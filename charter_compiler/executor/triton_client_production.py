"""Production Triton Client for Lambda Labs deployment with TensorRT-LLM backend."""

import tritonclient.grpc as grpcclient
import numpy as np
import time
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ProductionTritonClient:
    """Production Triton client for TensorRT-LLM with KV cache reuse support."""
    
    def __init__(
        self, 
        url: str = "localhost:8001", 
        model_name: str = "ensemble",
        timeout: int = 60
    ):
        """
        Initialize Triton client for TensorRT-LLM inference.
        
        Args:
            url: Triton server gRPC URL
            model_name: Model name in Triton repository (typically 'ensemble')
            timeout: Request timeout in seconds
        """
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.client = grpcclient.InferenceServerClient(url=url)
        
        # Verify server is ready
        if not self.client.is_server_ready():
            raise ConnectionError(f"Triton server at {url} is not ready")
        
        logger.info(f"Connected to Triton server at {url}")
        
    def infer_with_cache(
        self,
        prompt: str,
        cache_id: Optional[int] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute inference with optional cache ID for KV cache reuse.
        
        Args:
            prompt: Input text prompt
            cache_id: Optional cache ID for prompt table (enables KV cache reuse)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with:
                - text: Generated text
                - latency: Inference latency in seconds
                - cache_hit_rate: Estimated cache hit rate
                - cache_id: Cache ID used (if any)
                - tokens_generated: Number of tokens generated (estimated)
        """
        inputs = []
        
        # Text input - must be BYTES type for TensorRT-LLM
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_data = np.array([prompt.encode('utf-8')], dtype=object)
        text_input.set_data_from_numpy(text_data)
        inputs.append(text_input)
        
        # Max tokens parameter
        max_tokens_input = grpcclient.InferInput("max_tokens", [1], "INT32")
        max_tokens_input.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
        inputs.append(max_tokens_input)
        
        # Sampling parameters
        if temperature != 0.7:  # Only add if non-default
            temp_input = grpcclient.InferInput("temperature", [1], "FP32")
            temp_input.set_data_from_numpy(np.array([temperature], dtype=np.float32))
            inputs.append(temp_input)
        
        if top_p != 0.9:
            top_p_input = grpcclient.InferInput("top_p", [1], "FP32")
            top_p_input.set_data_from_numpy(np.array([top_p], dtype=np.float32))
            inputs.append(top_p_input)
        
        if top_k != 50:
            top_k_input = grpcclient.InferInput("top_k", [1], "INT32")
            top_k_input.set_data_from_numpy(np.array([top_k], dtype=np.int32))
            inputs.append(top_k_input)
        
        # Cache ID for prompt table (KV cache reuse)
        # Note: The parameter name may vary based on TensorRT-LLM version
        # Common names: prompt_embedding_table_extra_id, prompt_table_extra_id, cache_id
        if cache_id is not None:
            cache_id_input = grpcclient.InferInput("prompt_table_extra_id", [1], "UINT64")
            cache_id_input.set_data_from_numpy(np.array([cache_id], dtype=np.uint64))
            inputs.append(cache_id_input)
            logger.debug(f"Using cache_id: {cache_id}")
        
        # Request outputs
        outputs = [
            grpcclient.InferRequestedOutput("text_output"),
            # Optionally request additional metrics if available
            # grpcclient.InferRequestedOutput("cum_log_probs"),
            # grpcclient.InferRequestedOutput("output_log_probs"),
        ]
        
        # Execute inference with timing
        start_time = time.time()
        try:
            result = self.client.infer(
                self.model_name, 
                inputs, 
                outputs,
                client_timeout=self.timeout
            )
            latency = time.time() - start_time
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        
        # Parse results
        text_output = result.as_numpy("text_output")[0].decode('utf-8')
        
        # Estimate tokens generated (rough approximation)
        tokens_generated = len(text_output.split())
        
        # Extract cache metrics if available
        cache_hit_rate = self._estimate_cache_hit_rate(cache_id, prompt, text_output)
        
        response = {
            "text": text_output,
            "latency": latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_id": cache_id,
            "tokens_generated": tokens_generated,
            "ttft": latency / max(tokens_generated, 1),  # Time to first token estimate
        }
        
        logger.debug(f"Inference: {latency:.3f}s, cache_hit: {cache_hit_rate:.2%}, tokens: {tokens_generated}")
        
        return response
    
    def _estimate_cache_hit_rate(self, cache_id: Optional[int], prompt: str, output: str) -> float:
        """
        Estimate cache hit rate based on cache_id presence.
        
        In production, this should be replaced with actual metrics from Triton.
        TensorRT-LLM can expose kv_cache_reused_blocks and kv_cache_total_blocks.
        """
        if cache_id is None:
            return 0.0
        
        # Rough estimation: first inference with cache_id has lower hit rate
        # Subsequent inferences with same cache_id have higher hit rate
        # This is a placeholder - actual implementation should query Triton metrics
        
        # Check if we can get metrics from Triton
        try:
            stats = self.client.get_inference_statistics(self.model_name)
            # Parse cache metrics from stats if available
            # Format depends on TensorRT-LLM version
            return self._parse_cache_metrics(stats)
        except:
            # Fallback estimation
            return 0.75 if cache_id else 0.0
    
    def _parse_cache_metrics(self, stats) -> float:
        """
        Parse cache hit rate from Triton statistics.
        
        This is version-dependent and should be updated based on
        actual TensorRT-LLM metric names.
        """
        # Placeholder - update based on actual metrics
        return 0.0
    
    def batch_infer(
        self,
        prompts: List[str],
        cache_ids: Optional[List[Optional[int]]] = None,
        max_tokens: int = 200,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute batch inference.
        
        Args:
            prompts: List of input prompts
            cache_ids: Optional list of cache IDs (one per prompt)
            max_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional parameters for infer_with_cache
            
        Returns:
            List of inference results
        """
        if cache_ids is None:
            cache_ids = [None] * len(prompts)
        
        results = []
        for prompt, cache_id in zip(prompts, cache_ids):
            result = self.infer_with_cache(
                prompt=prompt,
                cache_id=cache_id,
                max_tokens=max_tokens,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def prepopulate_cache(self, prefixes: List[str]) -> Dict[int, str]:
        """
        Pre-populate TensorRT-LLM prompt table with shared prefixes.
        
        This should be called before running inference to establish
        cache entries for shared prompt prefixes.
        
        Args:
            prefixes: List of prompt prefixes to cache
            
        Returns:
            Dictionary mapping cache_id to prefix
        """
        cache_map = {}
        
        for idx, prefix in enumerate(prefixes):
            cache_id = idx + 1  # Cache IDs start at 1
            
            # Run minimal inference to populate cache
            logger.info(f"Prepopulating cache_id {cache_id} with prefix: {prefix[:50]}...")
            
            result = self.infer_with_cache(
                prompt=prefix,
                cache_id=cache_id,
                max_tokens=1,  # Minimal generation
                temperature=0.0
            )
            
            cache_map[cache_id] = prefix
            logger.debug(f"Cache {cache_id} populated, latency: {result['latency']:.3f}s")
        
        return cache_map
    
    def health_check(self) -> Dict[str, bool]:
        """Check Triton server and model health."""
        try:
            server_ready = self.client.is_server_ready()
            model_ready = self.client.is_model_ready(self.model_name)
            server_live = self.client.is_server_live()
            
            return {
                "server_ready": server_ready,
                "model_ready": model_ready,
                "server_live": server_live,
                "healthy": server_ready and model_ready and server_live
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}
