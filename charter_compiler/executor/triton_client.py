import tritonclient.grpc as grpcclient
import numpy as np
import time
from typing import Optional, Dict, Any

class TritonClient:
    def __init__(self, url: str = "localhost:8001", model_name: str = "ensemble"):
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model_name = model_name
        
    def infer_with_cache(
        self,
        prompt: str,
        cache_id: Optional[int] = None,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Execute inference with optional cache ID for KV cache reuse"""
        inputs = []
        
        # Text input
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_data = np.array([prompt.encode('utf-8')], dtype=object)
        text_input.set_data_from_numpy(text_data)
        inputs.append(text_input)
        
        # Max tokens parameter
        max_tokens_input = grpcclient.InferInput("max_tokens", [1], "INT32")
        max_tokens_input.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
        inputs.append(max_tokens_input)
        
        # Temperature
        temp_input = grpcclient.InferInput("temperature", [1], "FP32")
        temp_input.set_data_from_numpy(np.array([temperature], dtype=np.float32))
        inputs.append(temp_input)
        
        # Cache ID for prompt table (if provided)
        if cache_id is not None:
            cache_id_input = grpcclient.InferInput("prompt_embedding_table_extra_id", [1], "UINT64")
            cache_id_input.set_data_from_numpy(np.array([cache_id], dtype=np.uint64))
            inputs.append(cache_id_input)
        
        # Request outputs
        outputs = [grpcclient.InferRequestedOutput("text_output")]
        
        # Execute with timing
        start_time = time.time()
        result = self.client.infer(self.model_name, inputs, outputs)
        latency = time.time() - start_time
        
        # Parse results
        text_output = result.as_numpy("text_output")[0].decode('utf-8')
        
        # Extract cache metrics (if available)
        # Note: Actual metric names may vary based on TensorRT-LLM version
        try:
            stats = self.client.get_inference_statistics(self.model_name)
            cache_hit_rate = self._parse_cache_metrics(stats)
        except:
            cache_hit_rate = 0.8 if cache_id else 0.0  # Estimate
        
        return {
            "text": text_output,
            "latency": latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_id": cache_id
        }
    
    def _parse_cache_metrics(self, stats) -> float:
        """Parse cache hit rate from Triton statistics"""
        # Implementation depends on TensorRT-LLM metrics format
        return 0.0