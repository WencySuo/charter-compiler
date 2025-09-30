#!/usr/bin/env python3
"""
Test runner for Charter Compiler on Lambda Labs infrastructure.

This script runs the full test suite against a Triton Inference Server
running on Lambda Labs with TensorRT-LLM backend.
"""

import sys
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from charter_compiler.executor.triton_client_production import ProductionTritonClient
from examples.agents.document_processor import DocumentProcessor
from examples.agents.delta_processor import DeltaProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LambdaTestRunner:
    """Test runner for Lambda Labs deployment."""
    
    def __init__(self, config_path: str = None):
        """Initialize test runner with configuration."""
        self.config = self._load_config(config_path)
        self.client = ProductionTritonClient(
            url=self.config.get('triton_url', 'localhost:8001'),
            model_name=self.config.get('model_name', 'ensemble')
        )
        self.results = {}
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load test configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            'triton_url': 'localhost:8001',
            'model_name': 'ensemble',
            'test_cases': {
                'sequential_chain': {
                    'iterations': 10,
                    'warmup': 2
                },
                'delta_matching': {
                    'iterations': 5,
                    'variations': 5
                }
            }
        }
    
    def run_all_tests(self):
        """Run all test cases."""
        print("=" * 80)
        print("Charter Compiler - Lambda Labs Test Suite")
        print("=" * 80)
        print(f"\nTriton Server: {self.config['triton_url']}")
        print(f"Model: {self.config['model_name']}\n")
        
        # Health check
        if not self._health_check():
            print("❌ Health check failed. Aborting tests.")
            return
        
        # Run test cases
        self._test_sequential_chain()
        self._test_delta_matching()
        self._test_cache_prepopulation()
        
        # Print summary
        self._print_summary()
    
    def _health_check(self) -> bool:
        """Verify Triton server is healthy."""
        print("[Health Check]")
        print("-" * 80)
        
        health = self.client.health_check()
        
        for key, value in health.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        
        print()
        return health.get('healthy', False)
    
    def _test_sequential_chain(self):
        """Test sequential chain pattern (Document Processor)."""
        print("[Test 1] Sequential Chain - Document Processor")
        print("-" * 80)
        
        config = self.config['test_cases']['sequential_chain']
        iterations = config.get('iterations', 10)
        warmup = config.get('warmup', 2)
        
        test_documents = [
            """John Smith joined TechCorp in 2020 as Chief Technology Officer.
            The company, based in San Francisco, has raised $50M in Series B funding.
            Their main product is an AI-powered analytics platform.""",
            
            """Sarah Johnson founded DataInc in 2018. The Boston-based startup
            specializes in real-time data analytics. They recently announced
            a partnership with Microsoft Azure.""",
            
            """CloudCo, headquartered in Seattle, provides cloud infrastructure
            services. CEO Michael Chen led the company to profitability in 2022.
            The company employs 200 people across 3 offices."""
        ]
        
        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        for doc in test_documents[:warmup]:
            self.client.infer_with_cache(
                prompt=f"Extract entities from: {doc}",
                max_tokens=100
            )
        
        # Actual test
        print(f"Running test ({iterations} iterations)...")
        latencies = []
        cache_hits = []
        
        start_time = time.time()
        
        for i, doc in enumerate(test_documents[:iterations]):
            # Stage 1: Entity Extraction
            result1 = self.client.infer_with_cache(
                prompt=f"Extract named entities from this text:\n{doc}",
                cache_id=1,  # Shared system prompt
                max_tokens=150
            )
            
            # Stage 2: Validation (uses output from stage 1)
            result2 = self.client.infer_with_cache(
                prompt=f"Validate these entities:\n{result1['text']}\n\nAgainst source:\n{doc}",
                cache_id=1,
                max_tokens=150
            )
            
            # Stage 3: Classification
            result3 = self.client.infer_with_cache(
                prompt=f"Classify this document:\n{doc}\n\nWith entities:\n{result2['text']}",
                cache_id=1,
                max_tokens=100
            )
            
            # Track metrics
            total_latency = result1['latency'] + result2['latency'] + result3['latency']
            avg_cache_hit = (result1['cache_hit_rate'] + result2['cache_hit_rate'] + result3['cache_hit_rate']) / 3
            
            latencies.append(total_latency)
            cache_hits.append(avg_cache_hit)
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{iterations}: {total_latency:.3f}s, cache: {avg_cache_hit:.1%}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        avg_cache_hit = sum(cache_hits) / len(cache_hits)
        throughput = iterations / total_time
        
        print(f"\nResults:")
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Average latency: {avg_latency:.3f}s")
        print(f"  ✓ Cache hit rate: {avg_cache_hit:.1%}")
        print(f"  ✓ Throughput: {throughput:.2f} docs/sec")
        
        self.results['sequential_chain'] = {
            'avg_latency': avg_latency,
            'cache_hit_rate': avg_cache_hit,
            'throughput': throughput,
            'total_time': total_time
        }
        
        print()
    
    def _test_delta_matching(self):
        """Test delta matching pattern (Delta Processor)."""
        print("[Test 2] Delta Matching - Company Analysis")
        print("-" * 80)
        
        config = self.config['test_cases']['delta_matching']
        variations = config.get('variations', 5)
        
        companies = DeltaProcessor.generate_test_data()[:variations]
        
        # Base prompt template
        base_prompt = """You are a business analyst. Analyze this company:

Company: {company}
Revenue: {revenue}
Employees: {employees}
Industry: {industry}
Founded: {year}

Provide analysis:"""
        
        print(f"Processing {variations} companies...")
        
        latencies = []
        cache_hits = []
        
        start_time = time.time()
        
        for i, company in enumerate(companies):
            prompt = base_prompt.format(**company)
            
            # Use same cache_id for all (shared base prompt)
            result = self.client.infer_with_cache(
                prompt=prompt,
                cache_id=2,  # Different from sequential chain
                max_tokens=200
            )
            
            latencies.append(result['latency'])
            cache_hits.append(result['cache_hit_rate'])
            
            print(f"  {company['company']}: {result['latency']:.3f}s, cache: {result['cache_hit_rate']:.1%}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        avg_cache_hit = sum(cache_hits) / len(cache_hits)
        
        # Cache should improve after first iteration
        first_iter_cache = cache_hits[0]
        later_iter_cache = sum(cache_hits[1:]) / len(cache_hits[1:]) if len(cache_hits) > 1 else 0
        
        print(f"\nResults:")
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Average latency: {avg_latency:.3f}s")
        print(f"  ✓ Overall cache hit: {avg_cache_hit:.1%}")
        print(f"  ✓ First iteration cache: {first_iter_cache:.1%}")
        print(f"  ✓ Later iterations cache: {later_iter_cache:.1%}")
        
        self.results['delta_matching'] = {
            'avg_latency': avg_latency,
            'cache_hit_rate': avg_cache_hit,
            'first_iter_cache': first_iter_cache,
            'later_iter_cache': later_iter_cache,
            'total_time': total_time
        }
        
        print()
    
    def _test_cache_prepopulation(self):
        """Test cache prepopulation strategy."""
        print("[Test 3] Cache Prepopulation")
        print("-" * 80)
        
        # Define shared prefixes
        prefixes = [
            "You are an expert document analyzer. Extract entities from:",
            "You are a business analyst. Analyze the following company:",
            "You are a technical writer. Summarize this article:"
        ]
        
        print("Prepopulating cache with shared prefixes...")
        cache_map = self.client.prepopulate_cache(prefixes)
        
        print(f"✓ Prepopulated {len(cache_map)} cache entries")
        
        # Test that prepopulated cache is used
        print("\nTesting prepopulated cache...")
        test_prompt = prefixes[0] + "\n\nSample document here..."
        
        result = self.client.infer_with_cache(
            prompt=test_prompt,
            cache_id=1,  # Should match first prefix
            max_tokens=50
        )
        
        print(f"  ✓ Latency: {result['latency']:.3f}s")
        print(f"  ✓ Cache hit rate: {result['cache_hit_rate']:.1%}")
        
        self.results['cache_prepopulation'] = {
            'prefixes_cached': len(cache_map),
            'test_latency': result['latency'],
            'test_cache_hit': result['cache_hit_rate']
        }
        
        print()
    
    def _print_summary(self):
        """Print test summary."""
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)
        
        for test_name, metrics in self.results.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'cache' in key or 'rate' in key:
                        print(f"  {key}: {value:.1%}")
                    else:
                        print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Charter Compiler tests on Lambda Labs')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to test configuration YAML'
    )
    parser.add_argument(
        '--triton-url',
        type=str,
        default='localhost:8001',
        help='Triton server URL (default: localhost:8001)'
    )
    
    args = parser.parse_args()
    
    # Override config with CLI args
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    config['triton_url'] = args.triton_url
    
    # Save to temp file
    temp_config = Path('/tmp/lambda_test_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run tests
    runner = LambdaTestRunner(config_path=str(temp_config))
    runner.run_all_tests()


if __name__ == "__main__":
    main()
