#!/usr/bin/env python3
"""
Benchmark script for comparing baseline vs cache-optimized performance on Lambda Labs.

This script measures the performance improvement from KV cache reuse with TensorRT-LLM.
"""

import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict
import argparse

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from charter_compiler.executor.triton_client_production import ProductionTritonClient


class LambdaBenchmark:
    """Benchmark runner for Lambda Labs Triton deployment."""
    
    def __init__(self, triton_url: str = "localhost:8001"):
        """Initialize benchmark with Triton client."""
        self.client = ProductionTritonClient(url=triton_url)
        
    def benchmark_cache_performance(self, iterations: int = 10):
        """
        Benchmark cache reuse performance.
        
        Compares:
        1. Baseline: No cache reuse (cache_id=None)
        2. Optimized: With cache reuse (cache_id=1)
        """
        print("=" * 80)
        print("Lambda Labs Cache Performance Benchmark")
        print("=" * 80)
        print(f"\nIterations: {iterations}\n")
        
        # Shared base prompt (simulates shared system prompt + template)
        base_prompt = """You are an expert business analyst specializing in company analysis.
Provide detailed financial and strategic analysis based on company metrics.

Company Profile:
Company Name: {company}
Annual Revenue: {revenue}
Employee Count: {employees}
Industry Sector: {industry}
Year Founded: {year}

Required Analysis:
1. Financial health assessment
2. Market position evaluation
3. Growth potential analysis
4. Risk factor identification
5. Investment recommendation

Analysis:"""
        
        # Test companies with only variable changes
        companies = [
            {"company": f"TechCorp{i}", "revenue": f"${i*5}M", "employees": f"{i*10}", 
             "industry": "Software", "year": "2020"}
            for i in range(1, iterations + 1)
        ]
        
        # Benchmark 1: No cache reuse (baseline)
        print("[Baseline] No Cache Reuse (cache_id=None)")
        print("-" * 80)
        
        baseline_latencies = []
        
        for i, company in enumerate(companies):
            prompt = base_prompt.format(**company)
            result = self.client.infer_with_cache(
                prompt=prompt,
                cache_id=None,  # No cache reuse
                max_tokens=150
            )
            baseline_latencies.append(result['latency'])
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{iterations}: {result['latency']:.3f}s")
        
        baseline_avg = statistics.mean(baseline_latencies)
        baseline_median = statistics.median(baseline_latencies)
        baseline_stdev = statistics.stdev(baseline_latencies) if len(baseline_latencies) > 1 else 0
        
        print(f"\nBaseline Results:")
        print(f"  Average: {baseline_avg:.3f}s")
        print(f"  Median: {baseline_median:.3f}s")
        print(f"  Std Dev: {baseline_stdev:.3f}s")
        
        # Small delay between benchmarks
        time.sleep(2)
        
        # Benchmark 2: With cache reuse (optimized)
        print(f"\n[Optimized] With Cache Reuse (cache_id=1)")
        print("-" * 80)
        
        optimized_latencies = []
        cache_hits = []
        
        for i, company in enumerate(companies):
            prompt = base_prompt.format(**company)
            result = self.client.infer_with_cache(
                prompt=prompt,
                cache_id=1,  # Shared cache ID for all
                max_tokens=150
            )
            optimized_latencies.append(result['latency'])
            cache_hits.append(result['cache_hit_rate'])
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{iterations}: {result['latency']:.3f}s, "
                      f"cache: {result['cache_hit_rate']:.1%}")
        
        optimized_avg = statistics.mean(optimized_latencies)
        optimized_median = statistics.median(optimized_latencies)
        optimized_stdev = statistics.stdev(optimized_latencies) if len(optimized_latencies) > 1 else 0
        avg_cache_hit = statistics.mean(cache_hits)
        
        print(f"\nOptimized Results:")
        print(f"  Average: {optimized_avg:.3f}s")
        print(f"  Median: {optimized_median:.3f}s")
        print(f"  Std Dev: {optimized_stdev:.3f}s")
        print(f"  Avg Cache Hit: {avg_cache_hit:.1%}")
        
        # Calculate improvements
        print(f"\n" + "=" * 80)
        print("Performance Improvement")
        print("=" * 80)
        
        latency_reduction = (baseline_avg - optimized_avg) / baseline_avg * 100
        speedup = baseline_avg / optimized_avg
        
        print(f"\n  Latency Reduction: {latency_reduction:.1f}%")
        print(f"  Speedup Factor: {speedup:.2f}x")
        print(f"  Cache Hit Rate: {avg_cache_hit:.1%}")
        
        # Expected vs Actual
        print(f"\n  Expected (from spec): 40-60% latency reduction")
        if latency_reduction >= 40:
            print(f"  ✓ Target met: {latency_reduction:.1f}% reduction")
        else:
            print(f"  ⚠ Below target: {latency_reduction:.1f}% reduction")
        
        print("\n" + "=" * 80)
        
        return {
            'baseline': {
                'avg': baseline_avg,
                'median': baseline_median,
                'stdev': baseline_stdev
            },
            'optimized': {
                'avg': optimized_avg,
                'median': optimized_median,
                'stdev': optimized_stdev,
                'cache_hit_rate': avg_cache_hit
            },
            'improvement': {
                'latency_reduction_pct': latency_reduction,
                'speedup_factor': speedup
            }
        }
    
    def benchmark_batch_sizes(self, batch_sizes: List[int] = [1, 2, 4, 8]):
        """
        Benchmark different batch sizes to find optimal throughput.
        
        Args:
            batch_sizes: List of batch sizes to test
        """
        print("\n" + "=" * 80)
        print("Batch Size Optimization Benchmark")
        print("=" * 80)
        
        base_prompt = "Analyze this company: {company}\n\nProvide detailed analysis:"
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n[Batch Size: {batch_size}]")
            print("-" * 80)
            
            # Generate prompts
            prompts = [
                base_prompt.format(company=f"Company{i}")
                for i in range(batch_size)
            ]
            
            # Run batch inference
            start_time = time.time()
            batch_results = self.client.batch_infer(
                prompts=prompts,
                cache_ids=[1] * batch_size,  # Shared cache
                max_tokens=100
            )
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_latency = statistics.mean([r['latency'] for r in batch_results])
            throughput = batch_size / total_time
            
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg latency: {avg_latency:.3f}s")
            print(f"  Throughput: {throughput:.2f} req/sec")
            
            results[batch_size] = {
                'total_time': total_time,
                'avg_latency': avg_latency,
                'throughput': throughput
            }
        
        # Find optimal batch size
        optimal_batch = max(results.items(), key=lambda x: x[1]['throughput'])
        
        print(f"\n" + "=" * 80)
        print(f"Optimal Batch Size: {optimal_batch[0]}")
        print(f"  Throughput: {optimal_batch[1]['throughput']:.2f} req/sec")
        print("=" * 80)
        
        return results
    
    def benchmark_prefix_lengths(self, prefix_lengths: List[int] = [100, 500, 1000, 2000]):
        """
        Benchmark cache reuse with different shared prefix lengths.
        
        Tests how prefix length affects cache hit rate and performance.
        """
        print("\n" + "=" * 80)
        print("Prefix Length Impact Benchmark")
        print("=" * 80)
        
        results = {}
        
        for prefix_len in prefix_lengths:
            print(f"\n[Prefix Length: ~{prefix_len} tokens]")
            print("-" * 80)
            
            # Generate prefix of approximate token length
            # Rough estimate: 1 token ≈ 4 characters
            prefix = "You are an expert analyst. " + "Context information. " * (prefix_len // 4)
            suffix_template = "\n\nAnalyze item {item}: {data}"
            
            latencies = []
            cache_hits = []
            
            for i in range(10):
                prompt = prefix + suffix_template.format(item=i, data=f"data_{i}")
                result = self.client.infer_with_cache(
                    prompt=prompt,
                    cache_id=prefix_len,  # Different cache_id per prefix length
                    max_tokens=50
                )
                latencies.append(result['latency'])
                cache_hits.append(result['cache_hit_rate'])
            
            avg_latency = statistics.mean(latencies)
            avg_cache_hit = statistics.mean(cache_hits)
            
            print(f"  Avg latency: {avg_latency:.3f}s")
            print(f"  Avg cache hit: {avg_cache_hit:.1%}")
            
            results[prefix_len] = {
                'avg_latency': avg_latency,
                'cache_hit_rate': avg_cache_hit
            }
        
        print(f"\n" + "=" * 80)
        print("Summary: Longer prefixes should show higher cache hit rates")
        print("=" * 80)
        
        for prefix_len, metrics in results.items():
            print(f"  {prefix_len} tokens: {metrics['cache_hit_rate']:.1%} cache hit")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Benchmark Lambda Labs Triton deployment')
    parser.add_argument(
        '--triton-url',
        type=str,
        default='localhost:8001',
        help='Triton server URL (default: localhost:8001)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations for cache benchmark (default: 10)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        choices=['cache', 'batch', 'prefix', 'all'],
        default='cache',
        help='Benchmark type to run (default: cache)'
    )
    
    args = parser.parse_args()
    
    benchmark = LambdaBenchmark(triton_url=args.triton_url)
    
    if args.benchmark == 'cache' or args.benchmark == 'all':
        benchmark.benchmark_cache_performance(iterations=args.iterations)
    
    if args.benchmark == 'batch' or args.benchmark == 'all':
        benchmark.benchmark_batch_sizes()
    
    if args.benchmark == 'prefix' or args.benchmark == 'all':
        benchmark.benchmark_prefix_lengths()


if __name__ == "__main__":
    main()
