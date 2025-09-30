from typing import Dict, List

from .triton_client import TritonClient
from .cache_manager import CacheIDManager
from ..monitor.metrics import MetricsCollector, ExecutionMetrics


class TritonOrchestrator:
	def __init__(self, client: TritonClient | None = None):
		self.triton_client = client or TritonClient()
		self.metrics = MetricsCollector()

	def prepopulate_cache(self, shared_prefixes: List[str]):
		for idx, prefix in enumerate(shared_prefixes):
			self.triton_client.infer_with_cache(prompt=prefix, cache_id=idx + 1, max_tokens=1)

	def execute(self, compiled_agent, inputs: Dict):
		"""Execute compiled agent sequentially as per execution plan.

		Simulates cache-aware execution, recording metrics for each node.
		"""

		results: Dict[str, str] = {}
		cache_manager: CacheIDManager = compiled_agent.cache_config and getattr(compiled_agent, "cache_manager", None)  # type: ignore
		for node_id in compiled_agent.execution_plan:
			node = compiled_agent.dag.nodes[node_id]["data"]
			prompt = getattr(node, "prompt_template", "")
			cache_id = getattr(compiled_agent, "node_to_cache_id", {}).get(node_id)

			start_time = self.metrics._now() if hasattr(self.metrics, "_now") else None
			result = self.triton_client.infer_with_cache(prompt=prompt, cache_id=cache_id)
			results[node_id] = result["text"]
			end_time = self.metrics._now() if hasattr(self.metrics, "_now") else None

			# Record metrics (simulate tokens)
			self.metrics.record_execution(
				ExecutionMetrics(
					node_id=node_id,
					start_time=start_time or 0.0,
					end_time=end_time or result["latency"],
					cache_hit_rate=result["cache_hit_rate"],
					tokens_processed=128,
					memory_usage_mb=1024.0,
				)
			)

		return results



