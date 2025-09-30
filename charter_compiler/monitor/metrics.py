from dataclasses import dataclass
from typing import List, Dict
import time


@dataclass
class ExecutionMetrics:
	node_id: str
	start_time: float
	end_time: float
	cache_hit_rate: float
	tokens_processed: int
	memory_usage_mb: float

	@property
	def latency(self) -> float:
		return self.end_time - self.start_time if self.end_time >= self.start_time else self.end_time

	@property
	def tokens_per_second(self) -> float:
		return self.tokens_processed / self.latency if self.latency > 0 else 0


class MetricsCollector:
	def __init__(self):
		self.metrics: List[ExecutionMetrics] = []

	def _now(self) -> float:
		return time.time()

	def record_execution(self, metrics: ExecutionMetrics):
		self.metrics.append(metrics)

	def get_summary(self) -> Dict:
		if not self.metrics:
			return {}

		latencies = [m.latency for m in self.metrics]
		cache_hits = [m.cache_hit_rate for m in self.metrics]

		return {
			"total_requests": len(self.metrics),
			"avg_latency": sum(latencies) / len(latencies),
			"p50_latency": sorted(latencies)[len(latencies) // 2],
			"p99_latency": sorted(latencies)[max(int(len(latencies) * 0.99) - 1, 0)],
			"avg_cache_hit_rate": sum(cache_hits) / len(cache_hits),
			"total_tokens": sum(m.tokens_processed for m in self.metrics),
			"avg_tokens_per_second": sum(m.tokens_per_second for m in self.metrics) / len(self.metrics),
			"peak_memory_mb": max(m.memory_usage_mb for m in self.metrics),
		}



