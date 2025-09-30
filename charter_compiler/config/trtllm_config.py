from dataclasses import dataclass, asdict
from typing import List, Optional

from ..models import MemoizationOpportunity


@dataclass
class RetentionConfig:
	start_token: int
	end_token: int
	priority: int  # 0-100
	duration: int  # seconds


@dataclass
class TRTLLMConfig:
	enable_kv_cache_reuse: bool = True
	tokens_per_block: int = 32
	kv_cache_free_gpu_mem_fraction: float = 0.95
	batch_scheduler_policy: str = "max_utilization"
	enable_chunked_context: bool = True
	kv_cache_host_memory_bytes: int = 45000000000
	retention_configs: Optional[List[RetentionConfig]] = None

	def to_triton_config(self) -> dict:
		"""Convert to Triton-like config dictionary structure."""
		config = {"parameters": []}
		for key, value in asdict(self).items():
			if value is not None and key != "retention_configs":
				config["parameters"].append({"key": key, "value": {"string_value": str(value)}})
		return config


class TRTLLMConfigGenerator:
	def generate_from_analysis(self, opportunities: List[MemoizationOpportunity]) -> TRTLLMConfig:
		"""Generate optimal config from memoization analysis."""
		config = TRTLLMConfig()

		# Optimize block size based on reuse density
		if self._has_high_reuse(opportunities):
			config.tokens_per_block = 32
		else:
			config.tokens_per_block = 64

		# Set retention configs for high-priority prefixes
		retention_configs: List[RetentionConfig] = []
		for opp in opportunities:
			if getattr(opp, "shared_prefix_length", 0) and opp.priority > 80:
				retention_configs.append(
					RetentionConfig(
						start_token=0,
						end_token=opp.shared_prefix_length,
						priority=opp.priority,
						duration=60,
					)
				)
		config.retention_configs = retention_configs or None
		return config

	@staticmethod
	def _has_high_reuse(opportunities: List[MemoizationOpportunity]) -> bool:
		return any(getattr(opp, "estimated_cache_hit_rate", 0) >= 0.8 for opp in opportunities)



