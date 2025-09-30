from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from typing import Any


@dataclass
class LLMCall:
	"""Represents a single LLM inference call"""
	id: str
	prompt_template: str
	input_variables: List[str]
	output_variable: str
	line_number: int
	dependencies: List[str]
	metadata: Dict[str, Any]


@dataclass
class MemoizationOpportunity:
	"""Represents a caching opportunity"""
	pattern: "MemoizationPattern"
	nodes: List[str]
	shared_prefix_length: int
	estimated_cache_hit_rate: float
	priority: int


class MemoizationPattern(Enum):
	SHARED_PREFIX = "shared_prefix"
	SEQUENTIAL_CHAIN = "sequential_chain"
	BRANCHING = "branching"
	LOOP_INVARIANT = "loop_invariant"
	DELTA_MATCHING = "delta_matching"


@dataclass
class ParsedAgent:
	"""Container for parsed agent artifacts."""
	calls: List[LLMCall]
	control_flow: Any | None = None
	data_flow: Any | None = None


@dataclass
class CompiledAgent:
	"""Compiled representation ready for execution/orchestration."""
	dag: Any
	opportunities: List[MemoizationOpportunity]
	cache_config: Any
	execution_plan: List[str]
