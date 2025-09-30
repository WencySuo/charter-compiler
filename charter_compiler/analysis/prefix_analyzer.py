from typing import List, Dict
import networkx as nx

from ..models import MemoizationOpportunity, MemoizationPattern


class SimpleTokenizer:
	"""Very simple whitespace tokenizer placeholder.

	Replace with a model-accurate tokenizer for production.
	"""

	def encode(self, text: str) -> List[str]:
		return text.split()


class PrefixAnalyzer:
	def __init__(self, tokenizer=None, min_prefix_length: int = 10):
		self.tokenizer = tokenizer or SimpleTokenizer()
		self.min_prefix_length = min_prefix_length

	def find_shared_prefixes(self, dag: nx.DiGraph) -> List[MemoizationOpportunity]:
		"""Identify shared prompt prefixes in DAG.

		Heuristic: Longest common token-prefix among prompts between node pairs.
		"""

		opportunities: List[MemoizationOpportunity] = []

		# Extract prompts
		prompts: Dict[str, str] = {}
		for node_id in dag.nodes():
			node_data = dag.nodes[node_id]["data"]
			prompts[node_id] = getattr(node_data, "prompt_template", "")

		# Pairwise compare
		node_ids = list(prompts.keys())
		for i in range(len(node_ids)):
			for j in range(i + 1, len(node_ids)):
				n1, n2 = node_ids[i], node_ids[j]
				p1, p2 = prompts[n1], prompts[n2]
				prefix_len = self._longest_common_prefix_tokens(p1, p2)
				if prefix_len >= self.min_prefix_length:
					opportunities.append(
						MemoizationOpportunity(
							pattern=MemoizationPattern.SHARED_PREFIX,
							nodes=[n1, n2],
							shared_prefix_length=prefix_len,
							estimated_cache_hit_rate=self._estimate_hit_rate(prefix_len, p1, p2),
							priority=self._calculate_priority(prefix_len),
						)
					)

		return opportunities

	def _estimate_hit_rate(self, prefix_len: int, p1: str, p2: str) -> float:
		t1 = len(self.tokenizer.encode(p1)) or 1
		t2 = len(self.tokenizer.encode(p2)) or 1
		return min(prefix_len / max(t1, t2), 1.0)

	def _calculate_priority(self, prefix_len: int) -> int:
		# Very simple mapping
		if prefix_len >= 200:
			return 95
		if prefix_len >= 100:
			return 85
		if prefix_len >= 50:
			return 80
		return 70

	def _longest_common_prefix_tokens(self, p1: str, p2: str) -> int:
		t1 = self.tokenizer.encode(p1)
		t2 = self.tokenizer.encode(p2)
		match = 0
		for a, b in zip(t1, t2):
			if a == b:
				match += 1
			else:
				break
		return match



