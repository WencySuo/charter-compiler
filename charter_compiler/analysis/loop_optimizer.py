from typing import List
import networkx as nx

from ..models import MemoizationOpportunity, MemoizationPattern


class LoopOptimizer:
	"""Heuristic loop optimization estimator.

	Identifies nodes with loop metadata and suggests unrolling/memoization.
	"""

	def optimize(self, dag: nx.DiGraph) -> List[MemoizationOpportunity]:
		opportunities: List[MemoizationOpportunity] = []
		for node_id in dag.nodes():
			node_data = dag.nodes[node_id]["data"]
			if getattr(node_data, "metadata", {}).get("in_loop"):
				opportunities.append(
					MemoizationOpportunity(
						pattern=MemoizationPattern.LOOP_INVARIANT,
						nodes=[node_id],
						shared_prefix_length=0,
						estimated_cache_hit_rate=0.9,
						priority=95,
					)
				)
		return opportunities



