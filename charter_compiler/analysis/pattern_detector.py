from typing import List
import networkx as nx

from ..models import MemoizationOpportunity, MemoizationPattern


class SequencePatternDetector:
	"""Detects sequential chain and branching patterns in a DAG."""

	def detect(self, dag: nx.DiGraph) -> List[MemoizationOpportunity]:
		opportunities: List[MemoizationOpportunity] = []

		# Sequential chains: any path with length >= 2
		for node in dag.nodes():
			# simple heuristic: out-degree chain
			children = list(dag.successors(node))
			if len(children) == 1:
				chain = [node]
				cur = node
				while True:
					next_nodes = list(dag.successors(cur))
					if len(next_nodes) != 1:
						break
					cur = next_nodes[0]
					chain.append(cur)
				if len(chain) >= 3:
					opportunities.append(
						MemoizationOpportunity(
							pattern=MemoizationPattern.SEQUENTIAL_CHAIN,
							nodes=chain,
							shared_prefix_length=0,
							estimated_cache_hit_rate=0.65,
							priority=80,
						)
					)

		# Branching: nodes with out-degree >= 2
		for node in dag.nodes():
			children = list(dag.successors(node))
			if len(children) >= 2:
				opportunities.append(
					MemoizationOpportunity(
						pattern=MemoizationPattern.BRANCHING,
						nodes=[node] + children,
						shared_prefix_length=0,
						estimated_cache_hit_rate=0.5,
						priority=70,
					)
				)

		# Delta matching: nodes marked as inside a loop
		for node_id in dag.nodes():
			node_data = dag.nodes[node_id]["data"]
			if getattr(node_data, "metadata", {}).get("in_loop"):
				opportunities.append(
					MemoizationOpportunity(
						pattern=MemoizationPattern.DELTA_MATCHING,
						nodes=[node_id],
						shared_prefix_length=0,
						estimated_cache_hit_rate=0.85,
						priority=90,
					)
				)

		return opportunities

