from typing import List, Optional
import networkx as nx

from ..models import MemoizationOpportunity


class CacheIDManager:
	def __init__(self):
		self.cache_registry = {}
		self.next_cache_id = 1
		self.node_to_cache_id = {}

	def assign_cache_ids(self, dag: nx.DiGraph, opportunities: List[MemoizationOpportunity]):
		"""Assign cache IDs based on memoization opportunities.

		This simple implementation groups nodes that appear together in an
		opportunity with shared prefixes.
		"""

		groups = []
		for opp in opportunities:
			if getattr(opp, "shared_prefix_length", 0) > 0:
				groups.append(opp.nodes)

		# Assign IDs to groups
		for group in groups:
			cache_id = self.next_cache_id
			self.next_cache_id += 1
			for node_id in group:
				self.node_to_cache_id[node_id] = cache_id

	def get_cache_id(self, node_id: str) -> Optional[int]:
		return self.node_to_cache_id.get(node_id)

