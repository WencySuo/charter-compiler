import networkx as nx
from typing import List
from ..models import LLMCall


class DAGBuilder:
	def __init__(self):
		self.graph = nx.DiGraph()

	def build_from_calls(self, calls: List[LLMCall]) -> nx.DiGraph:
		for call in calls:
			self.graph.add_node(call.id, data=call)

		for call in calls:
			for dep_id in call.dependencies:
				self.graph.add_edge(dep_id, call.id)

		if not nx.is_directed_acyclic_graph(self.graph):
			cycles = list(nx.simple_cycles(self.graph))
			raise ValueError(f"Cycles detected: {cycles}")

		return self.graph

	def get_execution_order(self) -> List[str]:
		return list(nx.topological_sort(self.graph))


