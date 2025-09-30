from typing import List
import networkx as nx


class ExecutionScheduler:
	"""Simple scheduler that returns topological execution order."""

	def create_execution_plan(self, dag: nx.DiGraph) -> List[str]:
		return list(nx.topological_sort(dag))



