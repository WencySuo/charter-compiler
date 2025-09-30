from typing import List
import networkx as nx

from .prefix_analyzer import PrefixAnalyzer
from .pattern_detector import SequencePatternDetector
from .loop_optimizer import LoopOptimizer
from ..models import MemoizationOpportunity


class MemoizationAnalyzer:
	"""Aggregate analyzer that combines prefix, sequence, and loop analyses."""

	def __init__(self):
		self.prefix = PrefixAnalyzer()
		self.sequence = SequencePatternDetector()
		self.loop = LoopOptimizer()

	def analyze(self, dag: nx.DiGraph) -> List[MemoizationOpportunity]:
		opportunities: List[MemoizationOpportunity] = []
		opportunities.extend(self.prefix.find_shared_prefixes(dag))
		opportunities.extend(self.sequence.detect(dag))
		opportunities.extend(self.loop.optimize(dag))
		# De-duplicate by (pattern, nodes)
		unique = {}
		for opp in opportunities:
			key = (opp.pattern.value, tuple(opp.nodes))
			if key not in unique:
				unique[key] = opp
		return list(unique.values())

