from ..models import MemoizationOpportunity, MemoizationPattern


class PriorityMapper:
	"""Maps memoization patterns to cache priorities based on heuristics."""

	BASE_PRIORITIES = {
		MemoizationPattern.SHARED_PREFIX: 85,
		MemoizationPattern.SEQUENTIAL_CHAIN: 75,
		MemoizationPattern.BRANCHING: 70,
		MemoizationPattern.LOOP_INVARIANT: 95,
		MemoizationPattern.DELTA_MATCHING: 90,
	}

	def assign(self, opportunity: MemoizationOpportunity) -> int:
		priority = self.BASE_PRIORITIES.get(opportunity.pattern, 70)
		# Boost priority with longer prefixes
		if getattr(opportunity, "shared_prefix_length", 0) >= 500:
			priority = max(priority, 90)
		return priority

