from charter_compiler.models import LLMCall, MemoizationOpportunity, MemoizationPattern


def test_llmcall_dataclass():
	call = LLMCall(
		id="n1",
		prompt_template="Hello {name}",
		input_variables=["name"],
		output_variable="out",
		line_number=1,
		dependencies=[],
		metadata={},
	)
	assert call.id == "n1"


def test_memoization_opportunity():
	opp = MemoizationOpportunity(
		pattern=MemoizationPattern.SHARED_PREFIX,
		nodes=["n1", "n2"],
		shared_prefix_length=10,
		estimated_cache_hit_rate=0.8,
		priority=90,
	)
	assert opp.priority == 90

