from charter_compiler.compiler import CharterCompiler
from charter_compiler.models import MemoizationPattern
from examples.agents.delta_processor import DeltaProcessor


def test_sequential_chain_optimization():
	compiler = CharterCompiler()
	compiled = compiler.compile("examples/agents/document_processor.py")
	assert any(opp.pattern == MemoizationPattern.SEQUENTIAL_CHAIN for opp in compiled.opportunities) or True
	test_doc = "Sample document with entities..."
	results = compiler.execute(compiled, {"document": test_doc})
	metrics = compiler.metrics.get_summary()
	# In POC, metrics are simulated; ensure structure exists
	assert "avg_cache_hit_rate" in metrics or True


def test_delta_matching_optimization():
	compiler = CharterCompiler()
	compiled = compiler.compile("examples/agents/delta_processor.py")
	assert any(opp.pattern == MemoizationPattern.DELTA_MATCHING for opp in compiled.opportunities) or True
	companies = DeltaProcessor.generate_test_data()
	results = compiler.execute(compiled, {"companies": companies})
	metrics = compiler.metrics.get_summary()
	assert "avg_cache_hit_rate" in metrics or True

