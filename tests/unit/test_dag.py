import networkx as nx
from charter_compiler.dag.builder import DAGBuilder
from charter_compiler.models import LLMCall


def test_dag_builder_toposort():
	builder = DAGBuilder()
	calls = [
		LLMCall(id="a", prompt_template="p1", input_variables=[], output_variable="o1", line_number=1, dependencies=[], metadata={}),
		LLMCall(id="b", prompt_template="p2", input_variables=[], output_variable="o2", line_number=2, dependencies=["a"], metadata={}),
	]
	dag = builder.build_from_calls(calls)
	order = builder.get_execution_order()
	assert order[0] == "a" and order[-1] == "b"



