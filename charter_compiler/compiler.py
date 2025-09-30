from typing import Dict
import yaml
import networkx as nx

from .parser.ast_parser import AgentASTParser
from .dag.builder import DAGBuilder
from .analysis.memoization import MemoizationAnalyzer
from .config.trtllm_config import TRTLLMConfigGenerator
from .executor.cache_manager import CacheIDManager
from .executor.scheduler import ExecutionScheduler
from .executor.orchestrator import TritonOrchestrator
from .monitor.metrics import MetricsCollector
from .models import CompiledAgent, ParsedAgent, LLMCall


class CharterCompiler:
	"""Main compiler orchestrating all components."""

	def __init__(self, config_path: str = "charter_compiler/configs/default.yaml"):
		self.config = self._load_config(config_path)
		self.parser = AgentASTParser()
		self.dag_builder = DAGBuilder()
		self.analyzer = MemoizationAnalyzer()
		self.config_generator = TRTLLMConfigGenerator()
		self.cache_manager = CacheIDManager()
		self.scheduler = ExecutionScheduler()
		self.orchestrator = TritonOrchestrator()
		self.metrics = self.orchestrator.metrics

	def _load_config(self, path: str) -> Dict:
		try:
			with open(path, "r", encoding="utf-8") as f:
				return yaml.safe_load(f) or {}
		except FileNotFoundError:
			return {}

	def compile(self, agent_file: str) -> CompiledAgent:
		# Parse AST into calls (heuristic list for POC)
		calls = self.parser.parse_file(agent_file)
		if not isinstance(calls, list):
			calls = []

		# Build DAG
		dag = self.dag_builder.build_from_calls(calls)

		# Analyze opportunities
		opportunities = self.analyzer.analyze(dag)

		# Generate cache config
		cache_config = self.config_generator.generate_from_analysis(opportunities)

		# Assign cache IDs
		self.cache_manager.assign_cache_ids(dag, opportunities)

		# Execution plan
		execution_plan = self.scheduler.create_execution_plan(dag)

		compiled = CompiledAgent(
			dag=dag,
			opportunities=opportunities,
			cache_config=cache_config,
			execution_plan=execution_plan,
		)

		# Attach helper mappings for orchestrator
		compiled.node_to_cache_id = self.cache_manager.node_to_cache_id  # type: ignore
		compiled.cache_manager = self.cache_manager  # type: ignore
		return compiled

	def execute(self, compiled_agent: CompiledAgent, inputs: Dict) -> Dict:
		return self.orchestrator.execute(compiled_agent, inputs)

