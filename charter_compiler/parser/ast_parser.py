import ast
from typing import List
from ..models import LLMCall


class LLMCallExtractor(ast.NodeVisitor):
	def __init__(self):
		self.calls: List[LLMCall] = []
 		self.current_function_calls: List[str] = []

	def visit_Call(self, node: ast.Call):
		# Heuristic: detect attribute calls like self.llm.generate(
		is_attr = isinstance(node.func, ast.Attribute)
		func_name = node.func.attr if is_attr else ""
		owner = node.func.value.attr if is_attr and isinstance(node.func.value, ast.Attribute) else (
			node.func.value.id if is_attr and isinstance(node.func.value, ast.Name) else ""
		)
		if func_name in {"generate", "create", "__call__"}:
			# Extract prompt argument if present
			prompt_template = ""
			if node.args:
				prompt_template = self._stringify(node.args[0])
			elif node.keywords:
				for kw in node.keywords:
					if kw.arg in {"prompt", "messages", "input", "text"}:
						prompt_template = self._stringify(kw.value)
						break

			call_id = f"call_{node.lineno}_{node.col_offset}"
			dependencies = self.current_function_calls[-1:] if self.current_function_calls else []
			self.current_function_calls.append(call_id)
			self.calls.append(
				LLMCall(
					id=call_id,
					prompt_template=prompt_template,
					input_variables=[],
					output_variable="",
					line_number=node.lineno,
					dependencies=dependencies,
					metadata={"in_loop": self._in_loop(node)},
				)
			)

		self.generic_visit(node)

	def _stringify(self, node: ast.AST) -> str:
		if isinstance(node, ast.Constant) and isinstance(node.value, str):
			return node.value
		if isinstance(node, ast.JoinedStr):
			# f-string: concatenate constant parts and placeholders
			parts = []
			for value in node.values:
				if isinstance(value, ast.Constant) and isinstance(value.value, str):
					parts.append(value.value)
				else:
					parts.append("{var}")
			return "".join(parts)
		return ""

	def _in_loop(self, node: ast.AST) -> bool:
		# Walk up parents to see if inside For/While; requires parent links
		return any(isinstance(ancestor, (ast.For, ast.While)) for ancestor in self._parents(node))

	def _parents(self, node: ast.AST):
		# Best-effort: no direct parent links; rely on stack in visit. Here return empty.
		return []


class AgentASTParser:
	def parse_file(self, path: str) -> List[LLMCall]:
		with open(path, "r", encoding="utf-8") as f:
			source = f.read()
		module = ast.parse(source)
		extractor = LLMCallExtractor()
		extractor.visit(module)
		return extractor.calls
