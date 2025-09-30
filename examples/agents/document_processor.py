from typing import Dict, Any


class DummyLLM:
	def generate(self, prompt: str, max_tokens: int = 256) -> str:
		return f"GEN[{len(prompt)}]"


class DocumentProcessor:
	"""Test Case 1: Sequential Chain (Extraction → Validation → Classification)"""

	SYSTEM_PROMPT = """You are an expert document analyzer with capabilities in:
	- Entity extraction (people, organizations, dates, locations)
	- Information validation against source text
	- Document classification by type and purpose"""

	def __init__(self, llm_client=None):
		self.llm = llm_client or DummyLLM()

	def process(self, document: str) -> Dict[str, Any]:
		# Stage 1: Entity Extraction
		extraction_prompt = f"""{self.SYSTEM_PROMPT}

		Task: Extract all named entities from the following document.
		Format: JSON with categories (people, organizations, dates, locations)

		Document:
		{document}

		Entities:"""

		entities = self.llm.generate(extraction_prompt, max_tokens=500)

		# Stage 2: Validation (depends on extraction)
		validation_prompt = f"""{self.SYSTEM_PROMPT}

		Task: Validate the extracted entities against the source document.
		Mark each entity as: verified, uncertain, or incorrect.

		Extracted Entities:
		{entities}

		Source Document:
		{document}

		Validation Results:"""

		validated = self.llm.generate(validation_prompt, max_tokens=500)

		# Stage 3: Classification (depends on validation)
		classification_prompt = f"""{self.SYSTEM_PROMPT}

		Task: Classify this document based on its content and validated entities.
		Provide: document_type, primary_topic, confidence_score, key_insights

		Document:
		{document}

		Validated Entities:
		{validated}

		Classification:"""

		classification = self.llm.generate(classification_prompt, max_tokens=300)

		return {
			"entities": entities,
			"validated_entities": validated,
			"classification": classification,
		}

