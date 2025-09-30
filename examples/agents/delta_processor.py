from typing import List, Dict


class DummyLLM:
	def generate(self, prompt: str, max_tokens: int = 256) -> str:
		return f"GEN[{len(prompt)}]"


class DeltaProcessor:
	"""Test Case 2: Delta Matching Loop"""

	SYSTEM_PROMPT = """You are a business analyst specializing in company profiles.
	Generate detailed analysis based on provided company metrics."""

	TEMPLATE = """Analyze the following company:

	Company: {company}
	Annual Revenue: {revenue}
	Employee Count: {employees}
	Industry: {industry}
	Founded: {year}

	Provide a comprehensive analysis including:
	1. Financial health assessment
	2. Growth trajectory prediction
	3. Market position evaluation
	4. Risk factors
	5. Investment recommendation

	Analysis:"""

	def __init__(self, llm_client=None):
		self.llm = llm_client or DummyLLM()

	def batch_process(self, companies: List[Dict]) -> List[Dict]:
		results: List[Dict] = []
		base_prompt = f"{self.SYSTEM_PROMPT}\n\n{self.TEMPLATE}"
		for company_data in companies:
			full_prompt = base_prompt.format(**company_data)
			analysis = self.llm.generate(full_prompt, max_tokens=400)
			results.append({"company": company_data["company"], "analysis": analysis})
		return results

	@staticmethod
	def generate_test_data() -> List[Dict]:
		return [
			{"company": "TechCorp", "revenue": "$10M", "employees": "50", "industry": "Software", "year": "2019"},
			{"company": "DataInc", "revenue": "$15M", "employees": "75", "industry": "Analytics", "year": "2018"},
			{"company": "CloudCo", "revenue": "$8M", "employees": "40", "industry": "Cloud Services", "year": "2020"},
			{"company": "AIStart", "revenue": "$5M", "employees": "25", "industry": "AI/ML", "year": "2021"},
			{"company": "SecureNet", "revenue": "$12M", "employees": "60", "industry": "Cybersecurity", "year": "2017"},
			{"company": "DevTools", "revenue": "$20M", "employees": "100", "industry": "Developer Tools", "year": "2016"},
			{"company": "MobileApp", "revenue": "$7M", "employees": "35", "industry": "Mobile", "year": "2020"},
			{"company": "WebScale", "revenue": "$25M", "employees": "120", "industry": "Infrastructure", "year": "2015"},
			{"company": "DataFlow", "revenue": "$9M", "employees": "45", "industry": "Data Engineering", "year": "2019"},
			{"company": "MLOps", "revenue": "$11M", "employees": "55", "industry": "ML Operations", "year": "2020"},
		]



