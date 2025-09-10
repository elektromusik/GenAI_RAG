from typing import Protocol, List, Dict, Any, Iterable
from dataclasses import dataclass

@dataclass
class Query:
	text: str
	k: int = 5

@dataclass
class Doc:
	id: str
	text: str
	metadata: Dict[str, Any] | None = None

class Retriever(Protocol):
	def retrieve(self, q: Query) -> List[Doc]: ...

class Reranker(Protocol):
	def rerank(self, q: Query, docs: List[Doc]) -> List[Doc]: ...

class LLMClient(Protocol):
	def generate(self, prompt: str, **opts: Any) -> str: ...
