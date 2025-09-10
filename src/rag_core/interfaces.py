from typing import Any, Protocol 
# from collections.abc import Iterable # For the support of streaming-retrieval or generators, you should use Iterable[Doc] instead of lists only.
from dataclasses import dataclass

@dataclass
class Query:
	"""User query with top-k documents setting."""
	text: str
	k: int = 5 # (5-20)

@dataclass
class Doc:
	"""A retrieved document chunk."""
	id: str
	text: str
	metadata: dict[str, Any] | None = None

class Retriever(Protocol):
	"""Retrieves a ranked list of documents for a query."""
	def retrieve(self, q: Query) -> list[Doc]: ...

class Reranker(Protocol):
	"""Reorders a small candidate set for better relevance."""
	def rerank(self, q: Query, docs: list[Doc]) -> list[Doc]: ...

class LLMClient(Protocol):
	"""LLM Interface, just turn a prompt into text."""
	def generate(self, prompt: str, **opts: Any) -> str: ...
