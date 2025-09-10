from typing import Any, Protocol 
# Iterable (future) instead of lists only: For streaming retrieval (generators), prefer: 
# from collections.abc import Iterable 
from dataclasses import dataclass

@dataclass
class Query:
	"""End-user query with k top-ranked documents to retrieve."""
	text: str
	k: int = 5 # ~ 5-20

@dataclass
class Doc:
	"""A retrieved document chunk/passage with optional metadata."""
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
	"""LLM interface: Just turn a prompt into text (minimal)."""
	def generate(self, prompt: str, **opts: Any) -> str: ...
