"""
Quick smoke test for the RAGPipeline with an InMemoryBM25Retriever and a FakeLLM. Run with:

python scripts/repl_test.py
"""

from rag_core.interfaces import Query, Doc
from rag_core.pipeline import RAGPipeline
from rag_core.retrievers.bm25.inmem import InMemoryBM25Retriever


class FakeLLM:
