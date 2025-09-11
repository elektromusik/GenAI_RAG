from dataclasses import dataclass
from typing import Any
from .interfaces import Query, Retriever, Reranker, LLMClient, Doc
# Iterable (future): If Retriever returns Iterable[Doc], import it and materialize later.
# from collections.abc import Iterable

@dataclass
class RAGPipeline:
  """Composable RAG pipeline: Retrieve -> (optional) rerank -> prompt -> generate."""
  retriever: Retriever
  llm: LLMClient
  reranker: Reranker | None = None
  system_prompt: str = "Answer using only the provided context."

  def run(self, q: Query) -> dict[str, Any]:
    docs: list[Doc] = self.retriever.retrieve(q)

    # Iterable (future):
    # If Retriever returns Iterable[Doc], do:
    # docs_iter: Iterable[Doc] = self.retriever.retrieve(q)
    # docs = list(docs_iter)
    
    if self.reranker:
      docs = self.reranker.rerank(q, docs)
      
    context = "\n\n".join(d.text for d in docs)
    prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {q.text}"
    answer = self.llm.generate(prompt)
    return {"answer": answer, "docs": docs}
