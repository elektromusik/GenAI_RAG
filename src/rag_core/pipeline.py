from dataclass import dataclass
from .interfaces import Query, Retriever Reranker, LLMClient, Doc

@dataclass
class RAGPipeline:
  retriever: Retriever
  llm: LLMClient
  reranker: Reranker | None = None
  system_prompt: str = "Answer using only the provided context."

  def run(self, q: Query) -> dict:
    docs = self.retriever.retrieve(q)
    if self.reranker:
      docs = self.reranker.rerank(q, docs)
    context = "\n\n".join(d.text for d in docs)
    prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {q.text}"
    answer = self.llm.generate(prompt)
    return {"answer": answer, "docs": docs}
