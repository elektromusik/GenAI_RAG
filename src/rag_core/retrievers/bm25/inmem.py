from ...interfaces import Retriever, Query, Doc
from .base import to_doc
from ._inmem_engine import InMemoryBM25


class InMemoryBM25Retriever(Retriever):
    def __init__(self, corpus: dict[str, str]):
        self.engine = InMemoryBM25(corpus)

    def retrieve(self, q: Query) -> list[Doc]:
        ranked = self.engine.search(q.text, k=q.k)
        return [to_doc(doc_id, 
                       self.engine.docs[doc_id], score) 
                for doc_id, score in ranked]
