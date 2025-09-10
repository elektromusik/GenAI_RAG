from pyserini.search.lucene import LuceneSearcher
from ...interfaces import Retriever, Query, Doc
from .base import BM25Config, to_doc


class PyseriniBM25Retriever(Retriever):
    def __init__(self, index_dir: str, cfg: BM25Config | None = None):
        self.cfg = cfg or BM25Config()
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(self.cfg.k1, self.cfg.b)

    def retrieve(self, q: Query) -> list[Doc]:
        hits = self.searcher.search(q.text, k=q.k)
        docs: list[Doc] = []
        for h in hits:
            raw = self.searcher.doc(h.docid).raw() or ""
            docs.append(to_doc(h.docid, raw, h.score))
        return docs 
