from whoosh import index
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from ...interfaces import Retriever, Query, Doc
from .base import BM25Config, to_doc


class WhooshBM25Retriever(Retriever):
    def __init__(self, index_dir: str, cfg: BM25Config | None = None):
        self.ix  = index.open_dir(index_dir)
        self.cfg = cfg or BM25Config()

    def retrieve(self, q: Query) -> list[Doc]:
        # Whoosh-BM25F: Apply BM25F weighting with k1/b from config
        with self.ix.searcher(weighting=BM25F(B=self.cfg.b, K1=self.cfg.k1)) as searcher:
            fields  = self.cfg.fields or ["contents"]
            parser  = MultifieldParser(fields, schema=self.ix.schema)
            query   = parser.parse(q.text)
            results = searcher.search(query, limit=q.k)
            return [to_doc(hit["id"], hit.get("contents", ""), hit.score) for hit in results]
