from elasticsearch import Elasticsearch
from ...interfaces import Retriever, Query, Doc
from .base import BM25Config, to_doc


class ElasticBM25Retriever(Retriever):
    def __init__(self, host: str, index_name: str, cfg: BM25Config | None = None):
        self.es = Elasticsearch(hosts=[host])
        self.index_name= index_name
        self.cfg = cfg or BM25Config()

    def retrieve(self, q: Query) -> list[Doc]:
        fields = self.cfg.fields or ["contents"]
        
        # Simple match-query; for multi-field: "multi_match"
        body = {"size": q.k, 
                "query": {
                    "multi_match": {
                        "query": q.text, 
                        "fields": fields}}}
        
        res = self.es.search(index=self.index_name, body=body)
        out: list[Doc] = []
        for hit in res["hits"]["hits"]:
            src = hit.get("_source", {})
            out.append(
                to_doc(hit["_id"], 
                       src.get("contents", ""), 
                       hit["_score"]))
        return out
