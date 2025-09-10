from typing import Any, Callable
# from .base import BM25Config
from .whoosh import WhooshBM25Retriever
from .pyserini import PyseriniBM25Retriever
from .elastic import ElasticBM25Retriever
from .inmem import InMemoryBM25Retriever

_REGISTRY: dict[str, Callable[..., Any]] = {
    "whoosh": WhooshBM25Retriever,
    "pyserini": PyseriniBM25Retriever,
    "elastic": ElasticBM25Retriever,
    "inmem": InMemoryBM25Retriever
}

def create_bm25(backend: str, **kwargs: Any):
    """
    Factory for BM25 retrievers.
    Example:
        create_bm25("pyserini", index_dir="...", cfg=BM25Config(k1=1.2, b=0.75))
    """
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown BM25 backend '{backend}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[backend](**kwargs)
