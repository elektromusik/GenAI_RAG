"""
Quick smoke test for the RAGPipeline with an InMemoryBM25Retriever and a FakeLLM. Run with:

python scripts/repl_test.py
"""

import argparse
from rag_core.interfaces import Query, Doc
from rag_core.pipeline import RAGPipeline
from rag_core.retrievers.bm25 import create_bm25 # Our factory.


class FakeLLM:
    """A dummy LLM that just echoes the first part of the prompt."""
    def generate(self, prompt: str, **opts) -> str:
        return "[FAKE ANSWER] " + prompt[:80]


def main():
    parser = argparse.ArgumentParser(description="Smoke test for RAGPipeline.")
    parser.add_argument(
        "--backend",
        choices=["inmem", "whoosh", "pyserini", "elastic"],
        default="inmem",
        help="Which BM25 backend to use.")
    parser.add_argument("--index-dir", help="Path to index directory (for whoosh/pyserini).")
    parser.add_argument("--host", help="Elasticsearch host URL.")
    parser.add_argument("--index-name", help="Elasticsearch index name.")
    args = parser.parse_args()

    # Build retriever depending on backend
    if args.backend == "inmem":
        # Minimal corpus
        corpus = {
            "d1": "the quick brown fox jumps over the lazy dog",
            "d2": "slow turtle wins the race",
            "d3": "the green light in Gatsby is a powerful symbol of hope"}
        retriever = create_bm25("inmem", corpus=corpus)

    elif args.backend == "whoosh":
        if not args.index_dir:
            raise ValueError("Need --index-dir for whoosh backend.")
        retriever = create_bm25("whoosh", index_dir=args.index_dir)

    elif args.backend == "pyserini":
        if not args.index_dir:
            raise ValueError("Need --index-dir for pyserini backend.")
        retriever = create_bm25("pyserini", index_dir=args.index_dir)

    elif args.backend == "elastic":
        if not (args.host and args.index_name):
            raise ValueError("Need --host and --index-name for elastic backend.")
        retriever = create_bm25("elastic", host=args.host, index_name=args.index_name)

    pipe = RAGPipeline(retriever=retriever, llm=FakeLLM())

    query = Query("green light", k=2)
    result = pipe.run(query)

    print("\n____ Pipeline Result ____")
    print("Answer:")
    print(result["answer"])
    print("\nRetrieved docs:")
    for doc in result["docs"]:
        assert isinstance(doc, Doc)
        print(f"- {doc.id}: {doc.text!r} (score={doc.metadata['score']:.4f})")


if __name__ == "__main__":
    main()
