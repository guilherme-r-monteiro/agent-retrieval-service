from mcp.server.fastmcp import FastMCP

from retrieval_service.corpus import load_documents
from retrieval_service.hybrid import Mode
from retrieval_service.retriever import Retriever
from retrieval_service.settings import Settings

MAX_RESULTS = 20


def build_server(searcher) -> FastMCP:
    server = FastMCP("retrieval")

    @server.tool()
    def search_papers(query: str, k: int = 5, mode: str = "hybrid") -> list[dict[str, str]]:
        """Search scientific abstracts. mode is bm25, dense or hybrid. Returns id, title and a snippet. Use hybrid unless the query is a single exact term, then use bm25."""
        bounded = max(1, min(k, MAX_RESULTS))
        documents = searcher.search(query, bounded, Mode(mode))
        return [{"id": d.doc_id, "title": d.title, "snippet": d.text[:240]} for d in documents]

    return server


def main() -> None:
    settings = Settings.from_environment()
    documents = load_documents(settings.data_dir / "corpus.parquet")
    build_server(Retriever(documents, settings.device)).run()


if __name__ == "__main__":
    main()
