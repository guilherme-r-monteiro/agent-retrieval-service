import asyncio

from fastapi.testclient import TestClient

from retrieval_service.api import create_app
from retrieval_service.bm25 import Bm25Index
from retrieval_service.corpus import Document
from retrieval_service.hybrid import Mode, reciprocal_rank_fusion
from retrieval_service.mcp_server import build_server
from retrieval_service.metrics import ndcg_at, recall_at

DOCUMENTS = [
    Document("a", "Protein folding", "Proteins fold into stable shapes."),
    Document("b", "Vaccine trial", "A vaccine trial measured antibody response."),
    Document("c", "Graph theory", "Graphs have vertices and edges."),
]


class FakeSearcher:
    def search(self, query: str, k: int, mode: Mode) -> list[Document]:
        return DOCUMENTS[:k]


def test_bm25_ranks_matching_document_first() -> None:
    index = Bm25Index([d.body for d in DOCUMENTS])
    assert index.top("vaccine antibody", 3)[0] == 1


def test_bm25_returns_zero_scores_for_unknown_terms() -> None:
    assert Bm25Index([d.body for d in DOCUMENTS]).scores("zzzz").sum() == 0


def test_rrf_prefers_documents_ranked_in_both_lists() -> None:
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]], 1) == [2]


def test_metrics_on_known_ranking() -> None:
    assert recall_at(["x", "a", "b"], {"a", "b"}, 2) == 0.5
    assert ndcg_at(["a"], {"a"}, 10) == 1.0 and ndcg_at(["x"], {"a"}, 10) == 0.0


def test_stream_emits_start_hits_and_done_in_order() -> None:
    client = TestClient(create_app(FakeSearcher()))
    text = client.get("/search/stream", params={"q": "vaccine", "k": 2}).text
    names = [line.split(": ")[1] for line in text.splitlines() if line.startswith("event:")]
    assert names == ["start", "hit", "hit", "done"]


def test_stream_rejects_empty_query_and_large_k() -> None:
    client = TestClient(create_app(FakeSearcher()))
    assert client.get("/search/stream", params={"q": ""}).status_code == 422
    assert client.get("/search/stream", params={"q": "x", "k": 500}).status_code == 422


def test_mcp_tool_is_registered_and_bounded() -> None:
    server = build_server(FakeSearcher())
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == ["search_papers"]
    result = asyncio.run(server.call_tool("search_papers", {"query": "x", "k": 2}))
    assert "Protein folding" in str(result)
