import json
import time
from collections.abc import Iterator

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from retrieval_service.hybrid import Mode
from retrieval_service.protocol import Searcher

MAX_RESULTS = 20
SNIPPET_CHARACTERS = 240


def event(name: str, payload: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def stream_search(searcher: Searcher, query: str, k: int, mode: Mode) -> Iterator[str]:
    started = time.perf_counter()
    yield event("start", {"query": query, "mode": mode.value})
    documents = searcher.search(query, k, mode)
    for rank, document in enumerate(documents, start=1):
        yield event("hit", {"rank": rank, "id": document.doc_id, "title": document.title, "snippet": document.text[:SNIPPET_CHARACTERS]})
    yield event("done", {"count": len(documents), "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})


def create_app(searcher: Searcher) -> FastAPI:
    app = FastAPI(title="retrieval-service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search/stream")
    def search_stream(q: str = Query(min_length=1), k: int = Query(5, ge=1, le=MAX_RESULTS), mode: Mode = Mode.HYBRID) -> StreamingResponse:
        return StreamingResponse(stream_search(searcher, q, k, mode), media_type="text/event-stream")

    return app
