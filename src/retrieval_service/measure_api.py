import json
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from retrieval_service.api import create_app
from retrieval_service.corpus import load_documents, load_queries
from retrieval_service.retriever import Retriever

REQUESTS = 100


def first_event_and_total(client: TestClient, query: str) -> tuple[float, float]:
    started = time.perf_counter()
    first = 0.0
    with client.stream("GET", "/search/stream", params={"q": query, "k": 5}) as response:
        for line in response.iter_lines():
            if line.startswith("event:") and not first:
                first = (time.perf_counter() - started) * 1000
    return first, (time.perf_counter() - started) * 1000


def main(data_dir: Path, output: Path, device: str) -> None:
    documents = load_documents(data_dir / "corpus.parquet")
    queries, _ = load_queries(data_dir / "queries.parquet", data_dir / "test.tsv")
    client = TestClient(create_app(Retriever(documents, device)))
    measured = [first_event_and_total(client, text) for text in list(queries.values())[:REQUESTS]]
    firsts, totals = [m[0] for m in measured], [m[1] for m in measured]
    report = {
        "requests": len(measured),
        "first_event_ms_p50": statistics.median(firsts),
        "total_ms_p50": statistics.median(totals),
        "total_ms_p95": sorted(totals)[int(0.95 * len(totals))],
    }
    output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "cpu")
