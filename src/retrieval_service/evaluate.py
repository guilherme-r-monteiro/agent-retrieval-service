import json
import statistics
import sys
import time
from pathlib import Path

from retrieval_service.corpus import load_documents, load_queries
from retrieval_service.hybrid import Mode
from retrieval_service.metrics import ndcg_at, recall_at
from retrieval_service.retriever import Retriever

DEPTH = 100


def evaluate_mode(retriever: Retriever, queries: dict[str, str], relevant: dict[str, set[str]], mode: Mode) -> dict[str, float]:
    ndcg, recall10, recall100, latencies = [], [], [], []
    for query_id, text in queries.items():
        started = time.perf_counter()
        ranked = [d.doc_id for d in retriever.search(text, DEPTH, mode)]
        latencies.append((time.perf_counter() - started) * 1000)
        ndcg.append(ndcg_at(ranked, relevant[query_id], 10))
        recall10.append(recall_at(ranked, relevant[query_id], 10))
        recall100.append(recall_at(ranked, relevant[query_id], 100))
    return {
        "ndcg_at_10": statistics.mean(ndcg),
        "recall_at_10": statistics.mean(recall10),
        "recall_at_100": statistics.mean(recall100),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(0.95 * len(latencies))],
    }


def main(data_dir: Path, output: Path, device: str) -> None:
    documents = load_documents(data_dir / "corpus.parquet")
    queries, relevant = load_queries(data_dir / "queries.parquet", data_dir / "test.tsv")
    started = time.perf_counter()
    retriever = Retriever(documents, device)
    report = {
        "documents": len(documents),
        "queries": len(queries),
        "index_build_seconds": time.perf_counter() - started,
        "modes": {mode.value: evaluate_mode(retriever, queries, relevant, mode) for mode in Mode},
    }
    output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps(report["modes"]) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "cpu")
