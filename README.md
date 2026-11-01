# agent-retrieval-service

Hybrid retrieval (BM25, dense, reciprocal rank fusion) exposed as an MCP tool and as a Server-Sent Events API, evaluated on SciFact, with a Dockerfile, a Cloud Build file, a CI workflow and validated Terraform for Cloud Run.

Design note: `docs/design.md`. Raw results: `results/retrieval-results.json`, `results/api-results.json`.

## Results

SciFact test set: 5,183 abstracts, 300 queries. The dense model runs on the GPU for these numbers.

| Mode | nDCG@10 | Recall@10 | Recall@100 | Latency p50 |
|---|---|---|---|---|
| BM25 (own implementation) | 0.666 | 0.782 | 0.886 | 0.15 ms |
| Dense (MiniLM, FAISS) | 0.648 | 0.788 | 0.925 | 2.9 ms |
| Hybrid (RRF) | 0.688 | 0.815 | 0.955 | 3.2 ms |

The BM25 value matches the published BEIR BM25 figure for SciFact (0.665), so the metric code and the BM25 index behave as expected.

Streaming API, 100 queries, in-process test client, hybrid mode, k = 5: first event p50 2.2 ms, whole response p50 9.5 ms, p95 18.1 ms. The first event is `start`, sent before the search runs, so it shows streaming order and not search speed.

Checks that ran:
- 7 unit tests: BM25 ranking, fusion, metrics, event order, input validation (empty query and k over 20 return HTTP 422), and the MCP tool list and call.
- `terraform init -backend=false`, `terraform validate` and `terraform fmt -check` succeed.
- The Docker image builds (1.83 GB, CPU PyTorch). I ran the container once with the data mounted. `/health` returned ok and `/search/stream` streamed hits for a vaccine query. The container was removed afterwards.

What the results say:
- Hybrid fusion gives +2.2 points nDCG@10 over BM25 and +4.0 over dense, and the largest gain is in recall@100 (0.955). This helps a re-ranker or an agent that reads many candidates.
- The dense model alone is below BM25 on nDCG@10. A scientific-claim corpus rewards exact terms.

## Limits

- One dataset, one embedding model, one run. Differences of one to two points between modes are small relative to 300 queries.
- No LLM is in the loop. This service shows the retrieval tool, the interfaces and the delivery path. It does not show agent orchestration, model routing, Google ADK, OpenAI or Gemini integration, a React interface or Backstage publishing.
- Terraform was validated and never applied. No cloud account was used, so Cloud Run, Cloud Build and Artifact Registry behaviour is untested.
- The container indexes 5,183 abstracts at start-up on the CPU. A real deployment would build the index ahead of time.
- The service has no authentication. Add it before any public deployment.
- Latency is measured in-process on a laptop, not over a network.

## Run it

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e . pytest
B=https://huggingface.co/datasets/BeIR
curl -L -o data/corpus.parquet $B/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet
curl -L -o data/queries.parquet $B/scifact/resolve/main/queries/queries-00000-of-00001.parquet
curl -L -o data/test.tsv $B/scifact-qrels/resolve/main/test.tsv
.venv/bin/python -m pytest
.venv/bin/python -m retrieval_service.evaluate data results/retrieval-results.json cpu
.venv/bin/python -m retrieval_service.measure_api data results/api-results.json cpu
RETRIEVAL_DATA_DIR=data .venv/bin/python -m retrieval_service.mcp_server
```
