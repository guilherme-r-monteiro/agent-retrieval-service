# Design note

Target: BairesDev, Senior Agentic AI Engineer. The posting asks for Python (FastAPI) services, custom MCP servers, streaming APIs (Server-Sent Events), routing and evaluation suites for agents, containers on GCP Cloud Run, CI/CD with Cloud Build, Terraform, and, as a plus, vector databases with hybrid retrieval.

## Question

Can one small service cover that chain end to end: a retrieval tool for agents, exposed over MCP and over a streaming API, evaluated against labelled data, with deployment code?

## Method

- Data: SciFact from the BEIR benchmark (5,183 abstracts, 300 test queries with relevance labels).
- Retrieval: BM25 written from scratch on sparse matrices, dense retrieval with all-MiniLM-L6-v2 and a FAISS inner-product index, and hybrid retrieval with reciprocal rank fusion (constant 60) over the top 100 of each.
- Interfaces: an MCP tool `search_papers` (FastMCP) and a FastAPI endpoint `/search/stream` that emits `start`, `hit` and `done` events over Server-Sent Events.
- Evaluation: nDCG@10, recall@10 and recall@100 for the three modes, plus latency, and a streaming latency measurement over 100 queries.
- Deployment code: Dockerfile, Cloud Build file, GitHub Actions workflow, and Terraform for Artifact Registry, a service account and a Cloud Run service. Terraform is validated only.

## Scope limits chosen before the run

- No LLM is called. The service is the retrieval tool an agent would use, not the agent.
- No Google Agent Development Kit, no OpenAI or Gemini calls, no React interface, no Backstage.
- Nothing is deployed to a cloud account.
- Sanity check: the BM25 result must land near the published BEIR BM25 value for SciFact (about 0.665 nDCG@10) before the other numbers are trusted.
