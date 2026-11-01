FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .
ENV RETRIEVAL_DATA_DIR=/data RETRIEVAL_DEVICE=cpu PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "python -c \"import uvicorn; from retrieval_service.api import create_app; from retrieval_service.corpus import load_documents; from retrieval_service.retriever import Retriever; from retrieval_service.settings import Settings; s = Settings.from_environment(); uvicorn.run(create_app(Retriever(load_documents(s.data_dir / 'corpus.parquet'), s.device)), host='0.0.0.0', port=8080)\""]
