import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


class DenseIndex:
    def __init__(self, texts: list[str], device: str) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        embeddings = self.encode(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def top(self, query: str, k: int) -> list[int]:
        _, indexes = self.index.search(self.encode([query]), k)
        return [int(i) for i in indexes[0]]
