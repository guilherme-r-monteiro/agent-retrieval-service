from retrieval_service.bm25 import Bm25Index
from retrieval_service.corpus import Document
from retrieval_service.dense import DenseIndex
from retrieval_service.hybrid import CANDIDATES, Mode, reciprocal_rank_fusion


class Retriever:
    def __init__(self, documents: list[Document], device: str) -> None:
        self.documents = documents
        texts = [d.body for d in documents]
        self.bm25 = Bm25Index(texts)
        self.dense = DenseIndex(texts, device)

    def search(self, query: str, k: int, mode: Mode) -> list[Document]:
        if mode is Mode.BM25:
            ranking = self.bm25.top(query, k)
        elif mode is Mode.DENSE:
            ranking = self.dense.top(query, k)
        else:
            ranking = reciprocal_rank_fusion([self.bm25.top(query, CANDIDATES), self.dense.top(query, CANDIDATES)], k)
        return [self.documents[i] for i in ranking]
