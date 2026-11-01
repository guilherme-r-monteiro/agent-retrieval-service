import re
from collections import Counter

import numpy as np
from scipy import sparse

K1 = 1.5
B = 0.75
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an the of in on at to for and or is are was were be by with as that this from it its".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Bm25Index:
    def __init__(self, texts: list[str]) -> None:
        self.vocabulary: dict[str, int] = {}
        rows, columns, values = [], [], []
        lengths = np.zeros(len(texts))
        for doc_index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[doc_index] = sum(counts.values())
            for term, count in counts.items():
                rows.append(doc_index)
                columns.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                values.append(count)
        frequencies = sparse.csr_matrix((values, (rows, columns)), shape=(len(texts), len(self.vocabulary)), dtype=np.float64)
        document_frequency = np.asarray((frequencies > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (len(texts) - document_frequency + 0.5) / (document_frequency + 0.5))
        norm = K1 * (1 - B + B * lengths / lengths.mean())
        frequencies = frequencies.tocoo()
        weights = frequencies.data * (K1 + 1) / (frequencies.data + norm[frequencies.row])
        self.matrix = sparse.csc_matrix((weights, (frequencies.row, frequencies.col)), shape=frequencies.shape)

    def scores(self, query: str) -> np.ndarray:
        terms = [self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary]
        if not terms:
            return np.zeros(self.matrix.shape[0])
        return np.asarray(self.matrix[:, terms].multiply(self.idf[terms]).sum(axis=1)).ravel()

    def top(self, query: str, k: int) -> list[int]:
        scores = self.scores(query)
        order = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [int(i) for i in order[np.argsort(-scores[order])]]
