from enum import StrEnum

RRF_CONSTANT = 60
CANDIDATES = 100


class Mode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_CONSTANT + rank + 1)
    return sorted(scores, key=lambda doc: -scores[doc])[:k]
