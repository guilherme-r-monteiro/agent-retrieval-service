import math


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    gains = sum(1.0 / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return gains / ideal if ideal else 0.0


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0
