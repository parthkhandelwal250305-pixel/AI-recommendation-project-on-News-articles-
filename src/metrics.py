"""
Precision@K and NDCG@K, defined once so every system is scored identically.
"""
import math


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int = 10) -> float:
    """Fraction of the top-k recommended articles that are in the relevant set."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(top_k)


def dcg_at_k(ranked_ids: list, relevant_ids: set, k: int = 10) -> float:
    dcg = 0.0
    for i, article_id in enumerate(ranked_ids[:k]):
        rel = 1.0 if article_id in relevant_ids else 0.0
        # rank position is 1-indexed inside the log
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids: list, relevant_ids: set, k: int = 10) -> float:
    """
    Binary relevance NDCG@K. Ideal DCG assumes the best case: the first
    min(k, |relevant|) positions are all hits.
    """
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg
