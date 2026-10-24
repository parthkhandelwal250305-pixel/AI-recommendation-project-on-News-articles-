"""
Popularity baseline: one global top-10 feed, same for every persona.

Popularity = number of times an article was clicked across all training
impressions. This is intentionally the simplest thing that could work --
the number every other system has to beat.

Usage:
    python src/run_baseline.py
"""
import os
import json
from collections import Counter

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def clicked_ids_from_impressions(impressions: str) -> list:
    if not isinstance(impressions, str):
        return []
    clicked = []
    for token in impressions.strip().split():
        if "-" in token:
            nid, label = token.rsplit("-", 1)
            if label == "1":
                clicked.append(nid)
    return clicked


def main():
    train_dir = os.path.join(RAW_DIR, "MINDsmall_train")
    behaviors_path = os.path.join(train_dir, "behaviors.tsv")
    behaviors_df = pd.read_csv(
        behaviors_path, sep="\t", header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )

    pool_df = pd.read_csv(os.path.join(PROCESSED_DIR, "content_pool.csv"))
    pool_ids = set(pool_df["news_id"])

    click_counts = Counter()
    for impressions in behaviors_df["impressions"]:
        for nid in clicked_ids_from_impressions(impressions):
            if nid in pool_ids:
                click_counts[nid] += 1

    # Articles in the pool with zero observed clicks still need a defined
    # (last-place) rank so retrieval isn't silently dropping them.
    ranked = [nid for nid, _ in click_counts.most_common()]
    unclicked = [nid for nid in pool_df["news_id"] if nid not in click_counts]
    ranked.extend(unclicked)

    top_10_global = ranked[:10]

    out_path = os.path.join(PROCESSED_DIR, "baseline_feed.json")
    with open(out_path, "w") as f:
        json.dump({"global_top_10": top_10_global, "full_ranking": ranked}, f, indent=2)

    print(f"Global popularity top 10: {top_10_global}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
