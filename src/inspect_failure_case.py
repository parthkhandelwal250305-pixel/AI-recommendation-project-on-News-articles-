"""
Inspect a specific persona's baseline vs. reranked feed side by side,
with titles and (sub)categories, to understand a failure case concretely
instead of just looking at the aggregate number.

Usage:
    python src/inspect_failure_case.py p09
"""
import sys
import json
import os

import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    persona_id = sys.argv[1] if len(sys.argv) > 1 else "p09"

    pool_df = pd.read_csv(os.path.join(PROCESSED_DIR, "content_pool.csv"))
    id_to_row = pool_df.set_index("news_id")

    with open(os.path.join(PROCESSED_DIR, "baseline_feed.json")) as f:
        baseline = json.load(f)["global_top_10"]
    with open(os.path.join(PROCESSED_DIR, "rerank_feed.json")) as f:
        rerank_feed = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "persona_ground_truth.json")) as f:
        ground_truth = json.load(f)

    relevant = set(ground_truth.get(persona_id, []))
    reranked = rerank_feed.get(persona_id, [])

    def describe(article_id):
        hit = "RELEVANT" if article_id in relevant else "not relevant"
        if article_id not in id_to_row.index:
            return f"{article_id:10s} [{hit:12s}] <not in pool>"
        row = id_to_row.loc[article_id]
        return f"{article_id:10s} [{hit:12s}] ({row['category']}/{row['subcategory']}) {row['title']}"

    print(f"=== Persona: {persona_id} ===\n")
    print(f"Ground truth size: {len(relevant)} relevant articles in pool\n")

    print("--- BASELINE top 10 (global popularity, same for everyone) ---")
    for aid in baseline:
        print(describe(aid))

    print("\n--- RERANKED top 10 (persona-conditioned) ---")
    for aid in reranked:
        print(describe(aid))


if __name__ == "__main__":
    main()
