"""
System 2: retrieval-only. Same embeddings, same persona query, same cosine
ranking as stage 1 of run_retrieval_rerank.py -- but we take the top 10
straight from the cosine ranking. No LLM call.

This is the missing control in the experiment: it isolates how much of the
rerank system's lift comes from semantic retrieval versus the LLM step.

It also measures retrieval latency (query embedding + cosine ranking), so
the final comparison can report cost for every system on the same footing.

Usage:
    python src/run_retrieval_only.py
"""
import os
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "personas.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match build_embeddings.py
TOP_K = 10
N_TIMING_REPEATS = 5  # latency = median of this many runs (after one warm-up)


def persona_query(persona: dict) -> str:
    # MUST stay identical to the query built in run_retrieval_rerank.py main().
    # If you ever change one, change both -- the sanity check below will
    # flag it if they drift apart.
    # Category labels are deliberately NOT included (they define ground truth).
    return f"{persona['name']}. {persona['description']}"


def rank_articles(model, query_text, article_embs, article_ids, k):
    query_emb = model.encode(query_text, normalize_embeddings=True)
    sims = article_embs @ query_emb  # embeddings are normalized -> cosine similarity
    top_idx = np.argsort(-sims)[:k]
    return [str(article_ids[i]) for i in top_idx]


def main():
    article_embs = np.load(os.path.join(PROCESSED_DIR, "article_embeddings.npy"))
    article_ids = np.load(os.path.join(PROCESSED_DIR, "article_ids.npy"), allow_pickle=True)

    with open(CONFIG_PATH) as f:
        personas = json.load(f)["personas"]

    # Optional sanity check against the rerank run's stage-1 candidates.
    rerank_log_path = os.path.join(PROCESSED_DIR, "rerank_log.json")
    rerank_log = {}
    if os.path.exists(rerank_log_path):
        with open(rerank_log_path) as f:
            rerank_log = json.load(f)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    model.encode("warm-up", normalize_embeddings=True)  # first call is slow; don't time it

    feed = {}
    log = {}
    mismatches = 0

    for persona in personas:
        pid = persona["id"]
        query_text = persona_query(persona)

        timings = []
        top = None
        for _ in range(N_TIMING_REPEATS):
            t0 = time.perf_counter()
            top = rank_articles(model, query_text, article_embs, article_ids, TOP_K)
            timings.append(time.perf_counter() - t0)
        latency = float(np.median(timings))

        feed[pid] = top
        entry = {"retrieval_latency_seconds": round(latency, 4)}

        logged = rerank_log.get(pid, {}).get("retrieval_candidates")
        if logged is not None:
            same = [str(x) for x in logged[:TOP_K]] == top
            entry["matches_rerank_stage1_top10"] = same
            if not same:
                mismatches += 1
                print(f"[{pid}] WARNING: retrieval top-10 differs from the stage-1 candidates "
                      f"logged by run_retrieval_rerank.py (query text or embeddings changed?)")
        log[pid] = entry
        print(f"[{pid}] {persona['name']}: top {TOP_K} in {latency * 1000:.1f} ms")

    with open(os.path.join(PROCESSED_DIR, "retrieval_feed.json"), "w") as f:
        json.dump(feed, f, indent=2)
    with open(os.path.join(PROCESSED_DIR, "retrieval_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    if rerank_log:
        if mismatches == 0:
            print("\nSanity check OK: retrieval-only top-10 equals stage 1 of the rerank pipeline "
                  "for every persona (same query, same embeddings).")
        else:
            print(f"\nSanity check: {mismatches}/{len(personas)} personas differ from the logged "
                  "stage-1 candidates. Do NOT compare systems until this is explained.")

    print(f"\nSaved: {os.path.join(PROCESSED_DIR, 'retrieval_feed.json')}")
    print(f"Saved: {os.path.join(PROCESSED_DIR, 'retrieval_log.json')}")


if __name__ == "__main__":
    main()
