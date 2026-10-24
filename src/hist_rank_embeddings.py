"""
Retrieval-only system for the history-conditioned evaluation.

User profile = mean embedding of the user's last N clicked articles
(title + abstract, all-MiniLM-L6-v2). Candidates are ranked by cosine
similarity to that profile. No LLM.

PRIMARY setting: last 20 clicks (fixed in advance). Other history lengths
(5, 10, 50) are computed too, but only as a sensitivity analysis.

Latency note: article embeddings are computed once at ingestion time in a real
system, so the per-request cost measured here is only "build user vector +
cosine over candidates". The one-off embedding throughput is reported separately.

Usage:
    python src/hist_rank_embeddings.py
    python src/hist_rank_embeddings.py --force     # recompute cached article embeddings
"""
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from hist_common import OUT_DIR, HISTORY_N, HISTORY_SENSITIVITY, article_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "eval_set.json")) as f:
        impressions = json.load(f)["impressions"]
    arts = pd.read_csv(os.path.join(OUT_DIR, "articles.csv"), keep_default_na=False)
    ids = arts["news_id"].tolist()
    id2i = {n: i for i, n in enumerate(ids)}

    emb_path = os.path.join(OUT_DIR, "article_embeddings.npy")
    ids_path = os.path.join(OUT_DIR, "article_ids.npy")
    meta_path = os.path.join(OUT_DIR, "embedding_meta.json")

    cached = (
        not args.force and os.path.exists(emb_path) and os.path.exists(ids_path)
        and os.path.exists(meta_path)
        and list(np.load(ids_path, allow_pickle=True)) == ids
    )
    if cached:
        print("Using cached article embeddings.")
        E = np.load(emb_path)
        throughput = json.load(open(meta_path))["articles_per_second"]
    else:
        texts = [article_text(t, a) for t, a in zip(arts["title"], arts["abstract"])]
        print(f"Loading {MODEL_NAME} and embedding {len(texts):,} articles ...")
        model = SentenceTransformer(MODEL_NAME)
        t0 = time.perf_counter()
        E = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        dt = time.perf_counter() - t0
        throughput = len(texts) / dt
        np.save(emb_path, E)
        np.save(ids_path, np.array(ids, dtype=object))
        json.dump({"articles_per_second": throughput}, open(meta_path, "w"))
        print(f"  embedded {len(texts):,} articles in {dt:.0f}s ({throughput:.0f}/s on this CPU)")

    scores = {}
    latencies = []
    for imp in impressions:
        hist_idx = np.array([id2i[h] for h in imp["history"]])
        cand_idx = np.array([id2i[c] for c in imp["candidates"]])
        per_variant = {}
        for N in HISTORY_SENSITIVITY:
            t0 = time.perf_counter()
            u = E[hist_idx[-N:]].mean(axis=0)
            u = u / (np.linalg.norm(u) + 1e-12)
            sims = E[cand_idx] @ u
            dt = time.perf_counter() - t0
            if N == HISTORY_N:
                latencies.append(dt)
            per_variant[f"N{N}"] = [round(float(x), 5) for x in sims]
        scores[imp["impression_id"]] = per_variant

    out = {
        "model": MODEL_NAME,
        "primary_history_n": HISTORY_N,
        "article_embedding_throughput_per_sec": throughput,
        "scoring_latency_ms_median": float(np.median(latencies) * 1000),
        "scoring_latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
        "scores": scores,
    }
    with open(os.path.join(OUT_DIR, "emb_scores.json"), "w") as f:
        json.dump(out, f)

    print(f"\nScored {len(impressions):,} cases. Per-request scoring latency (precomputed article embeddings): "
          f"median {out['scoring_latency_ms_median']:.2f} ms, p95 {out['scoring_latency_ms_p95']:.2f} ms")
    print(f"One-off embedding cost for new articles: ~{throughput:.0f} articles/second on this CPU")
    print(f"Saved: {os.path.join(OUT_DIR, 'emb_scores.json')}")


if __name__ == "__main__":
    main()
