"""
Evaluate three systems against persona ground truth:

  1. Popularity baseline      (non-personalized, no compute)
  2. Retrieval only           (embeddings + cosine top-10, no LLM)
  3. Retrieval + LLM rerank   (cosine top-30 -> qwen2.5:3b picks top-10)

Plus two DIAGNOSTIC reference rows (not product candidates -- they exist to
tell us whether the benchmark itself is discriminating):

  - Random: expected score of 10 random articles from the pool
  - Category filter + popularity: no ML at all; keep pool articles whose MIND
    category is one of the persona's categories, rank them by click count.
    Because ground truth is defined via category membership, this is close
    to an oracle. If it matches or beats the ML systems, the benchmark can't
    tell "semantic understanding" apart from "category lookup".

Also runs: paired bootstrap CIs for every pairwise comparison, failure-mode
checks, feed-length checks, and a persona-overlap check (the bootstrap treats
personas as independent; overlapping personas make that optimistic).

Usage (run run_retrieval_only.py first):
    python src/evaluate.py
"""
import os
import json
import itertools

import numpy as np
import pandas as pd

from metrics import precision_at_k, ndcg_at_k

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "personas.json")

K = 10
REPETITION_SUBCAT_THRESHOLD = 0.5   # >50% of top-10 sharing one subcategory
POPULARITY_OVERLAP_THRESHOLD = 0.7  # >70% overlap with global popularity feed = not adding value
N_BOOT = 10000
N_RANDOM_DRAWS = 1000
SEED = 42
PERSONA_OVERLAP_JACCARD = 0.5       # flag persona pairs whose ground-truth sets overlap this much


def load_json(path):
    with open(path) as f:
        return json.load(f)


def failure_mode_checks(ranked_ids, id_to_subcat, global_pop_top10):
    subcats = [id_to_subcat.get(i, "unknown") for i in ranked_ids]
    top_subcat_share = (
        max(subcats.count(s) for s in set(subcats)) / len(subcats) if subcats else 0
    )
    overlap_with_pop = len(set(ranked_ids) & set(global_pop_top10)) / max(len(ranked_ids), 1)
    return {
        "over_specialized": top_subcat_share > REPETITION_SUBCAT_THRESHOLD,
        "top_subcategory_share": round(top_subcat_share, 2),
        "popularity_bias": overlap_with_pop > POPULARITY_OVERLAP_THRESHOLD,
        "overlap_with_global_popularity": round(overlap_with_pop, 2),
        "has_duplicate_ids": len(set(ranked_ids)) != len(ranked_ids),
    }


def paired_bootstrap(diffs, n_boot=N_BOOT, seed=SEED):
    """Mean of per-persona differences with a 95% percentile bootstrap CI."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def expected_random_scores(pool_ids, relevant, rng):
    ps, ns = [], []
    for _ in range(N_RANDOM_DRAWS):
        draw = list(rng.choice(pool_ids, size=K, replace=False))
        ps.append(precision_at_k(draw, relevant, K))
        ns.append(ndcg_at_k(draw, relevant, K))
    return float(np.mean(ps)), float(np.mean(ns))


def overlapping_persona_pairs(ground_truth, persona_ids):
    pairs = []
    for a, b in itertools.combinations(persona_ids, 2):
        A, B = set(ground_truth.get(a, [])), set(ground_truth.get(b, []))
        if not A or not B:
            continue
        jac = len(A & B) / len(A | B)
        if jac >= PERSONA_OVERLAP_JACCARD:
            pairs.append((a, b, jac))
    return pairs


def verdict(lo, hi):
    if lo > 0:
        return "CI excludes zero -> first system reliably better"
    if hi < 0:
        return "CI excludes zero -> first system reliably WORSE"
    return "CI includes zero -> no reliable difference at this sample size"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    retrieval_feed_path = os.path.join(PROCESSED_DIR, "retrieval_feed.json")
    if not os.path.exists(retrieval_feed_path):
        raise SystemExit("Missing retrieval_feed.json -- run `python src/run_retrieval_only.py` first.")

    ground_truth = load_json(os.path.join(PROCESSED_DIR, "persona_ground_truth.json"))
    baseline = load_json(os.path.join(PROCESSED_DIR, "baseline_feed.json"))
    rerank_feed = load_json(os.path.join(PROCESSED_DIR, "rerank_feed.json"))
    rerank_log = load_json(os.path.join(PROCESSED_DIR, "rerank_log.json"))
    retrieval_feed = load_json(retrieval_feed_path)
    retrieval_log = load_json(os.path.join(PROCESSED_DIR, "retrieval_log.json"))
    persona_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "persona_stats.csv"))
    persona_config = load_json(CONFIG_PATH)["personas"]
    pool_df = pd.read_csv(os.path.join(PROCESSED_DIR, "content_pool.csv"))

    id_to_subcat = dict(zip(pool_df["news_id"], pool_df["subcategory"]))
    id_to_cat = dict(zip(pool_df["news_id"], pool_df["category"]))
    pool_ids = np.array(pool_df["news_id"].tolist())
    pool_id_set = set(pool_ids)
    global_top_10 = baseline["global_top_10"]
    full_popularity_ranking = baseline["full_ranking"]

    sufficient_ids = set(persona_stats[persona_stats["sufficient_data"]]["persona_id"])
    rng = np.random.default_rng(SEED)

    rows = []
    short_feeds = []
    for persona in persona_config:
        pid = persona["id"]
        relevant = set(ground_truth.get(pid, []))
        if not relevant:
            continue

        feeds = {
            "baseline": global_top_10,                      # same feed for everyone
            "retrieval": retrieval_feed.get(pid, []),
            "rerank": rerank_feed.get(pid, []),
            "catfilter": [i for i in full_popularity_ranking
                          if id_to_cat.get(i) in set(persona["categories"])][:K],
        }
        for name, feed in feeds.items():
            if len(feed) != K:
                short_feeds.append((pid, name, len(feed)))

        r_lat = retrieval_log.get(pid, {}).get("retrieval_latency_seconds", np.nan)
        llm_lat = rerank_log.get(pid, {}).get("rerank_latency_seconds", np.nan)

        row = {
            "persona_id": pid,
            "persona_name": persona["name"],
            "sufficient_ground_truth": pid in sufficient_ids,
            "n_relevant": len(relevant),
            "n_relevant_in_pool": len(relevant & pool_id_set),
        }
        for name, feed in feeds.items():
            row[f"{name}_precision_at_10"] = precision_at_k(feed, relevant, K)
            row[f"{name}_ndcg_at_10"] = ndcg_at_k(feed, relevant, K)
        row["random_precision_at_10"], row["random_ndcg_at_10"] = expected_random_scores(
            pool_ids, relevant, rng)

        row["retrieval_latency_seconds"] = r_lat
        row["rerank_llm_latency_seconds"] = llm_lat
        # Total cost of the rerank system = stage 1 retrieval + the LLM call.
        row["rerank_total_latency_seconds"] = r_lat + llm_lat
        row["rerank_used_fallback"] = bool(rerank_log.get(pid, {}).get("used_fallback", False))
        row["rerank_n_padded"] = int(rerank_log.get(pid, {}).get("n_padded_from_retrieval", 0))
        row["rerank_overlap_with_retrieval_top10"] = round(
            len(set(feeds["rerank"]) & set(feeds["retrieval"])) / K, 2)

        for sys_name in ("retrieval", "rerank"):
            checks = failure_mode_checks(feeds[sys_name], id_to_subcat, global_top_10)
            row.update({f"{sys_name}_{k}": v for k, v in checks.items()})
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)

    reliable = results_df[results_df["sufficient_ground_truth"]]
    n = len(reliable)

    # ------------------------------------------------------------------ per persona
    print("=" * 78)
    print(f"PER-PERSONA Precision@10 (personas with sufficient ground truth: {n})")
    print("=" * 78)
    print(reliable[[
        "persona_name", "n_relevant_in_pool",
        "baseline_precision_at_10", "retrieval_precision_at_10", "rerank_precision_at_10",
        "catfilter_precision_at_10",
    ]].rename(columns={
        "baseline_precision_at_10": "popularity", "retrieval_precision_at_10": "retrieval",
        "rerank_precision_at_10": "rerank", "catfilter_precision_at_10": "catfilter(diag)",
        "n_relevant_in_pool": "rel_in_pool",
    }).to_string(index=False))

    if short_feeds:
        print("\nWARNING: some feeds do not have exactly 10 items. precision_at_k divides by the number of "
              "items returned, so short feeds are scored on fewer slots:")
        for pid, name, ln in short_feeds:
            print(f"   {pid} {name}: {ln} items")

    if n == 0:
        print("No personas with sufficient ground truth; nothing to summarize.")
        return

    # ------------------------------------------------------------------ aggregate
    systems = [
        ("Popularity baseline", "baseline", 0.0),
        ("Retrieval only", "retrieval", reliable["retrieval_latency_seconds"].mean()),
        ("Retrieval + LLM rerank", "rerank", reliable["rerank_total_latency_seconds"].mean()),
    ]
    diagnostics = [
        ("[diag] Random (expected)", "random", 0.0),
        ("[diag] Category filter + popularity", "catfilter", 0.0),
    ]
    print("\n" + "=" * 78)
    print("AGGREGATE (mean across personas)")
    print("=" * 78)
    print(f"{'System':38s} {'P@10':>7s} {'NDCG@10':>9s} {'Latency/request':>18s}")
    agg = {}
    for label, key, lat in systems + diagnostics:
        p = reliable[f"{key}_precision_at_10"].mean()
        nd = reliable[f"{key}_ndcg_at_10"].mean()
        agg[key] = (p, nd)
        lat_str = "~0" if lat == 0.0 else (f"{lat * 1000:.1f} ms" if lat < 1 else f"{lat:.2f} s")
        print(f"{label:38s} {p:7.3f} {nd:9.3f} {lat_str:>18s}")

    base_p, retr_p, rer_p = agg["baseline"][0], agg["retrieval"][0], agg["rerank"][0]
    print()
    if base_p > 0:
        print(f"Relative P@10 lift over popularity:  retrieval-only {(retr_p - base_p) / base_p:+.0%}   "
              f"rerank {(rer_p - base_p) / base_p:+.0%}")
    if rer_p - base_p > 0:
        share = (retr_p - base_p) / (rer_p - base_p)
        print(f"Share of the rerank system's lift that retrieval-only already captures: {share:.0%}"
              + ("  (>100% means retrieval-only scored HIGHER than rerank)" if share > 1 else ""))
    llm_s = reliable["rerank_llm_latency_seconds"].mean()
    print(f"LLM step adds ~{llm_s:.1f}s/request on top of ~{reliable['retrieval_latency_seconds'].mean() * 1000:.1f} ms "
          f"of retrieval.  Rerank fallback rate: {reliable['rerank_used_fallback'].mean():.0%}")
    n_pad = int((reliable["rerank_n_padded"] > 0).sum())
    print(f"Rerank lists the LLM left short and we padded from retrieval order: {n_pad}/{n} personas "
          f"({int(reliable['rerank_n_padded'].sum())} slots total)")
    print(f"Rerank top-10 overlap with retrieval top-10 (avg): "
          f"{reliable['rerank_overlap_with_retrieval_top10'].mean():.0%}  "
          "(low overlap = the LLM is genuinely reshuffling; high = it mostly confirms retrieval)")

    # ------------------------------------------------------------------ pairwise stats
    comparisons = [
        ("Retrieval-only vs Popularity", "retrieval", "baseline"),
        ("Rerank vs Popularity", "rerank", "baseline"),
        ("Rerank vs Retrieval-only  (what the LLM adds)", "rerank", "retrieval"),
        ("[diag] Rerank vs Category filter", "rerank", "catfilter"),
    ]
    pair_rows = []
    print("\n" + "=" * 78)
    print(f"PAIRWISE COMPARISONS (paired bootstrap, {N_BOOT:,} resamples over {n} personas)")
    print("=" * 78)
    for metric, metric_label in (("precision_at_10", "Precision@10"), ("ndcg_at_10", "NDCG@10")):
        print(f"\n--- {metric_label} ---")
        for label, a, b in comparisons:
            diffs = (reliable[f"{a}_{metric}"] - reliable[f"{b}_{metric}"]).to_numpy()
            mean_d, lo, hi = paired_bootstrap(diffs)
            wins, ties, losses = int((diffs > 0).sum()), int((diffs == 0).sum()), int((diffs < 0).sum())
            print(f"{label}")
            print(f"    mean diff {mean_d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
                  f"W/T/L = {wins}/{ties}/{losses}   -> {verdict(lo, hi)}")
            pair_rows.append({
                "metric": metric_label, "comparison": label, "mean_diff": round(mean_d, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "wins": wins, "ties": ties, "losses": losses,
            })
    pd.DataFrame(pair_rows).to_csv(os.path.join(RESULTS_DIR, "pairwise_comparisons.csv"), index=False)

    # ------------------------------------------------------------------ effective sample size
    pairs = overlapping_persona_pairs(ground_truth, reliable["persona_id"].tolist())
    print("\n" + "=" * 78)
    print("PERSONA OVERLAP CHECK (the bootstrap assumes personas are independent)")
    print("=" * 78)
    if pairs:
        names = dict(zip(results_df["persona_id"], results_df["persona_name"]))
        for a, b, jac in pairs:
            print(f"   {a} {names[a]}  <->  {b} {names[b]}: ground-truth Jaccard {jac:.2f}")
        print(f"   -> {len(pairs)} overlapping pair(s): the effective sample is smaller than {n}, "
              "so the CIs above are somewhat optimistic.")
    else:
        print("   No persona pairs with heavily overlapping ground truth.")

    ceilings = reliable[reliable["n_relevant_in_pool"] < K]
    if len(ceilings):
        print("\nPersonas with fewer than 10 relevant articles in the pool (P@10 is capped for them):")
        print(ceilings[["persona_name", "n_relevant_in_pool"]].to_string(index=False))

    # ------------------------------------------------------------------ failure modes
    print("\n" + "=" * 78)
    print("FAILURE-MODE CHECKS")
    print("=" * 78)
    for sys_name in ("retrieval", "rerank"):
        osp = reliable[reliable[f"{sys_name}_over_specialized"]]
        pb = reliable[reliable[f"{sys_name}_popularity_bias"]]
        dup = reliable[reliable[f"{sys_name}_has_duplicate_ids"]]
        print(f"{sys_name:10s} over-specialized: {len(osp)}/{n}   popularity-biased: {len(pb)}/{n}   "
              f"duplicate IDs: {len(dup)}/{n}")
        if len(osp):
            print(f"           over-specialized personas: {osp['persona_name'].tolist()}")

    print(f"\nFull results saved to: {results_path}")
    print(f"Pairwise stats saved to: {os.path.join(RESULTS_DIR, 'pairwise_comparisons.csv')}")


if __name__ == "__main__":
    main()
