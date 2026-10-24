"""
Optional, short check for "drift": does a ranker built on OLDER data get worse as time passes?

Two plain questions, answered on the same 3,000 test readers:

 1. STALENESS. Popularity can be computed two ways:
      - "frozen": click rates counted only from the earlier training days (never updated)
      - "live"  : also updated as the test day goes on (what the main experiment uses)
    If live beats frozen, then what is popular changes quickly, and yesterday's data goes stale.
    We also report how much of today's feed is NEW (never seen in the training days).

 2. STABILITY OVER THE DAY. Split the test readers into the earliest, middle and latest third
    of the day. Does the blend's edge over popularity hold up across the day, or fade?

Limits: the test data covers ONE day, so this cannot show drift over weeks or months.

Usage (after hist_evaluate.py has been run once):
    python src/hist_drift_check.py
"""
import os
import json
from collections import Counter

import numpy as np
import pandas as pd

from hist_common import (
    RAW_DIR, OUT_DIR, RES_DIR, SEED, HISTORY_N, POP_PRIOR_M, BLEND_PATH,
    all_metrics, order_by_score, bootstrap_ci, stage1_order,
    load_behaviors, parse_impressions, parse_times,
)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    mean, lo, hi = bootstrap_ci(d, n_boot=5000)
    return mean, lo, hi


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    lam = float(json.load(open(BLEND_PATH))["lambda"])
    imps = json.load(open(os.path.join(OUT_DIR, "eval_set.json")))["impressions"]
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]
    test = [i for i in imps if i["split"] == "test"]

    # ---- click counts from the TRAIN days only (this is the "frozen" popularity)
    print("Counting clicks from the training days ...")
    clicks, shown = Counter(), Counter()
    for s in load_behaviors(os.path.join(RAW_DIR, "MINDsmall_train"))["impressions"]:
        ids, labs = parse_impressions(s)
        for i, l in zip(ids, labs):
            shown[i] += 1
            clicks[i] += l
    g = sum(clicks.values()) / max(sum(shown.values()), 1)
    m = POP_PRIOR_M

    # ---- score every test reader with: frozen popularity, live popularity, embedding-only, blend
    rows = []
    n_cand = n_seen = n_clicked = n_clicked_seen = 0
    for imp in test:
        y = imp["labels"]
        cands = imp["candidates"]
        sims = emb[imp["impression_id"]][f"N{HISTORY_N}"]
        frozen = [(clicks[c] + m * g) / (shown[c] + m) for c in cands]
        seen = [shown[c] > 0 for c in cands]
        n_cand += len(cands)
        n_seen += sum(seen)
        n_clicked += sum(y)
        n_clicked_seen += sum(1 for s_, l_ in zip(seen, y) if s_ and l_)
        rows.append({
            "time": imp["time"],
            "frozen_pop": all_metrics(order_by_score(frozen, rng), y)["ndcg10"],
            "live_pop": all_metrics(order_by_score(imp["pop"], rng), y)["ndcg10"],
            "embedding": all_metrics(np.argsort(-np.asarray(sims), kind="stable"), y)["ndcg10"],
            "blend": all_metrics(stage1_order(sims, imp["pop"], lam), y)["ndcg10"],
        })
    df = pd.DataFrame(rows)
    df["t"] = parse_times(df["time"])

    print("\n" + "=" * 74)
    print("1. IS POPULARITY GOING STALE?")
    print("=" * 74)
    print(f"Share of shown articles that had ALREADY appeared in the training days: {n_seen / n_cand:.1%}")
    print(f"Share of CLICKED articles that had already appeared there:             {n_clicked_seen / max(n_clicked, 1):.1%}")
    print(f"  -> so about {1 - n_clicked_seen / max(n_clicked, 1):.0%} of what readers clicked was new that day.\n")
    for name in ("frozen_pop", "live_pop"):
        mean, lo, hi = bootstrap_ci(df[name], n_boot=5000)
        print(f"  popularity, {name.split('_')[0]:6s}: nDCG@10 {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
    mean, lo, hi = paired(df["live_pop"], df["frozen_pop"])
    print(f"  live minus frozen: {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("  -> YES, popularity goes stale: click rates from earlier days rank today's feed noticeably worse.")
        print("     Fresh data matters a lot for a news feed.")
    elif hi < 0:
        print("  -> Frozen popularity did BETTER than live. Unexpected; double-check before drawing conclusions.")
    else:
        print("  -> No clear difference between frozen and live popularity on this data.")

    print("\n" + "=" * 74)
    print("2. DOES THE BLEND'S EDGE HOLD ACROSS THE DAY?")
    print("=" * 74)
    d = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    if len(d) < 90:
        print("Not enough readers with a valid time to split into thirds.")
        return
    d["part"] = pd.qcut(np.arange(len(d)), 3, labels=["earliest third", "middle third", "latest third"])
    out = []
    print(f"{'part of the day':16s} {'n':>5s} {'popularity':>11s} {'embedding':>10s} {'blend':>8s}   blend minus popularity")
    for part, g_ in d.groupby("part", observed=True):
        mean, lo, hi = paired(g_["blend"], g_["live_pop"])
        print(f"{str(part):16s} {len(g_):5d} {g_['live_pop'].mean():11.4f} {g_['embedding'].mean():10.4f} "
              f"{g_['blend'].mean():8.4f}   {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        out.append({"part": str(part), "n": len(g_), "popularity": g_["live_pop"].mean(),
                    "embedding": g_["embedding"].mean(), "blend": g_["blend"].mean(),
                    "blend_minus_pop": mean, "ci_lo": lo, "ci_hi": hi})
    print("  -> If the three 'blend minus popularity' ranges overlap, there is no sign the blend's edge")
    print("     fades over the day. This data covers ONE day, so it cannot say anything about weeks or months.")
    pd.DataFrame(out).to_csv(os.path.join(RES_DIR, "drift_check.csv"), index=False)
    print(f"\nSaved: {os.path.join(RES_DIR, 'drift_check.csv')}")


if __name__ == "__main__":
    main()
