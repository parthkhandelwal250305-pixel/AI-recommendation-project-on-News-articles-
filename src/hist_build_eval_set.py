"""
Build the history-conditioned evaluation set from MIND-small.

Each eval case = one real user at one moment: their past clicks (history) and
the candidate articles they were shown, with which ones they clicked.

Key design choices (each is a talking point):
  - ONE impression per user, so cases are independent (the bootstrap assumes this).
  - Fixed, seeded sample; a TUNE split (to pick the popularity-blend weight) is kept
    strictly separate from the TEST split (to report results).
  - The popularity baseline is a time-aware click-through rate: it only uses
    clicks/impressions that happened BEFORE the case's timestamp. It is a strong,
    realistic non-personalized baseline, and it cannot peek at the answer.

Usage:
    python src/hist_build_eval_set.py
    python src/hist_build_eval_set.py --n-tune 150 --n-test 400   # smaller, for a quick test
"""
import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from hist_common import (
    RAW_DIR, OUT_DIR, SEED, MIN_HISTORY, MIN_CANDIDATES, N_TUNE, N_TEST,
    HISTORY_SENSITIVITY, POP_PRIOR_M,
    load_news, load_behaviors, parse_impressions, parse_times,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tune", type=int, default=N_TUNE)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    train_dir = os.path.join(RAW_DIR, "MINDsmall_train")
    dev_dir = os.path.join(RAW_DIR, "MINDsmall_dev")
    for d in (train_dir, dev_dir):
        if not os.path.exists(os.path.join(d, "behaviors.tsv")):
            raise SystemExit(f"Missing {d}. Run `python src/download_data.py` first.")

    print("Loading article metadata (train + dev) ...")
    news = pd.concat([load_news(train_dir), load_news(dev_dir)], ignore_index=True)
    news = news.drop_duplicates("news_id").set_index("news_id")
    known = set(news.index)

    # Popularity prior from the TRAIN period (earlier days than dev).
    print("Counting historical clicks/impressions from the train period ...")
    clicks, imps = Counter(), Counter()
    for s in load_behaviors(train_dir)["impressions"]:
        ids, labs = parse_impressions(s)
        for i, l in zip(ids, labs):
            imps[i] += 1
            clicks[i] += l
    global_ctr = sum(clicks.values()) / max(sum(imps.values()), 1)
    print(f"  global click-through rate in train: {global_ctr:.4f}")

    print("Loading dev behaviors and sorting by time ...")
    dev = load_behaviors(dev_dir)
    dev["t"] = parse_times(dev["time"])
    dev = dev.sort_values("t", kind="stable").reset_index(drop=True)
    users = dev["user_id"].to_numpy()

    parsed = []
    qualifies = []
    for r, row in enumerate(dev.itertuples(index=False)):
        hist = [h for h in row.history.split() if h in known] if isinstance(row.history, str) else []
        cands, labs = parse_impressions(row.impressions)
        ok = (
            len(hist) >= MIN_HISTORY
            and len(cands) >= MIN_CANDIDATES
            and all(c in known for c in cands)
            and 1 <= sum(labs) < len(labs)          # at least one click and one non-click
        )
        parsed.append((hist, cands, labs))
        if ok:
            qualifies.append(r)
    print(f"  eligible impressions: {len(qualifies):,} of {len(dev):,}")

    # One impression per user, seeded shuffle.
    seen, picked = set(), []
    for r in rng.permutation(qualifies):
        u = users[r]
        if u in seen:
            continue
        seen.add(u)
        picked.append(int(r))
    want = args.n_tune + args.n_test
    if len(picked) < want:
        print(f"  WARNING: only {len(picked)} eligible users; requested {want}.")
    picked = picked[:want]
    tune_rows = picked[: args.n_tune]
    test_rows = picked[args.n_tune:]
    sampled = {r: ("tune", k) for k, r in enumerate(tune_rows)}
    sampled.update({r: ("test", k) for k, r in enumerate(test_rows)})

    # Time-ordered pass: popularity uses ONLY events before each impression.
    print("Computing leak-free popularity scores ...")
    m = POP_PRIOR_M
    pop_by_row = {}
    for r in range(len(dev)):
        hist, cands, labs = parsed[r]
        if r in sampled:
            pop_by_row[r] = [
                round((clicks[c] + m * global_ctr) / (imps[c] + m), 6) for c in cands
            ]
        for c, l in zip(cands, labs):   # update AFTER scoring this impression
            imps[c] += 1
            clicks[c] += l

    max_hist = max(HISTORY_SENSITIVITY)
    impressions = []
    for r, (split, k) in sorted(sampled.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        hist, cands, labs = parsed[r]
        impressions.append({
            "impression_id": str(dev.at[r, "impression_id"]),
            "user_id": str(users[r]),
            "time": str(dev.at[r, "time"]),
            "split": split,
            "rank_in_split": k,
            "history": hist[-max_hist:],
            "candidates": cands,
            "labels": labs,
            "pop": pop_by_row[r],
        })

    needed = set()
    for imp in impressions:
        needed.update(imp["history"])
        needed.update(imp["candidates"])
    art = news.loc[sorted(needed), ["category", "subcategory", "title", "abstract"]].reset_index()
    art.to_csv(os.path.join(OUT_DIR, "articles.csv"), index=False)

    with open(os.path.join(OUT_DIR, "eval_set.json"), "w") as f:
        json.dump({
            "config": {
                "seed": SEED, "min_history": MIN_HISTORY, "min_candidates": MIN_CANDIDATES,
                "n_tune": len(tune_rows), "n_test": len(test_rows),
                "pop_prior_m": POP_PRIOR_M, "global_ctr_train": global_ctr,
            },
            "impressions": impressions,
        }, f)

    test = [i for i in impressions if i["split"] == "test"]
    n_cand = np.array([len(i["candidates"]) for i in test])
    n_pos = np.array([sum(i["labels"]) for i in test])
    h_len = np.array([len(i["history"]) for i in test])
    print("\n" + "=" * 60)
    print("EVAL SET SUMMARY")
    print("=" * 60)
    print(f"tune cases: {len(tune_rows)}   test cases: {len(test_rows)}   (one per user)")
    print(f"test: candidates/impression mean {n_cand.mean():.1f}, clicks/impression mean {n_pos.mean():.2f}, "
          f"click rate {n_pos.sum() / n_cand.sum():.3f}")
    print(f"test: history length (capped at {max_hist}) median {np.median(h_len):.0f}, "
          f"share with >=20 clicks {np.mean(h_len >= 20):.0%}")
    print(f"unique articles needed: {len(art):,}")
    print(f"Saved: {os.path.join(OUT_DIR, 'eval_set.json')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'articles.csv')}")


if __name__ == "__main__":
    main()
