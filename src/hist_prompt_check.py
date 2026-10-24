"""
Choose the LLM prompt variant using ONLY the tuning split (never the test split).

SELECTION RULE (fixed in advance):
  1. Pick the variant with the highest mean nDCG@10 gain over the blend on the tuning cases.
  2. If two variants are within 0.005 of each other, prefer fewer unparseable outputs,
     then the one that is less position-biased.
Small samples (~30 cases) are noisy, so this is a sanity filter for catastrophic prompts
and obvious biases, NOT a claim that one prompt is better. The test split decides the real question.

Also prints diagnostics:
  - position bias (shuffled variant): share of picks that sit in display positions 1-5.
    With no bias this is 5/15 = 33%.
  - copying (both variants): how many of the LLM's 5 picks are also the blend's own top 5.

Usage (after running hist_llm_rerank.py --split tune for both variants):
    python src/hist_prompt_check.py
"""
import os
import json

import numpy as np

from hist_common import (
    OUT_DIR, HISTORY_N, PICK_N, VARIANTS, CHOSEN_PATH, BLEND_PATH,
    all_metrics, stage1_order, llm_full_order, llm_path, bootstrap_ci,
)

MIN_CASES = 20


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    lam = float(json.load(open(BLEND_PATH))["lambda"])
    imps = {i["impression_id"]: i for i in json.load(open(os.path.join(OUT_DIR, "eval_set.json")))["impressions"]}
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]

    rows = {}
    for v in VARIANTS:
        res = read_jsonl(llm_path("tune", v))
        if len(res) < MIN_CASES:
            raise SystemExit(f"Variant '{v}' has {len(res)} tune cases; need at least {MIN_CASES}. Run:\n"
                             f"  python src/hist_llm_rerank.py --split tune --variant {v} --n 30")
        diffs, copy_share, low_pos, n_picks = [], [], 0, 0
        for r in res:
            imp = imps[r["impression_id"]]
            sims = emb[r["impression_id"]][f"N{HISTORY_N}"]
            s1 = stage1_order(sims, imp["pop"], lam)
            b = all_metrics(s1, imp["labels"])["ndcg10"]
            l = all_metrics(llm_full_order(s1, r["order"]), imp["labels"])["ndcg10"]
            diffs.append(l - b)
            llm_picks = [int(c) for c in r["order"][:r["n_returned"]]]
            copy_share.append(len(set(llm_picks) & set(int(c) for c in s1[:PICK_N])) / PICK_N)
            low_pos += sum(1 for p in r["picks"] if p <= PICK_N)
            n_picks += len(r["picks"])
        diffs = np.array(diffs)
        mean, lo, hi = bootstrap_ci(diffs, n_boot=5000)
        rows[v] = {
            "n": len(res), "gain": mean, "lo": lo, "hi": hi,
            "wins": int((diffs > 1e-9).sum()), "ties": int((np.abs(diffs) <= 1e-9).sum()), "losses": int((diffs < -1e-9).sum()),
            "fallback": float(np.mean([r["used_fallback"] for r in res])),
            "short": float(np.mean([r["n_padded"] > 0 for r in res])),
            "low_pos": low_pos / max(n_picks, 1), "copy": float(np.mean(copy_share)),
            "latency": float(np.mean([r["latency_seconds"] for r in res])),
        }

    print("=" * 78)
    print(f"PROMPT CHECK on the TUNING split (n~{min(r['n'] for r in rows.values())} cases per variant)")
    print("=" * 78)
    for v, r in rows.items():
        print(f"\n{v}")
        print(f"  nDCG@10 change vs blend: {r['gain']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]   "
              f"W/T/L = {r['wins']}/{r['ties']}/{r['losses']}")
        print(f"  unparseable: {r['fallback']:.0%}   fewer than {PICK_N} picks: {r['short']:.0%}   "
              f"avg latency {r['latency']:.1f}s")
        if v == "pick5_shuffled":
            print(f"  position bias: {r['low_pos']:.0%} of picks are display positions 1-{PICK_N} (no bias = 33%)")
        print(f"  copying: {r['copy']:.0%} of the LLM's picks are the blend's own top {PICK_N}")

    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["gain"])
    best, second = ranked[0], ranked[1]
    if best[1]["gain"] - second[1]["gain"] < 0.005:
        cands = sorted([best, second], key=lambda kv: (kv[1]["fallback"], kv[1]["low_pos"]))
        best = cands[0]
    chosen = best[0]
    json.dump({"variant": chosen, "tune_gain": best[1]["gain"], "n_tune_cases": best[1]["n"]}, open(CHOSEN_PATH, "w"))
    print("\n" + "=" * 78)
    print(f"CHOSEN VARIANT (by the rule above): {chosen}")
    print("Next: python src/hist_llm_rerank.py --n 150     (test split, uses this variant)")
    print("=" * 78)


if __name__ == "__main__":
    main()
