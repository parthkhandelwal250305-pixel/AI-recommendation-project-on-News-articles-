"""
Choose the prompt variant and the fusion weight using ONLY the tuning split (never the test split).

SELECTION RULES (fixed in advance, before any LLM-score result existed):
  1. For each variant and each weight w in W_GRID, compute the mean change in nDCG@10 versus the blend
     over the tuning cases (w = 0 is the blend itself, so the change is 0 by definition).
  2. Weight: the SMALLEST w whose mean gain is within one standard error of the best w ("1-SE rule").
     This is deliberately cautious: the LLM gets more weight only if more weight clearly helps.
  3. Variant: the one with the higher gain at its chosen weight. If they are within 0.005, prefer the one with
     fewer unparseable outputs, then the faster one.
If the chosen weight is 0, the LLM signal earned no weight on the tuning cases. That is a result, not an error.

About 60 tuning cases is a small sample, so this is a sensible-defaults step, not a verdict. The test split
(with a pre-set bar and a paired interval) decides the real question.

Usage (after both tune runs of hist_llm_score.py):
    python src/hist_score_check.py
"""
import os
import json

import numpy as np
import pandas as pd

from hist_common import (
    OUT_DIR, HISTORY_N, RERANK_K, SCORE_VARIANTS, W_GRID, BLEND_PATH, CHOSEN_SCORE_PATH,
    all_metrics, blend_scores, fused_full_order, llm_only_order, score_path, bootstrap_ci,
)

MIN_CASES = 40


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def spearman(a, b):
    sa, sb = pd.Series(a).rank(), pd.Series(b).rank()
    if sa.nunique() < 2 or sb.nunique() < 2:
        return np.nan
    return float(sa.corr(sb))


def main():
    lam = float(json.load(open(BLEND_PATH))["lambda"])
    imps = {i["impression_id"]: i for i in json.load(open(os.path.join(OUT_DIR, "eval_set.json")))["impressions"]}
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]

    summary = {}
    for v in SCORE_VARIANTS:
        res = read_jsonl(score_path("tune", v))
        if len(res) < MIN_CASES:
            raise SystemExit(f"Variant '{v}' has {len(res)} tune cases; need at least {MIN_CASES}. Run:\n"
                             f"  python src/hist_llm_score.py --split tune --variant {v} --n 60")
        per_w = {w: [] for w in W_GRID}
        llm_only, spear, top1, full5, copy5, first_is_1, low5 = [], [], [], [], [], [], []
        for r in res:
            imp = imps[r["impression_id"]]
            sims = emb[r["impression_id"]][f"N{HISTORY_N}"]
            sc = blend_scores(sims, imp["pop"], lam)
            stage1 = np.argsort(-sc, kind="stable")
            if [int(c) for c in stage1[:RERANK_K]] != [int(c) for c in r["top"]]:
                raise SystemExit("A stored shortlist does not match the current blend. The blend weight or the "
                                 "embeddings changed since this run; delete the llm_score_* files and re-run.")
            y = imp["labels"]
            base = all_metrics(stage1, y)["ndcg10"]
            for w in W_GRID:
                per_w[w].append(all_metrics(fused_full_order(stage1, sc, r["scores"], w), y)["ndcg10"] - base)
            llm_only.append(all_metrics(llm_only_order(stage1, r["scores"]), y)["ndcg10"] - base)
            if r["scores"] is not None:
                s = np.asarray(r["scores"], dtype=float)
                spear.append(spearman(s, -np.arange(RERANK_K)))
                top1.append(int(np.argmax(s)) == 0)
                chosen_idx = [i for i, x in enumerate(s) if x > 0]
                full5.append(len(chosen_idx) == 5)
                pk = r.get("picks") or []
                if pk:
                    first_is_1.append(pk[0] == 1)
                    low5.extend(int(x <= 5) for x in pk)
                copy5.append(len(set(chosen_idx) & set(range(5))) / 5)

        means = {w: float(np.mean(d)) for w, d in per_w.items()}
        ses = {w: float(np.std(d, ddof=1) / np.sqrt(len(d))) for w, d in per_w.items()}
        best_w = max(W_GRID, key=lambda w: means[w])
        chosen_w = min(w for w in W_GRID if means[w] >= means[best_w] - ses[best_w])
        mean, lo, hi = bootstrap_ci(per_w[chosen_w], n_boot=5000)
        summary[v] = {
            "n": len(res), "w": chosen_w, "gain": mean, "lo": lo, "hi": hi, "means": means, "ses": ses,
            "llm_only": float(np.mean(llm_only)),
            "fallback": float(np.mean([r["used_fallback"] for r in res])),
            "latency": float(np.mean([r["latency_seconds"] for r in res])),
            "spearman": float(np.nanmean(spear)) if spear else np.nan,
            "top1": float(np.mean(top1)) if top1 else np.nan,
            "full5": float(np.mean(full5)) if full5 else np.nan,
            "first_is_1": float(np.mean(first_is_1)) if first_is_1 else np.nan,
            "low5": float(np.mean(low5)) if low5 else np.nan,
            "copy5": float(np.mean(copy5)) if copy5 else np.nan,
        }

    print("=" * 78)
    print("PROMPT + WEIGHT CHECK on the TUNING split (change in nDCG@10 versus the blend)")
    print("=" * 78)
    for v, s in summary.items():
        print(f"\n{v}   (n={s['n']} tuning cases)")
        print("  weight on LLM score:  " + "  ".join(f"{w:>6}" for w in W_GRID))
        print("  mean gain vs blend:   " + "  ".join(f"{s['means'][w]:+6.3f}" for w in W_GRID))
        print("  standard error:       " + "  ".join(f"{s['ses'][w]:6.3f}" for w in W_GRID))
        print(f"  chosen weight (1-SE rule): {s['w']}   gain at that weight {s['gain']:+.4f}  [{s['lo']:+.4f}, {s['hi']:+.4f}]")
        print(f"  LLM's 5 picks first (rest in blend order) vs blend: {s['llm_only']:+.4f}")
        print(f"  unparseable: {s['fallback']:.0%}   avg latency {s['latency']:.1f}s")
        print(f"  does the LLM just copy the blend?  its top pick is the blend's #1 in {s['top1']:.0%} of cases; "
              f"its 5 picks overlap the blend's own top 5 by {s['copy5']:.0%} on average")
        print(f"  gave a full set of 5 picks in {s['full5']:.0%} of cases")
        if v == "content":   # candidates are shown in RANDOM order here, so any preference for early slots is pure bias
            print(f"  POSITION BIAS: its first pick is the first-listed candidate in {s['first_is_1']:.0%} of cases "
                  f"(no bias = 7%); {s['low5']:.0%} of its picks are in the first 5 slots (no bias = 33%)")

    ranked = sorted(summary.items(), key=lambda kv: -kv[1]["gain"])
    best, second = ranked[0], ranked[1]
    if best[1]["gain"] - second[1]["gain"] < 0.005:
        best = sorted([best, second], key=lambda kv: (kv[1]["fallback"], kv[1]["latency"]))[0]
    chosen = best[0]
    json.dump({"variant": chosen, "w": summary[chosen]["w"], "tune_gain": summary[chosen]["gain"],
               "n_tune_cases": summary[chosen]["n"]}, open(CHOSEN_SCORE_PATH, "w"))

    print("\n" + "=" * 78)
    print(f"CHOSEN (by the rules above): variant = {chosen}, weight on the LLM score = {summary[chosen]['w']}")
    if summary[chosen]["w"] == 0:
        print("The LLM signal earned NO weight on the tuning cases. The test run is still worth doing: it tells us")
        print("whether that holds on fresh cases. Tell me and I will advise how many test cases to run.")
    print(f"Next: python src/hist_llm_score.py --n 300     (test split; resumable; avg {summary[chosen]['latency']:.0f}s per case)")
    print("=" * 78)


if __name__ == "__main__":
    main()
