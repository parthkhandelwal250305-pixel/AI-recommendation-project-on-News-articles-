"""
Evaluate the EQUAL-INFORMATION experiment.

PRE-SET PLAN (written before any LLM-score result existed)
  QUESTION       : when the LLM is given everything the blend has, does adding its judgment to the blend help?
  PRIMARY TEST   : "Blend + LLM" minus "Blend", nDCG@10, paired bootstrap over test cases, 95% interval.
                   The LLM's weight and prompt variant were chosen on separate TUNING cases (hist_score_check.py).
                   If the tuning step gave the LLM weight 0, "Blend + LLM" would equal the blend by construction
                   and tell us nothing, so the primary comparison then uses weight 1 (equal weight).
  BAR            : +0.02 nDCG@10 (my assumption of the smallest gain worth an LLM call per request).
  SAMPLE         : the first 600 test cases, ONE look at the end, no extension rule. Decided before any test case
                   was run (the laptop runs unattended overnight, about 8.5 hours at ~50 s per case). History: the plan
                   first said 300, extendable to 450; 900 was briefly considered when the model looked 10x faster,
                   which turned out to be a transient. On the tuning cases the per-case differences at a light LLM
                   weight were small (SD ~0.04), so 600 cases should give an interval of roughly +/-0.003.
  SECONDARY      : the LLM's 5 picks placed first (rest in blend order); the earlier titles-only design on the same cases;
                   the nDCG@5 / AUC / MRR versions. These are exploratory.
  WHAT IT CANNOT SHOW: anything about stronger models. The conclusion holds for this 3B model and these prompts.

Usage:
    python src/hist_evaluate_equal.py
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hist_common import (
    OUT_DIR, RES_DIR, SEED, HISTORY_N, RERANK_K, MIN_WORTHWHILE_GAIN, W_GRID, BLEND_PATH, CHOSEN_PATH,
    CHOSEN_SCORE_PATH, all_metrics, order_by_score, bootstrap_ci, blend_scores, fused_full_order,
    llm_only_order, llm_full_order, score_path, llm_path, selftest,
)
from hist_evaluate import Collector, paired, fmt_pair, verdict, table, oracle_order, METRIC_LABEL

N_PLANNED, N_EXTEND = 600, 600
FALLBACK_W = 1.0


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    selftest()
    out_dir = os.path.join(RES_DIR, "equal_info")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if not os.path.exists(CHOSEN_SCORE_PATH):
        raise SystemExit("No chosen_score_config.json. Run the tune steps and hist_score_check.py first.")
    cfg = json.load(open(CHOSEN_SCORE_PATH))
    variant, tuned_w = cfg["variant"], float(cfg["w"])
    lam = float(json.load(open(BLEND_PATH))["lambda"])
    imps = json.load(open(os.path.join(OUT_DIR, "eval_set.json")))["impressions"]
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]
    test = sorted([i for i in imps if i["split"] == "test"], key=lambda i: i["rank_in_split"])

    results = {r["impression_id"]: r for r in read_jsonl(score_path("test", variant))}
    if any(r.get("stage1_lambda") != lam for r in results.values()):
        raise SystemExit("Some results were made with a different blend weight. Delete the llm_score_test_* file and re-run.")
    cases = [i for i in test if i["impression_id"] in results]
    if len(cases) < 20:
        raise SystemExit(f"Only {len(cases)} test cases have results; run `python src/hist_llm_score.py --n {N_PLANNED}`.")

    old = {}
    if os.path.exists(CHOSEN_PATH):
        ov = json.load(open(CHOSEN_PATH))["variant"]
        old = {r["impression_id"]: r for r in read_jsonl(llm_path("test", ov))}

    primary_w = tuned_w if tuned_w > 0 else FALLBACK_W
    labels = {
        "popularity": "Popularity (time-aware CTR)",
        "emb": "Embedding-only",
        "blend": "Blend (similarity + popularity)",
        "fused": f"Blend + LLM scores (w={primary_w:g}, prompt '{variant}')",
        "llm_only": "LLM's 5 picks first (rest in blend order)",
        "oracle": "Oracle reorder of the shortlist (upper bound)",
    }
    col, col_old = Collector(), Collector()
    has_click, changed, spear_top1, scores_seen = [], [], [], []
    for imp in cases:
        iid = imp["impression_id"]
        y = imp["labels"]
        sims = emb[iid][f"N{HISTORY_N}"]
        sc = blend_scores(sims, imp["pop"], lam)
        stage1 = np.argsort(-sc, kind="stable")
        r = results[iid]
        if [int(c) for c in stage1[:RERANK_K]] != [int(c) for c in r["top"]]:
            raise SystemExit("A stored shortlist does not match the current blend; delete llm_score_* and re-run.")
        s = r["scores"]
        orders = {
            "popularity": order_by_score(imp["pop"], rng),
            "emb": np.argsort(-np.asarray(sims), kind="stable"),
            "blend": stage1,
            "fused": fused_full_order(stage1, sc, s, primary_w),
            "llm_only": llm_only_order(stage1, s),
            "oracle": oracle_order(stage1, y),
        }
        for w in W_GRID:
            orders[f"w={w:g}"] = fused_full_order(stage1, sc, s, w)
        ms = {k: all_metrics(o, y) for k, o in orders.items()}
        for k, m in ms.items():
            col.add(k, m)
        has_click.append(int(np.sum(np.array(y)[[int(c) for c in stage1[:RERANK_K]]])) > 0)
        changed.append([int(c) for c in orders["fused"][:10]] != [int(c) for c in stage1[:10]])
        if iid in old:
            col_old.add("blend", ms["blend"])
            col_old.add("fused", ms["fused"])
            col_old.add("old_pick5", all_metrics(llm_full_order(stage1, old[iid]["order"]), y))
    n = len(cases)
    has_click = np.array(has_click)

    # =============================================================== REPORT
    hr(f"EQUAL-INFORMATION EXPERIMENT: {n} test cases with LLM scores (planned {N_PLANNED}, max {N_EXTEND})")
    print(f"Prompt variant chosen on tuning cases: '{variant}'   |   weight chosen on tuning cases: {tuned_w:g}")
    if tuned_w == 0:
        print(f"The tuning step gave the LLM weight 0, so 'Blend + LLM' would equal the blend by construction.")
        print(f"Per the pre-set plan, the primary comparison therefore uses weight {FALLBACK_W:g} (equal weight).")
    print("'content' = the LLM saw the reader's last 20 clicked titles and each candidate's title+abstract snippet, in random order.")
    print("'full'    = the same plus each candidate's click rate and similarity numbers (value and rank), in the blend's order.")

    hr("A. SYSTEMS")
    table(col, ["popularity", "emb", "blend", "fused", "llm_only", "oracle"], labels, n)

    hr("B. PRIMARY (pre-set): does adding the LLM to the blend help?")
    prim = paired(col, "fused", "blend", "ndcg10")
    print(fmt_pair(f"Blend + LLM minus Blend [nDCG@10]", prim))
    print(f"    -> {verdict(prim)}")
    pair_rows = [{"comparison": "Blend + LLM minus Blend (PRIMARY)", "metric": "nDCG@10", **prim}]
    print(f"    The LLM changed the blend's top-10 order in {np.mean(changed):.0%} of cases.")

    print("\nSecondary / exploratory (read with care):")
    for a, b, lab, ms_ in [("fused", "blend", "Blend + LLM minus Blend", ("ndcg5", "auc", "mrr")),
                           ("llm_only", "blend", "LLM's 5 picks first minus Blend", ("ndcg10",))]:
        for m in ms_:
            r = paired(col, a, b, m)
            print(fmt_pair(f"{lab} [{METRIC_LABEL[m]}]", r))
            pair_rows.append({"comparison": lab, "metric": METRIC_LABEL[m], **r})
    d = col.get("fused", "ndcg10") - col.get("blend", "ndcg10")
    if has_click.sum() >= 20:
        mean_s, lo_s, hi_s = bootstrap_ci(d[has_click], n_boot=5000)
        print(f"\nOnly cases where the shortlist contains a clicked article ({has_click.sum()} of {n}; elsewhere no reorder can matter):")
        print(f"    Blend + LLM minus Blend [nDCG@10]: {mean_s:+.4f}  95% CI [{lo_s:+.4f}, {hi_s:+.4f}]")
    if len(col_old.data.get("blend", {}).get("ndcg10", [])) >= 20:
        n_o = len(col_old.data["blend"]["ndcg10"])
        r_new = paired(col_old, "fused", "blend", "ndcg10")
        r_old = paired(col_old, "old_pick5", "blend", "ndcg10")
        r_dir = paired(col_old, "fused", "old_pick5", "ndcg10")
        print(f"\nNew design vs the earlier titles-only 'pick 5' design, on the same {n_o} readers (exploratory):")
        print(f"    earlier design minus blend: {r_old['mean']:+.4f} [{r_old['lo']:+.4f}, {r_old['hi']:+.4f}]")
        print(f"    new design minus blend:     {r_new['mean']:+.4f} [{r_new['lo']:+.4f}, {r_new['hi']:+.4f}]")
        print(f"    new minus earlier:          {r_dir['mean']:+.4f} [{r_dir['lo']:+.4f}, {r_dir['hi']:+.4f}]")

    hr("C. HOW MUCH WEIGHT HELPS? (on TEST cases; exploratory, never used to choose anything)")
    print("Weight on the LLM score -> change in nDCG@10 versus the blend")
    sweep_rows = []
    for w in W_GRID:
        r = paired(col, f"w={w:g}", "blend", "ndcg10")
        print(f"  w = {w:>5g}:  {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        sweep_rows.append({"w": w, **{k: r[k] for k in ("mean", "lo", "hi")}})
    print("  (If no weight clearly beats zero, the LLM's judgment adds nothing beyond what the blend already has.)")

    hr("D. WHAT DID THE LLM DO?")
    fb = np.mean([results[i['impression_id']]["used_fallback"] for i in cases])
    good = [results[i["impression_id"]] for i in cases if results[i["impression_id"]]["scores"] is not None]
    sp, t1, full5, copy5 = [], [], [], []
    for r in good:
        sv = np.asarray(r["scores"], dtype=float)
        a, b = pd.Series(sv).rank(), pd.Series(-np.arange(RERANK_K)).rank()
        if a.nunique() > 1:
            sp.append(float(a.corr(b)))
        t1.append(int(np.argmax(sv)) == 0)
        idx = [i for i, x in enumerate(sv) if x > 0]
        full5.append(len(idx) == 5)
        copy5.append(len(set(idx) & set(range(5))) / 5)
    print(f"Unparseable outputs (that case equals the blend): {fb:.1%}")
    print(f"Agreement of its picks with the blend's order (Spearman, +1 = identical): {np.mean(sp):+.2f}")
    print(f"Its first pick is the blend's #1 in {np.mean(t1):.0%} of cases; its 5 picks overlap the blend's own top 5 by "
          f"{np.mean(copy5):.0%} on average; it gave a full set of 5 picks in {np.mean(full5):.0%} of cases")
    headroom = col.get("oracle", "ndcg10").mean() - col.get("blend", "ndcg10").mean()
    gain = col.get("fused", "ndcg10").mean() - col.get("blend", "ndcg10").mean()
    print(f"Headroom (perfect reorder of the shortlist minus blend): {headroom:+.4f}; Blend + LLM captured {gain / headroom:.1%} of it." if headroom > 0 else "")

    hr("E. COST, RELIABILITY AND POWER")
    lat = np.array([results[i["impression_id"]]["latency_seconds"] for i in cases])
    print(f"LLM latency per request: mean {lat.mean():.1f}s, median {np.median(lat):.1f}s, p95 {np.percentile(lat, 95):.1f}s "
          f"(about {lat.mean() * 1000 / 3600:.1f} h of LLM time per 1,000 requests on this laptop)")
    toks = [(results[i["impression_id"]].get("prompt_tokens"), results[i["impression_id"]].get("prompt_seconds")) for i in cases]
    toks = [(a, b) for a, b in toks if a and b]
    if toks:
        print(f"Prompt length: about {np.mean([a for a, _ in toks]):.0f} tokens on average; processing the prompt took "
              f"{np.mean([b for _, b in toks]):.1f}s of the {lat.mean():.1f}s per call, about "
              f"{1000 * sum(b for _, b in toks) / sum(a for a, _ in toks):.0f} ms per prompt token (Ollama may skip cached text).")
    se = prim["sd"] / np.sqrt(prim["n"])
    print(f"Power: SD of per-case differences {prim['sd']:.3f}, standard error {se:.4f} at n={n}. "
          f"Effects of about +/-{2.8 * se:.3f} nDCG@10 or larger are detectable with ~80% power.")
    verdict_text = verdict(prim)
    if n < N_PLANNED:
        print(f"Only {n} of the planned {N_PLANNED} cases are done. Finish the planned run before reading the verdict.")
    elif verdict_text.startswith("INCONCLUSIVE") and n < N_EXTEND:
        print(f"\nPRE-SET RULE: the verdict is inconclusive at {n} cases, so extend to {N_EXTEND} (and then stop):")
        print(f"    python src/hist_llm_score.py --n {N_EXTEND}")
    elif verdict_text.startswith("INCONCLUSIVE"):
        print(f"\nFinal analysis at {n} cases (the pre-set sample). The result is what it is: report it as inconclusive,")
        print("with the interval above, and say what sample size would settle it.")
    else:
        print(f"\nThe verdict is decisive at {n} cases. No extension is needed.")

    # ---------------------------------------------------------------- save
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "pairwise_equal.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(out_dir, "weight_sweep_equal.csv"), index=False)
    summ = []
    for k in ["popularity", "emb", "blend", "fused", "llm_only", "oracle"]:
        m, lo, hi = bootstrap_ci(col.get(k, "ndcg10"), n_boot=5000)
        summ.append({"system": labels[k], "n": n, "ndcg10": m, "lo": lo, "hi": hi,
                     "auc": col.get(k, "auc").mean(), "mrr": col.get(k, "mrr").mean(), "ndcg5": col.get(k, "ndcg5").mean()})
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, "summary_equal.csv"), index=False)

    # ---------------------------------------------------------------- charts
    rows = [("Blend + LLM vs Blend (PRIMARY)", prim),
            ("LLM's 5 picks first vs Blend", paired(col, "llm_only", "blend", "ndcg10"))]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (l, r) in enumerate(rows[::-1]):
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", capsize=4, color="#333333")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for l, _ in rows[::-1]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(MIN_WORTHWHILE_GAIN, color="#d62728", linestyle="--", linewidth=1,
               label=f"+{MIN_WORTHWHILE_GAIN:.2f} = smallest gain worth an LLM call (assumption)")
    ax.set_xlabel(f"Difference in nDCG@10 (paired, 95% CI), n={n}")
    ax.set_title("Equal-information test: does the LLM add to the blend?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_equal_paired.png"), dpi=150)
    plt.close(fig)

    sw = pd.DataFrame(sweep_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(range(len(sw)), sw["mean"], yerr=[sw["mean"] - sw["lo"], sw["hi"] - sw["mean"]], fmt="o-", capsize=4, color="#ff7f0e")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(MIN_WORTHWHILE_GAIN, color="#d62728", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(sw)))
    ax.set_xticklabels([f"{w:g}" for w in sw["w"]])
    ax.set_xlabel("Weight on the LLM's picks (0 = the blend alone)")
    ax.set_ylabel("Change in nDCG@10 vs blend")
    ax.set_title("Does more LLM weight help? (test cases, exploratory)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_weight_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved tables and charts to: {out_dir}")


if __name__ == "__main__":
    main()
