"""
Evaluate the history-conditioned experiment.

ANALYSIS PLAN
  PRIMARY QUESTION : does adding an LLM rerank on top of the BEST CHEAP ranker help?
                     (best cheap ranker = embedding + popularity blend, tuned on a separate split)
  PRIMARY METRIC   : nDCG@10
  PRIMARY TEST     : LLM rerank vs the blend (paired bootstrap over cases, 95% CI)
  SIZE THAT MATTERS: +0.02 nDCG@10 (an ASSUMPTION about what justifies an LLM call per request)
  PLAN REVISION    : the original plan compared the LLM against embedding-only. After the non-LLM
                     results showed the blend is clearly the best cheap system, the primary comparison
                     was changed to LLM-vs-blend, and the LLM's shortlist now comes from the blend.
                     This was decided BEFORE any LLM result existed.
Everything else (AUC, MRR, nDCG@5, history lengths, fusion, segments) is secondary/exploratory.

Systems
  Random                      floor; AUC should be ~0.5 (a check that the metric code is right)
  Popularity                  time-aware click-through rate (only uses events BEFORE each case)
  Embedding-only              profile = mean embedding of last 20 clicks
  Embedding + popularity      z-scored blend; weight tuned on a SEPARATE tuning split  <-- best cheap ranker
  LLM rerank                  qwen2.5:3b picks its 5 favourites from the blend's top-15; the rest keep blend order
                              (prompt variant chosen on the TUNE split; LLM subset only)
  Rank fusion (exploratory)   reciprocal-rank fusion of blend and LLM rankings (LLM subset only)
  Oracle rerank (upper bound) perfect ordering of the blend's top-15: the MOST any reranker could add

Usage:
    python src/hist_evaluate.py
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hist_common import (
    OUT_DIR, RES_DIR, SEED, HISTORY_N, HISTORY_SENSITIVITY, RERANK_K, PICK_N, MIN_WORTHWHILE_GAIN,
    all_metrics, order_by_score, zscore, bootstrap_ci, selftest, stage1_order, BLEND_PATH,
    CHOSEN_PATH, llm_path, llm_full_order,
)

METRICS = ["auc", "mrr", "ndcg5", "ndcg10"]
METRIC_LABEL = {"auc": "AUC", "mrr": "MRR", "ndcg5": "nDCG@5", "ndcg10": "nDCG@10"}
LAMBDA_GRID = [0, 0.25, 0.5, 1, 2, 4, 8]
RRF_K = 60


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ------------------------------------------------------------------ ranking constructors
def emb_order(sims):
    return np.argsort(-np.asarray(sims), kind="stable")


def oracle_order(stage1, labels):
    so = [int(i) for i in stage1]
    top, rest = so[:RERANK_K], so[RERANK_K:]
    top_sorted = sorted(top, key=lambda i: (-labels[i], top.index(i)))
    return np.array(top_sorted + rest)


def rrf_order(order_a, order_b):
    n = len(order_a)
    ra, rb = np.empty(n), np.empty(n)
    ra[np.asarray(order_a)] = np.arange(n)
    rb[np.asarray(order_b)] = np.arange(n)
    score = 1.0 / (RRF_K + ra) + 1.0 / (RRF_K + rb)
    return np.argsort(-score, kind="stable")


# ------------------------------------------------------------------ evaluation helpers
class Collector:
    """Per-system, per-metric arrays over cases."""
    def __init__(self):
        self.data = {}

    def add(self, system, m):
        d = self.data.setdefault(system, {k: [] for k in METRICS})
        for k in METRICS:
            d[k].append(m[k])

    def get(self, system, metric):
        return np.array(self.data[system][metric])


def paired(col, a, b, metric):
    diffs = col.get(a, metric) - col.get(b, metric)
    mean, lo, hi = bootstrap_ci(diffs)
    return {
        "mean": mean, "lo": lo, "hi": hi,
        "wins": int((diffs > 1e-9).sum()), "ties": int((np.abs(diffs) <= 1e-9).sum()),
        "losses": int((diffs < -1e-9).sum()), "sd": float(diffs.std(ddof=1)), "n": len(diffs),
    }


def fmt_pair(label, r):
    return (f"{label}\n    {r['mean']:+.4f}  95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]   "
            f"W/T/L = {r['wins']}/{r['ties']}/{r['losses']}   (n={r['n']})")


def verdict(r):
    lo, hi, g = r["lo"], r["hi"], MIN_WORTHWHILE_GAIN
    if hi < 0:
        return "RELIABLY WORSE (entire CI below zero)"
    if lo > g:
        return f"gain is reliable AND clears the +{g:.2f} bar"
    if lo > 0:
        return f"gain is reliable but may be smaller than the +{g:.2f} bar"
    if hi < g:
        return f"no reliable gain; a gain of +{g:.2f} or more is RULED OUT (CI upper bound below it)"
    return f"INCONCLUSIVE: CI contains both 0 and +{g:.2f}; more cases are needed"


def table(col, systems, labels, n):
    print(f"{'System':42s} " + " ".join(f"{METRIC_LABEL[m]:>8s}" for m in METRICS) + "   nDCG@10 95% CI")
    rows = []
    for s in systems:
        vals = [col.get(s, m).mean() for m in METRICS]
        mean, lo, hi = bootstrap_ci(col.get(s, "ndcg10"))
        print(f"{labels[s]:42s} " + " ".join(f"{v:8.4f}" for v in vals) + f"   [{lo:.4f}, {hi:.4f}]")
        rows.append({"system": labels[s], "n": n, **{m: v for m, v in zip(METRICS, vals)},
                     "ndcg10_ci_lo": lo, "ndcg10_ci_hi": hi})
    return rows


def main():
    selftest()
    os.makedirs(RES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    es = json.load(open(os.path.join(OUT_DIR, "eval_set.json")))
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))
    emb_scores = emb["scores"]
    imps = es["impressions"]
    tune = [i for i in imps if i["split"] == "tune"]
    test = sorted([i for i in imps if i["split"] == "test"], key=lambda i: i["rank_in_split"])

    llm = {}
    variant = None
    if os.path.exists(CHOSEN_PATH):
        variant = json.load(open(CHOSEN_PATH))["variant"]
        path = llm_path("test", variant)
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    llm[r["impression_id"]] = r

    # ---------------------------------------------------------------- tune blend weight
    def metrics_for(imp, order):
        return all_metrics(order, imp["labels"])

    best_lam, best_score = 0, -1
    tune_curve = []
    for lam in LAMBDA_GRID:
        vals = []
        for imp in tune:
            sims = emb_scores[imp["impression_id"]][f"N{HISTORY_N}"]
            vals.append(metrics_for(imp, stage1_order(sims, imp["pop"], lam))["ndcg10"])
        mean = float(np.mean(vals))
        tune_curve.append((lam, mean))
        if mean > best_score:
            best_lam, best_score = lam, mean

    stale = [r for r in llm.values() if r.get("stage1_lambda") != best_lam]
    if stale:
        raise SystemExit(f"{len(stale)} LLM results were produced with a different stage-1 blend weight than the "
                         f"one tuned now (lambda={best_lam}). Delete data/history_eval/llm_rerank_test_*.jsonl and re-run "
                         "hist_llm_rerank.py so the comparison is like for like.")
    with open(BLEND_PATH, "w") as f:
        json.dump({"lambda": best_lam, "n_tune": len(tune), "tune_curve": tune_curve}, f)

    # ---------------------------------------------------------------- score the test split
    labels = {
        "random": "Random (floor)",
        "popularity": "Popularity (time-aware CTR)",
        "emb": f"Embedding-only (last {HISTORY_N} clicks)",
        "hybrid": f"Embedding + popularity blend (lambda={best_lam})",
        "llm": f"LLM rerank: picks {PICK_N} of blend top-15 ({variant})",
        "fusion": "Fusion: blend + LLM (exploratory)",
        "oracle": "Oracle rerank of blend top-15 (upper bound)",
    }
    for n_ in HISTORY_SENSITIVITY:
        labels[f"emb_N{n_}"] = f"Embedding-only, last {n_} clicks (sensitivity)"

    col_all = Collector()          # all test cases
    col_llm = Collector()          # LLM subset
    per_imp_rows = []
    bucket_of = []
    llm_cases = []

    for imp in test:
        iid = imp["impression_id"]
        sims = emb_scores[iid][f"N{HISTORY_N}"]
        y = imp["labels"]
        stage1 = stage1_order(sims, imp["pop"], best_lam)
        systems_order = {
            "random": order_by_score(np.zeros(len(y)), rng),
            "popularity": order_by_score(imp["pop"], rng),
            "emb": emb_order(sims),
            "hybrid": stage1,
        }
        for n_ in HISTORY_SENSITIVITY:
            systems_order[f"emb_N{n_}"] = emb_order(emb_scores[iid][f"N{n_}"])
        systems_order["oracle"] = oracle_order(stage1, y)
        has_llm = iid in llm
        if has_llm:
            lo_ = llm_full_order(stage1, llm[iid]["order"])
            systems_order["llm"] = lo_
            systems_order["fusion"] = rrf_order(stage1, lo_)

        ms = {s: metrics_for(imp, o) for s, o in systems_order.items()}
        for s, m in ms.items():
            if s in ("llm", "fusion"):
                continue
            col_all.add(s, m)
        if has_llm:
            for s in ("random", "popularity", "emb", "hybrid", "llm", "fusion", "oracle"):
                col_llm.add(s, ms[s])
            llm_cases.append(imp)
        hl = len(imp["history"])
        bucket_of.append("5-9" if hl < 10 else "10-19" if hl < 20 else "20-49" if hl < 50 else "50+")
        for s, m in ms.items():
            per_imp_rows.append({"impression_id": iid, "system": s, "in_llm_subset": has_llm, **m})

    n_all, n_llm = len(test), len(llm_cases)

    # ================================================================ REPORT
    hr(f"DATA: {len(tune)} tuning cases, {n_all} test cases (one per user), {n_llm} test cases with LLM results")
    avg_c = np.mean([len(i['candidates']) for i in test])
    avg_p = np.mean([sum(i['labels']) for i in test])
    print(f"Avg {avg_c:.1f} candidates and {avg_p:.2f} clicked per case. "
          f"Random AUC check (should be ~0.5): {col_all.get('random', 'auc').mean():.3f}")
    print("Blend-weight tuning (nDCG@10 on the TUNING split only): "
          + ", ".join(f"{l}: {s:.4f}" for l, s in tune_curve) + f"  -> chose lambda={best_lam}")

    hr(f"A. MAIN COMPARISON on all {n_all} test cases (no LLM)")
    main_sys = ["random", "popularity", "emb", "hybrid"]
    rows_a = table(col_all, main_sys, labels, n_all)
    pair_rows = []
    print()
    for a, b, lab in [("emb", "popularity", "Embedding-only minus Popularity"),
                      ("hybrid", "emb", "Blend minus Embedding-only"),
                      ("hybrid", "popularity", "Blend minus Popularity")]:
        for m in ("ndcg10", "auc"):
            r = paired(col_all, a, b, m)
            print(fmt_pair(f"{lab} [{METRIC_LABEL[m]}]", r))
            pair_rows.append({"split": "all_test", "comparison": lab, "metric": METRIC_LABEL[m], **r})

    hr(f"B. SENSITIVITY: how many past clicks does the profile need? (nDCG@10, n={n_all})")
    for n_ in HISTORY_SENSITIVITY:
        mean, lo, hi = bootstrap_ci(col_all.get(f"emb_N{n_}", "ndcg10"))
        flag = "  <-- primary (fixed in advance)" if n_ == HISTORY_N else ""
        print(f"  last {n_:>2d} clicks: {mean:.4f}  [{lo:.4f}, {hi:.4f}]{flag}")

    hr("C. SEGMENT: does personalization depend on how much history we have? (nDCG@10)")
    bucket_of = np.array(bucket_of)
    print(f"{'history length':16s} {'n':>5s} {'popularity':>11s} {'embedding':>10s} {'blend':>8s} {'emb - pop':>10s}")
    seg_rows = []
    for b in ["5-9", "10-19", "20-49", "50+"]:
        mask = bucket_of == b
        if mask.sum() < 20:
            continue
        p, e, h = (col_all.get(s, "ndcg10")[mask].mean() for s in ("popularity", "emb", "hybrid"))
        print(f"{b:16s} {mask.sum():5d} {p:11.4f} {e:10.4f} {h:8.4f} {e - p:+10.4f}")
        seg_rows.append({"history_bucket": b, "n": int(mask.sum()), "popularity": p, "embedding": e, "blend": h})
    print("  (history is capped at 50 stored clicks, so '50+' means 50 or more.)")

    llm_rows = []
    if n_llm >= 10:
        hr(f"D. THE KEY QUESTION: does the LLM add value on top of the best cheap ranker (the blend)?  ({n_llm} cases with LLM results)")
        llm_sys = ["random", "popularity", "emb", "hybrid", "llm", "fusion", "oracle"]
        llm_rows = table(col_llm, llm_sys, labels, n_llm)

        print(f"\nPRIMARY: LLM rerank vs the blend (best cheap ranker), nDCG@10")
        prim = paired(col_llm, "llm", "hybrid", "ndcg10")
        print(fmt_pair("LLM rerank minus Blend [nDCG@10]", prim))
        print(f"    -> {verdict(prim)}")
        pair_rows.append({"split": "llm_subset", "comparison": "LLM minus Blend (PRIMARY)",
                          "metric": "nDCG@10", **prim})

        print("\nSecondary / exploratory comparisons (read with care):")
        for a, b, lab in [("llm", "hybrid", "LLM minus Blend"),
                          ("fusion", "hybrid", "Fusion minus Blend"),
                          ("llm", "emb", "LLM minus Embedding-only"),
                          ("hybrid", "popularity", "Blend minus Popularity")]:
            for m in ("ndcg5", "auc", "mrr") if (a, b) == ("llm", "hybrid") else ("ndcg10",):
                r = paired(col_llm, a, b, m)
                print(fmt_pair(f"{lab} [{METRIC_LABEL[m]}]", r))
                pair_rows.append({"split": "llm_subset", "comparison": lab, "metric": METRIC_LABEL[m], **r})

        hr("E. HEADROOM: how much COULD any reranker add on top of the blend?")
        base_n = col_llm.get("hybrid", "ndcg10").mean()
        orc_n = col_llm.get("oracle", "ndcg10").mean()
        llm_n = col_llm.get("llm", "ndcg10").mean()
        recs, hits = [], []
        for i in llm_cases:
            s1 = stage1_order(emb_scores[i["impression_id"]][f"N{HISTORY_N}"], i["pop"], best_lam)[:RERANK_K]
            lab = np.array(i["labels"])
            recs.append(lab[s1].sum() / lab.sum())
            hits.append(lab[s1].sum() > 0)
        print(f"Stage-1 recall@{RERANK_K}: {np.mean(recs):.1%} of clicked articles are inside the blend's shortlist "
              f"({np.mean(hits):.1%} of cases have at least one).")
        print(f"nDCG@10: blend {base_n:.4f}  ->  perfect reranker {orc_n:.4f}  (headroom {orc_n - base_n:+.4f})")
        if orc_n > base_n:
            print(f"The LLM captured {(llm_n - base_n) / (orc_n - base_n):.1%} of that headroom "
                  f"({llm_n - base_n:+.4f} of {orc_n - base_n:+.4f}).")

        hr("F. RELIABILITY, COST AND STATISTICAL POWER")
        lr = [llm[i['impression_id']] for i in llm_cases]
        lat = np.array([r['latency_seconds'] for r in lr])
        fb = np.mean([r['used_fallback'] for r in lr])
        pad = np.mean([r['n_padded'] > 0 for r in lr])
        e_ms = emb["scoring_latency_ms_median"]
        print(f"LLM unparseable (fell back to blend order): {fb:.1%}   cases where the LLM gave fewer than {PICK_N} valid picks: {pad:.1%}")
        print(f"LLM latency per request: mean {lat.mean():.1f}s, median {np.median(lat):.1f}s, p95 {np.percentile(lat, 95):.1f}s")
        print(f"Embedding scoring per request (precomputed article embeddings): median {e_ms:.2f} ms")
        print(f"-> LLM step is ~{lat.mean() * 1000 / max(e_ms, 1e-9):,.0f}x slower; ranking 1,000 requests takes "
              f"~{lat.mean() * 1000 / 3600:.1f} h of LLM time vs ~{e_ms / 1000:.2f} s.")
        se = prim["sd"] / np.sqrt(prim["n"])
        print(f"Power: SD of per-case differences {prim['sd']:.3f}, standard error {se:.4f} at n={prim['n']}. "
              f"Smallest effect detectable with ~80% power: about +/-{2.8 * se:.3f} nDCG@10.")
        need = (2.8 * prim['sd'] / MIN_WORTHWHILE_GAIN) ** 2
        print(f"To reliably detect the +{MIN_WORTHWHILE_GAIN:.2f} bar you would need roughly {need:,.0f} cases "
              f"(you have {prim['n']}).")
    else:
        print("\n(No/too few test-split LLM results yet. Section D onward needs: tune-split prompt check, then "
              "`python src/hist_llm_rerank.py --n 150`.)")

    # ---------------------------------------------------------------- save
    pd.DataFrame(rows_a).to_csv(os.path.join(RES_DIR, "summary_test.csv"), index=False)
    if llm_rows:
        pd.DataFrame(llm_rows).to_csv(os.path.join(RES_DIR, "summary_llm_subset.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(RES_DIR, "pairwise.csv"), index=False)
    pd.DataFrame(per_imp_rows).to_csv(os.path.join(RES_DIR, "per_case_metrics.csv"), index=False)
    pd.DataFrame(seg_rows).to_csv(os.path.join(RES_DIR, "history_segments.csv"), index=False)

    # ---------------------------------------------------------------- charts
    colors = {"random": "#cccccc", "popularity": "#1f77b4", "emb": "#2ca02c", "hybrid": "#17becf",
              "llm": "#ff7f0e", "fusion": "#9467bd", "oracle": "#999999"}
    short = {"random": "Random", "popularity": "Popularity", "emb": "Embedding", "hybrid": "Emb + pop",
             "llm": "LLM rerank", "fusion": "Fusion", "oracle": "Oracle\n(upper bound)"}

    def bar_panel(ax, col, systems, title):
        means, los, his = [], [], []
        for s in systems:
            m, lo, hi = bootstrap_ci(col.get(s, "ndcg10"))
            means.append(m); los.append(m - lo); his.append(hi - m)
        ax.bar(range(len(systems)), means, yerr=[los, his], capsize=4,
               color=[colors[s] for s in systems], hatch=None)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([short[s] for s in systems], fontsize=9)
        ax.set_ylabel("nDCG@10 (mean, 95% CI)")
        ax.set_title(title)
        for i, m in enumerate(means):
            ax.text(i, m + his[i] + 0.012, f"{m:.3f}", ha="center", fontsize=8)

    if n_llm >= 10:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        bar_panel(axes[0], col_all, main_sys, f"All test cases (n={n_all})")
        bar_panel(axes[1], col_llm, ["popularity", "emb", "hybrid", "llm", "fusion", "oracle"],
                  f"LLM subset (n={n_llm})")
        for ax in axes:
            ax.set_ylim(0, max(a.get_ylim()[1] for a in axes))
    else:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        bar_panel(ax, col_all, main_sys, f"All test cases (n={n_all})")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "chart_systems.png"), dpi=150)
    plt.close(fig)

    forest = [(l, r) for l, r in [
        ("Embedding vs Popularity", paired(col_all, "emb", "popularity", "ndcg10")),
        ("Blend vs Embedding", paired(col_all, "hybrid", "emb", "ndcg10")),
        ("Blend vs Popularity", paired(col_all, "hybrid", "popularity", "ndcg10")),
    ]]
    if n_llm >= 10:
        forest += [("LLM vs Blend (PRIMARY)", paired(col_llm, "llm", "hybrid", "ndcg10")),
                   ("Fusion vs Blend (exploratory)", paired(col_llm, "fusion", "hybrid", "ndcg10")),
                   ("LLM vs Embedding-only", paired(col_llm, "llm", "emb", "ndcg10"))]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(forest)))
    for i, (l, r) in enumerate(forest[::-1]):
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", capsize=4, color="#333333")
    ax.set_yticks(range(len(forest)))
    ax.set_yticklabels([l for l, _ in forest[::-1]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(MIN_WORTHWHILE_GAIN, color="#d62728", linestyle="--", linewidth=1,
               label=f"+{MIN_WORTHWHILE_GAIN:.2f} = smallest gain worth an LLM call (assumption)")
    ax.set_xlabel("Difference in nDCG@10 (paired, 95% CI)")
    ax.set_title("Paired differences between systems")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "chart_paired_differences.png"), dpi=150)
    plt.close(fig)

    if seg_rows:
        sd = pd.DataFrame(seg_rows)
        x = np.arange(len(sd)); w = 0.27
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w, sd["popularity"], w, label="Popularity", color=colors["popularity"])
        ax.bar(x, sd["embedding"], w, label="Embedding-only", color=colors["emb"])
        ax.bar(x + w, sd["blend"], w, label="Embedding + popularity", color=colors["hybrid"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(sd["history_bucket"], sd["n"])])
        ax.set_xlabel("Number of past clicks available")
        ax.set_ylabel("nDCG@10")
        ax.set_title("Does personalization need history?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "chart_history_segments.png"), dpi=150)
        plt.close(fig)

    print(f"\nSaved tables and charts to: {RES_DIR}")


if __name__ == "__main__":
    main()
