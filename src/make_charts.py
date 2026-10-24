"""
Generate the charts for the project writeup from results/evaluation_results.csv
(three systems: popularity, retrieval-only, retrieval+LLM rerank).

Usage:
    python src/make_charts.py
"""
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COLORS = {"baseline": "#1f77b4", "retrieval": "#2ca02c", "rerank": "#ff7f0e"}
LABELS = {"baseline": "Popularity baseline", "retrieval": "Retrieval only", "rerank": "Retrieval + LLM rerank"}


def chart_precision_by_persona(df, out_path):
    d = df.sort_values("rerank_precision_at_10", ascending=False)
    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, key in enumerate(["baseline", "retrieval", "rerank"]):
        ax.bar(x + (i - 1) * w, d[f"{key}_precision_at_10"], w, label=LABELS[key], color=COLORS[key])
    ax.set_xticks(x)
    ax.set_xticklabels(d["persona_name"], rotation=45, ha="right")
    ax.set_ylabel("Precision@10")
    ax.set_ylim(0, 1)
    ax.set_title("Precision@10 by persona: popularity vs. retrieval-only vs. retrieval+rerank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_lift_two_panels(df, out_path):
    d = df.copy()
    d["retr_minus_base"] = d["retrieval_precision_at_10"] - d["baseline_precision_at_10"]
    d["rerank_minus_retr"] = d["rerank_precision_at_10"] - d["retrieval_precision_at_10"]
    d = d.sort_values("rerank_minus_retr")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, col, title in (
        (axes[0], "retr_minus_base", "Retrieval-only minus popularity\n(value of semantic retrieval)"),
        (axes[1], "rerank_minus_retr", "Rerank minus retrieval-only\n(value of the LLM step)"),
    ):
        colors = ["#d62728" if v < 0 else ("#999999" if v == 0 else "#2ca02c") for v in d[col]]
        ax.barh(d["persona_name"], d[col], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Precision@10 difference")
        ax.set_title(title)
        ax.set_xlim(-1, 1)
    fig.suptitle("Per-persona lift (green = better, red = worse, grey = tie)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_latency_vs_quality(df, out_path):
    lat = {
        "baseline": 0.0,
        "retrieval": df["retrieval_latency_seconds"].mean(),
        "rerank": df["rerank_total_latency_seconds"].mean(),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in ["baseline", "retrieval", "rerank"]:
        p = df[f"{key}_precision_at_10"].mean()
        ax.scatter(lat[key], p, s=120, color=COLORS[key], zorder=3)
        lat_txt = "~0" if lat[key] == 0 else (f"{lat[key] * 1000:.1f} ms" if lat[key] < 1 else f"{lat[key]:.1f} s")
        ax.annotate(f"{LABELS[key]}\nP@10 {p:.2f}, {lat_txt}", (lat[key], p),
                    xytext=(10, -18 if key == "baseline" else 8), textcoords="offset points", fontsize=9)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.002, 200)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean latency per request (seconds, symlog scale)")
    ax.set_ylabel("Mean Precision@10")
    ax.set_title("Quality vs. latency cost (CPU-only, 8GB laptop)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_aggregate_summary(df, out_path):
    metrics = [("precision_at_10", "Precision@10"), ("ndcg_at_10", "NDCG@10")]
    series = [
        ("random", "[diag] Random (expected)", "#cccccc", "//"),
        ("baseline", LABELS["baseline"], COLORS["baseline"], None),
        ("retrieval", LABELS["retrieval"], COLORS["retrieval"], None),
        ("rerank", LABELS["rerank"], COLORS["rerank"], None),
        ("catfilter", "[diag] Category filter + popularity (no ML)", "#999999", "//"),
    ]
    x = np.arange(len(metrics))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, label, color, hatch) in enumerate(series):
        vals = [df[f"{key}_{m}"].mean() for m, _ in metrics]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=label, color=color, hatch=hatch,
                      edgecolor="white" if hatch else None)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Aggregate metrics (mean over personas). Hatched bars = diagnostic references")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"))
    df = df[df["sufficient_ground_truth"]]

    chart_precision_by_persona(df, os.path.join(RESULTS_DIR, "precision_by_persona.png"))
    chart_lift_two_panels(df, os.path.join(RESULTS_DIR, "lift_by_persona.png"))
    chart_latency_vs_quality(df, os.path.join(RESULTS_DIR, "latency_vs_lift.png"))
    chart_aggregate_summary(df, os.path.join(RESULTS_DIR, "aggregate_summary.png"))

    print("\nAll charts saved to results/")


if __name__ == "__main__":
    main()
