"""
Shared helpers for the history-conditioned evaluation (v3).

Design (the standard MIND evaluation, adapted):
  - Each test case is one real user at one moment in time (an "impression").
  - INPUT: that user's past clicks (their history).
  - TASK: rank the candidate articles they were actually shown.
  - TRUTH: which candidates they actually clicked.
There is no persona text and no category label anywhere, so the "category
answer key" problem from v1/v2 does not exist here.
"""
import os
import math

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "history_eval")
RES_DIR = os.path.join(ROOT, "results", "history_eval")

# ---- Experiment design constants (fixed BEFORE looking at any results) ----
SEED = 42
MIN_HISTORY = 5               # users need at least this many past clicks
MIN_CANDIDATES = 15           # impressions need at least this many candidates (so top-15 rerank is meaningful)
N_TUNE = 1000                 # impressions used only to tune the popularity-blend weight
N_TEST = 3000                 # impressions for the main comparison (embeddings, popularity, blend)
HISTORY_N = 20                # PRIMARY: last 20 clicks form the user profile
HISTORY_SENSITIVITY = [5, 10, 20, 50]   # reported as a sensitivity analysis only
RERANK_K = 15                 # stage 1 hands the top 15 candidates to the LLM
PICK_N = 5                    # the LLM picks its 5 favourites; the rest keep the blend's order
VARIANTS = ["pick5_shuffled", "pick5_ranked"]   # prompt variants; chosen on the TUNE split only
POP_PRIOR_M = 20              # smoothing strength for click-through-rate popularity
MIN_WORTHWHILE_GAIN = 0.02    # ASSUMPTION: smallest nDCG@10 gain that would justify an LLM call per request

NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


# ------------------------------------------------------------------ data loading
def load_news(split_dir: str) -> pd.DataFrame:
    path = os.path.join(split_dir, "news.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS,
                       index_col=False, quoting=3)  # quoting=3: never treat " as a quote char


def load_behaviors(split_dir: str) -> pd.DataFrame:
    path = os.path.join(split_dir, "behaviors.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS,
                       index_col=False, quoting=3)


def parse_impressions(s):
    """'N1-0 N2-1 ...' -> (['N1','N2'], [0,1])."""
    ids, labels = [], []
    if not isinstance(s, str):
        return ids, labels
    for tok in s.split():
        if "-" in tok:
            nid, lab = tok.rsplit("-", 1)
            ids.append(nid)
            labels.append(1 if lab == "1" else 0)
    return ids, labels


def parse_times(series):
    t = pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if t.isna().mean() > 0.01:
        t = pd.to_datetime(series, errors="coerce")
    return t


def article_text(title, abstract) -> str:
    title = title if isinstance(title, str) else ""
    abstract = abstract if isinstance(abstract, str) else ""
    return f"{title}. {abstract}".strip()


# ------------------------------------------------------------------ ranking + metrics
def order_by_score(scores, rng):
    """Indices sorted by descending score; exact ties are broken randomly (seeded)."""
    scores = np.asarray(scores, dtype=float)
    tiebreak = rng.random(len(scores))
    return np.lexsort((tiebreak, -scores))


def zscore(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def all_metrics(order, labels):
    """AUC, MRR, nDCG@5, nDCG@10 for ONE impression (same definitions as the MIND benchmark)."""
    y = np.asarray(labels)[np.asarray(order)]
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pos_positions = np.where(y == 1)[0]
    # negatives ranked below each positive
    below = 0
    for k, p in enumerate(pos_positions):
        positives_below = n_pos - 1 - k
        below += (n - 1 - p) - positives_below
    auc = below / (n_pos * n_neg)
    mrr = float((y / (np.arange(n) + 1)).sum() / n_pos)

    def ndcg(k):
        dcg = float((y[:k] / np.log2(np.arange(min(k, n)) + 2)).sum())
        ideal = float((1.0 / np.log2(np.arange(min(k, n_pos)) + 2)).sum())
        return dcg / ideal if ideal > 0 else 0.0

    return {"auc": float(auc), "mrr": mrr, "ndcg5": ndcg(5), "ndcg10": ndcg(10)}


def selftest():
    """Tiny known-answer checks so a metric bug can't silently corrupt every result."""
    m = all_metrics([0, 1, 2], [1, 0, 0])          # perfect
    assert abs(m["auc"] - 1) < 1e-9 and abs(m["mrr"] - 1) < 1e-9 and abs(m["ndcg10"] - 1) < 1e-9
    m = all_metrics([1, 2, 0], [1, 0, 0])          # worst
    assert abs(m["auc"] - 0) < 1e-9 and abs(m["mrr"] - 1 / 3) < 1e-9
    assert abs(m["ndcg10"] - (1 / math.log2(4))) < 1e-9
    m = all_metrics([0, 1, 2, 3], [0, 1, 0, 1])    # positives at ranks 2 and 4
    assert abs(m["auc"] - 0.25) < 1e-9             # rank-2 positive beats 1 of 2 negatives, rank-4 beats none: 1/4
    assert abs(m["mrr"] - (1 / 2 + 1 / 4) / 2) < 1e-9
    assert all_metrics([0, 1], [1, 1]) is None     # no negatives -> undefined, skipped upstream
    return True


# ------------------------------------------------------------------ stage-1 blend (shared by evaluate + LLM script)
BLEND_PATH = os.path.join(OUT_DIR, "blend_lambda.json")
CHOSEN_PATH = os.path.join(OUT_DIR, "chosen_variant.json")


def llm_path(split, variant):
    return os.path.join(OUT_DIR, f"llm_rerank_{split}_{variant}.jsonl")


def llm_full_order(stage1, llm_top):
    """LLM's candidates first, then every remaining candidate in the blend's order."""
    top = [int(c) for c in llm_top]
    top_set = set(top)
    return np.array(top + [int(i) for i in stage1 if int(i) not in top_set])


def stage1_order(sims, pop, lam):
    """Deterministic ranking by z(similarity) + lam * z(popularity). Used as the LLM's shortlist AND as the blend system."""
    score = zscore(sims) + lam * zscore(pop)
    return np.argsort(-score, kind="stable")


# ------------------------------------------------------------------ statistics
def _boot_means(x, n_boot, seed, chunk=500):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    out = []
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        idx = rng.integers(0, len(x), size=(c, len(x)))
        out.append(x[idx].mean(axis=1))
        done += c
    return np.concatenate(out)


def bootstrap_ci(x, n_boot=10000, seed=SEED):
    """(mean, lo, hi): 95% percentile bootstrap CI of the mean."""
    means = _boot_means(x, n_boot, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(x)), float(lo), float(hi)
