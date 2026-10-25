"""
EQUAL-INFORMATION experiment: does adding an LLM's judgment to the blend help when the LLM is given
everything the blend has?

The LLM picks the 5 of the blend's top-15 candidates THIS reader is most likely to click, most likely first.
The picks become scores (5, 4, 3, 2, 1; everything else 0) that are combined with the blend by a weight
chosen on tuning cases (see hist_score_check.py). A weight of 0 reproduces the blend exactly, so adding the
LLM can only help, or be switched off by the tuning step.

DESIGN REVISION (recorded openly): the first version asked the model to give every candidate a 0-10 score.
On the first 6 tuning cases it answered all zeros in 4 of them, i.e. no usable judgment. Picking a few
favourites is a task this model did reliably in the earlier experiment (0% unparseable output), so the
output format was changed BEFORE any test case was run. The information given to the model is unchanged.

Two prompt variants (compared on the TUNING split only):
  content : the reader's last 20 clicked titles + each candidate's title and abstract snippet.
            Candidates shown in a seeded RANDOM order, so the LLM cannot lean on the blend's order.
            An independent, content-based opinion.
  full    : everything in 'content' PLUS what the blend used: each candidate's estimated click rate and
            its similarity to the reader's history (value and rank among the 15), and the candidates
            are listed in the blend's order. A superset of the blend's information.

The TEST split can only be run with the variant chosen on the tune split.
Unparseable output (not exactly 15 numbers) -> logged as a fallback; that case then equals the blend.
RESUMABLE: one line per case is appended; Ctrl+C is safe; re-running continues.

Requires Ollama running locally:  ollama pull qwen2.5:3b

Usage:
    python src/hist_llm_score.py --split tune --variant content --n 60
    python src/hist_llm_score.py --split tune --variant full    --n 60
    python src/hist_score_check.py                  # chooses variant + weight on tuning cases only
    python src/hist_llm_score.py --n 300            # TEST split, chosen variant
"""
import os
import json
import time
import zlib
import argparse

import numpy as np
import pandas as pd

from hist_common import (
    OUT_DIR, SEED, HISTORY_N, RERANK_K, PICK_N, BLEND_PATH, SCORE_VARIANTS, CHOSEN_SCORE_PATH,
    blend_scores, score_path,
)

LLM_MODEL_NAME = "qwen2.5:3b"
ABSTRACT_WORDS = 40
EXAMPLE = "[7, 12, 3, 9, 14]"
assert len(json.loads(EXAMPLE)) == PICK_N
FORMAT = "pick5_scores_v2"   # recorded in every result line; runs with a different format are never mixed

INTRO = "You are helping rank news articles for one reader.\n\nArticles the reader recently clicked (oldest to newest):\n{history}\n\n"
TAIL_CONTENT = ("Choose the {p} candidates this reader is MOST likely to click next, judging by how well each matches "
                "what they have been reading. List the most likely first.\n")
TAIL_FULL = ("Choose the {p} candidates this reader is MOST likely to click next. Use the reader's history and the "
             "measurements. List the most likely first.\n")
ASK = "Respond with ONLY a JSON array of exactly {p} candidate numbers, most likely first, for example {ex}. No other text."


def snippet(text, n=ABSTRACT_WORDS):
    words = (text or "").split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def ranks(values):
    """1 = highest."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    r = np.empty(len(order), dtype=int)
    r[order] = np.arange(1, len(order) + 1)
    return r


def build_prompt(variant, hist_titles, rows):
    """rows: list of dicts in DISPLAY order with title, abstract, pop, sim, pop_rank, sim_rank."""
    history = "\n".join(f"- {t}" for t in hist_titles)
    if variant == "full":
        head = ("Candidate articles. They are listed in order of a baseline score that combines recent popularity and "
                "similarity to this reader's history. Under each one are two measurements: its estimated click rate "
                "(how often readers click it when it is shown) and its similarity to what this reader has clicked "
                "before. Ranks are among these 15 candidates (1 = highest).\n\n")
        lines = []
        for i, r in enumerate(rows, 1):
            lines.append(f"[{i}] {r['title']}: {snippet(r['abstract'])}\n"
                         f"    click rate {r['pop'] * 100:.1f}% (rank {r['pop_rank']} of {len(rows)}); "
                         f"similarity {r['sim']:.2f} (rank {r['sim_rank']} of {len(rows)})")
        tail = TAIL_FULL
    else:
        head = "Candidate articles:\n\n"
        lines = [f"[{i}] {r['title']}: {snippet(r['abstract'])}" for i, r in enumerate(rows, 1)]
        tail = TAIL_CONTENT
    return (INTRO.format(history=history) + head + "\n".join(lines) + "\n\n" + tail.format(p=PICK_N) +
            ASK.format(p=PICK_N, ex=EXAMPLE))


def parse_picks(text, k, p):
    """First p unique candidate numbers (1..k) from a JSON array in the text, or None."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    seen, out = set(), []
    for x in arr:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= k and x not in seen:
            seen.add(x)
            out.append(x)
    return out[:p] or None


def _field(resp, name):
    try:
        return resp[name]
    except Exception:
        return None


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "tune"], default="test")
    ap.add_argument("--variant", choices=SCORE_VARIANTS, default=None)
    ap.add_argument("--n", type=int, default=300, help="total number of cases to have results for")
    args = ap.parse_args()

    if args.split == "tune":
        if args.variant is None:
            raise SystemExit(f"Tune runs need --variant, one of: {SCORE_VARIANTS}")
        variant = args.variant
    else:
        if not os.path.exists(CHOSEN_SCORE_PATH):
            raise SystemExit("No variant has been chosen yet. Run both tune-split runs, then "
                             "`python src/hist_score_check.py`, first.")
        variant = json.load(open(CHOSEN_SCORE_PATH))["variant"]
        if args.variant not in (None, variant):
            raise SystemExit(f"The variant chosen on the tune split is '{variant}'; the test split must use it.")

    if not os.path.exists(BLEND_PATH):
        raise SystemExit("Missing blend_lambda.json. Run `python src/hist_evaluate.py` first.")
    lam = float(json.load(open(BLEND_PATH))["lambda"])
    out_path = score_path(args.split, variant)
    print(f"Split: {args.split} | variant: {variant} | stage 1 = blend (lambda={lam}); LLM scores the top {RERANK_K}.")

    existing = read_jsonl(out_path)
    if existing and existing[0].get("stage1_lambda") != lam:
        raise SystemExit(f"{out_path} was produced with a different stage-1 ranker. Delete it to start over.")
    if existing and existing[0].get("format") != FORMAT:
        raise SystemExit(f"{out_path} was produced with an older output format. Delete it and re-run (do not mix formats).")

    import ollama

    imps = json.load(open(os.path.join(OUT_DIR, "eval_set.json")))["impressions"]
    cases = sorted([i for i in imps if i["split"] == args.split], key=lambda i: i["rank_in_split"])
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]
    arts = pd.read_csv(os.path.join(OUT_DIR, "articles.csv"), keep_default_na=False)
    title = dict(zip(arts["news_id"], arts["title"]))
    abstract = dict(zip(arts["news_id"], arts["abstract"]))

    target = cases[: args.n]
    done = {r["impression_id"] for r in existing}
    todo = [i for i in target if i["impression_id"] not in done]
    print(f"Target {len(target)} cases; already done {len(target) - len(todo)}; to run now: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    latencies = []
    t_start = time.time()
    with open(out_path, "a") as out:
        for n_done, imp in enumerate(todo, 1):
            iid = imp["impression_id"]
            sims_all = np.array(emb[iid][f"N{HISTORY_N}"])
            pop_all = np.array(imp["pop"])
            sc = blend_scores(sims_all, pop_all, lam)
            stage1 = np.argsort(-sc, kind="stable")
            top = [int(c) for c in stage1[:RERANK_K]]                    # candidate indices, in blend order

            sims15, pops15 = sims_all[top], pop_all[top]
            sim_rank, pop_rank = ranks(sims15), ranks(pops15)
            if variant == "content":
                rng = np.random.default_rng([SEED, zlib.crc32(iid.encode())])
                display_pos = list(rng.permutation(RERANK_K))             # display slot j shows top[display_pos[j]]
            else:
                display_pos = list(range(RERANK_K))
            rows = []
            for j in display_pos:
                cid = imp["candidates"][top[j]]
                rows.append({"title": title[cid], "abstract": abstract[cid], "pop": float(pops15[j]),
                             "sim": float(sims15[j]), "pop_rank": int(pop_rank[j]), "sim_rank": int(sim_rank[j])})

            prompt = build_prompt(variant, [title[h] for h in imp["history"][-HISTORY_N:]], rows)
            t0 = time.time()
            try:
                resp = ollama.generate(
                    model=LLM_MODEL_NAME, prompt=prompt, keep_alive="30m",
                    options={"temperature": 0, "seed": SEED, "num_predict": 120, "num_ctx": 4096},
                )
            except Exception as e:
                raise SystemExit(f"\nCould not call Ollama ({e}).\nIs it running and is `{LLM_MODEL_NAME}` pulled?")
            latency = time.time() - t0
            raw = resp["response"].strip()
            p_tok, g_tok, p_ns = _field(resp, "prompt_eval_count"), _field(resp, "eval_count"), _field(resp, "prompt_eval_duration")

            picks = parse_picks(raw, RERANK_K, PICK_N)
            used_fallback = picks is None
            scores_top = None
            if not used_fallback:                                         # picks -> scores, aligned to blend order
                scores_top = [0.0] * RERANK_K
                for q, num in enumerate(picks):
                    scores_top[display_pos[num - 1]] = float(PICK_N - q)

            out.write(json.dumps({
                "impression_id": iid, "stage1_lambda": lam, "variant": variant, "format": FORMAT,
                "top": top, "scores": scores_top, "picks": picks or [], "n_returned": len(picks or []),
                "used_fallback": used_fallback,
                "latency_seconds": round(latency, 2), "prompt_tokens": p_tok, "gen_tokens": g_tok,
                "prompt_seconds": round(p_ns / 1e9, 2) if p_ns else None, "raw": raw[:200],
            }) + "\n")
            out.flush()

            latencies.append(latency)
            avg = float(np.mean(latencies))
            eta = avg * (len(todo) - n_done) / 60
            tag = " FALLBACK" if used_fallback else (f"  picks={picks}" + (f"  (only {len(picks)})" if len(picks) < PICK_N else ""))
            print(f"[{n_done}/{len(todo)}] {latency:5.1f}s  avg {avg:5.1f}s  ETA {eta:5.1f} min{tag}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Saved: {out_path}")


if __name__ == "__main__":
    main()
