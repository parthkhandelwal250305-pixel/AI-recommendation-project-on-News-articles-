"""
Stage 2: local LLM reranks the top-15 candidates of the BEST CHEAP ranker (the
embedding + popularity blend), conditioned on the user's recent click history
(titles). The question: does an LLM step add value ON TOP OF the best cheap system?

The LLM picks its PICK_N (=5) favourites; every other candidate keeps the blend's
order. (A first pilot asked for a full 15-item ordering and the 3B model showed strong
position bias and lazy runs like 15,14,13,...; picking a few is a task small models
do better, and it leaves the tail to the blend, which has real signal.)

Two prompt variants, compared on the TUNING split only (hist_prompt_check.py picks one):
  pick5_shuffled : candidates shown in random order -> measures the LLM's own judgment
  pick5_ranked   : candidates shown in the blend's order, with a note saying so -> the
                   realistic "reranker sees stage-1 order" setup (may mostly copy it)
The TEST split can only be run with the variant chosen on the tune split. This prevents
trying prompts on the test cases until one looks good (forking paths).

Other choices: temperature 0 + fixed seed; unparseable output -> fall back to the blend's
order (logged); fewer than 5 valid picks -> the rest filled from the blend (logged);
titles only (no abstracts) to keep prompts short on a CPU. RESUMABLE: one line is
appended per case; Ctrl+C is safe and re-running continues.

Requires Ollama running locally:  ollama pull qwen2.5:3b

Usage:
    python src/hist_llm_rerank.py --split tune --variant pick5_shuffled --n 30
    python src/hist_llm_rerank.py --split tune --variant pick5_ranked   --n 30
    python src/hist_prompt_check.py                      # chooses the variant
    python src/hist_llm_rerank.py --n 150                # TEST split, chosen variant
"""
import os
import json
import time
import zlib
import argparse

import numpy as np
import pandas as pd

from hist_common import (
    OUT_DIR, SEED, HISTORY_N, RERANK_K, PICK_N, VARIANTS, BLEND_PATH, CHOSEN_PATH,
    stage1_order, llm_path,
)

LLM_MODEL_NAME = "qwen2.5:3b"

HEADER = """You are a news recommendation engine. Below are articles a reader recently clicked \
(oldest to newest), followed by {k} candidate articles. Choose the {p} candidates this reader is \
MOST likely to click next, most likely first.
"""
BODY = """
Recently clicked:
{history}

Candidates:
{candidates}

Respond with ONLY a JSON array of exactly {p} candidate numbers, for example [7, 12, 3, 9, 14]. \
No other text."""
RANKED_NOTE = """The candidates are listed in order of a baseline score that combines recent popularity \
and similarity to the reader's history. Keep that order unless the reader's clicks give a clear reason \
to prefer a different article.
"""


def build_prompt(variant, hist_titles, cand_titles):
    header = HEADER.format(k=RERANK_K, p=PICK_N)
    if variant == "pick5_ranked":
        header += RANKED_NOTE
    return header + BODY.format(
        history="\n".join(f"- {t}" for t in hist_titles),
        candidates="\n".join(f"[{i + 1}] {t}" for i, t in enumerate(cand_titles)),
        p=PICK_N,
    )


def parse_picks(text, k, p):
    """First p unique ints in 1..k from a JSON array in the text, or None."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    seen, out = set(), []
    for x in arr:
        if isinstance(x, int) and 1 <= x <= k and x not in seen:
            seen.add(x)
            out.append(x)
    return out[:p] or None


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "tune"], default="test")
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--n", type=int, default=150, help="total number of cases to have results for")
    args = ap.parse_args()

    # ---- which prompt variant? (test split may only use the one chosen on tune)
    if args.split == "tune":
        if args.variant is None:
            raise SystemExit(f"Tune runs need --variant, one of: {VARIANTS}")
        variant = args.variant
    else:
        if not os.path.exists(CHOSEN_PATH):
            raise SystemExit("No prompt variant has been chosen yet. Run the two tune-split runs and then "
                             "`python src/hist_prompt_check.py` first.")
        variant = json.load(open(CHOSEN_PATH))["variant"]
        if args.variant not in (None, variant):
            raise SystemExit(f"The variant chosen on the tune split is '{variant}'. The test split must use "
                             f"it (you passed '{args.variant}').")

    if not os.path.exists(BLEND_PATH):
        raise SystemExit("Missing blend_lambda.json. Run `python src/hist_evaluate.py` first.")
    lam = float(json.load(open(BLEND_PATH))["lambda"])
    out_path = llm_path(args.split, variant)
    print(f"Split: {args.split} | variant: {variant} | stage 1 = blend (lambda={lam}); LLM picks {PICK_N} of top {RERANK_K}.")

    existing = read_jsonl(out_path)
    if existing and existing[0].get("stage1_lambda") != lam:
        raise SystemExit(f"{out_path} was produced with a different stage-1 ranker. Delete it to start over.")

    import ollama

    with open(os.path.join(OUT_DIR, "eval_set.json")) as f:
        impressions = json.load(f)["impressions"]
    cases = sorted([i for i in impressions if i["split"] == args.split], key=lambda i: i["rank_in_split"])
    emb = json.load(open(os.path.join(OUT_DIR, "emb_scores.json")))["scores"]
    arts = pd.read_csv(os.path.join(OUT_DIR, "articles.csv"), keep_default_na=False)
    title = dict(zip(arts["news_id"], arts["title"]))

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
            sims = np.array(emb[iid][f"N{HISTORY_N}"])
            stage1 = stage1_order(sims, imp["pop"], lam)
            top = [int(c) for c in stage1[:RERANK_K]]          # shortlist (candidate indices, blend order)

            if variant == "pick5_shuffled":
                rng = np.random.default_rng([SEED, zlib.crc32(iid.encode())])
                display = [top[j] for j in rng.permutation(RERANK_K)]
            else:
                display = list(top)                              # display number p+1 == blend rank p+1

            prompt = build_prompt(
                variant,
                [title[h] for h in imp["history"][-HISTORY_N:]],
                [title[imp["candidates"][c]] for c in display],
            )

            t0 = time.time()
            try:
                resp = ollama.generate(
                    model=LLM_MODEL_NAME, prompt=prompt, keep_alive="30m",
                    options={"temperature": 0, "seed": SEED, "num_predict": 60, "num_ctx": 4096},
                )
            except Exception as e:
                raise SystemExit(f"\nCould not call Ollama ({e}).\nIs it running and is `{LLM_MODEL_NAME}` pulled?")
            latency = time.time() - t0
            raw = resp["response"].strip()

            picks = parse_picks(raw, RERANK_K, PICK_N)
            used_fallback = picks is None
            llm_top = [] if used_fallback else [display[p - 1] for p in picks]
            n_returned = len(llm_top)
            order = list(llm_top) + [c for c in top if c not in llm_top]   # 15 candidate indices
            n_padded = 0 if used_fallback else PICK_N - n_returned

            out.write(json.dumps({
                "impression_id": iid, "stage1_lambda": lam, "variant": variant,
                "order": [int(c) for c in order], "picks": picks or [],
                "used_fallback": used_fallback, "n_returned": n_returned, "n_padded": n_padded,
                "latency_seconds": round(latency, 2), "raw": raw[:200],
            }) + "\n")
            out.flush()

            latencies.append(latency)
            avg = float(np.mean(latencies))
            eta = avg * (len(todo) - n_done) / 60
            tag = " FALLBACK" if used_fallback else (f" only {n_returned} picks" if n_padded else "")
            print(f"[{n_done}/{len(todo)}] {latency:5.1f}s  avg {avg:5.1f}s  ETA {eta:5.1f} min  picks={picks}{tag}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Saved: {out_path}")


if __name__ == "__main__":
    main()
