"""
Stage 1: semantic retrieval (embeddings + cosine similarity) -> top 20-50
Stage 2: local LLM reranks those candidates conditioned on the persona -> top 10

Requires Ollama running locally with the model pulled:
    ollama pull qwen2.5:3b

Usage:
    python src/run_retrieval_rerank.py
"""
import os
import json
import time

import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "personas.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "qwen2.5:3b"
RETRIEVAL_K = 30
RERANK_K = 10

RERANK_PROMPT_TEMPLATE = """You are ranking news articles for a specific reader persona.

Persona: {persona_name}
Persona interests: {persona_description}

Below are {n} candidate articles, each with an ID and title. Choose the {k} \
that this persona is most likely to click on, and order them from most to \
least likely.

Consider:
- Topical fit with the persona's stated interests
- Avoid recommending near-duplicate stories back to back
- Avoid over-narrowing to a single subtopic if the persona has broad interests

Candidates:
{candidate_list}

Respond with ONLY a JSON array of exactly {k} article IDs, ordered from most \
to least recommended, and nothing else. Example format:
["N12345", "N67890", ...]
"""


def article_text(row) -> str:
    title = row["title"] if isinstance(row["title"], str) else ""
    abstract = row["abstract"] if isinstance(row["abstract"], str) else ""
    return f"{title}. {abstract}".strip()


def retrieve_candidates(query_emb, article_embs, article_ids, k=RETRIEVAL_K):
    sims = article_embs @ query_emb  # embeddings are normalized -> cosine sim
    top_idx = np.argsort(-sims)[:k]
    return [article_ids[i] for i in top_idx], sims[top_idx]


def build_prompt(persona, candidates_df, k=RERANK_K):
    lines = [f'{row["news_id"]}: {row["title"]}' for _, row in candidates_df.iterrows()]
    candidate_list = "\n".join(lines)
    return RERANK_PROMPT_TEMPLATE.format(
        persona_name=persona["name"],
        persona_description=persona["description"],
        n=len(candidates_df),
        k=k,
        candidate_list=candidate_list,
    )


def call_llm_rerank(prompt: str, valid_ids: set, k=RERANK_K) -> list:
    response = ollama.generate(model=LLM_MODEL_NAME, prompt=prompt, options={"temperature": 0.2})
    text = response["response"].strip()

    # Be forgiving about the model wrapping the array in prose or code fences.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        ids = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    ids = [i for i in ids if i in valid_ids]
    # De-duplicate while preserving order, in case the model repeats an ID.
    seen = set()
    deduped = []
    for i in ids:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    return deduped[:k] if deduped else None


def main():
    pool_df = pd.read_csv(os.path.join(PROCESSED_DIR, "content_pool.csv"))
    article_embs = np.load(os.path.join(PROCESSED_DIR, "article_embeddings.npy"))
    article_ids = np.load(os.path.join(PROCESSED_DIR, "article_ids.npy"), allow_pickle=True)
    id_to_row = pool_df.set_index("news_id")

    with open(CONFIG_PATH) as f:
        personas = json.load(f)["personas"]

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    results = {}
    retrieval_log = {}
    fallback_count = 0

    for persona in personas:
        pid = persona["id"]
        # Category labels are deliberately NOT included (they define ground truth).
        query_text = f"{persona['name']}. {persona['description']}"
        query_emb = embed_model.encode(query_text, normalize_embeddings=True)

        candidate_ids, sims = retrieve_candidates(query_emb, article_embs, article_ids)
        candidates_df = id_to_row.loc[candidate_ids].reset_index()

        prompt = build_prompt(persona, candidates_df)
        t0 = time.time()
        reranked = call_llm_rerank(prompt, valid_ids=set(candidate_ids))
        latency = time.time() - t0

        used_fallback = reranked is None
        if used_fallback:
            # LLM output didn't parse -- fall back to retrieval order rather
            # than crash. This fallback rate itself is a metric worth
            # reporting: an unreliable rerank step is part of the cost story.
            reranked = list(candidate_ids[:RERANK_K])
            fallback_count += 1
            print(f"[{pid}] LLM output unparseable, used retrieval-order fallback")

        # If the LLM returned fewer than 10 valid IDs, pad from retrieval order
        # so every system is scored on a full 10 slots; record how many were padded.
        llm_returned_n = 0 if used_fallback else len(reranked)
        n_padded = 0
        if not used_fallback and len(reranked) < RERANK_K:
            for cid in candidate_ids:
                if len(reranked) >= RERANK_K:
                    break
                if cid not in reranked:
                    reranked.append(cid)
                    n_padded += 1

        results[pid] = reranked
        retrieval_log[pid] = {
            "retrieval_candidates": list(candidate_ids),
            "rerank_latency_seconds": round(latency, 2),
            "used_fallback": used_fallback,   # True only if the LLM output could not be parsed
            "llm_returned_n": llm_returned_n,
            "n_padded_from_retrieval": n_padded,
        }
        print(f"[{pid}] {persona['name']}: reranked top 10 in {latency:.2f}s")

    out_path = os.path.join(PROCESSED_DIR, "rerank_feed.json")
    log_path = os.path.join(PROCESSED_DIR, "rerank_log.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w") as f:
        json.dump(retrieval_log, f, indent=2)

    print(f"\nFallback rate: {fallback_count}/{len(personas)} personas")
    print(f"Saved: {out_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
