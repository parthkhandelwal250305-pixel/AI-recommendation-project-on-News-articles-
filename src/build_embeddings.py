"""
Compute sentence-transformer embeddings for the content pool.

Uses all-MiniLM-L6-v2: small enough to run fast on CPU, no GPU required.

Usage:
    python src/build_embeddings.py
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def article_text(row) -> str:
    title = row["title"] if isinstance(row["title"], str) else ""
    abstract = row["abstract"] if isinstance(row["abstract"], str) else ""
    return f"{title}. {abstract}".strip()


def main():
    pool_path = os.path.join(PROCESSED_DIR, "content_pool.csv")
    df = pd.read_csv(pool_path)
    texts = df.apply(article_text, axis=1).tolist()

    print(f"Loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} articles ...")
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
    )

    emb_path = os.path.join(PROCESSED_DIR, "article_embeddings.npy")
    ids_path = os.path.join(PROCESSED_DIR, "article_ids.npy")
    np.save(emb_path, embeddings)
    np.save(ids_path, df["news_id"].values)

    print(f"Saved embeddings: {emb_path} {embeddings.shape}")
    print(f"Saved ids: {ids_path}")


if __name__ == "__main__":
    main()
