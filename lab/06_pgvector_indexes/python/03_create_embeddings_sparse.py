#!/usr/bin/env python3
"""
Lab 06 — 03_create_embeddings_sparse.py
Generate sparse embeddings with SPLADE (naver/splade-cocondenser-ensembledistil).
Resume-safe: only processes rows where the sparse column IS NULL.
"""

import sys
import time

import psycopg2
import torch
from pgvector.psycopg2 import register_vector
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    """Load SPLADE model and tokenizer."""
    print(f"Loading SPLADE model: {config.SPLADE_MODEL}")
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(config.SPLADE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(config.SPLADE_MODEL)
    model.to(DEVICE)
    model.eval()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer, model


def get_sparse_embedding(tokenizer, model, text):
    """Generate a sparse embedding string for pgvector sparsevec."""
    if len(text) > 2000:
        text = text[:2000] + "..."

    tokens = tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens).logits

    sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1).values
    vec = sparse_weights[0]

    indices = vec.nonzero().squeeze().cpu()
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    values = vec[indices].cpu()

    if len(indices) == 0:
        return f"{{}}/{tokenizer.vocab_size}"

    pairs = [f"{idx}:{val:.6f}" for idx, val in zip(indices.tolist(), values.tolist())]
    return "{" + ",".join(pairs) + f"}}/{tokenizer.vocab_size}"


def update_sparse_embeddings(conn, tokenizer, model, embedding_type="both"):
    """Generate sparse embeddings for articles."""
    if embedding_type == "both":
        update_sparse_embeddings(conn, tokenizer, model, "title")
        update_sparse_embeddings(conn, tokenizer, model, "content")
        return

    cur = conn.cursor()

    if embedding_type == "title":
        cur.execute(
            "SELECT id, title FROM articles "
            "WHERE title_sparse IS NULL AND title IS NOT NULL AND title != '' "
            "ORDER BY id;"
        )
        column = "title_sparse"
    else:
        cur.execute(
            "SELECT id, content FROM articles "
            "WHERE content_sparse IS NULL AND content IS NOT NULL AND content != '' "
            "ORDER BY id;"
        )
        column = "content_sparse"

    rows = cur.fetchall()
    if not rows:
        print(f"No articles need {embedding_type} sparse embeddings.")
        cur.close()
        return

    # Build batches
    batches = []
    batch_ids, batch_texts = [], []
    for article_id, text in rows:
        if not text or not text.strip():
            continue
        batch_ids.append(article_id)
        batch_texts.append(text)
        if len(batch_texts) >= config.BATCH_SIZE_SPARSE:
            batches.append((list(batch_ids), list(batch_texts)))
            batch_ids, batch_texts = [], []
    if batch_texts:
        batches.append((batch_ids, batch_texts))

    total = sum(len(ids) for ids, _ in batches)
    print(f"Processing {total} {embedding_type} sparse embeddings in {len(batches)} batches...")

    processed = 0
    for i, (ids, texts) in enumerate(batches):
        try:
            sparse_vecs = [get_sparse_embedding(tokenizer, model, t) for t in texts]
            for article_id, svec in zip(ids, sparse_vecs):
                cur.execute(
                    f"UPDATE articles SET {column} = %s WHERE id = %s;",
                    (svec, article_id),
                )
            conn.commit()
            processed += len(ids)
            pct = 100 * processed / total
            print(f"  Batch {i+1}/{len(batches)}: {len(ids)} rows. Progress: {processed}/{total} ({pct:.1f}%)")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR batch {i+1} (IDs {ids[:3]}...): {e}")
            conn.rollback()
            continue

    cur.close()
    print(f"Completed {embedding_type} sparse embeddings: {processed}/{total}")


def verify(conn):
    """Print sparse embedding coverage stats."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(title_sparse) AS title_sparse,
            COUNT(content_sparse) AS content_sparse
        FROM articles;
    """)
    total, ts, cs = cur.fetchone()
    print(f"\nSparse embedding coverage:")
    print(f"  total: {total}")
    print(f"  title_sparse: {ts}")
    print(f"  content_sparse: {cs}")
    cur.close()


def main():
    print(f"Connecting to {config.DATABASE_URL}")
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)

    tokenizer, model = initialize_model()

    update_sparse_embeddings(conn, tokenizer, model, "both")
    verify(conn)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
