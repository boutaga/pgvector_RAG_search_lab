#!/usr/bin/env python3
"""
Lab 06 — 02_create_embeddings_dense.py
Generate dense embeddings with OpenAI text-embedding-3-large (3072d).
After embedding, populate halfvec and binary_quantize columns via SQL CAST.
Resume-safe: only processes rows where the vector column IS NULL.
"""

import sys
import time

import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector

import config


def get_batch_embeddings(client, texts, max_retries=5):
    """Retrieve embeddings for a batch of texts with exponential backoff."""
    delay = 0.2
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=config.EMBEDDING_MODEL,
                dimensions=config.EMBEDDING_DIMENSION,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                print(f"  Rate limit hit. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise Exception("Max retries exceeded for batch embedding request.")


def update_embeddings(conn, embedding_type="both"):
    """
    Generate embeddings for articles.
    embedding_type: 'title', 'content', or 'both'
    """
    if embedding_type == "both":
        update_embeddings(conn, "title")
        update_embeddings(conn, "content")
        return

    cur = conn.cursor()

    if embedding_type == "title":
        cur.execute(
            "SELECT id, title FROM articles "
            "WHERE title_vector IS NULL AND title IS NOT NULL AND title != '' "
            "ORDER BY id;"
        )
        column = "title_vector"
    else:
        cur.execute(
            "SELECT id, content FROM articles "
            "WHERE content_vector IS NULL AND content IS NOT NULL AND content != '' "
            "ORDER BY id;"
        )
        column = "content_vector"

    rows = cur.fetchall()
    if not rows:
        print(f"No articles need {embedding_type} embeddings.")
        cur.close()
        return

    # Build batches
    batches = []
    batch_ids, batch_texts = [], []
    for article_id, text in rows:
        if not text or not text.strip():
            continue
        # Truncate very long content to stay within token limits
        if embedding_type == "content" and len(text) > 32000:
            text = text[:32000] + "..."
        batch_ids.append(article_id)
        batch_texts.append(text)
        if len(batch_texts) >= config.BATCH_SIZE_DENSE:
            batches.append((list(batch_ids), list(batch_texts)))
            batch_ids, batch_texts = [], []
    if batch_texts:
        batches.append((batch_ids, batch_texts))

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    total = sum(len(ids) for ids, _ in batches)
    print(f"Processing {total} {embedding_type} embeddings in {len(batches)} batches...")

    processed = 0
    for i, (ids, texts) in enumerate(batches):
        try:
            embeddings = get_batch_embeddings(client, texts)
            for article_id, embedding in zip(ids, embeddings):
                cur.execute(
                    f"UPDATE articles SET {column} = %s WHERE id = %s;",
                    (embedding, article_id),
                )
            conn.commit()
            processed += len(ids)
            pct = 100 * processed / total
            print(f"  Batch {i+1}/{len(batches)}: {len(ids)} rows. Progress: {processed}/{total} ({pct:.1f}%)")
            time.sleep(0.1)
        except Exception as e:
            print(f"  ERROR batch {i+1} (IDs {ids[:3]}...): {e}")
            conn.rollback()
            continue

    cur.close()
    print(f"Completed {embedding_type} embeddings: {processed}/{total}")


def populate_derived_columns(conn):
    """Populate halfvec and binary_quantize columns from the dense vectors."""
    cur = conn.cursor()

    # halfvec columns
    print("Populating title_halfvec...")
    cur.execute(
        "UPDATE articles SET title_halfvec = title_vector::halfvec "
        "WHERE title_vector IS NOT NULL AND title_halfvec IS NULL;"
    )
    print(f"  Updated {cur.rowcount} rows.")

    print("Populating content_halfvec...")
    cur.execute(
        "UPDATE articles SET content_halfvec = content_vector::halfvec "
        "WHERE content_vector IS NOT NULL AND content_halfvec IS NULL;"
    )
    print(f"  Updated {cur.rowcount} rows.")

    # binary_quantize
    print("Populating content_bq (binary_quantize)...")
    cur.execute(
        "UPDATE articles SET content_bq = binary_quantize(content_vector)::bit(3072) "
        "WHERE content_vector IS NOT NULL AND content_bq IS NULL;"
    )
    print(f"  Updated {cur.rowcount} rows.")

    conn.commit()
    cur.close()


def verify(conn):
    """Print embedding coverage stats."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(title_vector) AS title_dense,
            COUNT(content_vector) AS content_dense,
            COUNT(title_halfvec) AS title_hv,
            COUNT(content_halfvec) AS content_hv,
            COUNT(content_bq) AS content_bq
        FROM articles;
    """)
    row = cur.fetchone()
    labels = ["total", "title_dense", "content_dense", "title_halfvec", "content_halfvec", "content_bq"]
    print("\nEmbedding coverage:")
    for label, val in zip(labels, row):
        print(f"  {label}: {val}")
    cur.close()


def main():
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Add it to DEV/.env or export it.")
        sys.exit(1)

    print(f"Connecting to {config.DATABASE_URL}")
    print(f"Model: {config.EMBEDDING_MODEL}, Dimensions: {config.EMBEDDING_DIMENSION}")

    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)

    update_embeddings(conn, "both")
    populate_derived_columns(conn)
    verify(conn)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
