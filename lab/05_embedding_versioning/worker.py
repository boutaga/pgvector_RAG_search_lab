#!/usr/bin/env python3
"""
Lab 5 — Embedding Worker (SKIP LOCKED pattern)

Picks items from embedding_queue using SELECT FOR UPDATE SKIP LOCKED,
generates embeddings via OpenAI, and stores versioned results.

Usage:
    python worker.py                        # single worker
    python worker.py --workers 4            # multi-process mode
    python worker.py --batch-size 20        # process 20 items per cycle
    python worker.py --model text-embedding-3-small
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import psycopg
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 2000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks
MAX_RETRIES = 3
POLL_INTERVAL = 5       # seconds between polls when queue is empty
STALE_PROCESSING_MINUTES = 10  # recover items stuck in 'processing' longer than this

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    log.info("Shutdown signal received, finishing current batch...")
    shutdown_requested = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_article(article_id: int, content: str, content_hash: str,
                    model: str, conn) -> int:
    """Chunk, embed, and store versioned embeddings for one article.

    Returns the number of chunks embedded.
    """
    chunks = chunk_text(content)
    if not chunks:
        log.warning("Article %d has no content to embed", article_id)
        return 0

    embeddings = generate_embeddings(chunks, model=model)

    with conn.cursor() as cur:
        # Mark old embeddings as non-current
        cur.execute("""
            UPDATE article_embeddings_versioned
            SET is_current = false, replaced_at = now()
            WHERE article_id = %s AND model_version = %s AND is_current = true
        """, (article_id, model))

        # Insert new embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute("""
                INSERT INTO article_embeddings_versioned
                    (article_id, chunk_index, chunk_text, embedding,
                     model_version, source_hash, is_current)
                VALUES (%s, %s, %s, %s::vector, %s, %s, true)
            """, (article_id, idx, chunk, json.dumps(emb), model, content_hash))

    return len(chunks)


def recover_stale_processing(db_url: str):
    """Reset items stuck in 'processing' state (e.g. after worker crash).

    Items older than STALE_PROCESSING_MINUTES are returned to 'pending'.
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE embedding_queue
                SET status = 'pending', worker_id = NULL, started_at = NULL
                WHERE status = 'processing'
                  AND started_at < now() - (%s * interval '1 minute')
                RETURNING id, article_id
            """, (STALE_PROCESSING_MINUTES,))
            recovered = cur.fetchall()
        conn.commit()

    if recovered:
        log.warning("Recovered %d stuck items: %s",
                    len(recovered), [r[0] for r in recovered])
    return len(recovered)


def fetch_and_process(db_url: str, batch_size: int, model: str, worker_id: str) -> int:
    """Fetch a batch from the queue and process each item.

    Returns the number of items processed.
    """
    processed = 0

    with psycopg.connect(db_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Claim items with SKIP LOCKED
            cur.execute("""
                UPDATE embedding_queue
                SET status = 'processing',
                    started_at = now(),
                    worker_id = %s
                WHERE id IN (
                    SELECT id FROM embedding_queue
                    WHERE status = 'pending'
                    ORDER BY priority, queued_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                RETURNING id, article_id, content_hash
            """, (worker_id, batch_size))

            items = cur.fetchall()
            conn.commit()

        if not items:
            return 0

        log.info("Worker %s claimed %d items", worker_id, len(items))

        for queue_id, article_id, queued_hash in items:
            try:
                with conn.cursor() as cur:
                    # Fetch current article content
                    cur.execute(
                        "SELECT content, content_hash FROM articles WHERE id = %s",
                        (article_id,),
                    )
                    row = cur.fetchone()

                if row is None:
                    mark_completed(conn, queue_id, "skipped", "Article not found")
                    continue

                content, current_hash = row

                # Verify content hasn't changed again since queuing
                expected_hash = hashlib.md5(content.encode()).hexdigest()
                if current_hash != expected_hash:
                    log.warning("Article %d hash mismatch, recomputing", article_id)

                num_chunks = process_article(article_id, content, current_hash, model, conn)
                conn.commit()

                mark_completed(conn, queue_id, "completed")
                log.info("Article %d: embedded %d chunks", article_id, num_chunks)
                processed += 1

            except Exception as exc:
                conn.rollback()
                log.error("Article %d failed: %s", article_id, exc)
                mark_failed(conn, queue_id, str(exc))

    return processed


def mark_completed(conn, queue_id: int, status: str, error_msg: str = None):
    """Mark a queue item as completed or skipped."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE embedding_queue
            SET status = %s, completed_at = now(), error_message = %s
            WHERE id = %s
        """, (status, error_msg, queue_id))
    conn.commit()


def mark_failed(conn, queue_id: int, error_msg: str):
    """Mark a queue item as failed, incrementing retry count."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE embedding_queue
            SET status = CASE WHEN retry_count + 1 >= %s THEN 'failed' ELSE 'pending' END,
                error_message = %s,
                retry_count = retry_count + 1,
                started_at = NULL,
                worker_id = NULL
            WHERE id = %s
        """, (MAX_RETRIES, error_msg, queue_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(db_url: str, batch_size: int, model: str, worker_id: str):
    """Continuously poll the queue until shutdown."""
    log.info("Worker %s started (model=%s, batch=%d)", worker_id, model, batch_size)

    # Recover any items left in 'processing' by crashed workers
    recover_stale_processing(db_url)

    while not shutdown_requested:
        try:
            processed = fetch_and_process(db_url, batch_size, model, worker_id)
            if processed == 0:
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            log.error("Worker loop error: %s", exc)
            time.sleep(POLL_INTERVAL)

    log.info("Worker %s stopped", worker_id)


def run_worker(args_tuple):
    """Entry point for multi-process workers."""
    db_url, batch_size, model, worker_id = args_tuple
    worker_loop(db_url, batch_size, model, worker_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embedding queue worker")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Items per batch")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database connection URL")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    if args.once:
        processed = fetch_and_process(args.db_url, args.batch_size, args.model, "worker-once")
        log.info("Processed %d items", processed)
        return

    if args.workers == 1:
        worker_loop(args.db_url, args.batch_size, args.model, "worker-0")
    else:
        log.info("Starting %d workers", args.workers)
        worker_args = [
            (args.db_url, args.batch_size, args.model, f"worker-{i}")
            for i in range(args.workers)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            executor.map(run_worker, worker_args)


if __name__ == "__main__":
    main()
