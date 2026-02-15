#!/usr/bin/env python3
"""
Lab 5 — Demo: SKIP LOCKED Multi-Worker Concurrency

Demonstrates that multiple workers can safely process the queue
concurrently without overlap, using SELECT FOR UPDATE SKIP LOCKED.

Usage:
    python examples/demo_skip_locked.py --workers 4
    python examples/demo_skip_locked.py --workers 4 --items 50
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)


def ensure_queue_items(db_url: str, min_items: int):
    """Ensure there are enough pending items in the queue for the demo."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM embedding_queue WHERE status = 'pending'")
            current = cur.fetchone()[0]

        if current >= min_items:
            log.info("Queue already has %d pending items (need %d)", current, min_items)
            return current

        # Add synthetic queue entries for articles that aren't already queued
        needed = min_items - current
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO embedding_queue (article_id, content_hash, change_type, priority)
                SELECT a.id, a.content_hash, 'demo', 5
                FROM articles a
                WHERE a.content_hash IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM embedding_queue eq
                      WHERE eq.article_id = a.id AND eq.status = 'pending'
                  )
                ORDER BY random()
                LIMIT %s
            """, (needed,))
            added = cur.rowcount
        conn.commit()
        log.info("Added %d demo queue entries (total pending: %d)", added, current + added)
        return current + added


def worker_grab_items(args_tuple) -> dict:
    """A single worker grabs items using SKIP LOCKED and records what it got.

    Does NOT call OpenAI — just claims items and marks them completed
    to demonstrate the concurrency pattern.
    """
    db_url, worker_id, batch_size = args_tuple
    claimed = []

    try:
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
                    RETURNING id, article_id
                """, (worker_id, batch_size))

                rows = cur.fetchall()
                claimed = [(qid, aid) for qid, aid in rows]
                conn.commit()

            # Mark as completed (simulating work done)
            if claimed:
                with conn.cursor() as cur:
                    queue_ids = [qid for qid, _ in claimed]
                    cur.execute("""
                        UPDATE embedding_queue
                        SET status = 'completed', completed_at = now()
                        WHERE id = ANY(%s)
                    """, (queue_ids,))
                conn.commit()

    except Exception as exc:
        log.error("Worker %s error: %s", worker_id, exc)

    return {"worker_id": worker_id, "claimed": claimed}


def demo_skip_locked(db_url: str, num_workers: int, min_items: int):
    """Run the SKIP LOCKED concurrency demo."""
    print(f"\n{'=' * 60}")
    print(f"  Demo: SKIP LOCKED Multi-Worker Concurrency")
    print(f"  Workers: {num_workers}  |  Target items: {min_items}")
    print(f"{'=' * 60}\n")

    # Ensure we have enough queue items
    total = ensure_queue_items(db_url, min_items)
    items_per_worker = max(1, min_items // num_workers + 5)

    # Launch workers in parallel
    print(f"Launching {num_workers} workers (each requesting up to {items_per_worker} items)...\n")

    worker_args = [
        (db_url, f"demo-worker-{i}", items_per_worker)
        for i in range(num_workers)
    ]

    results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_grab_items, args) for args in worker_args]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.time() - start_time

    # Analyze results
    all_article_ids = []
    for r in sorted(results, key=lambda x: x["worker_id"]):
        article_ids = [aid for _, aid in r["claimed"]]
        all_article_ids.extend(article_ids)
        print(f"  {r['worker_id']}: claimed {len(r['claimed'])} items"
              f"  (articles: {article_ids[:5]}{'...' if len(article_ids) > 5 else ''})")

    # Check for overlaps
    print(f"\n{'=' * 40}")
    total_claimed = len(all_article_ids)
    unique_articles = len(set(all_article_ids))

    # Count duplicates (same article claimed by multiple workers)
    article_counts = defaultdict(int)
    for aid in all_article_ids:
        article_counts[aid] += 1
    duplicates = {aid: cnt for aid, cnt in article_counts.items() if cnt > 1}

    print(f"  Total items claimed:  {total_claimed}")
    print(f"  Unique articles:      {unique_articles}")
    print(f"  Elapsed time:         {elapsed:.2f}s")

    if duplicates:
        print(f"\n  WARNING: {len(duplicates)} articles claimed by multiple workers!")
        for aid, cnt in list(duplicates.items())[:5]:
            print(f"    Article {aid}: claimed {cnt} times")
    else:
        print(f"\n  ZERO OVERLAP — SKIP LOCKED working correctly!")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="SKIP LOCKED concurrency demo")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--items", type=int, default=50, help="Minimum queue items")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    demo_skip_locked(args.db_url, args.workers, args.items)


if __name__ == "__main__":
    main()
