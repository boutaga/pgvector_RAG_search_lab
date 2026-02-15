#!/usr/bin/env python3
"""
Lab 5 — Demo: Blue-Green Model Upgrade

Demonstrates the full model upgrade lifecycle:
1. Show current model status
2. Queue a subset of articles for new model
3. (Optionally) embed them
4. Compare search results between versions
5. Cutover to new model
6. Rollback to old model

Usage:
    python examples/demo_model_upgrade.py
    python examples/demo_model_upgrade.py --embed   # actually generate embeddings
"""

import argparse
import logging
import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_upgrade import queue_upgrade, cutover, rollback, show_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEMO] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

OLD_MODEL = "text-embedding-3-small"
NEW_MODEL = "text-embedding-3-large"


def demo_model_upgrade(db_url: str, do_embed: bool = False, limit: int = 10):
    """Run the blue-green model upgrade demo."""
    print(f"\n{'=' * 60}")
    print(f"  Demo: Blue-Green Model Upgrade")
    print(f"  Old model: {OLD_MODEL}")
    print(f"  New model: {NEW_MODEL}")
    print(f"{'=' * 60}")

    # Step 1: Current status
    print(f"\nStep 1: Current model version status")
    print("-" * 40)
    show_status(db_url)

    # Step 2: Queue articles for upgrade
    print(f"\nStep 2: Queue {limit} articles for model upgrade")
    print("-" * 40)
    queued = queue_upgrade(db_url, NEW_MODEL, limit)
    print(f"  Queued {queued} articles for {NEW_MODEL}")

    # Step 3: Show queue state
    print(f"\nStep 3: Queue state after scheduling upgrade")
    print("-" * 40)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT change_type, status, count(*)
                FROM embedding_queue
                WHERE change_type = 'model_upgrade'
                GROUP BY change_type, status
            """)
            rows = cur.fetchall()
            for change_type, status, cnt in rows:
                print(f"  {change_type} / {status}: {cnt}")

    if do_embed:
        from worker import fetch_and_process, OPENAI_API_KEY
        if not OPENAI_API_KEY:
            print(f"\n  [SKIP] No OPENAI_API_KEY — cannot generate embeddings")
            print(f"  Run: OPENAI_API_KEY=... python worker.py --model {NEW_MODEL} --once")
        else:
            print(f"\nStep 4: Generating embeddings with {NEW_MODEL}...")
            print("-" * 40)
            processed = fetch_and_process(db_url, limit, NEW_MODEL, "demo-upgrade-worker")
            print(f"  Processed {processed} items with {NEW_MODEL}")

            # Compare
            print(f"\nStep 5: Updated status after embedding")
            print("-" * 40)
            show_status(db_url)
    else:
        print(f"\n  [INFO] Embedding generation skipped (use --embed to generate)")
        print(f"  Run manually: python worker.py --model {NEW_MODEL} --once")

    # Step 4/6: Demonstrate cutover
    print(f"\nStep {'6' if do_embed else '4'}: Cutover demonstration")
    print("-" * 40)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(DISTINCT article_id)
                FROM article_embeddings_versioned
                WHERE model_version = %s AND is_current = true
            """, (NEW_MODEL,))
            new_count = cur.fetchone()[0]

    if new_count > 0:
        print(f"  {new_count} articles have {NEW_MODEL} embeddings — performing cutover...")
        retired = cutover(db_url, NEW_MODEL)
        print(f"  Cutover: retired {retired} old embedding rows")
        show_status(db_url)

        # Rollback
        print(f"\nStep {'7' if do_embed else '5'}: Rollback demonstration")
        print("-" * 40)
        restored = rollback(db_url, OLD_MODEL)
        print(f"  Rollback: restored {restored} rows for {OLD_MODEL}")
        show_status(db_url)
    else:
        print(f"  No {NEW_MODEL} embeddings exist yet — cutover/rollback skipped")
        print(f"  Generate embeddings first, then re-run this demo")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete!")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Model upgrade demo")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings (needs OPENAI_API_KEY)")
    parser.add_argument("--limit", type=int, default=10, help="Articles to upgrade")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    demo_model_upgrade(args.db_url, args.embed, args.limit)


if __name__ == "__main__":
    main()
