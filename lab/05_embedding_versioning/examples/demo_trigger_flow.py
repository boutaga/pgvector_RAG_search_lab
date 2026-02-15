#!/usr/bin/env python3
"""
Lab 5 — Demo: End-to-End Trigger Flow

Demonstrates the complete pipeline:
1. Modify an article's content
2. Trigger fires → queue entry created
3. Run worker for one cycle
4. Verify embedding was generated

Usage:
    python examples/demo_trigger_flow.py
    python examples/demo_trigger_flow.py --article-id 42
    python examples/demo_trigger_flow.py --skip-embed  # don't call OpenAI
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import psycopg

# Add parent dir to path for worker import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker import fetch_and_process, OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEMO] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)


def demo_trigger_flow(db_url: str, article_id: int = None, skip_embed: bool = False):
    """Run the full trigger → queue → embed flow."""
    with psycopg.connect(db_url) as conn:
        # Step 1: Pick an article
        if article_id is None:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title FROM articles ORDER BY random() LIMIT 1")
                row = cur.fetchone()
                if not row:
                    log.error("No articles found")
                    return
                article_id, title = row
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT title FROM articles WHERE id = %s", (article_id,))
                row = cur.fetchone()
                if not row:
                    log.error("Article %d not found", article_id)
                    return
                title = row[0]

        print(f"\n{'=' * 60}")
        print(f"  Demo: End-to-End Trigger Flow")
        print(f"  Article: [{article_id}] {title[:50]}")
        print(f"{'=' * 60}")

        # Check queue before
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM embedding_queue WHERE article_id = %s AND status = 'pending'",
                (article_id,),
            )
            before_count = cur.fetchone()[0]
        print(f"\n1. Queue entries (pending) for article {article_id} BEFORE update: {before_count}")

        # Step 2: Modify the content
        suffix = f"\n\n[Updated at {time.strftime('%Y-%m-%d %H:%M:%S')} for embedding versioning demo]"
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET content = content || %s WHERE id = %s",
                (suffix, article_id),
            )
        conn.commit()
        print(f"\n2. Updated article content (appended demo text)")

        # Step 3: Verify trigger fired
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, status, content_hash, queued_at
                FROM embedding_queue
                WHERE article_id = %s
                ORDER BY queued_at DESC
                LIMIT 1
            """, (article_id,))
            queue_entry = cur.fetchone()

        if queue_entry:
            print(f"\n3. Trigger fired! Queue entry created:")
            print(f"   Queue ID:     {queue_entry[0]}")
            print(f"   Status:       {queue_entry[1]}")
            print(f"   Content Hash: {queue_entry[2][:16]}...")
            print(f"   Queued At:    {queue_entry[3]}")
        else:
            print("\n3. WARNING: No queue entry found (trigger may not have fired)")
            return

        # Step 4: Verify content_hash was updated
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash, updated_at FROM articles WHERE id = %s",
                (article_id,),
            )
            hash_row = cur.fetchone()
        print(f"\n4. Article metadata updated:")
        print(f"   Content Hash: {hash_row[0][:16]}...")
        print(f"   Updated At:   {hash_row[1]}")

        # Step 5: Run worker (if not skipping)
        if skip_embed:
            print(f"\n5. [SKIPPED] Embedding generation (--skip-embed)")
            print(f"   Run 'python worker.py --once' to process the queue")
        else:
            if not OPENAI_API_KEY:
                print(f"\n5. [SKIPPED] No OPENAI_API_KEY set")
                print(f"   Export OPENAI_API_KEY and run: python worker.py --once")
            else:
                print(f"\n5. Running worker for one batch...")
                processed = fetch_and_process(db_url, 10, "text-embedding-3-small", "demo-worker")
                print(f"   Processed {processed} items")

                # Verify embeddings created
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT count(*), max(created_at)
                        FROM article_embeddings_versioned
                        WHERE article_id = %s AND is_current = true
                    """, (article_id,))
                    emb_row = cur.fetchone()
                print(f"\n6. Embeddings for article {article_id}:")
                print(f"   Current chunks: {emb_row[0]}")
                print(f"   Last created:   {emb_row[1]}")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete!")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Demo trigger flow")
    parser.add_argument("--article-id", type=int, help="Specific article to use")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    demo_trigger_flow(args.db_url, args.article_id, args.skip_embed)


if __name__ == "__main__":
    main()
