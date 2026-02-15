#!/usr/bin/env python3
"""
Lab 5 — Demo: Quality Drift & Staleness Detection

Demonstrates how embedding quality degrades when articles change
without re-embedding, and how the quality feedback loop detects
and triggers re-embedding.

Usage:
    python examples/demo_quality_drift.py
    python examples/demo_quality_drift.py --simulate-queries 20
"""

import argparse
import logging
import math
import os
import random
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_feedback_loop import run_quality_check, find_poor_queries, force_reembed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEMO] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

MODEL_VERSION = "text-embedding-3-small"


def simulate_quality_queries(conn, num_queries: int, stale_ratio: float = 0.4):
    """Simulate retrieval quality logs with some degraded results.

    Creates fake quality entries where a fraction have poor nDCG scores,
    simulating what happens when embeddings become stale.
    """
    # Get some article titles to use as query material
    with conn.cursor() as cur:
        cur.execute("SELECT id, title FROM articles ORDER BY random() LIMIT %s", (num_queries * 2,))
        articles = cur.fetchall()

    if not articles:
        log.warning("No articles found for query simulation")
        return

    queries_logged = 0
    for i in range(num_queries):
        art = random.choice(articles)
        article_id, title = art

        # Generate query from title words
        words = title.split()[:3]
        query = " ".join(words)

        # Simulate result IDs (pick random articles)
        result_ids = [random.choice(articles)[0] for _ in range(5)]
        # Ensure the actual article is sometimes in results
        if random.random() > 0.3:
            result_ids[0] = article_id

        # Simulate relevance scores
        if random.random() < stale_ratio:
            # Poor results (stale embeddings scenario)
            relevance_scores = [round(random.uniform(0.0, 0.4), 2) for _ in range(5)]
            feedback = random.choice(["negative", "negative", None])
        else:
            # Good results
            relevance_scores = sorted(
                [round(random.uniform(0.5, 1.0), 2) for _ in range(5)],
                reverse=True,
            )
            feedback = random.choice(["positive", None, None])

        ndcg = run_quality_check(conn, query, result_ids, relevance_scores, MODEL_VERSION)

        # Add user feedback if any
        if feedback:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE retrieval_quality_log
                    SET user_feedback = %s
                    WHERE query_text = %s AND logged_at = (
                        SELECT max(logged_at) FROM retrieval_quality_log WHERE query_text = %s
                    )
                """, (feedback, query, query))
            conn.commit()

        queries_logged += 1

    return queries_logged


def demo_quality_drift(db_url: str, num_queries: int):
    """Run the quality drift demo."""
    print(f"\n{'=' * 60}")
    print(f"  Demo: Quality Drift & Staleness Detection")
    print(f"{'=' * 60}")

    with psycopg.connect(db_url) as conn:
        # Step 1: Check current staleness
        print(f"\nStep 1: Current embedding staleness")
        print("-" * 40)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    count(*) AS total,
                    count(*) FILTER (WHERE a.content_hash IS DISTINCT FROM e.source_hash) AS stale
                FROM articles a
                JOIN article_embeddings_versioned e
                    ON e.article_id = a.id AND e.is_current = true AND e.chunk_index = 0
            """)
            row = cur.fetchone()
            if row and row[0] > 0:
                total, stale = row
                print(f"  Embedded articles:  {total}")
                print(f"  Stale:              {stale} ({100 * stale / total:.1f}%)")
            else:
                print("  No versioned embeddings found yet")
                print("  Run worker.py first to generate baseline embeddings")

        # Step 2: Simulate retrieval quality logs
        print(f"\nStep 2: Simulating {num_queries} retrieval quality logs")
        print("-" * 40)
        logged = simulate_quality_queries(conn, num_queries, stale_ratio=0.4)
        print(f"  Logged {logged} simulated queries (40% with poor results)")

        # Step 3: Show quality summary
        print(f"\nStep 3: Quality summary")
        print("-" * 40)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    count(*) AS total,
                    round(avg(ndcg_score), 4) AS avg_ndcg,
                    count(*) FILTER (WHERE ndcg_score < 0.5) AS poor,
                    count(*) FILTER (WHERE user_feedback = 'negative') AS negative
                FROM retrieval_quality_log
                WHERE logged_at > now() - interval '1 hour'
            """)
            row = cur.fetchone()
            if row:
                print(f"  Total queries (last hour): {row[0]}")
                print(f"  Average nDCG:              {row[1]}")
                print(f"  Poor quality (nDCG<0.5):   {row[2]}")
                print(f"  Negative feedback:         {row[3]}")

        # Step 4: Run quality feedback detection
        print(f"\nStep 4: Quality feedback loop — finding poor queries")
        print("-" * 40)
        poor_queries = find_poor_queries(conn, ndcg_threshold=0.5, lookback_hours=1)
        print(f"  Found {len(poor_queries)} poor queries")
        for pq in poor_queries[:5]:
            print(f"    nDCG={pq['ndcg_score']:.4f}  feedback={pq['user_feedback']}"
                  f"  query='{pq['query_text'][:40]}'")
        if len(poor_queries) > 5:
            print(f"    ... and {len(poor_queries) - 5} more")

        # Step 5: Identify articles for re-embedding
        print(f"\nStep 5: Identifying articles correlated with poor queries")
        print("-" * 40)
        affected_ids = set()
        for pq in poor_queries[:10]:
            from quality_feedback_loop import find_affected_articles
            affected = find_affected_articles(conn, pq["query_text"])
            for row in affected:
                affected_ids.add(row[0])
                if len(affected_ids) <= 10:
                    print(f"    Article {row[0]}: {row[1][:40]}")

        if affected_ids:
            print(f"\n  Total affected articles: {len(affected_ids)}")
            queued = force_reembed(conn, list(affected_ids))
            print(f"  Queued {queued} articles for re-embedding")
        else:
            print("  No articles identified (queries may not match article titles)")

    # Step 6: Show queue state
    print(f"\nStep 6: Queue state after quality feedback")
    print("-" * 40)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT change_type, status, count(*)
                FROM embedding_queue
                GROUP BY change_type, status
                ORDER BY change_type, status
            """)
            for change_type, status, cnt in cur.fetchall():
                print(f"  {change_type:<20} {status:<12} {cnt}")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete!")
    print(f"  Next step: run 'python worker.py --once' to re-embed flagged articles")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Quality drift demo")
    parser.add_argument("--simulate-queries", type=int, default=20,
                        help="Number of quality queries to simulate")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    demo_quality_drift(args.db_url, args.simulate_queries)


if __name__ == "__main__":
    main()
