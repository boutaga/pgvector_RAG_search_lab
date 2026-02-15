#!/usr/bin/env python3
"""
Lab 5 — Quality Feedback Loop

Detects quality degradation by analyzing retrieval_quality_log entries
and triggers re-embedding for articles correlated with poor results.

Uses ILIKE-based title correlation to find affected articles (since
there's no direct affected_doc_ids column).

Usage:
    python quality_feedback_loop.py --check          # run quality check
    python quality_feedback_loop.py --ndcg-threshold 0.5  # custom threshold
"""

import argparse
import json
import logging
import os
import sys

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QUALITY] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

DEFAULT_NDCG_THRESHOLD = 0.5   # below this → investigate
DEFAULT_LOOKBACK_HOURS = 24


# ---------------------------------------------------------------------------
# Find articles correlated with poor queries
# ---------------------------------------------------------------------------

def find_poor_queries(conn, ndcg_threshold: float, lookback_hours: int) -> list[dict]:
    """Find queries with low nDCG scores or negative feedback."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, query_text, model_version, ndcg_score, user_feedback, logged_at
            FROM retrieval_quality_log
            WHERE (ndcg_score < %s OR user_feedback = 'negative')
              AND logged_at > now() - (%s * interval '1 hour')
            ORDER BY ndcg_score ASC NULLS FIRST
            LIMIT 50
        """, (ndcg_threshold, lookback_hours))
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "query_text": r[1],
            "model_version": r[2],
            "ndcg_score": float(r[3]) if r[3] is not None else None,
            "user_feedback": r[4],
            "logged_at": str(r[5]),
        }
        for r in rows
    ]


def find_affected_articles(conn, query_text: str, limit: int = 5) -> list[tuple]:
    """Find articles whose titles match keywords from a poor query.

    Uses ILIKE-based correlation since we don't have a direct
    affected_doc_ids column.
    """
    # Extract significant words (>3 chars) from the query
    words = [w for w in query_text.split() if len(w) > 3]
    if not words:
        return []

    # Build ILIKE conditions for each word
    conditions = " OR ".join(["a.title ILIKE %s"] * len(words))
    params = [f"%{w}%" for w in words[:5]]  # limit to 5 keywords

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.id, a.title, a.content_hash, e.source_hash, e.created_at
            FROM articles a
            LEFT JOIN article_embeddings_versioned e
                ON e.article_id = a.id AND e.is_current = true AND e.chunk_index = 0
            WHERE ({conditions})
            ORDER BY a.id
            LIMIT %s
        """, params + [limit])
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Force re-embedding
# ---------------------------------------------------------------------------

def force_reembed(conn, article_ids: list[int], reason: str = "quality_degradation"):
    """Queue articles for re-embedding due to quality issues."""
    queued = 0
    with conn.cursor() as cur:
        for article_id in article_ids:
            cur.execute("""
                INSERT INTO embedding_queue (article_id, content_hash, change_type, priority)
                SELECT a.id, a.content_hash, 'quality_reembed', 2
                FROM articles a
                WHERE a.id = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM embedding_queue eq
                      WHERE eq.article_id = a.id
                        AND eq.status IN ('pending', 'processing')
                  )
            """, (article_id,))
            queued += cur.rowcount

    conn.commit()
    log.info("Queued %d articles for re-embedding (reason: %s)", queued, reason)
    return queued


# ---------------------------------------------------------------------------
# Quality check and log
# ---------------------------------------------------------------------------

def run_quality_check(conn, query: str, result_ids: list[int],
                      relevance_scores: list[float], model_version: str) -> float:
    """Compute nDCG for a single query and log it.

    Simple nDCG@k implementation (no external deps).
    Returns the nDCG score.
    """
    import math

    def dcg(scores: list[float]) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    if not relevance_scores:
        return 0.0

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))
    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO retrieval_quality_log
                (query_text, model_version, result_ids, relevance_scores, ndcg_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (query, model_version, result_ids, relevance_scores, round(ndcg, 4)))

    conn.commit()
    return ndcg


# ---------------------------------------------------------------------------
# Main check cycle
# ---------------------------------------------------------------------------

def quality_check_cycle(db_url: str, ndcg_threshold: float, lookback_hours: int):
    """Full quality check: find poor queries → correlate articles → queue re-embeds."""
    with psycopg.connect(db_url) as conn:
        poor_queries = find_poor_queries(conn, ndcg_threshold, lookback_hours)

        if not poor_queries:
            log.info("No quality issues found (threshold=%.2f, lookback=%dh)",
                     ndcg_threshold, lookback_hours)
            return

        log.info("Found %d poor queries", len(poor_queries))

        all_affected = set()
        for pq in poor_queries:
            affected = find_affected_articles(conn, pq["query_text"])
            for row in affected:
                article_id = row[0]
                source_hash = row[3]
                content_hash = row[2]

                # If embedding is stale OR quality is poor, flag it
                if source_hash != content_hash or pq.get("ndcg_score", 1.0) < ndcg_threshold:
                    all_affected.add(article_id)
                    log.info("  Article %d (%s) correlated with poor query: %s",
                             article_id, row[1][:40], pq["query_text"][:60])

        if all_affected:
            queued = force_reembed(conn, list(all_affected))
            log.info("Quality feedback: queued %d articles for re-embedding", queued)
        else:
            log.info("No articles identified for re-embedding")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quality feedback loop")
    parser.add_argument("--check", action="store_true", help="Run quality check cycle")
    parser.add_argument("--ndcg-threshold", type=float, default=DEFAULT_NDCG_THRESHOLD,
                        help="nDCG threshold for poor quality (default 0.5)")
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS,
                        help="Hours to look back for poor queries")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    if args.check:
        quality_check_cycle(args.db_url, args.ndcg_threshold, args.lookback_hours)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
