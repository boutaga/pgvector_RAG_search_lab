#!/usr/bin/env python3
"""
Lab 5 — Embedding Freshness Monitor

Runs diagnostic queries against the Wikipedia database to detect:
- Articles with no embeddings
- Stale embeddings (content changed since last embed)
- Queue health (pending/failed/processing counts)
- Version coverage across model versions
- Quality signal trends

Usage:
    python freshness_monitor.py --report        # full report
    python freshness_monitor.py --stale         # stale documents only
    python freshness_monitor.py --queue         # queue health only
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_table(headers: list[str], rows: list[tuple], max_col_width: int = 40):
    """Print a simple ASCII table."""
    if not rows:
        print("  (no results)")
        return

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], min(len(str(val)), max_col_width))

    fmt = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "  " + "-+-".join("-" * w for w in widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        vals = [str(v)[:max_col_width] for v in row]
        print(fmt.format(*vals))


# ---------------------------------------------------------------------------
# Query 1: Articles with no current embeddings
# ---------------------------------------------------------------------------

def query_no_embeddings(conn) -> list[tuple]:
    """Find articles that have no current embeddings at all."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.id, a.title, a.updated_at
            FROM articles a
            LEFT JOIN article_embeddings_versioned e
                ON e.article_id = a.id AND e.is_current = true
            WHERE e.id IS NULL
            ORDER BY a.updated_at DESC NULLS LAST
            LIMIT 20
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 2: Stale articles (content changed since last embedding)
# ---------------------------------------------------------------------------

def query_stale(conn) -> list[tuple]:
    """Find articles where content_hash differs from the embedding's source_hash."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                a.id,
                a.title,
                a.content_hash AS current_hash,
                e.source_hash  AS embedded_hash,
                a.updated_at   AS content_updated,
                e.created_at   AS embed_created
            FROM articles a
            JOIN article_embeddings_versioned e
                ON e.article_id = a.id AND e.is_current = true AND e.chunk_index = 0
            WHERE a.content_hash IS DISTINCT FROM e.source_hash
            ORDER BY a.updated_at DESC
            LIMIT 20
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 3: Queue health
# ---------------------------------------------------------------------------

def query_queue_health(conn) -> list[tuple]:
    """Show queue status distribution."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                status,
                count(*) AS cnt,
                min(queued_at) AS oldest,
                max(queued_at) AS newest
            FROM embedding_queue
            GROUP BY status
            ORDER BY
                CASE status
                    WHEN 'processing' THEN 1
                    WHEN 'pending'    THEN 2
                    WHEN 'failed'     THEN 3
                    WHEN 'completed'  THEN 4
                    WHEN 'skipped'    THEN 5
                END
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 4: Version coverage
# ---------------------------------------------------------------------------

def query_version_coverage(conn) -> list[tuple]:
    """Show embedding counts per model version."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                model_version,
                count(DISTINCT article_id) AS articles,
                count(*) AS total_chunks,
                count(*) FILTER (WHERE is_current) AS current_chunks,
                min(created_at) AS first_created,
                max(created_at) AS last_created
            FROM article_embeddings_versioned
            GROUP BY model_version
            ORDER BY last_created DESC
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 5: Freshness summary
# ---------------------------------------------------------------------------

def query_freshness_summary(conn) -> list[tuple]:
    """Overall freshness statistics."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH stats AS (
                SELECT
                    count(*) AS total_articles,
                    count(*) FILTER (WHERE content_hash IS NOT NULL) AS hashed_articles
                FROM articles
            ),
            embedded AS (
                SELECT count(DISTINCT article_id) AS embedded_articles
                FROM article_embeddings_versioned
                WHERE is_current = true
            ),
            stale AS (
                SELECT count(DISTINCT a.id) AS stale_articles
                FROM articles a
                JOIN article_embeddings_versioned e
                    ON e.article_id = a.id AND e.is_current = true AND e.chunk_index = 0
                WHERE a.content_hash IS DISTINCT FROM e.source_hash
            )
            SELECT
                s.total_articles,
                e.embedded_articles,
                s.total_articles - e.embedded_articles AS not_embedded,
                st.stale_articles,
                CASE WHEN s.total_articles > 0
                    THEN round(100.0 * e.embedded_articles / s.total_articles, 1)
                    ELSE 0
                END AS coverage_pct,
                CASE WHEN e.embedded_articles > 0
                    THEN round(100.0 * st.stale_articles / e.embedded_articles, 1)
                    ELSE 0
                END AS staleness_pct
            FROM stats s, embedded e, stale st
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 6: Change log summary (embed/skip decisions)
# ---------------------------------------------------------------------------

def query_change_decisions(conn) -> list[tuple]:
    """Show recent embed/skip decision distribution."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                decision,
                count(*) AS cnt,
                round(avg(similarity), 4) AS avg_similarity,
                min(decided_at) AS first,
                max(decided_at) AS last
            FROM embedding_change_log
            GROUP BY decision
            ORDER BY decision
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Query 7: Quality signal (nDCG trends)
# ---------------------------------------------------------------------------

def query_quality_trends(conn) -> list[tuple]:
    """Show recent quality log entries grouped by model version."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                model_version,
                count(*) AS queries,
                round(avg(ndcg_score), 4) AS avg_ndcg,
                count(*) FILTER (WHERE user_feedback = 'negative') AS negative_feedback,
                max(logged_at) AS last_entry
            FROM retrieval_quality_log
            GROUP BY model_version
            ORDER BY last_entry DESC
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def full_report(db_url: str):
    """Print a comprehensive freshness report."""
    print(f"\nEmbedding Freshness Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with psycopg.connect(db_url) as conn:
        # Freshness summary
        print_section("Freshness Summary")
        rows = query_freshness_summary(conn)
        if rows:
            r = rows[0]
            print(f"  Total articles:        {r[0]}")
            print(f"  With embeddings:       {r[1]}  ({r[4]}%)")
            print(f"  Without embeddings:    {r[2]}")
            print(f"  Stale embeddings:      {r[3]}  ({r[5]}%)")

        # Stale documents
        print_section("Stale Articles (content changed since embedding)")
        rows = query_stale(conn)
        print_table(
            ["ID", "Title", "Current Hash", "Embed Hash", "Content Updated", "Embed Created"],
            rows,
        )

        # No embeddings
        print_section("Articles Without Embeddings (top 20)")
        rows = query_no_embeddings(conn)
        print_table(["ID", "Title", "Updated At"], rows)

        # Queue health
        print_section("Queue Health")
        rows = query_queue_health(conn)
        print_table(["Status", "Count", "Oldest", "Newest"], rows)

        # Version coverage
        print_section("Embedding Version Coverage")
        rows = query_version_coverage(conn)
        print_table(
            ["Model Version", "Articles", "Total Chunks", "Current Chunks", "First", "Last"],
            rows,
        )

        # Change decisions
        print_section("Change Detection Decisions")
        rows = query_change_decisions(conn)
        print_table(["Decision", "Count", "Avg Similarity", "First", "Last"], rows)

        # Quality trends
        print_section("Quality Signal Trends")
        rows = query_quality_trends(conn)
        print_table(["Model", "Queries", "Avg nDCG", "Negative Feedback", "Last Entry"], rows)

    print(f"\n{'=' * 60}")
    print("  End of report")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embedding freshness monitor")
    parser.add_argument("--report", action="store_true", help="Full freshness report")
    parser.add_argument("--stale", action="store_true", help="Show stale articles only")
    parser.add_argument("--queue", action="store_true", help="Show queue health only")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    if args.report or not (args.stale or args.queue):
        full_report(args.db_url)
    else:
        with psycopg.connect(args.db_url) as conn:
            if args.stale:
                print_section("Stale Articles")
                rows = query_stale(conn)
                print_table(
                    ["ID", "Title", "Current Hash", "Embed Hash", "Content Updated", "Embed Created"],
                    rows,
                )
            elif args.queue:
                print_section("Queue Health")
                rows = query_queue_health(conn)
                print_table(["Status", "Count", "Oldest", "Newest"], rows)


if __name__ == "__main__":
    main()
