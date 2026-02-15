#!/usr/bin/env python3
"""
Lab 5 — Blue-Green Model Version Upgrade

Supports upgrading embedding models with zero downtime:
1. Create new embeddings alongside old ones (blue-green)
2. Compare search quality between versions
3. Cutover: mark new version as current
4. Rollback: restore previous version if needed

Usage:
    python model_upgrade.py --queue-upgrade --new-model text-embedding-3-large --limit 100
    python model_upgrade.py --compare --old-model text-embedding-3-small --new-model text-embedding-3-large
    python model_upgrade.py --cutover --new-model text-embedding-3-large
    python model_upgrade.py --rollback --old-model text-embedding-3-small
"""

import argparse
import json
import logging
import os
import sys

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [UPGRADE] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wikipedia",
)

# Model dimensions lookup
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# ---------------------------------------------------------------------------
# Queue articles for re-embedding with new model
# ---------------------------------------------------------------------------

def queue_upgrade(db_url: str, new_model: str, limit: int):
    """Queue articles that have current embeddings for re-embedding with new model.

    Queues articles that already have embeddings under the old model
    so the worker can create new version alongside.
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Find articles with current embeddings that don't have the new model
            cur.execute("""
                INSERT INTO embedding_queue (article_id, content_hash, change_type, priority)
                SELECT DISTINCT a.id, a.content_hash, 'model_upgrade', 4
                FROM articles a
                JOIN article_embeddings_versioned e
                    ON e.article_id = a.id AND e.is_current = true
                WHERE NOT EXISTS (
                    SELECT 1 FROM article_embeddings_versioned e2
                    WHERE e2.article_id = a.id
                      AND e2.model_version = %s
                      AND e2.is_current = true
                )
                LIMIT %s
            """, (new_model, limit))

            queued = cur.rowcount
        conn.commit()

    log.info("Queued %d articles for model upgrade to %s", queued, new_model)
    return queued


# ---------------------------------------------------------------------------
# Create DiskANN index for new model version
# ---------------------------------------------------------------------------

def create_version_index(db_url: str, model_version: str):
    """Create a partial DiskANN index for a specific model version.

    Note: text-embedding-3-large (3072 dims) requires the embedding column
    to support that dimension, or a separate column. For this lab, we
    demonstrate the concept with the existing 1536-dim column.
    """
    dims = MODEL_DIMENSIONS.get(model_version, 1536)
    index_name = f"ix_embed_diskann_{model_version.replace('-', '_')}"

    log.info("Creating DiskANN index '%s' for model %s (%d dims)",
             index_name, model_version, dims)

    with psycopg.connect(db_url) as conn:
        # DiskANN index creation can take a while on large datasets
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON article_embeddings_versioned
                USING diskann (embedding)
                WHERE is_current = true AND model_version = %s
            """, (model_version,))

    log.info("Index '%s' created successfully", index_name)


# ---------------------------------------------------------------------------
# Compare versions (search quality)
# ---------------------------------------------------------------------------

def compare_versions(db_url: str, old_model: str, new_model: str,
                     test_queries: list[str] = None):
    """Compare search results between two model versions.

    For each test query, runs similarity search against both versions
    and shows overlap/differences.
    """
    if test_queries is None:
        test_queries = [
            "machine learning algorithms",
            "world war two history",
            "climate change effects",
            "quantum computing principles",
            "ancient roman empire",
        ]

    log.info("Comparing %s vs %s with %d test queries",
             old_model, new_model, len(test_queries))

    # We need an embedding for the query — use old model for consistency
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    with psycopg.connect(db_url) as conn:
        for query in test_queries:
            # Generate query embedding
            resp = client.embeddings.create(input=[query], model=old_model)
            query_emb = resp.data[0].embedding

            print(f"\nQuery: '{query}'")
            print("-" * 50)

            for model in [old_model, new_model]:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT e.article_id, a.title,
                               e.embedding <=> %s::vector AS distance
                        FROM article_embeddings_versioned e
                        JOIN articles a ON a.id = e.article_id
                        WHERE e.is_current = true
                          AND e.model_version = %s
                          AND e.chunk_index = 0
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT 5
                    """, (json.dumps(query_emb), model, json.dumps(query_emb)))
                    results = cur.fetchall()

                print(f"\n  {model}:")
                if not results:
                    print("    (no embeddings for this model)")
                for rank, (aid, title, dist) in enumerate(results, 1):
                    print(f"    {rank}. [{aid}] {title[:50]} (dist={dist:.4f})")


# ---------------------------------------------------------------------------
# Cutover: make new version current, retire old
# ---------------------------------------------------------------------------

def cutover(db_url: str, new_model: str):
    """Switch the current version to the new model.

    For articles that have both old and new embeddings:
    - Mark new model embeddings as is_current = true (should already be)
    - Mark other model embeddings as is_current = false
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Find articles that have the new model ready
            cur.execute("""
                UPDATE article_embeddings_versioned
                SET is_current = false, replaced_at = now()
                WHERE is_current = true
                  AND model_version != %s
                  AND article_id IN (
                      SELECT DISTINCT article_id
                      FROM article_embeddings_versioned
                      WHERE model_version = %s AND is_current = true
                  )
            """, (new_model, new_model))
            retired = cur.rowcount

        conn.commit()

    log.info("Cutover complete: retired %d old embedding rows, %s is now primary",
             retired, new_model)
    return retired


# ---------------------------------------------------------------------------
# Rollback: restore previous version
# ---------------------------------------------------------------------------

def rollback(db_url: str, old_model: str):
    """Rollback to the previous model version.

    Re-activates the old model's embeddings (sets is_current = true,
    clears replaced_at) for articles where the old model still has rows.
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Restore old model embeddings
            cur.execute("""
                UPDATE article_embeddings_versioned
                SET is_current = true, replaced_at = NULL
                WHERE model_version = %s
                  AND is_current = false
                  AND replaced_at IS NOT NULL
            """, (old_model,))
            restored = cur.rowcount

            # Deactivate new model embeddings for those articles
            cur.execute("""
                UPDATE article_embeddings_versioned
                SET is_current = false
                WHERE model_version != %s
                  AND is_current = true
                  AND article_id IN (
                      SELECT DISTINCT article_id
                      FROM article_embeddings_versioned
                      WHERE model_version = %s AND is_current = true
                  )
            """, (old_model, old_model))
            deactivated = cur.rowcount

        conn.commit()

    log.info("Rollback complete: restored %d rows for %s, deactivated %d newer rows",
             restored, old_model, deactivated)
    return restored


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status(db_url: str):
    """Show current model version status."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    model_version,
                    count(DISTINCT article_id) AS articles,
                    count(*) AS chunks,
                    bool_or(is_current) AS has_current,
                    min(created_at) AS first,
                    max(created_at) AS last
                FROM article_embeddings_versioned
                GROUP BY model_version
                ORDER BY last DESC
            """)
            rows = cur.fetchall()

    print("\nModel Version Status:")
    print("-" * 80)
    print(f"  {'Model':<30} {'Articles':>8} {'Chunks':>8} {'Current':>8} {'First':>12} {'Last':>12}")
    print("-" * 80)
    for model, articles, chunks, has_current, first, last in rows:
        print(f"  {model:<30} {articles:>8} {chunks:>8} {'YES' if has_current else 'no':>8}"
              f" {str(first)[:10]:>12} {str(last)[:10]:>12}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Blue-green model upgrade")
    parser.add_argument("--queue-upgrade", action="store_true", help="Queue articles for new model")
    parser.add_argument("--create-index", action="store_true", help="Create DiskANN index for model")
    parser.add_argument("--compare", action="store_true", help="Compare two model versions")
    parser.add_argument("--cutover", action="store_true", help="Switch to new model")
    parser.add_argument("--rollback", action="store_true", help="Rollback to old model")
    parser.add_argument("--status", action="store_true", help="Show version status")
    parser.add_argument("--old-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--new-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--limit", type=int, default=100, help="Articles to queue for upgrade")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    if args.queue_upgrade:
        queue_upgrade(args.db_url, args.new_model, args.limit)
    elif args.create_index:
        create_version_index(args.db_url, args.new_model)
    elif args.compare:
        compare_versions(args.db_url, args.old_model, args.new_model)
    elif args.cutover:
        cutover(args.db_url, args.new_model)
    elif args.rollback:
        rollback(args.db_url, args.old_model)
    elif args.status:
        show_status(args.db_url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
