#!/usr/bin/env python3
"""
Test script to verify sparse search functionality and debug search issues.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService
from lab.core.embeddings import SPLADEEmbedder


def test_sparse_embeddings():
    """Test if sparse embeddings are properly stored and searchable."""

    # Connect to database
    config = ConfigService()
    db_service = DatabaseService(
        config.database.connection_string,
        config.database.min_connections,
        config.database.max_connections
    )

    print("=" * 60)
    print("SPARSE EMBEDDING TEST")
    print("=" * 60)

    # 1. Check if sparse embeddings exist
    query = """
        SELECT
            COUNT(*) as total,
            COUNT(content_sparse) as with_sparse,
            MIN(LENGTH(content_sparse::text)) as min_length,
            MAX(LENGTH(content_sparse::text)) as max_length
        FROM articles
    """

    result = db_service.execute_query(query, dict_cursor=True)
    print("\n1. Sparse Embedding Statistics:")
    print(f"   Total articles: {result[0]['total']}")
    print(f"   With sparse embeddings: {result[0]['with_sparse']}")
    print(f"   Min sparse length: {result[0]['min_length']}")
    print(f"   Max sparse length: {result[0]['max_length']}")

    # 2. Check a sample sparse embedding format
    query = """
        SELECT
            id,
            title,
            SUBSTRING(content_sparse::text, 1, 100) as sparse_sample
        FROM articles
        WHERE content_sparse IS NOT NULL
        LIMIT 3
    """

    results = db_service.execute_query(query, dict_cursor=True)
    print("\n2. Sample Sparse Embeddings:")
    for r in results:
        print(f"   ID {r['id']}: {r['title'][:50]}...")
        print(f"      Sparse: {r['sparse_sample']}...")

    # 3. Search for "July 4 2004" article directly
    query = """
        SELECT
            id,
            title,
            SUBSTRING(content, 1, 500) as content_preview
        FROM articles
        WHERE
            (title ILIKE '%2004%' OR content ILIKE '%2004%')
            AND (content ILIKE '%july%4%' OR content ILIKE '%july 4%')
        LIMIT 5
    """

    results = db_service.execute_query(query, dict_cursor=True)
    print("\n3. Direct Text Search for 'July 4 2004':")
    if results:
        for r in results:
            print(f"   Found: {r['title']}")
            print(f"   Content: {r['content_preview'][:200]}...")
    else:
        print("   No articles found with July 4 2004")

    # 4. Test sparse search with a query
    print("\n4. Testing Sparse Search Query:")

    # Initialize SPLADE embedder
    sparse_embedder = SPLADEEmbedder(
        model_name=config.embedding.splade_model,
        dimensions=config.embedding.splade_dimensions,
        batch_size=1
    )

    # Generate sparse embedding for query
    test_query = "July 4 2004 sporting upset"
    print(f"   Query: '{test_query}'")

    sparse_dict = sparse_embedder.generate_embeddings([test_query])[0]
    sparse_vector = sparse_embedder.format_for_pgvector(sparse_dict)

    print(f"   Generated sparse vector (first 100 chars): {sparse_vector[:100]}...")

    # Check if indices are 1-based
    if sparse_vector.startswith("{0:"):
        print("   ❌ WARNING: Sparse vector has 0-based indexing!")
    else:
        print("   ✅ Sparse vector has correct 1-based indexing")

    # 5. Execute sparse search
    query = """
        SELECT
            id,
            title,
            1 - (content_sparse <=> %s::sparsevec) as score
        FROM articles
        WHERE content_sparse IS NOT NULL
        ORDER BY content_sparse <=> %s::sparsevec
        LIMIT 5
    """

    try:
        results = db_service.execute_query(
            query,
            (sparse_vector, sparse_vector),
            dict_cursor=True
        )

        print("\n5. Sparse Search Results:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['title'][:60]}... (Score: {r['score']:.4f})")

    except Exception as e:
        print(f"\n5. Sparse Search Failed: {e}")

    # 6. Search for specific year articles
    query = """
        SELECT
            id,
            title,
            CASE
                WHEN content ILIKE '%july%4%' THEN 'Has July 4'
                ELSE 'No July 4'
            END as has_july4
        FROM articles
        WHERE title ~ '\\y2004\\y'
        LIMIT 10
    """

    results = db_service.execute_query(query, dict_cursor=True)
    print("\n6. Articles with '2004' in title:")
    if results:
        for r in results:
            print(f"   - {r['title']} [{r['has_july4']}]")
    else:
        print("   No articles with 2004 in title")

    # 7. Check if dense embeddings exist for comparison
    query = """
        SELECT
            COUNT(*) as with_dense_3072,
            COUNT(CASE WHEN content_vector_3072 IS NOT NULL
                       AND content_sparse IS NOT NULL THEN 1 END) as with_both
        FROM articles
    """

    result = db_service.execute_query(query, dict_cursor=True)
    print("\n7. Embedding Coverage:")
    print(f"   Articles with dense (3072): {result[0]['with_dense_3072']}")
    print(f"   Articles with both dense and sparse: {result[0]['with_both']}")

    db_service.close()
    print("\n" + "=" * 60)


def test_specific_search():
    """Test why a specific search isn't working."""

    config = ConfigService()
    db_service = DatabaseService(
        config.database.connection_string,
        config.database.min_connections,
        config.database.max_connections
    )

    print("\n" + "=" * 60)
    print("DEBUGGING SPECIFIC SEARCH ISSUE")
    print("=" * 60)

    # Look for the exact content
    query = """
        SELECT
            id,
            title,
            position('July 4' in content) as july4_pos,
            position('2004' in content) as year_pos,
            SUBSTRING(content FROM position('July 4' in content) - 50 FOR 200) as context
        FROM articles
        WHERE
            content LIKE '%July 4%'
            AND content LIKE '%2004%'
        LIMIT 5
    """

    results = db_service.execute_query(query, dict_cursor=True)
    print("\nArticles containing both 'July 4' and '2004':")
    if results:
        for r in results:
            print(f"\nID: {r['id']}")
            print(f"Title: {r['title']}")
            print(f"July 4 position: {r['july4_pos']}, 2004 position: {r['year_pos']}")
            print(f"Context: ...{r['context']}...")
    else:
        print("No articles found with both 'July 4' and '2004'")

    db_service.close()


if __name__ == "__main__":
    print("Testing Sparse Search Functionality\n")

    # Test sparse embeddings
    test_sparse_embeddings()

    # Debug specific search
    test_specific_search()

    print("\nTest complete!")