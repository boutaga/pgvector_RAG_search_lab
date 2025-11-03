#!/usr/bin/env python3
"""
Diagnostic script to see what the vector search is actually retrieving.
This helps you understand if your ground truth IDs are correct.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab.core.database import DatabaseService
from lab.core.embeddings import OpenAIEmbedder
from lab.core.search import VectorSearch

def diagnose_query(query: str, vector_column: str = "content_vector_3072", top_k: int = 10):
    """
    Run a query and show what articles were actually retrieved.

    Args:
        query: The search query
        vector_column: Vector column to search
        top_k: Number of results to show
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Initialize services
    db = DatabaseService()
    embedder = OpenAIEmbedder()

    # Initialize search
    search = VectorSearch(
        db_service=db,
        embedding_service=embedder,
        table_name="articles",
        vector_column=vector_column,
        content_columns=["title", "content"],
        id_column="id"
    )

    # Perform search
    results = search.search(query, top_k=top_k)

    # Display results
    print(f"Retrieved {len(results)} articles:\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] ID: {result.id}")
        print(f"    Title: {result.title}")
        print(f"    Score: {result.score:.4f}")
        print(f"    Preview: {result.content[:200]}...")
        print()

    # Show the IDs for easy copy-paste into test_cases.json
    ids = [r.id for r in results]
    print(f"\nArticle IDs (for test_cases.json):")
    print(f"  \"expected_doc_ids\": {ids[:5]},")
    print(f"\nAll IDs: {ids}")
    print()

if __name__ == "__main__":
    # Test queries from your test_cases.json
    queries = [
        "What is machine learning?",
        "Who invented the telephone?",
        "Explain quantum computing applications in cryptography",
        "How do neural networks learn?",
    ]

    for query in queries:
        diagnose_query(query, top_k=10)
        print("\n" + "="*80)
        print("Review these results. Do they answer the query?")
        print("If yes, use the first 3-5 IDs as your expected_doc_ids")
        print("="*80 + "\n")
        input("Press Enter to continue to next query...")
