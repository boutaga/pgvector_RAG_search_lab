#!/usr/bin/env python3
"""
Example: Database Integration

Demonstrates how to use RelevanceManager for evaluation tracking.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lab.evaluation.relevance_manager import RelevanceManager, TestQuery, RelevanceGrade
from lab.evaluation.metrics import ndcg_at_k_with_grades

# Update this with your database connection string
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost/wikipedia")


def example_create_test_query():
    """Example: Create a test query"""
    print("=" * 60)
    print("Example: Creating a Test Query")
    print("=" * 60)

    with RelevanceManager(DATABASE_URL) as manager:
        query = TestQuery(
            query_id=None,
            query_text="What is machine learning?",
            query_type="conceptual",
            category="AI",
            created_by="example_script",
            notes="Basic ML concept question"
        )

        query_id = manager.create_test_query(query)
        print(f"\n✓ Created test query with ID: {query_id}")
        print(f"  Query: {query.query_text}")
        print(f"  Type: {query.query_type}")
        print()

        return query_id


def example_add_relevance_grades(query_id: int):
    """Example: Add relevance grades for a query"""
    print("=" * 60)
    print("Example: Adding Relevance Grades")
    print("=" * 60)

    with RelevanceManager(DATABASE_URL) as manager:
        # Add individual grades
        grades = [
            RelevanceGrade(query_id, 101, 2, "admin", "human", "Perfect match"),
            RelevanceGrade(query_id, 102, 1, "admin", "human", "Good but not comprehensive"),
            RelevanceGrade(query_id, 103, 0, "admin", "human", "Unrelated"),
            RelevanceGrade(query_id, 104, 1, "admin", "human", "Relevant section"),
        ]

        # Bulk add
        count = manager.bulk_add_grades(grades)
        print(f"\n✓ Added {count} relevance grades")

        # Retrieve and display
        retrieved_grades = manager.get_relevance_grades(query_id)
        print(f"\nGrades for query {query_id}:")
        for doc_id, grade in sorted(retrieved_grades.items()):
            grade_label = ["Irrelevant", "Relevant", "Highly Relevant"][grade]
            print(f"  Doc {doc_id}: Grade {grade} ({grade_label})")
        print()


def example_calculate_ndcg(query_id: int):
    """Example: Calculate nDCG for search results"""
    print("=" * 60)
    print("Example: Calculating nDCG")
    print("=" * 60)

    with RelevanceManager(DATABASE_URL) as manager:
        # Get relevance grades
        grades = manager.get_relevance_grades(query_id)

        # Simulate search results (document IDs in retrieval order)
        # Scenario 1: Good ranking
        good_results = [101, 102, 104, 103]  # Highly relevant first
        # Scenario 2: Poor ranking
        poor_results = [103, 104, 102, 101]  # Irrelevant first

        k = 4

        ndcg_good = ndcg_at_k_with_grades(good_results, grades, k)
        ndcg_poor = ndcg_at_k_with_grades(poor_results, grades, k)

        print(f"\nSearch Results Evaluation:")
        print(f"\nGood ranking: {good_results}")
        print(f"  nDCG@{k}: {ndcg_good:.3f}")
        print(f"\nPoor ranking: {poor_results}")
        print(f"  nDCG@{k}: {ndcg_poor:.3f}")
        print(f"\nDifference: {(ndcg_good - ndcg_poor) / ndcg_poor * 100:.1f}% improvement")
        print()


def example_list_queries():
    """Example: List all test queries"""
    print("=" * 60)
    print("Example: Listing Test Queries")
    print("=" * 60)

    with RelevanceManager(DATABASE_URL) as manager:
        queries = manager.list_test_queries(limit=10)

        print(f"\n✓ Found {len(queries)} test queries:\n")
        for q in queries:
            print(f"  [{q.query_id}] {q.query_text[:50]}...")
            print(f"      Type: {q.query_type or 'N/A'}, Category: {q.category or 'N/A'}")

            # Check if it has grades
            grades = manager.get_relevance_grades(q.query_id)
            if grades:
                print(f"      Labeled: {len(grades)} documents")
            print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Database Integration Examples")
    print("=" * 60)
    print(f"\nUsing database: {DATABASE_URL}")
    print("\nNote: Ensure evaluation schema is installed:")
    print("  psql -f lab/setup/evaluation_schema.sql")
    print()

    try:
        # Example 1: Create query
        query_id = example_create_test_query()

        # Example 2: Add grades
        example_add_relevance_grades(query_id)

        # Example 3: Calculate nDCG
        example_calculate_ndcg(query_id)

        # Example 4: List all queries
        example_list_queries()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. View queries in Streamlit UI: Evaluation > Test Queries")
        print("2. Add more relevance grades via UI or API")
        print("3. Run evaluations to track nDCG over time")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check DATABASE_URL environment variable")
        print("2. Ensure evaluation schema is installed")
        print("3. Verify database connection")
        print()


if __name__ == "__main__":
    main()
