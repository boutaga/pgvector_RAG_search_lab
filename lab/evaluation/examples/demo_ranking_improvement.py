#!/usr/bin/env python3
"""
Quick Demo: Ranking Quality Improvement
========================================

Shows before/after comparison of search ranking optimizations.

Usage:
    python lab/evaluation/examples/demo_ranking_improvement.py

This script demonstrates:
1. Baseline vector search (content-only)
2. Title-weighted search (70% title, 30% content)
3. Side-by-side comparison showing improvement

Expected improvement:
- Recall@10: 75% → 100%
- nDCG: 0.779 → 0.85+
- Precision@10: ~20% → 40%
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab.core.database import DatabaseService
from lab.core.embeddings import OpenAIEmbedder
from lab.core.search import VectorSearch
from lab.evaluation.evaluator import TestCase, RetrievalEvaluator


@dataclass
class SearchResult:
    id: int
    title: str
    content: str
    score: float
    metadata: Dict = None


class TitleWeightedSearch:
    """Search that weights title matches higher than content"""

    def __init__(
        self,
        db_service: DatabaseService,
        embedding_service,
        title_weight: float = 0.7,
        content_weight: float = 0.3
    ):
        self.db = db_service
        self.embedder = embedding_service
        self.title_weight = title_weight
        self.content_weight = content_weight

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search with title weighting"""

        # Get query embedding (generate_embeddings returns list, take first)
        query_embedding = self.embedder.generate_embeddings([query])[0]

        # SQL with weighted combination
        sql = """
        SELECT
            id,
            title,
            content,
            (
                %s * (1 - (title_vector_3072 <=> %s::vector)) +
                %s * (1 - (content_vector_3072 <=> %s::vector))
            ) as combined_score
        FROM articles
        WHERE title_vector_3072 IS NOT NULL
          AND content_vector_3072 IS NOT NULL
        ORDER BY combined_score DESC
        LIMIT %s
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    self.title_weight,
                    query_embedding,
                    self.content_weight,
                    query_embedding,
                    top_k
                ))
                results = cur.fetchall()

        return [
            SearchResult(
                id=row[0],
                title=row[1],
                content=row[2],
                score=row[3],
                metadata={"method": "title_weighted"}
            )
            for row in results
        ]


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)

    return [
        TestCase(
            query=item['query'],
            expected_doc_ids=item.get('expected_doc_ids', []),
            expected_answer=item.get('expected_answer'),
            metadata=item.get('metadata', {})
        )
        for item in data
    ]


def evaluate_search(search, test_cases: List[TestCase], k: int = 10):
    """Evaluate search method on test cases"""

    results = []

    for test_case in test_cases:
        start_time = time.time()

        # Perform search
        search_results = search.search(test_case.query, top_k=k)
        latency_ms = (time.time() - start_time) * 1000

        # Extract IDs
        retrieved_ids = [r.id for r in search_results]

        # Calculate metrics
        precision = RetrievalEvaluator.precision_at_k(
            retrieved_ids, test_case.expected_doc_ids, k
        )
        recall = RetrievalEvaluator.recall_at_k(
            retrieved_ids, test_case.expected_doc_ids, k
        )
        ndcg = RetrievalEvaluator.ndcg_at_k(
            retrieved_ids, test_case.expected_doc_ids, k
        )
        mrr = RetrievalEvaluator.mean_reciprocal_rank(
            retrieved_ids, test_case.expected_doc_ids
        )

        relevant_found = len(set(retrieved_ids) & set(test_case.expected_doc_ids))

        results.append({
            'query': test_case.query,
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'mrr': mrr,
            'latency_ms': latency_ms,
            'relevant_found': relevant_found,
            'relevant_total': len(test_case.expected_doc_ids),
            'retrieved_ids': retrieved_ids[:10]  # Top 10 for display
        })

    return results


def print_comparison(baseline_results, improved_results):
    """Print side-by-side comparison"""

    print("\n" + "="*100)
    print("RANKING IMPROVEMENT DEMONSTRATION")
    print("="*100)

    for i, (baseline, improved) in enumerate(zip(baseline_results, improved_results)):
        print(f"\n📝 Query {i+1}: \"{baseline['query']}\"")
        print("-"*100)

        # Metrics comparison
        metrics_data = [
            [
                "Recall @ k=10",
                f"{baseline['recall']:.1%}",
                f"{improved['recall']:.1%}",
                f"{'+' if improved['recall'] > baseline['recall'] else ''}{(improved['recall'] - baseline['recall'])*100:.0f}%"
            ],
            [
                "Precision @ k=10",
                f"{baseline['precision']:.1%}",
                f"{improved['precision']:.1%}",
                f"{'+' if improved['precision'] > baseline['precision'] else ''}{(improved['precision'] - baseline['precision'])*100:.0f}%"
            ],
            [
                "nDCG",
                f"{baseline['ndcg']:.3f}",
                f"{improved['ndcg']:.3f}",
                f"{'+' if improved['ndcg'] > baseline['ndcg'] else ''}{(improved['ndcg'] - baseline['ndcg']):.3f}"
            ],
            [
                "MRR",
                f"{baseline['mrr']:.3f}",
                f"{improved['mrr']:.3f}",
                f"{'+' if improved['mrr'] > baseline['mrr'] else ''}{(improved['mrr'] - baseline['mrr']):.3f}"
            ],
            [
                "Relevant Found",
                f"{baseline['relevant_found']}/{baseline['relevant_total']}",
                f"{improved['relevant_found']}/{improved['relevant_total']}",
                "✅" if improved['relevant_found'] == improved['relevant_total'] else "❌"
            ]
        ]

        headers = ["Metric", "Baseline (Content)", "Improved (Title-Weighted)", "Change"]
        print(tabulate(metrics_data, headers=headers, tablefmt="grid"))

    # Overall summary
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)

    avg_baseline_recall = sum(r['recall'] for r in baseline_results) / len(baseline_results)
    avg_improved_recall = sum(r['recall'] for r in improved_results) / len(improved_results)

    avg_baseline_ndcg = sum(r['ndcg'] for r in baseline_results) / len(baseline_results)
    avg_improved_ndcg = sum(r['ndcg'] for r in improved_results) / len(improved_results)

    avg_baseline_precision = sum(r['precision'] for r in baseline_results) / len(baseline_results)
    avg_improved_precision = sum(r['precision'] for r in improved_results) / len(improved_results)

    summary_data = [
        [
            "Average Recall @ k=10",
            f"{avg_baseline_recall:.1%}",
            f"{avg_improved_recall:.1%}",
            f"+{(avg_improved_recall - avg_baseline_recall)*100:.0f}%"
        ],
        [
            "Average Precision @ k=10",
            f"{avg_baseline_precision:.1%}",
            f"{avg_improved_precision:.1%}",
            f"+{(avg_improved_precision - avg_baseline_precision)*100:.0f}%"
        ],
        [
            "Average nDCG",
            f"{avg_baseline_ndcg:.3f}",
            f"{avg_improved_ndcg:.3f}",
            f"+{(avg_improved_ndcg - avg_baseline_ndcg):.3f}"
        ]
    ]

    print(tabulate(summary_data, headers=headers[:3] + ["Improvement"], tablefmt="grid"))

    print("\n🎯 KEY INSIGHT:")
    if avg_improved_recall > avg_baseline_recall:
        print(f"   ✅ Title weighting improved recall by {(avg_improved_recall - avg_baseline_recall)*100:.0f}%")
        print(f"   ✅ More relevant documents are now in the top-10!")
    if avg_improved_ndcg > avg_baseline_ndcg:
        print(f"   ✅ nDCG improved by {(avg_improved_ndcg - avg_baseline_ndcg):.3f}")
        print(f"   ✅ Relevant documents are ranked higher!")

    print()


def main():
    print("="*100)
    print("RANKING IMPROVEMENT DEMO")
    print("="*100)
    print("\nThis demo compares:")
    print("  1. Baseline: Vector search on content only")
    print("  2. Improved: Title-weighted search (70% title, 30% content)")
    print()

    # Check for test file
    test_file = "lab/evaluation/test_cases_expanded.json"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("\nUsing simple test case instead...")

        # Create simple test case
        test_cases = [
            TestCase(
                query="What is machine learning?",
                expected_doc_ids=[38080, 6360, 44742, 44678],  # From your actual data
                expected_answer=None,
                metadata={}
            )
        ]
    else:
        print(f"📋 Loading test cases from {test_file}...")
        test_cases = load_test_cases(test_file)

    print(f"✓ Loaded {len(test_cases)} test case(s)")

    # Initialize services
    print("\n🔌 Initializing database and embedder...")
    db = DatabaseService()
    embedder = OpenAIEmbedder()

    # Create baseline search (content-only)
    print("\n1️⃣ Creating baseline search (content-only vector search)...")
    baseline_search = VectorSearch(
        db_service=db,
        embedding_service=embedder,
        table_name="articles",
        vector_column="content_vector_3072",
        content_columns=["title", "content"],
        id_column="id"
    )

    # Create improved search (title-weighted)
    print("2️⃣ Creating improved search (title-weighted 70/30)...")
    improved_search = TitleWeightedSearch(
        db_service=db,
        embedding_service=embedder,
        title_weight=0.7,
        content_weight=0.3
    )

    # Run evaluations
    print("\n🧪 Running baseline evaluation...")
    baseline_results = evaluate_search(baseline_search, test_cases, k=10)

    print("🧪 Running improved evaluation...")
    improved_results = evaluate_search(improved_search, test_cases, k=10)

    # Print comparison
    print_comparison(baseline_results, improved_results)


if __name__ == "__main__":
    main()
