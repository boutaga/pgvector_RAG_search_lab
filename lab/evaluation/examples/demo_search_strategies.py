#!/usr/bin/env python3
"""
Search Strategies Comparison Demo
=================================

Compares multiple search strategies to demonstrate how different approaches
affect retrieval quality for various query types.

Strategies compared:
1. Content-only: Vector search on content embeddings only
2. Title-only: Vector search on title embeddings only
3. Title-Weighted: 70% title + 30% content vector weighted combination
4. Hybrid (Dense+Sparse): Combines dense and sparse vectors (if available)
5. Title FTS Rerank: Hybrid search with PostgreSQL title full-text reranking

Usage:
    python lab/evaluation/examples/demo_search_strategies.py
    python lab/evaluation/examples/demo_search_strategies.py --test-file lab/evaluation/my_tests.json
    python lab/evaluation/examples/demo_search_strategies.py --k 20

This demo helps identify:
- Which strategy works best for factual vs conceptual queries
- How title matching improves recall for entity-focused queries
- When hybrid search outperforms pure dense search
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab.core.database import DatabaseService
from lab.core.embeddings import OpenAIEmbedder, SPLADEEmbedder
from lab.core.search import VectorSearch, SparseSearch, HybridSearch, SearchResult
from lab.evaluation.evaluator import TestCase, RetrievalEvaluator


@dataclass
class StrategyResult:
    """Results for a single strategy on a single query."""
    strategy_name: str
    query: str
    precision: float
    recall: float
    ndcg: float
    mrr: float
    relevant_found: int
    total_relevant: int
    latency_ms: float
    top_ids: List[int]


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from JSON file."""
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


class SearchStrategyEvaluator:
    """Evaluates multiple search strategies."""

    def __init__(
        self,
        db_service: DatabaseService,
        dense_embedder: OpenAIEmbedder,
        sparse_embedder: Optional[SPLADEEmbedder] = None,
        table_name: str = "articles",
        content_vector_col: str = "content_vector_3072",
        title_vector_col: str = "title_vector_3072",
        sparse_col: str = "content_sparse",
        id_column: str = "id"
    ):
        self.db = db_service
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.table_name = table_name
        self.content_vector_col = content_vector_col
        self.title_vector_col = title_vector_col
        self.sparse_col = sparse_col
        self.id_column = id_column

        # Initialize search services
        self._init_search_services()

    def _init_search_services(self):
        """Initialize all search service variants."""
        content_columns = ["title", "content"]

        # Content-only vector search
        self.content_search = VectorSearch(
            db_service=self.db,
            embedding_service=self.dense_embedder,
            table_name=self.table_name,
            vector_column=self.content_vector_col,
            content_columns=content_columns,
            id_column=self.id_column,
            title_vector_column=self.title_vector_col
        )

        # Title-only vector search
        self.title_search = VectorSearch(
            db_service=self.db,
            embedding_service=self.dense_embedder,
            table_name=self.table_name,
            vector_column=self.title_vector_col,
            content_columns=content_columns,
            id_column=self.id_column
        )

        # Sparse search (if embedder available)
        self.sparse_search = None
        if self.sparse_embedder:
            self.sparse_search = SparseSearch(
                db_service=self.db,
                embedding_service=self.sparse_embedder,
                table_name=self.table_name,
                sparse_column=self.sparse_col,
                content_columns=content_columns,
                id_column=self.id_column
            )

    def _evaluate_strategy(
        self,
        strategy_name: str,
        search_func,
        test_case: TestCase,
        k: int
    ) -> StrategyResult:
        """Evaluate a single strategy on a single query."""
        start_time = time.time()
        results = search_func(test_case.query, top_k=k)
        latency_ms = (time.time() - start_time) * 1000

        retrieved_ids = [r.id for r in results]

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

        return StrategyResult(
            strategy_name=strategy_name,
            query=test_case.query,
            precision=precision,
            recall=recall,
            ndcg=ndcg,
            mrr=mrr,
            relevant_found=relevant_found,
            total_relevant=len(test_case.expected_doc_ids),
            latency_ms=latency_ms,
            top_ids=retrieved_ids[:5]
        )

    def evaluate_all_strategies(
        self,
        test_cases: List[TestCase],
        k: int = 10,
        include_sparse: bool = True
    ) -> Dict[str, List[StrategyResult]]:
        """
        Evaluate all strategies on all test cases.

        Returns:
            Dict mapping strategy name to list of results per query
        """
        strategies = {
            "Content-Only": lambda q, top_k: self.content_search.search(
                q, top_k=top_k, search_mode="content"
            ),
            "Title-Only": lambda q, top_k: self.content_search.search(
                q, top_k=top_k, search_mode="title"
            ),
            "Title-Weighted (70/30)": lambda q, top_k: self.content_search.search(
                q, top_k=top_k, search_mode="combined", title_weight=0.7
            ),
            "Title-Weighted (50/50)": lambda q, top_k: self.content_search.search(
                q, top_k=top_k, search_mode="combined", title_weight=0.5
            ),
            "Title-Weighted (30/70)": lambda q, top_k: self.content_search.search(
                q, top_k=top_k, search_mode="combined", title_weight=0.3
            ),
        }

        # Add hybrid strategies if sparse search is available
        if include_sparse and self.sparse_search:
            hybrid_dense_heavy = HybridSearch(
                self.content_search,
                self.sparse_search,
                dense_weight=0.7,
                sparse_weight=0.3
            )
            hybrid_balanced = HybridSearch(
                self.content_search,
                self.sparse_search,
                dense_weight=0.5,
                sparse_weight=0.5
            )
            hybrid_sparse_heavy = HybridSearch(
                self.content_search,
                self.sparse_search,
                dense_weight=0.3,
                sparse_weight=0.7
            )
            # Hybrid with title FTS reranking
            hybrid_with_fts = HybridSearch(
                self.content_search,
                self.sparse_search,
                dense_weight=0.5,
                sparse_weight=0.5
            )

            strategies.update({
                "Hybrid (70% Dense)": lambda q, top_k: hybrid_dense_heavy.search(q, top_k=top_k),
                "Hybrid (50/50)": lambda q, top_k: hybrid_balanced.search(q, top_k=top_k),
                "Hybrid (70% Sparse)": lambda q, top_k: hybrid_sparse_heavy.search(q, top_k=top_k),
                "Hybrid + Title FTS": lambda q, top_k: hybrid_with_fts.search(
                    q, top_k=top_k, title_fts_rerank=True, title_fts_weight=0.2
                ),
            })

        results = {name: [] for name in strategies}

        for test_case in test_cases:
            print(f"\n  Query: \"{test_case.query[:50]}...\"" if len(test_case.query) > 50 else f"\n  Query: \"{test_case.query}\"")

            for strategy_name, search_func in strategies.items():
                result = self._evaluate_strategy(
                    strategy_name, search_func, test_case, k
                )
                results[strategy_name].append(result)
                print(f"    {strategy_name}: Recall={result.recall:.0%}, nDCG={result.ndcg:.3f}")

        return results


def print_per_query_comparison(
    results: Dict[str, List[StrategyResult]],
    test_cases: List[TestCase]
):
    """Print detailed per-query comparison across all strategies."""
    print("\n" + "=" * 120)
    print("PER-QUERY BREAKDOWN")
    print("=" * 120)

    strategy_names = list(results.keys())

    for i, test_case in enumerate(test_cases):
        category = test_case.metadata.get('category', 'unknown')
        difficulty = test_case.metadata.get('difficulty', 'unknown')

        print(f"\n{'─' * 120}")
        print(f"📝 Query {i+1}: \"{test_case.query}\"")
        print(f"   Category: {category} | Difficulty: {difficulty} | Expected docs: {len(test_case.expected_doc_ids)}")
        print(f"{'─' * 120}")

        # Build comparison table
        table_data = []
        for strategy_name in strategy_names:
            r = results[strategy_name][i]
            found_status = "✅" if r.relevant_found == r.total_relevant else f"❌ ({r.relevant_found}/{r.total_relevant})"
            table_data.append([
                strategy_name,
                f"{r.recall:.0%}",
                f"{r.precision:.0%}",
                f"{r.ndcg:.3f}",
                f"{r.mrr:.3f}",
                found_status,
                f"{r.latency_ms:.0f}ms"
            ])

        headers = ["Strategy", "Recall", "Precision", "nDCG", "MRR", "Found", "Latency"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Find best strategy for this query
        best_recall = max(results[s][i].recall for s in strategy_names)
        best_ndcg = max(results[s][i].ndcg for s in strategy_names)
        best_strategies_recall = [s for s in strategy_names if results[s][i].recall == best_recall]
        best_strategies_ndcg = [s for s in strategy_names if results[s][i].ndcg == best_ndcg]

        print(f"\n   🏆 Best Recall: {best_strategies_recall[0]} ({best_recall:.0%})")
        print(f"   🏆 Best nDCG: {best_strategies_ndcg[0]} ({best_ndcg:.3f})")


def print_overall_summary(results: Dict[str, List[StrategyResult]]):
    """Print overall summary comparing all strategies."""
    print("\n" + "=" * 120)
    print("OVERALL STRATEGY COMPARISON")
    print("=" * 120)

    summary_data = []
    strategy_names = list(results.keys())

    for strategy_name in strategy_names:
        strategy_results = results[strategy_name]
        n = len(strategy_results)

        avg_recall = sum(r.recall for r in strategy_results) / n
        avg_precision = sum(r.precision for r in strategy_results) / n
        avg_ndcg = sum(r.ndcg for r in strategy_results) / n
        avg_mrr = sum(r.mrr for r in strategy_results) / n
        avg_latency = sum(r.latency_ms for r in strategy_results) / n
        total_found = sum(r.relevant_found for r in strategy_results)
        total_relevant = sum(r.total_relevant for r in strategy_results)

        summary_data.append([
            strategy_name,
            f"{avg_recall:.1%}",
            f"{avg_precision:.1%}",
            f"{avg_ndcg:.3f}",
            f"{avg_mrr:.3f}",
            f"{total_found}/{total_relevant}",
            f"{avg_latency:.0f}ms"
        ])

    headers = ["Strategy", "Avg Recall", "Avg Precision", "Avg nDCG", "Avg MRR", "Total Found", "Avg Latency"]
    print("\n" + tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Find overall winners
    metrics = ['recall', 'precision', 'ndcg', 'mrr']
    print("\n🏆 WINNERS BY METRIC:")
    print("-" * 60)

    for metric in metrics:
        best_value = -1
        best_strategy = None
        for strategy_name in strategy_names:
            strategy_results = results[strategy_name]
            avg = sum(getattr(r, metric) for r in strategy_results) / len(strategy_results)
            if avg > best_value:
                best_value = avg
                best_strategy = strategy_name

        print(f"   {metric.upper():12} → {best_strategy} ({best_value:.3f})")


def print_category_breakdown(
    results: Dict[str, List[StrategyResult]],
    test_cases: List[TestCase]
):
    """Print breakdown by query category (factual, conceptual, etc.)."""
    print("\n" + "=" * 120)
    print("BREAKDOWN BY QUERY CATEGORY")
    print("=" * 120)

    # Group queries by category
    categories = {}
    for i, tc in enumerate(test_cases):
        cat = tc.metadata.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    strategy_names = list(results.keys())

    for category, indices in sorted(categories.items()):
        print(f"\n📂 Category: {category.upper()} ({len(indices)} queries)")
        print("-" * 100)

        table_data = []
        for strategy_name in strategy_names:
            cat_results = [results[strategy_name][i] for i in indices]
            n = len(cat_results)

            avg_recall = sum(r.recall for r in cat_results) / n
            avg_ndcg = sum(r.ndcg for r in cat_results) / n
            total_found = sum(r.relevant_found for r in cat_results)
            total_relevant = sum(r.total_relevant for r in cat_results)

            table_data.append([
                strategy_name,
                f"{avg_recall:.0%}",
                f"{avg_ndcg:.3f}",
                f"{total_found}/{total_relevant}"
            ])

        headers = ["Strategy", "Avg Recall", "Avg nDCG", "Found"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Best strategy for this category
        best_recall_strategy = max(
            strategy_names,
            key=lambda s: sum(results[s][i].recall for i in indices) / len(indices)
        )
        print(f"\n   🏆 Best for {category}: {best_recall_strategy}")


def print_recommendations(results: Dict[str, List[StrategyResult]], test_cases: List[TestCase]):
    """Print recommendations based on results."""
    print("\n" + "=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    strategy_names = list(results.keys())

    # Calculate average metrics for each strategy
    strategy_metrics = {}
    for strategy_name in strategy_names:
        strategy_results = results[strategy_name]
        n = len(strategy_results)
        strategy_metrics[strategy_name] = {
            'recall': sum(r.recall for r in strategy_results) / n,
            'precision': sum(r.precision for r in strategy_results) / n,
            'ndcg': sum(r.ndcg for r in strategy_results) / n,
            'latency': sum(r.latency_ms for r in strategy_results) / n
        }

    # Find best strategies
    best_recall = max(strategy_names, key=lambda s: strategy_metrics[s]['recall'])
    best_ndcg = max(strategy_names, key=lambda s: strategy_metrics[s]['ndcg'])
    fastest = min(strategy_names, key=lambda s: strategy_metrics[s]['latency'])

    print(f"""
Key Insights:
─────────────
1. **Best Overall Recall**: {best_recall}
   - Recall: {strategy_metrics[best_recall]['recall']:.1%}
   - Use when: You need to find ALL relevant documents

2. **Best Ranking Quality (nDCG)**: {best_ndcg}
   - nDCG: {strategy_metrics[best_ndcg]['ndcg']:.3f}
   - Use when: Ranking order matters (e.g., for RAG context)

3. **Fastest**: {fastest}
   - Latency: {strategy_metrics[fastest]['latency']:.0f}ms
   - Use when: Response time is critical

Strategy Selection Guide:
─────────────────────────
- For FACTUAL queries (who, when, where):
  → Prefer Title-Weighted or Hybrid with sparse emphasis
  → Title matching helps surface exact entity articles

- For CONCEPTUAL queries (why, how, explain):
  → Prefer Content-Only or Hybrid with dense emphasis
  → Semantic understanding captures abstract concepts better

- For LOW RECALL scenarios:
  → Increase k_retrieve to build larger candidate pool
  → Try Hybrid search to combine lexical + semantic matching

- For LOW nDCG scenarios:
  → Add re-ranking stage (Title FTS, cross-encoder)
  → Tune title_weight parameter for your query types

Next Steps:
───────────
1. Experiment with different title_weight values (0.3, 0.5, 0.7)
2. Try Hybrid + Title FTS for better ranking
3. Test with more diverse query types
4. Consider adding cross-encoder re-ranking for production
""")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple search strategies for RAG retrieval"
    )
    parser.add_argument(
        "--test-file",
        default="lab/evaluation/test_cases_expanded.json",
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--no-sparse", action="store_true",
        help="Disable sparse/hybrid search strategies"
    )
    parser.add_argument(
        "--content-vector",
        default=os.getenv("CONTENT_VECTOR_COLUMN", "content_vector_3072"),
        help="Content vector column name"
    )
    parser.add_argument(
        "--title-vector",
        default=os.getenv("TITLE_VECTOR_COLUMN", "title_vector_3072"),
        help="Title vector column name"
    )

    args = parser.parse_args()

    print("=" * 120)
    print("SEARCH STRATEGIES COMPARISON DEMO")
    print("=" * 120)
    print(f"""
This demo compares multiple search strategies:

Dense Vector Strategies:
  1. Content-Only        - Search only on content embeddings
  2. Title-Only          - Search only on title embeddings
  3. Title-Weighted      - Combine title and content vectors with different weights

Hybrid Strategies (Dense + Sparse):
  4. Hybrid Dense-Heavy  - 70% dense, 30% sparse
  5. Hybrid Balanced     - 50% dense, 50% sparse
  6. Hybrid Sparse-Heavy - 30% dense, 70% sparse
  7. Hybrid + Title FTS  - Hybrid with PostgreSQL title full-text reranking
""")

    # Load test cases
    if not os.path.exists(args.test_file):
        print(f"❌ Test file not found: {args.test_file}")
        print("\nCreating sample test cases...")
        test_cases = [
            TestCase(
                query="What is machine learning?",
                expected_doc_ids=[38080, 6360, 44742, 44678],
                metadata={"category": "conceptual", "difficulty": "easy"}
            ),
            TestCase(
                query="Who invented the telephone?",
                expected_doc_ids=[4579, 18368],
                metadata={"category": "factual", "difficulty": "easy"}
            ),
            TestCase(
                query="neural networks deep learning",
                expected_doc_ids=[44742, 6360],
                metadata={"category": "conceptual", "difficulty": "medium"}
            ),
            TestCase(
                query="animal whales",
                expected_doc_ids=[7597, 15244, 31022, 30442],
                metadata={"category": "factual", "difficulty": "easy"}
            )
        ]
    else:
        print(f"📋 Loading test cases from {args.test_file}...")
        test_cases = load_test_cases(args.test_file)

    print(f"✓ Loaded {len(test_cases)} test case(s)")
    print(f"✓ Using k={args.k}")

    # Initialize services
    print("\n🔌 Initializing services...")
    db = DatabaseService()
    dense_embedder = OpenAIEmbedder()

    sparse_embedder = None
    if not args.no_sparse:
        try:
            print("   Loading SPLADE embedder (this may take a moment)...")
            sparse_embedder = SPLADEEmbedder()
            print("   ✓ SPLADE embedder loaded")
        except Exception as e:
            print(f"   ⚠ Could not load SPLADE embedder: {e}")
            print("   → Continuing without hybrid strategies")

    # Create evaluator
    evaluator = SearchStrategyEvaluator(
        db_service=db,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        content_vector_col=args.content_vector,
        title_vector_col=args.title_vector
    )

    # Run evaluation
    print("\n🧪 Running strategy evaluations...")
    results = evaluator.evaluate_all_strategies(
        test_cases,
        k=args.k,
        include_sparse=(sparse_embedder is not None)
    )

    # Print results
    print_per_query_comparison(results, test_cases)
    print_overall_summary(results)
    print_category_breakdown(results, test_cases)
    print_recommendations(results, test_cases)

    print("\n✓ Demo completed!")


if __name__ == "__main__":
    main()
