#!/usr/bin/env python3
"""
Comprehensive Search Configuration Comparison
==============================================

Compares different search configurations across multiple dimensions:
- k_retrieve values (candidate pool size)
- k_context values (LLM context size)
- Search methods (vector, hybrid, adaptive)
- Vector models (1536 vs 3072 dimensions)

Outputs a comparison table and optionally saves results to JSON/CSV.
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from tabulate import tabulate
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab.core.database import DatabaseService
from lab.core.embeddings import OpenAIEmbedder, SPLADEEmbedder
from lab.core.search import VectorSearch, SparseSearch, HybridSearch, AdaptiveSearch
from lab.evaluation.evaluator import TestCase, RetrievalEvaluator


@dataclass
class SearchConfig:
    """Configuration for a search experiment"""
    name: str
    search_type: str  # 'vector', 'sparse', 'hybrid', 'adaptive'
    vector_column: str = "content_vector_3072"
    k_retrieve: int = 100
    k_context: int = 8
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config_name: str
    search_type: str
    k_retrieve: int
    k_context: int
    query: str
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float
    latency_ms: float
    docs_retrieved: int
    docs_relevant_found: int
    docs_relevant_total: int


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append(TestCase(
            query=item['query'],
            expected_doc_ids=item.get('expected_doc_ids', []),
            expected_answer=item.get('expected_answer'),
            metadata=item.get('metadata', {})
        ))

    return test_cases


def run_experiment(
    test_cases: List[TestCase],
    config: SearchConfig,
    db: DatabaseService,
    embedder: OpenAIEmbedder,
    sparse_embedder: SPLADEEmbedder = None
) -> List[ExperimentResult]:
    """Run experiment with given configuration"""

    results = []

    # Initialize appropriate search
    if config.search_type == 'vector':
        search = VectorSearch(
            db_service=db,
            embedding_service=embedder,
            table_name="articles",
            vector_column=config.vector_column,
            content_columns=["title", "content"],
            id_column="id"
        )
    elif config.search_type == 'sparse':
        if sparse_embedder is None:
            print(f"⚠️  Skipping {config.name} - sparse embedder not available")
            return results
        search = SparseSearch(
            db_service=db,
            embedding_service=sparse_embedder,
            table_name="articles",
            sparse_column="content_sparse",
            content_columns=["title", "content"],
            id_column="id"
        )
    elif config.search_type == 'hybrid':
        if sparse_embedder is None:
            print(f"⚠️  Skipping {config.name} - sparse embedder not available")
            return results

        dense_search = VectorSearch(
            db_service=db,
            embedding_service=embedder,
            table_name="articles",
            vector_column=config.vector_column,
            content_columns=["title", "content"],
            id_column="id"
        )
        sparse_search = SparseSearch(
            db_service=db,
            embedding_service=sparse_embedder,
            table_name="articles",
            sparse_column="content_sparse",
            content_columns=["title", "content"],
            id_column="id"
        )
        search = HybridSearch(
            dense_search=dense_search,
            sparse_search=sparse_search,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight
        )
    elif config.search_type == 'adaptive':
        if sparse_embedder is None:
            print(f"⚠️  Skipping {config.name} - sparse embedder not available")
            return results

        dense_search = VectorSearch(
            db_service=db,
            embedding_service=embedder,
            table_name="articles",
            vector_column=config.vector_column,
            content_columns=["title", "content"],
            id_column="id"
        )
        sparse_search = SparseSearch(
            db_service=db,
            embedding_service=sparse_embedder,
            table_name="articles",
            sparse_column="content_sparse",
            content_columns=["title", "content"],
            id_column="id"
        )
        search = AdaptiveSearch(
            dense_search=dense_search,
            sparse_search=sparse_search
        )
    else:
        raise ValueError(f"Unknown search type: {config.search_type}")

    # Run each test case
    for test_case in test_cases:
        start_time = time.time()

        try:
            # Perform search
            search_results = search.search(test_case.query, top_k=config.k_retrieve)
            latency_ms = (time.time() - start_time) * 1000

            # Extract IDs
            retrieved_ids = [r.id for r in search_results]

            # Calculate metrics
            precision = RetrievalEvaluator.precision_at_k(
                retrieved_ids, test_case.expected_doc_ids, config.k_retrieve
            )
            recall = RetrievalEvaluator.recall_at_k(
                retrieved_ids, test_case.expected_doc_ids, config.k_retrieve
            )
            f1 = RetrievalEvaluator.f1_at_k(
                retrieved_ids, test_case.expected_doc_ids, config.k_retrieve
            )
            ndcg = RetrievalEvaluator.ndcg_at_k(
                retrieved_ids, test_case.expected_doc_ids, config.k_retrieve
            )
            mrr = RetrievalEvaluator.mean_reciprocal_rank(
                retrieved_ids, test_case.expected_doc_ids
            )

            # Count relevant docs found
            relevant_found = len(set(retrieved_ids) & set(test_case.expected_doc_ids))

            results.append(ExperimentResult(
                config_name=config.name,
                search_type=config.search_type,
                k_retrieve=config.k_retrieve,
                k_context=config.k_context,
                query=test_case.query,
                precision=precision,
                recall=recall,
                f1=f1,
                ndcg=ndcg,
                mrr=mrr,
                latency_ms=latency_ms,
                docs_retrieved=len(search_results),
                docs_relevant_found=relevant_found,
                docs_relevant_total=len(test_case.expected_doc_ids)
            ))

        except Exception as e:
            print(f"❌ Error processing query '{test_case.query}': {e}")
            continue

    return results


def aggregate_results(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Aggregate results by configuration"""

    config_groups = {}

    for result in results:
        key = f"{result.config_name}"
        if key not in config_groups:
            config_groups[key] = {
                'config_name': result.config_name,
                'search_type': result.search_type,
                'k_retrieve': result.k_retrieve,
                'k_context': result.k_context,
                'results': []
            }
        config_groups[key]['results'].append(result)

    # Calculate aggregates
    aggregated = []
    for key, group in config_groups.items():
        res_list = group['results']

        aggregated.append({
            'Configuration': group['config_name'],
            'Search Type': group['search_type'],
            'k_retrieve': group['k_retrieve'],
            'k_context': group['k_context'],
            'Queries': len(res_list),
            'Avg Precision': np.mean([r.precision for r in res_list]),
            'Avg Recall': np.mean([r.recall for r in res_list]),
            'Avg F1': np.mean([r.f1 for r in res_list]),
            'Avg nDCG': np.mean([r.ndcg for r in res_list]),
            'Avg MRR': np.mean([r.mrr for r in res_list]),
            'Avg Latency (ms)': np.mean([r.latency_ms for r in res_list]),
            'Total Relevant Found': sum([r.docs_relevant_found for r in res_list]),
            'Total Relevant Expected': sum([r.docs_relevant_total for r in res_list])
        })

    return aggregated


def print_comparison_table(aggregated: List[Dict[str, Any]]):
    """Print a nice comparison table"""

    # Prepare table data
    table_data = []
    for row in aggregated:
        table_data.append([
            row['Configuration'],
            row['Search Type'],
            f"{row['k_retrieve']}/{row['k_context']}",
            f"{row['Avg Recall']:.3f}",
            f"{row['Avg Precision']:.3f}",
            f"{row['Avg F1']:.3f}",
            f"{row['Avg nDCG']:.3f}",
            f"{row['Avg MRR']:.3f}",
            f"{row['Avg Latency (ms)']:.0f}",
            f"{row['Total Relevant Found']}/{row['Total Relevant Expected']}"
        ])

    headers = [
        "Configuration",
        "Type",
        "k_r/k_c",
        "Recall",
        "Precision",
        "F1",
        "nDCG",
        "MRR",
        "Latency",
        "Found"
    ]

    print("\n" + "="*120)
    print("SEARCH CONFIGURATION COMPARISON")
    print("="*120)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def print_detailed_results(results: List[ExperimentResult]):
    """Print detailed per-query results"""

    print("\n" + "="*120)
    print("DETAILED RESULTS BY QUERY")
    print("="*120)

    # Group by query
    by_query = {}
    for r in results:
        if r.query not in by_query:
            by_query[r.query] = []
        by_query[r.query].append(r)

    for query, query_results in by_query.items():
        print(f"\nQuery: {query}")
        print("-" * 100)

        table_data = []
        for r in query_results:
            table_data.append([
                r.config_name,
                r.search_type,
                f"{r.k_retrieve}/{r.k_context}",
                f"{r.recall:.3f}",
                f"{r.precision:.3f}",
                f"{r.ndcg:.3f}",
                f"{r.latency_ms:.0f}",
                f"{r.docs_relevant_found}/{r.docs_relevant_total}"
            ])

        headers = ["Config", "Type", "k_r/k_c", "Recall", "Precision", "nDCG", "Latency", "Found"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description='Compare search configurations')
    parser.add_argument('--test-file', required=True, help='Test cases JSON file')
    parser.add_argument('--db-url', help='Database URL (or use DATABASE_URL env)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--output-csv', help='Output file for results (CSV)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-query results')
    parser.add_argument('--skip-sparse', action='store_true', help='Skip sparse/hybrid searches')

    args = parser.parse_args()

    print("="*120)
    print("COMPREHENSIVE SEARCH CONFIGURATION COMPARISON")
    print("="*120)

    # Load test cases
    print(f"\n📋 Loading test cases from {args.test_file}...")
    test_cases = load_test_cases(args.test_file)
    print(f"✓ Loaded {len(test_cases)} test cases")

    # Initialize services
    print("\n🔌 Initializing database and embedders...")
    db = DatabaseService(connection_string=args.db_url)
    dense_embedder = OpenAIEmbedder()

    sparse_embedder = None
    if not args.skip_sparse:
        try:
            sparse_embedder = SPLADEEmbedder()
            print("✓ Sparse embedder initialized")
        except Exception as e:
            print(f"⚠️  Sparse embedder not available: {e}")

    # Define configurations to compare
    configs = [
        # Vector search with different k values
        SearchConfig("Vec-3072-k10", "vector", "content_vector_3072", k_retrieve=10, k_context=5),
        SearchConfig("Vec-3072-k50", "vector", "content_vector_3072", k_retrieve=50, k_context=8),
        SearchConfig("Vec-3072-k100", "vector", "content_vector_3072", k_retrieve=100, k_context=8),
        SearchConfig("Vec-3072-k200", "vector", "content_vector_3072", k_retrieve=200, k_context=10),

        # Compare vector dimensions (NOTE: Requires content_vector column with 1536 dims)
        # SearchConfig("Vec-1536-k100", "vector", "content_vector", k_retrieve=100, k_context=8),

        # Sparse search
        SearchConfig("Sparse-k100", "sparse", k_retrieve=100, k_context=8),

        # Hybrid search
        SearchConfig("Hybrid-Balanced", "hybrid", "content_vector_3072", k_retrieve=100, k_context=8,
                    dense_weight=0.5, sparse_weight=0.5),
        SearchConfig("Hybrid-DenseHeavy", "hybrid", "content_vector_3072", k_retrieve=100, k_context=8,
                    dense_weight=0.7, sparse_weight=0.3),
        SearchConfig("Hybrid-SparseHeavy", "hybrid", "content_vector_3072", k_retrieve=100, k_context=8,
                    dense_weight=0.3, sparse_weight=0.7),

        # Adaptive search (auto-adjusts weights based on query type)
        SearchConfig("Adaptive-Auto", "adaptive", "content_vector_3072", k_retrieve=100, k_context=8),
    ]

    # Run experiments
    print(f"\n🧪 Running {len(configs)} configurations on {len(test_cases)} queries...")
    print("="*120)

    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config.name} ({config.search_type})...")

        results = run_experiment(test_cases, config, db, dense_embedder, sparse_embedder)
        all_results.extend(results)

        if results:
            avg_recall = np.mean([r.recall for r in results])
            avg_ndcg = np.mean([r.ndcg for r in results])
            print(f"  ✓ Avg Recall: {avg_recall:.3f} | Avg nDCG: {avg_ndcg:.3f}")

    # Aggregate and display results
    aggregated = aggregate_results(all_results)

    # Sort by nDCG descending
    aggregated.sort(key=lambda x: x['Avg nDCG'], reverse=True)

    print_comparison_table(aggregated)

    if args.detailed:
        print_detailed_results(all_results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'aggregated': aggregated,
                'detailed': [asdict(r) for r in all_results]
            }, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")

    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            if aggregated:
                writer = csv.DictWriter(f, fieldnames=aggregated[0].keys())
                writer.writeheader()
                writer.writerows(aggregated)
        print(f"✓ CSV saved to {args.output_csv}")

    # Print recommendations
    print("\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120)

    best_recall = max(aggregated, key=lambda x: x['Avg Recall'])
    best_ndcg = max(aggregated, key=lambda x: x['Avg nDCG'])
    fastest = min(aggregated, key=lambda x: x['Avg Latency (ms)'])

    print(f"\n🏆 Best Recall: {best_recall['Configuration']} ({best_recall['Avg Recall']:.3f})")
    print(f"🏆 Best Ranking (nDCG): {best_ndcg['Configuration']} ({best_ndcg['Avg nDCG']:.3f})")
    print(f"⚡ Fastest: {fastest['Configuration']} ({fastest['Avg Latency (ms)']:.0f}ms)")
    print()


if __name__ == "__main__":
    main()
