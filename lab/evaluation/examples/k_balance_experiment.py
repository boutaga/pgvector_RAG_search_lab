#!/usr/bin/env python3
"""
Experimenting with k_retrieve and k_context for RAG Retrieval
==============================================================

This example script helps researchers and students explore the trade-offs
between:
- **k_retrieve**: Number of candidate documents fetched from vector database
- **k_context**: Number of documents fed into LLM after re-ranking/filtering

Educational Context:
-------------------
Larger k values can increase recall and evidence sufficiency (less likely to
miss important documents), but may decrease precision and increase latency/cost
by including marginally relevant items.

Metrics Computed:
----------------
- Precision@k: Proportion of retrieved documents that are relevant
- Recall@k: Proportion of relevant documents that are retrieved
- F1@k: Harmonic mean of precision and recall
- nDCG@k: Normalized discounted cumulative gain (ranking quality)
- MRR: Mean reciprocal rank (position of first relevant document)
- Latency: Time taken for retrieval (milliseconds)
- Context Size: Approximate token count for LLM input

Usage:
------
    python k_balance_experiment.py \\
        --test-file lab/evaluation/test_cases.json \\
        --k-retrieve 200 \\
        --k-context 8 \\
        --vector-column content_vector_3072

    # Run with multiple k values for comparison
    python k_balance_experiment.py \\
        --test-file lab/evaluation/test_cases.json \\
        --k-retrieve-values 50 100 200 \\
        --k-context-values 5 8 10 \\
        --output results.json

Environment Variables Required:
------------------------------
- DATABASE_URL: PostgreSQL connection string
- OPENAI_API_KEY: OpenAI API key for embeddings

Note: This script focuses on retrieval evaluation only. It does not call
the LLM or perform re-ranking. For advanced re-ranking, use HybridSearch
or AdaptiveSearch classes.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab.core.database import DatabaseService
from lab.core.embeddings import OpenAIEmbedder
from lab.core.search import VectorSearch, SearchResult
from lab.evaluation.evaluator import TestCase, RetrievalEvaluator


def load_test_cases(path: str) -> List[TestCase]:
    """
    Load test cases from a JSON file.

    The file should contain a list of objects with:
    - query (str): The search query
    - expected_doc_ids (list[int]): Ground truth document IDs
    - expected_answer (str, optional): Expected answer text
    - metadata (dict, optional): Additional metadata

    Args:
        path: Path to JSON file containing test cases

    Returns:
        List of TestCase objects

    Raises:
        FileNotFoundError: If test file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases: List[TestCase] = []
    for item in data:
        cases.append(TestCase(
            query=item["query"],
            expected_doc_ids=item.get("expected_doc_ids", []),
            expected_answer=item.get("expected_answer"),
            metadata=item.get("metadata", {})
        ))

    return cases


def estimate_context_tokens(results: List[SearchResult]) -> int:
    """
    Estimate token count by word splitting (rough approximation).

    A more accurate estimate would use tiktoken, but this provides
    a quick proxy for context size comparison.

    Args:
        results: Search results to estimate tokens for

    Returns:
        Approximate token count
    """
    total_words = sum(len(res.content.split()) for res in results)
    # Rough approximation: 1 token ≈ 0.75 words (varies by tokenizer)
    return int(total_words * 1.33)


def run_single_experiment(
    test_cases: List[TestCase],
    db_url: str,
    table_name: str,
    vector_column: str,
    content_columns: List[str],
    k_retrieve: int,
    k_context: int,
    id_column: str = "id"
) -> List[Dict[str, Any]]:
    """
    Run retrieval experiment for all test cases with given parameters.

    Args:
        test_cases: List of test cases with queries and ground truth
        db_url: PostgreSQL connection string
        table_name: Table containing documents (e.g., 'articles')
        vector_column: Column with dense embeddings (e.g., 'content_vector_3072')
        content_columns: Columns to return as content (e.g., ['title', 'content'])
        k_retrieve: Number of candidates to retrieve from vector database
        k_context: Number of documents to include in final context
        id_column: Name of ID column (default: 'id')

    Returns:
        List of dictionaries with metrics for each query
    """
    # Initialize services
    print(f"Initializing database connection to {table_name}...")
    db_service = DatabaseService(connection_string=db_url)

    print("Initializing OpenAI embedder...")
    embedder = OpenAIEmbedder()

    print(f"Setting up vector search on {vector_column}...")
    search_service = VectorSearch(
        db_service=db_service,
        embedding_service=embedder,
        table_name=table_name,
        vector_column=vector_column,
        content_columns=content_columns,
        id_column=id_column
    )

    results_summary: List[Dict[str, Any]] = []

    print(f"\nRunning experiments (k_retrieve={k_retrieve}, k_context={k_context})...")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Query: {case.query[:60]}...")

        # Perform vector search
        start_time = time.time()
        try:
            search_results: List[SearchResult] = search_service.search(
                case.query, top_k=k_retrieve
            )
        except Exception as exc:
            print(f"  ❌ Error during search: {exc}")
            continue

        latency_ms = (time.time() - start_time) * 1000.0
        print(f"  ⏱  Latency: {latency_ms:.1f} ms")

        # Extract retrieved IDs
        retrieved_ids = [res.id for res in search_results]
        print(f"  📊 Retrieved {len(retrieved_ids)} documents")

        # Compute retrieval metrics at k_retrieve
        precision = RetrievalEvaluator.precision_at_k(
            retrieved_ids, case.expected_doc_ids, k_retrieve
        )
        recall = RetrievalEvaluator.recall_at_k(
            retrieved_ids, case.expected_doc_ids, k_retrieve
        )
        f1 = RetrievalEvaluator.f1_at_k(
            retrieved_ids, case.expected_doc_ids, k_retrieve
        )
        ndcg = RetrievalEvaluator.ndcg_at_k(
            retrieved_ids, case.expected_doc_ids, k_retrieve
        )
        mrr = RetrievalEvaluator.mean_reciprocal_rank(
            retrieved_ids, case.expected_doc_ids
        )

        # Build context of top k_context documents
        context_docs = search_results[:k_context]
        context_tokens = estimate_context_tokens(context_docs)

        # Count relevant documents found
        relevant_found = len(set(retrieved_ids[:k_retrieve]) & set(case.expected_doc_ids))
        total_relevant = len(case.expected_doc_ids)

        print(f"  ✓ Found {relevant_found}/{total_relevant} relevant docs")
        print(f"  📈 Precision@{k_retrieve}: {precision:.3f} | Recall@{k_retrieve}: {recall:.3f} | nDCG@{k_retrieve}: {ndcg:.3f}")
        print(f"  🎯 Context size: {context_tokens} tokens (top {k_context} docs)")

        results_summary.append({
            "query": case.query,
            "query_metadata": case.metadata,
            "k_retrieve": k_retrieve,
            "k_context": k_context,
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4),
            "ndcg@k": round(ndcg, 4),
            "mrr": round(mrr, 4),
            "latency_ms": round(latency_ms, 1),
            "context_tokens": context_tokens,
            "relevant_found": relevant_found,
            "total_relevant": total_relevant,
            "total_retrieved": len(retrieved_ids)
        })

    # Close database connections
    db_service.close()

    print("\n" + "=" * 80)
    print(f"✓ Completed {len(results_summary)} experiments")

    return results_summary


def run_multi_k_experiment(
    test_cases: List[TestCase],
    db_url: str,
    table_name: str,
    vector_column: str,
    content_columns: List[str],
    k_retrieve_values: List[int],
    k_context_values: List[int],
    id_column: str = "id"
) -> Dict[str, Any]:
    """
    Run experiments across multiple k value combinations.

    Args:
        test_cases: Test cases to evaluate
        db_url: Database connection string
        table_name: Table name
        vector_column: Vector column name
        content_columns: Content columns
        k_retrieve_values: List of k_retrieve values to test
        k_context_values: List of k_context values to test
        id_column: ID column name

    Returns:
        Dictionary with all experiment results and summary statistics
    """
    all_results = []

    for k_ret in k_retrieve_values:
        for k_ctx in k_context_values:
            if k_ctx > k_ret:
                print(f"\n⚠️  Skipping k_context={k_ctx} > k_retrieve={k_ret}")
                continue

            print(f"\n{'='*80}")
            print(f"Experiment: k_retrieve={k_ret}, k_context={k_ctx}")
            print(f"{'='*80}")

            results = run_single_experiment(
                test_cases=test_cases,
                db_url=db_url,
                table_name=table_name,
                vector_column=vector_column,
                content_columns=content_columns,
                k_retrieve=k_ret,
                k_context=k_ctx,
                id_column=id_column
            )

            all_results.extend(results)

    # Compute summary statistics
    summary = compute_summary_statistics(all_results, k_retrieve_values, k_context_values)

    return {
        "metadata": {
            "table_name": table_name,
            "vector_column": vector_column,
            "num_test_cases": len(test_cases),
            "k_retrieve_values": k_retrieve_values,
            "k_context_values": k_context_values
        },
        "detailed_results": all_results,
        "summary": summary
    }


def compute_summary_statistics(
    results: List[Dict[str, Any]],
    k_retrieve_values: List[int],
    k_context_values: List[int]
) -> Dict[str, Any]:
    """
    Compute summary statistics across all experiments.

    Args:
        results: List of experiment results
        k_retrieve_values: List of k_retrieve values tested
        k_context_values: List of k_context values tested

    Returns:
        Dictionary with summary statistics
    """
    import numpy as np

    summary = {}

    # Group by k_retrieve and k_context combinations
    for k_ret in k_retrieve_values:
        for k_ctx in k_context_values:
            if k_ctx > k_ret:
                continue

            key = f"k_ret_{k_ret}_ctx_{k_ctx}"

            # Filter results for this combination
            filtered = [r for r in results
                       if r["k_retrieve"] == k_ret and r["k_context"] == k_ctx]

            if not filtered:
                continue

            # Compute means
            summary[key] = {
                "k_retrieve": k_ret,
                "k_context": k_ctx,
                "num_queries": len(filtered),
                "avg_precision@k": round(np.mean([r["precision@k"] for r in filtered]), 4),
                "avg_recall@k": round(np.mean([r["recall@k"] for r in filtered]), 4),
                "avg_f1@k": round(np.mean([r["f1@k"] for r in filtered]), 4),
                "avg_ndcg@k": round(np.mean([r["ndcg@k"] for r in filtered]), 4),
                "avg_mrr": round(np.mean([r["mrr"] for r in filtered]), 4),
                "avg_latency_ms": round(np.mean([r["latency_ms"] for r in filtered]), 1),
                "avg_context_tokens": round(np.mean([r["context_tokens"] for r in filtered]), 0),
                "std_precision@k": round(np.std([r["precision@k"] for r in filtered]), 4),
                "std_recall@k": round(np.std([r["recall@k"] for r in filtered]), 4),
                "std_ndcg@k": round(np.std([r["ndcg@k"] for r in filtered]), 4)
            }

    return summary


def print_summary_table(summary: Dict[str, Any]):
    """Print summary statistics in a formatted table."""
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'nDCG':<12} {'MRR':<12} {'Latency':<12} {'Ctx Tokens':<12}")
    print("-" * 100)

    # Sort by k_retrieve, then k_context
    sorted_keys = sorted(summary.keys(),
                        key=lambda k: (summary[k]['k_retrieve'], summary[k]['k_context']))

    for key in sorted_keys:
        s = summary[key]
        config = f"k_r={s['k_retrieve']}, k_c={s['k_context']}"
        print(f"{config:<20} "
              f"{s['avg_precision@k']:<12.3f} "
              f"{s['avg_recall@k']:<12.3f} "
              f"{s['avg_f1@k']:<12.3f} "
              f"{s['avg_ndcg@k']:<12.3f} "
              f"{s['avg_mrr']:<12.3f} "
              f"{s['avg_latency_ms']:<12.1f} "
              f"{s['avg_context_tokens']:<12.0f}")

    print("=" * 100)


def print_interpretation_guide():
    """Print guide for interpreting results."""
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Key Insights:
------------
1. **Increasing k_retrieve**:
   - ✓ Improves Recall (finds more relevant documents)
   - ✓ Improves Evidence Sufficiency (less likely to miss important info)
   - ✗ May decrease Precision (more marginal documents included)
   - ✗ Increases latency (more computation)

2. **Decreasing k_context**:
   - ✓ Reduces LLM cost (fewer tokens in prompt)
   - ✓ May improve answer quality (less noise)
   - ✓ Faster LLM response
   - ✗ May miss relevant context if filtering is poor

3. **Optimal Strategy**:
   - Use high k_retrieve (100-200) to build large candidate pool
   - Use moderate k_context (5-10) after re-ranking
   - Consider implementing re-ranker (cross-encoder) between retrieve and context

4. **Metric Guidance**:
   - Low Recall → Increase k_retrieve
   - Good Recall but Low Precision → Add re-ranker, decrease k_context
   - High Latency → Optimize indexes, consider smaller k_context
   - Low nDCG → Improve ranking (use hybrid/adaptive search)

Next Steps:
----------
- Experiment with HybridSearch (dense + sparse vectors)
- Add cross-encoder re-ranking between k_retrieve and k_context
- Test different embedding models (e.g., text-embedding-3-large)
- Measure end-to-end answer quality with LLM evaluation
""")
    print("=" * 80)


def main() -> None:
    """Main function to run k-balance experiment."""
    parser = argparse.ArgumentParser(
        description="Experiment with k_retrieve and k_context parameters in RAG retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Single k configuration
  python k_balance_experiment.py --test-file test_cases.json --k-retrieve 100 --k-context 8

  # Multiple k values for comparison
  python k_balance_experiment.py --test-file test_cases.json \\
      --k-retrieve-values 50 100 200 --k-context-values 5 8 10

  # Save results to JSON
  python k_balance_experiment.py --test-file test_cases.json \\
      --k-retrieve 200 --k-context 8 --output results.json
        """
    )

    # Required arguments
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to JSON file with test cases (query + expected_doc_ids)"
    )

    # Database configuration
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string (default: DATABASE_URL env var)"
    )
    parser.add_argument(
        "--table",
        default="articles",
        help="Table name to search (default: articles)"
    )
    parser.add_argument(
        "--vector-column",
        default=os.getenv("CONTENT_VECTOR_COLUMN", "content_vector_3072"),
        help="Vector column name (default: content_vector_3072)"
    )
    parser.add_argument(
        "--content-columns",
        nargs="+",
        default=["title", "content"],
        help="Content columns to return (default: title content)"
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="ID column name (default: id)"
    )

    # K parameter configuration
    parser.add_argument(
        "--k-retrieve",
        type=int,
        help="Number of documents to retrieve (single value)"
    )
    parser.add_argument(
        "--k-context",
        type=int,
        help="Number of documents for context (single value)"
    )
    parser.add_argument(
        "--k-retrieve-values",
        type=int,
        nargs="+",
        help="Multiple k_retrieve values to test"
    )
    parser.add_argument(
        "--k-context-values",
        type=int,
        nargs="+",
        help="Multiple k_context values to test"
    )

    # Output configuration
    parser.add_argument(
        "--output",
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Validation
    if not args.db_url:
        parser.error("Database URL must be provided via --db-url or DATABASE_URL environment variable")

    # Determine k values to test
    if args.k_retrieve and args.k_context:
        # Single configuration
        k_retrieve_values = [args.k_retrieve]
        k_context_values = [args.k_context]
    elif args.k_retrieve_values and args.k_context_values:
        # Multiple configurations
        k_retrieve_values = args.k_retrieve_values
        k_context_values = args.k_context_values
    else:
        parser.error("Either (--k-retrieve and --k-context) or (--k-retrieve-values and --k-context-values) must be provided")

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 80)
        print("RAG k-Balance Experiment")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Database: {args.db_url}")
        print(f"  Table: {args.table}")
        print(f"  Vector Column: {args.vector_column}")
        print(f"  Test File: {args.test_file}")
        print(f"  k_retrieve values: {k_retrieve_values}")
        print(f"  k_context values: {k_context_values}")
        print()

    # Load test cases
    try:
        test_cases = load_test_cases(args.test_file)
        print(f"✓ Loaded {len(test_cases)} test cases")
    except Exception as e:
        print(f"❌ Error loading test cases: {e}")
        sys.exit(1)

    # Run experiments
    try:
        if len(k_retrieve_values) == 1 and len(k_context_values) == 1:
            # Single experiment
            results = run_single_experiment(
                test_cases=test_cases,
                db_url=args.db_url,
                table_name=args.table,
                vector_column=args.vector_column,
                content_columns=args.content_columns,
                k_retrieve=k_retrieve_values[0],
                k_context=k_context_values[0],
                id_column=args.id_column
            )

            experiment_data = {
                "metadata": {
                    "table_name": args.table,
                    "vector_column": args.vector_column,
                    "k_retrieve": k_retrieve_values[0],
                    "k_context": k_context_values[0]
                },
                "results": results
            }
        else:
            # Multiple experiments
            experiment_data = run_multi_k_experiment(
                test_cases=test_cases,
                db_url=args.db_url,
                table_name=args.table,
                vector_column=args.vector_column,
                content_columns=args.content_columns,
                k_retrieve_values=k_retrieve_values,
                k_context_values=k_context_values,
                id_column=args.id_column
            )

            # Print summary table
            if not args.quiet and "summary" in experiment_data:
                print_summary_table(experiment_data["summary"])

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")

        # Print interpretation guide
        if not args.quiet:
            print_interpretation_guide()

        print("\n✓ Experiment completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
