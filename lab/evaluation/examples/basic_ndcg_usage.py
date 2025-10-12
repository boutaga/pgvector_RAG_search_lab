#!/usr/bin/env python3
"""
Example: Basic nDCG Usage

Demonstrates how to use the enhanced nDCG metrics for evaluation.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lab.evaluation.metrics import dcg_at_k, ndcg_at_k, ndcg_at_k_binary


def example_1_basic_calculation():
    """Example 1: Basic nDCG calculation with multi-level relevance"""
    print("=" * 60)
    print("Example 1: Basic nDCG Calculation")
    print("=" * 60)

    # Relevance grades for retrieved documents (in rank order)
    # 2 = highly relevant, 1 = relevant, 0 = not relevant
    relevances = [2, 1, 0, 1, 2]
    k = 5

    dcg = dcg_at_k(relevances, k)
    ndcg = ndcg_at_k(relevances, k)

    print(f"\nRelevances (in retrieval order): {relevances}")
    print(f"Ideal order would be: {sorted(relevances, reverse=True)}")
    print(f"\nDCG@{k}: {dcg:.3f}")
    print(f"nDCG@{k}: {ndcg:.3f}")
    print(f"\nInterpretation: nDCG of {ndcg:.3f} means retrieval quality is ")
    print(f"{ndcg*100:.1f}% of ideal (1.0 = perfect ranking)")
    print()


def example_2_perfect_vs_imperfect():
    """Example 2: Compare perfect vs imperfect ranking"""
    print("=" * 60)
    print("Example 2: Perfect vs Imperfect Ranking")
    print("=" * 60)

    # Perfect ranking (best docs first)
    perfect = [2, 2, 1, 1, 0]
    # Imperfect ranking (mixed)
    imperfect = [1, 0, 2, 1, 2]

    k = 5

    ndcg_perfect = ndcg_at_k(perfect, k)
    ndcg_imperfect = ndcg_at_k(imperfect, k)

    print(f"\nPerfect ranking: {perfect}")
    print(f"  nDCG@{k}: {ndcg_perfect:.3f}")
    print()
    print(f"Imperfect ranking: {imperfect}")
    print(f"  nDCG@{k}: {ndcg_imperfect:.3f}")
    print()
    print(f"Quality loss: {(1 - ndcg_imperfect/ndcg_perfect)*100:.1f}%")
    print(f"This means imperfect ranking is {ndcg_imperfect/ndcg_perfect*100:.1f}% as good as perfect")
    print()


def example_3_binary_compatibility():
    """Example 3: Binary relevance (backward compatibility)"""
    print("=" * 60)
    print("Example 3: Binary Relevance Mode")
    print("=" * 60)

    # Retrieved document IDs
    retrieved_ids = [101, 305, 202, 404, 501]
    # Relevant document IDs (any order)
    relevant_ids = [101, 202, 203]

    k = 5

    ndcg = ndcg_at_k_binary(retrieved_ids, relevant_ids, k)

    print(f"\nRetrieved document IDs: {retrieved_ids}")
    print(f"Relevant document IDs:  {relevant_ids}")
    print(f"\nMatches found at positions:")
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            print(f"  Position {i}: Doc {doc_id} ✓")

    print(f"\nnDCG@{k}: {ndcg:.3f}")
    print(f"  (Found {len([d for d in retrieved_ids if d in relevant_ids])} out of {len(relevant_ids)} relevant docs in top {k})")
    print()


def example_4_exponential_weighting():
    """Example 4: Demonstrate exponential weighting effect"""
    print("=" * 60)
    print("Example 4: Exponential Weighting Effect")
    print("=" * 60)

    # Scenario A: One highly relevant doc at top
    scenario_a = [2, 0, 0, 0, 0]
    # Scenario B: Two moderately relevant docs at top
    scenario_b = [1, 1, 0, 0, 0]

    k = 5

    dcg_a = dcg_at_k(scenario_a, k)
    dcg_b = dcg_at_k(scenario_b, k)

    print(f"\nScenario A (one highly relevant doc): {scenario_a}")
    print(f"  DCG@{k}: {dcg_a:.3f}")
    print(f"  Contribution from grade 2: (2^2 - 1) = 3")
    print()
    print(f"Scenario B (two moderately relevant docs): {scenario_b}")
    print(f"  DCG@{k}: {dcg_b:.3f}")
    print(f"  Contribution from each grade 1: (2^1 - 1) = 1")
    print()
    print(f"Ratio: {dcg_a / dcg_b:.2f}x")
    print(f"This shows that highly relevant docs are valued ~{dcg_a / dcg_b:.1f}x more than ")
    print(f"moderately relevant docs (after accounting for position)")
    print()


def example_5_real_world_scenario():
    """Example 5: Real-world scenario - comparing search methods"""
    print("=" * 60)
    print("Example 5: Real-World Scenario - Search Method Comparison")
    print("=" * 60)

    # Simulate results from different search methods
    # Query: "What is machine learning?"

    # Method A (Hybrid search): Good ranking
    hybrid_results = [2, 2, 1, 1, 0]

    # Method B (Vector only): Decent ranking
    vector_results = [2, 1, 0, 1, 2]

    # Method C (Keyword only): Poor ranking
    keyword_results = [0, 1, 0, 2, 1]

    k = 5

    ndcg_hybrid = ndcg_at_k(hybrid_results, k)
    ndcg_vector = ndcg_at_k(vector_results, k)
    ndcg_keyword = ndcg_at_k(keyword_results, k)

    print(f"\nQuery: 'What is machine learning?'")
    print(f"\nMethod A (Hybrid Search):")
    print(f"  Results: {hybrid_results}")
    print(f"  nDCG@{k}: {ndcg_hybrid:.3f}")
    print()
    print(f"Method B (Vector Only):")
    print(f"  Results: {vector_results}")
    print(f"  nDCG@{k}: {ndcg_vector:.3f}")
    print()
    print(f"Method C (Keyword Only):")
    print(f"  Results: {keyword_results}")
    print(f"  nDCG@{k}: {ndcg_keyword:.3f}")
    print()
    print(f"Comparison:")
    print(f"  Hybrid is {(ndcg_hybrid - ndcg_vector) / ndcg_vector * 100:+.1f}% better than Vector")
    print(f"  Hybrid is {(ndcg_hybrid - ndcg_keyword) / ndcg_keyword * 100:+.1f}% better than Keyword")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Enhanced nDCG Metrics - Usage Examples")
    print("=" * 60 + "\n")

    example_1_basic_calculation()
    example_2_perfect_vs_imperfect()
    example_3_binary_compatibility()
    example_4_exponential_weighting()
    example_5_real_world_scenario()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Try modifying the relevance grades to see how nDCG changes")
    print("2. Test with your own search results")
    print("3. Integrate with your evaluation pipeline")
    print()


if __name__ == "__main__":
    main()
