#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics Module

Implements advanced retrieval metrics including multi-level nDCG
following the approach described in:
https://www.dbi-services.com/blog/rag-series-adaptive-rag-understanding-confidence-precision-ndcg/

This module provides:
- Enhanced DCG/nDCG calculation with exponential weighting
- Multi-level relevance support (0-2 scale)
- Backward compatibility with binary relevance
- Statistical utilities for evaluation analysis
"""

import math
from typing import List, Dict, Union, Optional, Tuple
import numpy as np


# ============================================================================
# Core nDCG Functions (Blog Post Implementation)
# ============================================================================

def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    Uses the standard formula: DCG@k = Σ (2^rel[i] - 1) / log2(i + 2)
    This formula emphasizes highly relevant documents through exponential
    weighting (2^rel - 1).

    Args:
        relevances: List of relevance scores (0=irrelevant, 1=relevant, 2=highly relevant)
                   Can also support higher grades (e.g., 0-3 scale)
        k: Number of results to consider (top-k)

    Returns:
        DCG score (float, >= 0)

    Raises:
        ValueError: If k is negative

    Examples:
        >>> dcg_at_k([2, 1, 0, 1], 3)
        4.631  # (2^2-1)/log2(2) + (2^1-1)/log2(3) + (2^0-1)/log2(4)

        >>> dcg_at_k([1, 1, 1], 3)
        2.131  # All relevant but not highly relevant

        >>> dcg_at_k([0, 0, 0], 3)
        0.0    # No relevant documents
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    if not relevances or k == 0:
        return 0.0

    # Limit to top-k results
    relevances = relevances[:k]

    # Calculate DCG using standard formula
    dcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )

    return dcg


def ndcg_at_k(actual_relevances: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Normalizes DCG by the ideal DCG (IDCG) where documents are
    sorted by relevance in descending order. This provides a score
    between 0 and 1 indicating ranking quality.

    Args:
        actual_relevances: Relevance scores in retrieval order (as returned by search)
        k: Number of results to consider (top-k)

    Returns:
        NDCG score between 0 and 1 (float)
        - 1.0 = perfect ranking (all docs in ideal order)
        - 0.0 = worst ranking (no relevant docs or all irrelevant)

    Raises:
        ValueError: If k is negative

    Examples:
        >>> ndcg_at_k([2, 1, 1, 0], 4)
        1.0  # Perfect ranking (best docs first)

        >>> ndcg_at_k([0, 0, 1, 2], 4)
        0.487  # Poor ranking (best docs last)

        >>> ndcg_at_k([1, 0, 2, 1], 4)
        0.897  # Good but not perfect
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    if not actual_relevances or k == 0:
        return 0.0

    # Calculate actual DCG
    dcg = dcg_at_k(actual_relevances, k)

    # Calculate ideal DCG (sort relevances in descending order)
    ideal_relevances = sorted(actual_relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    # Normalize by IDCG
    if idcg == 0:
        # All documents are irrelevant
        return 0.0

    return dcg / idcg


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def ndcg_at_k_binary(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int
) -> float:
    """
    Calculate nDCG with binary relevance (backward compatible).

    Adapter for existing code that uses document IDs instead of
    relevance grades. Converts binary relevance (relevant/not) to
    grade format where relevant docs get grade 1.

    Args:
        retrieved_ids: IDs of retrieved documents in rank order
        relevant_ids: IDs of relevant documents (any order)
        k: Number of results to consider

    Returns:
        NDCG score between 0 and 1

    Examples:
        >>> ndcg_at_k_binary([1, 3, 5, 7], [1, 5], 4)
        1.0  # Both relevant docs at top

        >>> ndcg_at_k_binary([2, 4, 1, 5], [1, 5], 4)
        0.631  # Relevant docs at positions 3 and 4
    """
    # Convert to binary relevance scores
    relevances = [
        1.0 if doc_id in relevant_ids else 0.0
        for doc_id in retrieved_ids[:k]
    ]

    return ndcg_at_k(relevances, k)


def ndcg_at_k_with_grades(
    retrieved_ids: List[int],
    relevance_grades: Dict[int, int],
    k: int
) -> float:
    """
    Calculate nDCG with multi-level relevance grades.

    Converts document IDs and grade dictionary to relevance list
    for nDCG calculation. Documents not in grades dict are assumed
    to be irrelevant (grade 0).

    Args:
        retrieved_ids: IDs of retrieved documents in rank order
        relevance_grades: Dict mapping doc_id -> relevance grade (0-2)
                         Missing docs are assumed grade 0
        k: Number of results to consider

    Returns:
        NDCG score between 0 and 1

    Examples:
        >>> ndcg_at_k_with_grades(
        ...     [101, 102, 103],
        ...     {101: 2, 102: 1, 103: 0},
        ...     3
        ... )
        1.0  # Perfect ranking

        >>> ndcg_at_k_with_grades(
        ...     [101, 102, 103],
        ...     {101: 0, 102: 1, 103: 2},
        ...     3
        ... )
        0.487  # Worst ranking
    """
    # Convert to relevance scores
    relevances = [
        float(relevance_grades.get(doc_id, 0))
        for doc_id in retrieved_ids[:k]
    ]

    return ndcg_at_k(relevances, k)


# ============================================================================
# Utility Functions
# ============================================================================

def validate_relevance_grades(
    grades: List[float],
    min_grade: int = 0,
    max_grade: int = 2
) -> bool:
    """
    Validate that all relevance grades are within allowed range.

    Args:
        grades: List of relevance grades to validate
        min_grade: Minimum allowed grade (default: 0)
        max_grade: Maximum allowed grade (default: 2)

    Returns:
        True if all grades are valid

    Raises:
        ValueError: If any grade is outside the allowed range

    Examples:
        >>> validate_relevance_grades([0, 1, 2, 1])
        True

        >>> validate_relevance_grades([0, 1, 3, 1])
        Traceback (most recent call last):
        ...
        ValueError: Relevance grade 3 is outside allowed range [0, 2]
    """
    for grade in grades:
        if grade < min_grade or grade > max_grade:
            raise ValueError(
                f"Relevance grade {grade} is outside allowed range "
                f"[{min_grade}, {max_grade}]"
            )
    return True


def convert_binary_to_graded(
    relevant_ids: List[int],
    all_ids: List[int]
) -> Dict[int, int]:
    """
    Convert binary relevance (relevant/not) to graded format.

    Creates a dictionary mapping doc IDs to grades where relevant
    documents get grade 1 and others get grade 0.

    Args:
        relevant_ids: List of IDs for relevant documents
        all_ids: List of all document IDs to include

    Returns:
        Dictionary mapping doc_id -> grade (0 or 1)

    Examples:
        >>> convert_binary_to_graded([1, 3], [1, 2, 3, 4])
        {1: 1, 2: 0, 3: 1, 4: 0}
    """
    relevant_set = set(relevant_ids)
    return {
        doc_id: (1 if doc_id in relevant_set else 0)
        for doc_id in all_ids
    }


def calculate_idcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate the Ideal DCG at k.

    This is the DCG score for the ideal ranking (documents sorted
    by relevance in descending order). Used for normalizing DCG.

    Args:
        relevances: List of relevance scores
        k: Number of results to consider

    Returns:
        IDCG score (float, >= 0)

    Examples:
        >>> calculate_idcg_at_k([0, 1, 2, 1], 4)
        5.824  # Ideal: [2, 1, 1, 0]
    """
    ideal_relevances = sorted(relevances, reverse=True)
    return dcg_at_k(ideal_relevances, k)


# ============================================================================
# Statistical Functions
# ============================================================================

def mean_ndcg(ndcg_scores: List[float]) -> float:
    """
    Calculate mean nDCG across multiple queries.

    Args:
        ndcg_scores: List of nDCG scores from multiple queries

    Returns:
        Mean nDCG score (float)

    Examples:
        >>> mean_ndcg([0.9, 0.8, 0.95, 0.85])
        0.875
    """
    if not ndcg_scores:
        return 0.0

    return float(np.mean(ndcg_scores))


def median_ndcg(ndcg_scores: List[float]) -> float:
    """
    Calculate median nDCG across multiple queries.

    Args:
        ndcg_scores: List of nDCG scores from multiple queries

    Returns:
        Median nDCG score (float)

    Examples:
        >>> median_ndcg([0.9, 0.8, 0.95, 0.85])
        0.875
    """
    if not ndcg_scores:
        return 0.0

    return float(np.median(ndcg_scores))


def ndcg_std(ndcg_scores: List[float]) -> float:
    """
    Calculate standard deviation of nDCG scores.

    Args:
        ndcg_scores: List of nDCG scores from multiple queries

    Returns:
        Standard deviation of nDCG scores (float)

    Examples:
        >>> ndcg_std([0.9, 0.8, 0.95, 0.85])
        0.0629
    """
    if not ndcg_scores:
        return 0.0

    return float(np.std(ndcg_scores))


def ndcg_confidence_interval(
    ndcg_scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for nDCG scores.

    Uses normal approximation for the confidence interval.

    Args:
        ndcg_scores: List of nDCG scores from multiple queries
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Examples:
        >>> ndcg_confidence_interval([0.9, 0.8, 0.95, 0.85, 0.88])
        (0.815, 0.925)  # Approximate 95% CI
    """
    if not ndcg_scores or len(ndcg_scores) < 2:
        return (0.0, 0.0)

    n = len(ndcg_scores)
    mean = np.mean(ndcg_scores)
    std_err = np.std(ndcg_scores, ddof=1) / np.sqrt(n)

    # Z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)

    lower = mean - z_score * std_err
    upper = mean + z_score * std_err

    return (float(lower), float(upper))


def ndcg_percentile(ndcg_scores: List[float], percentile: float) -> float:
    """
    Calculate percentile of nDCG scores.

    Args:
        ndcg_scores: List of nDCG scores from multiple queries
        percentile: Percentile to calculate (0-100)

    Returns:
        nDCG score at the specified percentile

    Examples:
        >>> ndcg_percentile([0.5, 0.6, 0.7, 0.8, 0.9], 50)
        0.7  # Median (50th percentile)

        >>> ndcg_percentile([0.5, 0.6, 0.7, 0.8, 0.9], 95)
        0.89  # 95th percentile
    """
    if not ndcg_scores:
        return 0.0

    return float(np.percentile(ndcg_scores, percentile))


# ============================================================================
# Comparison and Analysis Functions
# ============================================================================

def compare_ndcg_distributions(
    scores_a: List[float],
    scores_b: List[float],
    method_a_name: str = "Method A",
    method_b_name: str = "Method B"
) -> Dict[str, Any]:
    """
    Compare nDCG score distributions between two methods.

    Provides statistical comparison including means, medians,
    and t-test results.

    Args:
        scores_a: nDCG scores for method A
        scores_b: nDCG scores for method B
        method_a_name: Name of method A for reporting
        method_b_name: Name of method B for reporting

    Returns:
        Dictionary with comparison statistics

    Examples:
        >>> compare_ndcg_distributions(
        ...     [0.9, 0.85, 0.88],
        ...     [0.75, 0.78, 0.80],
        ...     "Hybrid", "Vector"
        ... )
        {
            'method_a': 'Hybrid',
            'method_b': 'Vector',
            'mean_a': 0.877,
            'mean_b': 0.777,
            'improvement': 0.100,
            'improvement_percent': 12.9,
            ...
        }
    """
    from scipy import stats as scipy_stats

    result = {
        'method_a': method_a_name,
        'method_b': method_b_name,
        'mean_a': mean_ndcg(scores_a),
        'mean_b': mean_ndcg(scores_b),
        'median_a': median_ndcg(scores_a),
        'median_b': median_ndcg(scores_b),
        'std_a': ndcg_std(scores_a),
        'std_b': ndcg_std(scores_b),
        'n_a': len(scores_a),
        'n_b': len(scores_b)
    }

    # Calculate improvement
    if result['mean_b'] > 0:
        result['improvement'] = result['mean_a'] - result['mean_b']
        result['improvement_percent'] = (
            (result['mean_a'] - result['mean_b']) / result['mean_b'] * 100
        )
    else:
        result['improvement'] = 0.0
        result['improvement_percent'] = 0.0

    # Perform t-test if we have enough samples
    if len(scores_a) >= 2 and len(scores_b) >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(scores_a, scores_b)
        result['t_statistic'] = float(t_stat)
        result['p_value'] = float(p_value)
        result['significant'] = p_value < 0.05
    else:
        result['t_statistic'] = None
        result['p_value'] = None
        result['significant'] = None

    return result


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Core functions
    'dcg_at_k',
    'ndcg_at_k',
    'calculate_idcg_at_k',

    # Backward compatibility
    'ndcg_at_k_binary',
    'ndcg_at_k_with_grades',

    # Utilities
    'validate_relevance_grades',
    'convert_binary_to_graded',

    # Statistics
    'mean_ndcg',
    'median_ndcg',
    'ndcg_std',
    'ndcg_confidence_interval',
    'ndcg_percentile',

    # Comparison
    'compare_ndcg_distributions',
]
