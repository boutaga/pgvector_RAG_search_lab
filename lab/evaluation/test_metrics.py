#!/usr/bin/env python3
"""
Unit tests for enhanced nDCG metrics.

Tests the implementation of DCG, nDCG, and related functions
against known values and edge cases.
"""

import pytest
import math
import numpy as np
from lab.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    ndcg_at_k_binary,
    ndcg_at_k_with_grades,
    validate_relevance_grades,
    convert_binary_to_graded,
    calculate_idcg_at_k,
    mean_ndcg,
    median_ndcg,
    ndcg_std,
    ndcg_percentile,
    compare_ndcg_distributions
)


class TestDCGCalculation:
    """Tests for DCG formula"""

    def test_dcg_basic(self):
        """Test basic DCG calculation with manual verification"""
        relevances = [3, 2, 1, 0]
        k = 4

        # Manual calculation
        expected = (
            (2**3 - 1) / math.log2(2) +    # 7/1.0 = 7.0
            (2**2 - 1) / math.log2(3) +    # 3/1.585 = 1.893
            (2**1 - 1) / math.log2(4) +    # 1/2.0 = 0.5
            (2**0 - 1) / math.log2(5)      # 0/2.322 = 0.0
        )

        result = dcg_at_k(relevances, k)
        assert result == pytest.approx(expected, rel=1e-5)
        assert result == pytest.approx(9.393, rel=0.01)

    def test_dcg_binary_relevance(self):
        """Test DCG with binary relevance (0 or 1)"""
        relevances = [1, 1, 0, 1]
        k = 4

        # Manual calculation for binary
        expected = (
            (2**1 - 1) / math.log2(2) +  # 1.0
            (2**1 - 1) / math.log2(3) +  # 0.631
            (2**0 - 1) / math.log2(4) +  # 0.0
            (2**1 - 1) / math.log2(5)    # 0.431
        )

        result = dcg_at_k(relevances, k)
        assert result == pytest.approx(expected, rel=1e-5)
        assert result == pytest.approx(2.062, rel=0.01)

    def test_dcg_empty_list(self):
        """Test DCG with empty relevance list"""
        assert dcg_at_k([], 5) == 0.0

    def test_dcg_k_zero(self):
        """Test DCG with k=0"""
        assert dcg_at_k([2, 1, 1], 0) == 0.0

    def test_dcg_k_larger_than_list(self):
        """Test DCG when k > len(relevances)"""
        relevances = [2, 1]
        result = dcg_at_k(relevances, 10)
        expected = dcg_at_k(relevances, 2)
        assert result == pytest.approx(expected)

    def test_dcg_negative_k(self):
        """Test that negative k raises ValueError"""
        with pytest.raises(ValueError, match="k must be non-negative"):
            dcg_at_k([1, 2, 3], -1)

    def test_dcg_all_zeros(self):
        """Test DCG with all irrelevant documents"""
        relevances = [0, 0, 0, 0]
        assert dcg_at_k(relevances, 4) == 0.0

    def test_dcg_single_result(self):
        """Test DCG with k=1"""
        relevances = [2]
        expected = (2**2 - 1) / math.log2(2)  # 3.0
        assert dcg_at_k(relevances, 1) == pytest.approx(expected)


class TestNDCGCalculation:
    """Tests for nDCG calculation"""

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give nDCG = 1.0"""
        # Already in ideal order (descending relevance)
        relevances = [2, 2, 1, 1, 0]
        assert ndcg_at_k(relevances, 5) == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self):
        """Worst possible ranking (ascending relevance)"""
        relevances = [0, 0, 1, 1, 2]
        k = 5

        # Should be significantly less than 1.0
        score = ndcg_at_k(relevances, k)
        assert 0 < score < 0.5
        assert score == pytest.approx(0.487, rel=0.01)

    def test_ndcg_all_irrelevant(self):
        """All irrelevant documents should give nDCG = 0"""
        relevances = [0, 0, 0, 0]
        score = ndcg_at_k(relevances, 4)
        assert score == 0.0

    def test_ndcg_mixed_ranking(self):
        """Test with mixed relevance rankings"""
        # Suboptimal order: should have [2, 1, 1, 0]
        relevances = [1, 0, 2, 1]
        k = 4

        score = ndcg_at_k(relevances, k)
        assert 0.85 < score < 0.95  # Good but not perfect
        assert score == pytest.approx(0.897, rel=0.01)

    def test_ndcg_empty_list(self):
        """Empty list should give nDCG = 0"""
        assert ndcg_at_k([], 5) == 0.0

    def test_ndcg_k_zero(self):
        """k=0 should give nDCG = 0"""
        assert ndcg_at_k([1, 2, 3], 0) == 0.0

    def test_ndcg_single_relevant(self):
        """Single relevant doc at top should give nDCG = 1.0"""
        relevances = [1]
        assert ndcg_at_k(relevances, 1) == pytest.approx(1.0)

    def test_ndcg_single_relevant_not_at_top(self):
        """Single relevant doc not at top"""
        relevances = [0, 1]
        score = ndcg_at_k(relevances, 2)
        assert 0 < score < 1.0

    def test_ndcg_negative_k(self):
        """Test that negative k raises ValueError"""
        with pytest.raises(ValueError, match="k must be non-negative"):
            ndcg_at_k([1, 2, 3], -1)


class TestBackwardCompatibility:
    """Tests for binary relevance adapters"""

    def test_binary_adapter_perfect(self):
        """Test binary adapter with perfect ranking"""
        retrieved = [1, 5, 3, 7, 9]
        relevant = [1, 5]
        k = 5

        score = ndcg_at_k_binary(retrieved, relevant, k)
        assert score == pytest.approx(1.0)

    def test_binary_adapter_suboptimal(self):
        """Test binary adapter with suboptimal ranking"""
        retrieved = [1, 3, 5, 7, 9]
        relevant = [1, 5]
        k = 5

        score = ndcg_at_k_binary(retrieved, relevant, k)
        assert 0 < score <= 1.0

    def test_binary_adapter_no_relevant(self):
        """Test binary adapter with no relevant documents"""
        retrieved = [1, 2, 3, 4]
        relevant = [10, 20]
        k = 4

        score = ndcg_at_k_binary(retrieved, relevant, k)
        assert score == 0.0

    def test_binary_adapter_all_relevant(self):
        """Test binary adapter with all documents relevant"""
        retrieved = [1, 2, 3]
        relevant = [1, 2, 3, 4, 5]
        k = 3

        score = ndcg_at_k_binary(retrieved, relevant, k)
        assert score == pytest.approx(1.0)

    def test_graded_adapter_perfect(self):
        """Test graded adapter with perfect ranking"""
        retrieved = [101, 102, 103]
        grades = {101: 2, 102: 1, 103: 0}
        k = 3

        score = ndcg_at_k_with_grades(retrieved, grades, k)
        assert score == pytest.approx(1.0)

    def test_graded_adapter_suboptimal(self):
        """Test graded adapter with suboptimal ranking"""
        retrieved = [101, 102, 103]
        grades = {101: 0, 102: 1, 103: 2}  # Worst to best
        k = 3

        score = ndcg_at_k_with_grades(retrieved, grades, k)
        assert 0 < score < 0.5

    def test_graded_adapter_missing_grades(self):
        """Test graded adapter with missing grades (treated as 0)"""
        retrieved = [101, 102, 103, 104]
        grades = {101: 2, 103: 1}  # 102 and 104 missing
        k = 4

        score = ndcg_at_k_with_grades(retrieved, grades, k)
        assert 0 < score <= 1.0


class TestMultiLevelRelevance:
    """Tests specific to multi-level relevance"""

    def test_graded_vs_binary_discrimination(self):
        """Multi-level should distinguish quality better than binary"""
        # Graded: highly relevant doc first
        graded = [2, 1, 0]
        # Binary: all relevant docs equal
        binary = [1, 1, 0]

        ndcg_graded = ndcg_at_k(graded, 3)
        ndcg_binary = ndcg_at_k(binary, 3)

        # Graded should show perfect ranking
        assert ndcg_graded == pytest.approx(1.0)
        # Binary should also be perfect (all relevant at top)
        assert ndcg_binary == pytest.approx(1.0)

    def test_ordering_impact_highly_relevant(self):
        """Test that ordering highly relevant docs matters more"""
        # Best doc first
        scenario_a = [2, 1, 1, 0]
        # Best doc last
        scenario_b = [1, 1, 0, 2]

        ndcg_a = ndcg_at_k(scenario_a, 4)
        ndcg_b = ndcg_at_k(scenario_b, 4)

        # A should score significantly higher
        assert ndcg_a > ndcg_b + 0.3
        assert ndcg_a == pytest.approx(1.0)  # Perfect
        assert ndcg_b < 0.7  # Poor

    def test_exponential_weighting_effect(self):
        """Test that exponential weighting emphasizes high relevance"""
        # One highly relevant doc at top
        high_at_top = [2, 0, 0]
        # Two moderately relevant docs at top
        medium_at_top = [1, 1, 0]

        dcg_high = dcg_at_k(high_at_top, 3)
        dcg_medium = dcg_at_k(medium_at_top, 3)

        # 2^2 - 1 = 3, so highly relevant is worth 3x as much
        # But position matters, so not exactly 3:2 ratio
        assert dcg_high > dcg_medium


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_result_k_1(self):
        """Test with single result"""
        relevances = [2]
        assert ndcg_at_k(relevances, 1) == pytest.approx(1.0)

    def test_large_k(self):
        """Test with k larger than result set"""
        relevances = [2, 1, 0]
        result = ndcg_at_k(relevances, 100)
        expected = ndcg_at_k(relevances, 3)
        assert result == pytest.approx(expected)

    def test_float_relevances(self):
        """Test with float relevance values"""
        relevances = [2.0, 1.5, 1.0, 0.5]
        k = 4
        # Should work with floats
        result = ndcg_at_k(relevances, k)
        assert 0 <= result <= 1.0


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_validate_grades_valid(self):
        """Test validation with valid grades"""
        assert validate_relevance_grades([0, 1, 2, 1, 0]) is True

    def test_validate_grades_invalid_high(self):
        """Test validation with grade too high"""
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_relevance_grades([0, 1, 3, 1])

    def test_validate_grades_invalid_low(self):
        """Test validation with grade too low"""
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_relevance_grades([0, -1, 1, 2])

    def test_validate_grades_custom_range(self):
        """Test validation with custom range"""
        assert validate_relevance_grades([0, 1, 2, 3], min_grade=0, max_grade=3) is True

    def test_convert_binary_to_graded(self):
        """Test binary to graded conversion"""
        relevant = [1, 3]
        all_ids = [1, 2, 3, 4]

        result = convert_binary_to_graded(relevant, all_ids)

        expected = {1: 1, 2: 0, 3: 1, 4: 0}
        assert result == expected

    def test_calculate_idcg(self):
        """Test IDCG calculation"""
        relevances = [0, 1, 2, 1]
        k = 4

        idcg = calculate_idcg_at_k(relevances, k)
        # Should be DCG of [2, 1, 1, 0]
        expected = dcg_at_k([2, 1, 1, 0], k)
        assert idcg == pytest.approx(expected)


class TestStatisticalFunctions:
    """Tests for statistical functions"""

    def test_mean_ndcg(self):
        """Test mean nDCG calculation"""
        scores = [0.9, 0.8, 0.95, 0.85]
        result = mean_ndcg(scores)
        assert result == pytest.approx(0.875)

    def test_mean_ndcg_empty(self):
        """Test mean with empty list"""
        assert mean_ndcg([]) == 0.0

    def test_median_ndcg(self):
        """Test median nDCG calculation"""
        scores = [0.9, 0.8, 0.95, 0.85]
        result = median_ndcg(scores)
        assert result == pytest.approx(0.875)

    def test_ndcg_std(self):
        """Test standard deviation calculation"""
        scores = [0.9, 0.8, 0.95, 0.85]
        result = ndcg_std(scores)
        assert result > 0
        assert result == pytest.approx(0.0629, rel=0.01)

    def test_ndcg_percentile(self):
        """Test percentile calculation"""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]

        p50 = ndcg_percentile(scores, 50)
        assert p50 == pytest.approx(0.7)  # Median

        p95 = ndcg_percentile(scores, 95)
        assert p95 == pytest.approx(0.89, rel=0.01)

    def test_compare_distributions(self):
        """Test distribution comparison"""
        scores_a = [0.9, 0.85, 0.88, 0.92]
        scores_b = [0.75, 0.78, 0.80, 0.77]

        result = compare_ndcg_distributions(scores_a, scores_b, "Hybrid", "Vector")

        assert result['method_a'] == "Hybrid"
        assert result['method_b'] == "Vector"
        assert result['mean_a'] > result['mean_b']
        assert result['improvement'] > 0
        assert result['improvement_percent'] > 0

        # Should have t-test results
        assert 't_statistic' in result
        assert 'p_value' in result


class TestRealWorldScenarios:
    """Tests simulating real-world scenarios"""

    def test_factual_query_scenario(self):
        """Simulate factual query with one perfect answer"""
        # Query: "When was Python created?"
        # Retrieved: perfect match, good match, unrelated, unrelated
        relevances = [2, 1, 0, 0]
        k = 4

        score = ndcg_at_k(relevances, k)
        # Good score because perfect match is at top
        assert score > 0.9

    def test_conceptual_query_scenario(self):
        """Simulate conceptual query with multiple good answers"""
        # Query: "Explain machine learning"
        # Retrieved: comprehensive, brief, related, unrelated
        relevances = [2, 1, 1, 0]
        k = 4

        score = ndcg_at_k(relevances, k)
        # Perfect ranking
        assert score == pytest.approx(1.0)

    def test_poor_retrieval_scenario(self):
        """Simulate poor retrieval with relevant docs buried"""
        # Relevant docs at positions 8, 9, 10
        relevances = [0, 0, 0, 0, 0, 0, 0, 1, 2, 1]
        k = 10

        score = ndcg_at_k(relevances, k)
        # Should be low score
        assert score < 0.5

    def test_comparison_scenario(self):
        """Simulate comparing two search methods"""
        # Method A (hybrid): better ranking
        results_a = [
            [2, 1, 1, 0, 0],  # Query 1
            [2, 1, 0, 1, 0],  # Query 2
            [1, 2, 1, 0, 0],  # Query 3
        ]

        # Method B (vector only): worse ranking
        results_b = [
            [1, 2, 0, 1, 0],  # Query 1
            [1, 0, 2, 0, 1],  # Query 2
            [0, 1, 2, 1, 0],  # Query 3
        ]

        k = 5

        scores_a = [ndcg_at_k(r, k) for r in results_a]
        scores_b = [ndcg_at_k(r, k) for r in results_b]

        # Method A should have higher average nDCG
        assert mean_ndcg(scores_a) > mean_ndcg(scores_b)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
