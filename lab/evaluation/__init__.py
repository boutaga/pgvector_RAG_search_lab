#!/usr/bin/env python3
"""
RAG Evaluation Framework

This module provides comprehensive evaluation tools for RAG systems including:
- Retrieval quality metrics (Precision, Recall, F1, MRR, nDCG)
- Enhanced nDCG with multi-level relevance (0-2 scale)
- Answer quality assessment
- Performance benchmarking
- A/B testing framework
- Streamlit UI for evaluation management
"""

# Import enhanced nDCG metrics
from lab.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    ndcg_at_k_binary,
    ndcg_at_k_with_grades,
    mean_ndcg,
    median_ndcg,
    ndcg_std,
    compare_ndcg_distributions,
)

# Import relevance management
from lab.evaluation.relevance_manager import (
    RelevanceManager,
    TestQuery,
    RelevanceGrade,
)

# Import existing evaluation framework
from lab.evaluation.evaluator import (
    RAGEvaluator,
    RetrievalEvaluator,
    AnswerEvaluator,
    EvaluationMetrics,
    TestCase,
)

from lab.evaluation.benchmark import (
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkSuite,
)

__all__ = [
    # Enhanced nDCG metrics
    'dcg_at_k',
    'ndcg_at_k',
    'ndcg_at_k_binary',
    'ndcg_at_k_with_grades',
    'mean_ndcg',
    'median_ndcg',
    'ndcg_std',
    'compare_ndcg_distributions',

    # Relevance management
    'RelevanceManager',
    'TestQuery',
    'RelevanceGrade',

    # Existing evaluation framework
    'RAGEvaluator',
    'RetrievalEvaluator',
    'AnswerEvaluator',
    'EvaluationMetrics',
    'TestCase',

    # Benchmarking
    'PerformanceBenchmark',
    'BenchmarkResult',
    'BenchmarkSuite',
]
