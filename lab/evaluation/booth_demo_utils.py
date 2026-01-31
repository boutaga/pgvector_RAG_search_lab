#!/usr/bin/env python3
"""
Booth Demo Utility Functions
=============================

Helper functions for the RAG Booth Demo Streamlit application.
Provides visualization utilities, color schemes, and data transformation functions.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os


# ============================================================================
# Color Scheme Configuration
# ============================================================================

@dataclass
class ColorScheme:
    """Color scheme for the booth demo."""
    # Recall/Quality colors
    PERFECT = "#22c55e"       # Green - 100% recall
    GOOD = "#eab308"          # Yellow - >= 80%
    NEEDS_WORK = "#f97316"    # Orange - >= 60%
    PROBLEM = "#ef4444"       # Red - < 60%

    # Strategy colors for charts
    CONTENT_ONLY = "#3b82f6"      # Blue
    TITLE_ONLY = "#10b981"        # Emerald
    TITLE_WEIGHTED = "#8b5cf6"    # Violet
    HYBRID_DENSE = "#f59e0b"      # Amber
    HYBRID_BALANCED = "#6366f1"   # Indigo
    HYBRID_SPARSE = "#ec4899"     # Pink
    HYBRID_FTS = "#14b8a6"        # Teal

    # UI colors
    BACKGROUND_CARD = "#1e293b"   # Slate-800
    TEXT_PRIMARY = "#f8fafc"      # Slate-50
    TEXT_SECONDARY = "#94a3b8"    # Slate-400
    BORDER = "#334155"            # Slate-700
    ACCENT = "#3b82f6"            # Blue-500

    @staticmethod
    def get_recall_color(recall: float) -> str:
        """Get color based on recall value."""
        if recall >= 1.0:
            return ColorScheme.PERFECT
        elif recall >= 0.8:
            return ColorScheme.GOOD
        elif recall >= 0.6:
            return ColorScheme.NEEDS_WORK
        else:
            return ColorScheme.PROBLEM

    @staticmethod
    def get_ndcg_color(ndcg: float) -> str:
        """Get color based on nDCG value."""
        if ndcg >= 0.9:
            return ColorScheme.PERFECT
        elif ndcg >= 0.7:
            return ColorScheme.GOOD
        elif ndcg >= 0.5:
            return ColorScheme.NEEDS_WORK
        else:
            return ColorScheme.PROBLEM

    @staticmethod
    def get_status_emoji(recall: float) -> str:
        """Get status emoji based on recall value."""
        if recall >= 1.0:
            return "✅"
        elif recall >= 0.8:
            return "🟡"
        elif recall >= 0.6:
            return "🟠"
        else:
            return "🔴"


COLORS = ColorScheme()


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_demo_data(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pre-computed demo data from JSON file.

    Args:
        file_path: Path to demo data JSON file.
                   If None, uses default booth_demo_data.json

    Returns:
        Dictionary containing demo data
    """
    if file_path is None:
        # Default to booth_demo_data.json in same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "demo_data", "booth_demo_data.json")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_test_cases(demo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract test cases from demo data."""
    return demo_data.get("test_cases", [])


def get_k_balance_results(demo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract k-balance experiment results from demo data."""
    return demo_data.get("k_balance_results", {})


def get_strategy_results(demo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract strategy comparison results from demo data."""
    return demo_data.get("strategy_results", {})


def get_corpus_analysis(demo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract corpus analysis data from demo data."""
    return demo_data.get("corpus_analysis", {})


def get_talking_points(demo_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract talking points for presenter from demo data."""
    return demo_data.get("talking_points", {})


# ============================================================================
# Data Transformation Functions
# ============================================================================

def prepare_k_balance_heatmap_data(k_balance_results: Dict[str, Any]) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Prepare data for recall heatmap visualization.

    Returns:
        Tuple of (queries, k_values, recall_matrix)
    """
    by_query = k_balance_results.get("by_query", {})

    queries = list(by_query.keys())
    k_values = ["k_10", "k_20", "k_50", "k_100"]

    recall_matrix = []
    for query in queries:
        query_data = by_query[query]
        row = [query_data.get(k, {}).get("recall", 0) for k in k_values]
        recall_matrix.append(row)

    # Convert k values to display format
    k_display = ["k=10", "k=20", "k=50", "k=100"]

    return queries, k_display, recall_matrix


def prepare_strategy_comparison_data(strategy_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare data for strategy comparison chart.

    Returns:
        List of dictionaries with strategy metrics
    """
    strategies = strategy_results.get("strategies", [])
    by_query = strategy_results.get("by_query", {})

    result = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        strategy_name = strategy["name"]

        # Calculate average metrics across all queries
        recalls = []
        ndcgs = []
        precisions = []

        for query, query_data in by_query.items():
            if strategy_id in query_data:
                recalls.append(query_data[strategy_id].get("recall", 0))
                ndcgs.append(query_data[strategy_id].get("ndcg", 0))
                precisions.append(query_data[strategy_id].get("precision", 0))

        if recalls:
            result.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "avg_recall": sum(recalls) / len(recalls),
                "avg_ndcg": sum(ndcgs) / len(ndcgs),
                "avg_precision": sum(precisions) / len(precisions),
                "description": strategy.get("description", "")
            })

    return result


def prepare_query_deep_dive_data(query: str, strategy_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare per-query strategy performance data.

    Args:
        query: Query string to analyze
        strategy_results: Strategy results dictionary

    Returns:
        List of dictionaries with per-strategy metrics for the query
    """
    strategies = strategy_results.get("strategies", [])
    by_query = strategy_results.get("by_query", {})

    query_data = by_query.get(query, {})
    result = []

    for strategy in strategies:
        strategy_id = strategy["id"]
        if strategy_id in query_data:
            metrics = query_data[strategy_id]
            result.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy["name"],
                "recall": metrics.get("recall", 0),
                "ndcg": metrics.get("ndcg", 0),
                "precision": metrics.get("precision", 0),
                "mrr": metrics.get("mrr", 0),
                "relevant_found": metrics.get("relevant_found", 0)
            })

    return result


def get_first_100_recall_summary(k_balance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get summary of when each query first achieves 100% recall.

    Returns:
        List of dictionaries with query, k value, and status
    """
    first_100 = k_balance_results.get("first_100_recall", {})

    result = []
    for query, data in first_100.items():
        result.append({
            "query": query,
            "k": data.get("k"),
            "status": data.get("status"),
            "note": data.get("note", ""),
            "emoji": "✅" if data.get("status") != "late" else "🟡" if data.get("k", 100) <= 50 else "🔴"
        })

    return result


# ============================================================================
# Formatting Functions
# ============================================================================

def format_percent(value: float, decimals: int = 0) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_metric(value: float, decimals: int = 3) -> str:
    """Format a metric value."""
    return f"{value:.{decimals}f}"


def truncate_query(query: str, max_length: int = 40) -> str:
    """Truncate query string for display."""
    if len(query) <= max_length:
        return query
    return query[:max_length - 3] + "..."


def get_query_short_name(query: str) -> str:
    """Get a short identifier for a query."""
    # Use first few significant words
    words = query.lower().replace("?", "").split()
    significant = [w for w in words if w not in ("what", "is", "the", "a", "an", "who", "how")]
    return "_".join(significant[:2]) if significant else query[:15]


# ============================================================================
# Insight Generation Functions
# ============================================================================

def generate_query_insight(query: str, strategy_results: Dict[str, Any]) -> str:
    """
    Generate an insight message for a specific query based on strategy results.

    Returns:
        Insight string explaining why results are the way they are
    """
    by_query = strategy_results.get("by_query", {})
    query_data = by_query.get(query, {})

    if not query_data:
        return "No data available for this query."

    insight = query_data.get("insight", "")
    if insight:
        return insight

    # Generate insight based on metrics
    best_strategy = query_data.get("best_strategy", "")
    max_recall = query_data.get("max_recall", 0)

    if max_recall >= 1.0:
        return f"This query achieves 100% recall. Best ranking with {best_strategy}."
    elif max_recall >= 0.75:
        return f"Good recall ({format_percent(max_recall)}), but room for improvement."
    else:
        return f"Low recall ({format_percent(max_recall)}) - may indicate corpus coverage issue."


def generate_act_summary(act: int, demo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary data for a specific act of the demo story.

    Args:
        act: Act number (1, 2, or 3)
        demo_data: Full demo data dictionary

    Returns:
        Dictionary with summary data for the act
    """
    talking_points = demo_data.get("talking_points", {})

    if act == 1:
        points = talking_points.get("act_1_problem", [])
        return {
            "title": "The Problem",
            "subtitle": "Some queries just won't reach 100% recall...",
            "points": points,
            "icon": "🔴"
        }
    elif act == 2:
        points = talking_points.get("act_2_investigation", [])
        return {
            "title": "The Investigation",
            "subtitle": "Trying different search strategies...",
            "points": points,
            "icon": "🔶"
        }
    elif act == 3:
        points = talking_points.get("act_3_revelation", [])
        return {
            "title": "The Revelation",
            "subtitle": "The AHA moment: It's the DATA!",
            "points": points,
            "icon": "🟢"
        }
    else:
        points = talking_points.get("summary_takeaways", [])
        return {
            "title": "Key Takeaways",
            "subtitle": "What we learned",
            "points": points,
            "icon": "📊"
        }


# ============================================================================
# Chart Helper Functions
# ============================================================================

def get_strategy_color(strategy_id: str) -> str:
    """Get color for a specific strategy."""
    color_map = {
        "content_only": COLORS.CONTENT_ONLY,
        "title_only": COLORS.TITLE_ONLY,
        "title_weighted_70_30": COLORS.TITLE_WEIGHTED,
        "title_weighted_50_50": COLORS.TITLE_WEIGHTED,
        "title_weighted_30_70": COLORS.TITLE_WEIGHTED,
        "hybrid_dense_heavy": COLORS.HYBRID_DENSE,
        "hybrid_balanced": COLORS.HYBRID_BALANCED,
        "hybrid_sparse_heavy": COLORS.HYBRID_SPARSE,
        "hybrid_title_fts": COLORS.HYBRID_FTS
    }
    return color_map.get(strategy_id, "#888888")


def get_all_strategy_colors() -> List[str]:
    """Get list of colors for all strategies in order."""
    return [
        COLORS.CONTENT_ONLY,
        COLORS.TITLE_ONLY,
        COLORS.TITLE_WEIGHTED,
        COLORS.TITLE_WEIGHTED,
        COLORS.TITLE_WEIGHTED,
        COLORS.HYBRID_DENSE,
        COLORS.HYBRID_BALANCED,
        COLORS.HYBRID_SPARSE,
        COLORS.HYBRID_FTS
    ]


def create_metric_card_style(color: str) -> str:
    """Generate CSS style for a metric card."""
    return f"""
        background-color: {COLORS.BACKGROUND_CARD};
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    """


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Classes
    'ColorScheme',
    'COLORS',

    # Data loading
    'load_demo_data',
    'get_test_cases',
    'get_k_balance_results',
    'get_strategy_results',
    'get_corpus_analysis',
    'get_talking_points',

    # Data transformation
    'prepare_k_balance_heatmap_data',
    'prepare_strategy_comparison_data',
    'prepare_query_deep_dive_data',
    'get_first_100_recall_summary',

    # Formatting
    'format_percent',
    'format_metric',
    'truncate_query',
    'get_query_short_name',

    # Insights
    'generate_query_insight',
    'generate_act_summary',

    # Chart helpers
    'get_strategy_color',
    'get_all_strategy_colors',
    'create_metric_card_style',
]
