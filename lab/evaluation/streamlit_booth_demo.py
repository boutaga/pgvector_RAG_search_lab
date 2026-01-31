#!/usr/bin/env python3
"""
RAG Booth Demo - Visual Story-Driven Streamlit App
===================================================

A narrative-driven Streamlit application for PostgreSQL conference booth
demonstrations. Combines k-balance experiments + search strategy comparisons
into a visual story that explains RAG retrieval quality.

THE STORY:
    Act 1 - The Problem: "Some queries work great, but 'machine learning' is stuck at 50%"
    Act 2 - The Investigation: "Trying different strategies - title weighting helps some, not all"
    Act 3 - The Revelation: "There's NO 'Machine Learning' article! This is a DATA problem"
    Summary: "RAG quality starts with your corpus"

Usage:
    streamlit run lab/evaluation/streamlit_booth_demo.py

Features:
    - Tabbed interface for guided presentations
    - Pre-loaded demo data (no API calls needed)
    - Responsive design for different screen sizes
    - Export results as JSON/CSV
    - Presenter notes in sidebar
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lab.evaluation.booth_demo_utils import (
    load_demo_data,
    get_test_cases,
    get_k_balance_results,
    get_strategy_results,
    get_corpus_analysis,
    get_talking_points,
    prepare_k_balance_heatmap_data,
    prepare_strategy_comparison_data,
    prepare_query_deep_dive_data,
    get_first_100_recall_summary,
    format_percent,
    format_metric,
    truncate_query,
    generate_query_insight,
    generate_act_summary,
    get_strategy_color,
    get_all_strategy_colors,
    COLORS,
    ColorScheme
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Search Quality: The Complete Picture",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for booth-friendly styling."""
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }

    .metric-card.success {
        border-left-color: #22c55e;
    }

    .metric-card.warning {
        border-left-color: #eab308;
    }

    .metric-card.danger {
        border-left-color: #ef4444;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Callout boxes */
    .callout {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        border-left: 4px solid #3b82f6;
    }

    .callout.insight {
        border-left-color: #22c55e;
        background-color: #052e16;
    }

    .callout.warning {
        border-left-color: #eab308;
        background-color: #422006;
    }

    .callout.danger {
        border-left-color: #ef4444;
        background-color: #450a0a;
    }

    /* Query badges */
    .query-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px;
    }

    .query-badge.success {
        background-color: #052e16;
        color: #22c55e;
    }

    .query-badge.warning {
        background-color: #422006;
        color: #eab308;
    }

    .query-badge.danger {
        background-color: #450a0a;
        color: #ef4444;
    }

    /* SQL code blocks */
    .sql-block {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }

    /* Presenter notes */
    .presenter-note {
        background-color: #1e3a5f;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.85rem;
        border-left: 3px solid #3b82f6;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.8rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 0.9rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = None
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = None
    if 'show_presenter_notes' not in st.session_state:
        st.session_state.show_presenter_notes = False


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_cached_demo_data():
    """Load and cache demo data."""
    return load_demo_data()


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render the sidebar with controls and presenter notes."""
    with st.sidebar:
        st.title("🎯 Booth Demo")
        st.markdown("---")

        # Demo mode toggle
        demo_mode = st.toggle("Demo Mode (Pre-loaded Data)", value=True)

        if not demo_mode:
            st.warning("Live mode requires database connection and API keys.")
            st.text_input("DATABASE_URL", type="password", key="db_url")
            st.text_input("OPENAI_API_KEY", type="password", key="api_key")

        st.markdown("---")

        # Presenter notes toggle
        st.session_state.show_presenter_notes = st.toggle(
            "Show Presenter Notes",
            value=st.session_state.show_presenter_notes
        )

        if st.session_state.show_presenter_notes:
            st.markdown("### 📝 Presenter Notes")
            if st.session_state.demo_data:
                talking_points = get_talking_points(st.session_state.demo_data)
                with st.expander("Act 1: The Problem", expanded=False):
                    for point in talking_points.get("act_1_problem", []):
                        st.markdown(f"- {point}")

                with st.expander("Act 2: Investigation", expanded=False):
                    for point in talking_points.get("act_2_investigation", []):
                        st.markdown(f"- {point}")

                with st.expander("Act 3: Revelation", expanded=False):
                    for point in talking_points.get("act_3_revelation", []):
                        st.markdown(f"- {point}")

                with st.expander("Summary", expanded=False):
                    for point in talking_points.get("summary_takeaways", []):
                        st.markdown(f"- {point}")

        st.markdown("---")

        # Export options
        st.markdown("### 📤 Export")
        if st.session_state.demo_data:
            json_str = json.dumps(st.session_state.demo_data, indent=2)
            st.download_button(
                label="Download Demo Data (JSON)",
                data=json_str,
                file_name="booth_demo_results.json",
                mime="application/json"
            )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **RAG Booth Demo** v1.0

        A visual story explaining why some
        RAG queries fail and how to diagnose
        corpus coverage issues.

        [GitHub Repository](https://github.com/boutaga/pgvector_RAG_search_lab)
        """)

    return demo_mode


# ============================================================================
# Act 1: The Problem
# ============================================================================

def render_act1_problem():
    """Render Act 1: The Problem tab."""
    st.header("🔴 Act 1: The Problem")
    st.markdown("### Some queries just won't reach 100% recall...")

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    k_results = get_k_balance_results(st.session_state.demo_data)
    test_cases = get_test_cases(st.session_state.demo_data)
    first_100 = get_first_100_recall_summary(k_results)

    # Big Metric Cards - 4 queries
    st.markdown("### Query Performance at k=10")

    cols = st.columns(4)
    by_query = k_results.get("by_query", {})

    for idx, test_case in enumerate(test_cases):
        query = test_case["query"]
        query_data = by_query.get(query, {}).get("k_10", {})
        recall = query_data.get("recall", 0)
        ndcg = query_data.get("ndcg", 0)

        with cols[idx]:
            color_class = "success" if recall >= 1.0 else "warning" if recall >= 0.8 else "danger"
            emoji = "✅" if recall >= 1.0 else "🟡" if recall >= 0.8 else "🔴"

            st.markdown(f"""
            <div class="metric-card {color_class}">
                <div class="metric-value">{emoji} {format_percent(recall)}</div>
                <div class="metric-label">Recall</div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                    nDCG: {format_metric(ndcg)}
                </div>
                <div style="margin-top: 8px; font-size: 0.8rem; color: #64748b;">
                    "{truncate_query(query, 35)}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Recall Achievement Summary
    st.markdown("### When does each query reach 100% recall?")

    col1, col2 = st.columns([2, 3])

    with col1:
        summary_df = pd.DataFrame(first_100)
        summary_df = summary_df.rename(columns={
            "query": "Query",
            "k": "First 100% at",
            "status": "Status",
            "emoji": ""
        })

        # Style the dataframe
        st.dataframe(
            summary_df[["", "Query", "First 100% at", "Status"]],
            hide_index=True,
            use_container_width=True
        )

        # Count summary
        reached_100 = len([x for x in first_100 if x["k"] is not None])
        total = len(first_100)
        st.metric(
            "Queries Reaching 100%",
            f"{reached_100}/{total}",
            delta=None if reached_100 == total else f"{total - reached_100} stuck"
        )

    with col2:
        # Recall Heatmap
        queries, k_values, recall_matrix = prepare_k_balance_heatmap_data(k_results)

        fig = go.Figure(data=go.Heatmap(
            z=recall_matrix,
            x=k_values,
            y=[truncate_query(q, 30) for q in queries],
            colorscale=[
                [0, COLORS.PROBLEM],
                [0.5, COLORS.NEEDS_WORK],
                [0.8, COLORS.GOOD],
                [1.0, COLORS.PERFECT]
            ],
            text=[[f"{v:.0%}" for v in row] for row in recall_matrix],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="Query: %{y}<br>k: %{x}<br>Recall: %{z:.0%}<extra></extra>",
            zmin=0,
            zmax=1
        ))

        fig.update_layout(
            title="Recall Heatmap: Queries vs k_retrieve",
            xaxis_title="k_retrieve",
            yaxis_title="Query",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recall vs K Line Chart
    st.markdown("### Recall vs k_retrieve for Each Query")

    # Prepare data for line chart
    line_data = []
    for query, query_data in by_query.items():
        for k_key, metrics in query_data.items():
            if k_key.startswith("k_"):
                k_val = int(k_key.replace("k_", ""))
                line_data.append({
                    "Query": truncate_query(query, 30),
                    "k_retrieve": k_val,
                    "Recall": metrics.get("recall", 0),
                    "nDCG": metrics.get("ndcg", 0)
                })

    line_df = pd.DataFrame(line_data)

    fig = px.line(
        line_df,
        x="k_retrieve",
        y="Recall",
        color="Query",
        markers=True,
        title="How Recall Changes with k_retrieve"
    )

    fig.update_layout(
        xaxis_title="k_retrieve (candidate pool size)",
        yaxis_title="Recall",
        yaxis=dict(range=[0, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=400
    )

    # Add 100% recall reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                  annotation_text="100% Recall", annotation_position="right")

    st.plotly_chart(fig, use_container_width=True)

    # Callout box
    st.markdown("""
    <div class="callout warning">
        <strong>🤔 Why is "What is machine learning?" stuck at 50%?</strong><br>
        Even at k=100, we only find 2 of 4 expected documents. Let's investigate...
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Act 2: The Investigation
# ============================================================================

def render_act2_investigation():
    """Render Act 2: The Investigation tab."""
    st.header("🔶 Act 2: The Investigation")
    st.markdown("### Trying different search strategies...")

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    strategy_results = get_strategy_results(st.session_state.demo_data)
    strategies = strategy_results.get("strategies", [])
    by_query = strategy_results.get("by_query", {})

    # Strategy Grid Overview
    st.markdown("### 9 Search Strategies Compared")

    strategy_data = prepare_strategy_comparison_data(strategy_results)
    strategy_df = pd.DataFrame(strategy_data)

    # Create strategy cards in a 3x3 grid
    rows = [strategies[i:i+3] for i in range(0, len(strategies), 3)]

    for row in rows:
        cols = st.columns(3)
        for idx, strategy in enumerate(row):
            with cols[idx]:
                s_id = strategy["id"]
                # Find metrics for this strategy
                s_metrics = next((s for s in strategy_data if s["strategy_id"] == s_id), None)

                if s_metrics:
                    recall = s_metrics["avg_recall"]
                    ndcg = s_metrics["avg_ndcg"]
                    color = get_strategy_color(s_id)

                    st.markdown(f"""
                    <div style="background-color: #1e293b; border-radius: 8px; padding: 12px;
                                border-left: 4px solid {color}; margin-bottom: 8px;">
                        <div style="font-weight: 600; color: #f8fafc;">{strategy['name']}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                            {strategy['description']}
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 16px;">
                            <span style="color: {ColorScheme.get_recall_color(recall)};">
                                Recall: {format_percent(recall)}
                            </span>
                            <span style="color: {ColorScheme.get_ndcg_color(ndcg)};">
                                nDCG: {format_metric(ndcg)}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Per-Query Deep Dive
    st.markdown("### Per-Query Strategy Performance")

    query_options = list(by_query.keys())
    selected_query = st.selectbox(
        "Select a query to analyze:",
        query_options,
        format_func=lambda x: truncate_query(x, 50)
    )

    if selected_query:
        query_insight = by_query[selected_query].get("insight", "")
        best_strategy = by_query[selected_query].get("best_strategy", "")
        max_recall = by_query[selected_query].get("max_recall", 0)

        # Show insight callout
        callout_class = "success" if max_recall >= 1.0 else "warning" if max_recall >= 0.8 else "danger"
        st.markdown(f"""
        <div class="callout {callout_class}">
            <strong>Best Strategy: {best_strategy}</strong> (Max Recall: {format_percent(max_recall)})<br>
            {query_insight}
        </div>
        """, unsafe_allow_html=True)

        # Strategy comparison chart for selected query
        query_data = prepare_query_deep_dive_data(selected_query, strategy_results)
        query_df = pd.DataFrame(query_data)

        col1, col2 = st.columns(2)

        with col1:
            # Recall bar chart
            fig_recall = px.bar(
                query_df,
                x="strategy_name",
                y="recall",
                color="recall",
                color_continuous_scale=[(0, COLORS.PROBLEM), (0.5, COLORS.NEEDS_WORK),
                                       (0.8, COLORS.GOOD), (1.0, COLORS.PERFECT)],
                title=f"Recall by Strategy",
                range_color=[0, 1]
            )
            fig_recall.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=40, b=100)
            )
            fig_recall.add_hline(y=1.0, line_dash="dash", line_color="green")
            st.plotly_chart(fig_recall, use_container_width=True)

        with col2:
            # nDCG bar chart
            fig_ndcg = px.bar(
                query_df,
                x="strategy_name",
                y="ndcg",
                color="ndcg",
                color_continuous_scale=[(0, COLORS.PROBLEM), (0.5, COLORS.NEEDS_WORK),
                                       (0.7, COLORS.GOOD), (1.0, COLORS.PERFECT)],
                title=f"nDCG by Strategy",
                range_color=[0, 1]
            )
            fig_ndcg.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=40, b=100)
            )
            st.plotly_chart(fig_ndcg, use_container_width=True)

    st.markdown("---")

    # Key Finding Highlight
    st.markdown("### Key Finding: The Contrast")

    key_finding = strategy_results.get("key_finding", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="callout success">
            <strong>✅ "animal whales" query</strong><br>
            Content-Only: 75% Recall<br>
            Title-Only: <strong>100% Recall</strong> (+25%!)<br><br>
            <em>Title weighting finds the "Whale" article at rank 1!</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="callout danger">
            <strong>🔴 "machine learning" query</strong><br>
            Content-Only: 50% Recall<br>
            Title-Only: 50% Recall<br>
            Hybrid: 50% Recall<br>
            <strong>ALL STRATEGIES: 50%</strong><br><br>
            <em>No strategy can break the 50% ceiling... Why?</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout warning">
        <strong>🤔 Investigation Conclusion</strong><br>
        Title weighting helps entity queries like "animal whales" significantly.
        But for "machine learning", <strong>ALL 9 strategies are stuck at 50%</strong>.
        Something is fundamentally wrong. Let's dig deeper...
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Act 3: The Revelation
# ============================================================================

def render_act3_revelation():
    """Render Act 3: The Revelation tab."""
    st.header("🟢 Act 3: The Revelation")
    st.markdown("### The AHA moment: It's the DATA!")

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    corpus_analysis = get_corpus_analysis(st.session_state.demo_data)

    # SQL Evidence Panel
    st.markdown("### Let's Check What's in Our Database")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Does a 'Machine Learning' article exist?")

        ml_title_search = corpus_analysis.get("ml_title_search", {})

        st.markdown(f"""
        <div class="sql-block">
        <span style="color: #3b82f6;">SELECT</span> title
        <span style="color: #3b82f6;">FROM</span> articles
        <span style="color: #3b82f6;">WHERE</span> title <span style="color: #22c55e;">ILIKE</span> '%machine learning%';
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top: 12px; padding: 12px; background-color: #450a0a;
                    border-radius: 8px; border: 1px solid #ef4444;">
            <strong style="color: #ef4444;">Result: ({ml_title_search.get('count', 0)} rows)</strong><br>
            <span style="color: #fca5a5;">No articles with "Machine Learning" in the title!</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### What does FTS find for 'machine learning'?")

        fts_results = corpus_analysis.get("ml_fts_results", {}).get("results", [])

        st.markdown("""
        <div class="sql-block" style="font-size: 0.75rem;">
        <span style="color: #3b82f6;">SELECT</span> id, title, ts_rank(...) as rank
        <span style="color: #3b82f6;">FROM</span> articles
        <span style="color: #3b82f6;">ORDER BY</span> rank <span style="color: #3b82f6;">DESC</span>
        <span style="color: #3b82f6;">LIMIT</span> 10;
        </div>
        """, unsafe_allow_html=True)

        # Show FTS results with highlighting
        for result in fts_results[:6]:
            is_expected = result.get("is_expected", False)
            title = result["title"]
            rank = result.get("rank", 0)
            note = result.get("note", "")

            bg_color = "#052e16" if is_expected else "#450a0a" if "machine" in title.lower() and not is_expected else "#1e293b"
            text_color = "#22c55e" if is_expected else "#ef4444" if not is_expected and "machine" in title.lower() else "#94a3b8"

            st.markdown(f"""
            <div style="padding: 6px 10px; background-color: {bg_color};
                        border-radius: 4px; margin: 4px 0; font-size: 0.85rem;">
                <span style="color: {text_color};">{title}</span>
                <span style="color: #64748b; float: right;">rank: {rank:.3f}</span>
                {f'<br><span style="color: #f97316; font-size: 0.75rem;">{note}</span>' if note else ''}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Expected vs Actual Explanation
    st.markdown("### What We Expected vs What Exists")

    expected_articles = corpus_analysis.get("expected_articles_explanation", {})

    st.markdown("""
    The 4 "expected" documents for the machine learning query are:
    """)

    cols = st.columns(4)
    for idx, (doc_id, info) in enumerate(expected_articles.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style="background-color: #1e293b; border-radius: 8px; padding: 12px;
                        border-top: 3px solid #eab308;">
                <div style="font-weight: 600; color: #f8fafc;">{info['title']}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">
                    {info['why_expected']}
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">
                    Mentions "ML": {info['ml_mentions']}x
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # The Core Problem
    problem_diagnosis = corpus_analysis.get("problem_diagnosis", {})

    st.markdown("### The Root Cause")

    issue_text = problem_diagnosis.get('issue', '')
    root_cause_text = problem_diagnosis.get('root_cause', '')
    impact_text = problem_diagnosis.get('impact', '')
    solution_text = problem_diagnosis.get('solution', '')

    st.markdown(f"""
    <div style="background-color: #052e16; border-radius: 8px; padding: 16px; margin: 16px 0; border-left: 4px solid #22c55e;">
        <h4 style="margin-top: 0; color: #22c55e;">Problem Diagnosis</h4>
        <p style="margin: 8px 0;"><strong style="color: #f8fafc;">Issue:</strong> <span style="color: #94a3b8;">{issue_text}</span></p>
        <p style="margin: 8px 0;"><strong style="color: #f8fafc;">Root Cause:</strong> <span style="color: #94a3b8;">{root_cause_text}</span></p>
        <p style="margin: 8px 0;"><strong style="color: #f8fafc;">Impact:</strong> <span style="color: #94a3b8;">{impact_text}</span></p>
        <p style="margin: 8px 0;"><strong style="color: #f8fafc;">Solution:</strong> <span style="color: #94a3b8;">{solution_text}</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Visual: The Mismatch
    st.markdown("### The Query-Corpus Mismatch")

    col1, col2, col3 = st.columns([1, 0.2, 1])

    with col1:
        st.markdown("""
        <div style="background-color: #1e3a5f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 3rem;">🔍</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #f8fafc; margin-top: 8px;">
                Query Intent
            </div>
            <div style="color: #94a3b8; margin-top: 8px;">
                "What is machine learning?"
            </div>
            <div style="margin-top: 12px; color: #3b82f6;">
                Wants: A comprehensive ML overview
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 150px;">
            <span style="font-size: 2rem; color: #ef4444;">≠</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color: #422006; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 3rem;">📚</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #f8fafc; margin-top: 8px;">
                Corpus Reality
            </div>
            <div style="color: #94a3b8; margin-top: 8px;">
                No "Machine Learning" article
            </div>
            <div style="margin-top: 12px; color: #f97316;">
                Has: Related articles that mention ML
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Takeaway
    st.markdown("""
    <div style="background: linear-gradient(135deg, #052e16 0%, #0f172a 100%);
                border-radius: 12px; padding: 24px; margin: 20px 0; text-align: center;
                border: 2px solid #22c55e;">
        <div style="font-size: 2rem;">💡</div>
        <h3 style="color: #22c55e; margin-top: 8px;">The Key Insight</h3>
        <p style="font-size: 1.2rem; color: #f8fafc; margin-top: 12px;">
            <strong>No search algorithm can find documents that don't exist.</strong>
        </p>
        <p style="color: #94a3b8; margin-top: 8px;">
            RAG Quality = Corpus Coverage × Search Algorithm × Generation Model
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Summary Tab
# ============================================================================

def render_summary():
    """Render the Summary tab with key takeaways."""
    st.header("📊 Summary: Key Takeaways")

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    # Big Dashboard Cards
    st.markdown("### The Numbers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #052e16 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; text-align: center; height: 180px;">
            <div style="font-size: 3.5rem; font-weight: 700; color: #22c55e;">3 of 4</div>
            <div style="font-size: 1rem; color: #94a3b8; margin-top: 8px;">queries</div>
            <div style="font-size: 1.2rem; color: #f8fafc; margin-top: 8px;">
                ✅ solvable with tuning
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #422006 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; text-align: center; height: 180px;">
            <div style="font-size: 3.5rem; font-weight: 700; color: #eab308;">+25%</div>
            <div style="font-size: 1rem; color: #94a3b8; margin-top: 8px;">recall gain</div>
            <div style="font-size: 1.2rem; color: #f8fafc; margin-top: 8px;">
                with title weighting
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #450a0a 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; text-align: center; height: 180px;">
            <div style="font-size: 3.5rem; font-weight: 700; color: #ef4444;">DATA</div>
            <div style="font-size: 1rem; color: #94a3b8; margin-top: 8px;">is the</div>
            <div style="font-size: 1.2rem; color: #f8fafc; margin-top: 8px;">
                bottleneck (not search)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.markdown("### Recommendations")

    recommendations = [
        {
            "icon": "1️⃣",
            "title": "Start with k=20-50 for most queries",
            "description": "Higher k gives better recall but increases latency. Find your sweet spot."
        },
        {
            "icon": "2️⃣",
            "title": "Use title-weighting for entity/factual queries",
            "description": "Queries looking for specific things benefit from title matching (e.g., 'animal whales')."
        },
        {
            "icon": "3️⃣",
            "title": "Audit your corpus before blaming the search",
            "description": "No algorithm can find documents that don't exist. Check coverage first!"
        },
        {
            "icon": "4️⃣",
            "title": "Use ground truth evaluation",
            "description": "Without labeled test cases, you're flying blind. Invest in evaluation data."
        }
    ]

    for rec in recommendations:
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px;
                    margin: 8px 0; display: flex; align-items: flex-start; gap: 16px;">
            <div style="font-size: 1.5rem;">{rec['icon']}</div>
            <div>
                <div style="font-weight: 600; color: #f8fafc;">{rec['title']}</div>
                <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 4px;">
                    {rec['description']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # The Formula
    st.markdown("### The RAG Quality Formula")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
                border-radius: 12px; padding: 32px; text-align: center; margin: 20px 0;">
        <div style="font-size: 1.5rem; color: #f8fafc; font-family: 'Georgia', serif;">
            RAG Quality = <span style="color: #ef4444;">Corpus Coverage</span>
            × <span style="color: #eab308;">Search Algorithm</span>
            × <span style="color: #22c55e;">Generation Model</span>
        </div>
        <div style="margin-top: 16px; color: #94a3b8;">
            If any factor is zero, the product is zero.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What We Learned Table
    st.markdown("### Query Analysis Summary")

    summary_data = [
        {
            "Query": "Who invented the telephone?",
            "Status": "✅ Perfect",
            "First 100%": "k=10",
            "Best Strategy": "Any",
            "Insight": "Easy factual query with clear entity match"
        },
        {
            "Query": "neural networks deep learning",
            "Status": "✅ Perfect",
            "First 100%": "k=10",
            "Best Strategy": "Content-Only",
            "Insight": "Semantic search captures conceptual queries well"
        },
        {
            "Query": "animal whales",
            "Status": "✅ Fixed",
            "First 100%": "k=20",
            "Best Strategy": "Title-Only",
            "Insight": "Title weighting boosted recall from 75% to 100%"
        },
        {
            "Query": "What is machine learning?",
            "Status": "🔴 Corpus Issue",
            "First 100%": "k=100 (barely)",
            "Best Strategy": "None helps",
            "Insight": "No dedicated ML article - need corpus curation"
        }
    ]

    st.dataframe(
        pd.DataFrame(summary_data),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # Resources
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Learn More")
        st.markdown("""
        - [Understanding nDCG](lab/evaluation/examples/UNDERSTANDING_NDCG.md)
        - [K-Balance Experiments](lab/evaluation/examples/README_K_BALANCE.md)
        - [Search Strategy Comparison](lab/evaluation/examples/demo_search_strategies.py)
        """)

    with col2:
        st.markdown("### 🔗 Repository")
        st.markdown("""
        **pgvector RAG Search Lab**

        [github.com/boutaga/pgvector_RAG_search_lab](https://github.com/boutaga/pgvector_RAG_search_lab)

        Educational toolkit for PostgreSQL + pgvector + RAG
        """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    apply_custom_css()
    init_session_state()

    # Load demo data
    try:
        st.session_state.demo_data = load_cached_demo_data()
    except Exception as e:
        st.error(f"Failed to load demo data: {e}")
        st.stop()

    # Render sidebar
    demo_mode = render_sidebar()

    # Main header
    st.title("🎯 RAG Search Quality: The Complete Picture")
    st.markdown("*A visual story about retrieval quality, search strategies, and corpus coverage*")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 The Problem",
        "🔶 Investigation",
        "🟢 The Insight",
        "📊 Summary"
    ])

    with tab1:
        render_act1_problem()

    with tab2:
        render_act2_investigation()

    with tab3:
        render_act3_revelation()

    with tab4:
        render_summary()


if __name__ == "__main__":
    main()
