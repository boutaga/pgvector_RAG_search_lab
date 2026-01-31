#!/usr/bin/env python3
"""
K-Balance Experiment Streamlit UI
=================================

A booth-friendly Streamlit UI for visualizing RAG k-balance experiments
at PostgreSQL conferences. Designed for non-DBAs to understand the
trade-offs between retrieval quality, latency, and cost.

Key Features:
- Two intuitive knobs: k_retrieve and k_context with plain-English labels
- Three key visualizations: Quality vs Cost, Latency Breakdown, Tradeoff Frontier
- Demo mode with pre-computed results for reliable booth demos
- Progress tracking during live experiments

Usage:
------
    streamlit run lab/evaluation/streamlit_k_balance.py

Environment Variables:
----------------------
    DATABASE_URL: PostgreSQL connection string
    OPENAI_API_KEY: OpenAI API key for embeddings (not needed in demo mode)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Lazy imports to allow streamlit to start faster
def get_experiment_functions():
    """Lazy import of experiment functions."""
    from lab.evaluation.examples.k_balance_experiment import (
        load_test_cases,
        run_single_experiment_with_timing,
        run_multi_k_experiment_with_timing,
        estimate_context_tokens
    )
    from lab.evaluation.evaluator import TestCase
    return {
        'load_test_cases': load_test_cases,
        'run_single_experiment_with_timing': run_single_experiment_with_timing,
        'run_multi_k_experiment_with_timing': run_multi_k_experiment_with_timing,
        'estimate_context_tokens': estimate_context_tokens,
        'TestCase': TestCase
    }


# ============================================================================
# Configuration and Constants
# ============================================================================

# Non-DBA friendly labels
LABELS = {
    'k_retrieve': "How many candidates we fetch from the database",
    'k_context': "How many chunks we send to the LLM",
    'recall': "Did we find all the relevant documents?",
    'ndcg': "Are the best results ranked at the top?",
    'precision': "Of the documents we found, how many were relevant?",
    'latency': "How long did it take?",
    'embed_time': "Time to convert query to vector (API call)",
    'db_time': "Time to search the database",
    'context_tokens': "LLM cost (more tokens = more expensive)"
}

# Default k values for sliders
DEFAULT_K_RETRIEVE = 100
DEFAULT_K_CONTEXT = 8
K_RETRIEVE_OPTIONS = [10, 25, 50, 100, 150, 200, 300]
K_CONTEXT_OPTIONS = [3, 5, 8, 10, 15, 20]

# Demo data path
DEMO_DATA_PATH = Path(__file__).parent / "demo_data" / "k_balance_demo.json"


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'k_balance_config': {
            'table_name': 'articles',
            'vector_column': os.getenv('CONTENT_VECTOR_COLUMN', 'content_vector_3072'),
            'content_columns': ['title', 'content'],
            'id_column': 'id'
        },
        'k_balance_test_cases': None,
        'k_balance_current_results': None,
        'k_balance_history': [],
        'use_demo_mode': True,
        'k_retrieve': DEFAULT_K_RETRIEVE,
        'k_context': DEFAULT_K_CONTEXT,
        'test_multiple_values': False,
        'k_retrieve_values': [50, 100, 200],
        'k_context_values': [5, 8, 10]
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# Demo Data Loading
# ============================================================================

def load_demo_data() -> Optional[Dict[str, Any]]:
    """Load pre-computed demo data for booth demos."""
    if not DEMO_DATA_PATH.exists():
        return None

    try:
        with open(DEMO_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load demo data: {e}")
        return None


def get_demo_results() -> Optional[Dict[str, Any]]:
    """Get demo results, loading if needed."""
    demo_data = load_demo_data()
    if demo_data:
        return demo_data.get('results')
    return None


# ============================================================================
# Sidebar Configuration
# ============================================================================

def render_sidebar() -> Dict[str, Any]:
    """
    Render the configuration sidebar.

    Returns:
        Dictionary with configuration values
    """
    st.sidebar.title("K-Balance Experiment")
    st.sidebar.markdown("---")

    # Demo mode toggle
    use_demo = st.sidebar.checkbox(
        "Use Demo Data",
        value=st.session_state.use_demo_mode,
        help="Use pre-computed sample results for instant visualization (no API calls needed)"
    )
    st.session_state.use_demo_mode = use_demo

    if use_demo:
        st.sidebar.success("Demo mode - instant results!")
    else:
        # Test file upload
        st.sidebar.subheader("Test Cases")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Test Cases JSON",
            type=['json'],
            help="JSON file with query + expected_doc_ids"
        )

        if uploaded_file:
            try:
                test_data = json.load(uploaded_file)
                st.session_state.k_balance_test_cases = test_data
                st.sidebar.success(f"Loaded {len(test_data)} test cases")
            except Exception as e:
                st.sidebar.error(f"Failed to load: {e}")

        # Check for environment variables
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            st.sidebar.warning("DATABASE_URL not set")
        else:
            st.sidebar.info("DB connected via DATABASE_URL")

    st.sidebar.markdown("---")

    # Configuration section
    st.sidebar.subheader("Configuration")

    config = st.session_state.k_balance_config

    table_name = st.sidebar.text_input(
        "Table",
        value=config['table_name'],
        help="Database table containing documents"
    )
    config['table_name'] = table_name

    vector_column = st.sidebar.text_input(
        "Vector Column",
        value=config['vector_column'],
        help="Column containing vector embeddings"
    )
    config['vector_column'] = vector_column

    st.sidebar.markdown("---")

    # K Parameters section
    st.sidebar.subheader("k Parameters")

    # k_retrieve with friendly label
    st.sidebar.caption(f"**k_retrieve**: {LABELS['k_retrieve']}")
    k_retrieve = st.sidebar.select_slider(
        "k_retrieve",
        options=K_RETRIEVE_OPTIONS,
        value=st.session_state.k_retrieve,
        label_visibility="collapsed"
    )
    st.session_state.k_retrieve = k_retrieve

    # k_context with friendly label
    st.sidebar.caption(f"**k_context**: {LABELS['k_context']}")
    k_context_options = [k for k in K_CONTEXT_OPTIONS if k <= k_retrieve]
    k_context = st.sidebar.select_slider(
        "k_context",
        options=k_context_options if k_context_options else [K_CONTEXT_OPTIONS[0]],
        value=min(st.session_state.k_context, k_retrieve),
        label_visibility="collapsed"
    )
    st.session_state.k_context = k_context

    # Multi-value testing
    st.sidebar.markdown("---")
    test_multiple = st.sidebar.checkbox(
        "Test multiple values",
        value=st.session_state.test_multiple_values,
        help="Run experiments across multiple k configurations"
    )
    st.session_state.test_multiple_values = test_multiple

    if test_multiple:
        k_ret_values = st.sidebar.multiselect(
            "k_retrieve values",
            options=K_RETRIEVE_OPTIONS,
            default=st.session_state.k_retrieve_values,
            help="Select multiple k_retrieve values to test"
        )
        st.session_state.k_retrieve_values = k_ret_values if k_ret_values else [100]

        k_ctx_values = st.sidebar.multiselect(
            "k_context values",
            options=K_CONTEXT_OPTIONS,
            default=st.session_state.k_context_values,
            help="Select multiple k_context values to test"
        )
        st.session_state.k_context_values = k_ctx_values if k_ctx_values else [8]

    st.sidebar.markdown("---")

    # Run button
    run_clicked = st.sidebar.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
        disabled=not use_demo and st.session_state.k_balance_test_cases is None
    )

    return {
        'use_demo': use_demo,
        'run_clicked': run_clicked,
        'k_retrieve': k_retrieve,
        'k_context': k_context,
        'test_multiple': test_multiple,
        'k_retrieve_values': st.session_state.k_retrieve_values,
        'k_context_values': st.session_state.k_context_values,
        'config': config
    }


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(sidebar_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run the k-balance experiment.

    Args:
        sidebar_config: Configuration from sidebar

    Returns:
        Experiment results or None if failed
    """
    if sidebar_config['use_demo']:
        # Load demo results
        demo_results = get_demo_results()
        if demo_results:
            st.session_state.k_balance_current_results = demo_results
            return demo_results
        else:
            st.error("Demo data not found. Please create demo data first.")
            return None

    # Real experiment
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        st.error("DATABASE_URL environment variable not set")
        return None

    test_cases_data = st.session_state.k_balance_test_cases
    if not test_cases_data:
        st.error("No test cases loaded")
        return None

    # Import experiment functions
    funcs = get_experiment_functions()

    # Convert test case data to TestCase objects
    test_cases = []
    for item in test_cases_data:
        test_cases.append(funcs['TestCase'](
            query=item["query"],
            expected_doc_ids=item.get("expected_doc_ids", []),
            expected_answer=item.get("expected_answer"),
            metadata=item.get("metadata", {})
        ))

    config = sidebar_config['config']

    # Progress tracking
    progress_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(current, total, message=""):
        progress = current / max(total, 1)
        progress_bar.progress(progress)
        status_text.text(message or f"Processing {current}/{total}...")

    try:
        if sidebar_config['test_multiple']:
            # Multi-k experiment
            results = funcs['run_multi_k_experiment_with_timing'](
                test_cases=test_cases,
                db_url=db_url,
                table_name=config['table_name'],
                vector_column=config['vector_column'],
                content_columns=config['content_columns'],
                k_retrieve_values=sidebar_config['k_retrieve_values'],
                k_context_values=sidebar_config['k_context_values'],
                id_column=config['id_column'],
                progress_callback=progress_callback
            )
        else:
            # Single configuration
            detailed_results = funcs['run_single_experiment_with_timing'](
                test_cases=test_cases,
                db_url=db_url,
                table_name=config['table_name'],
                vector_column=config['vector_column'],
                content_columns=config['content_columns'],
                k_retrieve=sidebar_config['k_retrieve'],
                k_context=sidebar_config['k_context'],
                id_column=config['id_column'],
                progress_callback=progress_callback
            )

            # Wrap in standard format
            results = {
                "metadata": {
                    "table_name": config['table_name'],
                    "vector_column": config['vector_column'],
                    "num_test_cases": len(test_cases),
                    "k_retrieve_values": [sidebar_config['k_retrieve']],
                    "k_context_values": [sidebar_config['k_context']]
                },
                "detailed_results": detailed_results,
                "summary": {}
            }

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Store results
        st.session_state.k_balance_current_results = results
        st.session_state.k_balance_history.append(results)

        return results

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Experiment failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ============================================================================
# Visualization Charts
# ============================================================================

def create_quality_vs_cost_chart(summary: Dict[str, Any], metric: str = 'recall') -> go.Figure:
    """
    Create Quality vs Cost scatter chart.

    Args:
        summary: Summary statistics from experiment
        metric: 'recall' or 'ndcg'

    Returns:
        Plotly figure
    """
    if not summary:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    data = []
    for key, stats in summary.items():
        metric_key = f'avg_{metric}@k' if metric in ['recall', 'precision', 'ndcg', 'f1'] else f'avg_{metric}'
        data.append({
            'k_retrieve': stats['k_retrieve'],
            'k_context': stats['k_context'],
            'config': f"k_r={stats['k_retrieve']}, k_c={stats['k_context']}",
            'metric_value': stats.get(metric_key, 0),
            'context_tokens': stats.get('avg_context_tokens', 0),
            'latency_ms': stats.get('avg_latency_ms', 0)
        })

    df = pd.DataFrame(data)

    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Friendly metric labels
    metric_labels = {
        'recall': LABELS['recall'],
        'ndcg': LABELS['ndcg'],
        'precision': LABELS['precision']
    }

    fig = px.scatter(
        df,
        x='context_tokens',
        y='metric_value',
        color='k_retrieve',
        size='k_context',
        hover_data=['config', 'latency_ms'],
        title=f"Quality vs Cost: {metric_labels.get(metric, metric)}",
        labels={
            'context_tokens': LABELS['context_tokens'],
            'metric_value': metric_labels.get(metric, metric),
            'k_retrieve': 'Candidates Fetched',
            'k_context': 'Chunks to LLM'
        },
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        yaxis_range=[0, 1.05],
        hovermode='closest',
        font=dict(size=14)
    )

    return fig


def create_latency_breakdown_chart(summary: Dict[str, Any]) -> go.Figure:
    """
    Create stacked bar chart showing latency breakdown.

    Args:
        summary: Summary statistics from experiment

    Returns:
        Plotly figure
    """
    if not summary:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    configs = []
    embed_times = []
    db_times = []

    for key, stats in sorted(summary.items(), key=lambda x: (x[1]['k_retrieve'], x[1]['k_context'])):
        configs.append(f"k_r={stats['k_retrieve']}\nk_c={stats['k_context']}")
        embed_times.append(stats.get('avg_embed_time_ms', 0))
        db_times.append(stats.get('avg_db_time_ms', 0))

    fig = go.Figure(data=[
        go.Bar(
            name=LABELS['embed_time'],
            x=configs,
            y=embed_times,
            marker_color='#636EFA',
            hovertemplate='%{y:.1f}ms<extra>Embed Time</extra>'
        ),
        go.Bar(
            name=LABELS['db_time'],
            x=configs,
            y=db_times,
            marker_color='#EF553B',
            hovertemplate='%{y:.1f}ms<extra>DB Time</extra>'
        )
    ])

    fig.update_layout(
        barmode='stack',
        title=f"Latency Breakdown: {LABELS['latency']}",
        xaxis_title='Configuration',
        yaxis_title='Time (ms)',
        legend_title='Component',
        font=dict(size=14),
        hovermode='x unified'
    )

    return fig


def create_tradeoff_frontier_chart(summary: Dict[str, Any]) -> go.Figure:
    """
    Create tradeoff frontier scatter chart with Pareto frontier.

    Args:
        summary: Summary statistics from experiment

    Returns:
        Plotly figure
    """
    if not summary:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    data = []
    for key, stats in summary.items():
        data.append({
            'k_retrieve': stats['k_retrieve'],
            'k_context': stats['k_context'],
            'config': f"k_r={stats['k_retrieve']}, k_c={stats['k_context']}",
            'latency_ms': stats.get('avg_latency_ms', 0),
            'ndcg': stats.get('avg_ndcg@k', 0),
            'recall': stats.get('avg_recall@k', 0),
            'context_tokens': stats.get('avg_context_tokens', 0)
        })

    df = pd.DataFrame(data)

    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Calculate Pareto frontier (lower latency, higher nDCG is better)
    pareto_points = []
    for _, row in df.iterrows():
        is_pareto = True
        for _, other in df.iterrows():
            # Another point dominates if it has lower latency AND higher nDCG
            if other['latency_ms'] < row['latency_ms'] and other['ndcg'] > row['ndcg']:
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(row)

    pareto_df = pd.DataFrame(pareto_points)
    pareto_df = pareto_df.sort_values('latency_ms')

    # Create scatter plot
    fig = px.scatter(
        df,
        x='latency_ms',
        y='ndcg',
        size='context_tokens',
        color='recall',
        hover_data=['config', 'k_retrieve', 'k_context'],
        title='Tradeoff Frontier: Latency vs Quality',
        labels={
            'latency_ms': LABELS['latency'] + ' (ms)',
            'ndcg': LABELS['ndcg'],
            'context_tokens': LABELS['context_tokens'],
            'recall': LABELS['recall']
        },
        color_continuous_scale='RdYlGn'
    )

    # Add Pareto frontier line
    if not pareto_df.empty and len(pareto_df) > 1:
        fig.add_trace(go.Scatter(
            x=pareto_df['latency_ms'],
            y=pareto_df['ndcg'],
            mode='lines',
            name='Pareto Frontier',
            line=dict(color='black', width=2, dash='dash'),
            hoverinfo='skip'
        ))

    fig.update_layout(
        yaxis_range=[0, 1.05],
        font=dict(size=14),
        hovermode='closest'
    )

    return fig


# ============================================================================
# Tab Renderers
# ============================================================================

def render_overview_tab(results: Optional[Dict[str, Any]]):
    """Render the Overview tab with summary metrics."""
    st.subheader("Summary Metrics")

    if not results:
        st.info("Run an experiment or enable Demo Mode to see results.")
        return

    metadata = results.get('metadata', {})
    summary = results.get('summary', {})
    detailed = results.get('detailed_results', [])

    # Key metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Test Cases", metadata.get('num_test_cases', len(detailed)))

    with col2:
        num_configs = len(summary) if summary else 1
        st.metric("Configs Tested", num_configs)

    with col3:
        if summary:
            best_recall = max(s.get('avg_recall@k', 0) for s in summary.values())
            st.metric("Best Recall", f"{best_recall:.1%}")
        elif detailed:
            best_recall = max(r.get('recall@k', 0) for r in detailed)
            st.metric("Best Recall", f"{best_recall:.1%}")

    with col4:
        if summary:
            best_ndcg = max(s.get('avg_ndcg@k', 0) for s in summary.values())
            st.metric("Best nDCG", f"{best_ndcg:.3f}")
        elif detailed:
            best_ndcg = max(r.get('ndcg@k', 0) for r in detailed)
            st.metric("Best nDCG", f"{best_ndcg:.3f}")

    st.divider()

    # Best configuration finder
    if summary:
        st.subheader("Best Configurations")

        col1, col2 = st.columns(2)

        with col1:
            # Best for quality
            best_quality_key = max(summary.keys(), key=lambda k: summary[k].get('avg_ndcg@k', 0))
            best_quality = summary[best_quality_key]
            st.success(f"""
            **Best Quality (nDCG)**
            - Configuration: k_retrieve={best_quality['k_retrieve']}, k_context={best_quality['k_context']}
            - nDCG: {best_quality.get('avg_ndcg@k', 0):.3f}
            - Recall: {best_quality.get('avg_recall@k', 0):.1%}
            """)

        with col2:
            # Best for speed
            best_speed_key = min(summary.keys(), key=lambda k: summary[k].get('avg_latency_ms', float('inf')))
            best_speed = summary[best_speed_key]
            st.info(f"""
            **Fastest**
            - Configuration: k_retrieve={best_speed['k_retrieve']}, k_context={best_speed['k_context']}
            - Latency: {best_speed.get('avg_latency_ms', 0):.0f}ms
            - nDCG: {best_speed.get('avg_ndcg@k', 0):.3f}
            """)

        # Summary table
        st.subheader("All Configurations")
        summary_data = []
        for key, stats in sorted(summary.items(), key=lambda x: -x[1].get('avg_ndcg@k', 0)):
            summary_data.append({
                'k_retrieve': stats['k_retrieve'],
                'k_context': stats['k_context'],
                'Recall': f"{stats.get('avg_recall@k', 0):.1%}",
                'nDCG': f"{stats.get('avg_ndcg@k', 0):.3f}",
                'Latency (ms)': f"{stats.get('avg_latency_ms', 0):.0f}",
                'Embed (ms)': f"{stats.get('avg_embed_time_ms', 0):.0f}",
                'DB (ms)': f"{stats.get('avg_db_time_ms', 0):.0f}",
                'Tokens': f"{stats.get('avg_context_tokens', 0):.0f}"
            })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


def render_visualizations_tab(results: Optional[Dict[str, Any]]):
    """Render the Visualizations tab with three key charts."""
    if not results or not results.get('summary'):
        st.info("Run a multi-configuration experiment to see visualizations.")
        return

    summary = results['summary']

    # Chart 1: Quality vs Cost
    st.subheader("1. Quality vs Cost")
    st.caption("How does retrieval quality change with context size (token cost)?")

    metric_choice = st.radio(
        "Quality Metric",
        ['recall', 'ndcg', 'precision'],
        horizontal=True,
        format_func=lambda x: {'recall': 'Recall', 'ndcg': 'nDCG', 'precision': 'Precision'}[x]
    )

    fig1 = create_quality_vs_cost_chart(summary, metric_choice)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # Chart 2: Latency Breakdown
    st.subheader("2. Latency Breakdown")
    st.caption("Where is time spent? Embedding generation vs database search.")

    fig2 = create_latency_breakdown_chart(summary)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Chart 3: Tradeoff Frontier
    st.subheader("3. Tradeoff Frontier")
    st.caption("Find the sweet spot between speed and quality. Points on the dashed line are optimal.")

    fig3 = create_tradeoff_frontier_chart(summary)
    st.plotly_chart(fig3, use_container_width=True)


def render_detailed_tab(results: Optional[Dict[str, Any]]):
    """Render the Detailed Results tab with per-query breakdown."""
    if not results or not results.get('detailed_results'):
        st.info("Run an experiment to see detailed per-query results.")
        return

    detailed = results['detailed_results']

    st.subheader("Per-Query Results")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        # Filter by category if available
        categories = set()
        for r in detailed:
            cat = r.get('query_metadata', {}).get('category')
            if cat:
                categories.add(cat)

        if categories:
            selected_category = st.selectbox(
                "Filter by Category",
                ['All'] + sorted(categories)
            )
        else:
            selected_category = 'All'

    with col2:
        # Filter by k_retrieve
        k_values = sorted(set(r['k_retrieve'] for r in detailed))
        if len(k_values) > 1:
            selected_k = st.selectbox(
                "Filter by k_retrieve",
                ['All'] + k_values,
                format_func=lambda x: f"k_retrieve = {x}" if x != 'All' else 'All'
            )
        else:
            selected_k = 'All'

    # Apply filters
    filtered = detailed
    if selected_category != 'All':
        filtered = [r for r in filtered if r.get('query_metadata', {}).get('category') == selected_category]
    if selected_k != 'All':
        filtered = [r for r in filtered if r['k_retrieve'] == selected_k]

    st.write(f"Showing {len(filtered)} results")

    # Results table
    table_data = []
    for r in filtered:
        table_data.append({
            'Query': r['query'][:50] + '...' if len(r['query']) > 50 else r['query'],
            'k_ret': r['k_retrieve'],
            'k_ctx': r['k_context'],
            'Recall': f"{r.get('recall@k', 0):.1%}",
            'nDCG': f"{r.get('ndcg@k', 0):.3f}",
            'Found': f"{r.get('relevant_found', 0)}/{r.get('total_relevant', 0)}",
            'Latency': f"{r.get('latency_ms', 0):.0f}ms",
            'Embed': f"{r.get('embed_time_ms', 0):.0f}ms",
            'DB': f"{r.get('db_time_ms', 0):.0f}ms"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable details
    st.subheader("Query Details")
    for i, r in enumerate(filtered[:10]):  # Limit to 10 for performance
        with st.expander(f"Query {i+1}: {r['query'][:60]}..."):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Recall", f"{r.get('recall@k', 0):.1%}")
                st.metric("nDCG", f"{r.get('ndcg@k', 0):.3f}")

            with col2:
                st.metric("Precision", f"{r.get('precision@k', 0):.1%}")
                st.metric("MRR", f"{r.get('mrr', 0):.3f}")

            with col3:
                st.metric("Latency", f"{r.get('latency_ms', 0):.0f}ms")
                st.metric("Tokens", f"{r.get('context_tokens', 0)}")

            st.caption(f"Found {r.get('relevant_found', 0)} of {r.get('total_relevant', 0)} relevant documents")

            if r.get('query_metadata'):
                st.json(r['query_metadata'])

    # Export buttons
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            "k_balance_results.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        json_data = json.dumps(filtered, indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            "k_balance_results.json",
            "application/json",
            use_container_width=True
        )


def render_comparison_tab():
    """Render the Comparison tab for multi-run analysis."""
    st.subheader("Run Comparison")

    history = st.session_state.k_balance_history

    if len(history) < 2:
        st.info("Run multiple experiments to compare results across runs.")
        st.caption("Your experiment history will appear here.")
        return

    st.write(f"**{len(history)} runs available**")

    # Select runs to compare
    run_options = [f"Run {i+1}" for i in range(len(history))]
    selected_runs = st.multiselect(
        "Select runs to compare",
        run_options,
        default=run_options[-2:]  # Last two runs
    )

    if len(selected_runs) < 2:
        st.warning("Select at least 2 runs to compare")
        return

    # Comparison chart
    comparison_data = []
    for run_name in selected_runs:
        run_idx = int(run_name.split()[1]) - 1
        run_results = history[run_idx]
        summary = run_results.get('summary', {})

        for key, stats in summary.items():
            comparison_data.append({
                'Run': run_name,
                'Config': f"k_r={stats['k_retrieve']}, k_c={stats['k_context']}",
                'nDCG': stats.get('avg_ndcg@k', 0),
                'Recall': stats.get('avg_recall@k', 0),
                'Latency': stats.get('avg_latency_ms', 0)
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        fig = px.bar(
            df,
            x='Config',
            y='nDCG',
            color='Run',
            barmode='group',
            title='nDCG Comparison Across Runs'
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main entry point for the Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="K-Balance Experiment",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for booth-friendly display
    st.markdown("""
    <style>
    .stMetric .metric-label {
        font-size: 1.1rem !important;
    }
    .stMetric .metric-value {
        font-size: 2rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Page header
    st.title("K-Balance Experiment")
    st.markdown("""
    Explore the trade-offs between **retrieval quality**, **latency**, and **cost** in RAG systems.

    - **k_retrieve**: How many candidates we fetch from the database
    - **k_context**: How many chunks we send to the LLM
    """)

    # Render sidebar and get configuration
    sidebar_config = render_sidebar()

    # Run experiment if button clicked
    if sidebar_config['run_clicked']:
        with st.spinner("Running experiment..."):
            run_experiment(sidebar_config)

    # Get current results
    results = st.session_state.k_balance_current_results

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Visualizations",
        "Detailed Results",
        "Run Comparison"
    ])

    with tab1:
        render_overview_tab(results)

    with tab2:
        render_visualizations_tab(results)

    with tab3:
        render_detailed_tab(results)

    with tab4:
        render_comparison_tab()


if __name__ == "__main__":
    main()
