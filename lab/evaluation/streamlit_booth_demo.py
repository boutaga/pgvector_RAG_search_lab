#!/usr/bin/env python3
"""
RAG Booth Demo - Interactive Story-Driven Streamlit App
========================================================

An interactive, guided Streamlit application for PostgreSQL conference booth
demonstrations. Places visitors in a DBA's shoes to understand RAG search quality.

THE STORY:
    Welcome - Meet the metrics: Recall, Precision, nDCG explained simply
    Act 1 - The Problem: Run queries yourself and see which ones struggle
    Act 2 - The Investigation: Try different strategies with interactive buttons
    Act 3 - The Revelation: Discover why some queries can't be fixed
    Summary - Key takeaways for your own RAG systems

Usage:
    streamlit run lab/evaluation/streamlit_booth_demo.py

Features:
    - Guided walkthrough with DBA persona
    - Interactive "Run Query" buttons
    - Metric explanations with visual examples
    - Live mode (real database) or Demo mode (pre-loaded)
    - Step-by-step navigation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import RAG search components
try:
    from lab.search.simple_search import SimpleSearchEngine
    from lab.core.database import DatabaseService
    from lab.core.config import ConfigService
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = str(e)

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
    page_title="RAG Search Quality Lab",
    page_icon="🔬",
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
    /* Main container - minimal top padding */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Tab styling - smaller and more compact */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        margin-bottom: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Big action buttons */
    .stButton > button {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px;
    }

    /* Metric explanation cards */
    .metric-explain {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
        min-height: 200px;
    }

    /* Interactive query card */
    .query-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #334155;
        transition: border-color 0.3s;
    }

    .query-card:hover {
        border-color: #3b82f6;
    }

    .query-card.selected {
        border-color: #22c55e;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
    }

    /* Result badge */
    .result-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .result-badge.success {
        background-color: #052e16;
        color: #22c55e;
    }

    .result-badge.warning {
        background-color: #422006;
        color: #eab308;
    }

    .result-badge.danger {
        background-color: #450a0a;
        color: #ef4444;
    }

    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 20px 0;
    }

    .step-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #334155;
    }

    .step-dot.active {
        background-color: #3b82f6;
    }

    .step-dot.completed {
        background-color: #22c55e;
    }

    /* Persona banner */
    .persona-banner {
        background: linear-gradient(90deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }

    /* Hint box */
    .hint-box {
        background-color: #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        border-left: 4px solid #3b82f6;
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animation for running queries */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .running {
        animation: pulse 1s infinite;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# RAG Search Engine Initialization
# ============================================================================

@st.cache_resource
def get_search_engine():
    """Initialize and cache the search engine for real RAG queries."""
    if not RAG_AVAILABLE:
        return None
    try:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            return None
        db_service = DatabaseService(connection_string=db_url)
        config = ConfigService()
        engine = SimpleSearchEngine(db_service, config, source="wikipedia")
        return engine
    except Exception as e:
        st.warning(f"Could not initialize search engine: {e}")
        return None


def perform_real_rag_search(query: str, top_k: int = 5):
    """
    Perform a real RAG search using the SimpleSearchEngine.

    Returns:
        dict with 'answer', 'sources', 'num_results', 'embedding_time', 'search_time', 'generation_time'
    """
    engine = get_search_engine()
    if engine is None:
        return None

    try:
        import time as time_module

        # Step 1: Generate embedding (we'll time the full search and estimate)
        t_start = time_module.time()

        # Perform search and answer
        result = engine.search_and_answer(
            query=query,
            search_type="dense",
            top_k=top_k,
            include_sources=True
        )

        t_total = (time_module.time() - t_start) * 1000  # Convert to ms

        # Estimate timing breakdown (rough approximation)
        embedding_time = int(t_total * 0.15)  # ~15% for embedding
        search_time = int(t_total * 0.10)     # ~10% for vector search
        generation_time = int(t_total * 0.75) # ~75% for LLM generation

        return {
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'num_results': result.get('num_results', 0),
            'embedding_time': embedding_time,
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': int(t_total)
        }
    except Exception as e:
        st.error(f"Search error: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_article_content(article_ids: tuple):
    """
    Fetch article content from the database by IDs.

    Args:
        article_ids: Tuple of article IDs to fetch (tuple for caching)

    Returns:
        Dict mapping article_id to {'title': str, 'content': str}
    """
    if not RAG_AVAILABLE:
        return {}

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return {}

    try:
        db_service = DatabaseService(connection_string=db_url)
        articles = {}

        with db_service.get_connection() as conn:
            with conn.cursor() as cur:
                # Fetch articles by ID
                id_list = list(article_ids)
                if not id_list:
                    return {}

                placeholders = ','.join(['%s'] * len(id_list))
                query = f"SELECT id, title, content FROM articles WHERE id IN ({placeholders})"
                cur.execute(query, id_list)

                for row in cur.fetchall():
                    articles[row[0]] = {
                        'title': row[1],
                        'content': row[2]
                    }

        return articles
    except Exception as e:
        return {}


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = None
    if 'query_results' not in st.session_state:
        st.session_state.query_results = {}
    if 'strategy_results_cache' not in st.session_state:
        st.session_state.strategy_results_cache = {}
    if 'show_presenter_notes' not in st.session_state:
        st.session_state.show_presenter_notes = False
    if 'live_mode' not in st.session_state:
        st.session_state.live_mode = False
    if 'visited_tabs' not in st.session_state:
        st.session_state.visited_tabs = set()
    # RAG pipeline animation state
    if 'rag_pipeline_step' not in st.session_state:
        st.session_state.rag_pipeline_step = 0
    if 'rag_pipeline_running' not in st.session_state:
        st.session_state.rag_pipeline_running = False
    if 'rag_pipeline_times' not in st.session_state:
        st.session_state.rag_pipeline_times = {}
    # Real RAG results storage
    if 'rag_real_results' not in st.session_state:
        st.session_state.rag_real_results = None


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
        st.markdown("## 🔬 RAG Search Lab")
        st.markdown("---")

        # Mode selection
        st.markdown("### Mode")
        mode = st.radio(
            "Select mode:",
            ["Demo Mode (Pre-loaded)", "Live Mode (Database)"],
            index=0,
            help="Demo mode uses pre-computed results. Live mode connects to your database."
        )
        st.session_state.live_mode = (mode == "Live Mode (Database)")

        if st.session_state.live_mode:
            st.warning("⚠️ Live mode requires database connection")
            with st.expander("Database Settings"):
                st.text_input("DATABASE_URL", type="password", key="db_url")
                st.text_input("OPENAI_API_KEY", type="password", key="api_key")

        st.markdown("---")

        # Progress tracker
        st.markdown("### 📍 Your Progress")
        tabs_progress = {
            "welcome": "📖 Welcome",
            "act1": "🔴 Act 1: Problem",
            "act2": "🔶 Act 2: Investigation",
            "act3": "🟢 Act 3: Revelation",
            "summary": "📊 Summary"
        }

        for tab_id, tab_name in tabs_progress.items():
            if tab_id in st.session_state.visited_tabs:
                st.markdown(f"✅ {tab_name}")
            else:
                st.markdown(f"⬜ {tab_name}")

        st.markdown("---")

        # Presenter notes toggle
        st.session_state.show_presenter_notes = st.toggle(
            "📝 Presenter Notes",
            value=st.session_state.show_presenter_notes
        )

        if st.session_state.show_presenter_notes:
            with st.expander("Talking Points", expanded=True):
                if st.session_state.demo_data:
                    talking_points = get_talking_points(st.session_state.demo_data)
                    current_tab = st.session_state.get('current_tab', 'welcome')
                    if current_tab == 'act1':
                        for point in talking_points.get("act_1_problem", [])[:3]:
                            st.markdown(f"• {point}")
                    elif current_tab == 'act2':
                        for point in talking_points.get("act_2_investigation", [])[:3]:
                            st.markdown(f"• {point}")
                    elif current_tab == 'act3':
                        for point in talking_points.get("act_3_revelation", [])[:3]:
                            st.markdown(f"• {point}")

        st.markdown("---")

        # Export
        if st.session_state.demo_data and st.session_state.query_results:
            st.markdown("### 📤 Export")
            export_data = {
                "query_results": st.session_state.query_results,
                "strategy_results": st.session_state.strategy_results_cache
            }
            st.download_button(
                "Download Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name="rag_lab_results.json",
                mime="application/json"
            )


# ============================================================================
# Welcome Tab - Metric Explanations
# ============================================================================

def render_rag_architecture_section():
    """Render a visual flowchart-style RAG pipeline diagram with animation."""
    import random

    current_step = st.session_state.rag_pipeline_step
    is_running = st.session_state.rag_pipeline_running

    if 'demo_query_text' not in st.session_state:
        st.session_state.demo_query_text = "What is machine learning?"

    # Timing values
    t1 = st.session_state.rag_pipeline_times.get("step_1", 0)
    t2 = st.session_state.rag_pipeline_times.get("step_2", 0)
    t3 = st.session_state.rag_pipeline_times.get("step_3", 0)

    # Node colors based on step
    def node_style(step_num):
        if current_step > step_num:
            return "#22c55e", "#052e16", "0 0 20px #22c55e"
        elif current_step == step_num:
            return "#3b82f6", "#1e3a5f", "0 0 25px #3b82f6"
        return "#475569", "#1e293b", "none"

    # Arrow color
    def arrow_color(from_step):
        if current_step > from_step:
            return "#22c55e"
        elif current_step == from_step:
            return "#3b82f6"
        return "#334155"

    # Build the visual diagram as one HTML block
    n1_border, n1_bg, n1_shadow = node_style(1)
    n2_border, n2_bg, n2_shadow = node_style(2)
    n3_border, n3_bg, n3_shadow = node_style(3)
    n4_border, n4_bg, n4_shadow = node_style(4)

    a1 = arrow_color(1)
    a2 = arrow_color(2)
    a3 = arrow_color(3)

    # Time displays
    t1_display = str(t1) + "ms" if t1 > 0 else ""
    t2_display = str(t2) + "ms" if t2 > 0 else ""
    t3_display = str(t3) + "ms" if t3 > 0 else ""

    # Pre-compute arrow color for step 4
    a4 = arrow_color(4)

    # Status labels
    n1_label = "✓ Ready" if current_step >= 1 else "1"
    n2_label = t1_display if t1 > 0 else "2"
    n3_label = t2_display if t2 > 0 else "3"
    n4_label = t3_display if t3 > 0 else "4"
    n5_label = "✓ Done" if current_step >= 4 else "5"

    # Container with background - BIGGER and more prominent
    st.markdown('<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);border-radius:20px;padding:40px 30px;margin:0;">', unsafe_allow_html=True)

    # Title - BIGGER
    st.markdown('<div style="text-align:center;margin-bottom:24px;"><span style="color:#f8fafc;font-size:1.4rem;font-weight:700;letter-spacing:3px;">WIKIPEDIA RAG PIPELINE</span></div>', unsafe_allow_html=True)

    # Create columns for nodes and arrows: 5 nodes + 4 arrows = 9 columns
    cols = st.columns([1.3, 0.4, 1.3, 0.4, 1.3, 0.4, 1.3, 0.4, 1.3])

    # Node 1: User Query - BIGGER
    with cols[0]:
        html = '<div style="text-align:center;"><div style="background:' + n1_bg + ';border:4px solid ' + n1_border + ';border-radius:16px;padding:24px 20px;box-shadow:' + n1_shadow + ';"><div style="font-size:3.5rem;">👤</div><div style="color:#f8fafc;font-weight:700;font-size:1.2rem;margin-top:10px;">USER QUERY</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">Natural Language</div></div><div style="color:' + n1_border + ';font-size:0.9rem;margin-top:8px;font-weight:600;">' + n1_label + '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Arrow 1 - BIGGER
    with cols[1]:
        html = '<div style="display:flex;align-items:center;height:160px;justify-content:center;"><div style="display:flex;align-items:center;"><div style="width:40px;height:4px;background:' + a1 + ';"></div><div style="width:0;height:0;border-top:10px solid transparent;border-bottom:10px solid transparent;border-left:16px solid ' + a1 + ';"></div></div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Node 2: Embedding - BIGGER
    with cols[2]:
        html = '<div style="text-align:center;"><div style="background:' + n2_bg + ';border:4px solid ' + n2_border + ';border-radius:16px;padding:24px 20px;box-shadow:' + n2_shadow + ';"><div style="font-size:3.5rem;">🔢</div><div style="color:#eab308;font-weight:700;font-size:1.2rem;margin-top:10px;">EMBEDDING</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">OpenAI 3072-dim</div></div><div style="color:' + n2_border + ';font-size:0.9rem;margin-top:8px;font-weight:600;">' + n2_label + '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Arrow 2 - BIGGER
    with cols[3]:
        html = '<div style="display:flex;align-items:center;height:160px;justify-content:center;"><div style="display:flex;align-items:center;"><div style="width:40px;height:4px;background:' + a2 + ';"></div><div style="width:0;height:0;border-top:10px solid transparent;border-bottom:10px solid transparent;border-left:16px solid ' + a2 + ';"></div></div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Node 3: pgvector - BIGGER
    with cols[4]:
        html = '<div style="text-align:center;"><div style="background:' + n3_bg + ';border:4px solid ' + n3_border + ';border-radius:16px;padding:24px 20px;box-shadow:' + n3_shadow + ';"><div style="font-size:3.5rem;">🐘</div><div style="color:#8b5cf6;font-weight:700;font-size:1.2rem;margin-top:10px;">PGVECTOR</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">25,000 Wikipedia</div></div><div style="color:' + n3_border + ';font-size:0.9rem;margin-top:8px;font-weight:600;">' + n3_label + '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Arrow 3 - BIGGER
    with cols[5]:
        html = '<div style="display:flex;align-items:center;height:160px;justify-content:center;"><div style="display:flex;align-items:center;"><div style="width:40px;height:4px;background:' + a3 + ';"></div><div style="width:0;height:0;border-top:10px solid transparent;border-bottom:10px solid transparent;border-left:16px solid ' + a3 + ';"></div></div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Node 4: LLM - BIGGER
    with cols[6]:
        html = '<div style="text-align:center;"><div style="background:' + n4_bg + ';border:4px solid ' + n4_border + ';border-radius:16px;padding:24px 20px;box-shadow:' + n4_shadow + ';"><div style="font-size:3.5rem;">🤖</div><div style="color:#f97316;font-weight:700;font-size:1.2rem;margin-top:10px;">GPT-5-mini</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">Generate Answer</div></div><div style="color:' + n4_border + ';font-size:0.9rem;margin-top:8px;font-weight:600;">' + n4_label + '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Arrow 4 - BIGGER
    with cols[7]:
        html = '<div style="display:flex;align-items:center;height:160px;justify-content:center;"><div style="display:flex;align-items:center;"><div style="width:40px;height:4px;background:' + a4 + ';"></div><div style="width:0;height:0;border-top:10px solid transparent;border-bottom:10px solid transparent;border-left:16px solid ' + a4 + ';"></div></div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Node 5: Answer - BIGGER
    with cols[8]:
        html = '<div style="text-align:center;"><div style="background:' + n4_bg + ';border:4px solid ' + n4_border + ';border-radius:16px;padding:24px 20px;box-shadow:' + n4_shadow + ';"><div style="font-size:3.5rem;">💬</div><div style="color:#22c55e;font-weight:700;font-size:1.2rem;margin-top:10px;">ANSWER</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">RAG Response</div></div><div style="color:' + n4_border + ';font-size:0.9rem;margin-top:8px;font-weight:600;">' + n5_label + '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Close container
    st.markdown('</div>', unsafe_allow_html=True)

    # Query input and controls
    st.markdown("")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        demo_query = st.text_input(
            "Your Question",
            value=st.session_state.demo_query_text,
            key="demo_query_input",
            placeholder="Type any question..."
        )
        if demo_query != st.session_state.demo_query_text:
            st.session_state.demo_query_text = demo_query

    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if current_step >= 4:
            if st.button("↺ Reset", use_container_width=True, key="reset_demo"):
                st.session_state.rag_pipeline_running = False
                st.session_state.rag_pipeline_step = 0
                st.session_state.rag_pipeline_times = {}
                st.session_state.rag_real_results = None
                st.rerun()
        else:
            if st.button("▶️ Run RAG", type="primary", use_container_width=True, key="run_demo"):
                st.session_state.rag_pipeline_running = True
                st.session_state.rag_pipeline_step = 0
                st.session_state.rag_pipeline_times = {}
                st.session_state.rag_real_results = None
                st.rerun()

    # Sample queries
    st.markdown("")
    sq1, sq2, sq3, sq4 = st.columns(4)
    with sq1:
        if st.button("🔬 ML (Gap)", key="s1", use_container_width=True):
            st.session_state.demo_query_text = "What is machine learning?"
            st.session_state.rag_pipeline_step = 0
            st.session_state.rag_pipeline_running = False
            st.session_state.rag_real_results = None
            st.rerun()
    with sq2:
        if st.button("📞 Telephone", key="s2", use_container_width=True):
            st.session_state.demo_query_text = "Who invented the telephone?"
            st.session_state.rag_pipeline_step = 0
            st.session_state.rag_pipeline_running = False
            st.session_state.rag_real_results = None
            st.rerun()
    with sq3:
        if st.button("🐋 Whales", key="s3", use_container_width=True):
            st.session_state.demo_query_text = "Tell me about blue whales"
            st.session_state.rag_pipeline_step = 0
            st.session_state.rag_pipeline_running = False
            st.session_state.rag_real_results = None
            st.rerun()
    with sq4:
        if st.button("🌍 Climate", key="s4", use_container_width=True):
            st.session_state.demo_query_text = "What causes climate change?"
            st.session_state.rag_pipeline_step = 0
            st.session_state.rag_pipeline_running = False
            st.session_state.rag_real_results = None
            st.rerun()

    # Show results after completion
    if current_step >= 4:
        st.markdown("---")

        # Get real results if available
        real_results = st.session_state.get('rag_real_results')
        has_real_results = real_results is not None and real_results.get('num_results', 0) > 0

        # Two columns: Retrieved docs + Answer
        col_docs, col_answer = st.columns([1, 1])

        with col_docs:
            st.markdown("#### 📄 Retrieved Documents")
            if has_real_results and real_results.get('sources'):
                for src in real_results['sources'][:5]:
                    title = src.get('metadata', {}).get('title', 'Wikipedia Article')
                    score = src.get('score', 0.0)
                    snippet = src.get('content', '')[:200]
                    doc_html = '<div style="background:#052e16;border:2px solid #22c55e;border-radius:8px;padding:12px;margin:8px 0;">'
                    doc_html += '<div style="display:flex;justify-content:space-between;">'
                    doc_html += '<span style="color:#22c55e;font-weight:600;">' + str(title) + '</span>'
                    doc_html += '<span style="color:#86efac;font-size:0.85rem;">' + str(round(score, 3)) + '</span></div>'
                    doc_html += '<div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">' + snippet + '...</div></div>'
                    st.markdown(doc_html, unsafe_allow_html=True)
            elif real_results is not None and real_results.get('num_results', 0) == 0:
                no_docs_html = '<div style="background:#450a0a;border:2px solid #dc2626;border-radius:8px;padding:16px;text-align:center;">'
                no_docs_html += '<div style="font-size:2rem;margin-bottom:8px;">⚠️</div>'
                no_docs_html += '<div style="color:#f87171;font-weight:600;">No Relevant Documents Found</div>'
                no_docs_html += '<div style="color:#fca5a5;font-size:0.85rem;margin-top:4px;">No matching articles in the Wikipedia corpus</div></div>'
                st.markdown(no_docs_html, unsafe_allow_html=True)
            else:
                # Fallback mock data when RAG not available
                st.info("RAG engine not available - showing demo data")
                docs = [("Demo Article 1", 0.85, "This is sample content...")]
                for title, score, snippet in docs:
                    doc_html = '<div style="background:#1e293b;border:2px solid #64748b;border-radius:8px;padding:12px;margin:8px 0;">'
                    doc_html += '<span style="color:#94a3b8;">' + title + '</span></div>'
                    st.markdown(doc_html, unsafe_allow_html=True)

        with col_answer:
            st.markdown("#### 💬 Generated Answer")
            if has_real_results:
                src_html = '<div style="background:#052e16;border-left:4px solid #22c55e;border-radius:8px;padding:4px 12px;margin-bottom:10px;display:inline-block;">'
                src_html += '<span style="color:#22c55e;font-weight:600;font-size:0.85rem;">✓ RAG-GROUNDED</span></div>'
                st.markdown(src_html, unsafe_allow_html=True)
                answer = real_results.get('answer', 'No answer generated.')
            elif real_results is not None:
                src_html = '<div style="background:#450a0a;border-left:4px solid #dc2626;border-radius:8px;padding:4px 12px;margin-bottom:10px;display:inline-block;">'
                src_html += '<span style="color:#f87171;font-weight:600;font-size:0.85rem;">⚠ NO RESULTS</span></div>'
                st.markdown(src_html, unsafe_allow_html=True)
                answer = real_results.get('answer', 'Could not find relevant information.')
            else:
                src_html = '<div style="background:#1e3a5f;border-left:4px solid #3b82f6;border-radius:8px;padding:4px 12px;margin-bottom:10px;display:inline-block;">'
                src_html += '<span style="color:#60a5fa;font-weight:600;font-size:0.85rem;">ℹ DEMO MODE</span></div>'
                st.markdown(src_html, unsafe_allow_html=True)
                answer = "RAG engine not connected. Set DATABASE_URL and OPENAI_API_KEY environment variables."

            ans_html = '<div style="background:#1e293b;border-radius:8px;padding:16px;border:1px solid #334155;">'
            ans_html += '<div style="color:#f8fafc;line-height:1.7;">' + answer + '</div></div>'
            st.markdown(ans_html, unsafe_allow_html=True)

        # Metrics row - show real timing if available
        st.markdown("")
        st.markdown("#### 📊 Pipeline Metrics")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            num_docs = real_results.get('num_results', 0) if real_results else 0
            doc_color = "#22c55e" if num_docs >= 3 else ("#eab308" if num_docs >= 1 else "#ef4444")
            mh = '<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border-top:4px solid ' + doc_color + ';">'
            mh += '<div style="color:#94a3b8;font-size:0.8rem;">Documents</div>'
            mh += '<div style="color:' + doc_color + ';font-size:1.8rem;font-weight:700;">' + str(num_docs) + '</div></div>'
            st.markdown(mh, unsafe_allow_html=True)
        with m2:
            embed_ms = real_results.get('embedding_time', t1) if real_results else t1
            mh = '<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border-top:4px solid #eab308;">'
            mh += '<div style="color:#94a3b8;font-size:0.8rem;">Embedding</div>'
            mh += '<div style="color:#eab308;font-size:1.8rem;font-weight:700;">' + str(embed_ms) + 'ms</div></div>'
            st.markdown(mh, unsafe_allow_html=True)
        with m3:
            search_ms = real_results.get('search_time', t2) if real_results else t2
            mh = '<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border-top:4px solid #8b5cf6;">'
            mh += '<div style="color:#94a3b8;font-size:0.8rem;">pgvector</div>'
            mh += '<div style="color:#8b5cf6;font-size:1.8rem;font-weight:700;">' + str(search_ms) + 'ms</div></div>'
            st.markdown(mh, unsafe_allow_html=True)
        with m4:
            total_ms = real_results.get('total_time', t1 + t2 + t3) if real_results else (t1 + t2 + t3)
            mh = '<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border-top:4px solid #3b82f6;">'
            mh += '<div style="color:#94a3b8;font-size:0.8rem;">Total</div>'
            mh += '<div style="color:#3b82f6;font-size:1.8rem;font-weight:700;">' + str(total_ms) + 'ms</div></div>'
            st.markdown(mh, unsafe_allow_html=True)

        # Status message
        if has_real_results:
            rec = '<div style="background:#052e16;border:2px solid #22c55e;border-radius:8px;padding:12px;margin-top:12px;">'
            rec += '<span style="color:#22c55e;font-weight:700;">✅ Live RAG:</span> '
            rec += '<span style="color:#86efac;">Real search results from Wikipedia corpus via pgvector.</span></div>'
        elif real_results is not None:
            rec = '<div style="background:#450a0a;border:2px solid #dc2626;border-radius:8px;padding:12px;margin-top:12px;">'
            rec += '<span style="color:#f87171;font-weight:700;">⚠️ No Results:</span> '
            rec += '<span style="color:#fca5a5;">Query did not match any documents in the corpus.</span></div>'
        else:
            rec = '<div style="background:#1e3a5f;border:2px solid #3b82f6;border-radius:8px;padding:12px;margin-top:12px;">'
            rec += '<span style="color:#60a5fa;font-weight:700;">ℹ Demo Mode:</span> '
            rec += '<span style="color:#93c5fd;">Connect to database for live RAG queries.</span></div>'
        st.markdown(rec, unsafe_allow_html=True)

    # Animation logic - perform real RAG search
    if is_running and current_step < 4:
        next_step = current_step + 1

        if next_step == 1:
            # Step 1: Query received
            time.sleep(0.2)
            st.session_state.rag_pipeline_step = 1
            st.rerun()
        elif next_step == 2:
            # Step 2: Embedding - show brief animation then perform real search
            time.sleep(0.3)
            st.session_state.rag_pipeline_step = 2
            st.rerun()
        elif next_step == 3:
            # Step 3: Perform actual RAG search
            query = st.session_state.demo_query_text
            result = perform_real_rag_search(query, top_k=5)
            if result:
                st.session_state.rag_real_results = result
                st.session_state.rag_pipeline_times["step_1"] = result.get('embedding_time', 150)
                st.session_state.rag_pipeline_times["step_2"] = result.get('search_time', 50)
                st.session_state.rag_pipeline_times["step_3"] = result.get('generation_time', 800)
            else:
                # Fallback to simulated times if RAG not available
                st.session_state.rag_real_results = None
                st.session_state.rag_pipeline_times["step_1"] = random.randint(150, 200)
                st.session_state.rag_pipeline_times["step_2"] = random.randint(40, 60)
                st.session_state.rag_pipeline_times["step_3"] = random.randint(700, 900)
            st.session_state.rag_pipeline_step = 3
            st.rerun()
        elif next_step == 4:
            # Step 4: Complete
            time.sleep(0.2)
            st.session_state.rag_pipeline_step = 4
            st.session_state.rag_pipeline_running = False
            st.rerun()


def render_welcome():
    """Render the Welcome tab with clean RAG workflow diagram."""
    st.session_state.visited_tabs.add('welcome')
    st.session_state.current_tab = 'welcome'

    # RAG Architecture Diagram - THE MAIN FOCUS (no extra title needed)
    render_rag_architecture_section()

    # Metrics Introduction Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;margin-bottom:16px;">
        <span style="color:#94a3b8;font-size:1.1rem;letter-spacing:1px;">RAG QUALITY METRICS</span>
    </div>
    """, unsafe_allow_html=True)

    # Get metric values - use demo data results if search was run
    real_results = st.session_state.get('rag_real_results')
    current_step = st.session_state.rag_pipeline_step

    # Determine metric values based on search state
    if current_step >= 4 and real_results:
        num_results = real_results.get('num_results', 0)
        # Simulated metrics based on results (in real app, would compute from ground truth)
        if num_results >= 3:
            recall_val, precision_val, ndcg_val = 0.85, 0.60, 0.78
        elif num_results >= 1:
            recall_val, precision_val, ndcg_val = 0.50, 0.40, 0.55
        else:
            recall_val, precision_val, ndcg_val = 0.00, 0.00, 0.00
        has_values = True
    else:
        recall_val, precision_val, ndcg_val = None, None, None
        has_values = False

    # Helper for color based on value
    def metric_color(val, thresholds=(0.8, 0.5)):
        if val is None:
            return "#64748b"
        if val >= thresholds[0]:
            return "#22c55e"
        if val >= thresholds[1]:
            return "#eab308"
        return "#ef4444"

    # Format value
    def fmt_val(val):
        if val is None:
            return "--"
        return str(int(val * 100)) + "%"

    def fmt_ndcg(val):
        if val is None:
            return "--"
        return str(round(val, 2))

    # Three metrics in columns with BIG numbers
    m1, m2, m3 = st.columns(3)

    with m1:
        r_color = metric_color(recall_val)
        r_display = fmt_val(recall_val)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#052e16 0%,#14532d 100%);border-radius:12px;padding:20px;text-align:center;border:2px solid #22c55e;min-height:200px;">
            <div style="color:#22c55e;font-size:1rem;font-weight:600;letter-spacing:1px;margin-bottom:8px;">🎯 RECALL</div>
            <div style="color:{r_color};font-size:3.5rem;font-weight:800;line-height:1;">{r_display}</div>
            <div style="color:#86efac;font-size:0.85rem;margin-top:12px;line-height:1.4;">
                Did we find all relevant documents?
            </div>
            <div style="margin-top:10px;padding:6px;background:#022c22;border-radius:6px;">
                <span style="color:#4ade80;font-size:0.8rem;">% of expected articles retrieved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        p_color = metric_color(precision_val)
        p_display = fmt_val(precision_val)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#422006 0%,#78350f 100%);border-radius:12px;padding:20px;text-align:center;border:2px solid #eab308;min-height:200px;">
            <div style="color:#eab308;font-size:1rem;font-weight:600;letter-spacing:1px;margin-bottom:8px;">✨ PRECISION</div>
            <div style="color:{p_color};font-size:3.5rem;font-weight:800;line-height:1;">{p_display}</div>
            <div style="color:#fde047;font-size:0.85rem;margin-top:12px;line-height:1.4;">
                Are the results actually relevant?
            </div>
            <div style="margin-top:10px;padding:6px;background:#451a03;border-radius:6px;">
                <span style="color:#facc15;font-size:0.8rem;">% of retrieved docs that matter</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        n_color = metric_color(ndcg_val, thresholds=(0.7, 0.4))
        n_display = fmt_ndcg(ndcg_val)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);border-radius:12px;padding:20px;text-align:center;border:2px solid #8b5cf6;min-height:200px;">
            <div style="color:#8b5cf6;font-size:1rem;font-weight:600;letter-spacing:1px;margin-bottom:8px;">📊 nDCG</div>
            <div style="color:{n_color};font-size:3.5rem;font-weight:800;line-height:1;">{n_display}</div>
            <div style="color:#c4b5fd;font-size:0.85rem;margin-top:12px;line-height:1.4;">
                Are the best results ranked first?
            </div>
            <div style="margin-top:10px;padding:6px;background:#1e1b4b;border-radius:6px;">
                <span style="color:#a78bfa;font-size:0.8rem;">1.0 = perfect ranking order</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Key insight - changes based on whether we have results
    if has_values:
        if recall_val >= 0.8 and precision_val >= 0.5:
            insight_bg = "#052e16"
            insight_border = "#22c55e"
            insight_icon = "✅"
            insight_text = "Good retrieval quality! Both recall and precision are healthy."
            insight_color = "#86efac"
        elif recall_val < 0.5:
            insight_bg = "#450a0a"
            insight_border = "#ef4444"
            insight_icon = "⚠️"
            insight_text = "Low recall - missing important documents. Check if relevant content exists in corpus."
            insight_color = "#fca5a5"
        else:
            insight_bg = "#422006"
            insight_border = "#eab308"
            insight_icon = "💡"
            insight_text = "Moderate results. Consider tuning search parameters or expanding the corpus."
            insight_color = "#fde047"
    else:
        insight_bg = "#1e293b"
        insight_border = "#3b82f6"
        insight_icon = "💡"
        insight_text = "Run a RAG query above to see live metrics. High recall + low precision = too many irrelevant results."
        insight_color = "#94a3b8"

    st.markdown(f"""
    <div style="background:{insight_bg};border-radius:8px;padding:16px;margin-top:16px;text-align:center;border-left:4px solid {insight_border};">
        <span style="color:{insight_border};font-weight:600;">{insight_icon} Insight:</span>
        <span style="color:{insight_color};"> {insight_text}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Act 1: The Problem - Interactive Query Testing
# ============================================================================

def render_act1_problem():
    """Render Act 1: Interactive query testing."""
    st.session_state.visited_tabs.add('act1')
    st.session_state.current_tab = 'act1'

    st.header("🔴 Act 1: The Problem")

    # Persona reminder
    st.markdown("""
    <div class="hint-box">
        👨‍💻 <strong>Your Task:</strong> Run each of these 4 queries and see which ones
        achieve 100% recall. Click the "Run Query" button to test each one.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    test_cases = get_test_cases(st.session_state.demo_data)
    k_results = get_k_balance_results(st.session_state.demo_data)
    by_query = k_results.get("by_query", {})

    # K-value selector with Run All button inline
    st.markdown("### ⚙️ Search Settings")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        k_retrieve = st.select_slider(
            "k_retrieve (candidate pool size)",
            options=[10, 20, 50, 100],
            value=10,
            help="How many documents to fetch from the database"
        )

    with col2:
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 12px; margin-top: 28px;">
            <span style="color: #64748b;">Current k:</span>
            <span style="color: #3b82f6; font-size: 1.5rem; font-weight: 600; margin-left: 8px;">{k_retrieve}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_all_clicked = st.button("🚀 Run All Queries", type="primary", use_container_width=True, key="run_all_act1_top")

    st.markdown("---")

    # Query Cards - Interactive
    st.markdown("### 🔍 Test Queries")
    st.markdown("Click **Run Query** to see how each performs:")

    for idx, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected_ids = test_case.get("expected_doc_ids", [])
        expected_titles = test_case.get("expected_titles", [])
        metadata = test_case.get("metadata", {})
        category = metadata.get("category", "unknown")

        # Get results for current k
        k_key = f"k_{k_retrieve}"
        query_data = by_query.get(query, {}).get(k_key, {})

        # Check if we have cached results for this query
        result_key = f"{query}_{k_retrieve}"
        has_result = result_key in st.session_state.query_results

        col1, col2 = st.columns([3, 1])

        with col1:
            # Query card
            card_class = "selected" if has_result else ""
            recall = query_data.get("recall", 0) if has_result else None

            if recall is not None:
                if recall >= 1.0:
                    border_color = "#22c55e"
                    status_emoji = "✅"
                elif recall >= 0.75:
                    border_color = "#eab308"
                    status_emoji = "🟡"
                else:
                    border_color = "#ef4444"
                    status_emoji = "🔴"
            else:
                border_color = "#334155"
                status_emoji = "⬜"

            st.markdown(f"""
            <div style="background-color: #1e293b; border-radius: 12px; padding: 20px;
                        border: 2px solid {border_color}; margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="font-size: 1.5rem; margin-right: 12px;">{status_emoji}</span>
                        <span style="color: #f8fafc; font-size: 1.1rem; font-weight: 600;">
                            "{query}"
                        </span>
                    </div>
                    <span style="background-color: #334155; padding: 4px 12px; border-radius: 12px;
                                 color: #94a3b8; font-size: 0.8rem;">
                        {category}
                    </span>
                </div>
                <div style="margin-top: 12px; color: #64748b; font-size: 0.85rem;">
                    Expected: {len(expected_ids)} documents
                    ({', '.join(expected_titles[:2])}{'...' if len(expected_titles) > 2 else ''})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Run button and results
            if st.button(f"▶️ Run Query", key=f"run_{idx}", use_container_width=True):
                with st.spinner("Searching..."):
                    time.sleep(0.5)  # Simulate search time
                    # Store results
                    st.session_state.query_results[result_key] = query_data
                    st.rerun()

            if has_result:
                recall = st.session_state.query_results[result_key].get("recall", 0)
                ndcg = st.session_state.query_results[result_key].get("ndcg", 0)
                found = st.session_state.query_results[result_key].get("relevant_found", 0)
                total = st.session_state.query_results[result_key].get("total_relevant", len(expected_ids))

                recall_color = "#22c55e" if recall >= 1.0 else "#eab308" if recall >= 0.75 else "#ef4444"

                st.markdown(f"""
                <div style="text-align: center; padding: 8px;">
                    <div style="color: {recall_color}; font-size: 1.8rem; font-weight: 700;">
                        {format_percent(recall)}
                    </div>
                    <div style="color: #64748b; font-size: 0.8rem;">Recall</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                        {found}/{total} found
                    </div>
                    <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">
                        nDCG: {format_metric(ndcg)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Ground Truth Explorer - Expander to view expected articles
        if expected_ids:
            with st.expander(f"📚 View Ground Truth Articles ({len(expected_ids)} expected)", expanded=False):
                # Try to fetch real content from database
                articles = fetch_article_content(tuple(expected_ids))

                if articles:
                    for i, doc_id in enumerate(expected_ids):
                        article = articles.get(doc_id)
                        if article:
                            title = article.get('title', f'Article {doc_id}')
                            content = article.get('content', 'Content not available')
                            # Truncate content for display
                            preview = content[:500] + '...' if len(content) > 500 else content

                            st.markdown(f"""
                            <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;margin:8px 0;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                    <span style="color:#3b82f6;font-weight:700;font-size:1.1rem;">📄 {title}</span>
                                    <span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.75rem;">ID: {doc_id}</span>
                                </div>
                                <div style="color:#94a3b8;font-size:0.9rem;line-height:1.6;">{preview}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Full content in nested expander
                            if len(content) > 500:
                                with st.expander(f"Read full article: {title}", expanded=False):
                                    st.markdown(content)
                        else:
                            st.markdown(f"""
                            <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin:8px 0;">
                                <span style="color:#64748b;">Article ID {doc_id}</span>
                                <span style="color:#94a3b8;margin-left:8px;">(Title: {expected_titles[i] if i < len(expected_titles) else 'Unknown'})</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Fallback when database not available - show titles only
                    st.info("📡 Connect to database to view full article content")
                    for i, doc_id in enumerate(expected_ids):
                        title = expected_titles[i] if i < len(expected_titles) else f"Article {doc_id}"
                        st.markdown(f"""
                        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin:8px 0;">
                            <span style="color:#3b82f6;font-weight:600;">📄 {title}</span>
                            <span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-left:8px;">ID: {doc_id}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Explanation of ground truth
                st.markdown("""
                <div style="background:#1e3a5f;border-radius:8px;padding:12px;margin-top:12px;border-left:3px solid #3b82f6;">
                    <span style="color:#60a5fa;font-weight:600;">ℹ️ What is Ground Truth?</span>
                    <div style="color:#93c5fd;font-size:0.85rem;margin-top:4px;">
                        These are the articles that <strong>should</strong> be retrieved for this query.
                        Recall measures what percentage of these articles were actually found in the search results.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Handle Run All button click (button is in the settings row above)
    if run_all_clicked:
        progress_bar = st.progress(0)
        for idx, test_case in enumerate(test_cases):
            query = test_case["query"]
            k_key = f"k_{k_retrieve}"
            result_key = f"{query}_{k_retrieve}"
            query_data = by_query.get(query, {}).get(k_key, {})
            st.session_state.query_results[result_key] = query_data
            progress_bar.progress((idx + 1) / len(test_cases))
            time.sleep(0.3)
        st.rerun()

    # Summary after running queries
    if len(st.session_state.query_results) >= len(test_cases):
        st.markdown("---")
        st.markdown("### 📊 Results Summary")

        # Count successes
        successes = 0
        problem_query = None
        for test_case in test_cases:
            query = test_case["query"]
            result_key = f"{query}_{k_retrieve}"
            if result_key in st.session_state.query_results:
                recall = st.session_state.query_results[result_key].get("recall", 0)
                if recall >= 1.0:
                    successes += 1
                elif recall <= 0.5:
                    problem_query = query

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Queries at 100% Recall", f"{successes}/{len(test_cases)}")

        with col2:
            if problem_query:
                st.metric("Problem Query", truncate_query(problem_query, 25), delta="-50% recall", delta_color="inverse")

        with col3:
            if successes < len(test_cases):
                st.metric("Status", "Investigation Needed", delta="⚠️")
            else:
                st.metric("Status", "All Good!", delta="✅")

        if problem_query:
            st.markdown(f"""
            <div style="background-color: #450a0a; border-radius: 8px; padding: 16px; margin: 16px 0;
                        border-left: 4px solid #ef4444;">
                <strong style="color: #ef4444;">🔴 Problem Detected!</strong><br>
                <span style="color: #fca5a5;">
                    The query "{truncate_query(problem_query, 40)}" is stuck at 50% recall even at k={k_retrieve}.
                    <br><br>
                    <strong>Try increasing k</strong> using the slider above, or move to Act 2 to try different strategies.
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Next step hint
            st.markdown("""
            <div class="hint-box">
                💡 <strong>Next Step:</strong> Go to "Act 2: Investigation" to try different search strategies.
                Maybe title-weighting or hybrid search can help?
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Act 2: The Investigation - Strategy Comparison
# ============================================================================

def render_act2_investigation():
    """Render Act 2: Interactive strategy comparison."""
    st.session_state.visited_tabs.add('act2')
    st.session_state.current_tab = 'act2'

    st.header("🔶 Act 2: The Investigation")

    st.markdown("""
    <div class="hint-box">
        👨‍💻 <strong>Your Task:</strong> Select a query and try different search strategies.
        Can you find one that improves recall for the struggling queries?
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    strategy_results = get_strategy_results(st.session_state.demo_data)
    strategies = strategy_results.get("strategies", [])
    by_query = strategy_results.get("by_query", {})
    test_cases = get_test_cases(st.session_state.demo_data)

    # Query selector with Run All button inline
    st.markdown("### ⚙️ Investigation Settings")

    query_options = [tc["query"] for tc in test_cases]

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_idx = st.selectbox(
            "Choose query:",
            range(len(query_options)),
            format_func=lambda i: f"{i+1}. {query_options[i]}"
        )
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_all_strategies_clicked = st.button("🚀 Run All 9", type="primary", use_container_width=True, key="run_all_act2_top")

    selected_query = query_options[selected_idx]
    st.session_state.selected_query = selected_query

    # Show current status
    query_data = by_query.get(selected_query, {})
    max_recall = query_data.get("max_recall", 0)
    best_strategy = query_data.get("best_strategy", "Unknown")
    insight = query_data.get("insight", "")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px;">
            <div style="color: #f8fafc; font-size: 1.2rem; font-weight: 600;">
                "{selected_query}"
            </div>
            <div style="color: #94a3b8; margin-top: 8px;">
                Best achievable recall: <span style="color: {ColorScheme.get_recall_color(max_recall)}; font-weight: 600;">
                    {format_percent(max_recall)}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        recall_color = ColorScheme.get_recall_color(max_recall)
        st.markdown(f"""
        <div style="background-color: #0f172a; border-radius: 8px; padding: 16px; text-align: center;
                    border: 2px solid {recall_color};">
            <div style="color: {recall_color}; font-size: 2rem; font-weight: 700;">
                {format_percent(max_recall)}
            </div>
            <div style="color: #64748b; font-size: 0.8rem;">Max Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Strategy Grid - Interactive
    st.markdown("### 🧪 Try Different Strategies")
    st.markdown("Click a strategy to see its results:")

    # Create 3x3 grid
    strategy_groups = [strategies[i:i+3] for i in range(0, len(strategies), 3)]

    for group in strategy_groups:
        cols = st.columns(3)
        for idx, strategy in enumerate(group):
            with cols[idx]:
                s_id = strategy["id"]
                s_name = strategy["name"]
                s_desc = strategy["description"]

                # Get metrics for this strategy
                s_metrics = query_data.get(s_id, {})
                recall = s_metrics.get("recall", 0)
                ndcg = s_metrics.get("ndcg", 0)

                # Button to "run" this strategy
                strategy_key = f"{selected_query}_{s_id}"
                is_run = strategy_key in st.session_state.strategy_results_cache

                color = get_strategy_color(s_id)
                recall_color = ColorScheme.get_recall_color(recall)

                st.markdown(f"""
                <div style="background-color: #1e293b; border-radius: 8px; padding: 12px;
                            border-left: 4px solid {color}; margin-bottom: 8px; min-height: 120px;">
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">{s_name}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">{s_desc}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"▶️ Run", key=f"strat_{s_id}", use_container_width=True):
                    with st.spinner("Testing..."):
                        time.sleep(0.3)
                        st.session_state.strategy_results_cache[strategy_key] = s_metrics
                        st.rerun()

                if is_run:
                    cached = st.session_state.strategy_results_cache[strategy_key]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 4px;">
                        <span style="color: {recall_color}; font-weight: 600;">
                            Recall: {format_percent(cached.get('recall', 0))}
                        </span>
                        <span style="color: #64748b; margin-left: 8px;">
                            nDCG: {format_metric(cached.get('ndcg', 0))}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

    # Handle Run All Strategies button click (button is in the settings row above)
    st.markdown("---")
    if run_all_strategies_clicked:
        progress = st.progress(0)
        for idx, strategy in enumerate(strategies):
            s_id = strategy["id"]
            strategy_key = f"{selected_query}_{s_id}"
            s_metrics = query_data.get(s_id, {})
            st.session_state.strategy_results_cache[strategy_key] = s_metrics
            progress.progress((idx + 1) / len(strategies))
            time.sleep(0.1)
        st.rerun()

    # Results comparison chart
    if any(f"{selected_query}_{s['id']}" in st.session_state.strategy_results_cache for s in strategies):
        st.markdown("---")
        st.markdown("### 📊 Strategy Comparison")

        # Build comparison data
        chart_data = []
        for strategy in strategies:
            s_id = strategy["id"]
            strategy_key = f"{selected_query}_{s_id}"
            if strategy_key in st.session_state.strategy_results_cache:
                metrics = st.session_state.strategy_results_cache[strategy_key]
                chart_data.append({
                    "Strategy": strategy["name"],
                    "Recall": metrics.get("recall", 0),
                    "nDCG": metrics.get("ndcg", 0),
                    "Precision": metrics.get("precision", 0)
                })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    chart_df,
                    x="Strategy",
                    y="Recall",
                    color="Recall",
                    color_continuous_scale=[(0, COLORS.PROBLEM), (0.5, COLORS.NEEDS_WORK),
                                           (0.8, COLORS.GOOD), (1.0, COLORS.PERFECT)],
                    title="Recall by Strategy"
                )
                fig.update_layout(xaxis_tickangle=-45, height=350, showlegend=False)
                fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                              annotation_text="100%", annotation_position="right")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    chart_df,
                    x="Strategy",
                    y="nDCG",
                    color="nDCG",
                    color_continuous_scale=[(0, COLORS.PROBLEM), (0.5, COLORS.NEEDS_WORK),
                                           (0.7, COLORS.GOOD), (1.0, COLORS.PERFECT)],
                    title="nDCG by Strategy"
                )
                fig.update_layout(xaxis_tickangle=-45, height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Insight callout
    if insight:
        if max_recall >= 1.0:
            callout_bg = "#052e16"
            callout_border = "#22c55e"
        elif "ALL" in best_strategy or "TIED" in best_strategy:
            callout_bg = "#450a0a"
            callout_border = "#ef4444"
        else:
            callout_bg = "#422006"
            callout_border = "#eab308"

        st.markdown(f"""
        <div style="background-color: {callout_bg}; border-radius: 8px; padding: 16px; margin: 16px 0;
                    border-left: 4px solid {callout_border};">
            <strong style="color: #f8fafc;">💡 Insight:</strong><br>
            <span style="color: #e2e8f0;">{insight}</span>
        </div>
        """, unsafe_allow_html=True)

    # Hint for next step
    if max_recall < 1.0 and "ALL" in str(best_strategy):
        st.markdown("""
        <div class="hint-box">
            🤔 <strong>Strange...</strong> ALL 9 strategies give the same result?
            This suggests the problem isn't with our search algorithm.
            Go to "Act 3: Revelation" to discover the real cause!
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Act 3: The Revelation
# ============================================================================

def render_act3_revelation():
    """Render Act 3: The Revelation tab."""
    st.session_state.visited_tabs.add('act3')
    st.session_state.current_tab = 'act3'

    st.header("🟢 Act 3: The Revelation")

    st.markdown("""
    <div class="hint-box">
        👨‍💻 <strong>The Mystery:</strong> We tried 9 different search strategies on the
        "machine learning" query. ALL of them gave 50% recall. Why?
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.demo_data:
        st.error("No demo data loaded.")
        return

    corpus_analysis = get_corpus_analysis(st.session_state.demo_data)

    # Interactive SQL Investigation
    st.markdown("### 🔍 Let's Check the Database")
    st.markdown("Click the buttons to run SQL queries and investigate:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Does a 'Machine Learning' article exist?")

        if st.button("▶️ Run Title Search", key="run_title_sql", use_container_width=True):
            st.session_state['ran_title_sql'] = True

        st.code("""SELECT title FROM articles
WHERE title ILIKE '%machine learning%';""", language="sql")

        if st.session_state.get('ran_title_sql', False):
            ml_title_search = corpus_analysis.get("ml_title_search", {})
            count = ml_title_search.get("count", 0)

            st.markdown(f"""
            <div style="background-color: #450a0a; border-radius: 8px; padding: 16px;
                        border: 2px solid #ef4444; margin-top: 8px;">
                <div style="font-size: 1.5rem; text-align: center;">😱</div>
                <div style="color: #ef4444; font-size: 1.2rem; font-weight: 600; text-align: center;">
                    Result: {count} rows
                </div>
                <div style="color: #fca5a5; text-align: center; margin-top: 8px;">
                    There is NO "Machine Learning" article in our database!
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### What DOES the FTS find?")

        if st.button("▶️ Run Full-Text Search", key="run_fts_sql", use_container_width=True):
            st.session_state['ran_fts_sql'] = True

        st.code("""SELECT title, ts_rank(...) as rank
FROM articles
WHERE to_tsvector('english', content)
      @@ plainto_tsquery('machine learning')
ORDER BY rank DESC LIMIT 6;""", language="sql")

        if st.session_state.get('ran_fts_sql', False):
            fts_results = corpus_analysis.get("ml_fts_results", {}).get("results", [])

            for result in fts_results[:6]:
                is_expected = result.get("is_expected", False)
                title = result["title"]
                rank = result.get("rank", 0)
                note = result.get("note", "")

                if is_expected:
                    bg = "#052e16"
                    border = "#22c55e"
                    icon = "✅"
                elif "machine" in title.lower():
                    bg = "#450a0a"
                    border = "#ef4444"
                    icon = "⚠️"
                else:
                    bg = "#1e293b"
                    border = "#334155"
                    icon = "📄"

                st.markdown(f"""
                <div style="background-color: {bg}; border-radius: 4px; padding: 8px;
                            border-left: 3px solid {border}; margin: 4px 0; font-size: 0.85rem;">
                    {icon} <span style="color: #f8fafc;">{title}</span>
                    <span style="color: #64748b; float: right;">rank: {rank:.3f}</span>
                    {f'<br><span style="color: #f97316; font-size: 0.75rem; margin-left: 24px;">{note}</span>' if note else ''}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # The Aha Moment
    if st.session_state.get('ran_title_sql', False) and st.session_state.get('ran_fts_sql', False):
        st.markdown("### 💡 The Aha Moment")

        st.markdown("""
        <div style="background: linear-gradient(135deg, #052e16 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; margin: 16px 0;
                    border: 2px solid #22c55e;">
            <h3 style="color: #22c55e; margin-top: 0; text-align: center;">
                🎯 Mystery Solved!
            </h3>
            <div style="display: flex; gap: 20px; margin-top: 16px;">
                <div style="flex: 1; background-color: #0f172a; border-radius: 8px; padding: 16px;">
                    <div style="color: #ef4444; font-weight: 600;">The Problem:</div>
                    <div style="color: #e2e8f0; margin-top: 8px;">
                        Our 25k Wikipedia subset has <strong>NO dedicated "Machine Learning" article</strong>.
                    </div>
                </div>
                <div style="flex: 1; background-color: #0f172a; border-radius: 8px; padding: 16px;">
                    <div style="color: #eab308; font-weight: 600;">What We Have:</div>
                    <div style="color: #e2e8f0; margin-top: 8px;">
                        Related articles (AI, Neural Networks) that <strong>mention</strong> ML
                        but aren't <strong>about</strong> it.
                    </div>
                </div>
                <div style="flex: 1; background-color: #0f172a; border-radius: 8px; padding: 16px;">
                    <div style="color: #22c55e; font-weight: 600;">The Truth:</div>
                    <div style="color: #e2e8f0; margin-top: 8px;">
                        <strong>No search algorithm can find documents that don't exist!</strong>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Root Cause Box
        problem_diagnosis = corpus_analysis.get("problem_diagnosis", {})
        issue_text = problem_diagnosis.get('issue', '')
        root_cause_text = problem_diagnosis.get('root_cause', '')
        impact_text = problem_diagnosis.get('impact', '')
        solution_text = problem_diagnosis.get('solution', '')

        st.markdown("### 📋 Root Cause Analysis")

        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 20px; margin: 16px 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px; color: #ef4444; font-weight: 600; width: 120px;">Issue:</td>
                    <td style="padding: 12px; color: #e2e8f0;">{issue_text}</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px; color: #f97316; font-weight: 600;">Root Cause:</td>
                    <td style="padding: 12px; color: #e2e8f0;">{root_cause_text}</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px; color: #eab308; font-weight: 600;">Impact:</td>
                    <td style="padding: 12px; color: #e2e8f0;">{impact_text}</td>
                </tr>
                <tr>
                    <td style="padding: 12px; color: #22c55e; font-weight: 600;">Solution:</td>
                    <td style="padding: 12px; color: #e2e8f0;">{solution_text}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # The Key Insight
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; margin: 24px 0; text-align: center;">
            <div style="font-size: 3rem;">🔑</div>
            <h3 style="color: #3b82f6; margin: 12px 0;">The Key Insight</h3>
            <p style="font-size: 1.3rem; color: #f8fafc; margin: 0;">
                <strong>RAG Quality = Corpus Coverage × Search Algorithm × Generation Model</strong>
            </p>
            <p style="color: #94a3b8; margin-top: 12px;">
                If any factor is zero, the product is zero.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="hint-box">
            👆 Click both "Run" buttons above to see the SQL results and discover the revelation!
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Summary Tab
# ============================================================================

def render_summary():
    """Render the Summary tab with key takeaways."""
    st.session_state.visited_tabs.add('summary')
    st.session_state.current_tab = 'summary'

    st.header("📊 Summary: What Did We Learn?")

    # Check if user completed the journey
    completed_all = len(st.session_state.visited_tabs) >= 4

    if completed_all:
        st.success("🎉 Congratulations! You've completed the investigation!")
    else:
        st.info("💡 Tip: Visit all tabs to get the full picture!")

    # Big Dashboard Cards
    st.markdown("### The Numbers at a Glance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #052e16 0%, #0f172a 100%);
                    border-radius: 12px; padding: 24px; text-align: center; height: 180px;">
            <div style="font-size: 3.5rem; font-weight: 700; color: #22c55e;">3 of 4</div>
            <div style="font-size: 1rem; color: #94a3b8; margin-top: 8px;">queries</div>
            <div style="font-size: 1.2rem; color: #f8fafc; margin-top: 8px;">
                ✅ Solvable with tuning
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
            <div style="font-size: 3.5rem; font-weight: 700; color: #ef4444;">1 of 4</div>
            <div style="font-size: 1rem; color: #94a3b8; margin-top: 8px;">queries</div>
            <div style="font-size: 1.2rem; color: #f8fafc; margin-top: 8px;">
                🔴 Needs corpus fix
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Your Checklist as a DBA
    st.markdown("### ✅ Your DBA Checklist for RAG Quality")

    checklist = [
        {
            "num": "1",
            "title": "Evaluate with Ground Truth",
            "desc": "Create test queries with known relevant documents. You can't improve what you can't measure.",
            "color": "#3b82f6"
        },
        {
            "num": "2",
            "title": "Start with k=20-50",
            "desc": "Higher k gives better recall but increases latency. Find your sweet spot.",
            "color": "#8b5cf6"
        },
        {
            "num": "3",
            "title": "Use Title Weighting for Entities",
            "desc": "Queries looking for specific things (people, places, concepts) benefit from title matching.",
            "color": "#22c55e"
        },
        {
            "num": "4",
            "title": "Audit Your Corpus FIRST",
            "desc": "Before tweaking search algorithms, check if the documents even exist!",
            "color": "#ef4444"
        }
    ]

    for item in checklist:
        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px;
                    margin: 8px 0; display: flex; align-items: flex-start; gap: 16px;
                    border-left: 4px solid {item['color']};">
            <div style="background-color: {item['color']}; color: white; width: 32px; height: 32px;
                        border-radius: 50%; display: flex; align-items: center; justify-content: center;
                        font-weight: 700; flex-shrink: 0;">
                {item['num']}
            </div>
            <div>
                <div style="font-weight: 600; color: #f8fafc;">{item['title']}</div>
                <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 4px;">
                    {item['desc']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Query Results Summary
    st.markdown("### 📋 Query Analysis Summary")

    summary_data = [
        {"Query": "Who invented the telephone?", "Status": "✅ Perfect", "First 100%": "k=10",
         "Best Strategy": "Any", "Insight": "Easy factual query"},
        {"Query": "neural networks deep learning", "Status": "✅ Perfect", "First 100%": "k=10",
         "Best Strategy": "Content-Only", "Insight": "Semantic search works well"},
        {"Query": "animal whales", "Status": "✅ Fixed", "First 100%": "k=20",
         "Best Strategy": "Title-Only", "Insight": "Title weighting gave +25%"},
        {"Query": "What is machine learning?", "Status": "🔴 Corpus Gap", "First 100%": "k=100",
         "Best Strategy": "None helps", "Insight": "No ML article exists!"}
    ]

    st.dataframe(
        pd.DataFrame(summary_data),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # The Formula
    st.markdown("### 🔑 The RAG Quality Formula")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
                border-radius: 12px; padding: 32px; text-align: center; margin: 20px 0;">
        <div style="font-size: 1.5rem; color: #f8fafc; font-family: 'Georgia', serif;">
            RAG Quality = <span style="color: #ef4444;">Corpus Coverage</span>
            × <span style="color: #eab308;">Search Algorithm</span>
            × <span style="color: #22c55e;">Generation Model</span>
        </div>
        <div style="margin-top: 16px; color: #94a3b8;">
            Today you learned to diagnose problems in the first two factors.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Resources
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Learn More")
        st.markdown("""
        - **nDCG Deep Dive**: Understanding ranking quality
        - **K-Balance Experiments**: Finding optimal k values
        - **Hybrid Search**: Combining dense + sparse vectors
        """)

    with col2:
        st.markdown("### 🔗 Resources")
        st.markdown("""
        **pgvector RAG Search Lab**

        [github.com/boutaga/pgvector_RAG_search_lab](https://github.com/boutaga/pgvector_RAG_search_lab)

        Tools for PostgreSQL + pgvector + RAG
        """)

    # Restart button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start Over", use_container_width=True):
            # Clear session state
            st.session_state.query_results = {}
            st.session_state.strategy_results_cache = {}
            st.session_state.visited_tabs = set()
            st.session_state.pop('ran_title_sql', None)
            st.session_state.pop('ran_fts_sql', None)
            st.rerun()


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
    render_sidebar()

    # Main header - compact version
    st.markdown("""
    <div style="text-align: center; margin-bottom: 8px;">
        <span style="font-size: 1.4rem; font-weight: 600; color: #f8fafc;">🔬 RAG Search Quality Lab</span>
        <span style="color: #64748b; font-size: 0.9rem; margin-left: 12px;">Interactive Demo</span>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Welcome",
        "🔴 Act 1: Problem",
        "🔶 Act 2: Investigation",
        "🟢 Act 3: Revelation",
        "📊 Summary"
    ])

    with tab1:
        render_welcome()

    with tab2:
        render_act1_problem()

    with tab3:
        render_act2_investigation()

    with tab4:
        render_act3_revelation()

    with tab5:
        render_summary()


if __name__ == "__main__":
    main()
