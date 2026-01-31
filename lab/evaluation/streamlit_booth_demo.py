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
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 16px;
        font-size: 1rem;
        font-weight: 600;
    }

    /* Big action buttons */
    .stButton > button {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
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
    """Render the interactive RAG architecture diagram with live animation."""
    import random

    st.markdown("## 📊 Understanding the RAG Pipeline")
    st.markdown("Click **Run Demo Query** to see how your query flows through the system:")

    # Get current animation state
    current_step = st.session_state.rag_pipeline_step
    is_running = st.session_state.rag_pipeline_running

    # Define styles for each component based on animation state
    # Step 0 = idle, Step 1 = embedding, Step 2 = search, Step 3 = LLM generation, Step 4 = complete
    def get_card_style(card_step, current_step, is_running):
        """Get card styling based on animation state."""
        base_bg = "#1e293b"
        dim_bg = "#0f172a"

        if not is_running or current_step == 0:
            # Idle state - all cards normal
            return {
                "bg": base_bg,
                "border": "#334155",
                "shadow": "none",
                "opacity": "1",
                "icon_filter": "grayscale(0%)"
            }
        elif current_step == card_step:
            # This card is currently active - bright and glowing
            return {
                "bg": "#1e3a5f",
                "border": "#3b82f6",
                "shadow": "0 0 30px rgba(59, 130, 246, 0.5)",
                "opacity": "1",
                "icon_filter": "grayscale(0%) brightness(1.2)"
            }
        elif current_step > card_step:
            # This card is completed - green check, slight glow
            return {
                "bg": "#052e16",
                "border": "#22c55e",
                "shadow": "0 0 15px rgba(34, 197, 94, 0.3)",
                "opacity": "1",
                "icon_filter": "grayscale(0%)"
            }
        else:
            # This card is upcoming - dimmed
            return {
                "bg": dim_bg,
                "border": "#1e293b",
                "shadow": "none",
                "opacity": "0.5",
                "icon_filter": "grayscale(50%)"
            }

    # Card 1: User Query
    user_style = get_card_style(0, current_step, is_running)
    # Card 2: Embedding API (Step 1)
    embed_style = get_card_style(1, current_step, is_running)
    # Card 3: PostgreSQL/pgvector (Step 2)
    db_style = get_card_style(2, current_step, is_running)
    # Card 4: LLM (Step 3)
    llm_style = get_card_style(3, current_step, is_running)

    # Get timing info for completed steps
    embed_time = st.session_state.rag_pipeline_times.get("step_1", "~200")
    search_time = st.session_state.rag_pipeline_times.get("step_2", "~50")
    llm_time = st.session_state.rag_pipeline_times.get("step_3", "~800")

    # Arrow colors based on state
    def get_arrow_color(from_step, current_step, is_running):
        if not is_running:
            return "#334155"
        elif current_step > from_step:
            return "#22c55e"
        elif current_step == from_step:
            return "#3b82f6"
        else:
            return "#1e293b"

    arrow1_color = get_arrow_color(1, current_step, is_running)
    arrow2_color = get_arrow_color(2, current_step, is_running)
    arrow3_color = get_arrow_color(3, current_step, is_running)

    # Render the large responsive diagram (70-80% of screen)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 20px; padding: 40px; margin: 20px 0; min-height: 500px;">

        <!-- Title -->
        <div style="text-align: center; margin-bottom: 40px;">
            <span style="color: #3b82f6; font-size: 1.5rem; font-weight: 700; letter-spacing: 3px;
                        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);">
                RAG SEARCH ARCHITECTURE
            </span>
            <div style="color: #64748b; font-size: 0.9rem; margin-top: 8px;">
                Retrieval-Augmented Generation Pipeline
            </div>
        </div>

        <!-- Top Row: User -> Embedding -> Database -->
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;
                    margin-bottom: 30px;">

            <!-- User Query Card -->
            <div style="background: {user_style['bg']}; border: 3px solid {user_style['border']};
                        border-radius: 16px; padding: 28px 24px; min-width: 180px; text-align: center;
                        box-shadow: {user_style['shadow']}; opacity: {user_style['opacity']};
                        transition: all 0.4s ease;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: {user_style['icon_filter']};">👤</div>
                <div style="color: #22c55e; font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;">USER</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">Query Input</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 12px; font-style: italic;">
                    "What is ML?"
                </div>
            </div>

            <!-- Arrow 1 -->
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="color: {arrow1_color}; font-size: 2rem; font-weight: bold;
                            text-shadow: {'0 0 10px ' + arrow1_color if current_step >= 1 else 'none'};">
                    ───▶
                </div>
                <div style="color: {arrow1_color}; font-size: 0.7rem; margin-top: 4px;">(1) Query</div>
            </div>

            <!-- Embedding API Card -->
            <div style="background: {embed_style['bg']}; border: 3px solid {embed_style['border']};
                        border-radius: 16px; padding: 28px 24px; min-width: 200px; text-align: center;
                        box-shadow: {embed_style['shadow']}; opacity: {embed_style['opacity']};
                        transition: all 0.4s ease;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: {embed_style['icon_filter']};">🔢</div>
                <div style="color: #eab308; font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;">EMBEDDING</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">OpenAI API</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">text-embedding-3-large</div>
                <div style="background: {'#052e16' if current_step > 1 else '#0f172a'}; border-radius: 20px;
                            padding: 6px 16px; margin-top: 16px; display: inline-block;">
                    <span style="color: {'#22c55e' if current_step > 1 else '#64748b'}; font-size: 1rem;
                                font-weight: {'700' if current_step > 1 else '400'};">
                        {embed_time}ms
                    </span>
                </div>
            </div>

            <!-- Arrow 2 -->
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="color: {arrow2_color}; font-size: 2rem; font-weight: bold;
                            text-shadow: {'0 0 10px ' + arrow2_color if current_step >= 2 else 'none'};">
                    ───▶
                </div>
                <div style="color: {arrow2_color}; font-size: 0.7rem; margin-top: 4px;">(2) Vector</div>
            </div>

            <!-- PostgreSQL Card -->
            <div style="background: {db_style['bg']}; border: 3px solid {db_style['border']};
                        border-radius: 16px; padding: 28px 24px; min-width: 200px; text-align: center;
                        box-shadow: {db_style['shadow']}; opacity: {db_style['opacity']};
                        transition: all 0.4s ease;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: {db_style['icon_filter']};">🐘</div>
                <div style="color: #8b5cf6; font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;">POSTGRESQL</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">pgvector Extension</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">25,000 Wikipedia docs</div>
                <div style="background: {'#052e16' if current_step > 2 else '#0f172a'}; border-radius: 20px;
                            padding: 6px 16px; margin-top: 16px; display: inline-block;">
                    <span style="color: {'#22c55e' if current_step > 2 else '#64748b'}; font-size: 1rem;
                                font-weight: {'700' if current_step > 2 else '400'};">
                        {search_time}ms
                    </span>
                </div>
            </div>
        </div>

        <!-- Arrow down from DB to LLM -->
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="color: {arrow3_color}; font-size: 1.5rem;
                            text-shadow: {'0 0 10px ' + arrow3_color if current_step >= 3 else 'none'};">│</div>
                <div style="color: {arrow3_color}; font-size: 0.8rem; margin: 8px 0; font-weight: 600;">
                    (3) Top-K Documents
                </div>
                <div style="color: {arrow3_color}; font-size: 1.5rem;
                            text-shadow: {'0 0 10px ' + arrow3_color if current_step >= 3 else 'none'};">▼</div>
            </div>
        </div>

        <!-- Bottom Row: LLM + Final Answer -->
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;">

            <!-- LLM Card -->
            <div style="background: {llm_style['bg']}; border: 3px solid {llm_style['border']};
                        border-radius: 16px; padding: 28px 24px; min-width: 220px; text-align: center;
                        box-shadow: {llm_style['shadow']}; opacity: {llm_style['opacity']};
                        transition: all 0.4s ease;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: {llm_style['icon_filter']};">🤖</div>
                <div style="color: #f97316; font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;">LLM GENERATION</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">OpenAI GPT-5-mini</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">Context + Query → Answer</div>
                <div style="background: {'#052e16' if current_step > 3 else '#0f172a'}; border-radius: 20px;
                            padding: 6px 16px; margin-top: 16px; display: inline-block;">
                    <span style="color: {'#22c55e' if current_step > 3 else '#64748b'}; font-size: 1rem;
                                font-weight: {'700' if current_step > 3 else '400'};">
                        {llm_time}ms
                    </span>
                </div>
            </div>

            <!-- Arrow to Answer -->
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="color: {'#22c55e' if current_step >= 4 else '#334155'}; font-size: 2rem; font-weight: bold;
                            text-shadow: {'0 0 15px rgba(34, 197, 94, 0.5)' if current_step >= 4 else 'none'};">
                    ───▶
                </div>
                <div style="color: {'#22c55e' if current_step >= 4 else '#334155'}; font-size: 0.7rem; margin-top: 4px;">
                    (4) Answer
                </div>
            </div>

            <!-- Answer Card -->
            <div style="background: {'linear-gradient(135deg, #052e16 0%, #064e3b 100%)' if current_step >= 4 else '#0f172a'};
                        border: 3px solid {'#22c55e' if current_step >= 4 else '#1e293b'};
                        border-radius: 16px; padding: 28px 24px; min-width: 180px; text-align: center;
                        box-shadow: {'0 0 40px rgba(34, 197, 94, 0.4)' if current_step >= 4 else 'none'};
                        opacity: {'1' if current_step >= 4 or not is_running else '0.4'};
                        transition: all 0.4s ease;">
                <div style="font-size: 3.5rem; margin-bottom: 12px;">{'✅' if current_step >= 4 else '📝'}</div>
                <div style="color: {'#22c55e' if current_step >= 4 else '#64748b'}; font-weight: 700; font-size: 1.1rem;
                            letter-spacing: 1px;">
                    {'COMPLETE!' if current_step >= 4 else 'ANSWER'}
                </div>
                <div style="color: {'#86efac' if current_step >= 4 else '#475569'}; font-size: 0.85rem; margin-top: 8px;">
                    {'Response Ready' if current_step >= 4 else 'Waiting...'}
                </div>
            </div>
        </div>

        <!-- Total Time (shown when complete) -->
        {'<div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 2px solid #334155;">' +
         '<span style="color: #94a3b8; font-size: 1.1rem;">Total Pipeline Latency: </span>' +
         '<span style="color: #22c55e; font-size: 2rem; font-weight: 700; text-shadow: 0 0 20px rgba(34, 197, 94, 0.4);">' +
         str(sum(st.session_state.rag_pipeline_times.values())) + 'ms</span>' +
         '<div style="color: #64748b; font-size: 0.85rem; margin-top: 8px;">Retrieved 10 documents from 25,000</div></div>'
         if current_step >= 4 else ''}

    </div>
    """, unsafe_allow_html=True)

    # Demo Query Input and Button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        demo_query = st.text_input(
            "Demo Query:",
            value="What is machine learning?",
            key="demo_query_input",
            label_visibility="collapsed",
            placeholder="Enter a query to see the pipeline in action..."
        )

    with col2:
        run_demo = st.button("🚀 Run Demo Query", type="primary", use_container_width=True, key="run_demo_pipeline")

    with col3:
        if is_running and current_step >= 4:
            if st.button("🔄 Reset Demo", use_container_width=True, key="reset_demo"):
                st.session_state.rag_pipeline_running = False
                st.session_state.rag_pipeline_step = 0
                st.session_state.rag_pipeline_times = {}
                st.rerun()

    # Handle Run button click
    if run_demo:
        st.session_state.rag_pipeline_running = True
        st.session_state.rag_pipeline_step = 0
        st.session_state.rag_pipeline_times = {}
        st.rerun()

    # Animation logic - advance through steps
    if is_running and current_step < 4:
        # Define step timing
        steps_config = [
            (1, 180, 220),  # Step 1: Embedding
            (2, 40, 60),    # Step 2: Search
            (3, 750, 850),  # Step 3: LLM
            (4, 0, 0),      # Step 4: Complete (no delay)
        ]

        next_step = current_step + 1
        if next_step <= 4:
            step_idx = next_step - 1
            if step_idx < len(steps_config):
                step_num, min_ms, max_ms = steps_config[step_idx]
                if step_num <= 3:
                    simulated_time = random.randint(min_ms, max_ms)
                    time.sleep(simulated_time / 1000.0 * 0.5)  # Slightly slowed for visual effect
                    st.session_state.rag_pipeline_times[f"step_{step_num}"] = simulated_time
                st.session_state.rag_pipeline_step = next_step
                st.rerun()

    # Status message below diagram
    if is_running:
        if current_step == 1:
            st.info("🔄 **Step 1/3:** Generating vector embedding from your query...")
        elif current_step == 2:
            st.info("🔄 **Step 2/3:** Searching pgvector for similar documents...")
        elif current_step == 3:
            st.info("🔄 **Step 3/3:** LLM is generating the final answer...")
        elif current_step >= 4:
            total_time = sum(st.session_state.rag_pipeline_times.values())
            st.success(f"✅ **Complete!** Retrieved 10 documents and generated answer in **{total_time}ms**")

            # Show LLM Response Output with Source Attribution
            st.markdown("### 💬 LLM Response")

            # Determine if this is a RAG-grounded response or LLM knowledge
            # For "machine learning" query, we know from Act 3 there's no dedicated article
            demo_query_lower = demo_query.lower().strip()
            is_ml_query = "machine learning" in demo_query_lower

            # Source attribution banner
            if is_ml_query:
                # No dedicated ML article - response is from LLM knowledge
                source_banner = """
                <div style="background: linear-gradient(90deg, #7c2d12 0%, #450a0a 100%);
                            border-radius: 8px; padding: 16px; margin-bottom: 16px;
                            border: 2px solid #dc2626; display: flex; align-items: center; gap: 16px;">
                    <div style="font-size: 2rem;">⚠️</div>
                    <div style="flex: 1;">
                        <div style="color: #fca5a5; font-weight: 700; font-size: 1rem;">
                            SOURCE: LLM GENERAL KNOWLEDGE
                        </div>
                        <div style="color: #fecaca; font-size: 0.85rem; margin-top: 4px;">
                            No dedicated "Machine Learning" article found in corpus. Response is based on
                            LLM's pre-trained knowledge, NOT retrieved documents.
                        </div>
                    </div>
                    <div style="background: #450a0a; border-radius: 20px; padding: 8px 16px;
                                border: 1px solid #dc2626;">
                        <span style="color: #f87171; font-weight: 600; font-size: 0.9rem;">0% RAG</span>
                    </div>
                </div>
                """
                response_border = "#dc2626"
                retrieved_docs_section = """
                <div style="background: #1c1917; border-radius: 8px; padding: 16px; margin-top: 16px;
                            border: 1px solid #44403c;">
                    <div style="color: #a8a29e; font-weight: 600; margin-bottom: 12px;">
                        📄 Retrieved Documents (Related but NOT about ML):
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #292524; padding: 6px 12px; border-radius: 6px;
                                    color: #78716c; font-size: 0.8rem; border: 1px solid #44403c;">
                            ❌ Artificial Intelligence (mentions ML)
                        </span>
                        <span style="background: #292524; padding: 6px 12px; border-radius: 6px;
                                    color: #78716c; font-size: 0.8rem; border: 1px solid #44403c;">
                            ❌ Neural Network (mentions ML)
                        </span>
                        <span style="background: #292524; padding: 6px 12px; border-radius: 6px;
                                    color: #78716c; font-size: 0.8rem; border: 1px solid #44403c;">
                            ❌ Data Mining (tangential)
                        </span>
                    </div>
                    <div style="color: #78716c; font-size: 0.75rem; margin-top: 12px; font-style: italic;">
                        These documents mention "machine learning" but are not dedicated articles about the topic.
                    </div>
                </div>
                """
            else:
                # Other queries - assume RAG-grounded response
                source_banner = """
                <div style="background: linear-gradient(90deg, #052e16 0%, #064e3b 100%);
                            border-radius: 8px; padding: 16px; margin-bottom: 16px;
                            border: 2px solid #22c55e; display: flex; align-items: center; gap: 16px;">
                    <div style="font-size: 2rem;">✅</div>
                    <div style="flex: 1;">
                        <div style="color: #86efac; font-weight: 700; font-size: 1rem;">
                            SOURCE: RAG-GROUNDED RESPONSE
                        </div>
                        <div style="color: #bbf7d0; font-size: 0.85rem; margin-top: 4px;">
                            Response is generated using retrieved documents from the PostgreSQL database.
                            Information is grounded in your corpus data.
                        </div>
                    </div>
                    <div style="background: #052e16; border-radius: 20px; padding: 8px 16px;
                                border: 1px solid #22c55e;">
                        <span style="color: #22c55e; font-weight: 600; font-size: 0.9rem;">100% RAG</span>
                    </div>
                </div>
                """
                response_border = "#22c55e"
                retrieved_docs_section = """
                <div style="background: #052e16; border-radius: 8px; padding: 16px; margin-top: 16px;
                            border: 1px solid #166534;">
                    <div style="color: #86efac; font-weight: 600; margin-bottom: 12px;">
                        📄 Retrieved Documents Used:
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #064e3b; padding: 6px 12px; border-radius: 6px;
                                    color: #6ee7b7; font-size: 0.8rem; border: 1px solid #059669;">
                            ✓ Relevant Article 1 (score: 0.92)
                        </span>
                        <span style="background: #064e3b; padding: 6px 12px; border-radius: 6px;
                                    color: #6ee7b7; font-size: 0.8rem; border: 1px solid #059669;">
                            ✓ Relevant Article 2 (score: 0.88)
                        </span>
                        <span style="background: #064e3b; padding: 6px 12px; border-radius: 6px;
                                    color: #6ee7b7; font-size: 0.8rem; border: 1px solid #059669;">
                            ✓ Relevant Article 3 (score: 0.85)
                        </span>
                    </div>
                    <div style="color: #6ee7b7; font-size: 0.75rem; margin-top: 12px; font-style: italic;">
                        Response is grounded in these retrieved documents from your corpus.
                    </div>
                </div>
                """

            st.markdown(f"""
            {source_banner}
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        border-radius: 12px; padding: 24px; margin: 16px 0;
                        border-left: 4px solid {response_border};">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    <span style="font-size: 1.5rem; margin-right: 12px;">🤖</span>
                    <span style="color: {response_border}; font-weight: 600; font-size: 1rem;">GPT-5-mini Response</span>
                    <span style="color: #64748b; font-size: 0.8rem; margin-left: auto;">
                        {'⚠️ Based on general knowledge' if is_ml_query else '✅ Based on retrieved documents'}
                    </span>
                </div>
                <div style="color: #f8fafc; font-size: 1rem; line-height: 1.7;">
                    <strong style="color: #eab308;">Machine Learning</strong> is a subset of artificial intelligence
                    that enables computers to learn and improve from experience without being explicitly programmed.
                    It focuses on developing algorithms that can access data, learn from it, and make predictions
                    or decisions.<br><br>
                    Key concepts include:
                    <ul style="color: #94a3b8; margin-top: 12px;">
                        <li><strong style="color: #f8fafc;">Supervised Learning:</strong> Training with labeled data</li>
                        <li><strong style="color: #f8fafc;">Unsupervised Learning:</strong> Finding patterns in unlabeled data</li>
                        <li><strong style="color: #f8fafc;">Neural Networks:</strong> Deep learning architectures inspired by the brain</li>
                    </ul>
                </div>
                {retrieved_docs_section}
                <div style="border-top: 1px solid #334155; margin-top: 20px; padding-top: 16px;">
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                        <div style="color: #64748b; font-size: 0.8rem;">
                            <span style="color: #94a3b8;">Grounding:</span>
                            {'<span style="color: #f87171;">LLM Pre-trained Knowledge</span>' if is_ml_query else '<span style="color: #22c55e;">Corpus Documents</span>'}
                        </div>
                        <div style="color: #64748b; font-size: 0.8rem;">
                            <span style="color: #3b82f6;">Tokens:</span> ~180 |
                            <span style="color: #22c55e;">Latency:</span> {st.session_state.rag_pipeline_times.get('step_3', 800)}ms
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Educational callout for ML query
            if is_ml_query:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
                            border-radius: 8px; padding: 16px; margin-top: 16px;
                            border-left: 4px solid #3b82f6;">
                    <div style="color: #93c5fd; font-weight: 600; margin-bottom: 8px;">
                        💡 This is exactly the problem we investigate in Act 3!
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        The RAG system retrieved documents that <em>mention</em> "machine learning" but there's
                        <strong style="color: #f8fafc;">no dedicated article about ML</strong> in our 25k Wikipedia subset.
                        The LLM falls back to its general knowledge, which may be outdated or hallucinated.
                        <br><br>
                        <strong style="color: #fbbf24;">This is a corpus coverage problem, not a search algorithm problem!</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")


def render_welcome():
    """Render the Welcome tab with metric explanations and DBA persona."""
    st.session_state.visited_tabs.add('welcome')
    st.session_state.current_tab = 'welcome'

    # Persona Banner
    st.markdown("""
    <div class="persona-banner">
        <div style="font-size: 3rem;">👨‍💻</div>
        <div>
            <div style="font-size: 1.3rem; font-weight: 600; color: #f8fafc;">
                Welcome, Database Administrator!
            </div>
            <div style="color: #94a3b8; margin-top: 4px;">
                Your team has deployed a RAG search system. Users are complaining some queries
                "don't find the right documents." Let's investigate together.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # RAG Architecture Diagram with Live Animation
    render_rag_architecture_section()

    st.markdown("## 📖 Before We Start: Understanding the Metrics")
    st.markdown("""
    To diagnose search quality issues, we need to understand three key metrics.
    Don't worry - we'll explain them in plain English!
    """)

    # Metric Explanation Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-explain" style="border-left-color: #22c55e;">
            <div style="font-size: 2.5rem; text-align: center;">🎯</div>
            <h3 style="text-align: center; color: #22c55e; margin: 12px 0;">Recall</h3>
            <p style="color: #f8fafc; font-size: 1.1rem; text-align: center;">
                "Did we find all the relevant documents?"
            </p>
            <hr style="border-color: #334155; margin: 16px 0;">
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong>Example:</strong> If there are 4 relevant articles about "whales"
                and we found 3 of them:
            </p>
            <p style="color: #22c55e; font-size: 1.5rem; text-align: center; margin-top: 8px;">
                Recall = 75%
            </p>
            <p style="color: #64748b; font-size: 0.8rem; text-align: center;">
                (3 found ÷ 4 total = 0.75)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-explain" style="border-left-color: #eab308;">
            <div style="font-size: 2.5rem; text-align: center;">📊</div>
            <h3 style="text-align: center; color: #eab308; margin: 12px 0;">Precision</h3>
            <p style="color: #f8fafc; font-size: 1.1rem; text-align: center;">
                "Of what we found, how much was relevant?"
            </p>
            <hr style="border-color: #334155; margin: 16px 0;">
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong>Example:</strong> If we retrieved 10 documents but only 3
                were actually about "whales":
            </p>
            <p style="color: #eab308; font-size: 1.5rem; text-align: center; margin-top: 8px;">
                Precision = 30%
            </p>
            <p style="color: #64748b; font-size: 0.8rem; text-align: center;">
                (3 relevant ÷ 10 retrieved = 0.30)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-explain" style="border-left-color: #3b82f6;">
            <div style="font-size: 2.5rem; text-align: center;">📈</div>
            <h3 style="text-align: center; color: #3b82f6; margin: 12px 0;">nDCG</h3>
            <p style="color: #f8fafc; font-size: 1.1rem; text-align: center;">
                "Are the best results at the top?"
            </p>
            <hr style="border-color: #334155; margin: 16px 0;">
            <p style="color: #94a3b8; font-size: 0.9rem;">
                <strong>Example:</strong> Finding the "Whale" article at position 1
                is better than finding it at position 10.
            </p>
            <p style="color: #3b82f6; font-size: 1.5rem; text-align: center; margin-top: 8px;">
                nDCG = 0.0 to 1.0
            </p>
            <p style="color: #64748b; font-size: 0.8rem; text-align: center;">
                (1.0 = perfect ranking)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Quiz
    st.markdown("### 🧠 Quick Check: Which metric matters most?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px; margin: 8px 0;">
            <strong style="color: #f8fafc;">Scenario A:</strong>
            <span style="color: #94a3b8;">User asks "Who invented the telephone?"</span>
            <br><br>
            <span style="color: #64748b;">They need THE answer, not 100 related articles.</span>
            <br><br>
            <span style="color: #3b82f6;">→ <strong>Precision</strong> and <strong>nDCG</strong> matter most</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 16px; margin: 8px 0;">
            <strong style="color: #f8fafc;">Scenario B:</strong>
            <span style="color: #94a3b8;">User asks "Tell me about machine learning"</span>
            <br><br>
            <span style="color: #64748b;">They want comprehensive coverage of the topic.</span>
            <br><br>
            <span style="color: #22c55e;">→ <strong>Recall</strong> matters most</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # The Mission
    st.markdown("### 🎯 Your Mission")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
                border-radius: 12px; padding: 24px; margin: 16px 0;">
        <p style="color: #f8fafc; font-size: 1.1rem; margin: 0;">
            We have <strong>4 test queries</strong> with known "ground truth" - documents that
            SHOULD be returned. Your job is to:
        </p>
        <ol style="color: #94a3b8; margin-top: 16px; padding-left: 24px;">
            <li style="margin: 8px 0;">Run each query and check if we achieve <strong>100% Recall</strong></li>
            <li style="margin: 8px 0;">Identify which queries are <strong>struggling</strong></li>
            <li style="margin: 8px 0;">Try different <strong>search strategies</strong> to fix them</li>
            <li style="margin: 8px 0;">Discover why some queries <strong>can't be fixed</strong> with search alone</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Start button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start the Investigation →", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

    # Hint
    st.markdown("""
    <div class="hint-box">
        💡 <strong>Tip:</strong> Click the button above or use the tabs to navigate.
        You can always come back to this page to review the metrics.
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

    # Main header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="margin-bottom: 8px;">🔬 RAG Search Quality Lab</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            An interactive investigation into why some RAG queries fail
        </p>
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
