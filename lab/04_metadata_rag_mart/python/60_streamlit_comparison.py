#!/usr/bin/env python3
"""
60_streamlit_comparison.py — Multi-Model Comparison Dashboard

Interactive Streamlit dashboard for comparing embedding models, LLM providers,
and visualizing RAG quality metrics. Designed for conference demos.

Usage:
    streamlit run python/60_streamlit_comparison.py

Tabs:
    1. Single Query    — Interactive RAG search with model selectors
    2. Benchmark       — 16 golden queries, IR metrics comparison, Plotly charts
    3. Embedding Explorer — t-SNE 2D projection, cosine similarity heatmap
    4. Broker Intelligence — Agent 3 alerts with approve/reject UI
    5. Evaluation History — Historical metrics from rag_monitor
"""

import sys
import os
import json
import time
import math
import importlib

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
from embedding_provider import get_embedding_provider, FakeEmbeddingProvider
from llm_provider import get_llm_provider, FakeLLMProvider

# Lazy imports for agents (avoid import-time DB connections)
_agent_rag = None
_agent_broker = None


def _get_agent_rag():
    global _agent_rag
    if _agent_rag is None:
        _agent_rag = importlib.import_module("20_agent_rag_search")
    return _agent_rag


def _get_agent_broker():
    global _agent_broker
    if _agent_broker is None:
        _agent_broker = importlib.import_module("35_agent_broker")
    return _agent_broker


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lab 04 — Multi-Model RAG Comparison",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏦 Lab 04")
    st.caption("Multi-Model RAG Comparison")

    st.divider()
    st.subheader("Embedding Model")
    embedding_models = list(config.EMBEDDING_MODELS.keys())
    selected_embedding = st.selectbox(
        "Select embedding model",
        embedding_models,
        index=embedding_models.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in embedding_models else 0,
        help="All models output 1024-dimension vectors",
    )

    st.subheader("LLM Model")
    llm_models = list(config.LLM_MODELS.keys())
    selected_llm = st.selectbox(
        "Select LLM model",
        llm_models,
        index=llm_models.index(config.CHAT_MODEL) if config.CHAT_MODEL in llm_models else 0,
    )

    st.divider()
    st.subheader("RAG Parameters")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, config.SIMILARITY_THRESHOLD, 0.05)
    top_k = st.slider("Top-K", 1, 30, config.TOP_K)

    # Ollama endpoint (if ollama model selected)
    emb_info = config.EMBEDDING_MODELS.get(selected_embedding, {})
    if emb_info.get("provider") == "ollama":
        st.divider()
        st.subheader("Ollama")
        ollama_endpoint = st.text_input("Endpoint", config.OLLAMA_ENDPOINT)
        config.OLLAMA_ENDPOINT = ollama_endpoint

    st.divider()
    st.subheader("API Key Status")
    keys = {
        "Voyage AI": bool(config.VOYAGE_API_KEY),
        "OpenAI": bool(config.OPENAI_API_KEY),
        "Anthropic": bool(config.ANTHROPIC_API_KEY),
    }
    for name, available in keys.items():
        icon = "🟢" if available else "🔴"
        st.write(f"{icon} {name}")


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Query",
    "📊 Benchmark Comparison",
    "🧠 Embedding Explorer",
    "🏦 Broker Intelligence",
    "📈 Evaluation History",
])


# ===========================================================================
# TAB 1: Single Query
# ===========================================================================

with tab1:
    st.header("Interactive RAG Metadata Search")
    st.caption(f"Embedding: **{selected_embedding}** | LLM: **{selected_llm}**")

    query = st.text_input(
        "Business question",
        placeholder="e.g., Show me portfolio exposure by sector with risk metrics",
        key="single_query",
    )

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Threshold: {threshold} | Top-K: {top_k}")

    if run_btn and query:
        # Update config for this search
        original_threshold = config.SIMILARITY_THRESHOLD
        original_topk = config.TOP_K
        config.SIMILARITY_THRESHOLD = threshold
        config.TOP_K = top_k

        try:
            agent_rag = _get_agent_rag()
            with st.spinner(f"Searching with {selected_embedding} + {selected_llm}..."):
                t0 = time.time()
                rec = agent_rag.search(
                    query,
                    embedding_model=selected_embedding,
                    llm_model=selected_llm,
                )
                elapsed = time.time() - t0

            # Metrics bar
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tables found", len(rec.tables))
            col2.metric("Columns found", len(rec.columns))
            col3.metric("KPI patterns", len(rec.kpi_patterns))
            col4.metric("Total time", f"{rec.total_time_ms}ms")

            # Governance badge
            cls_colors = {"public": "green", "internal": "blue", "confidential": "orange", "restricted": "red"}
            cls_color = cls_colors.get(rec.max_classification, "gray")
            st.markdown(f"**Governance:** :{cls_color}[{rec.max_classification}]")
            if rec.pii_fields:
                st.warning(f"PII fields detected: {', '.join(rec.pii_fields)}")

            # Reasoning
            st.subheader("LLM Reasoning")
            st.info(rec.reasoning)

            # Results tables
            col_tables, col_cols = st.columns(2)

            with col_tables:
                st.subheader("Tables")
                if rec.tables:
                    df_tables = pd.DataFrame([
                        {"Table": t["table_name"],
                         "Similarity": round(t["similarity"], 3),
                         "Classification": t["classification"],
                         "Rows": t.get("row_count", "?")}
                        for t in rec.tables
                    ])
                    st.dataframe(df_tables, use_container_width=True, hide_index=True)
                else:
                    st.caption("No tables found above threshold")

            with col_cols:
                st.subheader("Columns")
                if rec.columns:
                    df_cols = pd.DataFrame([
                        {"Column": f"{c['table_name']}.{c['column_name']}",
                         "Type": c["data_type"],
                         "Similarity": round(c["similarity"], 3),
                         "PII": "🔒" if c.get("is_pii") else ""}
                        for c in rec.columns[:15]
                    ])
                    st.dataframe(df_cols, use_container_width=True, hide_index=True)
                else:
                    st.caption("No columns found above threshold")

            if rec.kpi_patterns:
                st.subheader("KPI Patterns")
                df_kpis = pd.DataFrame([
                    {"KPI": k["kpi_name"],
                     "Domain": k["domain"],
                     "Similarity": round(k["similarity"], 3)}
                    for k in rec.kpi_patterns
                ])
                st.dataframe(df_kpis, use_container_width=True, hide_index=True)

        finally:
            config.SIMILARITY_THRESHOLD = original_threshold
            config.TOP_K = original_topk


# ===========================================================================
# TAB 2: Benchmark Comparison
# ===========================================================================

with tab2:
    st.header("Benchmark Comparison")
    st.caption("Run golden queries against multiple embedding models")

    golden_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "golden_queries.json")

    if not os.path.exists(golden_path):
        st.warning(f"Golden queries file not found: {golden_path}")
    else:
        with open(golden_path) as f:
            golden_queries = json.load(f)
        st.caption(f"{len(golden_queries)} benchmark queries loaded")

        # Model selection for comparison
        compare_models = st.multiselect(
            "Select embedding models to compare",
            [m for m in embedding_models if m != "fake"],
            default=[config.EMBEDDING_MODEL],
        )

        if st.button("▶ Run Benchmark", type="primary") and compare_models:
            eval_mod = importlib.import_module("50_evaluate_rag")
            all_results = {}

            progress = st.progress(0)
            for i, model in enumerate(compare_models):
                with st.spinner(f"Evaluating {model}..."):
                    try:
                        metrics = eval_mod.run_evaluation(
                            golden_path, k=5, verbose=False,
                            embedding_model=model, llm_model=selected_llm,
                        )
                        all_results[model] = metrics
                    except Exception as e:
                        st.error(f"Error with {model}: {e}")
                progress.progress((i + 1) / len(compare_models))

            if all_results:
                st.session_state["benchmark_results"] = all_results

        # Display results from session state
        if "benchmark_results" in st.session_state:
            results = st.session_state["benchmark_results"]

            # Metrics table
            st.subheader("IR Metrics Comparison")
            metric_keys = ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "map", "avg_latency_ms"]
            metric_labels = ["P@5", "R@5", "nDCG@5", "MRR", "MAP", "Avg Latency (ms)"]

            rows = []
            for model, metrics in results.items():
                row = {"Model": model}
                for label, key in zip(metric_labels, metric_keys):
                    row[label] = metrics.get(key, 0)
                rows.append(row)

            df_compare = pd.DataFrame(rows)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            # Plotly bar chart
            st.subheader("Visual Comparison")
            quality_keys = ["precision_at_5", "recall_at_5", "ndcg_at_5", "mrr", "map"]
            quality_labels = ["P@5", "R@5", "nDCG@5", "MRR", "MAP"]

            chart_data = []
            for model, metrics in results.items():
                for label, key in zip(quality_labels, quality_keys):
                    chart_data.append({"Model": model, "Metric": label, "Value": metrics.get(key, 0)})

            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(df_chart, x="Metric", y="Value", color="Model", barmode="group",
                         title="IR Metrics by Embedding Model")
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Latency comparison
            if len(results) > 1:
                latency_data = [{"Model": m, "Avg Latency (ms)": v.get("avg_latency_ms", 0)}
                                for m, v in results.items()]
                df_lat = pd.DataFrame(latency_data)
                fig_lat = px.bar(df_lat, x="Model", y="Avg Latency (ms)",
                                 title="Average Latency by Model")
                st.plotly_chart(fig_lat, use_container_width=True)


# ===========================================================================
# TAB 3: Embedding Explorer
# ===========================================================================

with tab3:
    st.header("Embedding Explorer")
    st.caption("Visualize embedding space and similarity relationships")

    explore_texts = st.text_area(
        "Enter texts to embed (one per line)",
        value="Portfolio exposure by sector\nRisk limit utilization\nAML investigation pipeline\nClient AUM analysis\nExecution quality TCA",
        height=150,
    )

    if st.button("🧠 Generate Embeddings", key="explore_btn") and explore_texts.strip():
        texts = [t.strip() for t in explore_texts.strip().split("\n") if t.strip()]

        with st.spinner(f"Embedding {len(texts)} texts with {selected_embedding}..."):
            try:
                provider = get_embedding_provider(selected_embedding)
                embeddings = provider.embed_texts(texts)
                emb_array = np.array(embeddings)

                st.session_state["explore_texts"] = texts
                st.session_state["explore_embeddings"] = emb_array
            except Exception as e:
                st.error(f"Error: {e}")

    if "explore_embeddings" in st.session_state:
        texts = st.session_state["explore_texts"]
        emb_array = st.session_state["explore_embeddings"]

        col_tsne, col_heatmap = st.columns(2)

        with col_tsne:
            st.subheader("t-SNE Projection")
            if len(texts) >= 3:
                from sklearn.manifold import TSNE
                perplexity = min(5, len(texts) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                coords = tsne.fit_transform(emb_array)
                df_tsne = pd.DataFrame({
                    "x": coords[:, 0], "y": coords[:, 1],
                    "text": [t[:40] for t in texts],
                })
                fig_tsne = px.scatter(df_tsne, x="x", y="y", text="text",
                                      title=f"t-SNE ({selected_embedding})")
                fig_tsne.update_traces(textposition="top center")
                fig_tsne.update_layout(showlegend=False)
                st.plotly_chart(fig_tsne, use_container_width=True)
            else:
                st.caption("Need at least 3 texts for t-SNE")

        with col_heatmap:
            st.subheader("Cosine Similarity")
            # Compute cosine similarity matrix
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            normalized = emb_array / (norms + 1e-10)
            sim_matrix = normalized @ normalized.T

            labels = [t[:30] for t in texts]
            fig_heat = go.Figure(data=go.Heatmap(
                z=sim_matrix,
                x=labels, y=labels,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=np.round(sim_matrix, 3),
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(title=f"Cosine Similarity ({selected_embedding})",
                                   height=400)
            st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# TAB 4: Broker Intelligence
# ===========================================================================

with tab4:
    st.header("Broker Intelligence")
    st.caption("Agent 3: Scan financial signals, cross-reference with positions")

    if st.button("🏦 Run Intelligence Scan", type="primary", key="broker_btn"):
        with st.spinner("Scanning financial intelligence datasets..."):
            try:
                broker = _get_agent_broker()
                brief = broker.run_analysis(portfolio_manager="Demo PM")
                st.session_state["broker_brief"] = brief
            except Exception as e:
                st.error(f"Error: {e}")

    if "broker_brief" in st.session_state:
        brief = st.session_state["broker_brief"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Critical", brief.critical_count)
        col2.metric("🟠 Warning", brief.warning_count)
        col3.metric("🔵 Info", brief.info_count)
        col4.metric("Total Alerts", len(brief.alerts))

        st.subheader("Executive Brief")
        st.info(brief.summary)

        # Alerts
        st.subheader("Alerts")
        severity_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        source_icon = {"news": "📰", "recommendation": "📊", "report": "📈", "filing": "📋"}

        for i, alert in enumerate(brief.alerts[:15]):
            sev = severity_icon.get(alert.severity, "⚪")
            src = source_icon.get(alert.signal.source_type, "📄")

            with st.expander(f"{sev} [{alert.alert_id}] {alert.signal.headline[:80]}", expanded=(alert.severity == "critical")):
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.write(f"**Source:** {src} {alert.signal.source_type} | **Date:** {alert.signal.signal_date}")
                    st.write(f"**Sectors:** {', '.join(alert.signal.related_sectors) if alert.signal.related_sectors else 'N/A'}")
                    st.write(f"**Exposure:** {alert.exposure_summary}")
                with col_action:
                    st.write(f"**Action:** {alert.recommended_action}")
                    if alert.recommended_action == "request_mart":
                        st.caption(f"Suggested: {alert.suggested_mart_question[:80]}...")
                        approve = st.button(f"✅ Approve Mart", key=f"approve_{i}")
                        reject = st.button(f"❌ Dismiss", key=f"reject_{i}")
                        if approve:
                            st.success("Mart request approved (would trigger Agent 1 → Agent 2)")
                        if reject:
                            st.info("Alert dismissed")


# ===========================================================================
# TAB 5: Evaluation History
# ===========================================================================

with tab5:
    st.header("Evaluation History")
    st.caption("Historical RAG quality metrics from rag_monitor")

    try:
        import psycopg2
        conn = psycopg2.connect(config.DATABASE_URL)

        # Evaluation runs
        df_runs = pd.read_sql("""
            SELECT run_name, embedding_model, num_queries,
                   precision_at_5, recall_at_5, ndcg_at_5, mrr, map,
                   avg_latency_ms, p50_latency_ms,
                   notes, run_date
            FROM rag_monitor.evaluation_runs
            ORDER BY run_date DESC
            LIMIT 50
        """, conn)

        if not df_runs.empty:
            st.subheader("Evaluation Runs")
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            # Time series chart
            if len(df_runs) > 1:
                st.subheader("Quality Trend")
                fig_trend = go.Figure()
                for metric, color in [("precision_at_5", "blue"), ("recall_at_5", "green"),
                                      ("ndcg_at_5", "orange"), ("mrr", "red")]:
                    if metric in df_runs.columns:
                        fig_trend.add_trace(go.Scatter(
                            x=df_runs["run_date"], y=df_runs[metric],
                            name=metric.replace("_", " ").title(),
                            mode="lines+markers",
                        ))
                fig_trend.update_layout(
                    title="IR Metrics Over Time",
                    yaxis_range=[0, 1],
                    xaxis_title="Run Date",
                    yaxis_title="Score",
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            # Filter by model
            models = df_runs["embedding_model"].dropna().unique().tolist()
            if len(models) > 1:
                st.subheader("Filter by Model")
                selected_filter = st.selectbox("Model", ["All"] + models)
                if selected_filter != "All":
                    filtered = df_runs[df_runs["embedding_model"] == selected_filter]
                    st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.info("No evaluation runs found. Run `python 50_evaluate_rag.py` first.")

        # Search log summary
        df_search = pd.read_sql("""
            SELECT embedding_model, llm_model,
                   COUNT(*) AS queries,
                   AVG(total_time_ms) AS avg_time_ms,
                   AVG(total_results) AS avg_results
            FROM rag_monitor.search_log
            GROUP BY embedding_model, llm_model
            ORDER BY queries DESC
            LIMIT 20
        """, conn)

        if not df_search.empty:
            st.subheader("Search Log by Model")
            st.dataframe(df_search, use_container_width=True, hide_index=True)

        conn.close()

    except Exception as e:
        st.warning(f"Could not connect to database: {e}")
        st.caption("Start the Docker environment to see evaluation history.")
