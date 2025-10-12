#!/usr/bin/env python3
"""
Streamlit UI components for evaluation metrics.

This module provides reusable Streamlit components for displaying
and managing nDCG evaluation metrics and relevance grades.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.evaluation.metrics import ndcg_at_k, ndcg_at_k_binary, ndcg_at_k_with_grades
from lab.evaluation.relevance_manager import RelevanceManager, TestQuery, RelevanceGrade


# ============================================================================
# Evaluation Dashboard Components
# ============================================================================

def render_evaluation_overview(relevance_manager: RelevanceManager):
    """
    Render the evaluation overview section.

    Shows:
    - Total test queries
    - Total relevance grades
    - Average nDCG by method
    - Recent evaluation activity
    """
    st.subheader("📊 Evaluation Overview")

    try:
        # Get summary statistics
        summary = relevance_manager.get_evaluation_summary(days_back=30)

        # Get labeled queries
        labeled_queries = relevance_manager.get_labeled_queries()
        num_queries = len(labeled_queries)

        # Count total grades
        total_grades = 0
        for qid in labeled_queries:
            total_grades += len(relevance_manager.get_relevance_grades(qid))

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Test Queries", num_queries)

        with col2:
            st.metric("Relevance Grades", total_grades)

        with col3:
            if summary and summary.get('avg_ndcg'):
                avg_ndcg = summary['avg_ndcg']
                st.metric("Avg nDCG (30d)", f"{avg_ndcg:.3f}")
            else:
                st.metric("Avg nDCG (30d)", "N/A")

        with col4:
            if summary and summary.get('num_evaluations'):
                num_evals = summary['num_evaluations']
                st.metric("Evaluations (30d)", num_evals)
            else:
                st.metric("Evaluations (30d)", 0)

        # Additional stats if available
        if summary and summary.get('avg_ndcg'):
            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                if summary.get('min_ndcg') is not None:
                    st.metric("Min nDCG", f"{summary['min_ndcg']:.3f}")

            with col2:
                if summary.get('max_ndcg') is not None:
                    st.metric("Max nDCG", f"{summary['max_ndcg']:.3f}")

            with col3:
                if summary.get('stddev_ndcg') is not None:
                    st.metric("Std Dev", f"{summary['stddev_ndcg']:.3f}")

    except Exception as e:
        st.error(f"Failed to load overview: {e}")


def render_ndcg_trends(relevance_manager: RelevanceManager):
    """
    Render nDCG trend visualization over time.

    Shows line chart of nDCG scores by method over the past 30 days.
    """
    st.subheader("📈 nDCG Trends")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox(
            "Time Period",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )

    with col2:
        methods = ['vector', 'hybrid', 'adaptive']
        selected_methods = st.multiselect(
            "Methods to Display",
            methods,
            default=methods
        )

    if not selected_methods:
        st.warning("Please select at least one method to display.")
        return

    try:
        # Get trend data for each method
        trend_data = []
        for method in selected_methods:
            method_trend = relevance_manager.get_ndcg_trend(method, days_back)
            for row in method_trend:
                trend_data.append({
                    'Date': row['date'],
                    'nDCG': row['avg_ndcg'],
                    'Method': method.title(),
                    'Num Queries': row['num_queries']
                })

        if trend_data:
            df = pd.DataFrame(trend_data)

            # Create line chart
            fig = px.line(
                df,
                x='Date',
                y='nDCG',
                color='Method',
                title='nDCG Trends by Search Method',
                markers=True,
                hover_data=['Num Queries']
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="nDCG Score",
                yaxis_range=[0, 1],
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation data available for the selected time period.")

    except Exception as e:
        st.error(f"Failed to load trends: {e}")


def render_method_comparison(relevance_manager: RelevanceManager):
    """
    Render comparison of different retrieval methods.

    Shows bar chart comparing average nDCG across methods.
    """
    st.subheader("🔍 Method Comparison")

    # Get comparison data
    methods = ['vector', 'hybrid', 'adaptive']
    k_value = st.selectbox("Top-K Results", [5, 10, 20], index=1)

    try:
        comparison = relevance_manager.compare_methods(methods, k_value)

        if comparison:
            # Prepare data for visualization
            chart_data = []
            for method, metrics in comparison.items():
                chart_data.append({
                    'Method': method.title(),
                    'nDCG': metrics.get('mean_ndcg', 0),
                    'Precision': metrics.get('mean_precision', 0),
                    'Recall': metrics.get('mean_recall', 0),
                    'F1': metrics.get('mean_f1', 0),
                    'MRR': metrics.get('mean_mrr', 0)
                })

            df = pd.DataFrame(chart_data)

            # Create grouped bar chart
            fig = go.Figure()

            metrics_to_plot = ['nDCG', 'Precision', 'Recall', 'F1', 'MRR']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df['Method'],
                        y=df[metric],
                        marker_color=colors[i]
                    ))

            fig.update_layout(
                title=f'Method Comparison at k={k_value}',
                xaxis_title='Search Method',
                yaxis_title='Score',
                yaxis_range=[0, 1],
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show detailed table
            st.subheader("Detailed Metrics")
            display_df = df.round(3)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No comparison data available. Run evaluations first.")

    except Exception as e:
        st.error(f"Failed to load comparison: {e}")


def render_test_queries_manager(relevance_manager: RelevanceManager):
    """
    Render test queries management interface.

    Allows viewing, adding, and managing test queries.
    """
    st.subheader("📝 Test Queries")

    tab1, tab2 = st.tabs(["View Queries", "Add New Query"])

    with tab1:
        # List existing queries
        try:
            queries = relevance_manager.list_test_queries(limit=100)

            if queries:
                st.write(f"**Total Test Queries**: {len(queries)}")

                # Create DataFrame
                queries_data = []
                for q in queries:
                    queries_data.append({
                        'ID': q.query_id,
                        'Query': q.query_text[:60] + ('...' if len(q.query_text) > 60 else ''),
                        'Type': q.query_type or 'N/A',
                        'Category': q.category or 'N/A',
                        'Created By': q.created_by or 'N/A'
                    })

                df = pd.DataFrame(queries_data)

                # Display with selection
                if not df.empty:
                    selected_query = st.selectbox(
                        "Select a query to view details",
                        options=df['ID'].tolist(),
                        format_func=lambda x: f"Q{x}: {df[df['ID']==x]['Query'].iloc[0]}"
                    )

                    if selected_query:
                        query_details = next(q for q in queries if q.query_id == selected_query)
                        st.info(f"**Query**: {query_details.query_text}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Type**: {query_details.query_type or 'N/A'}")
                            st.write(f"**Category**: {query_details.category or 'N/A'}")
                        with col2:
                            st.write(f"**Created By**: {query_details.created_by or 'N/A'}")
                            if query_details.created_at:
                                st.write(f"**Created**: {query_details.created_at.strftime('%Y-%m-%d')}")

                        if query_details.notes:
                            st.write(f"**Notes**: {query_details.notes}")

                        # Show relevance grades for this query
                        grades = relevance_manager.get_relevance_grades(selected_query)
                        if grades:
                            st.write(f"**Relevance Grades** ({len(grades)} documents labeled):")

                            grades_df = pd.DataFrame([
                                {
                                    'Document ID': doc_id,
                                    'Relevance Grade': grade,
                                    'Label': ['Irrelevant', 'Relevant', 'Highly Relevant'][grade]
                                }
                                for doc_id, grade in grades.items()
                            ])
                            st.dataframe(grades_df, use_container_width=True)

                            # Show distribution
                            dist = grades_df['Relevance Grade'].value_counts().sort_index()
                            fig = px.pie(
                                values=dist.values,
                                names=[f"Grade {i}" for i in dist.index],
                                title="Relevance Grade Distribution",
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No relevance grades for this query yet.")
            else:
                st.info("No test queries yet. Add some in the 'Add New Query' tab.")

        except Exception as e:
            st.error(f"Failed to load queries: {e}")

    with tab2:
        # Form to add new query
        with st.form("add_query_form"):
            query_text = st.text_area(
                "Query Text",
                placeholder="Enter your test query...",
                help="The search query to evaluate"
            )

            col1, col2 = st.columns(2)
            with col1:
                query_type = st.selectbox(
                    "Query Type",
                    ["factual", "conceptual", "exploratory", "other"],
                    help="Type of question being asked"
                )

            with col2:
                category = st.text_input(
                    "Category (optional)",
                    placeholder="e.g., technical, general",
                    help="Optional category for organizing queries"
                )

            created_by = st.text_input(
                "Created By",
                placeholder="Your name or 'llm'",
                help="Who is creating this test query"
            )

            notes = st.text_area(
                "Notes (optional)",
                placeholder="Any additional information...",
                help="Optional notes about this query"
            )

            submitted = st.form_submit_button("Add Query", type="primary")

            if submitted and query_text:
                try:
                    test_query = TestQuery(
                        query_id=None,
                        query_text=query_text,
                        query_type=query_type,
                        category=category or None,
                        created_by=created_by or None,
                        notes=notes or None
                    )

                    query_id = relevance_manager.create_test_query(test_query)
                    st.success(f"✅ Query added successfully! (ID: {query_id})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add query: {str(e)}")


def render_relevance_labeling(relevance_manager: RelevanceManager, db_service):
    """
    Render relevance labeling interface.

    Allows labeling documents for test queries.
    """
    st.subheader("🏷️ Relevance Labeling")

    try:
        # Get test queries
        queries = relevance_manager.list_test_queries(limit=100)

        if not queries:
            st.warning("No test queries available. Add queries first in the Test Queries tab.")
            return

        # Select query
        query_options = {q.query_id: q.query_text for q in queries}
        selected_qid = st.selectbox(
            "Select Query to Label",
            options=list(query_options.keys()),
            format_func=lambda x: f"Q{x}: {query_options[x][:60]}..."
        )

        if selected_qid:
            st.info(f"**Query**: {query_options[selected_qid]}")

            st.write("**Label documents by entering Document ID and Relevance Grade:**")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                doc_id = st.number_input(
                    "Document ID",
                    min_value=1,
                    step=1,
                    key="manual_doc_id",
                    help="Enter the ID of the document to label"
                )

            with col2:
                grade = st.selectbox(
                    "Relevance Grade",
                    [0, 1, 2],
                    format_func=lambda x: ["0: Irrelevant", "1: Relevant", "2: Highly Relevant"][x],
                    help="0=irrelevant, 1=relevant, 2=highly relevant"
                )

            with col3:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Add Label", type="primary", use_container_width=True):
                    try:
                        relevance_grade = RelevanceGrade(
                            query_id=selected_qid,
                            doc_id=doc_id,
                            rel_grade=grade,
                            labeler=st.session_state.get('user_name', 'anonymous'),
                            label_method='human',
                            notes=None
                        )

                        relevance_manager.add_relevance_grade(relevance_grade)
                        st.success(f"✅ Label added: Doc {doc_id} = Grade {grade}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to add label: {str(e)}")

            # Show existing labels
            existing_grades = relevance_manager.get_relevance_grades(selected_qid)
            if existing_grades:
                st.write(f"**Existing Labels** ({len(existing_grades)} documents):")
                grades_df = pd.DataFrame([
                    {
                        'Doc ID': doc_id,
                        'Grade': grade,
                        'Label': ["Irrelevant", "Relevant", "Highly Relevant"][grade]
                    }
                    for doc_id, grade in sorted(existing_grades.items())
                ])
                st.dataframe(grades_df, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load labeling interface: {e}")


def render_quick_evaluation(relevance_manager: RelevanceManager, db_service, engines):
    """
    Render quick evaluation interface.

    Allows running evaluations on test queries and viewing results.
    """
    st.subheader("⚡ Quick Evaluation")

    try:
        # Get labeled queries
        labeled_queries = relevance_manager.get_labeled_queries()

        if not labeled_queries:
            st.warning("No labeled queries available. Add test queries and relevance grades first.")
            return

        st.write(f"**Available Queries**: {len(labeled_queries)} with relevance grades")

        # Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            source = st.selectbox("Data Source", ["wikipedia", "movies"])

        with col2:
            method = st.selectbox("Search Method", ["simple", "hybrid", "adaptive"])

        with col3:
            k_value = st.selectbox("Top-K", [5, 10, 20], index=1)

        # Select queries to evaluate
        num_queries = st.slider(
            "Number of queries to evaluate",
            1,
            min(len(labeled_queries), 50),
            min(10, len(labeled_queries))
        )

        if st.button("🚀 Run Evaluation", type="primary"):
            with st.spinner(f"Evaluating {num_queries} queries..."):
                results = []

                # Get engine
                engine = engines.get(source, {}).get(method)

                if not engine:
                    st.error("Search engine not available")
                    return

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, qid in enumerate(labeled_queries[:num_queries]):
                    status_text.text(f"Evaluating query {i+1}/{num_queries}...")

                    # Get query text
                    test_query = relevance_manager.get_test_query(qid)
                    if not test_query:
                        continue

                    # Get relevance grades
                    grades = relevance_manager.get_relevance_grades(qid)

                    try:
                        # Perform search based on method
                        if method == "simple":
                            search_results = engine.search_dense(test_query.query_text, top_k=k_value)
                        elif method == "hybrid":
                            search_data = engine.search_hybrid(test_query.query_text, top_k=k_value)
                            search_results = search_data.get('hybrid_results', [])
                        else:  # adaptive
                            search_data = engine.search_adaptive(test_query.query_text, top_k=k_value)
                            search_results = search_data.get('results', [])

                        # Extract doc IDs
                        retrieved_ids = []
                        for r in search_results:
                            if isinstance(r, dict):
                                retrieved_ids.append(r.get('id', 0))
                            else:
                                retrieved_ids.append(getattr(r, 'id', 0))

                        # Calculate nDCG
                        ndcg = ndcg_at_k_with_grades(retrieved_ids, grades, k_value)

                        results.append({
                            'Query ID': qid,
                            'Query': test_query.query_text[:50] + '...',
                            'nDCG': ndcg,
                            'Num Results': len(search_results)
                        })

                        # Log to database
                        relevance_manager.log_retrieval(
                            qid,
                            [(doc_id, 1.0) for doc_id in retrieved_ids],
                            method,
                            None
                        )

                        # Save evaluation result
                        relevance_manager.save_evaluation_result(
                            qid, method, k_value,
                            ndcg, 0, 0,  # DCG/IDCG not calculated separately
                            None, None, None, None, None
                        )

                    except Exception as e:
                        st.warning(f"Query {qid} failed: {e}")

                    progress_bar.progress((i + 1) / num_queries)

                status_text.empty()

                # Display results
                if results:
                    st.success(f"✅ Evaluation completed!")

                    # Summary metrics
                    df = pd.DataFrame(results)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean nDCG", f"{df['nDCG'].mean():.3f}")
                    with col2:
                        st.metric("Median nDCG", f"{df['nDCG'].median():.3f}")
                    with col3:
                        st.metric("Std Dev", f"{df['nDCG'].std():.3f}")

                    # Results table
                    st.subheader("Results by Query")
                    st.dataframe(df.round(3), use_container_width=True)

                    # Distribution plot
                    fig = px.histogram(
                        df,
                        x='nDCG',
                        nbins=20,
                        title='nDCG Score Distribution',
                        labels={'nDCG': 'nDCG Score', 'count': 'Number of Queries'}
                    )
                    fig.update_layout(
                        xaxis_range=[0, 1],
                        yaxis_title='Number of Queries'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to run evaluation: {e}")
        st.exception(e)


# ============================================================================
# Main Evaluation Page Function
# ============================================================================

def evaluation_page(db_service, config, engines):
    """
    Main evaluation page for Streamlit app.

    This is the function to be called from streamlit_app.py

    Args:
        db_service: Database service instance
        config: Configuration object
        engines: Dictionary of search engines by source and method
    """
    st.header("📊 Evaluation & Metrics")

    st.markdown("""
    This page provides tools for evaluating RAG system performance using enhanced nDCG metrics.

    **Features**:
    - 📊 Evaluation overview and trends
    - 🔍 Method comparison (vector, hybrid, adaptive)
    - 📝 Test query management
    - 🏷️ Document relevance labeling (0-2 scale)
    - ⚡ Quick evaluation runner
    """)

    try:
        # Initialize RelevanceManager
        relevance_manager = RelevanceManager(config.database.connection_string)

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview",
            "🔍 Method Comparison",
            "📝 Test Queries",
            "🏷️ Labeling",
            "⚡ Quick Eval"
        ])

        with tab1:
            render_evaluation_overview(relevance_manager)
            st.divider()
            render_ndcg_trends(relevance_manager)

        with tab2:
            render_method_comparison(relevance_manager)

        with tab3:
            render_test_queries_manager(relevance_manager)

        with tab4:
            render_relevance_labeling(relevance_manager, db_service)

        with tab5:
            render_quick_evaluation(relevance_manager, db_service, engines)

    except Exception as e:
        st.error(f"Failed to load evaluation page: {str(e)}")
        st.exception(e)
        st.info("""
        **Troubleshooting**:
        1. Ensure the evaluation schema is installed: `psql -f lab/setup/evaluation_schema.sql`
        2. Check database connection in config
        3. Verify permissions on evaluation tables
        """)


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    'evaluation_page',
    'render_evaluation_overview',
    'render_ndcg_trends',
    'render_method_comparison',
    'render_test_queries_manager',
    'render_relevance_labeling',
    'render_quick_evaluation',
]
