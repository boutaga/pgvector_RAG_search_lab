#!/usr/bin/env python3
"""
Streamlit interface for pgvector RAG search.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.search.simple_search import SimpleSearchEngine
from lab.search.hybrid_search import HybridSearchEngine
from lab.search.adaptive_search import AdaptiveSearchEngine
from lab.evaluation.streamlit_evaluation import evaluation_page


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'db_service' not in st.session_state:
        st.session_state.db_service = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'engines' not in st.session_state:
        st.session_state.engines = {}
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []


@st.cache_resource
def load_services():
    """Load and cache database services."""
    config = ConfigService()
    
    db_service = DatabaseService(
        config.database.connection_string,
        config.database.min_connections,
        config.database.max_connections
    )
    
    engines = {
        'wikipedia': {
            'simple': SimpleSearchEngine(db_service, config, 'wikipedia'),
            'hybrid': HybridSearchEngine(db_service, config, 'wikipedia'),
            'adaptive': AdaptiveSearchEngine(db_service, config, 'wikipedia')
        },
        'movies': {
            'simple': SimpleSearchEngine(db_service, config, 'movies'),
            'hybrid': HybridSearchEngine(db_service, config, 'movies'),
            'adaptive': AdaptiveSearchEngine(db_service, config, 'movies')
        }
    }
    
    return db_service, config, engines


def get_source_statistics(db_service: DatabaseService) -> Dict[str, Dict[str, int]]:
    """Get statistics for data sources."""
    stats = {}
    
    try:
        # Wikipedia statistics
        with db_service.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM articles")
                total_articles = cur.fetchone()[0]
                
                # Check for new 3072 vectors first, fall back to old columns
                cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector_3072 IS NOT NULL")
                with_dense = cur.fetchone()[0]
                if with_dense == 0:
                    # Fall back to old column if new one is empty
                    cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector IS NOT NULL")
                    with_dense = cur.fetchone()[0]
                
                try:
                    cur.execute("SELECT COUNT(*) FROM articles WHERE content_sparse IS NOT NULL")
                    with_sparse = cur.fetchone()[0]
                except:
                    with_sparse = 0
        
        stats['Wikipedia'] = {
            'Total Articles': total_articles,
            'With Dense Embeddings': with_dense,
            'With Sparse Embeddings': with_sparse,
            'Dense Completion': f"{(with_dense/total_articles*100):.1f}%" if total_articles > 0 else "0%",
            'Sparse Completion': f"{(with_sparse/total_articles*100):.1f}%" if total_articles > 0 else "0%"
        }
    except Exception as e:
        st.warning(f"Could not load Wikipedia statistics: {e}")
        stats['Wikipedia'] = {}
    
    try:
        # Movies statistics
        with db_service.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM netflix_shows")
                total_shows = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM netflix_shows WHERE embedding IS NOT NULL")
                with_dense = cur.fetchone()[0]
                
                try:
                    cur.execute("SELECT COUNT(*) FROM netflix_shows WHERE sparse_embedding IS NOT NULL")
                    with_sparse = cur.fetchone()[0]
                except:
                    with_sparse = 0
        
        stats['Movies/Netflix'] = {
            'Total Shows': total_shows,
            'With Dense Embeddings': with_dense,
            'With Sparse Embeddings': with_sparse,
            'Dense Completion': f"{(with_dense/total_shows*100):.1f}%" if total_shows > 0 else "0%",
            'Sparse Completion': f"{(with_sparse/total_shows*100):.1f}%" if total_shows > 0 else "0%"
        }
    except Exception as e:
        st.warning(f"Could not load Movies/Netflix statistics: {e}")
        stats['Movies/Netflix'] = {}
    
    return stats


def display_search_results(results: List[Any], show_scores: bool = True):
    """Display search results in a nice format."""
    if not results:
        st.info("No results found.")
        return

    for i, result in enumerate(results, 1):
        # Handle both dict and object formats
        if isinstance(result, dict):
            score = result.get('score', 0.0)
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            result_id = result.get('id', 'N/A')
            source = result.get('source', '')
        else:
            score = getattr(result, 'score', 0.0)
            content = getattr(result, 'content', '')
            metadata = getattr(result, 'metadata', {})
            result_id = getattr(result, 'id', 'N/A')
            source = getattr(result, 'source', '')

        with st.expander(f"Result {i} {f'(Score: {score:.4f})' if show_scores else ''}",
                        expanded=i <= 3):

            # Content
            st.markdown("**Content:**")
            st.write(content)

            # Metadata
            if metadata:
                st.markdown("**Metadata:**")
                metadata_df = pd.DataFrame([metadata])
                st.dataframe(metadata_df, use_container_width=True)

            # Additional info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ID", result_id)
            with col2:
                if show_scores:
                    st.metric("Score", f"{score:.4f}")
            with col3:
                if source:
                    st.metric("Source", source)


def create_comparison_chart(comparison_results: Dict[str, List[Any]]):
    """Create visualization for method comparison."""
    # Prepare data for plotting
    methods = []
    num_results = []
    avg_scores = []
    
    for method, results in comparison_results.items():
        methods.append(method.title())
        num_results.append(len(results))
        if results:
            # Handle both dict and object formats
            scores = []
            for r in results:
                if isinstance(r, dict):
                    scores.append(r.get('score', 0.0))
                else:
                    scores.append(getattr(r, 'score', 0.0))
            avg_scores.append(sum(scores) / len(scores) if scores else 0)
        else:
            avg_scores.append(0)
    
    # Create subplot
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            x=methods,
            y=num_results,
            title="Number of Results by Method",
            labels={'x': 'Search Method', 'y': 'Number of Results'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            x=methods,
            y=avg_scores,
            title="Average Scores by Method",
            labels={'x': 'Search Method', 'y': 'Average Score'}
        )
        st.plotly_chart(fig2, use_container_width=True)


def search_page():
    """Main search page."""
    st.header("🔍 Vector Search Interface")
    
    # Load services
    try:
        db_service, config, engines = load_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Search Configuration")
        
        # Data source selection
        source = st.selectbox(
            "Data Source",
            ["wikipedia", "movies"],
            help="Choose the dataset to search"
        )
        
        # Search method
        method = st.selectbox(
            "Search Method",
            ["adaptive", "hybrid", "simple"],
            help="Choose the search algorithm"
        )
        
        # Method-specific options
        if method == "simple":
            search_type = st.selectbox(
                "Search Type",
                ["dense", "sparse"],
                help="Type of vector search"
            )
        elif method == "hybrid":
            st.subheader("Hybrid Weights")
            dense_weight = st.slider(
                "Dense Weight", 0.0, 1.0, 0.5, 0.1,
                help="Weight for dense search results"
            )
            sparse_weight = 1.0 - dense_weight
            st.write(f"Sparse Weight: {sparse_weight:.1f}")
        
        # Common options
        top_k = st.slider("Number of Results", 1, 20, 10)
        generate_answer = st.checkbox("Generate Answer", value=True)
        show_metadata = st.checkbox("Show Metadata", value=False)
        default_title_weight = getattr(config.embedding, 'title_weight', 0.4)
        title_search_mode = 'content'
        title_weight_value = default_title_weight
        title_vectors_enabled = False
        title_fts_rerank = False
        title_fts_weight = 0.2
        title_fts_max_candidates = None

        st.subheader("Title Search Options")
        if source == "wikipedia":
            title_vectors_enabled = st.toggle(
                "Enable Title Vector Search",
                value=False,
                help="Search article titles in addition to content vectors."
            )
            if title_vectors_enabled:
                title_only_mode = st.toggle(
                    "Title-only Mode",
                    value=False,
                    help="When enabled, only title embeddings are queried."
                )
                if title_only_mode:
                    title_search_mode = 'title'
                else:
                    title_search_mode = 'combined'
                    title_weight_value = st.slider(
                        "Title Vector Weight",
                        0.0,
                        1.0,
                        default_title_weight,
                        0.05,
                        help="Blend ratio between title and content vectors when combining results."
                    )
        else:
            st.toggle(
                "Enable Title Vector Search",
                value=False,
                disabled=True,
                help="Title vector search is only available for the Wikipedia dataset."
            )

        if method in ("hybrid", "adaptive"):
            if source == "wikipedia":
                title_fts_rerank = st.toggle(
                    "Boost titles with PostgreSQL FTS",
                    value=False,
                    help="Re-rank results using PostgreSQL full-text search scores over article titles."
                )
                if title_fts_rerank:
                    title_fts_weight = st.slider(
                        "Title FTS Weight",
                        0.0,
                        1.0,
                        0.2,
                        0.05,
                        help="How strongly the FTS score influences the final ranking."
                    )
                    title_fts_max_candidates = st.slider(
                        "Title FTS Candidate Count",
                        max(top_k, 10),
                        100,
                        min(max(top_k * 2, top_k), 50),
                        help="How many candidates to consider when applying title FTS reranking."
                    )
            else:
                st.toggle(
                    "Boost titles with PostgreSQL FTS",
                    value=False,
                    disabled=True,
                    help="Title FTS reranking is only available for the Wikipedia dataset."
                )

    
    effective_title_weight = title_weight_value if title_search_mode != 'content' else default_title_weight
    dense_search_kwargs = {'search_mode': title_search_mode}
    if title_search_mode != 'content':
        dense_search_kwargs['title_weight'] = title_weight_value

    hybrid_search_kwargs = {
        'search_mode': title_search_mode,
        'title_weight': effective_title_weight,
        'title_fts_rerank': title_fts_rerank,
        'title_fts_weight': title_fts_weight,
    }
    if title_fts_max_candidates is not None:
        hybrid_search_kwargs['title_fts_max_candidates'] = title_fts_max_candidates
    # Main search interface
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., What is machine learning? or romantic comedy movies"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        compare_button = st.button("📊 Compare Methods")
    with col3:
        if method == "adaptive":
            analyze_button = st.button("🧠 Analyze Query")
    
    if query and search_button:
        with st.spinner("Searching..."):
            try:
                engine = engines[source][method]
                start_time = time.time()
                
                if method == "simple":
                    if generate_answer:
                        response = engine.search_and_answer(
                            query=query,
                            search_type=search_type,
                            top_k=top_k,
                            search_mode=dense_search_kwargs['search_mode'],
                            title_weight=dense_search_kwargs.get('title_weight')
                        )
                        results = response.get('sources', [])
                        answer = response.get('answer')
                    else:
                        if search_type == "dense":
                            results = engine.search_dense(query, top_k, **dense_search_kwargs)
                        else:
                            results = engine.search_sparse(query, top_k)
                        answer = None
                
                elif method == "hybrid":
                    engine.update_weights(dense_weight, sparse_weight)
                    if generate_answer:
                        response = engine.generate_answer_from_hybrid(
                            query,
                            top_k,
                            **hybrid_search_kwargs
                        )
                        results = response.get('sources', [])
                        answer = response.get('answer')
                    else:
                        search_data = engine.search_hybrid(
                            query,
                            top_k,
                            **hybrid_search_kwargs
                        )
                        results = search_data['hybrid_results']
                        answer = None
                
                elif method == "adaptive":
                    if generate_answer:
                        response = engine.generate_adaptive_answer(
                            query,
                            top_k,
                            search_mode=title_search_mode,
                            title_weight=effective_title_weight,
                            title_fts_rerank=title_fts_rerank,
                            title_fts_weight=title_fts_weight,
                            title_fts_max_candidates=title_fts_max_candidates
                        )
                        results = response.get('sources', [])
                        answer = response.get('answer')

                        # Show query analysis
                        analysis = response.get('query_analysis', {})
                        if analysis:
                            st.info(
                                f"**Query Analysis:** {analysis.get('type', 'Unknown').title()} "
                                f"(Confidence: {analysis.get('confidence', 0):.3f}) - "
                                f"Weights: Dense {analysis.get('weights', {}).get('dense', 0):.2f}, "
                                f"Sparse {analysis.get('weights', {}).get('sparse', 0):.2f}"
                            )
                    else:
                        search_data = engine.search_adaptive(
                            query,
                            top_k,
                            show_analysis=True,
                            search_mode=title_search_mode,
                            title_weight=effective_title_weight,
                            title_fts_rerank=title_fts_rerank,
                            title_fts_weight=title_fts_weight,
                            title_fts_max_candidates=title_fts_max_candidates
                        )
                        results = search_data['results']
                        answer = None

                        # Show query analysis
                        st.info(
                            f"**Query Analysis:** {search_data.get('query_type', 'Unknown').title()} "
                            f"(Confidence: {search_data.get('classification_confidence', 0):.3f}) - "
                            f"Weights: Dense {search_data.get('recommended_weights', {}).get('dense', 0):.2f}, "
                            f"Sparse {search_data.get('recommended_weights', {}).get('sparse', 0):.2f}"
                        )
                
                search_time = time.time() - start_time
                
                # Display results
                st.success(f"Found {len(results)} results in {search_time:.2f} seconds")
                
                if answer:
                    st.subheader("💬 Generated Answer")
                    st.write(answer)
                
                if results:
                    st.subheader("📄 Search Results")
                    display_search_results(results, show_scores=True)
                
                # Add to history
                st.session_state.search_history.append({
                    'query': query,
                    'method': method,
                    'source': source,
                    'num_results': len(results),
                    'time': search_time,
                    'timestamp': time.time()
                })
            
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    if query and compare_button:
        with st.spinner("Comparing search methods..."):
            try:
                # Get all engines for comparison
                simple_engine = engines[source]['simple']
                hybrid_engine = engines[source]['hybrid']
                adaptive_engine = engines[source]['adaptive']
                
                # Perform different searches
                comparison_results = {}
                
                # Simple dense
                comparison_results['Dense'] = simple_engine.search_dense(query, top_k, **dense_search_kwargs)
                
                # Simple sparse
                comparison_results['Sparse'] = simple_engine.search_sparse(query, top_k)
                
                # Hybrid
                hybrid_data = hybrid_engine.search_hybrid(
                    query,
                    top_k,
                    **hybrid_search_kwargs
                )
                comparison_results['Hybrid'] = hybrid_data['hybrid_results']
                
                # Adaptive
                adaptive_data = adaptive_engine.search_adaptive(
                    query,
                    top_k,
                    show_analysis=True,
                    search_mode=title_search_mode,
                    title_weight=effective_title_weight,
                    title_fts_rerank=title_fts_rerank,
                    title_fts_weight=title_fts_weight,
                    title_fts_max_candidates=title_fts_max_candidates
                )
                comparison_results['Adaptive'] = adaptive_data['results']
                
                st.subheader("📊 Method Comparison")
                
                # Show comparison chart
                create_comparison_chart(comparison_results)
                
                # Show results by method
                for method_name, method_results in comparison_results.items():
                    st.subheader(f"{method_name} Search Results")
                    if method_results:
                        display_search_results(method_results[:3], show_scores=True)
                    else:
                        st.info("No results found.")
            
            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")
    
    if query and method == "adaptive" and 'analyze_button' in locals() and analyze_button:
        try:
            engine = engines[source]['adaptive']
            analysis = engine.classifier.analyze_query(query)
            
            st.subheader("🧠 Query Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query Type", analysis.query_type.value.title())
            with col2:
                st.metric("Confidence", f"{analysis.confidence:.3f}")
            with col3:
                st.metric("Recommended Weights", 
                         f"D: {analysis.recommended_weights[0]:.2f}, S: {analysis.recommended_weights[1]:.2f}")
            
            # Features
            st.subheader("Query Features")
            features_df = pd.DataFrame([analysis.features])
            st.dataframe(features_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def statistics_page():
    """Statistics and monitoring page."""
    st.header("📈 Statistics & Monitoring")
    
    try:
        db_service, config, engines = load_services()
        
        # Data source statistics
        st.subheader("📊 Data Source Statistics")
        
        stats = get_source_statistics(db_service)
        
        for source_name, source_stats in stats.items():
            if source_stats:
                st.write(f"**{source_name}**")
                
                # Create metrics columns
                cols = st.columns(len(source_stats))
                for i, (key, value) in enumerate(source_stats.items()):
                    with cols[i]:
                        st.metric(key, value)
                
                # Create completion chart if we have completion data
                if 'Dense Completion' in source_stats and 'Sparse Completion' in source_stats:
                    completion_data = {
                        'Type': ['Dense Embeddings', 'Sparse Embeddings'],
                        'Completion': [
                            float(source_stats['Dense Completion'].rstrip('%')),
                            float(source_stats['Sparse Completion'].rstrip('%'))
                        ]
                    }
                    
                    fig = px.bar(
                        completion_data,
                        x='Type',
                        y='Completion',
                        title=f"{source_name} Embedding Completion Rate",
                        labels={'Completion': 'Completion %'}
                    )
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
        
        # Search history
        if st.session_state.search_history:
            st.subheader("🔍 Recent Search History")
            
            # Convert to DataFrame for display
            history_df = pd.DataFrame(st.session_state.search_history)
            
            # Show summary statistics
            if not history_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Searches", len(history_df))
                with col2:
                    st.metric("Avg Results", f"{history_df['num_results'].mean():.1f}")
                with col3:
                    st.metric("Avg Time", f"{history_df['time'].mean():.2f}s")
                with col4:
                    most_used_method = history_df['method'].mode().iloc[0] if not history_df['method'].mode().empty else "N/A"
                    st.metric("Most Used Method", most_used_method.title())
                
                # Search method distribution
                method_counts = history_df['method'].value_counts()
                fig = px.pie(
                    values=method_counts.values,
                    names=method_counts.index,
                    title="Search Method Usage"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent searches table
                st.subheader("Recent Searches")
                display_df = history_df[['query', 'method', 'source', 'num_results', 'time']].tail(10)
                st.dataframe(display_df, use_container_width=True)
                
                # Clear history button
                if st.button("Clear Search History"):
                    st.session_state.search_history = []
                    st.success("Search history cleared!")
                    st.rerun()
        else:
            st.info("No search history yet. Perform some searches to see statistics here.")
    
    except Exception as e:
        st.error(f"Failed to load statistics: {str(e)}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PGVector RAG Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🔍 PGVector RAG Search Interface")
    st.markdown("Search Wikipedia articles and Netflix shows using advanced vector similarity methods.")
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📈 Statistics", "📊 Evaluation"])

    with tab1:
        search_page()

    with tab2:
        statistics_page()

    with tab3:
        # Load services for evaluation page
        try:
            db_service, config, engines = load_services()
            evaluation_page(db_service, config, engines)
        except Exception as e:
            st.error(f"Failed to load evaluation page: {e}")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with Streamlit • Powered by pgvector and OpenAI • 
        <a href='https://github.com/pgvector/pgvector' target='_blank'>Learn more about pgvector</a>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()