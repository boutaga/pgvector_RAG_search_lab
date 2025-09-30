#!/usr/bin/env python3
"""
Professional BI Analyst Interface for Data Mart Exploration and Rapid Prototyping
Powered by GPT-5 and RAG for intelligent schema generation
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import psycopg2
import plotly.graph_objects as go
import uuid
from typing import Dict, List, Optional, Any

# Add services and models to path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
services_dir = lab_dir / 'services'
models_dir = lab_dir / 'models'
sys.path.insert(0, str(services_dir))
sys.path.insert(0, str(models_dir))

from mart_agent_service import MartPlanningAgent, AgentConfig
from metadata_search_service import MetadataSearchService, SearchConfig
from mart_plan import MartPlan, FactDefinition, DimensionDefinition, MeasureDefinition

# Page configuration
st.set_page_config(
    page_title="BI Mart Explorer | GPT-5 Powered",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_agent_cached():
    """Get cached agent instance with default config."""
    config = AgentConfig(
        primary_model="gpt-5",
        fast_model="gpt-5-mini",
        fallback_model="gpt-4",
        max_tokens=4000,
        max_retries=3
    )
    return MartPlanningAgent(config)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .field-bucket {
        background-color: #f1f5f9;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        min-height: 100px;
    }
    .field-item {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .field-item:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    .field-item.secondary {
        background-color: #10b981;
    }
    .model-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        margin: 0.25rem;
    }
    .model-active {
        background-color: #10b981;
        color: white;
    }
    .model-fallback {
        background-color: #f59e0b;
        color: white;
    }
    .model-unavailable {
        background-color: #ef4444;
        color: white;
    }
    .erd-node {
        fill: #e0e7ff;
        stroke: #6366f1;
        stroke-width: 2;
    }
    .erd-node:hover {
        fill: #c7d2fe;
        cursor: pointer;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

class MartExplorerApp:
    """Main application class for BI Mart Explorer"""

    def __init__(self):
        self.initialize_session_state()
        self.initialize_services()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'mart_plan' not in st.session_state:
            st.session_state.mart_plan = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_fields' not in st.session_state:
            st.session_state.selected_fields = {'primary': [], 'secondary': []}
        if 'kpi_queries' not in st.session_state:
            st.session_state.kpi_queries = []
        if 'mart_created' not in st.session_state:
            st.session_state.mart_created = False
        if 'execution_history' not in st.session_state:
            st.session_state.execution_history = []
        if 'model_status' not in st.session_state:
            st.session_state.model_status = {}

    def initialize_services(self):
        """Initialize AI services and database connections"""
        try:
            # Check for required environment variables
            if not os.environ.get('OPENAI_API_KEY'):
                st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                st.stop()

            if not os.environ.get('DATABASE_URL'):
                st.warning("⚠️ Database URL not set. Using demo mode.")
                self.demo_mode = True
            else:
                self.demo_mode = False

            # Initialize services with proper GPT-5 configuration
            self.search_service = MetadataSearchService()

            # Get cached agent (initialized once per session)
            self.agent = get_agent_cached()

            # Get current sidebar settings (defaults if not yet created)
            top_k = st.session_state.get("top_k", 10)
            thr = st.session_state.get("similarity_threshold", 0.5)
            search_cfg = SearchConfig(top_k=top_k, similarity_threshold=thr, include_relationships=True)

            # Apply search config dynamically (not cached, can change per interaction)
            self.agent._ui_search_config = search_cfg

            # Store model status (check actual agent config)
            st.session_state.model_status = {
                'GPT-5': self.agent.config.primary_model == 'gpt-5',
                'GPT-5-mini': self.agent.config.fast_model == 'gpt-5-mini',
                'GPT-4': True  # Always available as fallback
            }

        except Exception as e:
            st.error(f"❌ Failed to initialize services: {e}")
            st.stop()

    def render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            st.markdown('<h1 class="main-header">📊 BI Mart Explorer</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Rapid Data Mart Prototyping & Exploration</p>', unsafe_allow_html=True)

        with col3:
            st.markdown("### 🤖 Model Status")
            for model, available in st.session_state.model_status.items():
                if available:
                    st.markdown(f'<span class="model-status model-active">{model} ✓</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="model-status model-fallback">{model} (fallback)</span>', unsafe_allow_html=True)

    def render_kpi_input(self):
        """Render KPI requirement input section"""
        st.markdown("## 🎯 Define Your KPI Requirements")
        st.markdown("*Describe the business metrics you need to track. Be specific about measures, dimensions, and time periods.*")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Use selected template if available
            default_value = st.session_state.get('selected_template', '')
            kpi_requirement = st.text_area(
                "KPI Description",
                value=default_value,
                placeholder="Example: I need to analyze product sales velocity by category over the last 30 days, including revenue contribution and inventory turnover metrics...",
                height=120,
                key="kpi_input"
            )

        with col2:
            st.markdown("### 💡 Quick Templates")
            templates = {
                "Sales Analysis": "Analyze product sales performance including revenue, quantity, and growth trends by category and region",
                "Customer Analytics": "Track customer lifetime value, purchase frequency, and churn indicators with demographic segmentation",
                "Inventory Metrics": "Monitor inventory turnover, stock levels, and reorder points by product category and supplier",
                "Financial KPIs": "Calculate gross margin, profit trends, and discount effectiveness across product lines"
            }

            for template_name, template_text in templates.items():
                if st.button(template_name, key=f"template_{template_name}", use_container_width=True):
                    # Set the template text in a different session state key to avoid conflicts
                    st.session_state.selected_template = template_text
                    st.rerun()

        return kpi_requirement

    def render_mart_planning(self, kpi_requirement: str):
        """Render mart planning and field selection interface"""
        # Dynamic button text based on model selection
        use_gpt5 = st.session_state.get("use_gpt5_planning", True)
        button_text = "🧠 Generate Mart Plan with GPT-5" if use_gpt5 else "⚡ Generate Mart Plan with GPT-5-mini"

        if st.button(button_text, type="primary", use_container_width=True):
            with st.spinner("🔍 Searching relevant metadata... 🧠 Planning optimal mart structure..."):
                try:
                    # Update model configuration based on sidebar settings
                    use_gpt5_for_planning = st.session_state.get("use_gpt5_planning", True)
                    use_gpt5_mini_for_validation = st.session_state.get("use_gpt5_mini_validation", True)
                    reasoning_effort = st.session_state.get("reasoning_effort", "low")

                    # Dynamically update agent task routing
                    if use_gpt5_for_planning:
                        self.agent.config.task_routing["complex_planning"] = "gpt-5"
                    else:
                        self.agent.config.task_routing["complex_planning"] = "gpt-5-mini"

                    if use_gpt5_mini_for_validation:
                        self.agent.config.task_routing["plan_validation"] = "gpt-5-mini"
                    else:
                        self.agent.config.task_routing["plan_validation"] = "gpt-5"

                    # Store reasoning effort for use in API calls
                    self.agent._reasoning_effort = reasoning_effort

                    # Update search configuration from sidebar
                    top_k = st.session_state.get("top_k", 10)
                    similarity_threshold = st.session_state.get("similarity_threshold", 0.5)
                    search_cfg = SearchConfig(
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        include_relationships=True
                    )
                    self.agent._ui_search_config = search_cfg

                    # Generate mart plan
                    mart_plan, search_results = self.agent.plan_mart_from_question(kpi_requirement)
                    st.session_state.mart_plan = mart_plan
                    st.session_state.search_results = search_results

                    # Clear cached validation result when new plan is generated
                    if 'validation_result' in st.session_state:
                        del st.session_state.validation_result

                    # Extract suggested fields
                    self.extract_suggested_fields(mart_plan)

                    st.success(f"✅ Mart plan generated successfully! Found {len(search_results)} relevant metadata elements.")
                    st.balloons()

                except ValueError as e:
                    if "No relevant metadata found" in str(e):
                        st.error("❌ No relevant metadata found for your question. Try:"
                               "\n- Using more general terms\n- Checking if the database contains relevant tables\n- Lowering the similarity threshold in settings")
                    else:
                        st.error(f"❌ Failed to parse mart plan: {e}")
                        st.warning("💡 The AI may be having trouble with the response format. Try rephrasing your question or using a template.")
                except Exception as e:
                    st.error(f"❌ Unexpected error during mart planning: {e}")
                    with st.expander("🔍 Debug Information"):
                        st.text(f"Error type: {type(e).__name__}")
                        st.text(f"Error details: {str(e)}")
                        if hasattr(e, '__traceback__'):
                            import traceback
                            st.text("Traceback:")
                            st.code(traceback.format_exc())

        # Display mart plan if available
        if st.session_state.mart_plan:
            self.render_mart_structure()

    def extract_suggested_fields(self, mart_plan: MartPlan):
        """Extract primary and secondary field suggestions from mart plan"""
        primary_fields = []
        secondary_fields = []

        # Extract fact table fields as primary
        for fact in mart_plan.facts:
            for measure in fact.measures:
                primary_fields.append({
                    'name': measure.name,
                    'type': 'measure',
                    'description': measure.description,
                    'expression': measure.expression,
                    'table': fact.name
                })

        # Extract dimension attributes
        for dim in mart_plan.dimensions:
            # Key column as primary
            primary_fields.append({
                'name': dim.key_column,
                'type': 'dimension_key',
                'description': f"Primary key for {dim.name}",
                'table': dim.name
            })

            # Other attributes as secondary
            for attr in dim.attributes:
                secondary_fields.append({
                    'name': attr,
                    'type': 'dimension_attribute',
                    'description': f"Attribute from {dim.name}",
                    'table': dim.name
                })

        st.session_state.selected_fields['primary'] = primary_fields
        st.session_state.selected_fields['secondary'] = secondary_fields

    def render_mart_structure(self):
        """Render interactive mart structure with field selection"""
        st.markdown("## 📐 Proposed Mart Structure")

        tabs = st.tabs(["📊 Field Selection", "🔗 ERD Visualization", "📋 Plan Details", "✅ Validation"])

        with tabs[0]:
            self.render_field_selection()

        with tabs[1]:
            self.render_erd_diagram()

        with tabs[2]:
            self.render_plan_details()

        with tabs[3]:
            self.render_validation()

    def render_field_selection(self):
        """Render dynamic field selection interface"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Primary Fields (Recommended)")
            st.markdown("*These fields are essential for your KPIs*")

            primary_container = st.container()
            with primary_container:
                for field in st.session_state.selected_fields['primary']:
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{field['name']}** ({field['type']})")
                        st.caption(field.get('description', ''))
                    with cols[1]:
                        st.caption(field.get('table', ''))
                    with cols[2]:
                        if st.button("➖", key=f"remove_primary_{field['name']}"):
                            st.session_state.selected_fields['primary'].remove(field)
                            st.session_state.selected_fields['secondary'].append(field)
                            st.rerun()

        with col2:
            st.markdown("### 💚 Additional Fields (Optional)")
            st.markdown("*Consider these for enhanced analysis*")

            secondary_container = st.container()
            with secondary_container:
                for field in st.session_state.selected_fields['secondary']:
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{field['name']}** ({field['type']})")
                        st.caption(field.get('description', ''))
                    with cols[1]:
                        st.caption(field.get('table', ''))
                    with cols[2]:
                        if st.button("➕", key=f"add_secondary_{field['name']}"):
                            st.session_state.selected_fields['secondary'].remove(field)
                            st.session_state.selected_fields['primary'].append(field)
                            st.rerun()

        # Custom field addition
        st.markdown("### ➕ Add Custom Field")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            custom_name = st.text_input("Field Name", key="custom_field_name")
        with col2:
            custom_type = st.selectbox("Type", ["measure", "dimension", "attribute"], key="custom_field_type")
        with col3:
            custom_desc = st.text_input("Description", key="custom_field_desc")
        with col4:
            if st.button("Add Field", key="add_custom_field"):
                if custom_name:
                    new_field = {
                        'name': custom_name,
                        'type': custom_type,
                        'description': custom_desc,
                        'table': 'custom'
                    }
                    st.session_state.selected_fields['secondary'].append(new_field)
                    st.rerun()

    def render_erd_diagram(self):
        """Render interactive ERD diagram using Plotly"""
        st.markdown("### 🔗 Entity Relationship Diagram")

        if not st.session_state.mart_plan:
            st.info("Generate a mart plan first to see the ERD")
            return

        # Create interactive ERD using Plotly
        fig = go.Figure()

        # Node positions (simple layout)
        nodes = []
        edges = []

        # Add fact tables
        y_pos = 0
        for i, fact in enumerate(st.session_state.mart_plan.facts):
            x_pos = i * 3
            nodes.append({
                'name': fact.name,
                'type': 'fact',
                'x': x_pos,
                'y': y_pos,
                'fields': [m.name for m in fact.measures]
            })

        # Add dimension tables
        y_pos = 3
        for i, dim in enumerate(st.session_state.mart_plan.dimensions):
            x_pos = i * 3
            nodes.append({
                'name': dim.name,
                'type': 'dimension',
                'x': x_pos,
                'y': y_pos,
                'fields': dim.attributes
            })

            # Add edges to fact tables
            for fact in st.session_state.mart_plan.facts:
                if dim.key_column in fact.dimension_keys:
                    edges.append((fact.name, dim.name))

        # Draw nodes
        for node in nodes:
            color = '#FFE4B5' if node['type'] == 'fact' else '#E6F3FF'
            border_color = '#FFA500' if node['type'] == 'fact' else '#4169E1'

            # Node box
            fig.add_shape(
                type="rect",
                x0=node['x']-1, x1=node['x']+1,
                y0=node['y']-0.5, y1=node['y']+0.5,
                fillcolor=color,
                line=dict(color=border_color, width=2)
            )

            # Node label
            fig.add_annotation(
                x=node['x'], y=node['y'],
                text=f"<b>{node['name']}</b>",
                showarrow=False,
                font=dict(size=12)
            )

            # Field list (hover text)
            hover_text = f"<b>{node['name']}</b><br>" + "<br>".join(node['fields'][:5])
            if len(node['fields']) > 5:
                hover_text += f"<br>... and {len(node['fields'])-5} more"

            fig.add_trace(go.Scatter(
                x=[node['x']], y=[node['y']],
                mode='markers',
                marker=dict(size=0.1, opacity=0),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

        # Draw edges
        for source, target in edges:
            source_node = next(n for n in nodes if n['name'] == source)
            target_node = next(n for n in nodes if n['name'] == target)

            fig.add_shape(
                type="line",
                x0=source_node['x'], y0=source_node['y'],
                x1=target_node['x'], y1=target_node['y'],
                line=dict(color="gray", width=1, dash="dash")
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            hoverlabel=dict(bgcolor="white", font_size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display legend
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🟧 **Fact Tables**: Contains measures and metrics")
        with col2:
            st.markdown("🟦 **Dimension Tables**: Contains descriptive attributes")

    def render_plan_details(self):
        """Render detailed mart plan information"""
        if not st.session_state.mart_plan:
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Fact Tables")
            for fact in st.session_state.mart_plan.facts:
                with st.expander(f"📈 {fact.name}"):
                    st.markdown(f"**Description:** {fact.description}")
                    st.markdown(f"**Grain:** {', '.join(fact.grain)}")
                    st.markdown("**Measures:**")
                    for measure in fact.measures:
                        st.markdown(f"- **{measure.name}**: {measure.expression} ({measure.aggregation})")
                        if measure.description:
                            st.caption(f"  {measure.description}")
                    st.markdown(f"**Source Tables:** {', '.join(fact.source_tables)}")

        with col2:
            st.markdown("### 🏷️ Dimension Tables")
            for dim in st.session_state.mart_plan.dimensions:
                with st.expander(f"📁 {dim.name}"):
                    st.markdown(f"**Description:** {dim.description}")
                    st.markdown(f"**Key Column:** {dim.key_column}")
                    st.markdown(f"**Attributes:** {', '.join(dim.attributes)}")
                    st.markdown(f"**Source Table:** {dim.source_table}")

    def render_validation(self):
        """Render mart plan validation results"""
        if not st.session_state.mart_plan:
            return

        st.markdown("### ✅ Plan Validation")

        # Run validation only once and cache the result
        if 'validation_result' not in st.session_state:
            with st.spinner("Validating mart plan..."):
                st.session_state.validation_result = self.agent.validate_mart_plan(st.session_state.mart_plan)

        validation_result = st.session_state.validation_result

        if validation_result.is_valid:
            st.success("✅ Mart plan is valid and ready for execution")
        else:
            st.error("❌ Validation found issues that need attention")

        # Display validation details
        col1, col2, col3 = st.columns(3)

        with col1:
            if validation_result.errors:
                st.markdown("#### 🔴 Errors")
                for error in validation_result.errors:
                    st.markdown(f"- {error}")

        with col2:
            if validation_result.warnings:
                st.markdown("#### 🟡 Warnings")
                for warning in validation_result.warnings:
                    st.markdown(f"- {warning}")

        with col3:
            if validation_result.suggestions:
                st.markdown("#### 💡 Suggestions")
                for suggestion in validation_result.suggestions:
                    st.markdown(f"- {suggestion}")

    def render_execution_controls(self):
        """Render mart execution controls"""
        if not st.session_state.mart_plan:
            st.info("📋 Generate and validate a mart plan first")
            return

        st.markdown("## 🚀 Execute Mart Plan")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Data Sampling")
            sampling_method = st.radio(
                "Limit data for testing",
                ["No limit", "By date range", "By row count"],
                key="sampling_method"
            )

            if sampling_method == "By date range":
                date_col = st.selectbox("Date column", ["order_date", "shipped_date"], key="date_column")
                date_range = st.date_input(
                    "Date range",
                    value=(date.today().replace(day=1), date.today()),
                    key="date_range"
                )
            elif sampling_method == "By row count":
                row_limit = st.number_input("Maximum rows", min_value=100, max_value=100000, value=10000, key="row_limit")

        with col2:
            st.markdown("### ⚙️ Execution Options")
            create_indexes = st.checkbox("Create indexes", value=True, key="create_indexes")
            add_constraints = st.checkbox("Add foreign key constraints", value=True, key="add_constraints")
            analyze_tables = st.checkbox("Run ANALYZE after load", value=True, key="analyze_tables")

        with col3:
            st.markdown("### 📝 Target Schema")
            schema_name = st.text_input("Schema name", value=st.session_state.mart_plan.target_schema, key="target_schema")

            if st.button("🔨 Create Data Mart", type="primary", use_container_width=True):
                self.execute_mart_creation()

    def execute_mart_creation(self):
        """Execute the mart creation process"""
        if self.demo_mode:
            # Simulate mart creation in demo mode
            with st.spinner("Creating data mart structure... Populating tables... Generating indexes..."):
                import time
                progress_bar = st.progress(0)
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(0.01)

                st.session_state.mart_created = True
                st.success("✅ Data mart created successfully (Demo Mode)")

                # Generate sample KPI queries
                st.session_state.kpi_queries = self.generate_sample_kpi_queries()
        else:
            # Actual mart creation would go here
            st.info("🚧 Mart execution coming soon...")

    def generate_sample_kpi_queries(self):
        """Generate sample KPI queries based on the mart plan"""
        queries = []

        if st.session_state.mart_plan and st.session_state.mart_plan.facts:
            fact = st.session_state.mart_plan.facts[0]

            # Generate basic aggregation query
            queries.append({
                'name': 'Total Sales by Period',
                'query': f"""
SELECT
    DATE_TRUNC('month', order_date) as period,
    SUM(gross_sales) as total_sales,
    COUNT(DISTINCT order_id) as order_count
FROM {st.session_state.mart_plan.target_schema}.{fact.name}
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY period DESC
LIMIT 12;"""
            })

            # Generate top products query
            queries.append({
                'name': 'Top Products by Revenue',
                'query': f"""
SELECT
    p.product_name,
    SUM(f.gross_sales) as total_revenue,
    SUM(f.quantity_sold) as units_sold
FROM {st.session_state.mart_plan.target_schema}.{fact.name} f
JOIN {st.session_state.mart_plan.target_schema}.dim_product p ON f.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_revenue DESC
LIMIT 10;"""
            })

        return queries

    def render_kpi_queries(self):
        """Render KPI query generation and testing interface"""
        if not st.session_state.mart_created:
            st.info("🏗️ Create the data mart first to generate KPI queries")
            return

        st.markdown("## 📈 KPI Queries")

        tabs = st.tabs(["🎯 Generated Queries", "✏️ Custom Query", "📊 Query Results"])

        with tabs[0]:
            st.markdown("### Generated KPI Queries")
            for i, query_info in enumerate(st.session_state.kpi_queries):
                with st.expander(f"📊 {query_info['name']}"):
                    st.code(query_info['query'], language='sql')
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button(f"Run Query", key=f"run_query_{i}"):
                            st.info("Query execution would run here")
                    with col2:
                        if st.button(f"Copy to Custom", key=f"copy_query_{i}"):
                            st.session_state.custom_query = query_info['query']
                            st.rerun()

        with tabs[1]:
            st.markdown("### Custom Query Editor")
            custom_query = st.text_area(
                "SQL Query",
                value=st.session_state.get('custom_query', ''),
                height=200,
                key="custom_query_input"
            )
            if st.button("🔍 Execute Query", key="execute_custom"):
                st.info("Custom query execution would run here")

        with tabs[2]:
            st.markdown("### Query Results")
            # Placeholder for query results
            st.info("Query results would be displayed here")

    def render_export_section(self):
        """Render export options for production pipeline"""
        st.markdown("## 📦 Export for Production")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📄 Export Mart Plan")
            if st.button("💾 Download JSON Plan", use_container_width=True):
                if st.session_state.mart_plan:
                    json_str = json.dumps(st.session_state.mart_plan.dict(), indent=2)
                    st.download_button(
                        label="📥 Download mart_plan.json",
                        data=json_str,
                        file_name=f"mart_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        with col2:
            st.markdown("### 📜 Export DDL Scripts")
            if st.button("💾 Generate SQL Scripts", use_container_width=True):
                st.info("DDL script generation would be implemented here")

        with col3:
            st.markdown("### 🚀 Pipeline Integration")
            pipeline_format = st.selectbox(
                "Export format",
                ["Airflow DAG", "dbt Models", "Prefect Flow", "Custom JSON"],
                key="pipeline_format"
            )
            if st.button("💾 Export Pipeline", use_container_width=True):
                st.info(f"{pipeline_format} export would be implemented here")

    def render_sidebar(self):
        """Render sidebar with history and settings"""
        with st.sidebar:
            st.markdown("## 📚 Exploration History")

            # Add current plan to history if created
            if st.session_state.mart_created and st.session_state.mart_plan:
                plan_summary = {
                    'timestamp': datetime.now(),
                    'name': st.session_state.mart_plan.target_schema,
                    'tables': len(st.session_state.mart_plan.facts) + len(st.session_state.mart_plan.dimensions)
                }
                if plan_summary not in st.session_state.execution_history:
                    st.session_state.execution_history.append(plan_summary)

            # Display history
            for item in st.session_state.execution_history[-5:]:
                with st.expander(f"📅 {item['timestamp'].strftime('%H:%M')} - {item['name']}"):
                    st.write(f"Tables created: {item['tables']}")

            st.markdown("---")

            st.markdown("## ⚙️ Settings")

            # Model preferences
            st.markdown("### 🤖 Model Preferences")
            use_gpt5 = st.checkbox("Use GPT-5 for planning", value=True, key="use_gpt5_planning")
            use_gpt5_mini = st.checkbox("Use GPT-5-mini for validation", value=True, key="use_gpt5_mini_validation")

            # Reasoning effort control for GPT-5 models
            if use_gpt5:
                reasoning_effort = st.selectbox(
                    "GPT-5 Reasoning Effort",
                    options=["low", "medium", "high"],
                    index=0,  # Default to "low" for faster responses
                    help="Lower effort = faster responses, less deep reasoning",
                    key="reasoning_effort"
                )
            else:
                reasoning_effort = None

            # Search settings
            st.markdown("### 🔍 Search Settings")
            top_k = st.slider("Metadata results", min_value=5, max_value=20, value=10, key="top_k")
            similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, key="similarity_threshold")

            st.markdown("---")

            # Help section
            with st.expander("📖 Help & Documentation"):
                st.markdown("""
                ### How to Use
                1. **Define KPI**: Describe your business metrics
                2. **Generate Plan**: Let GPT-5 create optimal structure
                3. **Customize Fields**: Add/remove fields as needed
                4. **Validate**: Review the proposed structure
                5. **Execute**: Create and populate the mart
                6. **Test Queries**: Run KPI queries
                7. **Export**: Download for production

                ### Tips
                - Be specific about time periods and dimensions
                - Use templates for common scenarios
                - Review the ERD before execution
                - Test with limited data first
                """)

    def run(self):
        """Main application flow"""
        self.render_header()
        self.render_sidebar()

        # Main workflow
        kpi_requirement = self.render_kpi_input()

        if kpi_requirement:
            self.render_mart_planning(kpi_requirement)

            if st.session_state.mart_plan:
                st.markdown("---")
                self.render_execution_controls()

                if st.session_state.mart_created:
                    st.markdown("---")
                    self.render_kpi_queries()
                    st.markdown("---")
                    self.render_export_section()

def main():
    """Main entry point"""
    app = MartExplorerApp()
    app.run()

if __name__ == "__main__":
    main()