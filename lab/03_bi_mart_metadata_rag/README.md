# Lab 03: Business Intelligence Mart with Metadata RAG

This lab demonstrates an end-to-end pipeline from CSV ingestion to automated data mart creation using RAG-powered metadata discovery and LLM-based schema planning. The demo uses the Northwind retail dataset and showcases how RAG can accelerate BI development by automatically identifying relevant relationships and suggesting optimal mart structures.

## 🎯 Demo Overview

**What this lab shows (2-3 minutes live execution):**

1. **Data Ingestion**: Download Northwind CSVs → Load into source schema (`src_northwind`)
2. **Metadata Discovery**: Scan PostgreSQL catalog → Generate semantic embeddings with pgvector
3. **RAG Query**: User asks business question → Retrieve relevant schema elements
4. **Agent Planning**: LLM generates MartPlan JSON with facts/dimensions/measures
5. **Mart Creation**: Execute plan → Create optimized mart schema → Populate with CTAS
6. **KPI Execution**: Run analytical queries on the new mart

## 🏗️ Architecture

### Key Components

- **pgvector Extension**: Vector similarity search infrastructure
- **Metadata Scanner**: PostgreSQL catalog introspection with statistical analysis
- **RAG Search**: Semantic search across database metadata using embeddings
- **LLM Agent**: GPT-5 powered mart planning with GPT-5-mini for fast validation and explanations
- **Mart Executor**: DDL/DML generation and execution engine
- **KPI Generator**: Business metrics query generation

### Data Flow

```
Northwind CSV → PostgreSQL → Metadata Extraction → Vector Embeddings
     ↓
Business Question → RAG Search → LLM Planning → Mart Generation → KPIs
```

## 📁 Directory Structure

```
lab/03_bi_mart_metadata_rag/
├── README.md                          # This file
├── data/                              # Downloaded CSV files (Git LFS)
├── sql/                               # Database setup scripts
│   ├── 00_setup_extensions.sql       # Enable pgvector, etc.
│   ├── 01_create_source_schema.sql   # Northwind schema DDL
│   ├── 02_create_metadata_schema.sql # Catalog schema for metadata
│   └── 03_sample_kpis.sql            # Example KPI queries
├── python/                           # Main execution scripts
│   ├── 00_download_northwind.py      # Download CSV data
│   ├── 10_load_csv_to_pg.py         # Load CSV → PostgreSQL
│   ├── 20_scan_metadata.py          # Extract & analyze metadata
│   ├── 30_embed_metadata.py         # Generate embeddings
│   ├── 40_metadata_rag_search.py    # Interactive RAG search
│   ├── 50_mart_planning_agent.py    # LLM mart planning
│   ├── 60_mart_executor.py          # Execute mart plans
│   ├── 70_kpi_generator.py          # Generate KPI queries
│   └── 80_streamlit_demo.py         # Web interface
├── services/                         # Core service modules
│   ├── mart_embedding_service.py     # Metadata embedding generation
│   ├── metadata_search_service.py    # RAG search implementation
│   └── mart_agent_service.py         # LLM planning agent
├── models/                           # Pydantic data models
│   └── mart_plan.py                  # MartPlan JSON schema
└── samples/                          # Generated plans & examples
    └── mart_plan_example.json        # Sample mart plan
```

## 🚀 Quick Start

### Prerequisites

1. **PostgreSQL** with pgvector extension
2. **Python 3.8+** with required packages
3. **OpenAI API Key** for embeddings and LLM

### Environment Setup

```bash
# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/dbname"
export OPENAI_API_KEY="your_openai_api_key_here"

# Install Python dependencies
pip install psycopg2-binary openai pgvector pydantic streamlit
```

### Step-by-Step Execution

```bash
cd lab/03_bi_mart_metadata_rag

# 1. Download and load Northwind data
python3 python/00_download_northwind.py
python3 python/10_load_csv_to_pg.py

# 2. Extract and embed metadata
python3 python/20_scan_metadata.py
python3 python/30_embed_metadata.py

# 3. Test RAG search (interactive)
python3 python/40_metadata_rag_search.py

# 4. Generate mart plans (interactive)
python3 python/50_mart_planning_agent.py

# 5. Execute mart creation
python3 python/60_mart_executor.py samples/mart_plan_xxxxx.json

# 6. Generate and run KPIs
python3 python/70_kpi_generator.py mart_sales

# 7. Launch web demo
python3 python/80_streamlit_demo.py
```

## 💡 Sample Business Questions

The system can handle various types of business questions:

### Sales Analytics
- "What are the fastest-selling products and their revenue contribution?"
- "How can I analyze customer purchasing patterns and lifetime value?"
- "What are the most profitable product categories and their trends?"

### Inventory Management
- "What metrics should I track for inventory turnover by category?"
- "How can I identify seasonal trends in product sales?"

### Performance Analysis
- "How can I compare sales performance across different regions and time periods?"
- "How can I analyze employee sales performance and commission calculations?"

### Operational Metrics
- "What shipping costs and delivery performance metrics should I track?"
- "How can I analyze discount effectiveness and pricing strategies?"

## 🔍 RAG Search Examples

### Query Types Supported

1. **Metric Queries**: Looking for measures, KPIs, numerical columns
2. **Dimension Queries**: Looking for grouping attributes, categorical data
3. **Relationship Queries**: Looking for joins, foreign keys, dependencies
4. **Table Queries**: Looking for specific entities or data sources

### Sample RAG Interactions

```bash
# Interactive search
python3 python/40_metadata_rag_search.py

🤔 Your question: show me sales-related columns
📊 COLUMN RESULTS (8):
1. [████████████████████] 0.892
   Column: order_details.unit_price
   Type: numeric [FK -> orders]
   Description: Price per unit at time of order

2. [███████████████████] 0.874
   Column: order_details.quantity
   Type: integer
   Description: Number of units ordered
```

## 🏗️ Mart Planning Examples

### Input Business Question
> "What are the fastest-selling products and their revenue contribution?"

### Generated Mart Plan
```json
{
  "source_schema": "src_northwind",
  "target_schema": "mart_sales",
  "facts": [
    {
      "name": "fact_sales",
      "grain": ["order_id", "product_id"],
      "measures": [
        {
          "name": "gross_sales",
          "expression": "unit_price * quantity * (1 - discount)",
          "aggregation": "sum",
          "description": "Total sales revenue"
        },
        {
          "name": "quantity_sold",
          "expression": "quantity",
          "aggregation": "sum",
          "description": "Total units sold"
        }
      ],
      "dimension_keys": ["order_date", "product_id", "customer_id"],
      "source_tables": ["orders", "order_details"],
      "join_conditions": ["orders.order_id = order_details.order_id"]
    }
  ],
  "dimensions": [
    {
      "name": "dim_product",
      "source_table": "products",
      "key_column": "product_id",
      "attributes": ["product_name", "category_id", "unit_price"]
    },
    {
      "name": "dim_customer",
      "source_table": "customers",
      "key_column": "customer_id",
      "attributes": ["company_name", "country", "city"]
    }
  ]
}
```

## 📊 Generated KPI Examples

### Product Velocity Analysis
```sql
-- Fastest-selling products (velocity over last 30 days)
SELECT
    p.product_name,
    SUM(f.quantity_sold) AS total_quantity,
    ROUND(SUM(f.quantity_sold) / 30.0, 2) AS daily_velocity,
    SUM(f.gross_sales) AS revenue_30d
FROM mart_sales.fact_sales f
JOIN mart_sales.dim_product p ON f.product_id = p.product_id
WHERE f.order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY p.product_name
ORDER BY daily_velocity DESC
LIMIT 10;
```

### Revenue Analysis
```sql
-- Top revenue-generating products
SELECT
    p.product_name,
    SUM(f.gross_sales) AS total_revenue,
    COUNT(DISTINCT f.order_id) AS order_count,
    AVG(f.gross_sales) AS avg_order_value
FROM mart_sales.fact_sales f
JOIN mart_sales.dim_product p ON f.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_revenue DESC
LIMIT 20;
```

## 🎨 Web Interface

Launch the Streamlit demo for an interactive web interface:

```bash
python3 python/80_streamlit_demo.py
```

Features:
- **Question Input**: Natural language business questions
- **Real-time Search**: Live metadata search results
- **Mart Visualization**: Interactive mart plan display
- **KPI Generation**: Automatic query generation and execution
- **Export Options**: Download mart plans and query results

## 🔧 Configuration

### Embedding Configuration
- **Model**: text-embedding-3-small (1536 dimensions)
- **Batch Size**: 50 for optimal API usage
- **Similarity Threshold**: 0.7 for relevant results

### LLM Configuration
- **Primary Model**: GPT-5 for complex mart planning with enhanced reasoning
- **Fast Model**: GPT-5-mini for validation, explanations, and quick tasks
- **Fallback Model**: GPT-4 for compatibility when GPT-5 unavailable
- **Max Completion Tokens**: 8000 for GPT-5/GPT-5-mini (optimized for reasoning + output)
- **Reasoning Effort**: Configurable (low/medium/high) - default: low for demos

#### Task Routing Strategy (Configurable in UI)
- **Complex Planning**: GPT-5 or GPT-5-mini (selectable in Streamlit UI)
- **Plan Validation**: GPT-5-mini (fast, thorough validation)
- **Plan Explanation**: GPT-5-mini (business-friendly descriptions)
- **Error Analysis**: GPT-5 (deep problem diagnosis)
- **Optimization**: GPT-5 (performance recommendations)

#### Model Selection (Streamlit UI)
The Streamlit interface allows real-time model switching:
- ☑/☐ **Use GPT-5 for planning**: Toggle between GPT-5 and GPT-5-mini
- ☑/☐ **Use GPT-5-mini for validation**: Toggle validation model
- **Reasoning Effort**: Select low/medium/high for GPT-5 (affects speed/quality)

**Recommended for Demos:**
- Planning: GPT-5-mini (faster, ~10-15s, $0.004/query)
- Validation: GPT-5-mini
- Reasoning Effort: low

### Database Configuration
- **Source Schema**: src_northwind (Northwind tables)
- **Catalog Schema**: catalog (metadata with embeddings)
- **Mart Schemas**: mart_* (generated data marts)

## 🧪 Testing & Validation

### Metadata Validation
```bash
# Check metadata extraction
python3 -c "
from services.metadata_search_service import MetadataSearchService
service = MetadataSearchService()
results = service.search_metadata('sales revenue', None)
print(f'Found {len(results[1])} results')
"
```

### Plan Validation
```bash
# Validate generated plans
python3 -c "
from models.mart_plan import MartPlan
from services.mart_agent_service import MartPlanningAgent

agent = MartPlanningAgent()
plan = MartPlan.parse_file('samples/mart_plan_xxxxx.json')
validation = agent.validate_mart_plan(plan)
print(f'Valid: {validation.is_valid}')
"
```

## 🚀 Demo Script (Live Presentation)

### Setup (30 seconds)
```bash
# Ensure data is loaded and embeddings are ready
python3 python/20_scan_metadata.py --check
python3 python/30_embed_metadata.py --status
```

### Interactive Demo (90 seconds)
1. **Ask Business Question**: "What metrics should I track for fastest-selling products?"
2. **Show RAG Results**: Display top-10 relevant schema elements (powered by embeddings)
3. **Generate MartPlan**: GPT-5 creates sophisticated JSON plan with facts/dimensions
4. **Validate Plan**: GPT-5-mini provides instant validation and suggestions
5. **Execute Plan**: Create mart schema and populate (simulation)
6. **Run KPIs**: Show fastest-selling products analysis with GPT-5-mini explanations

### Performance Comparison (30 seconds)
- Show query performance: source tables vs. optimized mart
- Demonstrate mart indexing benefits
- Highlight business-friendly column names

## 🔍 Key Differentiators

1. **Metadata-Centric RAG**: Searches database structure, not documents
2. **Generative BI**: LLM generates entire data mart schemas
3. **Catalog-Driven**: Uses PostgreSQL's system tables as knowledge base
4. **Business-Oriented**: Focuses on KPIs and business metrics
5. **Live Schema Evolution**: Real-time database schema creation
6. **GPT-5 Enhanced Planning**: Leverages advanced reasoning for superior mart designs

## 🧠 GPT-5 Integration Benefits

### Enhanced Capabilities
- **Deep Analytical Reasoning**: GPT-5's advanced reasoning improves business requirement analysis
- **Pattern Recognition**: Better identification of optimal dimensional modeling patterns
- **Performance Optimization**: Sophisticated understanding of query performance implications
- **Structured Output**: Enhanced JSON generation with better validation

### Intelligent Task Routing
- **GPT-5 for Complex Tasks**:
  - Mart schema generation with sophisticated business logic
  - Complex error analysis and troubleshooting
  - Performance optimization recommendations

- **GPT-5-mini for Fast Tasks**:
  - Real-time plan validation (10x faster than GPT-5)
  - Business-friendly explanations
  - Quick consistency checks
  - User interface responses

### Fallback Strategy
- **Automatic Failover**: Falls back to GPT-4 if GPT-5 models unavailable
- **Graceful Degradation**: Maintains full functionality with older models
- **Cost Optimization**: Uses GPT-5-mini for appropriate tasks to reduce costs

## 🎯 Success Metrics

- ✅ Demo completion in under 3 minutes
- ✅ Mart creation without manual SQL
- ✅ KPI queries return in <100ms
- ✅ Audience understanding of RAG→Agent→Deploy pipeline
- ✅ Reusability for other datasets

## 🔗 Integration with Existing Labs

This lab builds upon and integrates with the existing lab infrastructure:

- **pgvector Setup**: Reuses vector similarity search from other labs
- **Embedding Patterns**: Follows embedding generation patterns from Wikipedia/Movie labs
- **Database Patterns**: Uses established PostgreSQL connection and query patterns
- **Configuration**: Integrates with existing environment variable patterns

## 🛠️ Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL is running
   pg_isready

   # Test connection
   psql $DATABASE_URL -c "SELECT version();"
   ```

3. **pgvector Extension Missing**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Embedding Generation Fails**
   ```bash
   # Check API quota and limits
   python3 -c "import openai; print(openai.api_key[:10] + '...')"
   ```

5. **"Empty completion content" Errors**
   - **Cause**: GPT-5 models using all tokens for reasoning
   - **Solution**: Token limits have been increased to 8000 (already applied)
   - **If persists**: Increase further in `services/mart_agent_service.py` line 154
   - **Alternative**: Use GPT-5-mini with reasoning_effort="low" in UI

6. **Pydantic Deprecation Warnings**
   - **Cause**: Using Pydantic V1 syntax with V2 installed
   - **Solution**: Already fixed in v1.1 - ensure you have latest code
   - **Check version**: `python3 -c "import pydantic; print(pydantic.__version__)"`

7. **Streamlit Model Selection Not Working**
   - **Cause**: Cached agent not picking up new settings
   - **Solution**: Restart Streamlit (Ctrl+C and rerun)
   - **Alternative**: Clear cache: Press "C" in terminal or UI menu → "Clear cache"

8. **Validation Running Forever**
   - **Cause**: Fixed in v1.1 - validation now cached
   - **Solution**: Update to latest code and restart Streamlit

## 📈 Future Enhancements

- **Incremental Refresh**: CDC-based mart updates
- **Multi-Source Integration**: Join across different databases
- **Semantic Layer**: Business glossary integration
- **Query Optimization**: Cost-based plan selection
- **Monitoring Dashboard**: Track mart usage and performance

## 📚 References

- [Northwind Database](https://github.com/pthom/northwind_psql)
- [PostgreSQL System Catalogs](https://www.postgresql.org/docs/current/catalogs.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Dimensional Modeling (Kimball)](https://www.kimballgroup.com/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)

---

This lab provides a comprehensive blueprint for implementing automated BI mart generation using the power of RAG and LLM technologies, demonstrating the future of data warehouse automation.