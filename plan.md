# Repository Evolution Plan for Conference Presentation
## "What the RAG? From Naive to Advanced RAG Search"

## Executive Summary
Transform the current Movies_pgvector_lab repository into a comprehensive demonstration platform for PostgreSQL conferences, showcasing the evolution from naive to advanced RAG search techniques using pgvector and pgvectorscale's latest capabilities.

## Key Objectives
1. Create a unified lab environment supporting both educational purposes and live conference demonstrations
2. Integrate n8n workflows (Docker) alongside LangChain for comprehensive pipeline demonstrations
3. Provide hands-on LAB materials (publicly available) for conference attendees
4. Demonstrate performance comparisons across search techniques (LIKE → FTS → Dense → Sparse → Hybrid)
5. Showcase pgvector's sparsevec capabilities and pgvectorscale's DiskANN indexing
6. Implement Streamlit UI for interactive search comparison and visualization
7. Demonstrate context window optimization strategies for efficient RAG workflows (critical even with 400K window)

## Proposed Architecture

### Core Components
1. **Enhanced Wikipedia Dataset** (Primary Focus)
   - Maintain existing 25,000 articles with embeddings
   - Add comprehensive benchmarking queries for DBA-specific topics
   - Include performance metrics and evaluation framework
   - Implement context window optimization strategies

2. **Search Pipeline Comparisons**
   - LIKE pattern matching (baseline)
   - PostgreSQL Full-Text Search (FTS with BM25)
   - Dense embeddings (OpenAI text-embedding-3-small/large)
   - Sparse embeddings (SPLADE with sparsevec)
   - Hybrid approaches with adaptive weighting
   - Rerankers (cross-encoders) for result optimization

3. **Integration Stack**
   - **Streamlit**: Interactive web UI for search comparison and visualization
   - **LangChain**: Production-ready RAG pipeline orchestration
   - **n8n** (Docker): Visual workflow automation for demonstrations
   - **FastAPI Service**: REST API for programmatic access
   - **Command-line tools**: Direct Python scripts for testing

4. **Context Window Optimization**
   - Chunk size optimization strategies
   - Semantic chunking with overlap
   - Dynamic context selection based on relevance
   - Token counting and management
   - Compression techniques for long documents

## Implementation Phases

### Phase 1: Repository Restructuring

#### Directory Structure
```
pgvector_rag_lab/
├── lab/                          # Public conference lab materials
│   ├── 01_setup/                 # Environment setup
│   │   ├── setup.sql             # Schema + extensions (pgvector + pgvectorscale)
│   │   ├── requirements.txt      # Python venv dependencies
│   │   ├── docker-compose.yml    # n8n Docker setup
│   │   └── install.sh            # Automated setup script
│   ├── 02_data/                  # Data management
│   │   ├── load_wikipedia.sql
│   │   ├── sample_queries.txt    # DBA-focused test queries
│   │   └── chunk_strategies.py   # Context window optimization
│   ├── 03_embeddings/             # Embedding generation
│   │   ├── embed_dense.py
│   │   ├── embed_sparse.py
│   │   ├── batch_processor.py
│   │   └── token_counter.py      # Token management utilities
│   ├── 04_search/                 # Search implementations
│   │   ├── compare_searches.py   # All search methods comparison
│   │   ├── naive_rag.py
│   │   ├── hybrid_rag.py
│   │   ├── adaptive_rag.py       # Query classification + routing
│   │   └── langchain_rag.py      # LangChain implementation
│   ├── 05_api/                   # Service layer
│   │   ├── rag_service.py        # FastAPI endpoints
│   │   ├── streamlit_app.py      # Interactive UI
│   │   └── openapi.json
│   ├── 06_workflows/              # n8n integration
│   │   ├── docker-compose.yml    # n8n Docker setup
│   │   ├── n8n_workflows/
│   │   │   ├── naive_rag.json
│   │   │   ├── hybrid_rag.json
│   │   │   └── adaptive_rag.json
│   │   └── README.md
│   └── 07_evaluation/             # Performance metrics
│       ├── benchmark.py
│       ├── metrics.py
│       ├── context_analysis.py   # Context window usage analysis
│       └── results/               # Public benchmark results
├── original/                      # Legacy movie/Netflix code
│   └── [existing scripts]
└── docs/                          # Public documentation
    ├── LAB_INSTRUCTIONS.md
    ├── CONTEXT_OPTIMIZATION.md
    ├── TROUBLESHOOTING.md
    └── SETUP_GUIDE.md
```

#### Key Changes
1. Move existing movie/Netflix scripts to `original/` directory
2. Create new `lab/` structure with public materials for attendees
3. Implement Streamlit UI for interactive demonstration
4. Add LangChain for production-ready RAG pipelines
5. Include context window optimization utilities
6. Use Docker for n8n, host-level for PostgreSQL/pgvector/pgvectorscale

### Phase 2: Core Functionality Enhancement

#### 2.1 Search Comparison Framework
```python
# lab/04_search/compare_searches.py
class SearchComparator:
    - like_search()
    - fts_search()
    - dense_search()
    - sparse_search()
    - hybrid_search()
    - adaptive_hybrid_search()
    - measure_performance()
    - generate_report()
```

#### 2.2 Query Classification System
```python
# lab/04_search/query_classifier.py
class QueryClassifier:
    - classify_query() -> QueryType (factual/conceptual/exploratory/structured)
    - optimize_weights() -> (dense_weight, sparse_weight)
    - route_query() -> SearchMethod
```

#### 2.3 Performance Monitoring
```python
# lab/07_evaluation/benchmark.py
class BenchmarkSuite:
    - measure_latency()
    - calculate_recall_at_k()
    - compute_mrr()
    - analyze_index_usage()
    - generate_explain_plans()
```

### Phase 3: Integration Layer Development

#### 3.1 Streamlit Interactive UI (Inspired by RAG-essentials Capstone Lab)
Create a comprehensive web interface with:

**Search Page Features:**
1. **Search Method Selection**: Radio buttons for LIKE/FTS/Dense/Sparse/Hybrid/Adaptive/RAG-Open/RAG-Context-Only
2. **Parameter Tuning**: Sliders for alpha weights, k values, chunk sizes, page size
3. **Results Visualization**: 
   - Side-by-side comparison of different methods
   - Source documents with relevance scores
   - Pagination for large result sets
4. **Real-time Performance Metrics**:
   - Embedding generation time (ms)
   - Database query time (ms)
   - LLM generation time (ms)
   - Total execution time (ms)
   - Visual breakdown bar chart showing time distribution
5. **Context Analysis**: 
   - Token usage counter
   - Chunk distribution visualization
   - Cost estimation per query
6. **EXPLAIN Plans**: Collapsible sections showing PostgreSQL query plans

**Metrics Dashboard Page:**
1. **Historical Metrics Table**: All recorded search queries with performance data
2. **Metric Descriptions**: Hover tooltips explaining each metric
3. **Performance Trends**: Charts showing latency patterns over time
4. **Comparison Views**: Compare metrics across different search modes
5. **Export Capabilities**: Download metrics as CSV/JSON

#### 3.2 LangChain Integration
Implement production-ready pipelines:
1. **Document Loaders**: Wikipedia articles with smart chunking
2. **Text Splitters**: Multiple strategies (fixed, semantic, recursive)
3. **Vector Stores**: PGVector integration with metadata filtering
4. **Retrievers**: Dense, sparse, and hybrid implementations
5. **Chains**: RetrievalQA, ConversationalRetrievalChain
6. **Memory**: Buffer and summary memory for conversations

#### 3.3 n8n Workflows (Docker)
Visual workflow demonstrations:
1. **Naive RAG Flow**: HTTP → Embed → Search → GPT-5-mini → Response
2. **Hybrid RAG Flow**: HTTP → Classify → Parallel Search → Fusion → Rerank → Response
3. **Context-Optimized Flow**: HTTP → Chunk Analysis → Smart Retrieval → Compressed Context → Response

### Phase 4: Lab Materials Development

#### 4.1 Hands-on Exercises
1. **Exercise 1**: Set up pgvector with sparsevec
2. **Exercise 2**: Compare search methods on DBA queries
3. **Exercise 3**: Build and tune HNSW vs IVFFlat vs DiskANN indexes
4. **Exercise 4**: Implement hybrid search with custom weights
5. **Exercise 5**: Deploy n8n workflow for production RAG

#### 4.2 Sample Queries (DBA-focused)
```
- "What is WAL in PostgreSQL and why is it important?"
- "How does MVCC reduce locking?"
- "Explain checkpoint tuning for write-heavy workloads"
- "Compare HNSW vs IVFFlat indexing strategies"
- "When to use VACUUM vs VACUUM FULL"
- "How to diagnose and fix bloat in PostgreSQL"
```

#### 4.3 Performance Scenarios
- Small dataset (1K docs): Compare all methods
- Medium dataset (25K docs): Current Wikipedia set
- Large dataset simulation: DiskANN advantages
- High concurrency: Connection pooling impact

### Phase 5: Demo Optimization and Testing

#### 5.1 Live Demo Script (12-15 minutes)
1. **Minutes 1-3**: Problem statement - LLMs are static, need fresh data
2. **Minutes 4-6**: Search spectrum demonstration (LIKE → FTS → Vector)
3. **Minutes 7-9**: Hybrid search live with n8n visual flow
4. **Minutes 10-12**: Performance comparison with EXPLAIN ANALYZE
5. **Minutes 13-15**: Q&A buffer

#### 5.2 Fallback Strategies
- Pre-recorded video backup for network issues
- Local n8n instance (no cloud dependencies)
- Cached embeddings to avoid API calls
- Sample results JSON for worst-case scenarios

## Technical Specifications

### Database Requirements
- PostgreSQL 17.x (host-level installation)
- pgvector 0.8+ (for sparsevec support)
- pgvectorscale (required for DiskANN indexing)
- Full-text search configuration (english)

### Database Schema (Based on RAG-essentials Best Practices)

#### 1. Main Articles Table with Vector Extensions
```sql
-- Wikipedia articles with vector and full-text capabilities
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    content TEXT NOT NULL,
    
    -- Vector embeddings (pgvector)
    title_vector vector(1536),
    content_vector vector(1536),
    
    -- Sparse embeddings (pgvectorscale sparsevec)
    title_sparse sparsevec(30522),
    content_sparse sparsevec(30522),
    
    -- Full-text search vectors
    content_tsv tsvector,
    title_content_tsvector tsvector,
    
    -- Metadata
    vector_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Advanced full-text setup with weighted ranking (titles priority over content)
UPDATE articles 
SET title_content_tsvector = 
    setweight(to_tsvector('english', COALESCE(title, '')), 'A') || 
    setweight(to_tsvector('english', COALESCE(content, '')), 'B');

-- Auto-update trigger for full-text vectors
CREATE OR REPLACE FUNCTION articles_tsvector_trigger() RETURNS trigger AS $$
BEGIN
  NEW.title_content_tsvector := 
    setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') || 
    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON articles FOR EACH ROW EXECUTE FUNCTION articles_tsvector_trigger();
```

#### 2. Performance Indexes (Production-Ready)
```sql
-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_articles_content_tsv 
    ON articles USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_title_content_tsvector 
    ON articles USING GIN (title_content_tsvector);

-- Dense vector indexes (HNSW for speed)
CREATE INDEX IF NOT EXISTS idx_articles_title_vec_hnsw 
    ON articles USING hnsw (title_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_articles_content_vec_hnsw 
    ON articles USING hnsw (content_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- Sparse vector indexes (pgvectorscale)
CREATE INDEX IF NOT EXISTS idx_articles_title_sparse_hnsw 
    ON articles USING hnsw (title_sparse sparsevec_ip_ops) 
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_articles_content_sparse_hnsw 
    ON articles USING hnsw (content_sparse sparsevec_ip_ops) 
    WITH (m = 16, ef_construction = 64);

-- DiskANN indexes for large-scale performance (pgvectorscale required)
CREATE INDEX IF NOT EXISTS idx_articles_content_diskann 
    ON articles USING diskann (content_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_articles_content_sparse_diskann 
    ON articles USING diskann (content_sparse sparsevec_ip_ops);

-- Traditional indexes for exact matches
CREATE INDEX IF NOT EXISTS idx_articles_title_gin 
    ON articles USING GIN (title gin_trgm_ops);
```

#### 3. Metrics Tracking Schema
```sql
-- Enhanced performance metrics table (based on RAG-essentials)
CREATE TABLE IF NOT EXISTS search_metrics (
    log_id SERIAL PRIMARY KEY,
    query_id TEXT,                    -- Hash of the query for grouping
    description TEXT,                 -- First 20 chars of query for readability
    query_time TIMESTAMPTZ DEFAULT NOW(),
    mode TEXT,                        -- Search method used
    top_score REAL,                   -- Best similarity score/distance
    token_usage INTEGER,              -- Total tokens consumed
    precision REAL DEFAULT 0,        -- Proportion of relevant results
    embedding_ms REAL,                -- Embedding generation time
    db_ms REAL,                       -- Database query execution time
    llm_ms REAL,                      -- LLM response generation time
    total_ms REAL,                    -- Total end-to-end latency
    
    -- Extended metrics for advanced analysis
    context_tokens INTEGER,           -- Tokens used in context
    output_tokens INTEGER,            -- Tokens generated in response
    chunk_count INTEGER,              -- Number of chunks retrieved
    rerank_ms REAL,                   -- Reranking execution time
    index_used TEXT,                  -- Which index was used by query planner
    buffer_hits INTEGER,              -- PostgreSQL buffer cache hits
    disk_reads INTEGER                -- Disk reads for performance analysis
);

-- Metric descriptions for UI tooltips (RAG-essentials approach)
CREATE TABLE IF NOT EXISTS metric_descriptions (
    metric_name TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO metric_descriptions(metric_name, description) VALUES
    ('query_id', 'Short hash representing the query text'),
    ('description', 'First 20 characters of the query'),
    ('query_time', 'Timestamp when the query was executed'),
    ('mode', 'Search mode used for this query'),
    ('top_score', 'Best similarity distance or score'),
    ('token_usage', 'Total tokens used in the LLM call'),
    ('precision', 'Proportion of relevant results'),
    ('embedding_ms', 'Milliseconds spent generating the embedding'),
    ('db_ms', 'Milliseconds spent executing the database search'),
    ('llm_ms', 'Milliseconds spent generating the LLM answer'),
    ('total_ms', 'Total execution time in milliseconds'),
    ('context_tokens', 'Number of tokens used in the context'),
    ('output_tokens', 'Number of tokens generated in the response'),
    ('chunk_count', 'Number of document chunks retrieved'),
    ('rerank_ms', 'Time spent on result reranking'),
    ('index_used', 'PostgreSQL index used by the query planner'),
    ('buffer_hits', 'Number of buffer cache hits'),
    ('disk_reads', 'Number of disk reads performed')
ON CONFLICT (metric_name) DO NOTHING;
```

### Python Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install psycopg[binary]>=3.2
pip install openai>=1.30
pip install langchain>=0.1.0
pip install langchain-community
pip install langchain-openai
pip install streamlit>=1.28
pip install fastapi>=0.111
pip install uvicorn>=0.30
pip install torch>=2.0
pip install transformers>=4.30
pip install sentencepiece
pip install tiktoken
pip install tqdm
pip install numpy
pip install pandas
pip install plotly  # for visualization
pip install python-dotenv
```

### n8n Requirements (Docker)
```yaml
# docker-compose.yml for n8n
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=false
      - N8N_HOST=0.0.0.0
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - WEBHOOK_URL=http://localhost:5678/
    volumes:
      - ./n8n_data:/home/node/.n8n
    restart: unless-stopped
```

### Model Specifications
- **Dense**: OpenAI text-embedding-3-small (1536d) or text-embedding-3-large (3072d)
- **Sparse**: naver/splade-cocondenser-ensembledistil (30522d vocab)
- **LLM**: GPT-5-mini (primary for cost efficiency and speed)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-12-v2
- **Context Window**: 400K tokens input, 128K tokens output (GPT-5-mini)
- **Context Optimization**: Still critical for cost reduction, latency, and relevance despite large window

## Configuration Management

### Environment Variables (.env file)
```bash
# PostgreSQL (host-level)
DATABASE_URL=postgresql://user:pass@localhost:5432/pgvector_lab
PG_POOL_SIZE=20
PG_VECTOR_EXTENSION_VERSION=0.8.0
PG_VECTORSCALE_ENABLED=true

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_EMB=text-embedding-3-small
OPENAI_MODEL_CHAT=gpt-5-mini

# SPLADE
SPLADE_MODEL=naver/splade-cocondenser-ensembledistil
SPLADE_DEVICE=cuda  # or cpu

# Hybrid Search
HYBRID_ALPHA=0.5  # Default dense weight
TOPK_RETRIEVAL=50
FINAL_K=10

# Context Window Optimization
MAX_CONTEXT_TOKENS=50000  # Conservative limit for cost efficiency (GPT-5-mini supports 400K)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_OUTPUT_TOKENS=16000  # Conservative output limit (GPT-5-mini supports 128K)

# Streamlit
STREAMLIT_PORT=8501
STREAMLIT_THEME=dark

# n8n (Docker)
N8N_PORT=5678
N8N_WEBHOOK_URL=http://localhost:5678/webhook
```

## Testing Strategy

### Unit Tests
- Embedding generation accuracy
- Query classification correctness
- Score fusion mathematics
- Database connection handling

### Integration Tests
- End-to-end RAG pipeline
- n8n workflow execution
- API endpoint responses
- Index usage verification

### Performance Tests
- Query latency (p50, p95, p99)
- Throughput under load
- Memory usage patterns
- Index build times

## Deployment Options

### Option 1: Hybrid Setup (Recommended for Workshop)
```bash
# Host-level installations
# 1. PostgreSQL 17.x with pgvector and pgvectorscale
sudo apt-get install postgresql-17 postgresql-17-pgvector
# Install pgvectorscale from source or package

# 2. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Docker for n8n only
docker-compose -f lab/06_workflows/docker-compose.yml up -d

# 4. Run Streamlit UI
streamlit run lab/05_api/streamlit_app.py
```

### Option 2: Local Installation (For Development)
- Direct PostgreSQL installation
- Python venv
- n8n via npm or Docker

### Option 3: Cloud Deployment (For Production Demo)
- Managed PostgreSQL (with pgvector)
- n8n.cloud or self-hosted
- API on cloud run/lambda

## Migration Path from Current Repository

### Phase 1 Tasks (Foundation)
1. ✅ Create new directory structure
2. ✅ Move existing scripts to `original/`
3. ✅ Set up git branching (main, development, demo)
4. ✅ Update .gitignore for presentation materials
5. ⬜ Create Python venv setup script
6. ⬜ Install pgvectorscale extension

### Phase 2 Tasks (Core Development)
1. ⬜ Port embedding scripts with context optimization
2. ⬜ Implement search comparison framework
3. ⬜ Create Streamlit UI application
4. ⬜ Integrate LangChain pipelines
5. ⬜ Develop token counting utilities

### Phase 3 Tasks (Integration)
1. ⬜ Set up n8n Docker environment
2. ⬜ Create workflow JSON exports
3. ⬜ Build FastAPI service layer
4. ⬜ Implement context window optimization strategies
5. ⬜ Connect all components

### Phase 4 Tasks (Lab Materials)
1. ⬜ Complete lab exercises
2. ⬜ Develop evaluation framework
3. ⬜ Create demo scripts
4. ⬜ Write context optimization guide
5. ⬜ Test all components end-to-end

### Phase 5 Tasks (Polish and Testing)
1. ⬜ Performance optimization with pgvectorscale
2. ⬜ Documentation completion
3. ⬜ Demo rehearsal with timing
4. ⬜ Package for distribution
5. ⬜ Create fallback scenarios

## Success Metrics

### Technical
- All 5 search methods operational
- <100ms p95 latency for hybrid search
- >0.8 MRR@10 for DBA queries
- Zero API failures during demo

### Educational
- Clear progression from simple to advanced
- Attendees can reproduce locally
- Working n8n workflows exported
- Comprehensive troubleshooting guide

### Presentation
- Smooth 15-minute demo
- Visual n8n flow impresses audience
- Performance gains clearly demonstrated
- Q&A shows understanding

## Risk Mitigation

### Risk 1: API Rate Limits
- **Mitigation**: Pre-computed embeddings, caching layer, fallback to local models

### Risk 2: Network Issues
- **Mitigation**: Local deployment option, cached responses, offline mode

### Risk 3: Database Performance
- **Mitigation**: Pre-warmed indexes, connection pooling, explain plan analysis

### Risk 4: n8n Complexity
- **Mitigation**: Simple flows first, detailed documentation, video tutorials

## Alternative Considerations

### Technology Stack Rationale

#### Streamlit for UI
1. **Interactive**: Real-time parameter adjustment
2. **Visual**: Charts and comparisons built-in
3. **Simple**: Easy for attendees to understand
4. **Pythonic**: Natural fit with existing codebase

#### LangChain for RAG
1. **Production Ready**: Battle-tested components
2. **Modular**: Easy to swap components
3. **Well Documented**: Extensive examples
4. **Community**: Large ecosystem of integrations

#### n8n for Visual Workflows
1. **Visual Appeal**: Great for demonstrations
2. **Docker Based**: Easy deployment
3. **PostgreSQL Native**: Direct database integration
4. **Export/Import**: JSON workflows for sharing

#### Context Window Optimization Focus
1. **Cost Efficiency**: Reduce token usage
2. **Performance**: Faster responses
3. **Quality**: Better relevant context selection
4. **Scalability**: Handle larger documents

## Next Steps

### Immediate Actions (This Week)
1. Review and approve this plan
2. Set up new directory structure
3. Begin porting existing code
4. Install and test n8n locally

### Following Week
1. Implement core search comparisons
2. Build n8n workflows
3. Create initial lab materials
4. Test with sample queries

### Pre-Conference
1. Full rehearsal with timing
2. Backup preparations
3. Attendee materials packaging
4. Final performance tuning

## Detailed Presentation Structure (Based on RAG-essentials Insights)

### 🎯 "What the RAG? From Naive to Advanced RAG Search" - Complete PPT Blueprint

#### **SECTION 1: Hook & Problem Statement (3 minutes)**

##### Slide 1: Title & Hook
**"What the RAG? From Naive to Advanced RAG Search"**  
*Your subtitle: Making PostgreSQL the Heart of Intelligent Systems*

**Opening Hook**: 
> "Your CEO just saw a competitor's AI search and asked why your database search is 'so dumb.' Today we'll fix that."

**Speaker Notes**: 
- Start with relatable DBA frustration
- Set expectation: we'll build something impressive
- Preview the demo we'll end with

##### Slide 2: The Search Evolution Challenge
**The Journey**: LIKE → FTS → Dense → Sparse → Hybrid → Adaptive RAG

**Real Example Progression**:
```sql
-- The CEO's question: "Find articles about the biggest ocean animal"

-- LIKE search: 0 results
WHERE content ILIKE '%biggest ocean animal%'

-- FTS search: 5 results, some relevant  
WHERE content_tsv @@ plainto_tsquery('biggest ocean animal')

-- Vector search: Perfect match - "Blue Whale" articles
ORDER BY embedding <=> query_embedding

-- Hybrid: Best of all worlds with pgvectorscale DiskANN
```

**Key Message**: Each technique has specific use cases, but hybrid approaches win

#### **SECTION 2: Database Foundations (8 minutes)**

##### Slide 3: Why DBAs Should Care About RAG
**The New Reality**:
- LLMs are static (training data cutoff)
- RAG injects fresh, audited data at query time
- Reduces retraining costs, improves factuality
- Context window limitations require optimization strategies
- RAG builds are more future-proof than any 2-week-old LLM model

**DBA Value Proposition**: You're not just storing data anymore—you're enabling intelligent retrieval

##### Slide 4: PostgreSQL Capability Map
**Your Existing Arsenal + New Weapons**:

| Technique | PostgreSQL Tool | Index Type | Use Case |
|-----------|----------------|------------|----------|
| LIKE | Text operators | btree/gin_trgm | Exact matches |
| FTS | tsvector/GIN | GIN | BM25 ranking |
| Dense | pgvector | HNSW/IVFFlat | Semantic similarity |
| Sparse | sparsevec | HNSW | Lexical precision |
| ANN Scale | pgvectorscale | DiskANN | Large-scale performance |

**Live Demo Tease**: "We'll show all of these running on the same query"

##### Slide 5: The pgvector + pgvectorscale Power Combo
**What's New in the Ecosystem**:
- **pgvector 0.8+**: sparsevec support for SPLADE embeddings
- **pgvectorscale**: StreamingDiskANN for production scale
- **GPT-5-mini**: 400K context window, 128K output tokens

**Performance Numbers** (based on Wikipedia 25K articles):
```
HNSW:     ~10-50ms, 97% recall
DiskANN:  ~5-25ms, 98% recall, better memory efficiency
IVFFlat:  ~80ms, 85% recall
```

##### Slide 6: Schema Design for Hybrid Search
**Production-Ready Table Structure**:
```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    
    -- Dense embeddings (OpenAI)
    title_vector vector(1536),
    content_vector vector(1536),
    
    -- Sparse embeddings (SPLADE)  
    title_sparse sparsevec(30522),
    content_sparse sparsevec(30522),
    
    -- FTS vectors with weighted ranking
    title_content_tsvector tsvector -- titles get 'A' weight, content 'B'
);
```

**Index Strategy Demo**: Show index creation and EXPLAIN plans

#### **SECTION 3: The SPLADE Revolution (5 minutes)**

##### Slide 7: Sparse Embeddings 101
**SPLADE: The Best of Both Worlds**
- Produces ~30K-dimensional vectors over vocabulary
- Great for terms, acronyms, exactish matches  
- Pairs perfectly with dense embeddings
- Model: `naver/splade-cocondenser-ensembledistil`

**Why This Matters**:
```sql
-- Dense finds: "AI", "artificial intelligence", "machine learning"
-- Sparse finds: "AI", "AI-powered", "AI/ML", "A.I."
-- Hybrid finds: ALL of the above with proper ranking
```

##### Slide 8: Sparse vs Dense - Live Comparison
**Query**: "What is WAL in PostgreSQL?"

**Dense Results** (semantic):
1. PostgreSQL Architecture (0.15)
2. Database Theory (0.18)  
3. Transaction Processing (0.21)

**Sparse Results** (lexical):
1. Write-Ahead Logging (0.12)
2. WAL Configuration (0.15)
3. PostgreSQL WAL Files (0.18)

**Hybrid Results** (adaptive weights):
1. Write-Ahead Logging in PostgreSQL (0.89 combined)
2. WAL Configuration Best Practices (0.85)
3. Transaction Recovery Using WAL (0.82)

#### **SECTION 4: Context Window Optimization (4 minutes)**

##### Slide 9: The 400K Token Reality
**GPT-5-mini Specs**: 400K input, 128K output tokens
**Challenge**: Even with massive context, optimization still critical for:
- **Cost efficiency**: More tokens = higher API costs
- **Latency**: Larger contexts take longer to process  
- **Quality**: Better context selection improves answers
- **Scalability**: Optimization techniques matter at scale

##### Slide 10: Smart Context Strategies  
**1. Adaptive Chunking**:
```python
# Semantic-aware chunking
chunks = smart_chunker(document, 
    max_tokens=1000,
    overlap=200, 
    strategy='sentence_boundary',
    preserve_structure=True)
```

**2. Relevance-Based Selection**:
```sql
-- Multi-factor context scoring
SELECT content, 
    (sparse <#> query_sparse) * -1 * 0.4 +          -- Lexical match
    (1 - (dense <=> query_dense)) * 0.6 +           -- Semantic match  
    CASE WHEN title ILIKE '%' || query_terms || '%' 
         THEN 0.1 ELSE 0 END +                      -- Title bonus
    LOG(view_count + 1) * 0.05                      -- Popularity boost
    AS context_score
FROM articles 
ORDER BY context_score DESC 
LIMIT 10;
```

#### **SECTION 5: Live Demo Architecture (10 minutes)**

##### Slide 11: Demo Stack Overview
**What We'll Build Live**:
```
┌─ Streamlit UI (Interactive) ─┐
│  ├─ Search method selector   │
│  ├─ Real-time metrics        │  
│  ├─ Performance comparisons  │
│  └─ Context optimization     │
└───────────────────────────────┘
           │
┌─────── FastAPI Backend ──────┐
│  ├─ Query classification     │
│  ├─ Multi-method search      │
│  ├─ Score fusion (RRF)       │
│  └─ Performance tracking     │
└───────────────────────────────┘
           │
┌── PostgreSQL + Extensions ──┐
│  ├─ pgvector (dense+sparse) │
│  ├─ pgvectorscale (DiskANN)  │
│  ├─ Full-text search (FTS)   │
│  └─ Performance metrics DB   │
└───────────────────────────────┘
```

##### Slide 12: Query Classification Demo
**The Demo Question**: "How does MVCC reduce locking in PostgreSQL?"

**Live Classification**:
1. **Factual Query** (dense-heavy: 0.3 sparse, 0.7 dense)
   - Needs conceptual understanding
   - MVCC concepts span multiple articles
   
2. **Hybrid Execution**:
   - Sparse finds: "MVCC", "locking", "PostgreSQL"  
   - Dense finds: concurrency concepts, isolation levels
   - RRF fusion: Reciprocal Rank Fusion for robust ranking

##### Slide 13: Performance Metrics in Real-Time
**Live Metrics Display**:
```
┌── Query: "How does MVCC reduce locking?" ──┐
├─ Embedding Generation: 45ms                │
├─ Sparse Search: 23ms (HNSW)               │  
├─ Dense Search: 31ms (DiskANN)             │
├─ Score Fusion: 8ms                         │
├─ GPT-5-mini Generation: 1,247ms            │  
├─ Total Latency: 1,354ms                    │
├─ Context Tokens: 4,832 / 400,000          │
├─ Output Tokens: 387 / 128,000              │
├─ Cost: $0.0043                            │
└─────────────────────────────────────────────┘
```

**Visual Breakdown**: Horizontal bar chart showing time distribution

##### Slide 14: n8n Workflow Visualization
**Visual Pipeline Demo**:
```
HTTP Request → Query Classifier → Parallel Search → Score Fusion → Context Builder → GPT-5-mini → Response
     │              │                    │               │              │              │           │
   ┌─┘              └─ Factual?         ┌┴┐              │              │              │           │
   │                   Conceptual?    Dense│Sparse       │              │              │           │  
   └─ User Question     Exploratory?      └┬┘             │              │              │           │
                                          │               │              │              │           │
                                    Search Results   RRF Ranking   Token Counting   API Call   Answer + Sources
```

#### **SECTION 6: Production Best Practices (5 minutes)**

##### Slide 15: Hybrid Search Query Patterns
**The Five Search Archetypes**:

1. **Exact Match**: Use LIKE/equality 
   ```sql
   WHERE article_id = 12345  -- 0.1ms
   ```

2. **Partial Match**: Use FTS
   ```sql  
   WHERE title_content_tsvector @@ plainto_tsquery('PostgreSQL MVCC')  -- 5-20ms
   ```

3. **Semantic Search**: Use dense vectors
   ```sql
   ORDER BY content_vector <=> query_dense LIMIT 10  -- 10-50ms
   ```

4. **Lexical Precision**: Use sparse vectors
   ```sql
   ORDER BY content_sparse <#> query_sparse LIMIT 10  -- 15-100ms  
   ```

5. **Hybrid Intelligence**: Combine all techniques
   ```sql
   -- Adaptive weighting based on query classification
   -- 50-200ms but highest accuracy
   ```

##### Slide 16: Performance Tuning & Monitoring
**Essential Metrics** (based on RAG-essentials):
```sql
-- Performance monitoring view
CREATE VIEW search_performance AS
SELECT 
    mode,
    COUNT(*) as query_count,
    AVG(total_ms) as avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms) as p95_latency,
    AVG(CASE WHEN top_score > 0.8 THEN 1.0 ELSE 0.0 END) as high_confidence_rate
FROM search_metrics 
WHERE query_time >= NOW() - INTERVAL '24 hours'
GROUP BY mode;
```

**Index Health Monitoring**:
```sql
-- Track index usage and effectiveness
SELECT 
    schemaname, tablename, indexname,
    idx_scan as times_used,
    idx_tup_read / NULLIF(idx_scan, 0) as avg_tuples_per_scan
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%vector%' OR indexname LIKE '%sparse%'
ORDER BY idx_scan DESC;
```

#### **SECTION 7: Wrap-up & Resources (3 minutes)**

##### Slide 17: When Each Method Wins
**Decision Matrix**:

| Query Type | Best Method | Why |
|------------|-------------|-----|
| "maternity leave policy" | FTS | Exact policy terms |
| "work from home benefits" | Hybrid | "WFH" needs expansion + concepts |  
| "parental support options" | Dense | Pure semantic concept |
| "ERROR: duplicate key violates" | LIKE | Exact error message |
| "database performance tuning" | Hybrid | Technical terms + concepts |

##### Slide 18: Key Takeaways
**For DBAs**:
1. **Vector search is just specialized indexing** - your skills apply directly
2. **Hybrid approaches beat single techniques** for most real-world queries  
3. **Context optimization remains critical** even with 400K token windows
4. **Performance monitoring evolves** but core principles stay the same
5. **Data quality still rules** - garbage in, garbage out applies to AI too

**For Organizations**:
- PostgreSQL + pgvector/pgvectorscale rivals specialized vector databases
- Existing backup, replication, and HA strategies still apply  
- One system to manage instead of technology proliferation

##### Slide 19: Resources & Next Steps
**Take Home Materials**:
- GitHub repo: `pgvector-rag-lab` with complete lab setup
- Streamlit UI for interactive experimentation  
- n8n workflows for production deployment
- SQL scripts for performance analysis
- Benchmarking tools and datasets

**Production Deployment Checklist**:
- [ ] pgvector 0.8+ and pgvectorscale installed
- [ ] Hybrid search implementation tested
- [ ] Performance monitoring dashboard deployed
- [ ] Context optimization strategies implemented
- [ ] Cost tracking and budgeting in place

### 🎬 Demo Script (12-15 minutes live coding)

#### **Demo Flow**:
1. **Minutes 1-3**: Show the "dumb" search problem with LIKE and FTS
2. **Minutes 4-6**: Demonstrate vector search finding semantic matches  
3. **Minutes 7-9**: Live hybrid search with real-time metrics in Streamlit
4. **Minutes 10-12**: n8n workflow visualization with adaptive weighting
5. **Minutes 13-15**: Performance comparison and cost analysis

#### **Fallback Strategies**:
- Pre-recorded video backup for network issues
- Local environment (no cloud dependencies) 
- Cached embeddings to avoid API rate limits
- Sample results JSON for worst-case scenarios

### 🔧 Technical Demo Setup

#### **Required Components**:
1. **PostgreSQL 17.x** with pgvector 0.8+ and pgvectorscale
2. **Python 3.11+ venv** with dependencies
3. **Streamlit app** running on localhost:8501
4. **n8n Docker container** on localhost:5678  
5. **Wikipedia dataset** with pre-computed embeddings
6. **GPT-5-mini API access** with fallback to GPT-4o-mini

#### **Demo Queries** (DBA-focused):
- "What is WAL in PostgreSQL and why is it important?"
- "How does MVCC reduce locking in PostgreSQL?"
- "Explain PostgreSQL checkpoint tuning strategies"
- "Compare HNSW vs IVFFlat vs DiskANN indexing"
- "When should I use VACUUM vs VACUUM FULL?"

## Key Features and Demonstrations

### Streamlit Application Architecture

#### Backend Service (FastAPI)
```python
# lab/05_api/rag_service.py
@app.post("/search")
async def search(request: SearchRequest):
    metrics = {}
    total_start = time.perf_counter()
    
    # Track embedding generation
    emb_start = time.perf_counter()
    query_embedding = get_embedding(request.query)
    metrics['embedding_ms'] = (time.perf_counter() - emb_start) * 1000
    
    # Track database query
    db_start = time.perf_counter()
    results = await execute_search(request.mode, query_embedding)
    metrics['db_ms'] = (time.perf_counter() - db_start) * 1000
    
    # Track LLM generation (if applicable)
    if request.mode in ['rag_open', 'rag_context_only']:
        llm_start = time.perf_counter()
        answer = generate_answer(results, request.query)
        metrics['llm_ms'] = (time.perf_counter() - llm_start) * 1000
    
    metrics['total_ms'] = (time.perf_counter() - total_start) * 1000
    
    # Store metrics in database
    await store_metrics(metrics, request)
    
    return SearchResponse(results=results, metrics=metrics)
```

#### Frontend UI (Streamlit)
```python
# lab/05_api/streamlit_app.py
import streamlit as st
import plotly.graph_objects as go

# Sidebar configuration
st.sidebar.title("🔍 RAG Search Lab")
page = st.sidebar.radio("Navigation", ["Search", "Metrics", "Comparison"])

if page == "Search":
    # Search interface with real-time metrics
    query = st.text_input("Enter your question:")
    search_mode = st.selectbox("Search Mode", 
        ["LIKE", "FTS", "Dense", "Sparse", "Hybrid", "Adaptive"])
    
    if st.button("Search"):
        # Execute search and display metrics
        with st.spinner("Searching..."):
            response = execute_search(query, search_mode)
            
        # Display results
        st.subheader("Results")
        for result in response.results:
            st.write(f"**{result.title}** (Score: {result.score:.4f})")
            st.text(result.snippet)
            
        # Display performance metrics with visual bar
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Embedding", f"{response.metrics.embedding_ms:.2f} ms")
        col2.metric("Database", f"{response.metrics.db_ms:.2f} ms")
        col3.metric("LLM", f"{response.metrics.llm_ms:.2f} ms")
        col4.metric("Total", f"{response.metrics.total_ms:.2f} ms")
        
        # Visual breakdown bar
        fig = create_latency_breakdown_chart(response.metrics)
        st.plotly_chart(fig)

elif page == "Metrics":
    # Historical metrics dashboard
    st.title("📊 Query Metrics Dashboard")
    metrics_df = load_metrics_from_db()
    
    # Display metrics table with tooltips
    st.dataframe(metrics_df, use_container_width=True)
    
    # Performance trends chart
    fig = create_performance_trends(metrics_df)
    st.plotly_chart(fig)
    
    # Export options
    col1, col2 = st.columns(2)
    if col1.button("Export as CSV"):
        csv = metrics_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "metrics.csv", "text/csv")
    if col2.button("Export as JSON"):
        json_data = metrics_df.to_json(orient='records')
        st.download_button("Download JSON", json_data, "metrics.json", "application/json")
```

### Search Method Comparison Framework (RAG-essentials Inspired)

#### 1. Technique Limitations Demonstration
```sql
-- lab/02_data/technique_limitations_demo.sql

-- KEYWORD SEARCH (LIKE) limitations:
-- ✗ Synonym problem: "car" won't find "automobile"
-- ✗ Word order sensitivity  
-- ✗ No semantic understanding
-- ✗ Case sensitivity issues

-- FULL-TEXT SEARCH limitations:
-- ✗ Acronym expansion failure (AI ≠ Artificial Intelligence)
-- ✗ Cross-language issues
-- ✗ Conceptual gaps (can't bridge different terminology)
-- ✗ Stop word filtering removes important terms

-- VECTOR SEARCH limitations:  
-- ✗ Poor performance for exact matches
-- ✗ Ambiguous terms without context
-- ✗ Overkill for simple lookups
-- ✗ Requires quality embeddings

-- RAG SEARCH limitations:
-- ✗ Temporal limitations (data cutoff dates)
-- ✗ Hallucination risk with insufficient context  
-- ✗ High latency for simple queries
-- ✗ Token/cost overhead
```

#### 2. Performance Comparison Queries
```sql
-- Real-time performance analysis with EXPLAIN ANALYZE
-- Exact string match (fastest - ~0.1ms)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM articles WHERE title = 'Albert Einstein' LIMIT 1;

-- LIKE pattern search (fast with trigram index - ~1-5ms)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM articles WHERE title LIKE 'Albert%' LIMIT 5;

-- Full-text search (moderate - ~5-20ms)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM articles 
WHERE title_content_tsvector @@ plainto_tsquery('english', 'Albert Einstein') 
LIMIT 5;

-- Dense vector search with HNSW (fast - ~10-50ms)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM articles 
ORDER BY content_vector <=> $1::vector LIMIT 5;

-- Sparse vector search (moderate - ~15-100ms) 
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM articles 
ORDER BY content_sparse <#> $1::sparsevec LIMIT 5;

-- Hybrid search (slowest but most accurate - ~50-200ms)
-- Complex query combining multiple techniques
```

#### 3. Edge Cases and Data Quality Analysis
```sql
-- Unicode and special characters handling
SELECT COUNT(*) FROM articles WHERE content LIKE '%café%';     -- Misses "cafe"
SELECT COUNT(*) FROM articles WHERE content ILIKE '%café%' OR content ILIKE '%cafe%';

-- Data quality assessment
SELECT 
    COUNT(*) FILTER (WHERE length(content) < 100) as too_short,
    COUNT(*) FILTER (WHERE length(content) > 50000) as too_long,
    COUNT(*) FILTER (WHERE content LIKE '%{{%}}%') as wiki_markup,
    COUNT(*) FILTER (WHERE content LIKE '%<ref>%') as citations,
    COUNT(*) as total
FROM articles;
```

### Context Window Optimization Strategies
1. **Smart Chunking**: Semantic vs fixed-size chunking comparison
2. **Overlap Tuning**: Demonstrate impact on retrieval quality  
3. **Token Counting**: Real-time usage monitoring in Streamlit
4. **Compression**: Summarization for long documents
5. **Dynamic Selection**: Relevance-based context inclusion
6. **Cost Analysis**: Track API usage and optimization savings

### Performance Comparisons
1. **Search Methods**: LIKE vs FTS vs Dense vs Sparse vs Hybrid
2. **Index Types**: HNSW vs IVFFlat vs DiskANN (pgvectorscale)
3. **Chunk Sizes**: Impact on retrieval and generation quality
4. **Model Sizes**: text-embedding-3-small vs large trade-offs
5. **Context Lengths**: 2K vs 4K vs 8K token impacts

### Live Demonstrations
1. **Streamlit Dashboard**: Interactive search comparison
2. **n8n Workflows**: Visual pipeline execution
3. **Token Usage**: Real-time monitoring and optimization
4. **EXPLAIN Plans**: PostgreSQL query optimization
5. **Cost Analysis**: API usage and optimization savings

## Conclusion

This plan transforms the Movies_pgvector_lab into a comprehensive educational and demonstration platform suitable for PostgreSQL conferences. The focus on n8n provides a more production-oriented approach while maintaining the educational value of the original repository. The structured lab format ensures attendees can follow along and reproduce the results independently.

The evolution maintains backward compatibility with existing movie/Netflix demonstrations while elevating the Wikipedia RAG search as the primary showcase, perfectly aligned with DBA interests and PostgreSQL capabilities.