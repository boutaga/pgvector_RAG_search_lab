#!/bin/bash

# PostgreSQL pgvector RAG Lab Setup Script
# This script sets up the complete environment for the RAG lab

set -e  # Exit on any error

echo "🚀 Setting up PostgreSQL pgvector RAG Lab..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python 3.11+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            print_status "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Create and activate virtual environment
setup_venv() {
    echo "📦 Setting up Python virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists, removing it..."
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    source .venv/bin/activate || source .venv/Scripts/activate 2>/dev/null
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    print_status "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    echo "📚 Installing Python dependencies..."
    
    # Core dependencies
    pip install psycopg[binary]>=3.2
    pip install openai>=1.30
    pip install langchain>=0.1.0
    pip install langchain-community
    pip install langchain-openai
    pip install langchain-postgres
    pip install streamlit>=1.28
    pip install fastapi>=0.111
    pip install uvicorn>=0.30
    
    # ML/AI dependencies
    pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu
    pip install transformers>=4.30
    pip install sentencepiece
    pip install tiktoken
    
    # Utilities
    pip install tqdm
    pip install numpy
    pip install pandas
    pip install plotly
    pip install python-dotenv
    pip install requests
    
    print_status "Python dependencies installed"
}

# Create requirements.txt
create_requirements() {
    echo "📄 Creating requirements.txt..."
    
    cat > lab/01_setup/requirements.txt << 'EOF'
# Core Database and API
psycopg[binary]>=3.2
openai>=1.30

# LangChain Framework
langchain>=0.1.0
langchain-community
langchain-openai
langchain-postgres

# Web Framework
streamlit>=1.28
fastapi>=0.111
uvicorn>=0.30

# Machine Learning
torch>=2.0
transformers>=4.30
sentencepiece
tiktoken

# Data Processing
tqdm
numpy
pandas
plotly
python-dotenv
requests
EOF
    
    print_status "requirements.txt created"
}

# Create environment template
create_env_template() {
    echo "🔐 Creating environment template..."
    
    cat > lab/01_setup/.env.template << 'EOF'
# PostgreSQL Configuration (host-level installation)
DATABASE_URL=postgresql://user:password@localhost:5432/pgvector_lab
PG_POOL_SIZE=20
PG_VECTOR_EXTENSION_VERSION=0.8.0
PG_VECTORSCALE_ENABLED=true

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_EMB=text-embedding-3-small
OPENAI_MODEL_CHAT=gpt-4-mini

# SPLADE Configuration
SPLADE_MODEL=naver/splade-cocondenser-ensembledistil
SPLADE_DEVICE=cuda  # or cpu

# Hybrid Search Parameters
HYBRID_ALPHA=0.5  # Default dense weight (0-1)
TOPK_RETRIEVAL=50
FINAL_K=10

# Context Window Optimization
MAX_CONTEXT_TOKENS=50000  # Conservative limit for cost efficiency
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_OUTPUT_TOKENS=16000  # Conservative output limit

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_THEME=dark

# n8n Configuration (Docker)
N8N_PORT=5678
WEBHOOK_URL=http://localhost:5678/webhook
EOF
    
    print_status "Environment template created"
}

# Create setup SQL script
create_setup_sql() {
    echo "🗄️ Creating PostgreSQL setup script..."
    
    cat > lab/01_setup/setup.sql << 'EOF'
-- PostgreSQL pgvector RAG Lab Setup
-- This script sets up the complete database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create the main articles table with all search capabilities
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
CREATE OR REPLACE FUNCTION update_article_tsvector() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.title_content_tsvector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') || 
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER articles_tsvector_update 
    BEFORE INSERT OR UPDATE ON articles 
    FOR EACH ROW EXECUTE FUNCTION update_article_tsvector();

-- Performance indexes
-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_articles_content_tsv 
    ON articles USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_title_content_tsvector 
    ON articles USING GIN (title_content_tsvector);

-- Traditional indexes for exact matches
CREATE INDEX IF NOT EXISTS idx_articles_title_gin 
    ON articles USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_articles_id 
    ON articles USING btree (id);

-- Note: Vector indexes will be created after embeddings are populated
-- Dense vector indexes (HNSW for speed) - uncomment after populating embeddings
/*
CREATE INDEX IF NOT EXISTS idx_articles_title_vec_hnsw 
    ON articles USING hnsw (title_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_articles_content_vec_hnsw 
    ON articles USING hnsw (content_vector vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- Sparse vector indexes (pgvectorscale) - uncomment after populating sparse embeddings
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
*/

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

-- Performance monitoring views
CREATE OR REPLACE VIEW search_performance AS
SELECT 
    mode,
    COUNT(*) as query_count,
    AVG(total_ms) as avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms) as p95_latency,
    AVG(CASE WHEN top_score > 0.8 THEN 1.0 ELSE 0.0 END) as high_confidence_rate
FROM search_metrics 
WHERE query_time >= NOW() - INTERVAL '24 hours'
GROUP BY mode;

SELECT 'PostgreSQL pgvector RAG Lab setup completed successfully!' as status;
EOF
    
    print_status "PostgreSQL setup script created"
}

# Create Docker Compose for n8n
create_docker_compose() {
    echo "🐳 Creating Docker Compose for n8n..."
    
    mkdir -p lab/06_workflows
    
    cat > lab/06_workflows/docker-compose.yml << 'EOF'
version: '3.8'

services:
  n8n:
    image: n8nio/n8n:latest
    container_name: pgvector_lab_n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=false
      - N8N_HOST=0.0.0.0
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - WEBHOOK_URL=http://localhost:5678/
      - N8N_LOG_LEVEL=info
    volumes:
      - ./n8n_data:/home/node/.n8n
      - /var/run/docker.sock:/var/run/docker.sock:ro
    restart: unless-stopped
    command: n8n start
    networks:
      - pgvector_lab

networks:
  pgvector_lab:
    driver: bridge
EOF
    
    print_status "Docker Compose for n8n created"
}

# Main setup function
main() {
    echo "🎯 PostgreSQL pgvector RAG Lab Setup"
    echo "===================================="
    
    # Check prerequisites
    if ! check_python; then
        print_error "Python 3.11+ is required. Please install Python 3.11 or later."
        exit 1
    fi
    
    # Setup Python environment
    setup_venv
    install_dependencies
    
    # Create configuration files
    create_requirements
    create_env_template
    create_setup_sql
    create_docker_compose
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Copy .env.template to .env and fill in your credentials"
    echo "2. Install PostgreSQL 17.x with pgvector 0.8+ and pgvectorscale"
    echo "3. Run: psql -f lab/01_setup/setup.sql your_database"
    echo "4. Start n8n: cd lab/06_workflows && docker-compose up -d"
    echo ""
    echo "💡 To activate the virtual environment later:"
    echo "   source .venv/bin/activate  # Linux/Mac"
    echo "   .venv\\Scripts\\activate     # Windows"
    echo ""
    print_status "Environment ready for Phase 2 development!"
}

# Run main function
main "$@"