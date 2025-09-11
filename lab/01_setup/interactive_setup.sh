#!/bin/bash

# Interactive PostgreSQL pgvector RAG Lab Setup Script
# This script provides an interactive setup experience for the RAG lab

set -e  # Exit on any error

echo "🚀 Interactive PostgreSQL pgvector RAG Lab Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables
VENV_PATH=""
VENV_NAME=""
PG_INSTALL_CHOICE=""
PG_CONNECTION=""
CREATE_REQUIREMENTS=true
CREATE_ENV_FILE=true
CREATE_SQL_SCRIPTS=true
CREATE_DOCKER_COMPOSE=true
INSTALL_DEPENDENCIES=true

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

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_question() {
    echo -e "${BLUE}❓ $1${NC}"
}

# Function to ask yes/no questions
ask_yes_no() {
    local question="$1"
    local default="${2:-y}"
    local prompt
    local response
    
    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    while true; do
        printf "${BLUE}❓ %s %s: ${NC}" "$question" "$prompt" >&2
        read -r response
        response=${response:-$default}
        case "$response" in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) print_warning "Please answer yes or no.";;
        esac
    done
}

# Function to ask for user input with default
ask_input() {
    local question="$1"
    local default="$2"
    local response
    
    printf "${BLUE}❓ %s [%s]: ${NC}" "$question" "$default" >&2
    read -r response
    echo "${response:-$default}"
}

# Check if Python 3.11+ is available
check_python() {
    print_info "Checking Python installation..."
    
    # Try different Python commands
    for cmd in python3 python python3.11 python3.12; do
        if command -v $cmd &> /dev/null; then
            PYTHON_VERSION=$($cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
            PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
            
            if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
                print_status "Python $PYTHON_VERSION found (using $cmd)"
                PYTHON_CMD=$cmd
                return 0
            fi
        fi
    done
    
    print_error "Python 3.11+ not found"
    return 1
}

# PostgreSQL setup options
setup_postgresql() {
    echo ""
    echo "📦 PostgreSQL Configuration"
    echo "=========================="
    
    print_info "Choose PostgreSQL setup option:"
    echo "  1) Install PostgreSQL locally (requires sudo)"
    echo "  2) Use existing PostgreSQL instance"
    echo "  3) Use Docker PostgreSQL container"
    echo "  4) Skip PostgreSQL setup (manual configuration later)"
    
    while true; do
        PG_INSTALL_CHOICE=$(ask_input "Enter your choice (1-4)" "2")
        case $PG_INSTALL_CHOICE in
            1)
                install_postgresql_local
                break
                ;;
            2)
                configure_existing_postgresql
                break
                ;;
            3)
                setup_docker_postgresql
                break
                ;;
            4)
                print_warning "Skipping PostgreSQL setup. You'll need to configure it manually."
                PG_CONNECTION="postgresql://user:password@localhost:5432/pgvector_lab"
                break
                ;;
            *)
                print_error "Invalid choice. Please enter 1-4."
                ;;
        esac
    done
}

# Install PostgreSQL locally
install_postgresql_local() {
    print_info "Installing PostgreSQL locally..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            print_info "Detected Ubuntu/Debian system"
            if ask_yes_no "Install PostgreSQL 17?" "y"; then
                sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
                wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
                sudo apt-get update
                sudo apt-get install -y postgresql-17 postgresql-contrib-17 postgresql-17-pgvector
                print_status "PostgreSQL 17 installed"
            fi
        elif command -v yum &> /dev/null; then
            # RHEL/CentOS
            print_info "Detected RHEL/CentOS system"
            if ask_yes_no "Install PostgreSQL 17?" "y"; then
                sudo yum install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-7-x86_64/pgdg-redhat-repo-latest.noarch.rpm
                sudo yum install -y postgresql17-server postgresql17-contrib
                sudo /usr/pgsql-17/bin/postgresql-17-setup initdb
                sudo systemctl enable postgresql-17
                sudo systemctl start postgresql-17
                print_status "PostgreSQL 17 installed"
            fi
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_info "Detected macOS"
        if command -v brew &> /dev/null; then
            if ask_yes_no "Install PostgreSQL 17 with Homebrew?" "y"; then
                brew install postgresql@17
                brew services start postgresql@17
                print_status "PostgreSQL 17 installed"
            fi
        else
            print_warning "Homebrew not found. Please install PostgreSQL manually."
        fi
    else
        print_warning "Unsupported OS. Please install PostgreSQL manually."
    fi
    
    # Configure connection
    PG_HOST=$(ask_input "PostgreSQL host" "localhost")
    PG_PORT=$(ask_input "PostgreSQL port" "5432")
    PG_DATABASE=$(ask_input "Database name" "pgvector_lab")
    PG_USER=$(ask_input "Database user" "postgres")
    
    printf "${CYAN}ℹ️  Enter database password (will not be displayed): ${NC}" >&2
    read -s PG_PASSWORD
    echo ""
    
    PG_CONNECTION="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DATABASE}"
}

# Configure existing PostgreSQL
configure_existing_postgresql() {
    print_info "Configuring connection to existing PostgreSQL instance..."
    
    PG_HOST=$(ask_input "PostgreSQL host" "localhost")
    PG_PORT=$(ask_input "PostgreSQL port" "5432")
    PG_DATABASE=$(ask_input "Database name" "pgvector_lab")
    PG_USER=$(ask_input "Database user" "postgres")
    
    printf "${CYAN}ℹ️  Enter database password (will not be displayed): ${NC}" >&2
    read -s PG_PASSWORD
    echo ""
    
    PG_CONNECTION="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DATABASE}"
    
    # Test connection
    if command -v psql &> /dev/null; then
        if PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d postgres -c "SELECT 1" &> /dev/null; then
            print_status "Successfully connected to PostgreSQL"
            
            # Check for pgvector extension
            if PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d postgres -c "SELECT * FROM pg_available_extensions WHERE name = 'vector'" | grep -q vector; then
                print_status "pgvector extension is available"
            else
                print_warning "pgvector extension not found. You'll need to install it."
                if ask_yes_no "Show installation instructions?" "y"; then
                    echo ""
                    print_info "To install pgvector:"
                    echo "  1. Install from package manager (if available)"
                    echo "  2. Or build from source:"
                    echo "     git clone https://github.com/pgvector/pgvector.git"
                    echo "     cd pgvector"
                    echo "     make"
                    echo "     sudo make install"
                    echo ""
                fi
            fi
        else
            print_warning "Could not connect to PostgreSQL. Please verify credentials."
        fi
    else
        print_warning "psql client not found. Cannot test connection."
    fi
}

# Setup Docker PostgreSQL
setup_docker_postgresql() {
    print_info "Setting up PostgreSQL with Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        return 1
    fi
    
    CONTAINER_NAME=$(ask_input "Container name" "pgvector_lab_db")
    PG_PORT=$(ask_input "PostgreSQL port" "5432")
    PG_PASSWORD=$(ask_input "PostgreSQL password" "postgres")
    PG_DATABASE=$(ask_input "Database name" "pgvector_lab")
    
    # Create docker-compose for PostgreSQL
    cat > lab/01_setup/docker-compose-db.yml << EOF
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: ${CONTAINER_NAME}
    ports:
      - "${PG_PORT}:5432"
    environment:
      - POSTGRES_PASSWORD=${PG_PASSWORD}
      - POSTGRES_DB=${PG_DATABASE}
      - POSTGRES_HOST_AUTH_METHOD=trust
    volumes:
      - ./pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
EOF
    
    if ask_yes_no "Start PostgreSQL container now?" "y"; then
        cd lab/01_setup
        docker-compose -f docker-compose-db.yml up -d
        cd ../..
        print_status "PostgreSQL container started"
    fi
    
    PG_CONNECTION="postgresql://postgres:${PG_PASSWORD}@localhost:${PG_PORT}/${PG_DATABASE}"
}

# Virtual environment setup
setup_venv() {
    echo ""
    echo "🐍 Python Virtual Environment Setup"
    echo "==================================="
    
    # Ask for virtual environment location
    VENV_PATH=$(ask_input "Virtual environment directory path" ".")
    VENV_NAME=$(ask_input "Virtual environment folder name" ".venv")
    
    FULL_VENV_PATH="${VENV_PATH}/${VENV_NAME}"
    
    if [ -d "$FULL_VENV_PATH" ]; then
        if ask_yes_no "Virtual environment already exists. Remove and recreate?" "n"; then
            rm -rf "$FULL_VENV_PATH"
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    print_info "Creating virtual environment at $FULL_VENV_PATH..."
    $PYTHON_CMD -m venv "$FULL_VENV_PATH"
    
    # Activate virtual environment
    if [ -f "$FULL_VENV_PATH/bin/activate" ]; then
        source "$FULL_VENV_PATH/bin/activate"
    elif [ -f "$FULL_VENV_PATH/Scripts/activate" ]; then
        source "$FULL_VENV_PATH/Scripts/activate"
    else
        print_error "Could not find activation script"
        return 1
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip
    print_status "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    echo ""
    echo "📚 Python Dependencies"
    echo "====================="
    
    if ask_yes_no "Install Python dependencies?" "y"; then
        print_info "Choose installation method:"
        echo "  1) Install all dependencies (recommended for full lab)"
        echo "  2) Install core dependencies only"
        echo "  3) Install from existing requirements.txt"
        echo "  4) Custom selection"
        
        choice=$(ask_input "Enter your choice (1-4)" "1")
        
        case $choice in
            1)
                install_all_dependencies
                ;;
            2)
                install_core_dependencies
                ;;
            3)
                install_from_requirements
                ;;
            4)
                install_custom_dependencies
                ;;
        esac
    else
        INSTALL_DEPENDENCIES=false
    fi
}

# Install all dependencies
install_all_dependencies() {
    print_info "Installing all dependencies..."
    
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
    if ask_yes_no "Install PyTorch (large download)?" "y"; then
        if ask_yes_no "Do you have CUDA-capable GPU?" "n"; then
            pip install torch>=2.0
        else
            pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    
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
    
    print_status "All dependencies installed"
}

# Install core dependencies only
install_core_dependencies() {
    print_info "Installing core dependencies only..."
    
    pip install psycopg[binary]>=3.2
    pip install openai>=1.30
    pip install python-dotenv
    pip install tqdm
    pip install numpy
    
    print_status "Core dependencies installed"
}

# Install from requirements.txt
install_from_requirements() {
    local req_file=$(ask_input "Path to requirements.txt" "lab/01_setup/requirements.txt")
    
    if [ -f "$req_file" ]; then
        pip install -r "$req_file"
        print_status "Dependencies installed from $req_file"
    else
        print_error "Requirements file not found: $req_file"
    fi
}

# Custom dependency selection
install_custom_dependencies() {
    print_info "Select dependencies to install:"
    
    if ask_yes_no "Install psycopg (PostgreSQL driver)?" "y"; then
        pip install psycopg[binary]>=3.2
    fi
    
    if ask_yes_no "Install OpenAI SDK?" "y"; then
        pip install openai>=1.30
    fi
    
    if ask_yes_no "Install LangChain framework?" "y"; then
        pip install langchain>=0.1.0 langchain-community langchain-openai langchain-postgres
    fi
    
    if ask_yes_no "Install Streamlit (UI framework)?" "y"; then
        pip install streamlit>=1.28
    fi
    
    if ask_yes_no "Install FastAPI (API framework)?" "y"; then
        pip install fastapi>=0.111 uvicorn>=0.30
    fi
    
    if ask_yes_no "Install ML libraries (PyTorch, Transformers)?" "y"; then
        if ask_yes_no "Do you have CUDA-capable GPU?" "n"; then
            pip install torch>=2.0
        else
            pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        pip install transformers>=4.30 sentencepiece
    fi
    
    if ask_yes_no "Install data processing libraries?" "y"; then
        pip install numpy pandas plotly tqdm
    fi
    
    print_status "Selected dependencies installed"
}

# Create configuration files
create_config_files() {
    echo ""
    echo "📄 Configuration Files"
    echo "====================="
    
    if ask_yes_no "Create requirements.txt?" "y"; then
        create_requirements_file
    else
        CREATE_REQUIREMENTS=false
    fi
    
    if ask_yes_no "Create .env template?" "y"; then
        create_env_template
    else
        CREATE_ENV_FILE=false
    fi
    
    if ask_yes_no "Create PostgreSQL setup scripts?" "y"; then
        create_setup_sql
    else
        CREATE_SQL_SCRIPTS=false
    fi
    
    if ask_yes_no "Create Docker Compose for n8n workflows?" "y"; then
        create_docker_compose
    else
        CREATE_DOCKER_COMPOSE=false
    fi
}

# Create requirements.txt
create_requirements_file() {
    print_info "Creating requirements.txt..."
    
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
    print_info "Creating environment template..."
    
    # Use configured PostgreSQL connection if available
    local db_url="${PG_CONNECTION:-postgresql://user:password@localhost:5432/pgvector_lab}"
    
    cat > lab/01_setup/.env.template << EOF
# PostgreSQL Configuration
DATABASE_URL=${db_url}
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
MAX_CONTEXT_TOKENS=50000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_OUTPUT_TOKENS=16000

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_THEME=dark

# n8n Configuration (Docker)
N8N_PORT=5678
N8N_WEBHOOK_URL=http://localhost:5678/webhook
EOF
    
    print_status "Environment template created"
    
    if ask_yes_no "Copy .env.template to .env now?" "y"; then
        cp lab/01_setup/.env.template lab/01_setup/.env
        print_info "Created .env file. Please edit it with your API keys."
        
        if ask_yes_no "Open .env file in editor?" "n"; then
            ${EDITOR:-nano} lab/01_setup/.env
        fi
    fi
}

# Create setup SQL script
create_setup_sql() {
    print_info "Creating PostgreSQL setup script..."
    
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

-- Advanced full-text setup with weighted ranking
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
CREATE INDEX IF NOT EXISTS idx_articles_content_tsv 
    ON articles USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_title_content_tsvector 
    ON articles USING GIN (title_content_tsvector);
CREATE INDEX IF NOT EXISTS idx_articles_title_gin 
    ON articles USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_articles_id 
    ON articles USING btree (id);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS search_metrics (
    log_id SERIAL PRIMARY KEY,
    query_id TEXT,
    description TEXT,
    query_time TIMESTAMPTZ DEFAULT NOW(),
    mode TEXT,
    top_score REAL,
    token_usage INTEGER,
    precision REAL DEFAULT 0,
    embedding_ms REAL,
    db_ms REAL,
    llm_ms REAL,
    total_ms REAL,
    context_tokens INTEGER,
    output_tokens INTEGER,
    chunk_count INTEGER,
    rerank_ms REAL,
    index_used TEXT,
    buffer_hits INTEGER,
    disk_reads INTEGER
);

SELECT 'PostgreSQL pgvector RAG Lab setup completed successfully!' as status;
EOF
    
    print_status "PostgreSQL setup script created"
    
    if [ "$PG_INSTALL_CHOICE" != "4" ] && ask_yes_no "Run setup.sql on database now?" "y"; then
        if [ -n "$PG_PASSWORD" ]; then
            PGPASSWORD=$PG_PASSWORD psql "$PG_CONNECTION" -f lab/01_setup/setup.sql
            print_status "Database schema created"
        else
            print_warning "Database password not set. Please run manually:"
            echo "  psql \"$PG_CONNECTION\" -f lab/01_setup/setup.sql"
        fi
    fi
}

# Create Docker Compose for n8n
create_docker_compose() {
    print_info "Creating Docker Compose for n8n workflows..."
    
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
    
    if command -v docker &> /dev/null && ask_yes_no "Start n8n container now?" "n"; then
        cd lab/06_workflows
        docker-compose up -d
        cd ../..
        print_status "n8n container started on http://localhost:5678"
    fi
}

# Generate summary and next steps
generate_summary() {
    echo ""
    echo "======================================"
    echo "🎉 Setup Complete!"
    echo "======================================"
    echo ""
    echo "📋 Configuration Summary:"
    echo "------------------------"
    
    if [ -n "$PYTHON_CMD" ]; then
        echo "  ✓ Python: $PYTHON_CMD (version $PYTHON_VERSION)"
    fi
    
    if [ -n "$FULL_VENV_PATH" ]; then
        echo "  ✓ Virtual Environment: $FULL_VENV_PATH"
    fi
    
    if [ "$PG_INSTALL_CHOICE" != "4" ]; then
        echo "  ✓ PostgreSQL: Configured"
        if [ -n "$PG_CONNECTION" ]; then
            echo "    Connection: ${PG_CONNECTION//:*@//:****@}"
        fi
    fi
    
    if [ "$INSTALL_DEPENDENCIES" = true ]; then
        echo "  ✓ Python Dependencies: Installed"
    fi
    
    if [ "$CREATE_REQUIREMENTS" = true ]; then
        echo "  ✓ requirements.txt: Created"
    fi
    
    if [ "$CREATE_ENV_FILE" = true ]; then
        echo "  ✓ .env.template: Created"
    fi
    
    if [ "$CREATE_SQL_SCRIPTS" = true ]; then
        echo "  ✓ PostgreSQL scripts: Created"
    fi
    
    if [ "$CREATE_DOCKER_COMPOSE" = true ]; then
        echo "  ✓ Docker Compose (n8n): Created"
    fi
    
    echo ""
    echo "📋 Next Steps:"
    echo "-------------"
    
    local step=1
    
    if [ "$CREATE_ENV_FILE" = true ]; then
        echo "  $step. Edit lab/01_setup/.env with your API keys"
        ((step++))
    fi
    
    if [ "$PG_INSTALL_CHOICE" = "4" ]; then
        echo "  $step. Configure PostgreSQL and run: psql -f lab/01_setup/setup.sql"
        ((step++))
    fi
    
    if [ -n "$FULL_VENV_PATH" ]; then
        echo ""
        echo "💡 To activate the virtual environment later:"
        if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
            echo "   $FULL_VENV_PATH\\Scripts\\activate"
        else
            echo "   source $FULL_VENV_PATH/bin/activate"
        fi
    fi
    
    echo ""
    echo "📚 Documentation:"
    echo "  - README.md: Project overview"
    echo "  - lab/README.md: Lab instructions"
    echo "  - CLAUDE.md: AI assistant guidelines"
    
    echo ""
    print_status "Environment ready for development!"
}

# Main setup function
main() {
    # Check prerequisites
    if ! check_python; then
        print_error "Python 3.11+ is required. Please install Python 3.11 or later."
        
        if ask_yes_no "Show installation instructions?" "y"; then
            echo ""
            print_info "Python Installation Options:"
            echo "  Ubuntu/Debian: sudo apt-get install python3.11"
            echo "  RHEL/CentOS: sudo yum install python311"
            echo "  macOS: brew install python@3.11"
            echo "  Windows: Download from https://python.org"
            echo ""
        fi
        exit 1
    fi
    
    # PostgreSQL setup
    setup_postgresql
    
    # Virtual environment setup
    if ask_yes_no "Create Python virtual environment?" "y"; then
        setup_venv
    fi
    
    # Install dependencies
    install_dependencies
    
    # Create configuration files
    create_config_files
    
    # Generate summary
    generate_summary
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Interactive PostgreSQL pgvector RAG Lab Setup"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --help     Show this help message"
            echo "  --quick        Run with minimal prompts (use defaults)"
            echo "  --no-color     Disable colored output"
            echo ""
            exit 0
            ;;
        --quick)
            # Set defaults for quick mode
            export QUICK_MODE=true
            shift
            ;;
        --no-color)
            # Disable colors
            RED=''
            GREEN=''
            YELLOW=''
            BLUE=''
            CYAN=''
            NC=''
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"