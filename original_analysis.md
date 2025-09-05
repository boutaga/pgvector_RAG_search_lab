# Comprehensive Analysis of Original/ Python Files

This document analyzes all 11 Python files in the `/mnt/c/Users/oba/source/repos/Movies_pgvector_lab/original/` directory to understand the functionality, patterns, and requirements for porting to a new modular lab/ structure.

## Executive Summary

The original codebase contains two distinct scenarios:
1. **Movie/Netflix Recommendations** (7 files) - DVD rental database with Netflix shows
2. **Wikipedia RAG Search** (4 files) - 25,000 Wikipedia articles database

Both scenarios implement progressive complexity from simple dense embeddings to advanced hybrid search with query classification.

## File-by-File Analysis

### 1. Embedding Generation Scripts

#### create_emb.py (Movie/Netflix - Dense)
- **Purpose**: Batch generation of dense embeddings for film and netflix_shows tables
- **Model**: text-embedding-ada-002 (1536 dimensions)
- **Batch Size**: 30 
- **Key Features**:
  - Exponential backoff retry logic for rate limits
  - Batch processing for efficiency
  - Updates two tables: `film` and `netflix_shows`
- **Database Schema**: 
  - `film.embedding` (vector)
  - `netflix_shows.embedding` (vector)

#### create_emb_sparse.py (Movie/Netflix - Sparse)
- **Purpose**: Generate SPLADE sparse embeddings for the same tables
- **Model**: naver/splade-cocondenser-ensembledistil
- **Batch Size**: 10 (smaller due to memory requirements)
- **Key Features**:
  - CUDA/CPU auto-detection
  - Memory management with garbage collection
  - Sophisticated sparse vector formatting for pgvector
  - Updates only `netflix_shows.sparse_embedding` (sparsevec)
- **Notable**: More complex error handling and memory management

#### create_emb_wiki.py (Wikipedia - Dense)
- **Purpose**: Generate embeddings for Wikipedia articles table
- **Model**: text-embedding-3-small (newer, more efficient)
- **Batch Size**: 50 (increased efficiency)
- **Key Features**:
  - Handles both title and content embeddings separately
  - Content truncation for token limits (32,000 chars ≈ 8,000 tokens)
  - Comprehensive verification reporting
  - Progressive processing (titles first, then content)
- **Database Schema**:
  - `articles.title_vector` (vector)
  - `articles.content_vector` (vector)

#### create_emb_sparse_wiki.py (Wikipedia - Sparse)
- **Purpose**: Add sparse embeddings to Wikipedia articles
- **Key Features**:
  - Dynamically adds sparse columns if missing
  - Automatic index creation
  - Smaller batch size (5) due to large article content
  - Content truncation for memory management
- **Database Schema**: Adds `title_sparse` and `content_sparse` columns

### 2. RAG Search Scripts

#### RAG_search.py (Simple Movie/Netflix)
- **Purpose**: Basic RAG implementation
- **Features**:
  - Single embedding generation per query
  - Simple vector similarity search
  - Direct OpenAI chat completion
  - Searches only `netflix_shows` table
- **Limitations**: No hybrid search, basic error handling

#### RAG_search_Open.py (Alternative Movie/Netflix)
- **Purpose**: Variant of basic RAG with different prompt strategy
- **Key Difference**: Allows external knowledge if context insufficient
- **Otherwise**: Nearly identical to RAG_search.py

#### RAG_search_hybrid.py (Advanced Movie/Netflix)
- **Purpose**: Full-featured hybrid search system
- **Key Features**:
  - **Query Classification**: Structured vs semantic query detection
  - **Dual Search**: Dense + sparse embedding searches
  - **SQL Generation**: Direct SQL queries for structured questions
  - **Result Merging**: Weighted combination of search results
  - **Re-ranking**: Configurable weights (DENSE_WEIGHT=0.5, SPARSE_WEIGHT=0.5)
- **Architecture**: Most sophisticated in the movie/Netflix category

#### RAG_search_hybrid_simple.py (Simplified Hybrid)
- **Purpose**: Hybrid search without query classification
- **Features**: Dense + sparse search with merging, but no SQL generation
- **Use Case**: Experimentation and simpler hybrid scenarios

#### RAG_search_wiki.py (Simple Wikipedia)
- **Purpose**: Basic Wikipedia RAG search
- **Features**:
  - Title, content, or combined search modes  
  - Article deduplication in combined mode
  - Enhanced context formatting with article previews
  - Interactive loop with commands
- **User Experience**: More polished interface than movie equivalents

#### RAG_search_wiki_hybrid.py (Advanced Wikipedia)
- **Purpose**: Most sophisticated implementation in the codebase
- **Key Features**:
  - **Intelligent Query Classification**: factual/conceptual/exploratory
  - **Adaptive Weighting**: Changes dense/sparse weights based on query type
    - Factual: 0.3 dense, 0.7 sparse
    - Conceptual: 0.7 dense, 0.3 sparse
    - Exploratory: 0.5 dense, 0.5 sparse
  - **Multiple Search Modes**: dense_only, sparse_only, hybrid, adaptive
  - **Comprehensive Results**: Shows scoring details
  - **Global Model Loading**: SPLADE model loaded once and reused

### 3. Recommendation Script

#### recommend_netflix.py
- **Purpose**: Customer-based recommendation system
- **Approach**: 
  - Aggregate customer rental history embeddings
  - Compute average embedding as user profile
  - Find most similar Netflix shows
- **Database Integration**: Links DVD rental data with Netflix catalog
- **Unique Feature**: Only script that uses customer transaction history

## Common Patterns and Utilities

### Database Connection Patterns
1. **Basic Pattern**: Direct psycopg2.connect() with environment variables
2. **Vector Registration**: All scripts call register_vector(conn)
3. **Error Handling**: Varies from basic to sophisticated
4. **Connection Management**: Generally good cleanup practices

### Embedding Generation Patterns
1. **OpenAI API Calls**: 
   - Batch processing with configurable sizes
   - Exponential backoff retry logic
   - Rate limit handling (429 errors)
2. **SPLADE Processing**:
   - Model loading with device detection
   - Memory management considerations
   - Sparse vector formatting for pgvector

### Search Patterns
1. **Vector Similarity**: pgvector operators (<->, <=>)
2. **Result Merging**: Various approaches to combine dense/sparse results
3. **Re-ranking**: Weighted scoring systems
4. **Context Building**: Formatting retrieved content for LLM consumption

### Answer Generation Patterns
1. **Prompt Engineering**: System + user message patterns
2. **Context Limits**: Content truncation strategies
3. **Model Selection**: gpt-4o vs gpt-3.5-turbo
4. **Error Handling**: Graceful degradation approaches

## External Dependencies

### Required Python Packages
- `psycopg2` or `psycopg2-binary` - PostgreSQL adapter
- `openai` - OpenAI API client (using new interface)
- `pgvector` - PostgreSQL vector extension Python support
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for SPLADE model
- `sentencepiece` - Required for some tokenizers
- `numpy` - Array operations (used in recommend_netflix.py)

### Model Dependencies
- **OpenAI Models**:
  - `text-embedding-ada-002` (legacy, 1536 dimensions)
  - `text-embedding-3-small` (newer, 1536 dimensions)
  - `gpt-3.5-turbo` and `gpt-4o` for text generation
- **SPLADE Model**: `naver/splade-cocondenser-ensembledistil`

### Database Requirements
- PostgreSQL with pgvector extension
- Vector types: `vector(1536)`, `sparsevec(30522)`
- Indexes: HNSW or IVFFlat for performance

## Configuration Patterns

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection strings
- `OPENAI_API_KEY`: OpenAI API access

### Configurable Parameters
- **Batch Sizes**: 5-50 depending on content size and memory constraints
- **Search Limits**: TOP_K values from 5-10
- **Model Parameters**: temperature, max_tokens for generation
- **Weights**: Dense/sparse combination ratios

## Opportunities for Refactoring

### 1. Database Connection Management
**Current**: Each script manages its own connections
**Opportunity**: Centralized connection pool/manager

### 2. Embedding Generation
**Current**: Duplicated batch processing logic across scripts
**Opportunity**: Shared embedding service with pluggable models

### 3. Search Infrastructure
**Current**: Similar vector search patterns repeated
**Opportunity**: Unified search interface supporting multiple strategies

### 4. Model Management
**Current**: Model loading scattered across scripts
**Opportunity**: Model registry/cache system

### 5. Configuration Management
**Current**: Hardcoded constants in each file
**Opportunity**: Centralized configuration system

### 6. Error Handling
**Current**: Inconsistent error handling approaches
**Opportunity**: Standardized error handling and logging

### 7. Result Processing
**Current**: Different result merging implementations
**Opportunity**: Pluggable re-ranking strategies

## Architectural Insights

### Progressive Complexity
The codebase shows clear evolution from simple to sophisticated:
1. **Simple**: Single embedding type, basic search
2. **Hybrid**: Multiple embedding types, result merging
3. **Advanced**: Query classification, adaptive strategies

### Domain Separation
Clear separation between movie/Netflix and Wikipedia scenarios:
- Different database schemas
- Different embedding models
- Different search strategies
- Different user interfaces

### Model Evolution
Migration from older to newer OpenAI models shows adaptability:
- Legacy: text-embedding-ada-002
- Modern: text-embedding-3-small

## Recommendations for Modular Structure

### Core Components Needed
1. **Database Layer**: Connection management, schema abstraction
2. **Embedding Layer**: Pluggable dense/sparse embedding services
3. **Search Layer**: Unified search interface with multiple strategies
4. **Ranking Layer**: Configurable result merging and re-ranking
5. **Generation Layer**: Consistent answer generation interface
6. **Configuration Layer**: Centralized parameter management
7. **Utilities Layer**: Common functions (retry logic, formatting, etc.)

### Domain Modules
1. **Movies Module**: DVD rental + Netflix recommendation logic
2. **Wikipedia Module**: Wikipedia-specific search and processing
3. **Common Module**: Shared functionality across domains

### Service Interfaces
1. **EmbeddingService**: Abstract interface for dense/sparse embedding generation
2. **SearchService**: Abstract interface for vector similarity search
3. **RankingService**: Abstract interface for result combination and re-ranking
4. **GenerationService**: Abstract interface for answer generation

This analysis provides the foundation for designing a modular, maintainable architecture that preserves the sophisticated functionality while improving code organization, reusability, and maintainability.