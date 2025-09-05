# Phase 2 Implementation Plan - Core Functionality Enhancement

## Overview
**Phase**: 2 - Core Functionality Enhancement  
**Status**: Planning - Awaiting Human Review  
**Estimated Duration**: 2-3 weeks  
**Goal**: Transform existing monolithic scripts into modular, reusable components while adding enhanced functionality

## Analysis Summary

Based on comprehensive analysis of the original/ folder:
- **11 Python scripts** implementing two scenarios (Movie/Netflix + Wikipedia)
- **Progressive complexity** from simple dense search to advanced hybrid with query classification
- **Common patterns** identified for database, embedding, search, and generation operations
- **Significant code duplication** that can be eliminated through modularization

## Proposed Architecture

### Core Service Layer (`lab/core/`)

#### 1. Database Service (`lab/core/database.py`)
```python
class DatabaseService:
    - connection_pool management
    - pgvector registration
    - retry logic with exponential backoff
    - schema validation
    - index management utilities
```

#### 2. Embedding Service (`lab/core/embeddings.py`)
```python
class EmbeddingService:
    - AbstractEmbedder interface
    - OpenAIEmbedder (dense)
    - SPLADEEmbedder (sparse)
    - batch processing with configurable sizes
    - rate limit handling
    - memory management
```

#### 3. Search Service (`lab/core/search.py`)
```python
class SearchService:
    - VectorSearch (dense similarity)
    - SparseSearch (sparse similarity)
    - HybridSearch (combined)
    - AdaptiveSearch (query-based weights)
    - result pagination
    - metadata filtering
```

#### 4. Ranking Service (`lab/core/ranking.py`)
```python
class RankingService:
    - reciprocal rank fusion (RRF)
    - weighted linear combination
    - score normalization
    - deduplication strategies
    - configurable re-ranking algorithms
```

#### 5. Generation Service (`lab/core/generation.py`)
```python
class GenerationService:
    - prompt templates
    - context window management
    - token counting utilities
    - model selection logic
    - streaming support
```

#### 6. Configuration Service (`lab/core/config.py`)
```python
class ConfigService:
    - environment variable management
    - model parameters
    - search configurations
    - performance settings
    - cost tracking parameters
```

### Domain Modules

#### 1. Data Processing (`lab/02_data/`)
```
processor.py         - Base data processing utilities
chunking.py         - Smart document chunking strategies
cleaner.py          - Text cleaning and normalization
loader.py           - Data import/export utilities
wikipedia_loader.py - Wikipedia-specific processing
movie_loader.py     - Movie/Netflix data processing
```

#### 2. Embedding Generation (`lab/03_embeddings/`)
```
generate_embeddings.py - CLI for batch embedding generation
update_embeddings.py   - Incremental embedding updates
verify_embeddings.py   - Validation and statistics
embedding_manager.py   - Orchestration layer
```

#### 3. Search Implementation (`lab/04_search/`)
```
simple_search.py      - Basic dense vector search
hybrid_search.py      - Dense + sparse combination
adaptive_search.py    - Query classification + adaptive weights
sql_search.py         - Structured query generation
langchain_search.py   - LangChain integration
comparison_tool.py    - Side-by-side search comparison
```

#### 4. API Layer (`lab/05_api/`)
```
fastapi_server.py     - REST API endpoints
streamlit_app.py      - Interactive web UI
gradio_interface.py   - Alternative UI option
metrics_collector.py  - Performance tracking
cost_tracker.py       - Token usage and cost monitoring
```

#### 5. Evaluation (`lab/07_evaluation/`)
```
benchmark.py          - Performance benchmarking
metrics.py            - Evaluation metrics (MRR, NDCG, etc.)
visualizer.py         - Results visualization
report_generator.py   - Automated reporting
```

## Implementation Steps

### Step 1: Core Services (Week 1, Days 1-3)

1. **Day 1: Database & Configuration Services**
   - [ ] Create `lab/core/` directory structure
   - [ ] Implement DatabaseService with connection pooling
   - [ ] Implement ConfigService with environment management
   - [ ] Add comprehensive error handling and logging

2. **Day 2: Embedding Service**
   - [ ] Create abstract EmbeddingService interface
   - [ ] Implement OpenAIEmbedder with batch processing
   - [ ] Implement SPLADEEmbedder with CUDA support
   - [ ] Add rate limit handling and retry logic

3. **Day 3: Search & Ranking Services**
   - [ ] Implement SearchService with multiple strategies
   - [ ] Create RankingService with RRF and weighted combination
   - [ ] Add result deduplication and normalization
   - [ ] Implement comprehensive testing

### Step 2: Data Processing (Week 1, Days 4-5)

4. **Day 4: Data Loading & Processing**
   - [ ] Create base data processor
   - [ ] Implement smart chunking strategies
   - [ ] Add Wikipedia and Movie-specific loaders
   - [ ] Create data validation utilities

5. **Day 5: Embedding Generation Tools**
   - [ ] Port existing embedding scripts to use new services
   - [ ] Create CLI tools for batch processing
   - [ ] Add incremental update capabilities
   - [ ] Implement verification and statistics

### Step 3: Search Implementation (Week 2, Days 6-8)

6. **Day 6: Basic Search Implementations**
   - [ ] Port simple_search using new services
   - [ ] Implement hybrid_search with configurable weights
   - [ ] Add metadata filtering capabilities
   - [ ] Create search result formatting utilities

7. **Day 7: Advanced Search Features**
   - [ ] Implement query classification system
   - [ ] Create adaptive_search with dynamic weights
   - [ ] Add SQL generation for structured queries
   - [ ] Implement LangChain integration

8. **Day 8: Search Comparison Tool**
   - [ ] Create side-by-side comparison framework
   - [ ] Add performance metrics collection
   - [ ] Implement result visualization
   - [ ] Create export capabilities

### Step 4: API & UI (Week 2, Days 9-10)

9. **Day 9: FastAPI Backend**
   - [ ] Create REST API endpoints for all search types
   - [ ] Add authentication and rate limiting
   - [ ] Implement metrics collection
   - [ ] Add OpenAPI documentation

10. **Day 10: Streamlit Frontend**
    - [ ] Create interactive search interface
    - [ ] Add real-time performance visualization
    - [ ] Implement comparison mode
    - [ ] Add export and sharing features

### Step 5: Testing & Documentation (Week 3, Days 11-12)

11. **Day 11: Testing & Benchmarking**
    - [ ] Create comprehensive test suite
    - [ ] Run performance benchmarks
    - [ ] Generate evaluation metrics
    - [ ] Create performance reports

12. **Day 12: Documentation & Polish**
    - [ ] Write API documentation
    - [ ] Create user guides
    - [ ] Add code examples
    - [ ] Final testing and bug fixes

## Migration Strategy

### Phase 2.1: Core Infrastructure (Priority 1)
1. Build core services without breaking existing functionality
2. Test services in isolation
3. Create compatibility layer for gradual migration

### Phase 2.2: Feature Parity (Priority 2)
1. Port existing functionality using new services
2. Maintain backward compatibility
3. Validate against original implementations

### Phase 2.3: Enhanced Features (Priority 3)
1. Add new capabilities (LangChain, metrics, etc.)
2. Implement performance optimizations
3. Add production-ready features

## Success Criteria

### Functional Requirements
- [ ] All original functionality preserved
- [ ] Services work independently and together
- [ ] Clean separation of concerns
- [ ] Comprehensive error handling

### Performance Requirements
- [ ] Query latency <100ms (P95)
- [ ] Batch processing 100+ items/minute
- [ ] Memory usage <2GB for standard operations
- [ ] Support for concurrent requests

### Quality Requirements
- [ ] 80%+ code coverage in tests
- [ ] All functions documented
- [ ] Type hints throughout
- [ ] Logging at appropriate levels

## Risk Mitigation

### Technical Risks
1. **Database Connection Issues**
   - Mitigation: Connection pooling with retry logic
   
2. **Rate Limiting**
   - Mitigation: Exponential backoff, request queuing

3. **Memory Issues with SPLADE**
   - Mitigation: Batch size optimization, garbage collection

4. **API Compatibility**
   - Mitigation: Version management, deprecation warnings

### Process Risks
1. **Scope Creep**
   - Mitigation: Strict adherence to plan, defer enhancements

2. **Testing Gaps**
   - Mitigation: Test-driven development, continuous validation

## Dependencies

### Required Before Starting
- [ ] PostgreSQL with pgvector installed
- [ ] OpenAI API key configured
- [ ] Python environment set up
- [ ] Git branch strategy confirmed

### Can Be Added During Development
- [ ] pgvectorscale for production performance
- [ ] Additional embedding models
- [ ] Authentication system
- [ ] Monitoring infrastructure

## File Structure After Phase 2

```
lab/
├── core/                    # Core service layer
│   ├── __init__.py
│   ├── database.py         # Database service
│   ├── embeddings.py       # Embedding service
│   ├── search.py           # Search service
│   ├── ranking.py          # Ranking service
│   ├── generation.py       # Generation service
│   └── config.py           # Configuration service
│
├── 02_data/                # Data processing
│   ├── processor.py
│   ├── chunking.py
│   ├── loader.py
│   └── scripts/
│       ├── load_wikipedia.py
│       └── load_movies.py
│
├── 03_embeddings/          # Embedding generation
│   ├── generate_embeddings.py
│   ├── update_embeddings.py
│   └── verify_embeddings.py
│
├── 04_search/              # Search implementations
│   ├── simple_search.py
│   ├── hybrid_search.py
│   ├── adaptive_search.py
│   └── comparison_tool.py
│
├── 05_api/                 # API and UI
│   ├── fastapi_server.py
│   ├── streamlit_app.py
│   └── static/
│
├── 07_evaluation/          # Evaluation tools
│   ├── benchmark.py
│   ├── metrics.py
│   └── results/
│
└── tests/                  # Test suite
    ├── test_core/
    ├── test_search/
    └── test_api/
```

## Next Actions

Upon approval of this plan:
1. Create core/ directory structure
2. Begin with DatabaseService implementation
3. Set up testing framework
4. Start incremental migration

## Questions for Review

1. Is the proposed architecture aligned with conference presentation needs?
2. Should we prioritize any specific search method for demo purposes?
3. Are there specific performance metrics you want to showcase?
4. Do you want to maintain compatibility with existing scripts during migration?
5. Should we focus on one scenario (Wikipedia or Movies) first?

---

**Plan Status**: READY FOR REVIEW  
**Author**: Claude  
**Date**: September 5, 2025  
**Approval Required**: YES

Please review this plan and provide feedback or approval to proceed with Phase 2 implementation.