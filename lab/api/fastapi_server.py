#!/usr/bin/env python3
"""
FastAPI server for pgvector RAG search API.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.search.simple_search import SimpleSearchEngine
from lab.search.hybrid_search import HybridSearchEngine
from lab.search.adaptive_search import AdaptiveSearchEngine


# Global services (initialized in lifespan)
db_service = None
config = None
engines = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    global db_service, config, engines
    
    # Startup
    logging.info("Starting FastAPI server...")
    
    # Load configuration
    config_file = os.environ.get('CONFIG_FILE')
    config = load_config(config_file) if config_file else ConfigService()
    
    # Initialize database service
    db_service = DatabaseService(
        config.database.connection_string,
        config.database.min_connections,
        config.database.max_connections
    )
    
    # Initialize search engines
    engines['wikipedia'] = {
        'simple': SimpleSearchEngine(db_service, config, 'wikipedia'),
        'hybrid': HybridSearchEngine(db_service, config, 'wikipedia'),
        'adaptive': AdaptiveSearchEngine(db_service, config, 'wikipedia')
    }
    
    engines['movies'] = {
        'simple': SimpleSearchEngine(db_service, config, 'movies'),
        'hybrid': HybridSearchEngine(db_service, config, 'movies'),
        'adaptive': AdaptiveSearchEngine(db_service, config, 'movies')
    }
    
    logging.info("FastAPI server started successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down FastAPI server...")
    if db_service:
        db_service.close()


# Initialize FastAPI app
app = FastAPI(
    title="PGVector RAG Search API",
    description="API for searching Wikipedia articles and movie/Netflix data using various vector search methods",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    source: str = Field("wikipedia", description="Data source (wikipedia or movies)")
    method: str = Field("adaptive", description="Search method (simple, hybrid, adaptive)")
    search_type: str = Field("dense", description="Search type for simple method (dense or sparse)")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    dense_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Dense weight for hybrid search")
    sparse_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Sparse weight for hybrid search")
    generate_answer: bool = Field(True, description="Whether to generate an answer")


class SearchResult(BaseModel):
    id: Any = Field(..., description="Result ID")
    content: str = Field(..., description="Result content")
    score: float = Field(..., description="Similarity score")
    source: Optional[str] = Field(None, description="Result source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    method: str = Field(..., description="Search method used")
    source: str = Field(..., description="Data source")
    results: List[SearchResult] = Field(..., description="Search results")
    num_results: int = Field(..., description="Number of results")
    answer: Optional[str] = Field(None, description="Generated answer")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional search metadata")


class ComparisonRequest(BaseModel):
    query: str = Field(..., description="Search query")
    source: str = Field("wikipedia", description="Data source")
    top_k: int = Field(10, ge=1, le=50, description="Number of results per method")


class ComparisonResponse(BaseModel):
    query: str = Field(..., description="Original query")
    source: str = Field(..., description="Data source")
    results: Dict[str, List[SearchResult]] = Field(..., description="Results by method")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Comparison metadata")


class QueryAnalysisResponse(BaseModel):
    query: str = Field(..., description="Analyzed query")
    query_type: str = Field(..., description="Classified query type")
    confidence: float = Field(..., description="Classification confidence")
    recommended_weights: Dict[str, float] = Field(..., description="Recommended search weights")
    features: Dict[str, Any] = Field(..., description="Extracted query features")


# Dependency functions
def get_engine(source: str, method: str):
    """Get search engine for source and method."""
    if source not in engines:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
    if method not in engines[source]:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    return engines[source][method]


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PGVector RAG Search API",
        "version": "1.0.0",
        "description": "API for searching Wikipedia articles and movie/Netflix data",
        "endpoints": {
            "search": "/search",
            "compare": "/compare",
            "analyze": "/analyze",
            "health": "/health"
        },
        "sources": ["wikipedia", "movies"],
        "methods": ["simple", "hybrid", "adaptive"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Test database connection
        with db_service.get_connection():
            pass
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform search using specified method."""
    try:
        engine = get_engine(request.source, request.method)
        
        if request.method == "simple":
            # Simple search
            if request.generate_answer:
                response_data = engine.search_and_answer(
                    query=request.query,
                    search_type=request.search_type,
                    top_k=request.top_k
                )
                results = response_data.get('sources', [])
                answer = response_data.get('answer')
            else:
                # Just search results
                if request.search_type == "dense":
                    search_results = engine.search_dense(request.query, request.top_k)
                else:
                    search_results = engine.search_sparse(request.query, request.top_k)
                results = [
                    {
                        'id': r.id,
                        'content': r.content,
                        'score': r.score,
                        'source': r.source,
                        'metadata': r.metadata
                    }
                    for r in search_results
                ]
                answer = None
        
        elif request.method == "hybrid":
            # Hybrid search
            if request.dense_weight and request.sparse_weight:
                engine.update_weights(request.dense_weight, request.sparse_weight)
            
            if request.generate_answer:
                response_data = engine.generate_answer_from_hybrid(
                    query=request.query,
                    top_k=request.top_k
                )
                results = response_data.get('sources', [])
                answer = response_data.get('answer')
                metadata = {
                    'weights': response_data.get('weights', {}),
                    'cost': response_data.get('cost', 0)
                }
            else:
                search_data = engine.search_hybrid(request.query, request.top_k)
                search_results = search_data['hybrid_results']
                results = [
                    {
                        'id': r.id,
                        'content': r.content,
                        'score': r.score,
                        'source': r.source,
                        'metadata': r.metadata
                    }
                    for r in search_results
                ]
                answer = None
                metadata = {'weights': search_data['weights']}
        
        elif request.method == "adaptive":
            # Adaptive search
            if request.generate_answer:
                response_data = engine.generate_adaptive_answer(
                    query=request.query,
                    top_k=request.top_k
                )
                results = response_data.get('sources', [])
                answer = response_data.get('answer')
                metadata = {
                    'query_analysis': response_data.get('query_analysis', {}),
                    'cost': response_data.get('generation_cost', 0)
                }
            else:
                search_data = engine.search_adaptive(request.query, request.top_k)
                search_results = search_data['results']
                results = [
                    {
                        'id': r.id,
                        'content': r.content,
                        'score': r.score,
                        'source': r.source,
                        'metadata': r.metadata
                    }
                    for r in search_results
                ]
                answer = None
                metadata = {
                    'query_type': search_data.get('query_type'),
                    'classification_confidence': search_data.get('classification_confidence'),
                    'recommended_weights': search_data.get('recommended_weights')
                }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return SearchResponse(
            query=request.query,
            method=request.method,
            source=request.source,
            results=results,
            num_results=len(results),
            answer=answer,
            metadata=metadata if 'metadata' in locals() else None
        )
    
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_methods(request: ComparisonRequest):
    """Compare different search methods."""
    try:
        # Get engines for comparison
        simple_engine = engines[request.source]['simple']
        hybrid_engine = engines[request.source]['hybrid']
        adaptive_engine = engines[request.source]['adaptive']
        
        # Perform searches
        results = {}
        
        # Simple dense
        dense_results = simple_engine.search_dense(request.query, request.top_k)
        results['dense'] = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                source=r.source,
                metadata=r.metadata
            )
            for r in dense_results
        ]
        
        # Simple sparse
        sparse_results = simple_engine.search_sparse(request.query, request.top_k)
        results['sparse'] = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                source=r.source,
                metadata=r.metadata
            )
            for r in sparse_results
        ]
        
        # Hybrid
        hybrid_data = hybrid_engine.search_hybrid(request.query, request.top_k)
        results['hybrid'] = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                source=r.source,
                metadata=r.metadata
            )
            for r in hybrid_data['hybrid_results']
        ]
        
        # Adaptive
        adaptive_data = adaptive_engine.search_adaptive(request.query, request.top_k)
        results['adaptive'] = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                source=r.source,
                metadata=r.metadata
            )
            for r in adaptive_data['results']
        ]
        
        # Metadata
        metadata = {
            'hybrid_weights': hybrid_data.get('weights', {}),
            'adaptive_analysis': {
                'query_type': adaptive_data.get('query_type'),
                'confidence': adaptive_data.get('classification_confidence'),
                'weights': adaptive_data.get('recommended_weights')
            }
        }
        
        return ComparisonResponse(
            query=request.query,
            source=request.source,
            results=results,
            metadata=metadata
        )
    
    except Exception as e:
        logging.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    query: str = Query(..., description="Query to analyze"),
    source: str = Query("wikipedia", description="Data source")
):
    """Analyze query and classify type."""
    try:
        adaptive_engine = engines[source]['adaptive']
        analysis = adaptive_engine.classifier.analyze_query(query)
        
        return QueryAnalysisResponse(
            query=query,
            query_type=analysis.query_type.value,
            confidence=analysis.confidence,
            recommended_weights={
                'dense': analysis.recommended_weights[0],
                'sparse': analysis.recommended_weights[1]
            },
            features=analysis.features
        )
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def get_sources():
    """Get available data sources and their statistics."""
    try:
        stats = {}
        
        for source in engines.keys():
            if source == 'wikipedia':
                # Get Wikipedia stats
                engine = engines[source]['simple']
                with engine.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM articles")
                        total_articles = cur.fetchone()[0]
                        
                        cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector IS NOT NULL")
                        with_dense = cur.fetchone()[0]
                        
                        cur.execute("SELECT COUNT(*) FROM articles WHERE content_sparse IS NOT NULL")
                        with_sparse = cur.fetchone()[0]
                
                stats[source] = {
                    'total_items': total_articles,
                    'with_dense_embeddings': with_dense,
                    'with_sparse_embeddings': with_sparse,
                    'table': 'articles'
                }
            
            elif source == 'movies':
                # Get movies stats
                engine = engines[source]['simple']
                with engine.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM netflix_shows")
                        total_shows = cur.fetchone()[0]
                        
                        cur.execute("SELECT COUNT(*) FROM netflix_shows WHERE embedding IS NOT NULL")
                        with_dense = cur.fetchone()[0]
                        
                        cur.execute("SELECT COUNT(*) FROM netflix_shows WHERE sparse_embedding IS NOT NULL")
                        with_sparse = cur.fetchone()[0]
                
                stats[source] = {
                    'total_items': total_shows,
                    'with_dense_embeddings': with_dense,
                    'with_sparse_embeddings': with_sparse,
                    'table': 'netflix_shows'
                }
        
        return {
            'available_sources': list(engines.keys()),
            'statistics': stats
        }
    
    except Exception as e:
        logging.error(f"Sources error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def setup_logging():
    """Setup logging configuration."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    import uvicorn
    
    setup_logging()
    
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', '8000'))
    workers = int(os.environ.get('API_WORKERS', '1'))
    
    uvicorn.run(
        "fastapi_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )