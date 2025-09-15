"""
Search service for vector similarity and hybrid search operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .database import DatabaseService
from .embeddings import EmbeddingService, SPLADEEmbedder

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for classification."""
    FACTUAL = "factual"  # Specific facts, dates, names
    CONCEPTUAL = "conceptual"  # Abstract concepts, ideas
    EXPLORATORY = "exploratory"  # Broad, open-ended questions
    STRUCTURED = "structured"  # Queries that can be answered with SQL


@dataclass
class SearchResult:
    """Container for search results."""
    id: Any
    content: str
    score: float
    metadata: Dict[str, Any] = None
    source: str = None  # 'dense', 'sparse', 'hybrid', 'sql'


class SearchService(ABC):
    """Abstract base class for search services."""
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a search operation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def get_search_type(self) -> str:
        """Get the type of search performed."""
        pass


class VectorSearch(SearchService):
    """
    Dense vector similarity search using pgvector.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        embedding_service: EmbeddingService,
        table_name: str,
        vector_column: str,
        content_columns: List[str],
        id_column: str = "id",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector search service.
        
        Args:
            db_service: Database service instance
            embedding_service: Embedding service for query encoding
            table_name: Table to search
            vector_column: Name of vector column
            content_columns: Columns to return as content
            id_column: Name of ID column
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.db = db_service
        self.embedder = embedding_service
        self.table_name = table_name
        self.vector_column = vector_column
        self.content_columns = content_columns
        self.id_column = id_column
        self.distance_metric = distance_metric
        
        # Distance operators for pgvector
        self.distance_ops = {
            'cosine': '<=>',
            'l2': '<->',
            'ip': '<#>'  # Inner product (negative for similarity)
        }
        self.distance_op = self.distance_ops.get(distance_metric, '<=>')
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform dense vector similarity search.
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_conditions: Optional SQL WHERE conditions
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embeddings([query])[0]
        
        # Build SQL query
        content_select = ", ".join(self.content_columns)
        
        base_query = f"""
            SELECT 
                {self.id_column},
                {content_select},
                1 - ({self.vector_column} {self.distance_op} %s::vector) as score
            FROM {self.table_name}
        """
        
        if filter_conditions:
            base_query += f" WHERE {filter_conditions}"
        
        base_query += f"""
            ORDER BY {self.vector_column} {self.distance_op} %s::vector
            LIMIT %s
        """
        
        # Execute search
        params = (query_embedding, query_embedding, top_k)
        results = self.db.execute_query(base_query, params, dict_cursor=True)
        
        # Format results
        search_results = []
        for row in results:
            content = " ".join(str(row.get(col, "")) for col in self.content_columns)
            result = SearchResult(
                id=row[self.id_column],
                content=content,
                score=row['score'],
                metadata={k: v for k, v in row.items() if k not in [self.id_column, 'score'] + self.content_columns},
                source='dense'
            )
            search_results.append(result)
        
        return search_results
    
    def get_search_type(self) -> str:
        """Get search type."""
        return "dense_vector"


class SparseSearch(SearchService):
    """
    Sparse vector similarity search using pgvector sparsevec.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        embedding_service: SPLADEEmbedder,
        table_name: str,
        sparse_column: str,
        content_columns: List[str],
        id_column: str = "id"
    ):
        """
        Initialize sparse search service.
        
        Args:
            db_service: Database service instance
            embedding_service: SPLADE embedding service
            table_name: Table to search
            sparse_column: Name of sparse vector column
            content_columns: Columns to return as content
            id_column: Name of ID column
        """
        self.db = db_service
        self.embedder = embedding_service
        self.table_name = table_name
        self.sparse_column = sparse_column
        self.content_columns = content_columns
        self.id_column = id_column
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform sparse vector similarity search.
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_conditions: Optional SQL WHERE conditions
            
        Returns:
            List of search results
        """
        # Generate sparse query embedding
        sparse_embedding = self.embedder.generate_embeddings([query])[0]
        sparse_vector = self.embedder.format_for_pgvector(sparse_embedding)
        
        # Build SQL query
        content_select = ", ".join(self.content_columns)
        
        base_query = f"""
            SELECT 
                {self.id_column},
                {content_select},
                1 - ({self.sparse_column} <=> %s::sparsevec) as score
            FROM {self.table_name}
        """
        
        if filter_conditions:
            base_query += f" WHERE {filter_conditions}"
        
        base_query += f"""
            ORDER BY {self.sparse_column} <=> %s::sparsevec
            LIMIT %s
        """
        
        # Execute search
        params = (sparse_vector, sparse_vector, top_k)
        results = self.db.execute_query(base_query, params, dict_cursor=True)
        
        # Format results
        search_results = []
        for row in results:
            content = " ".join(str(row.get(col, "")) for col in self.content_columns)
            result = SearchResult(
                id=row[self.id_column],
                content=content,
                score=row['score'],
                metadata={k: v for k, v in row.items() if k not in [self.id_column, 'score'] + self.content_columns},
                source='sparse'
            )
            search_results.append(result)
        
        return search_results
    
    def get_search_type(self) -> str:
        """Get search type."""
        return "sparse_vector"


class HybridSearch(SearchService):
    """
    Hybrid search combining dense and sparse vector search.
    """
    
    def __init__(
        self,
        dense_search: VectorSearch,
        sparse_search: SparseSearch,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ):
        """
        Initialize hybrid search service.
        
        Args:
            dense_search: Dense vector search service
            sparse_search: Sparse vector search service
            dense_weight: Weight for dense search results
            sparse_weight: Weight for sparse search results
        """
        self.dense_search = dense_search
        self.sparse_search = sparse_search
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        if abs(dense_weight + sparse_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {dense_weight + sparse_weight}, normalizing to 1.0")
            total = dense_weight + sparse_weight
            self.dense_weight = dense_weight / total
            self.sparse_weight = sparse_weight / total
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse results.
        
        Args:
            query: Search query text
            top_k: Number of results
            rerank: Whether to rerank combined results
            
        Returns:
            List of search results
        """
        # Get results from both searches (fetch more for merging)
        fetch_k = min(top_k * 2, 50)
        dense_results = self.dense_search.search(query, top_k=fetch_k, **kwargs)
        sparse_results = self.sparse_search.search(query, top_k=fetch_k, **kwargs)
        
        if rerank:
            return self._merge_and_rerank(dense_results, sparse_results, top_k)
        else:
            return self._simple_merge(dense_results, sparse_results, top_k)
    
    def _merge_and_rerank(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Merge and rerank results using weighted scores.
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of results to return
            
        Returns:
            Merged and reranked results
        """
        # Create score dictionaries
        dense_scores = {r.id: r.score * self.dense_weight for r in dense_results}
        sparse_scores = {r.id: r.score * self.sparse_weight for r in sparse_results}
        
        # Combine all unique IDs
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Calculate combined scores
        combined_results = []
        result_map = {r.id: r for r in dense_results + sparse_results}
        
        for id in all_ids:
            # Get scores (default to 0 if not present)
            d_score = dense_scores.get(id, 0)
            s_score = sparse_scores.get(id, 0)
            combined_score = d_score + s_score
            
            # Get result object (prefer dense if available)
            if id in result_map:
                result = result_map[id]
                new_result = SearchResult(
                    id=result.id,
                    content=result.content,
                    score=combined_score,
                    metadata=result.metadata,
                    source='hybrid'
                )
                combined_results.append(new_result)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]
    
    def _simple_merge(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Simple merge without reranking (interleave results).
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of results to return
            
        Returns:
            Merged results
        """
        merged = []
        seen_ids = set()
        
        # Interleave results
        for d, s in zip(dense_results, sparse_results):
            if d.id not in seen_ids:
                d.source = 'hybrid'
                merged.append(d)
                seen_ids.add(d.id)
            if len(merged) >= top_k:
                break
            
            if s.id not in seen_ids:
                s.source = 'hybrid'
                merged.append(s)
                seen_ids.add(s.id)
            if len(merged) >= top_k:
                break
        
        # Add remaining if needed
        for r in dense_results + sparse_results:
            if len(merged) >= top_k:
                break
            if r.id not in seen_ids:
                r.source = 'hybrid'
                merged.append(r)
                seen_ids.add(r.id)
        
        return merged[:top_k]
    
    def get_search_type(self) -> str:
        """Get search type."""
        return "hybrid"


class AdaptiveSearch(SearchService):
    """
    Adaptive search that adjusts weights based on query classification.
    """
    
    def __init__(
        self,
        dense_search: VectorSearch,
        sparse_search: SparseSearch,
        classification_service: Optional['QueryClassifier'] = None
    ):
        """
        Initialize adaptive search service.
        
        Args:
            dense_search: Dense vector search service
            sparse_search: Sparse vector search service
            classification_service: Optional query classifier
        """
        self.dense_search = dense_search
        self.sparse_search = sparse_search
        self.classifier = classification_service or QueryClassifier()
        
        # Predefined weights for different query types
        self.weight_profiles = {
            QueryType.FACTUAL: (0.3, 0.7),  # More sparse
            QueryType.CONCEPTUAL: (0.7, 0.3),  # More dense
            QueryType.EXPLORATORY: (0.5, 0.5),  # Balanced
            QueryType.STRUCTURED: (0.2, 0.8)  # Heavily sparse
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform adaptive search with dynamic weight adjustment.
        
        Args:
            query: Search query text
            top_k: Number of results
            
        Returns:
            List of search results
        """
        # Classify query - handle both old and new classifier interfaces
        if hasattr(self.classifier, 'analyze_query'):
            # EnhancedQueryClassifier
            analysis = self.classifier.analyze_query(query)
            query_type = analysis.query_type
            dense_weight, sparse_weight = analysis.recommended_weights
        elif hasattr(self.classifier, 'classify'):
            # Original QueryClassifier
            query_type = self.classifier.classify(query)
            dense_weight, sparse_weight = self.weight_profiles[query_type]
        else:
            # Fallback to balanced weights
            query_type = QueryType.EXPLORATORY
            dense_weight, sparse_weight = 0.5, 0.5
        
        logger.info(f"Query classified as {query_type.value}: using weights dense={dense_weight}, sparse={sparse_weight}")
        
        # Create hybrid search with adaptive weights
        hybrid = HybridSearch(
            self.dense_search,
            self.sparse_search,
            dense_weight,
            sparse_weight
        )
        
        return hybrid.search(query, top_k, **kwargs)
    
    def get_search_type(self) -> str:
        """Get search type."""
        return "adaptive"


class QueryClassifier:
    """
    Classify queries to determine optimal search strategy.
    """
    
    def __init__(self):
        """Initialize query classifier."""
        # Keywords for different query types
        self.factual_keywords = [
            'when', 'where', 'who', 'how many', 'how much', 'date', 'year',
            'name', 'list', 'number', 'count', 'specific', 'exact'
        ]
        self.conceptual_keywords = [
            'why', 'how', 'explain', 'describe', 'understand', 'concept',
            'theory', 'idea', 'meaning', 'significance', 'relationship'
        ]
        self.structured_keywords = [
            'table', 'column', 'row', 'database', 'sql', 'select',
            'filter', 'sort', 'group', 'aggregate', 'join'
        ]
    
    def classify(self, query: str) -> QueryType:
        """
        Classify a query into a type.
        
        Args:
            query: Query text to classify
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower()
        
        # Check for structured query patterns
        if any(keyword in query_lower for keyword in self.structured_keywords):
            return QueryType.STRUCTURED
        
        # Count keyword matches
        factual_score = sum(1 for kw in self.factual_keywords if kw in query_lower)
        conceptual_score = sum(1 for kw in self.conceptual_keywords if kw in query_lower)
        
        # Classify based on scores
        if factual_score > conceptual_score:
            return QueryType.FACTUAL
        elif conceptual_score > factual_score:
            return QueryType.CONCEPTUAL
        else:
            return QueryType.EXPLORATORY