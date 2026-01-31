"""
Timing instrumentation wrappers for measuring embedding and database operations.

This module provides wrappers that track timing for:
- Embedding generation (API calls to OpenAI, local SPLADE inference)
- Database operations (vector similarity queries)

These wrappers are designed for the K-Balance experiment UI to show
where time is spent in the RAG retrieval pipeline.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from abc import ABC

from .embeddings import EmbeddingService, OpenAIEmbedder
from .search import VectorSearch, SearchResult
from .database import DatabaseService

logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    """
    Container for timing measurements across the search pipeline.

    Attributes:
        embed_time_ms: Time spent generating query embeddings (API/model inference)
        db_time_ms: Time spent executing database queries
        total_time_ms: Total end-to-end search time
        additional: Optional dictionary for extra timing details
    """
    embed_time_ms: float = 0.0
    db_time_ms: float = 0.0
    total_time_ms: float = 0.0
    additional: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total if not provided."""
        if self.total_time_ms == 0.0 and (self.embed_time_ms > 0 or self.db_time_ms > 0):
            self.total_time_ms = self.embed_time_ms + self.db_time_ms

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        result = {
            'embed_time_ms': round(self.embed_time_ms, 2),
            'db_time_ms': round(self.db_time_ms, 2),
            'total_time_ms': round(self.total_time_ms, 2)
        }
        if self.additional:
            result['additional'] = {k: round(v, 2) for k, v in self.additional.items()}
        return result


class TimedEmbedder:
    """
    Wrapper for EmbeddingService that tracks embedding generation time.

    This wrapper measures the time spent calling the embedding service,
    which includes API calls (OpenAI) or local model inference (SPLADE).

    Usage:
        embedder = OpenAIEmbedder()
        timed_embedder = TimedEmbedder(embedder)
        embeddings = timed_embedder.generate_embeddings(["query text"])
        print(f"Embed time: {timed_embedder.last_embed_time_ms}ms")
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize timed embedder wrapper.

        Args:
            embedding_service: The underlying embedding service to wrap
        """
        self._embedder = embedding_service
        self._last_embed_time_ms: float = 0.0
        self._total_embed_time_ms: float = 0.0
        self._call_count: int = 0

    @property
    def last_embed_time_ms(self) -> float:
        """Get the time for the most recent embedding call."""
        return self._last_embed_time_ms

    @property
    def total_embed_time_ms(self) -> float:
        """Get cumulative embedding time across all calls."""
        return self._total_embed_time_ms

    @property
    def call_count(self) -> int:
        """Get number of embedding calls made."""
        return self._call_count

    @property
    def avg_embed_time_ms(self) -> float:
        """Get average embedding time per call."""
        if self._call_count == 0:
            return 0.0
        return self._total_embed_time_ms / self._call_count

    def reset_timing(self):
        """Reset all timing counters."""
        self._last_embed_time_ms = 0.0
        self._total_embed_time_ms = 0.0
        self._call_count = 0

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Union[List[float], Dict[int, float]]]:
        """
        Generate embeddings with timing measurement.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override

        Returns:
            List of embeddings (dense vectors or sparse dicts)
        """
        start_time = time.perf_counter()

        embeddings = self._embedder.generate_embeddings(texts, batch_size)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_embed_time_ms = elapsed_ms
        self._total_embed_time_ms += elapsed_ms
        self._call_count += 1

        logger.debug(f"Embedding generation took {elapsed_ms:.2f}ms for {len(texts)} texts")

        return embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions from underlying service."""
        return self._embedder.get_dimensions()

    def get_embedding_type(self) -> str:
        """Get embedding type from underlying service."""
        return self._embedder.get_embedding_type()

    # Pass through other attributes to underlying embedder
    def __getattr__(self, name):
        """Delegate attribute access to underlying embedder."""
        return getattr(self._embedder, name)


class TimedVectorSearch:
    """
    Wrapper for VectorSearch that tracks timing for embed and db operations separately.

    This wrapper provides a breakdown of where time is spent:
    - embed_time_ms: Time generating the query embedding
    - db_time_ms: Time executing the database query

    Usage:
        vector_search = VectorSearch(db, embedder, ...)
        timed_search = TimedVectorSearch(vector_search)
        results = timed_search.search("query", top_k=10)
        timing = timed_search.get_timing_breakdown()
        print(f"Embed: {timing.embed_time_ms}ms, DB: {timing.db_time_ms}ms")
    """

    def __init__(self, vector_search: VectorSearch):
        """
        Initialize timed vector search wrapper.

        Args:
            vector_search: The underlying VectorSearch to wrap
        """
        self._search = vector_search
        self._last_timing: TimingBreakdown = TimingBreakdown()
        self._total_embed_time_ms: float = 0.0
        self._total_db_time_ms: float = 0.0
        self._search_count: int = 0

        # Wrap the embedder with timing
        if hasattr(self._search, 'embedder'):
            self._timed_embedder = TimedEmbedder(self._search.embedder)

    @property
    def last_timing(self) -> TimingBreakdown:
        """Get timing breakdown for the most recent search."""
        return self._last_timing

    def get_timing_breakdown(self) -> TimingBreakdown:
        """Get timing breakdown for the most recent search."""
        return self._last_timing

    def reset_timing(self):
        """Reset all timing counters."""
        self._last_timing = TimingBreakdown()
        self._total_embed_time_ms = 0.0
        self._total_db_time_ms = 0.0
        self._search_count = 0
        if hasattr(self, '_timed_embedder'):
            self._timed_embedder.reset_timing()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform vector search with timing measurement.

        Separates timing into:
        - Embedding generation time (API call)
        - Database query time (vector similarity search)

        Args:
            query: Search query text
            top_k: Number of results
            filter_conditions: Optional SQL WHERE conditions

        Returns:
            List of search results
        """
        total_start = time.perf_counter()

        # Step 1: Generate query embedding (timed)
        embed_start = time.perf_counter()
        query_embedding = self._timed_embedder.generate_embeddings([query])[0]
        embed_time_ms = (time.perf_counter() - embed_start) * 1000.0

        # Step 2: Execute database query (timed)
        db_start = time.perf_counter()

        # Call the internal query execution method directly
        results = self._search._execute_vector_query(
            query_embedding,
            self._search.vector_column,
            top_k,
            filter_conditions,
            "dense"
        )

        db_time_ms = (time.perf_counter() - db_start) * 1000.0

        # Calculate total time
        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        # Store timing breakdown
        self._last_timing = TimingBreakdown(
            embed_time_ms=embed_time_ms,
            db_time_ms=db_time_ms,
            total_time_ms=total_time_ms
        )

        # Update cumulative stats
        self._total_embed_time_ms += embed_time_ms
        self._total_db_time_ms += db_time_ms
        self._search_count += 1

        logger.debug(
            f"Search timing - Embed: {embed_time_ms:.2f}ms, "
            f"DB: {db_time_ms:.2f}ms, Total: {total_time_ms:.2f}ms"
        )

        return results

    def get_cumulative_stats(self) -> Dict[str, float]:
        """Get cumulative timing statistics across all searches."""
        return {
            'total_embed_time_ms': round(self._total_embed_time_ms, 2),
            'total_db_time_ms': round(self._total_db_time_ms, 2),
            'search_count': self._search_count,
            'avg_embed_time_ms': round(self._total_embed_time_ms / max(1, self._search_count), 2),
            'avg_db_time_ms': round(self._total_db_time_ms / max(1, self._search_count), 2)
        }

    def get_search_type(self) -> str:
        """Get search type from underlying service."""
        return self._search.get_search_type()

    # Pass through other attributes to underlying search
    def __getattr__(self, name):
        """Delegate attribute access to underlying search."""
        return getattr(self._search, name)


class TimedDatabaseService:
    """
    Wrapper for DatabaseService that tracks query execution time.

    Useful for detailed timing analysis of database operations.
    """

    def __init__(self, db_service: DatabaseService):
        """
        Initialize timed database service wrapper.

        Args:
            db_service: The underlying DatabaseService to wrap
        """
        self._db = db_service
        self._last_query_time_ms: float = 0.0
        self._total_query_time_ms: float = 0.0
        self._query_count: int = 0

    @property
    def last_query_time_ms(self) -> float:
        """Get time for the most recent query."""
        return self._last_query_time_ms

    def reset_timing(self):
        """Reset all timing counters."""
        self._last_query_time_ms = 0.0
        self._total_query_time_ms = 0.0
        self._query_count = 0

    def execute_query(
        self,
        query: str,
        params: Any = None,
        fetch: bool = True,
        dict_cursor: bool = False
    ):
        """Execute query with timing measurement."""
        start_time = time.perf_counter()

        result = self._db.execute_query(query, params, fetch, dict_cursor)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_query_time_ms = elapsed_ms
        self._total_query_time_ms += elapsed_ms
        self._query_count += 1

        logger.debug(f"Query execution took {elapsed_ms:.2f}ms")

        return result

    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        return {
            'total_query_time_ms': round(self._total_query_time_ms, 2),
            'query_count': self._query_count,
            'avg_query_time_ms': round(self._total_query_time_ms / max(1, self._query_count), 2)
        }

    # Pass through other attributes to underlying db service
    def __getattr__(self, name):
        """Delegate attribute access to underlying db service."""
        return getattr(self._db, name)


# Export public API
__all__ = [
    'TimingBreakdown',
    'TimedEmbedder',
    'TimedVectorSearch',
    'TimedDatabaseService'
]
