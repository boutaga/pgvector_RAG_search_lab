"""
Core service layer for pgvector RAG lab.

This module provides the foundational services for database operations,
embeddings, search, ranking, generation, and configuration management.
"""

from .config import ConfigService
from .database import DatabaseService
from .embeddings import (
    EmbeddingService,
    OpenAIEmbedder,
    SPLADEEmbedder
)
from .search import (
    SearchService,
    VectorSearch,
    SparseSearch,
    HybridSearch,
    AdaptiveSearch
)
from .ranking import RankingService
from .generation import GenerationService

__all__ = [
    'ConfigService',
    'DatabaseService',
    'EmbeddingService',
    'OpenAIEmbedder',
    'SPLADEEmbedder',
    'SearchService',
    'VectorSearch', 
    'SparseSearch',
    'HybridSearch',
    'AdaptiveSearch',
    'RankingService',
    'GenerationService'
]

__version__ = '0.1.0'