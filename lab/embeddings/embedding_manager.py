"""
Embedding generation manager and orchestration.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.database import DatabaseService
from ..core.embeddings import EmbeddingService, OpenAIEmbedder, SPLADEEmbedder
from ..core.config import ConfigService
from ..data.loaders import UniversalDataLoader, Document

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Types of embeddings."""
    DENSE = "dense"
    SPARSE = "sparse"
    BOTH = "both"


@dataclass
class EmbeddingJob:
    """Configuration for an embedding generation job."""
    source: str  # wikipedia, movies
    embedding_type: EmbeddingType
    table_name: str
    content_columns: List[str]  # Columns to generate embeddings for
    embedding_columns: List[str]  # Where to store embeddings
    batch_size: Optional[int] = None
    start_id: int = 0
    limit: Optional[int] = None
    update_existing: bool = False


@dataclass
class EmbeddingProgress:
    """Progress tracking for embedding generation."""
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    current_batch: int
    total_batches: int
    errors: List[str]


class EmbeddingManager:
    """
    Manager for embedding generation operations.
    
    Features:
    - Orchestrates embedding generation for different data sources
    - Manages dense and sparse embedding services
    - Provides progress tracking and error handling
    - Supports incremental updates
    - Handles batch processing with retry logic
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        config: Optional[ConfigService] = None
    ):
        """
        Initialize embedding manager.
        
        Args:
            db_service: Database service
            config: Configuration service
        """
        self.db = db_service
        self.config = config
        self.data_loader = UniversalDataLoader(db_service)
        
        # Initialize embedders
        self.dense_embedder = None
        self.sparse_embedder = None
        self._initialize_embedders()
    
    def _initialize_embedders(self):
        """Initialize embedding services based on configuration."""
        if self.config:
            # Dense embedder (OpenAI)
            self.dense_embedder = OpenAIEmbedder(
                model=self.config.embedding.openai_model,
                dimensions=self.config.embedding.openai_dimensions,
                batch_size=self.config.embedding.batch_size_dense,
                max_retries=self.config.embedding.max_retries,
                retry_delay=self.config.embedding.retry_delay
            )
            
            # Sparse embedder (SPLADE)
            self.sparse_embedder = SPLADEEmbedder(
                model_name=self.config.embedding.splade_model,
                dimensions=self.config.embedding.splade_dimensions,
                batch_size=self.config.embedding.batch_size_sparse
            )
        else:
            # Use defaults
            self.dense_embedder = OpenAIEmbedder()
            self.sparse_embedder = SPLADEEmbedder()
    
    def generate_embeddings(self, job: EmbeddingJob) -> EmbeddingProgress:
        """
        Generate embeddings according to job specification.
        
        Args:
            job: Embedding generation job
            
        Returns:
            Progress tracking object
        """
        logger.info(f"Starting embedding generation job: {job.source} - {job.embedding_type.value}")
        
        # Initialize progress tracking
        progress = EmbeddingProgress(
            total_items=0,
            processed_items=0,
            successful_items=0,
            failed_items=0,
            current_batch=0,
            total_batches=0,
            errors=[]
        )
        
        # Get total count for progress tracking
        progress.total_items = self._get_item_count(job)
        if progress.total_items == 0:
            logger.warning("No items found for embedding generation")
            return progress
        
        # Calculate batch size
        batch_size = job.batch_size or self._get_default_batch_size(job.embedding_type)
        progress.total_batches = (progress.total_items + batch_size - 1) // batch_size
        
        # Process in batches
        for batch in self._iterate_batches(job, batch_size):
            progress.current_batch += 1
            
            try:
                batch_success = self._process_batch(job, batch, progress)
                if batch_success:
                    progress.successful_items += len(batch)
                else:
                    progress.failed_items += len(batch)
            except Exception as e:
                error_msg = f"Batch {progress.current_batch} failed: {str(e)}"
                logger.error(error_msg)
                progress.errors.append(error_msg)
                progress.failed_items += len(batch)
            
            progress.processed_items += len(batch)
            
            # Log progress
            if progress.current_batch % 10 == 0:
                self._log_progress(progress)
        
        # Final progress log
        self._log_progress(progress)
        logger.info(f"Embedding generation completed: {progress.successful_items}/{progress.total_items} successful")
        
        return progress
    
    def _get_item_count(self, job: EmbeddingJob) -> int:
        """Get total number of items to process."""
        if job.update_existing:
            query = f"SELECT COUNT(*) FROM {job.table_name}"
        else:
            # Count items without embeddings
            where_clauses = []
            for col in job.embedding_columns:
                where_clauses.append(f"{col} IS NULL")
            where_clause = " OR ".join(where_clauses)
            query = f"SELECT COUNT(*) FROM {job.table_name} WHERE {where_clause}"
        
        if job.limit:
            query += f" LIMIT {job.limit}"
        
        result = self.db.execute_query(query)
        return result[0][0] if result else 0
    
    def _get_default_batch_size(self, embedding_type: EmbeddingType) -> int:
        """Get default batch size for embedding type."""
        if embedding_type == EmbeddingType.DENSE:
            return self.config.embedding.batch_size_dense if self.config else 50
        elif embedding_type == EmbeddingType.SPARSE:
            return self.config.embedding.batch_size_sparse if self.config else 5
        else:  # BOTH
            return min(
                self.config.embedding.batch_size_dense if self.config else 50,
                self.config.embedding.batch_size_sparse if self.config else 5
            )
    
    def _iterate_batches(self, job: EmbeddingJob, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Iterate through data in batches."""
        offset = job.start_id
        
        while True:
            # Build query for batch
            if job.update_existing:
                query = f"""
                    SELECT id, {', '.join(job.content_columns)}
                    FROM {job.table_name}
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                params = (batch_size, offset)
            else:
                # Only get items without embeddings
                where_clauses = []
                for col in job.embedding_columns:
                    where_clauses.append(f"{col} IS NULL")
                where_clause = " OR ".join(where_clauses)
                
                query = f"""
                    SELECT id, {', '.join(job.content_columns)}
                    FROM {job.table_name}
                    WHERE {where_clause}
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                params = (batch_size, offset)
            
            results = self.db.execute_query(query, params, dict_cursor=True)
            
            if not results:
                break
            
            yield results
            offset += len(results)
            
            # Check limit
            if job.limit and offset >= job.limit:
                break
    
    def _process_batch(
        self,
        job: EmbeddingJob,
        batch: List[Dict[str, Any]],
        progress: EmbeddingProgress
    ) -> bool:
        """
        Process a single batch of items.
        
        Args:
            job: Embedding job configuration
            batch: Batch of database rows
            progress: Progress tracker
            
        Returns:
            True if batch processed successfully
        """
        try:
            # Extract content for each column
            for col_idx, content_col in enumerate(job.content_columns):
                embedding_col = job.embedding_columns[col_idx]
                
                # Extract texts
                texts = []
                valid_indices = []
                for i, row in enumerate(batch):
                    text = row.get(content_col, '')
                    if text and str(text).strip():
                        texts.append(str(text).strip())
                        valid_indices.append(i)
                    else:
                        texts.append('')
                        valid_indices.append(i)
                
                if not any(texts):
                    continue
                
                # Generate embeddings based on type
                embeddings = None
                if job.embedding_type == EmbeddingType.DENSE:
                    embeddings = self._generate_dense_embeddings(texts)
                elif job.embedding_type == EmbeddingType.SPARSE:
                    embeddings = self._generate_sparse_embeddings(texts)
                elif job.embedding_type == EmbeddingType.BOTH:
                    # Handle both types - this would need column mapping
                    dense_embeddings = self._generate_dense_embeddings(texts)
                    sparse_embeddings = self._generate_sparse_embeddings(texts)
                    # For now, prioritize dense
                    embeddings = dense_embeddings
                
                if embeddings:
                    self._update_embeddings(job.table_name, batch, embedding_col, embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process batch: {str(e)}")
            progress.errors.append(str(e))
            return False
    
    def _approx_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return max(1, len(text) // 4)

    def _truncate_text(self, text: str, max_tokens: int = 7900) -> str:
        """Truncate text to fit within token limit."""
        if self._approx_tokens(text) <= max_tokens:
            return text
        max_chars = max_tokens * 4
        return text[:max_chars]

    def _chunk_text(self, text: str, max_tokens: int = 7500) -> List[str]:
        """Split text into chunks that fit within token limits.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (conservative to avoid errors)

        Returns:
            List of text chunks
        """
        # If text fits in one chunk, return as-is
        if self._approx_tokens(text) <= max_tokens:
            return [text]

        # Calculate chunk size in characters (conservative estimate)
        chunk_size_chars = max_tokens * 3  # More conservative: 3 chars per token
        overlap_chars = 200  # Small overlap to maintain context

        chunks = []
        position = 0
        text_length = len(text)

        while position < text_length:
            # Calculate end position for this chunk
            end_pos = min(position + chunk_size_chars, text_length)

            # Extract chunk
            chunk = text[position:end_pos]

            # Try to break at a sentence or word boundary if not at the end
            if end_pos < text_length:
                # Look for sentence end (.!?) in the last 100 chars
                last_sentence = max(
                    chunk.rfind('. '),
                    chunk.rfind('! '),
                    chunk.rfind('? '),
                    chunk.rfind('\n\n')  # Paragraph break
                )

                if last_sentence > len(chunk) * 0.8:  # If we found a good break point
                    chunk = chunk[:last_sentence + 1]
                    end_pos = position + last_sentence + 1
                else:
                    # Fall back to word boundary
                    last_space = chunk.rfind(' ')
                    if last_space > len(chunk) * 0.8:
                        chunk = chunk[:last_space]
                        end_pos = position + last_space

            chunks.append(chunk.strip())

            # Move position forward with small overlap
            position = end_pos - overlap_chars if end_pos < text_length else end_pos

        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Average multiple embeddings into a single vector.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Averaged embedding vector
        """
        if not embeddings:
            return []

        if len(embeddings) == 1:
            return embeddings[0]

        # Convert to numpy array for efficient averaging
        embedding_array = np.array(embeddings)

        # Average across chunks (axis=0)
        averaged = np.mean(embedding_array, axis=0)

        # Normalize the averaged vector to maintain semantic properties
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged.tolist()

    def _generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings with chunking and averaging for long texts.

        For texts that exceed token limits, this method:
        1. Splits the text into manageable chunks
        2. Generates embeddings for each chunk
        3. Averages the chunk embeddings into a single vector

        This preserves semantic information from the entire document
        while respecting API token limits.
        """
        if not self.dense_embedder:
            raise ValueError("Dense embedder not initialized")

        final_embeddings = []

        for text in texts:
            if not text or not text.strip():
                # Empty text gets zero embedding
                final_embeddings.append([0.0] * self.config.embeddings.dimensions)
                continue

            # Check if we need to chunk this text
            if self._approx_tokens(text) > 7500:
                # Text is too long, use chunking with averaging
                chunks = self._chunk_text(text.strip())

                if len(chunks) > 1:
                    logger.info(f"Chunking long text into {len(chunks)} chunks for embedding")

                # Generate embeddings for all chunks
                chunk_embeddings = self.dense_embedder.generate_embeddings(chunks)

                # Average the chunk embeddings
                averaged_embedding = self._average_embeddings(chunk_embeddings)
                final_embeddings.append(averaged_embedding)

                if len(chunks) > 1:
                    logger.debug(f"Successfully averaged {len(chunks)} chunk embeddings")
            else:
                # Text fits in one chunk, process normally
                embedding = self.dense_embedder.generate_embeddings([text.strip()])
                final_embeddings.extend(embedding)

        return final_embeddings
    
    def _generate_sparse_embeddings(self, texts: List[str]) -> List[str]:
        """Generate sparse embeddings in pgvector format."""
        if not self.sparse_embedder:
            raise ValueError("Sparse embedder not initialized")
        
        sparse_dicts = self.sparse_embedder.generate_embeddings(texts)
        return [self.sparse_embedder.format_for_pgvector(d) for d in sparse_dicts]
    
    def _update_embeddings(
        self,
        table_name: str,
        batch: List[Dict[str, Any]],
        embedding_column: str,
        embeddings: List[Any]
    ):
        """Update embeddings in database."""
        update_data = []
        for row, embedding in zip(batch, embeddings):
            if embedding:  # Skip empty embeddings
                update_data.append((embedding, row['id']))
        
        if not update_data:
            return
        
        # Batch update
        query = f"UPDATE {table_name} SET {embedding_column} = %s WHERE id = %s"
        self.db.execute_many(query, update_data)
    
    def _log_progress(self, progress: EmbeddingProgress):
        """Log current progress."""
        percentage = (progress.processed_items / progress.total_items * 100) if progress.total_items > 0 else 0
        logger.info(
            f"Progress: {progress.processed_items}/{progress.total_items} ({percentage:.1f}%) "
            f"- Success: {progress.successful_items}, Failed: {progress.failed_items}"
        )
    
    def create_wikipedia_dense_job(
        self,
        update_existing: bool = False,
        limit: Optional[int] = None
    ) -> EmbeddingJob:
        """Create job for Wikipedia dense embeddings (3072-dimension)."""
        return EmbeddingJob(
            source="wikipedia",
            embedding_type=EmbeddingType.DENSE,
            table_name="articles",
            content_columns=["title", "content"],
            embedding_columns=["title_vector_3072", "content_vector_3072"],
            update_existing=update_existing,
            limit=limit
        )
    
    def create_wikipedia_sparse_job(
        self,
        update_existing: bool = False,
        limit: Optional[int] = None
    ) -> EmbeddingJob:
        """Create job for Wikipedia sparse embeddings."""
        return EmbeddingJob(
            source="wikipedia",
            embedding_type=EmbeddingType.SPARSE,
            table_name="articles",
            content_columns=["title", "content"],
            embedding_columns=["title_sparse", "content_sparse"],
            update_existing=update_existing,
            limit=limit
        )
    
    def create_movie_dense_job(
        self,
        include_films: bool = True,
        include_netflix: bool = True,
        update_existing: bool = False,
        limit: Optional[int] = None
    ) -> List[EmbeddingJob]:
        """Create jobs for movie data dense embeddings."""
        jobs = []
        
        if include_films:
            jobs.append(EmbeddingJob(
                source="movies",
                embedding_type=EmbeddingType.DENSE,
                table_name="film",
                content_columns=["description"],
                embedding_columns=["embedding"],
                update_existing=update_existing,
                limit=limit
            ))
        
        if include_netflix:
            jobs.append(EmbeddingJob(
                source="movies",
                embedding_type=EmbeddingType.DENSE,
                table_name="netflix_shows",
                content_columns=["description"],
                embedding_columns=["embedding"],
                update_existing=update_existing,
                limit=limit
            ))
        
        return jobs
    
    def create_movie_sparse_job(
        self,
        include_netflix: bool = True,
        update_existing: bool = False,
        limit: Optional[int] = None
    ) -> List[EmbeddingJob]:
        """Create jobs for movie data sparse embeddings."""
        jobs = []
        
        # Only Netflix has sparse embeddings in original setup
        if include_netflix:
            jobs.append(EmbeddingJob(
                source="movies",
                embedding_type=EmbeddingType.SPARSE,
                table_name="netflix_shows",
                content_columns=["description"],
                embedding_columns=["sparse_embedding"],
                update_existing=update_existing,
                limit=limit
            ))
        
        return jobs
    
    def get_embedding_statistics(self, table_name: str, embedding_column: str) -> Dict[str, Any]:
        """Return counts and completion % for one embedding column."""
        stats: Dict[str, Any] = {}

        total_query = f"SELECT COUNT(*) FROM {table_name}"
        embed_query = f"SELECT COUNT(*) FROM {table_name} WHERE {embedding_column} IS NOT NULL"

        total_rows = self.db.execute_query(total_query)[0][0]
        rows_with = self.db.execute_query(embed_query)[0][0]

        stats["total_rows"] = total_rows
        stats["rows_with_embeddings"] = rows_with
        stats["rows_missing_embeddings"] = total_rows - rows_with
        stats["completion_percentage"] = (rows_with / total_rows * 100) if total_rows > 0 else 0.0
        return stats

    def verify_embeddings(self, job: EmbeddingJob) -> Dict[str, Any]:
        """
        Verify embedding generation results for all embedding columns in a job.
        Returns a dict keyed by embedding column name.
        """
        results: Dict[str, Any] = {}
        for col in job.embedding_columns:
            results[col] = self.get_embedding_statistics(job.table_name, col)
        return results