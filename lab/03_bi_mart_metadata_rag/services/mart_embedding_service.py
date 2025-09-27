#!/usr/bin/env python3
"""
Embedding service for metadata RAG.
Adapted from the existing embedding service for metadata-specific use cases.
"""

import os
import sys
import openai
import psycopg2
from typing import List, Dict, Any, Optional
import time
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 50
    max_retries: int = 3
    retry_delay: int = 2
    max_tokens_per_text: int = 8000

class MetadataEmbeddingService:
    """
    Embedding service specialized for database metadata.
    Generates embeddings for table metadata, column metadata, and relationships.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding service."""
        self.config = config or EmbeddingConfig()

        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

        logger.info(f"Initialized embedding service with model: {self.config.model}")

    def get_db_connection(self):
        """Create a database connection."""
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            return psycopg2.connect(db_url)
        else:
            return psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                port=os.environ.get('DB_PORT', '5432'),
                database=os.environ.get('DB_NAME', 'postgres'),
                user=os.environ.get('DB_USER', 'postgres'),
                password=os.environ.get('DB_PASSWORD', '')
            )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)

        if not valid_texts:
            # Return zero embeddings for empty texts
            return [[0.0] * self.config.dimensions for _ in texts]

        # Generate embeddings with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=valid_texts,
                    dimensions=self.config.dimensions
                )

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                # Create result list with zero embeddings for empty texts
                result = []
                valid_idx = 0
                for i in range(len(texts)):
                    if i in valid_indices:
                        result.append(embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        result.append([0.0] * self.config.dimensions)

                return result

            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise

        return []

    def embed_table_metadata(self, conn) -> int:
        """Generate embeddings for table metadata."""
        logger.info("Generating embeddings for table metadata...")

        with conn.cursor() as cursor:
            # Get table metadata without embeddings
            cursor.execute("""
                SELECT id, metadata_text
                FROM catalog.table_metadata
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cursor.fetchall()
            if not rows:
                logger.info("No table metadata to embed")
                return 0

            logger.info(f"Found {len(rows)} table metadata records to embed")

            # Process in batches
            embedded_count = 0
            for i in range(0, len(rows), self.config.batch_size):
                batch = rows[i:i + self.config.batch_size]
                ids = [row[0] for row in batch]
                texts = [row[1] or '' for row in batch]

                # Generate embeddings
                embeddings = self.generate_embeddings(texts)

                # Update database
                for row_id, embedding in zip(ids, embeddings):
                    if embedding and any(embedding):  # Skip zero embeddings
                        cursor.execute("""
                            UPDATE catalog.table_metadata
                            SET embedding = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (embedding, row_id))
                        embedded_count += 1

                conn.commit()
                logger.info(f"Embedded batch {i//self.config.batch_size + 1}: {len(embeddings)} embeddings")

        logger.info(f"✓ Generated embeddings for {embedded_count} table metadata records")
        return embedded_count

    def embed_column_metadata(self, conn) -> int:
        """Generate embeddings for column metadata."""
        logger.info("Generating embeddings for column metadata...")

        with conn.cursor() as cursor:
            # Get column metadata without embeddings
            cursor.execute("""
                SELECT id, metadata_text
                FROM catalog.column_metadata
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cursor.fetchall()
            if not rows:
                logger.info("No column metadata to embed")
                return 0

            logger.info(f"Found {len(rows)} column metadata records to embed")

            # Process in batches
            embedded_count = 0
            for i in range(0, len(rows), self.config.batch_size):
                batch = rows[i:i + self.config.batch_size]
                ids = [row[0] for row in batch]
                texts = [row[1] or '' for row in batch]

                # Generate embeddings
                embeddings = self.generate_embeddings(texts)

                # Update database
                for row_id, embedding in zip(ids, embeddings):
                    if embedding and any(embedding):  # Skip zero embeddings
                        cursor.execute("""
                            UPDATE catalog.column_metadata
                            SET embedding = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (embedding, row_id))
                        embedded_count += 1

                conn.commit()
                logger.info(f"Embedded batch {i//self.config.batch_size + 1}: {len(embeddings)} embeddings")

        logger.info(f"✓ Generated embeddings for {embedded_count} column metadata records")
        return embedded_count

    def embed_relationship_metadata(self, conn) -> int:
        """Generate embeddings for relationship metadata."""
        logger.info("Generating embeddings for relationship metadata...")

        with conn.cursor() as cursor:
            # Get relationship metadata without embeddings
            cursor.execute("""
                SELECT id, metadata_text
                FROM catalog.relationship_metadata
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cursor.fetchall()
            if not rows:
                logger.info("No relationship metadata to embed")
                return 0

            logger.info(f"Found {len(rows)} relationship metadata records to embed")

            # Process in batches
            embedded_count = 0
            for i in range(0, len(rows), self.config.batch_size):
                batch = rows[i:i + self.config.batch_size]
                ids = [row[0] for row in batch]
                texts = [row[1] or '' for row in batch]

                # Generate embeddings
                embeddings = self.generate_embeddings(texts)

                # Update database
                for row_id, embedding in zip(ids, embeddings):
                    if embedding and any(embedding):  # Skip zero embeddings
                        cursor.execute("""
                            UPDATE catalog.relationship_metadata
                            SET embedding = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (embedding, row_id))
                        embedded_count += 1

                conn.commit()
                logger.info(f"Embedded batch {i//self.config.batch_size + 1}: {len(embeddings)} embeddings")

        logger.info(f"✓ Generated embeddings for {embedded_count} relationship metadata records")
        return embedded_count

    def embed_suggested_kpis(self, conn) -> int:
        """Generate embeddings for suggested KPIs."""
        logger.info("Generating embeddings for suggested KPIs...")

        with conn.cursor() as cursor:
            # Get KPI metadata without embeddings
            cursor.execute("""
                SELECT id, metadata_text
                FROM catalog.suggested_kpis
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cursor.fetchall()
            if not rows:
                logger.info("No KPI metadata to embed")
                return 0

            logger.info(f"Found {len(rows)} KPI metadata records to embed")

            # Process in batches
            embedded_count = 0
            for i in range(0, len(rows), self.config.batch_size):
                batch = rows[i:i + self.config.batch_size]
                ids = [row[0] for row in batch]
                texts = [row[1] or '' for row in batch]

                # Generate embeddings
                embeddings = self.generate_embeddings(texts)

                # Update database
                for row_id, embedding in zip(ids, embeddings):
                    if embedding and any(embedding):  # Skip zero embeddings
                        cursor.execute("""
                            UPDATE catalog.suggested_kpis
                            SET embedding = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (embedding, row_id))
                        embedded_count += 1

                conn.commit()
                logger.info(f"Embedded batch {i//self.config.batch_size + 1}: {len(embeddings)} embeddings")

        logger.info(f"✓ Generated embeddings for {embedded_count} KPI metadata records")
        return embedded_count

    def embed_all_metadata(self) -> Dict[str, int]:
        """Generate embeddings for all metadata types."""
        logger.info("Starting metadata embedding generation...")

        results = {}
        conn = self.get_db_connection()

        try:
            # Embed all metadata types
            results['tables'] = self.embed_table_metadata(conn)
            results['columns'] = self.embed_column_metadata(conn)
            results['relationships'] = self.embed_relationship_metadata(conn)
            results['kpis'] = self.embed_suggested_kpis(conn)

            total = sum(results.values())
            logger.info(f"✓ Total embeddings generated: {total}")

            return results

        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about embedding completion."""
        conn = self.get_db_connection()

        try:
            with conn.cursor() as cursor:
                stats = {}

                # Table metadata stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(embedding) as with_embeddings
                    FROM catalog.table_metadata
                """)
                result = cursor.fetchone()
                stats['tables'] = {
                    'total': result[0],
                    'with_embeddings': result[1],
                    'completion_pct': (result[1] / result[0] * 100) if result[0] > 0 else 0
                }

                # Column metadata stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(embedding) as with_embeddings
                    FROM catalog.column_metadata
                """)
                result = cursor.fetchone()
                stats['columns'] = {
                    'total': result[0],
                    'with_embeddings': result[1],
                    'completion_pct': (result[1] / result[0] * 100) if result[0] > 0 else 0
                }

                # Relationship metadata stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(embedding) as with_embeddings
                    FROM catalog.relationship_metadata
                """)
                result = cursor.fetchone()
                stats['relationships'] = {
                    'total': result[0],
                    'with_embeddings': result[1],
                    'completion_pct': (result[1] / result[0] * 100) if result[0] > 0 else 0
                }

                # KPI metadata stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(embedding) as with_embeddings
                    FROM catalog.suggested_kpis
                """)
                result = cursor.fetchone()
                stats['kpis'] = {
                    'total': result[0],
                    'with_embeddings': result[1],
                    'completion_pct': (result[1] / result[0] * 100) if result[0] > 0 else 0
                }

                return stats

        finally:
            conn.close()