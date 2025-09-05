"""
Database service for managing PostgreSQL connections with pgvector support.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import connection, cursor
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Manages database connections with connection pooling and pgvector support.
    
    Features:
    - Connection pooling for efficient resource management
    - Automatic pgvector registration
    - Retry logic with exponential backoff
    - Context manager support for transactions
    - Query execution helpers
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 20,
        enable_pgvector: bool = True
    ):
        """
        Initialize database service with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string (defaults to DATABASE_URL env)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            enable_pgvector: Whether to register pgvector extension
        """
        self.connection_string = connection_string or os.environ.get('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("Database connection string not provided")
        
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_pgvector = enable_pgvector
        self._pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                self.connection_string
            )
            logger.info(f"Database pool initialized with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> connection:
        """
        Get a connection from the pool as a context manager.
        
        Yields:
            psycopg2 connection object
        """
        conn = None
        try:
            conn = self._pool.getconn()
            if self.enable_pgvector:
                register_vector(conn)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False) -> cursor:
        """
        Get a cursor from a pooled connection.
        
        Args:
            dict_cursor: Whether to use RealDictCursor for dict-like rows
            
        Yields:
            psycopg2 cursor object
        """
        with self.get_connection() as conn:
            cursor_factory = extras.RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
            finally:
                cur.close()
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True,
        dict_cursor: bool = False
    ) -> Optional[List[Any]]:
        """
        Execute a query with optional parameters.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch: Whether to fetch results
            dict_cursor: Whether to return rows as dictionaries
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        with self.get_cursor(dict_cursor=dict_cursor) as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple],
        batch_size: int = 100
    ):
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            batch_size: Batch size for execution
        """
        with self.get_cursor() as cur:
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                extras.execute_batch(cur, query, batch, page_size=batch_size)
                logger.debug(f"Executed batch {i//batch_size + 1} with {len(batch)} items")
    
    def execute_with_retry(
        self,
        query: str,
        params: Optional[Tuple] = None,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Optional[List[Any]]:
        """
        Execute a query with exponential backoff retry logic.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponentially increased)
            
        Returns:
            Query results or None
        """
        for attempt in range(max_retries):
            try:
                return self.execute_query(query, params)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Query attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
        
        return None
    
    def table_exists(self, table_name: str, schema: str = 'public') -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: public)
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """
        result = self.execute_query(query, (schema, table_name))
        return result[0][0] if result else False
    
    def column_exists(self, table_name: str, column_name: str, schema: str = 'public') -> bool:
        """
        Check if a column exists in a table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            schema: Schema name (default: public)
            
        Returns:
            True if column exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
            )
        """
        result = self.execute_query(query, (schema, table_name, column_name))
        return result[0][0] if result else False
    
    def add_vector_column(
        self,
        table_name: str,
        column_name: str,
        dimensions: int = 1536,
        column_type: str = 'vector'
    ):
        """
        Add a vector column to a table if it doesn't exist.
        
        Args:
            table_name: Name of the table
            column_name: Name of the vector column
            dimensions: Vector dimensions (default: 1536 for OpenAI)
            column_type: Type of vector column ('vector' or 'sparsevec')
        """
        if not self.column_exists(table_name, column_name):
            if column_type == 'vector':
                query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} vector({dimensions})"
            elif column_type == 'sparsevec':
                query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} sparsevec({dimensions})"
            else:
                raise ValueError(f"Unknown column type: {column_type}")
            
            self.execute_query(query, fetch=False)
            logger.info(f"Added {column_type} column {column_name} to {table_name}")
    
    def create_index(
        self,
        table_name: str,
        column_name: str,
        index_type: str = 'hnsw',
        distance_metric: str = 'cosine'
    ):
        """
        Create an index on a vector column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            index_type: Type of index ('hnsw' or 'ivfflat')
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        index_name = f"idx_{table_name}_{column_name}_{index_type}"
        
        # Check if index exists
        query = """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE indexname = %s
            )
        """
        exists = self.execute_query(query, (index_name,))[0][0]
        
        if not exists:
            if index_type == 'hnsw':
                ops = {
                    'cosine': 'vector_cosine_ops',
                    'l2': 'vector_l2_ops',
                    'ip': 'vector_ip_ops'
                }
                op = ops.get(distance_metric, 'vector_cosine_ops')
                create_query = f"""
                    CREATE INDEX {index_name} ON {table_name} 
                    USING hnsw ({column_name} {op})
                """
            elif index_type == 'ivfflat':
                ops = {
                    'cosine': 'vector_cosine_ops',
                    'l2': 'vector_l2_ops',
                    'ip': 'vector_ip_ops'
                }
                op = ops.get(distance_metric, 'vector_cosine_ops')
                create_query = f"""
                    CREATE INDEX {index_name} ON {table_name} 
                    USING ivfflat ({column_name} {op}) WITH (lists = 100)
                """
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            self.execute_query(create_query, fetch=False)
            logger.info(f"Created {index_type} index on {table_name}.{column_name}")
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table statistics
        """
        stats = {}
        
        # Row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        stats['row_count'] = self.execute_query(count_query)[0][0]
        
        # Table size
        size_query = f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))"
        stats['total_size'] = self.execute_query(size_query)[0][0]
        
        # Column info
        col_query = """
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        stats['columns'] = self.execute_query(col_query, (table_name,), dict_cursor=True)
        
        return stats
    
    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()