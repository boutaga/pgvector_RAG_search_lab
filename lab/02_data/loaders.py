"""
Data loaders for Wikipedia and Movie/Netflix datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass

from ..core.database import DatabaseService
from .processor import Document, DataProcessor, TextCleaner, DataValidator

logger = logging.getLogger(__name__)


class WikipediaLoader(DataProcessor):
    """
    Loader for Wikipedia articles dataset.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        table_name: str = "articles",
        clean_content: bool = True,
        max_content_length: int = 32000
    ):
        """
        Initialize Wikipedia loader.
        
        Args:
            db_service: Database service
            table_name: Table containing Wikipedia articles
            clean_content: Whether to clean article content
            max_content_length: Maximum content length (for truncation)
        """
        self.db = db_service
        self.table_name = table_name
        self.max_content_length = max_content_length
        
        # Initialize text cleaner if requested
        if clean_content:
            self.cleaner = TextCleaner(
                remove_html=True,
                normalize_whitespace=True,
                remove_special_chars=False
            )
        else:
            self.cleaner = None
        
        # Initialize validator
        self.validator = DataValidator(
            min_length=50,
            max_length=max_content_length * 2,  # Allow some flexibility
            required_fields=['title'],
            forbidden_patterns=[r'^(redirect|disambiguation)', r'^\s*$']
        )
    
    def process(self, data: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load Wikipedia articles from database.
        
        Args:
            data: Optional filter conditions
            
        Returns:
            List of Wikipedia documents
        """
        # Build query
        query = f"""
            SELECT id, title, content
            FROM {self.table_name}
        """
        
        params = None
        if data and 'limit' in data:
            query += " LIMIT %s"
            params = (data['limit'],)
        
        if data and 'offset' in data:
            if params:
                query += " OFFSET %s"
                params = params + (data['offset'],)
            else:
                query += " OFFSET %s"
                params = (data['offset'],)
        
        # Execute query
        results = self.db.execute_query(query, params, dict_cursor=True)
        
        documents = []
        for row in results:
            doc = self._create_document(row)
            if doc and self.validate(doc):
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} Wikipedia articles")
        return documents
    
    def load_by_ids(self, article_ids: List[int]) -> List[Document]:
        """
        Load specific articles by IDs.
        
        Args:
            article_ids: List of article IDs
            
        Returns:
            List of documents
        """
        if not article_ids:
            return []
        
        placeholders = ','.join(['%s'] * len(article_ids))
        query = f"""
            SELECT id, title, content
            FROM {self.table_name}
            WHERE id IN ({placeholders})
        """
        
        results = self.db.execute_query(query, tuple(article_ids), dict_cursor=True)
        
        documents = []
        for row in results:
            doc = self._create_document(row)
            if doc and self.validate(doc):
                documents.append(doc)
        
        return documents
    
    def load_by_title_pattern(self, pattern: str, limit: int = 100) -> List[Document]:
        """
        Load articles matching title pattern.
        
        Args:
            pattern: SQL LIKE pattern for titles
            limit: Maximum number of articles
            
        Returns:
            List of documents
        """
        query = f"""
            SELECT id, title, content
            FROM {self.table_name}
            WHERE title ILIKE %s
            ORDER BY title
            LIMIT %s
        """
        
        results = self.db.execute_query(query, (pattern, limit), dict_cursor=True)
        
        documents = []
        for row in results:
            doc = self._create_document(row)
            if doc and self.validate(doc):
                documents.append(doc)
        
        return documents
    
    def iterate_articles(
        self,
        batch_size: int = 100,
        start_id: int = 0
    ) -> Iterator[List[Document]]:
        """
        Iterate through articles in batches.
        
        Args:
            batch_size: Size of each batch
            start_id: Starting article ID
            
        Yields:
            Batches of documents
        """
        current_id = start_id
        
        while True:
            query = f"""
                SELECT id, title, content
                FROM {self.table_name}
                WHERE id >= %s
                ORDER BY id
                LIMIT %s
            """
            
            results = self.db.execute_query(query, (current_id, batch_size), dict_cursor=True)
            
            if not results:
                break
            
            batch = []
            for row in results:
                doc = self._create_document(row)
                if doc and self.validate(doc):
                    batch.append(doc)
            
            if batch:
                yield batch
            
            # Update current_id for next batch
            current_id = results[-1]['id'] + 1
    
    def _create_document(self, row: Dict[str, Any]) -> Optional[Document]:
        """
        Create document from database row.
        
        Args:
            row: Database row
            
        Returns:
            Document or None if invalid
        """
        title = row.get('title', '').strip()
        content = row.get('content', '').strip()
        
        if not title or not content:
            return None
        
        # Clean content if cleaner is configured
        if self.cleaner:
            content = self.cleaner.clean(content)
            content = self.cleaner.remove_citations(content)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return Document(
            id=row['id'],
            title=title,
            content=content,
            source='wikipedia',
            metadata={
                'title': title,
                'content_length': len(content),
                'source_table': self.table_name
            }
        )
    
    def validate(self, document: Document) -> bool:
        """
        Validate Wikipedia document.
        
        Args:
            document: Document to validate
            
        Returns:
            True if valid
        """
        is_valid, error = self.validator.validate_document(document)
        if not is_valid:
            logger.debug(f"Invalid Wikipedia document {document.id}: {error}")
        return is_valid
    
    def get_article_count(self) -> int:
        """
        Get total number of articles.
        
        Returns:
            Article count
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        result = self.db.execute_query(query)
        return result[0][0] if result else 0
    
    def get_articles_with_embeddings(self, embedding_column: str) -> int:
        """
        Get count of articles with embeddings.
        
        Args:
            embedding_column: Name of embedding column
            
        Returns:
            Count of articles with embeddings
        """
        query = f"""
            SELECT COUNT(*) 
            FROM {self.table_name} 
            WHERE {embedding_column} IS NOT NULL
        """
        result = self.db.execute_query(query)
        return result[0][0] if result else 0


class MovieNetflixLoader(DataProcessor):
    """
    Loader for Movie/Netflix datasets.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        film_table: str = "film",
        netflix_table: str = "netflix_shows"
    ):
        """
        Initialize Movie/Netflix loader.
        
        Args:
            db_service: Database service
            film_table: Table containing film data
            netflix_table: Table containing Netflix shows data
        """
        self.db = db_service
        self.film_table = film_table
        self.netflix_table = netflix_table
        
        # Initialize validator
        self.validator = DataValidator(
            min_length=20,
            max_length=5000,
            forbidden_patterns=[r'^\s*$', r'^null$', r'^n/a$']
        )
    
    def process(self, data: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load movie/Netflix data.
        
        Args:
            data: Optional configuration (table selection, limits)
            
        Returns:
            List of movie/Netflix documents
        """
        documents = []
        
        # Determine which tables to load
        load_films = data is None or data.get('include_films', True)
        load_netflix = data is None or data.get('include_netflix', True)
        
        if load_films:
            film_docs = self.load_films(data)
            documents.extend(film_docs)
        
        if load_netflix:
            netflix_docs = self.load_netflix_shows(data)
            documents.extend(netflix_docs)
        
        logger.info(f"Loaded {len(documents)} movie/Netflix documents")
        return documents
    
    def load_films(self, data: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load film data.
        
        Args:
            data: Optional filter/limit data
            
        Returns:
            List of film documents
        """
        query = f"""
            SELECT film_id, title, description
            FROM {self.film_table}
            WHERE description IS NOT NULL
        """
        
        params = None
        if data and 'limit' in data:
            query += " LIMIT %s"
            params = (data['limit'],)
        
        results = self.db.execute_query(query, params, dict_cursor=True)
        
        documents = []
        for row in results:
            doc = self._create_film_document(row)
            if doc and self.validate(doc):
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} films")
        return documents
    
    def load_netflix_shows(self, data: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load Netflix shows data.
        
        Args:
            data: Optional filter/limit data
            
        Returns:
            List of Netflix show documents
        """
        query = f"""
            SELECT show_id, title, description, type, genre, country, release_year
            FROM {self.netflix_table}
            WHERE description IS NOT NULL
        """
        
        params = None
        if data and 'limit' in data:
            query += " LIMIT %s"
            params = (data['limit'],)
        
        results = self.db.execute_query(query, params, dict_cursor=True)
        
        documents = []
        for row in results:
            doc = self._create_netflix_document(row)
            if doc and self.validate(doc):
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} Netflix shows")
        return documents
    
    def load_customer_rental_history(self, customer_id: int) -> List[Dict[str, Any]]:
        """
        Load customer rental history for recommendations.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of rental records
        """
        query = """
            SELECT f.film_id, f.title, f.description,
                   r.rental_date, r.return_date
            FROM rental r
            JOIN inventory i ON r.inventory_id = i.inventory_id
            JOIN film f ON i.film_id = f.film_id
            WHERE r.customer_id = %s
            ORDER BY r.rental_date DESC
        """
        
        results = self.db.execute_query(query, (customer_id,), dict_cursor=True)
        return results or []
    
    def _create_film_document(self, row: Dict[str, Any]) -> Optional[Document]:
        """
        Create document from film row.
        
        Args:
            row: Database row
            
        Returns:
            Document or None
        """
        title = row.get('title', '').strip()
        description = row.get('description', '').strip()
        
        if not title or not description:
            return None
        
        return Document(
            id=f"film_{row['film_id']}",
            title=title,
            content=description,
            source='dvd_rental_films',
            metadata={
                'film_id': row['film_id'],
                'title': title,
                'content_type': 'film',
                'source_table': self.film_table
            }
        )
    
    def _create_netflix_document(self, row: Dict[str, Any]) -> Optional[Document]:
        """
        Create document from Netflix show row.
        
        Args:
            row: Database row
            
        Returns:
            Document or None
        """
        title = row.get('title', '').strip()
        description = row.get('description', '').strip()
        
        if not title or not description:
            return None
        
        return Document(
            id=f"netflix_{row['show_id']}",
            title=title,
            content=description,
            source='netflix_shows',
            metadata={
                'show_id': row['show_id'],
                'title': title,
                'type': row.get('type'),
                'genre': row.get('genre'),
                'country': row.get('country'),
                'release_year': row.get('release_year'),
                'content_type': 'netflix_show',
                'source_table': self.netflix_table
            }
        )
    
    def validate(self, document: Document) -> bool:
        """
        Validate movie/Netflix document.
        
        Args:
            document: Document to validate
            
        Returns:
            True if valid
        """
        is_valid, error = self.validator.validate_document(document)
        if not is_valid:
            logger.debug(f"Invalid movie document {document.id}: {error}")
        return is_valid
    
    def get_film_count(self) -> int:
        """Get total number of films."""
        query = f"SELECT COUNT(*) FROM {self.film_table}"
        result = self.db.execute_query(query)
        return result[0][0] if result else 0
    
    def get_netflix_count(self) -> int:
        """Get total number of Netflix shows."""
        query = f"SELECT COUNT(*) FROM {self.netflix_table}"
        result = self.db.execute_query(query)
        return result[0][0] if result else 0
    
    def get_films_with_embeddings(self, embedding_column: str = "embedding") -> int:
        """
        Get count of films with embeddings.
        
        Args:
            embedding_column: Name of embedding column
            
        Returns:
            Count of films with embeddings
        """
        query = f"""
            SELECT COUNT(*) 
            FROM {self.film_table} 
            WHERE {embedding_column} IS NOT NULL
        """
        result = self.db.execute_query(query)
        return result[0][0] if result else 0
    
    def get_netflix_with_embeddings(self, embedding_column: str = "embedding") -> int:
        """
        Get count of Netflix shows with embeddings.
        
        Args:
            embedding_column: Name of embedding column
            
        Returns:
            Count of Netflix shows with embeddings
        """
        query = f"""
            SELECT COUNT(*) 
            FROM {self.netflix_table} 
            WHERE {embedding_column} IS NOT NULL
        """
        result = self.db.execute_query(query)
        return result[0][0] if result else 0


class UniversalDataLoader:
    """
    Universal data loader that can handle different data sources.
    """
    
    def __init__(self, db_service: DatabaseService):
        """
        Initialize universal loader.
        
        Args:
            db_service: Database service
        """
        self.db = db_service
        self.loaders = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default loaders."""
        self.loaders['wikipedia'] = WikipediaLoader(self.db)
        self.loaders['movies'] = MovieNetflixLoader(self.db)
    
    def register_loader(self, name: str, loader: DataProcessor):
        """
        Register a custom loader.
        
        Args:
            name: Loader name
            loader: Loader instance
        """
        self.loaders[name] = loader
    
    def load(self, source: str, **kwargs) -> List[Document]:
        """
        Load data from specified source.
        
        Args:
            source: Data source name
            **kwargs: Source-specific parameters
            
        Returns:
            List of documents
        """
        if source not in self.loaders:
            raise ValueError(f"Unknown data source: {source}. Available: {list(self.loaders.keys())}")
        
        loader = self.loaders[source]
        return loader.process(kwargs if kwargs else None)
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.
        
        Returns:
            List of source names
        """
        return list(self.loaders.keys())
    
    def get_source_info(self, source: str) -> Dict[str, Any]:
        """
        Get information about a data source.
        
        Args:
            source: Data source name
            
        Returns:
            Source information
        """
        if source not in self.loaders:
            raise ValueError(f"Unknown data source: {source}")
        
        loader = self.loaders[source]
        info = {
            'name': source,
            'type': type(loader).__name__
        }
        
        # Add source-specific info
        if isinstance(loader, WikipediaLoader):
            info.update({
                'article_count': loader.get_article_count(),
                'table_name': loader.table_name
            })
        elif isinstance(loader, MovieNetflixLoader):
            info.update({
                'film_count': loader.get_film_count(),
                'netflix_count': loader.get_netflix_count(),
                'film_table': loader.film_table,
                'netflix_table': loader.netflix_table
            })
        
        return info