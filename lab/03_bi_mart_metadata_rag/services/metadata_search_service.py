#!/usr/bin/env python3
"""
RAG search service for database metadata.
Adapted from the existing adaptive search for metadata-specific queries.
"""

import os
import sys
import psycopg2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
import logging
import json
from pgvector.psycopg2 import register_vector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of metadata queries."""
    METRIC = "metric"          # Looking for measures/KPIs
    DIMENSION = "dimension"    # Looking for dimensions/grouping
    RELATIONSHIP = "relationship"  # Looking for joins/connections
    TABLE = "table"           # Looking for specific tables
    GENERAL = "general"       # General schema exploration

@dataclass
class SearchResult:
    """Search result from metadata catalog."""
    id: int
    metadata_type: str  # 'table', 'column', 'relationship', 'kpi'
    schema_name: str
    table_name: str
    column_name: Optional[str]
    description: str
    metadata_text: str
    similarity_score: float
    additional_info: Dict[str, Any]

@dataclass
class SearchConfig:
    """Configuration for metadata search."""
    top_k: int = 10
    similarity_threshold: float = 0.7
    include_relationships: bool = True
    boost_primary_keys: bool = True
    boost_foreign_keys: bool = True
    boost_measures: bool = True

class MetadataSearchService:
    """
    RAG search service for database metadata.
    Performs semantic search across table, column, relationship, and KPI metadata.
    """

    def __init__(self):
        """Initialize the search service."""
        # Initialize OpenAI client for query embeddings
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536

    def get_db_connection(self):
        """Create a database connection."""
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            conn = psycopg2.connect(db_url)
        else:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                port=os.environ.get('DB_PORT', '5432'),
                database=os.environ.get('DB_NAME', 'postgres'),
                user=os.environ.get('DB_USER', 'postgres'),
                password=os.environ.get('DB_PASSWORD', '')
            )
        # Register vector types
        register_vector(conn)
        return conn

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=query.strip(),
                dimensions=self.embedding_dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise

    def classify_query_type(self, query: str) -> QueryType:
        """Classify the type of metadata query."""
        query_lower = query.lower()

        # Keywords that suggest different query types
        metric_keywords = [
            'sales', 'revenue', 'profit', 'total', 'sum', 'count', 'average', 'max', 'min',
            'kpi', 'metric', 'measure', 'performance', 'amount', 'value', 'cost',
            'fastest', 'top', 'bottom', 'growth', 'trend'
        ]

        dimension_keywords = [
            'by', 'group', 'category', 'region', 'time', 'date', 'customer', 'product',
            'segment', 'type', 'status', 'location', 'territory', 'supplier'
        ]

        relationship_keywords = [
            'join', 'connect', 'related', 'link', 'association', 'foreign key',
            'relationship', 'dependency', 'hierarchy'
        ]

        table_keywords = [
            'table', 'entity', 'data', 'information', 'records'
        ]

        # Count keyword matches
        metric_score = sum(1 for keyword in metric_keywords if keyword in query_lower)
        dimension_score = sum(1 for keyword in dimension_keywords if keyword in query_lower)
        relationship_score = sum(1 for keyword in relationship_keywords if keyword in query_lower)
        table_score = sum(1 for keyword in table_keywords if keyword in query_lower)

        # Determine query type based on highest score
        scores = {
            QueryType.METRIC: metric_score,
            QueryType.DIMENSION: dimension_score,
            QueryType.RELATIONSHIP: relationship_score,
            QueryType.TABLE: table_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.GENERAL

        # Return the type with the highest score
        for query_type, score in scores.items():
            if score == max_score:
                return query_type

        return QueryType.GENERAL

    def search_table_metadata(
        self,
        query_embedding: List[float],
        config: SearchConfig
    ) -> List[SearchResult]:
        """Search table metadata."""
        conn = self.get_db_connection()

        try:
            with conn.cursor() as cursor:
                # Use JSON format for vector (consistent with other labs)
                embedding_json = json.dumps(query_embedding)

                cursor.execute("""
                    SELECT
                        id,
                        schema_name,
                        table_name,
                        table_type,
                        row_count,
                        table_comment,
                        metadata_text,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM catalog.table_metadata
                    WHERE embedding IS NOT NULL
                        AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (embedding_json, embedding_json, config.similarity_threshold, config.top_k))

                results = []
                for row in cursor.fetchall():
                    results.append(SearchResult(
                        id=row[0],
                        metadata_type='table',
                        schema_name=row[1],
                        table_name=row[2],
                        column_name=None,
                        description=row[5] or f"{row[2]} table",
                        metadata_text=row[6],
                        similarity_score=row[7],
                        additional_info={
                            'table_type': row[3],
                            'row_count': row[4]
                        }
                    ))

                return results

        finally:
            conn.close()

    def search_column_metadata(
        self,
        query_embedding: List[float],
        config: SearchConfig,
        query_type: QueryType
    ) -> List[SearchResult]:
        """Search column metadata with type-specific boosting."""
        conn = self.get_db_connection()

        try:
            with conn.cursor() as cursor:
                # Build boost conditions based on query type
                boost_conditions = []
                if query_type == QueryType.METRIC and config.boost_measures:
                    boost_conditions.append("""
                        CASE
                            WHEN data_type IN ('numeric', 'integer', 'bigint', 'real', 'double precision')
                                AND (column_name LIKE '%price%' OR column_name LIKE '%amount%'
                                     OR column_name LIKE '%cost%' OR column_name LIKE '%total%'
                                     OR column_name LIKE '%quantity%' OR column_name LIKE '%revenue%')
                            THEN 0.1
                            ELSE 0.0
                        END
                    """)

                if config.boost_primary_keys:
                    boost_conditions.append("""
                        CASE WHEN is_primary_key THEN 0.05 ELSE 0.0 END
                    """)

                if config.boost_foreign_keys:
                    boost_conditions.append("""
                        CASE WHEN is_foreign_key THEN 0.05 ELSE 0.0 END
                    """)

                boost_expression = " + ".join(boost_conditions) if boost_conditions else "0.0"

                # Build the SQL query with proper parameter substitution
                sql_query = """
                    SELECT
                        id,
                        schema_name,
                        table_name,
                        column_name,
                        data_type,
                        is_primary_key,
                        is_foreign_key,
                        referenced_table,
                        column_comment,
                        metadata_text,
                        (1 - (embedding <=> %s::vector)) + """ + boost_expression + """ as similarity
                    FROM catalog.column_metadata
                    WHERE embedding IS NOT NULL
                        AND ((1 - (embedding <=> %s::vector)) + """ + boost_expression + """) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """

                logger.info(f"Executing query with embedding length: {len(query_embedding)}")
                logger.info(f"Config threshold: {config.similarity_threshold}, top_k: {config.top_k}")

                # Use JSON format for vector (consistent with other labs)
                embedding_json = json.dumps(query_embedding)

                # Debug parameter values and ensure they're the right types
                threshold = float(config.similarity_threshold)
                top_k = int(config.top_k)

                params = (embedding_json, embedding_json, threshold, top_k)
                logger.info(f"Parameters: embedding_json_len={len(embedding_json)}, threshold={threshold}, top_k={top_k}")

                cursor.execute(sql_query, params)

                results = []
                for row in cursor.fetchall():
                    try:
                        results.append(SearchResult(
                            id=row[0],
                            metadata_type='column',
                            schema_name=row[1],
                            table_name=row[2],
                            column_name=row[3],
                            description=row[8] or f"{row[3]} column in {row[2]}",
                            metadata_text=row[9],
                            similarity_score=row[10],
                            additional_info={
                                'data_type': row[4],
                                'is_primary_key': row[5],
                                'is_foreign_key': row[6],
                                'referenced_table': row[7]
                            }
                        ))
                    except IndexError as e:
                        logger.error(f"Row has {len(row)} columns, expected 11: {row}")
                        raise e

                return results

        finally:
            conn.close()

    def search_relationship_metadata(
        self,
        query_embedding: List[float],
        config: SearchConfig
    ) -> List[SearchResult]:
        """Search relationship metadata."""
        if not config.include_relationships:
            return []

        conn = self.get_db_connection()

        try:
            with conn.cursor() as cursor:
                # Use JSON format for vector (consistent with other labs)
                embedding_json = json.dumps(query_embedding)

                cursor.execute("""
                    SELECT
                        id,
                        source_schema,
                        source_table,
                        source_column,
                        target_table,
                        target_column,
                        relationship_type,
                        metadata_text,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM catalog.relationship_metadata
                    WHERE embedding IS NOT NULL
                        AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (embedding_json, embedding_json, config.similarity_threshold, config.top_k))

                results = []
                for row in cursor.fetchall():
                    results.append(SearchResult(
                        id=row[0],
                        metadata_type='relationship',
                        schema_name=row[1],
                        table_name=row[2],
                        column_name=row[3],
                        description=f"Relationship: {row[2]}.{row[3]} -> {row[4]}.{row[5]}",
                        metadata_text=row[7],
                        similarity_score=row[8],
                        additional_info={
                            'source_table': row[2],
                            'source_column': row[3],
                            'target_table': row[4],
                            'target_column': row[5],
                            'relationship_type': row[6]
                        }
                    ))

                return results

        finally:
            conn.close()

    def search_kpi_metadata(
        self,
        query_embedding: List[float],
        config: SearchConfig
    ) -> List[SearchResult]:
        """Search KPI metadata."""
        conn = self.get_db_connection()

        try:
            with conn.cursor() as cursor:
                # Use JSON format for vector (consistent with other labs)
                embedding_json = json.dumps(query_embedding)

                cursor.execute("""
                    SELECT
                        id,
                        kpi_name,
                        kpi_description,
                        kpi_category,
                        measure_expression,
                        required_tables,
                        metadata_text,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM catalog.suggested_kpis
                    WHERE embedding IS NOT NULL
                        AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (embedding_json, embedding_json, config.similarity_threshold, config.top_k))

                results = []
                for row in cursor.fetchall():
                    results.append(SearchResult(
                        id=row[0],
                        metadata_type='kpi',
                        schema_name='catalog',
                        table_name='suggested_kpis',
                        column_name=None,
                        description=row[2] or row[1],
                        metadata_text=row[6],
                        similarity_score=row[7],
                        additional_info={
                            'kpi_name': row[1],
                            'kpi_category': row[3],
                            'measure_expression': row[4],
                            'required_tables': row[5]
                        }
                    ))

                return results

        finally:
            conn.close()

    def search_metadata(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> Tuple[QueryType, List[SearchResult]]:
        """
        Perform comprehensive metadata search.

        Args:
            query: Natural language search query
            config: Search configuration

        Returns:
            Tuple of (query_type, search_results)
        """
        if not config:
            config = SearchConfig()

        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)

        # Classify query type
        query_type = self.classify_query_type(query)

        logger.info(f"Query type classified as: {query_type.value}")

        # Search all metadata types
        all_results = []

        # Search tables
        table_results = self.search_table_metadata(query_embedding, config)
        all_results.extend(table_results)

        # Search columns with type-specific boosting
        column_results = self.search_column_metadata(query_embedding, config, query_type)
        all_results.extend(column_results)

        # Search relationships
        relationship_results = self.search_relationship_metadata(query_embedding, config)
        all_results.extend(relationship_results)

        # Search KPIs
        kpi_results = self.search_kpi_metadata(query_embedding, config)
        all_results.extend(kpi_results)

        # Sort by similarity score and limit
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        top_results = all_results[:config.top_k]

        logger.info(f"Found {len(top_results)} relevant metadata elements")

        return query_type, top_results

    def explain_results(self, query_type: QueryType, results: List[SearchResult]) -> str:
        """Generate a human-readable explanation of search results."""
        if not results:
            return "No relevant metadata found for your query."

        explanation_parts = [
            f"Based on your query, I identified this as a {query_type.value} question.",
            f"Found {len(results)} relevant metadata elements:\n"
        ]

        # Group results by type
        by_type = {}
        for result in results:
            if result.metadata_type not in by_type:
                by_type[result.metadata_type] = []
            by_type[result.metadata_type].append(result)

        for metadata_type, type_results in by_type.items():
            explanation_parts.append(f"\n{metadata_type.title()} ({len(type_results)}):")
            for i, result in enumerate(type_results[:5], 1):  # Show top 5 per type
                if metadata_type == 'column':
                    explanation_parts.append(
                        f"  {i}. {result.table_name}.{result.column_name} "
                        f"({result.additional_info.get('data_type', 'unknown')}) "
                        f"- {result.description}"
                    )
                elif metadata_type == 'relationship':
                    explanation_parts.append(
                        f"  {i}. {result.additional_info.get('source_table')}.{result.additional_info.get('source_column')} "
                        f"-> {result.additional_info.get('target_table')}.{result.additional_info.get('target_column')}"
                    )
                elif metadata_type == 'kpi':
                    explanation_parts.append(
                        f"  {i}. {result.additional_info.get('kpi_name')} "
                        f"({result.additional_info.get('kpi_category')}) - {result.description}"
                    )
                else:  # table
                    explanation_parts.append(
                        f"  {i}. {result.table_name} - {result.description}"
                    )

        return "\n".join(explanation_parts)