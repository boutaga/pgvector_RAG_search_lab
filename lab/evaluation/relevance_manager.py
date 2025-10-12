#!/usr/bin/env python3
"""
Relevance Manager Service

Handles database operations for relevance grades and evaluation tracking.
Provides CRUD operations for test queries, relevance grades, retrieval logs,
and evaluation results.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """Represents a test query"""
    query_id: Optional[int]
    query_text: str
    query_type: Optional[str] = None
    category: Optional[str] = None
    created_by: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class RelevanceGrade:
    """Represents a relevance grade for a query-doc pair"""
    query_id: int
    doc_id: int
    rel_grade: int  # 0, 1, or 2
    labeler: str
    label_method: str
    notes: Optional[str] = None
    label_date: Optional[datetime] = None


class RelevanceManager:
    """
    Manages relevance grades and evaluation data in PostgreSQL.

    This class provides a clean API for all evaluation-related database operations.
    """

    def __init__(self, connection_string: str):
        """
        Initialize with database connection.

        Args:
            connection_string: PostgreSQL connection string
                Example: "postgresql://user:pass@localhost/dbname"
        """
        self.connection_string = connection_string
        self._connection = None

    def _get_connection(self):
        """Get or create database connection"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(self.connection_string)
        return self._connection

    def close(self):
        """Close database connection"""
        if self._connection and not self._connection.closed:
            self._connection.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    # ========================================================================
    # Test Query Management
    # ========================================================================

    def create_test_query(self, query: TestQuery) -> int:
        """
        Create a new test query and return its ID.

        Args:
            query: TestQuery object (query_id will be ignored)

        Returns:
            Generated query_id

        Raises:
            psycopg2.Error: On database errors
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO test_queries
                        (query_text, query_type, category, created_by, notes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING query_id
                """, (
                    query.query_text,
                    query.query_type,
                    query.category,
                    query.created_by,
                    query.notes
                ))

                query_id = cur.fetchone()[0]
                conn.commit()

                logger.info(f"Created test query {query_id}: {query.query_text[:50]}...")
                return query_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create test query: {e}")
            raise

    def get_test_query(self, query_id: int) -> Optional[TestQuery]:
        """
        Retrieve a test query by ID.

        Args:
            query_id: ID of the query to retrieve

        Returns:
            TestQuery object or None if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM test_queries
                    WHERE query_id = %s
                """, (query_id,))

                row = cur.fetchone()
                if row:
                    return TestQuery(**dict(row))
                return None

        except Exception as e:
            logger.error(f"Failed to get test query {query_id}: {e}")
            raise

    def list_test_queries(
        self,
        query_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TestQuery]:
        """
        List test queries with optional filtering.

        Args:
            query_type: Filter by query type
            category: Filter by category
            limit: Maximum number of queries to return

        Returns:
            List of TestQuery objects
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = "SELECT * FROM test_queries WHERE 1=1"
                params = []

                if query_type:
                    sql += " AND query_type = %s"
                    params.append(query_type)

                if category:
                    sql += " AND category = %s"
                    params.append(category)

                sql += " ORDER BY created_at DESC"

                if limit:
                    sql += " LIMIT %s"
                    params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()

                return [TestQuery(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list test queries: {e}")
            raise

    def update_test_query(self, query: TestQuery) -> bool:
        """
        Update an existing test query.

        Args:
            query: TestQuery object with query_id set

        Returns:
            True if updated, False if not found
        """
        if query.query_id is None:
            raise ValueError("query_id must be set for update")

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE test_queries
                    SET query_text = %s,
                        query_type = %s,
                        category = %s,
                        notes = %s
                    WHERE query_id = %s
                """, (
                    query.query_text,
                    query.query_type,
                    query.category,
                    query.notes,
                    query.query_id
                ))

                updated = cur.rowcount > 0
                conn.commit()

                if updated:
                    logger.info(f"Updated test query {query.query_id}")
                return updated

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update test query: {e}")
            raise

    def delete_test_query(self, query_id: int) -> bool:
        """
        Delete a test query (cascades to related records).

        Args:
            query_id: ID of query to delete

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM test_queries WHERE query_id = %s", (query_id,))
                deleted = cur.rowcount > 0
                conn.commit()

                if deleted:
                    logger.info(f"Deleted test query {query_id}")
                return deleted

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete test query: {e}")
            raise

    # ========================================================================
    # Relevance Grade Management
    # ========================================================================

    def add_relevance_grade(self, grade: RelevanceGrade) -> bool:
        """
        Add or update a relevance grade.

        Args:
            grade: RelevanceGrade object

        Returns:
            True on success

        Raises:
            psycopg2.Error: On database errors
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO relevance_grades
                        (query_id, doc_id, rel_grade, labeler, label_method, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (query_id, doc_id)
                    DO UPDATE SET
                        rel_grade = EXCLUDED.rel_grade,
                        labeler = EXCLUDED.labeler,
                        label_method = EXCLUDED.label_method,
                        label_date = NOW(),
                        notes = EXCLUDED.notes
                """, (
                    grade.query_id,
                    grade.doc_id,
                    grade.rel_grade,
                    grade.labeler,
                    grade.label_method,
                    grade.notes
                ))

                conn.commit()
                logger.debug(f"Added/updated grade: Q{grade.query_id} D{grade.doc_id} = {grade.rel_grade}")
                return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add relevance grade: {e}")
            raise

    def get_relevance_grades(self, query_id: int) -> Dict[int, int]:
        """
        Get all relevance grades for a query as {doc_id: grade}.

        Args:
            query_id: Query ID

        Returns:
            Dictionary mapping doc_id -> relevance grade
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT doc_id, rel_grade
                    FROM relevance_grades
                    WHERE query_id = %s
                """, (query_id,))

                return {row[0]: row[1] for row in cur.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get relevance grades for query {query_id}: {e}")
            raise

    def bulk_add_grades(self, grades: List[RelevanceGrade]) -> int:
        """
        Add multiple relevance grades efficiently.

        Args:
            grades: List of RelevanceGrade objects

        Returns:
            Number of grades added/updated
        """
        if not grades:
            return 0

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO relevance_grades
                        (query_id, doc_id, rel_grade, labeler, label_method, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (query_id, doc_id)
                    DO UPDATE SET
                        rel_grade = EXCLUDED.rel_grade,
                        label_date = NOW()
                """, [
                    (g.query_id, g.doc_id, g.rel_grade, g.labeler, g.label_method, g.notes)
                    for g in grades
                ])

                count = cur.rowcount
                conn.commit()

                logger.info(f"Bulk added/updated {count} relevance grades")
                return count

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to bulk add grades: {e}")
            raise

    def get_labeled_queries(self) -> List[int]:
        """
        Get list of query IDs that have relevance grades.

        Returns:
            List of query IDs with at least one relevance grade
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT query_id
                    FROM relevance_grades
                    ORDER BY query_id
                """)

                return [row[0] for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get labeled queries: {e}")
            raise

    # ========================================================================
    # Retrieval Logging
    # ========================================================================

    def log_retrieval(
        self,
        query_id: int,
        retrieved_docs: List[Tuple[int, float]],  # (doc_id, score)
        retrieval_method: str,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Log a retrieval attempt.

        Args:
            query_id: Query ID
            retrieved_docs: List of (doc_id, score) tuples in rank order
            retrieval_method: Search method used
            session_id: Optional session identifier

        Returns:
            True on success
        """
        if not retrieved_docs:
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO retrieval_log
                        (query_id, doc_id, rank, score, retrieval_method, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, [
                    (query_id, doc_id, rank + 1, score, retrieval_method, session_id)
                    for rank, (doc_id, score) in enumerate(retrieved_docs)
                ])

                conn.commit()
                logger.debug(f"Logged retrieval: Q{query_id}, {len(retrieved_docs)} docs, method={retrieval_method}")
                return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log retrieval: {e}")
            raise

    # ========================================================================
    # Evaluation Results
    # ========================================================================

    def save_evaluation_result(
        self,
        query_id: int,
        retrieval_method: str,
        k_value: int,
        ndcg_score: float,
        dcg_score: float,
        idcg_score: float,
        precision_at_k: Optional[float] = None,
        recall_at_k: Optional[float] = None,
        f1_at_k: Optional[float] = None,
        mrr: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Save evaluation results. Returns eval_id.

        Args:
            query_id: Query ID
            retrieval_method: Method used
            k_value: Top-k value
            ndcg_score: nDCG score
            dcg_score: DCG score
            idcg_score: IDCG score
            precision_at_k: Optional precision score
            recall_at_k: Optional recall score
            f1_at_k: Optional F1 score
            mrr: Optional MRR score
            session_id: Optional session identifier

        Returns:
            Generated eval_id
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO evaluation_results
                        (query_id, retrieval_method, k_value, ndcg_score, dcg_score, idcg_score,
                         precision_at_k, recall_at_k, f1_at_k, mrr, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING eval_id
                """, (
                    query_id, retrieval_method, k_value, ndcg_score, dcg_score, idcg_score,
                    precision_at_k, recall_at_k, f1_at_k, mrr, session_id
                ))

                eval_id = cur.fetchone()[0]
                conn.commit()

                logger.debug(f"Saved evaluation result {eval_id}: Q{query_id}, nDCG={ndcg_score:.3f}")
                return eval_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save evaluation result: {e}")
            raise

    # ========================================================================
    # Analysis Functions
    # ========================================================================

    def get_evaluation_summary(
        self,
        method: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Get evaluation summary statistics.

        Args:
            method: Optional method filter
            days_back: Days to look back

        Returns:
            Dictionary with summary statistics
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT
                        COUNT(*) as num_evaluations,
                        COUNT(DISTINCT query_id) as num_queries,
                        AVG(ndcg_score) as avg_ndcg,
                        STDDEV(ndcg_score) as stddev_ndcg,
                        MIN(ndcg_score) as min_ndcg,
                        MAX(ndcg_score) as max_ndcg,
                        AVG(precision_at_k) as avg_precision,
                        AVG(recall_at_k) as avg_recall
                    FROM evaluation_results
                    WHERE timestamp > NOW() - INTERVAL '%s days'
                """
                params = [days_back]

                if method:
                    sql += " AND retrieval_method = %s"
                    params.append(method)

                cur.execute(sql, params)
                result = cur.fetchone()

                return dict(result) if result else {}

        except Exception as e:
            logger.error(f"Failed to get evaluation summary: {e}")
            raise

    def get_ndcg_trend(self, method: str, days_back: int = 30) -> List[Dict]:
        """
        Get nDCG trend over time.

        Args:
            method: Retrieval method
            days_back: Days to look back

        Returns:
            List of {date, avg_ndcg, num_queries} dicts
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM get_ndcg_trend(%s, %s)", (method, days_back))
                return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get nDCG trend: {e}")
            raise

    def compare_methods(
        self,
        methods: List[str],
        k_value: int = 10
    ) -> Dict[str, Dict]:
        """
        Compare different retrieval methods.

        Args:
            methods: List of method names to compare
            k_value: Top-k value to compare

        Returns:
            Dictionary mapping method -> metrics dict
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        retrieval_method,
                        COUNT(*) as num_evaluations,
                        AVG(ndcg_score) as mean_ndcg,
                        AVG(precision_at_k) as mean_precision,
                        AVG(recall_at_k) as mean_recall,
                        AVG(f1_at_k) as mean_f1,
                        AVG(mrr) as mean_mrr,
                        STDDEV(ndcg_score) as stddev_ndcg
                    FROM evaluation_results
                    WHERE retrieval_method = ANY(%s)
                      AND k_value = %s
                      AND timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY retrieval_method
                """, (methods, k_value))

                result = {}
                for row in cur.fetchall():
                    method = row['retrieval_method']
                    result[method] = dict(row)

                return result

        except Exception as e:
            logger.error(f"Failed to compare methods: {e}")
            raise


# ============================================================================
# Convenience Functions
# ============================================================================

def create_sample_queries(manager: RelevanceManager) -> List[int]:
    """
    Create sample test queries for demonstration.

    Args:
        manager: RelevanceManager instance

    Returns:
        List of created query IDs
    """
    sample_queries = [
        TestQuery(
            query_id=None,
            query_text="What is machine learning?",
            query_type="conceptual",
            category="AI",
            created_by="system",
            notes="Basic ML concept question"
        ),
        TestQuery(
            query_id=None,
            query_text="Who invented Python?",
            query_type="factual",
            category="programming",
            created_by="system",
            notes="Creator of Python programming language"
        ),
        TestQuery(
            query_id=None,
            query_text="Explain quantum computing",
            query_type="exploratory",
            category="physics",
            created_by="system",
            notes="Advanced quantum computing topic"
        ),
    ]

    query_ids = []
    for query in sample_queries:
        query_id = manager.create_test_query(query)
        query_ids.append(query_id)

    logger.info(f"Created {len(query_ids)} sample queries")
    return query_ids


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    'RelevanceManager',
    'TestQuery',
    'RelevanceGrade',
    'create_sample_queries',
]
