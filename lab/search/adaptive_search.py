#!/usr/bin/env python3
"""
Adaptive search implementation with query classification and dynamic weight adjustment.
"""

import argparse
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.core.embeddings import OpenAIEmbedder, SPLADEEmbedder
from lab.core.search import VectorSearch, SparseSearch, AdaptiveSearch, QueryType, SearchResult
from lab.core.generation import GenerationService
from lab.core.ranking import RankingService


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    query: str
    query_type: QueryType
    confidence: float
    features: Dict[str, Any]
    recommended_weights: Tuple[float, float]


class EnhancedQueryClassifier:
    """
    Enhanced query classifier with more sophisticated analysis.
    """
    
    def __init__(self):
        """Initialize enhanced classifier."""
        # Expanded keyword sets
        self.factual_keywords = {
            'when', 'where', 'who', 'how many', 'how much', 'what year', 'what date',
            'name', 'list', 'number', 'count', 'specific', 'exact', 'which',
            'born', 'died', 'founded', 'established', 'located', 'capital',
            'population', 'area', 'height', 'weight', 'age', 'distance'
        }
        
        self.conceptual_keywords = {
            'why', 'how', 'explain', 'describe', 'understand', 'concept',
            'theory', 'idea', 'meaning', 'significance', 'relationship',
            'compare', 'contrast', 'difference', 'similarity', 'impact',
            'influence', 'effect', 'cause', 'reason', 'purpose', 'philosophy',
            'principle', 'mechanism', 'process', 'approach', 'method'
        }
        
        self.exploratory_keywords = {
            'overview', 'summary', 'introduction', 'about', 'general',
            'broad', 'explore', 'discover', 'find out', 'learn about',
            'tell me about', 'what is', 'what are', 'kinds of', 'types of',
            'examples of', 'related to', 'associated with', 'connected to'
        }
        
        self.structured_keywords = {
            'table', 'column', 'row', 'database', 'sql', 'select',
            'filter', 'sort', 'group', 'aggregate', 'join', 'query',
            'search for', 'find all', 'show me', 'get', 'retrieve'
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: Query to analyze
            
        Returns:
            Query analysis results
        """
        query_lower = query.lower()
        
        # Feature extraction
        features = self._extract_features(query, query_lower)
        
        # Classification with confidence
        query_type, confidence = self._classify_with_confidence(features)
        
        # Determine recommended weights based on classification
        recommended_weights = self._get_recommended_weights(query_type, features)
        
        return QueryAnalysis(
            query=query,
            query_type=query_type,
            confidence=confidence,
            features=features,
            recommended_weights=recommended_weights
        )
    
    def _extract_features(self, query: str, query_lower: str) -> Dict[str, Any]:
        """Extract features from the query."""
        features = {
            'length': len(query.split()),
            'has_question_mark': '?' in query,
            'has_numbers': any(char.isdigit() for char in query),
            'has_quotes': '"' in query or "'" in query,
            'capitalized_words': len([w for w in query.split() if w[0].isupper()]),
            'factual_score': 0,
            'conceptual_score': 0,
            'exploratory_score': 0,
            'structured_score': 0,
            'starts_with_wh': False,
            'comparative': False,
            'specific_entities': False
        }
        
        # Count keyword matches
        features['factual_score'] = sum(1 for kw in self.factual_keywords if kw in query_lower)
        features['conceptual_score'] = sum(1 for kw in self.conceptual_keywords if kw in query_lower)
        features['exploratory_score'] = sum(1 for kw in self.exploratory_keywords if kw in query_lower)
        features['structured_score'] = sum(1 for kw in self.structured_keywords if kw in query_lower)
        
        # WH-questions typically factual
        features['starts_with_wh'] = any(query_lower.startswith(wh) for wh in ['what', 'when', 'where', 'who', 'which'])
        
        # Comparative language suggests conceptual
        features['comparative'] = any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better', 'worse'])
        
        # Proper nouns suggest specific factual queries
        features['specific_entities'] = features['capitalized_words'] > 1
        
        return features
    
    def _classify_with_confidence(self, features: Dict[str, Any]) -> Tuple[QueryType, float]:
        """
        Classify query with confidence score.
        
        Args:
            features: Extracted features
            
        Returns:
            Tuple of (query_type, confidence)
        """
        scores = {
            QueryType.FACTUAL: 0,
            QueryType.CONCEPTUAL: 0,
            QueryType.EXPLORATORY: 0,
            QueryType.STRUCTURED: 0
        }
        
        # Base scores from keyword matches
        scores[QueryType.FACTUAL] += features['factual_score'] * 2
        scores[QueryType.CONCEPTUAL] += features['conceptual_score'] * 2
        scores[QueryType.EXPLORATORY] += features['exploratory_score'] * 2
        scores[QueryType.STRUCTURED] += features['structured_score'] * 3  # Higher weight
        
        # Bonus points based on other features
        if features['starts_with_wh']:
            scores[QueryType.FACTUAL] += 1
        
        if features['comparative']:
            scores[QueryType.CONCEPTUAL] += 2
        
        if features['specific_entities']:
            scores[QueryType.FACTUAL] += 1
        
        if features['has_numbers']:
            scores[QueryType.FACTUAL] += 1
        
        if features['length'] > 10:  # Long queries tend to be exploratory
            scores[QueryType.EXPLORATORY] += 1
        
        if features['length'] < 4:  # Short queries often factual
            scores[QueryType.FACTUAL] += 1
        
        # Find best classification
        max_score = max(scores.values())
        
        if max_score == 0:
            # No clear indicators, default to exploratory
            return QueryType.EXPLORATORY, 0.3
        
        best_type = max(scores, key=scores.get)
        
        # Calculate confidence based on score gap
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Minimum confidence threshold
        confidence = max(confidence, 0.3)
        
        return best_type, confidence
    
    def _get_recommended_weights(self, query_type: QueryType, features: Dict[str, Any]) -> Tuple[float, float]:
        """
        Get recommended weights based on classification.
        
        Args:
            query_type: Classified query type
            features: Query features
            
        Returns:
            Tuple of (dense_weight, sparse_weight)
        """
        # Base weight profiles
        base_weights = {
            QueryType.FACTUAL: (0.3, 0.7),
            QueryType.CONCEPTUAL: (0.7, 0.3),
            QueryType.EXPLORATORY: (0.5, 0.5),
            QueryType.STRUCTURED: (0.2, 0.8)
        }
        
        dense_w, sparse_w = base_weights[query_type]
        
        # Fine-tune based on features
        if features['specific_entities']:
            # More factual, favor sparse
            sparse_w += 0.1
            dense_w -= 0.1
        
        if features['comparative']:
            # More conceptual, favor dense
            dense_w += 0.1
            sparse_w -= 0.1
        
        if features['length'] > 8:
            # Longer queries benefit from semantic understanding
            dense_w += 0.05
            sparse_w -= 0.05
        
        # Ensure weights sum to 1 and are within bounds
        total = dense_w + sparse_w
        dense_w = max(0.1, min(0.9, dense_w / total))
        sparse_w = 1.0 - dense_w
        
        return dense_w, sparse_w


class AdaptiveSearchEngine:
    """
    Adaptive search engine with query classification and dynamic weights.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        config: Optional[ConfigService] = None,
        source: str = "wikipedia"
    ):
        """
        Initialize adaptive search engine.
        
        Args:
            db_service: Database service
            config: Configuration service
            source: Data source
        """
        self.db = db_service
        self.config = config or ConfigService()
        self.source = source
        
        # Initialize services
        self._initialize_services()

        # Initialize classifier first (needed by searches)
        self.classifier = EnhancedQueryClassifier()

        # Initialize searches (uses classifier)
        self._initialize_searches()
    
    def _initialize_services(self):
        """Initialize embedding and generation services."""
        self.dense_embedder = OpenAIEmbedder(
            model=self.config.embedding.openai_model,
            dimensions=self.config.embedding.openai_dimensions,
            batch_size=1
        )
        
        self.sparse_embedder = SPLADEEmbedder(
            model_name=self.config.embedding.splade_model,
            dimensions=self.config.embedding.splade_dimensions,
            batch_size=1
        )
        
        self.generator = GenerationService(
            model=self.config.generation.model,
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_tokens
        )
        
        self.ranker = RankingService()
    
    def _initialize_searches(self):
        """Initialize search services based on data source."""
        title_vector_column = getattr(self.config.embedding, "title_vector_column", None)
        default_title_weight = getattr(self.config.embedding, "title_weight", 0.4)
        combined_fetch_multiplier = getattr(self.config.embedding, "combined_fetch_multiplier", 2)
        max_combined_fetch = getattr(self.config.embedding, "max_combined_fetch", 50)

        if self.source == "wikipedia":
            vector_col = getattr(self.config.embedding, "vector_column", "content_vector")
            self.dense_search = VectorSearch(
                db_service=self.db,
                embedding_service=self.dense_embedder,
                table_name="articles",
                vector_column=vector_col,
                content_columns=["title", "content"],
                id_column="id",
                title_vector_column=title_vector_column,
                default_title_weight=default_title_weight,
                combined_fetch_multiplier=combined_fetch_multiplier,
                max_combined_fetch=max_combined_fetch
            )

            self.sparse_search = SparseSearch(
                db_service=self.db,
                embedding_service=self.sparse_embedder,
                table_name="articles",
                sparse_column="content_sparse",
                content_columns=["title", "content"],
                id_column="id"
            )

        elif self.source == "movies":
            self.dense_search = VectorSearch(
                db_service=self.db,
                embedding_service=self.dense_embedder,
                table_name="netflix_shows",
                vector_column="embedding",
                content_columns=["title", "description"],
                id_column="show_id",
                title_vector_column=None,
                default_title_weight=default_title_weight,
                combined_fetch_multiplier=combined_fetch_multiplier,
                max_combined_fetch=max_combined_fetch
            )

            self.sparse_search = SparseSearch(
                db_service=self.db,
                embedding_service=self.sparse_embedder,
                table_name="netflix_shows",
                sparse_column="sparse_embedding",
                content_columns=["title", "description"],
                id_column="show_id"
            )

        else:
            raise ValueError(f"Unknown source: {self.source}")

        self.adaptive_search = AdaptiveSearch(
            self.dense_search,
            self.sparse_search,
            self.classifier
        )
    def search_adaptive(
        self,
        query: str,
        top_k: int = 10,
        show_analysis: bool = True,
        search_mode: str = "content",
        title_weight: Optional[float] = None,
        title_fts_rerank: bool = False,
        title_fts_weight: float = 0.2,
        title_fts_max_candidates: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform adaptive search with query analysis.
        
        Args:
            query: Search query
            top_k: Number of results
            show_analysis: Whether to include analysis details
            search_mode: Mode for dense search ('content', 'title', 'combined')
            title_weight: Optional override for combined dense/title weighting
            title_fts_rerank: Apply PostgreSQL title FTS reranking to hybrid results
            title_fts_weight: Weight for FTS when blending scores
            title_fts_max_candidates: Optional cap on candidates considered for FTS rerank
        
        Returns:
            Search results with analysis
        """
        logging.info(f"Performing adaptive search for: {query}")
        
        analysis = self.classifier.analyze_query(query)

        effective_title_weight = (
            title_weight if title_weight is not None else getattr(self.config.embedding, "title_weight", 0.4)
        )

        results = self.adaptive_search.search(
            query,
            top_k,
            search_mode=search_mode,
            title_weight=effective_title_weight,
            title_fts_rerank=title_fts_rerank,
            title_fts_weight=title_fts_weight,
            title_fts_max_candidates=title_fts_max_candidates
        )

        response = {
            'query': query,
            'results': results,
            'num_results': len(results)
        }
        
        if show_analysis:
            response.update({
                'query_type': analysis.query_type.value,
                'classification_confidence': analysis.confidence,
                'recommended_weights': {
                    'dense': analysis.recommended_weights[0],
                    'sparse': analysis.recommended_weights[1]
                },
                'query_features': analysis.features
            })
        
        return response
    def compare_adaptive_vs_fixed(
        self,
        query: str,
        fixed_weights: List[Tuple[float, float]] = None,
        top_k: int = 10,
        search_mode: str = "content",
        title_weight: Optional[float] = None,
        title_fts_rerank: bool = False,
        title_fts_weight: float = 0.2,
        title_fts_max_candidates: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare adaptive search against fixed weight combinations.
        
        Args:
            query: Search query
            fixed_weights: List of fixed weight combinations
            top_k: Number of results
            search_mode: Mode for dense search ('content', 'title', 'combined')
            title_weight: Optional override for combined dense/title weighting
            title_fts_rerank: Apply PostgreSQL title FTS reranking to hybrid results
            title_fts_weight: Weight for FTS when blending scores
            title_fts_max_candidates: Optional cap on candidates considered for FTS rerank
        
        Returns:
            Comparison results
        """
        if fixed_weights is None:
            fixed_weights = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
        
        analysis = self.classifier.analyze_query(query)

        adaptive_results = self.search_adaptive(
            query,
            top_k,
            show_analysis=True,
            search_mode=search_mode,
            title_weight=title_weight,
            title_fts_rerank=title_fts_rerank,
            title_fts_weight=title_fts_weight,
            title_fts_max_candidates=title_fts_max_candidates
        )
        
        from lab.core.search import HybridSearch
        
        fixed_results = {}
        effective_title_weight = (
            title_weight if title_weight is not None else getattr(self.config.embedding, "title_weight", 0.4)
        )
        hybrid_kwargs: Dict[str, Any] = {
            "search_mode": search_mode,
            "title_weight": effective_title_weight,
            "title_fts_rerank": title_fts_rerank,
            "title_fts_weight": title_fts_weight,
        }
        if title_fts_max_candidates is not None:
            hybrid_kwargs["title_fts_max_candidates"] = title_fts_max_candidates
        
        for dense_w, sparse_w in fixed_weights:
            hybrid_search = HybridSearch(
                self.dense_search,
                self.sparse_search,
                dense_w,
                sparse_w
            )
            results = hybrid_search.search(query, top_k, **hybrid_kwargs)
            fixed_results[f"fixed_{dense_w:.1f}_{sparse_w:.1f}"] = results
        
        return {
            'query': query,
            'analysis': {
                'query_type': analysis.query_type.value,
                'confidence': analysis.confidence,
                'adaptive_weights': analysis.recommended_weights
            },
            'adaptive_results': adaptive_results['results'],
            'fixed_results': fixed_results,
            'result_counts': {
                'adaptive': len(adaptive_results['results']),
                **{k: len(v) for k, v in fixed_results.items()}
            }
        }
    def batch_analyze_queries(
        self,
        queries: List[str]
    ) -> List[QueryAnalysis]:
        """
        Analyze multiple queries in batch.
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            List of analysis results
        """
        return [self.classifier.analyze_query(q) for q in queries]
    
    def generate_adaptive_answer(
        self,
        query: str,
        top_k: int = 10,
        max_context_length: int = 8000,
        search_mode: str = "content",
        title_weight: Optional[float] = None,
        title_fts_rerank: bool = False,
        title_fts_weight: float = 0.2,
        title_fts_max_candidates: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using adaptive search.
        
        Args:
            query: Search query
            top_k: Number of search results
            max_context_length: Maximum context length
            search_mode: Mode for dense search ('content', 'title', 'combined')
            title_weight: Optional override for combined dense/title weighting
            title_fts_rerank: Apply PostgreSQL title FTS reranking to hybrid results
            title_fts_weight: Weight for FTS when blending scores
            title_fts_max_candidates: Optional cap on candidates considered for FTS rerank
        
        Returns:
            Answer with adaptive search metadata
        """
        search_data = self.search_adaptive(
            query,
            top_k,
            show_analysis=True,
            search_mode=search_mode,
            title_weight=title_weight,
            title_fts_rerank=title_fts_rerank,
            title_fts_weight=title_fts_weight,
            title_fts_max_candidates=title_fts_max_candidates
        )
        results = search_data['results']
        
        if not results:
            return {
                'query': query,
                'answer': "I couldn't find relevant information to answer your question.",
                'search_method': 'adaptive',
                'query_analysis': {
                    'type': search_data['query_type'],
                    'confidence': search_data['classification_confidence'],
                    'weights': search_data['recommended_weights']
                },
                'num_results': 0
            }
        
        response = self.generator.generate_rag_response(
            query=query,
            search_results=results,
            max_context_length=max_context_length
        )
        
        return {
            'query': query,
            'answer': response.content,
            'search_method': 'adaptive',
            'query_analysis': {
                'type': search_data['query_type'],
                'confidence': search_data['classification_confidence'],
                'weights': search_data['recommended_weights'],
                'features': search_data['query_features']
            },
            'num_results': len(results),
            'generation_cost': response.cost,
            'generation_usage': response.usage,
            'sources': [
                {
                    'id': result.id,
                    'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'score': result.score,
                    'source': result.source
                }
                for result in results
            ]
        }
def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Adaptive search with query classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive adaptive search
  python adaptive_search.py --source wikipedia --interactive

  # Single adaptive query with analysis
  python adaptive_search.py --source wikipedia --query "When was Python created?" --show-analysis

  # Compare adaptive vs fixed weights
  python adaptive_search.py --source movies --query "romantic comedy" --compare-fixed

  # Batch analyze query types
  python adaptive_search.py --source wikipedia --analyze-queries "queries.txt"

  # Generate answer with adaptive search
  python adaptive_search.py --source wikipedia --query "What is machine learning?" --generate-answer
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--source",
        choices=["wikipedia", "movies"],
        required=True,
        help="Data source to search"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Search query"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )
    
    parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Show query analysis details"
    )
    
    parser.add_argument(
        "--compare-fixed",
        action="store_true",
        help="Compare against fixed weight combinations"
    )
    
    parser.add_argument(
        "--analyze-queries",
        type=str,
        help="File with queries to analyze (one per line)"
    )
    
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate answer using adaptive search"
    )
    
    return parser


def print_query_analysis(analysis: QueryAnalysis):
    """Print query analysis results."""
    print(f"\nQUERY ANALYSIS:")
    print("=" * 40)
    print(f"Query: {analysis.query}")
    print(f"Type: {analysis.query_type.value.upper()}")
    print(f"Confidence: {analysis.confidence:.3f}")
    print(f"Recommended Weights: Dense {analysis.recommended_weights[0]:.2f}, Sparse {analysis.recommended_weights[1]:.2f}")
    print(f"Features:")
    for key, value in analysis.features.items():
        if isinstance(value, bool) and value:
            print(f"  - {key}: {value}")
        elif isinstance(value, (int, float)) and value > 0:
            print(f"  - {key}: {value}")


def print_search_results(results: List[SearchResult], title: str = "Results"):
    """Print search results."""
    print(f"\n{title} ({len(results)} results):")
    print("=" * 50)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f} | Source: {result.source}")
        content = result.content[:200] + "..." if len(result.content) > 200 else result.content
        print(f"    {content}")


def interactive_adaptive_search(engine: AdaptiveSearchEngine, top_k: int):
    """Run interactive adaptive search session."""
    print(f"\n{'='*60}")
    print(f"Interactive Adaptive Search - {engine.source.title()}")
    print("="*60)
    print("Commands:")
    print("  /analyze <query>  - Analyze query without searching")
    print("  /compare          - Compare adaptive vs fixed for last query")
    print("  /answer           - Generate answer for last query")
    print("  /help             - Show this help")
    print("  /quit             - Exit")
    print("=" * 60)
    
    last_query = None
    
    while True:
        try:
            query = input(f"\n[adaptive] Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q', '/quit']:
                print("Goodbye!")
                break
            
            # Handle commands
            if query.startswith('/'):
                if query.startswith('/analyze'):
                    analyze_query = query[8:].strip()
                    if analyze_query:
                        analysis = engine.classifier.analyze_query(analyze_query)
                        print_query_analysis(analysis)
                    else:
                        print("Usage: /analyze <query>")
                    continue
                
                elif query == '/compare':
                    if last_query:
                        print(f"Comparing adaptive vs fixed weights for: '{last_query}'")
                        comparison = engine.compare_adaptive_vs_fixed(last_query, top_k=top_k)
                        
                        print(f"\nQuery Type: {comparison['analysis']['query_type'].upper()}")
                        print(f"Adaptive Weights: {comparison['analysis']['adaptive_weights']}")
                        
                        print_search_results(comparison['adaptive_results'], "ADAPTIVE SEARCH")
                        
                        for method, results in comparison['fixed_results'].items():
                            weights = method.split('_')[1:]  # Extract weights from method name
                            print_search_results(results, f"FIXED WEIGHTS ({weights[0]}/{weights[1]})")
                    else:
                        print("No previous query to compare. Run a search first.")
                    continue
                
                elif query == '/answer':
                    if last_query:
                        print(f"Generating adaptive answer for: '{last_query}'")
                        response = engine.generate_adaptive_answer(last_query, top_k)
                        print(f"\nAnswer:")
                        print("-" * 40)
                        print(response['answer'])
                        print(f"\nQuery Analysis:")
                        print(f"  Type: {response['query_analysis']['type']}")
                        print(f"  Weights: Dense {response['query_analysis']['weights']['dense']:.2f}, Sparse {response['query_analysis']['weights']['sparse']:.2f}")
                        if 'generation_cost' in response and response['generation_cost'] > 0:
                            print(f"  Cost: ${response['generation_cost']:.4f}")
                    else:
                        print("No previous query. Run a search first.")
                    continue
                
                elif query == '/help':
                    print("Commands:")
                    print("  /analyze <query>  - Analyze query without searching")
                    print("  /compare          - Compare adaptive vs fixed")
                    print("  /answer           - Generate answer")
                    print("  /help             - Show this help")
                    print("  /quit             - Exit")
                    continue
                
                else:
                    print("Unknown command. Type /help for available commands.")
                    continue
            
            # Perform adaptive search
            last_query = query
            print(f"\nSearching for: '{query}'")
            
            search_data = engine.search_adaptive(query, top_k, show_analysis=True)
            
            # Show analysis
            print(f"\nQuery Type: {search_data['query_type'].upper()}")
            print(f"Classification Confidence: {search_data['classification_confidence']:.3f}")
            print(f"Adaptive Weights: Dense {search_data['recommended_weights']['dense']:.2f}, Sparse {search_data['recommended_weights']['sparse']:.2f}")
            
            # Show results
            print_search_results(search_data['results'], "ADAPTIVE SEARCH")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            logging.error(f"Search error: {str(e)}")


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.interactive and not args.query and not args.analyze_queries:
        print("Error: Must specify --interactive, --query, or --analyze-queries")
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config) if args.config else ConfigService()
        
        # Initialize services
        db_service = DatabaseService(
            config.database.connection_string,
            config.database.min_connections,
            config.database.max_connections
        )
        
        engine = AdaptiveSearchEngine(db_service, config, args.source)
        
        if args.analyze_queries:
            # Batch analyze queries from file
            with open(args.analyze_queries, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            analyses = engine.batch_analyze_queries(queries)
            
            print("BATCH QUERY ANALYSIS:")
            print("=" * 60)
            for analysis in analyses:
                print_query_analysis(analysis)
            
            # Summary
            type_counts = {}
            for analysis in analyses:
                type_counts[analysis.query_type] = type_counts.get(analysis.query_type, 0) + 1
            
            print("\nSUMMARY:")
            print("=" * 30)
            for query_type, count in type_counts.items():
                print(f"{query_type.value}: {count} queries")
        
        elif args.interactive:
            # Interactive mode
            interactive_adaptive_search(engine, args.top_k)
        
        else:
            # Single query mode
            if args.generate_answer:
                # Generate answer
                response = engine.generate_adaptive_answer(args.query, args.top_k)
                print(f"Answer:")
                print("=" * 50)
                print(response['answer'])
                
                print(f"\nQuery Analysis:")
                analysis = response['query_analysis']
                print(f"  Type: {analysis['type']}")
                print(f"  Confidence: {analysis['confidence']:.3f}")
                print(f"  Weights: Dense {analysis['weights']['dense']:.2f}, Sparse {analysis['weights']['sparse']:.2f}")
                
                if 'generation_cost' in response and response['generation_cost'] > 0:
                    print(f"  Generation Cost: ${response['generation_cost']:.4f}")
            
            elif args.compare_fixed:
                # Compare adaptive vs fixed
                comparison = engine.compare_adaptive_vs_fixed(args.query, top_k=args.top_k)
                
                print(f"ADAPTIVE VS FIXED COMPARISON:")
                print("=" * 60)
                print(f"Query: {args.query}")
                print(f"Classified as: {comparison['analysis']['query_type'].upper()}")
                print(f"Confidence: {comparison['analysis']['confidence']:.3f}")
                print(f"Adaptive weights: {comparison['analysis']['adaptive_weights']}")
                
                print_search_results(comparison['adaptive_results'], "ADAPTIVE SEARCH")
                
                for method, results in comparison['fixed_results'].items():
                    weights = method.replace('fixed_', '').replace('_', '/')
                    print_search_results(results, f"FIXED WEIGHTS ({weights})")
            
            else:
                # Regular adaptive search
                search_data = engine.search_adaptive(args.query, args.top_k, args.show_analysis)
                
                if args.show_analysis:
                    print(f"Query Type: {search_data['query_type'].upper()}")
                    print(f"Confidence: {search_data['classification_confidence']:.3f}")
                    print(f"Adaptive Weights: Dense {search_data['recommended_weights']['dense']:.2f}, Sparse {search_data['recommended_weights']['sparse']:.2f}")
                
                print_search_results(search_data['results'], "ADAPTIVE SEARCH")
    
    except Exception as e:
        logger.error(f"Adaptive search failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()