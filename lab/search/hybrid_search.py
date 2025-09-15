#!/usr/bin/env python3
"""
Hybrid search implementation combining dense and sparse vector search.
"""

import argparse
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.core.embeddings import OpenAIEmbedder, SPLADEEmbedder
from lab.core.search import VectorSearch, SparseSearch, HybridSearch, SearchResult
from lab.core.generation import GenerationService
from lab.core.ranking import RankingService


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class HybridSearchEngine:
    """
    Hybrid search engine combining dense and sparse vector search.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        config: Optional[ConfigService] = None,
        source: str = "wikipedia",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            db_service: Database service
            config: Configuration service
            source: Data source (wikipedia or movies)
            dense_weight: Weight for dense search results
            sparse_weight: Weight for sparse search results
        """
        self.db = db_service
        self.config = config or ConfigService()
        self.source = source
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize services
        self._initialize_services()
        
        # Initialize searches
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
        if self.source == "wikipedia":
            # Wikipedia searches
            vector_col = getattr(self.config.embedding, "vector_column", "content_vector")
            self.dense_search = VectorSearch(
                db_service=self.db,
                embedding_service=self.dense_embedder,
                table_name="articles",
                vector_column=vector_col,
                content_columns=["title", "content"],
                id_column="id"
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
            # Netflix shows searches
            self.dense_search = VectorSearch(
                db_service=self.db,
                embedding_service=self.dense_embedder,
                table_name="netflix_shows",
                vector_column="embedding",
                content_columns=["title", "description"],
                id_column="show_id"
            )
            
            self.sparse_search = SparseSearch(
                db_service=self.db,
                embedding_service=self.sparse_embedder,
                table_name="netflix_shows",
                sparse_column="sparse_embedding",
                content_columns=["title", "description"],
                id_column="show_id"
            )
        
        # Create hybrid search
        self.hybrid_search = HybridSearch(
            self.dense_search,
            self.sparse_search,
            self.dense_weight,
            self.sparse_weight
        )
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        show_individual_results: bool = False
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining dense and sparse results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to rerank combined results
            show_individual_results: Whether to include individual search results
            
        Returns:
            Dictionary with hybrid results and optional individual results
        """
        logging.info(f"Performing hybrid search for: {query}")
        
        # Get individual results if requested
        individual_results = {}
        if show_individual_results:
            individual_results['dense'] = self.dense_search.search(query, top_k * 2)
            individual_results['sparse'] = self.sparse_search.search(query, top_k * 2)
        
        # Perform hybrid search
        hybrid_results = self.hybrid_search.search(query, top_k, rerank=rerank)
        
        return {
            'query': query,
            'hybrid_results': hybrid_results,
            'individual_results': individual_results,
            'weights': {
                'dense': self.dense_weight,
                'sparse': self.sparse_weight
            },
            'reranked': rerank,
            'num_results': len(hybrid_results)
        }
    
    def compare_search_methods(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Compare different search methods side by side.
        
        Args:
            query: Search query
            top_k: Number of results for each method
            
        Returns:
            Dictionary with results from each method
        """
        logging.info(f"Comparing search methods for: {query}")
        
        results = {
            'dense': self.dense_search.search(query, top_k),
            'sparse': self.sparse_search.search(query, top_k),
            'hybrid_reranked': self.hybrid_search.search(query, top_k, rerank=True),
            'hybrid_interleaved': self.hybrid_search.search(query, top_k, rerank=False)
        }
        
        return results
    
    def search_with_different_weights(
        self,
        query: str,
        weight_combinations: List[Tuple[float, float]] = None,
        top_k: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Search with different weight combinations.
        
        Args:
            query: Search query
            weight_combinations: List of (dense_weight, sparse_weight) tuples
            top_k: Number of results
            
        Returns:
            Results for each weight combination
        """
        if weight_combinations is None:
            weight_combinations = [
                (1.0, 0.0),  # Dense only
                (0.7, 0.3),  # Dense heavy
                (0.5, 0.5),  # Balanced
                (0.3, 0.7),  # Sparse heavy
                (0.0, 1.0)   # Sparse only
            ]
        
        results = {}
        
        for dense_w, sparse_w in weight_combinations:
            # Create temporary hybrid search with these weights
            temp_hybrid = HybridSearch(
                self.dense_search,
                self.sparse_search,
                dense_w,
                sparse_w
            )
            
            search_results = temp_hybrid.search(query, top_k, rerank=True)
            weight_key = f"dense_{dense_w:.1f}_sparse_{sparse_w:.1f}"
            results[weight_key] = search_results
        
        return results
    
    def generate_answer_from_hybrid(
        self,
        query: str,
        top_k: int = 10,
        max_context_length: int = 8000
    ) -> Dict[str, Any]:
        """
        Generate answer using hybrid search results.
        
        Args:
            query: Search query
            top_k: Number of search results
            max_context_length: Maximum context length
            
        Returns:
            Answer and metadata
        """
        # Perform hybrid search
        search_data = self.search_hybrid(query, top_k, rerank=True)
        results = search_data['hybrid_results']
        
        if not results:
            return {
                'query': query,
                'answer': "I couldn't find relevant information to answer your question.",
                'method': 'hybrid',
                'weights': search_data['weights'],
                'num_results': 0,
                'sources': []
            }
        
        # Generate answer
        response = self.generator.generate_rag_response(
            query=query,
            search_results=results,
            max_context_length=max_context_length
        )
        
        return {
            'query': query,
            'answer': response.content,
            'method': 'hybrid',
            'weights': search_data['weights'],
            'num_results': len(results),
            'cost': response.cost,
            'usage': response.usage,
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
    
    def update_weights(self, dense_weight: float, sparse_weight: float):
        """
        Update search weights.
        
        Args:
            dense_weight: New dense weight
            sparse_weight: New sparse weight
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Update hybrid search
        self.hybrid_search = HybridSearch(
            self.dense_search,
            self.sparse_search,
            dense_weight,
            sparse_weight
        )
        
        logging.info(f"Updated weights: dense={dense_weight}, sparse={sparse_weight}")


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Hybrid search combining dense and sparse methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive hybrid search
  python hybrid_search.py --source wikipedia --interactive

  # Single query with custom weights
  python hybrid_search.py --source wikipedia --query "machine learning" --dense-weight 0.7 --sparse-weight 0.3

  # Compare search methods
  python hybrid_search.py --source movies --query "romantic comedy" --compare-methods

  # Test different weight combinations
  python hybrid_search.py --source wikipedia --query "Python programming" --test-weights

  # Generate answer using hybrid search
  python hybrid_search.py --source wikipedia --query "What is AI?" --generate-answer
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
        help="Search query (if not interactive)"
    )
    
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Weight for dense search (default: 0.5)"
    )
    
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
        help="Weight for sparse search (default: 0.5)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of search results (default: 10)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )
    
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare different search methods"
    )
    
    parser.add_argument(
        "--test-weights",
        action="store_true",
        help="Test different weight combinations"
    )
    
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate answer using hybrid search"
    )
    
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Don't rerank results (use simple interleaving)"
    )
    
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Show individual dense and sparse results"
    )
    
    return parser


def print_search_results(results: List[SearchResult], title: str = "Results", show_source: bool = True):
    """Print search results in a formatted way."""
    print(f"\n{title} ({len(results)} results):")
    print("=" * 60)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] ID: {result.id} | Score: {result.score:.4f}")
        if show_source and result.source:
            print(f"    Source: {result.source}")
        
        # Truncate content for display
        content = result.content
        if len(content) > 250:
            content = content[:250] + "..."
        
        print(f"    Content: {content}")


def print_comparison_results(results: Dict[str, List[SearchResult]]):
    """Print comparison of different search methods."""
    print("\n" + "=" * 80)
    print("SEARCH METHOD COMPARISON")
    print("=" * 80)
    
    for method, search_results in results.items():
        print_search_results(search_results, f"{method.upper()} SEARCH")
        
        # Show overlap analysis
        if method != 'dense':  # Compare with dense as baseline
            dense_ids = {r.id for r in results['dense']}
            current_ids = {r.id for r in search_results}
            overlap = len(dense_ids & current_ids)
            unique_to_current = len(current_ids - dense_ids)
            
            print(f"    Overlap with dense: {overlap}/{len(search_results)} results")
            print(f"    Unique to {method}: {unique_to_current} results")


def interactive_hybrid_search(engine: HybridSearchEngine, top_k: int, rerank: bool, show_individual: bool):
    """Run interactive hybrid search session."""
    print(f"\n{'='*60}")
    print(f"Interactive Hybrid Search - {engine.source.title()}")
    print(f"Dense Weight: {engine.dense_weight} | Sparse Weight: {engine.sparse_weight}")
    print("="*60)
    print("Commands:")
    print("  /weights <dense> <sparse> - Change weights (e.g., /weights 0.7 0.3)")
    print("  /compare                  - Compare all search methods")
    print("  /answer                   - Generate answer for last query")
    print("  /help                     - Show this help")
    print("  /quit                     - Exit")
    print("=" * 60)
    
    last_query = None
    
    while True:
        try:
            query = input(f"\n[hybrid {engine.dense_weight:.1f}/{engine.sparse_weight:.1f}] Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q', '/quit']:
                print("Goodbye!")
                break
            
            # Handle commands
            if query.startswith('/'):
                if query.startswith('/weights'):
                    try:
                        parts = query.split()
                        if len(parts) >= 3:
                            dense_w = float(parts[1])
                            sparse_w = float(parts[2])
                            engine.update_weights(dense_w, sparse_w)
                            print(f"Updated weights: dense={dense_w}, sparse={sparse_w}")
                        else:
                            print("Usage: /weights <dense_weight> <sparse_weight>")
                    except ValueError:
                        print("Invalid weight values. Use numbers like: /weights 0.7 0.3")
                    continue
                
                elif query == '/compare':
                    if last_query:
                        print(f"Comparing search methods for: '{last_query}'")
                        comparison = engine.compare_search_methods(last_query, top_k)
                        print_comparison_results(comparison)
                    else:
                        print("No previous query to compare. Run a search first.")
                    continue
                
                elif query == '/answer':
                    if last_query:
                        print(f"Generating answer for: '{last_query}'")
                        response = engine.generate_answer_from_hybrid(last_query, top_k)
                        print(f"\nAnswer:")
                        print("-" * 40)
                        print(response['answer'])
                        print(f"\nBased on {response['num_results']} results using hybrid search")
                        if 'cost' in response and response['cost'] > 0:
                            print(f"Generation cost: ${response['cost']:.4f}")
                    else:
                        print("No previous query. Run a search first.")
                    continue
                
                elif query == '/help':
                    print("Commands:")
                    print("  /weights <dense> <sparse> - Change weights")
                    print("  /compare                  - Compare search methods")
                    print("  /answer                   - Generate answer")
                    print("  /help                     - Show this help")
                    print("  /quit                     - Exit")
                    continue
                
                else:
                    print("Unknown command. Type /help for available commands.")
                    continue
            
            # Perform hybrid search
            last_query = query
            print(f"\nSearching for: '{query}'")
            
            search_data = engine.search_hybrid(
                query, top_k, rerank=rerank, show_individual_results=show_individual
            )
            
            # Show hybrid results
            print_search_results(search_data['hybrid_results'], "HYBRID SEARCH")
            
            # Show individual results if requested
            if show_individual and search_data['individual_results']:
                print_search_results(
                    search_data['individual_results']['dense'][:top_k],
                    "DENSE SEARCH",
                    show_source=False
                )
                print_search_results(
                    search_data['individual_results']['sparse'][:top_k],
                    "SPARSE SEARCH",
                    show_source=False
                )
        
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
    
    if not args.interactive and not args.query:
        print("Error: Either --interactive or --query must be specified")
        sys.exit(1)
    
    # Validate weights
    if abs(args.dense_weight + args.sparse_weight - 1.0) > 0.01:
        print("Warning: Dense and sparse weights should sum to 1.0")
        # Normalize
        total = args.dense_weight + args.sparse_weight
        args.dense_weight /= total
        args.sparse_weight /= total
        print(f"Normalized to: dense={args.dense_weight:.3f}, sparse={args.sparse_weight:.3f}")
    
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
        
        engine = HybridSearchEngine(
            db_service, config, args.source,
            args.dense_weight, args.sparse_weight
        )
        
        if args.interactive:
            # Interactive mode
            interactive_hybrid_search(
                engine, args.top_k,
                not args.no_rerank, args.show_individual
            )
        else:
            # Single query mode
            print(f"Hybrid searching {args.source} for: '{args.query}'")
            print(f"Weights: dense={args.dense_weight}, sparse={args.sparse_weight}")
            
            if args.compare_methods:
                # Compare all methods
                results = engine.compare_search_methods(args.query, args.top_k)
                print_comparison_results(results)
            
            elif args.test_weights:
                # Test different weight combinations
                results = engine.search_with_different_weights(args.query, top_k=args.top_k)
                print("\nTESTING DIFFERENT WEIGHT COMBINATIONS:")
                print("=" * 60)
                for weight_combo, search_results in results.items():
                    print_search_results(search_results, weight_combo.upper())
            
            elif args.generate_answer:
                # Generate answer
                response = engine.generate_answer_from_hybrid(args.query, args.top_k)
                print(f"\nAnswer:")
                print("=" * 60)
                print(response['answer'])
                
                print(f"\nBased on {response['num_results']} results")
                print(f"Weights: dense={response['weights']['dense']}, sparse={response['weights']['sparse']}")
                if 'cost' in response and response['cost'] > 0:
                    print(f"Generation cost: ${response['cost']:.4f}")
                
                if response['sources']:
                    print(f"\nTop sources:")
                    print("-" * 40)
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"[{i}] {source['content']} (Score: {source['score']:.4f})")
            
            else:
                # Regular hybrid search
                search_data = engine.search_hybrid(
                    args.query, args.top_k,
                    not args.no_rerank, args.show_individual
                )
                
                print_search_results(search_data['hybrid_results'], "HYBRID SEARCH")
                
                if args.show_individual and search_data['individual_results']:
                    print_search_results(
                        search_data['individual_results']['dense'][:args.top_k],
                        "DENSE SEARCH"
                    )
                    print_search_results(
                        search_data['individual_results']['sparse'][:args.top_k],
                        "SPARSE SEARCH"
                    )
    
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()