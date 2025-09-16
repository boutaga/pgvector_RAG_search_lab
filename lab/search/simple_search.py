#!/usr/bin/env python3
"""
Simple search implementations using the modular service architecture.
"""

import argparse
import sys
import os
import logging
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.core.embeddings import OpenAIEmbedder, SPLADEEmbedder
from lab.core.search import VectorSearch, SparseSearch, SearchResult
from lab.core.generation import GenerationService
from lab.core.ranking import RankingService


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class SimpleSearchEngine:
    """
    Simple search engine implementing basic RAG functionality.
    """
    
    def __init__(
        self,
        db_service: DatabaseService,
        config: Optional[ConfigService] = None,
        source: str = "wikipedia"
    ):
        """
        Initialize search engine.
        
        Args:
            db_service: Database service
            config: Configuration service
            source: Data source (wikipedia or movies)
        """
        self.db = db_service
        self.config = config or ConfigService()
        self.source = source
        
        # Initialize services
        self._initialize_services()
        
        # Initialize searches
        self._initialize_searches()
    
    def _initialize_services(self):
        """Initialize embedding and generation services."""
        self.dense_embedder = OpenAIEmbedder(
            model=self.config.embedding.openai_model,
            dimensions=self.config.embedding.openai_dimensions,
            batch_size=1  # Single query processing
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

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        search_mode: str = "content",
        title_weight: Optional[float] = None,
        combined_fetch_multiplier: Optional[int] = None,
        max_combined_fetch: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results
            search_mode: Which vector columns to consult ('content', 'title', 'combined')
            title_weight: Optional override weight when combining title and content vectors
            combined_fetch_multiplier: Optional override for fetching extra rows in combined mode
            max_combined_fetch: Optional override for maximum combined fetch size
        
        Returns:
            Search results
        """
        logging.info(f"Performing dense search for: {query}")
        options = {"search_mode": search_mode}
        if title_weight is not None:
            options["title_weight"] = title_weight
        if combined_fetch_multiplier is not None:
            options["combined_fetch_multiplier"] = combined_fetch_multiplier
        if max_combined_fetch is not None:
            options["max_combined_fetch"] = max_combined_fetch
        return self.dense_search.search(query, top_k, **options)

    def search_sparse(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform sparse vector search.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results
        """
        logging.info(f"Performing sparse search for: {query}")
        return self.sparse_search.search(query, top_k)
    
    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_length: int = 8000
    ) -> str:
        """
        Generate answer from search results.
        
        Args:
            query: Original query
            search_results: Search results to use as context
            max_context_length: Maximum context length
            
        Returns:
            Generated answer
        """
        if not search_results:
            return "I couldn't find relevant information to answer your question."
        
        # Format search results as context
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            part = f"[{i}] {result.content}\n"
            part_length = len(part)
            
            if current_length + part_length > max_context_length:
                if current_length > 0:  # We have some context
                    break
                # Truncate if first result is too long
                part = part[:max_context_length] + "..."
                context_parts.append(part)
                break
            
            context_parts.append(part)
            current_length += part_length
        
        context = "\n".join(context_parts)
        
        # Generate answer
        response = self.generator.generate_rag_response(
            query=query,
            search_results=search_results,
            result_formatter=lambda _: context
        )
        
        return response.content
    
    def search_and_answer(
        self,
        query: str,
        search_type: str = "dense",
        top_k: int = 10,
        include_sources: bool = True,
        search_mode: str = "content",
        title_weight: Optional[float] = None,
        combined_fetch_multiplier: Optional[int] = None,
        max_combined_fetch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform search and generate answer.
        
        Args:
            query: Search query
            search_type: Type of search ('dense' or 'sparse')
            top_k: Number of search results
            include_sources: Whether to include source information
            search_mode: Mode for dense search ('content', 'title', 'combined')
            title_weight: Optional override for combined title/content weighting
            combined_fetch_multiplier: Optional override for combined fetch multiplier
            max_combined_fetch: Optional override for maximum combined fetch size
        
        Returns:
            Dictionary with answer and metadata
        """
        # Perform search
        if search_type == "dense":
            results = self.search_dense(
                query,
                top_k,
                search_mode=search_mode,
                title_weight=title_weight,
                combined_fetch_multiplier=combined_fetch_multiplier,
                max_combined_fetch=max_combined_fetch
            )
        elif search_type == "sparse":
            results = self.search_sparse(query, top_k)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        if not results:
            return {
                'query': query,
                'answer': "I couldn't find relevant information to answer your question.",
                'search_type': search_type,
                'num_results': 0,
                'sources': [] if include_sources else None
            }

        answer = self.generate_answer(query, results)

        response = {
            'query': query,
            'answer': answer,
            'search_type': search_type,
            'num_results': len(results)
        }

        if include_sources:
            response['sources'] = [
                {
                    'id': result.id,
                    'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'score': result.score,
                    'metadata': result.metadata
                }
                for result in results
            ]

        return response

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Simple search using modular architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive Wikipedia search
  python simple_search.py --source wikipedia --interactive

  # Single dense search query
  python simple_search.py --source wikipedia --query "What is machine learning?" --type dense

  # Sparse search with more results
  python simple_search.py --source movies --query "romantic comedy" --type sparse --top-k 15

  # Search without answer generation
  python simple_search.py --source wikipedia --query "Python programming" --no-answer
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
        "--type",
        choices=["dense", "sparse"],
        default="dense",
        help="Search type (default: dense)"
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
        "--no-answer",
        action="store_true",
        help="Don't generate answers, just show search results"
    )
    
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include similarity scores in output"
    )
    
    return parser


def print_search_results(results: List[SearchResult], include_scores: bool = False):
    """Print search results in a formatted way."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] ID: {result.id}")
        if include_scores:
            print(f"    Score: {result.score:.4f}")
        
        # Truncate long content for display
        content = result.content
        if len(content) > 300:
            content = content[:300] + "..."
        
        print(f"    Content: {content}")
        
        if result.metadata:
            metadata_str = ", ".join(f"{k}: {v}" for k, v in result.metadata.items() if k not in ['title', 'content'])
            if metadata_str:
                print(f"    Metadata: {metadata_str}")


def interactive_search(engine: SimpleSearchEngine, search_type: str, top_k: int, no_answer: bool, include_scores: bool):
    """Run interactive search session."""
    print(f"\n{'='*60}")
    print(f"Interactive Search Mode - {engine.source.title()} ({search_type})")
    print("='*60")
    print("Enter your queries (type 'quit', 'exit', or 'q' to stop)")
    print("Commands:")
    print("  /dense    - Switch to dense search")
    print("  /sparse   - Switch to sparse search")
    print("  /help     - Show this help")
    print("=" * 60)
    
    current_type = search_type
    
    while True:
        try:
            query = input(f"\n[{current_type}] Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Handle commands
            if query.startswith('/'):
                if query == '/dense':
                    current_type = 'dense'
                    print("Switched to dense search")
                    continue
                elif query == '/sparse':
                    current_type = 'sparse'
                    print("Switched to sparse search")
                    continue
                elif query == '/help':
                    print("Commands:")
                    print("  /dense    - Switch to dense search")
                    print("  /sparse   - Switch to sparse search")
                    print("  /help     - Show this help")
                    continue
                else:
                    print("Unknown command. Type /help for available commands.")
                    continue
            
            # Perform search
            print(f"\nSearching for: '{query}'")
            
            if no_answer:
                # Just show search results
                if current_type == "dense":
                    results = engine.search_dense(query, top_k)
                else:
                    results = engine.search_sparse(query, top_k)
                
                print_search_results(results, include_scores)
            else:
                # Full search and answer
                response = engine.search_and_answer(
                    query, current_type, top_k, include_sources=True
                )
                
                print(f"\nAnswer:")
                print("-" * 40)
                print(response['answer'])
                
                print(f"\nBased on {response['num_results']} results:")
                print("-" * 40)
                for i, source in enumerate(response['sources'][:5], 1):  # Show top 5
                    print(f"[{i}] {source['content']}")
                    if include_scores:
                        print(f"    Score: {source['score']:.4f}")
                
                if len(response['sources']) > 5:
                    print(f"... and {len(response['sources']) - 5} more results")
        
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
        
        engine = SimpleSearchEngine(db_service, config, args.source)
        
        if args.interactive:
            # Interactive mode
            interactive_search(engine, args.type, args.top_k, args.no_answer, args.include_scores)
        else:
            # Single query mode
            print(f"Searching {args.source} for: '{args.query}'")
            
            if args.no_answer:
                # Just search results
                if args.type == "dense":
                    results = engine.search_dense(args.query, args.top_k)
                else:
                    results = engine.search_sparse(args.query, args.top_k)
                
                print_search_results(results, args.include_scores)
            else:
                # Full search and answer
                response = engine.search_and_answer(
                    args.query, args.type, args.top_k, include_sources=True
                )
                
                print(f"\nAnswer:")
                print("=" * 60)
                print(response['answer'])
                
                if response['sources']:
                    print(f"\nSources ({response['num_results']} results):")
                    print("=" * 60)
                    for i, source in enumerate(response['sources'], 1):
                        print(f"\n[{i}] {source['content']}")
                        if args.include_scores:
                            print(f"    Score: {source['score']:.4f}")
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()