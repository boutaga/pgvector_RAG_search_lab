#!/usr/bin/env python3
"""
Agentic RAG search implementation using LLM function calling.

This module implements an agentic approach to RAG where the LLM acts as an autonomous
agent that decides when and how to retrieve information. Unlike traditional RAG patterns
that always perform retrieval, the agent can:
- Skip retrieval for simple questions it can answer directly
- Invoke a search tool when additional information is needed
- Potentially refine queries and search multiple times (future enhancement)

The implementation uses OpenAI's function calling capability to expose the vector search
as a tool the agent can invoke during its reasoning process.
"""

import argparse
import sys
import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple

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


# Agentic system prompt - instructs the LLM on how to use the search tool
AGENTIC_SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a Wikipedia database.
Your goal is to answer the user's question with factual information from the database.

You can use the function `search_wikipedia` to retrieve relevant snippets **if needed**.

Guidelines:
1. ANALYZE the question first:
   - If it's a simple factual question you know with certainty (like basic math, common knowledge), you MAY answer directly
   - If it's specific to Wikipedia content, complex, or you're uncertain, USE the search_wikipedia function
   - When in doubt, prefer using the search tool to ensure accuracy

2. USING THE SEARCH TOOL:
   - Call search_wikipedia(query, top_k) with a relevant query
   - You can reformulate the user's question into a better search query
   - Review the returned snippets carefully

3. GENERATING THE ANSWER:
   - Base your answer ONLY on retrieved snippets (if you searched)
   - Cite sources using snippet indices, e.g., "According to [1], ..."
   - If snippets don't contain sufficient information, say: "I don't have enough information to answer this completely based on the database."
   - Do NOT hallucinate or add information not present in the snippets

4. ANSWER FORMAT:
   - Provide a clear, concise answer
   - Include source citations when using search results
   - If you used the search tool, acknowledge it: "Based on the database search..."

Remember: Accuracy and groundedness are paramount. Use the search tool when in doubt."""


class AgenticSearchEngine:
    """
    Agentic search engine where LLM decides when to retrieve information.

    This implements an autonomous agent pattern where the LLM orchestrates the
    retrieval process rather than following a fixed pipeline.
    """

    def __init__(
        self,
        db_service: DatabaseService,
        config: Optional[ConfigService] = None,
        source: str = "wikipedia"
    ):
        """
        Initialize agentic search engine.

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

    def _create_search_tool(self) -> Tuple[callable, Dict[str, Any]]:
        """
        Create search tool function and its OpenAI specification.

        The tool function executes the actual vector search and formats results
        for LLM consumption. The specification tells the LLM how to call it.

        Returns:
            Tuple of (search_function, function_spec)
        """
        def search_tool(query: str, top_k: int = 5) -> str:
            """
            Execute search and format results for LLM consumption.

            Args:
                query: Search query
                top_k: Number of results to retrieve

            Returns:
                Formatted string with numbered snippets
            """
            # Perform vector search
            results = self.dense_search.search(query, top_k)

            # Format as numbered snippets
            snippets = []
            for i, result in enumerate(results, 1):
                # Extract title from metadata
                title = result.metadata.get('title', 'Unknown') if result.metadata else 'Unknown'

                # Truncate content to reasonable length
                content = result.content[:300].replace('\n', ' ').strip()
                if len(result.content) > 300:
                    content += "..."

                snippet = f"[{i}] {title}: {content}"
                snippets.append(snippet)

            if not snippets:
                return "No relevant results found in the database."

            return "\n\n".join(snippets)

        # OpenAI function specification
        function_name = "search_wikipedia" if self.source == "wikipedia" else "search_database"

        spec = {
            "name": function_name,
            "description": f"Search the {self.source} database for relevant information and return text snippets",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (can be rephrased from the original user question for better results)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }

        return search_tool, spec

    def search_and_answer_agentic(
        self,
        query: str,
        top_k: int = 5,
        max_iterations: int = 1,
        include_sources: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute agentic search with LLM decision-making.

        The LLM acts as an agent that decides whether to invoke the search tool
        or answer directly. This enables more flexible and efficient retrieval.

        Args:
            query: User query
            top_k: Number of results per search
            max_iterations: Maximum search iterations allowed (default 1)
            include_sources: Whether to include source information
            system_prompt: Optional override for system prompt

        Returns:
            Response dict with answer, decision metadata, and optional sources
        """
        logging.info(f"Starting agentic search for: {query}")

        # Create search tool
        search_tool, tool_spec = self._create_search_tool()

        # Build initial messages
        messages = [
            {"role": "system", "content": system_prompt or AGENTIC_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        # Tracking variables
        tool_used = False
        search_count = 0
        decision = "unknown"
        sources = []
        total_cost = 0.0

        # Initial LLM call with function definition
        logging.info("Sending initial request to LLM agent...")
        response = self.generator.generate_with_functions(
            messages=messages,
            functions=[tool_spec],
            function_call="auto"  # Let model decide
        )

        total_cost += response.cost

        # Check if function was called
        if response.function_call:
            # Agent decided to search
            tool_used = True
            decision = "search"
            search_count = 1

            # Parse function call
            fn_name = response.function_call['name']
            fn_args = json.loads(response.function_call['arguments'])

            search_query = fn_args.get('query', query)
            search_k = fn_args.get('top_k', top_k)

            logging.info(f"Agent calling {fn_name}(query='{search_query}', top_k={search_k})")

            # Execute search tool
            search_results_str = search_tool(search_query, search_k)

            # Get actual SearchResult objects for sources
            if include_sources:
                actual_results = self.dense_search.search(search_query, search_k)
                sources = [
                    {
                        'id': r.id,
                        'content': r.content[:200] + "..." if len(r.content) > 200 else r.content,
                        'score': r.score,
                        'metadata': r.metadata
                    }
                    for r in actual_results
                ]

            # Add function call and result to conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": response.function_call
            })
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": search_results_str
            })

            # Get final answer from LLM (no more function calls)
            logging.info("Getting final answer from LLM with search results...")
            final_response = self.generator.generate_from_messages(
                messages=messages,
                function_call="none"  # Force answer, no more tool calls
            )

            answer = final_response.content
            total_cost += final_response.cost

        else:
            # Agent answered directly without search
            decision = "direct"
            answer = response.content
            logging.info("Agent answered directly without using search tool")

        result = {
            'query': query,
            'answer': answer,
            'decision': decision,
            'tool_used': tool_used,
            'search_count': search_count,
            'sources': sources,
            'cost': total_cost,
            'method': 'agentic',
            'num_results': len(sources)
        }

        logging.info(f"Agentic search complete: decision={decision}, cost=${total_cost:.4f}")

        return result

    def batch_agentic_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform agentic search on multiple queries.

        Args:
            queries: List of queries to process
            top_k: Number of results per search

        Returns:
            List of response dictionaries
        """
        results = []
        for query in queries:
            try:
                result = self.search_and_answer_agentic(query, top_k)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'answer': f"Error: {str(e)}",
                    'decision': 'error',
                    'tool_used': False,
                    'search_count': 0,
                    'sources': [],
                    'cost': 0.0,
                    'method': 'agentic',
                    'num_results': 0
                })
        return results


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG search with LLM-driven retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive agentic search
  python agentic_search.py --source wikipedia --interactive

  # Single query
  python agentic_search.py --source wikipedia --query "What is PostgreSQL?"

  # Batch queries from file
  python agentic_search.py --source wikipedia --batch-file queries.txt

  # Compare agentic decisions
  python agentic_search.py --source wikipedia --query "What is 2+2?" --show-decision
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
        default=5,
        help="Number of results per search (default: 5)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )

    parser.add_argument(
        "--batch-file",
        type=str,
        help="File with queries to process (one per line)"
    )

    parser.add_argument(
        "--show-decision",
        action="store_true",
        help="Show agent's decision reasoning"
    )

    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source snippets"
    )

    return parser


def print_agentic_result(result: Dict[str, Any], show_decision: bool = False, show_sources: bool = False):
    """Print agentic search result in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Query: {result['query']}")
    print('='*70)

    # Decision indicator
    if result['decision'] == 'search':
        print("🔄 Agent Decision: USE SEARCH TOOL")
    elif result['decision'] == 'direct':
        print("⚡ Agent Decision: ANSWER DIRECTLY")
    else:
        print(f"❓ Agent Decision: {result['decision'].upper()}")

    if show_decision:
        print(f"   Tool Used: {result['tool_used']}")
        print(f"   Search Count: {result['search_count']}")
        print(f"   Cost: ${result['cost']:.4f}")

    print(f"\nAnswer:")
    print("-"*70)
    print(result['answer'])

    # Sources
    if show_sources and result['sources']:
        print(f"\nSources ({len(result['sources'])} retrieved):")
        print("-"*70)
        for i, source in enumerate(result['sources'], 1):
            title = source.get('metadata', {}).get('title', 'Unknown')
            print(f"[{i}] {title}")
            print(f"    {source['content']}")
            print(f"    Score: {source['score']:.4f}")


def interactive_agentic_search(engine: AgenticSearchEngine, top_k: int, show_decision: bool, show_sources: bool):
    """Run interactive agentic search session."""
    print(f"\n{'='*70}")
    print(f"Interactive Agentic RAG Search - {engine.source.title()}")
    print("="*70)
    print("The LLM agent will decide whether to search or answer directly.")
    print("\nCommands:")
    print("  /decision    - Toggle decision info display")
    print("  /sources     - Toggle source display")
    print("  /help        - Show this help")
    print("  /quit        - Exit")
    print("="*70)

    while True:
        try:
            query = input(f"\n[agentic] Query: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q', '/quit']:
                print("Goodbye!")
                break

            # Handle commands
            if query.startswith('/'):
                if query == '/decision':
                    show_decision = not show_decision
                    print(f"Decision info: {'ON' if show_decision else 'OFF'}")
                    continue
                elif query == '/sources':
                    show_sources = not show_sources
                    print(f"Source display: {'ON' if show_sources else 'OFF'}")
                    continue
                elif query == '/help':
                    print("Commands:")
                    print("  /decision - Toggle decision info")
                    print("  /sources  - Toggle sources")
                    print("  /help     - Show this help")
                    print("  /quit     - Exit")
                    continue
                else:
                    print("Unknown command. Type /help for available commands.")
                    continue

            # Perform agentic search
            result = engine.search_and_answer_agentic(query, top_k)
            print_agentic_result(result, show_decision, show_sources)

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

    if not args.interactive and not args.query and not args.batch_file:
        print("Error: Must specify --interactive, --query, or --batch-file")
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

        engine = AgenticSearchEngine(db_service, config, args.source)

        if args.batch_file:
            # Batch mode
            with open(args.batch_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]

            print(f"Processing {len(queries)} queries from {args.batch_file}...")
            results = engine.batch_agentic_search(queries, args.top_k)

            # Summary statistics
            direct_count = sum(1 for r in results if r['decision'] == 'direct')
            search_count = sum(1 for r in results if r['decision'] == 'search')
            total_cost = sum(r['cost'] for r in results)

            for result in results:
                print_agentic_result(result, args.show_decision, args.show_sources)

            print(f"\n{'='*70}")
            print("BATCH SUMMARY")
            print('='*70)
            print(f"Total Queries: {len(queries)}")
            print(f"Direct Answers: {direct_count} ({direct_count/len(queries)*100:.1f}%)")
            print(f"Search Used: {search_count} ({search_count/len(queries)*100:.1f}%)")
            print(f"Total Cost: ${total_cost:.4f}")
            print(f"Avg Cost per Query: ${total_cost/len(queries):.4f}")

        elif args.interactive:
            # Interactive mode
            interactive_agentic_search(
                engine, args.top_k,
                args.show_decision, args.show_sources
            )

        else:
            # Single query mode
            result = engine.search_and_answer_agentic(args.query, args.top_k)
            print_agentic_result(result, args.show_decision, args.show_sources)

    except Exception as e:
        logger.error(f"Agentic search failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()
