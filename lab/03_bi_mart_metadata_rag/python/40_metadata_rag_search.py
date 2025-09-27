#!/usr/bin/env python3
"""
Interactive RAG search interface for database metadata.
Allows users to ask natural language questions about the database schema.
"""

import os
import sys
from pathlib import Path

# Add the services directory to the path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
services_dir = lab_dir / 'services'
sys.path.insert(0, str(services_dir))

from metadata_search_service import MetadataSearchService, SearchConfig, QueryType

def print_search_results(query_type, results):
    """Print search results in a formatted way."""
    print(f"\nQuery Type: {query_type.value.title()}")
    print(f"Found {len(results)} relevant metadata elements\n")

    if not results:
        print("No relevant metadata found for your query.")
        return

    # Group results by type for better presentation
    by_type = {}
    for result in results:
        if result.metadata_type not in by_type:
            by_type[result.metadata_type] = []
        by_type[result.metadata_type].append(result)

    for metadata_type, type_results in by_type.items():
        print(f"📊 {metadata_type.upper()} RESULTS ({len(type_results)}):")
        print("-" * 50)

        for i, result in enumerate(type_results, 1):
            score_bar = "█" * int(result.similarity_score * 20)
            print(f"{i:2d}. [{score_bar:<20}] {result.similarity_score:.3f}")

            if metadata_type == 'table':
                print(f"    Table: {result.schema_name}.{result.table_name}")
                if result.additional_info.get('row_count'):
                    print(f"    Rows: {result.additional_info['row_count']:,}")
                print(f"    Description: {result.description}")

            elif metadata_type == 'column':
                fk_info = ""
                if result.additional_info.get('is_foreign_key'):
                    fk_info = f" -> {result.additional_info.get('referenced_table')}"
                pk_info = " [PK]" if result.additional_info.get('is_primary_key') else ""

                print(f"    Column: {result.table_name}.{result.column_name}")
                print(f"    Type: {result.additional_info.get('data_type', 'unknown')}{pk_info}{fk_info}")
                print(f"    Description: {result.description}")

            elif metadata_type == 'relationship':
                print(f"    Relationship: {result.additional_info.get('source_table')}.{result.additional_info.get('source_column')}")
                print(f"    -> {result.additional_info.get('target_table')}.{result.additional_info.get('target_column')}")
                print(f"    Type: {result.additional_info.get('relationship_type', 'unknown')}")

            elif metadata_type == 'kpi':
                print(f"    KPI: {result.additional_info.get('kpi_name')}")
                print(f"    Category: {result.additional_info.get('kpi_category')}")
                print(f"    Description: {result.description}")
                if result.additional_info.get('required_tables'):
                    tables = result.additional_info['required_tables']
                    if isinstance(tables, list):
                        print(f"    Tables: {', '.join(tables)}")

            print()

def get_sample_queries():
    """Return a list of sample queries to help users get started."""
    return [
        "What are the sales-related tables and columns?",
        "Show me columns that can be used to calculate revenue",
        "What are the foreign key relationships in the database?",
        "Find columns related to customer information",
        "What metrics can I create with order data?",
        "Show me date/time columns for trend analysis",
        "What are the primary keys in each table?",
        "Find columns that contain price or cost information",
        "What KPIs are available for sales analysis?",
        "Show me product-related tables and relationships"
    ]

def interactive_search_loop():
    """Main interactive search loop."""
    print("=" * 70)
    print("🔍 Interactive Metadata RAG Search")
    print("=" * 70)

    # Initialize search service
    try:
        service = MetadataSearchService()
        print("✓ Connected to metadata search service")
    except Exception as e:
        print(f"✗ Error initializing search service: {e}")
        return

    # Show sample queries
    print("\n💡 Sample queries to get you started:")
    for i, query in enumerate(get_sample_queries()[:5], 1):
        print(f"  {i}. {query}")

    print("\n" + "=" * 70)
    print("Enter your questions about the database schema.")
    print("Type 'samples' to see more sample queries.")
    print("Type 'config' to adjust search settings.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    # Initialize search configuration
    config = SearchConfig(
        top_k=10,
        similarity_threshold=0.7,
        include_relationships=True,
        boost_primary_keys=True,
        boost_foreign_keys=True,
        boost_measures=True
    )

    while True:
        try:
            # Get user input
            query = input("\n🤔 Your question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            if query.lower() == 'samples':
                print("\n💡 Sample queries:")
                for i, sample in enumerate(get_sample_queries(), 1):
                    print(f"  {i:2d}. {sample}")
                continue

            if query.lower() == 'config':
                print(f"\n⚙️  Current Configuration:")
                print(f"    Top-K results: {config.top_k}")
                print(f"    Similarity threshold: {config.similarity_threshold}")
                print(f"    Include relationships: {config.include_relationships}")
                print(f"    Boost primary keys: {config.boost_primary_keys}")
                print(f"    Boost foreign keys: {config.boost_foreign_keys}")
                print(f"    Boost measures: {config.boost_measures}")

                change = input("\nChange settings? (y/n): ").lower()
                if change in ['y', 'yes']:
                    try:
                        new_k = input(f"Top-K results ({config.top_k}): ").strip()
                        if new_k:
                            config.top_k = int(new_k)

                        new_threshold = input(f"Similarity threshold ({config.similarity_threshold}): ").strip()
                        if new_threshold:
                            config.similarity_threshold = float(new_threshold)

                        print("✓ Configuration updated")
                    except ValueError:
                        print("✗ Invalid input, keeping current settings")
                continue

            # Perform search
            print(f"\n🔍 Searching for: '{query}'")
            print("..." * 20)

            query_type, results = service.search_metadata(query, config)

            # Display results
            print_search_results(query_type, results)

            # Offer to show explanation
            if results:
                show_explanation = input("\nShow detailed explanation? (y/n): ").lower()
                if show_explanation in ['y', 'yes']:
                    explanation = service.explain_results(query_type, results)
                    print(f"\n📝 Explanation:\n{explanation}")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n✗ Error during search: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    # Check for required environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("✗ Error: OPENAI_API_KEY environment variable is required")
        print("\nPlease set your OpenAI API key:")
        print('  export OPENAI_API_KEY="your_api_key_here"')
        sys.exit(1)

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("✗ Error: DATABASE_URL environment variable is required")
        print("\nPlease set your database connection:")
        print('  export DATABASE_URL="postgresql://user:password@localhost/dbname"')
        sys.exit(1)

    # Check if we're running in interactive mode
    if len(sys.argv) > 1:
        # Non-interactive mode - search for provided query
        query = ' '.join(sys.argv[1:])
        print(f"🔍 Searching for: '{query}'")

        try:
            service = MetadataSearchService()
            config = SearchConfig()
            query_type, results = service.search_metadata(query, config)
            print_search_results(query_type, results)

            if results:
                explanation = service.explain_results(query_type, results)
                print(f"\n📝 Explanation:\n{explanation}")

        except Exception as e:
            print(f"✗ Error during search: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_search_loop()

if __name__ == "__main__":
    main()