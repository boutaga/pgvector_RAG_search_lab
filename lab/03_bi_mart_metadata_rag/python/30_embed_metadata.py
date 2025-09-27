#!/usr/bin/env python3
"""
Generate embeddings for database metadata.
This script creates vector embeddings for table, column, and relationship metadata
to enable RAG-based search for mart planning.
"""

import os
import sys
from pathlib import Path

# Add the services directory to the path
script_dir = Path(__file__).parent
lab_dir = script_dir.parent
services_dir = lab_dir / 'services'
sys.path.insert(0, str(services_dir))

from mart_embedding_service import MetadataEmbeddingService, EmbeddingConfig

def main():
    """Main execution function."""
    print("=" * 60)
    print("Generating Metadata Embeddings")
    print("=" * 60)

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

    try:
        # Initialize embedding service
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            dimensions=1536,
            batch_size=50,
            max_retries=3,
            retry_delay=2
        )

        service = MetadataEmbeddingService(config)
        print(f"✓ Initialized embedding service with model: {config.model}")

        # Check current embedding status
        print("\nChecking current embedding status...")
        stats = service.get_embedding_statistics()

        print("\nCurrent Statistics:")
        for metadata_type, stat in stats.items():
            print(f"  {metadata_type.title()}: {stat['with_embeddings']}/{stat['total']} "
                  f"({stat['completion_pct']:.1f}%) complete")

        # Ask user for confirmation
        total_missing = sum(stat['total'] - stat['with_embeddings'] for stat in stats.values())
        if total_missing == 0:
            print("\n✓ All metadata already has embeddings!")
            return

        print(f"\nFound {total_missing} metadata records without embeddings.")
        response = input("Generate embeddings for all missing records? (y/n): ")

        if response.lower() not in ['y', 'yes']:
            print("Embedding generation cancelled.")
            return

        # Generate embeddings
        print("\nGenerating embeddings...")
        results = service.embed_all_metadata()

        print("\n" + "=" * 60)
        print("✓ Embedding generation completed!")
        print("=" * 60)

        print("\nResults:")
        for metadata_type, count in results.items():
            print(f"  {metadata_type.title()}: {count} new embeddings")

        total_generated = sum(results.values())
        print(f"\nTotal new embeddings generated: {total_generated}")

        # Show updated statistics
        print("\nUpdated Statistics:")
        final_stats = service.get_embedding_statistics()
        for metadata_type, stat in final_stats.items():
            print(f"  {metadata_type.title()}: {stat['with_embeddings']}/{stat['total']} "
                  f"({stat['completion_pct']:.1f}%) complete")

        print("\nNext steps:")
        print("1. Run 40_metadata_rag_search.py to test RAG search")
        print("2. Run 50_mart_planning_agent.py to generate mart plans")
        print("3. Run 80_streamlit_demo.py for interactive demo")

    except KeyboardInterrupt:
        print("\n\nEmbedding generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during embedding generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()