#!/usr/bin/env python3
"""
CLI tool for generating embeddings using the modular service architecture.
"""

import argparse
import sys
import os
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.embeddings.embedding_manager import EmbeddingManager, EmbeddingType


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('embedding_generation.log')
        ]
    )


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Wikipedia articles or Movie/Netflix data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dense embeddings for Wikipedia articles
  python generate_embeddings.py --source wikipedia --type dense

  # Generate sparse embeddings for Wikipedia (titles and content)
  python generate_embeddings.py --source wikipedia --type sparse --limit 1000

  # Generate dense embeddings for movies (films and Netflix)
  python generate_embeddings.py --source movies --type dense --include-films --include-netflix

  # Generate sparse embeddings for Netflix shows only
  python generate_embeddings.py --source movies --type sparse --include-netflix --batch-size 5

  # Update existing embeddings
  python generate_embeddings.py --source wikipedia --type dense --update-existing

  # Use custom configuration file
  python generate_embeddings.py --config config.json --source wikipedia --type both
        """
    )
    
    # General arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    # Data source arguments
    parser.add_argument(
        "--source",
        choices=["wikipedia", "movies"],
        required=True,
        help="Data source for embedding generation"
    )
    
    parser.add_argument(
        "--type",
        choices=["dense", "sparse", "both"],
        required=True,
        help="Type of embeddings to generate"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing (overrides config)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of items to process"
    )
    
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting ID for processing (default: 0)"
    )
    
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing embeddings (default: only process missing embeddings)"
    )
    
    # Movie-specific arguments
    parser.add_argument(
        "--include-films",
        action="store_true",
        help="Include DVD rental films (for movies source)"
    )
    
    parser.add_argument(
        "--include-netflix",
        action="store_true",
        help="Include Netflix shows (for movies source)"
    )
    
    return parser


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Validate movie source arguments
    if args.source == "movies":
        if not args.include_films and not args.include_netflix:
            errors.append("For movies source, must specify --include-films and/or --include-netflix")
        
        # Sparse embeddings only available for Netflix in original setup
        if args.type in ["sparse", "both"] and not args.include_netflix:
            errors.append("Sparse embeddings for movies are only available for Netflix shows")
    
    # Validate batch size
    if args.batch_size is not None and args.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if args.limit is not None and args.limit <= 0:
        errors.append("Limit must be positive")
    
    if args.start_id < 0:
        errors.append("Start ID must be non-negative")
    
    return errors


def print_job_summary(args, manager: EmbeddingManager):
    """Print summary of what will be processed."""
    print("\n" + "="*60)
    print("EMBEDDING GENERATION JOB SUMMARY")
    print("="*60)
    print(f"Source: {args.source}")
    print(f"Embedding Type: {args.type}")
    print(f"Update Existing: {args.update_existing}")
    
    if args.limit:
        print(f"Limit: {args.limit}")
    if args.start_id > 0:
        print(f"Start ID: {args.start_id}")
    if args.batch_size:
        print(f"Batch Size: {args.batch_size}")
    
    if args.source == "movies":
        print(f"Include Films: {args.include_films}")
        print(f"Include Netflix: {args.include_netflix}")
    
    print("="*60)
    
    # Show current embedding status
    print("\nCURRENT STATUS:")
    if args.source == "wikipedia":
        # Check new 3072 columns
        stats = manager.get_embedding_statistics("articles", "title_vector_3072")
        print(f"Wikipedia Articles: {stats['total_rows']} total, {stats['rows_with_embeddings']} with title embeddings (3072)")

        content_stats = manager.get_embedding_statistics("articles", "content_vector_3072")
        print(f"                   {content_stats['rows_with_embeddings']} with content embeddings (3072)")
    
    elif args.source == "movies":
        if args.include_films:
            film_stats = manager.get_embedding_statistics("film", "embedding")
            print(f"Films: {film_stats['total_rows']} total, {film_stats['rows_with_embeddings']} with embeddings")
        
        if args.include_netflix:
            netflix_stats = manager.get_embedding_statistics("netflix_shows", "embedding")
            print(f"Netflix Shows: {netflix_stats['total_rows']} total, {netflix_stats['rows_with_embeddings']} with dense embeddings")
            
            if args.type in ["sparse", "both"]:
                sparse_stats = manager.get_embedding_statistics("netflix_shows", "sparse_embedding")
                print(f"              {sparse_stats['rows_with_embeddings']} with sparse embeddings")


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        print("Argument validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config) if args.config else ConfigService()
        
        # Initialize services
        logger.info("Initializing services...")
        db_service = DatabaseService(
            config.database.connection_string,
            config.database.min_connections,
            config.database.max_connections
        )
        
        manager = EmbeddingManager(db_service, config)
        
        # Print job summary
        print_job_summary(args, manager)
        
        if args.dry_run:
            print("\nDRY RUN MODE - No embeddings will be generated")
            return
        
        # Confirm before proceeding
        print("\nProceed with embedding generation? (y/N): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        
        # Create jobs based on arguments
        jobs = []
        embedding_type = EmbeddingType(args.type)
        
        if args.source == "wikipedia":
            if embedding_type in [EmbeddingType.DENSE, EmbeddingType.BOTH]:
                job = manager.create_wikipedia_dense_job(
                    update_existing=args.update_existing,
                    limit=args.limit
                )
                if args.batch_size:
                    job.batch_size = args.batch_size
                job.start_id = args.start_id
                jobs.append(job)
            
            if embedding_type in [EmbeddingType.SPARSE, EmbeddingType.BOTH]:
                job = manager.create_wikipedia_sparse_job(
                    update_existing=args.update_existing,
                    limit=args.limit
                )
                if args.batch_size:
                    job.batch_size = args.batch_size
                job.start_id = args.start_id
                jobs.append(job)
        
        elif args.source == "movies":
            if embedding_type in [EmbeddingType.DENSE, EmbeddingType.BOTH]:
                movie_jobs = manager.create_movie_dense_job(
                    include_films=args.include_films,
                    include_netflix=args.include_netflix,
                    update_existing=args.update_existing,
                    limit=args.limit
                )
                for job in movie_jobs:
                    if args.batch_size:
                        job.batch_size = args.batch_size
                    job.start_id = args.start_id
                jobs.extend(movie_jobs)
            
            if embedding_type in [EmbeddingType.SPARSE, EmbeddingType.BOTH]:
                movie_jobs = manager.create_movie_sparse_job(
                    include_netflix=args.include_netflix,
                    update_existing=args.update_existing,
                    limit=args.limit
                )
                for job in movie_jobs:
                    if args.batch_size:
                        job.batch_size = args.batch_size
                    job.start_id = args.start_id
                jobs.extend(movie_jobs)
        
        # Execute jobs
        total_successful = 0
        total_failed = 0
        
        for i, job in enumerate(jobs, 1):
            print(f"\n{'='*60}")
            print(f"EXECUTING JOB {i}/{len(jobs)}")
            print(f"Table: {job.table_name}")
            print(f"Type: {job.embedding_type.value}")
            print(f"Columns: {job.content_columns} -> {job.embedding_columns}")
            print('='*60)
            
            progress = manager.generate_embeddings(job)
            
            total_successful += progress.successful_items
            total_failed += progress.failed_items
            
            print(f"\nJob {i} completed:")
            print(f"  Successful: {progress.successful_items}")
            print(f"  Failed: {progress.failed_items}")
            if progress.errors:
                print(f"  Errors: {len(progress.errors)}")
                for error in progress.errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(progress.errors) > 3:
                    print(f"    ... and {len(progress.errors) - 3} more")
        
        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total items processed: {total_successful + total_failed}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        
        if total_successful > 0:
            success_rate = (total_successful / (total_successful + total_failed)) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # Show final embedding status
        print("\nFINAL EMBEDDING STATUS:")
        print_job_summary(args, manager)
        
        logger.info("Embedding generation completed successfully")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()