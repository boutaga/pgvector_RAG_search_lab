#!/usr/bin/env python3
"""
CLI tool for verifying and analyzing embedding generation results.
"""

import argparse
import sys
import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lab.core.database import DatabaseService
from lab.core.config import ConfigService, load_config
from lab.embeddings.embedding_manager import EmbeddingManager


@dataclass
class EmbeddingAnalysis:
    """Container for embedding analysis results."""
    table_name: str
    column_name: str
    total_rows: int
    with_embeddings: int
    missing_embeddings: int
    completion_rate: float
    avg_embedding_size: Optional[float] = None
    embedding_type: str = "unknown"
    sample_embeddings: List[Any] = None


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Verify and analyze embedding generation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all Wikipedia embeddings
  python verify_embeddings.py --source wikipedia

  # Verify specific table and column
  python verify_embeddings.py --table articles --column title_vector

  # Detailed analysis with samples
  python verify_embeddings.py --source wikipedia --detailed --samples 3

  # Export results to JSON
  python verify_embeddings.py --source movies --output results.json

  # Check specific embedding types
  python verify_embeddings.py --source wikipedia --embedding-type dense
        """
    )
    
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
    
    # Source specification
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        choices=["wikipedia", "movies", "all"],
        help="Data source to verify"
    )
    
    source_group.add_argument(
        "--table",
        type=str,
        help="Specific table to verify (use with --column)"
    )
    
    parser.add_argument(
        "--column",
        type=str,
        help="Specific embedding column to verify (requires --table)"
    )
    
    parser.add_argument(
        "--embedding-type",
        choices=["dense", "sparse", "all"],
        default="all",
        help="Type of embeddings to verify (default: all)"
    )
    
    # Analysis options
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Perform detailed analysis including embedding quality checks"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample embeddings to analyze (default: 5)"
    )
    
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Check embedding quality (non-zero values, reasonable magnitudes)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output results to JSON file"
    )
    
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    
    return parser


class EmbeddingVerifier:
    """Verifier for embedding generation results."""
    
    def __init__(self, db_service: DatabaseService, config: Optional[ConfigService] = None):
        """Initialize verifier."""
        self.db = db_service
        self.config = config
        self.manager = EmbeddingManager(db_service, config)
        
        # Define embedding configurations
        self.embedding_configs = {
            "wikipedia": {
                "dense": [
                    ("articles", "title_vector", "title"),
                    ("articles", "content_vector", "content")
                ],
                "sparse": [
                    ("articles", "title_sparse", "title"),
                    ("articles", "content_sparse", "content")
                ]
            },
            "movies": {
                "dense": [
                    ("film", "embedding", "description"),
                    ("netflix_shows", "embedding", "description")
                ],
                "sparse": [
                    ("netflix_shows", "sparse_embedding", "description")
                ]
            }
        }
    
    def verify_source(
        self,
        source: str,
        embedding_type: str = "all",
        detailed: bool = False,
        samples: int = 5,
        check_quality: bool = False
    ) -> List[EmbeddingAnalysis]:
        """
        Verify embeddings for a data source.
        
        Args:
            source: Data source to verify
            embedding_type: Type of embeddings to check
            detailed: Perform detailed analysis
            samples: Number of samples to analyze
            check_quality: Check embedding quality
            
        Returns:
            List of analysis results
        """
        if source not in self.embedding_configs:
            raise ValueError(f"Unknown source: {source}")
        
        analyses = []
        configs = self.embedding_configs[source]
        
        # Determine which embedding types to check
        types_to_check = []
        if embedding_type == "all":
            types_to_check = list(configs.keys())
        else:
            if embedding_type in configs:
                types_to_check = [embedding_type]
        
        # Verify each configuration
        for emb_type in types_to_check:
            if emb_type not in configs:
                continue
                
            for table_name, column_name, content_column in configs[emb_type]:
                analysis = self.verify_table_column(
                    table_name,
                    column_name,
                    content_column,
                    emb_type,
                    detailed,
                    samples,
                    check_quality
                )
                analyses.append(analysis)
        
        return analyses
    
    def verify_table_column(
        self,
        table_name: str,
        column_name: str,
        content_column: str = None,
        embedding_type: str = "unknown",
        detailed: bool = False,
        samples: int = 5,
        check_quality: bool = False
    ) -> EmbeddingAnalysis:
        """
        Verify embeddings for a specific table/column.
        
        Args:
            table_name: Table name
            column_name: Embedding column name
            content_column: Content column name
            embedding_type: Type of embedding
            detailed: Perform detailed analysis
            samples: Number of samples
            check_quality: Check quality
            
        Returns:
            Analysis results
        """
        logging.info(f"Verifying {table_name}.{column_name}")
        
        # Check if table and column exist
        if not self.db.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        if not self.db.column_exists(table_name, column_name):
            raise ValueError(f"Column {column_name} does not exist in table {table_name}")
        
        # Get basic statistics
        stats = self.manager.get_embedding_statistics(table_name, column_name)
        
        analysis = EmbeddingAnalysis(
            table_name=table_name,
            column_name=column_name,
            total_rows=stats['total_rows'],
            with_embeddings=stats['rows_with_embeddings'],
            missing_embeddings=stats['rows_missing_embeddings'],
            completion_rate=stats['completion_percentage'],
            embedding_type=embedding_type
        )
        
        # Detailed analysis
        if detailed or check_quality:
            self._perform_detailed_analysis(analysis, content_column, samples, check_quality)
        
        return analysis
    
    def _perform_detailed_analysis(
        self,
        analysis: EmbeddingAnalysis,
        content_column: Optional[str],
        samples: int,
        check_quality: bool
    ):
        """Perform detailed analysis on embeddings."""
        # Get sample embeddings
        sample_query = f"""
            SELECT id, {analysis.column_name}
            {f', {content_column}' if content_column else ''}
            FROM {analysis.table_name}
            WHERE {analysis.column_name} IS NOT NULL
            ORDER BY id
            LIMIT %s
        """
        
        sample_results = self.db.execute_query(
            sample_query,
            (samples,),
            dict_cursor=True
        )
        
        if sample_results:
            analysis.sample_embeddings = []
            embedding_sizes = []
            
            for row in sample_results:
                embedding = row[analysis.column_name]
                
                # Analyze embedding
                embedding_info = {
                    'id': row['id'],
                    'has_embedding': embedding is not None
                }
                
                if content_column:
                    content = row.get(content_column, '')
                    embedding_info['content_preview'] = content[:100] + "..." if len(content) > 100 else content
                    embedding_info['content_length'] = len(content) if content else 0
                
                if embedding is not None:
                    if analysis.embedding_type == "dense":
                        # Dense embedding analysis
                        if isinstance(embedding, (list, tuple)):
                            embedding_info['dimensions'] = len(embedding)
                            embedding_info['non_zero_count'] = sum(1 for x in embedding if x != 0)
                            embedding_info['magnitude'] = sum(x*x for x in embedding) ** 0.5
                            embedding_sizes.append(len(embedding))
                        else:
                            # Stored as vector type
                            embedding_info['type'] = 'vector'
                    
                    elif analysis.embedding_type == "sparse":
                        # Sparse embedding analysis
                        if isinstance(embedding, str):
                            # Parse sparsevec format: {1:0.5,3:0.8}/30522
                            if embedding.startswith('{') and '}/' in embedding:
                                sparse_part = embedding.split('}/')[0] + '}'
                                dimension_part = embedding.split('/')[1]
                                
                                try:
                                    # Count non-zero entries
                                    non_zero_count = len([x for x in sparse_part.split(',') if x.strip()])
                                    embedding_info['non_zero_count'] = non_zero_count
                                    embedding_info['total_dimensions'] = int(dimension_part)
                                    embedding_info['sparsity'] = 1 - (non_zero_count / int(dimension_part))
                                except:
                                    embedding_info['format_error'] = True
                
                if check_quality:
                    embedding_info['quality_check'] = self._check_embedding_quality(
                        embedding, analysis.embedding_type
                    )
                
                analysis.sample_embeddings.append(embedding_info)
            
            # Calculate average size for dense embeddings
            if embedding_sizes:
                analysis.avg_embedding_size = sum(embedding_sizes) / len(embedding_sizes)
    
    def _check_embedding_quality(self, embedding: Any, embedding_type: str) -> Dict[str, Any]:
        """Check the quality of an embedding."""
        quality = {
            'is_valid': True,
            'issues': []
        }
        
        if embedding is None:
            quality['is_valid'] = False
            quality['issues'].append("Embedding is None")
            return quality
        
        if embedding_type == "dense":
            if isinstance(embedding, (list, tuple)):
                # Check for all zeros
                if all(x == 0 for x in embedding):
                    quality['is_valid'] = False
                    quality['issues'].append("All values are zero")
                
                # Check for reasonable magnitude
                magnitude = sum(x*x for x in embedding) ** 0.5
                if magnitude < 0.01:
                    quality['issues'].append("Very low magnitude")
                elif magnitude > 100:
                    quality['issues'].append("Very high magnitude")
                
                # Check for NaN or infinite values
                try:
                    if any(not isinstance(x, (int, float)) or x != x for x in embedding):
                        quality['is_valid'] = False
                        quality['issues'].append("Contains NaN or invalid values")
                except:
                    quality['is_valid'] = False
                    quality['issues'].append("Could not validate values")
        
        elif embedding_type == "sparse":
            if isinstance(embedding, str):
                # Check format
                if not (embedding.startswith('{') and '}/' in embedding):
                    quality['is_valid'] = False
                    quality['issues'].append("Invalid sparsevec format")
                else:
                    # Check if it parses correctly
                    try:
                        sparse_part = embedding.split('}/')[0]
                        if not sparse_part or sparse_part == '{':
                            quality['issues'].append("Empty sparse vector")
                    except:
                        quality['is_valid'] = False
                        quality['issues'].append("Could not parse sparse vector")
        
        return quality
    
    def verify_all_sources(
        self,
        embedding_type: str = "all",
        detailed: bool = False,
        samples: int = 5,
        check_quality: bool = False
    ) -> Dict[str, List[EmbeddingAnalysis]]:
        """Verify all available data sources."""
        results = {}
        
        for source in self.embedding_configs.keys():
            try:
                results[source] = self.verify_source(
                    source, embedding_type, detailed, samples, check_quality
                )
            except Exception as e:
                logging.error(f"Failed to verify {source}: {e}")
                results[source] = []
        
        return results


def format_analysis_table(analyses: List[EmbeddingAnalysis]) -> str:
    """Format analysis results as a table."""
    if not analyses:
        return "No embedding data found."
    
    # Calculate column widths
    widths = {
        'table': max(len(a.table_name) for a in analyses) + 2,
        'column': max(len(a.column_name) for a in analyses) + 2,
        'type': max(len(a.embedding_type) for a in analyses) + 2,
        'total': 8,
        'with': 8,
        'missing': 8,
        'rate': 8
    }
    
    # Header
    header = (
        f"{'Table':<{widths['table']}} "
        f"{'Column':<{widths['column']}} "
        f"{'Type':<{widths['type']}} "
        f"{'Total':<{widths['total']}} "
        f"{'With':<{widths['with']}} "
        f"{'Missing':<{widths['missing']}} "
        f"{'Rate %':<{widths['rate']}}"
    )
    
    separator = "-" * len(header)
    
    # Rows
    rows = []
    for analysis in analyses:
        row = (
            f"{analysis.table_name:<{widths['table']}} "
            f"{analysis.column_name:<{widths['column']}} "
            f"{analysis.embedding_type:<{widths['type']}} "
            f"{analysis.total_rows:<{widths['total']}} "
            f"{analysis.with_embeddings:<{widths['with']}} "
            f"{analysis.missing_embeddings:<{widths['missing']}} "
            f"{analysis.completion_rate:<{widths['rate']}.1f}"
        )
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def format_detailed_analysis(analyses: List[EmbeddingAnalysis]) -> str:
    """Format detailed analysis results."""
    output = []
    
    for analysis in analyses:
        output.append(f"\n{'='*60}")
        output.append(f"DETAILED ANALYSIS: {analysis.table_name}.{analysis.column_name}")
        output.append('='*60)
        output.append(f"Embedding Type: {analysis.embedding_type}")
        output.append(f"Total Rows: {analysis.total_rows}")
        output.append(f"With Embeddings: {analysis.with_embeddings}")
        output.append(f"Missing Embeddings: {analysis.missing_embeddings}")
        output.append(f"Completion Rate: {analysis.completion_rate:.1f}%")
        
        if analysis.avg_embedding_size:
            output.append(f"Average Embedding Size: {analysis.avg_embedding_size:.0f}")
        
        if analysis.sample_embeddings:
            output.append("\nSAMPLE ANALYSIS:")
            for i, sample in enumerate(analysis.sample_embeddings, 1):
                output.append(f"\n  Sample {i} (ID: {sample['id']}):")
                
                if 'content_preview' in sample:
                    output.append(f"    Content: {sample['content_preview']}")
                    output.append(f"    Content Length: {sample['content_length']}")
                
                if 'dimensions' in sample:
                    output.append(f"    Dimensions: {sample['dimensions']}")
                    output.append(f"    Non-zero Values: {sample['non_zero_count']}")
                    output.append(f"    Magnitude: {sample['magnitude']:.3f}")
                
                if 'non_zero_count' in sample and 'total_dimensions' in sample:
                    output.append(f"    Non-zero Count: {sample['non_zero_count']}")
                    output.append(f"    Total Dimensions: {sample['total_dimensions']}")
                    if 'sparsity' in sample:
                        output.append(f"    Sparsity: {sample['sparsity']:.3f}")
                
                if 'quality_check' in sample:
                    quality = sample['quality_check']
                    status = "✓ Valid" if quality['is_valid'] else "✗ Invalid"
                    output.append(f"    Quality: {status}")
                    if quality['issues']:
                        for issue in quality['issues']:
                            output.append(f"      - {issue}")
    
    return "\n".join(output)


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.table and not args.column:
        print("Error: --column is required when using --table")
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
        
        verifier = EmbeddingVerifier(db_service, config)
        
        # Perform verification
        analyses = []
        
        if args.table and args.column:
            # Single table/column verification
            analysis = verifier.verify_table_column(
                args.table,
                args.column,
                detailed=args.detailed,
                samples=args.samples,
                check_quality=args.check_quality
            )
            analyses = [analysis]
        
        elif args.source == "all":
            # All sources
            results = verifier.verify_all_sources(
                args.embedding_type,
                args.detailed,
                args.samples,
                args.check_quality
            )
            for source_analyses in results.values():
                analyses.extend(source_analyses)
        
        else:
            # Specific source
            analyses = verifier.verify_source(
                args.source,
                args.embedding_type,
                args.detailed,
                args.samples,
                args.check_quality
            )
        
        # Output results
        if args.format == "json":
            # Convert to dict for JSON serialization
            results_dict = []
            for analysis in analyses:
                result = {
                    'table_name': analysis.table_name,
                    'column_name': analysis.column_name,
                    'embedding_type': analysis.embedding_type,
                    'total_rows': analysis.total_rows,
                    'with_embeddings': analysis.with_embeddings,
                    'missing_embeddings': analysis.missing_embeddings,
                    'completion_rate': analysis.completion_rate
                }
                if analysis.avg_embedding_size:
                    result['avg_embedding_size'] = analysis.avg_embedding_size
                if analysis.sample_embeddings:
                    result['sample_embeddings'] = analysis.sample_embeddings
                results_dict.append(result)
            
            output_text = json.dumps(results_dict, indent=2)
        
        else:
            # Table format
            output_text = format_analysis_table(analyses)
            
            if args.detailed and any(a.sample_embeddings for a in analyses):
                output_text += format_detailed_analysis(analyses)
        
        # Output to file or stdout
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"Results saved to {args.output}")
        else:
            print(output_text)
        
        # Summary
        total_embeddings = sum(a.with_embeddings for a in analyses)
        total_rows = sum(a.total_rows for a in analyses)
        
        print(f"\nSUMMARY:")
        print(f"Tables analyzed: {len(analyses)}")
        print(f"Total rows: {total_rows}")
        print(f"Total embeddings: {total_embeddings}")
        if total_rows > 0:
            print(f"Overall completion rate: {(total_embeddings / total_rows) * 100:.1f}%")
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_service' in locals():
            db_service.close()


if __name__ == "__main__":
    main()