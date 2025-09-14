"""
Configuration service for managing environment variables and application settings.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    openai_model: str = "text-embedding-3-large"
    openai_dimensions: int = 3072
    splade_model: str = "naver/splade-cocondenser-ensembledistil"
    splade_dimensions: int = 30522
    batch_size_dense: int = 50
    batch_size_sparse: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    content_max_length: int = 32000  # Characters, roughly 8000 tokens


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    top_k: int = 10
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rerank: bool = True
    min_score: float = 0.0
    # Adaptive search weights for different query types
    factual_dense_weight: float = 0.3
    factual_sparse_weight: float = 0.7
    conceptual_dense_weight: float = 0.7
    conceptual_sparse_weight: float = 0.3
    exploratory_dense_weight: float = 0.5
    exploratory_sparse_weight: float = 0.5


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = "gpt-5-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = "You are a helpful AI assistant specializing in information retrieval."
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    connection_string: Optional[str] = None
    min_connections: int = 1
    max_connections: int = 20
    enable_pgvector: bool = True
    connection_timeout: int = 30
    command_timeout: int = 60


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Costs tracking
    track_costs: bool = True
    openai_embedding_cost_per_1k: float = 0.00002  # $0.020 per 1M tokens
    openai_generation_cost_per_1k_input: float = 0.0025  # GPT-4o
    openai_generation_cost_per_1k_output: float = 0.01  # GPT-4o


class ConfigService:
    """
    Service for managing application configuration.
    
    Features:
    - Environment variable loading
    - Configuration file support (JSON)
    - Default values with override capability
    - Type validation
    - Configuration export
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration service.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.config = ApplicationConfig()
        self._config_file = config_file
        self._load_environment()
        if config_file:
            self._load_file(config_file)
        self._validate()
        self._setup_logging()
    
    def _load_environment(self):
        """Load configuration from environment variables."""
        # Database configuration
        if db_url := os.getenv('DATABASE_URL'):
            self.config.database.connection_string = db_url
        
        # OpenAI configuration
        if api_key := os.getenv('OPENAI_API_KEY'):
            # Store in environment for OpenAI client
            os.environ['OPENAI_API_KEY'] = api_key
        
        # Embedding models
        if model := os.getenv('OPENAI_EMBEDDING_MODEL'):
            self.config.embedding.openai_model = model
        if model := os.getenv('SPLADE_MODEL'):
            self.config.embedding.splade_model = model
        
        # Batch sizes
        if batch := os.getenv('BATCH_SIZE_DENSE'):
            self.config.embedding.batch_size_dense = int(batch)
        if batch := os.getenv('BATCH_SIZE_SPARSE'):
            self.config.embedding.batch_size_sparse = int(batch)
        
        # Search configuration
        if top_k := os.getenv('SEARCH_TOP_K'):
            self.config.search.top_k = int(top_k)
        if weight := os.getenv('DENSE_WEIGHT'):
            self.config.search.dense_weight = float(weight)
        if weight := os.getenv('SPARSE_WEIGHT'):
            self.config.search.sparse_weight = float(weight)
        
        # Generation configuration
        if model := os.getenv('GENERATION_MODEL'):
            self.config.generation.model = model
        if temp := os.getenv('GENERATION_TEMPERATURE'):
            self.config.generation.temperature = float(temp)
        if tokens := os.getenv('GENERATION_MAX_TOKENS'):
            self.config.generation.max_tokens = int(tokens)
        
        # API configuration
        if host := os.getenv('API_HOST'):
            self.config.api_host = host
        if port := os.getenv('API_PORT'):
            self.config.api_port = int(port)
        
        # Logging
        if level := os.getenv('LOG_LEVEL'):
            self.config.log_level = level.upper()
        
        # Performance
        if metrics := os.getenv('ENABLE_METRICS'):
            self.config.enable_metrics = metrics.lower() in ('true', '1', 'yes')
        if caching := os.getenv('ENABLE_CACHING'):
            self.config.enable_caching = caching.lower() in ('true', '1', 'yes')
        
        # Cost tracking
        if track := os.getenv('TRACK_COSTS'):
            self.config.track_costs = track.lower() in ('true', '1', 'yes')
    
    def _load_file(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            path = Path(config_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self._update_config(data)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
    
    def _update_config(self, data: Dict[str, Any], prefix: str = ''):
        """
        Recursively update configuration from dictionary.
        
        Args:
            data: Configuration data
            prefix: Prefix for nested keys
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Handle nested configuration
                if hasattr(self.config, key):
                    sub_config = getattr(self.config, key)
                    if hasattr(sub_config, '__dataclass_fields__'):
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_config, sub_key):
                                setattr(sub_config, sub_key, sub_value)
            else:
                # Handle top-level configuration
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def _validate(self):
        """Validate configuration values."""
        # Check required environment variables
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not set - OpenAI features will not work")
        
        if not self.config.database.connection_string:
            logger.warning("DATABASE_URL not set - database features will not work")
        
        # Validate weights sum to 1.0
        if abs(self.config.search.dense_weight + self.config.search.sparse_weight - 1.0) > 0.01:
            logger.warning("Dense and sparse weights should sum to 1.0")
        
        # Validate batch sizes
        if self.config.embedding.batch_size_dense <= 0:
            raise ValueError("Dense batch size must be positive")
        if self.config.embedding.batch_size_sparse <= 0:
            raise ValueError("Sparse batch size must be positive")
        
        # Validate API configuration
        if self.config.api_port < 1 or self.config.api_port > 65535:
            raise ValueError("API port must be between 1 and 65535")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'database.max_connections')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        obj = self.config
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        
        return obj
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'database.max_connections')
            value: Value to set
        """
        parts = key.split('.')
        obj = self.config
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
        else:
            raise KeyError(f"Configuration key not found: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return asdict(self.config)
    
    def to_json(self, indent: int = 2) -> str:
        """
        Export configuration as JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON configuration string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Configuration saved to {filepath}")
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database
    
    @property
    def embedding(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.config.embedding
    
    @property
    def search(self) -> SearchConfig:
        """Get search configuration."""
        return self.config.search
    
    @property
    def generation(self) -> GenerationConfig:
        """Get generation configuration."""
        return self.config.generation


# Global configuration instance
_config_instance: Optional[ConfigService] = None


def get_config() -> ConfigService:
    """
    Get global configuration instance.
    
    Returns:
        ConfigService instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigService()
    return _config_instance


def load_config(config_file: Optional[str] = None) -> ConfigService:
    """
    Load and return configuration.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        ConfigService instance
    """
    global _config_instance
    _config_instance = ConfigService(config_file)
    return _config_instance