"""
Base data processing utilities for pgvector RAG lab.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Container for document data."""
    id: Any
    content: str
    metadata: Dict[str, Any] = None
    title: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ProcessedChunk:
    """Container for processed text chunks."""
    chunk_id: str
    document_id: Any
    content: str
    metadata: Dict[str, Any] = None
    position: int = 0
    overlap_with_previous: int = 0
    overlap_with_next: int = 0


class DataProcessor(ABC):
    """
    Abstract base class for data processing.
    """
    
    @abstractmethod
    def process(self, data: Any) -> List[Document]:
        """
        Process raw data into documents.
        
        Args:
            data: Raw data to process
            
        Returns:
            List of processed documents
        """
        pass
    
    @abstractmethod
    def validate(self, document: Document) -> bool:
        """
        Validate a processed document.
        
        Args:
            document: Document to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass


class TextCleaner:
    """
    Text cleaning and normalization utilities.
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False
    ):
        """
        Initialize text cleaner.
        
        Args:
            remove_html: Remove HTML tags
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_whitespace: Normalize whitespace
            remove_special_chars: Remove special characters
            lowercase: Convert to lowercase
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
    
    def clean(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&[a-z]+;', '', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-]', '', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def remove_citations(self, text: str) -> str:
        """
        Remove citation markers like [1], [2], etc.
        
        Args:
            text: Text with citations
            
        Returns:
            Text without citations
        """
        return re.sub(r'\[\d+\]', '', text)
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters.
        
        Args:
            text: Text with unicode
            
        Returns:
            Normalized text
        """
        import unicodedata
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


class DataValidator:
    """
    Data validation utilities.
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None
    ):
        """
        Initialize data validator.
        
        Args:
            min_length: Minimum content length
            max_length: Maximum content length
            required_fields: Required metadata fields
            forbidden_patterns: Patterns that invalidate content
        """
        self.min_length = min_length
        self.max_length = max_length
        self.required_fields = required_fields or []
        self.forbidden_patterns = forbidden_patterns or []
    
    def validate_document(self, document: Document) -> Tuple[bool, Optional[str]]:
        """
        Validate a document.
        
        Args:
            document: Document to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check content length
        if len(document.content) < self.min_length:
            return False, f"Content too short ({len(document.content)} < {self.min_length})"
        
        if self.max_length and len(document.content) > self.max_length:
            return False, f"Content too long ({len(document.content)} > {self.max_length})"
        
        # Check required fields
        if document.metadata:
            for field in self.required_fields:
                if field not in document.metadata:
                    return False, f"Missing required field: {field}"
        elif self.required_fields:
            return False, "No metadata present but required fields specified"
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, document.content):
                return False, f"Forbidden pattern found: {pattern}"
        
        return True, None
    
    def validate_batch(self, documents: List[Document]) -> Tuple[List[Document], List[Tuple[Document, str]]]:
        """
        Validate a batch of documents.
        
        Args:
            documents: Documents to validate
            
        Returns:
            Tuple of (valid_documents, invalid_documents_with_reasons)
        """
        valid = []
        invalid = []
        
        for doc in documents:
            is_valid, error = self.validate_document(doc)
            if is_valid:
                valid.append(doc)
            else:
                invalid.append((doc, error))
        
        return valid, invalid


class ContentDeduplicator:
    """
    Content deduplication utilities.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        use_hash: bool = True
    ):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering duplicates
            use_hash: Use hash-based deduplication
        """
        self.similarity_threshold = similarity_threshold
        self.use_hash = use_hash
        self.seen_hashes = set()
    
    def generate_hash(self, content: str) -> str:
        """
        Generate hash for content.
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash
        """
        # Normalize before hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, content: str) -> bool:
        """
        Check if content is duplicate.
        
        Args:
            content: Content to check
            
        Returns:
            True if duplicate
        """
        if not self.use_hash:
            return False
        
        content_hash = self.generate_hash(content)
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents.
        
        Args:
            documents: Documents to deduplicate
            
        Returns:
            Deduplicated documents
        """
        unique_docs = []
        
        for doc in documents:
            if not self.is_duplicate(doc.content):
                unique_docs.append(doc)
            else:
                logger.debug(f"Removed duplicate document: {doc.id}")
        
        return unique_docs
    
    def reset(self):
        """Reset seen hashes."""
        self.seen_hashes.clear()


class MetadataExtractor:
    """
    Extract metadata from documents.
    """
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text.
        
        Args:
            text: Text to extract dates from
            
        Returns:
            List of found dates
        """
        # Common date patterns
        patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b',  # Month DD, YYYY
        ]
        
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return dates
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of numbers
        """
        # Match integers and decimals
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (simple pattern-based).
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            'capitalized_words': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        }
        
        return entities
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text (simple frequency-based).
        
        Args:
            text: Text to extract keywords from
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        # Simple word frequency approach
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequencies
        from collections import Counter
        word_counts = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(top_k)]


class DataStatistics:
    """
    Calculate statistics for data.
    """
    
    @staticmethod
    def calculate_document_stats(documents: List[Document]) -> Dict[str, Any]:
        """
        Calculate statistics for documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {
                'count': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'total_characters': 0
            }
        
        lengths = [len(doc.content) for doc in documents]
        
        return {
            'count': len(documents),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_characters': sum(lengths),
            'unique_sources': len(set(doc.source for doc in documents if doc.source)),
            'with_metadata': sum(1 for doc in documents if doc.metadata)
        }
    
    @staticmethod
    def calculate_chunk_stats(chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """
        Calculate statistics for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'count': 0,
                'avg_length': 0,
                'avg_overlap': 0
            }
        
        lengths = [len(chunk.content) for chunk in chunks]
        overlaps = [chunk.overlap_with_next for chunk in chunks if chunk.overlap_with_next > 0]
        
        return {
            'count': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_overlap': sum(overlaps) / len(overlaps) if overlaps else 0,
            'unique_documents': len(set(chunk.document_id for chunk in chunks))
        }