#!/usr/bin/env python3

# This script generates sparse embeddings for Wikipedia articles using SPLADE model
# and updates the PostgreSQL database. Since the Wikipedia articles table doesn't have
# sparse embedding columns by default, this script will add them if needed.
#
# Required environment variables:
# export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
#
# Dependencies: psycopg2-binary, transformers, torch, sentencepiece, pgvector
#
# The script processes the articles table and adds:
# - title_sparse (sparsevec) for title embeddings
# - content_sparse (sparsevec) for content embeddings

import os
import time
import psycopg2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pgvector.psycopg2 import register_vector

# Configuration
BATCH_SIZE = 5  # Smaller batch for memory efficiency with large articles
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_splade_model():
    """Load SPLADE model and tokenizer for sparse embeddings."""
    print(f"Loading SPLADE model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    
    print(f"Model vocabulary size: {tokenizer.vocab_size}")
    return tokenizer, model

def get_sparse_embedding(tokenizer, model, text):
    """Generate a sparse embedding for the given text using SPLADE model."""
    # Truncate text to avoid memory issues
    if len(text) > 2000:  # Approximate token limit
        text = text[:2000] + "..."
    
    tokens = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    
    with torch.no_grad():
        output = model(**tokens).logits
    
    # Apply SPLADE weighting: log(1 + ReLU(output))
    sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1).values
    vec = sparse_weights[0]
    
    # Get non-zero indices and values
    indices = vec.nonzero().squeeze().cpu()
    if indices.dim() == 0:  # Single element
        indices = indices.unsqueeze(0)
    values = vec[indices].cpu()
    
    # Format as pgvector sparsevec string: '{index:value,...}/dimension'
    if len(indices) == 0:
        return f"{{}}/%d" % tokenizer.vocab_size
    
    value_strs = [f"{v:.6f}" for v in values.tolist()]
    pairs = [f"{idx}:{val}" for idx, val in zip(indices.tolist(), value_strs)]
    dict_str = "{" + ",".join(pairs) + "}"
    sparse_vec_str = f"{dict_str}/{tokenizer.vocab_size}"
    
    return sparse_vec_str

def ensure_sparse_columns(conn):
    """Add sparse embedding columns to articles table if they don't exist."""
    cur = conn.cursor()
    
    # Check if sparse columns exist
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'articles' 
        AND column_name IN ('title_sparse', 'content_sparse');
    """)
    existing_columns = {row[0] for row in cur.fetchall()}
    
    # Add missing columns
    if 'title_sparse' not in existing_columns:
        print("Adding title_sparse column...")
        cur.execute("ALTER TABLE articles ADD COLUMN title_sparse sparsevec(30522);")  # SPLADE vocab size
        
    if 'content_sparse' not in existing_columns:
        print("Adding content_sparse column...")
        cur.execute("ALTER TABLE articles ADD COLUMN content_sparse sparsevec(30522);")
    
    conn.commit()
    
    # Create indexes for sparse columns if they don't exist
    try:
        print("Creating sparse indexes...")
        cur.execute("CREATE INDEX IF NOT EXISTS articles_title_sparse_idx ON articles USING ivfflat (title_sparse) WITH (lists = 1000);")
        cur.execute("CREATE INDEX IF NOT EXISTS articles_content_sparse_idx ON articles USING ivfflat (content_sparse) WITH (lists = 1000);")
        conn.commit()
    except Exception as e:
        print(f"Index creation warning: {e}")
    
    cur.close()

def update_sparse_embeddings(conn, tokenizer, model, embedding_type="both"):
    """
    Update sparse embeddings for Wikipedia articles.
    embedding_type: "title", "content", or "both"
    """
    cur = conn.cursor()
    
    # Get articles that need sparse embeddings
    if embedding_type == "title":
        cur.execute("SELECT id, title FROM articles WHERE title_sparse IS NULL AND title IS NOT NULL AND title != '';")
        column_to_update = "title_sparse"
        text_column = "title"
    elif embedding_type == "content":
        cur.execute("SELECT id, content FROM articles WHERE content_sparse IS NULL AND content IS NOT NULL AND content != '';")
        column_to_update = "content_sparse"
        text_column = "content"
    else:  # both
        update_sparse_embeddings(conn, tokenizer, model, "title")
        update_sparse_embeddings(conn, tokenizer, model, "content")
        return
    
    rows = cur.fetchall()
    if not rows:
        print(f"No articles found needing {embedding_type} sparse embeddings.")
        cur.close()
        return

    # Process in batches
    batches = []
    current_ids = []
    current_texts = []
    
    for row in rows:
        article_id, text = row
        if not text or text.strip() == "":
            continue
            
        current_ids.append(article_id)
        current_texts.append(text)
        
        if len(current_texts) >= BATCH_SIZE:
            batches.append((current_ids, current_texts))
            current_ids = []
            current_texts = []
    
    if current_texts:
        batches.append((current_ids, current_texts))

    total = sum(len(batch_ids) for batch_ids, _ in batches)
    print(f"Processing {total} {embedding_type} sparse embeddings in {len(batches)} batches (batch size = {BATCH_SIZE})...")

    processed = 0
    for i, (batch_ids, batch_texts) in enumerate(batches):
        try:
            # Process each text in the batch
            sparse_embeddings = []
            for text in batch_texts:
                sparse_vec = get_sparse_embedding(tokenizer, model, text)
                sparse_embeddings.append(sparse_vec)
            
            # Update database
            for article_id, sparse_embedding in zip(batch_ids, sparse_embeddings):
                cur.execute(f"UPDATE articles SET {column_to_update} = %s WHERE id = %s;", (sparse_embedding, article_id))
            
            conn.commit()
            processed += len(batch_ids)
            print(f"Updated {embedding_type} sparse batch {i+1}/{len(batches)}: {len(batch_ids)} articles. Progress: {processed}/{total} ({100*processed/total:.1f}%)")
            
            # Memory cleanup and brief pause
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing {embedding_type} sparse batch with IDs {batch_ids}: {e}")
            continue
    
    cur.close()
    print(f"Completed {embedding_type} sparse embeddings: {processed}/{total} articles processed.")

def verify_sparse_embeddings(conn):
    """Verify sparse embedding generation results."""
    cur = conn.cursor()
    
    # Check total articles
    cur.execute("SELECT COUNT(*) FROM articles;")
    total_articles = cur.fetchone()[0]
    
    # Check sparse embedding coverage
    cur.execute("SELECT COUNT(*) FROM articles WHERE title_sparse IS NOT NULL;")
    title_sparse = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM articles WHERE content_sparse IS NOT NULL;")
    content_sparse = cur.fetchone()[0]
    
    print(f"\nSparse Embedding Status:")
    print(f"Total articles: {total_articles}")
    print(f"Title sparse embeddings: {title_sparse} ({100*title_sparse/total_articles:.1f}%)")
    print(f"Content sparse embeddings: {content_sparse} ({100*content_sparse/total_articles:.1f}%)")
    
    # Show sample sparse embedding info
    cur.execute("""
        SELECT id, title,
               regexp_split_to_array(regexp_replace(title_sparse, '.*{(.*)}.*', '\\1'), ',') as sparse_elements
        FROM articles 
        WHERE title_sparse IS NOT NULL 
        LIMIT 1;
    """)
    sample = cur.fetchone()
    if sample:
        article_id, title, sparse_elements = sample
        non_zero_count = len([e for e in sparse_elements if e.strip() != ''])
        print(f"Sample sparse embedding for '{title[:50]}...': {non_zero_count} non-zero elements")
    
    cur.close()

def main():
    # Connect to PostgreSQL database
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost/wikipedia")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        print(f"Connected to Wikipedia database. Using SPLADE model: {MODEL_NAME}")
        
        # Initialize SPLADE model
        tokenizer, model = initialize_splade_model()
        
        # Ensure sparse columns exist
        ensure_sparse_columns(conn)
        
        # Generate sparse embeddings
        print("\nStarting Wikipedia sparse embedding generation...")
        update_sparse_embeddings(conn, tokenizer, model, embedding_type="both")
        
        # Verify results
        verify_sparse_embeddings(conn)
        
        conn.close()
        print("\nWikipedia sparse embedding generation completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your DATABASE_URL environment variable is set correctly.")
        print("Also ensure you have sufficient GPU memory or the script will use CPU (slower).")

if __name__ == '__main__':
    main()