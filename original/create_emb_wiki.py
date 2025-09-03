#!/usr/bin/env python3

# This script generates embeddings for Wikipedia articles using OpenAI's text-embedding-3-small model
# and updates the PostgreSQL database with both title and content embeddings.
# 
# Required environment variables:
# export OPENAI_API_KEY="your_openai_api_key" 
# export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
#
# Dependencies: psycopg2-binary, openai, pgvector
#
# The script processes the articles table with columns:
# - id, title, content, title_vector, content_vector
# - Uses text-embedding-3-small (1536 dimensions) for better performance and cost

import os
import time
import psycopg2
import numpy as np
from openai import OpenAI
from pgvector.psycopg2 import register_vector

# Configuration
BATCH_SIZE = 50  # Increased for text-embedding-3-small efficiency
MODEL_NAME = "text-embedding-3-small"  # Updated to newer model
EMBEDDING_DIMENSION = 1536

def get_batch_embeddings(client, texts, model, max_retries=5):
    """
    Retrieve embeddings for a list of texts in one API call.
    Uses exponential backoff for rate-limit errors.
    Returns a list of embeddings.
    """
    delay = 0.2  # initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts, 
                model=model,
                dimensions=EMBEDDING_DIMENSION  # Explicitly set dimensions for consistency
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "insufficient_quota" in error_str or "rate_limit" in error_str.lower():
                print(f"Rate limit/quota error: {error_str}. Retrying in {delay} seconds for batch of {len(texts)} texts...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                print(f"Non-retryable error: {error_str}")
                raise
    raise Exception("Failed to get embeddings after maximum retries for a batch.")

def update_wikipedia_embeddings(conn, embedding_type="both"):
    """
    Update embeddings for Wikipedia articles.
    embedding_type: "title", "content", or "both"
    """
    cur = conn.cursor()
    
    # Get articles that need embeddings
    if embedding_type == "title":
        cur.execute("SELECT id, title FROM articles WHERE title_vector IS NULL AND title IS NOT NULL AND title != '';")
        column_to_update = "title_vector"
        text_column = "title"
    elif embedding_type == "content":
        cur.execute("SELECT id, content FROM articles WHERE content_vector IS NULL AND content IS NOT NULL AND content != '';")
        column_to_update = "content_vector"
        text_column = "content"
    else:  # both
        # Process titles first, then content
        update_wikipedia_embeddings(conn, "title")
        update_wikipedia_embeddings(conn, "content")
        return
    
    rows = cur.fetchall()
    if not rows:
        print(f"No articles found needing {embedding_type} embeddings.")
        cur.close()
        return

    # Prepare batches
    batches = []
    current_ids = []
    current_texts = []
    
    for row in rows:
        article_id, text = row
        if not text or text.strip() == "":
            print(f"Article {article_id} has empty {text_column}; skipping.")
            continue
            
        # Truncate very long content to avoid token limits (approximate)
        if embedding_type == "content" and len(text) > 32000:  # ~8000 tokens
            text = text[:32000] + "..."
            
        current_ids.append(article_id)
        current_texts.append(text)
        
        if len(current_texts) >= BATCH_SIZE:
            batches.append((current_ids, current_texts))
            current_ids = []
            current_texts = []
    
    if current_texts:
        batches.append((current_ids, current_texts))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    total = sum(len(batch_ids) for batch_ids, _ in batches)
    print(f"Processing {total} {embedding_type} embeddings in {len(batches)} batches (batch size = {BATCH_SIZE})...")

    processed = 0
    for i, (batch_ids, batch_texts) in enumerate(batches):
        try:
            embeddings = get_batch_embeddings(client, batch_texts, MODEL_NAME)
            
            # Update database
            for article_id, embedding in zip(batch_ids, embeddings):
                cur.execute(f"UPDATE articles SET {column_to_update} = %s WHERE id = %s;", (embedding, article_id))
            
            conn.commit()
            processed += len(batch_ids)
            print(f"Updated {embedding_type} batch {i+1}/{len(batches)}: {len(batch_ids)} articles. Progress: {processed}/{total} ({100*processed/total:.1f}%)")
            
            # Rate limiting - be conservative with API calls
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing {embedding_type} batch with IDs {batch_ids}: {e}")
            # Continue with next batch rather than failing completely
            continue
    
    cur.close()
    print(f"Completed {embedding_type} embeddings: {processed}/{total} articles processed.")

def verify_embeddings(conn):
    """Verify embedding generation results."""
    cur = conn.cursor()
    
    # Check total articles
    cur.execute("SELECT COUNT(*) FROM articles;")
    total_articles = cur.fetchone()[0]
    
    # Check embedding coverage
    cur.execute("SELECT COUNT(*) FROM articles WHERE title_vector IS NOT NULL;")
    title_embeddings = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector IS NOT NULL;")
    content_embeddings = cur.fetchone()[0]
    
    print(f"\nEmbedding Status:")
    print(f"Total articles: {total_articles}")
    print(f"Title embeddings: {title_embeddings} ({100*title_embeddings/total_articles:.1f}%)")
    print(f"Content embeddings: {content_embeddings} ({100*content_embeddings/total_articles:.1f}%)")
    
    # Check embedding dimensions
    cur.execute("SELECT array_length(title_vector, 1) as title_dim, array_length(content_vector, 1) as content_dim FROM articles WHERE title_vector IS NOT NULL AND content_vector IS NOT NULL LIMIT 1;")
    result = cur.fetchone()
    if result:
        title_dim, content_dim = result
        print(f"Embedding dimensions - Title: {title_dim}, Content: {content_dim}")
    
    cur.close()

def main():
    # Connect to PostgreSQL database
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost/wikipedia")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        print(f"Connected to database. Using model: {MODEL_NAME}")
        
        # Verify table structure
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'articles' 
            AND column_name IN ('id', 'title', 'content', 'title_vector', 'content_vector')
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        print(f"Table structure verified: {columns}")
        cur.close()
        
        # Generate embeddings
        print("\nStarting Wikipedia embedding generation...")
        update_wikipedia_embeddings(conn, embedding_type="both")
        
        # Verify results
        verify_embeddings(conn)
        
        conn.close()
        print("\nWikipedia embedding generation completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your DATABASE_URL and OPENAI_API_KEY environment variables are set correctly.")

if __name__ == '__main__':
    main()