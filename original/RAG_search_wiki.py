#!/usr/bin/env python3

# Wikipedia RAG Search using dense embeddings
# This script performs retrieval-augmented generation on Wikipedia articles
# using OpenAI's text-embedding-3-large for retrieval and GPT-5 mini for generation.
#
# Required environment variables:
# export DATABASE_URL="postgresql://postgres@localhost/wikipedia"
# export OPENAI_API_KEY="your_openai_api_key"
#
# Dependencies: psycopg2-binary, openai, pgvector

import os
import psycopg2
import json
from openai import OpenAI
from pgvector.psycopg2 import register_vector

# Configuration
MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
TOP_K = 5  # Number of articles to retrieve
GPT_MODEL = "gpt-5-mini"  # For answer generation

def get_embedding(client, text, model=MODEL_NAME):
    """Generate embedding for query text."""
    response = client.embeddings.create(
        input=text,
        model=model,
        dimensions=EMBEDDING_DIMENSION
    )
    return response.data[0].embedding

def search_wikipedia_articles(query_embedding, search_type="content", limit=TOP_K):
    """
    Search Wikipedia articles using vector similarity.
    search_type: "title", "content", or "both"
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()
    
    vec_str = json.dumps(query_embedding)
    
    if search_type == "title":
        sql = """
        SELECT id, title, content, title_vector <-> %s::vector AS distance
        FROM articles
        WHERE title_vector IS NOT NULL
        ORDER BY title_vector <-> %s::vector
        LIMIT %s;
        """
    elif search_type == "content":
        sql = """
        SELECT id, title, content, content_vector <-> %s::vector AS distance
        FROM articles
        WHERE content_vector IS NOT NULL
        ORDER BY content_vector <-> %s::vector
        LIMIT %s;
        """
    else:  # both - combine results
        # Search both title and content, then merge
        title_results = search_wikipedia_articles(query_embedding, "title", limit)
        content_results = search_wikipedia_articles(query_embedding, "content", limit)
        
        # Merge and deduplicate by article ID
        merged = {}
        for result in title_results + content_results:
            article_id = result[0]
            if article_id not in merged or result[3] < merged[article_id][3]:  # Keep best distance
                merged[article_id] = result
        
        # Sort by distance and return top results
        results = sorted(merged.values(), key=lambda x: x[3])[:limit]
        cur.close()
        conn.close()
        return results
    
    cur.execute(sql, (vec_str, vec_str, limit))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results

def generate_answer(client, query, context_articles, model=GPT_MODEL):
    """Generate answer using GPT with retrieved Wikipedia context."""
    
    # Prepare context from retrieved articles
    context_parts = []
    for i, (article_id, title, content, distance) in enumerate(context_articles):
        # Limit content length for context window
        content_preview = content[:800] if content else ""
        context_parts.append(f"Article {i+1}: {title}\n{content_preview}...")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for GPT
    prompt = f"""Based on the following Wikipedia articles, please answer the user's question. 
If the information needed to answer the question is not available in the provided articles, please say so.

Wikipedia Articles:
{context}

Question: {query}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided Wikipedia articles. Be accurate and cite which article(s) you're referencing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {e}"

def display_search_results(results, query):
    """Display search results in a formatted way."""
    print(f"\n🔍 Search Results for: '{query}'\n")
    print("=" * 80)
    
    for i, (article_id, title, content, distance) in enumerate(results):
        print(f"\n{i+1}. {title}")
        print(f"   Article ID: {article_id}")
        print(f"   Distance: {distance:.4f}")
        
        # Show content preview
        content_preview = content[:200] if content else "No content available"
        if len(content) > 200:
            content_preview += "..."
        
        print(f"   Preview: {content_preview}")
        print("-" * 80)

def main():
    """Main interactive loop for Wikipedia RAG search."""
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print("📚 Wikipedia RAG Search System")
        print("Using model:", MODEL_NAME)
        print("Connected to Wikipedia database")
        print("\nType 'quit' to exit, 'help' for commands")
        
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set correctly.")
        return
    
    # Check database connection
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        
        # Verify embeddings exist
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector IS NOT NULL;")
        embedded_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles;")
        total_count = cur.fetchone()[0]
        
        print(f"Database status: {embedded_count}/{total_count} articles have embeddings")
        
        if embedded_count == 0:
            print("⚠️  No embeddings found! Run create_emb_wiki.py first.")
            return
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n❓ Enter your question: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  help    - Show this help")
                print("  quit    - Exit the program")
                print("  Any other text will be used as a search query")
                continue
            elif not user_input:
                continue
            
            print(f"\n🔍 Searching for: '{user_input}'")
            
            # Generate embedding for query
            query_embedding = get_embedding(client, user_input)
            
            # Search Wikipedia articles
            results = search_wikipedia_articles(query_embedding, search_type="content")
            
            if not results:
                print("No results found.")
                continue
            
            # Display search results
            display_search_results(results, user_input)
            
            # Generate answer
            print("\n🤖 Generating answer...\n")
            answer = generate_answer(client, user_input, results)
            
            print("📝 Answer:")
            print(answer)
            print("\n" + "=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()