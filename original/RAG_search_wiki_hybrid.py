#!/usr/bin/env python3

# Wikipedia Hybrid RAG Search with dense/sparse embeddings and query classification
# This script performs advanced retrieval-augmented generation on Wikipedia articles
# using both dense (OpenAI) and sparse (SPLADE) embeddings with intelligent query routing.
#
# Required environment variables:
# export DATABASE_URL="postgresql://postgres@localhost/wikipedia" 
# export OPENAI_API_KEY="your_openai_api_key"
#
# Dependencies: psycopg2-binary, openai, transformers, torch, pgvector

import os
import psycopg2
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pgvector.psycopg2 import register_vector

# Configuration
DENSE_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "naver/splade-cocondenser-ensembledistil"
GPT_MODEL = "gpt-5-mini"
EMBEDDING_DIMENSION = 3072

TOP_K = 8  # Results per search type before re-ranking
FINAL_K = 5  # Final results after re-ranking

# Weights for hybrid re-ranking
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for SPLADE model (loaded once)
sparse_tokenizer = None
sparse_model = None

def initialize_sparse_model():
    """Load SPLADE model and tokenizer for sparse embeddings."""
    global sparse_tokenizer, sparse_model
    
    if sparse_tokenizer is None:
        print(f"Loading SPLADE model: {SPARSE_MODEL} on {DEVICE}")
        sparse_tokenizer = AutoTokenizer.from_pretrained(SPARSE_MODEL)
        sparse_model = AutoModelForMaskedLM.from_pretrained(SPARSE_MODEL)
        sparse_model.to(DEVICE)
        sparse_model.eval()

def get_dense_embedding(client, text, model=DENSE_MODEL):
    """Generate dense embedding using OpenAI."""
    response = client.embeddings.create(
        input=text,
        model=model,
        dimensions=EMBEDDING_DIMENSION
    )
    return response.data[0].embedding

def get_sparse_embedding(text):
    """Generate sparse embedding using SPLADE model."""
    initialize_sparse_model()
    
    # Truncate text if too long
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    tokens = sparse_tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    
    with torch.no_grad():
        output = sparse_model(**tokens).logits
    
    sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1).values
    vec = sparse_weights[0]
    
    indices = vec.nonzero().squeeze().cpu()
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    values = vec[indices].cpu()
    
    if len(indices) == 0:
        return f"{{}}/%d" % sparse_tokenizer.vocab_size
    
    value_strs = [f"{v:.6f}" for v in values.tolist()]
    pairs = [f"{idx}:{val}" for idx, val in zip(indices.tolist(), value_strs)]
    dict_str = "{" + ",".join(pairs) + "}"
    sparse_vec_str = f"{dict_str}/{sparse_tokenizer.vocab_size}"
    
    return sparse_vec_str

def classify_query(client, query):
    """
    Classify query as 'factual', 'conceptual', or 'exploratory' to determine search strategy.
    """
    classification_prompt = f"""Classify the following query into one of these categories:

1. 'factual' - Asks for specific facts, dates, names, numbers, or direct information
2. 'conceptual' - Asks about ideas, theories, explanations, or how things work  
3. 'exploratory' - Broad questions seeking general information or overviews

Query: "{query}"

Respond with only the category name (factual, conceptual, or exploratory):"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=10,
            temperature=0
        )
        
        category = response.choices[0].message.content.strip().lower()
        if category in ['factual', 'conceptual', 'exploratory']:
            return category
        else:
            return 'conceptual'  # default
    except:
        return 'conceptual'  # default fallback

def search_dense_articles(query_embedding, search_field="content", limit=TOP_K):
    """Search using dense embeddings."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()
    
    vec_str = json.dumps(query_embedding)
    
    if search_field == "title":
        vector_column = "title_vector"
    else:
        vector_column = "content_vector"
    
    sql = f"""
    SELECT id, title, content, {vector_column} <-> %s::vector AS distance
    FROM articles
    WHERE {vector_column} IS NOT NULL
    ORDER BY {vector_column} <-> %s::vector
    LIMIT %s;
    """
    
    cur.execute(sql, (vec_str, vec_str, limit))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results

def search_sparse_articles(sparse_embedding, search_field="content", limit=TOP_K):
    """Search using sparse embeddings."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()
    
    if search_field == "title":
        sparse_column = "title_sparse"
    else:
        sparse_column = "content_sparse"
    
    sql = f"""
    SELECT id, title, content, {sparse_column} <=> %s::sparsevec AS distance
    FROM articles
    WHERE {sparse_column} IS NOT NULL
    ORDER BY {sparse_column} <=> %s::sparsevec
    LIMIT %s;
    """
    
    cur.execute(sql, (sparse_embedding, sparse_embedding, limit))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results

def merge_and_rerank(dense_results, sparse_results, dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT):
    """Merge and re-rank results from dense and sparse searches."""
    merged = {}
    
    # Add dense results
    for article_id, title, content, distance in dense_results:
        merged[article_id] = {
            "title": title,
            "content": content,
            "dense_dist": distance,
            "sparse_dist": float('inf')  # Default high distance
        }
    
    # Add sparse results
    for article_id, title, content, distance in sparse_results:
        if article_id in merged:
            merged[article_id]["sparse_dist"] = distance
        else:
            merged[article_id] = {
                "title": title,
                "content": content,
                "dense_dist": float('inf'),
                "sparse_dist": distance
            }
    
    # Calculate weighted scores and sort
    ranked_results = []
    for article_id, data in merged.items():
        # Normalize distances (lower is better)
        dense_score = data["dense_dist"] if data["dense_dist"] != float('inf') else 2.0
        sparse_score = data["sparse_dist"] if data["sparse_dist"] != float('inf') else 2.0
        
        # Weighted combination
        combined_score = dense_weight * dense_score + sparse_weight * sparse_score
        
        ranked_results.append((
            article_id,
            data["title"],
            data["content"],
            combined_score,
            data["dense_dist"],
            data["sparse_dist"]
        ))
    
    # Sort by combined score and return top results
    ranked_results.sort(key=lambda x: x[3])
    return ranked_results[:FINAL_K]

def hybrid_search(client, query, search_strategy="adaptive"):
    """
    Perform hybrid search using both dense and sparse embeddings.
    search_strategy: "adaptive", "dense_only", "sparse_only", or "hybrid"
    """
    
    if search_strategy == "adaptive":
        query_type = classify_query(client, query)
        print(f"🧠 Query classified as: {query_type}")
        
        if query_type == "factual":
            # Factual queries benefit more from sparse search
            strategy = "sparse_heavy"
            dense_w, sparse_w = 0.3, 0.7
        elif query_type == "conceptual":
            # Conceptual queries benefit from dense search
            strategy = "dense_heavy" 
            dense_w, sparse_w = 0.7, 0.3
        else:  # exploratory
            # Balanced approach for exploratory queries
            strategy = "balanced"
            dense_w, sparse_w = 0.5, 0.5
        
        print(f"🎯 Using {strategy} search (dense: {dense_w}, sparse: {sparse_w})")
        
    else:
        dense_w, sparse_w = DENSE_WEIGHT, SPARSE_WEIGHT
    
    if search_strategy == "dense_only":
        query_embedding = get_dense_embedding(client, query)
        dense_results = search_dense_articles(query_embedding, limit=FINAL_K)
        return [(r[0], r[1], r[2], r[3], r[3], float('inf')) for r in dense_results]
    
    elif search_strategy == "sparse_only":
        sparse_embedding = get_sparse_embedding(query)
        sparse_results = search_sparse_articles(sparse_embedding, limit=FINAL_K)
        return [(r[0], r[1], r[2], r[3], float('inf'), r[3]) for r in sparse_results]
    
    else:  # hybrid or adaptive
        print("🔍 Performing dense search...")
        query_embedding = get_dense_embedding(client, query)
        dense_results = search_dense_articles(query_embedding)
        
        print("🔍 Performing sparse search...")
        sparse_embedding = get_sparse_embedding(query)
        sparse_results = search_sparse_articles(sparse_embedding)
        
        print("⚖️  Merging and re-ranking results...")
        return merge_and_rerank(dense_results, sparse_results, dense_w, sparse_w)

def generate_comprehensive_answer(client, query, context_articles, model=GPT_MODEL):
    """Generate comprehensive answer using GPT with retrieved context."""
    
    # Prepare context from articles
    context_parts = []
    for i, (article_id, title, content, combined_score, dense_dist, sparse_dist) in enumerate(context_articles):
        content_preview = content[:1000] if content else ""
        context_parts.append(f"""Article {i+1}: "{title}"
{content_preview}{"..." if len(content) > 1000 else ""}
[Relevance scores - Combined: {combined_score:.3f}, Dense: {dense_dist:.3f}, Sparse: {sparse_dist:.3f}]
""")
    
    context = "\n" + "="*50 + "\n".join(context_parts)
    
    prompt = f"""You are a knowledgeable Wikipedia assistant. Based on the following Wikipedia articles, provide a comprehensive and accurate answer to the user's question.

Guidelines:
- If the information is available in the articles, provide a detailed answer
- Reference specific articles when appropriate (e.g., "According to the article on...")  
- If information is incomplete or missing, acknowledge this
- Maintain factual accuracy and cite sources when possible

Wikipedia Context:
{context}

User Question: {query}

Comprehensive Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful Wikipedia assistant that provides accurate, well-sourced answers based on provided articles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {e}"

def display_hybrid_results(results, query):
    """Display hybrid search results with scoring details."""
    print(f"\n🔍 Hybrid Search Results for: '{query}'\n")
    print("=" * 100)
    
    for i, (article_id, title, content, combined_score, dense_dist, sparse_dist) in enumerate(results):
        print(f"\n{i+1}. {title}")
        print(f"   Article ID: {article_id}")
        print(f"   📊 Scores - Combined: {combined_score:.4f} | Dense: {dense_dist:.4f} | Sparse: {sparse_dist:.4f}")
        
        content_preview = content[:200] if content else "No content available"
        if len(content) > 200:
            content_preview += "..."
        
        print(f"   Preview: {content_preview}")
        print("-" * 100)

def main():
    """Main interactive loop for Wikipedia Hybrid RAG search."""
    
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print("📚 Wikipedia Hybrid RAG Search System")
        print(f"Dense Model: {DENSE_MODEL}")
        print(f"Sparse Model: {SPARSE_MODEL}")
        print(f"Device: {DEVICE}")
        
        # Verify database connection and embeddings
        DATABASE_URL = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        cur = conn.cursor()
        
        # Check embedding coverage
        cur.execute("SELECT COUNT(*) FROM articles WHERE content_vector IS NOT NULL;")
        dense_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM articles WHERE content_sparse IS NOT NULL;")
        sparse_count = cur.fetchone()[0] if cur.fetchone() else 0
        
        cur.execute("SELECT COUNT(*) FROM articles;")
        total_count = cur.fetchone()[0]
        
        print(f"📈 Database status:")
        print(f"   Dense embeddings: {dense_count}/{total_count} articles")
        print(f"   Sparse embeddings: {sparse_count}/{total_count} articles")
        
        if dense_count == 0:
            print("⚠️  No dense embeddings found! Run create_emb_wiki.py first.")
            return
        
        cur.close()
        conn.close()
        
        print("\n🔧 Commands:")
        print("  help     - Show help")
        print("  quit     - Exit")
        print("  dense    - Use dense search only")
        print("  sparse   - Use sparse search only") 
        print("  hybrid   - Use hybrid search (default)")
        print("  adaptive - Use adaptive search strategy")
        
    except Exception as e:
        print(f"Initialization error: {e}")
        return
    
    search_mode = "adaptive"
    
    # Interactive loop
    while True:
        try:
            user_input = input(f"\n❓ [{search_mode.upper()}] Enter your question: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands and search modes...")
                continue
            elif user_input.lower() in ['dense', 'sparse', 'hybrid', 'adaptive']:
                search_mode = user_input.lower()
                print(f"🔧 Search mode changed to: {search_mode}")
                continue
            elif not user_input:
                continue
            
            print(f"\n🚀 Starting {search_mode} search for: '{user_input}'")
            
            # Perform hybrid search
            results = hybrid_search(client, user_input, search_strategy=search_mode)
            
            if not results:
                print("❌ No results found.")
                continue
            
            # Display results
            display_hybrid_results(results, user_input)
            
            # Generate answer
            print("\n🤖 Generating comprehensive answer...\n")
            answer = generate_comprehensive_answer(client, user_input, results)
            
            print("📝 Answer:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()