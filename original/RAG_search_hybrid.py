import os
import psycopg2
import openai
import json
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Set your environment variables: DATABASE_URL and OPENAI_API_KEY
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://postgres@localhost/dvdrental"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Model configuration
DENSE_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
GPT_MODEL = "gpt-5-mini"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of results to retrieve from each search before re-ranking
TOP_K = 10

# Weights for combining dense and sparse similarity scores (distance)
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

def get_dense_embedding(text, model=DENSE_MODEL):
    """Generate a dense embedding for the given text using OpenAI API."""
    response = openai.embeddings.create(
        input=text,
        model=model,
        dimensions=EMBEDDING_DIMENSION
    )
    embedding = response.data[0].embedding
    return embedding

def initialize_sparse_model_and_tokenizer(model_name):
    """Load SPLADE model and tokenizer for sparse embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def get_sparse_embedding(tokenizer, model, text):
    """Generate a sparse embedding for the given text using SPLADE model."""
    tokens = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens).logits
    sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1).values
    vec = sparse_weights[0]
    indices = vec.nonzero().squeeze().cpu()
    values = vec[indices].cpu()
    # Format sparse vector as pgvector sparsevec string '{index:value,...}/dimension'
    value_strs = [f"{v:.6f}" for v in values.tolist()]
    pairs = [f"{idx}:{val}" for idx, val in zip(indices.tolist(), value_strs)]
    dict_str = "{" + ",".join(pairs) + "}"
    sparse_vec_str = f"{dict_str}/{tokenizer.vocab_size}"
    return sparse_vec_str

def query_dense_similar_items(query_embedding, limit=TOP_K):
    """Perform similarity search on dense embeddings."""
    vec_str = json.dumps(query_embedding)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    sql = """
    SELECT show_id, title, description, embedding <-> %s::vector AS distance
    FROM netflix_shows
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """
    cur.execute(sql, (vec_str, vec_str, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def query_sparse_similar_items(sparse_vec_str, limit=TOP_K):
    """Perform similarity search on sparse embeddings."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    sql = """
    SELECT show_id, title, description, sparse_embedding <=> %s::sparsevec AS distance
    FROM netflix_shows
    WHERE sparse_embedding IS NOT NULL
    ORDER BY sparse_embedding <=> %s::sparsevec
    LIMIT %s;
    """
    cur.execute(sql, (sparse_vec_str, sparse_vec_str, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def merge_and_rerank(dense_results, sparse_results, dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT, top_k=TOP_K):
    """Merge dense and sparse search results and re-rank by weighted sum of distances."""
    merged = {}
    for show_id, title, description, dist in dense_results:
        merged[show_id] = {
            "title": title,
            "description": description,
            "dense_dist": dist,
            "sparse_dist": None
        }
    for show_id, title, description, dist in sparse_results:
        if show_id in merged:
            merged[show_id]["sparse_dist"] = dist
        else:
            merged[show_id] = {
                "title": title,
                "description": description,
                "dense_dist": None,
                "sparse_dist": dist
            }
    LARGE_DIST = 1e6
    scored = []
    for show_id, data in merged.items():
        d_dist = data["dense_dist"] if data["dense_dist"] is not None else LARGE_DIST
        s_dist = data["sparse_dist"] if data["sparse_dist"] is not None else LARGE_DIST
        combined_score = dense_weight * d_dist + sparse_weight * s_dist
        scored.append((combined_score, show_id, data["title"], data["description"]))
    scored.sort(key=lambda x: x[0])
    top_results = [(item[2], item[3]) for item in scored[:top_k]]
    return top_results

def generate_answer(query, context):
    """Generate an answer using OpenAI ChatCompletion API."""
    messages = [
        {"role": "system", "content":  "You are a helpful assistant. You must answer the following question using only the context provided from the local database. "
                "Do not include any external information. If the answer is not present in the context, respond with 'No relevant information is available.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def classify_query(query):
    messages = [
        {"role": "system", "content": "Classify the following query as either 'structured' if it can be answered directly from database fields (e.g., release year, director, cast) or 'semantic' if it requires semantic understanding. Respond with only 'structured' or 'semantic'."},
        {"role": "user", "content": query}
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=1,
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

def structured_query(query):
    messages = [
        {"role": "system", "content": "Given the following PostgreSQL table structure:\n\n"
                                      "Table: netflix_shows\n"
                                      "Columns:\n"
                                      "- show_id (text)\n"
                                      "- type (text)\n"
                                      "- title (text)\n"
                                      "- director (text)\n"
                                      "- cast (text)\n"
                                      "- country (text)\n"
                                      "- date_added (text)\n"
                                      "- release_year (integer)\n"
                                      "- rating (text)\n"
                                      "- duration (text)\n"
                                      "- listed_in (text)\n"
                                      "- description (text)\n\n"
                                      "Generate an SQL query to answer the user's question. Only return the SQL query without explanation or markdown formatting."},
        {"role": "user", "content": query}
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=100,
        temperature=0
    )
    sql_query = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    print("\nGenerated SQL Query:\n", sql_query)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(sql_query)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def main():
    query = input("Enter your question: ")
    query_type = classify_query(query)

    if query_type == "structured":
        results = structured_query(query)
        if not results:
            print("No relevant documents were found.")
            return
        print("\nStructured Query Results:")
        context = ""
        for idx, row in enumerate(results, 1):
            print(f"{idx}. {row}")
            context += f"{row}\n"
        answer = generate_answer(query, context)
        print("\nAnswer:", answer)
        return
    query_dense_emb = get_dense_embedding(query)
    tokenizer, model = initialize_sparse_model_and_tokenizer(SPLADE_MODEL)
    query_sparse_emb = get_sparse_embedding(tokenizer, model, query)
    dense_results = query_dense_similar_items(query_dense_emb)
    print("\nDense Embedding Results:")
    for idx, (show_id, title, description, distance) in enumerate(dense_results, 1):
        print(f"{idx}. Title: {title}, Distance: {distance:.4f}")

    sparse_results = query_sparse_similar_items(query_sparse_emb)
    print("\nSparse Embedding Results:")
    for idx, (show_id, title, description, distance) in enumerate(sparse_results, 1):
        print(f"{idx}. Title: {title}, Distance: {distance:.4f}")
    if not dense_results and not sparse_results:
        print("No relevant documents were found.")
        return
    top_results = merge_and_rerank(dense_results, sparse_results)
    print("\nCombined Weighted Results:")
    for idx, (title, description) in enumerate(top_results, 1):
        print(f"{idx}. Title: {title}")
    context = ""
    for title, description in top_results:
        context += f"Title: {title}\nDescription: {description}\n\n"
    answer = generate_answer(query, context)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
