import os
import psycopg2
import openai
import json

# Set your environment variables: DATABASE_URL and OPENAI_API_KEY
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://postgres@localhost/dvdrental"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for the given text using OpenAI API."""
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response["data"][0]["embedding"]
    return embedding

def query_similar_items(query_embedding, limit=5):
    """
    Connect to PostgreSQL and perform a similarity search using pgvector.
    It uses the <-> operator to measure distance between the stored embedding
    and the query embedding.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    sql = """
    SELECT title, description, embedding <-> %s::vector AS distance
    FROM netflix_shows
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """
    # Pass the embedding twice (for ordering) and the limit.
    cur.execute(sql, (json.dumps(query_embedding), json.dumps(query_embedding), limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def generate_answer(query, context):
    """
    Use the OpenAI completion API to generate an answer based on the user query
    and the retrieved context.
    """
    prompt = f"""You are a helpful assistant. Answer the following question using only the context provided.
    
Context:
{context}

Question: {query}
Answer:"""
    response = openai.Completion.create(
        engine="text-davinci-003",  # Alternatively, use ChatCompletion with gpt-3.5-turbo or gpt-4.
        prompt=prompt,
        max_tokens=150,
        temperature=0.2,
    )
    return response["choices"][0]["text"].strip()

def main():
    query = input("Enter your question: ")
    # Generate an embedding for the query
    query_embedding = get_embedding(query)
    # Retrieve top similar documents using pgvector similarity search
    similar_items = query_similar_items(query_embedding)
    
    if not similar_items:
        print("No relevant documents were found.")
        return

    # Build a context string by concatenating retrieved titles and descriptions.
    context = ""
    for title, description, distance in similar_items:
        context += f"Title: {title}\nDescription: {description}\n\n"
    
    # Generate the final answer using the query and the retrieved context.
    answer = generate_answer(query, context)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
