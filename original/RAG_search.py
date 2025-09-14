import os
import psycopg2
import openai
import json

# Set your environment variables: DATABASE_URL and OPENAI_API_KEY
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://postgres@localhost/dvdrental"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configuration
MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
GPT_MODEL = "gpt-5-mini"

def get_embedding(text, model=MODEL_NAME):
    """
    Generate an embedding for the given text using the latest OpenAI API.
    Access attributes using dot notation.
    """
    response = openai.embeddings.create(
        input=text,
        model=model,
        dimensions=EMBEDDING_DIMENSION
    )
    embedding = response.data[0].embedding  # dot notation for attribute access
    return embedding

def query_similar_items(query_embedding, limit=5):
    """
    Connect to PostgreSQL and perform a similarity search using pgvector.
    Convert the embedding (a list of floats) into a JSON string.
    """
    vec_str = json.dumps(query_embedding)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    sql = """
    SELECT title, description, embedding <-> %s::vector AS distance
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

def generate_answer(query, context):
    """
    Generate an answer using the OpenAI ChatCompletion API with model GPT-4o.
    This uses the new openai.chat.completions.create interface.
    """
    messages = [
        {"role": "system", "content":  "You are a helpful assistant. You must answer the following question using only the context provided from the local database. "
                "Do not include any external information. If the answer is not present in the context, respond with 'No relevant information is available.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL,  # Using GPT-5 mini
        messages=messages,
        max_tokens=150,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

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