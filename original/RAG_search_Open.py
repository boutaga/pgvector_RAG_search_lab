import os
import psycopg2
import openai
import json

# Set your environment variables: DATABASE_URL and OPENAI_API_KEY
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://postgres@localhost/dvdrental"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate an embedding for the given text using the latest OpenAI API.
    Access attributes using dot notation.
    """
    response = openai.embeddings.create(
        input=text,
        model=model
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
        {"role": "system", "content":"You are a helpful assistant. Your primary goal is to answer the user's 'Question' using the provided 'Context'."
        "1. First, determine if the 'Context' contains information relevant to the 'Question'."
        "2. If yes, formulate your answer based *solely* on the relevant information found in the 'Context'."
        "3. If no, or if the 'Context' is insufficient, answer the 'Question' using your general knowledge (external sources)."
        "4. **Crucially:** If you use external sources (step 3), you MUST start your response with: 'The provided context did not contain the relevant information. Based on external information:'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",  # Ensure you have access to GPT-4o
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