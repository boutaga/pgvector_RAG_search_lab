#!/usr/bin/env python3

# This script fetches rows from a PostgreSQL database, generates embeddings using the OpenAI API, and updates the database with the embeddings.
# In order to run this script, you need to have the OpenAI API key and the database URL set as environment variables.
# example : export OPENAI_API_KEY="your_openai_api_key" and export DATABASE_URL="postgresql://postgres@localhost/dvdrental"
# dependencies: psycopg2, openai, pgvector
# Additionally, depending on the tokens quota of your OpenAI account, you may need to adjust the BATCH_SIZE variable to optimize throughput and avoid rate limits with errors like "429" or "insufficient_quota".

# This script retrieves embe
import os
import time
import psycopg2
import numpy as np
from openai import OpenAI
from pgvector.psycopg2 import register_vector

# Tune this batch size to optimize throughput.
# For example, try 10, then 20, then 50 depending on the average token count in your texts.
BATCH_SIZE = 30

def get_batch_embeddings(client, texts, model, max_retries=5):
    """
    Retrieve embeddings for a list of texts in one API call.
    Uses exponential backoff for rate-limit errors.
    Returns a list of embeddings.
    """
    delay = 0.2  # initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "insufficient_quota" in error_str:
                print(f"Encountered rate limit/quota error: {error_str}. Retrying in {delay} seconds for batch of size {len(texts)}...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                print(f"Non-retryable error: {error_str}")
                raise
    raise Exception("Failed to get embeddings after maximum retries for a batch.")

def update_table_embeddings(conn, table_name, id_column, text_column, model):
    """
    Update the embedding column for all rows in a given table using batched API calls.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT {id_column}, {text_column} FROM {table_name};")
    rows = cur.fetchall()
    if not rows:
        print(f"No rows found in table {table_name}.")
        cur.close()
        return

    # Prepare batches: each batch is a tuple of (list_of_ids, list_of_texts)
    batches = []
    current_ids = []
    current_texts = []
    for row in rows:
        row_id, text = row
        if not text:
            print(f"{table_name.capitalize()} {id_column} {row_id} has no {text_column}; skipping.")
            continue
        current_ids.append(row_id)
        current_texts.append(text)
        if len(current_texts) >= BATCH_SIZE:
            batches.append((current_ids, current_texts))
            current_ids = []
            current_texts = []
    if current_texts:
        batches.append((current_ids, current_texts))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    total = sum(len(batch_ids) for batch_ids, _ in batches)
    print(f"Processing {total} rows from {table_name} in {len(batches)} batches (batch size = {BATCH_SIZE})...")

    for batch_ids, batch_texts in batches:
        try:
            embeddings = get_batch_embeddings(client, batch_texts, model)
            for row_id, embedding in zip(batch_ids, embeddings):
                cur.execute(f"UPDATE {table_name} SET embedding = %s WHERE {id_column} = %s;", (embedding, row_id))
            conn.commit()
            print(f"Updated batch for {table_name}: {batch_ids}")
            # Optional: adjust this delay if you are not hitting rate limits.
            time.sleep(0.5)
        except Exception as e:
            print(f"Error processing batch for {table_name} with IDs {batch_ids}: {e}")
    cur.close()

def main():
    # We use text-embedding-ada-002 which returns 1536-dimensional vectors.
    model = "text-embedding-ada-002"

    # Connect to your PostgreSQL database
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost/dvdrental")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)

    # Update embeddings in the film table
    print("Updating film embeddings...")
    update_table_embeddings(conn, table_name="film", id_column="film_id", text_column="description", model=model)

    # Update embeddings in the netflix_shows table
    print("Updating netflix_shows embeddings...")
    update_table_embeddings(conn, table_name="netflix_shows", id_column="show_id", text_column="description", model=model)

    conn.close()

if __name__ == '__main__':
    main()