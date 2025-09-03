#!/usr/bin/env python3

# This script retrieves embeddings for a customer's rented films and recommends Netflix shows based on similarity.
# It uses the pgvector library to handle vector columns in PostgreSQL.
# The database schema is based on the DVD rental example database in which we added the netflix_shows table.
# In order to run this script, you need to have the database URL set as an environment variable.
# example : export DATABASE_URL="postgresql://postgres@localhost/dvdrental"


import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

def get_customer_profile(customer_id, cur):
    """
    Retrieve all film embeddings for films that the customer has rented,
    and compute the average embedding as the user's profile.
    """
    query = """
    SELECT f.film_id, f.embedding
    FROM rental r
    JOIN inventory i ON r.inventory_id = i.inventory_id
    JOIN film f ON i.film_id = f.film_id
    WHERE r.customer_id = %s;
    """
    cur.execute(query, (customer_id,))
    films = cur.fetchall()
    if not films:
        print("No rentals found for this customer.")
        return None

    embeddings = []
    for film_id, emb in films:
        if emb is None:
            print(f"Film ID {film_id} does not have an embedding; skipping.")
            continue
        embeddings.append(np.array(emb))

    if not embeddings:
        print("No valid embeddings found for this customer.")
        return None

    # Compute the average embedding as a list of floats
    profile_embedding = np.mean(embeddings, axis=0)
    return profile_embedding.tolist()

def recommend_netflix():
    customer_input = input("Enter customer id: ").strip()
    if not customer_input.isdigit():
        print("Invalid customer id.")
        return
    customer_id = int(customer_input)

    # Connect to the dvdrental database
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost/dvdrental")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    # Create the customer profile by averaging film embeddings
    user_profile = get_customer_profile(customer_id, cur)
    if user_profile is None:
        cur.close()
        conn.close()
        return

    # Modify the SQL query to cast the parameter to vector and filter out rows with NULL embeddings
    query = """
    SELECT title, description, embedding <-> (%s)::vector AS distance
    FROM netflix_shows
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> (%s)::vector
    LIMIT 5;
    """
    cur.execute(query, (user_profile, user_profile))
    recommendations = cur.fetchall()

    if recommendations:
        print(f"\nTop 5 Netflix recommendations for customer {customer_id}:")
        for idx, (title, description, distance) in enumerate(recommendations, start=1):
            # Check if distance is None, though it shouldn't be with the WHERE clause.
            if distance is None:
                print(f"{idx}. {title} (no valid similarity distance)")
            else:
                print(f"{idx}. {title} (similarity distance: {distance:.4f})")
    else:
        print("No recommendations found.")

    cur.close()
    conn.close()

if __name__ == '__main__':
    recommend_netflix()