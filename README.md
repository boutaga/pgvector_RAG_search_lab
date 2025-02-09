# Movies_pgvector_lab 

This repository is intended for educational purpose, by following the instructions you will be able to have a LAB and play with pgvector and similarity searches on PostgreSQL.

The aim of this LAB is to performa a similarity search to be able to recommend Netflix shows to DVDRental's users based on their renting profile. 


## Install pgvector

**Clone the Repository:**  
    Clone the pgvector extension from GitHub:
    
```bash
    cd /tmp
    git clone https://github.com/pgvector/pgvector.git
    cd pgvector
```
    
**Build and Install:**  
    Use make to compile the extension and then install it:
    
```bash
    make
    sudo make install
```
    
**Verify Installation:**  
    In psql, connect to your database and run:
    
```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    SELECT * FROM pg_extension WHERE extname = 'vector';
```
    
    This should show that pgvector is installed.






**Download the Sample Database:**  
    Visit [Neon’s PostgreSQL Sample Database](https://neon.tech/postgresql/postgresql-getting-started/postgresql-sample-database) and download the SQL dump file.
    Visit [Kaggle’s Netflix Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) and download the CSV file (e.g., `netflix_titles.csv`).






```sql

--psql 
CREATE DATABASE dvdrental;

--bash
unzip dvdrental.zip
  
pg_restore -U postgres -d dvdrental dvdrental.tar


--psql    
\c dvdrental  
    
CREATE TABLE netflix_shows (                                                                                        
    show_id TEXT PRIMARY KEY,
    type VARCHAR(20),
    title TEXT,
    director TEXT,
    "cast" TEXT,
    country TEXT,
    date_added TEXT,      -- stored as text; later you can convert to DATE with TO_DATE if desired
    release_year INTEGER,
    rating VARCHAR(10),
    duration VARCHAR(20),
    listed_in TEXT,
    description TEXT
);

--import from the csv file
COPY netflix_shows(show_id, type, title, director, "cast", country, date_added, release_year, rating, duration, listed_in, description)
FROM '/home/postgres/LAB_vector/netflix_titles.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

    
ALTER TABLE film ADD COLUMN embedding vector(1536);
ALTER TABLE netflix_shows ADD COLUMN embedding vector(1536);
    
CREATE INDEX IF NOT EXISTS film_embedding_idx ON film USING hnsw (embedding vector_l2_ops);   
CREATE INDEX IF NOT EXISTS netflix_shows_embedding_idx ON netflix_shows USING hnsw (embedding vector_l2_ops);
```
