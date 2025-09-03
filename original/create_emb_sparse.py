#!/usr/bin/env python3
import os
import time
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pgvector.psycopg2 import register_vector # Use sparse vector registration

# --- Configuration ---
# Tune this batch size based on your available GPU memory (if using GPU) or RAM/CPU
BATCH_SIZE = 10 # SPLADE models can be memory intensive, start smaller
# Choose your SPLADE model
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Database connection string
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost/dvdrental")
# Column to store sparse embeddings
SPARSE_EMBEDDING_COLUMN = "sparse_embedding"
# --- End Configuration ---

def initialize_model_and_tokenizer(model_name):
    """Loads the SPLADE model and tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name} onto device: {DEVICE}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model and tokenizer loaded.")
    return tokenizer, model

def format_sparse_vector(indices, values, dim):
    """Formats sparse vector data into pgvector sparsevec string format."""
    if not isinstance(indices, list):
        indices = indices.tolist()
    if not isinstance(values, list):
        values = values.tolist()

    if len(indices) == 0:
        # Handle cases with no activation (empty sparse vector)
        return f'{{}}/{{dim}}'

    # Ensure values are floats for correct formatting
    value_strs = [f"{v:.6f}" for v in values] # Format to 6 decimal places
    
    # Create dictionary string manually
    pairs = [f"{idx}:{val}" for idx, val in zip(indices, value_strs)]
    dict_str = "{" + ",".join(pairs) + "}"
    
    # Return in pgvector format '{index:value,...}/dimension'
    return f"{dict_str}/{dim}"


def get_batch_sparse_embeddings(tokenizer, model, texts, max_retries=3):
    """
    Generates sparse embeddings for a list of texts using a SPLADE model.
    Returns a list of sparse vectors in pgvector string format.
    """
    if not texts:
        return []

    for attempt in range(max_retries):
        try:
            # Tokenize texts
            tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512) # Adjust max_length if needed
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()} # Move tensors to the correct device

            # Get model output (logits) - no gradient calculation needed
            with torch.no_grad():
                output = model(**tokens).logits

            # Compute sparse weights (ReLU + max pooling over sequence length)
            # This follows common SPLADE processing: apply ReLU to logits and then max-pool
            # across the sequence length dimension (dim=1) to get one vector per document.
            sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1).values
            
            # Get indices and values for non-zero weights
            batch_sparse_vectors = []
            vocab_size = tokenizer.vocab_size
            
            for vec in sparse_weights:
                indices = vec.nonzero().squeeze().cpu() # Get indices of non-zero elements
                values = vec[indices].cpu() # Get corresponding values

                # Filter out potential NaN/Inf values if any (shouldn't happen with ReLU)
                is_finite = torch.isfinite(values)
                indices = indices[is_finite]
                values = values[is_finite]

                # Thresholding (optional but recommended): Keep only weights above a certain value
                # threshold = 0.1 # Example threshold - adjust as needed
                # keep_mask = values > threshold
                # indices = indices[keep_mask]
                # values = values[keep_mask]

                # Format for pgvector
                sparse_vec_str = format_sparse_vector(indices, values, vocab_size)
                batch_sparse_vectors.append(sparse_vec_str)

            return batch_sparse_vectors

        except RuntimeError as e:
            # Catch CUDA out of memory errors specifically
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM Error on batch size {len(texts)}. Attempt {attempt + 1}/{max_retries}. Try reducing BATCH_SIZE.")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed due to CUDA OOM after {max_retries} retries. Reduce BATCH_SIZE.") from e
                time.sleep(2) # Wait before retrying
                # Optional: Could try garbage collection here `torch.cuda.empty_cache()`
            else:
                print(f"Runtime error processing batch: {e}. Attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                     raise Exception(f"Failed after {max_retries} retries.") from e
                time.sleep(1) # Wait before retrying non-OOM runtime errors
        except Exception as e:
            # Handle other potential errors
            print(f"Unexpected error processing batch: {e}")
            raise # Re-raise unexpected errors

    raise Exception("Failed to get sparse embeddings after maximum retries for a batch.")


def update_table_sparse_embeddings(conn, table_name, id_column, text_column, tokenizer, model):
    """
    Update the sparse_embedding column for all rows in a given table using batched processing.
    """
    cur = conn.cursor()
    # Select rows where the sparse embedding is NULL or doesn't exist yet
    cur.execute(f"SELECT {id_column}, {text_column} FROM {table_name} WHERE {SPARSE_EMBEDDING_COLUMN} IS NULL;")
    rows = cur.fetchall()
    if not rows:
        print(f"No rows found needing sparse embeddings update in table {table_name}.")
        cur.close()
        return

    total_rows = len(rows)
    print(f"Found {total_rows} rows to process in {table_name}.")

    # Prepare batches
    batches = []
    current_ids = []
    current_texts = []
    processed_count = 0
    start_time_total = time.time()

    for row_id, text in rows:
        if not text or not text.strip():
            print(f"{table_name.capitalize()} {id_column} {row_id} has empty text; skipping.")
            processed_count += 1
            continue
        current_ids.append(row_id)
        current_texts.append(text.strip()) # Ensure no leading/trailing whitespace

        if len(current_texts) >= BATCH_SIZE:
            batches.append((list(current_ids), list(current_texts)))
            current_ids.clear()
            current_texts.clear()

    if current_texts: # Add the last partial batch
        batches.append((current_ids, current_texts))

    print(f"Processing {total_rows} rows ({processed_count} skipped) from {table_name} in {len(batches)} batches (batch size = {BATCH_SIZE})...")

    for batch_ids, batch_texts in batches:
        batch_start_time = time.time()
        try:
            sparse_embeddings = get_batch_sparse_embeddings(tokenizer, model, batch_texts)

            # Prepare data for batch update (executemany is generally faster)
            update_data = []
            for row_id, sparse_embedding_str in zip(batch_ids, sparse_embeddings):
                update_data.append((sparse_embedding_str, row_id))

            # Use executemany for potentially faster updates
            update_query = f"UPDATE {table_name} SET {SPARSE_EMBEDDING_COLUMN} = %s WHERE {id_column} = %s;"
            cur.executemany(update_query, update_data)
            conn.commit()

            processed_count += len(batch_ids)
            batch_end_time = time.time()
            elapsed_time = batch_end_time - batch_start_time
            total_elapsed = time.time() - start_time_total
            rows_per_sec_batch = len(batch_ids) / elapsed_time if elapsed_time > 0 else float('inf')
            rows_per_sec_total = processed_count / total_elapsed if total_elapsed > 0 else float('inf')

            print(f"Updated batch for {table_name} (IDs: {batch_ids[:3]}...{batch_ids[-1:]}). "
                  f"Batch time: {elapsed_time:.2f}s ({rows_per_sec_batch:.2f} rows/s). "
                  f"Processed: {processed_count}/{total_rows} ({rows_per_sec_total:.2f} rows/s total avg).")

            # Optional small delay to prevent overwhelming the system, adjust if needed
            # time.sleep(0.1)

        except Exception as e:
            conn.rollback() # Rollback the transaction on error for this batch
            print(f"ERROR processing batch for {table_name} with IDs {batch_ids}: {e}")
            # Consider adding logic here to skip the batch or retry individual items
            processed_count += len(batch_ids) # Increment count even if failed to avoid infinite loops if skipping

    cur.close()
    print(f"Finished processing {table_name}. Total time: {time.time() - start_time_total:.2f} seconds.")


def main():
    print(f"Using device: {DEVICE}")
    tokenizer, model = initialize_model_and_tokenizer(MODEL_NAME)
    vocab_size = tokenizer.vocab_size
    print(f"Model vocabulary size: {vocab_size}. Ensure your sparsevec column uses this dimension.")

    # Connect to your PostgreSQL database
    conn = None
    try:
        print(f"Connecting to database: {DATABASE_URL.split('@')[-1]}") # Avoid logging password
        conn = psycopg2.connect(DATABASE_URL)
        # VERY IMPORTANT: Register the sparse vector adapter
        register_vector(conn)
        print("Database connection successful and sparse vector type registered.")

        # Update sparse embeddings in the film table
        print("\nUpdating film sparse embeddings...")
        update_table_sparse_embeddings(conn,
                                       table_name="film",
                                       id_column="film_id",
                                       text_column="description",
                                       tokenizer=tokenizer,
                                       model=model)

        # Update sparse embeddings in the netflix_shows table
        print("\nUpdating netflix_shows sparse embeddings...")
        update_table_sparse_embeddings(conn,
                                       table_name="netflix_shows",
                                       id_column="show_id",
                                       text_column="description",
                                       tokenizer=tokenizer,
                                       model=model)

    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    main()