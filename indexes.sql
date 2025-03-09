-- execution plan without an index
------------------------------------------------------------------------------------------------------------------------->
 Limit  (cost=133.11..133.12 rows=5 width=27) (actual time=7.658..7.660 rows=5 loops=1)
   ->  Sort  (cost=133.11..135.61 rows=1000 width=27) (actual time=7.656..7.657 rows=5 loops=1)
         Sort Key: ((embedding <-> '[-0.0060701305,-0.008093507,-0.0019467601,0.015574081,0.012467623,0.032596912,-0.0284?'>
         Sort Method: top-N heapsort  Memory: 25kB
         ->  Seq Scan on film  (cost=0.00..116.50 rows=1000 width=27) (actual time=0.041..7.389 rows=1000 loops=1)
 Planning Time: 0.104 ms
 Execution Time: 7.680 ms
(7 rows)




-- create a HNSW index on the embedding column of the film table
CREATE INDEX film_embedding_idx ON public.film USING hnsw (embedding vector_l2_ops);
CREATE INDEX film_embedding_ivfflat_idx ON public.film USING ivfflat (embedding) WITH (lists='100')

-- create a HNSW index on the embedding column of the netflix_shows table
CREATE INDEX netflix_shows_embedding_idx ON public.netflix_shows USING hnsw (embedding vector_l2_ops);
 

EXPLAIN ANALYZE 
    SELECT film_id, title
    FROM film
    ORDER BY embedding <-> '[-0.0060701305,-0.008093507,-0.0019467601,0.015574081,0.012467623,0.032596912,-0.0284...]' --choose any embedding from the table
    LIMIT 5;


-- execution plan for a query that uses the HNSW index
 Limit  (cost=133.11..133.12 rows=5 width=27) (actual time=7.498..7.500 rows=5 loops=1)
   ->  Sort  (cost=133.11..135.61 rows=1000 width=27) (actual time=7.497..7.497 rows=5 loops=1)
         Sort Key: ((embedding <-> '[-0.0060701305,-0.008093507,-0.0019467601,0.015574081,0.012467623,0.032596912,-0.0284'>
         Sort Method: top-N heapsort  Memory: 25kB
         ->  Seq Scan on film  (cost=0.00..116.50 rows=1000 width=27) (actual time=0.034..7.243 rows=1000 loops=1)
 Planning Time: 0.115 ms
 Execution Time: 7.521 ms
(7 rows)


-- create an IVFFLAT index on the embedding column of the film table
CREATE INDEX film_embedding_ivfflat_idx ON film USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

SET ivfflat.probes = 5;

    -- New execution plan
    Limit  (cost=133.11..133.12 rows=5 width=27) (actual time=7.270..7.272 rows=5 loops=1)
   ->  Sort  (cost=133.11..135.61 rows=1000 width=27) (actual time=7.268..7.269 rows=5 loops=1)
         Sort Key: ((embedding <-> '[-0.0060701305,-0.008093507,-0.0019467601,0.015574081,0.012467623,0.032596912,-0.0284'>
         Sort Method: top-N heapsort  Memory: 25kB
         ->  Seq Scan on film  (cost=0.00..116.50 rows=1000 width=27) (actual time=0.054..6.984 rows=1000 loops=1)
 Planning Time: 0.140 ms
 Execution Time: 7.293 ms
(7 rows)



SET ivfflat.probes = 10;
   
   -- New execution plan 
    Limit  (cost=104.75..120.17 rows=5 width=27) (actual time=0.459..0.499 rows=5 loops=1)
   ->  Index Scan using film_embedding_ivfflat_idx on film  (cost=104.75..3188.50 rows=1000 width=27) (actual time=0.458.>
         Order By: (embedding <-> '[-0.0060701305,-0.008093507,-0.0019467601,0.015574081,0.012467623,0.032596912,-0.02841'>
 Planning Time: 0.153 ms
 Execution Time: 0.524 ms
(5 rows)




------------------------------------------------------------------------------------------------------------------------->



CREATE INDEX film_embedding_cosine_idx 
  ON public.film USING hnsw (embedding vector_cosine_ops);

CREATE INDEX film_embedding_idx 
ON public.film USING hnsw (embedding vector_l2_ops);

--------------

CREATE INDEX netflix_shows_embedding_idx 
ON public.netflix_shows USING hnsw (embedding vector_l2_ops);

CREATE INDEX netflix_shows_embedding_cosine_idx 
  ON public.netflix_shows USING hnsw (embedding vector_cosine_ops);

---------------

CREATE INDEX film_embedding_ivfflat_idx 
ON film USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

CREATE INDEX film_embedding_ivfflat_cosine_idx 
  ON film USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);



CREATE INDEX netflix_embedding_ivfflat_cosine_idx 
  ON netflix_shows USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

---------------


CREATE INDEX netflix_embedding_cosine_idx 
  ON netflix_shows USING diskann (embedding vector_cosine_ops);

CREATE INDEX film_embedding_cosine_idx 
  ON film USING diskann (embedding vector_cosine_ops);  

CREATE INDEX netflix_embedding_idx 
  ON netflix_shows USING diskann (embedding vector_l2_ops);