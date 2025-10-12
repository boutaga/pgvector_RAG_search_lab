20:29:58 postgres@PG1:/home/postgres/RAG_lab_demo/lab/ [PG17] ls -la
total 60
drwxrwxrwx 12 postgres postgres  4096 Sep 13 10:39 .
drwxrwxr-x  3 postgres postgres  4096 Sep 13 10:32 ..
drwxrwxrwx  3 postgres postgres  4096 Sep 11 15:45 01_setup
drwxrwxrwx  2 postgres postgres  4096 Sep  5 20:20 02_data
drwxrwxrwx  2 postgres postgres  4096 Sep 13 13:45 03_embeddings
drwxrwxrwx  5 postgres postgres  4096 Sep 13 15:23 06_workflows
drwxrwxrwx  2 postgres postgres  4096 Sep 11 12:18 07_evaluation
drwxrwxrwx  3 postgres postgres  4096 Sep 13 14:11 api
drwxrwxrwx  3 postgres postgres  4096 Sep 13 14:42 core
-rw-r--r--  1 postgres postgres     0 Sep 13 10:39 __init__.py
drwxr-xr-x  2 postgres postgres  4096 Sep 13 10:39 __pycache__
-rw-rw-rw-  1 postgres postgres 11871 Sep  5 20:20 README.md
drwxrwxrwx  3 postgres postgres  4096 Sep 13 11:18 search
drwxr-xr-x  7 postgres postgres  4096 Sep 13 10:36 .venv
20:30:01 postgres@PG1:/home/postgres/RAG_lab_demo/lab/ [PG17] source .venv/bin/activate


python -m uvicorn lab.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload

(.venv) 20:40:46 postgres@PG1:/home/postgres/RAG_lab_demo/ [PG17] streamlit run lab/api/streamlit_app.py   --server.port 850
1   --server.address 0.0.0.0

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501



 curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is WAL in PostgreSQL and why is it important?","source":"wikipedia","method":"simple","search_type":"dense","top_k":5,"generate_answer":true}'



  curl http://localhost:8000/health



  ssh -i ~/PG1.pem -L 8000:127.0.0.1:8000 -N adrien@98.71.89.168

  ssh -i ~/PG1.pem -L 8501:127.0.0.1:8501 -N adrien@98.71.89.168

  ssh -i ~/PG1.pem -L 5678:127.0.0.1:5678 -N adrien@98.71.89.168

  

export OPENAI_API_KEY="secret"
export DATABASE_URL="postgresql://postgres@localhost:5435/wikipedia"



export DATABASE_URL="postgresql://postgres@localhost:5435/northwind_rag"
export OPENAI_API_KEY="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"




cd /home/postgres/RAG_lab_demo/lab/03_bi_mart_metadata_rag

source venv_lab3/bin/activate

python3 python/50_mart_planning_agent.py

streamlit run python/80_streamlit_demo.py



wikipedia=# select id, title, url from articles where title like '%zombie%';
  id   |         title         |                              url
-------+-----------------------+---------------------------------------------------------------
 66253 | List of zombie movies | https://simple.wikipedia.org/wiki/List%20of%20zombie%20movies
(1 row)

wikipedia=# WITH current_article AS (
    SELECT content_vector
    FROM articles
    WHERE id = 66253
)
SELECT
    a.title,
    a.url,
    a.content_vector <-> current_article.content_vector AS distance
FROM
    articles a,
    current_article
WHERE
    a.id <> 66353
ORDER BY
    distance
LIMIT 5;
               title                |                                     url                                     |      distance
------------------------------------+-----------------------------------------------------------------------------+--------------------
 List of zombie movies              | https://simple.wikipedia.org/wiki/List%20of%20zombie%20movies               |                  0
 Night of the Living Dead           | https://simple.wikipedia.org/wiki/Night%20of%20the%20Living%20Dead          | 0.7672253445926926
 List of Universal Pictures movies  | https://simple.wikipedia.org/wiki/List%20of%20Universal%20Pictures%20movies | 0.8052875540692092
 Zombie                             | https://simple.wikipedia.org/wiki/Zombie                                    | 0.8328822265980188
 List of Metro-Goldwyn-Mayer movies | https://simple.wikipedia.org/wiki/List%20of%20Metro-Goldwyn-Mayer%20movies  | 0.8753209206755904
(5 rows)