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

  

export OPENAI_API_KEY="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"
export DATABASE_URL="postgresql://postgres@localhost:5435/wikipedia"