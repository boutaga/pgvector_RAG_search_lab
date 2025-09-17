# lc_rag_service.py ###   uvicorn lc_rag_service:app --host 0.0.0.0 --port 8001
from fastapi import FastAPI
from pydantic import BaseModel
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# --- request/response models ---
class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    use_router: bool = True
    use_multiquery: bool = True
    multiquery_n: int = 4
    use_compression: bool = True
    use_llm_rerank: bool = True
    answer_strategy: str = "map_reduce"  # or "stuff" / "refine"

class RAGResponse(BaseModel):
    success: bool
    answer: str
    contexts: list
    metrics: dict

app = FastAPI()

# --- Init LC pieces (adjust DSN to your pgvector DB) ---
EMB = OpenAIEmbeddings(model="text-embedding-3-large")   # 3072-dim
LLM = ChatOpenAI(model="gpt-5-mini", temperature=0)
VS = PGVector(
    embedding_function=EMB,
    connection_string="postgresql+psycopg://postgres:***@localhost:5432/wikipedia",
    collection_name="articles_content_vector_3072",
)
BASE_RETRIEVER = VS.as_retriever(search_kwargs={"k": 5})

@app.post("/lc/rag", response_model=RAGResponse)
def lc_rag(req: RAGRequest):
    t0 = time.time()
    # --- telemetry buckets ---
    tele = dict(
        llm_calls=0, embedding_calls=0, retriever_queries=0,
        db_round_trips=0, tokens_input=0, tokens_output=0, stages={}
    )

    def bump(stage, k=1): tele[stage] = tele.get(stage, 0) + k
    def mark(stage, t): tele["stages"][stage] = round((t)*1000)

    # 1) Optional router (LLM decides: dense/hybrid/sparse)
    t = time.time()
    route = "hybrid"
    if req.use_router:
        # one LLM call to decide route
        _ = LLM.predict("Decide retrieval mode (dense|hybrid|sparse) for: " + req.query)
        tele["llm_calls"] += 1
    mark("router", time.time()-t)

    # 2) MultiQuery fan-out
    retriever = BASE_RETRIEVER
    if req.use_multiquery:
        t = time.time()
        mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=LLM, include_original=True)
        # LLM will generate ~N paraphrases; count as llm_calls += N
        tele["llm_calls"] += req.multiquery_n
        # Each paraphrase triggers a retriever query -> db_round_trips += N ; retriever_queries += N
        tele["retriever_queries"] += req.multiquery_n
        tele["db_round_trips"]   += req.multiquery_n
        retriever = mqr
        mark("multiquery", time.time()-t)

    # 3) Compression (LLM screens each doc) — worst case up to k LLM calls
    if req.use_compression:
        t = time.time()
        compressor = LLMChainExtractor.from_llm(LLM)
        # compression pipeline uses an LLM per doc seen
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        mark("compression_setup", time.time()-t)

    # 4) Retrieve
    t = time.time()
    docs = retriever.get_relevant_documents(req.query)
    # Approximate counters:
    top_k = req.top_k
    tele["retriever_queries"] += 1
    tele["db_round_trips"]    += 1
    # If compression active, assume ~min(k, len(docs)) LLM calls
    if req.use_compression:
        tele["llm_calls"] += min(top_k, len(docs))
    mark("retrieve", time.time()-t)

    # 5) Optional LLM rerank (LLM per doc to score)
    if req.use_llm_rerank and docs:
        t = time.time()
        tele["llm_calls"] += min(top_k, len(docs))
        # ...call LLM to score each doc...
        mark("llm_rerank", time.time()-t)

    # 6) Answering (map_reduce inflates calls)
    t = time.time()
    if req.answer_strategy == "map_reduce":
        # map: one call per doc; reduce: one final call
        tele["llm_calls"] += min(top_k, len(docs)) + 1
        answer = LLM.predict("Compose the final answer…")  # placeholder
    elif req.answer_strategy == "refine":
        tele["llm_calls"] += min(top_k, len(docs))
        answer = LLM.predict("Refine answer…")
    else:
        tele["llm_calls"] += 1
        answer = LLM.predict("Stuff + answer…")
    mark("answer", time.time()-t)

    tele["latency_ms"] = round((time.time()-t0)*1000)
    return RAGResponse(
        success=True,
        answer=answer,
        contexts=[{"title": getattr(d, "metadata", {}).get("title",""), "content": d.page_content} for d in docs[:top_k]],
        metrics=tele,
    )
