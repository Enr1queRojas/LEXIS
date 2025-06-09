#rag_app\app\main.py

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_app.retriever.retriever import retrieve_relevant_chunks
from rag_app.generator.generator import generate_answer
from rag_app.agents.mcp_agent import answer_with_mcp
from rag_app.ingestion.indexer import index_url

app = FastAPI(title="LEXIS RAG API")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


class IndexRequest(BaseModel):
    url: str

@app.get("/", response_class=FileResponse)
def read_index():
    return static_dir / "index.html"

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    chunks = retrieve_relevant_chunks(req.query, top_k=3)
    if not chunks:
        answer = answer_with_mcp(req.query)
    else:
        answer = generate_answer(req.query, chunks)
    return QueryResponse(answer=answer)


@app.post("/index")
def add_url(req: IndexRequest):
    doc_ids = index_url(req.url)
    return {"documents_indexed": len(doc_ids)}
