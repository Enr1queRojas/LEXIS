import json
import logging
from typing import Optional, List
from pathlib import Path

from rag_app.agents.web_agent import summarize_html_with_ai
from rag_app.ingestion.chunker import chunk_text
from rag_app.ingestion.embedder import Embedder
from rag_app.generator.generator import generate_answer
from chromadb import Client
from chromadb.config import Settings

PERSIST_DIRECTORY = "data"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_FILE = DATA_DIR / "documents.json"

_embedder = Embedder()

class CustomEmbeddingFunction:
    def is_legacy(self):
        return True

    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input):
        return self.embedder.embed(input).tolist()

    def name(self):
        return "custom_embedder"

_embedding_function = CustomEmbeddingFunction(_embedder)

_chroma_client = Client(Settings(
    anonymized_telemetry=False,
    persist_directory=PERSIST_DIRECTORY
))

collection_name_default = "rag_index"
collection = _chroma_client.get_or_create_collection(
    name=collection_name_default,
    embedding_function=_embedding_function
)

def infer_title_from_text(summary: str) -> str:
    system_prompt = (
        "You are a professional assistant generating a clear and concise product title from a legal web summary.\n\n"
        "Summary:\n"
        f"{summary}\n\n"
        "Return only the title:"
    )
    return generate_answer(query="", chunks=[], system_prompt=system_prompt).strip()

def infer_keywords_from_text(text: str, max_keywords: int = 6) -> List[str]:
    system_prompt = (
        f"Extract {max_keywords} concise, relevant keywords from the following legal content. "
        "Avoid duplicates and keep them short:\n\n"
        f"{text}\n\n"
        "Keywords:"
    )
    raw = generate_answer(query="", chunks=[], system_prompt=system_prompt).strip()
    keywords = [k.strip().strip(",.") for k in raw.split(",") if k.strip()]
    return keywords[:max_keywords]

def index_url(
    url: str,
    collection_name: str = collection_name_default,
    max_tokens: int = 100,
    overlap: int = 20,
    persist: bool = True
) -> List[str]:
    logger.info(f"\U0001f310 Summarizing content from: {url}")
    summary = summarize_html_with_ai(url)
    if not summary.strip():
        logger.warning("⚠️ Empty summary returned from summarizer.")
        return []

    logger.info(f"\U0001f9e0 Inferring dynamic title...")
    title = infer_title_from_text(summary)
    keywords_page = infer_keywords_from_text(summary)

    logger.info(f"✂️ Chunking summary into segments (max_tokens={max_tokens}, overlap={overlap})")
    chunks = chunk_text(summary, max_tokens=max_tokens, overlap=overlap)
    if not chunks:
        logger.warning("⚠️ No chunks generated from the summary.")
        return []

    logger.info(f"📎 Generated {len(chunks)} chunks.")
    logger.info(f"📦 Generating embeddings for {len(chunks)} chunks...")
    embeddings = _embedder.embed(chunks)

    existing_ids = collection.get(ids=None, where={"source": url}).get("ids", [])
    if existing_ids:
        logger.info(f"🗑️ Removing {len(existing_ids)} existing chunks from index for {url}")
        collection.delete(ids=existing_ids)

    doc_ids = []
    documents_to_save = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{url}__chunk_{i}"

        metadata = {
            "source": url,
            "chunk_index": i,
            "title": title,
            "product": title,
            "contains_ai": "ai" in chunk.lower(),
            "keywords": ", ".join(keywords_page)
        }

        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[doc_id],
            metadatas=[metadata]
        )

        doc_ids.append(doc_id)
        documents_to_save.append({
            "id": doc_id,
            "text": chunk,
            "source": url,
            "chunk_index": i,
            "metadata": metadata
        })

    if persist:
        logger.info(f"📝 Preparing to update {DOCUMENTS_FILE}")
        existing = []
        if DOCUMENTS_FILE.exists():
            try:
                with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing = [doc for doc in existing if doc["source"] != url]
            except json.JSONDecodeError:
                logger.warning("⚠️ documents.json is empty or malformed. Starting fresh.")

        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing + documents_to_save, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Indexing complete. Total documents added: {len(doc_ids)}")
    return doc_ids

def reset_index():
    global collection
    logger.info(f"🧹 Previous collection '{collection_name_default}' deleted.")
    _chroma_client.delete_collection(collection_name_default)
    collection = _chroma_client.create_collection(
        name=collection_name_default,
        embedding_function=_embedding_function
    )
    logger.info("🗑️ Index reset — collection cleared and re-initialized.")

def get_collection():
    return collection

def query_index(query: str, top_k: int = 5) -> List[str]:
    logger.info(f"🔎 Querying index for: {query}")
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

if __name__ == "__main__":
    url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
    index_url(url=url, max_tokens=50, overlap=10)
