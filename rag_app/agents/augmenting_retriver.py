# rag_app/agents/augmenting_retriever.py

import logging
from rag_app.retriever.retriever import retrieve_relevant_chunks
from rag_app.generator.generator import generate_answer
from rag_app.utils.helpers import search_relevant_urls
from rag_app.ingestion.indexer import index_url

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def rag_with_fallback(query: str, top_k: int = 3, relevance_threshold: float = 0.3):
    logger.info(f"❓ Initial retrieval for query: {query}")
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    if not chunks or all(len(chunk.strip()) < 20 for chunk in chunks):
        logger.warning("⚠️ Low relevance or no chunks found — invoking fallback indexer.")

        url_candidates = search_relevant_urls(query, top_k=3)
        for entry in url_candidates:
            try:
                logger.info(f"🌐 Indexing fallback URL: {entry['url']} | Title: {entry['title']}")
                index_url(entry["url"], persist=True)
            except Exception as e:
                logger.warning(f"⛔ Error indexing {entry['url']}: {e}")

        logger.info("🔁 Retrying retrieval after fallback indexing...")
        chunks = retrieve_relevant_chunks(query, top_k=top_k)
        print("🔍 Retrieved chunks:", chunks)

    return generate_answer(query, chunks)
