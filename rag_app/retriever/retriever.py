#rag_app\retriever\retriever.py

import logging
from typing import List

from rag_app.ingestion.embedder import Embedder
from rag_app.ingestion.indexer import get_collection

logger = logging.getLogger(__name__)

_embedder = Embedder()

def retrieve_relevant_chunks(query: str, top_k: int = 10, threshold: float = 0.5) -> List[str]:
    """
    Retrieve the most relevant chunks from the index based on a query.

    Args:
        query (str): Natural language question.
        top_k (int): Number of top documents to consider before filtering.
        threshold (float): Minimum similarity threshold (cosine similarity).

    Returns:
        List[str]: Filtered chunks with similarity above the threshold.
    """
    logger.info(f"🔍 Retrieving top-{top_k} chunks for query: '{query}'")

    query_embedding = _embedder.embed([query])[0].tolist()

    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    logger.info("📊 Similarities returned:")
    for i, (doc, score) in enumerate(zip(documents, distances)):
        logger.info(f"   - Chunk {i}: score={score:.4f} | preview={doc[:60]}")

    filtered_chunks = [
        doc for doc, score in zip(documents, distances) if score >= threshold
    ]

    logger.info(f"✅ Filtered {len(filtered_chunks)} relevant chunks (threshold: {threshold})")
    return filtered_chunks
