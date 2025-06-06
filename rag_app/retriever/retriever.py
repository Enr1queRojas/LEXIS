import logging
from typing import List

from rag_app.ingestion.embedder import Embedder
from rag_app.ingestion.indexer import get_collection


logger = logging.getLogger(__name__)

_embedder = Embedder()

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant chunks from the index based on a query.

    Args:
        query (str): Natural language question.
        top_k (int): Number of top documents to retrieve.

    Returns:
        List[str]: Top-k document chunks relevant to the query.
    """
    logger.info(f"🔍 Retrieving top-{top_k} chunks for query: '{query}'")
    
    # Embed the query into the same vector space
    query_embedding = _embedder.embed([query])[0].tolist()

    # Perform semantic search in the collection
    results = get_collection().query(
    query_embeddings=[query_embedding],
    n_results=top_k
    )


    documents = results.get("documents", [[]])[0]
    logger.info(f"✅ Retrieved {len(documents)} relevant chunks.")
    return documents
