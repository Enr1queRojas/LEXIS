import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cargamos el modelo Cross-Encoder solo una vez
_reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Reordena los chunks según su relevancia con el query usando CrossEncoder.
    
    Retorna los `top_k` chunks más relevantes junto con su score.
    """
    if not chunks:
        logger.warning("⚠️ No chunks provided to rerank.")
        return []

    logger.info(f"📊 Reranking {len(chunks)} chunks for query: '{query}'")

    inputs = [(query, chunk) for chunk in chunks]
    scores = _reranker_model.predict(inputs)

    # Forzamos a que los scores sean tipo float (no np.float32)
    scored_chunks = [(chunk, float(score)) for chunk, score in zip(chunks, scores)]

    top_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:top_k]

    logger.info(f"✅ Top-{top_k} chunks reranked successfully.")
    return top_chunks

