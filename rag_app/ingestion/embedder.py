import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Embedder:
    """
    Wrapper class for embedding text chunks using SentenceTransformer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedder with a pre-trained SentenceTransformer model.

        Args:
            model_name (str): Name of the model to use from sentence-transformers hub.
        """
        logger.info(f"🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ Model loaded successfully.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            np.ndarray: 2D array of embeddings, shape (n_texts, embedding_dim).
        """
        if not texts:
            logger.warning("⚠️ Received empty list of texts for embedding.")
            return np.array([])

        logger.info(f"📦 Embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        logger.info("✅ Embedding complete.")
        return np.array(embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single string.

        Args:
            text (str): Text string to embed.

        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,).
        """
        if not text.strip():
            logger.warning("⚠️ Empty or whitespace-only string received for embedding.")
            return np.array([])

        logger.info(f"📄 Embedding single text of length {len(text)}...")
        embedding = self.model.encode([text], show_progress_bar=False)
        logger.info("✅ Single embedding complete.")
        return embedding[0]
