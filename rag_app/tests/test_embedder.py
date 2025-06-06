import unittest
import numpy as np
from rag_app.ingestion.embedder import Embedder


class TestEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder(model_name="all-MiniLM-L6-v2")

    def test_embed_multiple_texts(self):
        texts = [
            "LexisNexis provides legal and business analytics.",
            "They integrate AI in legal search tools.",
            "This is a test chunk for embedding."
        ]
        embeddings = self.embedder.embed(texts)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertGreater(embeddings.shape[1], 0)  # embedding dim should be > 0

    def test_embed_empty_list(self):
        embeddings = self.embedder.embed([])
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 0)

    def test_embed_single_text(self):
        text = "LexisNexis uses machine learning."
        embedding = self.embedder.embed_single(text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(embedding.shape[0], 0)

    def test_embed_single_empty(self):
        text = "   "
        embedding = self.embedder.embed_single(text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
