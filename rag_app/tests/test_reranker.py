import unittest
from rag_app.retriever.reranker import rerank_chunks

class TestReranker(unittest.TestCase):
    def test_rerank_chunks(self):
        query = "What is LexisNexis known for?"
        chunks = [
            "LexisNexis provides legal and business information.",
            "Apple designs smartphones and computers.",
            "LexisNexis also develops AI-powered legal research tools."
        ]

        results = rerank_chunks(query, chunks, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, tuple) and isinstance(r[0], str) and isinstance(r[1], float) for r in results))
        self.assertIn("LexisNexis", results[0][0])

if __name__ == '__main__':
    unittest.main()
