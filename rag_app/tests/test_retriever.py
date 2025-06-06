import unittest
from rag_app.retriever.retriever import retrieve_relevant_chunks
from rag_app.ingestion.indexer import index_url, reset_index

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
        self.query = "What AI products does LexisNexis have?"

        reset_index()
        index_url(self.url)

    def test_retrieve_chunks(self):
        chunks = retrieve_relevant_chunks(self.query, top_k=3)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for i, c in enumerate(chunks):
            print(f"\n[Chunk {i+1}] {c[:150]}...")

if __name__ == "__main__":
    unittest.main()
