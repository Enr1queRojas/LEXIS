import unittest
from rag_app.agents.web_agent import summarize_html_with_ai
from rag_app.ingestion.chunker import chunk_text


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.test_url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
        self.max_tokens = 50
        self.overlap = 10

    def test_chunking_from_web_summary(self):
        print("🔍 Getting summary from web agent...")
        summary = summarize_html_with_ai(self.test_url)

        print(f"\n📄 Summary received (first 300 chars):\n{summary[:300]}...\n")
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)

        print("🔪 Splitting summary into chunks...")
        chunks = chunk_text(summary, max_tokens=self.max_tokens, overlap=self.overlap)

        print(f"\n✅ Total chunks generated: {len(chunks)}\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"[Chunk {i}]\n{chunk}\n")

        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(isinstance(c, str) for c in chunks))

    def test_chunker_empty_text(self):
        empty_text = ""
        chunks = chunk_text(empty_text)
        self.assertEqual(chunks, [], msg="Chunker should return empty list for empty input.")

    def test_chunker_short_text(self):
        short_text = "Only one sentence here."
        chunks = chunk_text(short_text, max_tokens=10)
        self.assertEqual(chunks, [short_text], msg="Chunker should return single chunk for short input.")

    def test_chunker_exact_token_limit(self):
        text = ("This is a sentence. " * 12) + "Hi there."  # Exactly 50 tokens
        chunks = chunk_text(text.strip(), max_tokens=50, overlap=0)
        self.assertGreaterEqual(len(chunks), 1, msg="Should produce at least one chunk.")
        self.assertTrue(all(isinstance(c, str) for c in chunks))


if __name__ == "__main__":
    unittest.main()
