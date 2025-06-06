import unittest
from rag_app.agents.web_agent import summarize_html_with_ai, batch_scrape_and_summarize

TEST_URLS = [
    "https://www.lexisnexis.com/en-us/about-us/innovation.page",
    "https://www.lexisnexis.com/community/insights/legal/practical-guidance-news/b/blog/posts/rag-trusted-generative-ai"
]

class TestWebAgent(unittest.TestCase):

    def test_summarize_html_with_ai(self):
        """Should return non-empty summary from a single URL"""
        summary = summarize_html_with_ai(TEST_URLS[0])
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)  # Puedes ajustar el mínimo esperado

    def test_batch_scrape_and_summarize(self):
        """Should return a summary for each URL in the batch"""
        results = batch_scrape_and_summarize(TEST_URLS)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(TEST_URLS))
        for url, summary in results.items():
            self.assertIsInstance(summary, str)
            self.assertGreater(len(summary), 50)

if __name__ == "__main__":
    unittest.main()
