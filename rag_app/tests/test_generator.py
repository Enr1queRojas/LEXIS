# rag_app/tests/test_generator.py

import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from rag_app.generator import generator
except ModuleNotFoundError:
    # Create minimal stubs if optional dependencies are missing
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda *args, **kwargs: None

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, prompt):
            pass

    fake_genai.GenerativeModel = FakeModel
    sys.modules.setdefault("google", types.ModuleType("google"))
    sys.modules["google.generativeai"] = fake_genai

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

    os.environ.setdefault("GEMINI_API_KEY", "dummy")

    from rag_app.generator import generator


class TestGenerator(unittest.TestCase):
    def test_generate_answer(self):
        query = "What is LexisNexis known for?"
        context = [
            "LexisNexis is a global provider of legal, regulatory and business information.",
            "They offer AI-powered tools like Lexis+ to support litigation and research."
        ]

        mock_response = MagicMock()
        mock_response.text = "Mocked answer"

        with patch.object(generator.model, "generate_content", return_value=mock_response):
            answer = generator.generate_answer(query, context)

        self.assertEqual(answer, "Mocked answer")

if __name__ == "__main__":
    unittest.main()
