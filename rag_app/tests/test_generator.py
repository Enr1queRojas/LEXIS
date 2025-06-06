# rag_app/tests/test_generator.py

import unittest
from rag_app.generator.generator import generate_answer

class TestGenerator(unittest.TestCase):
    def test_generate_answer(self):
        query = "What is LexisNexis known for?"
        context = [
            "LexisNexis is a global provider of legal, regulatory and business information.",
            "They offer AI-powered tools like Lexis+ to support litigation and research."
        ]
        answer = generate_answer(query, context)
        print("\n📤 Generated answer:\n", answer)
        self.assertTrue(isinstance(answer, str) and len(answer) > 0)

if __name__ == "__main__":
    unittest.main()
