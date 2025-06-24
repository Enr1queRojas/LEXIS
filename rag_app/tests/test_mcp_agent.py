import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from rag_app.agents import mcp_agent
except ModuleNotFoundError:
    # Minimal stubs for optional dependencies
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

    os.environ.setdefault("GEMINI_API_KEY", "dummy")

    from rag_app.agents import mcp_agent


class TestMcpAgent(unittest.TestCase):
    def test_answer_with_mcp_uses_system_prompt(self):
        query = "Who are you?"
        with patch("rag_app.agents.mcp_agent.generate_answer", return_value="hi") as mock_gen:
            result = mcp_agent.answer_with_mcp(query, persist=False)

        self.assertEqual(result, "hi")
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs.get("system_prompt"), mcp_agent.MCP_SYSTEM_PROMPT)
        self.assertEqual(kwargs.get("chunks"), [])


if __name__ == "__main__":
    unittest.main()
