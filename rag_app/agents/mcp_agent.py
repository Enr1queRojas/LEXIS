import logging
from rag_app.generator.generator import generate_answer

logger = logging.getLogger(__name__)

def answer_with_mcp(query: str) -> str:
    """
    Fallback when no relevant chunks are found in vector index.
    This uses the LLM directly to answer the query (e.g., Gemini).
    """
    logger.info("🧠 No relevant chunks found — triggering MCP agent.")
    
    # You may send only the query or wrap it in a custom instruction.
    # The generate_answer function expects context_chunks as the second
    # argument. Previously the call used the wrong keyword "chunks",
    # which caused a TypeError.
    answer = generate_answer(query=query, context_chunks=[])
    
    logger.info("✅ MCP agent generated a fallback answer.")
    return answer
