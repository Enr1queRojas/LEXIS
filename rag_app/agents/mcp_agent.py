import logging
from rag_app.generator.generator import generate_answer

logger = logging.getLogger(__name__)

def answer_with_mcp(query: str) -> str:
    """
    Fallback when no relevant chunks are found in vector index.
    This uses the LLM directly to answer the query (e.g., Gemini).
    """
    logger.info("🧠 No relevant chunks found — triggering MCP agent.")
    
    # Puedes mandar solo el query o envolverlo en una instrucción especial
    answer = generate_answer(query=query, chunks=[])
    
    logger.info("✅ MCP agent generated a fallback answer.")
    return answer
