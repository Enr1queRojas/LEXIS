from rag_app.retriever.retriever import retrieve_relevant_chunks
from rag_app.agents.mcp_agent import answer_with_mcp
import logging

logger = logging.getLogger(__name__)

def search_with_fallback(query: str, top_k: int = 3) -> list[str]:
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    if not chunks or all(len(c.strip()) < 30 for c in chunks):
        logger.warning("⚠️ No sufficient chunks found — falling back to MCP agent.")
        return [answer_with_mcp(query)]

    return chunks



if __name__ == "__main__":
    user_query = input("🧠 Ask something: ")
    print("\n📤 Response:\n")
    print(search_with_fallback(user_query))
