import logging

from rag_app.ingestion.indexer import index_url
from rag_app.retriever.search import search_with_fallback
from rag_app.generator.generator import generate_answer

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
    query = "What AI tools does LexisNexis offer?"

    logger.info("🌐 Indexing URL content...")
    index_url(url)

    logger.info(f"❓ Searching with fallback for query: '{query}'")
    chunks = search_with_fallback(query, top_k=3)

    logger.info(f"🧠 Generating answer using LLM...")
    answer = generate_answer(query, chunks)

    print("\n🔎 Query:")
    print(query)
    print("\n📤 Answer:")
    print(answer)

if __name__ == "__main__":
    main()
