#rag_app\agents\mcp_agent.py

import json
import logging
from pathlib import Path
from rag_app.generator.generator import generate_answer
from rag_app.ingestion.indexer import index_url
from rag_app.ingestion.embedder import Embedder
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("rag_app/data")
DOCUMENTS_FILE = DATA_DIR / "documents.json"
URL_INDEX_FILE = DATA_DIR / "url_index.json"

MCP_SYSTEM_PROMPT = (
    "You are an advanced fallback agent (MCP). "
    "When no contextual information is found in the knowledge base, "
    "you must still respond clearly and accurately based only on your own trained knowledge. "
    "Be concise, direct, and helpful."
)





def suggest_top_k_urls_with_llm(query: str, k: int = 3) -> list[str]:
    if not URL_INDEX_FILE.exists():
        logger.warning("⚠️ url_index.json not found.")
        return []

    with open(URL_INDEX_FILE, encoding="utf-8") as f:
        entries = json.load(f)

    prompt = f"""You are helping route user questions to the most relevant web pages.

QUESTION:
{query}

Here is a list of available pages:
"""
    for entry in entries:
        prompt += f"- {entry['title']} → {entry['url']}\n"

    prompt += f"\nReturn up to {k} matching URLs as a numbered list. If none are relevant, respond with 'NONE'."


    response = generate_answer(query="", chunks=[], system_prompt=prompt).strip()

    if response.upper() == "NONE":
        logger.info("🛑 LLM decided no URL is relevant.")
        return []

    # Extraer URLs desde la respuesta tipo:
    # 1. https://example.com/page1
    # 2. https://example.com/page2
    urls = []
    for line in response.splitlines():
        if "http" in line:
            _, url = line.split("http", 1)
            urls.append("http" + url.strip())
    logger.info(f"🤖 LLM suggested URLs: {urls}")
    return urls



def answer_with_mcp(query: str, persist: bool = True, interactive: bool = True) -> str:
    logger.info("🧠 No chunks available — checking URL suggestions before fallback.")
    suggestions = suggest_top_k_urls_with_llm(query)

    if not suggestions:
        logger.info("🛑 No related URLs found — using fallback LLM answer.")
        answer = generate_answer(query=query, chunks=[], system_prompt=MCP_SYSTEM_PROMPT)

        if persist:
            logger.info(f"💾 Persisting fallback answer to {DOCUMENTS_FILE}")
            existing = []
            if DOCUMENTS_FILE.exists():
                try:
                    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("📄 documents.json is empty or invalid, starting fresh.")
            entry = {
                "id": f"mcp__fallback__{hash(query)}",
                "text": answer,
                "source": "MCP",
                "chunk_index": 0
            }
            with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(existing + [entry], f, indent=2, ensure_ascii=False)

        return f"📘 This topic is not covered in our current knowledge base. Here's a general answer:\n\n{answer}"

    logger.info(f"🤖 Automatically selecting up to 3 URLs to index.")
    selected_urls = suggestions[:3]
    logger.info(f"🌐 Indexing selected URLs: {selected_urls}")
    for url in selected_urls:
        index_url(url)
    return "🔄 Updating knowledge base with new content. Please wait..."



    # Fallback model response
    answer = generate_answer(query=query, chunks=[], system_prompt=MCP_SYSTEM_PROMPT)

    if persist:
        logger.info(f"💾 Persisting fallback answer to {DOCUMENTS_FILE}")
        existing = []
        if DOCUMENTS_FILE.exists():
            with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)

        entry = {
            "id": f"mcp__fallback__{hash(query)}",
            "text": answer,
            "source": "MCP",
            "chunk_index": 0
        }
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing + [entry], f, indent=2, ensure_ascii=False)

    logger.info("✅ Fallback answer generated and (optionally) persisted.")
    return answer

