import json
import logging
from pathlib import Path
from rag_app.generator.generator import generate_answer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ruta para guardar respuestas generadas
DATA_DIR = Path("rag_app/data")
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_FILE = DATA_DIR / "documents.json"

# Prompt oculto (puedes refinarlo más si quieres que sea multi-turn)
MCP_SYSTEM_PROMPT = (
    "You are an advanced fallback agent (MCP). "
    "When no contextual information is found in the knowledge base, "
    "you must still respond clearly and accurately based only on your own trained knowledge. "
    "Be concise, direct, and helpful."
)

def answer_with_mcp(query: str, persist: bool = True) -> str:
    """
    Answer user query directly using LLM (e.g., Gemini) without chunks.
    Optionally save response to local store to enhance future context.
    """
    logger.info("🧠 No chunks available — generating fallback answer using MCP logic.")
    
    # Prompt directo con query, sin contexto
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
