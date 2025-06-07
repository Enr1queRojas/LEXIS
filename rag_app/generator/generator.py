# rag_app/generator/generator.py

import os
import logging
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Cargar API Key desde entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("❌ Environment variable 'GEMINI_API_KEY' is not set.")

# Configurar el cliente de Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")


def generate_answer(query: str, chunks: List[str], system_prompt: Optional[str] = None) -> str:
    logger.info(f"🤖 Generating answer with Gemini for query: '{query}'")

    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("✅ Answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"❌ Failed to generate answer: {e}")
        return "An error occurred while generating the answer."
