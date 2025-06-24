# rag_app/generator/generator.py

import logging
from typing import List, Optional
import google.generativeai as genai
from rag_app.utils.config import * 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("✅ Answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"❌ Failed to generate answer: {e}")
        return "An error occurred while generating the answer."
