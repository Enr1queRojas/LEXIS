import nltk
import re
import json
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from pathlib import Path

# Setup NLTK
nltk.data.path.append("C:/Users/Usuario/nltk_data")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_tokens: int = 100, overlap: int = 1, min_tokens: int = 20) -> List[str]:
    sentences = sent_tokenize(clean_text(text))
    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = []
        token_count = 0

        for j in range(i, len(sentences)):
            tokens = sentences[j].split()
            if token_count + len(tokens) > max_tokens:
                break
            chunk_sentences.append(sentences[j])
            token_count += len(tokens)

        if token_count >= min_tokens:
            chunks.append(" ".join(chunk_sentences))

        i += max(1, len(chunk_sentences) - overlap)

    return chunks


def extract_metadata(text: str, source: str) -> Dict:
    text_lower = text.lower()
    keywords = []

    if "protégé" in text_lower:
        keywords.append("Protégé")
    if "lexis+" in text_lower or "lexis plus" in text_lower:
        keywords.append("Lexis+ AI")
    if "summarization" in text_lower:
        keywords.append("summarization")
    if "draft" in text_lower or "document" in text_lower:
        keywords.append("legal drafting")

    contains_ai_info = any(k in text_lower for k in ["ai", "artificial intelligence", "machine learning", "protégé"])

    title = "General"
    if "products/lexis-plus-ai" in source:
        title = "Lexis+ AI"
    elif "home.page" in source:
        title = "Homepage Overview"
    elif "about-us" in source:
        title = "Company Overview"

    return {
        "title": title,
        "keywords": keywords,
        "contains_ai_info": contains_ai_info
    }


def build_index(text: str, source: str, max_tokens: int = 100, overlap: int = 1) -> List[Dict]:
    chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)

    index = []

    for idx, chunk in enumerate(chunks):
        metadata = extract_metadata(chunk, source)
        index.append({
            "id": f"{source}__chunk_{idx}",
            "text": chunk,
            "source": source,
            "chunk_index": idx,
            "metadata": metadata
        })

    return index


def update_documents_json(source_url: str, text: str):
    """
    Procesa el texto y actualiza el archivo documents.json con los chunks generados.
    """
    file_path = Path("rag_app/data/documents.json")
    if not file_path.exists():
        raise FileNotFoundError("❌ Archivo documents.json no encontrado")

    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        if doc["url"] == source_url:
            try:
                chunks = build_index(text, source_url)
                doc["chunks"] = chunks
                doc["status"] = "fetched"
            except Exception as e:
                doc["status"] = "error"
                doc["error"] = str(e)
            break

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
