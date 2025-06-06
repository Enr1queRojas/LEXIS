import nltk
import re
from typing import List
from nltk.tokenize import sent_tokenize

# Download NLTK resources if not already present
nltk.data.path.append("C:/Users/Usuario/nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str) -> str:
    """
    Pre-clean text to remove extra whitespace and normalize spacing.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = 100, overlap: int = 20) -> List[str]:
    """
    Splits text into chunks based on sentence boundaries, with optional overlap.

    Args:
        text: Input text to chunk.
        max_tokens: Approximate maximum number of tokens per chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    sentences = sent_tokenize(clean_text(text))
    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        sentence_tokens = sentence.split()
        if token_count + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = sentence_tokens[-overlap:] if overlap > 0 else []
            token_count = len(current_chunk)
        else:
            current_chunk.extend(sentence_tokens)
            token_count += len(sentence_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
