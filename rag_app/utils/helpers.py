# rag_app/utils/helpers.py

import json
from typing import List
from rag_app.ingestion.indexer import index_url  # o desde donde definas index_url

def load_url_index() -> List[dict]:
    with open("rag_app/data/url_index.json", "r", encoding="utf-8") as f:
        return json.load(f)

def search_relevant_urls(query: str, top_k: int = 3) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    index = load_url_index()
    documents = [entry["title"] for entry in index]
    urls = [entry["url"] for entry in index]

    vectorizer = TfidfVectorizer().fit(documents + [query])
    vectors = vectorizer.transform(documents + [query])

    scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]

    return [urls[i] for i in top_indices]
