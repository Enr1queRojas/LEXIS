# rag_app/utils/cache.py
import hashlib
import json
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("rag_app/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_from_cache(prompt: str) -> Optional[str]:

    key = _hash_key(prompt)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("response")
    return None

def save_to_cache(prompt: str, response: str):
    key = _hash_key(prompt)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, ensure_ascii=False, indent=2)
