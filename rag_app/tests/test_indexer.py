import unittest
import os
import json
from pathlib import Path
from rag_app.ingestion import indexer

DOCUMENTS_PATH = Path("rag_app/data/documents.json")

class TestIndexer(unittest.TestCase):

    def setUp(self):
        # Asegurarse de un entorno limpio
        if DOCUMENTS_PATH.exists():
            DOCUMENTS_PATH.unlink()
        indexer.reset_index()

    def test_index_url_creates_documents_and_persists(self):
        url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
        doc_ids = indexer.index_url(url, max_tokens=100, overlap=20, persist=True)

        # Verificar que se agregaron documentos
        self.assertGreater(len(doc_ids), 0)

        # Verificar que el archivo de respaldo se creó
        self.assertTrue(DOCUMENTS_PATH.exists())

        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)

        # Verificar contenido
        self.assertGreater(len(docs), 0)
        self.assertIn("id", docs[0])
        self.assertIn("text", docs[0])
        self.assertIn("source", docs[0])

    def test_reset_index_deletes_documents(self):
        url = "https://www.lexisnexis.com/en-us/about-us/innovation.page"
        indexer.index_url(url, persist=True)

        indexer.reset_index()

        # Index vacío tras reset
        results = indexer.query_index("AI at LexisNexis", top_k=3)
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()
