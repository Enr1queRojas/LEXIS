# eval_metrics.py

import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
import pandas as pd

# Datos de ejemplo (reemplaza o carga desde un archivo si lo deseas)
data = [
    {
        "query": "What is Lexis+ AI?",
        "generated": "Lexis+ AI is an advanced research tool that helps with legal analytics.",
        "reference": "Lexis+ AI is a legal research assistant that uses artificial intelligence for analytics and recommendations.",
        "chunks_retrieved": ["Lexis+ AI is an advanced research tool...", "Lexis is a legal platform..."],
        "ground_truth_in_chunks": [1, 0]
    },
    {
        "query": "What is Juris by LexisNexis?",
        "generated": "Juris is a billing software for legal firms.",
        "reference": "LexisNexis Juris is a time and billing software designed for legal professionals.",
        "chunks_retrieved": ["Juris is a tool...", "LexisNexis also has billing software..."],
        "ground_truth_in_chunks": [1, 1]
    }
]

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

results = []

for entry in data:
    precision = precision_score([1]*len(entry["ground_truth_in_chunks"]), entry["ground_truth_in_chunks"])
    recall = recall_score([1]*len(entry["ground_truth_in_chunks"]), entry["ground_truth_in_chunks"])
    f1 = f1_score([1]*len(entry["ground_truth_in_chunks"]), entry["ground_truth_in_chunks"])
    rouge = scorer.score(entry["reference"], entry["generated"])
    rougeL = rouge['rougeL'].fmeasure

    results.append({
        "Query": entry["query"],
        "ROUGE-L": round(rougeL, 4),
        "Precision@k": round(precision, 4),
        "Recall@k": round(recall, 4),
        "F1@k": round(f1, 4)
    })

# Mostrar en tabla
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
