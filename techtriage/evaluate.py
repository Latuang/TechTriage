"""
Run: python evaluate.py
"""
import csv
from pathlib import Path
from classifier import train_from_csv
from retrieval import load_articles, Retriever

ROOT = Path(__file__).parent
rows = list(csv.DictReader((ROOT / "data/questions.csv").open()))
classifier = train_from_csv()
retriever = Retriever(load_articles())
category_hits, article_hits = 0, 0
for row in rows:
    label = classifier.predict(row["question"])[0][0]
    category_hits += label == row["category"]
    results = retriever.search(row["question"], category=None, top_k=3)
    article_hits += row["article_id"] in [r["id"] for r in results]
print("WARNING: TRAINING-SET CHECK on synthetic examples, NOT held-out evaluation")
print(f"Training-set classifier accuracy: {category_hits}/{len(rows)}")
print(f"Demo BM25 Recall@3 (same synthetic examples): {article_hits}/{len(rows)}")
