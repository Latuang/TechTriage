"""BM25, optional Sentence-BERT, and simple rank-fusion hybrid retrieval."""
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from classifier import tokenize


def load_articles(path=None):
    path = Path(path) if path else Path(__file__).parent / "data/articles.json"
    return json.loads(path.read_text(encoding="utf-8"))


class Retriever:
    def __init__(self, articles, use_embeddings=False):
        self.articles = articles
        self.bm25 = BM25Okapi([tokenize(a["title"] + " " + a["text"]) for a in articles])
        self.encoder = None
        self.embeddings = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = self.encoder.encode(
                [a["title"] + " " + a["text"] for a in articles],
                normalize_embeddings=True,
            )

    def search(self, query, category=None, method="BM25", top_k=3):
        options = [i for i, a in enumerate(self.articles)
                   if category is None or a["category"] == category]
        if not options:
            return []
        bm = self.bm25.get_scores(tokenize(query))
        if self.encoder is not None and method in ("Embedding", "Hybrid"):
            q = self.encoder.encode([query], normalize_embeddings=True)[0]
            sem = self.embeddings @ q
        else:
            sem = None
        if method == "Embedding" and sem is None:
            raise ValueError("Enable embeddings to use embedding retrieval")
        if method == "Hybrid" and sem is None:
            raise ValueError("Enable embeddings to use hybrid retrieval")
        bm_order = sorted(options, key=lambda i: -bm[i])
        sem_order = sorted(options, key=lambda i: -sem[i]) if sem is not None else []
        ranks_bm = {i: rank for rank, i in enumerate(bm_order)}
        ranks_sem = {i: rank for rank, i in enumerate(sem_order)}
        results = []
        for i in options:
            # Reciprocal rank fusion avoids mixing incompatible raw score scales.
            score = (1 / (60 + ranks_bm[i])) if method != "Embedding" else 0
            if method in ("Embedding", "Hybrid"):
                score += 1 / (60 + ranks_sem[i])
            if method == "BM25":
                score = float(bm[i])
            if method == "Embedding":
                score = float(sem[i])
            results.append({**self.articles[i], "score": float(score),
                            "bm25_score": float(bm[i]),
                            "semantic_score": float(sem[i]) if sem is not None else None})
        return sorted(results, key=lambda a: -a["score"])[:top_k]
