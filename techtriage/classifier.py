"""Small bag-of-words multiclass linear classifier trained with gradient descent.
Training examples are synthetic demo data, not a measured Stanford dataset.
"""
import csv
import re
from collections import Counter
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class IssueClassifier:
    def __init__(self):
        self.vocab = {}
        self.classes = []
        self.W = None
        self.b = None

    def features(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for word, count in Counter(tokenize(text)).items():
                if word in self.vocab:
                    X[i, self.vocab[word]] = count
        return X

    def fit(self, questions, labels, epochs=400, lr=0.25):
        self.classes = sorted(set(labels))
        self.vocab = {word: i for i, word in enumerate(sorted(set(
            w for text in questions for w in tokenize(text)
        )))}
        X = self.features(questions)
        y = np.array([self.classes.index(label) for label in labels])
        self.W = np.zeros((X.shape[1], len(self.classes)))
        self.b = np.zeros(len(self.classes))
        for _ in range(epochs):
            logits = X @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(y)), y] -= 1
            self.W -= lr * X.T @ p / len(y)
            self.b -= lr * p.mean(axis=0)
        return self

    def predict(self, question):
        x = self.features([question])[0]
        logits = x @ self.W + self.b
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return sorted(zip(self.classes, probs.tolist()), key=lambda item: -item[1])


def train_from_csv(path=None):
    path = Path(path) if path else Path(__file__).parent / "data/questions.csv"
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return IssueClassifier().fit([r["question"] for r in rows], [r["category"] for r in rows])
