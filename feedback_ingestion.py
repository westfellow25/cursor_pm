"""Feedback ingestion pipeline with CSV parsing, embedding generation, and simple clustering."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class FeedbackRecord:
    """Represents one feedback row from the CSV source."""

    feedback_id: str
    text: str
    source: str


@dataclass
class ClusterResult:
    """A clustering result for one feedback record."""

    feedback_id: str
    cluster_id: int


class CSVFeedbackParser:
    """Loads feedback records from a CSV file."""

    REQUIRED_FIELDS = {"feedback_id", "text", "source"}

    def parse(self, csv_path: str | Path) -> List[FeedbackRecord]:
        csv_path = Path(csv_path)
        records: List[FeedbackRecord] = []

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = self.REQUIRED_FIELDS - fieldnames
            if missing:
                missing_cols = ", ".join(sorted(missing))
                raise ValueError(f"Missing required CSV fields: {missing_cols}")

            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                records.append(
                    FeedbackRecord(
                        feedback_id=(row.get("feedback_id") or "").strip(),
                        text=text,
                        source=(row.get("source") or "unknown").strip() or "unknown",
                    )
                )

        return records


class HashingEmbedder:
    """Creates deterministic dense embeddings using token hashing and TF-IDF style weighting."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("Embedding dimension should be at least 8")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        tokenized = [self._tokenize(text) for text in texts]
        idf = self._compute_idf(tokenized)
        return [self._vectorize(tokens, idf) for tokens in tokenized]

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _compute_idf(self, tokenized_texts: Sequence[Sequence[str]]) -> dict[str, float]:
        doc_count = len(tokenized_texts)
        doc_freq: Counter[str] = Counter()

        for tokens in tokenized_texts:
            doc_freq.update(set(tokens))

        return {
            token: math.log((1 + doc_count) / (1 + freq)) + 1
            for token, freq in doc_freq.items()
        }

    def _vectorize(self, tokens: Sequence[str], idf: dict[str, float]) -> List[float]:
        vector = [0.0] * self.dimensions
        if not tokens:
            return vector

        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            idx = hash(token) % self.dimensions
            tf = count / len(tokens)
            vector[idx] += tf * idf.get(token, 1.0)

        norm = math.sqrt(sum(val * val for val in vector))
        if norm == 0:
            return vector
        return [val / norm for val in vector]


class KMeansClustering:
    """Very small K-Means implementation for embedding vectors."""

    def __init__(self, n_clusters: int = 3, max_iters: int = 30, seed: int = 7) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.seed = seed

    def fit_predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
        if not vectors:
            return []

        n_clusters = min(self.n_clusters, len(vectors))
        rng = Random(self.seed)
        indices = list(range(len(vectors)))
        rng.shuffle(indices)
        centroids = [list(vectors[idx]) for idx in indices[:n_clusters]]

        assignments = [0] * len(vectors)
        for _ in range(self.max_iters):
            updated = False

            for i, vector in enumerate(vectors):
                best_cluster = min(
                    range(n_clusters),
                    key=lambda cluster: self._distance(vector, centroids[cluster]),
                )
                if assignments[i] != best_cluster:
                    assignments[i] = best_cluster
                    updated = True

            centroids = self._recompute_centroids(vectors, assignments, n_clusters)
            if not updated:
                break

        return assignments

    def _recompute_centroids(
        self,
        vectors: Sequence[Sequence[float]],
        assignments: Sequence[int],
        n_clusters: int,
    ) -> List[List[float]]:
        dim = len(vectors[0])
        sums = [[0.0] * dim for _ in range(n_clusters)]
        counts = [0] * n_clusters

        for vector, cluster in zip(vectors, assignments):
            counts[cluster] += 1
            for i, value in enumerate(vector):
                sums[cluster][i] += value

        centroids: List[List[float]] = []
        for cluster in range(n_clusters):
            if counts[cluster] == 0:
                centroids.append([0.0] * dim)
            else:
                centroids.append([value / counts[cluster] for value in sums[cluster]])
        return centroids

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right))


def run_pipeline(csv_path: str | Path, n_clusters: int = 3) -> tuple[List[FeedbackRecord], List[ClusterResult]]:
    parser = CSVFeedbackParser()
    records = parser.parse(csv_path)

    embedder = HashingEmbedder(dimensions=128)
    vectors = embedder.embed([record.text for record in records])

    clusterer = KMeansClustering(n_clusters=n_clusters)
    assignments = clusterer.fit_predict(vectors)

    results = [
        ClusterResult(feedback_id=record.feedback_id, cluster_id=cluster_id)
        for record, cluster_id in zip(records, assignments)
    ]
    return records, results


def _format_cluster_summary(records: Iterable[FeedbackRecord], clusters: Iterable[ClusterResult]) -> str:
    rec_by_id = {record.feedback_id: record for record in records}
    grouped: dict[int, List[str]] = {}

    for cluster in clusters:
        grouped.setdefault(cluster.cluster_id, []).append(cluster.feedback_id)

    lines: List[str] = []
    for cluster_id in sorted(grouped):
        lines.append(f"Cluster {cluster_id}:")
        for feedback_id in grouped[cluster_id]:
            text = rec_by_id[feedback_id].text
            lines.append(f"  - {feedback_id}: {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    sample_path = Path("example_data/feedback.csv")
    parsed_records, clustered = run_pipeline(sample_path, n_clusters=3)
    print(_format_cluster_summary(parsed_records, clustered))
