from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def choose_cluster_count(n_rows: int, max_clusters: int = 12) -> int:
    if n_rows <= 3:
        return 1
    return max(3, min(max_clusters, int(np.ceil(np.sqrt(n_rows) * 1.5))))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cluster_feedback(
    feedback_df: pd.DataFrame,
    embeddings: list[list[float]],
    similarity_threshold: float = 0.62,
) -> dict[int, list[int]]:
    if feedback_df.empty:
        return {}

    n_clusters = choose_cluster_count(len(feedback_df))
    matrix = _normalize_rows(np.array(embeddings, dtype=float))

    if n_clusters == 1:
        return {0: list(range(len(feedback_df)))}

    model = KMeans(n_clusters=n_clusters, random_state=23, n_init=10)
    labels = model.fit_predict(matrix)
    centroids = _normalize_rows(model.cluster_centers_)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        similarity = float(np.dot(matrix[idx], centroids[int(label)]))
        target_cluster = int(label) if similarity >= similarity_threshold else -1
        grouped[target_cluster].append(idx)

    return dict(grouped)
