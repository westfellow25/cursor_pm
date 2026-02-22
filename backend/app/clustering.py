from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def choose_cluster_count(n_rows: int, max_clusters: int = 8) -> int:
    if n_rows <= 3:
        return 1
    return max(2, min(max_clusters, int(np.sqrt(n_rows))))


def cluster_feedback(feedback_df: pd.DataFrame, embeddings: list[list[float]]) -> dict[int, list[int]]:
    if feedback_df.empty:
        return {}

    n_clusters = choose_cluster_count(len(feedback_df))
    matrix = np.array(embeddings)

    if n_clusters == 1:
        return {0: list(range(len(feedback_df)))}

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(idx)

    return dict(grouped)
