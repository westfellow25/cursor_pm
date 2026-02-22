from __future__ import annotations

from io import BytesIO

import pandas as pd

from app.clustering import cluster_feedback
from app.openai_client import embed_texts
from app.schemas import DiscoveryResponse, OpportunitySummary

REQUIRED_COLUMNS = {"feedback"}


def load_feedback_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["feedback"]).copy()
    df["feedback"] = df["feedback"].astype(str).str.strip()
    df = df[df["feedback"] != ""]
    return df.reset_index(drop=True)


def generate_discovery(df: pd.DataFrame) -> DiscoveryResponse:
    texts = df["feedback"].tolist()
    embeddings = embed_texts(texts)
    clusters = cluster_feedback(df, embeddings)

    opportunities: list[OpportunitySummary] = []
    for cluster_id, indexes in sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True):
        cluster_texts = [texts[i] for i in indexes]
        representative = cluster_texts[0]
        theme = representative[:120]
        opportunities.append(
            OpportunitySummary(
                cluster_id=cluster_id,
                size=len(indexes),
                theme=theme,
                representative_feedback=representative,
            )
        )

    return DiscoveryResponse(
        total_feedback_items=len(df),
        total_clusters=len(opportunities),
        opportunities=opportunities,
    )
