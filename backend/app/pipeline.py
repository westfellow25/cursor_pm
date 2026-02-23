from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import pandas as pd

from .clustering import cluster_feedback
from .openai_client import embed_texts
from .schemas import AnalyzeResponse, DiscoveryResponse, OpportunitySummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from e2e_demo import (  # noqa: E402
    EVIDENCE_KEYWORD_GATE,
    OFF_THEME_TERMS,
    _select_supporting_evidence,
    _tokenize,
    analyze_feedback,
)
from generate_artifacts import build_artifact_content  # noqa: E402

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


def _summarize_clusters(ranked_clusters: list[dict[str, object]]) -> list[OpportunitySummary]:
    return [
        OpportunitySummary(
            cluster_id=int(cluster["cluster_id"]),
            size=int(cluster["frequency"]),
            theme=str(cluster["theme_label"]),
            representative_feedback=str(cluster["example_signal"]),
        )
        for cluster in ranked_clusters
    ]


def _build_evidence(analysis: dict[str, object]) -> list[str]:
    top = analysis["top"]
    if not top:
        return []

    records = analysis["records"]
    scored_pairs: list[tuple[float, str, str]] = []
    theme_tokens = set(_tokenize(top["theme_label"]))
    for record in records:
        quote = record.text
        quote_tokens = set(_tokenize(quote))
        if not (EVIDENCE_KEYWORD_GATE & quote_tokens):
            continue
        if OFF_THEME_TERMS & quote_tokens:
            continue
        overlap = len(theme_tokens & quote_tokens) / max(1, len(theme_tokens))
        scored_pairs.append((overlap, record.feedback_id, quote))

    selected = _select_supporting_evidence(scored_pairs, evidence_count=min(5, len(records)))
    if selected:
        return [quote for _, quote in selected]
    return [record.text for record in records[:3]]


def analyze_feedback_csv(file_bytes: bytes, n_clusters: int = 3) -> AnalyzeResponse:
    with NamedTemporaryFile(mode="wb", suffix=".csv", delete=True) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        analysis = analyze_feedback(csv_path=temp_file.name, n_clusters=n_clusters)

    prd_text, jira_tickets_text = build_artifact_content(analysis)
    ranked_clusters = analysis["ranked_clusters"]
    top_opportunities = ranked_clusters[:3]

    return AnalyzeResponse(
        clusters_summary=_summarize_clusters(ranked_clusters),
        top_opportunities=top_opportunities,
        recommended_action=str(analysis["proposed_solution"]),
        evidence=_build_evidence(analysis),
        prd_text=prd_text,
        jira_tickets_text=jira_tickets_text,
    )
