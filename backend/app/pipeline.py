from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import pandas as pd

from .schemas import AnalyzeResponse, OpportunitySummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from e2e_demo import (  # noqa: E402
    _centroid,
    _cosine_similarity,
    _select_supporting_evidence,
    analyze_feedback,
)
from feedback_ingestion import HashingEmbedder  # noqa: E402
from generate_artifacts import build_artifact_content  # noqa: E402

TEXT_COLUMN_CANDIDATES = ("text", "feedback")


def load_feedback_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes))
    text_column = next((col for col in TEXT_COLUMN_CANDIDATES if col in df.columns), None)
    if not text_column:
        raise ValueError(
            "CSV must contain a 'text' column (or legacy 'feedback' column) with user feedback strings."
        )
    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    if text_column != "text":
        df = df.rename(columns={text_column: "text"})
    return df.reset_index(drop=True)


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

    cluster_texts: list[str] = list(top.get("texts") or [])
    cluster_ids: list[str] = list(top.get("ids") or [])
    if not cluster_texts:
        return []

    embedder = HashingEmbedder(dimensions=256)
    vectors = embedder.embed(cluster_texts)
    centroid_vec = _centroid(vectors)
    scored_pairs = [
        (_cosine_similarity(centroid_vec, vector), feedback_id, quote)
        for feedback_id, quote, vector in zip(cluster_ids, cluster_texts, vectors)
    ]
    evidence_count = min(5, len(scored_pairs))
    selected = _select_supporting_evidence(scored_pairs, evidence_count=evidence_count)
    return [quote for _, quote in selected]


def analyze_feedback_csv(
    file_bytes: bytes,
    n_clusters: int = 3,
    run_id: str = "",
) -> AnalyzeResponse:
    # Normalize the incoming CSV to the canonical columns expected downstream
    # (text, feedback_id, source), regardless of whether the caller passes the
    # new `text` schema or the legacy `feedback` schema.
    df = load_feedback_csv(file_bytes)
    if "feedback_id" not in df.columns:
        df["feedback_id"] = [f"f{i:03d}" for i in range(1, len(df) + 1)]
    if "source" not in df.columns:
        df["source"] = "unknown"
    normalized_csv = df[["feedback_id", "text", "source"]].to_csv(index=False).encode("utf-8")

    with NamedTemporaryFile(mode="wb", suffix=".csv", delete=True) as temp_file:
        temp_file.write(normalized_csv)
        temp_file.flush()
        analysis = analyze_feedback(csv_path=temp_file.name, n_clusters=n_clusters)

    prd_text, jira_tickets_text = build_artifact_content(analysis)
    ranked_clusters = analysis["ranked_clusters"]
    top_opportunities = ranked_clusters[:3]

    return AnalyzeResponse(
        run_id=run_id,
        clusters_summary=_summarize_clusters(ranked_clusters),
        top_opportunities=top_opportunities,
        recommended_action=str(analysis["proposed_solution"]),
        evidence=_build_evidence(analysis),
        prd_text=prd_text,
        jira_tickets_text=jira_tickets_text,
    )
