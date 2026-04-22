"""Intelligence service — the core analysis engine.

Orchestrates clustering, scoring, theme extraction, and produces the
structured analysis output that drives the entire platform."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from pulse.ml.clustering import (
    cluster_embeddings,
    extract_keywords,
    extract_theme,
    find_optimal_k,
)
from pulse.ml.sentiment import analyze_urgency
from pulse.models import (
    AnalysisRun,
    Cluster,
    ClusterMember,
    FeedbackItem,
)

logger = logging.getLogger(__name__)

# Severity signal terms with weights
_SEVERITY_TERMS = {
    "blocked": 3.0, "blocker": 3.0, "crash": 2.8, "data loss": 3.0,
    "broken": 2.5, "outage": 2.8, "security": 2.5, "vulnerability": 3.0,
    "can't": 2.0, "unable": 2.0, "fails": 2.2, "error": 1.8,
    "slow": 1.5, "timeout": 2.0, "critical": 2.5, "urgent": 2.2,
    "not working": 2.5, "doesn't work": 2.5, "losing": 2.0, "lost": 2.0,
}


def _cluster_severity(texts: list[str]) -> float:
    """Score cluster severity based on urgency term prevalence."""
    if not texts:
        return 0.0
    total = 0.0
    for text in texts:
        text_lower = text.lower()
        for term, weight in _SEVERITY_TERMS.items():
            if term in text_lower:
                total += weight
    return min(1.0, total / (len(texts) * 2.0))


def _opportunity_score(
    frequency: float,
    severity: float,
    sentiment: float,
    prevalence: float,
    cluster_size: int = 0,
) -> float:
    """Compute opportunity score (0-10).

    Formula weights:
    - Frequency (35%): share of feedback in this cluster
    - Size boost: log-scaled bump so a 10-item cluster clearly outranks a
      1-item cluster even if the smaller one has very high severity
    - Severity (25%): how urgent/critical the feedback is
    - Negative sentiment (25%): how unhappy users are
    - Prevalence (15%): how many distinct segments report this
    """
    neg_sentiment = max(0.0, -sentiment)
    # Log-scaled size bonus: 1 item → 0.0, 3 → 0.48, 10 → 1.0 (capped).
    import math

    size_bonus = min(1.0, math.log1p(max(0, cluster_size - 1)) / math.log(10))

    raw = (
        0.35 * frequency
        + 0.25 * severity
        + 0.25 * neg_sentiment
        + 0.15 * prevalence
    )
    scaled = raw * 10.0
    # Size bonus adds up to +1.5 points on top of the weighted formula, which is
    # enough to push a 10-item negative cluster above a 1-item crisis-ticket.
    scaled += 1.5 * size_bonus
    return round(min(10.0, scaled), 1)


def run_analysis(
    db: Session,
    org_id: str,
    n_clusters: int | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> AnalysisRun:
    """Run a full analysis on an organisation's feedback.

    1. Load feedback items (with optional date range)
    2. Cluster by embeddings
    3. Score and rank clusters
    4. Generate themes, keywords, summaries
    5. Persist results

    Returns the completed AnalysisRun with all clusters populated.
    """
    run = AnalysisRun(org_id=org_id, status="running", started_at=datetime.now(timezone.utc))
    db.add(run)
    db.flush()

    # Load feedback
    query = db.query(FeedbackItem).filter(FeedbackItem.org_id == org_id)
    if date_from:
        query = query.filter(FeedbackItem.created_at >= date_from)
    if date_to:
        query = query.filter(FeedbackItem.created_at <= date_to)

    feedback_items = query.order_by(FeedbackItem.created_at.desc()).all()
    run.feedback_count = len(feedback_items)

    if len(feedback_items) < 2:
        run.status = "completed"
        run.cluster_count = 0
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        return run

    # Build embedding matrix
    embeddings = []
    valid_items = []
    for item in feedback_items:
        if item.embedding:
            embeddings.append(item.embedding)
            valid_items.append(item)
    if not embeddings:
        run.status = "failed"
        run.error_message = "No embeddings available. Ensure feedback items have been enriched."
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        return run

    embed_matrix = np.array(embeddings, dtype=np.float64)

    # Cluster
    labels, similarities, centroids = cluster_embeddings(embed_matrix, n_clusters=n_clusters)

    # Group items by cluster
    cluster_groups: dict[int, list[tuple[FeedbackItem, float]]] = {}
    for i, (item, label, sim) in enumerate(zip(valid_items, labels, similarities)):
        label_int = int(label)
        if label_int not in cluster_groups:
            cluster_groups[label_int] = []
        cluster_groups[label_int].append((item, float(sim)))

    # Build cluster objects
    total_items = len(valid_items)
    all_segments = set()
    for item in valid_items:
        if item.author_segment:
            all_segments.add(item.author_segment)
    total_segments = max(1, len(all_segments))

    clusters_created: list[Cluster] = []
    for label_int, members in sorted(cluster_groups.items()):
        if label_int == -1:
            continue  # Skip outliers

        texts = [item.text for item, _ in members]
        sentiments = [item.sentiment or 0.0 for item, _ in members]
        segments_in_cluster = {item.author_segment for item, _ in members if item.author_segment}

        frequency = len(members) / total_items
        severity = _cluster_severity(texts)
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        prevalence = len(segments_in_cluster) / total_segments if total_segments > 0 else 0.0

        opp_score = _opportunity_score(
            frequency, severity, avg_sentiment, prevalence, cluster_size=len(members)
        )
        theme = extract_theme(texts)
        keywords = extract_keywords(texts)

        # LLM enrichment (if available)
        from pulse.ml.llm import generate_cluster_label, generate_cluster_summary
        llm_label = generate_cluster_label(keywords, texts[:8])
        if llm_label:
            theme = llm_label

        summary_text = f"{len(members)} feedback items about {theme.lower()}"
        llm_summary = generate_cluster_summary(
            theme, len(members), total_items, keywords, texts[:6], float(avg_sentiment),
        )
        if llm_summary:
            summary_text = llm_summary

        centroid_vec = centroids[label_int].tolist() if label_int < len(centroids) else None

        cluster = Cluster(
            run_id=run.id,
            org_id=org_id,
            label=theme,
            theme=theme,
            summary=summary_text,
            size=len(members),
            opportunity_score=opp_score,
            severity_score=round(severity, 3),
            frequency_score=round(frequency, 3),
            sentiment_avg=round(float(avg_sentiment), 3),
            centroid=centroid_vec,
            top_keywords=keywords,
            trend_direction="stable",
        )
        db.add(cluster)
        db.flush()

        # Add cluster members
        for item, sim in members:
            db.add(ClusterMember(
                cluster_id=cluster.id,
                feedback_id=item.id,
                similarity=round(sim, 4),
            ))

        clusters_created.append(cluster)

    # Sort by opportunity score
    clusters_created.sort(key=lambda c: c.opportunity_score, reverse=True)

    run.status = "completed"
    run.cluster_count = len(clusters_created)
    run.completed_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(
        "Analysis complete: %d items → %d clusters (org=%s, run=%s)",
        len(valid_items), len(clusters_created), org_id, run.id,
    )
    return run
