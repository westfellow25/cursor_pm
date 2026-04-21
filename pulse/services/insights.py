"""Insight generation service — transforms raw analysis into actionable intelligence.

Insights are the "aha moments" that make PMs love the product. They combine
quantitative signals with qualitative context to drive product decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from pulse.models import (
    AnalysisRun,
    Cluster,
    ClusterMember,
    FeedbackItem,
    Insight,
)

logger = logging.getLogger(__name__)


def generate_run_insights(db: Session, run: AnalysisRun) -> list[Insight]:
    """Generate insights from a completed analysis run."""
    if run.status != "completed":
        return []

    org_id = run.org_id
    clusters = (
        db.query(Cluster)
        .filter(Cluster.run_id == run.id)
        .order_by(desc(Cluster.opportunity_score))
        .all()
    )

    if not clusters:
        return []

    insights: list[Insight] = []

    # 1. Top opportunity insight
    top = clusters[0]
    insights.append(Insight(
        org_id=org_id,
        type="opportunity_found",
        title=f"Top opportunity: {top.label}",
        description=(
            f"{top.size} feedback items ({top.frequency_score * 100:.0f}% of total) "
            f"point to issues with {top.label.lower()}. "
            f"Opportunity score: {top.opportunity_score}/10. "
            f"Average sentiment: {top.sentiment_avg:.2f}."
        ),
        severity="warning" if top.opportunity_score >= 7 else "info",
        data={
            "cluster_id": top.id,
            "theme": top.label,
            "size": top.size,
            "score": top.opportunity_score,
            "keywords": top.top_keywords,
        },
    ))

    # 2. Critical severity clusters
    critical_clusters = [c for c in clusters if c.severity_score >= 0.6]
    if critical_clusters:
        themes = [c.label for c in critical_clusters[:3]]
        insights.append(Insight(
            org_id=org_id,
            type="churn_signal",
            title=f"{len(critical_clusters)} high-severity themes detected",
            description=(
                f"The following themes have high severity scores indicating potential churn risk: "
                f"{', '.join(themes)}. These contain urgent language like 'blocked', 'broken', "
                f"'critical' that suggests immediate attention is needed."
            ),
            severity="critical",
            data={
                "clusters": [
                    {"id": c.id, "theme": c.label, "severity": c.severity_score}
                    for c in critical_clusters[:5]
                ],
            },
        ))

    # 3. Sentiment distribution insight
    positive_clusters = [c for c in clusters if (c.sentiment_avg or 0) > 0.2]
    negative_clusters = [c for c in clusters if (c.sentiment_avg or 0) < -0.2]
    if positive_clusters and negative_clusters:
        insights.append(Insight(
            org_id=org_id,
            type="sentiment_shift",
            title="Mixed sentiment across themes",
            description=(
                f"{len(positive_clusters)} themes have positive sentiment "
                f"(e.g., {positive_clusters[0].label}) while "
                f"{len(negative_clusters)} themes are negative "
                f"(e.g., {negative_clusters[0].label}). "
                f"Consider doubling down on what works while addressing pain points."
            ),
            severity="info",
            data={
                "positive": [{"theme": c.label, "sentiment": c.sentiment_avg} for c in positive_clusters[:3]],
                "negative": [{"theme": c.label, "sentiment": c.sentiment_avg} for c in negative_clusters[:3]],
            },
        ))

    # 4. Concentration risk
    if clusters and clusters[0].frequency_score >= 0.3:
        insights.append(Insight(
            org_id=org_id,
            type="trend_spike",
            title=f"{clusters[0].label} dominates feedback ({clusters[0].frequency_score * 100:.0f}%)",
            description=(
                f"A single theme accounts for {clusters[0].frequency_score * 100:.0f}% of all feedback. "
                f"This concentration suggests a widespread issue affecting a large portion of users."
            ),
            severity="warning",
            data={"cluster_id": clusters[0].id, "theme": clusters[0].label, "pct": clusters[0].frequency_score},
        ))

    # 5. Quick wins (high opportunity, low severity = easy to address)
    quick_wins = [c for c in clusters if c.opportunity_score >= 5 and c.severity_score < 0.3]
    if quick_wins:
        insights.append(Insight(
            org_id=org_id,
            type="opportunity_found",
            title=f"{len(quick_wins)} potential quick wins identified",
            description=(
                f"These themes have high opportunity scores but low severity, suggesting "
                f"they could be addressed with moderate effort: "
                f"{', '.join(c.label for c in quick_wins[:3])}."
            ),
            severity="positive",
            data={
                "quick_wins": [
                    {"theme": c.label, "score": c.opportunity_score, "size": c.size}
                    for c in quick_wins[:5]
                ],
            },
        ))

    db.add_all(insights)
    db.commit()
    for i in insights:
        db.refresh(i)

    logger.info("Generated %d insights for run %s", len(insights), run.id)
    return insights


def get_insights(
    db: Session,
    org_id: str,
    limit: int = 20,
    type_filter: str | None = None,
    unread_only: bool = False,
) -> list[Insight]:
    """Retrieve insights for an organisation."""
    query = db.query(Insight).filter(Insight.org_id == org_id)
    if type_filter:
        query = query.filter(Insight.type == type_filter)
    if unread_only:
        query = query.filter(Insight.is_read == False)
    return query.order_by(desc(Insight.created_at)).limit(limit).all()


def mark_insight_read(db: Session, insight_id: str) -> None:
    """Mark an insight as read."""
    insight = db.get(Insight, insight_id)
    if insight:
        insight.is_read = True
        db.commit()
