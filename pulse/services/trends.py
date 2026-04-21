"""Temporal intelligence service — the core of the historical data moat.

Computes time-series metrics, detects trends, and generates snapshots
that power the trend dashboard and anomaly detection."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from pulse.ml.anomaly import (
    detect_emerging_topics,
    detect_segment_divergence,
    detect_sentiment_shift,
    detect_volume_spike,
    Anomaly,
)
from pulse.models import FeedbackItem, Insight, TrendSnapshot

logger = logging.getLogger(__name__)


def compute_trend_snapshot(
    db: Session,
    org_id: str,
    period_start: datetime,
    period_end: datetime,
    granularity: str = "week",
) -> TrendSnapshot:
    """Compute a trend snapshot for a given time period."""
    items = (
        db.query(FeedbackItem)
        .filter(
            FeedbackItem.org_id == org_id,
            FeedbackItem.created_at >= period_start,
            FeedbackItem.created_at < period_end,
        )
        .all()
    )

    # Category distribution
    cat_dist: dict[str, int] = defaultdict(int)
    for item in items:
        cat_dist[item.category or "uncategorized"] += 1

    # Segment breakdown
    segment_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "sentiments": []})
    for item in items:
        seg = item.author_segment or "unknown"
        segment_data[seg]["count"] += 1
        if item.sentiment is not None:
            segment_data[seg]["sentiments"].append(item.sentiment)

    segment_breakdown = {}
    for seg, data in segment_data.items():
        sents = data["sentiments"]
        segment_breakdown[seg] = {
            "count": data["count"],
            "sentiment": round(np.mean(sents), 3) if sents else 0.0,
        }

    # Average sentiment
    sentiments = [i.sentiment for i in items if i.sentiment is not None]
    avg_sentiment = round(float(np.mean(sentiments)), 3) if sentiments else None

    snapshot = TrendSnapshot(
        org_id=org_id,
        period_start=period_start,
        period_end=period_end,
        granularity=granularity,
        total_feedback=len(items),
        avg_sentiment=avg_sentiment,
        category_distribution=dict(cat_dist),
        segment_breakdown=segment_breakdown,
        top_themes=[],
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def compute_weekly_snapshots(
    db: Session,
    org_id: str,
    weeks: int = 26,
) -> list[TrendSnapshot]:
    """Generate weekly snapshots for the past N weeks."""
    now = datetime.now(timezone.utc)
    snapshots = []
    for i in range(weeks):
        end = now - timedelta(weeks=i)
        start = end - timedelta(weeks=1)
        # Check if snapshot already exists
        existing = (
            db.query(TrendSnapshot)
            .filter(
                TrendSnapshot.org_id == org_id,
                TrendSnapshot.period_start == start,
                TrendSnapshot.granularity == "week",
            )
            .first()
        )
        if existing:
            snapshots.append(existing)
        else:
            snapshot = compute_trend_snapshot(db, org_id, start, end, "week")
            snapshots.append(snapshot)
    return list(reversed(snapshots))  # Chronological order


def detect_anomalies(
    db: Session,
    org_id: str,
) -> list[Anomaly]:
    """Run anomaly detection on recent data and store as insights."""
    snapshots = (
        db.query(TrendSnapshot)
        .filter(TrendSnapshot.org_id == org_id, TrendSnapshot.granularity == "week")
        .order_by(TrendSnapshot.period_start.asc())
        .all()
    )

    if len(snapshots) < 4:
        return []

    anomalies: list[Anomaly] = []

    # Volume spike detection
    daily_counts = [s.total_feedback for s in snapshots]
    vol_anomaly = detect_volume_spike(daily_counts)
    if vol_anomaly:
        anomalies.append(vol_anomaly)

    # Sentiment shift detection
    sentiments = [s.avg_sentiment for s in snapshots if s.avg_sentiment is not None]
    if len(sentiments) >= 4:
        sent_anomaly = detect_sentiment_shift(sentiments)
        if sent_anomaly:
            anomalies.append(sent_anomaly)

    # Emerging topics
    if len(snapshots) >= 2:
        current_cats = snapshots[-1].category_distribution or {}
        # Average of previous snapshots as baseline
        historical_cats: dict[str, int] = defaultdict(int)
        for s in snapshots[:-1]:
            for cat, count in (s.category_distribution or {}).items():
                historical_cats[cat] += count
        n_prev = len(snapshots) - 1
        historical_avg = {cat: count // n_prev for cat, count in historical_cats.items()}
        topic_anomalies = detect_emerging_topics(current_cats, historical_avg)
        anomalies.extend(topic_anomalies)

    # Segment divergence
    if snapshots[-1].segment_breakdown:
        segment_sentiments = {
            seg: data.get("sentiment", 0.0)
            for seg, data in snapshots[-1].segment_breakdown.items()
        }
        overall = snapshots[-1].avg_sentiment or 0.0
        seg_anomalies = detect_segment_divergence(segment_sentiments, overall)
        anomalies.extend(seg_anomalies)

    # Store significant anomalies as insights
    for anomaly in anomalies:
        if anomaly.score >= 0.3:
            insight = Insight(
                org_id=org_id,
                type=anomaly.type,
                title=anomaly.title,
                description=anomaly.description,
                severity=anomaly.severity,
                data=anomaly.data,
            )
            db.add(insight)
    db.commit()

    return anomalies


def get_trend_data(
    db: Session,
    org_id: str,
    weeks: int = 12,
) -> dict:
    """Get formatted trend data for the frontend dashboard."""
    snapshots = (
        db.query(TrendSnapshot)
        .filter(TrendSnapshot.org_id == org_id, TrendSnapshot.granularity == "week")
        .order_by(TrendSnapshot.period_start.asc())
        .limit(weeks)
        .all()
    )

    return {
        "volume": [
            {
                "period": s.period_start.isoformat(),
                "count": s.total_feedback,
            }
            for s in snapshots
        ],
        "sentiment": [
            {
                "period": s.period_start.isoformat(),
                "value": s.avg_sentiment,
            }
            for s in snapshots
            if s.avg_sentiment is not None
        ],
        "categories": [
            {
                "period": s.period_start.isoformat(),
                **{cat: count for cat, count in (s.category_distribution or {}).items()},
            }
            for s in snapshots
        ],
        "segments": [
            {
                "period": s.period_start.isoformat(),
                **(s.segment_breakdown or {}),
            }
            for s in snapshots
        ],
    }
