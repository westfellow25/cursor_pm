"""Anomaly detection for feedback streams.

Detects spikes in volume, sentiment shifts, emerging topics, and
segment-specific divergences. This is a core piece of the temporal
intelligence moat — the system gets smarter with more historical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    type: str  # volume_spike | sentiment_shift | emerging_topic | segment_divergence
    title: str
    description: str
    severity: str  # critical | warning | info
    score: float  # 0-1, how anomalous
    data: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def detect_volume_spike(
    daily_counts: list[int],
    sensitivity: float = 2.0,
) -> Anomaly | None:
    """Detect if the latest day's feedback volume is anomalous."""
    if len(daily_counts) < 7:
        return None

    recent = daily_counts[-1]
    baseline = daily_counts[:-1]
    mean = np.mean(baseline)
    std = np.std(baseline) or 1.0
    z_score = (recent - mean) / std

    if z_score > sensitivity:
        pct_increase = ((recent - mean) / mean * 100) if mean > 0 else 0
        severity = "critical" if z_score > 3.0 else "warning"
        return Anomaly(
            type="volume_spike",
            title=f"Feedback volume spike: {pct_increase:.0f}% above normal",
            description=(
                f"Received {recent} feedback items today vs. {mean:.0f} daily average. "
                f"This is {z_score:.1f} standard deviations above the baseline."
            ),
            severity=severity,
            score=min(1.0, z_score / 5.0),
            data={"count": recent, "mean": round(mean, 1), "z_score": round(z_score, 2)},
        )
    return None


def detect_sentiment_shift(
    daily_sentiments: list[float],
    sensitivity: float = 2.0,
) -> Anomaly | None:
    """Detect sudden sentiment drops or improvements."""
    if len(daily_sentiments) < 7:
        return None

    recent = daily_sentiments[-1]
    baseline = daily_sentiments[:-1]
    mean = np.mean(baseline)
    std = np.std(baseline) or 0.1
    z_score = (recent - mean) / std

    if abs(z_score) > sensitivity:
        direction = "drop" if z_score < 0 else "improvement"
        severity = "critical" if z_score < -3.0 else "warning" if z_score < -2.0 else "info"
        return Anomaly(
            type="sentiment_shift",
            title=f"Sentiment {direction}: {recent:.2f} vs {mean:.2f} average",
            description=(
                f"Average sentiment {'dropped' if z_score < 0 else 'improved'} to {recent:.2f} "
                f"from a {mean:.2f} baseline ({abs(z_score):.1f}σ deviation)."
            ),
            severity=severity,
            score=min(1.0, abs(z_score) / 5.0),
            data={"current": round(recent, 3), "mean": round(mean, 3), "z_score": round(z_score, 2)},
        )
    return None


def detect_emerging_topics(
    current_categories: dict[str, int],
    historical_categories: dict[str, int],
    threshold: float = 2.0,
) -> list[Anomaly]:
    """Detect categories growing faster than expected."""
    anomalies: list[Anomaly] = []
    total_current = sum(current_categories.values()) or 1
    total_historical = sum(historical_categories.values()) or 1

    for cat, count in current_categories.items():
        current_pct = count / total_current
        historical_count = historical_categories.get(cat, 0)
        historical_pct = historical_count / total_historical if total_historical > 0 else 0

        if historical_pct > 0:
            growth = current_pct / historical_pct
        elif current_pct > 0.05:  # New category with >5% share
            growth = 10.0
        else:
            continue

        if growth > threshold and count >= 3:
            anomalies.append(Anomaly(
                type="emerging_topic",
                title=f"Rising topic: {cat} ({growth:.1f}x growth)",
                description=(
                    f"'{cat}' now represents {current_pct*100:.1f}% of feedback, "
                    f"up from {historical_pct*100:.1f}%. This {growth:.1f}x increase "
                    f"may indicate an emerging issue."
                ),
                severity="warning" if growth > 3 else "info",
                score=min(1.0, growth / 10.0),
                data={"category": cat, "current_pct": round(current_pct, 3), "growth": round(growth, 2)},
            ))

    return sorted(anomalies, key=lambda a: a.score, reverse=True)


def detect_segment_divergence(
    segment_sentiments: dict[str, float],
    overall_sentiment: float,
    threshold: float = 0.3,
) -> list[Anomaly]:
    """Detect segments whose sentiment diverges significantly from the overall."""
    anomalies: list[Anomaly] = []
    for segment, sentiment in segment_sentiments.items():
        diff = sentiment - overall_sentiment
        if abs(diff) > threshold:
            direction = "more negative" if diff < 0 else "more positive"
            anomalies.append(Anomaly(
                type="segment_divergence",
                title=f"{segment} segment is {direction} than average",
                description=(
                    f"The {segment} segment has a sentiment of {sentiment:.2f} vs. "
                    f"{overall_sentiment:.2f} overall (Δ{diff:+.2f}). This divergence "
                    f"may indicate segment-specific issues."
                ),
                severity="warning" if abs(diff) > 0.5 else "info",
                score=min(1.0, abs(diff)),
                data={"segment": segment, "sentiment": round(sentiment, 3), "diff": round(diff, 3)},
            ))
    return sorted(anomalies, key=lambda a: a.score, reverse=True)
