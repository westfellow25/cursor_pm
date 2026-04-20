"""Feedback ingestion service — multi-source, enrichment, and persistence.

This service is the single entry point for all feedback into the platform.
Every item gets enriched with AI metadata before storage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from pulse.ml.embeddings import embed_texts
from pulse.ml.sentiment import enrich_feedback
from pulse.models import FeedbackItem, FeedbackSource

logger = logging.getLogger(__name__)


def ingest_csv(
    db: Session,
    org_id: str,
    source_id: str,
    file_bytes: bytes,
) -> list[FeedbackItem]:
    """Ingest feedback from a CSV file.

    Expects at minimum a 'feedback' or 'text' column. Optional columns:
    feedback_id, source, author, channel, created_at, segment, metadata.*
    """
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find the text column
    text_col = None
    for candidate in ("feedback", "text", "comment", "message", "body", "review"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"CSV must contain a text column (feedback, text, comment, message, body, or review). "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().astype(bool)]

    if df.empty:
        return []

    texts = df[text_col].tolist()

    # Generate embeddings
    embeddings = embed_texts(texts)

    # Enrich each item
    items: list[FeedbackItem] = []
    for i, row in enumerate(df.itertuples(index=False)):
        text = getattr(row, text_col)
        enrichment = enrich_feedback(text)

        # Parse optional fields
        created_at = None
        if hasattr(row, "created_at") and pd.notna(row.created_at):
            try:
                created_at = pd.to_datetime(row.created_at).to_pydatetime()
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except Exception:
                created_at = None
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        metadata: dict[str, Any] = {}
        for col in df.columns:
            if col.startswith("metadata_") or col in ("nps", "mrr", "plan", "company"):
                val = getattr(row, col, None)
                if pd.notna(val):
                    metadata[col] = val

        item = FeedbackItem(
            org_id=org_id,
            source_id=source_id,
            external_id=getattr(row, "feedback_id", None) or getattr(row, "id", None),
            text=text,
            author=str(getattr(row, "author", "") or ""),
            author_segment=str(getattr(row, "segment", "") or getattr(row, "author_segment", "") or ""),
            channel=str(getattr(row, "channel", "") or getattr(row, "source", "") or ""),
            sentiment=enrichment["sentiment"],
            urgency=enrichment["urgency"],
            category=enrichment["category"],
            subcategory=enrichment["subcategory"],
            embedding=embeddings[i].tolist() if i < len(embeddings) else None,
            meta=metadata,
            created_at=created_at,
            ingested_at=datetime.now(timezone.utc),
        )
        items.append(item)

    db.add_all(items)

    # Update source stats
    source = db.get(FeedbackSource, source_id)
    if source:
        source.items_synced = (source.items_synced or 0) + len(items)
        source.last_sync_at = datetime.now(timezone.utc)
        source.status = "active"

    db.commit()
    for item in items:
        db.refresh(item)

    logger.info("Ingested %d feedback items for org %s from source %s", len(items), org_id, source_id)
    return items


def ingest_items(
    db: Session,
    org_id: str,
    source_id: str | None,
    items_data: list[dict[str, Any]],
) -> list[FeedbackItem]:
    """Ingest feedback from structured data (API, connectors).

    Each dict must have 'text'. Optional: author, channel, segment, metadata, created_at.
    """
    if not items_data:
        return []

    texts = [d["text"] for d in items_data]
    embeddings = embed_texts(texts)

    items: list[FeedbackItem] = []
    for i, data in enumerate(items_data):
        enrichment = enrich_feedback(data["text"])
        created_at = data.get("created_at", datetime.now(timezone.utc))
        if isinstance(created_at, str):
            created_at = pd.to_datetime(created_at).to_pydatetime()
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

        item = FeedbackItem(
            org_id=org_id,
            source_id=source_id,
            external_id=data.get("external_id"),
            text=data["text"],
            author=data.get("author", ""),
            author_segment=data.get("segment", ""),
            channel=data.get("channel", ""),
            sentiment=enrichment["sentiment"],
            urgency=enrichment["urgency"],
            category=enrichment["category"],
            subcategory=enrichment["subcategory"],
            embedding=embeddings[i].tolist() if i < len(embeddings) else None,
            meta=data.get("metadata", {}),
            created_at=created_at,
            ingested_at=datetime.now(timezone.utc),
        )
        items.append(item)

    db.add_all(items)

    if source_id:
        source = db.get(FeedbackSource, source_id)
        if source:
            source.items_synced = (source.items_synced or 0) + len(items)
            source.last_sync_at = datetime.now(timezone.utc)

    db.commit()
    for item in items:
        db.refresh(item)

    return items
