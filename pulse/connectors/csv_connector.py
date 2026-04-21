"""CSV file connector — the simplest ingestion path."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from pulse.connectors.base import BaseConnector, FeedbackRecord, register_connector


@register_connector
class CSVConnector(BaseConnector):
    connector_type = "csv"
    display_name = "CSV Upload"
    description = "Upload feedback from CSV files. Supports any CSV with a text column."
    config_schema = {
        "type": "object",
        "properties": {
            "text_column": {"type": "string", "default": "feedback"},
        },
    }

    def validate_config(self) -> bool:
        return True

    def test_connection(self) -> bool:
        return True

    def pull(self, since: datetime | None = None) -> list[FeedbackRecord]:
        raise NotImplementedError("CSV connector uses direct file upload via ingest_csv()")

    def parse_bytes(self, file_bytes: bytes) -> list[FeedbackRecord]:
        """Parse CSV bytes into FeedbackRecords."""
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        text_col = self.config.get("text_column", "feedback")
        candidates = [text_col, "feedback", "text", "comment", "message", "body", "review"]
        found_col = None
        for c in candidates:
            if c in df.columns:
                found_col = c
                break
        if found_col is None:
            raise ValueError(f"No text column found. Expected one of: {candidates}. Got: {list(df.columns)}")

        df = df.dropna(subset=[found_col])
        records: list[FeedbackRecord] = []
        for row in df.itertuples(index=False):
            text = str(getattr(row, found_col, "")).strip()
            if not text:
                continue

            created_at = None
            if hasattr(row, "created_at") and pd.notna(row.created_at):
                try:
                    created_at = pd.to_datetime(row.created_at).to_pydatetime()
                except Exception:
                    pass

            records.append(FeedbackRecord(
                text=text,
                external_id=str(getattr(row, "feedback_id", "") or getattr(row, "id", "") or ""),
                author=str(getattr(row, "author", "") or ""),
                channel=str(getattr(row, "channel", "") or getattr(row, "source", "") or "csv"),
                segment=str(getattr(row, "segment", "") or getattr(row, "author_segment", "") or ""),
                metadata={
                    col: getattr(row, col)
                    for col in df.columns
                    if col.startswith("metadata_") or col in ("nps", "mrr", "plan", "company")
                    if pd.notna(getattr(row, col, None))
                },
                created_at=created_at or datetime.utcnow(),
            ))

        return records
