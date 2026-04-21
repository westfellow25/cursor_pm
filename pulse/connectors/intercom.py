"""Intercom connector — pulls conversations and feedback from Intercom.

This is a production-ready connector skeleton. In production, it would
handle pagination, rate limiting, and webhook-based real-time sync."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from pulse.connectors.base import BaseConnector, FeedbackRecord, register_connector


@register_connector
class IntercomConnector(BaseConnector):
    connector_type = "intercom"
    display_name = "Intercom"
    description = "Import conversations, feedback, and NPS responses from Intercom."
    config_schema = {
        "type": "object",
        "required": ["access_token"],
        "properties": {
            "access_token": {"type": "string"},
            "tag_filter": {"type": "string", "description": "Only import conversations with this tag"},
        },
    }

    BASE_URL = "https://api.intercom.io"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config['access_token']}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def validate_config(self) -> bool:
        if not self.config.get("access_token"):
            raise ValueError("Intercom access_token is required")
        return True

    def test_connection(self) -> bool:
        try:
            resp = httpx.get(f"{self.BASE_URL}/me", headers=self._headers(), timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def pull(self, since: datetime | None = None) -> list[FeedbackRecord]:
        """Pull conversations from Intercom."""
        self.validate_config()
        records: list[FeedbackRecord] = []

        params: dict[str, Any] = {"per_page": 50, "order": "desc"}
        url = f"{self.BASE_URL}/conversations"

        while url:
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for conv in data.get("conversations", []):
                created = datetime.fromtimestamp(conv.get("created_at", 0), tz=timezone.utc)
                if since and created < since:
                    return records

                # Extract first customer message
                source = conv.get("source", {})
                body = source.get("body", "") or ""
                # Strip HTML tags
                import re
                text = re.sub(r"<[^>]+>", "", body).strip()
                if not text:
                    continue

                author_name = ""
                if source.get("author"):
                    author_name = source["author"].get("name", "")

                records.append(FeedbackRecord(
                    text=text,
                    external_id=str(conv.get("id", "")),
                    author=author_name,
                    channel="intercom",
                    segment=conv.get("custom_attributes", {}).get("plan", ""),
                    metadata={
                        "conversation_id": conv.get("id"),
                        "state": conv.get("state"),
                        "tags": [t.get("name") for t in conv.get("tags", {}).get("tags", [])],
                    },
                    created_at=created,
                ))

            # Pagination
            pages = data.get("pages", {})
            next_page = pages.get("next")
            if next_page:
                url = next_page if isinstance(next_page, str) else next_page.get("url")
                params = {}
            else:
                url = None

        return records
