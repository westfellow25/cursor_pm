"""Slack connector — pulls feedback from designated Slack channels.

Monitors channels like #product-feedback, #bug-reports, #feature-requests
for customer-facing feedback that teams forward from conversations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from pulse.connectors.base import BaseConnector, FeedbackRecord, register_connector


@register_connector
class SlackConnector(BaseConnector):
    connector_type = "slack"
    display_name = "Slack"
    description = "Monitor Slack channels for product feedback shared by your team."
    config_schema = {
        "type": "object",
        "required": ["bot_token", "channels"],
        "properties": {
            "bot_token": {"type": "string"},
            "channels": {"type": "array", "items": {"type": "string"}},
        },
    }

    BASE_URL = "https://slack.com/api"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config['bot_token']}"}

    def validate_config(self) -> bool:
        if not self.config.get("bot_token"):
            raise ValueError("Slack bot_token is required")
        if not self.config.get("channels"):
            raise ValueError("At least one Slack channel is required")
        return True

    def test_connection(self) -> bool:
        try:
            resp = httpx.post(
                f"{self.BASE_URL}/auth.test",
                headers=self._headers(),
                timeout=10,
            )
            return resp.json().get("ok", False)
        except Exception:
            return False

    def pull(self, since: datetime | None = None) -> list[FeedbackRecord]:
        """Pull messages from configured Slack channels."""
        self.validate_config()
        records: list[FeedbackRecord] = []

        for channel_id in self.config["channels"]:
            params: dict[str, Any] = {"channel": channel_id, "limit": 200}
            if since:
                params["oldest"] = str(since.timestamp())

            resp = httpx.get(
                f"{self.BASE_URL}/conversations.history",
                headers=self._headers(),
                params=params,
                timeout=30,
            )
            data = resp.json()
            if not data.get("ok"):
                continue

            for msg in data.get("messages", []):
                text = msg.get("text", "").strip()
                if not text or msg.get("subtype"):  # Skip system messages
                    continue

                ts = float(msg.get("ts", "0"))
                created = datetime.fromtimestamp(ts, tz=timezone.utc)

                records.append(FeedbackRecord(
                    text=text,
                    external_id=msg.get("ts", ""),
                    author=msg.get("user", ""),
                    channel="slack",
                    metadata={
                        "slack_channel": channel_id,
                        "thread_ts": msg.get("thread_ts"),
                        "reactions": [
                            r.get("name") for r in msg.get("reactions", [])
                        ],
                    },
                    created_at=created,
                ))

        return records
