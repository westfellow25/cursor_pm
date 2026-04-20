"""REST API connector — allows any system to push feedback via API.

This is the universal connector. Customers embed API calls in their
apps, support tools, or build custom pipelines."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pulse.connectors.base import BaseConnector, FeedbackRecord, register_connector


@register_connector
class APIConnector(BaseConnector):
    connector_type = "api"
    display_name = "REST API"
    description = "Push feedback from any system via REST API. Generate an API key to get started."
    config_schema = {
        "type": "object",
        "properties": {
            "api_key_id": {"type": "string"},
            "webhook_url": {"type": "string", "description": "Optional: URL to receive analysis results"},
        },
    }

    def validate_config(self) -> bool:
        return True

    def test_connection(self) -> bool:
        return True

    def pull(self, since: datetime | None = None) -> list[FeedbackRecord]:
        raise NotImplementedError("API connector receives push-based data via the /api/v1/feedback endpoint")
