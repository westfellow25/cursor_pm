"""Base connector interface — the pluggable integration framework.

Every feedback source implements this interface. The connector registry
makes it trivial to add new sources: implement pull(), register, done."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Normalized feedback record from any source."""
    text: str
    external_id: str = ""
    author: str = ""
    channel: str = ""
    segment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseConnector(ABC):
    """Abstract base for all feedback source connectors."""

    connector_type: str = "base"
    display_name: str = "Base Connector"
    description: str = ""
    config_schema: dict[str, Any] = {}  # JSON Schema for connector config

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate connector configuration. Raises ValueError on invalid."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test that the connection to the source works."""
        ...

    @abstractmethod
    def pull(self, since: datetime | None = None) -> list[FeedbackRecord]:
        """Pull feedback records from the source since a given timestamp."""
        ...

    def get_display_info(self) -> dict[str, str]:
        return {
            "type": self.connector_type,
            "name": self.display_name,
            "description": self.description,
        }


# ── Connector Registry ───────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseConnector]] = {}


def register_connector(cls: type[BaseConnector]) -> type[BaseConnector]:
    """Decorator to register a connector class."""
    _REGISTRY[cls.connector_type] = cls
    return cls


def get_connector(connector_type: str, config: dict[str, Any]) -> BaseConnector:
    """Instantiate a connector by type."""
    cls = _REGISTRY.get(connector_type)
    if cls is None:
        raise ValueError(f"Unknown connector type: {connector_type}. Available: {list(_REGISTRY.keys())}")
    return cls(config)


def list_connectors() -> list[dict[str, str]]:
    """List all available connector types."""
    return [cls({}).get_display_info() for cls in _REGISTRY.values()]
