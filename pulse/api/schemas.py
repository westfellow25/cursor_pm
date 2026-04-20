"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    org_name: str = Field(min_length=2, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8)
    industry: str = ""
    company_size: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    org_id: str
    user_id: str
    name: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    org_id: str


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    text: str
    author: str
    author_segment: str
    channel: str
    sentiment: float | None
    urgency: float | None
    category: str | None
    subcategory: str | None
    meta: dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    created_at: datetime
    ingested_at: datetime

class FeedbackSubmit(BaseModel):
    text: str = Field(min_length=1)
    author: str = ""
    channel: str = ""
    segment: str = ""
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None

class FeedbackListResponse(BaseModel):
    items: list[FeedbackItemResponse]
    total: int
    page: int
    page_size: int


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisRunResponse(BaseModel):
    id: str
    status: str
    run_type: str
    feedback_count: int
    cluster_count: int
    started_at: datetime | None
    completed_at: datetime | None

    class Config:
        from_attributes = True

class ClusterResponse(BaseModel):
    id: str
    label: str
    theme: str
    summary: str
    size: int
    opportunity_score: float
    severity_score: float
    frequency_score: float
    sentiment_avg: float | None
    trend_direction: str
    top_keywords: list[str]
    revenue_impact: float

    class Config:
        from_attributes = True

class AnalysisDetailResponse(BaseModel):
    run: AnalysisRunResponse
    clusters: list[ClusterResponse]
    top_opportunity: ClusterResponse | None
    evidence: list[str]


# ── Insights ──────────────────────────────────────────────────────────────────

class InsightResponse(BaseModel):
    id: str
    type: str
    title: str
    description: str
    severity: str
    data: dict[str, Any]
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_feedback: int
    total_sources: int
    active_clusters: int
    avg_sentiment: float | None
    unread_insights: int
    top_category: str
    feedback_this_week: int
    sentiment_trend: str  # improving | stable | declining

class TrendPoint(BaseModel):
    period: str
    value: float | None = None
    count: int | None = None

class TrendDataResponse(BaseModel):
    volume: list[TrendPoint]
    sentiment: list[TrendPoint]
    categories: list[dict[str, Any]]

class DashboardResponse(BaseModel):
    stats: DashboardStats
    trends: TrendDataResponse
    recent_insights: list[InsightResponse]
    top_clusters: list[ClusterResponse]


# ── Integrations ──────────────────────────────────────────────────────────────

class SourceResponse(BaseModel):
    id: str
    type: str
    name: str
    status: str
    items_synced: int
    last_sync_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True

class CreateSourceRequest(BaseModel):
    type: str
    name: str
    config: dict[str, Any] = {}

class ConnectorInfo(BaseModel):
    type: str
    name: str
    description: str


# ── Artifacts ─────────────────────────────────────────────────────────────────

class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    title: str
    content: str
    meta: dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    created_at: datetime
