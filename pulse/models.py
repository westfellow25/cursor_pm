"""Domain models — the schema is designed for multi-tenant SaaS with temporal
intelligence, cross-org benchmarking, and deep integration state tracking."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship

from pulse.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


# ── Organisation (tenant) ────────────────────────────────────────────────────

class Organisation(Base):
    __tablename__ = "organisations"

    id = Column(String(32), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(63), unique=True, nullable=False, index=True)
    plan = Column(String(32), default="trial")  # trial | growth | enterprise
    industry = Column(String(128), default="")
    company_size = Column(String(32), default="")  # 1-10 | 11-50 | 51-200 | 201-1000 | 1000+
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    users = relationship("User", back_populates="organisation", cascade="all, delete-orphan")
    feedback_sources = relationship("FeedbackSource", back_populates="organisation", cascade="all, delete-orphan")
    feedback_items = relationship("FeedbackItem", back_populates="organisation", cascade="all, delete-orphan")
    analysis_runs = relationship("AnalysisRun", back_populates="organisation", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="organisation", cascade="all, delete-orphan")


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), default="member")  # owner | admin | member | viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    organisation = relationship("Organisation", back_populates="users")


# ── API Keys ──────────────────────────────────────────────────────────────────

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    key_prefix = Column(String(8), nullable=False)  # first 8 chars for identification
    key_hash = Column(String(255), nullable=False)
    scopes = Column(JSON, default=list)  # ["feedback:write", "analysis:read", ...]
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ── Feedback Sources (integrations) ──────────────────────────────────────────

class FeedbackSource(Base):
    __tablename__ = "feedback_sources"

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(32), nullable=False)  # csv | intercom | zendesk | slack | appstore | api | g2 | survey
    name = Column(String(255), nullable=False)
    config = Column(JSON, default=dict)  # connector-specific configuration
    status = Column(String(32), default="active")  # active | paused | error | disconnected
    error_message = Column(Text, nullable=True)
    items_synced = Column(Integer, default=0)
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    organisation = relationship("Organisation", back_populates="feedback_sources")
    feedback_items = relationship("FeedbackItem", back_populates="source", cascade="all, delete-orphan")


# ── Feedback Items ────────────────────────────────────────────────────────────

class FeedbackItem(Base):
    __tablename__ = "feedback_items"
    __table_args__ = (
        Index("ix_feedback_org_created", "org_id", "created_at"),
        Index("ix_feedback_org_sentiment", "org_id", "sentiment"),
        Index("ix_feedback_org_category", "org_id", "category"),
    )

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(String(32), ForeignKey("feedback_sources.id", ondelete="SET NULL"), nullable=True)
    external_id = Column(String(255), nullable=True)  # ID in the source system

    text = Column(Text, nullable=False)
    author = Column(String(255), default="")
    author_segment = Column(String(64), default="")  # enterprise | mid-market | smb | free | unknown
    channel = Column(String(64), default="")  # web | mobile | api | email | chat | call | review

    # AI-enriched fields
    sentiment = Column(Float, nullable=True)  # -1.0 to 1.0
    urgency = Column(Float, nullable=True)  # 0.0 to 1.0
    category = Column(String(128), nullable=True)  # performance | ux | bug | feature-request | praise | ...
    subcategory = Column(String(128), nullable=True)
    language = Column(String(8), default="en")
    embedding = Column(JSON, nullable=True)  # vector as JSON array (swap for pgvector in prod)

    # Metadata from source
    meta = Column("metadata_json", JSON, default=dict)  # flexible: NPS score, plan, MRR, etc.

    created_at = Column(DateTime(timezone=True), default=_utcnow)  # when feedback was given
    ingested_at = Column(DateTime(timezone=True), default=_utcnow)  # when we received it

    organisation = relationship("Organisation", back_populates="feedback_items")
    source = relationship("FeedbackSource", back_populates="feedback_items")
    cluster_memberships = relationship("ClusterMember", back_populates="feedback_item", cascade="all, delete-orphan")


# ── Analysis Runs ─────────────────────────────────────────────────────────────

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(32), default="pending")  # pending | running | completed | failed
    run_type = Column(String(32), default="full")  # full | incremental | scheduled
    config = Column(JSON, default=dict)
    feedback_count = Column(Integer, default=0)
    cluster_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    organisation = relationship("Organisation", back_populates="analysis_runs")
    clusters = relationship("Cluster", back_populates="analysis_run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="analysis_run", cascade="all, delete-orphan")


# ── Clusters ──────────────────────────────────────────────────────────────────

class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(String(32), primary_key=True, default=_uuid)
    run_id = Column(String(32), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)

    label = Column(String(255), nullable=False)
    theme = Column(Text, default="")
    summary = Column(Text, default="")
    size = Column(Integer, default=0)

    # Scoring
    opportunity_score = Column(Float, default=0.0)  # 0–10
    severity_score = Column(Float, default=0.0)
    frequency_score = Column(Float, default=0.0)
    revenue_impact = Column(Float, default=0.0)  # estimated $ at risk
    trend_direction = Column(String(16), default="stable")  # rising | stable | declining

    # Centroid for similarity matching
    centroid = Column(JSON, nullable=True)  # embedding vector
    top_keywords = Column(JSON, default=list)
    sentiment_avg = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)

    analysis_run = relationship("AnalysisRun", back_populates="clusters")
    members = relationship("ClusterMember", back_populates="cluster", cascade="all, delete-orphan")


class ClusterMember(Base):
    __tablename__ = "cluster_members"
    __table_args__ = (
        Index("ix_clustermember_cluster", "cluster_id"),
        Index("ix_clustermember_feedback", "feedback_id"),
    )

    id = Column(String(32), primary_key=True, default=_uuid)
    cluster_id = Column(String(32), ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False)
    feedback_id = Column(String(32), ForeignKey("feedback_items.id", ondelete="CASCADE"), nullable=False)
    similarity = Column(Float, default=0.0)

    cluster = relationship("Cluster", back_populates="members")
    feedback_item = relationship("FeedbackItem", back_populates="cluster_memberships")


# ── Insights (AI-generated observations) ─────────────────────────────────────

class Insight(Base):
    __tablename__ = "insights"
    __table_args__ = (
        Index("ix_insight_org_type", "org_id", "type"),
    )

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(64), nullable=False)
    # Types: trend_spike | emerging_theme | sentiment_shift | segment_divergence |
    #        churn_signal | benchmark_outlier | opportunity_found | regression_detected

    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(16), default="info")  # critical | warning | info | positive
    data = Column(JSON, default=dict)  # insight-specific payload
    is_read = Column(Boolean, default=False)
    is_actionable = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    organisation = relationship("Organisation", back_populates="insights")


# ── Artifacts (generated documents) ──────────────────────────────────────────

class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(32), ForeignKey("analysis_runs.id", ondelete="SET NULL"), nullable=True)
    type = Column(String(32), nullable=False)  # prd | jira_tickets | roadmap | impact_report | executive_summary
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)  # markdown
    meta = Column("metadata_json", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ── Trend Snapshots (time-series data for temporal intelligence) ─────────────

class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"
    __table_args__ = (
        Index("ix_trend_org_period", "org_id", "period_start"),
    )

    id = Column(String(32), primary_key=True, default=_uuid)
    org_id = Column(String(32), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    granularity = Column(String(16), default="week")  # day | week | month

    total_feedback = Column(Integer, default=0)
    avg_sentiment = Column(Float, nullable=True)
    category_distribution = Column(JSON, default=dict)  # {category: count}
    top_themes = Column(JSON, default=list)  # [{theme, count, trend}]
    segment_breakdown = Column(JSON, default=dict)  # {segment: {count, sentiment}}
    anomalies = Column(JSON, default=list)  # detected anomalies in this period

    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ── Benchmark Data (cross-org anonymised intelligence — the moat) ────────────

class BenchmarkData(Base):
    __tablename__ = "benchmark_data"

    id = Column(String(32), primary_key=True, default=_uuid)
    industry = Column(String(128), nullable=False, index=True)
    company_size = Column(String(32), nullable=False)
    period = Column(String(16), nullable=False)  # 2024-Q1, 2024-W12, etc.
    metric = Column(String(128), nullable=False)
    # Metrics: avg_sentiment | top_category | nps_correlation | resolution_rate | ...
    value = Column(Float, nullable=False)
    sample_size = Column(Integer, default=0)  # how many orgs contributed
    meta = Column("metadata_json", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
