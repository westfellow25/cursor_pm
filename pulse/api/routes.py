"""API routes — all endpoints for the Pulse platform."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import desc, func

from pulse.config import settings
from pulse.api.deps import (
    DB,
    CurrentOrgId,
    CurrentUser,
    create_access_token,
    hash_password,
    verify_password,
)
from pulse.api.schemas import (
    AnalysisDetailResponse,
    AnalysisRunResponse,
    ArtifactResponse,
    ClusterResponse,
    ConnectorInfo,
    CreateSourceRequest,
    DashboardResponse,
    DashboardStats,
    FeedbackItemResponse,
    FeedbackListResponse,
    FeedbackSubmit,
    InsightResponse,
    LoginRequest,
    SignupRequest,
    SourceResponse,
    TokenResponse,
    TrendDataResponse,
    TrendPoint,
    UserResponse,
)
from pulse.connectors.base import list_connectors
from pulse.models import (
    AnalysisRun,
    Artifact,
    Cluster,
    ClusterMember,
    FeedbackItem,
    FeedbackSource,
    Insight,
    Organisation,
    TrendSnapshot,
    User,
)
from pulse.services.artifacts import (
    generate_executive_summary,
    generate_jira_tickets,
    generate_prd,
)
from pulse.services.ingestion import ingest_csv, ingest_items
from pulse.services.insights import generate_run_insights, get_insights, mark_insight_read
from pulse.services.intelligence import run_analysis
from pulse.services.trends import compute_weekly_snapshots, detect_anomalies, get_trend_data

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Auth ──────────────────────────────────────────────────────────────────────


@router.post("/auth/signup", response_model=TokenResponse, tags=["auth"])
def signup(body: SignupRequest, db: DB):
    """Register a new organisation and admin user."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    slug = body.org_name.lower().replace(" ", "-")[:63]
    # Make slug unique
    base_slug = slug
    counter = 1
    while db.query(Organisation).filter(Organisation.slug == slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1

    org = Organisation(
        name=body.org_name,
        slug=slug,
        industry=body.industry,
        company_size=body.company_size,
    )
    db.add(org)
    db.flush()

    user = User(
        org_id=org.id,
        email=body.email,
        name=body.name,
        password_hash=hash_password(body.password),
        role="owner",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, org.id)
    return TokenResponse(access_token=token, org_id=org.id, user_id=user.id, name=user.name)


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest, db: DB):
    """Authenticate and receive a JWT token."""
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(user.id, user.org_id)
    return TokenResponse(access_token=token, org_id=user.org_id, user_id=user.id, name=user.name)


@router.get("/auth/me", response_model=UserResponse, tags=["auth"])
def me(user: CurrentUser):
    return user


@router.get("/system/status", tags=["system"])
def system_status(verify: bool = False):
    """Return info about the active LLM + embedding providers and feature flags.

    Pass `?verify=true` to actually ping the LLM provider (1 token round-trip)
    and report whether the configured API key works.
    """
    from pulse.ml.llm import get_llm_info, verify_llm
    from pulse.ml.embeddings import get_embedding_info

    llm = dict(get_llm_info())
    if verify:
        llm["healthy"] = verify_llm()
    embeddings = get_embedding_info()
    return {
        "llm": llm,
        "embeddings": embeddings,
        "features": {
            "llm_configured": llm["provider"] != "none",
            "embeddings_semantic": embeddings["provider"] in ("openai", "sentence-transformers"),
        },
    }


# ── Dashboard ─────────────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=DashboardResponse, tags=["dashboard"])
def get_dashboard(db: DB, org_id: CurrentOrgId):
    """Main dashboard data — stats, trends, insights, top clusters."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Stats
    total_feedback = db.query(func.count(FeedbackItem.id)).filter(FeedbackItem.org_id == org_id).scalar() or 0
    total_sources = db.query(func.count(FeedbackSource.id)).filter(FeedbackSource.org_id == org_id).scalar() or 0
    feedback_this_week = (
        db.query(func.count(FeedbackItem.id))
        .filter(FeedbackItem.org_id == org_id, FeedbackItem.created_at >= week_ago)
        .scalar() or 0
    )
    avg_sentiment = (
        db.query(func.avg(FeedbackItem.sentiment))
        .filter(FeedbackItem.org_id == org_id, FeedbackItem.sentiment.isnot(None))
        .scalar()
    )
    unread_insights = (
        db.query(func.count(Insight.id))
        .filter(Insight.org_id == org_id, Insight.is_read == False)
        .scalar() or 0
    )

    # Top category
    cat_counts = (
        db.query(FeedbackItem.category, func.count(FeedbackItem.id))
        .filter(FeedbackItem.org_id == org_id, FeedbackItem.category.isnot(None))
        .group_by(FeedbackItem.category)
        .order_by(func.count(FeedbackItem.id).desc())
        .first()
    )
    top_category = cat_counts[0] if cat_counts else "N/A"

    # Active clusters (from latest run)
    latest_run = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.org_id == org_id, AnalysisRun.status == "completed")
        .order_by(desc(AnalysisRun.completed_at))
        .first()
    )
    active_clusters = latest_run.cluster_count if latest_run else 0

    # Sentiment trend
    snapshots = (
        db.query(TrendSnapshot)
        .filter(TrendSnapshot.org_id == org_id)
        .order_by(desc(TrendSnapshot.period_start))
        .limit(4)
        .all()
    )
    if len(snapshots) >= 2 and snapshots[0].avg_sentiment and snapshots[-1].avg_sentiment:
        diff = snapshots[0].avg_sentiment - snapshots[-1].avg_sentiment
        sentiment_trend = "improving" if diff > 0.05 else "declining" if diff < -0.05 else "stable"
    else:
        sentiment_trend = "stable"

    stats = DashboardStats(
        total_feedback=total_feedback,
        total_sources=total_sources,
        active_clusters=active_clusters,
        avg_sentiment=round(avg_sentiment, 3) if avg_sentiment else None,
        unread_insights=unread_insights,
        top_category=top_category,
        feedback_this_week=feedback_this_week,
        sentiment_trend=sentiment_trend,
    )

    # Trends
    trend_data = get_trend_data(db, org_id, weeks=12)
    trends = TrendDataResponse(
        volume=[TrendPoint(**t) for t in trend_data["volume"]],
        sentiment=[TrendPoint(**t) for t in trend_data["sentiment"]],
        categories=trend_data["categories"],
    )

    # Recent insights
    recent_insights_raw = get_insights(db, org_id, limit=5)
    recent_insights = [InsightResponse.model_validate(i) for i in recent_insights_raw]

    # Top clusters
    top_clusters_raw = []
    if latest_run:
        top_clusters_raw = (
            db.query(Cluster)
            .filter(Cluster.run_id == latest_run.id)
            .order_by(desc(Cluster.opportunity_score))
            .limit(5)
            .all()
        )
    top_clusters = [ClusterResponse.model_validate(c) for c in top_clusters_raw]

    return DashboardResponse(
        stats=stats,
        trends=trends,
        recent_insights=recent_insights,
        top_clusters=top_clusters,
    )


# ── Feedback ──────────────────────────────────────────────────────────────────


@router.get("/feedback", response_model=FeedbackListResponse, tags=["feedback"])
def list_feedback(
    db: DB,
    org_id: CurrentOrgId,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    category: str | None = None,
    sentiment_min: float | None = None,
    sentiment_max: float | None = None,
    channel: str | None = None,
    segment: str | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
):
    """List feedback items with filtering, search, and pagination."""
    query = db.query(FeedbackItem).filter(FeedbackItem.org_id == org_id)

    if category:
        query = query.filter(FeedbackItem.category == category)
    if sentiment_min is not None:
        query = query.filter(FeedbackItem.sentiment >= sentiment_min)
    if sentiment_max is not None:
        query = query.filter(FeedbackItem.sentiment <= sentiment_max)
    if channel:
        query = query.filter(FeedbackItem.channel == channel)
    if segment:
        query = query.filter(FeedbackItem.author_segment == segment)
    if search:
        query = query.filter(FeedbackItem.text.ilike(f"%{search}%"))

    total = query.count()

    # Sort
    sort_col = getattr(FeedbackItem, sort_by, FeedbackItem.created_at)
    query = query.order_by(desc(sort_col) if sort_dir == "desc" else sort_col)

    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return FeedbackListResponse(
        items=[FeedbackItemResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/feedback", response_model=list[FeedbackItemResponse], tags=["feedback"])
def submit_feedback(body: list[FeedbackSubmit], db: DB, org_id: CurrentOrgId):
    """Submit feedback items via API."""
    items = ingest_items(
        db, org_id, source_id=None,
        items_data=[b.model_dump() for b in body],
    )
    return [FeedbackItemResponse.model_validate(i) for i in items]


@router.post("/feedback/upload", response_model=list[FeedbackItemResponse], tags=["feedback"])
def upload_csv(file: UploadFile = File(...), db: DB = ..., org_id: CurrentOrgId = ...):
    """Upload a CSV file of feedback."""
    # Find or create CSV source
    source = (
        db.query(FeedbackSource)
        .filter(FeedbackSource.org_id == org_id, FeedbackSource.type == "csv")
        .first()
    )
    if not source:
        source = FeedbackSource(org_id=org_id, type="csv", name="CSV Upload")
        db.add(source)
        db.flush()

    file_bytes = file.file.read()
    items = ingest_csv(db, org_id, source.id, file_bytes)
    return [FeedbackItemResponse.model_validate(i) for i in items]


@router.get("/feedback/stats", tags=["feedback"])
def feedback_stats(db: DB, org_id: CurrentOrgId):
    """Get aggregate feedback statistics."""
    total = db.query(func.count(FeedbackItem.id)).filter(FeedbackItem.org_id == org_id).scalar() or 0

    categories = dict(
        db.query(FeedbackItem.category, func.count(FeedbackItem.id))
        .filter(FeedbackItem.org_id == org_id)
        .group_by(FeedbackItem.category)
        .all()
    )

    channels = dict(
        db.query(FeedbackItem.channel, func.count(FeedbackItem.id))
        .filter(FeedbackItem.org_id == org_id)
        .group_by(FeedbackItem.channel)
        .all()
    )

    segments = dict(
        db.query(FeedbackItem.author_segment, func.count(FeedbackItem.id))
        .filter(FeedbackItem.org_id == org_id)
        .group_by(FeedbackItem.author_segment)
        .all()
    )

    avg_sentiment = (
        db.query(func.avg(FeedbackItem.sentiment))
        .filter(FeedbackItem.org_id == org_id, FeedbackItem.sentiment.isnot(None))
        .scalar()
    )

    return {
        "total": total,
        "categories": categories,
        "channels": channels,
        "segments": segments,
        "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment else None,
    }


# ── Analysis ──────────────────────────────────────────────────────────────────


@router.post("/analysis/run", response_model=AnalysisRunResponse, tags=["analysis"])
def trigger_analysis(
    db: DB,
    org_id: CurrentOrgId,
    n_clusters: int | None = None,
):
    """Trigger a new analysis run."""
    analysis_run = run_analysis(db, org_id, n_clusters=n_clusters)

    # Generate insights and artifacts
    if analysis_run.status == "completed":
        generate_run_insights(db, analysis_run)
        generate_prd(db, analysis_run)
        generate_jira_tickets(db, analysis_run)
        generate_executive_summary(db, analysis_run)
        # Compute trend snapshots
        compute_weekly_snapshots(db, org_id, weeks=12)
        # Detect anomalies
        detect_anomalies(db, org_id)

    return AnalysisRunResponse.model_validate(analysis_run)


@router.get("/analysis/latest", response_model=AnalysisDetailResponse, tags=["analysis"])
def get_latest_analysis(db: DB, org_id: CurrentOrgId):
    """Get the latest completed analysis with full details."""
    run = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.org_id == org_id, AnalysisRun.status == "completed")
        .order_by(desc(AnalysisRun.completed_at))
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No completed analysis found. Upload feedback and run an analysis first.")

    clusters = (
        db.query(Cluster)
        .filter(Cluster.run_id == run.id)
        .order_by(desc(Cluster.opportunity_score))
        .all()
    )

    # Get diverse evidence for top cluster (deduped by first 50 chars)
    evidence: list[str] = []
    if clusters:
        top = clusters[0]
        members = (
            db.query(ClusterMember)
            .filter(ClusterMember.cluster_id == top.id)
            .order_by(desc(ClusterMember.similarity))
            .limit(60)
            .all()
        )
        seen_sigs: set[str] = set()
        for m in members:
            if len(evidence) >= 5:
                break
            item = db.get(FeedbackItem, m.feedback_id)
            if item:
                # Signature = first 6 words to catch template variations
                sig = " ".join(item.text.lower().split()[:6])
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    evidence.append(item.text)

    return AnalysisDetailResponse(
        run=AnalysisRunResponse.model_validate(run),
        clusters=[ClusterResponse.model_validate(c) for c in clusters],
        top_opportunity=ClusterResponse.model_validate(clusters[0]) if clusters else None,
        evidence=evidence,
    )


@router.get("/analysis/cluster/{cluster_id}/deep-dive", tags=["analysis"])
def cluster_deep_dive(cluster_id: str, db: DB, org_id: CurrentOrgId):
    """LLM-powered deep dive into a specific cluster."""
    cluster = db.get(Cluster, cluster_id)
    if not cluster or cluster.org_id != org_id:
        raise HTTPException(status_code=404, detail="Cluster not found")

    members = (
        db.query(ClusterMember)
        .filter(ClusterMember.cluster_id == cluster.id)
        .order_by(desc(ClusterMember.similarity))
        .limit(15)
        .all()
    )
    texts = []
    seen: set[str] = set()
    for m in members:
        item = db.get(FeedbackItem, m.feedback_id)
        if item and item.text not in seen:
            seen.add(item.text)
            texts.append(item.text)

    from pulse.ml.llm import generate_recommendation, generate_root_cause_analysis, get_llm_info
    recommendation = generate_recommendation(
        cluster.theme, cluster.severity_score, cluster.top_keywords or [], texts,
    )
    root_causes = generate_root_cause_analysis(
        cluster.theme, texts, cluster.top_keywords or [],
    )

    provider = get_llm_info()["provider"]
    if provider == "none":
        llm_fallback = (
            "Set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY in .env "
            "to enable AI-powered recommendations."
        )
    else:
        llm_fallback = (
            f"The configured {provider} call returned no result. "
            "Check server logs for API errors (invalid key, rate limit, or model name)."
        )

    return {
        "cluster": ClusterResponse.model_validate(cluster),
        "evidence": texts[:5],
        "recommendation": recommendation or llm_fallback,
        "root_cause_analysis": root_causes or llm_fallback,
    }


@router.get("/analysis/runs", response_model=list[AnalysisRunResponse], tags=["analysis"])
def list_runs(db: DB, org_id: CurrentOrgId, limit: int = 10):
    runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.org_id == org_id)
        .order_by(desc(AnalysisRun.created_at))
        .limit(limit)
        .all()
    )
    return [AnalysisRunResponse.model_validate(r) for r in runs]


# ── Insights ──────────────────────────────────────────────────────────────────


@router.get("/insights", response_model=list[InsightResponse], tags=["insights"])
def list_insights(
    db: DB,
    org_id: CurrentOrgId,
    limit: int = 20,
    type: str | None = None,
    unread_only: bool = False,
):
    items = get_insights(db, org_id, limit=limit, type_filter=type, unread_only=unread_only)
    return [InsightResponse.model_validate(i) for i in items]


@router.post("/insights/{insight_id}/read", tags=["insights"])
def read_insight(insight_id: str, db: DB, org_id: CurrentOrgId):
    mark_insight_read(db, insight_id)
    return {"ok": True}


# ── Artifacts ─────────────────────────────────────────────────────────────────


@router.get("/artifacts", response_model=list[ArtifactResponse], tags=["artifacts"])
def list_artifacts(
    db: DB,
    org_id: CurrentOrgId,
    type: str | None = None,
    limit: int = 20,
):
    query = db.query(Artifact).filter(Artifact.org_id == org_id)
    if type:
        query = query.filter(Artifact.type == type)
    items = query.order_by(desc(Artifact.created_at)).limit(limit).all()
    return [ArtifactResponse.model_validate(a) for a in items]


@router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse, tags=["artifacts"])
def get_artifact(artifact_id: str, db: DB, org_id: CurrentOrgId):
    artifact = db.get(Artifact, artifact_id)
    if not artifact or artifact.org_id != org_id:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactResponse.model_validate(artifact)


# ── Integrations ──────────────────────────────────────────────────────────────


@router.get("/integrations/available", response_model=list[ConnectorInfo], tags=["integrations"])
def available_connectors():
    return list_connectors()


@router.get("/integrations", response_model=list[SourceResponse], tags=["integrations"])
def list_sources(db: DB, org_id: CurrentOrgId):
    sources = (
        db.query(FeedbackSource)
        .filter(FeedbackSource.org_id == org_id)
        .order_by(desc(FeedbackSource.created_at))
        .all()
    )
    return [SourceResponse.model_validate(s) for s in sources]


@router.post("/integrations", response_model=SourceResponse, tags=["integrations"])
def create_source(body: CreateSourceRequest, db: DB, org_id: CurrentOrgId):
    source = FeedbackSource(
        org_id=org_id,
        type=body.type,
        name=body.name,
        config=body.config,
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return SourceResponse.model_validate(source)


@router.delete("/integrations/{source_id}", tags=["integrations"])
def delete_source(source_id: str, db: DB, org_id: CurrentOrgId):
    source = db.get(FeedbackSource, source_id)
    if not source or source.org_id != org_id:
        raise HTTPException(status_code=404, detail="Source not found")
    db.delete(source)
    db.commit()
    return {"ok": True}


# ── Trends ────────────────────────────────────────────────────────────────────


@router.get("/trends", tags=["trends"])
def get_trends(db: DB, org_id: CurrentOrgId, weeks: int = 12):
    return get_trend_data(db, org_id, weeks=weeks)
