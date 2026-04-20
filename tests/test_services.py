"""Integration tests for services using an in-memory SQLite DB."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pulse.database import Base
from pulse.models import FeedbackItem, FeedbackSource, Organisation
from pulse.services.ingestion import ingest_items
from pulse.services.intelligence import run_analysis


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def org(db):
    org = Organisation(name="Test Org", slug="test-org")
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def _sample_feedback():
    return [
        {"text": "The dashboard is extremely slow when loading reports"},
        {"text": "Dashboard loading performance is terrible"},
        {"text": "Dashboard takes forever to render"},
        {"text": "The app crashes every time I try to upload"},
        {"text": "Upload feature crashes the whole application"},
        {"text": "Love the new design, looks great!"},
        {"text": "The onboarding flow is excellent"},
    ]


def test_ingest_items_enriches_feedback(db, org):
    items = ingest_items(db, org.id, None, _sample_feedback())
    assert len(items) == 7
    for item in items:
        assert item.sentiment is not None
        assert item.category is not None
        assert item.embedding is not None


def test_run_analysis_creates_clusters(db, org):
    ingest_items(db, org.id, None, _sample_feedback())
    run = run_analysis(db, org.id, n_clusters=3)
    assert run.status == "completed"
    assert run.cluster_count >= 1
    assert run.feedback_count == 7
