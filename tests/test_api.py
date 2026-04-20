"""Integration tests for the FastAPI /analyze endpoint and its downloads."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.main import app  # noqa: E402


client = TestClient(app)


def _analyze(csv_bytes: bytes, filename: str = "feedback.csv") -> dict:
    response = client.post(
        "/analyze",
        files={"file": (filename, csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_analyze_legacy_feedback_column() -> None:
    csv_bytes = (ROOT / "data" / "sample_feedback.csv").read_bytes()
    data = _analyze(csv_bytes)

    assert data["run_id"]
    assert data["clusters_summary"], "expected at least one cluster"
    assert data["recommended_action"]
    assert data["prd_text"].startswith("# Product Requirements Document")
    assert data["jira_tickets_text"].startswith("# Jira Tickets")
    # The legacy dataset is about onboarding/Slack/exports. The top cluster must
    # not be wedged into a hardcoded "Dashboard performance" theme anymore.
    assert "Dashboard" not in data["clusters_summary"][0]["theme"]


def test_analyze_new_schema_and_download_endpoints() -> None:
    csv_bytes = (ROOT / "example_data" / "feedback.csv").read_bytes()
    data = _analyze(csv_bytes)
    run_id = data["run_id"]

    prd = client.get(f"/download/prd/{run_id}")
    jira = client.get(f"/download/jira/{run_id}")
    assert prd.status_code == 200
    assert jira.status_code == 200
    assert prd.text.startswith("# Product Requirements Document")
    assert jira.text.startswith("# Jira Tickets")


def test_download_returns_404_for_unknown_run() -> None:
    response = client.get("/download/prd/does-not-exist")
    assert response.status_code == 404


def test_analyze_rejects_missing_text_column() -> None:
    bad_csv = b"title,body\nfoo,bar\n"
    response = client.post(
        "/analyze",
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower()


def test_concurrent_runs_do_not_clobber_each_other() -> None:
    # Two different uploads must produce two independently downloadable run_ids.
    first = _analyze((ROOT / "data" / "sample_feedback.csv").read_bytes(), filename="a.csv")
    second = _analyze((ROOT / "example_data" / "feedback.csv").read_bytes(), filename="b.csv")

    assert first["run_id"] != second["run_id"]
    first_prd = client.get(f"/download/prd/{first['run_id']}").text
    second_prd = client.get(f"/download/prd/{second['run_id']}").text
    assert first_prd == first["prd_text"]
    assert second_prd == second["prd_text"]
