"""Generate PRD and Jira ticket artifacts from feedback CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from e2e_demo import (
    _feature_name,
    _problem_statement,
    _risks,
    _success_metrics,
    _target_users,
    analyze_feedback,
)


def _build_prd_markdown(top: dict[str, object], total_records: int, proposed_solution: str) -> str:
    lines = [
        "# Product Requirements Document",
        "",
        f"## Feature: {_feature_name(str(top['theme_label']))}",
        "",
        "## Problem Summary",
        _problem_statement(list(top["texts"])),
        "",
        "## Target Users",
        _target_users(top, total_records),
        "",
        "## Proposed Solution",
        proposed_solution,
        "",
        "## Context",
        f"- Theme label: {top['theme_label']}",
        f"- Cluster ID: {top['cluster_id']}",
        f"- Opportunity score: {top['opportunity_score']}",
        f"- Frequency: {top['frequency']} / {total_records} feedback items",
        f"- Severity: {top['severity']}",
        "",
        "## Success Metrics",
    ]
    lines.extend([f"- {metric}" for metric in _success_metrics(top)])
    lines.append("")
    lines.append("## Risks")
    lines.extend([f"- {risk}" for risk in _risks(top)])
    lines.append("")
    return "\n".join(lines)


def _build_tickets_markdown(tickets: list[dict[str, str]]) -> str:
    lines = ["# Jira Tickets", ""]
    for idx, ticket in enumerate(tickets, start=1):
        lines.extend(
            [
                f"## Ticket {idx}: {ticket['title']}",
                "",
                "**Description**",
                ticket["description"],
                "",
                "**Acceptance Criteria**",
                ticket["acceptance"],
                "",
            ]
        )
    return "\n".join(lines)


def build_artifact_content(analysis: dict[str, object]) -> tuple[str, str]:
    top = analysis["top"]
    if not top:
        raise RuntimeError("No non-misc clusters found, unable to generate artifacts.")

    prd_content = _build_prd_markdown(top, len(analysis["records"]), analysis["proposed_solution"])
    tickets_content = _build_tickets_markdown(analysis["tickets"])
    return prd_content, tickets_content


def generate_artifacts(csv_path: str | Path = "example_data/feedback.csv", n_clusters: int = 3) -> tuple[Path, Path]:
    analysis = analyze_feedback(csv_path=csv_path, n_clusters=n_clusters)

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    prd_path = docs_dir / "PRD.md"
    tickets_path = docs_dir / "jira_tickets.md"

    prd_content, tickets_content = build_artifact_content(analysis)

    prd_path.write_text(prd_content, encoding="utf-8", newline="\n")
    tickets_path.write_text(tickets_content, encoding="utf-8", newline="\n")
    return prd_path, tickets_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRD and Jira ticket artifacts from feedback CSV.")
    parser.add_argument(
        "--csv",
        default="example_data/feedback.csv",
        help="Path to feedback CSV (default: example_data/feedback.csv)",
    )
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters to request from the pipeline")
    args = parser.parse_args()

    prd_path, tickets_path = generate_artifacts(csv_path=args.csv, n_clusters=args.clusters)
    print(f"Generated: {prd_path}")
    print(f"Generated: {tickets_path}")


if __name__ == "__main__":
    main()
