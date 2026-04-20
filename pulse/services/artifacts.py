"""Artifact generation service — PRDs, Jira tickets, roadmap items, executive summaries.

These are the deliverables that make the platform actionable. PMs can take these
directly to stakeholder meetings and sprint planning."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from pulse.models import Artifact, AnalysisRun, Cluster, ClusterMember, FeedbackItem

logger = logging.getLogger(__name__)


def _get_cluster_evidence(db: Session, cluster: Cluster, limit: int = 5) -> list[str]:
    """Get representative feedback quotes for a cluster."""
    members = (
        db.query(ClusterMember)
        .filter(ClusterMember.cluster_id == cluster.id)
        .order_by(ClusterMember.similarity.desc())
        .limit(limit)
        .all()
    )
    quotes = []
    for m in members:
        item = db.get(FeedbackItem, m.feedback_id)
        if item:
            quotes.append(item.text)
    return quotes


def generate_prd(db: Session, run: AnalysisRun) -> Artifact | None:
    """Generate a Product Requirements Document from analysis results."""
    clusters = (
        db.query(Cluster)
        .filter(Cluster.run_id == run.id)
        .order_by(Cluster.opportunity_score.desc())
        .all()
    )
    if not clusters:
        return None

    top = clusters[0]
    evidence = _get_cluster_evidence(db, top, limit=5)

    md = f"""# Product Requirements Document

## {top.label}: Product Improvement Specification

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Analysis Run**: {run.id}
**Data Points**: {run.feedback_count} feedback items analysed

---

## Problem Statement

Analysis of {run.feedback_count} customer feedback items reveals that **{top.label.lower()}** \
is the highest-impact opportunity, affecting {top.size} users ({top.frequency_score * 100:.0f}% of feedback). \
The opportunity score is **{top.opportunity_score}/10** based on frequency, severity, and sentiment analysis.

## Evidence

"""
    for i, quote in enumerate(evidence, 1):
        md += f"> {i}. \"{quote}\"\n\n"

    md += f"""## Key Metrics

| Metric | Value |
|--------|-------|
| Feedback Volume | {top.size} items |
| Frequency | {top.frequency_score * 100:.1f}% of total |
| Severity Score | {top.severity_score:.2f} |
| Avg. Sentiment | {top.sentiment_avg:.2f} |
| Opportunity Score | {top.opportunity_score}/10 |
| Top Keywords | {', '.join(top.top_keywords[:5])} |

## Proposed Solution

### Goals
1. Reduce negative feedback related to {top.label.lower()} by 40% within 90 days
2. Improve overall sentiment score from {top.sentiment_avg:.2f} to {min(0.5, (top.sentiment_avg or 0) + 0.3):.2f}
3. Decrease support tickets related to this theme by 25%

### Scope
- Address the core issues identified in the top feedback cluster
- Implement improvements iteratively with measurable checkpoints
- Track impact through continued feedback monitoring

### Out of Scope
- Complete product redesign
- Changes to other unrelated feature areas
- Infrastructure migration (unless directly required)

## Success Criteria
1. **Quantitative**: {top.size * 40 // 100}+ fewer negative feedback items on this theme in the next quarter
2. **Sentiment**: Average sentiment for this theme improves by 0.3+ points
3. **Engagement**: Task completion rate related to this feature improves by 15%+

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scope creep beyond identified theme | Strict theme-based acceptance criteria |
| Regression in other areas | Comprehensive monitoring of all clusters post-launch |
| Insufficient impact | A/B test approach with rollback capability |

"""

    # Add secondary opportunities
    if len(clusters) > 1:
        md += "## Other Opportunities\n\n"
        md += "| Rank | Theme | Score | Size | Sentiment |\n"
        md += "|------|-------|-------|------|-----------|\n"
        for i, c in enumerate(clusters[1:6], 2):
            md += f"| {i} | {c.label} | {c.opportunity_score} | {c.size} | {c.sentiment_avg:.2f} |\n"

    artifact = Artifact(
        org_id=run.org_id,
        run_id=run.id,
        type="prd",
        title=f"PRD: {top.label}",
        content=md,
        meta={"cluster_id": top.id, "opportunity_score": top.opportunity_score},
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def generate_jira_tickets(db: Session, run: AnalysisRun) -> Artifact | None:
    """Generate structured Jira tickets from the top opportunity."""
    clusters = (
        db.query(Cluster)
        .filter(Cluster.run_id == run.id)
        .order_by(Cluster.opportunity_score.desc())
        .all()
    )
    if not clusters:
        return None

    top = clusters[0]
    theme_lower = top.label.lower()

    md = f"""# Jira Tickets: {top.label}

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Source**: Analysis run {run.id} | {top.size} feedback items | Score {top.opportunity_score}/10

---

## Ticket 1: [Frontend] {top.label} — UX Improvements

**Type**: Story | **Priority**: High | **Points**: 8

**Description**:
Improve the user-facing experience related to {theme_lower}. Based on {top.size} pieces of user feedback,
the current experience causes friction and negative sentiment (avg: {top.sentiment_avg:.2f}).

**Acceptance Criteria**:
- [ ] Audit current UX flows related to {theme_lower}
- [ ] Implement top 3 user-requested improvements
- [ ] Add loading states and error handling
- [ ] Verify accessibility compliance
- [ ] User testing with 5+ participants confirms improvement

---

## Ticket 2: [Backend] {top.label} — Reliability & Performance

**Type**: Story | **Priority**: High | **Points**: 8

**Description**:
Address backend reliability and performance issues contributing to {theme_lower}
feedback. Severity score: {top.severity_score:.2f}.

**Acceptance Criteria**:
- [ ] Profile and identify top 3 performance bottlenecks
- [ ] Implement caching strategy where applicable
- [ ] Add retry logic and graceful degradation
- [ ] Reduce p95 response time by 30%+
- [ ] Zero increase in error rate post-deployment

---

## Ticket 3: [Analytics] Impact Tracking Instrumentation

**Type**: Task | **Priority**: Medium | **Points**: 5

**Description**:
Instrument analytics to measure the impact of {theme_lower} improvements.
Track before/after metrics to validate the investment.

**Acceptance Criteria**:
- [ ] Track key user actions related to {theme_lower}
- [ ] Set up sentiment monitoring dashboard for this theme
- [ ] Configure alerts for regression detection
- [ ] Create weekly automated impact report
- [ ] Baseline metrics documented before changes ship

---

## Ticket 4: [QA] End-to-End Validation Plan

**Type**: Task | **Priority**: Medium | **Points**: 5

**Description**:
Create comprehensive test plan for {theme_lower} improvements.
Cover golden path, edge cases, and regression scenarios.

**Acceptance Criteria**:
- [ ] Test plan covers all user journeys related to the theme
- [ ] Automated regression tests for critical paths
- [ ] Performance benchmark tests established
- [ ] Cross-browser / cross-device validation
- [ ] Load test under realistic traffic patterns

---

## Ticket 5: [Rollout] Phased Launch & Monitoring

**Type**: Task | **Priority**: Medium | **Points**: 3

**Description**:
Plan and execute phased rollout of {theme_lower} improvements with
monitoring and rollback capability.

**Acceptance Criteria**:
- [ ] Feature flagged for gradual rollout (10% → 50% → 100%)
- [ ] Monitoring dashboards live before rollout begins
- [ ] Rollback procedure documented and tested
- [ ] Success criteria defined for each phase gate
- [ ] Post-launch retrospective scheduled

"""

    artifact = Artifact(
        org_id=run.org_id,
        run_id=run.id,
        type="jira_tickets",
        title=f"Jira Tickets: {top.label}",
        content=md,
        meta={"cluster_id": top.id, "ticket_count": 5},
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def generate_executive_summary(db: Session, run: AnalysisRun) -> Artifact | None:
    """Generate an executive summary for leadership."""
    clusters = (
        db.query(Cluster)
        .filter(Cluster.run_id == run.id)
        .order_by(Cluster.opportunity_score.desc())
        .all()
    )
    if not clusters:
        return None

    total_items = run.feedback_count
    top3 = clusters[:3]

    md = f"""# Executive Summary: Voice of Customer Analysis

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Period**: Latest analysis | **Data Points**: {total_items} feedback items

---

## Key Findings

We analysed **{total_items} customer feedback items** across all channels and identified
**{len(clusters)} distinct themes**. Here are the top priorities:

"""
    for i, c in enumerate(top3, 1):
        sentiment_label = "Negative" if (c.sentiment_avg or 0) < -0.2 else "Neutral" if (c.sentiment_avg or 0) < 0.2 else "Positive"
        md += f"""### {i}. {c.label} (Score: {c.opportunity_score}/10)
- **Volume**: {c.size} items ({c.frequency_score * 100:.0f}% of feedback)
- **Sentiment**: {sentiment_label} ({c.sentiment_avg:.2f})
- **Severity**: {c.severity_score:.2f}
- **Keywords**: {', '.join(c.top_keywords[:5])}

"""

    md += """## Recommended Actions

| Priority | Theme | Action | Expected Impact |
|----------|-------|--------|-----------------|
"""
    for i, c in enumerate(top3, 1):
        md += f"| P{i} | {c.label} | Address top feedback drivers | Reduce negative feedback by ~40% |\n"

    md += f"""
## Overall Health

| Metric | Value |
|--------|-------|
| Total Feedback | {total_items} |
| Themes Identified | {len(clusters)} |
| Average Opportunity Score | {sum(c.opportunity_score for c in clusters) / len(clusters):.1f}/10 |
| Critical Themes | {sum(1 for c in clusters if c.severity_score >= 0.6)} |
| Positive Themes | {sum(1 for c in clusters if (c.sentiment_avg or 0) > 0.2)} |

"""

    artifact = Artifact(
        org_id=run.org_id,
        run_id=run.id,
        type="executive_summary",
        title="Executive Summary: Voice of Customer",
        content=md,
        meta={"cluster_count": len(clusters), "feedback_count": total_items},
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact
