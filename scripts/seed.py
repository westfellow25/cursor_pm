"""Seed script — generates 6 months of realistic feedback data for demo.

Run with:
    python -m scripts.seed

Creates a demo org, user, sources, and ~2000 feedback items spanning
multiple channels, segments, and categories with realistic temporal patterns.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.config import settings
from pulse.database import SessionLocal, init_db
from pulse.api.deps import hash_password
from pulse.models import (
    FeedbackItem,
    FeedbackSource,
    Organisation,
    User,
)
from pulse.ml.sentiment import enrich_feedback
from pulse.ml.embeddings import embed_texts_local

# ── Feedback templates ────────────────────────────────────────────────────────

PERFORMANCE = [
    "The dashboard takes forever to load when I have more than 50 reports",
    "Page load times are getting worse every week, especially the analytics page",
    "The app is extremely slow when loading monthly reports",
    "Dashboard keeps timing out when I try to view quarterly data",
    "Reports page takes 15+ seconds to render, completely unusable",
    "The latency on the search page is terrible, it hangs for 5-10 seconds",
    "Everything slows to a crawl when multiple users are on the dashboard",
    "The loading spinner never stops on the analytics overview page",
    "Performance has degraded significantly since the last update",
    "Our team can't use the dashboard during peak hours because it's so slow",
    "API response times have doubled in the last month",
    "The real-time charts are lagging by 30+ seconds",
    "Export takes 3 minutes for a simple CSV, this is unacceptable",
    "The notification system delays are getting longer and longer",
    "Scrolling through the feed is incredibly laggy on our dataset",
]

BUGS = [
    "The app crashes every time I try to upload a file larger than 10MB",
    "Login fails intermittently with a timeout error, very frustrating",
    "Data is showing incorrectly after the recent update — numbers don't match",
    "The export button generates an empty file, this has been broken for weeks",
    "Notifications are not being delivered for critical alerts",
    "The search function returns completely irrelevant results",
    "The integration sync failed and now we have duplicate records everywhere",
    "Charts are displaying the wrong date range no matter what I select",
    "The mobile app crashes on startup after the latest iOS update",
    "Two-factor authentication is broken — I can't log in to my account",
    "The webhook endpoint returns 500 errors about half the time",
    "Sorting by date doesn't work correctly — items appear in random order",
    "Copy-paste doesn't work in the editor, breaks the entire workflow",
    "The save button sometimes doesn't actually save — lost my work twice",
    "Email notifications are sent with wrong recipient names",
]

UX_ISSUES = [
    "The navigation is confusing — I can never find the settings page",
    "Too many clicks to get to the feature I use most",
    "The onboarding flow is overwhelming, too many steps for new users",
    "Why is the most important action hidden in a dropdown menu?",
    "The color scheme makes it hard to distinguish between different states",
    "There's no way to bulk-select items, I have to do everything one by one",
    "The mobile layout is completely broken on smaller phones",
    "I wish there was a keyboard shortcut for common actions",
    "The settings page is a maze — nothing is where I expect it to be",
    "The table headers don't stay fixed when scrolling, very annoying",
    "Can't customize the dashboard layout to show what's important to me",
    "The form validation messages are cryptic and unhelpful",
    "Dark mode has poor contrast in several places",
    "The breadcrumbs don't make any logical sense",
    "It's impossible to find anything without using the search bar",
]

FEATURE_REQUESTS = [
    "We really need a Salesforce integration for our sales team",
    "Please add the ability to schedule recurring reports",
    "Can you add support for SSO? Our IT team requires it for compliance",
    "We need better filtering options in the analytics dashboard",
    "Would love to see a Slack bot for real-time notifications",
    "Please add an API endpoint for custom webhook integrations",
    "We need multi-language support — our team is global",
    "A bulk import/export tool would save us hours every week",
    "Can you add role-based access control? Not everyone should see everything",
    "We desperately need an audit log for compliance requirements",
    "Please add the ability to annotate and comment on reports",
    "A mobile push notification feature would be incredibly useful",
    "We need a way to compare data across different time periods side by side",
    "Custom fields would let us track the metrics that matter to us",
    "Can you add a workflow automation builder? Similar to Zapier",
]

INTEGRATION = [
    "The Slack integration keeps disconnecting every few days",
    "Salesforce sync is breaking our data — duplicate contacts everywhere",
    "The API documentation is outdated, half the examples don't work",
    "Can't get the webhook to trigger reliably, missing events constantly",
    "HubSpot integration is missing key fields that we need",
    "The Jira integration creates tickets with wrong priority levels",
    "Data sync between our CRM and your platform has a 6-hour delay",
    "The Google Workspace integration doesn't support shared drives",
    "Zapier connection fails silently — no error messages at all",
    "The API rate limits are too restrictive for our enterprise use case",
    "OAuth tokens expire too quickly, our automation keeps breaking",
    "The SSO setup process is needlessly complicated",
]

PRAISE = [
    "Love the new analytics dashboard — it's exactly what we needed!",
    "The customer support team is absolutely fantastic, resolved my issue in minutes",
    "This tool has completely transformed how our product team makes decisions",
    "The onboarding experience was smooth and well-designed, great job!",
    "Impressed with how fast new features are being shipped",
    "The reporting capabilities are best-in-class compared to competitors",
    "Your API is a pleasure to work with — clean, well-documented, reliable",
    "The recent performance improvements are noticeable and appreciated",
    "Love the dark mode implementation, looks beautiful",
    "The team collaboration features have improved our workflow significantly",
]

SECURITY = [
    "We need SOC2 compliance documentation before we can proceed",
    "The password policy is too weak — no complexity requirements",
    "Found a potential XSS vulnerability in the comment section",
    "Need granular permission controls — currently it's all or nothing",
    "The session timeout is too short, I keep getting logged out during work",
    "Can you add IP whitelisting for API access?",
    "We need encryption at rest for all customer data, this is a blocker",
    "The audit log doesn't capture enough detail for our compliance needs",
]

ONBOARDING = [
    "The setup wizard asks too many questions upfront, just let me start using it",
    "No tutorial for advanced features — had to figure everything out myself",
    "The getting started guide is outdated and references old UI",
    "Would love interactive tooltips for first-time users",
    "It took our team 3 weeks to fully onboard, that's way too long",
    "The sample data doesn't represent a real use case at all",
    "There should be video tutorials for the more complex features",
]

ALL_TEMPLATES = {
    "performance": PERFORMANCE,
    "bug": BUGS,
    "ux": UX_ISSUES,
    "feature-request": FEATURE_REQUESTS,
    "integration": INTEGRATION,
    "praise": PRAISE,
    "security": SECURITY,
    "onboarding": ONBOARDING,
}

SEGMENTS = ["enterprise", "mid-market", "smb", "free"]
SEGMENT_WEIGHTS = [0.25, 0.35, 0.25, 0.15]
CHANNELS = ["web", "mobile", "api", "email", "chat", "intercom", "slack"]
CHANNEL_WEIGHTS = [0.25, 0.2, 0.1, 0.15, 0.1, 0.12, 0.08]

AUTHORS = [
    "Sarah Chen", "Mike Johnson", "Priya Patel", "David Kim", "Emma Wilson",
    "James Rodriguez", "Lisa Tanaka", "Alex Kumar", "Maria Santos", "Tom O'Brien",
    "Sophie Martin", "Raj Gupta", "Anna Kowalski", "Chris Lee", "Hannah Schmidt",
    "Kevin Nakamura", "Rachel Green", "Omar Hassan", "Julia Costa", "Daniel Park",
    "Nina Petrov", "Carlos Mendez", "Yuki Takahashi", "Ben Archer", "Leila Rahman",
]

# Category weights shift over time to simulate trends
def _category_weights(month_offset: int) -> dict[str, float]:
    """Categories shift in frequency over time to create realistic trends."""
    base = {
        "performance": 0.18, "bug": 0.15, "ux": 0.14, "feature-request": 0.16,
        "integration": 0.12, "praise": 0.10, "security": 0.08, "onboarding": 0.07,
    }
    # Performance complaints increase over time (simulating real degradation)
    if month_offset >= 3:
        base["performance"] += 0.08
        base["praise"] -= 0.04
        base["ux"] -= 0.04
    # Integration issues spike in month 4-5
    if 3 <= month_offset <= 4:
        base["integration"] += 0.06
        base["feature-request"] -= 0.06
    return base


def _random_variation(text: str) -> str:
    """Add slight random variation to template text."""
    variations = [
        ("", ""),
        (" — really need this fixed", ""),
        (" This is blocking our team.", ""),
        ("", " Please prioritize this."),
        ("", " We might have to look at alternatives."),
        ("", ""),
        ("", ""),
    ]
    prefix, suffix = random.choice(variations)
    return text + prefix + suffix


def main():
    print("Initialising database...")
    init_db()

    db = SessionLocal()

    # Check if seed data already exists
    existing = db.query(Organisation).filter(Organisation.slug == "acme-corp").first()
    if existing:
        print("Seed data already exists. Delete pulse.db to re-seed.")
        db.close()
        return

    print("Creating demo organisation and user...")

    org = Organisation(
        name="Acme Corp",
        slug="acme-corp",
        plan="growth",
        industry="saas",
        company_size="51-200",
    )
    db.add(org)
    db.flush()

    user = User(
        org_id=org.id,
        email="demo@acme.com",
        name="Demo User",
        password_hash=hash_password("demo1234"),
        role="owner",
    )
    db.add(user)
    db.flush()

    # Create sources
    sources = {}
    source_defs = [
        ("csv", "CSV Imports", 0),
        ("intercom", "Intercom", 0),
        ("slack", "Slack #product-feedback", 0),
        ("api", "REST API", 0),
    ]
    for stype, sname, _ in source_defs:
        source = FeedbackSource(org_id=org.id, type=stype, name=sname, status="active")
        db.add(source)
        db.flush()
        sources[stype] = source

    source_types = list(sources.keys())
    source_weights = [0.3, 0.3, 0.25, 0.15]

    print("Generating ~2000 feedback items over 6 months...")

    now = datetime.now(timezone.utc)
    items = []
    total_items = 2000

    for i in range(total_items):
        # Distribute items across 6 months with increasing volume
        month_offset = random.choices(range(6), weights=[0.1, 0.12, 0.15, 0.18, 0.2, 0.25])[0]
        day_in_month = random.randint(1, 28)
        hour = random.randint(6, 22)
        minute = random.randint(0, 59)
        created_at = now - timedelta(days=(5 - month_offset) * 30 + (28 - day_in_month), hours=24 - hour, minutes=60 - minute)

        # Select category based on temporal weights
        weights = _category_weights(month_offset)
        cats = list(weights.keys())
        cat_weights = [weights[c] for c in cats]
        category = random.choices(cats, weights=cat_weights)[0]

        # Select template
        templates = ALL_TEMPLATES[category]
        text = _random_variation(random.choice(templates))

        # Select other attributes
        source_type = random.choices(source_types, weights=source_weights)[0]
        segment = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        channel = random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]
        author = random.choice(AUTHORS)

        # Enrich
        enrichment = enrich_feedback(text)

        items.append(FeedbackItem(
            org_id=org.id,
            source_id=sources[source_type].id,
            text=text,
            author=author,
            author_segment=segment,
            channel=channel,
            sentiment=enrichment["sentiment"],
            urgency=enrichment["urgency"],
            category=enrichment["category"],
            subcategory=enrichment["subcategory"],
            meta={"month": month_offset, "plan": segment},
            created_at=created_at,
            ingested_at=created_at + timedelta(seconds=random.randint(1, 3600)),
        ))

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{total_items} items...")

    # Generate embeddings in batches
    print("Generating embeddings...")
    texts = [item.text for item in items]
    embeddings = embed_texts_local(texts, dim=256)
    for i, item in enumerate(items):
        item.embedding = embeddings[i].tolist()

    print("Saving to database...")
    db.add_all(items)

    # Update source counts
    for source in sources.values():
        count = sum(1 for item in items if item.source_id == source.id)
        source.items_synced = count
        source.last_sync_at = now

    db.commit()
    db.close()

    print(f"""
Seed complete!
  Organisation: Acme Corp
  User: demo@acme.com / demo1234
  Feedback items: {total_items}
  Sources: {len(sources)}
  Time span: 6 months

Start the server:
  uvicorn platform.main:app --reload

Then log in at http://localhost:8000 and run an analysis.
""")


if __name__ == "__main__":
    main()
