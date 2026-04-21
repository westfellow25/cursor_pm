"""Seed script — generates 6 months of realistic feedback data for demo.

Run with:  python -m scripts.seed

Creates a demo org, user, sources, and ~2500 feedback items spanning
multiple channels, segments, and categories with realistic temporal patterns.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.config import settings
from pulse.database import SessionLocal, init_db
from pulse.api.deps import hash_password
from pulse.models import FeedbackItem, FeedbackSource, Organisation, User
from pulse.ml.sentiment import enrich_feedback
from pulse.ml.embeddings import embed_texts_local

# ── 500+ Feedback templates across 10 categories ─────────────────────────────

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
    "Auto-complete in the search bar takes 4-5 seconds to show suggestions",
    "The pipeline processing view locks up the browser for large datasets",
    "Switching between tabs in the analytics section causes visible lag",
    "Image thumbnails in the gallery section take ages to load",
    "The calendar view becomes unresponsive when displaying a full year",
    "Database queries behind the team overview are painfully slow",
    "Opening a project with more than 100 tasks freezes the UI for 10 seconds",
    "The PDF export has gotten noticeably slower over the past three months",
    "Drag and drop in the kanban board stutters badly on boards with 50+ cards",
    "The global search indexing seems broken — results lag behind by hours",
    "Rendering complex charts with 1000+ data points crashes the tab",
    "Switching workspaces takes 8+ seconds, losing my train of thought",
    "Batch operations on 500+ items cause the whole page to freeze",
    "The activity feed is unbelievably slow if you have more than a week of data",
    "Every page transition shows a blank screen for 2-3 seconds now",
    "Server response times during our morning standup window are terrible",
    "Loading the customer segment report takes so long people give up",
    "The rich text editor slows to a crawl on longer documents",
    "Video playback in the knowledge base stutters constantly",
    "Typeahead autocomplete has been broken since the last release — huge lag",
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
    "The invoice PDF generator creates blank pages for some line items",
    "Markdown rendering in comments strips out all links",
    "File attachments silently fail when the filename contains special characters",
    "The time zone conversion is off by one hour for users in Australia",
    "Clicking 'Undo' actually deletes the item instead of reverting the change",
    "The password reset flow sends you back to an expired token every time",
    "Dropdown menus clip behind modals on Firefox",
    "The bulk delete operation only removes the first 25 items silently",
    "CSV import maps columns incorrectly when headers contain spaces",
    "The auto-save feature conflicts with manual saves, overwriting changes",
    "Date pickers show the wrong month when crossing a year boundary",
    "Team mentions in comments don't trigger the notification at all",
    "The recurring task scheduler skips weekends even when it shouldn't",
    "Floating point rounding errors show $0.01 discrepancies in invoices",
    "The print view is completely broken — content overflows off the page",
    "Image uploads succeed but display a broken icon until you refresh",
    "The 'Remember me' checkbox on login does absolutely nothing",
    "Tags disappear when you edit an item and save without changing them",
    "The API returns a 200 status code even when the operation actually fails",
    "Concurrent edits silently overwrite each other with no conflict resolution",
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
    "Error messages just say 'Something went wrong' with no details",
    "The back button behavior is unpredictable — sometimes it loses my state",
    "There's no confirmation dialog before destructive actions like deleting a project",
    "The tooltip text is cut off on smaller monitors",
    "I can't resize table columns which makes long text unreadable",
    "The file manager feels like it's from 2005 — no drag and drop, no thumbnails",
    "Pagination resets to page 1 every time I apply a filter",
    "The empty states are unhelpful — they just show a blank area with no guidance",
    "Tab order in forms jumps around randomly, making keyboard use impossible",
    "There's no visual hierarchy on the dashboard — everything has the same weight",
    "The notification center opens in a new page instead of a popover",
    "Compact mode removes too much whitespace, making everything feel cramped",
    "The color-coded status labels are indistinguishable for colorblind users",
    "Switching between list and grid view loses my scroll position",
    "The onboarding checklist can't be dismissed, keeps appearing on every login",
    "Date inputs don't respect my locale — I keep getting MM/DD when I expect DD/MM",
    "The help center link opens in the same tab, losing my unsaved work",
    "Multi-select fields don't show how many items are selected when collapsed",
    "The chart legends overlap with the data on narrow screens",
    "There's no way to duplicate a template, I have to recreate from scratch every time",
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
    "Would love the ability to set up custom alerts based on thresholds",
    "We need white-labeling options for our client-facing reports",
    "Can you add a Gantt chart view for project timelines?",
    "Need the ability to @mention teammates directly in item comments",
    "It would be great to have saved search queries I can reuse",
    "We need custom user groups, not just the three default roles",
    "Please add support for markdown in all text fields",
    "We'd love a public API for building our own dashboards",
    "An in-app survey builder would let us collect feedback without third-party tools",
    "Need the ability to create custom views and share them with the team",
    "Can we get recurring task templates? We repeat the same sprint rituals every two weeks",
    "Time tracking built into task cards would replace our Toggl subscription",
    "A changelog page we can share with customers would be amazing",
    "Please add conditional logic to form fields — show/hide based on selection",
    "We need data retention policies to auto-archive items after 90 days",
    "Would love deeper analytics — cohort analysis, funnels, retention curves",
    "The ability to duplicate projects with all their settings intact",
    "Can you add approval workflows? We need manager sign-off before publishing",
    "Native Google Calendar sync instead of just iCal export",
    "A read-only guest link for sharing dashboards with external stakeholders",
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
    "The Stripe billing sync double-counts refunded transactions",
    "Linear integration doesn't map custom fields — only title and status come through",
    "The GitHub commit linking only works for the default branch",
    "Zendesk ticket sync is one-directional — changes here don't reflect back",
    "The Notion import mangles tables and nested bullet points",
    "Calendar sync creates duplicate events when you update a meeting",
    "The Figma embed viewer doesn't support components with variants",
    "Intercom data import maps all conversations to a single user",
    "The REST API doesn't support pagination cursors, only offset",
    "Webhook payloads are missing the 'updated_by' field",
    "The Microsoft Teams integration requires admin-level permissions for basic use",
    "SAML configuration fails silently if the IdP metadata URL has a redirect",
    "The Twilio SMS integration truncates messages over 160 characters",
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
    "The new notification preferences are exactly what I wanted — finally!",
    "Your support engineer Maria went above and beyond to help us migrate",
    "Honestly the best project management tool we've ever used as a team",
    "The keyboard shortcuts make me so much more productive, thank you",
    "The release notes are incredibly detailed — shows you care about transparency",
    "Just renewed for another year. This tool has become essential for our team",
    "The custom report builder is phenomenal. Replaced three separate tools for us",
    "Migrating from Asana was painless thanks to your import tool",
    "The real-time collaboration features rival Google Docs — impressive",
    "The mobile app is surprisingly full-featured, not a watered down version",
    "Your uptime has been flawless all quarter. Reliability matters and you deliver",
    "The recent UI refresh looks so much cleaner. Modern without being trendy",
    "CSV export works perfectly with our downstream data pipeline",
    "The granular permissions saved us from a compliance nightmare. Thank you",
    "Response time from the support team is consistently under 2 hours. Amazing",
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
    "The data export includes personally identifiable information without masking",
    "Shared links don't expire — anyone with the URL can access the data forever",
    "There's no way to enforce MFA for the entire organization",
    "User accounts aren't automatically deprovisioned when removed from our IdP",
    "File uploads aren't scanned for malware before being stored",
    "The API keys are visible in plaintext in the settings page",
    "GDPR data deletion request takes 30+ days to process, that's not compliant",
    "Can't restrict data access by geographic region for data sovereignty",
    "Inactive user sessions stay alive for 30 days — that's a security risk",
    "Webhook endpoints accept requests without signature verification",
    "The password reset link doesn't expire quickly enough — 72 hours is too long",
    "There's no login attempt throttling — brute force attacks are possible",
]

ONBOARDING = [
    "The setup wizard asks too many questions upfront, just let me start using it",
    "No tutorial for advanced features — had to figure everything out myself",
    "The getting started guide is outdated and references old UI",
    "Would love interactive tooltips for first-time users",
    "It took our team 3 weeks to fully onboard, that's way too long",
    "The sample data doesn't represent a real use case at all",
    "There should be video tutorials for the more complex features",
    "The welcome email links to a 404 page",
    "No one on my team understood the difference between projects and workspaces",
    "The import from CSV wizard crashed on our 5000-row file",
    "There's no sandbox or test environment to try things without breaking production",
    "The terminology in the app is confusing — 'items' vs 'tasks' vs 'tickets'",
    "First-time setup should suggest sensible defaults based on team size",
    "The demo video on the landing page shows features that don't exist yet",
    "Invite flow is confusing — my teammates couldn't find where to accept",
]

PRICING = [
    "The per-seat pricing doesn't work for our team model — we have many part-time users",
    "We'd love a startup discount program, the current pricing is steep for seed stage",
    "The jump from the Growth plan to Enterprise is too big — need something in between",
    "Hidden overage charges caught us off guard this month",
    "The free tier is too limited to properly evaluate the product",
    "Billing page doesn't show what you're actually paying for — no line items",
    "We need annual billing with a discount option, not just monthly",
    "The pricing page doesn't list what's included in each plan clearly",
    "Why do read-only viewers count as full seats? That's not fair pricing",
    "We were charged for deactivated users — that shouldn't happen",
    "The trial period is too short to run a proper evaluation with the team",
    "Volume discounts should kick in earlier — 100 seats is a high threshold",
]

DATA = [
    "The analytics dashboard doesn't let me drill down into individual data points",
    "Custom report builder is limited — can't create pivot tables or cross-tab views",
    "Data refresh intervals are too long — we need near real-time for our ops team",
    "The chart builder doesn't support scatter plots or heatmaps",
    "No way to create calculated metrics or derived fields",
    "Historical data only goes back 90 days — we need at least a year for trends",
    "The data export is capped at 10,000 rows which isn't enough for our analysis",
    "Can't schedule automated report delivery to Slack or email",
    "The cohort analysis feature is too simplistic — no custom date ranges",
    "Anomaly detection should highlight outliers automatically",
    "The funnel visualization doesn't support custom step ordering",
    "We need the ability to compare metrics across different customer segments",
    "The SQL query mode is great but has a 30-second timeout — too short for complex queries",
    "Dashboard sharing creates a static snapshot, not a live view",
    "No support for geospatial data visualization — we need map charts",
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
    "pricing": PRICING,
    "data": DATA,
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
    "Aisha Mohammed", "Marcus Webb", "Elena Volkov", "Diego Vargas", "Mia Foster",
    "Nathan Brooks", "Fatima Al-Sharif", "Lucas Bergmann", "Grace Okafor", "Ian MacLeod",
    "Chloe Dupont", "Ravi Sharma", "Olivia Chang", "Samuel Katz", "Hana Yoshida",
]

COMPANIES = [
    "TechFlow", "DataBridge", "ScaleUp AI", "Nexus Labs", "CloudFirst",
    "Quantum Ops", "PrimeStack", "SynapseIO", "VelocityHQ", "ClearPath",
    "Beacon Health", "TradeWind", "FintechPro", "EduSync", "RetailEdge",
]


def _category_weights(month_offset: int) -> dict[str, float]:
    base = {
        "performance": 0.15, "bug": 0.14, "ux": 0.12, "feature-request": 0.14,
        "integration": 0.10, "praise": 0.10, "security": 0.07, "onboarding": 0.06,
        "pricing": 0.06, "data": 0.06,
    }
    if month_offset >= 3:
        base["performance"] += 0.07
        base["praise"] -= 0.03
        base["ux"] -= 0.02
        base["onboarding"] -= 0.02
    if 3 <= month_offset <= 4:
        base["integration"] += 0.05
        base["feature-request"] -= 0.03
        base["pricing"] -= 0.02
    if month_offset >= 4:
        base["security"] += 0.03
        base["data"] += 0.02
        base["praise"] -= 0.03
        base["onboarding"] -= 0.02
    return base


def main():
    print("Initialising database...")
    init_db()

    db = SessionLocal()
    existing = db.query(Organisation).filter(Organisation.slug == "acme-corp").first()
    if existing:
        print("Seed data already exists. Delete pulse.db to re-seed.")
        db.close()
        return

    print("Creating demo organisation and user...")
    org = Organisation(name="Acme Corp", slug="acme-corp", plan="growth", industry="saas", company_size="51-200")
    db.add(org)
    db.flush()

    user = User(org_id=org.id, email="demo@acme.com", name="Demo User",
                password_hash=hash_password("demo1234"), role="owner")
    db.add(user)
    db.flush()

    sources = {}
    for stype, sname in [("csv", "CSV Imports"), ("intercom", "Intercom Conversations"),
                          ("slack", "Slack #product-feedback"), ("api", "REST API Submissions"),
                          ("intercom", "Intercom NPS Surveys")]:
        s = FeedbackSource(org_id=org.id, type=stype, name=sname, status="active")
        db.add(s)
        db.flush()
        sources[s.id] = s

    source_ids = list(sources.keys())

    total_items = 2500
    print(f"Generating {total_items} feedback items over 6 months ({sum(len(v) for v in ALL_TEMPLATES.values())} unique templates)...")

    now = datetime.now(timezone.utc)
    items = []

    for i in range(total_items):
        month_offset = random.choices(range(6), weights=[0.08, 0.10, 0.14, 0.18, 0.22, 0.28])[0]
        day = random.randint(1, 28)
        hour = random.randint(6, 23)
        minute = random.randint(0, 59)
        created_at = now - timedelta(days=(5 - month_offset) * 30 + (28 - day), hours=24 - hour, minutes=60 - minute)

        weights = _category_weights(month_offset)
        cats = list(weights.keys())
        category = random.choices(cats, weights=[weights[c] for c in cats])[0]

        text = random.choice(ALL_TEMPLATES[category])
        segment = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        channel = random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]
        author = random.choice(AUTHORS)
        company = random.choice(COMPANIES)
        source_id = random.choice(source_ids)

        nps = None
        if random.random() < 0.3:
            if category == "praise":
                nps = random.choices([9, 10, 8, 7], weights=[0.4, 0.3, 0.2, 0.1])[0]
            elif category in ("bug", "performance"):
                nps = random.choices([1, 2, 3, 4, 5], weights=[0.15, 0.2, 0.25, 0.25, 0.15])[0]
            else:
                nps = random.choices(range(1, 11), weights=[0.05, 0.05, 0.08, 0.1, 0.12, 0.15, 0.15, 0.12, 0.1, 0.08])[0]

        enrichment = enrich_feedback(text)

        items.append(FeedbackItem(
            org_id=org.id,
            source_id=source_id,
            text=text,
            author=author,
            author_segment=segment,
            channel=channel,
            sentiment=enrichment["sentiment"],
            urgency=enrichment["urgency"],
            category=enrichment["category"],
            subcategory=enrichment["subcategory"],
            meta={"month": month_offset, "plan": segment, "company": company, **({"nps": nps} if nps else {})},
            created_at=created_at,
            ingested_at=created_at + timedelta(seconds=random.randint(1, 3600)),
        ))

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{total_items} items...")

    print("Generating embeddings...")
    texts = [item.text for item in items]
    embeddings = embed_texts_local(texts, dim=256)
    for idx, item in enumerate(items):
        item.embedding = embeddings[idx].tolist()

    print("Saving to database...")
    db.add_all(items)

    for s in sources.values():
        s.items_synced = sum(1 for item in items if item.source_id == s.id)
        s.last_sync_at = now

    db.commit()
    db.close()

    unique_texts = len(set(texts))
    print(f"""
Seed complete!
  Organisation: Acme Corp
  User: demo@acme.com / demo1234
  Feedback items: {total_items} ({unique_texts} unique texts)
  Sources: {len(sources)}
  Templates: {sum(len(v) for v in ALL_TEMPLATES.values())}
  Time span: 6 months

Start the server:
  uvicorn pulse.main:app --reload
""")


if __name__ == "__main__":
    main()
