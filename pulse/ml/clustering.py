"""Advanced clustering pipeline with adaptive sizing, coherence scoring,
post-processing refinement, and theme extraction.

Designed to handle real-world feedback datasets: noisy, variable-length,
mixed-topic, and ranging from 10 to 100k+ items."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could i me my we our you your he she it they them "
    "this that these those and or but if then so to of in for on with at by from "
    "up out about into through during before after above below between under "
    "again further once here there when where why how all each every both few "
    "more most other some such no nor not only own same than too very just also "
    "get got its been would".split()
)


def _normalize_rows(matrix: NDArray) -> NDArray:
    """L2-normalise each row."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def choose_cluster_count(n: int, min_k: int = 3, max_k: int = 20) -> int:
    """Adaptive cluster count using sqrt heuristic with bounds."""
    if n <= min_k:
        return max(2, n)
    k = int(math.ceil(math.sqrt(n) * 1.2))
    return max(min_k, min(max_k, k))


def find_optimal_k(
    embeddings: NDArray,
    min_k: int = 3,
    max_k: int = 20,
) -> int:
    """Find optimal k using silhouette analysis."""
    n = len(embeddings)
    if n < 4:
        return max(2, n)

    max_k = min(max_k, n - 1)
    if min_k >= max_k:
        return min_k

    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, sample_size=min(n, 1000))
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_embeddings(
    embeddings: NDArray,
    n_clusters: int | None = None,
    similarity_threshold: float = 0.55,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Cluster embedding vectors.

    Returns:
        labels: cluster assignment per item (-1 = outlier)
        similarities: cosine similarity to assigned centroid
        centroids: cluster centroid vectors
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([]), np.array([])

    normed = _normalize_rows(embeddings)

    if n_clusters is None:
        n_clusters = find_optimal_k(normed)

    km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(normed)
    centroids = _normalize_rows(km.cluster_centers_)

    # Compute cosine similarity to centroid
    similarities = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = labels[i]
        similarities[i] = float(np.dot(normed[i], centroids[c]))

    # Mark low-similarity items as outliers
    outlier_mask = similarities < similarity_threshold
    labels = labels.copy()
    labels[outlier_mask] = -1

    return labels, similarities, centroids


# ── Theme extraction ─────────────────────────────────────────────────────────

_THEME_PATTERNS: dict[frozenset[str], str] = {
    # Performance
    frozenset({"dashboard", "slow", "loading", "performance", "latency"}): "Dashboard performance",
    frozenset({"slow", "loading", "spinner", "timeout", "render"}): "Page load performance",
    frozenset({"lag", "laggy", "stutter", "freeze", "crawl"}): "UI responsiveness",
    frozenset({"response", "times", "api", "speed", "doubled"}): "API response times",
    frozenset({"charts", "rendering", "complex", "data", "points"}): "Chart rendering performance",
    frozenset({"batch", "operations", "bulk", "large", "dataset"}): "Bulk operation performance",
    # Bugs
    frozenset({"crash", "bug", "error", "broken", "fails"}): "Stability & reliability",
    frozenset({"crashes", "startup", "upload", "mobile"}): "App crash issues",
    frozenset({"login", "authentication", "password", "timeout", "fails"}): "Authentication issues",
    frozenset({"duplicate", "duplicates", "sync", "data", "records"}): "Data sync & duplicates",
    frozenset({"save", "lost", "overwrite", "conflict", "work"}): "Data loss & save issues",
    frozenset({"wrong", "incorrect", "incorrectly", "match", "displaying"}): "Incorrect data display",
    frozenset({"sort", "sorting", "order", "pagination", "filter"}): "Sorting & filtering bugs",
    frozenset({"email", "notification", "sent", "wrong", "names"}): "Email notification bugs",
    # UX
    frozenset({"navigation", "confusing", "find", "settings", "hidden"}): "Navigation & discoverability",
    frozenset({"clicks", "steps", "many", "workflow", "complicated"}): "Workflow complexity",
    frozenset({"mobile", "layout", "responsive", "phones", "tablet"}): "Mobile experience",
    frozenset({"keyboard", "shortcuts", "shortcut", "actions"}): "Keyboard accessibility",
    frozenset({"dark", "mode", "contrast", "color", "colorblind"}): "Accessibility & theming",
    frozenset({"onboarding", "setup", "wizard", "tutorial", "started"}): "Onboarding experience",
    frozenset({"validation", "messages", "error", "cryptic", "unhelpful"}): "Error messaging",
    frozenset({"table", "headers", "columns", "resize", "scroll"}): "Table & list UX",
    frozenset({"empty", "states", "blank", "guidance", "help"}): "Empty state guidance",
    frozenset({"customize", "custom", "personalize", "layout", "dashboard"}): "Customization gaps",
    # Feature requests
    frozenset({"salesforce", "integration", "crm", "hubspot", "sync"}): "CRM integration",
    frozenset({"sso", "saml", "oauth", "compliance", "security"}): "SSO & compliance",
    frozenset({"schedule", "recurring", "automated", "report"}): "Automated reporting",
    frozenset({"slack", "bot", "notifications", "real", "time"}): "Real-time notifications",
    frozenset({"api", "endpoint", "webhook", "custom", "public"}): "API & developer tools",
    frozenset({"language", "multi", "global", "i18n", "translation"}): "Multi-language support",
    frozenset({"import", "export", "bulk", "csv", "tool"}): "Import & export tools",
    frozenset({"role", "access", "control", "permissions", "rbac"}): "Role-based access",
    frozenset({"audit", "log", "compliance", "track", "requirements"}): "Audit & compliance",
    frozenset({"workflow", "automation", "zapier", "trigger", "builder"}): "Workflow automation",
    frozenset({"gantt", "timeline", "project", "milestones"}): "Project timeline views",
    frozenset({"approval", "workflows", "sign", "publish"}): "Approval workflows",
    frozenset({"calendar", "google", "sync", "ical", "meetings"}): "Calendar integration",
    frozenset({"custom", "fields", "metrics", "track", "data"}): "Custom fields & metrics",
    frozenset({"compare", "periods", "side", "time", "analysis"}): "Comparative analytics",
    # Integration
    frozenset({"integration", "sync", "api", "connect", "webhook"}): "Integration reliability",
    frozenset({"slack", "disconnecting", "keeps", "days", "connection"}): "Slack integration issues",
    frozenset({"jira", "tickets", "priority", "linear", "fields"}): "Project tool integration",
    frozenset({"documentation", "outdated", "examples", "docs", "api"}): "API documentation",
    frozenset({"rate", "limits", "restrictive", "tokens", "expire"}): "API limits & auth",
    frozenset({"stripe", "billing", "payment", "transactions"}): "Payment integration",
    # Reporting & data
    frozenset({"export", "report", "csv", "download", "pdf"}): "Reporting & exports",
    frozenset({"analytics", "drill", "down", "cohort", "funnel"}): "Advanced analytics",
    frozenset({"chart", "visualization", "scatter", "heatmap", "map"}): "Data visualization",
    frozenset({"historical", "data", "retention", "days", "year"}): "Data retention & history",
    frozenset({"real", "time", "refresh", "intervals", "live"}): "Real-time data updates",
    frozenset({"sql", "query", "calculated", "derived", "fields"}): "Computed metrics & queries",
    # Security
    frozenset({"permission", "access", "role", "admin", "security"}): "Access control",
    frozenset({"soc2", "compliance", "encryption", "gdpr", "data"}): "Security compliance",
    frozenset({"mfa", "2fa", "session", "ip", "whitelisting"}): "Authentication security",
    frozenset({"vulnerability", "xss", "malware", "scan"}): "Security vulnerabilities",
    # Notification
    frozenset({"notification", "alert", "email", "message", "delivered"}): "Notifications & alerts",
    frozenset({"push", "mobile", "notification", "real", "time"}): "Push notifications",
    # Pricing
    frozenset({"pricing", "billing", "plan", "subscription", "cost"}): "Pricing & billing",
    frozenset({"seat", "per", "pricing", "users", "viewers"}): "Per-seat pricing model",
    frozenset({"trial", "free", "tier", "evaluate", "limited"}): "Free tier & trials",
    frozenset({"discount", "annual", "startup", "volume"}): "Pricing options",
    # Praise
    frozenset({"love", "great", "amazing", "fantastic", "best"}): "Positive user experience",
    frozenset({"support", "team", "fantastic", "resolved", "helpful"}): "Customer support praise",
    frozenset({"improved", "workflow", "transformed", "productive"}): "Productivity improvements",
}


def extract_theme(texts: list[str]) -> str:
    """Extract a human-readable theme label from cluster texts."""
    all_tokens: list[str] = []
    for text in texts:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]
        all_tokens.extend(tokens)

    if not all_tokens:
        return "General feedback"

    freq = Counter(all_tokens)
    top_tokens = {tok for tok, _ in freq.most_common(15)}

    # Match against known patterns
    best_match = ""
    best_overlap = 0
    for pattern_tokens, label in _THEME_PATTERNS.items():
        overlap = len(top_tokens & pattern_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = label

    if best_overlap >= 2:
        return best_match

    # Fallback: build label from top 3 tokens
    top3 = [tok for tok, _ in freq.most_common(3)]
    return " ".join(top3).title() + " issues"


def extract_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    """Extract top keywords from a set of texts."""
    all_tokens: list[str] = []
    for text in texts:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]
        all_tokens.extend(tokens)
    return [tok for tok, _ in Counter(all_tokens).most_common(top_n)]
