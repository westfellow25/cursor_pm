"""Multi-signal sentiment & urgency analysis.

Combines lexicon-based scoring with pattern detection. In production this
would be enhanced by a fine-tuned classifier, but the lexicon approach
gives us interpretable baselines with zero latency."""

from __future__ import annotations

import re

# ── Sentiment lexicon (curated for product feedback domain) ──────────────────

_POSITIVE = {
    "love": 0.8, "great": 0.7, "excellent": 0.9, "amazing": 0.8, "awesome": 0.8,
    "fantastic": 0.8, "helpful": 0.6, "intuitive": 0.7, "clean": 0.5, "fast": 0.6,
    "smooth": 0.6, "easy": 0.5, "beautiful": 0.6, "impressive": 0.7, "reliable": 0.6,
    "perfect": 0.9, "wonderful": 0.8, "solid": 0.5, "nice": 0.4, "pleased": 0.6,
    "happy": 0.6, "enjoy": 0.5, "improved": 0.5, "better": 0.4, "best": 0.7,
    "powerful": 0.6, "efficient": 0.6, "delightful": 0.7, "seamless": 0.7,
    "responsive": 0.5, "polished": 0.6, "recommend": 0.6, "valuable": 0.6,
}

_NEGATIVE = {
    "slow": -0.5, "bug": -0.6, "crash": -0.8, "broken": -0.8, "terrible": -0.9,
    "awful": -0.9, "horrible": -0.8, "frustrating": -0.7, "annoying": -0.6,
    "confusing": -0.5, "useless": -0.8, "laggy": -0.6, "unresponsive": -0.7,
    "difficult": -0.5, "complicated": -0.5, "fails": -0.7, "error": -0.6,
    "timeout": -0.6, "missing": -0.4, "lacks": -0.4, "poor": -0.6,
    "disappointed": -0.7, "worse": -0.6, "worst": -0.9, "unusable": -0.9,
    "clunky": -0.5, "bloated": -0.5, "inconsistent": -0.5, "unstable": -0.7,
    "regression": -0.7, "downgrade": -0.7, "hate": -0.8, "painful": -0.6,
    "waste": -0.6, "stuck": -0.5, "blocked": -0.7, "cumbersome": -0.5,
    "crashes": -0.8, "crashing": -0.8, "wrong": -0.5, "incorrectly": -0.5,
    "empty": -0.4, "outdated": -0.4, "silently": -0.4, "cryptic": -0.5,
    "overwhelming": -0.5, "needlessly": -0.5, "restrictive": -0.4,
    "duplicates": -0.5, "duplicate": -0.5, "delays": -0.5, "forever": -0.6,
    "maze": -0.5, "impossible": -0.7, "unacceptable": -0.8, "expires": -0.3,
    "expire": -0.3, "disconnecting": -0.5, "breaking": -0.6, "weak": -0.4,
    "hidden": -0.3, "hard": -0.4, "overlap": -0.3, "longer": -0.3,
}

# Phrase-level patterns (checked against full text) for cases the tokenizer misses
_NEGATIVE_PHRASES = {
    "not working": -0.7, "doesn't work": -0.7, "can't find": -0.5,
    "too long": -0.4, "too many": -0.4, "too short": -0.3,
    "no way to": -0.5, "hard to find": -0.5, "hard to": -0.4,
    "nothing is where": -0.5, "lost my work": -0.7, "this has been broken": -0.7,
    "data loss": -0.9, "we might have to look at alternatives": -0.7,
}
_POSITIVE_PHRASES = {
    "exactly what we needed": 0.8, "best-in-class": 0.8,
    "pleasure to work": 0.7, "completely transformed": 0.8,
    "well-designed": 0.6, "well-documented": 0.6,
}

# Negation flips sentiment
_NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't", "hardly", "barely"}

# Intensifiers amplify sentiment
_INTENSIFIERS = {"very": 1.3, "really": 1.3, "extremely": 1.5, "incredibly": 1.5, "absolutely": 1.4, "totally": 1.3, "super": 1.3, "so": 1.2}

# ── Urgency signals ──────────────────────────────────────────────────────────

_URGENCY_TERMS = {
    "blocked": 0.9, "blocker": 0.9, "critical": 0.9, "urgent": 0.85,
    "asap": 0.85, "immediately": 0.8, "broken": 0.8, "crash": 0.85,
    "data loss": 0.95, "security": 0.9, "vulnerability": 0.95,
    "production": 0.7, "outage": 0.9, "down": 0.7, "emergency": 0.9,
    "showstopper": 0.9, "deadline": 0.7, "escalat": 0.8, "cancel": 0.75,
    "churn": 0.8, "leaving": 0.7, "switching": 0.7, "competitor": 0.6,
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def analyze_sentiment(text: str) -> float:
    """Return sentiment score in [-1.0, 1.0]."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    score = 0.0
    count = 0
    for i, tok in enumerate(tokens):
        val = _POSITIVE.get(tok, 0.0) + _NEGATIVE.get(tok, 0.0)
        if val == 0.0:
            continue

        # Check for negation in preceding 3 tokens
        negated = any(tokens[j] in _NEGATORS for j in range(max(0, i - 3), i))
        if negated:
            val *= -0.8

        # Check for intensifiers
        for j in range(max(0, i - 2), i):
            mult = _INTENSIFIERS.get(tokens[j])
            if mult:
                val *= mult
                break

        score += val
        count += 1

    # Phrase-level patterns
    text_lower = text.lower()
    for phrase, val in _NEGATIVE_PHRASES.items():
        if phrase in text_lower:
            score += val
            count += 1
    for phrase, val in _POSITIVE_PHRASES.items():
        if phrase in text_lower:
            score += val
            count += 1

    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / count))


def analyze_urgency(text: str) -> float:
    """Return urgency score in [0.0, 1.0]."""
    text_lower = text.lower()
    max_urgency = 0.0
    for term, weight in _URGENCY_TERMS.items():
        if term in text_lower:
            max_urgency = max(max_urgency, weight)
    return max_urgency


# ── Category classification ──────────────────────────────────────────────────

_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("performance", ["slow", "latency", "timeout", "loading", "speed", "lag", "response time", "performance"]),
    ("bug", ["bug", "crash", "error", "broken", "fails", "glitch", "issue", "not working", "doesn't work"]),
    ("ux", ["confusing", "intuitive", "layout", "design", "navigation", "ui", "ux", "usability", "hard to find", "difficult to"]),
    ("feature-request", ["wish", "would be nice", "should have", "need", "want", "add", "please", "missing", "request", "suggest"]),
    ("integration", ["integration", "api", "connect", "sync", "webhook", "import", "export", "plugin", "third-party"]),
    ("onboarding", ["onboarding", "setup", "getting started", "first time", "tutorial", "wizard", "learn"]),
    ("security", ["security", "password", "auth", "login", "permission", "access", "vulnerability", "2fa", "sso"]),
    ("mobile", ["mobile", "ios", "android", "app", "phone", "tablet", "responsive"]),
    ("pricing", ["price", "pricing", "expensive", "cost", "billing", "subscription", "plan", "upgrade", "pay"]),
    ("praise", ["love", "great", "excellent", "amazing", "awesome", "thank", "fantastic", "wonderful", "best", "perfect"]),
    ("support", ["support", "help", "documentation", "docs", "customer service", "response time", "ticket"]),
    ("data", ["data", "analytics", "report", "dashboard", "chart", "metric", "export", "insight"]),
]


def classify_category(text: str) -> tuple[str, str]:
    """Return (category, subcategory) for a feedback text."""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for cat, keywords in _CATEGORY_PATTERNS:
        score = sum(1.0 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return ("general", "")

    # Primary category is highest scoring
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_cats[0][0]
    secondary = sorted_cats[1][0] if len(sorted_cats) > 1 else ""
    return (primary, secondary)


# ── Batch enrichment ─────────────────────────────────────────────────────────


def enrich_feedback(text: str) -> dict:
    """Run all analyses on a single feedback text. Returns enrichment dict."""
    sentiment = analyze_sentiment(text)
    urgency = analyze_urgency(text)
    category, subcategory = classify_category(text)
    return {
        "sentiment": round(sentiment, 3),
        "urgency": round(urgency, 3),
        "category": category,
        "subcategory": subcategory,
    }
