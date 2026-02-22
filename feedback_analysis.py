"""Cluster customer feedback into opportunities and generate feature recommendations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "too",
    "we",
    "with",
    "you",
    "your",
}

URGENCY_TERMS = {
    "blocked": 1.8,
    "broken": 1.7,
    "cannot": 1.7,
    "cant": 1.7,
    "crash": 1.8,
    "critical": 1.9,
    "difficult": 1.3,
    "frustrating": 1.4,
    "hard": 1.2,
    "impossible": 1.8,
    "manual": 1.3,
    "slow": 1.4,
    "time": 1.2,
    "unable": 1.7,
}


@dataclass
class FeedbackItem:
    text: str


TOKEN_ALIASES = {
    "integrations": "integration",
    "integrated": "integration",
    "syncing": "sync",
    "synced": "sync",
    "filters": "filter",
    "permissions": "permission",
    "reports": "report",
}


def _normalize_token(token: str) -> str:
    token = TOKEN_ALIASES.get(token, token)
    if token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    cleaned = []
    for raw in words:
        token = _normalize_token(raw.strip("'"))
        if token in STOPWORDS or len(token) <= 2:
            continue
        cleaned.append(token)
    return cleaned


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union


def _urgency_score(tokens: Iterable[str]) -> float:
    weights = [URGENCY_TERMS[t] for t in tokens if t in URGENCY_TERMS]
    if not weights:
        return 1.0
    return sum(weights) / len(weights)


def _problem_summary(counter: Counter[str]) -> str:
    common = [term for term, _ in counter.most_common(4)]
    if not common:
        return "General customer pain point"
    return "Users struggle with " + ", ".join(common[:3])


def _feature_recommendation(counter: Counter[str]) -> str:
    terms = {t for t, _ in counter.most_common(8)}
    if terms & {"search", "find", "filter"}:
        return "Add advanced search and saved filters so users can quickly locate what they need."
    if terms & {"export", "report", "csv", "download"}:
        return "Build one-click export/reporting with scheduled delivery to reduce manual reporting work."
    if terms & {"notify", "alert", "notification", "update"}:
        return "Create configurable notifications and digest updates so users stay informed automatically."
    if terms & {"integrate", "integration", "sync", "api"}:
        return "Deliver native integrations and API sync to remove repetitive data entry between tools."
    if terms & {"permission", "role", "access", "approve"}:
        return "Introduce granular role-based access and approval workflows for safer collaboration."
    return "Prioritize workflow automation for this pain point to reduce repetitive manual effort."


def cluster_feedback(feedback: list[FeedbackItem], similarity_threshold: float = 0.1) -> list[list[FeedbackItem]]:
    clusters: list[list[FeedbackItem]] = []
    cluster_tokens: list[set[str]] = []

    for item in feedback:
        tokens = set(_tokenize(item.text))
        if not clusters:
            clusters.append([item])
            cluster_tokens.append(set(tokens))
            continue

        best_idx = -1
        best_score = 0.0
        for idx, centroid_tokens in enumerate(cluster_tokens):
            score = _jaccard_similarity(tokens, centroid_tokens)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= similarity_threshold:
            clusters[best_idx].append(item)
            cluster_tokens[best_idx].update(tokens)
        else:
            clusters.append([item])
            cluster_tokens.append(set(tokens))

    return clusters


def opportunity_report(raw_feedback: list[str]) -> list[dict[str, object]]:
    feedback_items = [FeedbackItem(text=text) for text in raw_feedback if text and text.strip()]
    clusters = cluster_feedback(feedback_items)
    total_items = max(1, len(feedback_items))
    results: list[dict[str, object]] = []

    for group in clusters:
        all_tokens = [_tokenize(item.text) for item in group]
        flattened_tokens = [token for tokens in all_tokens for token in tokens]
        token_counter = Counter(flattened_tokens)

        frequency = len(group)
        avg_urgency = sum(_urgency_score(tokens) for tokens in all_tokens) / max(1, frequency)
        prevalence = frequency / total_items
        opportunity_score = round(frequency * avg_urgency * math.sqrt(1 + prevalence), 2)

        results.append(
            {
                "problem_summary": _problem_summary(token_counter),
                "frequency": frequency,
                "opportunity_score": opportunity_score,
                "suggested_feature": _feature_recommendation(token_counter),
                "supporting_evidence_quotes": [item.text for item in group[:3]],
            }
        )

    return sorted(results, key=lambda r: (r["opportunity_score"], r["frequency"]), reverse=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature recommendations from clustered feedback.")
    parser.add_argument("--feedback", required=True, help="Path to JSON file containing a list of feedback strings.")
    args = parser.parse_args()

    with open(args.feedback, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("Feedback JSON must be a list of strings")

    print(json.dumps(opportunity_report(payload), indent=2))


if __name__ == "__main__":
    main()
