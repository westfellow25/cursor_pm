"""LLM-powered intelligence layer.

Supports both Claude (Anthropic) and GPT (OpenAI). Claude is preferred
when ANTHROPIC_API_KEY is set — it handles product/business analysis
better and has a much larger context window.

Used for:
- Rich thematic labels for clusters
- Actionable summaries and recommendations
- Root cause analysis
- Impact narratives

Falls back gracefully to heuristic methods when no key is configured."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from pulse.config import settings

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai", "none"]


@lru_cache(maxsize=1)
def _active_provider() -> Provider:
    """Determine which LLM provider to use based on config."""
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.openai_api_key:
        return "openai"
    return "none"


def _call_anthropic(system: str, user: str, max_tokens: int) -> str | None:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.claude_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Extract text from content blocks
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text.strip()
        return None
    except Exception as exc:
        logger.warning("Claude call failed: %s", exc)
        return None


def _call_openai(system: str, user: str, max_tokens: int) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        return None


def _call_llm(system: str, user: str, max_tokens: int = 512) -> str | None:
    """Call the active LLM provider. Returns None if unavailable."""
    provider = _active_provider()
    if provider == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    if provider == "openai":
        return _call_openai(system, user, max_tokens)
    return None


def is_llm_available() -> bool:
    return _active_provider() != "none"


def get_llm_info() -> dict[str, str]:
    """Return info about the active LLM provider."""
    provider = _active_provider()
    if provider == "anthropic":
        return {"provider": "anthropic", "model": settings.claude_model}
    if provider == "openai":
        return {"provider": "openai", "model": settings.llm_model}
    return {"provider": "none", "model": "heuristic fallback"}


def verify_llm() -> bool:
    """Issue a minimal prompt to confirm the configured LLM key actually works.

    Returns False if the provider is `none`, the API key is invalid, the model
    name is wrong, or the provider is unreachable. Cheap (~1 token out).
    """
    if not is_llm_available():
        return False
    result = _call_llm(system="Reply with the single word: ok.", user="ping", max_tokens=4)
    return bool(result)


def generate_cluster_label(keywords: list[str], sample_texts: list[str]) -> str | None:
    """Generate a human-readable theme label for a cluster."""
    samples = "\n".join(f"- {t[:150]}" for t in sample_texts[:8])
    return _call_llm(
        system=(
            "You are a product analyst. Generate a concise, specific theme label "
            "(3-6 words) for a cluster of customer feedback. The label should be "
            "actionable and descriptive, like 'Dashboard loading performance' or "
            "'Mobile app crash on upload'. Return ONLY the label, nothing else."
        ),
        user=f"Keywords: {', '.join(keywords[:10])}\n\nSample feedback:\n{samples}",
        max_tokens=30,
    )


def generate_cluster_summary(
    theme: str,
    size: int,
    total: int,
    keywords: list[str],
    sample_texts: list[str],
    sentiment_avg: float,
) -> str | None:
    """Generate a rich summary paragraph for a cluster."""
    samples = "\n".join(f"- {t[:150]}" for t in sample_texts[:6])
    pct = size / total * 100 if total > 0 else 0
    return _call_llm(
        system=(
            "You are a senior product analyst writing for a VP of Product. "
            "Write a 2-3 sentence summary of this feedback cluster. "
            "Include: what the issue is, how widespread it is, and what the "
            "user impact is. Be specific and data-driven. No bullet points."
        ),
        user=(
            f"Theme: {theme}\n"
            f"Volume: {size} items ({pct:.0f}% of all feedback)\n"
            f"Avg sentiment: {sentiment_avg:.2f}\n"
            f"Keywords: {', '.join(keywords[:10])}\n\n"
            f"Sample feedback:\n{samples}"
        ),
        max_tokens=200,
    )


def generate_recommendation(
    theme: str,
    severity: float,
    keywords: list[str],
    sample_texts: list[str],
) -> str | None:
    """Generate an actionable recommendation for a cluster."""
    samples = "\n".join(f"- {t[:150]}" for t in sample_texts[:5])
    return _call_llm(
        system=(
            "You are a senior product manager. Based on the customer feedback cluster below, "
            "write a specific, actionable recommendation in 2-3 sentences. "
            "Include what to build/fix and the expected impact. Be concrete, not generic."
        ),
        user=(
            f"Theme: {theme}\n"
            f"Severity: {severity:.2f}\n"
            f"Keywords: {', '.join(keywords[:10])}\n\n"
            f"Sample feedback:\n{samples}"
        ),
        max_tokens=200,
    )


def generate_executive_narrative(
    clusters: list[dict[str, Any]],
    total_feedback: int,
) -> str | None:
    """Generate an executive narrative about the overall feedback landscape."""
    cluster_summary = "\n".join(
        f"- {c['theme']} (score={c['score']}, size={c['size']}, sentiment={c['sentiment']:.2f})"
        for c in clusters[:10]
    )
    return _call_llm(
        system=(
            "You are a Chief Product Officer writing a brief executive narrative. "
            "Summarize the current product health based on customer feedback analysis. "
            "Highlight the top 2-3 priorities, any positive signals, and the recommended "
            "strategic focus. Write 3-4 sentences. Be direct and insight-driven."
        ),
        user=(
            f"Total feedback analysed: {total_feedback}\n"
            f"Top clusters (by opportunity score):\n{cluster_summary}"
        ),
        max_tokens=300,
    )


def generate_root_cause_analysis(
    theme: str,
    sample_texts: list[str],
    keywords: list[str],
) -> str | None:
    """Generate a root cause analysis for a cluster."""
    samples = "\n".join(f"- {t[:150]}" for t in sample_texts[:8])
    return _call_llm(
        system=(
            "You are a technical product analyst. Based on the customer feedback, "
            "identify the likely root causes (technical or UX). "
            "List 2-3 probable root causes, each as one sentence. "
            "Format: numbered list."
        ),
        user=(
            f"Theme: {theme}\n"
            f"Keywords: {', '.join(keywords[:10])}\n\n"
            f"Feedback:\n{samples}"
        ),
        max_tokens=250,
    )


def enrich_clusters_with_llm(
    clusters_data: list[dict[str, Any]],
    total_feedback: int,
) -> list[dict[str, Any]]:
    """Enrich a list of cluster dicts with LLM-generated content."""
    if not is_llm_available():
        return clusters_data

    for cluster in clusters_data[:10]:
        label = generate_cluster_label(cluster.get("keywords", []), cluster.get("sample_texts", []))
        if label:
            cluster["llm_label"] = label

        summary = generate_cluster_summary(
            cluster.get("theme", ""), cluster.get("size", 0), total_feedback,
            cluster.get("keywords", []), cluster.get("sample_texts", []),
            cluster.get("sentiment_avg", 0.0),
        )
        if summary:
            cluster["llm_summary"] = summary

        recommendation = generate_recommendation(
            cluster.get("theme", ""), cluster.get("severity", 0.0),
            cluster.get("keywords", []), cluster.get("sample_texts", []),
        )
        if recommendation:
            cluster["llm_recommendation"] = recommendation

    return clusters_data
