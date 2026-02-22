"""Simple end-to-end demo for feedback ingestion, clustering, and recommendation output."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math
import re

from feedback_ingestion import HashingEmbedder, run_pipeline

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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _divider(title: str) -> str:
    return f"\n{'=' * 20} {title} {'=' * 20}"


def _impact_estimate(opportunity_score: float, frequency: int, total_records: int) -> str:
    share = (frequency / max(1, total_records)) * 100
    if opportunity_score >= 5:
        band = "High"
    elif opportunity_score >= 3:
        band = "Medium"
    else:
        band = "Low"
    return f"{band} impact (affects ~{share:.0f}% of sampled feedback)"


def _cluster_severity(texts: list[str]) -> float:
    """Estimate severity from urgency terms when sentiment/importance fields are unavailable."""
    if not texts:
        return 1.0

    text_scores = []
    for text in texts:
        tokens = _tokenize(text)
        weights = [URGENCY_TERMS[token] for token in tokens if token in URGENCY_TERMS]
        if weights:
            text_scores.append(sum(weights) / len(weights))
        else:
            text_scores.append(1.0)
    return sum(text_scores) / len(text_scores)


def _normalize_scores(raw_scores: dict[int, float]) -> dict[int, float]:
    if not raw_scores:
        return {}
    low = min(raw_scores.values())
    high = max(raw_scores.values())
    if math.isclose(low, high):
        return {cluster_id: 5.0 for cluster_id in raw_scores}
    return {
        cluster_id: round(1 + ((score - low) / (high - low)) * 9, 2)
        for cluster_id, score in raw_scores.items()
    }


def _problem_statement(texts: list[str]) -> str:
    token_counts = Counter(token for text in texts for token in _tokenize(text) if len(token) > 3)
    keywords = [token for token, _ in token_counts.most_common(3)]
    keyword_phrase = ", ".join(keywords) if keywords else "core workflow reliability"
    return (
        "Users are repeatedly reporting friction in this workflow cluster. "
        f"The most common signals point to issues around {keyword_phrase}."
    )


def _proposed_solution(example_signal: str, texts: list[str]) -> str:
    top_cluster_text = " ".join([example_signal, *texts])
    token_set = set(_tokenize(top_cluster_text))

    # Ensure recommendations stay anchored to the dominant pain signals from the top cluster.
    if {"slow", "loading", "lag", "latency", "performance", "timeout"} & token_set:
        return (
            "Improve dashboard and report performance by optimizing heavy queries, adding pagination for large views, "
            "caching high-traffic summaries, and moving long-running calculations to async jobs so pages load quickly "
            "under real usage." 
        )
    if {"search", "find", "filter"} & token_set:
        return (
            "Implement an upgraded discovery experience with smarter indexing, typo/punctuation tolerance, "
            "and saved filters so users can quickly locate records and repeat common queries without manual rework."
        )
    if {"export", "report", "csv", "download"} & token_set:
        return (
            "Build a guided reporting flow with prominent export actions, reusable report templates, and scheduled "
            "delivery so customers can consistently share insights without hunting for controls each time."
        )
    if {"notify", "notification", "urgent", "late"} & token_set:
        return (
            "Redesign the notifications pipeline with priority-aware delivery windows, configurable alert channels, "
            "and digest settings so urgent updates arrive on time while routine updates remain manageable."
        )
    if {"crash", "timeout", "fails", "error"} & token_set:
        return (
            "Prioritize reliability hardening for the affected workflow by adding upload and auth guardrails, "
            "clear retry paths, and proactive monitoring to reduce failure rates and user disruption."
        )
    return (
        "Deliver a focused workflow improvement initiative that removes repeated friction points, adds clearer "
        "guidance in key steps, and improves consistency for common customer tasks."
    )


def run_demo(csv_path: str | Path = "example_data/feedback.csv", n_clusters: int = 3) -> None:
    csv_path = Path(csv_path)

    print("\n" + "=" * 78)
    print("AI PRODUCT DISCOVERY — STAKEHOLDER DEMO OUTPUT")
    print("=" * 78)

    print(_divider("1) DATA INGESTION"))
    print(f"Source file        : {csv_path}")
    records, cluster_results = run_pipeline(csv_path, n_clusters=n_clusters)
    print(f"Feedback loaded    : {len(records)} records")

    print(_divider("2) CLUSTER SNAPSHOT"))
    grouped_ids: dict[int, list[str]] = defaultdict(list)
    grouped_similarity: dict[int, list[float]] = defaultdict(list)
    for row in cluster_results:
        grouped_ids[row.cluster_id].append(row.feedback_id)
        grouped_similarity[row.cluster_id].append(row.similarity_to_centroid)

    print(f"Clusters generated : {len(grouped_ids)}")
    for cluster_id in sorted(grouped_ids):
        ids = grouped_ids[cluster_id]
        preview = next((r.text for r in records if r.feedback_id == ids[0]), "")
        cluster_name = "misc/unclustered" if cluster_id == -1 else f"Cluster {cluster_id}"
        coherence = sum(grouped_similarity[cluster_id]) / max(1, len(grouped_similarity[cluster_id]))
        print(f"\n• {cluster_name}")
        print(f"  - Size            : {len(ids)} items")
        print(f"  - Coherence score : {coherence:.3f}")
        print(f"  - Record IDs      : {', '.join(ids)}")
        print(f"  - Example signal  : {preview}")

    print(_divider("3) OPPORTUNITY SCORING"))
    if not grouped_ids:
        print("No opportunities identified from this sample.")
        return

    records_by_id = {record.feedback_id: record for record in records}
    raw_scores: dict[int, float] = {}
    cluster_metrics: dict[int, dict[str, object]] = {}

    for cluster_id, ids in grouped_ids.items():
        if cluster_id == -1:
            continue
        texts = [records_by_id[feedback_id].text for feedback_id in ids if feedback_id in records_by_id]
        example_signal = next((records_by_id[feedback_id].text for feedback_id in ids if feedback_id in records_by_id), "")
        frequency = len(texts)
        severity = _cluster_severity(texts)
        prevalence = frequency / max(1, len(records))
        weighted_score = (0.65 * frequency) + (0.35 * severity * 3.0)
        raw_score = weighted_score * math.sqrt(1 + prevalence)
        raw_scores[cluster_id] = raw_score
        cluster_metrics[cluster_id] = {
            "cluster_id": cluster_id,
            "texts": texts,
            "ids": [feedback_id for feedback_id in ids if feedback_id in records_by_id],
            "example_signal": example_signal,
            "frequency": frequency,
            "severity": round(severity, 2),
        }

    if not raw_scores:
        print("Only misc/unclustered feedback remained after coherence filtering.")
        return

    normalized_scores = _normalize_scores(raw_scores)
    ranked_clusters = sorted(
        [
            {
                **cluster_metrics[cluster_id],
                "opportunity_score": normalized_scores[cluster_id],
            }
            for cluster_id in cluster_metrics
        ],
        key=lambda item: (item["opportunity_score"], item["frequency"]),
        reverse=True,
    )

    print(f"Clusters scored     : {len(ranked_clusters)}")
    print("Top opportunities   :")
    for index, cluster in enumerate(ranked_clusters[:3], start=1):
        impact = _impact_estimate(cluster["opportunity_score"], cluster["frequency"], len(records))
        print(
            f"  {index}. Cluster {cluster['cluster_id']}\n"
            f"     • Opportunity score : {cluster['opportunity_score']}\n"
            f"     • Frequency         : {cluster['frequency']} items\n"
            f"     • Severity          : {cluster['severity']}\n"
            f"     • Impact estimate   : {impact}"
        )

    top = ranked_clusters[0]
    print(_divider("4) RECOMMENDED ACTION"))
    print(f"Top cluster         : Cluster {top['cluster_id']}")
    print(f"Problem statement   : {_problem_statement(top['texts'])}")
    print("Proposed solution   :")
    print(f"  {_proposed_solution(top['example_signal'], top['texts'])}")
    print("Why this, why now   :")
    top_frequency_pct = (top["frequency"] / max(1, len(records))) * 100
    print(
        f"  This cluster appears in {top_frequency_pct:.0f}% of sampled feedback, with a severity score of {top['severity']}."
    )
    print(
        "  The repeated narrative suggests users hit this issue during core workflows, which increases friction, "
        "slows task completion, and elevates support risk if left unresolved."
    )
    print("Acceptance criteria :")
    print("  - Cluster-specific workflow completion improves in follow-up feedback.")
    print("  - Users can complete the target task without escalation or workaround.")
    print("  - Support tickets linked to this pain point trend downward after release.")
    print("  - PM dashboard tracks adoption and reliability for the shipped solution.")
    print("Open questions      :")
    print("  - Which user segment in this cluster should we prioritize for beta testing?")
    print("  - What success threshold (adoption, CSAT, ticket reduction) defines launch readiness?")
    print("  - Are there platform-specific constraints (web/mobile/api) that require phased rollout?")

    print(_divider("5) SUPPORTING EVIDENCE"))
    evidence_pairs = list(zip(top["ids"], top["texts"]))
    evidence_count = min(5, max(3, len(evidence_pairs)))
    if evidence_pairs:
        embedder = HashingEmbedder(dimensions=128)
        vectors = embedder.embed([top["example_signal"], *top["texts"]])
        signal_vector, quote_vectors = vectors[0], vectors[1:]
        scored_pairs = []
        for (feedback_id, quote), vector in zip(evidence_pairs, quote_vectors):
            similarity = sum(a * b for a, b in zip(signal_vector, vector))
            scored_pairs.append((similarity, feedback_id, quote))

        ranked_evidence = sorted(scored_pairs, key=lambda item: item[0], reverse=True)
        semantically_close = [item for item in ranked_evidence if item[0] >= 0.55]
        candidate_pool = semantically_close if len(semantically_close) >= 3 else ranked_evidence
        selected_evidence = [
            (feedback_id, quote)
            for _, feedback_id, quote in candidate_pool[:evidence_count]
        ]
        for feedback_id, quote in selected_evidence:
            print(f"  ▸ [{feedback_id}] “{quote}”")

    print("\n" + "-" * 78)
    print("Demo complete. This output is formatted for direct stakeholder review.")
    print("-" * 78)


if __name__ == "__main__":
    run_demo()
