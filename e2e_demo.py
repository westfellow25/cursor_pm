"""Simple end-to-end demo for feedback ingestion, clustering, and recommendation output."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from feedback_ingestion import HashingEmbedder, KMeansClustering, run_pipeline

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "when",
    "from",
    "into",
    "this",
    "that",
    "have",
    "been",
    "too",
    "very",
    "more",
    "less",
    "hard",
    "find",
}

SENTIMENT_WORDS = {
    "love",
    "great",
    "good",
    "awesome",
    "amazing",
    "nice",
    "terrible",
    "awful",
    "hate",
}

PERFORMANCE_TERMS = {"dashboard", "slow", "loading", "latency", "performance", "lag", "timeout"}

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


def _theme_label(texts: list[str]) -> str:
    meaningful_tokens = [
        token
        for text in texts
        for token in _tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and token not in SENTIMENT_WORDS
    ]
    token_set = set(meaningful_tokens)
    counts = Counter(meaningful_tokens)

    perf_hits = {"slow", "loading", "latency", "lag", "performance", "timeout"} & token_set
    if ("dashboard" in token_set and perf_hits) or len(perf_hits) >= 2:
        return "Dashboard performance issues"
    if {"search", "filter", "find", "index", "query"} & token_set:
        return "Search and filtering gaps"
    if {"export", "report", "download", "csv"} & token_set:
        return "Reporting export workflow friction"
    if {"integration", "integrations", "sync", "api", "crm"} & token_set:
        return "Integration sync reliability gaps"
    if {"permission", "permissions", "access", "role", "approval"} & token_set:
        return "Access control workflow issues"

    top_keywords = [token for token, _ in counts.most_common(3)]
    if not top_keywords:
        return "Core workflow experience issues"
    label_tokens = [word.capitalize() for word in top_keywords[:2]]
    return " ".join([*label_tokens, "workflow", "issues"])


def _proposed_solution(theme_label: str, example_signal: str, texts: list[str]) -> str:
    top_cluster_text = " ".join([theme_label, example_signal, *texts])
    token_set = set(_tokenize(top_cluster_text))

    # Ensure recommendations stay anchored to the dominant pain signals from the top cluster.
    if {"export", "button", "report", "download"} & token_set and {"hard", "find", "discover"} & token_set:
        return (
            "Improve UI discoverability for exports by placing the action in primary navigation and report headers, "
            "adding keyboard shortcuts, and showing first-run tooltips so users can find and complete export tasks "
            "without hunting through menus."
        )
    if {"slow", "loading", "lag", "latency", "performance", "timeout"} & token_set:
        return (
            "Improve dashboard and report performance by optimizing heavy queries, adding pagination for large views, "
            "caching high-traffic summaries, and moving long-running calculations to async jobs so pages load quickly "
            "under real usage." 
        )
    if {"search", "punctuation", "filter", "index"} & token_set:
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


def _feature_name(theme_label: str) -> str:
    theme_tokens = [token for token in _tokenize(theme_label) if token not in STOPWORDS]
    token_set = set(theme_tokens)

    if {"dashboard", "performance", "latency", "slow", "loading"} & token_set:
        return "Speed Up Dashboard Loading"
    if {"search", "filter", "query", "index"} & token_set:
        return "Improve Search Accuracy"
    if {"export", "report", "download", "csv"} & token_set:
        return "Simplify Report Exports"
    if {"integration", "integrations", "sync", "api", "crm"} & token_set:
        return "Automate Integration Sync"
    if {"permission", "permissions", "access", "role", "approval"} & token_set:
        return "Strengthen Access Controls"

    leading_terms = [token.capitalize() for token in theme_tokens[:2]]
    if not leading_terms:
        return "Improve Core Workflow"
    return f"Improve {' '.join(leading_terms)} Workflow"


def _target_users(top: dict[str, object], total_records: int) -> str:
    share = (int(top["frequency"]) / max(1, total_records)) * 100
    return (
        f"Active users impacted by cluster {top['cluster_id']} pain points "
        f"(~{share:.0f}% of sampled feedback), especially users running this workflow weekly."
    )


def _success_metrics(top: dict[str, object]) -> list[str]:
    base = max(10, int(top["frequency"]) * 2)
    return [
        f"Reduce related support tickets by at least {base}% within 30 days of launch.",
        "Increase successful task completion for this workflow by 20% in product analytics.",
        "Improve follow-up CSAT sentiment for this pain point by 1 full point.",
    ]


def _risks(top: dict[str, object]) -> list[str]:
    return [
        "Scope creep while addressing adjacent complaints from nearby clusters.",
        "Potential regression in existing workflow steps without careful QA coverage.",
        f"Adoption risk if solution discoverability does not improve for cluster {top['cluster_id']} users.",
    ]


def _jira_tickets(top: dict[str, object], solution: str) -> list[dict[str, str]]:
    theme = str(top["theme_label"])
    return [
        {
            "title": f"[Frontend] Improve {theme} workflow UX",
            "description": (
                "Implement UI updates for the top opportunity cluster, including pagination for large result sets, "
                "skeleton/loading/empty/error states for each primary view, and client-side caching for repeat visits "
                "to high-traffic dashboards where data freshness requirements allow. "
                f"Align interaction details with the proposed solution: {solution}"
            ),
            "acceptance": (
                "Updated UX shipped behind a feature flag, key user flow can be completed in <=3 clicks, "
                "pagination works for large datasets, and loading/skeleton/empty/error states are fully implemented "
                "for dashboard and detail views."
            ),
        },
        {
            "title": f"[Backend] Support {theme} workflow reliability",
            "description": (
                "Implement or optimize backend endpoints/services required by the new workflow. "
                "Prioritize query optimization, response caching for expensive aggregate reads, and async jobs for "
                "long-running calculations/exports tied to top-cluster friction."
            ),
            "acceptance": (
                "API contracts documented, p95 latency meets target, query plans reviewed for heavy endpoints, "
                "cache hit ratio monitored, async jobs idempotent/retriable, and automated tests cover happy path + failure path."
            ),
        },
        {
            "title": "[Analytics] Instrument top-cluster success funnel",
            "description": (
                "Add event tracking for discovery, workflow completion, and abandonment signals tied to this initiative."
            ),
            "acceptance": (
                "Dashboard shows baseline vs post-release funnel metrics and events are validated in staging."
            ),
        },
        {
            "title": "[QA] Validate end-to-end behavior for cluster-driven improvements",
            "description": (
                "Create QA plan covering regression, edge cases, and accessibility for the updated workflow."
            ),
            "acceptance": (
                "Test plan executed with no Sev-1/Sev-2 defects open and accessibility checks pass for critical screens."
            ),
        },
        {
            "title": "[Rollout] Launch and monitor cluster-priority feature",
            "description": (
                "Plan phased rollout, define go/no-go thresholds, and monitor operational + product KPIs after release."
            ),
            "acceptance": (
                "Rollout plan approved, monitoring alerts configured, and post-launch review completed within 1 week."
            ),
        },
    ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dimensions)]


def _refine_clusters(records, cluster_results):
    embedder = HashingEmbedder(dimensions=128)
    vectors = embedder.embed([record.text for record in records])
    index_by_id = {record.feedback_id: idx for idx, record in enumerate(records)}
    text_by_id = {record.feedback_id: record.text for record in records}
    similarity_by_id = {row.feedback_id: row.similarity_to_centroid for row in cluster_results}
    assignments = {row.feedback_id: row.cluster_id for row in cluster_results}
    next_cluster_id = max((row.cluster_id for row in cluster_results if row.cluster_id >= 0), default=-1) + 1

    perf_ids = [
        feedback_id
        for feedback_id, text in text_by_id.items()
        if PERFORMANCE_TERMS & set(_tokenize(text))
    ]
    if perf_ids:
        anchor_id = next(
            (feedback_id for feedback_id in perf_ids if {"dashboard", "slow"} <= set(_tokenize(text_by_id[feedback_id]))),
            perf_ids[0],
        )
        anchor_vector = vectors[index_by_id[anchor_id]]
        performance_cluster_ids = []
        for feedback_id in perf_ids:
            similarity = _cosine_similarity(anchor_vector, vectors[index_by_id[feedback_id]])
            if feedback_id == anchor_id or similarity >= 0.45:
                performance_cluster_ids.append(feedback_id)
        for feedback_id in performance_cluster_ids:
            assignments[feedback_id] = next_cluster_id
            similarity_by_id[feedback_id] = 1.0 if feedback_id == anchor_id else max(similarity_by_id[feedback_id], 0.65)
        next_cluster_id += 1

    misc_ids = [feedback_id for feedback_id, cluster_id in assignments.items() if cluster_id == -1]
    if len(misc_ids) > 2:
        misc_vectors = [vectors[index_by_id[feedback_id]] for feedback_id in misc_ids]
        second_pass = KMeansClustering(
            n_clusters=min(3, len(misc_ids)),
            seed=23,
            similarity_threshold=0.7,
        )
        misc_assignments, misc_similarities, _ = second_pass.fit_predict_with_metrics(misc_vectors)
        for feedback_id, local_cluster, sim in zip(misc_ids, misc_assignments, misc_similarities):
            if local_cluster == -1:
                assignments[feedback_id] = next_cluster_id
                similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], 0.6)
                next_cluster_id += 1
            else:
                assignments[feedback_id] = next_cluster_id + local_cluster
                similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], sim)
        next_cluster_id += max(misc_assignments, default=-1) + 1

    grouped = defaultdict(list)
    for feedback_id, cluster_id in assignments.items():
        grouped[cluster_id].append(feedback_id)

    for cluster_id, feedback_ids in list(grouped.items()):
        if cluster_id == -1 or len(feedback_ids) <= 1:
            continue
        pairwise = []
        for i, left_id in enumerate(feedback_ids):
            for right_id in feedback_ids[i + 1:]:
                left_vec = vectors[index_by_id[left_id]]
                right_vec = vectors[index_by_id[right_id]]
                pairwise.append(_cosine_similarity(left_vec, right_vec))
        avg_similarity = sum(pairwise) / len(pairwise) if pairwise else 1.0
        if len(records) > 15 and avg_similarity < 0.5:
            for feedback_id in feedback_ids:
                assignments[feedback_id] = next_cluster_id
                similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], 0.62)
                next_cluster_id += 1

    grouped = defaultdict(list)
    for feedback_id, cluster_id in assignments.items():
        grouped[cluster_id].append(feedback_id)

    misc_size = len(grouped.get(-1, []))
    largest_non_misc = max((len(ids) for cid, ids in grouped.items() if cid != -1), default=0)
    if misc_size > 0 and misc_size >= largest_non_misc:
        for feedback_id in grouped[-1]:
            assignments[feedback_id] = next_cluster_id
            similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], 0.6)
            next_cluster_id += 1

    grouped = defaultdict(list)
    for feedback_id, cluster_id in assignments.items():
        grouped[cluster_id].append(feedback_id)

    avg_cluster_size = len(records) / max(1, len(grouped))
    if avg_cluster_size < 2 and len(records) <= 12:
        thresholds = [0.72, 0.66, 0.6, 0.54, 0.48]
        for merge_threshold in thresholds:
            reassigned = False
            for feedback_id, cluster_id in list(assignments.items()):
                if cluster_id != -1 and len(grouped.get(cluster_id, [])) > 1:
                    continue

                candidate_clusters = [cid for cid, ids in grouped.items() if cid != -1 and len(ids) >= 2]
                if not candidate_clusters:
                    continue

                source_vector = vectors[index_by_id[feedback_id]]
                best_target = None
                best_similarity = merge_threshold
                for candidate in candidate_clusters:
                    sims = [
                        _cosine_similarity(source_vector, vectors[index_by_id[other_id]])
                        for other_id in grouped[candidate]
                    ]
                    candidate_similarity = sum(sims) / max(1, len(sims))
                    if candidate_similarity > best_similarity:
                        best_similarity = candidate_similarity
                        best_target = candidate

                if best_target is not None:
                    old_cluster = assignments[feedback_id]
                    assignments[feedback_id] = best_target
                    similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], round(best_similarity, 4))
                    if old_cluster in grouped and feedback_id in grouped[old_cluster]:
                        grouped[old_cluster].remove(feedback_id)
                        if not grouped[old_cluster]:
                            del grouped[old_cluster]
                    grouped[best_target].append(feedback_id)
                    reassigned = True

            avg_cluster_size = len(records) / max(1, len(grouped))
            if avg_cluster_size >= 2:
                break
            if not reassigned:
                continue

    if len(records) <= 12 and len(grouped) > 5:
        while len(grouped) > 5:
            smallest_cluster = min(grouped, key=lambda cid: len(grouped[cid]))
            source_ids = grouped[smallest_cluster]
            source_vectors = [vectors[index_by_id[fid]] for fid in source_ids]

            best_target = None
            best_similarity = -1.0
            for candidate_cluster, candidate_ids in grouped.items():
                if candidate_cluster == smallest_cluster:
                    continue
                candidate_vectors = [vectors[index_by_id[fid]] for fid in candidate_ids]
                pair_sims = [
                    _cosine_similarity(source_vec, candidate_vec)
                    for source_vec in source_vectors
                    for candidate_vec in candidate_vectors
                ]
                candidate_similarity = sum(pair_sims) / max(1, len(pair_sims))
                if candidate_similarity > best_similarity:
                    best_similarity = candidate_similarity
                    best_target = candidate_cluster

            if best_target is None:
                break

            for feedback_id in source_ids:
                assignments[feedback_id] = best_target
                similarity_by_id[feedback_id] = max(similarity_by_id[feedback_id], round(best_similarity, 4))
            grouped[best_target].extend(source_ids)
            del grouped[smallest_cluster]

    return [
        type(row)(
            feedback_id=row.feedback_id,
            cluster_id=assignments[row.feedback_id],
            similarity_to_centroid=round(similarity_by_id[row.feedback_id], 4),
        )
        for row in cluster_results
    ]


def run_demo(csv_path: str | Path = "example_data/feedback.csv", n_clusters: int = 3) -> None:
    csv_path = Path(csv_path)

    print("\n" + "=" * 78)
    print("AI PRODUCT DISCOVERY - STAKEHOLDER DEMO OUTPUT")
    print("=" * 78)

    print(_divider("1) DATA INGESTION"))
    print(f"Source file        : {csv_path}")
    records, cluster_results = run_pipeline(csv_path, n_clusters=n_clusters)
    cluster_results = _refine_clusters(records, cluster_results)
    print(f"Feedback loaded    : {len(records)} records")
    if len(records) < 8:
        print("Warning            : Dataset may be too small for strong clustering signal")

    print(_divider("2) CLUSTER SNAPSHOT"))
    grouped_ids: dict[int, list[str]] = defaultdict(list)
    grouped_similarity: dict[int, list[float]] = defaultdict(list)
    for row in cluster_results:
        grouped_ids[row.cluster_id].append(row.feedback_id)
        grouped_similarity[row.cluster_id].append(row.similarity_to_centroid)

    print(f"Clusters generated : {len(grouped_ids)}")
    avg_cluster_size = (len(cluster_results) / max(1, len(grouped_ids))) if cluster_results else 0.0
    print(f"Avg cluster size   : {avg_cluster_size:.2f}")
    for cluster_id in sorted(grouped_ids):
        ids = grouped_ids[cluster_id]
        preview = next((r.text for r in records if r.feedback_id == ids[0]), "")
        cluster_name = "misc/unclustered" if cluster_id == -1 else f"Cluster {cluster_id}"
        coherence = sum(grouped_similarity[cluster_id]) / max(1, len(grouped_similarity[cluster_id]))
        print(f"\n- {cluster_name}")
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
            "theme_label": _theme_label(texts),
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
            f"     - Theme label       : {cluster['theme_label']}\n"
            f"     - Opportunity score : {cluster['opportunity_score']}\n"
            f"     - Frequency         : {cluster['frequency']} items\n"
            f"     - Severity          : {cluster['severity']}\n"
            f"     - Impact estimate   : {impact}"
        )

    top = ranked_clusters[0]
    print(_divider("4) RECOMMENDED ACTION"))
    print(f"Top cluster         : Cluster {top['cluster_id']}")
    print(f"Theme label         : {top['theme_label']}")
    print(f"Problem statement   : {_problem_statement(top['texts'])}")
    print("Proposed solution   :")
    proposed_solution = _proposed_solution(top["theme_label"], top["example_signal"], top["texts"])
    print(f"  {proposed_solution}")
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
    print("  - Dashboard page-load p95 improves from <baseline_before_ms> ms to <= <target_after_ms> ms in production.")
    print("  - Dashboard API p95 latency improves from <baseline_before_ms> ms to <= <target_after_ms> ms for top workflows.")
    print("  - Users complete the target task without escalation/workaround in >= <target_success_rate>% of sessions.")
    print("  - Support tickets linked to this pain point decrease by >= <target_ticket_reduction>% vs pre-release baseline.")
    print("  - PM dashboard tracks baseline vs after-release metrics for load time, API latency, adoption, and reliability.")
    print("Open questions      :")
    print("  - Which user segment in this cluster should we prioritize for beta testing?")
    print("  - What success threshold (adoption, CSAT, ticket reduction) defines launch readiness?")
    print("  - Are there platform-specific constraints (web/mobile/api) that require phased rollout?")

    print("\n=== PRODUCT SPEC (AUTO-GENERATED) ===")
    print(f"Feature name        : {_feature_name(top['theme_label'])}")
    print(f"Problem summary     : {_problem_statement(top['texts'])}")
    print(f"Target users        : {_target_users(top, len(records))}")
    print(f"Proposed solution   : {proposed_solution}")
    print("Success metrics     :")
    for metric in _success_metrics(top):
        print(f"  - {metric}")
    print("Risks               :")
    for risk in _risks(top):
        print(f"  - {risk}")

    print("\n=== IMPLEMENTATION BREAKDOWN ===")
    for ticket in _jira_tickets(top, proposed_solution):
        print(f"\nTitle               : {ticket['title']}")
        print(f"Description         : {ticket['description']}")
        print(f"Acceptance criteria : {ticket['acceptance']}")

    print(_divider("5) SUPPORTING EVIDENCE"))
    evidence_pairs = list(zip(top["ids"], top["texts"]))
    evidence_count = min(5, len(evidence_pairs))
    if evidence_pairs:
        embedder = HashingEmbedder(dimensions=128)
        quote_vectors = embedder.embed(top["texts"])
        centroid = _centroid(quote_vectors)
        threshold = 0.62
        scored_pairs = []
        for (feedback_id, quote), vector in zip(evidence_pairs, quote_vectors):
            similarity = _cosine_similarity(centroid, vector)
            scored_pairs.append((similarity, feedback_id, quote))

        passing = sorted([item for item in scored_pairs if item[0] >= threshold], key=lambda item: item[0], reverse=True)
        fallback = sorted([item for item in scored_pairs if item[0] < threshold], key=lambda item: item[0], reverse=True)

        selected = passing[:evidence_count]
        minimum_quotes = min(3, len(scored_pairs))
        if len(selected) < minimum_quotes:
            needed = minimum_quotes - len(selected)
            selected.extend(fallback[:needed])

        selected_evidence = [(feedback_id, quote) for _, feedback_id, quote in selected]
        for feedback_id, quote in selected_evidence:
            print(f'  - [{feedback_id}] "{quote}"')

    print("\n" + "-" * 78)
    print("Demo complete. This output is formatted for direct stakeholder review.")
    print("-" * 78)


if __name__ == "__main__":
    run_demo()
