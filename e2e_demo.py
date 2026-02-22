"""Simple end-to-end demo for feedback ingestion, clustering, and recommendation output."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from feedback_analysis import opportunity_report
from feedback_ingestion import run_pipeline


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
    for row in cluster_results:
        grouped_ids[row.cluster_id].append(row.feedback_id)

    print(f"Clusters generated : {len(grouped_ids)}")
    for cluster_id in sorted(grouped_ids):
        ids = grouped_ids[cluster_id]
        preview = next((r.text for r in records if r.feedback_id == ids[0]), "")
        print(f"\n• Cluster {cluster_id}")
        print(f"  - Size            : {len(ids)} items")
        print(f"  - Record IDs      : {', '.join(ids)}")
        print(f"  - Example signal  : {preview}")

    print(_divider("3) OPPORTUNITY SCORING"))
    report = opportunity_report([record.text for record in records])

    if not report:
        print("No opportunities identified from this sample.")
        return

    print(f"Opportunities found: {len(report)}")
    print("Top opportunities   :")
    for index, item in enumerate(report[:3], start=1):
        impact = _impact_estimate(item["opportunity_score"], item["frequency"], len(records))
        print(
            f"  {index}. {item['problem_summary']}\n"
            f"     • Opportunity score : {item['opportunity_score']}\n"
            f"     • Frequency         : {item['frequency']} mentions\n"
            f"     • Impact estimate   : {impact}"
        )

    top = report[0]
    print(_divider("4) RECOMMENDED ACTION"))
    print(f"Recommendation      : {top['suggested_feature']}")
    print(
        f"Expected impact     : {_impact_estimate(top['opportunity_score'], top['frequency'], len(records))}"
    )

    print(_divider("5) SUPPORTING EVIDENCE"))
    for quote in top["supporting_evidence_quotes"]:
        print(f"  ▸ “{quote}”")

    print("\n" + "-" * 78)
    print("Demo complete. This output is formatted for direct stakeholder review.")
    print("-" * 78)


if __name__ == "__main__":
    run_demo()
