"""Simple end-to-end demo for feedback ingestion, clustering, and recommendation output."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from feedback_analysis import opportunity_report
from feedback_ingestion import run_pipeline


def run_demo(csv_path: str | Path = "example_data/feedback.csv", n_clusters: int = 3) -> None:
    csv_path = Path(csv_path)

    print("=" * 72)
    print("AI Product Discovery - End-to-End Demo")
    print("=" * 72)

    print(f"\n1) Loading example feedback CSV: {csv_path}")
    records, cluster_results = run_pipeline(csv_path, n_clusters=n_clusters)
    print(f"   Loaded {len(records)} feedback records.")

    print("\n2) Running ingestion + embedding + clustering")
    grouped_ids: dict[int, list[str]] = defaultdict(list)
    for row in cluster_results:
        grouped_ids[row.cluster_id].append(row.feedback_id)

    print(f"   Created {len(grouped_ids)} clusters:\n")
    for cluster_id in sorted(grouped_ids):
        ids = grouped_ids[cluster_id]
        preview = next((r.text for r in records if r.feedback_id == ids[0]), "")
        print(f"   - Cluster {cluster_id}: {len(ids)} items | IDs: {', '.join(ids)}")
        print(f"     Example: {preview}")

    print("\n3) Computing opportunity scores")
    report = opportunity_report([record.text for record in records])

    if not report:
        print("   No opportunities found.")
        return

    print(f"   Computed {len(report)} opportunities. Top opportunities:")
    for index, item in enumerate(report[:3], start=1):
        print(
            f"   {index}. score={item['opportunity_score']}, "
            f"freq={item['frequency']}, summary={item['problem_summary']}"
        )

    top = report[0]
    print("\n4) Generating one feature recommendation")
    print(f"   Recommended feature: {top['suggested_feature']}")

    print("\n5) Supporting evidence")
    for quote in top["supporting_evidence_quotes"]:
        print(f"   - {quote}")

    print("\nDone.")


if __name__ == "__main__":
    run_demo()
