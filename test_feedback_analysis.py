import unittest

from feedback_analysis import opportunity_report


from e2e_demo import _select_supporting_evidence


class FeedbackAnalysisTests(unittest.TestCase):
    def test_report_contains_required_fields(self):
        feedback = [
            "I cannot find old customer records quickly. Search is too limited.",
            "The search and filtering tools are weak; finding archived accounts is hard.",
            "Exporting reports is manual and takes too much time every week.",
        ]

        report = opportunity_report(feedback)

        self.assertGreaterEqual(len(report), 2)
        top = report[0]
        self.assertIn("problem_summary", top)
        self.assertIn("frequency", top)
        self.assertIn("suggested_feature", top)
        self.assertIn("supporting_evidence_quotes", top)
        self.assertIn("opportunity_score", top)

    def test_similar_feedback_clusters_together(self):
        feedback = [
            "We need better API sync with our CRM.",
            "CRM integration is missing so we manually re-enter everything.",
            "Please add role-based access permissions.",
        ]

        report = opportunity_report(feedback)
        integration_cluster = [
            item
            for item in report
            if "integrations" in item["suggested_feature"].lower()
            or "api sync" in item["suggested_feature"].lower()
        ]
        self.assertTrue(integration_cluster)
        self.assertEqual(integration_cluster[0]["frequency"], 2)


class EvidenceSelectionTests(unittest.TestCase):
    def test_top_cluster_with_three_items_returns_three_quotes_with_fallback(self):
        scored_pairs = [
            (0.95, True, "f-1", "search is slow"),
            (0.35, False, "f-2", "dashboard times out"),
            (0.22, False, "f-3", "api page lags"),
        ]

        selected = _select_supporting_evidence(scored_pairs, evidence_count=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0][0], "f-1")
        self.assertCountEqual([item[0] for item in selected], ["f-1", "f-2", "f-3"])

    def test_when_relevance_pool_has_three_items_only_same_cluster_quotes_used(self):
        scored_pairs = [
            (0.90, True, "f-1", "search misses records"),
            (0.75, True, "f-2", "filters feel broken"),
            (0.71, True, "f-3", "query parsing is weak"),
            (0.40, False, "f-4", "other low signal"),
        ]

        selected = _select_supporting_evidence(scored_pairs, evidence_count=4)

        self.assertGreaterEqual(len(selected), 3)
        self.assertTrue(set(["f-1", "f-2", "f-3"]).issubset({item[0] for item in selected}))


if __name__ == "__main__":
    unittest.main()
