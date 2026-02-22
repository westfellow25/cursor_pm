import unittest

from feedback_analysis import opportunity_report


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


if __name__ == "__main__":
    unittest.main()
