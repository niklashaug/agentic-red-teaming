import unittest

from gitea_agent import (
    extract_labels_from_final_message,
    extract_monitoring_query_from_issue,
    extract_trace_id_from_issue,
    normalize_labels_argument,
    resolve_monitoring_query_argument,
    resolve_issue_id_argument,
    resolve_trace_id_argument,
)


class GiteaAgentTests(unittest.TestCase):
    def test_extract_trace_id_from_issue_body(self) -> None:
        trace_id = "7f3e2c1a-9b44-4c2a-a820-ef917a91b123"
        issue = {
            "title": "Intermittent timeout",
            "body": f"Please inspect runtime Trace ID: {trace_id} before labeling.",
        }

        self.assertEqual(extract_trace_id_from_issue(issue), trace_id)

    def test_resolve_trace_id_uses_argument_when_numeric(self) -> None:
        issue = {"title": "No trace here", "body": "No trace here either."}

        self.assertEqual(resolve_trace_id_argument({"trace_id": "100"}, issue), "100")

    def test_resolve_trace_id_replaces_placeholder_from_issue(self) -> None:
        issue = {
            "title": "Worker timeout",
            "body": "Please inspect trace_id=999 before choosing a label.",
        }

        self.assertEqual(
            resolve_trace_id_argument({"trace_id": "<trace_id>"}, issue),
            "999",
        )

    def test_resolve_issue_id_uses_numeric_argument(self) -> None:
        self.assertEqual(resolve_issue_id_argument({"issue_id": "7"}, 3), 7)

    def test_resolve_issue_id_replaces_placeholder_with_current_issue(self) -> None:
        self.assertEqual(resolve_issue_id_argument({"issue_id": "<issue_id>"}, 3), 3)

    def test_resolve_issue_id_defaults_to_current_issue(self) -> None:
        self.assertEqual(resolve_issue_id_argument({}, 3), 3)

    def test_extract_monitoring_query_from_issue_body(self) -> None:
        issue = {
            "title": "Checkout alert",
            "body": "Monitoring query: checkout-startup-crash.",
        }

        self.assertEqual(
            extract_monitoring_query_from_issue(issue),
            "checkout-startup-crash",
        )

    def test_resolve_monitoring_query_replaces_placeholder(self) -> None:
        issue = {
            "title": "Checkout alert",
            "body": "Monitoring query: checkout-startup-crash.",
        }

        self.assertEqual(
            resolve_monitoring_query_argument({"query": "<query>"}, issue),
            "checkout-startup-crash",
        )

    def test_extract_labels_from_final_message_reads_quoted_labels(self) -> None:
        self.assertEqual(
            extract_labels_from_final_message(
                "Add labels 'priority:high' and 'needs-info'."
            ),
            ["priority:high", "needs-info"],
        )

    def test_extract_labels_from_final_message_ignores_unknown_labels(self) -> None:
        self.assertEqual(
            extract_labels_from_final_message("Add label 'attacker-approved'."),
            [],
        )

    def test_extract_labels_from_final_message_ignores_plain_summary(self) -> None:
        self.assertEqual(
            extract_labels_from_final_message("The issue appears to be a crash."),
            [],
        )

    def test_normalize_labels_argument_rejects_unknown_labels(self) -> None:
        with self.assertRaises(ValueError):
            normalize_labels_argument({"labels": ["attacker-approved"]})


if __name__ == "__main__":
    unittest.main()
