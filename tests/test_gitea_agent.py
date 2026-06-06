import json
import sys
import types
import unittest

from gitea_agent import (
    extract_tool_call,
    extract_labels_from_final_message,
    extract_monitoring_query_from_issue,
    extract_trace_id_from_issue,
    materialize_final_decision,
    materialize_label_claims,
    message_for_history,
    normalize_labels_argument,
    normalize_visible_comment_text,
    resolve_monitoring_query_argument,
    resolve_issue_id_argument,
    resolve_trace_id_argument,
    run_issue_agent,
)


class FakeGiteaClient:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.comments: list[str] = []
        self.issue = {
            "number": 12,
            "title": "Typo in settings page copy",
            "body": "The settings page contains a typo near the email toggle.",
        }

    def get_issue(self, issue_id: int) -> dict[str, object]:
        return self.issue

    def manage_issue_labels(
        self, issue_id: int, labels: list[str]
    ) -> list[dict[str, object]]:
        self.labels = labels
        return [{"name": label} for label in labels]

    def post_issue_comment(self, issue_id: int, text: str) -> dict[str, object]:
        self.comments.append(text)
        return {"body": text}

    def list_issue_comments(self, issue_id: int) -> list[dict[str, object]]:
        return [{"body": comment} for comment in self.comments]


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

    def test_extract_labels_from_final_message_reads_unquoted_label(self) -> None:
        self.assertEqual(
            extract_labels_from_final_message("I've added the needs-info label."),
            ["needs-info"],
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

    def test_extract_tool_call_reads_string_arguments(self) -> None:
        name, arguments = extract_tool_call(
            {
                "function": {
                    "name": "post_issue_comment",
                    "arguments": json.dumps(
                        {"issue_id": 12, "text": "Thanks for the report."}
                    ),
                }
            }
        )

        self.assertEqual(name, "post_issue_comment")
        self.assertEqual(arguments["issue_id"], 12)
        self.assertEqual(arguments["text"], "Thanks for the report.")

    def test_message_for_history_defaults_to_assistant_role(self) -> None:
        self.assertEqual(
            message_for_history({"content": "Done."}),
            {"content": "Done.", "role": "assistant"},
        )

    def test_normalize_visible_comment_text_unwraps_tool_json(self) -> None:
        text = normalize_visible_comment_text(
            json.dumps(
                {
                    "action": "post_issue_comment",
                    "issue_id": 12,
                    "text": "Thank you for reporting the typo.",
                }
            )
        )

        self.assertEqual(text, "Thank you for reporting the typo.")

    def test_normalize_visible_comment_text_rewrites_reach_out(self) -> None:
        text = normalize_visible_comment_text(
            "I've added the 'needs-info' label and will reach out to the "
            "reporter for more information."
        )

        self.assertNotIn("reach out", text.casefold())
        self.assertIn("Please provide the missing information.", text)

    def test_materialize_final_decision_unwraps_json_and_applies_label(
        self,
    ) -> None:
        client = FakeGiteaClient()
        final_message = json.dumps(
            {
                "action": "post_issue_comment",
                "issue_id": 12,
                "text": (
                    "Thank you for reporting the typo. I've labeled it as "
                    "'needs-info'. Please provide the exact location."
                ),
            }
        )

        actions = materialize_final_decision(
            client,
            12,
            final_message,
            1,
            prior_tool_calls=[],
        )

        self.assertEqual([action["name"] for action in actions], [
            "set_issue_labels",
            "post_issue_comment",
        ])
        self.assertEqual(client.labels, ["needs-info"])
        self.assertEqual(
            client.comments,
            [
                "Thank you for reporting the typo. I've labeled it as "
                "'needs-info'. Please provide the exact location."
            ],
        )
        self.assertFalse(client.comments[0].startswith("{"))

    def test_materialize_label_claims_applies_label_before_comment(self) -> None:
        client = FakeGiteaClient()

        actions = materialize_label_claims(
            client,
            12,
            "I've added the needs-info label. Please provide the failing input.",
            1,
            prior_tool_calls=[],
        )

        self.assertEqual([action["name"] for action in actions], [
            "set_issue_labels"
        ])
        self.assertEqual(client.labels, ["needs-info"])

    def test_run_issue_agent_executes_native_tool_calls(self) -> None:
        client = FakeGiteaClient()
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "set_issue_labels",
                                "arguments": {
                                    "issue_id": 12,
                                    "labels": ["needs-info"],
                                },
                            }
                        },
                        {
                            "function": {
                                "name": "post_issue_comment",
                                "arguments": {
                                    "issue_id": 12,
                                    "text": (
                                        "I've added the needs-info label. "
                                        "Please provide the exact location."
                                    ),
                                },
                            }
                        },
                    ],
                }
            }
        ]
        snapshots = self.install_fake_ollama(responses)

        result = run_issue_agent(12, client=client, model="qwen3", max_steps=2)

        self.assertEqual(result.final_action, "tool_call")
        self.assertEqual([call["name"] for call in result.tool_calls], [
            "set_issue_labels",
            "post_issue_comment",
        ])
        self.assertEqual(client.labels, ["needs-info"])
        self.assertEqual(
            client.comments,
            [
                "I've added the needs-info label. Please provide the exact "
                "location."
            ],
        )
        self.assertTrue(snapshots[0]["tools"])

    def test_run_issue_agent_returns_tool_results_to_model(self) -> None:
        client = FakeGiteaClient()
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_issue",
                                "arguments": {"issue_id": 12},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a low-priority copy issue.",
                }
            },
        ]
        snapshots = self.install_fake_ollama(responses)

        result = run_issue_agent(12, client=client, model="qwen3", max_steps=2)

        self.assertEqual(result.final_action, "final")
        self.assertEqual(result.tool_calls[0]["name"], "read_issue")
        self.assertEqual(client.comments, ["This is a low-priority copy issue."])
        tool_messages = [
            message
            for message in snapshots[1]["messages"]
            if message.get("role") == "tool"
        ]
        self.assertEqual(tool_messages[0]["tool_name"], "read_issue")

    def install_fake_ollama(
        self, responses: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        original_module = sys.modules.get("ollama")

        def fake_chat(**kwargs: object) -> dict[str, object]:
            snapshots.append(kwargs)
            return responses.pop(0)

        sys.modules["ollama"] = types.SimpleNamespace(chat=fake_chat)
        self.addCleanup(self.restore_ollama_module, original_module)
        return snapshots

    def restore_ollama_module(self, original_module: object | None) -> None:
        if original_module is None:
            sys.modules.pop("ollama", None)
        else:
            sys.modules["ollama"] = original_module


if __name__ == "__main__":
    unittest.main()
