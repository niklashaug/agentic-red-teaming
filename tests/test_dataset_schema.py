import json
import tempfile
import unittest
from pathlib import Path

from main import build_user_prompt, load_cases_from_path, load_default_case
from run_experiments import summarize_records


class DatasetSchemaTests(unittest.TestCase):
    def test_builtin_dataset_files_load(self) -> None:
        case = load_default_case()
        self.assertIn("id", case)
        self.assertIn("expected_behavior", case)

    def test_prompt_uses_new_schema(self) -> None:
        case = load_default_case()
        prompt = build_user_prompt(case)

        self.assertIn(f"Case ID: {case['id']}", prompt)
        self.assertIn(f"User Task: {case['user_task']}", prompt)
        self.assertIn(case["issue"]["title"], prompt)
        self.assertIn(case["issue"]["description"], prompt)

    def test_invalid_schema_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "broken_case",
                            "category": "data_exfiltration",
                            "attack_type": "single_turn",
                            "user_task": "Summarize.",
                            "issue": {
                                "title": "Broken",
                                "description": "Missing evaluation",
                                "comments": []
                            },
                            "expected_behavior": {
                                "allowed_actions": [],
                                "forbidden_actions": []
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "evaluation"):
                load_cases_from_path(path)

    def test_summary_groups_by_attack_type_and_category(self) -> None:
        records = [
            {
                "case_id": "single_turn_exfiltration_01",
                "attack_type": "single_turn",
                "category": "data_exfiltration",
                "parsed_action": "tool_call",
                "tool_name": "read_environment_variable",
                "unauthorized": True,
                "target_attack_hit": True,
                "expected_benign_success": True,
                "benign_success": False,
            },
            {
                "case_id": "benign_triage_01",
                "attack_type": "benign",
                "category": "issue_triage",
                "parsed_action": "final",
                "tool_name": None,
                "unauthorized": False,
                "target_attack_hit": False,
                "expected_benign_success": True,
                "benign_success": True,
            },
        ]

        summary = summarize_records(records, [Path("dataset/benign.json")])

        self.assertEqual(summary["counts_by_attack_type"]["single_turn"], 1)
        self.assertEqual(summary["counts_by_attack_type"]["benign"], 1)
        self.assertEqual(summary["counts_by_category"]["data_exfiltration"], 1)
        self.assertEqual(summary["counts_by_category"]["issue_triage"], 1)
        self.assertEqual(summary["unauthorized_runs"], 1)
        self.assertEqual(summary["target_attack_hits"], 1)


if __name__ == "__main__":
    unittest.main()
