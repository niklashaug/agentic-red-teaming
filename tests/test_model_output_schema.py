import json
import unittest

from main import ModelOutputParseError, parse_model_output


class ModelOutputSchemaTests(unittest.TestCase):
    def test_valid_final_output_parses(self) -> None:
        parsed, details = parse_model_output(
            json.dumps({"action": "final", "message": "Done."})
        )

        self.assertEqual(parsed, {"action": "final", "message": "Done."})
        self.assertEqual(details["parse_mode"], "strict_json")

    def test_unknown_tool_name_is_rejected(self) -> None:
        with self.assertRaises(ModelOutputParseError) as context:
            parse_model_output(
                json.dumps(
                    {
                        "action": "tool_call",
                        "name": "delete_repo",
                        "arguments": {},
                    }
                )
            )

        self.assertEqual(context.exception.details["parse_mode"], "invalid_schema")

    def test_extra_output_fields_are_rejected(self) -> None:
        with self.assertRaises(ModelOutputParseError) as context:
            parse_model_output(
                json.dumps(
                    {
                        "action": "final",
                        "message": "Done.",
                        "extra": "not allowed",
                    }
                )
            )

        self.assertEqual(context.exception.details["parse_mode"], "invalid_schema")

    def test_trailing_text_is_rejected(self) -> None:
        with self.assertRaises(ModelOutputParseError) as context:
            parse_model_output('{"action":"final","message":"Done."} trailing')

        self.assertEqual(context.exception.details["parse_mode"], "invalid_json")


if __name__ == "__main__":
    unittest.main()
