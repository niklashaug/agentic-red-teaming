import base64
import unittest

from gitea_tools import decode_gitea_content_payload, normalize_log_trace_id


class GiteaToolsTests(unittest.TestCase):
    def test_decode_gitea_content_payload_handles_wrapped_base64(self) -> None:
        content = "FLAG_SECRET_API_KEY=12345\n"
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        wrapped = f"{encoded[:12]}\n{encoded[12:]}"

        self.assertEqual(
            decode_gitea_content_payload({"content": wrapped}),
            content,
        )

    def test_decode_gitea_content_payload_requires_content_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "base64 string"):
            decode_gitea_content_payload({"name": "README.md"})

    def test_normalize_log_trace_id_accepts_model_variants(self) -> None:
        self.assertEqual(normalize_log_trace_id(999), "999")
        self.assertEqual(normalize_log_trace_id("trace_id=999"), "999")
        self.assertEqual(normalize_log_trace_id("Trace 999."), "999")
        self.assertEqual(normalize_log_trace_id({"trace_id": "999"}), "999")

    def test_normalize_log_trace_id_rejects_missing_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric trace_id"):
            normalize_log_trace_id("latest trace")


if __name__ == "__main__":
    unittest.main()
