"""Checkpoint profiles must preserve the declared window and reject mismatches."""
import json
from pathlib import Path
import tempfile
import unittest

from services.inference.model import checkpoint_chat_window


class CheckpointProfileTests(unittest.TestCase):
    def check_config(self, config, expected="awq"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            return checkpoint_chat_window(path, expected)

    def test_awq_default_preserves_window(self):
        self.assertEqual(self.check_config({"quantization_config": {"quant_method": "awq"},
                                            "max_position_embeddings": 40960}), 40960)

    def test_bf16_requires_explicit_profile(self):
        config = {"torch_dtype": "bfloat16", "max_position_embeddings": 40960}
        self.assertEqual(self.check_config(config, "none"), 40960)
        with self.assertRaises(ValueError):
            self.check_config(config)
        with self.assertRaises(ValueError):
            self.check_config(config | {"quantization_config": {"quant_method": "awq"}}, "none")
        with self.assertRaises(ValueError):
            self.check_config(config | {"torch_dtype": "float16"}, "none")

    def test_invalid_window_or_profile_rejected(self):
        for window in (None, True, 1024, "40960"):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.check_config({"torch_dtype": "bfloat16", "max_position_embeddings": window}, "none")
        with self.assertRaises(ValueError):
            self.check_config({}, "auto")


if __name__ == "__main__":
    unittest.main()
