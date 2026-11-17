import unittest
from types import SimpleNamespace

from services.inference.patches.patch_minicpmo_awq_stages import patch_stage_source


PINNED_HOOK = """
class Base:
    def create_model_config(self):
        return self.quantization, self.hf_overrides

class Args(Base):
    def create_model_config(self):
        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
        finally:
            pass
        return model_config
"""


class MiniCPMOStagePatchTest(unittest.TestCase):
    def setUp(self):
        namespace = {}
        exec(patch_stage_source(PINNED_HOOK), namespace)
        self.args = namespace["Args"]()
        self.args.model_arch = "MiniCPMO45OmniForConditionalGeneration"
        self.args.quantization = "awq"

    def test_thinker_and_other_models_keep_checkpoint_quantization(self):
        original = {"quantization_config": {"quant_method": "awq"}}
        self.args.hf_overrides = original
        self.args.model_stage = "llm"
        self.assertEqual(self.args.create_model_config(), ("awq", original))
        self.args.model_stage = "tts"
        self.args.model_arch = "OtherOmniModel"
        self.assertEqual(self.args.create_model_config(), ("awq", original))

    def test_talker_and_code2wav_are_unquantized_before_base_model_initialization(self):
        original = {"architectures": ["MiniCPMO45OmniForConditionalGeneration"],
                    "quantization_config": {"quant_method": "awq"}}
        for stage, architecture in (("tts", "MiniCPMO45OmniForConditionalGeneration"),
                                    ("code2wav", "MiniCPMO45Code2Wav")):
            with self.subTest(stage=stage):
                self.args.model_stage = stage
                self.args.model_arch = architecture
                self.args.quantization = "auto_awq"
                self.args.hf_overrides = original
                self.assertEqual(self.args.create_model_config(), (None, original | {"quantization_config": None}))
                self.assertEqual(original["quantization_config"], {"quant_method": "awq"})

    def test_existing_callable_overrides_are_preserved(self):
        def overrides(cfg):
            cfg.extra_value = 42
            return cfg

        self.args.model_stage = "tts"
        self.args.hf_overrides = overrides
        quantization, apply = self.args.create_model_config()
        result = apply(SimpleNamespace(quantization_config={"quant_method": "awq"}))
        self.assertIsNone(quantization)
        self.assertIsNone(result.quantization_config)
        self.assertEqual(result.extra_value, 42)

    def test_refuses_a_changed_or_already_patched_upstream(self):
        with self.assertRaises(RuntimeError):
            patch_stage_source("unrelated source")
        with self.assertRaises(RuntimeError):
            patch_stage_source(patch_stage_source(PINNED_HOOK))
