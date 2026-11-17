import unittest

from services.inference.patches.patch_reasoning_budget import patch_source, patch_minicpmo_config


class ReasoningBudgetPatchTests(unittest.TestCase):
    def test_thinker_budget_configuration_does_not_override_other_models_or_existing_policy(self):
        source = '        if self.reasoning_config is not None and self.model_config is not None:\n'
        result = patch_minicpmo_config(source)
        self.assertIn('self.reasoning_config is None', result)
        self.assertIn('"stage_id", 0) == 0', result)
        self.assertIn('"model_type", None) == "minicpmo"', result)
        with self.assertRaises(RuntimeError):
            patch_minicpmo_config(result)

    def test_conversion_hook_fails_closed_on_drift_or_reapplication(self):
        source = ('from vllm.sampling_params import (\n)\n'
                  '    reasoning: Reasoning | None = None\n    include_reasoning: bool = Field(\n'
                  '            max_tokens=max_tokens,\n')
        result = patch_source(source)
        self.assertIn('thinking_token_budget: ThinkingTokenBudget = None', result)
        self.assertIn('thinking_token_budget=self.thinking_token_budget', result)
        for invalid in (result, '', source * 2):
            with self.assertRaises(RuntimeError):
                patch_source(invalid)
