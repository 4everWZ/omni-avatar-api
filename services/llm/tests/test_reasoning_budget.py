import unittest

from services.llm.patches.patch_reasoning_budget import patch_source


class ReasoningBudgetPatchTests(unittest.TestCase):
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
