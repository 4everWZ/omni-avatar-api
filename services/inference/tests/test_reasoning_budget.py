import unittest

from services.inference.patches.patch_reasoning_budget import patch_minicpmo_config


class ReasoningBudgetPatchTests(unittest.TestCase):
    def test_thinker_budget_configuration_does_not_override_other_models_or_existing_policy(self):
        source = '        if self.reasoning_config is not None and self.model_config is not None:\n'
        result = patch_minicpmo_config(source)
        self.assertIn('self.reasoning_config is None', result)
        self.assertIn('"stage_id", 0) == 0', result)
        self.assertIn('"model_type", None) == "minicpmo"', result)
        with self.assertRaises(RuntimeError):
            patch_minicpmo_config(result)
