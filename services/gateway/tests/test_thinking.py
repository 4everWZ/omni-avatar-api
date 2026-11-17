import copy
import unittest

from services.gateway.thinking import apply_thinking_default


class ThinkingDefaultsTest(unittest.TestCase):
    def test_defaults_change_neither_official_budgets_nor_sampling(self):
        payload = {'model': 'fixture', 'include_reasoning': False}
        apply_thinking_default(payload)
        self.assertEqual(payload, {'model': 'fixture', 'include_reasoning': False,
                                  'chat_template_kwargs': {'enable_thinking': True}})

    def test_explicit_disable_effort_and_budget_remain_caller_owned(self):
        cases = [
            {'chat_template_kwargs': {'enable_thinking': False, 'other': 7}},
            {'reasoning': {'effort': 'none'}}, {'reasoning_effort': 'none'},
            {'thinking_token_budget': 0, 'max_output_tokens': 512},
        ]
        for original in cases:
            with self.subTest(original=original):
                payload = copy.deepcopy(original)
                apply_thinking_default(payload)
                enabled = original.get('chat_template_kwargs', {}).get('enable_thinking',
                    original.get('reasoning', {}).get('effort') != 'none'
                    and original.get('reasoning_effort') != 'none')
                self.assertEqual(payload['chat_template_kwargs']['enable_thinking'], enabled)
                for key, value in original.items():
                    if key != 'chat_template_kwargs':
                        self.assertEqual(payload[key], value)
