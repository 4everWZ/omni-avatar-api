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

    def test_default_budget_injection_and_explicit_overrides(self):
        payload = {'model': 'fixture'}
        apply_thinking_default(payload, default_budget=16384)
        self.assertEqual(payload, {
            'model': 'fixture',
            'chat_template_kwargs': {'enable_thinking': True},
            'thinking_token_budget': 16384,
        })

        zero_default = {'model': 'fixture'}
        apply_thinking_default(zero_default, default_budget=0)
        self.assertEqual(zero_default['thinking_token_budget'], 0)

        for original in [
            {'thinking_token_budget': 0},
            {'thinking_token_budget': None},
            {'thinking_token_budget': 4096},
        ]:
            with self.subTest(original=original):
                case = copy.deepcopy(original)
                apply_thinking_default(case, default_budget=16384)
                self.assertEqual(case['thinking_token_budget'], original['thinking_token_budget'])

        for disabled in [
            {'chat_template_kwargs': {'enable_thinking': False}},
            {'reasoning': {'effort': 'none'}},
            {'reasoning_effort': 'none'},
        ]:
            with self.subTest(disabled=disabled):
                case = copy.deepcopy(disabled)
                apply_thinking_default(case, default_budget=16384)
                self.assertNotIn('thinking_token_budget', case)
