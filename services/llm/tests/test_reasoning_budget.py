from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.llm.patches.patch_reasoning_budget import main, patch_source, verify_native


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

    def test_verify_native_accepts_defined_field_and_fails_closed_when_missing(self):
        native_source = (
            'class ResponsesRequest:\n'
            '    reasoning: Reasoning | None = None\n'
            '    thinking_token_budget: ThinkingTokenBudget = None\n'
        )
        verify_native(native_source)
        for missing in ('', '    reasoning: Reasoning | None = None\n',
                        '# thinking_token_budget: commented out\n', 'thinking_token_budget\n'):
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                verify_native(missing)

    def test_main_handles_032_034_and_rejects_other_versions(self):
        class FakePackage:
            def __init__(self, version, protocol_path):
                self.version = version
                self.protocol_path = protocol_path

            def locate_file(self, relative):
                self.located = relative
                return self.protocol_path

        with tempfile.TemporaryDirectory() as folder:
            protocol = Path(folder) / 'protocol.py'
            native = 'class ResponsesRequest:\n    thinking_token_budget: ThinkingTokenBudget = None\n'
            protocol.write_text(native)
            with patch('services.llm.patches.patch_reasoning_budget.distribution',
                       return_value=FakePackage('0.34.0+cu128', protocol)):
                main()
            self.assertEqual(protocol.read_text(), native)

            protocol.write_text('class ResponsesRequest:\n    pass\n')
            with patch('services.llm.patches.patch_reasoning_budget.distribution',
                       return_value=FakePackage('0.34.0', protocol)):
                with self.assertRaises(RuntimeError):
                    main()

            legacy = ('from vllm.sampling_params import (\n    SamplingParams,\n)\n'
                      'class ResponsesRequest:\n'
                      '    reasoning: Reasoning | None = None\n    include_reasoning: bool = Field(default=True)\n'
                      '    def to_params(self):\n'
                      '        return SamplingParams(\n'
                      '            max_tokens=max_tokens,\n'
                      '        )\n')
            protocol.write_text(legacy)
            with patch('services.llm.patches.patch_reasoning_budget.distribution',
                       return_value=FakePackage('0.32.0', protocol)):
                main()
            self.assertIn('thinking_token_budget: ThinkingTokenBudget = None', protocol.read_text())

            with patch('services.llm.patches.patch_reasoning_budget.distribution',
                       return_value=FakePackage('0.35.0', protocol)):
                with self.assertRaises(RuntimeError):
                    main()
