import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from services.inference.llm import serve
from services.inference.patches.patch_llm_admission import patch_core_source


class InferenceAdmissionTests(unittest.TestCase):
    def launch(self, extra):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.snapshot-ready').write_text('model@pin\n')
            (root / 'config.json').write_text(json.dumps({'max_position_embeddings': 262144}))
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
            with patch.dict(os.environ, extra, clear=True), patch('os.execvp') as execute:
                serve('model', 'pin', root)
                return execute.call_args.args[1]

    def test_shared_gpu_preserves_context_and_real_batching(self):
        command = self.launch({'LLM_GPU_STEP_PAUSE_MS': '40'})
        for flag, value in [('--max-model-len', '262144'), ('--max-num-seqs', '4'),
                            ('--kv-cache-memory-bytes', '9663676416')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--no-async-scheduling', command)
        self.assertNotIn('--no-async-scheduling', self.launch({}))

    def test_invalid_gpu_pause_rejected(self):
        for value in ('nan', 'inf', '-1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.launch({'LLM_GPU_STEP_PAUSE_MS': value})

    def test_compute_yield_only_after_executed_batch(self):
        source = ('class Core:\n'
                  '    def step(self, scheduler_output):\n'
                  '        engine_core_outputs = {}\n'
                  '        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0\n')
        clock = SimpleNamespace(sleep=Mock())
        namespace = {'os': os, 'time': clock}
        exec(patch_core_source(source), namespace)
        core = namespace['Core']()
        with patch.dict(os.environ, {'LLM_GPU_STEP_PAUSE_MS': '40'}):
            core.step(SimpleNamespace(total_num_scheduled_tokens=4))
            clock.sleep.assert_called_once_with(.04)
            clock.sleep.reset_mock()
            core.step(SimpleNamespace(total_num_scheduled_tokens=0))
            clock.sleep.assert_not_called()
        with self.assertRaises(RuntimeError):
            patch_core_source(patch_core_source(source))
