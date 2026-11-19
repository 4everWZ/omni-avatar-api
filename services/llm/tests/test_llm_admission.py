import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from services.llm.llm import serve
from services.llm.patches.patch_llm_admission import patch_api_source


class InferenceAdmissionTests(unittest.TestCase):
    def launch(self, extra):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.snapshot-ready').write_text('model@pin\n')
            (root / 'config.json').write_text(json.dumps({'max_position_embeddings': 262144}))
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
            with patch.dict(os.environ, extra, clear=True), patch('os.execvp') as execute:
                serve('model', 'pin', root)
                command = execute.call_args.args[1]
                command[command.index('--model') + 1] = '<checkpoint>'
                return command

    def test_shared_gpu_preserves_context_and_real_batching(self):
        command = self.launch({})
        for flag, value in [('--max-model-len', '262144'), ('--max-num-seqs', '4'),
                            ('--kv-cache-memory-bytes', '9663676416')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--no-async-scheduling', command)
        self.assertNotIn('--enforce-eager', command)
        config = json.loads(command[command.index('--compilation-config') + 1])
        self.assertEqual(config['cudagraph_mode'], 'FULL_DECODE_ONLY')
        self.assertEqual(config['cudagraph_capture_sizes'], [1, 2, 4])

    def test_background_qwen_enables_bounded_image_input(self):
        command = self.launch({})
        self.assertNotIn('--language-model-only', command)
        self.assertEqual(json.loads(command[command.index('--limit-mm-per-prompt') + 1]),
                         {'image': 8, 'video': 0})
        self.assertIn('--language-model-only', self.launch({'LLM_MAX_IMAGES_PER_REQUEST': '0'}))
        with self.assertRaises(ValueError):
            self.launch({'LLM_MAX_IMAGES_PER_REQUEST': '9'})

    def test_withdrawn_pause_setting_cannot_reenable_throttling(self):
        self.assertEqual(self.launch({}), self.launch({'LLM_GPU_STEP_PAUSE_MS': '250'}))

    def test_router_hook_fails_closed_on_version_drift_or_reapplication(self):
        source = 'def attach_router(app: FastAPI):\n    app.include_router(router)\n'
        patched = patch_api_source(source)
        self.assertIn('app.include_router(admission_router)', patched)
        with self.assertRaises(RuntimeError):
            patch_api_source(patched)
        with self.assertRaises(RuntimeError):
            patch_api_source('def attach_router(app):\n    pass\n')
