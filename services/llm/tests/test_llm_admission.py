from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.llm.llm import location, serve, validate
from services.llm.patches.patch_llm_admission import main as patch_admission_main, patch_api_source


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
                            ('--max-num-batched-tokens', '2048'),
                            ('--gpu-memory-utilization', '0.75'),
                            ('--kv-cache-memory-bytes', '9663676416'),
                            ('--mamba-ssm-cache-dtype', 'float32'),
                            ('--reasoning-parser', 'qwen3'),
                            ('--tool-call-parser', 'qwen3_coder')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--no-async-scheduling', command)
        self.assertIn('--no-enable-prefix-caching', command)
        self.assertIn('--enable-auto-tool-choice', command)
        self.assertNotIn('--enforce-eager', command)
        self.assertNotIn('--tensor-parallel-size', command)
        self.assertNotIn('--speculative-config', command)
        self.assertNotIn('--enable-lora', command)
        config = json.loads(command[command.index('--compilation-config') + 1])
        self.assertEqual(config['cudagraph_mode'], 'FULL_DECODE_ONLY')
        self.assertEqual(config['cudagraph_capture_sizes'], [1, 2, 4])

    def test_background_qwen_enables_bounded_image_input(self):
        command = self.launch({})
        self.assertNotIn('--language-model-only', command)
        self.assertEqual(json.loads(command[command.index('--limit-mm-per-prompt') + 1]),
                         {'image': 8, 'video': 0})
        language_only = self.launch({'LLM_MAX_IMAGES_PER_REQUEST': '0', 'LLM_MM_ENCODER_TP_MODE': 'weights'})
        self.assertIn('--language-model-only', language_only)
        self.assertNotIn('--mm-encoder-tp-mode', language_only)
        with self.assertRaises(ValueError):
            self.launch({'LLM_MAX_IMAGES_PER_REQUEST': '9'})

    def test_withdrawn_pause_setting_cannot_reenable_throttling(self):
        self.assertEqual(self.launch({}), self.launch({'LLM_GPU_STEP_PAUSE_MS': '250'}))

    def test_tp2_mtp_prefix_caching_and_lora_configuration(self):
        command = self.launch({
            'LLM_TENSOR_PARALLEL_SIZE': '2',
            'LLM_GPU_MEMORY_UTILIZATION': '0.80',
            'LLM_KV_CACHE_BYTES': '',
            'LLM_KV_CACHE_DTYPE': 'fp8',
            'LLM_MAX_MODEL_LEN': '131072',
            'LLM_MAX_NUM_SEQS': '8',
            'LLM_MAX_NUM_BATCHED_TOKENS': '4096',
            'LLM_CUDAGRAPH_CAPTURE_SIZES': '1,2,4,8,16,32',
            'LLM_ASYNC_SCHEDULING': '1',
            'LLM_ENABLE_PREFIX_CACHING': '1',
            'LLM_MAMBA_CACHE_MODE': 'align',
            'LLM_MAMBA_BLOCK_SIZE': '4096',
            'LLM_MM_ENCODER_TP_MODE': 'weights',
            'LLM_ENABLE_MTP': '1',
            'LLM_MTP_NUM_SPECULATIVE_TOKENS': '3',
            'LLM_ENABLE_LORA': '1',
            'LLM_MAX_LORAS': '2',
            'LLM_MAX_LORA_RANK': '32',
            'LLM_GPU_CONTROL_TOKEN': 'secret',
        })
        for flag, value in [('--tensor-parallel-size', '2'),
                            ('--gpu-memory-utilization', '0.80'),
                            ('--kv-cache-dtype', 'fp8'),
                            ('--max-model-len', '131072'),
                            ('--max-num-seqs', '8'),
                            ('--max-num-batched-tokens', '4096'),
                            ('--mamba-cache-mode', 'align'),
                            ('--mamba-block-size', '4096'),
                            ('--mm-encoder-tp-mode', 'weights'),
                            ('--max-loras', '2'),
                            ('--max-lora-rank', '32')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn('--kv-cache-memory-bytes', command)
        self.assertNotIn('--no-async-scheduling', command)
        self.assertNotIn('--no-enable-prefix-caching', command)
        self.assertIn('--enable-prefix-caching', command)
        self.assertIn('--enable-lora', command)
        self.assertIn('--enable-sleep-mode', command)
        self.assertEqual(json.loads(command[command.index('--speculative-config') + 1]),
                         {'method': 'mtp', 'num_speculative_tokens': 3})
        config = json.loads(command[command.index('--compilation-config') + 1])
        self.assertEqual(config['cudagraph_capture_sizes'], [1, 2, 4, 8, 16, 32])

    def test_auto_kv_and_optional_flag_omission(self):
        for kv_val in ('', '0', 'auto'):
            with self.subTest(kv_val=kv_val):
                self.assertNotIn('--kv-cache-memory-bytes', self.launch({'LLM_KV_CACHE_BYTES': kv_val}))

        omitted = self.launch({
            'LLM_MAMBA_SSM_CACHE_DTYPE': '',
            'LLM_REASONING_PARSER': '',
            'LLM_TOOL_CALL_PARSER': '',
        })
        for flag in ('--mamba-ssm-cache-dtype', '--reasoning-parser', '--tool-call-parser',
                     '--enable-auto-tool-choice'):
            self.assertNotIn(flag, omitted)

    def test_structured_outputs_backend_configuration(self):
        pinned = self.launch({'LLM_STRUCTURED_OUTPUTS_BACKEND': 'xgrammar'})
        self.assertEqual(
            json.loads(pinned[pinned.index('--structured-outputs-config') + 1]),
            {'backend': 'xgrammar'},
        )
        for empty_val in ('', '   '):
            with self.subTest(empty_val=empty_val):
                self.assertNotIn(
                    '--structured-outputs-config',
                    self.launch({'LLM_STRUCTURED_OUTPUTS_BACKEND': empty_val}),
                )
        self.assertNotIn('--structured-outputs-config', self.launch({}))

    def test_env_validation_rejects_invalid_values(self):
        invalid_envs = [
            {'LLM_TENSOR_PARALLEL_SIZE': '0'},
            {'LLM_GPU_MEMORY_UTILIZATION': '0'},
            {'LLM_GPU_MEMORY_UTILIZATION': '1.01'},
            {'LLM_GPU_MEMORY_UTILIZATION': 'nan'},
            {'LLM_KV_CACHE_BYTES': '-10'},
            {'LLM_MAX_NUM_SEQS': '0'},
            {'LLM_MAX_NUM_BATCHED_TOKENS': '0'},
            {'LLM_CUDAGRAPH_CAPTURE_SIZES': ''},
            {'LLM_CUDAGRAPH_CAPTURE_SIZES': '1,0,4'},
            {'LLM_ASYNC_SCHEDULING': 'true'},
            {'LLM_ENABLE_PREFIX_CACHING': 'yes'},
            {'LLM_MAMBA_BLOCK_SIZE': '0'},
            {'LLM_ENABLE_MTP': '2'},
            {'LLM_ENABLE_MTP': '1', 'LLM_MTP_NUM_SPECULATIVE_TOKENS': '0'},
            {'LLM_ENABLE_LORA': '2'},
            {'LLM_ENABLE_LORA': '1', 'LLM_MAX_LORAS': '0'},
            {'LLM_ENABLE_LORA': '1', 'LLM_MAX_LORA_RANK': '0'},
        ]
        for env in invalid_envs:
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.launch(env)

    def test_validate_supports_sharded_and_single_file_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'text_config': {'max_position_embeddings': 262144}}))
            with self.assertRaises(FileNotFoundError):
                validate(root)
            (root / 'model.safetensors').write_bytes(b'ok')
            self.assertEqual(validate(root), 262144)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({'max_position_embeddings': 131072}))
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'shard-01.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                validate(root)
            (root / 'shard-01.safetensors').write_bytes(b'ok')
            self.assertEqual(validate(root), 131072)

        with patch.dict(os.environ, {'LLM_MODEL_ID': 'm', 'LLM_CHECKPOINT': 'Z' * 40}, clear=True):
            with self.assertRaises(ValueError):
                location()

    def test_router_hook_fails_closed_on_version_drift_or_reapplication(self):
        source = 'def attach_router(app: FastAPI):\n    app.include_router(router)\n'
        patched = patch_api_source(source)
        self.assertIn('app.include_router(admission_router)', patched)
        with self.assertRaises(RuntimeError):
            patch_api_source(patched)
        with self.assertRaises(RuntimeError):
            patch_api_source('def attach_router(app):\n    pass\n')

    def test_admission_patch_version_guard_accepts_032_and_034_only(self):
        class FakePackage:
            def __init__(self, version, root):
                self.version = version
                self.root = root

            def locate_file(self, relative):
                return self.root / relative

        for version in ('0.32.0', '0.34.0+cu128'):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                api_router = root / 'vllm/entrypoints/openai/responses/api_router.py'
                api_router.parent.mkdir(parents=True)
                api_router.write_text('def attach_router(app: FastAPI):\n    app.include_router(router)\n')
                with patch('services.llm.patches.patch_llm_admission.distribution',
                           return_value=FakePackage(version, root)):
                    patch_admission_main()
                self.assertIn('admission_router', api_router.read_text())
                self.assertTrue((root / 'vllm/entrypoints/openai/llm_admission.py').is_file())
                self.assertTrue((root / 'vllm/entrypoints/openai/llm_gpu_control.py').is_file())

        for version in ('0.33.0', '0.35.0'):
            with self.subTest(version=version):
                with patch('services.llm.patches.patch_llm_admission.distribution',
                           return_value=FakePackage(version, Path('.'))):
                    with self.assertRaises(RuntimeError):
                        patch_admission_main()
        with patch('services.llm.patches.patch_llm_admission.distribution', side_effect=PackageNotFoundError('vllm')):
            with self.assertRaises(PackageNotFoundError):
                patch_admission_main()
