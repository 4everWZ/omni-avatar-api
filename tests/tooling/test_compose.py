import os
from pathlib import Path
import tempfile
import unittest

from tools.compose import COMPOSE, ROOT, TARGETS, command, main


class ComposeEntrypointTests(unittest.TestCase):
    def test_default_stack_and_persistent_paths_are_anchored_to_repository(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = command('workstation', ['up', '-d', '--wait'])
            finally:
                os.chdir(previous)
        self.assertEqual(args[:6], ['docker', 'compose', '--project-directory', str(ROOT),
                                   '--env-file', str(ROOT / '.env')])
        files = [args[i+1] for i, item in enumerate(args) if item == '-f']
        self.assertEqual(files, [str(COMPOSE / f'{name}.yml') for name in
                                 ('base', 'workstation', 'llm', 'inference', 'asr', 'llm-tp2', 'tts', 'omni-duplex')])
        self.assertEqual(args[-3:], ['up', '-d', '--wait'])

    def test_workstation_legacy_preserves_original_four_file_stack(self):
        args = command('workstation-legacy', ['config', '--quiet'])
        files = [Path(args[i+1]).stem for i, item in enumerate(args) if item == '-f']
        self.assertEqual(files, ['base', 'workstation', 'llm', 'inference'])

    def test_all_deployment_targets_use_existing_compose_files(self):
        for target in TARGETS:
            with self.subTest(target=target):
                args = command(target, ['config', '--quiet'])
                for i, item in enumerate(args):
                    if item == '-f':
                        self.assertTrue(Path(args[i+1]).is_file())

    def test_dedicated_image_routing_does_not_deploy_text_or_gpu_services(self):
        args = command('gateway', ['up'], remote_images=True)
        files = [Path(args[i+1]).stem for i, item in enumerate(args) if item == '-f']
        self.assertEqual(files, ['gateway-node', 'remote-images'])

    def test_optional_layers_do_not_replace_base_stack_or_consume_docker_arguments(self):
        args = command('workstation', ['--profile', 'assets', 'run', '--rm', 'image-model-fetch'],
                       images=True, vpn=True, media=True)
        files = [Path(args[i+1]).stem for i, item in enumerate(args) if item == '-f']
        self.assertEqual(files, ['base', 'workstation', 'llm', 'inference', 'asr', 'llm-tp2', 'tts', 'omni-duplex',
                                 'images', 'vpn', 'media'])
        self.assertEqual(args[-5:], ['--profile', 'assets', 'run', '--rm', 'image-model-fetch'])

    def test_shared_images_allowed_on_workstation_targets_and_rejected_elsewhere(self):
        for target in ('workstation', 'workstation-legacy'):
            with self.subTest(target=target):
                args = command(target, ['up'], images=True)
                files = [Path(args[i+1]).stem for i, item in enumerate(args) if item == '-f']
                self.assertEqual(files[-1], 'images')
        for target in set(TARGETS) - {'workstation', 'workstation-legacy'}:
            with self.subTest(target=target), self.assertRaises(ValueError):
                command(target, ['up'], images=True)

    def test_asr_overlay_deduplicates_workstation_appends_legacy_and_rejects_non_gateway(self):
        workstation_args = command('workstation', ['up'], asr=True)
        workstation_files = [Path(workstation_args[i+1]).stem for i, item in enumerate(workstation_args) if item == '-f']
        self.assertEqual(workstation_files.count('asr'), 1)
        legacy_args = command('workstation-legacy', ['up'], asr=True, images=True)
        legacy_files = [Path(legacy_args[i+1]).stem for i, item in enumerate(legacy_args) if item == '-f']
        self.assertEqual(legacy_files, ['base', 'workstation', 'llm', 'inference', 'asr', 'images'])
        for target in ('images', 'llm', 'omni'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                command(target, ['up'], asr=True)
        self.assertEqual(main(['--target', 'workstation-legacy', '--asr', '--dry-run', '--', 'config', '--quiet']), 0)

    def test_llm_tp2_overlay_configures_swift_mtp_compile_cache_and_opt_in_hooks(self):
        tp2_text = (COMPOSE / 'llm-tp2.yml').read_text(encoding='utf-8')
        expected_snippets = [
            'LLM_MODEL_ID: ${LLM_TP2_MODEL_ID:-ukisai/Swift-1.5-Qwen3.8-27b-INT4}',
            'LLM_CHECKPOINT: ${LLM_TP2_CHECKPOINT:-81a28ee75eb1cd469f209813a368a79a7e3db3bc}',
            'LLM_LORA_NAME: ${LLM_TP2_LORA_NAME:-}',
            'LLM_LORA_REPO_ID: ${LLM_TP2_LORA_REPO_ID:-}',
            'LLM_LORA_CHECKPOINT: ${LLM_TP2_LORA_CHECKPOINT:-}',
            'LLM_SPECULATIVE_METHOD: ${LLM_TP2_SPECULATIVE_METHOD:-mtp}',
            'LLM_MTP_NUM_SPECULATIVE_TOKENS: ${LLM_TP2_MTP_NUM_SPECULATIVE_TOKENS:-3}',
            'LLM_DRAFT_MODEL_ID: ${LLM_TP2_DRAFT_MODEL_ID:-}',
            'LLM_DRAFT_CHECKPOINT: ${LLM_TP2_DRAFT_CHECKPOINT:-}',
            'LLM_MAX_NUM_BATCHED_TOKENS: ${LLM_TP2_MAX_NUM_BATCHED_TOKENS:-8192}',
            'LLM_COMPILATION_MODE: ${LLM_TP2_COMPILATION_MODE:-0}',
            'LLM_MAMBA_CACHE_MODE: ${LLM_TP2_MAMBA_CACHE_MODE:-}',
            'LLM_ENABLE_LORA: ${LLM_TP2_ENABLE_LORA:-0}',
            'LLM_DRAFT_TENSOR_PARALLEL_SIZE: ${LLM_TP2_DRAFT_TENSOR_PARALLEL_SIZE:-2}',
            'LLM_NUM_SPECULATIVE_TOKENS: ${LLM_TP2_NUM_SPECULATIVE_TOKENS:-}',
            '- ${OMNI_MODEL_DATA:-./data/models}:/models:ro',
            '- vllm-compile-cache:/root/.cache/vllm',
            '- triton-cache:/root/.triton',
            'start_period: 8m',
            'LLM_EXTRA_MODEL_ALIASES: ${LLM_TP2_EXTRA_MODEL_ALIASES:-}',
            'vllm-compile-cache: {}',
            'triton-cache: {}',
        ]
        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, tp2_text)
        # Both llm-model-fetch and llm-engine receive the x-llm-tp2 anchor (base, LoRA, and draft vars).
        self.assertIn('  llm-model-fetch:\n    image: omni-api-llm:0.34.0\n    environment: *llm-tp2', tp2_text)
        self.assertIn('    environment:\n      <<: *llm-tp2', tp2_text)

    def test_omni_duplex_and_env_example_sync_tp2_defaults(self):
        duplex_text = (COMPOSE / 'omni-duplex.yml').read_text(encoding='utf-8')
        self.assertIn('OMNI_LLM_MODEL: ${OMNI_DUPLEX_LLM_MODEL:-llm-default}', duplex_text)
        self.assertIn('OMNI_MAX_MODEL_LEN: ${LLM_TP2_MAX_MODEL_LEN:-262144}', duplex_text)

        env_lines = {}
        for line in (ROOT / '.env.example').read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key, value = stripped.split('=', 1)
                env_lines[key] = value

        expected_env = {
            'LLM_TP2_MODEL_ID': 'ukisai/Swift-1.5-Qwen3.8-27b-INT4',
            'LLM_TP2_CHECKPOINT': '81a28ee75eb1cd469f209813a368a79a7e3db3bc',
            'LLM_TP2_LORA_NAME': '',
            'LLM_TP2_LORA_REPO_ID': '',
            'LLM_TP2_LORA_CHECKPOINT': '',
            'LLM_TP2_EXTRA_MODEL_ALIASES': '',
            'LLM_TP2_MAX_MODEL_LEN': '262144',
            'LLM_TP2_MAX_NUM_BATCHED_TOKENS': '8192',
            'LLM_TP2_COMPILATION_MODE': '0',
            'LLM_TP2_MAMBA_CACHE_MODE': '',
            'LLM_TP2_SPECULATIVE_METHOD': 'mtp',
            'LLM_TP2_MTP_NUM_SPECULATIVE_TOKENS': '3',
            'LLM_TP2_DRAFT_MODEL_ID': '',
            'LLM_TP2_DRAFT_CHECKPOINT': '',
            'LLM_TP2_DRAFT_TENSOR_PARALLEL_SIZE': '2',
            'LLM_TP2_NUM_SPECULATIVE_TOKENS': '',
            'LLM_TP2_ENABLE_LORA': '0',
            'OMNI_DUPLEX_LLM_MODEL': 'llm-default',
        }
        for key, value in expected_env.items():
            with self.subTest(key=key):
                self.assertEqual(env_lines.get(key), value)
