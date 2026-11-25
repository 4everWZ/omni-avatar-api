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
