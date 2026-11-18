import os
from pathlib import Path
import tempfile
import unittest

from tools.compose import COMPOSE, ROOT, TARGETS, command


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
                                 ('base', 'workstation', 'avatar', 'llm', 'inference')])
        self.assertEqual(args[-3:], ['up', '-d', '--wait'])

    def test_all_deployment_targets_use_existing_compose_files(self):
        for target in TARGETS:
            with self.subTest(target=target):
                args = command(target, ['config', '--quiet'])
                for i, item in enumerate(args):
                    if item == '-f':
                        self.assertTrue(Path(args[i+1]).is_file())

    def test_optional_layers_do_not_replace_base_stack_or_consume_docker_arguments(self):
        args = command('workstation', ['--profile', 'assets', 'run', '--rm', 'portrait-assets'],
                       portrait=True, vpn=True, media=True)
        files = [Path(args[i+1]).stem for i, item in enumerate(args) if item == '-f']
        self.assertEqual(files[-3:], ['portrait', 'vpn', 'media'])
        self.assertEqual(args[-5:], ['--profile', 'assets', 'run', '--rm', 'portrait-assets'])

    def test_portrait_cannot_allocate_workstation_gpu_on_legacy_or_orin_targets(self):
        for target in set(TARGETS) - {'workstation'}:
            with self.subTest(target=target), self.assertRaises(ValueError):
                command(target, ['up'], portrait=True)
