import json
import unittest
from unittest.mock import patch

from services.llm.patches.gpu_deadline import yield_to_renderer
from services.llm.patches.patch_llm_admission import patch_core_source


class GPUDeadlineTests(unittest.TestCase):
    def test_no_wait_without_a_live_clock_or_with_sufficient_headroom(self):
        with patch.dict('os.environ', {}, clear=True), patch('time.sleep') as sleep:
            self.assertEqual(yield_to_renderer(), 0)
            sleep.assert_not_called()
        for clock in ({'updated': 9, 'deadline': 9}, {'updated': 11, 'deadline': 9},
                      {'updated': 10, 'deadline': 10.25}, {'updated': 10, 'deadline': float('nan')}):
            with self.subTest(clock=clock), patch.dict('os.environ', {'OMNI_GPU_RENDER_CLOCK': 'fixture'}), \
                    patch('pathlib.Path.read_text', return_value=json.dumps(clock)), \
                    patch('time.monotonic', return_value=10), patch('time.sleep') as sleep:
                yield_to_renderer()
                sleep.assert_not_called()

    def test_wait_is_bounded_and_releases_early_when_frames_catch_up(self):
        for caught_up in (False, True):
            current = [10.]
            def advance(seconds):
                current[0] += seconds
            def clock():
                return json.dumps({'updated': current[0],
                    'deadline': current[0] + (.4 if caught_up and current[0] > 10.003 else 0)})
            with patch.dict('os.environ', {'OMNI_GPU_RENDER_CLOCK': 'fixture'}), \
                    patch('pathlib.Path.read_text', side_effect=clock), \
                    patch('time.monotonic', side_effect=lambda: current[0]), \
                    patch('time.sleep', side_effect=advance):
                waited = yield_to_renderer()
                self.assertGreater(waited, 0)
                self.assertLessEqual(waited, .0180001)
                if caught_up:
                    self.assertLess(waited, .01)

    def test_gpu_dispatch_hook_is_single_and_fails_closed(self):
        anchor = ('        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\n'
                  '        future = self.model_executor.execute_model(scheduler_output, non_block=True)\n')
        result = patch_core_source(anchor)
        self.assertEqual(result.count('        yield_to_renderer()'), 1)
        for invalid in (result, '', anchor * 2):
            with self.assertRaises(RuntimeError):
                patch_core_source(invalid)
