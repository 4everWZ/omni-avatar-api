import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from services.inference.patches.patch_minicpmo_cuda_headroom import (
    patch_ar_worker_source,
    patch_model_source,
    patch_runner_source,
)


class Code2WavCacheTest(unittest.TestCase):
    def setUp(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.memory_allocated.return_value = 4000 * 1024**2
        cuda.memory_reserved.return_value = 6000 * 1024**2
        namespace = {
            "torch": SimpleNamespace(cuda=cuda),
            "logger": Mock(),
            "OmniRequestPrewarm": object,
        }
        self.environ = {"OMNI_RELEASE_UNUSED_CUDA_CACHE": "1"}
        namespace["os"] = SimpleNamespace(environ=self.environ)
        exec(patch_model_source(
            "class Model:\n"
            "    def on_requests_added(self, prewarms: list[OmniRequestPrewarm]) -> None:\n"
            "        pass\n"
        ), namespace)
        self.cuda = cuda
        self.model = namespace["Model"]()
        self.model._states = {}
        self.model._prefetch_queue = {}

    def test_reclaims_only_significant_idle_allocator_reservations(self):
        self.model.release_idle_cuda_cache()
        self.cuda.empty_cache.assert_called_once_with()
        self.cuda.reset_mock()
        self.cuda.memory_reserved.return_value = 4511 * 1024**2
        self.model.release_idle_cuda_cache()
        self.cuda.empty_cache.assert_not_called()

    def test_skips_when_disabled_or_model_owns_live_work(self):
        for enabled, states, queue in ((False, {}, {}), (True, {"voice": object()}, {}),
                                       (True, {}, {"ref": object()})):
            with self.subTest(enabled=enabled, states=states, queue=queue):
                self.environ["OMNI_RELEASE_UNUSED_CUDA_CACHE"] = str(int(enabled))
                self.model._states = states
                self.model._prefetch_queue = queue
                self.model.release_idle_cuda_cache()
        self.cuda.empty_cache.assert_not_called()

    def test_runner_calls_hook_only_on_last_finish_without_scheduled_work(self):
        source = (
            "class Runner:\n"
            "    def step(self, num_scheduled_tokens, scheduler_output):\n"
            "        if True:\n"
            "            if has_ec_transfer() and not get_ec_transfer().is_consumer:\n"
            "                pass\n"
        )
        namespace = {"has_ec_transfer": lambda: False}
        exec(patch_runner_source(source), namespace)
        runner = namespace["Runner"]()
        runner._call_optional_model_hook = Mock()
        runner.requests = {"other": object()}
        runner.step(0, SimpleNamespace(finished_req_ids={"closed"}))
        runner.requests.clear()
        runner.step(10, SimpleNamespace(finished_req_ids={"closed"}))
        runner.step(0, SimpleNamespace(finished_req_ids=set()))
        runner._call_optional_model_hook.assert_not_called()
        runner.step(0, SimpleNamespace(finished_req_ids={"closed"}))
        runner._call_optional_model_hook.assert_called_once_with("release_idle_cuda_cache")

    def test_thinker_reclaims_only_after_warmup(self):
        source = (
            "class Base:\n"
            "    def compile_or_warm_up_model(self):\n"
            "        warm_up()\n"
            "        return times\n"
            "class Worker(Base):\n"
            "    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK:\n"
            "        pass\n"
        )
        calls = Mock()
        calls.cuda.memory_reserved.return_value = 6000 * 1024**2
        times = object()
        namespace = {"torch": calls, "os": SimpleNamespace(environ=self.environ), "times": times,
                     "warm_up": calls.warm_up, "logger": Mock(), "OmniSleepTask": str, "OmniACK": object}
        exec(patch_ar_worker_source(source), namespace)
        worker = namespace["Worker"]()
        worker.device = SimpleNamespace(type="cuda")
        worker.model_config = SimpleNamespace(stage_id=0)
        self.assertIs(worker.compile_or_warm_up_model(), times)
        self.assertEqual([call[0] for call in calls.mock_calls],
                         ["warm_up", "cuda.memory_reserved", "cuda.empty_cache", "cuda.memory_reserved"])

        calls.reset_mock()
        worker.model_config.stage_id = 1
        self.assertIs(worker.compile_or_warm_up_model(), times)
        calls.cuda.empty_cache.assert_not_called()
