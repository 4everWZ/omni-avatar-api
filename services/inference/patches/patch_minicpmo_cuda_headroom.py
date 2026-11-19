"""Return unused CUDA allocator memory after Thinker warmup and Code2Wav work."""

from importlib.metadata import distribution


def _insert_before(source, anchor, addition):
    if source.count(anchor) != 1 or addition in source:
        raise RuntimeError("Recheck the pinned MiniCPM-o CUDA cache hook")
    return source.replace(anchor, addition + anchor, 1)


def patch_model_source(source):
    return _insert_before(
        source,
        "    def on_requests_added(self, prewarms: list[OmniRequestPrewarm]) -> None:\n",
        '''    def release_idle_cuda_cache(self) -> None:
        if os.environ.get("OMNI_RELEASE_UNUSED_CUDA_CACHE") != "1":
            return
        if self._states or self._prefetch_queue or not torch.cuda.is_initialized():
            return

        device = torch.cuda.current_device()
        mib = 1024 * 1024
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        if reserved - allocated < 512 * mib:
            return

        # Return only inactive allocator blocks. Weights, live tensors, and
        # warm reference data stay in place.
        torch.cuda.empty_cache()
        released = reserved - torch.cuda.memory_reserved(device)
        logger.info("MiniCPM-o Code2Wav idle CUDA cache: allocated=%d MiB reserved=%d MiB released=%d MiB",
                    allocated // mib, reserved // mib, released // mib)

''',
    )


def patch_runner_source(source):
    return _insert_before(
        source,
        "            if has_ec_transfer() and not get_ec_transfer().is_consumer:\n",
        """            if num_scheduled_tokens <= 0 and scheduler_output.finished_req_ids and not self.requests:
                self._call_optional_model_hook("release_idle_cuda_cache")

""",
    )


def patch_ar_worker_source(source):
    return _insert_before(
        source,
        "    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK:\n",
        '''    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        if (
            getattr(self.model_config, "stage_id", None) == 0
            and self.device.type == "cuda"
            and os.environ.get("OMNI_RELEASE_UNUSED_CUDA_CACHE") == "1"
        ):
            # The explicit KV budget profiles with dummy inputs. Return the
            # unused profiling blocks before the other stages start.
            before = torch.cuda.memory_reserved(self.device)
            torch.cuda.empty_cache()
            released = before - torch.cuda.memory_reserved(self.device)
            logger.info("MiniCPM-o Thinker released %d MiB of warmup CUDA cache", released // (1024 * 1024))
        return result

''',
    )


def main():
    omni = distribution("vllm-omni")
    vllm = distribution("vllm")
    if omni.version != "0.32.0" or vllm.version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Recheck MiniCPM-o CUDA cache reclamation for these versions")

    patched_files = []
    for name, patch in (
        ("model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py", patch_model_source),
        ("worker/gpu_generation_model_runner.py", patch_runner_source),
        ("worker/gpu_ar_worker.py", patch_ar_worker_source),
    ):
        path = omni.locate_file(f"vllm_omni/{name}")
        patched = patch(path.read_text(encoding="utf-8"))
        compile(patched, str(path), "exec")
        patched_files.append((path, patched))
    for path, patched in patched_files:
        path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
