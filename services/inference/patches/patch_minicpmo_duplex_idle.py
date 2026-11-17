"""Keep MiniCPM-o's session-owned Code2Wav parked between native speech turns."""

from importlib.metadata import distribution


def _replace_once(source, old, new):
    if source.count(old) != 1 or new in source:
        raise RuntimeError("Omni no longer matches the pinned MiniCPM-o duplex idle path")
    return source.replace(old, new, 1)


def patch_transfer_source(source):
    anchor = "from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, unflatten_payload\n"
    source = _replace_once(
        source,
        anchor,
        anchor + "from vllm_omni.engine.duplex.contracts import (\n"
        "    DuplexFence,\n    duplex_resource_request_id,\n    duplex_session_id_from_request_id,\n)\n",
    )
    source = _replace_once(
        source,
        "        self._waiting_since: dict[str, float] = {}\n",
        """        self._waiting_since: dict[str, float] = {}
        self._minicpmo_code2wav = (
            self.connector.stage_id == 2 and getattr(model_config, "model_arch", None) == "MiniCPMO45Code2Wav"
        )
        self._minicpmo_speaking: set[str] = set()
""",
    )
    anchor = "            payload_segment_finished = self._is_truthy_scalar(meta.get(\"is_segment_finished\"))\n"
    source = _replace_once(
        source,
        anchor,
        anchor + """            if self._minicpmo_code2wav and getattr(request, "resumable", False):
                # A model-owned terminal snapshot ends speech, including the
                # duplicate control-only boundary. A transport-only segment
                # marker may simply flush an intermediate Talker condition.
                native_turn_finished = (
                    self._is_truthy_scalar(meta.get("turn_end"))
                    or self._is_truthy_scalar(meta.get("last_chunk"))
                    or self._is_truthy_scalar(meta.get("stream_finished"))
                    or (payload_segment_finished and meta.get("replace_runtime_additional_information") is True)
                )
                if payload_finished or native_turn_finished:
                    self._minicpmo_speaking.discard(req_id)
                elif self._is_truthy_scalar(meta.get("codec_chunk_frames")):
                    self._minicpmo_speaking.add(req_id)
""",
    )
    anchor = "        self._registered_load_entries.pop(request_id, None)\n        self._discard_from_chunk_deque("
    source = _replace_once(
        source,
        anchor,
        "        self._registered_load_entries.pop(request_id, None)\n"
        "        self._minicpmo_speaking.discard(request_id)\n        self._discard_from_chunk_deque(",
    )
    source = _replace_once(
        source,
        "    def collect_timed_out_request_ids(self, timeout_s: float) -> set[str]:\n",
        """    def _minicpmo_native_idle_wait(self, request: Request | None) -> bool:
        if not self._minicpmo_code2wav or getattr(request, "resumable", False) is not True:
            return False
        external_id = getattr(request, "external_req_id", None)
        if not isinstance(external_id, str):
            return False
        session_id = duplex_session_id_from_request_id(external_id)
        if session_id is None or not session_id.startswith("duplex-"):
            return False
        session_suffix = session_id.removeprefix("duplex-")
        if len(session_suffix) != 32 or any(char not in "0123456789abcdef" for char in session_suffix):
            return False
        try:
            epoch = int(external_id.split(".")[3])
        except ValueError:
            return False
        # The orchestrator copies the resident stage-0 identity and resume flag
        # to its prewarmed decoder. Turn-scoped and ordinary Chat IDs differ.
        if epoch < 0 or external_id != duplex_resource_request_id(DuplexFence(session_id, epoch), "stage0"):
            return False
        with self._receiver_state_lock:
            # The receiver marks a real codec chunk ready before the scheduler
            # can consume it and reset the wait clock.
            return (
                request.request_id in self._finished_load_reqs
                or request.request_id not in self._minicpmo_speaking
            )

    def collect_timed_out_request_ids(
        self,
        timeout_s: float,
        *,
        scheduler_requests: dict[str, Request] | None = None,
    ) -> set[str]:
""",
    )
    return _replace_once(
        source,
        "        timed_out_ids = {req_id for req_id, started in self._waiting_since.items() if now - started > timeout_s}\n",
        """        # Intentional silence before and between native turns is owned by the
        # duplex lease. Keep connector timeouts for ordinary, unowned, and
        # in-progress speech transfers. No clock or synthetic chunk is injected.
        timed_out_ids = {
            req_id
            for req_id, started in self._waiting_since.items()
            if now - started > timeout_s
            and not self._minicpmo_native_idle_wait((scheduler_requests or {}).get(req_id))
        }
""",
    )


def patch_scheduler_source(source):
    return _replace_once(
        source,
        "        timed_out_ids = adapter.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)\n",
        "        timed_out_ids = adapter.collect_timed_out_request_ids(\n"
        "            timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S, scheduler_requests=self.requests\n        )\n",
    )


def main():
    omni = distribution("vllm-omni")
    vllm = distribution("vllm")
    if omni.version != "0.32.0" or vllm.version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Recheck MiniCPM-o duplex idle handling for these versions")

    patched_files = []
    for name, patch in (
        ("distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py", patch_transfer_source),
        ("core/sched/omni_scheduler_mixin.py", patch_scheduler_source),
    ):
        path = omni.locate_file(f"vllm_omni/{name}")
        patched = patch(path.read_text(encoding="utf-8"))
        compile(patched, str(path), "exec")
        patched_files.append((path, patched))
    for path, patched in patched_files:
        path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
