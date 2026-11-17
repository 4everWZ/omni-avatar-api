"""Install packed INT4 MiniCPM-o duplex windowing into vLLM-Omni 0.32.0."""

from importlib.metadata import distribution
from pathlib import Path


def patch_window_source(source):
    changes = (
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\n"
            "from vllm.logger import init_logger\n"
            "from .minicpmo_int4_window import rotate_packed_int4_keys\n\n"
            "_int4_window_logger = init_logger(__name__)\n",
        ),
        (
            "        page_size_padded=layer_spec.page_size_padded,\n",
            "        page_size_padded=layer_spec.page_size_padded,\n"
            "        state_content_bytes=layer_spec.state_content_bytes,\n",
        ),
        (
            """    if last_dim not in (expected_head_dim, 2 * expected_head_dim):
        raise ValueError(
            f"expected head_dim={expected_head_dim} or {2 * expected_head_dim} for "
            f"{inv_freq.numel()} RoPE frequencies, got {last_dim} (shape {tuple(k_pool.shape)})"
        )
""",
            "    packed_int4 = k_pool.dtype == torch.uint8\n"
            "    supported_content = (2 * (expected_head_dim // 2 + 4),) if packed_int4 else (\n"
            "        expected_head_dim, 2 * expected_head_dim,\n"
            "    )\n"
            "    if last_dim not in supported_content:\n"
            "        raise ValueError(f\"Expected KV content dimension in {supported_content}, got {last_dim}\")\n",
        ),
        (
            "        if last_dim == 2 * expected_head_dim:\n",
            "        if packed_int4 or last_dim == 2 * expected_head_dim:\n",
        ),
        (
            '    if layout == "BHND":\n        keys = k_pool[b_idx, h_idx, o_idx, :expected_head_dim]\n',
            '    if packed_int4:\n'
            '        if layout != "BHND":\n'
            '            raise ValueError("Packed Triton INT4 cache requires BHND logical dimensions")\n'
            '        from vllm.v1.attention.ops.int4_per_token_head import single_rht\n\n'
            '        rotate_packed_int4_keys(\n'
            '            k_pool, (b_idx, h_idx, o_idx), expected_head_dim,\n'
            '            lambda keys: rotate_keys(keys, plan.delta, inv_freq), single_rht,\n'
            '        )\n'
            '    elif layout == "BHND":\n'
            '        keys = k_pool[b_idx, h_idx, o_idx, :expected_head_dim]\n',
        ),
        (
            "            # Evict worker-held unit history so state.window_units remains bounded\n",
            "                if runner.kv_caches[0].dtype == torch.uint8:\n"
            "                    _int4_window_logger.info(\n"
            '                        "MiniCPM-o INT4 native KV reanchor applied: shift_tokens=%d retained_tokens=%d layers=%d",\n'
            "                        plan.delta, positions.numel(), len(runner.kv_caches),\n"
            "                    )\n\n"
            "            # Evict worker-held unit history so state.window_units remains bounded\n",
        ),
    )
    for anchor, replacement in changes:
        if source.count(anchor) != 1:
            raise RuntimeError("Omni duplex KV source differs from the pinned version")
        source = source.replace(anchor, replacement, 1)
    return source


def main():
    omni = distribution("vllm-omni")
    vllm = distribution("vllm")
    if omni.version != "0.32.0" or vllm.version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Recheck MiniCPM-o duplex and the Triton INT4 KV layout for these versions")

    module = omni.locate_file("vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py")
    patched = patch_window_source(module.read_text(encoding="utf-8"))
    helper = Path(__file__).with_name("minicpmo_int4_window.py").read_text(encoding="utf-8")
    compile(patched, str(module), "exec")
    compile(helper, "minicpmo_int4_window.py", "exec")
    module.with_name("minicpmo_int4_window.py").write_text(helper, encoding="utf-8")
    module.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
