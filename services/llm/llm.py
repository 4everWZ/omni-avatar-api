"""Pinned, independent text and image inference; never executes tools or stores memory."""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

_VLM_LORA_OLD_PREFIX = 'base_model.model.model.layers.'
_VLM_LORA_NEW_PREFIX = 'base_model.model.model.language_model.layers.'


def _validate_revision(revision, env_name='LLM_CHECKPOINT'):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError(f'{env_name} must be an exact Hugging Face commit')
    return revision


def location():
    model, revision = os.environ['LLM_MODEL_ID'], os.environ['LLM_CHECKPOINT']
    _validate_revision(revision, 'LLM_CHECKPOINT')
    return model, revision, Path('/models') / revision


def lora_location():
    name = os.environ.get('LLM_LORA_NAME', '').strip()
    repo_id = os.environ.get('LLM_LORA_REPO_ID', '').strip()
    revision = os.environ.get('LLM_LORA_CHECKPOINT', '').strip()
    if not any((name, repo_id, revision)):
        return None
    if not all((name, repo_id, revision)):
        raise ValueError('LLM_LORA_NAME, LLM_LORA_REPO_ID, and LLM_LORA_CHECKPOINT must all be set together')
    _validate_revision(revision, 'LLM_LORA_CHECKPOINT')
    return name, repo_id, revision, Path('/models/lora') / revision


def draft_location():
    spec_method = os.environ.get('LLM_SPECULATIVE_METHOD', '').strip()
    repo_id = os.environ.get('LLM_DRAFT_MODEL_ID', '').strip()
    revision = os.environ.get('LLM_DRAFT_CHECKPOINT', '').strip()
    if not repo_id and not revision:
        return None
    if not repo_id or not revision:
        raise ValueError('LLM_DRAFT_MODEL_ID and LLM_DRAFT_CHECKPOINT must both be set')
    _validate_revision(revision, 'LLM_DRAFT_CHECKPOINT')
    # vLLM 0.34.0 DSparkSpeculatorConfig.matches() checks `"dspark" in model.lower()`,
    # so DSpark snapshots must live under `/models/dspark/<sha>`.
    if spec_method in ('dspark', 'dflash'):
        subdir = spec_method
    else:
        subdir = 'dspark' if 'dspark' in repo_id.lower() else 'dflash'
    return subdir, repo_id, revision, Path('/models') / subdir / revision


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f'Invalid safetensors file: {path}')
    with path.open('rb') as handle:
        (header_len,) = struct.unpack('<Q', handle.read(8))
        if header_len == 0 or 8 + header_len > file_size:
            raise ValueError(f'Invalid safetensors header length in {path}')
        header = json.loads(handle.read(header_len).decode('utf-8'))
    if not isinstance(header, dict):
        raise ValueError(f'Invalid safetensors header in {path}')
    return header, header_len


def _read_safetensors_keys(path: Path) -> list[str]:
    header, _ = _read_safetensors_header(path)
    return [key for key in header if key != '__metadata__']


def _normalize_vlm_lora_safetensors(lora_dir: Path) -> int:
    adapter_path = lora_dir / 'adapter_model.safetensors'
    if not adapter_path.is_file():
        raise FileNotFoundError(f'Missing LoRA weights: {adapter_path}')
    header, header_len = _read_safetensors_header(adapter_path)
    tensor_keys = [key for key in header if key != '__metadata__']
    old_keys = [key for key in tensor_keys if key.startswith(_VLM_LORA_OLD_PREFIX)]
    if not old_keys:
        return 0
    if any(key.startswith(_VLM_LORA_NEW_PREFIX) for key in tensor_keys):
        raise ValueError(f'Mixed VLM and text LoRA tensor keys in {adapter_path}')

    new_header = {}
    for key, value in header.items():
        if key != '__metadata__' and key.startswith(_VLM_LORA_OLD_PREFIX):
            new_key = _VLM_LORA_NEW_PREFIX + key[len(_VLM_LORA_OLD_PREFIX):]
        else:
            new_key = key
        if new_key in new_header:
            raise ValueError(f'Duplicate tensor key after normalization: {new_key}')
        new_header[new_key] = value

    new_tensor_keys = [key for key in new_header if key != '__metadata__']
    assert len(new_tensor_keys) == len(tensor_keys)
    assert not any(key.startswith(_VLM_LORA_OLD_PREFIX) for key in new_tensor_keys)

    raw_json = json.dumps(new_header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    pad = (-len(raw_json)) % 8
    if pad:
        raw_json += b' ' * pad

    tmp_path = lora_dir / f'.adapter_model.safetensors.tmp.{os.getpid()}'
    try:
        with adapter_path.open('rb') as src, tmp_path.open('wb') as dst:
            src.seek(8 + header_len)
            dst.write(struct.pack('<Q', len(raw_json)))
            dst.write(raw_json)
            shutil.copyfileobj(src, dst)
            dst.flush()
        written_keys = _read_safetensors_keys(tmp_path)
        assert len(written_keys) == len(tensor_keys)
        assert not any(key.startswith(_VLM_LORA_OLD_PREFIX) for key in written_keys)
        tmp_path.replace(adapter_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return len(old_keys)


def _write_snapshot_ready(path: Path, repo_id: str, revision: str):
    marker = path / f'.snapshot-ready.{os.getpid()}'
    marker.write_text(f'{repo_id}@{revision}\n')
    marker.replace(path / '.snapshot-ready')


def _require_snapshot_ready(path: Path, repo_id: str, revision: str, message: str):
    marker = path / '.snapshot-ready'
    if not marker.is_file() or marker.read_text().strip() != f'{repo_id}@{revision}':
        raise RuntimeError(message)


def _validate_lora_dir(lora_dir: Path):
    if not (lora_dir / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Incomplete LoRA snapshot: missing {lora_dir / "adapter_config.json"}')
    if not (lora_dir / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError(f'Incomplete LoRA snapshot: missing {lora_dir / "adapter_model.safetensors"}')


def _validate_draft_dir(draft_dir: Path):
    if not (draft_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Incomplete draft snapshot: missing {draft_dir / "config.json"}')
    index_path = draft_dir / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        missing = [name for name in set(index['weight_map'].values()) if not (draft_dir / name).is_file()]
        if missing:
            raise FileNotFoundError('Incomplete draft snapshot: ' + ', '.join(sorted(missing)))
    elif not (draft_dir / 'model.safetensors').is_file() and not any(draft_dir.glob('*.safetensors')):
        raise FileNotFoundError('Incomplete draft snapshot: missing safetensors weights')


def validate(path):
    config = json.loads((path / 'config.json').read_text())
    index_path = path / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        missing = [name for name in set(index['weight_map'].values()) if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError('Incomplete LLM snapshot: ' + ', '.join(sorted(missing)))
    elif not (path / 'model.safetensors').is_file():
        raise FileNotFoundError('Incomplete LLM snapshot: missing model.safetensors.index.json or model.safetensors')
    text = config.get('text_config', config)
    window = text['max_position_embeddings']
    if type(window) is not int or window < 4096:
        raise ValueError('Invalid LLM context window')
    return window


def fetch_lora(name, repo_id, revision, lora_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=lora_dir)
    _validate_lora_dir(lora_dir)
    _normalize_vlm_lora_safetensors(lora_dir)
    _write_snapshot_ready(lora_dir, repo_id, revision)


def fetch_draft(method, repo_id, revision, draft_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=draft_dir)
    _validate_draft_dir(draft_dir)
    _write_snapshot_ready(draft_dir, repo_id, revision)


def fetch(model, revision, path):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model, revision=revision, local_dir=path)
    validate(path)
    _write_snapshot_ready(path, model, revision)
    if lora := lora_location():
        fetch_lora(*lora)
    if draft := draft_location():
        fetch_draft(*draft)


def _env_int(name, default, *, minimum=1):
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _env_flag(name, default='0'):
    value = os.environ.get(name, default)
    if value not in ('0', '1'):
        raise ValueError(f"{name} must be '0' or '1'")
    return value == '1'


def _env_gpu_utilization(default='0.75'):
    raw = os.environ.get('LLM_GPU_MEMORY_UTILIZATION', default).strip()
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('LLM_GPU_MEMORY_UTILIZATION must be in (0, 1]')
    return raw


def _env_capture_sizes(default='1,2,4'):
    raw = os.environ.get('LLM_CUDAGRAPH_CAPTURE_SIZES', default)
    parts = [part.strip() for part in raw.split(',')]
    if not parts or any(not part for part in parts):
        raise ValueError('LLM_CUDAGRAPH_CAPTURE_SIZES must be comma-separated positive integers')
    sizes = [int(part) for part in parts]
    if any(size < 1 for size in sizes):
        raise ValueError('LLM_CUDAGRAPH_CAPTURE_SIZES values must be >= 1')
    return sizes


def _speculative_config():
    spec_method = os.environ.get('LLM_SPECULATIVE_METHOD', '').strip()
    if not spec_method:
        spec_method = 'mtp' if _env_flag('LLM_ENABLE_MTP', '0') else 'none'
    elif spec_method not in ('none', 'mtp', 'dspark', 'dflash'):
        raise ValueError("LLM_SPECULATIVE_METHOD must be '', 'none', 'mtp', 'dspark', or 'dflash'")

    if spec_method == 'none':
        return None
    if spec_method == 'mtp':
        return {
            'method': 'mtp',
            'num_speculative_tokens': _env_int('LLM_MTP_NUM_SPECULATIVE_TOKENS', 3),
        }

    draft = draft_location()
    if draft is None:
        raise ValueError(f'LLM_DRAFT_MODEL_ID and LLM_DRAFT_CHECKPOINT are required for {spec_method}')
    _, draft_repo, draft_rev, draft_path = draft
    _require_snapshot_ready(
        draft_path,
        draft_repo,
        draft_rev,
        'Run the pinned llm-model-fetch service first for draft model',
    )
    _validate_draft_dir(draft_path)
    config = {'method': spec_method, 'model': str(draft_path)}
    if raw_tokens := os.environ.get('LLM_NUM_SPECULATIVE_TOKENS', '').strip():
        tokens = int(raw_tokens)
        if tokens < 1:
            raise ValueError('LLM_NUM_SPECULATIVE_TOKENS must be >= 1')
        config['num_speculative_tokens'] = tokens
    if raw_draft_tp := os.environ.get('LLM_DRAFT_TENSOR_PARALLEL_SIZE', '').strip():
        draft_tp = int(raw_draft_tp)
        if draft_tp < 1:
            raise ValueError('LLM_DRAFT_TENSOR_PARALLEL_SIZE must be >= 1')
        config['draft_tensor_parallel_size'] = draft_tp
    return config


def serve(model, revision, path):
    _require_snapshot_ready(path, model, revision, 'Run the pinned llm-model-fetch service first')
    native = validate(path)
    window = int(os.environ.get('LLM_MAX_MODEL_LEN', native))
    if not 4096 <= window <= native:
        raise ValueError('LLM_MAX_MODEL_LEN is outside the checkpoint window')
    max_images = int(os.environ.get('LLM_MAX_IMAGES_PER_REQUEST', '8'))
    if not 0 <= max_images <= 8:
        raise ValueError('LLM_MAX_IMAGES_PER_REQUEST must be between 0 and 8')

    tp_size = _env_int('LLM_TENSOR_PARALLEL_SIZE', 1)
    max_num_seqs = _env_int('LLM_MAX_NUM_SEQS', 4)
    max_batched_tokens = _env_int('LLM_MAX_NUM_BATCHED_TOKENS', 2048)
    gpu_utilization = _env_gpu_utilization('0.75')
    compilation_mode = _env_int('LLM_COMPILATION_MODE', 0, minimum=0)
    if compilation_mode not in (0, 1, 2, 3):
        raise ValueError('LLM_COMPILATION_MODE must be 0, 1, 2, or 3')
    cudagraph_mode = os.environ.get('LLM_CUDAGRAPH_MODE', 'FULL_DECODE_ONLY').strip()
    if not cudagraph_mode:
        raise ValueError('LLM_CUDAGRAPH_MODE must not be empty')
    capture_sizes = _env_capture_sizes('1,2,4')
    async_scheduling = _env_flag('LLM_ASYNC_SCHEDULING', '0')
    prefix_caching = _env_flag('LLM_ENABLE_PREFIX_CACHING', '0')
    spec_config = _speculative_config()
    enable_lora = _env_flag('LLM_ENABLE_LORA', '0')

    command = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(path), '--served-model-name', model,
        '--host', '0.0.0.0', '--port', '8000', '--dtype', 'bfloat16',
        '--max-model-len', str(window),
        '--max-num-seqs', str(max_num_seqs),
        '--max-num-batched-tokens', str(max_batched_tokens),
        '--compilation-config', json.dumps({
            'mode': compilation_mode,
            'cudagraph_mode': cudagraph_mode,
            'cudagraph_capture_sizes': capture_sizes,
        }),
    ]
    if tp_size > 1:
        command.extend(['--tensor-parallel-size', str(tp_size)])
    if not async_scheduling:
        command.append('--no-async-scheduling')
    command.append('--enable-prefix-caching' if prefix_caching else '--no-enable-prefix-caching')
    # KV allocation can be fixed separately from the startup free-memory check or auto-sized.
    command.extend(['--gpu-memory-utilization', gpu_utilization])

    kv_cache_bytes = os.environ.get('LLM_KV_CACHE_BYTES', '9663676416').strip()
    if kv_cache_bytes.lower() not in ('', '0', 'auto'):
        if int(kv_cache_bytes) <= 0:
            raise ValueError('LLM_KV_CACHE_BYTES must be positive, 0, auto, or empty')
        command.extend(['--kv-cache-memory-bytes', kv_cache_bytes])
    if kv_cache_dtype := os.environ.get('LLM_KV_CACHE_DTYPE', '').strip():
        command.extend(['--kv-cache-dtype', kv_cache_dtype])
    if mamba_cache_mode := os.environ.get('LLM_MAMBA_CACHE_MODE', '').strip():
        command.extend(['--mamba-cache-mode', mamba_cache_mode])
    if raw_mamba_block := os.environ.get('LLM_MAMBA_BLOCK_SIZE', '').strip():
        mamba_block_size = int(raw_mamba_block)
        if mamba_block_size < 1:
            raise ValueError('LLM_MAMBA_BLOCK_SIZE must be >= 1')
        command.extend(['--mamba-block-size', str(mamba_block_size)])
    if mamba_ssm_dtype := os.environ.get('LLM_MAMBA_SSM_CACHE_DTYPE', 'float32').strip():
        command.extend(['--mamba-ssm-cache-dtype', mamba_ssm_dtype])
    if reasoning_parser := os.environ.get('LLM_REASONING_PARSER', 'qwen3').strip():
        command.extend(['--reasoning-parser', reasoning_parser])
    if tool_call_parser := os.environ.get('LLM_TOOL_CALL_PARSER', 'qwen3_coder').strip():
        command.extend(['--enable-auto-tool-choice', '--tool-call-parser', tool_call_parser])
    if structured_backend := os.environ.get('LLM_STRUCTURED_OUTPUTS_BACKEND', '').strip():
        command.extend(['--structured-outputs-config', json.dumps({'backend': structured_backend})])
    if spec_config is not None:
        command.extend(['--speculative-config', json.dumps(spec_config)])
    if enable_lora:
        max_loras = _env_int('LLM_MAX_LORAS', 2)
        max_lora_rank = _env_int('LLM_MAX_LORA_RANK', 32)
        command.extend(['--enable-lora', '--max-loras', str(max_loras), '--max-lora-rank', str(max_lora_rank)])
        if lora := lora_location():
            lora_name, lora_repo, lora_rev, lora_path = lora
            _require_snapshot_ready(
                lora_path,
                lora_repo,
                lora_rev,
                'Run the pinned llm-model-fetch service first for LoRA adapter',
            )
            _validate_lora_dir(lora_path)
            if any(key.startswith(_VLM_LORA_OLD_PREFIX) for key in _read_safetensors_keys(lora_path / 'adapter_model.safetensors')):
                raise RuntimeError('LoRA adapter keys are not normalized for VLM; re-run llm-model-fetch')
            command.extend(['--lora-modules', f'{lora_name}={lora_path}'])
    if max_images:
        command.extend(['--limit-mm-per-prompt', json.dumps({'image': max_images, 'video': 0})])
        if mm_encoder_tp_mode := os.environ.get('LLM_MM_ENCODER_TP_MODE', '').strip():
            command.extend(['--mm-encoder-tp-mode', mm_encoder_tp_mode])
    else:
        command.append('--language-model-only')
    if os.environ.get('LLM_GPU_CONTROL_TOKEN'):
        command.append('--enable-sleep-mode')
    os.execvp(command[0], command)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('fetch', 'fetch-lora', 'fetch-draft', 'serve'):
        raise SystemExit('usage: llm.py fetch|fetch-lora|fetch-draft|serve')
    action = sys.argv[1]
    if action == 'fetch':
        fetch(*location())
    elif action == 'serve':
        serve(*location())
    elif action == 'fetch-lora':
        lora = lora_location()
        if lora is None:
            raise ValueError('LLM_LORA_NAME, LLM_LORA_REPO_ID, and LLM_LORA_CHECKPOINT must be set')
        fetch_lora(*lora)
    elif action == 'fetch-draft':
        draft = draft_location()
        if draft is None:
            raise ValueError('LLM_DRAFT_MODEL_ID and LLM_DRAFT_CHECKPOINT must be set')
        fetch_draft(*draft)
