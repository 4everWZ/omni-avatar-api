"""Pinned, independent text and image inference; never executes tools or stores memory."""
import json
import math
import os
from pathlib import Path
import sys


def location():
    model, revision = os.environ['LLM_MODEL_ID'], os.environ['LLM_CHECKPOINT']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('LLM_CHECKPOINT must be an exact Hugging Face commit')
    return model, revision, Path('/models') / revision


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


def fetch(model, revision, path):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model, revision=revision, local_dir=path)
    validate(path)
    marker = path / f'.snapshot-ready.{os.getpid()}'
    marker.write_text(f'{model}@{revision}\n')
    marker.replace(path / '.snapshot-ready')


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


def serve(model, revision, path):
    marker = path / '.snapshot-ready'
    if not marker.is_file() or marker.read_text().strip() != f'{model}@{revision}':
        raise RuntimeError('Run the pinned llm-model-fetch service first')
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
    capture_sizes = _env_capture_sizes('1,2,4')
    async_scheduling = _env_flag('LLM_ASYNC_SCHEDULING', '0')
    prefix_caching = _env_flag('LLM_ENABLE_PREFIX_CACHING', '0')
    enable_mtp = _env_flag('LLM_ENABLE_MTP', '0')
    enable_lora = _env_flag('LLM_ENABLE_LORA', '0')

    command = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(path), '--served-model-name', model,
        '--host', '0.0.0.0', '--port', '8000', '--dtype', 'bfloat16',
        '--max-model-len', str(window),
        '--max-num-seqs', str(max_num_seqs),
        '--max-num-batched-tokens', str(max_batched_tokens),
        '--compilation-config', json.dumps({
            'mode': 0,
            'cudagraph_mode': 'FULL_DECODE_ONLY',
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
    if enable_mtp:
        mtp_tokens = _env_int('LLM_MTP_NUM_SPECULATIVE_TOKENS', 3)
        command.extend(['--speculative-config', json.dumps({
            'method': 'mtp',
            'num_speculative_tokens': mtp_tokens,
        })])
    if enable_lora:
        max_loras = _env_int('LLM_MAX_LORAS', 2)
        max_lora_rank = _env_int('LLM_MAX_LORA_RANK', 32)
        command.extend(['--enable-lora', '--max-loras', str(max_loras), '--max-lora-rank', str(max_lora_rank)])
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
    actions = {'fetch': fetch, 'serve': serve}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit('usage: llm.py fetch|serve')
    actions[sys.argv[1]](*location())
