"""Pinned, independent text and image inference; never executes tools or stores memory."""
import json
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
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    missing = [name for name in set(index['weight_map'].values()) if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError('Incomplete LLM snapshot: ' + ', '.join(sorted(missing)))
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
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', str(path), '--served-model-name', model,
               '--host', '0.0.0.0', '--port', '8000', '--dtype', 'bfloat16',
               '--max-model-len', str(window), '--max-num-seqs', os.environ.get('LLM_MAX_NUM_SEQS', '4'),
               '--max-num-batched-tokens', os.environ.get('LLM_MAX_NUM_BATCHED_TOKENS', '2048'),
               '--compilation-config', json.dumps({'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                                    'cudagraph_capture_sizes': [1, 2, 4]}),
               '--no-async-scheduling',
               '--no-enable-prefix-caching',
               # KV allocation is fixed separately from the startup free-memory check.
               '--gpu-memory-utilization', '0.75',
               '--kv-cache-memory-bytes', os.environ.get('LLM_KV_CACHE_BYTES', '9663676416'),
               '--mamba-ssm-cache-dtype', 'float32', '--reasoning-parser', 'qwen3',
               '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder']
    if max_images:
        command.extend(['--limit-mm-per-prompt', json.dumps({'image': max_images, 'video': 0})])
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
