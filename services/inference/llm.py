"""Pinned, independent text inference; never executes tools or stores memory."""
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
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', str(path), '--served-model-name', model,
               '--host', '0.0.0.0', '--port', '8000', '--dtype', 'bfloat16',
               '--max-model-len', str(window), '--max-num-seqs', os.environ.get('LLM_MAX_NUM_SEQS', '4'),
               '--max-num-batched-tokens', '2048', '--enforce-eager',
               '--language-model-only', '--no-enable-prefix-caching',
               '--gpu-memory-utilization', '0.90',
               '--kv-cache-memory-bytes', os.environ.get('LLM_KV_CACHE_BYTES', '9663676416'),
               '--mamba-ssm-cache-dtype', 'float32', '--reasoning-parser', 'qwen3',
               '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder']
    os.execvp(command[0], command)


if __name__ == '__main__':
    actions = {'fetch': fetch, 'serve': serve}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit('usage: llm.py fetch|serve')
    actions[sys.argv[1]](*location())
