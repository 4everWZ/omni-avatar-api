import os
import sys
from pathlib import Path

from .model import MODEL, REVISION, Klein


def main():
    path = Path('/models') / REVISION
    if sys.argv[1:] == ['fetch']:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=path,
                          allow_patterns=['model_index.json', 'scheduler/*', 'text_encoder/*',
                                          'tokenizer/*', 'transformer/*', 'vae/*'])
        return
    from aiohttp import web
    from .app import create_app
    dedicated = os.environ.get('OMNI_IMAGE_PLACEMENT') == 'dedicated'
    switch_root = None if dedicated else os.environ['OMNI_GPU_SWITCH_ROOT']
    web.run_app(create_app(Klein(path), os.environ['OMNI_IMAGE_INTERNAL_TOKEN'],
                           switch_root),
                host='0.0.0.0', port=8000, handler_cancellation=False, access_log=None)


if __name__ == '__main__':
    main()
