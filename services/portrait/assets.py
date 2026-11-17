"""Fetch pinned provider code and PyTorch weights in the assets profile only."""
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download
from .backend import CODE_REVISION, MODEL_REVISION

def main():
    root = Path('/portrait')

    code = root / 'code' / CODE_REVISION
    code.parent.mkdir(parents=True, exist_ok=True)
    if not code.exists():
        subprocess.run(['git', 'clone', '--quiet', '--no-checkout',
                        'https://github.com/antgroup/ditto-talkinghead', str(code)], check=True)
    subprocess.run(['git', '-C', str(code), 'checkout', '--quiet', '--detach', CODE_REVISION], check=True)
    snapshot_download(repo_id='digital-avatar/ditto-talkinghead', revision=MODEL_REVISION,
                      local_dir=root / 'models' / MODEL_REVISION,
                      allow_patterns=['ditto_pytorch/*', 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'])
    (root / '.assets-ready').write_text(CODE_REVISION + '\n' + MODEL_REVISION + '\n')


if __name__ == "__main__":
    main()
