"""Print installed GPU runtime versions and active requirement mismatches as JSON."""
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement


def report():
    packages = {}
    names = ('torch', 'torchvision', 'torchaudio', 'triton', 'nvidia-cuda-runtime',
             'nvidia-nccl-cu13', 'vllm', 'vllm-omni', 'transformers', 'diffusers',
             'tensorrt-cu13', 'onnxruntime', 'numpy', 'mediapipe', 'aiohttp', 'multidict', 'pillow')
    for name in names:
        try:
            installed = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        packages[name] = {'version': installed.version, 'location': str(installed.locate_file(''))}

    conflicts = []
    # system-site-packages may expose two distributions with the same name.
    # Match the distribution selected by the worker's active interpreter.
    for name in sorted({d.metadata['Name'].lower() for d in metadata.distributions() if d.metadata['Name']}):
        for raw in metadata.requires(name) or ():
            dep = Requirement(raw)
            if dep.marker and not dep.marker.evaluate({'extra': ''}):
                continue
            try:
                version = metadata.version(dep.name)
            except metadata.PackageNotFoundError:
                version = None
            if version is None or (dep.specifier and not dep.specifier.contains(version, prereleases=True)):
                conflicts.append({'package': name, 'requirement': str(dep), 'installed': version})
    return {'python': platform.python_version(), 'platform': platform.machine(),
            'prefix': str(Path(sys.prefix)), 'packages': packages, 'conflicts': conflicts}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
