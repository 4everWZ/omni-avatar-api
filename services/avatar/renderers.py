"""Rendering boundary shared by sprite and remote neural drivers."""
import math
from pathlib import Path
from typing import Awaitable, Protocol

from PIL import Image
from .inochi import InochiRenderer


class Renderer(Protocol):
    capabilities: dict
    def push_audio(self, pcm: bytes, sample_rate: int, start_sample: int, generation: int) -> None: ...
    def render(self, timestamp: float, rms: float) -> Image.Image: ...
    def clear(self, generation: int) -> None: ...
    def close(self) -> Awaitable[None] | None: ...


class SpriteRenderer:
    """Four registered poses: closed, small mouth, open mouth, blink."""
    capabilities = {'video': '640x720', 'fps': 25, 'alpha': False, 'driver': 'sprite2d'}
    def __init__(self, root: Path):
        with Image.open(root / 'sprites.png') as atlas:
            atlas = atlas.convert('RGBA')
            width, height = atlas.width // 2, atlas.height // 2
            self.poses = []
            for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
                pose = atlas.crop((x * width, y * height, (x + 1) * width, (y + 1) * height))
                pose.thumbnail((600, 680), Image.Resampling.LANCZOS)
                self.poses.append(pose)

    def push_audio(self, pcm, sample_rate, start_sample, generation):
        # Sprite animation samples amplitude from the shared timeline. Neural drivers
        # can enqueue this PCM in their worker without changing session or media APIs.
        pass

    def render(self, timestamp, rms):
        blink = timestamp % 4.7 < .14
        index = (2 if rms > .07 else 1 if rms > .012 else 0)
        if blink and index == 0:
            index = 3
        pose = self.poses[index]
        frame = Image.new('RGB', (640, 720), '#ece7df')
        y = 720 - pose.height + round(2 * math.sin(timestamp * 1.8))
        frame.paste(pose, ((640 - pose.width) // 2, y), pose)
        return frame

    def clear(self, generation):
        pass

    def close(self):
        self.poses.clear()


DRIVERS = {'sprite2d': SpriteRenderer, 'inochi2d': InochiRenderer}
