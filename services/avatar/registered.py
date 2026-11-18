"""Readiness of locally registered Inochi rigs, without opening a live session."""
import asyncio
from pathlib import Path

from aiohttp import web


class RegisteredRigs:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.validated = {}

    async def describe(self, identity, entry, factory):
        root = Path(entry['root'])
        file = root / 'model.inp' if root.is_dir() else root
        try:
            async with self.lock:
                stat = file.stat()
                version = (file, stat.st_mtime_ns, stat.st_size)
                if self.validated.get(identity) != version:
                    # Parse the same container and textures used by session creation.
                    def validate():
                        renderer = factory(root)
                        renderer.close()
                    await asyncio.to_thread(validate)
                    self.validated[identity] = version
        except Exception:
            raise web.HTTPServiceUnavailable(reason='Registered rig is unavailable') from None
        return web.json_response({'id': identity, 'driver': 'inochi2d', 'state': 'ready'})
