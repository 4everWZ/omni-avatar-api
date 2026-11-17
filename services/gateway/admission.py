"""Bounded backend admission, independent of HTTP transport."""
import asyncio
from contextlib import asynccontextmanager


class Busy(Exception):
    pass


class Admission:
    def __init__(self, config):
        self.config = config
        self.manual_mode = "batch"
        self.program_sessions = 0
        self.active = {"chat": 0, "realtime": 0}
        self.queued = 0
        self.changed = asyncio.Event()

    def _signal(self):
        event, self.changed = self.changed, asyncio.Event()
        event.set()

    def set_mode(self, mode):
        self.manual_mode = mode
        self._signal()

    @property
    def mode(self):
        return "live" if self.program_sessions or self.manual_mode == "live" else "batch"

    @asynccontextmanager
    async def program(self, enabled):
        if enabled:
            self.program_sessions += 1
            self._signal()
        try:
            yield
        finally:
            if enabled:
                self.program_sessions -= 1
                self._signal()

    def _available(self, lane):
        chat, realtime = self.active["chat"], self.active["realtime"]
        if self.mode == "batch":
            return chat + realtime < self.config.batch_cap and (lane == "chat" or realtime == 0)
        if lane == "chat":
            return chat == 0
        # Drain ordinary work admitted above the live cap before opening the reserved lane.
        return realtime == 0 and chat <= 1

    @asynccontextmanager
    async def hold(self, lane):
        if not self._available(lane):
            if self.queued >= self.config.queue_limit:
                raise Busy()
            self.queued += 1
            try:
                async with asyncio.timeout(self.config.queue_timeout):
                    while not self._available(lane):
                        await self.changed.wait()
            except TimeoutError as error:
                raise Busy() from error
            finally:
                self.queued -= 1
        self.active[lane] += 1
        try:
            yield
        finally:
            self.active[lane] -= 1
            self._signal()

    def view(self):
        return {"mode": self.mode, "active": self.active.copy(), "queued": self.queued,
                "batch_cap": self.config.batch_cap, "live_chat_cap": 1, "live_realtime_reserve": 1}
