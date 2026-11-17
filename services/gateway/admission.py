"""Bounded backend admission, independent of HTTP transport."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager


class Busy(Exception):
    pass


class BudgetExceeded(ValueError):
    pass


class Admission:
    def __init__(self, config):
        self.config = config
        self.manual_mode = "batch"
        self.program_sessions = 0
        self.active = {"chat": 0, "realtime": 0}
        self.queued = 0
        self.reserved_tokens = 0
        self.waiters = deque()
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

    def _available(self, lane, tokens=0, slots=1):
        chat, realtime = self.active["chat"], self.active["realtime"]
        if self.config.token_budget:
            return (chat + realtime + slots <= self.config.batch_cap
                    and self.reserved_tokens + tokens <= self.config.token_budget)
        if self.mode == "batch":
            return chat + realtime + slots <= self.config.batch_cap and (lane == "chat" or realtime == 0)
        if lane == "chat":
            return chat == 0
        # Drain ordinary work admitted above the live cap before opening the reserved lane.
        return realtime == 0 and chat <= 1

    @asynccontextmanager
    async def hold(self, lane, *, tokens=0, slots=1):
        if (tokens < 0 or slots < 1 or slots > self.config.batch_cap
                or (self.config.token_budget and (tokens <= 0 or tokens > self.config.token_budget))):
            raise BudgetExceeded('Request exceeds the configured token or sequence budget.')
        if (self.config.token_budget and self.waiters) or not self._available(lane, tokens, slots):
            if self.queued >= self.config.queue_limit:
                raise Busy()
            ticket = object()
            self.waiters.append(ticket)
            self.queued += 1
            try:
                async with asyncio.timeout(self.config.queue_timeout):
                    while ((self.config.token_budget and self.waiters[0] is not ticket)
                           or not self._available(lane, tokens, slots)):
                        await self.changed.wait()
            except TimeoutError as error:
                raise Busy() from error
            finally:
                self.waiters.remove(ticket)
                self.queued -= 1
                self._signal()
        self.active[lane] += slots
        self.reserved_tokens += tokens
        try:
            yield
        finally:
            self.active[lane] -= slots
            self.reserved_tokens -= tokens
            self._signal()

    def view(self):
        state = {"mode": self.mode, "active": self.active.copy(), "queued": self.queued,
                 "batch_cap": self.config.batch_cap, "live_chat_cap": 1, "live_realtime_reserve": 1}
        if self.config.token_budget:
            state.update(token_budget=self.config.token_budget, reserved_tokens=self.reserved_tokens,
                         available_tokens=self.config.token_budget - self.reserved_tokens,
                         scheduling='fifo', live_chat_cap=self.config.batch_cap, live_realtime_reserve=0)
        return state
