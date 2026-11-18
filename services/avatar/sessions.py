"""Owned sessions and a bounded PCM timeline, independent of transport."""
import asyncio
import base64
import binascii
import math
import secrets
import struct
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

RATE = 24000
MAX_AUDIO_SECONDS = 30


@dataclass
class Session:
    owner: str
    avatar_id: str
    source: str | None
    renderer: object
    id: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    generation: int = 0
    state: str = 'active'
    failure: str | None = None
    origin: float = field(default_factory=monotonic)
    touched: float = field(default_factory=monotonic)
    segments: deque = field(default_factory=deque)
    end_sample: int = 0
    events: set = field(default_factory=set)
    peers: dict = field(default_factory=dict)
    last_ack: dict = field(default_factory=dict)
    response_id: str | None = None
    epoch: int = -1
    blocked_responses: set = field(default_factory=set)
    response_samples: int = 0
    recent_events: deque = field(default_factory=lambda: deque(maxlen=64))

    def now_sample(self):
        return int((monotonic() - self.origin) * RATE)

    def view(self):
        base = f'/v1/avatar/sessions/{self.id}'
        return {'id': self.id, 'avatar_id': self.avatar_id, 'source_session_id': self.source,
                'state': self.state, 'failure': self.failure, 'generation': self.generation,
                'timeline_ms': self.now_sample() * 1000 // RATE,
                'format': {'transport': 'whep', 'video': '640x720', 'fps': 25,
                           'audio': 'opus', 'input': 'pcm16', 'sample_rate_hz': RATE,
                           **self.renderer.capabilities},
                'media_url': base + '/whep', 'events_url': base + '/events',
                'audio_url': base + '/audio', 'playback': self.last_ack,
                'renderer': self.renderer.status() if hasattr(self.renderer, 'status') else {}}

    def emit(self, kind, **fields):
        event = {'type': kind, 'session_id': self.id, 'generation': self.generation,
                 'source_epoch': self.epoch,
                 'timestamp_ms': self.now_sample() * 1000 // RATE, **fields}
        self.recent_events.append(event)
        for queue in tuple(self.events):
            if queue.full():
                # Never silently lose control events. A slow consumer must reconnect.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({'type': 'avatar.resync_required'})
            else:
                queue.put_nowait(event)

    def clear(self, response_id=None, epoch=None):
        if self.response_id:
            self.blocked_responses.add(self.response_id)
        if response_id:
            self.blocked_responses.add(response_id)
        # Only recent IDs matter; IDs must never be reused by the upstream.
        if len(self.blocked_responses) > 128:
            self.blocked_responses = {v for v in (response_id, self.response_id) if v}
        if epoch is not None:
            self.epoch = max(self.epoch, epoch)
        self.generation += 1
        self.last_ack = {}
        self.segments.clear()
        self.response_samples = 0
        self.end_sample = self.now_sample()
        self.renderer.clear(self.generation)
        self.emit('avatar.cleared')

    def append(self, event):
        if self.state != 'active':
            raise ValueError('Session is not active')
        generation = event.get('generation', self.generation)
        if type(generation) is not int or generation != self.generation:
            raise ValueError('Stale generation')
        if event.get('sample_rate_hz', RATE) != RATE or event.get('format', 'pcm16') not in ('pcm16', 'pcm'):
            raise ValueError('Expected mono 24000 Hz PCM16')
        encoded = event.get('audio', event.get('delta', ''))
        if not isinstance(encoded, str) or len(encoded) > RATE * 2 * 5 * 4 // 3 + 4:
            raise ValueError('Audio chunks must be at most five seconds')
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError('Invalid base64 audio') from error
        if not pcm or len(pcm) % 2:
            raise ValueError('Expected nonempty PCM16 samples')
        now = self.now_sample()
        while self.segments and self.segments[0][0] + len(self.segments[0][1]) // 2 < now - RATE:
            self.segments.popleft()
        lead = self.renderer.capabilities.get('input_lead_ms', 100)
        start = max(self.end_sample, now + round(RATE * lead / 1000))
        if start + len(pcm) // 2 - now > RATE * MAX_AUDIO_SECONDS:
            raise ValueError('Audio exceeds thirty-second playback buffer')
        self.renderer.push_audio(pcm, RATE, start, self.generation)
        self.segments.append((start, pcm))
        self.end_sample = start + len(pcm) // 2
        self.touched = monotonic()
        self.emit('avatar.audio.queued', start_ms=start * 1000 // RATE,
                  end_ms=self.end_sample * 1000 // RATE, response_id=self.response_id,
                  source_start_ms=self.response_samples * 1000 // RATE)
        self.response_samples += len(pcm) // 2

    def pcm(self, start, count):
        output = bytearray(count * 2)
        end = start + count
        for position, data in self.segments:
            # append() keeps segments ordered and non-overlapping.
            if position >= end:
                break
            segment_end = position + len(data) // 2
            if segment_end <= start:
                continue
            left, right = max(position, start), min(segment_end, end)
            output[(left-start)*2:(right-start)*2] = data[(left-position)*2:(right-position)*2]
        return bytes(output)

    def rms(self, start):
        samples = struct.unpack('<480h', self.pcm(start, 480))
        return math.sqrt(sum(value * value for value in samples) / 480) / 32768

    def ingest(self, event, direction='upstream'):
        kind = event.get('type', '')
        metadata = event.get('metadata') or {}
        epoch = event.get('epoch', metadata.get('epoch'))
        if type(epoch) is int:
            if epoch < self.epoch:
                return
            if self.epoch >= 0 and epoch > self.epoch:
                self.clear(epoch=epoch)
            self.epoch = epoch
        response = event.get('response_id')
        if kind in ('response.cancel', 'output_audio_buffer.clear', 'barge_in') and direction == 'downstream':
            # Fence audio already in flight, including a cancellation before the first delta.
            self.clear(response, epoch=max(0, self.epoch) + 1)
        elif kind == 'output_audio_buffer.cleared':
            self.clear(response)
        elif direction == 'upstream' and kind in ('response.created', 'response.speak'):
            new_response = response or event.get('response', {}).get('id')
            if new_response != self.response_id:
                self.response_samples = 0
            self.response_id = new_response
        elif direction == 'upstream' and kind in ('response.audio.delta', 'response.output_audio.delta'):
            if response in self.blocked_responses:
                return
            if self.response_id != response:
                self.response_samples = 0
            self.response_id = response
            self.append(event)
        elif direction == 'upstream' and kind in ('response.audio_transcript.delta',
                'response.output_audio_transcript.delta', 'response.text.delta', 'response.output_text.delta'):
            if response not in self.blocked_responses:
                self.emit('avatar.subtitle', text=event.get('delta', ''), response_id=response,
                          start_ms=(self.segments[-1][0] if self.segments else self.now_sample()) * 1000 // RATE)

    async def close(self):
        if self.state == 'closed':
            return
        self.state = 'closed'
        self.segments.clear()
        self.emit('avatar.closed')
        await asyncio.gather(*(pc.close() for pc in tuple(self.peers.values())), return_exceptions=True)
        self.peers.clear()
        pending = self.renderer.close()
        if pending is not None:
            await pending

    async def fail(self, reason):
        if self.state != 'active':
            return
        self.state, self.failure = 'failed', reason
        self.segments.clear()
        self.emit('avatar.failed', reason=reason, voice_continues=True)
        pending = self.renderer.close()
        if pending is not None:
            await pending
