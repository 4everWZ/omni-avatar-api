"""Pace both WHEP senders from the instant a subscription can send media."""
import asyncio
import logging
from fractions import Fraction
from time import monotonic, perf_counter

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame

from .sessions import RATE

VIDEO_SAMPLES = RATE // 25


class PlaybackClock:
    def __init__(self, session):
        # The SDP answer commits the content cursor for playback receipts.
        # ICE/DTLS may still take seconds. It does not advance this cursor.
        self.start_sample = ((session.now_sample() + VIDEO_SAMPLES - 1) // VIDEO_SAMPLES) * VIDEO_SAMPLES
        self.earliest = session.origin + self.start_sample / RATE
        self.started_at = None
        self.senders = set()
        self.ready = asyncio.Event()

    async def packet(self, kind, index, size):
        if self.started_at is None:
            self.senders.add(kind)
            if len(self.senders) == 2:
                self.started_at = perf_counter() + max(0, self.earliest - monotonic())
                self.ready.set()
            else:
                await self.ready.wait()
        deadline = self.started_at + index * size / RATE
        while (delay := deadline - perf_counter()) > 0:
            await asyncio.sleep(delay)
        if index == 0:
            return 0  # Both RTP streams start at the cursor advertised in SDP.
        # A late encoder skips expired packets on the same elapsed timeline
        # for either medium. Its RTP timestamp advances by the same amount.
        due = int((perf_counter() - self.started_at) * RATE) // size
        return max(index, due)


class TimelineTrack(MediaStreamTrack):
    def __init__(self, session, kind, clock):
        super().__init__()
        self.session, self.kind, self.clock = session, kind, clock
        self.index = 0

    async def recv(self):
        if self.readyState != 'live' or self.session.state != 'active':
            raise MediaStreamError
        samples_per_packet = 480 if self.kind == 'audio' else VIDEO_SAMPLES
        index = await self.clock.packet(self.kind, self.index, samples_per_packet)
        if self.readyState != 'live' or self.session.state != 'active':
            raise MediaStreamError
        position = self.clock.start_sample + index * samples_per_packet
        if self.kind == 'audio':
            frame = AudioFrame(format='s16', layout='mono', samples=480)
            frame.planes[0].update(self.session.pcm(position, 480))
            frame.sample_rate = RATE
            frame.pts, frame.time_base = index * samples_per_packet, Fraction(1, RATE)
        else:
            try:
                renderer = self.session.renderer
                if render_pcm := getattr(renderer, 'render_pcm', None):
                    offset, size = getattr(renderer, 'pcm_window', (0, 480))
                    while True:
                        generation = self.session.generation
                        audio = self.session.pcm(position + offset, size)
                        image = await asyncio.to_thread(render_pcm, position / RATE, audio)
                        if generation == self.session.generation:
                            break
                else:
                    image = renderer.render(position / RATE, self.session.rms(position))
                frame = VideoFrame.from_image(image)
            except Exception:
                logging.exception('Avatar renderer failed')
                await self.session.fail('renderer_failed')
                self.stop()
                raise MediaStreamError
            frame.pts, frame.time_base = index * 3600, Fraction(1, 90000)
        self.index = index + 1
        return frame
