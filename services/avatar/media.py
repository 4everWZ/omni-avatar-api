"""WHEP media tracks share one session clock and presentation timeline."""
import asyncio
import logging
from fractions import Fraction
from time import monotonic

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame

from .sessions import RATE

VIDEO_SAMPLES = RATE // 25


class TimelineTrack(MediaStreamTrack):
    def __init__(self, session, kind, start_sample):
        super().__init__()
        self.session, self.kind, self.start_sample = session, kind, start_sample
        self.index = 0

    async def recv(self):
        if self.readyState != 'live' or self.session.state == 'closed':
            raise MediaStreamError
        samples_per_packet = 480 if self.kind == 'audio' else VIDEO_SAMPLES
        position = self.start_sample + self.index * samples_per_packet
        timestamp = self.session.origin + position / RATE
        await asyncio.sleep(max(0, timestamp - monotonic()))
        if self.kind == 'audio':
            frame = AudioFrame(format='s16', layout='mono', samples=480)
            frame.planes[0].update(self.session.pcm(position, 480))
            frame.sample_rate = RATE
            frame.pts, frame.time_base = self.index * 480, Fraction(1, RATE)
        else:
            try:
                image = self.session.renderer.render(position / RATE, self.session.rms(position))
                frame = VideoFrame.from_image(image)
            except Exception:
                logging.exception('Avatar renderer failed')
                await self.session.fail('renderer_failed')
                self.stop()
                raise MediaStreamError
            frame.pts, frame.time_base = self.index * 3600, Fraction(1, 90000)
        self.index += 1
        return frame
