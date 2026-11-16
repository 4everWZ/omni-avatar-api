"""WHEP media tracks share one session clock and presentation timeline."""
import asyncio
from fractions import Fraction
from time import monotonic

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame

from sessions import RATE


class TimelineTrack(MediaStreamTrack):
    def __init__(self, session, kind, origin):
        super().__init__()
        self.session, self.kind, self.origin = session, kind, origin
        self.index = 0

    async def recv(self):
        if self.readyState != 'live' or self.session.state == 'closed':
            raise MediaStreamError
        period = .02 if self.kind == 'audio' else .04
        timestamp = self.origin + self.index * period
        await asyncio.sleep(max(0, timestamp - monotonic()))
        position = round((timestamp - self.session.origin) * RATE)
        if self.kind == 'audio':
            frame = AudioFrame(format='s16', layout='mono', samples=480)
            frame.planes[0].update(self.session.pcm(position, 480))
            frame.sample_rate = RATE
            frame.pts, frame.time_base = self.index * 480, Fraction(1, RATE)
        else:
            try:
                image = self.session.renderer.render(timestamp - self.session.origin, self.session.rms(position))
                frame = VideoFrame.from_image(image)
            except Exception:
                self.session.state, self.session.failure = 'failed', 'renderer_failed'
                self.session.emit('avatar.failed', reason='renderer_failed', voice_continues=True)
                self.stop()
                raise MediaStreamError
            frame.pts, frame.time_base = self.index * 3600, Fraction(1, 90000)
        self.index += 1
        return frame
