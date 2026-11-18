import struct
import unittest
from time import monotonic
from unittest.mock import Mock

from PIL import Image

from services.avatar.media import TimelineTrack, VIDEO_SAMPLES
from services.avatar.sessions import RATE, Session


class TimelineTests(unittest.TestCase):
    def test_reads_preserve_gaps_and_partial_segments_for_independent_peers(self):
        session = Session('owner', 'ava', None, None)
        session.segments.extend([(2, struct.pack('<3h', 10, 11, 12)),
                                 (7, struct.pack('<2h', 20, 21))])
        for start, expected in [(7, (20, 21, 0)), (3, (11, 12, 0, 0, 20)),
                                (0, (0, 0)), (9, (0,)), (4, ())]:
            self.assertEqual(session.pcm(start, len(expected)),
                             struct.pack(f'<{len(expected)}h', *expected))


class LateVideoTests(unittest.IsolatedAsyncioTestCase):
    async def test_backpressured_rig_reads_current_pcm_and_keeps_rtp_origin(self):
        renderer = Mock(spec=['render_pcm'])
        renderer.render_pcm.return_value = Image.new('RGB', (8, 8))
        session = Session('owner', 'rig', None, renderer, origin=monotonic() - 5)
        start = RATE
        current = start + 34 * VIDEO_SAMPLES
        session.now_sample = lambda: current + 100
        speech = struct.pack('<480h', *([9000] * 480))
        session.segments.append((current, speech))
        video = TimelineTrack(session, 'video', start)
        self.assertEqual((await video.recv()).pts, 0)
        self.assertEqual((await video.recv()).pts, 34 * 3600)
        renderer.render_pcm.assert_called_with(current / RATE, speech)
        self.assertEqual((await video.recv()).pts, 35 * 3600)
        audio = TimelineTrack(session, 'audio', start)
        self.assertEqual((await audio.recv()).pts, 0)
        self.assertEqual((await audio.recv()).pts, 480)
