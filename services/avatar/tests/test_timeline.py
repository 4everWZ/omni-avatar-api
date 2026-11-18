import asyncio
import base64
import struct
import unittest
from time import monotonic, perf_counter
from unittest.mock import Mock

from PIL import Image

from services.avatar.media import PlaybackClock, TimelineTrack, VIDEO_SAMPLES
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
    def tracks(self, session):
        clock = PlaybackClock(session)
        return TimelineTrack(session, 'audio', clock), TimelineTrack(session, 'video', clock)

    async def test_rig_context_crosses_chunks_without_advancing_audio(self):
        from services.avatar.inochi import InochiRenderer

        renderer = Mock(spec=InochiRenderer)
        renderer.pcm_window = InochiRenderer.pcm_window
        renderer.render_pcm.return_value = Image.new('RGB', (8, 8))
        session = Session('owner', 'rig', None, renderer, origin=monotonic() - 1)
        session.now_sample = lambda: 0
        first, next_chunk = struct.pack('<h', 700) * 240, struct.pack('<h', 2000) * 900
        session.segments.extend([(0, first), (240, next_chunk)])
        audio, video = self.tracks(session)
        await asyncio.gather(audio.recv(), video.recv())
        offset, size = renderer.pcm_window
        expected = bytes(-2 * offset) + first + next_chunk
        renderer.render_pcm.assert_called_once_with(0, expected.ljust(size*2, b'\0'))
        self.assertEqual(session.pcm(0, 480), first + next_chunk[:480])
        session.clear()
        audio, video = self.tracks(session)
        await asyncio.gather(audio.recv(), video.recv())
        renderer.render_pcm.assert_called_with(0, bytes(size*2))

    async def test_delayed_handshake_preserves_queued_pcm_and_paces_both_senders(self):
        renderer = Mock(spec=['render_pcm', 'capabilities', 'push_audio', 'clear'])
        renderer.capabilities = {'input_lead_ms': 0}
        renderer.render_pcm.return_value = Image.new('RGB', (8, 8))
        session = Session('owner', 'rig', None, renderer)
        audio, video = self.tracks(session)
        speech = struct.pack('<h', 9000) * 480
        session.segments.append((audio.clock.start_sample, speech))
        waiting_audio = asyncio.create_task(audio.recv())
        await asyncio.sleep(1.12)
        self.assertFalse(waiting_audio.done())
        session.append({'audio': base64.b64encode(speech).decode()})
        first_audio, first_video = await asyncio.gather(waiting_audio, video.recv())
        self.assertEqual((first_audio.pts, first_video.pts), (0, 0))
        self.assertEqual(bytes(first_audio.planes[0]), speech)
        renderer.render_pcm.assert_called_once_with(audio.clock.start_sample / RATE, speech)

        async def collect(track, count):
            started = perf_counter()
            for packet in range(count):
                await track.recv()
                if packet + 1 == count // 2:
                    self.assertGreater(perf_counter() - started, .055)
            elapsed = perf_counter() - started
            self.assertGreater(elapsed, .16)
            self.assertLess(elapsed, .5)

        await asyncio.gather(collect(audio, 10), collect(video, 5))
        session.clear()
        next_audio, next_video = self.tracks(session)
        self.assertGreater(next_audio.clock.start_sample, audio.clock.start_sample)
        a, v = await asyncio.gather(next_audio.recv(), next_video.recv())
        self.assertEqual((a.pts, v.pts, bytes(a.planes[0])), (0, 0, bytes(960)))

    async def test_backpressured_senders_skip_on_one_subscription_clock(self):
        renderer = Mock(spec=['render_pcm'])
        renderer.render_pcm.return_value = Image.new('RGB', (8, 8))
        session = Session('owner', 'rig', None, renderer, origin=monotonic() - 5)
        start = RATE
        current = start + 34 * VIDEO_SAMPLES
        session.now_sample = lambda: start
        speech = struct.pack('<480h', *([9000] * 480))
        session.segments.append((current, speech))
        audio, video = self.tracks(session)
        await asyncio.gather(audio.recv(), video.recv())
        audio.clock.started_at = perf_counter() - 34 / 25 - .002
        a, v = await asyncio.gather(audio.recv(), video.recv())
        self.assertEqual((a.pts, v.pts), (68 * 480, 34 * 3600))
        self.assertEqual(bytes(a.planes[0]), speech)
        renderer.render_pcm.assert_called_with(current / RATE, speech)
        self.assertEqual((await video.recv()).pts, 35 * 3600)
