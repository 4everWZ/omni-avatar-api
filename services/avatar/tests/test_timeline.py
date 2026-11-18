import struct
import unittest

from services.avatar.sessions import Session


class TimelineTests(unittest.TestCase):
    def test_reads_preserve_gaps_and_partial_segments_for_independent_peers(self):
        session = Session('owner', 'ava', None, None)
        session.segments.extend([(2, struct.pack('<3h', 10, 11, 12)),
                                 (7, struct.pack('<2h', 20, 21))])
        for start, expected in [(7, (20, 21, 0)), (3, (11, 12, 0, 0, 20)),
                                (0, (0, 0)), (9, (0,)), (4, ())]:
            self.assertEqual(session.pcm(start, len(expected)),
                             struct.pack(f'<{len(expected)}h', *expected))
