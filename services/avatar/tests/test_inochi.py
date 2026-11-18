import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.avatar.inochi import InochiRenderer
from services.avatar.speech_mouth import CONTEXT, HOP


def node(uid, kind='Node', x=0, y=0, z=0):
    return {
        'uuid': uid,
        'type': kind,
        'zsort': z,
        'enabled': True,
        'transform': {'trans': [x, y, 0], 'rot': [0, 0, 0], 'scale': [1, 1]},
        'children': [],
    }


def square(uid, texture, **position):
    return node(uid, 'Part', **position) | {
        'textures': [texture],
        'mesh': {
            'verts': [-4, -4, 4, -4, -4, 4, 4, 4],
            'uvs': [0, 0, 1, 0, 0, 1, 1, 1],
            'indices': [0, 1, 2, 2, 1, 3],
        },
    }


def binding(target, prop, values):
    return {
        'node': target,
        'param_name': prop,
        'values': [[v] for v in values],
        'interpolate_mode': 'Linear',
        'isSet': [[True] for _ in values],
    }


class InochiTest(unittest.TestCase):
    def load(self, children, bindings=()):
        root = node(1)
        root['children'] = children
        model = {
            'nodes': root,
            'param': [
                {
                    'name': 'Control',
                    'axis_points': [[0, 0.22, 1], [0]],
                    'min': [0, 0],
                    'max': [1, 1],
                    'defaults': [0, 0],
                    'bindings': bindings,
                }
            ],
        }
        payload = json.dumps(model).encode()
        file = b'TRNSRTS\0' + struct.pack('>I', len(payload)) + payload + b'TEX_SECT' + struct.pack('>I', 2)
        for color in ('red', 'blue'):
            buf = io.BytesIO()
            Image.new('RGBA', (8, 8), color).save(buf, 'PNG')
            file += struct.pack('>IB', len(buf.getvalue()), 0) + buf.getvalue()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'model.inp'
            path.write_bytes(file)
            return InochiRenderer(path)

    def test_zsort_opacity_and_nonuniform_keys(self):
        # Reverse traversal order: the more negative z must win regardless.
        top = square(3, 1, z=-3) | {'opacity': 0.5}
        rig = self.load([top, square(2, 0)], [binding(3, 'opacity', [0, 1, 0])])
        for u, blue in ((0, 0), (0.11, 64), (0.22, 128), (0.61, 64), (1, 0)):
            red, _, b = rig.render_pose({'Control': u}).getpixel((320, 360))
            self.assertLessEqual(abs(blue - b), 1)
            self.assertEqual(red + b, 255)

    def test_parent_scale_rotation_and_local_translation(self):
        parent = node(2, x=10)
        parent['children'] = [square(3, 0, y=8)]
        rig = self.load(
            [parent],
            [
                binding(2, 'transform.s.y', [1, 2, 2]),
                binding(2, 'transform.r.z', [0, math.pi / 2, math.pi / 2]),
                binding(3, 'transform.t.y', [0, 2, 2]),
            ],
        )
        self.assertEqual(rig.render_pose({}).getpixel((330, 368)), (255, 0, 0))
        pose = rig.render_pose({'Control': 0.22})
        self.assertEqual(pose.getpixel((310, 360)), (255, 0, 0))
        self.assertNotEqual(pose.getpixel((330, 368)), (255, 0, 0))

    def test_reject_unmapped_uvs(self):
        part = square(2, 0)
        part['mesh']['uvs'][2] = 0.75
        with self.assertRaisesRegex(ValueError, 'UV'):
            self.load([part])

    def test_pcm_controls_follow_local_voicing_and_resonance(self):
        rig = self.load([square(2, 1)])
        rig.render_pose = lambda values: values

        def packet(components, gain=1.):
            return struct.pack('<480h', *(round(gain * sum(
                amplitude * math.sin(i * 2 * math.pi * hz / 24000) for hz, amplitude in components
            )) for i in range(480)))

        def pose(center, neighbor=None):
            neighbor = center if neighbor is None else neighbor
            return rig.render_pcm(1, neighbor * 3 + center + neighbor * 3)

        front = [(200, 3000), (400, 1500), (2400, 1100)]
        back = [(200, 3000), (400, 1500), (800, 1100)]
        wide, rounded = pose(packet(front)), pose(packet(back))
        self.assertGreater(wide['MouthForm'], rounded['MouthForm'])
        self.assertGreater(min(wide['MouthOpen'], rounded['MouthOpen']), .2)
        self.assertAlmostEqual(wide['MouthForm'], pose(packet(front, .6))['MouthForm'], places=3)
        murmur = packet([(200, 1200), (400, 400), (1200, 60)])
        self.assertEqual(pose(murmur, packet(front))['MouthOpen'], 0)
        self.assertEqual(pose(bytes(960), packet(front))['MouthOpen'], 0)
        self.assertEqual(pose(struct.pack('<h', 3500) * 480)['MouthOpen'], 0)
        self.assertEqual(rig.render_pcm(1, bytes(960))['MouthForm'], 0)
        # No state from a previous generation, peer or render order.
        self.assertEqual(wide, pose(packet(front)))

    def test_isolated_click_after_silence_and_short_speech(self):
        rig = self.load([square(2, 1)])
        rig.render_pose = lambda values: values
        offset, size = rig.pcm_window

        def poses(*sounds):
            samples = [0] * size
            for start_ms, duration_ms, amplitude in sounds:
                for i in range(duration_ms * 24):
                    samples[-offset + start_ms * 24 + i] = round(
                        amplitude * math.sin(2 * math.pi * 800 * i / 24000))
            window = struct.pack(f'<{size}h', *samples)
            gain_start = (-offset - CONTEXT) * 2
            gain_size = (HOP + 2 * CONTEXT) * 2
            baseline = rig.render_pcm(0, window[gain_start:gain_start + gain_size])
            return baseline, rig.render_pcm(0, window)

        # Check a pulse inside the central packet and across each packet edge.
        for start in (-2, 8, 18):
            before, after = poses((start, 4, 800))
            self.assertGreater(before['MouthOpen'], 0)
            self.assertEqual((after['MouthOpen'], after['MouthForm']), (0., 0.))

        for sounds in (
            ((1, 16, 300),),                      # weak, sustained short onset
            ((16, 16, 700),),                     # onset continues into the next packet
            ((8, 4, 4000),),                      # strong attack after silence
            ((-100, 20, 1000), (8, 4, 800)),      # weak, brief sound inside speech
        ):
            before, after = poses(*sounds)
            self.assertGreater(before['MouthOpen'], 0)
            self.assertEqual(before, after)
