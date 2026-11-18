"""CPU renderer for registered Inochi2D 0.8 rectangular, full-texture INP rigs."""

import bisect
import io
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .speech_mouth import CONTEXT, HOP, ONSET_HISTORY, mouth_controls

CANVAS = (640, 720)
PROPERTIES = ('transform.t.x', 'transform.t.y', 'transform.s.x', 'transform.s.y', 'transform.r.z', 'opacity')


def read_inp(path):
    def read(count):
        if count > length - stream.tell():
            raise ValueError('Truncated INP')
        data = stream.read(count)
        if len(data) != count:
            raise ValueError('Truncated INP')
        return data

    def uint():
        return int.from_bytes(read(4), 'big')

    with path.open('rb') as stream:
        length = os.fstat(stream.fileno()).st_size
        if length > 128 << 20:
            raise ValueError('Registered INP exceeds loader limit')
        if read(8) != b'TRNSRTS\x00':
            raise ValueError('Expected INP1 container')
        model = json.loads(read(uint()))
        if read(8) != b'TEX_SECT':
            raise ValueError('Missing INP textures')
        textures = []
        pixels = 0
        for _ in range(uint()):
            size = uint()
            kind = read(1)[0]
            if not size:
                textures.append(None)
                continue
            if kind not in (0, 1):
                raise ValueError('Expected PNG or TGA texture')
            with Image.open(io.BytesIO(read(size)), formats=(('PNG', 'TGA')[kind],)) as im:
                pixels += im.width * im.height
                if pixels > 24_000_000:
                    raise ValueError('INP texture pixel budget exceeded')
                textures.append(im.convert('RGBA'))
    return model, textures


def compose(p, q):
    """Apply local affine q followed by parent affine p."""
    a, b, c, d, e, f = p
    xx, xy, tx, yx, yy, ty = q
    return (
        a * xx + b * yx,
        a * xy + b * yy,
        a * tx + b * ty + c,
        d * xx + e * yx,
        d * xy + e * yy,
        d * tx + e * ty + f,
    )


@dataclass
class Part:
    image: Image.Image
    local: tuple

    @classmethod
    def from_node(cls, node, textures):
        mesh = node['mesh']
        vertices, uvs = mesh['verts'], mesh['uvs']
        if len(vertices) < 8 or len(vertices) % 2 or len(vertices) != len(uvs):
            raise ValueError('Invalid INP rectangle geometry')
        xy = list(zip(vertices[::2], vertices[1::2]))
        xs, ys = zip(*xy)
        x0, y0 = min(xs), min(ys)
        width, height = max(xs) - x0, max(ys) - y0
        if width <= 0 or height <= 0:
            raise ValueError('Empty INP rectangle')
        for (x, y), u, v in zip(xy, uvs[::2], uvs[1::2]):
            if abs((x - x0) / width - u) > 1e-5 or abs((y - y0) / height - v) > 1e-5:
                raise ValueError('Expected full-texture rectangular UV mapping')
        indices = mesh['indices']
        if (
            not indices
            or len(indices) % 3
            or any(type(i) is not int or not 0 <= i < len(xy) for i in indices)
        ):
            raise ValueError('Invalid INP triangle indices')
        xs, ys = sorted(set(xs)), sorted(set(ys))
        if len(set(xy)) != len(xs) * len(ys):
            raise ValueError('Expected a complete rectangular vertex grid')
        columns, rows = {x: i for i, x in enumerate(xs)}, {y: i for i, y in enumerate(ys)}
        cells = {}
        for i, j, k in zip(indices[::3], indices[1::3], indices[2::3]):
            corners = {(columns[xy[n][0]], rows[xy[n][1]]) for n in (i, j, k)}
            cx, cy = zip(*corners)
            if len(corners) != 3 or max(cx) - min(cx) != 1 or max(cy) - min(cy) != 1:
                raise ValueError('Expected triangles within a rectangular grid cell')
            cells.setdefault((min(cx), min(cy)), []).append(corners)
        if len(cells) != (len(xs) - 1) * (len(ys) - 1):
            raise ValueError('Uncovered INP grid cells')
        for triangles in cells.values():
            if len(triangles) != 2:
                raise ValueError('Expected two triangles per cell')
            shared = triangles[0] & triangles[1]
            if len(shared) != 2 or len({p[0] for p in shared}) != 2 or len({p[1] for p in shared}) != 2:
                raise ValueError('Expected shared cell diagonal')
        slot = node['textures'][0]
        if type(slot) is not int or not 0 <= slot < len(textures) or textures[slot] is None:
            raise ValueError('Missing INP albedo texture')
        image = textures[slot]
        sx, sy = width / image.width, height / image.height
        ox, oy = mesh.get('origin', [0, 0])
        bbox = image.getbbox()
        if bbox is None:
            return None
        patch = ImageOps.expand(image.crop(bbox), border=1).convert('RGBa')
        return cls(patch, (sx, 0, x0 - ox + (bbox[0] - 1) * sx, 0, sy, y0 - oy + (bbox[1] - 1) * sy))

    def draw(self, frame, world, opacity):
        a, b, c, d, e, f = compose(world, self.local)
        det = a * e - b * d
        if abs(det) < 1e-9 or opacity <= 0:
            return
        w, h = self.image.size
        x = (c, a * w + c, b * h + c, a * w + b * h + c)
        y = (f, d * w + f, e * h + f, d * w + e * h + f)
        left, top = max(0, math.floor(min(x))), max(0, math.floor(min(y)))
        right, bottom = min(frame.width, math.ceil(max(x))), min(frame.height, math.ceil(max(y)))
        if right <= left or bottom <= top:
            return
        inverse = (
            e / det,
            -b / det,
            (e * (left - c) - b * (top - f)) / det,
            -d / det,
            a / det,
            (a * (top - f) - d * (left - c)) / det,
        )
        if b == d == 0 and a > 0 and e > 0:
            # The source has a transparent border. Resample the visible rectangle
            # with Pillow's separable kernel, keeping fractional translation.
            left, top = max(left, math.ceil(c)), max(top, math.ceil(f))
            right, bottom = min(right, math.floor(c + a * w)), min(bottom, math.floor(f + e * h))
            if right <= left or bottom <= top:
                return
            box = (
                max(0, (left - c) / a),
                max(0, (top - f) / e),
                min(w, (right - c) / a),
                min(h, (bottom - f) / e),
            )
            patch = self.image.resize((right - left, bottom - top), Image.Resampling.BILINEAR, box)
        else:
            patch = self.image.transform(
                (right - left, bottom - top),
                Image.Transform.AFFINE,
                inverse,
                resample=Image.Resampling.BILINEAR,
            )
        patch = patch.convert('RGBA')
        alpha = patch.getchannel('A')
        if opacity < 1:
            alpha = alpha.point(lambda value: round(value * opacity))
        frame.paste(patch, (left, top), alpha)


@dataclass
class Node:
    parent: int
    base: tuple
    order: float
    part: Part | None
    enabled: bool


class InochiRenderer:
    capabilities = {'video': '640x720', 'fps': 25, 'alpha': False, 'driver': 'inochi2d'}
    # Earlier samples distinguish a quiet onset; the lookahead stays at CONTEXT.
    pcm_window = (-ONSET_HISTORY, ONSET_HISTORY + HOP + CONTEXT)

    def __init__(self, root: Path):
        model, textures = read_inp(root / 'model.inp' if root.is_dir() else root)
        self.nodes, self.parameters = [], {}
        identifiers = {}

        def visit(raw, parent):
            kind = raw['type']
            if kind not in ('Node', 'Part') or raw.get('lockToRoot') or raw.get('masks'):
                raise ValueError('Unsupported INP node or mask')
            if raw.get('blend_mode', 'Normal') != 'Normal':
                raise ValueError('Expected Normal INP layer blending')
            transform = raw['transform']
            x, y, _ = transform['trans']
            rx, ry, angle = transform['rot']
            if rx or ry:
                raise ValueError('Only in-plane rotation is supported')
            sx, sy = transform['scale']
            base = (x, y, sx, sy, angle, raw.get('opacity', 1))
            if not all(math.isfinite(v) for v in base) or raw['uuid'] in identifiers:
                raise ValueError('Invalid INP node transform or duplicate UUID')
            part = Part.from_node(raw, textures) if kind == 'Part' else None
            index = identifiers[raw['uuid']] = len(self.nodes)
            order = raw.get('zsort', 0) + (self.nodes[parent].order if parent >= 0 else 0)
            enabled = raw.get('enabled', True) and (parent < 0 or self.nodes[parent].enabled)
            self.nodes.append(Node(parent, base, order, part, enabled))
            for child in raw.get('children', []):
                visit(child, index)

        visit(model['nodes'], -1)
        self.layers = sorted(
            (i for i, n in enumerate(self.nodes) if n.part and n.enabled),
            key=lambda i: self.nodes[i].order,
            reverse=True,
        )
        for p in model['param']:
            axis = p['axis_points'][0]
            lo, hi = p['min'][0], p['max'][0]
            if (
                p.get('is_vec2')
                or hi <= lo
                or len(axis) < 2
                or axis[0] != 0
                or axis[-1] != 1
                or any(not a < b for a, b in zip(axis, axis[1:]))
                or p['name'] in self.parameters
            ):
                raise ValueError('Expected scalar INP parameter with normalized ordered keys')
            bindings = []
            for binding in p['bindings']:
                if binding.get('interpolate_mode', 'Linear') != 'Linear':
                    raise ValueError('Only linear INP bindings are supported')
                key = PROPERTIES.index(binding['param_name'])
                values = binding['values']
                if len(values) != len(axis) or any(len(v) != 1 or not math.isfinite(v[0]) for v in values):
                    raise ValueError('Invalid INP scalar binding values')
                bindings.append((identifiers[binding['node']], key, [v[0] for v in values]))
            self.parameters[p['name']] = (lo, hi, p['defaults'][0], axis, bindings)

    def render_pose(self, values):
        pose = [list(node.base) for node in self.nodes]
        for name, (lo, hi, default, axis, bindings) in self.parameters.items():
            u = (max(lo, min(hi, values.get(name, default))) - lo) / (hi - lo)
            j = min(len(axis) - 1, bisect.bisect_right(axis, u))
            t = (u - axis[j - 1]) / (axis[j] - axis[j - 1])
            for index, key, curve in bindings:
                value = curve[j - 1] + t * (curve[j] - curve[j - 1])
                if key in (2, 3, 5):
                    pose[index][key] *= value
                else:
                    pose[index][key] += value
        matrices = []
        for node, (x, y, sx, sy, angle, _) in zip(self.nodes, pose):
            co, si = math.cos(angle), math.sin(angle)
            local = (co * sx, -si * sy, x, si * sx, co * sy, y)
            parent = matrices[node.parent] if node.parent >= 0 else (1, 0, CANVAS[0] / 2, 0, 1, CANVAS[1] / 2)
            matrices.append(compose(parent, local))
        frame = Image.new('RGB', CANVAS, '#ece7df')
        for i in self.layers:
            self.nodes[i].part.draw(frame, matrices[i], max(0, min(1, pose[i][5])))
        return frame

    @staticmethod
    def speech_pose(timestamp, opening, form):
        phase = timestamp % 4.1
        blink = max(0, 1 - abs(phase - 3.7) / 0.12)
        return {
            'MouthOpen': opening,
            'MouthForm': form,
            'BlinkL': blink,
            'BlinkR': blink,
            'GazeX': 0.45 * math.sin(timestamp * 0.67),
            'GazeY': 0.3 * math.sin(timestamp * 0.89),
            'HeadX': 0.65 * math.sin(timestamp * 0.93),
            'HeadY': 0.5 * math.sin(timestamp * 1.1),
            'Breath': 0.5 - 0.5 * math.cos(timestamp * 1.7),
        }

    def render_pcm(self, timestamp, pcm):
        return self.render_pose(self.speech_pose(timestamp, *mouth_controls(pcm)))

    def render(self, timestamp, rms):
        return self.render_pose(self.speech_pose(timestamp, max(0, min(1, (rms - 0.008) * 8)), 0))

    def push_audio(self, pcm, sample_rate, start_sample, generation):
        pass  # PCM is read from the session timeline at presentation time.

    def clear(self, generation):
        pass

    def close(self):
        pass  # In-flight CPU renders retain immutable textures until they finish.
