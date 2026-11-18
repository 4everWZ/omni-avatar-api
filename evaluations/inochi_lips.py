"""Compare PCM mouth controls on native 25 Hz INP frames using external synthesis events.

Run from the repository root as python -m evaluations.inochi_lips --help.
Events are read by the scorer only. The production renderer receives PCM alone.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
from time import perf_counter_ns

import cv2
import numpy as np

from services.avatar.inochi import InochiRenderer
from services.avatar.speech_mouth import mouth_controls

RATE = 24000
FRAME = 960
CENTERS = {'regular': (370, 418), 'q': (347, 465)}


def events(path):
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('viseme,'):
            _, offset, duration, value = line.split(',')
            shape = int(value)
        elif line.startswith('phoneme_ipa,'):
            _, offset, duration, value = line.split(',')
            # Only unambiguous monophthongs and bilabials. Unclassified
            # diphthongs, coarticulation and silence are left out of scoring.
            shape = {'m': 21, 'p': 21, 'b': 21, 'ph': 21, 'pʰ': 21,
                     'i': 6, 'u': 7, 'y': 7, 'o': 8, 'ɔ': 8}.get(value.strip('ˈˌː'))
        else:
            continue
        if float(duration) >= .04 and shape in (6, 7, 8, 21):
            yield float(offset) + float(duration) / 2, shape


def previous(pcm):
    x = struct.unpack('<480h', pcm)
    energy = sum(v*v for v in x)
    rms = math.sqrt(energy / 480) / 32768
    brightness = math.sqrt(sum((a-b)**2 for a, b in zip(x, x[1:])) / energy) if rms > .008 else 0
    return max(0, min(1, (rms-.008)*8)), max(-1, min(1, (brightness-.28)*4)) if rms > .008 else 0.


def component(frame, center):
    x, y = center
    rgb = np.asarray(frame.crop((x-49, y-42, x+50, y+43)), dtype=np.int16)
    red, green = rgb[:, :, 0], rgb[:, :, 1]
    mask = ((red-green > 50) & (green < 207) & (red > 136)).astype(np.uint8)
    _, _, boxes, centers = cv2.connectedComponentsWithStats(mask)
    boxes = [b for b, (cx, cy) in zip(boxes[1:], centers[1:]) if abs(cx-49)<36 and abs(cy-42)<32]
    _, _, w, h, area = max(boxes, key=lambda b: b[4], default=(0, 0, 0, 0, 0))
    return int(area), int(w), int(h)


def paired_vowels(vowels):
    available = list(vowels['round'])
    order = []
    for area, aspect in vowels['spread']:
        eligible = [(abs(a/area-1), i) for i, (a, _) in enumerate(available) if abs(a/area-1) <= .1]
        if eligible:
            _, i = min(eligible)
            order.append(aspect > available.pop(i)[1])
    return [sum(order), len(order)]


def measure(rig, center, pcm, labels, method, phase):
    # WHEP holds each native frame until the next PTS. Reproduce this sampling,
    # without claiming that an encoded or presented frame was observed.
    off, size = rig.pcm_window
    pad = bytes(2 * size)
    signal = pad + pcm + pad
    quiet = max(component(rig.render_pcm(t/25, bytes(size*2)), center)[0] for t in range(0, 25)) + 4
    tally = Counter()
    vowels = {'spread': [], 'round': []}
    for mid, shape in labels:
        index = math.floor((mid*RATE-phase)/FRAME) * FRAME + phase
        start = len(pad) + index*2
        ts = index/RATE
        if method == 'old':
            pose = rig.speech_pose(ts, *previous(signal[start:start+960]))
            image = rig.render_pose(pose)
        else:
            start += off*2
            image = rig.render_pcm(ts, signal[start:start+size*2])
        area, w, h = component(image, center)
        if shape == 21:
            tally['closures'] += area <= quiet
            tally['bilabials'] += 1
        else:
            tally['vowels'] += 1
            tally['vowels_open'] += area > quiet
            if area >= 180 and h > 4:
                vowels['spread' if shape == 6 else 'round'].append((area, w/h))
    return {'closed': [tally['closures'], tally['bilabials']],
            'vowels_open': [tally['vowels_open'], tally['vowels']],
            'matched_width_order': paired_vowels(vowels),
            'matched_candidates': [len(v) for v in vowels.values()]}


def frame_cost(rig, pcm):
    off, size = rig.pcm_window
    pad = bytes(size*2)
    signal = pad + pcm + pad
    costs = {'old': [], 'new': []}
    for n, index in enumerate(range(0, len(pcm)//2, FRAME)):
        start = len(pad) + 2 * index
        timestamp = index/RATE
        for method in (('old', 'new') if n % 2 else ('new', 'old')):
            before = perf_counter_ns()
            if method == 'old':
                rig.render_pose(rig.speech_pose(timestamp, *previous(signal[start:start+960])))
            else:
                left = start + 2*off
                rig.render_pcm(timestamp, signal[left:left+size*2])
            costs[method].append((perf_counter_ns()-before)/1e6)
    return {method: {'frames': len(t), 'p50_ms': statistics.median(t),
                     'p95_ms': sorted(t)[int(.95*len(t))], 'over_40ms': sum(x>40 for x in t)}
            for method, t in costs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rigs', type=Path, help='directory containing regular/model.inp and q/model.inp')
    parser.add_argument('audio', type=Path, help='source WAV or 24 kHz mono s16le .pcm')
    parser.add_argument('events', type=Path, help='generator CSV: viseme or phoneme_ipa,time_s,duration_s,value')
    parser.add_argument('--timing', action='store_true', help='time both renderers on every native frame')
    parser.add_argument('--phase-ms', type=int, choices=range(0, 40), default=0)
    args = parser.parse_args()
    pcm = (args.audio.read_bytes() if args.audio.suffix == '.pcm' else subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(args.audio), '-ar', str(RATE), '-ac', '1', '-f', 's16le', 'pipe:1']))
    labels = list(events(args.events))
    if not labels:
        parser.error('No labelled windows of 40 ms or longer')
    for rig_name, center in CENTERS.items():
        rig = InochiRenderer(args.rigs / rig_name)
        for method in ('old', 'new'):
            print(rig_name, method, json.dumps(measure(rig, center, pcm, labels, method, 24*args.phase_ms)), flush=True)
        if args.timing:
            print(rig_name, 'frame_cost', json.dumps(frame_cost(rig, pcm)), flush=True)
    _, size = rig.pcm_window
    times = []
    for start in range(0, len(pcm)-size*2, FRAME*2):
        frame = pcm[start:start+size*2]
        begin = perf_counter_ns()
        mouth_controls(frame)
        times.append((perf_counter_ns()-begin)/1e6)
    print('analysis_ms', {'frames': len(times), 'p50': statistics.median(times),
                          'p95': sorted(times)[int(len(times)*.95)]})


if __name__ == '__main__':
    main()
