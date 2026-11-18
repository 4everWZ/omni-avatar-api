"""Short-time spectral cues for the two continuous mouth controls of an INP rig.

The input is mono s16le at 24 kHz. Analysis uses no transcript, synthesis
events, persistent speaker state or network model. All windows are drawn
from audio already present in the playback timeline.
"""

import math
import struct

HOP = 480  # 20 ms
CONTEXT = 3 * HOP  # local level on either side of the analysis window
ONSET_HISTORY = 11 * HOP  # 200 ms of history before a 20 ms onset margin
_PACKET = struct.Struct('<480h')
_MILLISECOND = struct.Struct('<24h')
# Decimate to 12 kHz. The paired-sample average attenuates upper-band energy.
_SIZE = HOP // 2
_TAPER = tuple(.5 - .5 * math.cos(2 * math.pi * i / _SIZE) for i in range(_SIZE))


def _basis(bin_index, fn):
    return tuple(fn(2 * math.pi * bin_index * i / _SIZE) for i in range(1, _SIZE // 2))


# 50 Hz analysis bins up to 4 kHz; keep the quadrature filters precomputed.
_FILTERS = tuple((_basis(k, math.cos), _basis(k, math.sin), (-1)**k) for k in range(1, 81))


def _limit(value, lo=0., hi=1.):
    return max(lo, min(hi, value))


def _isolated_weak_onset(pcm):
    # Retain short consonants inside speech. Only inspect a possible click
    # after ten consecutive quiet packets before the onset margin.
    margin = (ONSET_HISTORY - HOP) * 2
    for offset in range(0, margin, _PACKET.size):
        block = _PACKET.unpack_from(pcm, offset)
        if math.sumprod(block, block) - sum(block)**2 / HOP >= HOP * (.003 * 32768)**2:
            return False
    # In a 60 ms neighbourhood of the current packet, require some energy
    # outside the transient. The next 20 ms is already in the renderer's
    # normal context; neither the track deadline nor the audio moves.
    energies = []
    for offset in range(margin, (ONSET_HISTORY + 2 * HOP) * 2, _MILLISECOND.size):
        block = _MILLISECOND.unpack_from(pcm, offset)
        energies.append(math.sumprod(block, block))
    total = sum(energies)
    # A six-ms pulse may straddle seven one-ms bins.
    peak = window = sum(energies[:7])
    for i in range(7, len(energies)):
        window += energies[i] - energies[i - 7]
        peak = max(peak, window)
    return peak >= .9 * total


def mouth_controls(pcm):
    """Return jaw opening (0..1) and spread (-1 rounded to +1 wide)."""
    if len(pcm) == _PACKET.size:
        center = 0
    elif len(pcm) == 2 * (2 * CONTEXT + HOP):
        center = CONTEXT * 2
    elif len(pcm) == 2 * (ONSET_HISTORY + HOP + CONTEXT):
        center = ONSET_HISTORY * 2
    else:
        raise ValueError('Expected a 20 ms PCM packet or its timeline context')
    samples = _PACKET.unpack_from(pcm, center)
    mean = sum(samples) / HOP
    power = max(0., math.sumprod(samples, samples) - HOP * mean * mean)
    rms = math.sqrt(power / HOP) / 32768
    if rms < .003:
        return 0., 0.
    if rms < .012 and center == ONSET_HISTORY * 2 and _isolated_weak_onset(pcm):
        return 0., 0.
    peak = power
    gain_start = max(0, center - CONTEXT * 2)
    for offset in range(gain_start, len(pcm), _PACKET.size):
        if offset != center:
            nearby = _PACKET.unpack_from(pcm, offset)
            peak = max(peak, math.sumprod(nearby, nearby) - sum(nearby)**2 / HOP)
    relative = math.sqrt(power / peak)
    # Attenuate DC before estimating resonances.
    signal = [w * ((a + b) * .5 - mean) for w, a, b in zip(_TAPER, samples[::2], samples[1::2])]
    # Fold a real DFT about the window midpoint: cosine is even, sine is
    # odd, and the Hann window's first sample is zero.
    half = _SIZE // 2
    left, right = signal[1:half], signal[:half:-1]
    even = tuple(a+b for a, b in zip(left, right))
    odd = tuple(a-b for a, b in zip(left, right))
    spectrum = [(math.sumprod(even, c) + sign*signal[half])**2 + math.sumprod(odd, s)**2
                for c, s, sign in _FILTERS]
    total = sum(spectrum) or 1.
    low = sum(spectrum[:10]) / total             # 50..500 Hz, voicing and nasal murmur
    middle = sum(spectrum[10:32]) / total        # 550..1600 Hz, back-vowel resonances
    front = sum(spectrum[36:70]) / total         # 1850..3500 Hz, front-vowel resonances
    # A nasal murmur can retain substantial energy with sealed lips. Some
    # speakers concentrate it below 500 Hz; others have a second nasal pole
    # above a spectral notch. A sustained quiet vowel can also be bass-heavy,
    # so require a local level dip before treating either cue as closure.
    oral_low = sum(spectrum[10:17]) / total       # 550..850 Hz
    oral_high = middle - oral_low                # 900..1600 Hz
    murmur = low > .95 and front < .0015 and middle < .035
    notch = low > .65 and oral_low < oral_high * .10 and front < oral_high * .15
    if relative < .15 or (murmur and relative < .85) or (notch and relative < .55):
        return 0., 0.
    # The prominence of the upper resonance relative to the lower oral
    # band is less sensitive to pitch and microphone gain than brightness.
    form = _limit(.35 + math.log((front + .0003) / (middle + .0003)) * .55, -1., 1.)
    opening = .85 * math.sqrt(_limit((relative - .15) / .85))
    # Quiet consonants and gaps should not expand to a full mouth in isolation.
    opening *= min(1., rms / .025)
    # Distribute the aperture over width and height: spreading stretches
    # the lip mesh horizontally, rounding draws it in.
    return _limit(opening / (1 + .23 * form)), form
