"""Generate a local Mandarin WAV and its same-pass IPA events (evaluation only).

Requires espeakng-loader. The service does not import this module.
"""
import argparse
import csv
import ctypes
from pathlib import Path
import wave

import espeakng_loader


class Event(ctypes.Structure):
    # espeak_EVENT from espeak-ng/speak_lib.h, using its eight-byte phoneme name union.
    _fields_ = [('type', ctypes.c_int), ('identifier', ctypes.c_uint), ('position', ctypes.c_int),
                ('length', ctypes.c_int), ('audio_ms', ctypes.c_int), ('sample', ctypes.c_int),
                ('user_data', ctypes.c_void_p), ('name', ctypes.c_char * 8)]


def generate(stem, text, voice):
    lib = ctypes.CDLL(str(espeakng_loader.get_library_path()))
    data = str(Path(espeakng_loader.get_data_path()).parent).encode()
    rate = lib.espeak_Initialize(2, 0, data, 0x8003)  # synchronous output, timed IPA, no exit on missing data
    if rate <= 0:
        raise RuntimeError('Could not initialize eSpeak NG')
    pcm, phones = bytearray(), []

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.POINTER(Event))
    def receive(audio, count, events):
        if count:
            pcm.extend(ctypes.string_at(audio, count * 2))
        index = 0
        while events[index].type:
            event = events[index]
            if event.type == 7:  # espeakEVENT_PHONEME
                phones.append((event.audio_ms, event.name.decode()))
            index += 1
        return 0

    try:
        if lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise ValueError(f'Unavailable eSpeak voice {voice}')
        lib.espeak_SetSynthCallback(receive)
        msg = text.encode() + b'\0'
        if lib.espeak_Synth(msg, len(msg), 0, 1, 0, 1, None, None) or not phones or not pcm:
            raise RuntimeError('Incomplete reference speech or timings')
    finally:
        lib.espeak_Terminate()

    with wave.open(str(stem.with_suffix('.wav')), 'wb') as out:
        out.setparams((1, 2, rate, 0, 'NONE', ''))
        out.writeframes(pcm)
    with stem.with_suffix('.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        for (start, name), (end, _) in zip(phones, phones[1:]):
            writer.writerow(('phoneme_ipa', start / 1000, (end-start) / 1000, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stem', type=Path)
    parser.add_argument('text')
    parser.add_argument('--voice', default='cmn')
    args = parser.parse_args()
    generate(args.stem, args.text, args.voice)
