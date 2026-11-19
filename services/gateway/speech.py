"""Reference-conditioned literal speech and a blind readback on the same Omni engine."""
import asyncio
import base64
import binascii
import io
import json
import re
import wave

from aiohttp import ClientError, web

from .admission import BudgetExceeded, Busy
from .token_budget import ReservationRejected, reservation


def spoken_words(value):
    # Preserve lexical symbols and decimal points. ASR supplies its own sentence punctuation.
    # Keep word boundaries in Latin text and numeric separators in dates or decimals.
    return re.findall(r"[A-Za-z0-9]+(?:[.,:][0-9]+)*|[^\s，。！？、；：‘’“”《》,.!?;:\"「」『』…]", value)


def validate_wav(data, *, output=False):
    with wave.open(io.BytesIO(data)) as reader:
        channels, width, rate, frames, *_ = reader.getparams()
        if not (1 <= channels <= 2 and 1 <= width <= 4 and 8000 <= rate <= 192000 and 0 < frames <= rate * 90):
            raise ValueError("Invalid PCM WAV")
        if output and (channels, width, rate) != (1, 2, 24000):
            raise ValueError("Unexpected output format")
        pcm = reader.readframes(frames)
        if len(pcm) != frames * channels * width or not pcm.strip(b"\x80" if width == 1 else b"\x00"):
            raise ValueError("Empty or incomplete PCM")


def synthesis_request(model, line, reference):
    budget = len(line.encode()) + 16
    # MiniCPM-o conditions the Thinker on system audio as well as conditioning
    # Code2Wav on the reference waveform. Both inputs use the same recording.
    voice_context = [
        {"type": "text", "text": "模仿音频样本的音色并生成新的内容。"},
        {"type": "input_audio", "input_audio": {"format": "wav", "data": reference}},
        {"type": "text", "text": "请用这种声音风格来为用户提供帮助。直接作答，不要有冗余内容。"},
    ]
    return {
        "model": model, "modalities": ["text", "audio"], "audio": {"format": "wav", "voice": "assistant"},
        "messages": [{"role": "system", "content": voice_context},
                     {"role": "user", "content": "请朗读以下内容。 " + line}],
        "ref_audio": "data:audio/wav;base64," + reference,
        "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
        "max_tokens": budget,
        # The workstation Talker's default EOS floor is 50 codec frames. It can
        # repeat a finished short utterance while waiting for EOS to be available.
        "sampling_params_list": [
            {"temperature": 0, "seed": 42, "max_tokens": budget},
            {"temperature": 0.8, "top_p": 0.85, "top_k": 25, "repetition_penalty": 1.05,
             "min_tokens": 1, "max_tokens": min(2048, 64 + len(line) * 24), "seed": 42, "detokenize": False},
        ],
    }


def transcription_request(model, audio):
    return {
        "model": model, "modalities": ["text"], "temperature": 0, "max_tokens": 1536,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"json": {"type": "object", "properties": {"transcript": {"type": "string"}},
                                        "required": ["transcript"], "additionalProperties": False}},
        "messages": [{"role": "system", "content": "逐字转写这段录音，只输出听到的原话。"},
                     {"role": "user", "content": [{"type": "text", "text": "听录音，逐字转写到 transcript。"},
                                                  {"type": "input_audio", "input_audio": {"format": "wav", "data": audio}}]}],
    }


async def complete(gateway, payload):
    budget = await reservation(gateway, "chat", payload)
    async with gateway.admission.hold("chat", **budget):
        async with gateway.engine.post("/v1/chat/completions", json=payload, allow_redirects=False) as upstream:
            if upstream.status == 429:
                raise Busy()
            upstream.raise_for_status()
            raw = bytearray()
            async for chunk in upstream.content.iter_chunked(65536):
                raw.extend(chunk)
                if len(raw) > 8_000_000:
                    raise ValueError("Response too large")
            choices = json.loads(raw)["choices"]
            if not choices or any(c["finish_reason"] != "stop" for c in choices):
                raise ValueError("Incomplete response")
            return [c["message"] for c in choices]


def failure(status, code, message, *, headers=None):
    return web.json_response({"error": {"code": code, "message": message}}, status=status, headers=headers)


async def speech(request):
    gateway = request.app["gateway"]
    try:
        async with asyncio.timeout(gateway.config.request_timeout):
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"input", "model", "reference_audio", "response_format", "voice"}:
                raise ValueError()
            line, ref = body["input"], body["reference_audio"]
            if (body["model"] != gateway.config.model_alias or body.get("response_format", "wav") != "wav"
                    or body.get("voice", "reference") != "reference"
                    or not isinstance(line, str) or not 1 <= len(line) <= 4096 or not spoken_words(line)
                    or not isinstance(ref, dict) or set(ref) != {"data", "format"} or ref["format"] != "wav"
                    or not isinstance(ref["data"], str) or len(ref["data"]) > 10_666_668):
                raise ValueError()
            validate_wav(base64.b64decode(ref["data"], validate=True))
    except (ValueError, KeyError, TypeError, EOFError, wave.Error, binascii.Error):
        return failure(400, "invalid_speech_request", "Supply the configured model, literal input and inline reference WAV.")
    except TimeoutError:
        return failure(408, "request_timeout", "Speech input timed out.")

    try:
        async with asyncio.timeout(gateway.config.request_timeout):
            messages = await complete(gateway, synthesis_request(gateway.config.model, line.strip(), ref["data"]))
            words = [m["content"] for m in messages if m.get("content")]
            clips = [m["audio"]["data"] for m in messages if m.get("audio")]
            if len(words) != 1 or spoken_words(words[0]) != spoken_words(line) or len(clips) != 1:
                raise ValueError("Text mismatch")
            data = base64.b64decode(clips[0], validate=True)
            validate_wav(data, output=True)
            heard, = await complete(gateway, transcription_request(gateway.config.model, clips[0]))
            transcript = json.loads(heard["content"])["transcript"]
            if not isinstance(transcript, str) or spoken_words(transcript) != spoken_words(line):
                raise ValueError("Spoken words mismatch")
            return web.Response(body=data, content_type="audio/wav", headers={
                "X-Omni-Transcript": base64.b64encode(transcript.encode()).decode("ascii"), "Cache-Control": "no-store",
            })
    except Busy:
        return failure(429, "gateway_busy", "Speech inference is at capacity.", headers={"Retry-After": "1"})
    except BudgetExceeded:
        return failure(400, "request_budget_exceeded", "Speech exceeds the admission budget.")
    except ReservationRejected as error:
        return error.response()
    except TimeoutError:
        return failure(504, "engine_timeout", "Speech exceeded the request deadline.")
    except ClientError:
        return failure(502, "engine_unavailable", "The inference engine could not serve the request.")
    except (ValueError, KeyError, TypeError, AttributeError, EOFError, wave.Error, binascii.Error):
        return failure(422, "speech_unverified", "The engine did not produce a complete reading matching the input.")
