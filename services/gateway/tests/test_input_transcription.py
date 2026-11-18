import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

from services.gateway.admission import BudgetExceeded, Busy
from services.gateway.input_transcription import InputTranscription, PREFIX


class SelectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.socket = AsyncMock()
        self.selection = InputTranscription(object(), self.socket, "fixture-client", "fixture-ticket")
        self.addAsyncCleanup(self.selection.stop)
        self.append = {"type":"input_audio_buffer.append", "format":"pcm16", "sample_rate_hz":16000,
                       "audio":base64.b64encode(b"\x00\x01"*16000).decode()}

    async def start(self):
        await self.selection.consume(json.dumps({"type":PREFIX+"start"}))
        return self.socket.send_json.await_args.args[0]["capture_id"]

    async def test_limit_clears_only_selection_and_does_not_truncate_a_transcript(self):
        await self.start()
        for _ in range(16):
            self.assertFalse(await self.selection.consume(json.dumps(self.append)))
        self.assertEqual(self.socket.send_json.await_args.args[0]["reason"], "audio_bounds")
        self.assertIsNone(self.selection.pcm)
        self.assertIsNone(self.selection.pending)

    async def test_disconnect_cancels_pending_inference_and_releases_audio(self):
        capture = await self.start()
        computing, aborted = asyncio.Event(), asyncio.Event()
        async def delayed(gateway, pcm):
            computing.set()
            try:
                await asyncio.Event().wait()
            finally:
                aborted.set()
        with patch("services.gateway.input_transcription.transcribe", delayed):
            await self.selection.consume(json.dumps(self.append))
            for _ in range(2):
                await self.selection.consume(json.dumps({"type":PREFIX+"finish", "capture_id":capture}))
            await asyncio.wait_for(computing.wait(), 1)
            await self.selection.stop()
        self.assertTrue(aborted.is_set())
        self.assertEqual(self.socket.send_json.await_count, 1)
        self.assertIsNone(self.selection.pcm)

    async def test_capacity_and_budget_errors_end_the_selected_attempt(self):
        for exception, reason in [(Busy(), "capacity"), (BudgetExceeded(), "budget")]:
            capture = await self.start()
            with patch("services.gateway.input_transcription.transcribe", side_effect=exception):
                await self.selection.consume(json.dumps(self.append))
                await self.selection.consume(json.dumps({"type":PREFIX+"finish", "capture_id":capture}))
                await self.selection.pending
            self.assertEqual(self.socket.send_json.await_args.args[0],
                             {"type":PREFIX+"failed", "capture_id":capture, "reason":reason})
