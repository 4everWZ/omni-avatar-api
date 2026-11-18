import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from aiohttp import web

from services.avatar.registered import RegisteredRigs


class RegisteredRigTest(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_tracks_actual_container(self):
        rigs = RegisteredRigs()
        factory = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {'root': root}
            model = root / 'model.inp'
            with self.assertRaises(web.HTTPServiceUnavailable):
                await rigs.describe('q', entry, factory)
            model.write_bytes(b'first')
            for _ in range(2):
                result = await rigs.describe('q', entry, factory)
                self.assertEqual(json.loads(result.text), {'id': 'q', 'driver': 'inochi2d', 'state': 'ready'})
            factory.assert_called_once_with(root)
            factory.return_value.close.assert_called_once_with()
            model.write_bytes(b'replacement')
            factory.side_effect = ValueError('unusable rig')
            with self.assertRaises(web.HTTPServiceUnavailable):
                await rigs.describe('q', entry, factory)
            factory.side_effect = None
            self.assertEqual((await rigs.describe('q', entry, factory)).status, 200)
            model.unlink()
            with self.assertRaises(web.HTTPServiceUnavailable):
                await rigs.describe('q', entry, factory)
