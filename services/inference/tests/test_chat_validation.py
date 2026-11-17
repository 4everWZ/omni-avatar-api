"""Pinned API adapter must keep caller validation separate from server faults."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from services.inference.patches.patch_chat_prompt_tokens import patch_api_source

SOURCE = '''@router.post(
    "/v1/chat/completions",
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = Omnichat(raw_request)
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return generator
'''


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class ValidationError(Exception):
    pass


class ApiValidationTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, name, error):
        async def raise_error(*args, **kwargs):
            raise error
        async def body():
            return {'messages': []}
        handler = SimpleNamespace(create_chat_completion=raise_error, _create_chat_completion=raise_error)
        namespace = {'router': SimpleNamespace(post=lambda *a, **k: lambda fn: fn),
                     'Depends': lambda value: value, 'validate_json_request': object(),
                     'with_cancellation': lambda fn: fn, 'ChatCompletionRequest': object,
                     'Request': object, 'HTTPException': HTTPError,
                     '_validate_chat_completion_raw_body': lambda value: None,
                     'Omnichat': lambda request: handler,
                     'EngineGenerateError': type('GenerateError', (Exception,), {}),
                     'EngineDeadError': type('DeadError', (Exception,), {})}
        exec(compile(patch_api_source(SOURCE), '<patched-api>', 'exec'), namespace)
        module = ModuleType('vllm.exceptions')
        module.VLLMValidationError = ValidationError
        with patch.dict(sys.modules, {'vllm': ModuleType('vllm'), 'vllm.exceptions': module}):
            with self.assertRaises(HTTPError) as caught:
                await namespace[name](SimpleNamespace(truncate_prompt_tokens=None), SimpleNamespace(json=body))
        return caught.exception.status_code

    async def test_chat_validation_is_a_client_error(self):
        self.assertEqual(await self.invoke('create_chat_completion', ValidationError('context length exceeded')), 400)

    async def test_tokenizer_validation_is_a_client_error(self):
        self.assertEqual(await self.invoke('count_chat_prompt_tokens', ValidationError('invalid request')), 400)

    async def test_server_failure_keeps_server_status(self):
        self.assertEqual(await self.invoke('create_chat_completion', RuntimeError('server failed')), 500)


if __name__ == '__main__':
    unittest.main()
