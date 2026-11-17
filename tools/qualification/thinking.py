"""Check real default thinking, explicit overrides and caller-owned budgets."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from aiohttp import ClientSession, ClientTimeout
from .workstation import read_env


async def structured_checks(http, record):
    async def post(path, body):
        async with http.post(path, json=body) as response:
            assert response.status == 200, (path, response.status)
            data = await response.json()
            assert 'error' not in data
            return data
    def text(response):
        return ''.join(part.get('text', '') for item in response.get('output', [])
                       if item.get('type') == 'message' for part in item.get('content', [])
                       if part.get('type') == 'output_text')
    schema = {'type': 'json_schema', 'name': 'thinking_fixture', 'strict': True,
              'schema': {'type': 'object', 'properties': {'answer': {'type': 'integer', 'enum': [5]}},
                         'required': ['answer'], 'additionalProperties': False}}
    for prefix in ('', '/llm'):
        result = await post(prefix + '/v1/chat/completions', {
            'model': 'omni-default' if not prefix else 'llm-default',
            'messages': [{'role': 'user', 'content': 'Return answer=2+3 in the required JSON.'}],
            'max_tokens': 4096, 'temperature': 0,
            'response_format': {'type': 'json_schema', 'json_schema': {k: v for k, v in schema.items() if k != 'type'}}})
        assert json.loads(result['choices'][0]['message']['content']) == {'answer': 5}
        record({'name': 'default_thinking_chat_json_schema', 'prefix': prefix or 'omni'})
    result = await post('/llm/v1/responses', {'model': 'llm-default', 'input': 'Return answer=2+3 in JSON.',
        'max_output_tokens': 4096, 'temperature': 0, 'text': {'format': schema}})
    assert json.loads(text(result)) == {'answer': 5}
    record({'name': 'default_thinking_responses_json_schema'})
    tools = [{'type': 'function', 'name': 'lookup_status', 'description': 'Return a ticket status.',
              'parameters': {'type': 'object', 'properties': {'ticket': {'type': 'string'}},
                             'required': ['ticket'], 'additionalProperties': False}, 'strict': True}]
    seed = await post('/llm/v1/responses', {'model': 'llm-default', 'input': 'Use lookup_status for ticket T-57.',
        'max_output_tokens': 4096, 'temperature': 0, 'tools': tools,
        'tool_choice': {'type': 'function', 'name': 'lookup_status'}})
    call = next(item for item in seed['output'] if item['type'] == 'function_call')
    assert call['name'] == 'lookup_status' and json.loads(call['arguments']) == {'ticket': 'T-57'}
    follow = await post('/llm/v1/responses', {'model': 'llm-default', 'max_output_tokens': 4096, 'temperature': 0,
        'previous_response_id': seed['id'], 'tools': tools,
        'instructions': 'Summarize the returned status without another tool call.',
        'input': [{'type': 'function_call_output', 'call_id': call['call_id'], 'output': '{"status":"ok"}'}]})
    assert 'ok' in text(follow).lower()
    record({'name': 'default_thinking_function_call_owned_continuation', 'execution': 'literal_fixture_only'})


async def run(args):
    report = {'state': 'running', 'scope': 'default_chat_responses_thinking_and_explicit_budgets',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': []}
    def save():
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    async with ClientSession(base_url=args.gateway, timeout=ClientTimeout(total=300),
            headers={'Authorization': 'Bearer ' + read_env(args.env)['OMNI_GATEWAY_TOKEN']}) as http:
        async def case(name, path, *, options=None, answer='1776', prompt=None, enabled=None):
            responses = path.endswith('/responses')
            question = prompt or 'Calculate 37 times 48 carefully. Return only the final integer as your answer.'
            body = {'model': 'llm-default' if path.startswith('/llm') else 'omni-default',
                    'temperature': 0, 'stream': True}
            if responses:
                body.update(input=question, max_output_tokens=4096, store=False)
            else:
                body.update(messages=[{'role': 'user', 'content': question}], max_tokens=4096,
                            stream_options={'include_usage': True})
            body.update(options or {})
            content, reasoning_chars, usage = [], 0, None
            async with http.post(path, json=body) as response:
                assert response.status == 200, (name, response.status)
                async for line in response.content:
                    if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                        continue
                    event = json.loads(line[6:])
                    assert 'error' not in event, (name, 'stream_error')
                    if responses:
                        kind = event.get('type', '')
                        if kind == 'response.output_text.delta':
                            content.append(event['delta'])
                        elif 'reasoning' in kind and kind.endswith('.delta'):
                            reasoning_chars += len(event.get('delta', ''))
                        elif kind in ('response.completed', 'response.incomplete'):
                            usage = event['response'].get('usage')
                    else:
                        usage = event.get('usage') or usage
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            content.append(delta.get('content') or '')
                            reasoning_chars += len(delta.get('reasoning') or delta.get('reasoning_content') or '')
            text = ''.join(content).strip()
            report['checks'].append({'name': name, 'path': path, 'options': options or {},
                'reasoning_characters': reasoning_chars, 'final_answer': text[-256:], 'usage': usage})
            save()
            numbers = [value.replace(',', '') for value in re.findall(r'[0-9][0-9,]*', text)]
            assert numbers and numbers[-1] == answer, (name, 'incorrect_final_answer')
            assert usage, (name, 'missing_usage')
            if enabled is False:
                assert reasoning_chars == 0, (name, 'reasoning_not_disabled')
            elif enabled is True:
                assert reasoning_chars > 0, (name, 'default_thinking_not_observed')
            print(name, flush=True)
        try:
            for prefix in ('', '/llm'):
                await case(('omni' if not prefix else 'llm') + '_chat_default_thinking',
                           prefix + '/v1/chat/completions', enabled=True)
            await case('responses_default_thinking', '/llm/v1/responses', enabled=True)
            for prefix in ('', '/llm'):
                await case(('omni' if not prefix else 'llm') + '_chat_explicit_disable',
                           prefix + '/v1/chat/completions', enabled=False,
                           options={'chat_template_kwargs': {'enable_thinking': False}})
            await case('responses_effort_none_disables_thinking', '/llm/v1/responses', enabled=False,
                       options={'reasoning': {'effort': 'none'}})
            for path in ('/v1/chat/completions', '/llm/v1/chat/completions', '/llm/v1/responses'):
                await case('explicit_thinking_budget_' + path, path, answer='5',
                           prompt='What is two plus three? Return only the final integer.',
                           options={'thinking_token_budget': 16})
                async with http.post(path, json={'model': 'omni-default' if path == '/v1/chat/completions' else 'llm-default',
                        ('input' if path.endswith('/responses') else 'messages'):
                            'test' if path.endswith('/responses') else [{'role': 'user', 'content': 'test'}],
                        'thinking_token_budget': -2}) as response:
                    assert 400 <= response.status < 500, ('invalid_budget_not_rejected', path, response.status)
                    report['checks'].append({'name': 'negative_budget_rejected', 'path': path, 'status': response.status})
            def record(item):
                report['checks'].append(item)
                save()
            await structured_checks(http, record)
            report['state'] = 'passed'
        except BaseException as error:
            report.update(state='failed', failure=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--report', default='tmp/thinking.json')
    asyncio.run(run(parser.parse_args()))
