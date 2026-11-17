"""Ask the model's renderer for an exact, continuation-aware reservation."""
from aiohttp import web


class ReservationRejected(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body

    def response(self):
        return web.Response(status=self.status, body=self.body, content_type='application/json')


async def reservation(gateway, kind, payload):
    if not gateway.config.token_budget:
        return {}
    async with gateway.engine.post('/internal/admission/' + kind, json=payload,
                                   allow_redirects=False) as response:
        if response.status != 200:
            raise ReservationRejected(response.status, await response.read())
        counted = await response.json()
    tokens, slots = counted['reserved_tokens'], counted['sequences']
    if type(tokens) is not int or tokens < 1 or type(slots) is not int or slots < 1:
        raise ValueError('Invalid engine reservation')
    return {'tokens': tokens, 'slots': slots}
