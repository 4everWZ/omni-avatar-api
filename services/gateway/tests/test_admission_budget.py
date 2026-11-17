import asyncio
import unittest
from services.gateway.admission import Admission, Busy, BudgetExceeded
from services.gateway.config import Config


class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.admission = Admission(Config(engine_url='http://engine', model='text', checkpoint='pin',
            tokens=('client',), admin_token='admin', batch_cap=4, token_budget=1000,
            queue_timeout=.15, realtime_enabled=False))

    async def test_four_slots_share_tokens_and_release_on_cancellation(self):
        entered, release = asyncio.Queue(), asyncio.Event()
        async def work():
            async with self.admission.hold('chat', tokens=250):
                await entered.put(True)
                await release.wait()
        tasks = [asyncio.create_task(work()) for _ in range(4)]
        try:
            for _ in tasks:
                await asyncio.wait_for(entered.get(), 1)
            self.assertEqual(self.admission.view()['reserved_tokens'], 1000)
            self.assertEqual(self.admission.active['chat'], 4)
            with self.assertRaises(Busy):
                async with self.admission.hold('chat', tokens=1):
                    self.fail('Fifth request entered')
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.admission.reserved_tokens, 0)
        self.assertEqual(self.admission.active['chat'], 0)

    async def test_long_request_waits_for_tokens_and_fifo_prevents_starvation(self):
        order = []
        release_long = asyncio.Event()
        async def long():
            async with self.admission.hold('chat', tokens=900):
                order.append('long')
                await release_long.wait()
        async def short():
            async with self.admission.hold('chat', tokens=50):
                order.append('short')
        async with self.admission.hold('chat', tokens=200):
            large = asyncio.create_task(long())
            await asyncio.sleep(.01)
            small = asyncio.create_task(short())
            await asyncio.sleep(.01)
            self.assertEqual(self.admission.queued, 2)
            self.assertEqual(order, [])
        await asyncio.gather(large_started := asyncio.create_task(self.wait_until(lambda: order)),
                             asyncio.sleep(.01))
        release_long.set()
        await asyncio.gather(large, small)
        self.assertEqual(order, ['long', 'short'])
        self.assertEqual(self.admission.reserved_tokens, 0)

    async def wait_until(self, condition):
        async with asyncio.timeout(1):
            while not condition():
                await asyncio.sleep(.001)

    async def test_impossible_request_and_multiple_choices_cannot_bypass_limits(self):
        for options in ({'tokens': 1001}, {'tokens': 0}, {'tokens': 1, 'slots': 5}):
            with self.assertRaises(BudgetExceeded):
                async with self.admission.hold('chat', **options):
                    self.fail('Impossible reservation entered')
        async with self.admission.hold('chat', tokens=400, slots=4):
            self.assertEqual(self.admission.active['chat'], 4)
            with self.assertRaises(Busy):
                async with self.admission.hold('chat', tokens=1):
                    self.fail('Sequence budget bypassed')

    async def test_cancelled_queue_head_wakes_the_next_waiter(self):
        entered = asyncio.Event()
        async def work(tokens):
            async with self.admission.hold('chat', tokens=tokens):
                entered.set()
        async with self.admission.hold('chat', tokens=900):
            head = asyncio.create_task(work(500))
            await asyncio.sleep(.01)
            next_task = asyncio.create_task(work(100))
            await asyncio.sleep(.01)
            head.cancel()
            await asyncio.gather(head, return_exceptions=True)
            await asyncio.wait_for(entered.wait(), 1)
            await next_task
        self.assertEqual(self.admission.queued, 0)
