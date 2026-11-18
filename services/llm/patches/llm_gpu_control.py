"""Authenticated, serialized subset of the pinned engine's sleep controls."""
import asyncio
import logging
import os
import secrets

from fastapi import HTTPException, Request


def attach_gpu_control(app):
    token = os.environ.get('LLM_GPU_CONTROL_TOKEN')
    if not token:
        return
    lock = asyncio.Lock()
    phase = 'awake'

    async def control(request: Request, action: str):
        nonlocal phase
        if not secrets.compare_digest(request.headers.get('X-GPU-Control', '').encode(), token.encode()):
            raise HTTPException(403)
        async with lock:
            engine = request.app.state.engine_client
            try:
                if action == 'state':
                    return {'phase': phase}
                if action == 'wake' and phase in ('kv_released', 'weights_offloaded'):
                    tags = ['kv_cache'] if phase == 'kv_released' else None
                    if not await engine.wake_up(tags):
                        raise RuntimeError('Engine did not completely wake')
                    phase = 'awake'
                elif action == 'wake' and phase == 'awake':
                    return {'phase': phase}
                elif action == 'release_kv' and phase == 'awake':
                    await engine.sleep(0, 'wait')
                    await engine.release_kv_cache_memory()
                    phase = 'kv_released'
                elif action == 'offload' and phase == 'awake':
                    await engine.sleep(1, 'wait')
                    phase = 'weights_offloaded'
                else:
                    raise HTTPException(409, 'Invalid GPU control transition')
                return {'phase': phase}
            except HTTPException:
                raise
            except (Exception, asyncio.CancelledError):
                phase = 'unknown'
                logging.exception('LLM GPU control failed during %s', action)
                raise

    app.add_api_route('/internal/gpu-control/{action}', control, methods=['POST'])
