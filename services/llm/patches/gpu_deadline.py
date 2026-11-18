"""Bounded cooperation with a live renderer; inactive/stale clocks never wait."""
import json
import math
import os
from pathlib import Path
import time


def yield_to_renderer():
    filename = os.environ.get('OMNI_GPU_RENDER_CLOCK')
    if not filename:
        return 0.
    path = Path(filename)
    started = time.monotonic()
    while True:
        now = time.monotonic()
        paths = path.parent.glob(path.name) if '*' in path.name else (path,)
        if not any(_urgent(clock, now) for clock in paths):
            break
        remaining = .018 - (time.monotonic() - started)
        if remaining <= .000001:
            break
        time.sleep(min(.002, remaining))
    return time.monotonic() - started


def _urgent(path, now):
    try:
        clock = json.loads(path.read_text())
        updated, deadline = float(clock['updated']), float(clock['deadline'])
        return (math.isfinite(updated) and math.isfinite(deadline)
                and 0 <= now - updated < .5 and deadline - now < .24)
    except (OSError, ValueError, KeyError, TypeError):
        return False
