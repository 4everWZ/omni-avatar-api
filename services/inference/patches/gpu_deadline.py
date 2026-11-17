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
    started = time.monotonic()
    while True:
        now = time.monotonic()
        try:
            clock = json.loads(Path(filename).read_text())
            updated, deadline = float(clock['updated']), float(clock['deadline'])
            if (not math.isfinite(updated) or not math.isfinite(deadline)
                    or not 0 <= now - updated < .5 or deadline - now >= .24):
                break
        except (OSError, ValueError, KeyError, TypeError):
            break
        remaining = .018 - (time.monotonic() - started)
        if remaining <= .000001:
            break
        time.sleep(min(.002, remaining))
    return time.monotonic() - started
