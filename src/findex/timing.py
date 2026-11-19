"""Small decorator that reports how long a call took."""

from __future__ import annotations

import functools
import sys
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

ENABLED = True


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if ENABLED:
                ms = (time.perf_counter() - t0) * 1000
                print(f"[timed] {fn.__qualname__}: {ms:.1f} ms", file=sys.stderr)
    return wrapper
