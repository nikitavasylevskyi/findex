"""Lab 5: a shared Counter updated from several threads, without and with a fix.

``counter[t] += 1`` is three steps: read the value, add one, write it back.
Another thread can write in between, and then one of the two increments is lost.
Under the GIL the switch rarely lands exactly there; without the GIL the threads
really run at the same time and losses become routine.
"""

from __future__ import annotations

import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

THREADS = 8
TOKENS = ["async", "await", "yield", "lambda"] * 50_000  # 200 000 increments per thread


def shared_no_lock() -> Counter[str]:
    counter: Counter[str] = Counter()

    def work() -> None:
        for t in TOKENS:
            counter[t] += 1  # read-modify-write, not atomic

    with ThreadPoolExecutor(THREADS) as pool:
        for _ in range(THREADS):
            pool.submit(work)
    return counter


def shared_with_lock() -> Counter[str]:
    counter: Counter[str] = Counter()
    lock = threading.Lock()

    def work() -> None:
        for t in TOKENS:
            with lock:
                counter[t] += 1

    with ThreadPoolExecutor(THREADS) as pool:
        for _ in range(THREADS):
            pool.submit(work)
    return counter


def own_counters() -> Counter[str]:
    """The fix findex uses: nothing is shared, every worker returns its own result and they are merged once."""
    def work() -> Counter[str]:
        return Counter(TOKENS)

    total: Counter[str] = Counter()
    with ThreadPoolExecutor(THREADS) as pool:
        for part in pool.map(lambda _: work(), range(THREADS)):
            total.update(part)
    return total


if __name__ == "__main__":
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    expected = THREADS * len(TOKENS)
    print(f"Python {sys.version.split()[0]}, sys._is_gil_enabled() = {gil}, expected {expected:,} increments")
    for fn in (shared_no_lock, shared_with_lock, own_counters):
        got = sum(fn().values())
        print(f"  {fn.__name__:16} {got:>9,}  lost {expected - got:,}")
