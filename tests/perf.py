"""Wall-clock assertions that stay meaningful on a loaded machine.

Every latency test states its *strict* budget (the documented number, e.g. "< 20 ms per call") and checks it through
`assert_fast`. Two things make that robust without hiding real regressions:

1. Best of N with early exit: the timed block is re-run (up to `tries` times) only while it is over budget, and the
   test passes as soon as one run is within budget. A genuinely slow implementation is slow on every run and still
   fails; a run that lost the CPU to another process is simply retried. Fast code costs a single run.
2. Load-scaled slack: unless PERF_STRICT=1, the strict budget is multiplied by `perf_factor()`, which grows with the
   1-minute load average per CPU (bounded to [MIN_FACTOR, MAX_FACTOR]). On an idle machine the slack is 1.5x; at
   load 60 on 10 cores (parallel agents running suites) it is capped at 8x.

PERF_STRICT=1 python -m pytest -m perf   # strict budgets, factor 1.0 (use on an idle machine before a release)
python -m pytest -m "not perf"           # skip all wall-clock tests
"""
from __future__ import annotations

import os
import time
from typing import Callable

MIN_FACTOR = 1.5
MAX_FACTOR = 8.0


def strict() -> bool:
    return os.environ.get("PERF_STRICT", "").strip() == "1"


def load_per_cpu() -> float:
    try:
        return os.getloadavg()[0] / (os.cpu_count() or 1)
    except (OSError, AttributeError):  # not available on every platform
        return 0.0


def perf_factor() -> float:
    """1.0 under PERF_STRICT=1; otherwise 1 + 1.5 * (load per CPU), clamped to [MIN_FACTOR, MAX_FACTOR]."""
    if strict():
        return 1.0
    return min(MAX_FACTOR, max(MIN_FACTOR, 1.0 + 1.5 * load_per_cpu()))


def limit(strict_budget: float) -> float:
    return strict_budget * perf_factor()


def assert_fast(fn: Callable[[], object], budget_s: float, *, per: int = 1, tries: int = 5, what: str = "") -> float:
    """Run `fn` until one run takes <= limit(budget_s) * per seconds (at most `tries` runs); return the best
    per-unit time in seconds. `per` divides the measured time (e.g. fn loops 10 calls -> per=10, budget per call)."""
    cap = limit(budget_s)
    times = []
    for _ in range(max(1, tries)):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) / per)
        if times[-1] <= cap:
            return times[-1]
    raise AssertionError(
        f"{what or getattr(fn, '__name__', 'block')}: best {min(times) * 1000:.1f} ms > limit {cap * 1000:.1f} ms "
        f"(strict budget {budget_s * 1000:.1f} ms x factor {perf_factor():.2f}, load/cpu {load_per_cpu():.2f}); "
        f"runs ms={[round(t * 1000, 1) for t in times]}")
