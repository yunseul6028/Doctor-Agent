"""Token-usage metering for evaluation runs (eval only; the submitted code never imports this).

`OpenAICompatClient` keeps the OpenAI SDK object in `.client` and calls `client.chat.completions.create(...)`. We swap
that attribute for a thin proxy that forwards every call and adds `resp.usage` to a thread-safe meter. No request is
changed, so src/ stays untouched and old commits (v5 baseline worktrees) work the same way.
"""
import threading

FIELDS = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "no_usage")


def _get(obj: object, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t = dict.fromkeys(FIELDS, 0)

    def add(self, resp: object) -> None:
        usage = _get(resp, "usage")
        with self._lock:
            self._t["calls"] += 1
            if usage is None:
                self._t["no_usage"] += 1  # server did not report usage for this response
                return
            p = _get(usage, "prompt_tokens") or 0
            c = _get(usage, "completion_tokens") or 0
            r = _get(_get(usage, "completion_tokens_details"), "reasoning_tokens") or 0
            self._t["prompt_tokens"] += int(p)
            self._t["completion_tokens"] += int(c)
            self._t["reasoning_tokens"] += int(r)
            self._t["total_tokens"] += int(_get(usage, "total_tokens") or (p + c))

    def merge(self, other: "UsageMeter | dict | None") -> None:
        d = other.totals() if isinstance(other, UsageMeter) else (other or {})
        with self._lock:
            for k in FIELDS:
                self._t[k] += int(d.get(k) or 0)

    def totals(self) -> dict:
        with self._lock:
            return dict(self._t)


class _Completions:
    def __init__(self, inner: object, meter: UsageMeter) -> None:
        self._inner, self._meter = inner, meter

    def create(self, *args, **kwargs):
        resp = self._inner.create(*args, **kwargs)
        self._meter.add(resp)
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: object, meter: UsageMeter) -> None:
        self._inner = inner
        self.completions = _Completions(inner.completions, meter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MeteredSDK:
    """Proxy for an OpenAI SDK client: `.chat.completions.create` is metered, everything else is forwarded."""

    def __init__(self, inner: object, meter: UsageMeter) -> None:
        self._inner = inner
        self.chat = _Chat(inner.chat, meter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def attach(llm: object, meter: UsageMeter) -> bool:
    """Meter an OpenAICompatClient-like object in place. False when it has no SDK client (e.g. DummyLLM)."""
    inner = getattr(llm, "client", None)
    if inner is None or not hasattr(inner, "chat"):
        return False
    llm.client = MeteredSDK(inner, meter)
    return True
