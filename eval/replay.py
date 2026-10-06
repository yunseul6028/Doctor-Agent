"""LLM record/replay cache for local experiments (eval only; the agent package never imports this).

Same approach as eval/usage.py: `OpenAICompatClient.client` (the OpenAI SDK object) is swapped for a proxy whose
`chat.completions.create(**kwargs)` first looks the request up in an on-disk cache. src/ is untouched.

Key = sha256 over (endpoint host+path, model, messages, temperature, max_tokens, reasoning_effort, response_format,
extra_body, any other request kwarg except `timeout`, --cache-salt, --cache-sample-idx). Any change to a prompt, a
patient answer or a sampling parameter is a miss; code changes that do not change the request replay the stored answer.

Storage: one append-only JSONL file per role under eval/cache/ (git-ignored), e.g. eval/cache/patient.jsonl:
  {"v": 1, "key": "<sha256>", "role": "patient", "model": "...", "sample_idx": 0, "salt": "", "at": "...",
   "response": {"content": ..., "reasoning": ..., "finish_reason": ..., "usage": {...}}}
Appends hold a process lock plus an fcntl file lock (safe for --workers N threads and parallel processes); a torn
last line from a crash is skipped on load. Duplicate keys are harmless (the last entry wins).

Modes:
  off     no wrapping
  record  always call the API and store the answer (refreshes the entry: the last stored entry wins)
  replay  answer from the cache only; a miss raises CacheMiss (a BaseException, so no retry/fallback layer in the
          client, policy or patient simulator can swallow it) and run_local aborts the batch -> no silent API calls
  auto    replay on hit, else call + record

Temperature > 0: a replayed run is deterministic (it reuses the first sampled answer). To keep several samples per
request use --cache-sample-idx k (k = 0, 1, 2 ... are separate entries) or a new --cache-salt for fresh answers.

Order with eval/usage.py: the cache is attached *after* the meter, so it wraps the metered SDK; hits never reach the
meter and `usage` keeps meaning "tokens actually billed". Tokens avoided are in the cache stats (`saved_*`).
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

try:
    import fcntl
except ImportError:  # Windows: process lock only
    fcntl = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "eval/cache"
MODES = ("off", "record", "replay", "auto")
ROLES = ("doctor", "patient", "judge")
FORMAT_VERSION = 1
_KEY_EXCLUDE = {"timeout"}  # per-call deadlines must not change the key
STAT_FIELDS = ("hits", "misses", "recorded", "calls", "saved_prompt_tokens", "saved_completion_tokens",
               "saved_reasoning_tokens", "saved_total_tokens")


class CacheMiss(BaseException):
    """Replay mode found no entry. BaseException on purpose: OpenAICompatClient retries every Exception, the policy
    and the patient simulator fall back on errors; a replay miss must stop the batch instead."""


def _get(obj: object, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            val = extra.get(name)
    return val


def _jsonable(x: object) -> object:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def endpoint_of(llm: object) -> str:
    base = str(getattr(getattr(llm, "cfg", None), "base_url", "") or "")
    u = urlparse(base)
    return (u.netloc + u.path.rstrip("/")) if u.netloc else base


def request_key(kwargs: dict, *, endpoint: str = "", salt: str = "", sample_idx: int = 0) -> str:
    req = {k: _jsonable(v) for k, v in sorted(kwargs.items()) if k not in _KEY_EXCLUDE}
    blob = json.dumps({"endpoint": endpoint, "req": req, "salt": salt, "sample_idx": int(sample_idx)},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def serialize_response(resp: object) -> dict:
    choice = (_get(resp, "choices") or [None])[0]
    msg = _get(choice, "message")
    content = _get(msg, "content")
    if isinstance(content, list):
        content = "".join(str(_get(p, "text") or "") for p in content)
    reasoning = None
    for k in ("reasoning_content", "reasoning"):
        r = _get(msg, k)
        if isinstance(r, str) and r:
            reasoning = r
            break
    u = _get(resp, "usage")
    usage = None
    if u is not None:
        usage = {"prompt_tokens": int(_get(u, "prompt_tokens") or 0),
                 "completion_tokens": int(_get(u, "completion_tokens") or 0),
                 "total_tokens": int(_get(u, "total_tokens") or 0),
                 "completion_tokens_details": {
                     "reasoning_tokens": int(_get(_get(u, "completion_tokens_details"), "reasoning_tokens") or 0)}}
    return {"content": content, "reasoning": reasoning, "finish_reason": _get(choice, "finish_reason"),
            "usage": usage}


def build_response(d: dict) -> SimpleNamespace:
    """A stored entry -> an object with the SDK response shape used by extract_text() and eval/usage.py."""
    msg = SimpleNamespace(role="assistant", content=d.get("content"), reasoning_content=d.get("reasoning"))
    u = d.get("usage")
    usage = None
    if u is not None:
        usage = SimpleNamespace(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                                total_tokens=u.get("total_tokens"),
                                completion_tokens_details=SimpleNamespace(
                                    reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens")))
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=msg, finish_reason=d.get("finish_reason"))],
                           usage=usage, cached=True)


class LLMCache:
    """One role's cache file + stats. Thread-safe; share one instance across worker threads."""

    def __init__(self, role: str, mode: str = "auto", directory: Path | str = DEFAULT_DIR, salt: str = "",
                 sample_idx: int = 0) -> None:
        if mode not in MODES:
            raise ValueError(f"cache mode {mode!r} not in {MODES}")
        self.role, self.mode, self.salt, self.sample_idx = role, mode, salt, int(sample_idx)
        self.path = Path(directory) / f"{role}.jsonl"
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._stats = dict.fromkeys(STAT_FIELDS, 0)
        self.miss_keys: list[str] = []
        if mode != "off":
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn line from an interrupted write
                if isinstance(e, dict) and e.get("key") and "response" in e:
                    self._index[e["key"]] = e["response"]

    def __len__(self) -> int:
        return len(self._index)

    def key(self, kwargs: dict, endpoint: str = "") -> str:
        return request_key(kwargs, endpoint=endpoint, salt=self.salt, sample_idx=self.sample_idx)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._index.get(key)

    def put(self, key: str, response: dict, model: str | None = None) -> None:
        entry = {"v": FORMAT_VERSION, "key": key, "role": self.role, "model": model, "sample_idx": self.sample_idx,
                 "salt": self.salt, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": response}
        data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        view = view[n:]
                finally:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            self._index[key] = response
            self._stats["recorded"] += 1

    def _count(self, field: str, n: int = 1) -> None:
        with self._lock:
            self._stats[field] += n

    def _hit(self, response: dict) -> None:
        u = response.get("usage") or {}
        p, c = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        with self._lock:
            self._stats["hits"] += 1
            self._stats["saved_prompt_tokens"] += p
            self._stats["saved_completion_tokens"] += c
            self._stats["saved_reasoning_tokens"] += int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            self._stats["saved_total_tokens"] += int(u.get("total_tokens") or (p + c))

    def create(self, inner_create, kwargs: dict, endpoint: str = ""):
        """The cached `chat.completions.create`."""
        if self.mode == "off":
            return inner_create(**kwargs)
        key = self.key(kwargs, endpoint)
        if self.mode in ("replay", "auto"):
            hit = self.get(key)
            if hit is not None:
                self._hit(hit)
                return build_response(hit)
            self._count("misses")
            if self.mode == "replay":
                with self._lock:
                    self.miss_keys.append(key)
                raise CacheMiss(f"llm cache miss (role={self.role}, key={key[:12]}, file={self.path}); "
                                "replay mode makes no API calls — run with --llm-cache auto/record first")
        resp = inner_create(**kwargs)
        self._count("calls")
        self.put(key, serialize_response(resp), model=kwargs.get("model"))
        return resp

    def stats(self) -> dict:
        with self._lock:
            s = dict(self._stats)
        looked = s["hits"] + s["misses"]
        s["hit_rate"] = round(s["hits"] / looked, 3) if looked else None
        return {"mode": self.mode, "entries": len(self), **s}


class _Completions:
    def __init__(self, inner: object, cache: LLMCache, endpoint: str) -> None:
        self._inner, self._cache, self._endpoint = inner, cache, endpoint

    def create(self, *args, **kwargs):
        if args:  # the SDK is keyword-only; never cache something we cannot key
            return self._inner.create(*args, **kwargs)
        return self._cache.create(self._inner.create, kwargs, self._endpoint)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: object, cache: LLMCache, endpoint: str) -> None:
        self._inner = inner
        self.completions = _Completions(inner.completions, cache, endpoint)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CachedSDK:
    """Proxy for an OpenAI SDK client: `.chat.completions.create` goes through the cache, the rest is forwarded."""

    def __init__(self, inner: object, cache: LLMCache, endpoint: str = "") -> None:
        self._inner = inner
        self.chat = _Chat(inner.chat, cache, endpoint)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def attach(llm: object, cache: LLMCache | None) -> bool:
    """Route an OpenAICompatClient-like object through `cache` in place. False when there is nothing to wrap."""
    if cache is None or cache.mode == "off":
        return False
    inner = getattr(llm, "client", None)
    if inner is None or not hasattr(inner, "chat"):
        return False
    llm.client = CachedSDK(inner, cache, endpoint_of(llm))
    return True


def make_caches(mode: str, *, cache_doctor: bool = False, doctor_mode: str | None = None,
                directory: Path | str = DEFAULT_DIR, salt: str = "", sample_idx: int = 0) -> dict[str, LLMCache | None]:
    """Per-role caches: patient/judge use `mode`; the doctor uses `doctor_mode` (default: `mode` if cache_doctor
    else off)."""
    dm = doctor_mode or (mode if cache_doctor else "off")
    out: dict[str, LLMCache | None] = {}
    for role in ROLES:
        m = dm if role == "doctor" else mode
        out[role] = LLMCache(role, m, directory, salt, sample_idx) if m != "off" else None
    return out


def summary(caches: dict[str, LLMCache | None], *, salt: str = "", sample_idx: int = 0,
            price_in: float | None = None, price_out: float | None = None) -> dict | None:
    """Result-JSON block. `saved_krw` (doctor only, when prices per 1M tokens are known): credits not spent."""
    roles = {r: c.stats() for r, c in caches.items() if c is not None}
    if not roles:
        return None
    out: dict = {"salt": salt, "sample_idx": sample_idx, "roles": roles}
    d = roles.get("doctor")
    if d and price_in is not None and price_out is not None:
        out["saved_krw_doctor"] = round(d["saved_prompt_tokens"] / 1e6 * price_in
                                        + d["saved_completion_tokens"] / 1e6 * price_out, 1)
    return out


def format_summary(block: dict | None) -> str:
    if not block:
        return "llm cache: off"
    parts = []
    for role, s in block["roles"].items():
        parts.append(f"{role}[{s['mode']}] hit={s['hits']} miss={s['misses']} rec={s['recorded']} "
                     f"saved={s['saved_prompt_tokens']:,} in/{s['saved_completion_tokens']:,} out")
    extra = f"  (~{block['saved_krw_doctor']:.0f} KRW doctor credits saved)" if "saved_krw_doctor" in block else ""
    return "llm cache: " + "  ".join(parts) + extra
