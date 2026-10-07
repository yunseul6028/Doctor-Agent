import json
import logging
import time
from typing import Callable, Protocol

from doctor_agent.config import LLMConfig
from doctor_agent.llm.harmony import split_harmony

log = logging.getLogger("doctor_agent.llm")


class BillingError(RuntimeError):
    """Out of credits / billing disabled: retrying or continuing other cases is pointless."""


class LLMTimeout(RuntimeError):
    """The caller's deadline passed before a usable answer arrived."""


def _is_billing_error(e: Exception) -> bool:
    msg = str(e)
    return "402" in msg or "credits are depleted" in msg or "billing" in msg.lower() and "RESOURCE_EXHAUSTED" in msg


def _is_bad_request(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    return status in (400, 422) or type(e).__name__ in ("BadRequestError", "UnprocessableEntityError")


class LLMClient(Protocol):
    call_count: int

    def chat(self, messages: list[dict]) -> str: ...


# Servers (base_url, model) that rejected structured output. Server capability only, never case content, so sharing it
# across cases does not break case independence.
_STRUCTURED_REJECTED: set[tuple[str, str]] = set()
_EFFORT_ORDER = ["low", "medium", "high"]


def _field(obj: object, name: str) -> object:
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None) or {}
        val = extra.get(name) if isinstance(extra, dict) else None
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return val


def extract_text(resp: object, expect_json: bool = False) -> tuple[str, str, str]:
    """(text, reasoning, finish_reason) from a chat.completions response.

    Prefers `content` (with harmony markers stripped). Reasoning comes from `reasoning_content` / `reasoning` or from
    leaked analysis channels. When `expect_json` and the content has no JSON object but the reasoning does, the reasoning
    is appended inside <analysis>…</analysis> so the parser can fall back to the action JSON written there.
    """
    choice = (_field(resp, "choices") or [None])[0]
    msg = _field(choice, "message") if choice is not None else None
    finish = str(_field(choice, "finish_reason") or "") if choice is not None else ""
    content = _field(msg, "content") if msg is not None else None
    if isinstance(content, list):  # content parts
        content = "".join(str(_field(p, "text") or "") for p in content)
    final, leaked = split_harmony(content if isinstance(content, str) else "")
    reasoning = ""
    for key in ("reasoning_content", "reasoning"):
        r = _field(msg, key) if msg is not None else None
        if isinstance(r, str) and r.strip():
            reasoning = r
            break
    reasoning = "\n".join(x for x in (reasoning, leaked) if x).strip()
    text = final
    if expect_json and "{" not in final and "{" in reasoning:
        text = (final + "\n" if final else "") + f"<analysis>{reasoning}</analysis>"
    return text, reasoning, finish


class OpenAICompatClient:
    """Client for the doctor LLM (any OpenAI-compatible chat-completions endpoint; default Gemini Pro, see config.py).
    Also handles gpt-oss harmony output and reasoning returned in `reasoning_content` (see extract_text).

    Inference talks to this one configured model endpoint only. Never add other external API calls.

    `chat(messages)` keeps the simple protocol. Extra keyword options (used by the agent runtime):
      json_schema  – request structured output when cfg.structured_output is on (falls back if the server rejects it)
      expect_json  – when the answer lands only in the reasoning, hand it to the parser (see extract_text)
      reasoning_effort – override cfg.reasoning_effort for this call
      deadline     – absolute time (self.clock) after which no new attempt starts; per-request timeout is capped by it
    """

    supports_options = True

    def __init__(self, cfg: LLMConfig, client: object | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        if client is None:
            from openai import OpenAI

            # our own retry loop is deadline-aware; the SDK's hidden retries are not
            client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s, max_retries=0)
        self.client = client
        self.clock, self.sleep = clock, sleep
        self.call_count = 0  # successful completions

    def _structured_on(self) -> bool:
        return self.cfg.structured_output in ("json_schema", "guided_json") and \
            (self.cfg.base_url, self.cfg.model) not in _STRUCTURED_REJECTED

    def _request(self, messages: list[dict], max_tokens: int, effort: str | None, json_schema: dict | None,
                 timeout: float | None) -> object:
        kwargs: dict = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
                        "max_tokens": max_tokens}
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
        if timeout is not None:
            kwargs["timeout"] = timeout
        if json_schema is not None and self._structured_on():
            if self.cfg.structured_output == "guided_json":
                kwargs["extra_body"] = {"guided_json": json_schema}
            else:
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "action", "schema": json_schema, "strict": False}}
        return self.client.chat.completions.create(**kwargs)

    def chat(self, messages: list[dict], *, json_schema: dict | None = None, expect_json: bool = False,
             reasoning_effort: str | None = None, deadline: float | None = None) -> str:
        last_err: Exception | None = None
        effort = reasoning_effort or self.cfg.reasoning_effort
        max_tokens = self.cfg.max_tokens
        length_retried = False
        attempt = 0
        while attempt <= self.cfg.max_retries:
            timeout = None
            if deadline is not None:
                left = deadline - self.clock()
                if left <= 0.5:
                    raise LLMTimeout(f"deadline reached before LLM attempt {attempt + 1} (last error: {last_err})")
                timeout = min(self.cfg.timeout_s, left)
            try:
                resp = self._request(messages, max_tokens, effort, json_schema, timeout)
            except Exception as e:  # noqa: BLE001 — never kill the case
                if _is_billing_error(e):
                    raise BillingError(f"LLM billing error: {e}") from e
                if json_schema is not None and self._structured_on() and _is_bad_request(e):
                    # server does not support structured output: disable it for the rest of the run and retry at once
                    log.warning("structured output rejected by %s (%s); disabled for this run", self.cfg.base_url, e)
                    _STRUCTURED_REJECTED.add((self.cfg.base_url, self.cfg.model))
                    continue
                last_err = e
                attempt += 1
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                wait = 30 if rate_limited else 1.5 * attempt
                if attempt > self.cfg.max_retries:
                    break
                if deadline is not None and self.clock() + wait >= deadline - 1.0:
                    raise LLMTimeout(f"no time left to retry before the deadline (last error: {e})") from e
                self.sleep(wait)
                continue
            self.call_count += 1
            text, reasoning, finish = extract_text(resp, expect_json)
            answer = text.split("<analysis>", 1)[0]  # the final-channel part only (not the reasoning fallback)
            needs_more = not answer.strip() or (expect_json and "{" not in answer)
            if finish == "length" and needs_more and not length_retried:
                # reasoning ate the whole budget: one retry with more tokens and the lowest effort
                length_retried = True
                max_tokens = min(max(int(max_tokens * self.cfg.length_retry_factor), max_tokens + 1), self.cfg.max_tokens_cap)
                if effort and effort != "none":
                    effort = _EFFORT_ORDER[0]
                log.info("empty content with finish_reason=length; retrying with max_tokens=%d effort=%s", max_tokens, effort)
                continue
            if not text.strip() and reasoning:
                # last resort: the reasoning itself (the parser pulls the final action JSON out of it)
                text = f"<analysis>{reasoning}</analysis>"
            return text
        raise RuntimeError(f"LLM call failed: {last_err}")


class DummyLLM:
    """Scripted responses for smoke tests without an LLM."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages: list[dict]) -> str:
        self.call_count += 1
        if self.call_count >= 4:
            return json.dumps({"type": "DIAGNOSE", "content": "급성 충수염", "confidence": 0.9, "ddx": []})
        script = [
            {"type": "ASK", "content": "어디가 제일 아프고, 아픈 곳이 옮겨갔나요?"},
            {"type": "EXAM", "content": "복부 진찰"},
            {"type": "TEST", "content": "일반혈액검사(CBC)"},
        ]
        return json.dumps(script[self.call_count - 1] | {"confidence": 0.3, "ddx": []})
