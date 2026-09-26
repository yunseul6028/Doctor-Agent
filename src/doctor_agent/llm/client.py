import json
import time
from typing import Protocol

from doctor_agent.config import LLMConfig


class LLMClient(Protocol):
    call_count: int

    def chat(self, messages: list[dict]) -> str: ...


class OpenAICompatClient:
    """Client for the fixed LLM gpt-oss-20b (OpenAI-compatible chat completions).

    The only endpoint allowed during inference is the fixed LLM on the evaluation server. Never add other external API calls.
    """

    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)
        self.call_count = 0

    def chat(self, messages: list[dict]) -> str:
        last_err: Exception | None = None
        extra = {}
        if self.cfg.reasoning_effort and self.cfg.reasoning_effort != "none":
            extra["reasoning_effort"] = self.cfg.reasoning_effort
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    **extra,
                )
                self.call_count += 1
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 — never kill the case
                last_err = e
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                time.sleep(30 if rate_limited else 1.5 * (attempt + 1))
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
