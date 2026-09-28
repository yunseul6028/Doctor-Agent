"""Per-case runtime guards: wall-clock budget, LLM call wrapper (timeouts, failure cap, gpt-oss options, prompt-size
logging). Everything here is created fresh per case (case independence)."""
import logging
import math
import threading
import time
from typing import Callable

from doctor_agent.agent import prompts
from doctor_agent.agent.parser import ACTION_SCHEMA
from doctor_agent.config import Config
from doctor_agent.llm.client import BillingError, LLMClient, LLMTimeout

log = logging.getLogger("doctor_agent.runtime")


class BudgetExceeded(RuntimeError):
    """Not enough wall-clock time left for another exploratory LLM call: finish the case now."""


class LLMUnavailable(RuntimeError):
    """The LLM failed too many times in a row in this case: stop calling it."""


class CaseBudget:
    """Wall-clock budget of one case. budget_s <= 0 means unlimited."""

    def __init__(self, budget_s: float, degrade_at_frac: float, final_reserve_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.start = clock()
        self.budget_s = budget_s if budget_s and budget_s > 0 else 0.0
        self.degrade_at_frac = degrade_at_frac
        # never reserve more than half the budget for the final answer
        self.final_reserve_s = min(final_reserve_s, self.budget_s / 2) if self.budget_s else 0.0

    @property
    def enabled(self) -> bool:
        return self.budget_s > 0

    def elapsed(self) -> float:
        return self.clock() - self.start

    def remaining(self) -> float:
        return self.budget_s - self.elapsed() if self.enabled else math.inf

    def degraded(self) -> bool:
        return self.enabled and self.elapsed() >= self.budget_s * self.degrade_at_frac

    def must_finish(self) -> bool:
        return self.enabled and self.remaining() <= self.final_reserve_s

    def deadline(self, final: bool) -> float | None:
        """Absolute clock time a call must finish by; exploratory calls must leave the final reserve untouched."""
        if not self.enabled:
            return None
        return self.start + self.budget_s - (0.0 if final else self.final_reserve_s)


def _is_action_prompt(messages: list[dict]) -> bool:
    return bool(messages) and messages[0].get("content") == prompts.SYSTEM


def _call_with_watchdog(fn: Callable[[], str], timeout: float) -> str:
    """Hard wall-clock guard around one LLM call (the HTTP timeout alone does not bound retries or a stuck socket).
    On timeout the call is abandoned in a daemon thread and its result discarded."""
    box: dict = {}

    def target() -> None:
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001 — re-raised in the caller thread
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise LLMTimeout(f"LLM call exceeded the {timeout:.0f}s watchdog")
    if "err" in box:
        raise box["err"]
    return box["out"]


class GuardedLLM:
    """Wraps the case's LLM client for the policy.

    - never lets an ordinary LLM error kill the case: the failed call returns "" (the policy treats it as a parse
      failure) until `max_llm_failures` consecutive failures, then raises LLMUnavailable
    - enforces the case budget (BudgetExceeded when only the final reserve is left) and a hard per-call watchdog
    - for clients that support options: structured output + JSON-from-reasoning fallback for action prompts,
      lower reasoning effort when the budget is running low, per-call deadline
    - logs prompt sizes
    """

    def __init__(self, inner: LLMClient, cfg: Config, budget: CaseBudget):
        self.inner = inner
        self.cfg = cfg
        self.budget = budget
        self.final_mode = False  # set when producing the forced final diagnosis (may use the reserve)
        self.degraded = False
        self.attempts = 0
        self.failures = 0  # consecutive
        self.dead = False
        self.errors: list[str] = []
        self.prompt_chars: list[int] = []
        self.subagent_calls = 0  # attempted sub-agent calls (agent/subagents/runner.py increments it)
        llm = cfg.llm
        self.call_timeout_s = llm.timeout_s * (llm.max_retries + 1) + 30.0

    @property
    def call_count(self) -> int:
        return self.inner.call_count

    @property
    def supports_options(self) -> bool:
        """Sub-agent calls may pass json_schema / expect_json / reasoning_effort / deadline (used only when the inner
        client supports options; ignored otherwise)."""
        return True

    def chat(self, messages: list[dict], **overrides) -> str:
        if self.dead:
            raise LLMUnavailable("LLM disabled for this case after repeated failures")
        remaining = self.budget.remaining()
        if self.budget.enabled:
            if not self.final_mode and remaining <= self.budget.final_reserve_s:
                raise BudgetExceeded(f"{remaining:.0f}s left (reserve {self.budget.final_reserve_s:.0f}s)")
            if remaining <= 0.5:
                raise BudgetExceeded("case time budget exhausted")
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        self.prompt_chars.append(chars)
        self.attempts += 1
        log.debug("LLM call #%d prompt_chars=%d remaining=%.0fs degraded=%s final=%s",
                  self.attempts, chars, remaining, self.degraded, self.final_mode)

        opts: dict = {}
        if getattr(self.inner, "supports_options", False):
            if _is_action_prompt(messages):
                opts["json_schema"] = ACTION_SCHEMA
                opts["expect_json"] = True
            if self.degraded or self.final_mode:
                opts["reasoning_effort"] = "low" if self.cfg.llm.reasoning_effort != "none" else None
            opts["deadline"] = self.budget.deadline(self.final_mode)
            for k in ("json_schema", "expect_json", "reasoning_effort"):
                if overrides.get(k) is not None:
                    opts[k] = overrides[k]
            if opts.get("reasoning_effort") and self.cfg.llm.reasoning_effort == "none":
                opts["reasoning_effort"] = None  # "none": the parameter is never sent
            if overrides.get("deadline") is not None:
                opts["deadline"] = min(d for d in (opts["deadline"], overrides["deadline"]) if d is not None)
        timeout = self.call_timeout_s
        if self.budget.enabled:
            # the watchdog runs on real time; the budget clock may be injected, so only use its *duration*
            timeout = max(1.0, min(timeout, (self.budget.deadline(self.final_mode) or 0) - self.budget.clock()))
        try:
            out = _call_with_watchdog(lambda: self.inner.chat(messages, **opts), timeout)
        except BillingError:
            if not self.cfg.agent.submission:
                raise  # dev: abort the whole evaluation batch
            self.dead = True
            self.errors.append("billing error")
            raise LLMUnavailable("billing error")
        except LLMTimeout as e:
            self.errors.append(f"timeout: {e}")
            self.failures += 1
            if self.budget.enabled:
                raise BudgetExceeded(str(e)) from e
            return self._failed()
        except Exception as e:  # noqa: BLE001 — an LLM error must never end the case without a diagnosis
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            log.warning("LLM call failed (%d in a row): %s", self.failures + 1, e)
            self.failures += 1
            return self._failed()
        self.failures = 0
        return out if isinstance(out, str) else str(out or "")

    def _failed(self) -> str:
        if self.failures >= self.cfg.agent.max_llm_failures:
            self.dead = True
            raise LLMUnavailable(f"{self.failures} consecutive LLM failures")
        return ""

    def stats(self) -> dict:
        return {"llm_attempts": self.attempts, "llm_errors": self.errors[-5:], "llm_disabled": self.dead,
                "prompt_chars_max": max(self.prompt_chars, default=0),
                "prompt_chars_total": sum(self.prompt_chars), "subagent_calls": self.subagent_calls}
