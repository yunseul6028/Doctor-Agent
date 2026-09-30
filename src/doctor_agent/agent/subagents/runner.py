"""One guarded call of the fixed LLM for a sub-agent (docs/architecture.md "Specialist sub-agents").

`run` never raises: any failure (LLM error, time-out, budget exhausted, empty or unparseable answer, a bug in the
content module's parser) comes back as SubagentResult(ok=False) with the reason in raw["error"]."""
import logging
import time
from typing import Callable

from doctor_agent.agent.parser import as_list
from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult, failed

log = logging.getLogger("doctor_agent.subagents")

ACTION_TYPES = ("ASK", "EXAM", "TEST")
MIN_CALL_S = 1.0  # do not start a call with less than this before the deadline
MAX_LIST = 6
MAX_ITEM_CHARS = 160


def _s(x: object, cap: int = MAX_ITEM_CHARS) -> str:
    return " ".join(str(x or "").split())[:cap]


def sanitize(res: object, call: SubagentCall) -> SubagentResult:
    """Coerce whatever a parser returned into a well-typed SubagentResult within the caps."""
    if not isinstance(res, SubagentResult):
        return failed(call.name, f"parser returned {type(res).__name__}")
    ddx = []
    for d in as_list(res.ddx_add)[:MAX_LIST]:
        if isinstance(d, dict):
            nm = _s(d.get("name") or d.get("dx"), 60)
            if nm:
                ddx.append({"name": nm, "why": _s(d.get("why") or d.get("reason"))})
        elif _s(d, 60):
            ddx.append({"name": _s(d, 60), "why": ""})
    acts = []
    for a in as_list(res.suggested_actions)[:MAX_LIST]:
        if not isinstance(a, dict):
            continue
        typ, content = str(a.get("type", "")).strip().upper(), _s(a.get("content"), 80)
        if typ in ACTION_TYPES and content:
            acts.append({"type": typ, "content": content, "why": _s(a.get("why") or a.get("reason"))})
    flags = [_s(f, 80) for f in as_list(res.red_flags)[:MAX_LIST] if _s(f, 80)]
    cap = max(0, int(call.max_chars_out or 0))
    hint = _s(res.hint_ko, 4000)
    if len(hint) > cap:
        hint = hint[:max(0, cap - 1)].rstrip() + "…" if cap else ""
    raw = res.raw if isinstance(res.raw, dict) else {"raw": _s(res.raw, 500)}
    return SubagentResult(call.name, bool(res.ok), hint, ddx, acts, flags, raw)


def run(llm, call: SubagentCall, deadline: float | None = None, *,
        parse: Callable[[str], SubagentResult] | None = None, clock: Callable[[], float] = time.monotonic,
        reasoning_effort: str | None = "low") -> SubagentResult:
    """Calls the fixed LLM once for `call` and parses the answer with the content module's `parse`. Never raises (no
    parser: ok=False without a call).

    With a client that supports options (OpenAICompatClient, GuardedLLM around it): structured output with
    call.json_schema, expect_json (JSON written only in the reasoning is still found), reasoning_effort, deadline (an
    absolute time on `clock`; GuardedLLM also applies the case budget deadline). Plain clients get chat(messages)."""
    if parse is None:
        return failed(call.name, "no parser")
    try:
        if deadline is not None and clock() >= deadline - MIN_CALL_S:
            return failed(call.name, "no time left before the deadline")
        if hasattr(llm, "subagent_calls"):
            llm.subagent_calls += 1
        if getattr(llm, "supports_options", False):
            opts: dict = {"expect_json": True}
            if call.json_schema is not None:
                opts["json_schema"] = call.json_schema
            if reasoning_effort and reasoning_effort != "none":
                opts["reasoning_effort"] = reasoning_effort
            if deadline is not None:
                opts["deadline"] = deadline
            text = llm.chat(call.messages, **opts)
        else:
            text = llm.chat(call.messages)
    except BaseException as e:  # noqa: BLE001 — BudgetExceeded / LLMUnavailable / timeouts: the main loop re-checks
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        log.warning("sub-agent %s call failed: %s", call.name, e)
        return failed(call.name, f"{type(e).__name__}: {e}")
    if not (text or "").strip():
        return failed(call.name, "empty answer")
    try:
        res = parse(text)
    except Exception as e:  # noqa: BLE001 — a content-module parser bug must not stop the case
        log.warning("sub-agent %s parser failed: %s", call.name, e)
        return failed(call.name, f"parser {type(e).__name__}: {e}")
    out = sanitize(res, call)
    if out.ok and not (out.hint_ko or out.ddx_add or out.suggested_actions or out.red_flags or out.raw):
        return failed(call.name, "empty result")
    return out
