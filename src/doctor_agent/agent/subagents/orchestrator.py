"""Triggers, caps, logging and the hint queue of the specialist sub-agents (docs/architecture.md "Specialist
sub-agents"). One SubagentManager per Policy, i.e. per case: nothing here is shared across cases.

Sub-agents never pick the action. Their output is a short hint for the next main prompt, "참고" DDx candidates
(DdxLedger.refs), LLM-read result findings (radiology) and safety_log entries. Every entry point is guarded by the
caller (Policy) and none of them raises on a content-module failure: the failing sub-agent is disabled for the case."""
import importlib
import logging
import time
from typing import Callable

from doctor_agent.agent import prompts
from doctor_agent.agent.ledger import LLM_RADIOLOGY_SOURCE, Finding
from doctor_agent.agent.subagents import runner
from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult, failed
from doctor_agent.agent.text import similarity

log = logging.getLogger("doctor_agent.subagents")

LAYER = "subagent"
# content modules (owned by the content branches), imported lazily so a missing / broken one only disables itself
MODULES = {"consult": "doctor_agent.agent.subagents.consult", "advocate": "doctor_agent.agent.subagents.advocate",
           "specialty": "doctor_agent.knowledge.specialty"}
PRIORITY = {"radiology": 0, "consult": 1, "advocate": 2}  # order inside the sub-agent hint budget
MIN_HINT_CHARS = 60  # a hint cut shorter than this is not worth showing
RADIOLOGY_TEXT_CHARS = 3000  # result text sent to the radiology call
MAX_RADIOLOGY_ITEMS = 6
RADIOLOGY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "finding": {"type": "string"}, "status": {"type": "string"}, "site": {"type": "string"},
            "value": {"type": "string"}, "critical": {"type": "boolean"}}}},
        "normal": {"type": "boolean"}, "unavailable": {"type": "boolean"}, "summary": {"type": "string"}},
    "required": ["items"],
}
_RAD_STATUS = {"있음": "양성", "관찰됨": "양성", "present": "양성", "의심": "양성", "uncertain": "양성", "possible": "양성",
               "없음": "음성", "absent": "음성", "정상": "음성", "negative": "음성"}
_SKIP_KO = {"low_time": "시간 부족", "few_turns": "남은 턴 부족", "call_cap": "호출 한도 도달", "no_specialty": "전문 분야 없음",
            "radiology_cap": "판독 호출 한도 도달", "module_missing": "모듈 없음", "disabled": "끔"}
_TRIGGER_KO = {"routed": "전문 분야 집중", "low_confidence": "확신도 정체", "anchoring": "조기 고정 의심",
               "pre_review": "낮은 확신도 진단 직전", "needs_llm": "복잡한 결과"}


def _kind(name: str) -> str:
    """"consult:cardio" -> "consult" (also tolerates the older "consult_cardio")."""
    kind = name.split(":", 1)[0]
    return kind.split("_", 1)[0] if kind.split("_", 1)[0] in PRIORITY else kind


class SubagentManager:
    def __init__(self, llm, cfg):
        self.llm, self.cfg = llm, cfg
        self.degraded = False  # mirrored from the Policy each turn (low-time mode)
        self.attempted = 0
        self.calls: dict[str, int] = {}
        self.n_ok = self.n_fail = 0
        self.used: set[str] = set()  # "consult" / "advocate": at most once per case
        self.radiology_used = 0
        self.radiology_findings = 0
        self.disabled: dict[str, str] = {}  # kind -> why (module missing / content error)
        self.skipped: dict[str, int] = {}
        self._skip_logged: set[tuple[str, str]] = set()
        self._pending: list[tuple[int, str]] = []  # (priority, hint) for the next main prompt
        self._conf_trace: list[float] = []  # model confidence after each turn (index = turn - 1)
        self._mods: dict[str, object] = {}

    # ------------------------------------------------------------------------------------------ guards / logging
    def enabled(self, kind: str) -> bool:
        c = self.cfg
        if not getattr(c, "use_subagents", False) or kind in self.disabled:
            return False
        return {"consult": getattr(c, "use_consult", False), "advocate": getattr(c, "use_advocate", False),
                "radiology": getattr(c, "use_llm_radiology", False)}.get(kind, False)

    def blocked(self, state) -> str:
        """Global skip reason ("" = a call may be made)."""
        if self.degraded:
            return "low_time"
        if self.cfg.max_turns - state.turn_count < self.cfg.subagent_min_remaining_turns:
            return "few_turns"
        if self.attempted >= self.cfg.max_subagent_calls:
            return "call_cap"
        return ""

    def _skip(self, state, name: str, reason: str, turn: int | None = None) -> None:
        key = f"{name}:{reason}"
        self.skipped[key] = self.skipped.get(key, 0) + 1
        if (name, reason) in self._skip_logged:
            return
        self._skip_logged.add((name, reason))
        state.safety_log.append({"turn": turn or state.turn_count + 1, "layer": LAYER, "kind": "skip", "name": name,
                                 "reason": reason,
                                 "msg": f"{self._label(name)} 건너뜀: {_SKIP_KO.get(reason, reason)}"})

    def _error(self, state, kind: str, where: str, e: BaseException) -> None:
        """A content module failed: disable that sub-agent for the rest of the case (logged once)."""
        self.disabled[kind] = f"{where}: {type(e).__name__}"
        log.warning("sub-agent %s disabled (%s): %s", kind, where, e)
        state.safety_log.append({"turn": state.turn_count + 1, "layer": LAYER, "kind": "error", "name": kind,
                                 "where": where, "error": f"{type(e).__name__}: {e}"[:200]})

    def _module(self, state, kind: str, key: str | None = None):
        key = key or kind
        if key in self._mods:
            return self._mods[key]
        try:
            mod = importlib.import_module(MODULES[key])
        except Exception as e:  # noqa: BLE001 — ImportError (not merged yet) or an import-time bug
            self.disabled[kind] = "module_missing"
            self._skip(state, kind, "module_missing")
            log.info("sub-agent module %s unavailable: %s", MODULES[key], e)
            return None
        self._mods[key] = mod
        return mod

    @staticmethod
    def _label(name: str) -> str:
        kind = _kind(name)
        base = prompts.SUBAGENT_LABELS.get(kind, "자문")
        return f"{base}({name.split(':', 1)[1]})" if ":" in name else base

    # ------------------------------------------------------------------------------------------ one call
    def _call(self, state, call: SubagentCall, trigger: str, parse: Callable[[str], SubagentResult] | None,
              turn: int | None = None) -> SubagentResult:
        self.attempted += 1
        self.calls[call.name] = self.calls.get(call.name, 0) + 1
        t0 = time.monotonic()
        res = runner.run(self.llm, call, None, parse=parse, reasoning_effort=self.cfg.subagent_reasoning_effort)
        elapsed = round(time.monotonic() - t0, 2)
        if res.ok:
            self.n_ok += 1
        else:
            self.n_fail += 1
        added = []
        if res.ok and _kind(call.name) != "radiology":
            for d in res.ddx_add:
                if state.ddx_ledger.add_ref(d["name"], d.get("why", "")):
                    added.append(d["name"])
        err = res.raw.get("error", "") if not res.ok else ""
        msg = (f"{self._label(call.name)} 호출 ({_TRIGGER_KO.get(trigger, trigger)}) · "
               + (f"성공: {res.hint_ko[:120]}" if res.ok else f"실패: {err}")
               + (f" · 참고 감별 추가: {', '.join(added)}" if added else ""))
        state.safety_log.append({
            "turn": turn or state.turn_count + 1, "layer": LAYER, "kind": "call", "name": call.name,
            "trigger": trigger, "ok": res.ok, "elapsed_s": elapsed, "hint": res.hint_ko, "ddx_add": added,
            "red_flags": res.red_flags, "suggested_actions": res.suggested_actions,
            **({"error": err} if err else {}), "msg": msg})
        return res

    def _queue(self, kind: str, text: str) -> None:
        if text:
            self._pending.append((PRIORITY.get(kind, 9), text))

    @staticmethod
    def _hint_text(res: SubagentResult) -> str:
        """hint_ko, or the suggested actions when the sub-agent gave no hint text."""
        if res.hint_ko:
            return res.hint_ko
        acts = "; ".join(f"{a['type']}: {a['content']}" for a in res.suggested_actions[:3])
        return f"제안 행동: {acts}" if acts else ""

    # ------------------------------------------------------------------------------------------ per-step triggers
    def observe(self, state) -> None:
        """Records the model's confidence once per finished turn (for the stuck-consult trigger)."""
        while len(self._conf_trace) < state.turn_count:
            self._conf_trace.append(float(state.confidence or 0.0))

    def before_step(self, state, anchoring: dict | None = None) -> None:
        """Consult and anchoring-moment advocate triggers, at the start of Policy.next_action (after the advisors).
        Their hints are queued for the main prompt of this step."""
        if not getattr(self.cfg, "use_subagents", False):
            return
        self.observe(state)
        self._consult(state)
        if anchoring:
            self._advocate(state, anchoring.get("dx"), str(anchoring.get("why") or anchoring.get("msg") or ""),
                           "anchoring", queue=True)

    def _consult_trigger(self, state) -> tuple[str, str | None]:
        """(trigger, specialty) or ("", None)."""
        c = self.cfg
        low = (state.turn_count >= c.consult_low_conf_after and len(self._conf_trace) >= c.consult_low_conf_turns
               and all(x < c.consult_low_conf for x in self._conf_trace[-c.consult_low_conf_turns:]))
        if state.turn_count < c.consult_min_turns:
            return "", None
        spec_mod = self._module(state, "consult", "specialty")
        if spec_mod is None:
            return "", None
        try:
            specialty, share, _reasons = spec_mod.route(state)
        except Exception as e:  # noqa: BLE001
            self._error(state, "consult", "specialty.route", e)
            return "", None
        if specialty and float(share or 0) >= c.consult_min_share:
            return "routed", str(specialty)
        if low:
            return "low_confidence", (str(specialty) if specialty else None)
        return "", None

    def _consult(self, state) -> None:
        if not self.enabled("consult") or "consult" in self.used or self.degraded:
            return  # low-time mode: the routing is not even evaluated (CPU)
        trigger, specialty = self._consult_trigger(state)
        if not trigger:
            return
        if not specialty:
            self._skip(state, "consult", "no_specialty")
            return
        if why := self.blocked(state):
            self._skip(state, "consult", why)
            if why in ("few_turns", "call_cap"):  # permanent for this case: stop evaluating the routing
                self.used.add("consult")
            return
        consult = self._module(state, "consult")
        spec_mod = self._module(state, "consult", "specialty")
        if consult is None or spec_mod is None:
            return
        try:
            call = consult.build_consult(state, specialty, spec_mod.resources(specialty, state))
        except Exception as e:  # noqa: BLE001
            self._error(state, "consult", "build_consult", e)
            return
        self.used.add("consult")
        res = self._call(state, call, trigger, consult.parse_consult)
        if res.ok:
            spec = getattr(consult, "SPECIALTIES", {}).get(specialty) if isinstance(
                getattr(consult, "SPECIALTIES", None), dict) else None
            label = str(getattr(spec, "name_ko", "") or getattr(spec, "label", "") or specialty)
            if text := self._hint_text(res):
                self._queue("consult", prompts.subagent_hint("consult", text, label))

    def _advocate(self, state, proposed: str | None, reason: str, trigger: str, queue: bool) -> SubagentResult | None:
        if not self.enabled("advocate") or "advocate" in self.used:
            return None
        if why := self.blocked(state):
            self._skip(state, "advocate", why)
            return None
        mod = self._module(state, "advocate")
        if mod is None:
            return None
        try:
            call = mod.build_advocate(state, proposed, reason)
        except Exception as e:  # noqa: BLE001
            self._error(state, "advocate", "build_advocate", e)
            return None
        self.used.add("advocate")
        res = self._call(state, call, trigger, mod.parse_advocate)
        if res.ok and queue and (text := self._hint_text(res)):
            self._queue("advocate", prompts.subagent_hint("advocate", text))
        return res

    def advocate_for_review(self, state, proposed: str, reason: str, score_fn: Callable[[], float]) -> str:
        """Pre-review advocate: when the code confidence of the proposed diagnosis is < advocate_conf_below, one call
        whose hint is returned as a note for the review view ("" = nothing to add)."""
        if not self.enabled("advocate") or "advocate" in self.used:
            return ""
        score = score_fn()
        if score >= self.cfg.advocate_conf_below:
            return ""
        res = self._advocate(state, proposed, reason, "pre_review", queue=False)
        if res is None or not res.ok or not (text := self._hint_text(res)):
            return ""
        return prompts.ADVOCATE_REVIEW_NOTE.format(text=text)

    # ------------------------------------------------------------------------------------------ radiology
    def radiology(self, state, turn: int, test_name: str, result_text: str, code_line: str, interp) -> SubagentResult | None:
        """LLM reading of one result the code flagged (needs_llm). Findings → ledger (source llm_radiology), critical
        present items → state.result_criticals (concept "llm:<finding>"), summary → hint. None when not called."""
        if not (self.enabled("radiology") and getattr(self.cfg, "use_result_interpreter", False)):
            return None
        if self.radiology_used >= self.cfg.max_llm_radiology:
            self._skip(state, "radiology", "radiology_cap", turn)
            return None
        if why := self.blocked(state):
            self._skip(state, "radiology", why, turn)
            return None
        self.radiology_used += 1
        name = (test_name or "검사").strip()[:40]
        call = SubagentCall("radiology", prompts.build_result_interpreter_messages(
            name, (result_text or "")[:RADIOLOGY_TEXT_CHARS], code_line or "(없음)"), RADIOLOGY_SCHEMA)
        res = self._call(state, call, "needs_llm", parse_radiology, turn)
        if not res.ok:
            return res
        n = 0
        for f in radiology_findings(res.raw, turn, name):
            state.findings.add(f)
            n += 1
        self.radiology_findings += n
        for it in _rad_items(res.raw):
            if it["critical"] and it["status"] == "양성" and not it["hedged"]:
                if any(c.get("turn") == turn and similarity(str(c.get("label", "")), it["finding"]) >= 0.6
                       for c in state.result_criticals):
                    continue  # the code already flagged it
                state.result_criticals.append({
                    "turn": turn, "test": name, "concept": f"llm:{it['finding']}", "label": it["finding"],
                    "polarity": "present", "kind": getattr(interp, "kind", ""), "laterality": "",
                    "summary": f"{it['finding']} (LLM 판독)"})
        if res.hint_ko:
            self._queue("radiology", prompts.subagent_hint("radiology", f"[{name}] {res.hint_ko}"))
        return res

    # ------------------------------------------------------------------------------------------ output
    def take_hints(self) -> list[str]:
        """Queued hints for this step's main prompt, within cfg.max_subagent_chars (priority radiology > consult >
        advocate; a hint that does not fit is cut, not dropped: it cost an LLM call). The queue is cleared."""
        budget = max(0, int(self.cfg.max_subagent_chars))
        out = []
        for _, text in sorted(self._pending, key=lambda x: x[0]):
            if budget < MIN_HINT_CHARS:
                break
            if len(text) > budget:
                text = text[:budget - 1].rstrip() + "…"
            out.append(text)
            budget -= len(text)
        self._pending = []
        return out

    def summary(self, state=None) -> dict:
        return {"enabled": bool(getattr(self.cfg, "use_subagents", False)), "calls": dict(self.calls),
                "ok": self.n_ok, "fail": self.n_fail, "skipped": dict(self.skipped), "disabled": dict(self.disabled),
                "refs": state.ddx_ledger.refs_list() if state is not None else [],
                "llm_radiology_findings": self.radiology_findings}


# ---------------------------------------------------------------------------------------------- radiology parsing
def _rad_items(raw: dict) -> list[dict]:
    out = []
    items = raw.get("items") if isinstance(raw, dict) else None
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        finding = " ".join(str(it.get("finding") or "").split())[:60]
        st = str(it.get("status") or "").strip().lower().replace(" ", "")
        status = _RAD_STATUS.get(st)
        if not finding or status is None:
            continue
        out.append({"finding": finding, "status": status, "hedged": st in ("의심", "uncertain", "possible"),
                    "site": " ".join(str(it.get("site") or "").split())[:30],
                    "value": " ".join(str(it.get("value") or "").split())[:30],
                    "critical": it.get("critical") is True or str(it.get("critical")).lower() == "true"})
    return out


def parse_radiology(text: str) -> SubagentResult:
    obj = runner.last_object(text, ("items", "summary"))
    if not obj or not isinstance(obj.get("items", []), list):
        return failed("radiology", "no radiology JSON in the answer")
    summary = " ".join(str(obj.get("summary") or "").split())
    items = _rad_items(obj)
    if not summary and items:
        summary = "; ".join(f"{i['finding']} {'의심' if i['hedged'] else i['status']}" for i in items[:4])
    return SubagentResult("radiology", bool(items or summary or obj.get("normal") or obj.get("unavailable")),
                          summary, raw=obj)


def radiology_findings(raw: dict, turn: int, test_name: str) -> list[Finding]:
    """Ledger findings from an LLM reading: source llm_radiology, verified=None (the grounding check verifies them
    against the result text like model-reported findings). 있음 → 양성, 의심 → 양성 "의심", 없음 → 음성. The ledger keeps
    a code reading of the same item (FindingsLedger.add)."""
    out = []
    for it in _rad_items(raw)[:MAX_RADIOLOGY_ITEMS]:
        detail = [x for x in (it["site"], it["value"]) if x]
        if it["hedged"]:
            detail.append("의심")
        if it["critical"] and it["status"] == "양성":
            detail.append("위급")
        detail += [test_name, "LLM 판독"]
        out.append(Finding(it["finding"], it["status"], ", ".join(detail), turn, source=LLM_RADIOLOGY_SOURCE))
    return out
