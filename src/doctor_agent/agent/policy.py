import logging
import re

from doctor_agent.agent import anchoring, confidence, grounding, kb_hints, prompts, question_planner, result_interpreter
from doctor_agent.agent.ledger import CODE_SOURCE, Finding
from doctor_agent.agent.parser import _json_objects, extract_action_json, parse_action
from doctor_agent.agent.state import CaseState
from doctor_agent.agent.subagents.orchestrator import SubagentManager
from doctor_agent.config import AgentConfig
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.llm.client import LLMClient
from doctor_agent.knowledge import clinical_rules, diagnostic_criteria
from doctor_agent.safety import danger_gate, preconditions, protocols, triage

log = logging.getLogger("doctor_agent.policy")

MAX_RULES_IN_PROMPT = 2  # keep prompts short for the small fixed LLM
MAX_ATTEMPTS = 4
MAX_REVIEWS = 2  # pre-diagnosis reviews per case (each hold costs a turn)
MAX_HINTS_DEGRADED = 2  # hints kept when the time budget runs low (can't-miss + pending safety checks come first)
# triage flags that make an "unknown" level (vitals not measured yet) urgent enough for the top-of-prompt alert
TRIAGE_RED_FLAGS = ("ams", "chest_pain", "syncope", "bleeding", "anaphylaxis", "airway", "respiratory", "seizure",
                    "sepsis_suspected", "trauma")
# a question sent as EXAM/TEST ("목이 뻣뻣한가요?") is really an ASK
_QUESTION_END = re.compile(r"(\?|？|나요|세요|니까|까요|있어요|없어요|하셨어요|드세요)\s*$")


def normalize_type(action: Action) -> Action:
    from doctor_agent.agent.text import looks_like_history_question

    text = action.content.strip()
    if action.type in (ActionType.EXAM, ActionType.TEST) and (_QUESTION_END.search(text) or looks_like_history_question(text)):
        return Action(ActionType.ASK, action.content, action.reason)
    return action


class Policy:
    """LLM proposes the action → rules enforce the budget, no repeats, and safety."""

    def __init__(self, llm: LLMClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg
        self._kb_seen: set = set()  # KB hints already shown in this case (Policy is created per case)
        self.degraded = False  # set by the loop when the case time budget runs low (runtime.CaseBudget)
        self._conf_params = None  # confidence parameters, loaded on first use (per case)
        self._planner_logged = ""  # last planner suggestions written to the safety log
        self._interp_done = 0  # turns already given to the result interpreter
        self.subagents = SubagentManager(llm, cfg)  # specialist sub-agents (per case, only when triggered)
        self._anchoring_fired: dict | None = None  # premature-closure entry shown in this step (advocate trigger)

    def next_action(self, state: CaseState) -> Action:
        self.subagents.degraded = self.degraded  # low-time mode: no sub-agent call from here on (radiology included)
        try:
            self._interpret_results(state)
        except Exception as e:  # noqa: BLE001 — each result is guarded; this only catches a bug in the bookkeeping
            self._advisor_error(state, "result_interp", e)
        remaining = self.cfg.max_turns - state.turn_count
        if remaining <= 1:
            return self._final_diagnosis(state)

        hints = self._hints(state)
        self._anchoring_fired = None
        try:
            alert, advice = self._advisors(state)
        except Exception as e:  # noqa: BLE001 — each advisor is guarded; this only catches a bug in the composition
            self._advisor_error(state, "advisors", e)
            alert, advice = "", []
        sub_hints = self._subagent_hints(state)
        if self.degraded:  # time budget running low: short prompt, no extra review/pushback calls
            hints = hints[:MAX_HINTS_DEGRADED] + [prompts.LOW_TIME_HINT]
        else:
            hints = hints + advice + sub_hints
        for attempt in range(MAX_ATTEMPTS):  # retries: parse failure, repeated action, safety pushback, review hold
            raw = self.llm.chat(prompts.build_step_messages(state.view(), state.turn_count, self.cfg.max_turns, hints,
                                                            alert=alert))
            parsed = parse_action(raw)
            if parsed is None:
                hints = hints + ["정해진 형식의 JSON 한 줄로만 출력하세요."]
                continue
            action, ddx, conf = parsed
            action = normalize_type(action)
            obj = extract_action_json(raw) or {}
            state.findings.update(obj.get("findings"), state.turn_count)
            state.ddx_ledger.update(ddx)
            state.ddx, state.confidence = state.ddx_ledger.as_list(), conf
            self._ground(state)
            if action.type != ActionType.DIAGNOSE and state.asked(action):
                hints = hints + [f"'{action.content}'은(는) 이미 했습니다. 다른 행동을 고르세요."]
                continue
            if action.type == ActionType.DIAGNOSE and not state.safety_pushback and remaining > 5 and not self.degraded:
                pending = self._pending(state)
                if pending:
                    # one pushback per case: diagnose anyway only with a stated reason
                    state.safety_pushback = True
                    hints = hints + ["진단 전에 아직 안 한 필수 안전 확인이 있습니다: " + ", ".join(c.name for c in pending)
                                     + ". 이 중 하나를 먼저 하세요. 정말 불필요하면 reason에 그 이유를 쓰고 진단하세요."]
                    continue
            if action.type == ActionType.DIAGNOSE and not self.degraded:
                gated, hint = self._gate(state, action, remaining)
                if gated is not None:
                    return self._safe(state, gated)
                if hint:
                    hints = hints + [hint]
                    continue
                # low code-computed confidence: one pushback per case, only with >= 2 attempts left (so the retry
                # budget cannot run out on it) and never close to the turn cap
                if attempt < MAX_ATTEMPTS - 2 and remaining > 3 and (hint := self._confidence(state, action)):
                    hints = hints + [hint]
                    continue
            if action.type == ActionType.DIAGNOSE and len(state.reviews) < MAX_REVIEWS and remaining > 3 and not self.degraded:
                reviewed = self._review(state, action, self._advocate_note(state, action))
                if reviewed.type != ActionType.DIAGNOSE and state.asked(reviewed):
                    hints = hints + ["검토의 지적: " + "; ".join(state.reviews[-1]["issues"]) + ". 이를 해결할 다른 행동을 고르세요."]
                    continue
                return self._safe(state, reviewed)
            if action.type != ActionType.DIAGNOSE:
                safe = self._safe(state, action)
                if safe is None:  # blocked with no safe alternative: ask the model for a different action
                    hints = hints + [state.safety_log[-1]["why"] + " 다른 행동을 고르세요."]
                    continue
                return safe
            return action
        return self._final_diagnosis(state)

    # ---- safety layers (each guarded: a bug here must never stop a case) ----
    def _ground(self, state: CaseState) -> None:
        """Mark findings / DDx evidence the environment never said as unverified (grounding.apply)."""
        if not self.cfg.use_grounding:
            return
        try:
            report = grounding.apply(state)
        except Exception as e:  # noqa: BLE001
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "grounding", "error": str(e)[:200]})
            return
        if report.get("findings_unverified") or report.get("ddx_removed") or report.get("reason_ungrounded"):
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "grounding", **report})

    def _gate(self, state: CaseState, action: Action, remaining: int) -> tuple[Action | None, str | None]:
        """Can't-miss rule-out gate before a diagnosis. Returns (forced action, None), (None, one-time hint) or
        (None, None) when the diagnosis may proceed."""
        if not self.cfg.use_danger_gate:
            return None, None
        try:
            g = danger_gate.gate(state, action.content, remaining, max_gate_turns=self.cfg.max_gate_turns,
                                 gate_turns_used=state.gate_turns)
        except Exception as e:  # noqa: BLE001
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "danger_gate", "error": str(e)[:200]})
            return None, None
        if not g.get("allow") and g.get("action"):
            typ, content, reason = g["action"]
            forced = normalize_type(Action(ActionType(getattr(typ, "value", typ)), content,
                                           f"위험 질환 배제({g.get('danger')}): {reason or g.get('why', '')}"))
            if not state.asked(forced):
                state.gate_turns += 1
                state.safety_log.append({"turn": state.turn_count + 1, "layer": "danger_gate", "kind": "rule_out",
                                         "proposed": action.content, "danger": g.get("danger"), "why": g.get("why"),
                                         "action": forced.to_dict()})
                return forced, None
        if g.get("kind") == "confirmed_other" and not state.confirmed_other_shown:
            state.confirmed_other_shown = True
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "danger_gate", "kind": "confirmed_other",
                                     "proposed": action.content, "danger": g.get("danger"), "why": g.get("why")})
            return None, f"참고: {g.get('why')} 제안한 진단이 이 소견을 설명하는지 확인하고 진단하세요."
        return None, None

    def _safe(self, state: CaseState, action: Action) -> Action | None:
        """Pre-test precondition check for TEST/EXAM: swap in the prerequisite, annotate warnings, or return None when
        the action is contraindicated with no alternative."""
        if not self.cfg.use_preconditions or action.type not in (ActionType.TEST, ActionType.EXAM):
            return action
        try:
            res = preconditions.check(action.type, action.content, state)
        except Exception as e:  # noqa: BLE001
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "preconditions", "error": str(e)[:200]})
            return action
        if res.get("ok", True) and res.get("severity") != "warn":
            return action
        cite = f" ({res['citation']})" if res.get("citation") else ""
        entry = {"turn": state.turn_count + 1, "layer": "preconditions", "severity": res.get("severity"),
                 "rule": res.get("rule"), "requested": action.content, "why": f"{res.get('why', '')}{cite}"}
        state.safety_log.append(entry)
        if res.get("severity") == "warn":
            return Action(action.type, action.content, f"{action.reason} [주의: {res.get('why', '')}]".strip())
        pre = res.get("prerequisite")
        if pre:
            typ, content = pre
            first = normalize_type(Action(ActionType(getattr(typ, "value", typ)), content,
                                          f"검사 전 안전 확인: {res.get('why', '')}{cite}"))
            if not state.asked(first):
                entry["replaced_with"] = first.to_dict()
                return first
        return None

    # ---- advisors (each guarded and switchable; they only add hints / one pushback, never pick the action) ----
    def _advisor_error(self, state: CaseState, layer: str, e: Exception) -> None:
        log.warning("advisor %s failed: %s", layer, e)
        state.safety_log.append({"turn": state.turn_count + 1, "layer": layer, "error": f"{type(e).__name__}: {e}"[:200]})

    def _confidence(self, state: CaseState, action: Action) -> str | None:
        """One pushback per case when the code-computed confidence of the proposed diagnosis is below
        cfg.confidence_pushback_below. "must_continue" (an actionable unresolved can't-miss danger) is left to the
        danger gate, which has already had its say, so a diagnosis is never blocked twice for the same reason."""
        if not self.cfg.use_confidence or state.confidence_pushback:
            return None
        try:
            if self._conf_params is None:
                self._conf_params = confidence.load_params()
            a = confidence.assess(state, action.content, self.cfg, self._conf_params)
            push = a.recommendation != "must_continue" and a.score < self.cfg.confidence_pushback_below
            state.safety_log.append({
                "turn": state.turn_count + 1, "layer": "confidence", "proposed": action.content, "score": a.score,
                "recommendation": a.recommendation, "pushback": push, "components": a.components,
                "msg": f"진단({action.content}) 확신도 {a.score:.2f} → {a.recommendation}" + (" · 재고 요청" if push else "")})
            if not push:
                return None
            state.confidence_pushback = True
            return prompts.confidence_pushback(action.content, a.score, a.reasons_ko)
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "confidence", e)
            return None

    def _advisors(self, state: CaseState) -> tuple[str, list[str]]:
        """(top-of-prompt alert, extra hints) from triage (+ new critical results in the alert), the reading of the
        latest result, the anchoring check, the starting DDx and the question planner. Within cfg.max_advisor_chars in
        total, filled in that priority order; a hint that does not fit is dropped (not cut) and is neither logged nor
        counted as shown."""
        alert, triage_hint = self._triage(state)
        if critical := self._critical_alert(state):  # triage text first, then the critical results (same slot)
            alert = f"{alert} / {critical}" if alert else critical
        if self.degraded:  # low-time mode: only the alert survives (the other hints would be cut anyway)
            return alert, []
        items = [(triage_hint, None), self._result_hint(state), self._anchoring(state), self._initial_ddx(state),
                 self._planner(state)]
        budget = max(0, self.cfg.max_advisor_chars - len(alert))
        out = []
        for text, entry in items:
            if not text or len(text) > budget:
                continue
            out.append(text)
            budget -= len(text)
            if entry:
                if entry.get("kind") == "premature_closure":
                    state.anchoring_shown = True  # shown once per case, counted only once really in the prompt
                    self._anchoring_fired = entry
                if entry["layer"] == "planner":  # shown every turn, logged only when the suggestions change
                    if entry["msg"] == self._planner_logged:
                        continue
                    self._planner_logged = entry["msg"]
                state.safety_log.append(entry)
        return alert, out

    def _triage(self, state: CaseState) -> tuple[str, str]:
        """(alert, hint). Unstable, or vitals unknown with a red flag → alert at the top of the prompt; concerning →
        ordinary hint; stable / unknown without red flags → nothing (a level change is still logged)."""
        if not self.cfg.use_triage:
            return "", ""
        try:
            res = triage.assess(state)
            level = res["level"]
            red = [k for k in TRIAGE_RED_FLAGS if (res.get("flags") or {}).get(k)]
            top = level == "unstable" or (level == "unknown" and bool(red))
            text = triage.render_for_prompt(state, res) if top or level == "concerning" else ""
            if level != state.triage_level:
                state.triage_level = level
                state.safety_log.append({
                    "turn": state.turn_count + 1, "layer": "triage", "level": level, "why": res.get("why_ko", ""),
                    "red_flags": red, "shown": ("top" if top else "hint") if text else "",
                    "msg": text or f"중증도 {level}: {res.get('why_ko', '')}"})
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "triage", e)
            return "", ""
        return (text, "") if top else ("", text)

    def _initial_ddx(self, state: CaseState) -> tuple[str, dict | None]:
        """Broad starting differential, turn 1 only."""
        if not self.cfg.use_anchoring or state.turn_count:
            return "", None
        try:
            ddx = anchoring.initial_differential(state.initial_info)
            names = [str(d.get("dx")) for d in ddx]
            return anchoring.render_for_prompt(ddx), {"turn": 1, "layer": "anchoring", "kind": "initial_ddx",
                                                      "ddx": names, "msg": "초기 감별 목록: " + ", ".join(names)}
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "anchoring", e)
            return "", None

    def _anchoring(self, state: CaseState) -> tuple[str, dict | None]:
        """Premature-closure check from turn anchoring.MIN_TURNS on, each turn until it fires; shown once per case."""
        if not self.cfg.use_anchoring or state.anchoring_shown or state.turn_count < anchoring.MIN_TURNS:
            return "", None
        try:
            chk = anchoring.anchoring_check(state)
            if not chk or not chk.get("prompt_ko"):
                return "", None
            return str(chk["prompt_ko"]), {
                "turn": state.turn_count + 1, "layer": "anchoring", "kind": "premature_closure", "dx": chk.get("dx"),
                "reasons": chk.get("reasons"), "msg": f"'{chk.get('dx')}' 조기 고정 의심: {chk.get('why_ko', '')}"}
        except Exception as e:  # noqa: BLE001
            state.anchoring_shown = True  # a failing check is not retried every turn
            self._advisor_error(state, "anchoring", e)
            return "", None

    def _planner(self, state: CaseState) -> tuple[str, dict | None]:
        """Most discriminating next actions (KB information gain), every turn; needs the KB (use_kb)."""
        if not (self.cfg.use_planner and self.cfg.use_kb):
            return "", None
        try:
            sugg = question_planner.suggest(state, k=self.cfg.planner_k)
            shown = [f"{s.type}: {s.content_ko}" for s in sugg if not s.source.startswith("protocol:")]
            return question_planner.render_for_prompt(sugg), {"turn": state.turn_count + 1, "layer": "planner",
                                                               "suggestions": shown, "msg": "; ".join(shown)}
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "planner", e)
            return "", None

    # ---- result interpreter (code-first reading of EXAM/TEST results; guarded, switchable) ----
    def _interpret_results(self, state: CaseState) -> None:
        """Read every EXAM/TEST response not read yet: store the reading on the Turn, merge its findings into the
        findings ledger (as code-verified findings), keep critical items for the alert and the can't-miss gate, count
        the results that would need an LLM reading (not called). Each turn is read once, even when it fails."""
        if not self.cfg.use_result_interpreter:
            return
        stats = state.interp_stats
        for k in ("n", "needs_llm", "critical", "unavailable", "errors"):
            stats.setdefault(k, 0)
        stats.setdefault("llm_reasons", {})
        while self._interp_done < len(state.turns):
            idx = self._interp_done
            self._interp_done += 1
            t = state.turns[idx]
            if t.action.type not in (ActionType.EXAM, ActionType.TEST) or not (t.response or "").strip() \
                    or t.response.startswith("(환경 응답 오류)"):
                continue
            try:
                self._interpret_turn(state, idx + 1, t)
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                self._advisor_error(state, "result_interp", e)

    def _interpret_turn(self, state: CaseState, turn: int, t) -> None:
        stats = state.interp_stats
        interp = result_interpreter.interpret(t.action.content, t.response, {"initial_info": state.initial_info})
        line = result_interpreter.render_for_prompt(
            interp, max(60, self.cfg.result_hint_chars - len(prompts.RESULT_HINT.format(line=""))))
        reasons = result_interpreter.llm_reasons(interp)
        t.interp = {**interp.as_dict(), "prompt": line, "needs_llm": bool(reasons), "pending": interp.pending}
        stats["n"] += 1
        stats["unavailable"] += int(interp.unavailable)
        if any(r == "error" for r in reasons):
            stats["errors"] += 1
        if reasons:  # counted here; the LLM reading itself is the radiology sub-agent (below, capped per case)
            stats["needs_llm"] += 1
            for r in reasons:
                stats["llm_reasons"][r] = stats["llm_reasons"].get(r, 0) + 1
        for f in _interp_findings(interp, turn):
            state.findings.add(f)
        called = None
        if reasons:  # optional extra gpt-oss reading (radiology sub-agent, prompts.RESULT_INTERPRETER_PROMPT)
            try:
                called = self.subagents.radiology(state, turn, interp.test_name or t.action.content, t.response, line,
                                                  interp)
            except Exception as e:  # noqa: BLE001
                self._advisor_error(state, "subagent", e)
        crit = []
        for i in interp.critical():
            if i.polarity == "absent":
                continue
            crit.append({"turn": turn, "test": interp.test_name, "concept": i.concept, "label": i.label,
                         "polarity": i.polarity, "kind": i.kind, "laterality": i.laterality,
                         "summary": i.summary_ko.lstrip("⚠")})
        state.result_criticals.extend(crit)
        stats["critical"] += len(crit)
        state.safety_log.append({
            "turn": turn, "layer": "result_interp", "kind": "reading", "test": interp.test_name,
            "test_kind": interp.kind, "critical": [c["summary"] for c in crit], "needs_llm": bool(reasons),
            "llm_reasons": reasons,
            "llm_called": called is not None,
            "msg": line + (f" · LLM 판독 필요({', '.join(reasons)}; {'호출함' if called is not None else '호출 안 함'})"
                           if reasons else "")})

    def _result_hint(self, state: CaseState) -> tuple[str, dict | None]:
        """Code reading of the latest action's result (EXAM/TEST), on the prompt right after it."""
        if not self.cfg.use_result_interpreter or not state.turns:
            return "", None
        interp = state.turns[-1].interp
        line = (interp or {}).get("prompt")
        return (prompts.result_hint(line), None) if line else ("", None)

    def _critical_alert(self, state: CaseState) -> str:
        """Critical results not shown yet → one alert line (each critical finding is shown once per case)."""
        if not self.cfg.use_result_interpreter:
            return ""
        try:
            new = []
            for c in state.result_criticals:
                key = (c.get("concept"), c.get("laterality"), c.get("polarity"))
                if key in state.critical_alerted:
                    continue
                state.critical_alerted.add(key)
                new.append(f"{c.get('summary') or c.get('label')} [{c.get('test', '')[:20]}]")
            if not new:
                return ""
            text = prompts.result_critical_alert(new)
            state.safety_log.append({"turn": state.turn_count + 1, "layer": "result_interp", "kind": "critical_alert",
                                     "items": new, "msg": "위급 결과 알림: " + ", ".join(new)})
            return text
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "result_interp", e)
            return ""

    # ---- specialist sub-agents (agent/subagents/; guarded, switchable, capped; see docs/architecture.md) ----
    def _subagent_hints(self, state: CaseState) -> list[str]:
        """Consult / anchoring-moment advocate triggers for this step, then every queued sub-agent hint (radiology
        from the result reading above, consult, advocate) within cfg.max_subagent_chars."""
        try:
            self.subagents.degraded = self.degraded
            self.subagents.before_step(state, self._anchoring_fired)
            return self.subagents.take_hints()
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "subagent", e)
            return []

    def _advocate_note(self, state: CaseState, action: Action) -> str:
        """Pre-review advocate (once per case, only when the code confidence of the proposal is low): a note for the
        review view, or ""."""
        def score() -> float:
            if self._conf_params is None:
                self._conf_params = confidence.load_params()
            return confidence.assess(state, action.content, self.cfg, self._conf_params).score
        try:
            self.subagents.degraded = self.degraded
            return self.subagents.advocate_for_review(state, action.content, action.reason, score)
        except Exception as e:  # noqa: BLE001
            self._advisor_error(state, "subagent", e)
            return ""

    def subagent_summary(self, state: CaseState) -> dict:
        return self.subagents.summary(state)

    def _review(self, state: CaseState, action: Action, note: str = "") -> Action:
        """Pre-diagnosis review by the same LLM in a reviewer role. The reviewer only fills structured fields
        (key findings explained or not, contradictions, confirmatory evidence, unresolved dangers); the verdict and
        any renaming are decided here in code. Returns the (possibly refined) diagnosis, or the reviewer's next
        action on hold."""
        view = state.view()
        if self.cfg.use_kb and (warn := kb_hints.normalize_hint(action.content, state.initial_info).get("warning")):
            view += "\n\n[지식베이스 경고] " + warn
        if criteria := diagnostic_criteria.render_for_review(action.content, _case_findings(state)):
            view += "\n\n" + criteria
        if note:  # pre-review advocate sub-agent (prompts.ADVOCATE_REVIEW_NOTE)
            view += "\n\n" + note
        raw = self.llm.chat(prompts.build_review_messages(view, action.content, action.reason,
                                                          state.turn_count, self.cfg.max_turns))
        objs = _json_objects(raw or "")
        obj = next((o for o in reversed(objs) if "key_findings" in o or "confirmation" in o), objs[-1] if objs else {})
        key = [(str(k.get("finding", "")).strip(), _explained(k.get("status"))) for k in _as_list(obj.get("key_findings"))
               if isinstance(k, dict) and str(k.get("finding", "")).strip()]
        contra = [str(x).strip() for x in _as_list(obj.get("contradicting")) if _meaningful(x)]
        danger = [str(x).strip() for x in _as_list(obj.get("unresolved_danger")) if _meaningful(x)]
        confirmation = str(obj.get("confirmation") or "").strip()
        confirmed = _meaningful(confirmation)
        reasons = ([f"설명 안 되는 소견: {f}" for f, ok in key if not ok] + [f"모순 소견: {x}" for x in contra]
                   + ([] if confirmed else ["확진 근거 없음"]) + [f"미배제 위험 질환: {x}" for x in danger])
        follow = _next_action(obj.get("next"), reasons)
        if not obj:
            reasons = []  # unparseable review → never block the diagnosis on it
        verdict = "보류" if reasons and follow is not None else "승인"
        record = {"turn": state.turn_count + 1, "proposed": action.content, "verdict": verdict, "issues": reasons,
                  "key_findings": [{"finding": f, "explained": ok} for f, ok in key], "contradicting": contra,
                  "confirmation": confirmation if confirmed else "없음", "unresolved_danger": danger}
        state.reviews.append(record)
        if verdict == "보류":
            record["next"] = {"type": follow.type.value, "content": follow.content}
            return follow
        refined = str(obj.get("final_diagnosis") or "").strip()
        if refined and _norm(refined) != _norm(action.content):
            evidence = str(obj.get("refine_evidence") or obj.get("근거") or "").strip()
            why_not = _refinement_problem(action.content, refined, evidence, state.transcript() + "\n" + state.findings.render(exclude_unverified=True))
            record["refinement"] = {"name": refined, "evidence": evidence, "accepted": not why_not, "rejected": why_not}
            if not why_not:
                record["final_diagnosis"] = refined
                return Action(ActionType.DIAGNOSE, refined, f"{action.reason} (검토의가 세부 진단명으로 수정: {evidence})")
        return action

    @staticmethod
    def _learned(state: CaseState) -> str:
        # everything learned so far; only used for conditional triggers, not for picking the category
        return " ".join(t.response for t in state.turns)

    def _pending(self, state: CaseState) -> list:
        actions = [t.action.content for t in state.turns]
        pending = protocols.pending_checks(state.initial_info, actions, self._learned(state))
        return [c for c in pending if c.kind != "treatment"]

    def _hints(self, state: CaseState) -> list[str]:
        hints = []
        text = state.initial_info
        cant = protocols.cant_miss_for(text)
        if cant:
            hints.append("반드시 배제할 위험 질환: " + ", ".join(cant))
        pending = self._pending(state)
        if pending:
            hints.append("아직 안 한 최소 안전 확인 (근거 지침): " + "; ".join(
                f"{c.name}{f' [{c.when}]' if c.when else ''} ({c.citation.short})" for c in pending))
        na = state.unavailable()
        if na:
            hints.append("결과가 제공되지 않은 요청 (다시 요청하지 말고 다른 방법으로 확인): " + "; ".join(x[:40] for x in na[-6:]))
        rules = clinical_rules.rules_for(text)[:MAX_RULES_IN_PROMPT]
        if rules:
            hints.append(clinical_rules.render_for_prompt(rules))
        if state.turn_count >= self.cfg.target_turns:
            hints.append("목표 턴 수를 넘었습니다. 충분히 확신하면 진단하세요.")
        if self.cfg.use_kb:
            hints += kb_hints.step_hints(state, self._kb_seen)
        return hints

    def _final_diagnosis(self, state: CaseState) -> Action:
        parsed = parse_action(self.llm.chat(prompts.build_final_messages(state.view())))
        if parsed and parsed[0].type == ActionType.DIAGNOSE:
            return parsed[0]
        top = state.ddx[0].get("dx") if state.ddx and isinstance(state.ddx[0], dict) else None
        return Action(ActionType.DIAGNOSE, top or "진단 불가")


# --- result interpreter → findings ledger -----------------------------------------------------------------------
MAX_LEDGER_ITEMS_PER_RESULT = 6
_NORMAL_CONCEPTS = ("IMG:normal_study", "ECG:normal_ecg")
_POL_STATUS = {"present": "양성", "uncertain": "양성", "absent": "음성"}
_LAT_KO = {"right": "우측", "left": "좌측", "bilateral": "양측"}


def _interp_findings(interp, turn: int) -> list[Finding]:
    """Ledger findings (source=CODE_SOURCE, verified) from one reading: present → 양성, uncertain → 양성 "의심",
    absent → 음성 (a normal measurement once, as its span "정상"), whole-normal study → the test 음성 "특이 소견 없음",
    "not provided" / pending-only → the test 결과없음 (never 음성). Unmapped wording (concept "") stays prompt-only.
    Abnormal items first, at most MAX_LEDGER_ITEMS_PER_RESULT."""
    name = (interp.test_name or "검사").strip()[:40]

    def f(item: str, status: str, detail: str = "", span: str = "") -> Finding:
        return Finding(item, status, detail, turn, verified=True, span=span[:120], source=CODE_SOURCE)

    if interp.unavailable:
        return [f(name, "결과없음", "결과 제공 안 됨")]
    out: list[Finding] = []
    abnormal = sorted((i for i in interp.items if i.concept and i.polarity != "absent" and i.concept not in _NORMAL_CONCEPTS),
                      key=lambda i: (not i.critical, i.polarity != "present"))
    for i in abnormal:
        detail = [x for x in (_LAT_KO.get(i.laterality, ""), i.site if i.kind in ("imaging", "ecg") else "") if x]
        if i.value is not None:
            detail.append(f"{i.value:g}{i.unit}")
        if i.polarity == "uncertain":
            detail.append("의심")
        if i.critical:
            detail.append("위급")
        detail.append(name)
        out.append(f(i.label, _POL_STATUS[i.polarity], ", ".join(detail), i.span))
    if interp.normal and any(i.concept in _NORMAL_CONCEPTS for i in interp.items):
        out.append(f(name, "음성", "특이 소견 없음", "정상"))
    seen: set[str] = set()
    for i in interp.items:
        if i.polarity != "absent" or not i.concept:
            continue
        item, detail = (i.span, "정상") if i.value is not None and i.span else (i.label, name)
        if item not in seen:
            seen.add(item)
            out.append(f(item, "음성", detail, i.span))
    if interp.pending and not out:
        out.append(f(name, "결과없음", "결과 대기 중"))
    return out[:MAX_LEDGER_ITEMS_PER_RESULT]


# --- pre-diagnosis review helpers -------------------------------------------------------------------------------
_NONE_WORDS = {"", "없음", "없다", "없습니다", "none", "null", "n/a", "na", "-", "해당없음", "해당 없음", "미확인", "추측", "불명"}
_NOT_EXPLAINED = re.compile(r"안\s*됨|안됨|않|불충분|미설명|no|false|unexplained", re.I)
# qualifiers that only add where the disease is or what caused/accompanies it; the standard name is preferred
_LOCATION = re.compile(r"상행|하행|횡행|S상|구불|맹장부|좌측|우측|양측|좌엽|우엽|상엽|중엽|하엽|전벽|하벽|측벽|후벽|근위부|원위부|기저부|첨부")
_CAUSE = re.compile(r"에\s*의한|(으)?로\s*인한|에\s*따른|에\s*동반된|의존성|유발성|연관|관련")
_PARENS = re.compile(r"\([^)]*\)")


def _as_list(x: object) -> list:
    return x if isinstance(x, list) else ([] if x in (None, "") else [x])


def _meaningful(x: object) -> bool:
    t = str(x or "").strip().strip(".").lower()
    return t not in _NONE_WORDS and not t.startswith("없음")


def _explained(status: object) -> bool:
    if isinstance(status, bool):
        return status
    return not _NOT_EXPLAINED.search(str(status or ""))


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", _PARENS.sub("", name)).lower()


def _next_action(nxt: object, reasons: list[str]) -> Action | None:
    if not isinstance(nxt, dict):
        return None
    try:
        kind = ActionType(str(nxt.get("type", "")).upper())
    except ValueError:
        return None
    content = str(nxt.get("content", "")).strip()
    if kind == ActionType.DIAGNOSE or not content:
        return None
    return normalize_type(Action(kind, content, "검토 보류: " + (str(nxt.get("reason", "")).strip() or "; ".join(reasons))))


_STOP = {"없음", "있음", "양성", "음성", "정상", "이상", "소견", "검사", "결과", "환자", "증상", "진단", "확인", "관련", "의한", "인한", "따른", "동반된"}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9a-z가-힣]+", _PARENS.sub("", text).lower()) if len(w) >= 2 and w not in _STOP]


def _grounded(text: str, case_text: str) -> bool:
    """True if at least half of the content words of `text` literally appear in `case_text` (a trailing Korean
    particle is tolerated)."""
    case = re.sub(r"\s+", "", case_text).lower()
    words = _words(text)
    hits = sum(w in case or (len(w) >= 3 and w[:-1] in case) for w in words)
    return bool(words) and hits * 2 >= len(words)


def _case_findings(state: CaseState) -> str:
    """Case facts for the criteria checkers: initial info, patient/test responses and the findings ledger (the doctor's
    own questions are left out so that "입원한 적 있나요?" is not read as a finding)."""
    return "\n".join([state.initial_info] + [t.response for t in state.turns] + [state.findings.render(exclude_unverified=True)])


def _criteria_refinement(proposed: str, refined: str, case_text: str) -> str | None:
    """Published criteria decide a subtype refinement when the findings explicitly satisfy one (knowledge/
    diagnostic_criteria.py): "" = accept, a reason = refuse, None = no criteria opinion (the usual checks apply).
    Only for a set that also covers the proposed name (same disease family or a listed cross-check such as arterial
    stenosis → Takayasu), and never when the new name adds a location qualifier."""
    if set(_LOCATION.findall(refined)) - set(_LOCATION.findall(proposed)):
        return None
    family = {c.id for c in diagnostic_criteria.criteria_for(proposed)}
    for c in diagnostic_criteria.criteria_for(refined, related=False):
        if c.id not in family:
            continue
        result = diagnostic_criteria.evaluate(c.id, case_text)
        if not result.decision:
            continue
        if other := diagnostic_criteria.conflicting_subtype(c, result.decision, refined):
            return f"진단 기준({c.short})상 {result.decision}에 해당 ({other} 아님)"
        if diagnostic_criteria.decision_matches(c, result.decision, refined):
            return ""
    return None


def _refinement_problem(proposed: str, refined: str, evidence: str, case_text: str) -> str:
    """Why the reviewer's renaming must be refused ("" = accept). A renaming is kept only when it is backed by a cited
    finding of this case. Location qualifiers are always refused (the standard disease name is preferred); a cause or
    trigger qualifier is refused unless the qualifier itself is in the case findings and in the cited evidence; adding a
    manifestation to a cause diagnosis ("B12 결핍" → "B12 결핍에 의한 인지장애") is refused. A published criteria set
    whose subtype rule is explicitly met (or contradicted) by the findings overrides these checks."""
    if (by_criteria := _criteria_refinement(proposed, refined, case_text)) is not None:
        return by_criteria
    if not _meaningful(evidence):
        return "근거 인용 없음"
    if not _grounded(evidence, case_text):
        return "인용한 근거가 이 환자의 소견에 없음"
    if set(_LOCATION.findall(refined)) - set(_LOCATION.findall(proposed)):
        return "위치 수식어 추가 (표준 질환명 유지)"
    markers = {m.group(0) for m in _CAUSE.finditer(refined)} - {m.group(0) for m in _CAUSE.finditer(proposed)}
    if markers:
        p, r = _norm(proposed), _norm(refined)
        if p in r and any(_norm(m) in r[r.index(p):] for m in markers):
            return "원인 진단에 증상·합병증을 덧붙임"
        base = set(_words(proposed))
        quals = [q for q in (_CAUSE.sub("", w) for w in _words(refined) if w not in base) if len(q) >= 2]
        if not quals or not all(_grounded(q, case_text) and _grounded(q, evidence) for q in quals):
            return "원인·유발 요인 수식어가 소견으로 뒷받침되지 않음"
    return ""
