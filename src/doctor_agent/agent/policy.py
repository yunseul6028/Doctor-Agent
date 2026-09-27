import re

from doctor_agent.agent import grounding, kb_hints, prompts
from doctor_agent.agent.parser import _json_objects, extract_action_json, parse_action
from doctor_agent.agent.state import CaseState
from doctor_agent.config import AgentConfig
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.llm.client import LLMClient
from doctor_agent.knowledge import clinical_rules, diagnostic_criteria
from doctor_agent.safety import danger_gate, preconditions, protocols

MAX_RULES_IN_PROMPT = 2  # keep prompts short for the small fixed LLM
MAX_ATTEMPTS = 4
MAX_REVIEWS = 2  # pre-diagnosis reviews per case (each hold costs a turn)
MAX_HINTS_DEGRADED = 2  # hints kept when the time budget runs low (can't-miss + pending safety checks come first)
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

    def next_action(self, state: CaseState) -> Action:
        remaining = self.cfg.max_turns - state.turn_count
        if remaining <= 1:
            return self._final_diagnosis(state)

        hints = self._hints(state)
        if self.degraded:  # time budget running low: short prompt, no extra review/pushback calls
            hints = hints[:MAX_HINTS_DEGRADED] + [prompts.LOW_TIME_HINT]
        for _ in range(MAX_ATTEMPTS):  # retries: parse failure, repeated action, safety pushback, review hold
            raw = self.llm.chat(prompts.build_step_messages(state.view(), state.turn_count, self.cfg.max_turns, hints))
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
            if action.type == ActionType.DIAGNOSE and len(state.reviews) < MAX_REVIEWS and remaining > 3 and not self.degraded:
                reviewed = self._review(state, action)
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

    def _review(self, state: CaseState, action: Action) -> Action:
        """Pre-diagnosis review by the same LLM in a reviewer role. The reviewer only fills structured fields
        (key findings explained or not, contradictions, confirmatory evidence, unresolved dangers); the verdict and
        any renaming are decided here in code. Returns the (possibly refined) diagnosis, or the reviewer's next
        action on hold."""
        view = state.view()
        if self.cfg.use_kb and (warn := kb_hints.normalize_hint(action.content, state.initial_info).get("warning")):
            view += "\n\n[지식베이스 경고] " + warn
        if criteria := diagnostic_criteria.render_for_review(action.content, _case_findings(state)):
            view += "\n\n" + criteria
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
