import re

from doctor_agent.agent import prompts
from doctor_agent.agent.parser import _json_objects, extract_action_json, parse_action
from doctor_agent.agent.state import CaseState
from doctor_agent.config import AgentConfig
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.llm.client import LLMClient
from doctor_agent.knowledge import clinical_rules
from doctor_agent.safety import protocols

MAX_RULES_IN_PROMPT = 2  # keep prompts short for the small fixed LLM
MAX_ATTEMPTS = 4
MAX_REVIEWS = 2  # pre-diagnosis reviews per case (each hold costs a turn)
# a question sent as EXAM/TEST ("목이 뻣뻣한가요?") is really an ASK
_QUESTION_END = re.compile(r"(\?|？|나요|세요|니까|까요|있어요|없어요|하셨어요|드세요)\s*$")


def normalize_type(action: Action) -> Action:
    if action.type in (ActionType.EXAM, ActionType.TEST) and _QUESTION_END.search(action.content.strip()):
        return Action(ActionType.ASK, action.content, action.reason)
    return action


class Policy:
    """LLM proposes the action → rules enforce the budget, no repeats, and safety."""

    def __init__(self, llm: LLMClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg

    def next_action(self, state: CaseState) -> Action:
        remaining = self.cfg.max_turns - state.turn_count
        if remaining <= 1:
            return self._final_diagnosis(state)

        hints = self._hints(state)
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
            if action.type != ActionType.DIAGNOSE and state.asked(action):
                hints = hints + [f"'{action.content}'은(는) 이미 했습니다. 다른 행동을 고르세요."]
                continue
            if action.type == ActionType.DIAGNOSE and not state.safety_pushback and remaining > 5:
                pending = self._pending(state)
                if pending:
                    # one pushback per case: diagnose anyway only with a stated reason
                    state.safety_pushback = True
                    hints = hints + ["진단 전에 아직 안 한 필수 안전 확인이 있습니다: " + ", ".join(c.name for c in pending)
                                     + ". 이 중 하나를 먼저 하세요. 정말 불필요하면 reason에 그 이유를 쓰고 진단하세요."]
                    continue
            if action.type == ActionType.DIAGNOSE and len(state.reviews) < MAX_REVIEWS and remaining > 3:
                reviewed = self._review(state, action)
                if reviewed.type != ActionType.DIAGNOSE and state.asked(reviewed):
                    hints = hints + ["검토의 지적: " + "; ".join(state.reviews[-1]["issues"]) + ". 이를 해결할 다른 행동을 고르세요."]
                    continue
                return reviewed
            return action
        return self._final_diagnosis(state)

    def _review(self, state: CaseState, action: Action) -> Action:
        """Pre-diagnosis review by the same LLM in a reviewer role. Returns the (possibly refined) diagnosis or the
        reviewer's next action on hold."""
        raw = self.llm.chat(prompts.build_review_messages(state.view(), action.content, action.reason,
                                                          state.turn_count, self.cfg.max_turns))
        obj = next((o for o in reversed(_json_objects(raw or "")) if "verdict" in o), {})
        verdict = "보류" if str(obj.get("verdict", "")).strip() == "보류" else "승인"
        issues = [str(x) for x in obj.get("issues", []) if str(x).strip()] if isinstance(obj.get("issues"), list) else []
        state.reviews.append({"turn": state.turn_count + 1, "proposed": action.content, "verdict": verdict, "issues": issues})
        if verdict == "승인":
            refined = str(obj.get("final_diagnosis") or "").strip()
            if refined and refined != action.content:
                state.reviews[-1]["final_diagnosis"] = refined
                return Action(ActionType.DIAGNOSE, refined, f"{action.reason} (검토의가 더 구체적인 진단명으로 수정)")
            return action
        nxt = obj.get("next") if isinstance(obj.get("next"), dict) else {}
        try:
            follow = Action(ActionType(str(nxt.get("type", "")).upper()), str(nxt.get("content", "")).strip(),
                            "검토 보류: " + (str(nxt.get("reason", "")).strip() or "; ".join(issues)))
        except ValueError:
            follow = None
        if follow is None or follow.type == ActionType.DIAGNOSE or not follow.content:
            return Action(ActionType.DIAGNOSE, action.content, action.reason)  # unusable hold → keep the diagnosis
        return normalize_type(follow)

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
        return hints

    def _final_diagnosis(self, state: CaseState) -> Action:
        parsed = parse_action(self.llm.chat(prompts.build_final_messages(state.view())))
        if parsed and parsed[0].type == ActionType.DIAGNOSE:
            return parsed[0]
        top = state.ddx[0].get("dx") if state.ddx and isinstance(state.ddx[0], dict) else None
        return Action(ActionType.DIAGNOSE, top or "진단 불가")
