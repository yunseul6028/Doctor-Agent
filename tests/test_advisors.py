"""Advisors wired into the policy: code-confidence pushback, starting DDx + anchoring check, question planner, triage.
Scripted fake LLM (no real calls); each advisor is switched on/off and made to raise."""
import json

import pytest

from doctor_agent.agent import anchoring, confidence, prompts, question_planner
from doctor_agent.agent.confidence import Assessment
from doctor_agent.agent.policy import MAX_REVIEWS, Policy
from doctor_agent.agent.question_planner import Suggestion
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.safety import triage

UNSTABLE = ("70세 남성. 주호소: 1시간 전부터 시작된 가슴 통증. 혈압 70/40 mmHg, 맥박 130회/분, 호흡수 28회/분, "
            "체온 36.5도, 산소포화도 88%")
UNKNOWN_RED = "61세 남성. 주호소: 40분 전부터 시작된 가슴 통증"
CONCERNING = "45세 남성. 주호소: 3일 전부터 기침과 발열. 혈압 118/76 mmHg, 맥박 118회/분, 호흡수 23회/분, 체온 38.9도, 산소포화도 95%"
STABLE = "25세 여성. 주호소: 2주 전부터 생긴 손목 통증. 혈압 120/80 mmHg, 맥박 72회/분, 호흡수 16회/분, 체온 36.8도"


class Recording:
    def __init__(self, *outputs):
        self.outputs, self.call_count, self.messages = list(outputs), 0, []

    def chat(self, messages):
        self.messages.append(messages)
        self.call_count += 1
        return self.outputs[min(self.call_count - 1, len(self.outputs) - 1)]

    def user(self, i=0) -> str:
        return self.messages[i][1]["content"]


def _step(type_, content):
    return json.dumps({"type": type_, "content": content, "reason": "r", "confidence": 0.5,
                       "ddx": [{"dx": "A", "p": 0.5}, {"dx": "B", "p": 0.4}]}, ensure_ascii=False)


ASK = _step("ASK", "언제부터 아팠나요?")
DX = _step("DIAGNOSE", "A")


def _cfg(**over):
    cfg = Config().agent
    # isolate the advisors from the safety layers / KB hints (tested elsewhere); advisors on unless overridden
    cfg.use_kb = False
    cfg.use_danger_gate = cfg.use_preconditions = cfg.use_grounding = False
    cfg.use_confidence = cfg.use_anchoring = cfg.use_planner = cfg.use_triage = True
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _state(initial="30세 남성. 주호소: 두통", n_turns=0, **flags):
    st = CaseState(initial_info=initial)
    for i in range(n_turns):
        st.turns.append(Turn(Action(ActionType.ASK, f"질문 {i}번째 내용"), "네"))
    for k, v in flags.items():
        setattr(st, k, v)
    return st


def _layers(st, layer):
    return [e for e in st.safety_log if e["layer"] == layer]


# ---------------------------------------------------------------- triage

def test_triage_unstable_alert_at_top_of_prompt():
    st = _state(UNSTABLE)
    llm = Recording(ASK)
    Policy(llm, _cfg()).next_action(st)
    user = llm.user()
    assert user.startswith("⚠ 우선 확인: [중증도: 불안정]") and user.index("⚠") < user.index("[처음 정보]")
    e = _layers(st, "triage")[0]
    assert e["level"] == "unstable" and e["shown"] == "top"


def test_triage_unknown_with_red_flag_goes_on_top_and_concerning_is_a_hint():
    st, llm = _state(UNKNOWN_RED), Recording(ASK)
    Policy(llm, _cfg()).next_action(st)
    assert llm.user().startswith("⚠ 우선 확인: [중증도: 미확인]")
    st, llm = _state(CONCERNING), Recording(ASK)
    Policy(llm, _cfg()).next_action(st)
    assert not llm.user().startswith("⚠") and "\n- [중증도: 주의]" in llm.user()
    assert _layers(st, "triage")[0]["shown"] == "hint"


def test_triage_stable_adds_nothing_and_logs_level_once():
    st, llm = _state(STABLE), Recording(ASK)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    st.turns.append(Turn(Action(ActionType.ASK, "언제부터 아팠나요?"), "2주 전부터요"))
    pol.next_action(st)
    assert "중증도" not in llm.user(0) and "중증도" not in llm.user(1)
    assert [e["level"] for e in _layers(st, "triage")] == ["stable"]  # logged on change only


def test_triage_off_switch():
    st, llm = _state(UNSTABLE), Recording(ASK)
    Policy(llm, _cfg(use_triage=False)).next_action(st)
    assert "중증도" not in llm.user() and not _layers(st, "triage")


def test_triage_alert_survives_low_time_mode():
    st, llm = _state(UNSTABLE), Recording(ASK)
    pol = Policy(llm, _cfg())
    pol.degraded = True
    pol.next_action(st)
    assert llm.user().startswith("⚠ 우선 확인") and "[초기 감별 목록" not in llm.user()


# ---------------------------------------------------------------- starting DDx + anchoring check

def test_initial_ddx_hint_turn_1_only():
    st, llm = _state("61세 남성. 주호소: 40분 전부터 시작된 가슴 통증"), Recording(ASK, ASK)
    pol = Policy(llm, _cfg(use_triage=False))
    pol.next_action(st)
    assert "[초기 감별 목록" in llm.user(0)
    st.turns.append(Turn(Action(ActionType.ASK, "언제부터 아팠나요?"), "40분 전부터요"))
    pol.next_action(st)
    assert "[초기 감별 목록" not in llm.user(1)
    assert len([e for e in _layers(st, "anchoring") if e["kind"] == "initial_ddx"]) == 1


def test_initial_ddx_off_switch():
    st, llm = _state("61세 남성. 주호소: 40분 전부터 시작된 가슴 통증"), Recording(ASK)
    Policy(llm, _cfg(use_anchoring=False)).next_action(st)
    assert "[초기 감별 목록" not in llm.user() and not _layers(st, "anchoring")


def _fake_check(calls):
    def check(state, **kw):
        calls.append(state.turn_count)
        return {"dx": "A", "p": 0.8, "reasons": ["weak_support"], "why_ko": "근거 부족", "prompt_ko": "[반론 점검] A 고정?",
                "suggested_actions": []}
    return check


def test_anchoring_check_from_min_turns_shown_once(monkeypatch):
    calls = []
    monkeypatch.setattr(anchoring, "anchoring_check", _fake_check(calls))
    llm = Recording(ASK)
    pol = Policy(llm, _cfg(use_triage=False))
    st = _state(n_turns=4)
    pol.next_action(st)
    assert calls == [] and "[반론 점검]" not in llm.user(0)  # not before cfg.anchoring_min_turns (5)
    st.turns.append(Turn(Action(ActionType.ASK, "다섯 번째 질문입니다"), "네"))
    pol.next_action(st)
    assert calls == [5] and "[반론 점검] A 고정?" in llm.user(1) and st.anchoring_shown
    st.turns.append(Turn(Action(ActionType.ASK, "여섯 번째 질문입니다"), "네"))
    pol.next_action(st)
    assert calls == [5] and "[반론 점검]" not in llm.user(2)
    assert [e["kind"] for e in _layers(st, "anchoring")] == ["premature_closure"]


def test_anchoring_check_min_turns_from_config(monkeypatch):
    seen = []
    monkeypatch.setattr(anchoring, "anchoring_check", lambda s, min_turns=None: seen.append((s.turn_count, min_turns)))
    Policy(Recording(ASK), _cfg(use_triage=False, anchoring_min_turns=3)).next_action(_state(n_turns=3))
    assert seen == [(3, 3)]


def test_anchoring_check_keeps_trying_until_it_fires(monkeypatch):
    results = [None, {"prompt_ko": "[반론 점검] 지금", "dx": "A"}]
    monkeypatch.setattr(anchoring, "anchoring_check", lambda s, **kw: results.pop(0))
    llm, st = Recording(ASK), _state(n_turns=5)
    pol = Policy(llm, _cfg(use_triage=False))
    pol.next_action(st)
    assert not st.anchoring_shown
    st.turns.append(Turn(Action(ActionType.ASK, "네 번째 질문입니다"), "네"))
    pol.next_action(st)
    assert st.anchoring_shown and "[반론 점검] 지금" in llm.user(1)


def test_anchoring_check_off_switch(monkeypatch):
    calls = []
    monkeypatch.setattr(anchoring, "anchoring_check", _fake_check(calls))
    Policy(Recording(ASK), _cfg(use_anchoring=False)).next_action(_state(n_turns=6))
    assert calls == []


def test_anchoring_hint_dropped_by_budget_is_not_marked_shown(monkeypatch):
    monkeypatch.setattr(anchoring, "anchoring_check", _fake_check([]))
    llm, st = Recording(ASK), _state(n_turns=5)
    Policy(llm, _cfg(use_triage=False, max_advisor_chars=5)).next_action(st)
    assert "[반론 점검]" not in llm.user() and not st.anchoring_shown and not _layers(st, "anchoring")


# ---------------------------------------------------------------- question planner

def _fake_suggest(calls):
    def suggest(state, k=3, include_safety=True):
        calls.append(k)
        return [Suggestion("ASK", "열이 나나요?", ["A", "B"], 0.5), Suggestion("TEST", "심전도", ["A"], 0.4)]
    return suggest


def test_planner_hint_each_turn_when_kb_on(monkeypatch):
    calls = []
    monkeypatch.setattr(question_planner, "suggest", _fake_suggest(calls))
    llm, st = Recording(ASK), _state()
    pol = Policy(llm, _cfg(use_kb=True, use_triage=False, use_anchoring=False))
    pol.next_action(st)
    st.turns.append(Turn(Action(ActionType.ASK, "언제부터 아팠나요?"), "어제부터요"))
    pol.next_action(st)
    assert calls == [3, 3]
    assert all("추천 다음 행동 (참고): 1) [문진] 열이 나나요?" in llm.user(i) for i in (0, 1))
    assert len(_layers(st, "planner")) == 1  # same suggestions: logged once


@pytest.mark.parametrize("over", [{"use_planner": False, "use_kb": True}, {"use_planner": True, "use_kb": False}])
def test_planner_off_switch_and_needs_kb(monkeypatch, over):
    calls = []
    monkeypatch.setattr(question_planner, "suggest", _fake_suggest(calls))
    llm = Recording(ASK)
    Policy(llm, _cfg(**over)).next_action(_state())
    assert calls == [] and "추천 다음 행동" not in llm.user()


# ---------------------------------------------------------------- confidence pushback

def _fake_assess(calls, score=0.1, rec="continue"):
    def assess(state, proposed_dx=None, cfg=None, params=None):
        calls.append(proposed_dx)
        return Assessment(score, {"margin": 0.1}, rec, ["확신도 낮음", "감별 우위 0.10", "확진 근거 없음"], proposed_dx or "")
    return assess


def _dx_state(**flags):
    # review and protocol pushback already used up, so only the confidence pushback is in play
    return _state(n_turns=4, safety_pushback=True, reviews=[{}] * MAX_REVIEWS, **flags)


def test_confidence_pushback_fires_once(monkeypatch):
    calls = []
    monkeypatch.setattr(confidence, "assess", _fake_assess(calls))
    llm, st = Recording(DX, DX, DX), _dx_state()
    pol = Policy(llm, _cfg())
    act = pol.next_action(st)
    assert act.type == ActionType.DIAGNOSE and llm.call_count == 2 and calls == ["A"]
    assert "확신도가 낮습니다(0.10" in llm.user(1) and "확신도가 낮습니다" not in llm.user(0)
    assert st.confidence_pushback
    act = pol.next_action(st)  # a later diagnosis in the same case is not pushed back again
    assert act.type == ActionType.DIAGNOSE and llm.call_count == 3 and calls == ["A"]
    e = _layers(st, "confidence")
    assert len(e) == 1 and e[0]["pushback"] and e[0]["score"] == 0.1


def test_confidence_high_score_or_must_continue_does_not_push_back(monkeypatch):
    for score, rec in ((0.9, "diagnose"), (0.1, "must_continue")):
        monkeypatch.setattr(confidence, "assess", _fake_assess([], score, rec))
        llm, st = Recording(DX), _dx_state()
        assert Policy(llm, _cfg()).next_action(st).type == ActionType.DIAGNOSE and llm.call_count == 1
        assert not st.confidence_pushback and _layers(st, "confidence")[0]["pushback"] is False


def test_confidence_respects_turn_and_time_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(confidence, "assess", _fake_assess(calls))
    cfg = _cfg()
    late = _state(n_turns=cfg.max_turns - 3, safety_pushback=True, reviews=[{}] * MAX_REVIEWS)
    llm = Recording(DX)
    assert Policy(llm, cfg).next_action(late).type == ActionType.DIAGNOSE and llm.call_count == 1
    pol = Policy(Recording(DX), cfg)
    pol.degraded = True  # low-time mode
    assert pol.next_action(_dx_state()).type == ActionType.DIAGNOSE and pol.llm.call_count == 1
    assert calls == []


def test_confidence_leaves_room_in_the_retry_budget(monkeypatch):
    """Two parse failures first: no pushback on the 3rd attempt, so the diagnosis is not lost to the attempt cap."""
    calls = []
    monkeypatch.setattr(confidence, "assess", _fake_assess(calls))
    llm = Recording("??", "??", DX)
    act = Policy(llm, _cfg()).next_action(_dx_state())
    assert act.type == ActionType.DIAGNOSE and act.content == "A" and calls == []


def test_confidence_off_switch(monkeypatch):
    calls = []
    monkeypatch.setattr(confidence, "assess", _fake_assess(calls))
    llm = Recording(DX)
    assert Policy(llm, _cfg(use_confidence=False)).next_action(_dx_state()).type == ActionType.DIAGNOSE
    assert calls == [] and llm.call_count == 1


def test_confidence_real_module_runs():
    st = _dx_state()
    Policy(Recording(DX, DX), _cfg()).next_action(st)
    e = _layers(st, "confidence")[0]
    assert 0.0 <= e["score"] <= 1.0 and e["recommendation"] in ("diagnose", "continue", "must_continue")


# ---------------------------------------------------------------- failures never break the case

def _boom(*a, **k):
    raise RuntimeError("advisor bug")


@pytest.mark.parametrize("target,attr,layer", [
    (triage, "assess", "triage"),
    (anchoring, "initial_differential", "anchoring"),
    (question_planner, "suggest", "planner"),
])
def test_advisor_exception_is_logged_not_raised(monkeypatch, target, attr, layer):
    monkeypatch.setattr(target, attr, _boom)
    st, llm = _state(UNSTABLE), Recording(ASK)
    act = Policy(llm, _cfg(use_kb=True)).next_action(st)
    assert act.type == ActionType.ASK and llm.call_count == 1
    assert any(e["layer"] == layer and "advisor bug" in e.get("error", "") for e in st.safety_log)


def test_anchoring_check_exception_is_logged_once(monkeypatch):
    calls = []

    def boom(state, **kw):
        calls.append(1)
        raise RuntimeError("advisor bug")
    monkeypatch.setattr(anchoring, "anchoring_check", boom)
    st, pol = _state(n_turns=5), Policy(Recording(ASK), _cfg())
    pol.next_action(st)
    st.turns.append(Turn(Action(ActionType.ASK, "네 번째 질문입니다"), "네"))
    pol.next_action(st)
    assert calls == [1] and len([e for e in _layers(st, "anchoring") if e.get("error")]) == 1


def test_confidence_exception_does_not_block_diagnosis(monkeypatch):
    monkeypatch.setattr(confidence, "assess", _boom)
    st, llm = _dx_state(), Recording(DX)
    assert Policy(llm, _cfg()).next_action(st).type == ActionType.DIAGNOSE and llm.call_count == 1
    assert _layers(st, "confidence")[0]["error"]


def test_whole_case_with_every_advisor_raising(monkeypatch):
    """Dev mode (bugs would propagate out of run_case) still finishes the case with a diagnosis."""
    from pathlib import Path

    from doctor_agent.agent.loop import run_case
    from doctor_agent.llm.client import DummyLLM
    from eval.simulator import CaseFileEnvironment

    case = Path(__file__).resolve().parents[1] / "data/sample_cases/synthetic_001.json"

    for target, attr in ((triage, "assess"), (triage, "render_for_prompt"), (anchoring, "initial_differential"),
                         (anchoring, "anchoring_check"), (question_planner, "suggest"), (confidence, "assess")):
        monkeypatch.setattr(target, attr, _boom)
    cfg = Config()
    assert not cfg.agent.submission
    res = run_case(CaseFileEnvironment.from_file(case), DummyLLM(), cfg)
    assert res["diagnosis"] and res["llm_calls"] >= 1
    assert {e["layer"] for e in res["safety_log"] if e.get("error")} >= {"triage", "anchoring", "planner", "confidence"}


# ---------------------------------------------------------------- budget / prompt wording

def test_advisor_hints_fit_the_budget(monkeypatch):
    monkeypatch.setattr(question_planner, "suggest", _fake_suggest([]))
    for budget in (0, 200, 900):
        st, llm = _state("61세 남성. 주호소: 40분 전부터 시작된 가슴 통증"), Recording(ASK)
        pol = Policy(llm, _cfg(use_kb=True, max_advisor_chars=budget))
        alert, hints = pol._advisors(st)
        assert len(alert) + sum(len(h) for h in hints) <= max(budget, len(alert))


def test_prompt_wording_lives_in_prompts():
    msg = prompts.build_step_messages("VIEW", 0, 60, ["h"], alert="X")
    assert msg[1]["content"].startswith(prompts.TRIAGE_ALERT.format(text="X"))
    assert "0.12" in prompts.confidence_pushback("A", 0.123, ["r1", "r2"])
    assert prompts.build_step_messages("VIEW", 0, 60, [])[1]["content"].startswith("VIEW")
