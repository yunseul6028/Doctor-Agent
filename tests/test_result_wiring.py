"""Result interpreter wired into the policy: code reading of EXAM/TEST results → findings ledger, next-prompt hint,
one-time critical-result alert, can't-miss gate. Scripted fake LLM (no real calls)."""
import json
from pathlib import Path

import pytest

from doctor_agent.agent import prompts, result_interpreter
from doctor_agent.agent.ledger import CODE_SOURCE, Finding, FindingsLedger
from doctor_agent.agent.policy import Policy, _interp_findings
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.safety import danger_gate

ROOT = Path(__file__).resolve().parents[1]
PTX = "우측 기흉 관찰됨, 종격동 이동 없음"


class Recording:
    def __init__(self, *outputs):
        self.outputs, self.call_count, self.messages = list(outputs), 0, []

    def chat(self, messages):
        self.messages.append(messages)
        self.call_count += 1
        return self.outputs[min(self.call_count - 1, len(self.outputs) - 1)]

    def user(self, i=-1) -> str:
        return self.messages[i][1]["content"]


def _step(type_, content, findings=None):
    return json.dumps({"type": type_, "content": content, "reason": "r", "confidence": 0.5, "findings": findings or [],
                       "ddx": [{"dx": "A", "p": 0.5}, {"dx": "B", "p": 0.4}]}, ensure_ascii=False)


ASK = _step("ASK", "언제부터 아팠나요?")


def _cfg(**over):
    cfg = Config().agent
    cfg.use_kb = False
    cfg.use_danger_gate = cfg.use_preconditions = cfg.use_grounding = False
    cfg.use_confidence = cfg.use_anchoring = cfg.use_planner = cfg.use_triage = False
    cfg.use_result_interpreter = True
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _state(*results, initial="30세 남성. 주호소: 흉통"):
    st = CaseState(initial_info=initial)
    for name, text in results:
        st.turns.append(Turn(Action(ActionType.TEST, name), text))
    return st


def _layers(st, kind=None):
    return [e for e in st.safety_log if e["layer"] == "result_interp" and (kind is None or e.get("kind") == kind)]


# ---------------------------------------------------------------- switch on / off

def test_reading_goes_to_turn_ledger_hint_alert_and_log():
    st, llm = _state(("흉부 X선", PTX)), Recording(ASK)
    act = Policy(llm, _cfg()).next_action(st)
    assert act.type == ActionType.ASK and llm.call_count == 1
    user = llm.user()
    # critical result: top-of-prompt alert (triage slot), above the case view
    assert user.startswith("⚠ 우선 확인: 즉시 조치가 필요한 결과: 기흉") and user.index("⚠") < user.index("[처음 정보]")
    # the reading of the latest result is a hint
    assert "\n- " + prompts.RESULT_HINT.split("{")[0] + "[흉부 X선]" in user
    interp = st.turns[0].interp
    assert interp["kind"] == "imaging" and any(i["concept"] == "IMG:cxr_ptx" for i in interp["items"])
    f = next(x for x in st.findings.items if x.item == "기흉")
    assert f.status == "양성" and f.source == CODE_SOURCE and f.verified is True and "위급" in f.detail
    assert [c["concept"] for c in st.result_criticals] == ["IMG:cxr_ptx"]
    assert _layers(st, "reading")[0]["turn"] == 1 and _layers(st, "critical_alert")
    assert st.interp_stats["n"] == 1 and st.interp_stats["critical"] == 1


def test_switch_off_does_nothing():
    st, llm = _state(("흉부 X선", PTX)), Recording(ASK)
    Policy(llm, _cfg(use_result_interpreter=False)).next_action(st)
    assert "판독" not in llm.user() and "즉시 조치" not in llm.user()
    assert st.turns[0].interp is None and not st.findings.items and not _layers(st) and st.interp_stats == {}
    assert not st.result_criticals


def test_env_switch(monkeypatch):
    monkeypatch.setenv("AGENT_USE_RESULT_INTERPRETER", "0")
    assert Config().agent.use_result_interpreter is False
    monkeypatch.delenv("AGENT_USE_RESULT_INTERPRETER")
    assert Config().agent.use_result_interpreter is True


def test_ask_turns_are_not_read_and_each_result_is_read_once():
    st = _state(("혈액검사", "WBC 15,000/μL"))
    st.turns.append(Turn(Action(ActionType.ASK, "열이 있나요?"), "네, 38도요"))
    pol = Policy(Recording(ASK, ASK), _cfg())
    pol.next_action(st)
    pol.next_action(st)
    assert st.turns[1].interp is None and st.interp_stats["n"] == 1 and len(_layers(st, "reading")) == 1


# ---------------------------------------------------------------- critical alert once

def test_critical_alert_shown_once_per_finding():
    st, llm = _state(("흉부 X선", PTX)), Recording(ASK, ASK, ASK)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    assert "즉시 조치가 필요한 결과" in llm.user(0)
    st.turns.append(Turn(Action(ActionType.ASK, "숨이 찬가요?"), "네"))
    pol.next_action(st)
    assert "즉시 조치가 필요한 결과" not in llm.user(1) and "판독" not in llm.user(1)  # hint only right after the result
    st.turns.append(Turn(Action(ActionType.TEST, "혈액검사"), "칼륨 6.8 mEq/L"))
    pol.next_action(st)
    assert "즉시 조치가 필요한 결과: 고칼륨혈증" in llm.user(2) and "기흉" not in llm.user(2).split("\n")[0]
    assert len(_layers(st, "critical_alert")) == 2


def test_critical_alert_joins_triage_alert_and_survives_low_time_mode():
    unstable = ("70세 남성. 주호소: 1시간 전부터 시작된 가슴 통증. 혈압 70/40 mmHg, 맥박 130회/분, 호흡수 28회/분, "
                "체온 36.5도, 산소포화도 88%")
    st, llm = _state(("흉부 X선", PTX), initial=unstable), Recording(ASK)
    pol = Policy(llm, _cfg(use_triage=True))
    pol.degraded = True
    pol.next_action(st)
    first = llm.user().split("\n")[0]
    assert first.startswith("⚠ 우선 확인: [중증도") and " / 즉시 조치가 필요한 결과: 기흉" in first
    assert "검사 결과 판독" not in llm.user()  # low-time mode: hints cut, alert kept


# ---------------------------------------------------------------- not provided / pending are not normal

@pytest.mark.parametrize("text,detail", [("결과가 제공되지 않습니다.", "결과 제공 안 됨"), ("결과 대기 중입니다.", "결과 대기 중")])
def test_not_provided_or_pending_is_not_recorded_as_normal(text, detail):
    st, llm = _state(("복부 CT", text)), Recording(ASK)
    Policy(llm, _cfg()).next_action(st)
    assert [(f.item, f.status, f.detail) for f in st.findings.items] == [("복부 CT", "결과없음", detail)]
    assert not any(f.status == "음성" for f in st.findings.items)
    assert "정상으로 보지 말 것" in llm.user() or "정상 아님" in llm.user()


def test_ledger_mapping_normal_uncertain_absent():
    ok = _interp_findings(result_interpreter.interpret("흉부 X선", "특이 소견 없음"), 1)
    assert [(f.item, f.status) for f in ok] == [("흉부 X선", "음성")]
    ex = _interp_findings(result_interpreter.interpret("복부 진찰", "우하복부 압통 있음, 반발통 의심"), 2)
    assert ("우하복부 압통", "양성") in [(f.item, f.status) for f in ex]
    reb = next(f for f in ex if f.item == "반발통")
    assert reb.status == "양성" and "의심" in reb.detail
    vit = _interp_findings(result_interpreter.interpret("활력징후", "혈압 120/80, 맥박 88"), 3)
    assert [(f.item, f.status, f.detail) for f in vit] == [("혈압 120/80", "음성", "정상"), ("맥박 88", "음성", "정상")]
    assert all(f.source == CODE_SOURCE and f.verified for f in ok + ex + vit)


def test_model_report_does_not_overwrite_code_reading_of_the_same_turn():
    led = FindingsLedger()
    led.add(Finding("기흉", "양성", "우측", 3, verified=True, source=CODE_SOURCE))
    led.update([{"item": "기흉", "status": "음성"}], 3)
    assert led.items[0].status == "양성" and led.items[0].source == CODE_SOURCE
    led.update([{"item": "기흉", "status": "음성"}], 5)  # later information still wins
    assert led.items[0].status == "음성" and led.items[0].source == ""


def test_grounding_keeps_code_findings_verified():
    from doctor_agent.agent import grounding
    st = _state(("흉부 X선", PTX))
    st.findings.add(Finding("아무 말이나 적힌 항목", "양성", "", 1, verified=True, source=CODE_SOURCE))
    grounding.apply(st)
    assert st.findings.items[0].verified is True


# ---------------------------------------------------------------- guarded

def _boom(*a, **k):
    raise RuntimeError("interp bug")


def test_interpreter_exception_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(result_interpreter, "interpret", _boom)
    st, llm = _state(("흉부 X선", PTX)), Recording(ASK)
    act = Policy(llm, _cfg()).next_action(st)
    assert act.type == ActionType.ASK and llm.call_count == 1
    assert any("interp bug" in e.get("error", "") for e in _layers(st)) and st.interp_stats["errors"] == 1


def test_whole_case_with_interpreter_raising(monkeypatch):
    from doctor_agent.agent.loop import run_case
    from doctor_agent.llm.client import DummyLLM
    from eval.simulator import CaseFileEnvironment

    for attr in ("interpret", "render_for_prompt", "llm_reasons"):
        monkeypatch.setattr(result_interpreter, attr, _boom)
    cfg = Config()
    cfg.agent.use_result_interpreter = True
    res = run_case(CaseFileEnvironment.from_file(ROOT / "data/sample_cases/synthetic_001.json"), DummyLLM(), cfg)
    assert res["diagnosis"] and res["llm_calls"] >= 1 and "result_interp" in res


def test_run_case_reports_readings_and_needs_llm_count():
    from doctor_agent.agent.loop import run_case
    from doctor_agent.llm.client import DummyLLM
    from eval.simulator import CaseFileEnvironment

    cfg = Config()
    cfg.agent.use_result_interpreter = True
    res = run_case(CaseFileEnvironment.from_file(ROOT / "data/sample_cases/synthetic_001.json"), DummyLLM(), cfg)
    stats = res["result_interp"]
    assert set(stats) >= {"n", "needs_llm", "llm_reasons", "critical", "unavailable", "errors"}
    n_read = sum(1 for t in res["turns"] if "interp" in t)
    assert stats["n"] == n_read and stats["needs_llm"] <= stats["n"]
    assert json.dumps(res, ensure_ascii=False)  # serialisable for eval/results + viewer


# ---------------------------------------------------------------- budget

def test_result_hint_respects_budgets():
    long = ", ".join(f"소견{i} 있음" for i in range(80))
    st = _state(("흉부 CT", "우하엽 경화, 좌측 흉수, 심비대, 기흉 없음, 종격동 정상, " + long))
    pol = Policy(Recording(ASK), _cfg())
    pol._interpret_results(st)
    text, _ = pol._result_hint(st)
    assert text and len(text) <= pol.cfg.result_hint_chars
    for budget in (0, 100, 900):
        st = _state(("흉부 X선", PTX))
        pol = Policy(Recording(ASK), _cfg(max_advisor_chars=budget, use_anchoring=True))
        pol._interpret_results(st)
        alert, hints = pol._advisors(st)
        assert alert  # the alert is never dropped
        assert len(alert) + sum(len(h) for h in hints) <= max(budget, len(alert))


# ---------------------------------------------------------------- can't-miss gate sees critical results

def test_danger_gate_confirms_danger_from_critical_result():
    assert set(danger_gate.CRITICAL_RESULT_DANGER) <= result_interpreter._CRITICAL_CONCEPTS
    assert set(danger_gate.CRITICAL_RESULT_DANGER.values()) <= set(danger_gate.RULE_OUT)
    st = _state(("흉부 X선", "소견 설명 없음"), initial="30세 남성. 주호소: 기침")
    assert danger_gate.status("긴장성 기흉", st)[0] != "confirmed"
    st.result_criticals.append({"turn": 1, "test": "흉부 X선", "concept": "IMG:cxr_ptx", "label": "기흉",
                                "polarity": "present", "summary": "기흉(우측): 있음"})
    st_, ev = danger_gate.status("긴장성 기흉", st)
    assert st_ == "confirmed" and "기흉" in ev[0]
    g = danger_gate.gate(st, "폐렴", remaining_turns=30)
    assert g["allow"] and g["kind"] == "confirmed_other" and g["danger"] == "긴장성 기흉"
    st.result_criticals[0]["polarity"] = "uncertain"  # hedged readings do not confirm
    assert danger_gate.status("긴장성 기흉", st)[0] != "confirmed"


# ---------------------------------------------------------------- prompt wording / tools

def test_prompt_wording_and_version():
    assert prompts.PROMPT_VERSION == "v8-result-interp"
    assert prompts.result_hint("X").startswith(prompts.RESULT_HINT.split("{")[0])
    assert prompts.result_critical_alert(["a", "b"]).startswith("즉시 조치가 필요한 결과: a, b")
