"""Integration of the safety layers into the policy (scripted fake LLM, no real calls)."""
import json

from doctor_agent.agent.policy import Policy
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType


class Scripted:
    def __init__(self, outputs):
        self.outputs, self.call_count = outputs, 0

    def chat(self, messages):
        self.call_count += 1
        return self.outputs[min(self.call_count - 1, len(self.outputs) - 1)]


def _cfg(**over):
    cfg = Config().agent
    cfg.use_kb = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _step(type_, content, **extra):
    return json.dumps({"type": type_, "content": content, "reason": "r", "confidence": 0.9, **extra}, ensure_ascii=False)


def test_precondition_swaps_ct_for_pregnancy_test():
    st = CaseState(initial_info="26세 여성. 주호소: 어제부터 시작된 우하복부 통증")
    act = Policy(Scripted([_step("TEST", "복부 CT")]), _cfg()).next_action(st)
    assert act.type == ActionType.TEST and ("임신" in act.content or "hcg" in act.content.lower())
    assert any(e["layer"] == "preconditions" for e in st.safety_log)


def test_precondition_off_switch():
    st = CaseState(initial_info="26세 여성. 주호소: 어제부터 시작된 우하복부 통증")
    act = Policy(Scripted([_step("TEST", "복부 CT")]), _cfg(use_preconditions=False)).next_action(st)
    assert act.content == "복부 CT"


def test_danger_gate_forces_rule_out_before_diagnosis():
    st = CaseState(initial_info="61세 남성. 주호소: 30분 전부터 시작된 가슴 통증")
    st.safety_pushback = True  # isolate the gate from the protocol pushback
    act = Policy(Scripted([_step("DIAGNOSE", "위식도 역류질환")]), _cfg()).next_action(st)
    assert act.type != ActionType.DIAGNOSE and st.gate_turns == 1
    assert any(e["layer"] == "danger_gate" and e.get("kind") == "rule_out" for e in st.safety_log)


def test_danger_gate_respects_turn_cap():
    st = CaseState(initial_info="61세 남성. 주호소: 30분 전부터 시작된 가슴 통증")
    st.safety_pushback, st.gate_turns = True, 3
    cfg = _cfg()
    outputs = [_step("DIAGNOSE", "위식도 역류질환"),
               json.dumps({"key_findings": [], "contradicting": [], "confirmation": "내시경", "unresolved_danger": []},
                          ensure_ascii=False)]
    act = Policy(Scripted(outputs), cfg).next_action(st)
    assert act.type == ActionType.DIAGNOSE


def test_grounding_marks_unverified_findings():
    st = CaseState(initial_info="30세 남성. 주호소: 기침")
    st.turns.append(Turn(Action(ActionType.ASK, "열이 있나요?"), "아니요, 열은 없어요."))
    out = _step("ASK", "가래가 있나요?", findings=[{"item": "발열", "status": "양성"}])
    Policy(Scripted([out]), _cfg()).next_action(st)
    f = st.findings.items[0]
    assert f.verified is False and "(미확인)" in st.findings.render()


def test_safety_layers_never_crash(monkeypatch):
    from doctor_agent.safety import preconditions

    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(preconditions, "check", boom)
    st = CaseState(initial_info="26세 여성. 주호소: 복통")
    act = Policy(Scripted([_step("TEST", "복부 CT")]), _cfg()).next_action(st)
    assert act.content == "복부 CT" and st.safety_log[-1].get("error")
