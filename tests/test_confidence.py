"""Code-computed confidence and stop rule (agent/confidence.py). No LLM calls."""
import json
import sys
from pathlib import Path

from doctor_agent.agent import confidence as C
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import AgentConfig
from doctor_agent.env.interface import Action, ActionType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_confidence as CC  # noqa: E402  (offline calibration, not shipped)


def _state(turns, ddx, initial="28세 남성. 주호소: 어제부터 시작된 복통") -> CaseState:
    st = CaseState(initial)
    for typ, content, resp in turns:
        st.turns.append(Turn(Action(ActionType(typ), content), resp))
    st.ddx_ledger.update(ddx)
    st.ddx = st.ddx_ledger.as_list()
    return st


APPENDICITIS_TURNS = [
    ("ASK", "통증이 어디서 시작됐나요?", "배꼽 주위에서 시작해서 오른쪽 아랫배로 옮겨갔어요."),
    ("EXAM", "복부 촉진", "우하복부 압통과 반발통이 있습니다."),
    ("TEST", "복부 CT", "복부 CT: 충수 직경 12mm로 비후, 주위 지방 침윤. 급성 충수염 소견."),
]


def test_confident_case_scores_high_and_diagnoses():
    st = _state(APPENDICITIS_TURNS, [
        {"dx": "급성 충수염", "p": 0.95, "for": ["우하복부 압통", "반발통", "충수 비후"], "against": []},
        {"dx": "게실염", "p": 0.05}])
    a = C.assess(st, "급성 충수염", AgentConfig())
    assert 0.0 <= a.score <= 1.0
    assert set(C.FEATURES) <= set(a.components)
    assert a.components["margin"] == 0.9
    assert a.components["verified_support"] > 0
    assert a.components["confirmatory_test"] == 1.0
    assert a.recommendation == "diagnose", a
    assert a.reasons_ko and all(isinstance(r, str) for r in a.reasons_ko)


def test_close_ddx_with_contradiction_continues():
    st = _state(APPENDICITIS_TURNS[:1], [
        {"dx": "급성 충수염", "p": 0.4, "for": ["우하복부 통증"], "against": ["배꼽 주위에서 시작"]},
        {"dx": "게실염", "p": 0.35}])
    a = C.assess(st, "급성 충수염", AgentConfig())
    assert a.components["margin"] < 0.1
    assert a.recommendation == "continue"
    hi = C.assess(_state(APPENDICITIS_TURNS, [{"dx": "급성 충수염", "p": 0.95, "for": ["우하복부 압통"]}]), "급성 충수염")
    assert a.score < hi.score


def test_ungrounded_support_does_not_count():
    st = _state(APPENDICITIS_TURNS[:1], [{"dx": "급성 충수염", "p": 0.9, "for": ["백혈구 18000", "발열 39도"]}])
    comp, det = C.features(st, "급성 충수염")
    assert comp["verified_support"] == 0.0 and det["support"] == []


def test_unresolved_cant_miss_danger_forces_continue():
    st = _state([("ASK", "통증이 어떤가요?", "가슴을 쥐어짜는 듯한 통증이 30분째 계속돼요.")],
                [{"dx": "위식도 역류", "p": 0.95, "for": ["가슴 통증"]}], initial="58세 남성. 주호소: 흉통")
    a = C.assess(st, "위식도 역류", AgentConfig())
    assert a.recommendation == "must_continue"
    assert a.dangers


def test_turn_cap_always_diagnoses():
    st = _state(APPENDICITIS_TURNS[:1] * 59, [{"dx": "급성 충수염", "p": 0.3}, {"dx": "게실염", "p": 0.3}])
    assert C.assess(st, "급성 충수염", AgentConfig()).recommendation == "diagnose"


def test_decide_bands():
    p = C.ConfidenceParams(theta_high=0.8, theta_low=0.5)
    assert C.decide(0.9, 3, 60, 20, [], p)[0] == "diagnose"
    assert C.decide(0.3, 3, 60, 20, [], p)[0] == "continue"
    assert C.decide(0.6, 3, 60, 20, [], p)[0] == "continue"
    assert C.decide(0.6, 20, 60, 20, [], p)[0] == "diagnose"
    assert C.decide(0.99, 3, 60, 20, ["대동맥 박리"], p)[0] == "must_continue"
    assert C.decide(0.99, 3, 60, 20, ["대동맥 박리"], p, gate_left=False)[0] == "diagnose"
    assert C.decide(0.99, 58, 60, 20, ["대동맥 박리"], p)[0] == "diagnose"  # 2 turns left: diagnosis goes ahead


def test_never_raises_on_empty_or_broken_state():
    st = CaseState("")
    a = C.assess(st, None)
    assert a.recommendation == "continue" and 0 <= a.score <= 1
    st.ddx_ledger = None  # broken state: features fail, assess still answers
    b = C.assess(st, "폐렴")
    assert b.recommendation in ("diagnose", "continue", "must_continue")


def test_no_state_kept_between_calls():
    st1 = _state(APPENDICITIS_TURNS, [{"dx": "급성 충수염", "p": 0.95, "for": ["우하복부 압통"]}])
    before = C.assess(st1, "급성 충수염").score
    C.assess(_state(APPENDICITIS_TURNS[:1], [{"dx": "게실염", "p": 0.2}]), "게실염")
    assert C.assess(st1, "급성 충수염").score == before


def test_fit_logistic_separates_and_respects_signs():
    xs = [[1.0, m] + [0.0] * (len(C.FEATURES) - 1) for m in (0.0, 0.1, 0.2, 0.7, 0.8, 0.9)]
    ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    w = CC.fit_logistic(xs, ys, l2=0.1)
    assert w[1] > 0
    wc = CC.fit_constrained(xs, ys, l2=0.1)
    for i, k in enumerate(C.FEATURES, 1):
        assert wc[i] * CC.SIGNS.get(k, 1) >= 0
    assert wc[C.FEATURES.index("turns_used") + 1] == CC.HAND_SET.weights["turns_used"]


def test_auc():
    assert CC.auc([0.9, 0.8, 0.2], [1, 1, 0]) == 1.0
    assert CC.auc([0.5, 0.5], [1, 0]) == 0.5
    assert CC.auc([0.5], [1]) is None


def _result_file(tmp_path, name, cases):
    p = tmp_path / name
    p.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return p


def _case(i, correct: bool):
    ddx_good = [{"dx": "급성 충수염", "p": 0.95, "status": "유력", "for": ["우하복부 압통"], "against": []}]
    ddx_bad = [{"dx": "게실염", "p": 0.4, "status": "유력", "for": [], "against": ["우하복부 압통"]},
               {"dx": "급성 충수염", "p": 0.35}]
    ddx = ddx_good if correct else ddx_bad
    turns = [{"type": t, "content": c, "response": r, "ddx": ddx} for t, c, r in APPENDICITIS_TURNS]
    turns.append({"type": "DIAGNOSE", "content": ddx[0]["dx"], "response": "", "ddx": ddx})
    return {"case": f"c{i}", "initial": "28세 남성. 주호소: 복통", "answer": "급성 충수염", "diagnosis": ddx[0]["dx"],
            "turns": turns, "scores": {"accuracy": 1.0 if correct else 0.0}}


def test_state_from_result_and_calibrate(tmp_path):
    p1 = _result_file(tmp_path, "run_a.json", [_case(i, i % 3 != 0) for i in range(6)])
    p2 = _result_file(tmp_path, "run_b.json", [_case(i, i % 2 == 0) for i in range(6)])
    st = CC.state_from_result(_case(0, True), 2)
    assert st.turn_count == 2 and st.ddx_ledger.entries[0].dx == "급성 충수염"
    smp = CC.samples([p1, p2])
    assert sum(s["final"] for s in smp) == 12
    rep = CC.calibrate([p1, p2])
    assert rep["auc_fitted_final_insample"] is not None and rep["auc_fitted_final_insample"] > 0.5
    out = tmp_path / "params.json"
    out.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    loaded = C.load_params(out)
    assert loaded.source == "calibrate" and set(loaded.weights) == set(C.FEATURES)
    assert CC.THETA_GRID[0] <= loaded.theta_high <= CC.THETA_GRID[-1]


def test_load_params_falls_back(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert C.load_params(bad) is C.DEFAULT_PARAMS
    monkeypatch.delenv("AGENT_CONFIDENCE_PARAMS", raising=False)
    assert C.load_params() is C.DEFAULT_PARAMS
