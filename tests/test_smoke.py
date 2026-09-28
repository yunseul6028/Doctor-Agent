import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.loop import run_case  # noqa: E402
from doctor_agent.agent.parser import parse_action  # noqa: E402
from doctor_agent.config import Config  # noqa: E402
from doctor_agent.env.interface import ActionType  # noqa: E402
from doctor_agent.llm.client import DummyLLM  # noqa: E402
from eval.simulator import CaseFileEnvironment  # noqa: E402

CASE = ROOT / "data/sample_cases/synthetic_001.json"


def test_run_case_with_dummy_llm():
    result = run_case(CaseFileEnvironment.from_file(CASE), DummyLLM(), Config())
    assert result["diagnosis"]
    assert result["llm_calls"] >= 1
    assert result["n_turns"] <= 60


def test_parse_action_handles_noise():
    parsed = parse_action('thinking...\n{"type": "ask", "content": "열이 있나요?", "confidence": 0.2}')
    assert parsed and parsed[0].type == ActionType.ASK
    assert parse_action("not json") is None


def test_case_file_is_utf8_json():
    json.loads(CASE.read_text(encoding="utf-8"))


def test_llm_patient_hides_answer():
    from eval.llm_patient import LLMPatientEnvironment

    case = json.loads(CASE.read_text(encoding="utf-8"))
    env = LLMPatientEnvironment(case, DummyLLM())
    assert case["diagnosis"] not in env.system


def test_llm_patient_hides_meta_fields():
    from eval.llm_patient import LLMPatientEnvironment

    for path in (ROOT / "data/sample_cases").glob("*.json"):
        case = json.loads(path.read_text(encoding="utf-8"))
        system = LLMPatientEnvironment(case, DummyLLM()).system
        for key in ("diagnosis", "teaching_point"):
            if case.get(key):
                assert case[key] not in system, (path.name, key)


def test_personas_build_and_mixed_is_deterministic():
    from eval.llm_patient import HARD_PERSONAS, PERSONAS, LLMPatientEnvironment, resolve_persona

    case = json.loads(CASE.read_text(encoding="utf-8"))
    for name in PERSONAS:
        env = LLMPatientEnvironment(case, DummyLLM(), name)
        assert case["diagnosis"] not in env.system
    assert resolve_persona("mixed", "x") == resolve_persona("mixed", "x") in HARD_PERSONAS


def test_role_config_fallback(monkeypatch):
    from doctor_agent.config import LLMConfig

    monkeypatch.setenv("LLM_MODEL", "shared-model")
    monkeypatch.setenv("DOCTOR_LLM_MODEL", "doctor-model")
    assert LLMConfig.from_env("DOCTOR_LLM").model == "doctor-model"
    assert LLMConfig.from_env("PATIENT_LLM").model == "shared-model"


def test_parse_action_skips_reasoning_block():
    raw = ('<thought>ddx: [{"dx": "충수염", "p": 0.2}] 그래서 JSON: {"type": "ASK", "content": "초안"</thought>\n'
           '{"type": "ASK", "content": "어디가 가장 아프세요?", "reason": "위치 확인", "ddx": [{"dx": "충수염", "p": 0.3}], "confidence": 0.1}')
    action, ddx, conf = parse_action(raw)
    assert action.content == "어디가 가장 아프세요?" and action.reason == "위치 확인"
    assert ddx[0]["dx"] == "충수염" and conf == 0.1


def test_parse_action_unclosed_reasoning_falls_back_to_last_json():
    raw = '<thought>생각 중... {"type": "TEST", "content": "복부 CT", "confidence": 0.5}'
    action, _, _ = parse_action(raw)
    assert action.type == ActionType.TEST and action.content == "복부 CT"


def test_clean_diagnosis_strips_sentence():
    from doctor_agent.agent.loop import clean_diagnosis

    assert clean_diagnosis("당신의 진단은 당뇨병성 케톤산증(Diabetic Ketoacidosis, DKA)입니다.") == "당뇨병성 케톤산증(Diabetic Ketoacidosis, DKA)"
    assert clean_diagnosis("최종 진단: 급성 충수염") == "급성 충수염"
    assert clean_diagnosis("급성 췌장염") == "급성 췌장염"


class _Scripted:
    """Scripted LLM: returns the outputs in order, repeating the last one."""

    def __init__(self, *outputs):
        self.outputs, self.call_count = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs], 0

    def chat(self, messages):
        self.call_count += 1
        return self.outputs[min(self.call_count - 1, len(self.outputs) - 1)]


def _diagnose(dx, reason="추정"):
    return {"type": "DIAGNOSE", "content": dx, "reason": reason, "confidence": 0.8}


def _review(confirmation="혈액검사 확인", unexplained=(), contradicting=(), danger=(), final="", evidence="",
            nxt=({"type": "TEST", "content": "자가항체 검사", "reason": "감별"})):
    key = [{"finding": "황달", "status": "설명됨"}] + [{"finding": f, "status": "설명 안 됨"} for f in unexplained]
    return {"key_findings": key, "contradicting": list(contradicting), "confirmation": confirmation,
            "unresolved_danger": list(danger), "next": nxt, "final_diagnosis": final, "refine_evidence": evidence}


def _state(*exchanges):
    from doctor_agent.agent.state import CaseState, Turn
    from doctor_agent.env.interface import Action

    st = CaseState(initial_info="40세. 주호소: 황달")
    # these tests exercise the diagnosis review only; 황달 now has a safety protocol, whose one-time pushback
    # before DIAGNOSE would consume a scripted output, so mark it as already given
    st.safety_pushback = True
    st.confidence_pushback = True  # same for the one-time low-confidence pushback (tests/test_advisors.py covers it)
    for q, a in exchanges:
        st.turns.append(Turn(Action(ActionType.ASK, q), a))
    return st


def _run(state, *outputs):
    from doctor_agent.agent.policy import Policy

    llm = _Scripted(*outputs)
    return Policy(llm, Config().agent).next_action(state), llm


def test_review_holds_without_confirmatory_evidence():
    state = _state()
    step = {"findings": [{"item": "B형·C형 간염 검사", "status": "음성"}], "ddx": [{"dx": "A형 간염", "p": 0.5}], **_diagnose("A형 간염")}
    action, _ = _run(state, step, _review(confirmation="없음"))
    assert action.type == ActionType.TEST and action.content == "자가항체 검사"
    r = state.reviews[0]
    assert r["verdict"] == "보류" and r["confirmation"] == "없음" and "확진 근거 없음" in r["issues"]
    assert state.findings.items[0].status == "음성" and state.ddx_ledger.entries[0].dx == "A형 간염"


def test_review_holds_on_unexplained_finding_or_open_danger():
    state = _state()
    action, _ = _run(state, _diagnose("A형 간염"), _review(unexplained=["혈소판 감소"]))
    assert action.type == ActionType.TEST and state.reviews[0]["key_findings"][1] == {"finding": "혈소판 감소", "explained": False}
    state = _state()
    action, _ = _run(state, _diagnose("A형 간염"), _review(danger=["급성 간부전"]))
    assert action.type == ActionType.TEST and state.reviews[0]["unresolved_danger"] == ["급성 간부전"]


def test_review_approves_complete_case_despite_free_text_verdict():
    state = _state()
    out = {**_review(nxt=None), "verdict": "보류"}  # a stray free-text verdict is ignored; code decides from fields
    action, _ = _run(state, _diagnose("A형 간염"), out)
    assert action.type == ActionType.DIAGNOSE and action.content == "A형 간염" and state.reviews[0]["verdict"] == "승인"


def test_review_without_usable_next_action_does_not_hold():
    state = _state()
    action, _ = _run(state, _diagnose("A형 간염"), _review(confirmation="없음", nxt={"type": "DIAGNOSE", "content": "x"}))
    assert action.type == ActionType.DIAGNOSE and state.reviews[0]["verdict"] == "승인"
    state = _state()
    action, _ = _run(state, _diagnose("A형 간염"), "검토 결과: 문제 없음")  # unparseable → never blocks
    assert action.type == ActionType.DIAGNOSE and action.content == "A형 간염"


def test_review_refuses_unsupported_refinements():
    from doctor_agent.agent.policy import _refinement_problem

    case = "대장내시경: 대장에 궤양성 종괴, 조직검사 선암\n운동 중 두드러기와 호흡곤란\n비타민 B12 낮음, 기억력 저하"
    assert _refinement_problem("대장암", "상행결장암", "조직검사 선암", case)  # location qualifier
    assert _refinement_problem("운동 유발성 아나필락시스", "밀가루 의존성 운동 유발성 아나필락시스", "운동 중 두드러기", case)
    assert _refinement_problem("비타민 B12 결핍", "비타민 B12 결핍에 의한 인지장애", "기억력 저하", case)
    assert _refinement_problem("대장암", "대장 선암", "", case)  # no cited evidence
    assert _refinement_problem("대장암", "대장 선암", "CEA 상승", case)  # evidence not in this case
    state = _state(("대장내시경", "대장에 궤양성 종괴"))
    action, _ = _run(state, _diagnose("대장암"), _review(nxt=None, final="상행결장암", evidence="대장에 궤양성 종괴"))
    assert action.content == "대장암" and state.reviews[0]["refinement"]["accepted"] is False
    assert "final_diagnosis" not in state.reviews[0]


def test_review_accepts_supported_refinements():
    from doctor_agent.agent.policy import _refinement_problem

    state = _state(("가장 길었던 들뜬 시기는 얼마나 갔나요?", "경조증 같은 시기가 4일 정도였고 입원한 적은 없어요"))
    action, _ = _run(state, _diagnose("양극성 장애", "조증 삽화"),
                     _review(nxt=None, final="양극성 II형 장애", evidence="경조증 4일, 입원 없음"))
    assert action.type == ActionType.DIAGNOSE and action.content == "양극성 II형 장애"
    assert state.reviews[0]["final_diagnosis"] == "양극성 II형 장애" and state.reviews[0]["refinement"]["accepted"]
    case = "밀가루 음식을 먹고 운동하면 두드러기와 호흡곤란"
    assert not _refinement_problem("운동 유발성 아나필락시스", "밀가루 의존성 운동 유발성 아나필락시스", "밀가루 음식 후 운동 시 발생", case)


def test_review_cap_and_turn_budget():
    from doctor_agent.agent.policy import MAX_REVIEWS
    from doctor_agent.agent.state import Turn

    state = _state()
    hold = _review(confirmation="없음")
    for i in range(MAX_REVIEWS):
        action, _ = _run(state, _diagnose("A형 간염"), {**hold, "next": {"type": "TEST", "content": f"검사 {i}"}})
        assert action.type == ActionType.TEST
        state.turns.append(Turn(action, "결과가 제공되지 않습니다"))
    action, llm = _run(state, _diagnose("A형 간염"), hold)
    assert action.type == ActionType.DIAGNOSE and llm.call_count == 1 and len(state.reviews) == MAX_REVIEWS
    late = _state(*[(f"질문 {i}", "네") for i in range(Config().agent.max_turns - 3)])
    action, llm = _run(late, _diagnose("A형 간염"), hold)
    assert action.type == ActionType.DIAGNOSE and not late.reviews


def test_ledgers_merge_across_turns():
    from doctor_agent.agent.ledger import DdxLedger, FindingsLedger

    f = FindingsLedger()
    f.update([{"item": "우하복부 압통", "status": "양성"}], 1)
    f.update([{"item": "우하복부 압통", "status": "양성", "detail": "반발통 동반"}, {"item": "발열", "status": "없음"}], 2)
    assert len(f.items) == 2 and f.items[0].detail == "반발통 동반" and f.items[1].status == "음성"
    d = DdxLedger()
    d.update([{"dx": "급성 충수염", "p": 0.5, "for": ["우하복부 압통"]}, {"dx": "장염", "p": 0.3}])
    d.update([{"dx": "급성 충수염", "p": 0.8, "for": ["백혈구 증가"]}, {"dx": "장염", "p": 0.1, "status": "배제", "against": ["설사 없음"]}])
    assert d.ranked()[0].dx == "급성 충수염" and d.ranked()[0].support == ["우하복부 압통", "백혈구 증가"]
    assert "배제됨: 장염" in d.render()


def test_missing_test_is_not_reported_as_normal():
    case = json.loads(CASE.read_text(encoding="utf-8"))
    from doctor_agent.env.interface import Action

    obs = CaseFileEnvironment(case).step(Action(ActionType.TEST, "갑상선 기능 검사"))
    assert "제공되지 않습니다" in obs.text


def test_near_duplicate_actions_are_detected():
    from doctor_agent.agent.state import CaseState, Turn
    from doctor_agent.env.interface import Action

    st = CaseState(initial_info="x")
    st.turns.append(Turn(Action(ActionType.ASK, "혹시 최근에 체중이 갑자기 줄었거나, 식은땀이 나시나요?"), "아니요"))
    assert st.asked(Action(ActionType.ASK, "혹시 최근에 갑자기 체중이 줄었거나 식은땀이 나시나요?"))
    assert not st.asked(Action(ActionType.ASK, "기침이 있나요?"))
    st.turns.append(Turn(Action(ActionType.ASK, "열이 있나요?"), "네"))
    assert not st.asked(Action(ActionType.ASK, "기침이 있나요?"))


def test_question_sent_as_exam_becomes_ask():
    from doctor_agent.agent.policy import normalize_type
    from doctor_agent.env.interface import Action

    assert normalize_type(Action(ActionType.EXAM, "목이 뻣뻣하거나 고개를 숙일 때 통증이 심해지나요?")).type == ActionType.ASK
    assert normalize_type(Action(ActionType.EXAM, "경부 강직 진찰")).type == ActionType.EXAM
