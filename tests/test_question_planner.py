import pytest

from doctor_agent.agent import question_planner as qp
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge import kb, kb_tests
from doctor_agent.safety import preconditions
from perf import assert_fast

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built")

CHEST = [("급성 심근경색", 0.4), ("폐색전증", 0.25), ("대동맥 박리", 0.2), ("급성 심낭염", 0.15)]
COUGH = [("폐렴", 0.5), ("급성 기관지염", 0.3), ("인플루엔자", 0.2)]


def _state(ddx, initial="45세 남성. 주호소: 2시간 전부터 시작된 흉통", turns=()):
    s = CaseState(initial_info=initial)
    for typ, content, resp in turns:
        s.turns.append(Turn(Action(ActionType(typ), content), resp))
    s.ddx_ledger.update([{"dx": d, "p": p} for d, p in ddx])
    return s


def _ranked(s, k=3):
    return qp.suggest(s, k=k)


def test_chest_pain_ranks_decisive_cardiac_tests():
    out = _ranked(_state(CHEST))
    assert 1 <= len(out) <= 3
    contents = [x.content_ko for x in out]
    assert "심전도" in contents and any("트로포닌" in c for c in contents)
    ecg = next(x for x in out if x.content_ko == "심전도")
    assert ecg.type == "TEST" and ecg.cost_tier == "lab" and ecg.expected_value > 0
    assert "급성 심근경색" in ecg.targets and ecg.source == "curated"
    assert all(a.expected_value >= b.expected_value for a, b in zip(out, out[1:]))


def test_done_tests_and_known_findings_are_excluded():
    s = _state(CHEST, turns=[("TEST", "12유도 심전도", "동리듬, ST 변화 없음")])
    assert all("심전도" not in x.content_ko for x in _ranked(s, k=5))
    base = _ranked(_state(COUGH, initial="30세 여성. 주호소: 3일 전부터 기침과 발열"), k=6)
    q = next(x for x in base if "피가 섞여" in x.content_ko)  # DDXPlus hemoptysis question
    s = _state(COUGH, initial="30세 여성. 주호소: 3일 전부터 기침과 발열",
               turns=[("ASK", "가래 색", "기침할 때 피는 안 나와요.")])
    assert all(x.content_ko != q.content_ko for x in _ranked(s, k=6))  # answered (negatively) already


def test_blocked_tests_are_not_suggested():
    ddx = [("세균성 수막염", 0.5), ("편두통", 0.5)]
    s = _state(ddx, initial="60세 남성. 주호소: 발열과 두통, 의식 저하")
    out = _ranked(s, k=8)
    assert all("요추천자" not in x.content_ko for x in out)
    for x in out:
        if x.type in ("TEST", "EXAM"):
            assert preconditions.check(ActionType(x.type), x.content_ko, s)["severity"] != "block"


def test_ranked_only_and_rendered():
    out = qp.suggest(_state(CHEST), k=3)
    assert 1 <= len(out) <= 3 and all(x.expected_value > 0 for x in out)  # no zero-value safety rows
    text = qp.render_for_prompt(out)
    assert text.startswith("추천 다음 행동 (참고") and len(text) <= qp.MAX_CHARS
    assert "1) [검사]" in text


def test_render_limits_and_empty():
    assert qp.render_for_prompt([]) == ""
    long = [qp.Suggestion("ASK", "가" * 400, ["폐렴"], 0.5), qp.Suggestion("ASK", "나" * 200, [], 0.4)]
    text = qp.render_for_prompt(long)
    assert len(text) <= qp.MAX_CHARS and text.endswith("…")


def test_request_wording_for_every_curated_result():
    for f in kb_tests.FINDINGS:
        req, tier, kws, typ = qp.request_for_result(f.id, f.ko)
        assert req and kws and tier in qp.TIER_W and typ in ("TEST", "EXAM"), f.id
        assert not req.endswith(("상승", "양성", "감소", "저하")), (f.id, req)
    assert qp.request_for_result("lipase_high", "리파아제 상승")[0] == "리파아제 검사"
    assert qp.request_for_result("ecg_stemi", "심전도: ST 분절 상승")[0] == "심전도"
    assert qp.request_for_result("csf_bacterial", "")[1] == "invasive"


def test_no_ddx_falls_back_to_kb_candidates_and_bad_state_is_safe():
    s = CaseState(initial_info="30세 여성. 주호소: 3일 전부터 기침과 발열")
    assert _ranked(s)
    assert qp.suggest(None) == []


def test_deterministic_and_does_not_mutate_state():
    s = _state(COUGH, initial="30세 여성. 주호소: 3일 전부터 기침과 발열")
    before = (len(s.turns), s.ddx_ledger.as_list(), s.findings.as_list())
    a, b = qp.suggest(s), qp.suggest(s)
    assert a == b
    assert (len(s.turns), s.ddx_ledger.as_list(), s.findings.as_list()) == before


@pytest.mark.perf
def test_latency_under_30ms():
    s = _state(CHEST, turns=[("ASK", "통증 양상", "가슴을 짓누르는 듯하고 왼팔로 퍼져요. 식은땀이 났어요."),
                             ("TEST", "심전도", "동리듬, ST 변화 없음")])
    qp.suggest(s)  # warm-up (lazy imports)
    assert_fast(lambda: qp.suggest(s), 0.030, what="question_planner.suggest()")  # strict budget: < 30 ms
