import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent import grounding as g  # noqa: E402
from doctor_agent.agent.ledger import DdxLedger, FindingsLedger  # noqa: E402
from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402

INITIAL = "64세 남성. 주호소: 1시간 전 갑자기 시작된 가슴 통증"
QA = [
    (ActionType.ASK, "통증 양상이 어떤가요?", "칼로 찢는 것처럼 아파요. 살면서 이렇게 아픈 건 처음이에요."),
    (ActionType.ASK, "통증이 퍼지나요?", "등 쪽 날개뼈 사이로 뻗치더니 지금은 허리 쪽으로 내려가는 것 같아요."),
    (ActionType.ASK, "식은땀이나 구토는요?", "식은땀이 나요. 토하지는 않았어요."),
    (ActionType.ASK, "숨차거나 기침은요?", "숨이 많이 차지는 않아요. 기침도 없어요."),
    (ActionType.ASK, "열은요?", "열은 없어요."),
    (ActionType.EXAM, "활력징후", "체온 36.7℃, 맥박 104회/분, 호흡수 22회/분, 혈압 우측 상지 182/98 mmHg · 좌측 상지 138/78 mmHg, "
                                "산소포화도 97%(실내공기)"),
    (ActionType.EXAM, "맥박 촉지", "좌측 요골동맥 맥박 우측에 비해 약함, 양측 대퇴동맥 맥박 촉지됨"),
    (ActionType.EXAM, "폐 청진", "양측 호흡음 깨끗함, 수포음, 천명음 없음"),
    (ActionType.TEST, "심전도", "동성빈맥 104회/분, 좌심실비대 소견과 V5–V6 비특이적 ST분절 하강, ST분절 상승 없음"),
    (ActionType.TEST, "트로포닌", "고감도 트로포닌 I 21 ng/L (참고치 <34)"),
    (ActionType.TEST, "혈액검사", "백혈구 14,200/μL, 혈색소 13.8 g/dL, 혈소판 212,000/μL, CRP 8.5 mg/dL (참고치 <0.5)"),
    (ActionType.TEST, "흉부 CT 혈관조영", "결과가 제공되지 않습니다."),
    (ActionType.ASK, "가슴이 두근거리나요?", "네, 좀 그래요."),
    (ActionType.ASK, "배가 아프신가요?", "아니요."),
]


def make_state(rows=QA) -> CaseState:
    s = CaseState(INITIAL)
    for kind, content, resp in rows:
        s.turns.append(Turn(Action(kind, content), resp))
    return s


@pytest.fixture(scope="module")
def ev():
    return g.Evidence.build(g.evidence_text(make_state()))


def grounded(claim, ev):
    return g.is_grounded(claim, ev)[0]


# --- evidence ---------------------------------------------------------------------------------------------------
def test_evidence_text_has_only_environment_words():
    text = g.evidence_text(make_state())
    assert INITIAL in text and "칼로 찢는" in text
    assert "통증 양상이 어떤가요" not in text  # the doctor's question is not evidence
    assert len(text.split("\n")) == len(QA) + 1


# --- true positives: patient language, exam, labs ---------------------------------------------------------------
@pytest.mark.parametrize("claim", ["가슴 통증", "흉통", "chest pain", "등으로 뻗치는 통증", "식은땀", "발한",
                                   "좌측 요골동맥 맥박 약함", "ST분절 하강", "빈맥",
                                   "좌심실비대", "64세 남성"])
def test_true_positive(claim, ev):
    assert grounded(claim, ev), claim


def test_measurement_concepts(ev):
    assert grounded("빈맥", ev)  # 맥박 104
    assert grounded("빈호흡", ev)  # 호흡수 22
    assert grounded("백혈구 증가", ev)  # 14,200
    assert grounded("leukocytosis", ev)
    assert not grounded("저산소증", ev)  # SpO2 97 %
    assert grounded("저산소증 없음", ev)
    assert not grounded("저혈압", ev)


# --- numbers ----------------------------------------------------------------------------------------------------
def test_numeric_match_and_synonym(ev):
    assert grounded("WBC 14200", ev)
    assert grounded("WBC 14.2", ev)  # ×10³/μL notation
    assert grounded("백혈구 14,200/μL", ev)
    assert grounded("우측 상지 혈압 182/98", ev)
    assert grounded("체온 36.7도", ev)


def test_numeric_mismatch(ev):
    assert not grounded("WBC 18000", ev)
    assert not grounded("체온 38.5도", ev)
    assert not grounded("혈소판 90,000", ev)


def test_reference_range_numbers_do_not_ground(ev):
    assert not grounded("트로포닌 34", ev)  # 34 is the reference limit, not the result
    assert grounded("트로포닌 21", ev)


def test_lab_direction_vs_reference(ev):
    assert not grounded("트로포닌 상승", ev)  # 21 < 34
    assert grounded("트로포닌 정상", ev)
    assert grounded("CRP 상승", ev)  # 8.5 > 0.5
    assert not grounded("CRP 정상", ev)


# --- negation ---------------------------------------------------------------------------------------------------
def test_positive_claim_not_grounded_by_negative_statement(ev):
    assert not grounded("발열", ev)  # "열은 없어요" (and 체온 36.7)
    assert not grounded("기침", ev)
    assert not grounded("구토", ev)
    assert not grounded("호흡곤란", ev)
    assert not grounded("수포음", ev)
    assert not grounded("ST분절 상승", ev)


def test_negative_claim_needs_negative_statement(ev):
    assert grounded("발열 없음", ev)
    assert grounded("기침 없음", ev)
    assert grounded("구토 없음", ev)
    assert grounded("호흡곤란 없음", ev)
    assert grounded("천명음 없음", ev)  # list: "수포음, 천명음 없음"
    assert grounded("수포음 없음", ev)
    assert grounded("ST 상승 없음", ev)
    assert not grounded("식은땀 없음", ev)  # patient said 식은땀이 나요
    assert not grounded("두통 없음", ev)  # never asked


def test_normal_breath_sounds(ev):
    assert grounded("호흡음 정상", ev)
    assert not grounded("호흡음 감소", ev)


def test_korean_pre_verb_negation():
    ev = g.Evidence.build("배는 안 아파요. 머리가 좀 아파요.")
    assert not grounded("복통", ev)
    assert grounded("복통 없음", ev)
    assert grounded("두통", ev)


def test_english_negation():
    ev = g.Evidence.build("Patient denies fever. Mild cough for 3 days.")
    assert not grounded("fever", ev)
    assert grounded("no fever", ev)
    assert grounded("기침", ev)


def test_uncertain_answer_grounds_nothing():
    ev = g.Evidence.build("열이 났는지는 잘 모르겠어요.")
    assert not grounded("발열", ev)
    assert not grounded("발열 없음", ev)


def test_idiom_is_not_negation():
    ev = g.Evidence.build("배가 너무 아파서 참을 수 없어요.")
    assert grounded("복통", ev)
    assert not grounded("복통 없음", ev)


def test_list_negation_only_for_bare_items():
    ev = g.Evidence.build("우상복부 압통, 머피 징후 양성, 반발통 없음, 장음 정상\n의식 명료, 국소 신경학적 결손 없음")
    assert grounded("우상복부 압통", ev)  # its own item, not negated by the later "반발통 없음"
    assert grounded("머피 징후 양성", ev)
    assert grounded("반발통 없음", ev)
    assert grounded("의식 명료", ev)
    assert not grounded("반발통", ev)


def test_parenthetical_remark_does_not_negate():
    ev = g.Evidence.build("혈압 152/94 mmHg(양측 상지 차이 없음)")
    assert grounded("혈압 152/94", ev)


def test_negated_claim_phrasing_and_korean_counts():
    ev = g.Evidence.build("오른쪽 허리가 뻐근해요. 배는 안 아파요.\n새벽에 두 번 토했어요.")
    assert grounded("배는 안 아파요", ev)  # negation inside the matched phrase of the claim
    assert not grounded("복통", ev)
    assert grounded("구토 2회", ev)
    assert not grounded("구토 1회", ev)


def test_direction_words_in_evidence():
    ev = g.Evidence.build("복부 팽만, 장음 항진")
    assert grounded("장음 항진", ev)
    assert not grounded("장음 감소", ev)


def test_concept_inside_longer_word_keeps_the_word():
    ev = g.Evidence.build("시신경 유두부종 없음")
    assert not grounded("폐부종 없음", ev)
    assert grounded("유두부종 없음", ev)
    assert not grounded("맥박", g.Evidence.build("대동맥 박리가 걱정돼요"))


# --- hallucinations / unavailable -------------------------------------------------------------------------------
def test_hallucinated_findings(ev):
    assert not grounded("우하복부 압통", ev)
    assert not grounded("반발통", ev)
    assert not grounded("D-dimer 상승", ev)
    assert not grounded("황달", ev)


def test_unavailable_result_never_grounds():
    text = "요청하신 흉부 CT, 복부 CT는 결과가 제공되지 않습니다.\n해당 검사 결과는 제공되지 않습니다"
    ev = g.Evidence.build(text)
    for claim in ("흉부 CT", "복부 CT", "CT 정상", "CT 이상 없음", "검사 결과"):
        assert not grounded(claim, ev), claim


def test_empty_inputs():
    assert g.is_grounded("", "열이 나요") == (False, 0.0, "")
    assert not grounded("발열", "")
    assert g.ungrounded_in_text("", "") == []
    report = g.apply(CaseState(""))
    assert report["findings_checked"] == 0 and report["ddx_removed"] == 0


# --- ledgers ----------------------------------------------------------------------------------------------------
def test_check_findings_marks_and_renders():
    state = make_state()
    state.findings.update([
        {"item": "가슴 통증", "status": "양성", "detail": "찢어지는 양상"},
        {"item": "우하복부 압통", "status": "양성"},
        {"item": "발열", "status": "음성"},
        {"item": "WBC", "status": "양성", "detail": "14,200"},
        {"item": "CRP", "status": "양성", "detail": "12.0 mg/dL"},
        {"item": "흉부 CT 혈관조영", "status": "결과없음"},
        {"item": "두근거림", "status": "양성"},
        {"item": "복통", "status": "음성"},
    ], 1)
    report = g.apply(state)
    v = {f.item: f.verified for f in state.findings.items}
    assert v == {"가슴 통증": True, "우하복부 압통": False, "발열": True, "WBC": True, "CRP": False,
                 "흉부 CT 혈관조영": True, "두근거림": True, "복통": True}
    assert report["findings_unverified"] == 2
    text = state.findings.render()
    assert "우하복부 압통 (미확인)" in text and "가슴 통증 (찢어지는 양상)" in text
    assert "우하복부" not in state.findings.render(exclude_unverified=True)
    span = next(f.span for f in state.findings.items if f.item == "두근거림")
    assert span.startswith("Q:")  # grounded by a yes answer to a question that named it


def test_yes_answer_that_denies_the_finding_does_not_ground():
    state = make_state([(ActionType.ASK, "기침하고 가래 있으세요?", "네, 가래는 있는데 기침은 없어요.")])
    state.findings.update([{"item": "기침", "status": "양성"}, {"item": "가래", "status": "양성"}], 1)
    g.apply(state)
    assert [f.verified for f in state.findings.items] == [False, True]


def test_render_unchanged_before_check():
    led = FindingsLedger()
    led.update([{"item": "기침", "status": "양성"}], 1)
    assert led.render() == "- 양성: 기침"
    assert led.as_list()[0]["verified"] is None


def test_check_ddx_support_drops_ungrounded(ev):
    led = DdxLedger()
    led.update([{"dx": "대동맥 박리", "p": 0.6, "for": ["찢어지는 흉통", "양팔 혈압 차이", "전형적 양상"],
                 "against": ["발열 없음"]},
                {"dx": "급성 충수염", "p": 0.1, "for": ["우하복부 압통", "WBC 14200"], "against": ["복막 자극 징후 없음"]}])
    removed = g.check_ddx_support(led, ev)
    items = {r["item"] for r in removed}
    assert "우하복부 압통" in items
    assert "전형적 양상" not in items  # reasoning, not a finding: left alone
    assert "WBC 14200" not in items and "발열 없음" not in items
    ad = led.entries[0]
    assert "찢어지는 흉통" in ad.support


def test_check_ddx_flag_mode_is_idempotent(ev):
    led = DdxLedger()
    led.update([{"dx": "충수염", "p": 0.3, "for": ["반발통", "구토"]}])
    first = g.check_ddx_support(led, ev, mode="flag")
    snapshot = list(led.entries[0].support)
    second = g.check_ddx_support(led, ev, mode="flag")
    assert len(first) == 2 and second == []
    assert led.entries[0].support == snapshot == ["반발통 (미확인)", "구토 (미확인)"]


def test_apply_is_idempotent():
    state = make_state()
    state.findings.update([{"item": "황달", "status": "양성"}, {"item": "식은땀", "status": "양성"}], 1)
    state.ddx_ledger.update([{"dx": "담관염", "p": 0.2, "for": ["황달", "발열"]}])
    r1 = g.apply(state)
    before = (state.findings.as_list(), state.ddx_ledger.as_list())
    r2 = g.apply(state)
    assert (state.findings.as_list(), state.ddx_ledger.as_list()) == before
    assert r1["findings_unverified"] == r2["findings_unverified"] == 1
    assert r1["ddx_removed"] == 2 and r2["ddx_removed"] == 0
    assert state.ddx == state.ddx_ledger.as_list()


def test_ungrounded_in_reason(ev):
    reason = "찢어지는 흉통과 등으로의 방사통, 우하복부 압통, WBC 18000으로 보아 대동맥 박리 의심"
    bad = g.ungrounded_in_text(reason, ev)
    assert "우하복부 압통" in bad
    assert any("18000" in b for b in bad)
    assert not any("흉통" in b for b in bad)
    assert not any("대동맥" in b for b in bad)


# --- normalisation-layer guards (precision first) ---------------------------------------------------------------
def test_ambiguous_report_list_grounds_neither_way():
    ev = g.Evidence.build("복부 부드러움, 장음 항진, 반발통 및 근육 강직 없음")
    assert not grounded("장음 항진", ev) and not grounded("장음 항진 없음", ev)
    assert grounded("반발통 없음", ev)


def test_list_item_with_severity_is_stated():
    ev = g.Evidence.build("명치 부위 경미한 압통, 반발통 없음, 간비종대 없음")
    assert grounded("명치 부위 경미한 압통", ev)
    assert not grounded("명치 압통 없음", ev)


def test_test_value_against_printed_reference_range():
    ev = g.Evidence.build("고감도 트로포닌 I 1240 ng/L (참고치 <34)")
    assert not grounded("트로포닌 I 음성", ev)
    assert grounded("트로포닌 1240", ev)


def test_site_narrowed_negation_does_not_cover_other_sites():
    ev = g.Evidence.build("흉벽 압통 없음, 촉진으로 통증 재현되지 않음")
    assert not grounded("맥버니 점 압통 없음", ev)


def test_claim_qualifier_must_be_in_the_evidence():
    ev = g.Evidence.build("우하복부 압통(맥버니 점), 반발통 있음")
    assert grounded("복부 압통", ev)  # a present child grounds its parent
    assert not grounded("하복부 전반의 압통", ev)
    ev = g.Evidence.build("구강 점막 약간 건조, 피부 긴장도 정상, 모세혈관 재충혈 2초 이내")
    assert not grounded("피부 긴장도 저하", ev)


def test_past_or_quit_does_not_ground_current():
    ev = g.Evidence.build("담배는 10년 전에 끊었어요")
    assert not grounded("하루 반 갑 피워요", ev)
    assert grounded("흡연력", ev)


def test_when_clause_is_not_a_symptom():
    ev = g.Evidence.build("기침할 때 심해지지도 않아요")
    assert not grounded("마른기침", ev) and not grounded("기침", ev)


def test_literal_reader_edge_cases():
    ev = g.Evidence.build("좌심실비대 소견과 V5–V6 비특이적 ST분절 하강, ST분절 상승 없음")
    assert not grounded("그 외 특이 소견 없음", ev)  # "비특이적" is not "특이"
    assert not grounded("오른쪽은 괜찮아요", g.Evidence.build("오른쪽 팔다리에 힘이 없다고 했어요"))


def test_family_question_answer_is_not_the_patients_finding():
    state = make_state([(ActionType.ASK, "가족 중에 당뇨 있는 분 계세요?", "네, 어머니가 당뇨가 있으세요.")])
    state.findings.update([{"item": "당뇨", "status": "양성"}], 1)
    g.apply(state)
    assert state.findings.items[0].verified is False


def test_parse_cache_lives_on_the_case_state():
    s1, s2 = make_state(), make_state(QA[:2])
    g.apply(s1)
    assert s1._grounding_cache and not hasattr(s2, "_grounding_cache")
    g.apply(s2)
    assert s2._grounding_cache is not s1._grounding_cache


# --- performance ------------------------------------------------------------------------------------------------
def test_performance_40_turn_case():
    rows = (QA * 3)[:40]
    state = make_state(rows)
    state.findings.update([{"item": x, "status": "양성"} for x in
                           ["가슴 통증", "식은땀", "빈맥", "좌측 요골동맥 맥박 약함", "백혈구 증가", "우하복부 압통", "황달",
                            "WBC 14200", "CRP 상승", "좌심실비대"]]
                          + [{"item": x, "status": "음성"} for x in ["발열", "기침", "구토", "호흡곤란", "ST 상승"]], 1)
    state.ddx_ledger.update([{"dx": f"dx{i}", "p": 0.1, "for": ["찢어지는 흉통", "반발통", "WBC 14200"],
                              "against": ["발열 없음", "기침 없음"]} for i in range(5)])
    g.apply(state)  # warm up (regex compile caches)
    batches = []
    for _ in range(5):  # best batch: robust to other processes loading the machine
        t = time.perf_counter()
        for _ in range(5):
            g.apply(state)
        batches.append((time.perf_counter() - t) / 5 * 1000)
    assert min(batches) <= 5.0, batches
