"""Diagnostic/classification criteria checkers (knowledge/diagnostic_criteria.py) and their use in the review.
No LLM calls: the review tests use a scripted fake LLM."""
import json

import pytest

from doctor_agent.agent.ledger import FindingsLedger
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge import diagnostic_criteria as dc

# (criteria id, positive findings, expected decision, negative-or-unknown findings)
CASES = [
    ("sle_2019",
     "28세 여성. 관절이 붓고 아파요. 뺨에 나비 모양 발진. ANA 1:320 양성, anti-dsDNA 양성, C3 감소.", "전신홍반루푸스",
     "28세 여성. 관절이 붓고 아파요. ANA 음성."),
    ("ra_2010",
     "45세 여성. 3개월 전부터 양측 손가락 관절이 붓고 아침에 뻣뻣해요. 류마티스 인자 양성, 항CCP 항체 고역가 양성, "
     "CRP 상승.", "류마티스 관절염",
     "45세 여성. 무릎이 아파요."),
    ("takayasu_2022",
     "25세 여성. 왼팔을 쓰면 저리고 아파요. 왼쪽 요골 맥박이 약함. 쇄골하 잡음. CT 혈관조영: 좌측 쇄골하동맥 협착.",
     "타카야수 동맥염",
     "72세 남성. CT 혈관조영: 좌측 쇄골하동맥 협착."),
    ("gca_2022",
     "72세 여성. 새로 생긴 관자놀이 두통, 씹을 때 턱이 아파요. ESR 85 mm/h. 측두동맥 생검에서 거대세포 혈관염.",
     "거대세포동맥염",
     "40세 여성. 관자놀이 두통. 측두동맥 생검 양성."),
    ("kawasaki_aha2017",
     "3세 남아. 열이 6일째 나요. 눈이 충혈됐고 입술이 갈라지고 딸기 혀, 몸에 발진, 손발이 붓고 목 림프절이 커져 있음.",
     "가와사키병",
     "3세 남아. 열이 2일째. 발진."),
    ("duke_iscvid_2023",
     "혈액 배양 3세트 모두 황색포도알균 자람. 경식도 심초음파에서 승모판 증식물. 체온 38.9도.", "감염성 심내막염",
     "체온 38.5도. 인공 판막. 혈액 배양 결과 대기 중."),
    ("dka_hhs_2024",
     "당뇨병 환자. 혈당 480 mg/dL, pH 7.12, HCO3 9, 소변 케톤 3+.", "당뇨병성 케톤산증",
     "혈당 250 mg/dL."),
    ("light_1972",
     "흉수 단백 4.5, 혈청 단백 6.0, 흉수 LDH 300, 혈청 LDH 200", "삼출성 흉수",
     "흉수 있음"),
    ("sepsis3_2016",
     "폐렴. 노르에피네프린 투여 중. 젖산 4.2 mmol/L. 수액 30 mL/kg 후에도 저혈압 지속.", "패혈성 쇼크",
     "폐렴. 호흡수 26회, 혈압 90/60, 의식 저하."),
    ("jones_2015",
     "10세. 2주 전 인후통. ASO 상승. 다발성 관절염, 새로운 심잡음, 체온 38.8도.", "급성 류마티스열",
     "10세. 다발성 관절염, 새로운 심잡음."),
    ("mcdonald_2017",
     "30세 여성. 시신경염. MRI: 뇌실 주위 병변, 척수 병변. 뇌척수액 올리고클론띠 양성.", "다발성 경화증",
     "30세 여성. 시신경염. MRI: 뇌실 주위 병변."),
    ("ichd3_migraine_tth",
     "한쪽 머리가 욱신거리고 심한 두통, 메스꺼움, 빛이 눈부셔요. 한 달에 3번 반복.", "편두통",
     "머리가 아파요."),
    ("bipolar_dsm5tr",
     "들뜬 시기가 10일 계속됐고 그때 입원했어요.", "양극성 I형 장애",
     "기분이 우울해요"),
    ("gout_2015",
     "55세 남성. 자다가 엄지발가락이 갑자기 붓고 빨개졌어요. 이불만 스쳐도 아파요. 걷기 힘들어요. 요산 9.2 mg/dL.",
     "통풍",
     "무릎이 아파요."),
]


def test_every_criteria_set_has_a_test_case():
    assert {c[0] for c in CASES} | {"kdigo_aki_2012"} == set(dc.CRITERIA_BY_ID)


@pytest.mark.parametrize("cid,positive,decision,negative", CASES, ids=[c[0] for c in CASES])
def test_positive_and_negative_examples(cid, positive, decision, negative):
    assert dc.evaluate(cid, positive).decision == decision
    neg = dc.evaluate(cid, negative)
    assert neg.decision == "" and neg.band


def test_registry_metadata_is_complete():
    for c in dc.CRITERIA:
        assert c.citation.verified and (c.citation.pmid or c.citation.doi)
        assert c.verification in ("primary", "secondary", "unverified")
        assert c.rule and c.note and c.copyright and c.items
        assert len({i.key for i in c.items}) == len(c.items)


def test_aki_staging():
    assert dc.evaluate("kdigo_aki_2012", "기저 크레아티닌 1.0 mg/dL. 현재 크레아티닌 2.4 mg/dL.").band.startswith("AKI 2단계")
    assert dc.evaluate("kdigo_aki_2012", "크레아티닌 1.0에서 3.5로 상승").band.startswith("AKI 3단계")
    assert dc.evaluate("kdigo_aki_2012", "크레아티닌 1.0에서 1.1로").band.startswith("AKI 크레아티닌 기준 미달")
    assert dc.evaluate("kdigo_aki_2012", "크레아티닌 2.4").band.startswith("병기 미확인")  # no baseline: no stage


def test_subtype_decisions():
    assert dc.evaluate("dka_hhs_2024", "혈당 850 mg/dL, 삼투압 345 mOsm/kg, pH 7.36, HCO3 22, 소변 케톤 음성.").decision \
        == "고혈당성 고삼투압 상태"
    kd = "3세 남아. 열이 6일째. 발진과 눈 충혈. CRP 8 mg/dL. 알부민 2.8 g/dL, 백혈구 18000, 농뇨."
    assert dc.evaluate("kawasaki_aha2017", kd).decision == "불완전 가와사키병"
    tth = "양쪽 머리가 조이는 느낌, 참을 만해요. 메스꺼움은 없어요. 움직여도 심해지지 않아요. 거의 매일 반복돼요."
    assert dc.evaluate("ichd3_migraine_tth", tth).decision == "긴장형 두통"
    assert dc.evaluate("ichd3_migraine_tth", tth.replace(" 거의 매일 반복돼요.", "")).decision == ""  # recurrence needed
    light = "흉수 단백 1.5, 혈청 단백 7.0, 흉수 LDH 60, 혈청 LDH 200, LDH 정상 상한 250"
    assert dc.evaluate("light_1972", light).decision == "누출성 흉수"
    assert dc.evaluate("gout_2015", "관절액에서 바늘 모양 요산 결정 관찰, 음성 복굴절").decision == "통풍"


def test_bipolar_two_needs_explicit_hypomania_no_admission_and_depression():
    two = "경조증 같은 시기가 4일 정도였고 입원한 적은 없어요. 2주 넘게 우울했던 적이 여러 번 있어요."
    assert dc.evaluate("bipolar_dsm5tr", two).decision == "양극성 II형 장애"
    no_mde = "경조증 같은 시기가 4일 정도였고 입원한 적은 없어요"
    assert dc.evaluate("bipolar_dsm5tr", no_mde).decision == ""  # depression not established
    long_no_impairment = "들뜬 시기가 10일 정도 있었어요. 입원한 적은 없어요."
    assert dc.evaluate("bipolar_dsm5tr", long_no_impairment).decision == ""  # >=7 days alone is not mania


def test_headache_red_flags_block_primary_headache_decision():
    migraine = "한쪽 머리가 욱신거리고 심한 두통, 메스꺼움, 빛이 눈부셔요. 한 달에 3번 반복."
    assert dc.evaluate("ichd3_migraine_tth", migraine).decision == "편두통"
    for flag in ("갑자기 망치로 맞은 듯 심한 두통", "목이 뻣뻣해요", "체온 39도, 열이 나요"):
        r = dc.evaluate("ichd3_migraine_tth", migraine + " " + flag)
        assert r.decision == "" and r.band.startswith("판정 보류")
    assert dc.evaluate("ichd3_migraine_tth", migraine + " 목이 뻣뻣하다기보다 어깨가 뭉쳐요.").decision == "편두통"


def test_focus_limits_features_to_the_body_part_unless_it_is_the_chief_complaint():
    story = "58세 여성. 주호소: 설사\n양측 폐 호흡음 정상. 배가 묵직해요. 설사를 9번 이상 했어요. 가끔 머리가 아파요."
    r = dc.evaluate("ichd3_migraine_tth", story)
    assert r.states["bilateral"][0] == dc.UNKNOWN and r.states["pressing"][0] == dc.UNKNOWN
    headache = "38세 여성. 주호소: 두통\n욱신욱신 뛰어요."  # answer without the word "head" still counts
    assert dc.evaluate("ichd3_migraine_tth", headache).states["pulsating"][0] == dc.MET


def test_conservative_extraction():
    # doctor questions and "not provided" ledger lines are not findings
    asked = "1. ASK: 들뜬 시기로 입원한 적 있나요?\n   → 잘 모르겠어요"
    r = dc.evaluate("bipolar_dsm5tr", asked)
    assert r.states["hosp"][0] == dc.UNKNOWN
    ledger = FindingsLedger()
    ledger.update([{"item": "ANA", "status": "결과없음"}, {"item": "관절 부종", "status": "음성"}], 1)
    r = dc.evaluate("sle_2019", ledger.render())
    assert r.states["ana"][0] == dc.UNKNOWN and r.states["joint"][0] == dc.NOT_MET
    # negated in the clause
    assert dc.evaluate("gca_2022", "72세. 턱 통증은 없어요.").states["jaw"][0] != dc.MET
    # CRP without a unit is not trusted (mg/L vs mg/dL)
    assert dc.evaluate("gca_2022", "72세. CRP 12").states["esr_crp"][0] == dc.UNKNOWN
    # a pair of stated values that fail the threshold is "not met", not "unknown"
    assert dc.evaluate("dka_hhs_2024", "pH 7.40, HCO3 24").states["ph"][0] == dc.NOT_MET


@pytest.mark.parametrize("name,expected", [
    ("양극성 장애", ["bipolar_dsm5tr"]),
    ("Bipolar II disorder", ["bipolar_dsm5tr"]),
    ("타카야수 동맥염(Takayasu arteritis)", ["takayasu_2022"]),
    ("좌측 쇄골하동맥 협착", ["takayasu_2022"]),  # cross-check: stenosis vs underlying arteritis
    ("대혈관 혈관염", ["takayasu_2022", "gca_2022"]),
    ("당뇨병성 케톤산증(DKA)", ["dka_hhs_2024"]),
    ("알코올성 케톤산증", []),
    ("가성통풍", []),
    ("통풍성 관절염", ["gout_2015"]),
    ("septic arthritis", []),
    ("승모판 협착증(MS)", []),
    ("다발성 경화증(MS)", ["mcdonald_2017"]),
    ("급성 세뇨관 괴사", ["kdigo_aki_2012"]),
    ("관절염", []),
    ("", []),
])
def test_criteria_for_names(name, expected):
    assert [c.id for c in dc.criteria_for(name)] == expected


def test_subtype_name_resolution():
    bp = dc.CRITERIA_BY_ID["bipolar_dsm5tr"]
    assert dc.which_subtype(bp, "양극성 II형 장애").name == "양극성 II형 장애"
    assert dc.which_subtype(bp, "양극성 I형 장애").name == "양극성 I형 장애"
    assert dc.which_subtype(bp, "bipolar I disorder").name == "양극성 I형 장애"
    assert dc.which_subtype(bp, "양극성 장애") is None
    kd = dc.CRITERIA_BY_ID["kawasaki_aha2017"]
    assert dc.which_subtype(kd, "불완전 가와사키병").name == "불완전 가와사키병"
    assert dc.conflicting_subtype(bp, "양극성 I형 장애", "양극성 II형 장애") == "양극성 II형 장애"
    assert dc.decision_matches(bp, "양극성 I형 장애", "양극성 1형 장애")


def test_render_for_review_is_short_and_cited():
    text = dc.render_for_review("대혈관 혈관염", "72세 여성. 관자놀이 두통. ESR 85 mm/h.")
    assert text.startswith("[진단 기준 대조:") and len(text) <= dc.MAX_RENDER
    assert "PMID" in text and "판정" in text
    assert dc.render_for_review("급성 충수염", "우하복부 통증") == ""
    for c in dc.CRITERIA:
        out = dc.render_for_review(c.synonyms[0].removeprefix("re:"), CASES[0][1])
        assert len(out) <= dc.MAX_RENDER


# --- review integration (scripted fake LLM, no real calls) -----------------------------------------------------------

class _Scripted:
    def __init__(self, *outputs):
        self.outputs = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs]
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.outputs[min(len(self.messages) - 1, len(self.outputs) - 1)]


def _state(initial, *exchanges):
    from doctor_agent.agent.state import CaseState, Turn

    st = CaseState(initial_info=initial)
    st.safety_pushback = True  # skip the one-time protocol pushback; these tests exercise the review only
    for q, a in exchanges:
        st.turns.append(Turn(Action(ActionType.ASK, q), a))
    return st


def _review(final="", evidence=""):
    return {"key_findings": [{"finding": "기분 변화", "status": "설명됨"}], "contradicting": [], "confirmation": "임상 기준",
            "unresolved_danger": [], "next": None, "final_diagnosis": final, "refine_evidence": evidence}


def _diagnose(dx):
    return {"type": "DIAGNOSE", "content": dx, "reason": "추정", "confidence": 0.8}


def _run(state, *outputs):
    from doctor_agent.agent.policy import Policy

    llm = _Scripted(*outputs)
    cfg = Config().agent
    cfg.use_danger_gate = cfg.use_preconditions = False  # isolate the review/criteria logic under test
    cfg.use_confidence = False  # its one-time pushback would consume a scripted output
    return Policy(llm, cfg).next_action(state), llm


def test_review_input_includes_matching_criteria():
    state = _state("30세 여성. 주호소: 기분 변화", ("들뜬 시기가 있었나요?", "들뜬 시기가 10일 계속됐고 그때 입원했어요"))
    action, llm = _run(state, _diagnose("양극성 장애"), _review())
    assert action.type == ActionType.DIAGNOSE
    review_user = llm.messages[1][1]["content"]
    assert "[진단 기준 대조: DSM-5-TR 양극성 I/II" in review_user and "양극성 I형" in review_user
    step_user = llm.messages[0][1]["content"]
    assert "[진단 기준 대조" not in step_user  # only the reviewer sees it


def test_review_input_unchanged_without_matching_criteria():
    state = _state("40세. 주호소: 우하복부 통증", ("언제부터요?", "어제부터요"))
    _, llm = _run(state, _diagnose("급성 충수염"), _review())
    assert "[진단 기준 대조" not in llm.messages[1][1]["content"]


def test_criteria_accept_subtype_refinement_without_quoted_evidence():
    state = _state("30세 여성. 주호소: 기분 변화", ("들뜬 시기가 있었나요?", "들뜬 시기가 10일 계속됐고 그때 입원했어요"))
    action, _ = _run(state, _diagnose("양극성 장애"), _review(final="양극성 I형 장애", evidence="없음"))
    assert action.content == "양극성 I형 장애"
    assert state.reviews[0]["refinement"]["accepted"]


def test_criteria_refuse_contradicted_subtype_refinement():
    state = _state("30세 여성. 주호소: 기분 변화", ("들뜬 시기가 있었나요?", "들뜬 시기가 10일 계속됐고 그때 입원했어요"))
    action, _ = _run(state, _diagnose("양극성 장애"), _review(final="양극성 II형 장애", evidence="들뜬 시기 10일"))
    assert action.content == "양극성 장애"
    ref = state.reviews[0]["refinement"]
    assert not ref["accepted"] and "양극성 I형" in ref["rejected"]


def test_criteria_accept_takayasu_over_stenosis():
    state = _state("25세 여성. 주호소: 왼팔 저림",
                   ("팔을 쓰면 어떤가요?", "왼팔을 쓰면 저리고 아파요"),
                   ("CT 혈관조영", "좌측 쇄골하동맥 협착, 좌측 총경동맥 벽 비후"),
                   ("맥박", "왼쪽 요골 맥박이 약함. 쇄골하 잡음"))
    action, _ = _run(state, _diagnose("좌측 쇄골하동맥 협착"), _review(final="타카야수 동맥염", evidence="없음"))
    assert action.content == "타카야수 동맥염"


def test_existing_refinement_rules_still_apply_without_criteria_decision():
    from doctor_agent.agent.policy import _refinement_problem

    case = "경조증 같은 시기가 4일 정도였고 입원한 적은 없어요"  # criteria: no decision (depression unknown)
    assert not _refinement_problem("양극성 장애", "양극성 II형 장애", "경조증 4일, 입원 없음", case)
    assert _refinement_problem("양극성 장애", "양극성 II형 장애", "", case) == "근거 인용 없음"
    assert _refinement_problem("대장암", "상행결장암", "조직검사 선암", "대장내시경 조직검사 선암")


def test_ra_joint_count_phrases_do_not_crash():
    from doctor_agent.knowledge import diagnostic_criteria as dc

    for text in ("손 관절 6개가 붓고 아파요. 8주째입니다.", "6개 관절이 부었어요", "관절 12곳 압통"):
        res = dc.evaluate("ra_2010", text)
        assert res is not None


# --- 2026-09-29 second verification pass -------------------------------------------------------------------------


def test_verification_levels_second_pass():
    levels = {c.id: c.verification for c in dc.CRITERIA}
    assert levels["dka_hhs_2024"] == "primary" and levels["ra_2010"] == "primary" and levels["jones_2015"] == "primary"
    assert levels["kawasaki_aha2017"] == "secondary"
    assert {k for k, v in levels.items() if v == "unverified"} == {"light_1972", "bipolar_dsm5tr"}


def test_hhs_uses_ada_2024_figure_2b_cutoffs():
    # bicarbonate 15-17 with pH >= 7.3 is "no acidosis" for HHS (Fig. 2B: HCO3 >= 15), not only >= 18
    r = dc.evaluate("dka_hhs_2024", "혈당 720 mg/dL, 삼투압 335 mOsm/kg, pH 7.33, HCO3 16, 소변 케톤 음성.")
    assert r.decision == "고혈당성 고삼투압 상태"
    # effective osmolality > 300 counts even when the total is not stated
    r = dc.evaluate("dka_hhs_2024", "혈당 720 mg/dL, 유효 삼투압 310 mOsm/kg, pH 7.35, HCO3 20, 소변 케톤 음성.")
    assert r.decision == "고혈당성 고삼투압 상태"
    # HCO3 < 15 is acidosis for HHS; ketones negative, so neither DKA nor HHS is decided
    r = dc.evaluate("dka_hhs_2024", "혈당 720 mg/dL, 삼투압 335 mOsm/kg, pH 7.33, HCO3 13, 소변 케톤 음성.")
    assert r.decision == ""
    assert dc.evaluate("dka_hhs_2024", "유효 삼투압 290 mOsm/kg").states["osm"][0] == dc.NOT_MET


def test_jones_pr_interval_not_minor_with_carditis():
    # Table 7: prolonged PR is a minor criterion only when carditis is not counted as a major one
    assert dc.evaluate("jones_2015", "ASO 상승. 새로 생긴 심잡음, 심장염. 발열 38.8도. PR 간격 연장.").decision == ""
    assert dc.evaluate("jones_2015", "ASO 상승. 다발성 관절염. 발열 38.8도. PR 간격 연장.").decision == "급성 류마티스열"
