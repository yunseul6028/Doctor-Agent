"""Specialist sub-agent content (agent/subagents/consult.py, advocate.py): prompts, branches, sizes, parsers. No LLM."""
import json

import pytest

from doctor_agent.agent.ledger import Finding
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.agent.subagents.advocate import ADVOCATE_SCHEMA, build_advocate, parse_advocate
from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult
from doctor_agent.agent.subagents.consult import (
    BRANCHES,
    CONSULT_SCHEMA,
    HINT_MAX_CHARS,
    SPECIALTIES,
    SPECIALTY_IDS,
    build_consult,
    parse_consult,
    patient_profile,
    render_profile,
)
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge.clinical_rules import RULES
from doctor_agent.knowledge.diagnostic_criteria import CRITERIA_BY_ID
from doctor_agent.safety.danger_gate import RULE_OUT_TABLE
from doctor_agent.safety.protocols import PROTOCOLS_BY_CATEGORY

A, E, T = ActionType.ASK, ActionType.EXAM, ActionType.TEST

ADULT_M = "58세 남성. 주호소: 2시간 전 시작된 가슴 통증"
ADULT_F = "25세 여성. 주호소: 어제부터 시작된 하복부 통증"
PREGNANT = "32세 여성, 임신 28주. 주호소: 어제부터 심해진 두통"
EARLY_PREG = "29세 여성, 임신 7주. 주호소: 오늘 아침 시작된 하복부 통증과 질출혈"
POSTPARTUM = "30세 여성. 주호소: 출산 후 5일째 심한 두통"
CHILD = "3세 남아. 주호소: 이틀 전부터 열이 나요"
NEONATE = "생후 20일 된 여아. 주호소: 오늘 열이 나요"


def case(initial: str, *turns) -> CaseState:
    s = CaseState(initial_info=initial)
    for typ, content, resp in turns:
        s.turns.append(Turn(Action(typ, content), resp))
    return s


def user_msg(call: SubagentCall) -> str:
    return call.messages[1]["content"]


def sys_msg(call: SubagentCall) -> str:
    return call.messages[0]["content"]


# --- spec references point to real ids in the existing modules -----------------------------------------------------

def test_specialty_ids():
    assert set(SPECIALTY_IDS) == {"cardio", "resp_id", "gi_liver", "neuro", "rheum_immune", "peds_obgyn",
                                  "heme_onc", "renal_uro", "endo_metab", "psych"}
    assert SPECIALTY_IDS[-2:] == ("endo_metab", "psych")  # appended after the existing entries


def test_criteria_owners_after_endo_psych():
    owners = lambda cid: {sid for sid, s in SPECIALTIES.items() if cid in s.criteria_ids}  # noqa: E731
    assert owners("bipolar_dsm5tr") == {"psych"}  # moved from neuro (2026-09-29)
    assert "endo_metab" in owners("dka_hhs_2024")


def test_endo_metab_and_psych_content():
    s = case("24세 남성. 주호소: 하루 전부터 복통과 구토")
    s.ddx_ledger.update([{"dx": "당뇨병성 케톤산증", "p": 0.5, "status": "유력"}])
    endo = user_msg(build_consult(s, "endo_metab"))
    assert "위장관 질환으로 오인" in endo and "케톤" in endo and "TSH" in endo and "코르티솔" in endo and "PTH" in endo
    assert "ADA 2024 DKA/HHS" in endo  # criteria linked for the DKA candidate
    assert "정상혈당 DKA" in user_msg(build_consult(case(PREGNANT), "endo_metab"))  # ADA 2024 (2026-09-29)
    psych = user_msg(build_consult(case("35세 남성. 주호소: 최근 우울하고 이상한 행동을 해요"), "psych"))
    assert "자살" in psych and "섬망" in psych and "클로누스" in psych
    assert "기질적 원인" in sys_msg(build_consult(case(ADULT_M), "psych"))
    assert "산후 정신병" in user_msg(build_consult(case(POSTPARTUM), "psych"))
    assert "산후 정신병" not in user_msg(build_consult(case(ADULT_M), "psych"))


def test_second_pass_wording_2026_09_29():
    """Claims that were reviewer knowledge now say what the source supports (resp_id, endo_metab, psych)."""
    for sid in ("resp_id", "endo_metab", "psych"):
        assert SPECIALTIES[sid].verification == "secondary" and "Reviewer knowledge:" not in SPECIALTIES[sid].note
    peds_resp = user_msg(build_consult(case(CHILD), "resp_id"))
    assert "후두개염 시사" in peds_resp and "기구 진찰" in peds_resp and "= 후두개염" not in peds_resp  # Tibballs 2011
    assert "융모양막염" in user_msg(build_consult(case(PREGNANT), "resp_id"))  # ACOG CO 712
    endo = user_msg(build_consult(case(ADULT_M), "endo_metab"))
    assert "리튬" not in endo and "자주색 선조" not in endo and "이측 반맹" not in endo  # not in the sources read
    assert "갑상선 중독 위기 소견" in endo and "일부뿐" in endo and "유리 T4/T3" in endo
    assert "뇌부종" in user_msg(build_consult(case(CHILD), "endo_metab"))
    psych_child = user_msg(build_consult(case(CHILD), "psych"))
    assert "자살·자해 생각을 직접 물음" in psych_child and "따로 면담" not in psych_child and "괴롭힘" not in psych_child
    assert "영아 살해" in user_msg(build_consult(case(POSTPARTUM), "psych"))


def test_kdigo_aki_owned_by_renal_uro_only():
    owners = {sid for sid, s in SPECIALTIES.items() if "kdigo_aki_2012" in s.criteria_ids}
    assert owners == {"renal_uro"}


def test_heme_onc_and_renal_uro_content():
    heme = user_msg(build_consult(case("67세 남성. 주호소: 두 달째 피곤하고 체중이 줄었어요"), "heme_onc"))
    assert "말초혈액 도말" in heme and "SPEP" in heme and "혈액 악성 종양(백혈병·림프종)" in heme  # fatigue protocol
    neck = user_msg(build_consult(case("45세 남성. 주호소: 한 달 전부터 만져지는 목 멍울"), "heme_onc"))
    assert "림프종" in neck.split("[환자 구분]")[1]  # neck_mass protocol can't-miss rendered
    assert "HUS" in user_msg(build_consult(case(CHILD), "heme_onc"))
    s = case("72세 남성. 주호소: 이틀째 소변이 거의 안 나와요")
    s.ddx_ledger.update([{"dx": "급성 신손상", "p": 0.5, "status": "유력"}])
    renal = user_msg(build_consult(s, "renal_uro"))
    assert "KDIGO" in renal and "수신증" in renal and "고칼륨" in renal
    assert "KDIGO" not in user_msg(build_consult(s, "gi_liver"))
    assert "고환 염전" in user_msg(build_consult(case(CHILD), "renal_uro"))


@pytest.mark.parametrize("sid", sorted(SPECIALTIES))
def test_spec_references_exist(sid):
    spec = SPECIALTIES[sid]
    rule_ids = {r.id for r in RULES}
    assert set(spec.rule_ids) <= rule_ids
    assert set(spec.criteria_ids) <= set(CRITERIA_BY_ID)
    assert set(spec.protocol_categories) <= set(PROTOCOLS_BY_CATEGORY)
    assert set(spec.rule_out_names) <= {r.name for r in RULE_OUT_TABLE}
    assert set(spec.branch_notes) <= set(BRANCHES)
    assert spec.verification in ("primary", "secondary", "unverified")
    assert spec.must_not_miss and spec.key_asks and spec.key_exams and spec.key_tests and spec.pitfalls
    for c in spec.citations:
        assert c.pmid or c.doi or not c.verified  # guideline documents without PMID/DOI (e.g. RCOG GTG 63)


def test_consult_sources_verified_used_and_in_ledger():
    """2026-09-29 verification pass: every new source is PubMed-verified, cited by a spec and in the license ledger."""
    from pathlib import Path

    from doctor_agent.agent.subagents.consult_sources import CONSULT_SOURCES
    ledger = (Path(__file__).resolve().parents[1] / "docs" / "licenses.md").read_text(encoding="utf-8")
    used = {id(c) for s in SPECIALTIES.values() for c in s.citations}
    for c in CONSULT_SOURCES:
        assert c.verified and c.pmid, c.title
        assert id(c) in used, c.title
        assert f"PMID {c.pmid}" in ledger, c.title
    assert len({c.pmid for c in CONSULT_SOURCES}) == len(CONSULT_SOURCES)


@pytest.mark.parametrize("sid", sorted(SPECIALTIES))
def test_spec_verification_level_matches_note(sid):
    """verification = weakest level of the spec's own claims: 'unverified' iff reviewer knowledge is still listed."""
    spec = SPECIALTIES[sid]
    left = "Reviewer knowledge:" in spec.note
    assert (spec.verification == "unverified") == left, sid
    assert "reviewer knowledge." not in spec.note.lower().replace("reviewer knowledge:", ""), sid


def test_verified_claim_wording_2026_09_29():
    preg = user_msg(build_consult(case(PREGNANT), "peds_obgyn"))
    assert "단백뇨 또는 중증 소견이면 전자간증" in preg and "4시간 간격 2회" in preg  # ACOG PB 222 Box 2
    assert "산후 6주:" not in preg and "혈소판 <10만" in preg
    assert "1분기 후 ACE 억제제·ARB" in preg and "16주부터 테트라사이클린" in preg  # Dathe & Schaefer 2019
    post = render_profile(patient_profile(case(POSTPARTUM)))
    assert "12주" in post and "산후 6주까지 전자간증" not in post  # Kamel 2014; PB 222 has no 6-week limit
    assert "차폐" not in user_msg(build_consult(case(PREGNANT), "resp_id"))  # shielding not in ACOG CO 723
    child_gi = user_msg(build_consult(case(CHILD), "gi_liver"))
    assert "증상만으로 배제 불가, 초음파" in child_gi and "다리 당김" not in child_gi  # Hom 2022
    assert "하벽" not in user_msg(build_consult(case(ADULT_M), "gi_liver"))  # Canto 2000 is about MI, not inferior MI
    assert "도플러 혈류 정상으로 난소 염전 배제 금지" in user_msg(build_consult(case(ADULT_F), "peds_obgyn"))  # ACOG 783
    renal = user_msg(build_consult(case("70세 남성. 주호소: 옆구리 통증"), "renal_uro"))
    assert "심전도는 고칼륨에 둔감" in renal and "영상 기다리지 말고" in renal and "근색소뇨(또는 혈색소뇨)" in renal
    assert "누우면 숨참" not in user_msg(build_consult(case(ADULT_M), "heme_onc"))  # not in Rice 2006


def test_every_protocol_category_has_an_owner():
    owned = {c for s in SPECIALTIES.values() for c in s.protocol_categories}
    # fatigue / neck_mass were unowned until heme_onc was added (2026-09-28)
    assert set(PROTOCOLS_BY_CATEGORY) <= owned, set(PROTOCOLS_BY_CATEGORY) - owned
    assert {"fatigue", "neck_mass"} <= set(SPECIALTIES["heme_onc"].protocol_categories)


# --- prompt builds, per specialty and branch --------------------------------------------------------------------

@pytest.mark.parametrize("sid", sorted(SPECIALTIES))
@pytest.mark.parametrize("initial", [ADULT_M, ADULT_F, PREGNANT, CHILD])
def test_build_every_specialty_and_branch(sid, initial):
    call = build_consult(case(initial), sid)
    assert isinstance(call, SubagentCall)
    assert call.name == f"consult:{sid}"
    assert call.json_schema is CONSULT_SCHEMA
    assert [m["role"] for m in call.messages] == ["system", "user"]
    spec = SPECIALTIES[sid]
    assert spec.name_ko in sys_msg(call) and "JSON" in sys_msg(call)
    u = user_msg(call)
    assert initial in u and "[소견 장부" in u and "[현재 감별 진단]" in u and "[환자 구분]" in u
    assert "{" not in sys_msg(call).split("\n")[0]  # format() braces resolved


def test_unknown_specialty_raises_keyerror():
    with pytest.raises(KeyError):
        build_consult(case(ADULT_M), "derm")


def test_profile_branches():
    assert patient_profile(case(ADULT_M))["branch"] == "adult"
    assert patient_profile(case(ADULT_F))["branch"] == "female_repro"
    p = patient_profile(case(PREGNANT))
    assert p["branch"] == "pregnant" and p["weeks"] == 28
    assert patient_profile(case(EARLY_PREG))["weeks"] == 7
    pp = patient_profile(case(POSTPARTUM))
    assert pp["branch"] == "pregnant" and pp["postpartum"]
    c = patient_profile(case(CHILD))
    assert c["branch"] == "peds" and c["age"] == 3
    n = patient_profile(case(NEONATE))
    assert n["branch"] == "peds" and n["age"] < 0.1


def test_pregnancy_learned_later_switches_branch():
    s = case(ADULT_F, (A, "임신 가능성이 있나요?", "네, 지금 임신 10주예요."))
    p = patient_profile(s)
    assert p["branch"] == "pregnant" and p["weeks"] == 10


def test_unavailable_response_is_not_read_for_pregnancy():
    s = case(ADULT_F, (T, "소변 임신 검사", "임신 반응 양성 결과는 제공되지 않습니다."))
    assert patient_profile(s)["branch"] == "female_repro"


def test_peds_profile_has_age_specific_vitals():
    t = render_profile(patient_profile(case(CHILD)))
    assert "3세" in t and "맥박 70~136/분" in t and "호흡수 17~33/분" in t and "76 mmHg" in t
    t = render_profile(patient_profile(case(NEONATE)))
    assert "생후 20일" in t and "60 mmHg" in t
    assert "나이 미상" in render_profile(patient_profile(case("아기가 열이 나요")))


def test_peds_obgyn_branch_content():
    peds = user_msg(build_consult(case(CHILD), "peds_obgyn"))
    preg = user_msg(build_consult(case(PREGNANT), "peds_obgyn"))
    fem = user_msg(build_consult(case(ADULT_F), "peds_obgyn"))
    adult = user_msg(build_consult(case(ADULT_M), "peds_obgyn"))
    assert "영아 발열" in peds and "전자간증" not in peds.split("[환자 구분]")[1].split("[처음 정보]")[0]
    assert "혈압 ≥140/90" in preg and "가돌리늄" in preg and "20주 이후" in preg and "임신 금기 약" in preg
    assert "β-hCG" in fem and "가임기 여성(임신 여부 미확인)" in fem
    assert "특이 위험 없음" in adult
    for text in (peds, adult):
        assert "임신 금기 약" not in text and "가돌리늄" not in text


def test_other_specialties_get_branch_notes():
    assert "주산기 심근병증" in user_msg(build_consult(case(PREGNANT), "cardio"))
    assert "주산기 심근병증" not in user_msg(build_consult(case(ADULT_M), "cardio"))
    assert "후두개염" in user_msg(build_consult(case(CHILD), "resp_id"))


# --- knowledge links only when they apply ------------------------------------------------------------------------

def test_rules_only_in_population_and_specialty():
    u = user_msg(build_consult(case(ADULT_M), "cardio"))
    assert "HEART" in u and "ADD-RS" in u
    assert "Alvarado" not in u
    assert "임상 결정 규칙" not in user_msg(build_consult(case(ADULT_M), "rheum_immune"))


def test_protocol_cant_miss_filtered_by_specialty():
    u = user_msg(build_consult(case(ADULT_M), "cardio"))
    assert "배제할 위험 질환(안전 프로토콜)] 급성 관상동맥 증후군" in u
    assert "안전 프로토콜" not in user_msg(build_consult(case(ADULT_M), "gi_liver"))


def test_criteria_follow_the_ddx():
    s = case("45세 남성. 주호소: 3주째 계속되는 발열")
    s.ddx_ledger.update([{"dx": "감염성 심내막염", "p": 0.4, "status": "유력"}])
    assert "Duke-ISCVID" in user_msg(build_consult(s, "cardio"))
    assert "Duke-ISCVID" not in user_msg(build_consult(case("45세 남성. 주호소: 3주째 계속되는 발열"), "cardio"))


# --- grounding of the case context -----------------------------------------------------------------------------

def test_unverified_findings_are_left_out():
    s = case(ADULT_M, (E, "양팔 혈압", "우측 150/90, 좌측 118/80 mmHg"))
    s.findings.add(Finding("양팔 혈압 차이", "양성", "32 mmHg", 1, verified=True))
    s.findings.add(Finding("심낭 마찰음", "양성", "", 1, verified=False))
    u = user_msg(build_consult(s, "cardio"))
    assert "양팔 혈압 차이" in u and "심낭 마찰음" not in u
    assert "EXAM 양팔 혈압" in u  # actions done are listed so the consult does not repeat them


def test_unavailable_results_are_marked_in_done_list():
    s = case(ADULT_M, (T, "관상동맥 CT", "해당 검사 결과는 제공되지 않습니다."))
    assert "TEST 관상동맥 CT (결과 없음)" in user_msg(build_consult(s, "cardio"))


def test_resources_optional_and_tolerant():
    s = case(ADULT_M)
    assert "[참고 자료]" not in user_msg(build_consult(s, "cardio", {}))
    assert "[참고 자료]" not in user_msg(build_consult(s, "cardio", None))
    assert "[참고 자료]" not in user_msg(build_consult(s, "cardio", {"kb": [], "x": ""}))
    u = user_msg(build_consult(s, "cardio", {"kb": ["대동맥 박리: 찢어지는 통증", {"a": 1}], "rules": {"HEART": "0-3 저위험"},
                                            "n": 5, "long": "가" * 5000}))
    assert "[참고 자료]" in u and "찢어지는 통증" in u and "HEART: 0-3 저위험" in u
    assert u.count("가") < 1000


# --- size bounds -------------------------------------------------------------------------------------------------

def _big_state(initial: str) -> CaseState:
    s = case(initial, *[(T if i % 3 else A, f"검사 항목 {i} 이름이 꽤 긴 편인 요청", "결과 " + "수치 정상 " * 40)
                        for i in range(60)])
    for i in range(80):
        s.findings.add(Finding(f"소견{i}번 항목", ("양성", "음성", "결과없음")[i % 3], "세부 " * 8, i, verified=True))
    s.ddx_ledger.update([{"dx": f"후보 질환 {i}", "p": 0.1, "status": "유력", "for": ["근거 " * 10] * 4,
                          "against": ["반대 " * 10] * 4} for i in range(8)])
    return s


SPEC_RENDER_MAX = 1000
SYSTEM_MAX = 900
USER_MAX = 6500  # worst case: 60 turns, 80 findings, 8 DDx entries, full resources


@pytest.mark.parametrize("sid", sorted(SPECIALTIES))
def test_prompt_size_bounds(sid):
    spec = SPECIALTIES[sid]
    for b in BRANCHES:
        assert len(spec.render(b)) <= SPEC_RENDER_MAX, (sid, b, len(spec.render(b)))
    for initial in (ADULT_M, PREGNANT, CHILD):
        call = build_consult(_big_state(initial), sid, {"kb": ["x" * 400] * 5})
        assert len(sys_msg(call)) <= SYSTEM_MAX
        assert len(user_msg(call)) <= USER_MAX, (sid, len(user_msg(call)))


def test_advocate_size_bound():
    call = build_advocate(_big_state(ADULT_M), "급성 심근경색", "심전도 ST 상승" * 50)
    assert len(sys_msg(call)) <= SYSTEM_MAX + 200
    assert len(user_msg(call)) <= USER_MAX


# --- consult parser ----------------------------------------------------------------------------------------------

GOOD = {"assessment": "허혈성 흉통 가능성, 대동맥 박리 미배제",
        "ddx_add": [{"name": "대동맥 박리", "why": "양팔 혈압 차이"}],
        "missed_dangers": ["대동맥 박리", "폐색전증"],
        "next_actions": [{"type": "TEST", "content": "흉부 대동맥 CT 혈관조영", "why": "박리 확인"},
                         {"type": "EXAM", "content": "새 이완기 심잡음 청진", "why": "대동맥판 역류"}],
        "confidence_note": "중간"}


def test_parse_plain_json():
    r = parse_consult(json.dumps(GOOD, ensure_ascii=False), "cardio")
    assert isinstance(r, SubagentResult) and r.ok and r.name == "consult:cardio"
    assert r.hint_ko.startswith("[심장·혈관 자문]")
    assert r.ddx_add == [{"name": "대동맥 박리", "why": "양팔 혈압 차이"}]
    assert [a["type"] for a in r.suggested_actions] == ["TEST", "EXAM"]
    assert r.red_flags == ["대동맥 박리", "폐색전증"]
    assert r.raw["confidence_note"] == "중간"


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```",
    "여기 답입니다:\n{}\n이상입니다. {{참고}}",
    "<|start|>assistant<|channel|>analysis<|message|>생각 {{\"assessment\": \"초안\"}}<|end|>"
    "<|start|>assistant<|channel|>final<|message|>{}<|return|>",
    "analysis먼저 생각해 보면...assistantfinal{}",
    "<think>{{\"assessment\": \"초안\"}}</think>\n{}",
])
def test_parse_wrappers(wrap):
    text = wrap.replace("{}", json.dumps(GOOD, ensure_ascii=False), 1)
    r = parse_consult(text, "cardio")
    assert r.ok and r.raw["assessment"] == GOOD["assessment"]


def test_parse_truncated_json():
    full = json.dumps(GOOD, ensure_ascii=False)
    cut = full[: full.index("새 이완기") + 3]  # cut inside the second action
    r = parse_consult(cut, "cardio")
    assert r.ok and r.raw["assessment"] == GOOD["assessment"]
    assert r.suggested_actions[0]["content"] == "흉부 대동맥 CT 혈관조영"


@pytest.mark.parametrize("bad", [None, "", "모르겠습니다", "{broken", "[1, 2, 3]", "{\"foo\": 1}", "}{", "{\"assessment\": "])
def test_parse_failures_never_raise(bad):
    r = parse_consult(bad, "cardio")
    assert r.ok is False and r.hint_ko == "" and r.ddx_add == [] and r.suggested_actions == [] and r.red_flags == []


def test_empty_json_gives_no_hint():
    for obj in ({"assessment": "", "ddx_add": [], "missed_dangers": [], "next_actions": [], "confidence_note": ""},
                {"assessment": None, "next_actions": None}):
        r = parse_consult(json.dumps(obj), "neuro")
        assert r.ok is False and r.hint_ko == ""


def test_hint_has_no_invented_lines():
    r = parse_consult(json.dumps({"assessment": "편두통 가능성", "ddx_add": [], "missed_dangers": [], "next_actions": []}),
                      "neuro")
    assert r.ok and r.hint_ko == "[신경 자문] 편두통 가능성"


def test_wrong_action_types_and_extra_keys():
    obj = {"assessment": "평가", "extra": {"x": 1}, "next_actions": [
        {"type": "DIAGNOSE", "content": "급성 심근경색"}, {"type": "PLAN", "content": "입원"},
        {"type": "test", "content": "트로포닌", "reason": "허혈"}, {"type": "검사", "content": "심전도"},
        {"type": "ASK", "content": ""}, "TEST 흉부 X선", {"type": "문진", "content": "통증 양상"},
        {"type": "EXAM", "content": "청진"}]}
    r = parse_consult(json.dumps(obj, ensure_ascii=False), "cardio")
    assert r.ok and r.raw["extra"] == {"x": 1}
    assert r.suggested_actions == [{"type": "TEST", "content": "트로포닌", "why": "허혈"},
                                   {"type": "TEST", "content": "심전도", "why": ""},
                                   {"type": "ASK", "content": "통증 양상", "why": ""}]
    assert "DIAGNOSE" not in r.hint_ko and "입원" not in r.hint_ko


def test_odd_field_types_are_coerced():
    obj = {"assessment": 42, "ddx_add": "심낭염", "missed_dangers": "대동맥 박리",
           "next_actions": {"type": "EXAM", "content": "심낭 마찰음 청진"}, "confidence_note": ["x"]}
    r = parse_consult(json.dumps(obj, ensure_ascii=False), "cardio")
    assert r.ok and r.ddx_add == [{"name": "심낭염", "why": ""}] and r.red_flags == ["대동맥 박리"]
    assert r.suggested_actions[0]["type"] == "EXAM"


def test_known_ddx_and_duplicates_dropped():
    obj = {"assessment": "a", "ddx_add": [{"name": "폐색전증"}, {"name": "폐동맥 색전증"}, {"dx": "심낭염 (pericarditis)"},
                                         {"name": "심낭염"}, {"name": "기흉"}, {"name": "식도 파열"}]}
    r = parse_consult(json.dumps(obj, ensure_ascii=False), "cardio", known_ddx=["폐색전증 의심"])
    names = [d["name"] for d in r.ddx_add]
    assert "폐색전증" not in names and names[0] == "폐동맥 색전증" and len(names) == 3
    assert sum("심낭염" in n for n in names) == 1


def test_hint_is_capped():
    obj = {"assessment": "가" * 1000, "missed_dangers": ["나" * 200] * 3,
           "next_actions": [{"type": "TEST", "content": "다" * 300, "why": "라" * 300}] * 3,
           "ddx_add": [{"name": f"질환{i}", "why": "마" * 200} for i in range(5)]}
    r = parse_consult(json.dumps(obj, ensure_ascii=False), "gi_liver")
    assert r.ok and len(r.hint_ko) <= HINT_MAX_CHARS
    assert len(parse_consult(json.dumps(obj, ensure_ascii=False), "gi_liver", max_chars=150).hint_ko) <= 150
    assert len(r.ddx_add) <= 3 and len(r.suggested_actions) <= 3 and len(r.red_flags) <= 3


# --- advocate ----------------------------------------------------------------------------------------------------

def test_build_advocate():
    s = case(ADULT_M, (T, "심전도", "II, III, aVF ST 분절 상승"))
    call = build_advocate(s, "급성 하벽 심근경색", "하벽 ST 상승")
    assert call.name == "advocate" and call.json_schema is ADVOCATE_SCHEMA
    u = user_msg(call)
    assert "[제안 진단] 급성 하벽 심근경색" in u and "[주치의 근거] 하벽 ST 상승" in u
    assert "대동맥 박리" in u  # chest-pain can't-miss list from safety/protocols.py
    assert "반대 의견" in sys_msg(call) and "유지" in sys_msg(call)


def test_build_advocate_peds_and_pregnant_profiles():
    assert "소아 3세" in user_msg(build_advocate(case(CHILD), "바이러스성 상기도 감염"))
    assert "임신 중 28주" in user_msg(build_advocate(case(PREGNANT), "긴장형 두통"))


ADV = {"alternatives": [{"name": "대동맥 박리", "why": "양팔 혈압 차이"}, {"name": "급성 심근경색", "why": "same"},
                        {"name": "심낭염", "why": "x"}],
       "unexplained": ["양팔 혈압 차이 32 mmHg"],
       "refuting_test": {"type": "TEST", "content": "흉부 대동맥 CT 혈관조영", "why": "박리면 진단이 바뀜"},
       "dangers_not_excluded": ["대동맥 박리"], "verdict": "재검토", "note": "혈압 차이가 설명 안 됨", "extra": 1}


def test_parse_advocate():
    r = parse_advocate("```json\n" + json.dumps(ADV, ensure_ascii=False) + "\n```", proposed_dx="급성 심근경색")
    assert r.ok and r.name == "advocate"
    assert [d["name"] for d in r.ddx_add] == ["대동맥 박리", "심낭염"]  # proposed dx dropped, max 2
    assert r.suggested_actions == [{"type": "TEST", "content": "흉부 대동맥 CT 혈관조영", "why": "박리면 진단이 바뀜"}]
    assert r.red_flags == ["대동맥 박리"] and r.raw["verdict"] == "재검토" and r.raw["extra"] == 1
    assert r.hint_ko.startswith("[반대 의견] 제안 진단 '급성 심근경색': 재검토")
    assert "설명 안 되는 소견: 양팔 혈압 차이 32 mmHg" in r.hint_ko


def test_parse_advocate_refuting_diagnose_dropped_and_null():
    obj = dict(ADV, refuting_test={"type": "DIAGNOSE", "content": "대동맥 박리"})
    assert parse_advocate(json.dumps(obj, ensure_ascii=False)).suggested_actions == []
    obj = dict(ADV, refuting_test=None, verdict="keep")
    r = parse_advocate(json.dumps(obj, ensure_ascii=False))
    assert r.suggested_actions == [] and r.raw["verdict"] == "유지"


def test_parse_advocate_empty_and_broken():
    for bad in (None, "", "없음", "{\"alternatives\": [], \"unexplained\": [], \"dangers_not_excluded\": []}"):
        r = parse_advocate(bad)
        assert r.ok is False and r.hint_ko == ""
    r = parse_advocate("{\"verdict\": \"유지\", \"note\": \"근거 충분\"}", "편두통")
    assert r.ok and r.hint_ko == "[반대 의견] 제안 진단 '편두통': 유지 — 근거 충분"
    trunc = json.dumps(ADV, ensure_ascii=False)[:120]
    assert parse_advocate(trunc).ok
