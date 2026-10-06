import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import clinical_rules as cr  # noqa: E402
from doctor_agent.safety import protocols as pr  # noqa: E402
from doctor_agent.safety.protocols import cant_miss_for  # noqa: E402

# --- scoring with known examples -------------------------------------------------------------


def test_wells_pe_bands():
    r = cr.score("wells_pe", {"dvt_signs": True, "pe_most_likely": True, "hr_gt_100": True})
    assert r.score == 7.5
    assert r.label == "고확률"
    assert "PE 가능성 높음" in r.secondary_label
    low = cr.score("wells_pe", {"hemoptysis": True})
    assert low.score == 1.0 and low.label == "저확률" and "가능성 낮음" in low.secondary_label
    assert cr.score("wells_pe", {"pe_most_likely": True, "hr_gt_100": True}).label == "중간 확률"  # 4.5
    assert cr.score("wells_pe", {k.key: False for k in cr.RULES_BY_ID["wells_pe"].items}).complete


def test_perc_rule_out_requires_all_items():
    keys = [i.key for i in cr.RULES_BY_ID["perc"].items]
    full = cr.score("perc", {k: False for k in keys})
    assert full.score == 0 and full.complete and full.label == "PERC 음성"
    partial = cr.score("perc", {k: False for k in keys[:5]})
    assert not partial.complete and "배제 불가" in partial.label
    assert cr.score("perc", {**{k: False for k in keys}, "age_ge_50": True}).label == "PERC 양성"


def test_heart_graded_items():
    r = cr.score("heart", {"history": 2, "ecg": 1, "age": "65세 이상", "risk_factors": 1, "troponin": 0})
    assert r.score == 6 and r.label == "중간 위험"
    assert cr.score("heart", {"history": 2, "ecg": 2, "age": 2, "risk_factors": 2, "troponin": 0}).label == "고위험"
    with pytest.raises(ValueError):
        cr.score("heart", {"history": 3})
    with pytest.raises(ValueError):
        cr.score("heart", {"not_an_item": True})


def test_qsofa_and_curb65():
    assert cr.score("qsofa", {"rr_ge_22": True, "sbp_le_100": True, "altered_mentation": False}).label == "qSOFA 양성"
    assert cr.score("qsofa", {"rr_ge_22": True}).label == "qSOFA 낮음"
    r = cr.score("curb65", {"confusion": True, "urea_gt_7": True, "rr_ge_30": True, "low_bp": False, "age_ge_65": True})
    assert r.score == 4 and r.label == "고위험"
    assert cr.score("curb65", {"age_ge_65": True, "urea_gt_7": True}).label == "중간 위험"


def test_abcd2_and_centor():
    r = cr.score("abcd2", {"age_ge_60": True, "bp_ge_140_90": True, "clinical": 2, "duration": 2, "diabetes": True})
    assert r.score == 7 and r.label == "고위험" and r.max_score == 7
    assert cr.score("centor", {"tonsillar_exudate": True, "tender_anterior_nodes": True, "no_cough": True,
                               "fever_history": True}).meaning.endswith("56%")


def test_group_and_tier_rules():
    add = cr.score("add_rs", {"marfan_ctd": True, "family_history": True, "tearing_pain": True})
    assert add.score == 2 and add.label == "고위험"  # two categories, not three items
    assert cr.score("cchr", {"amnesia_gt30": True}).label == "중등도 위험"
    assert cr.score("cchr", {"amnesia_gt30": True, "age_ge_65": True}).label == "고위험"
    keys = [i.key for i in cr.RULES_BY_ID["cchr"].items]
    assert cr.score("cchr", {k: False for k in keys}).label == "저위험"


def test_ottawa_sah_rule_out():
    keys = [i.key for i in cr.RULES_BY_ID["ottawa_sah"].items]
    assert cr.score("ottawa_sah", {k: False for k in keys}).label == "규칙 음성"
    assert cr.score("ottawa_sah", {"thunderclap": True}).label == "규칙 양성"


def test_thresholds_cover_every_reachable_score():
    for rule in cr.RULES:
        if rule.method != "sum":
            scores = range(0, int(rule.max_score) + 1)
        else:
            steps = sorted({0.0} | {p for i in rule.items for p in ([i.points] if not i.options else
                                                                     [p for _, p in i.options])})
            scores = [x / 2 for x in range(0, int(rule.max_score * 2) + 1)] if 0.5 in steps or 1.5 in steps \
                else range(0, int(rule.max_score) + 1)
        for s in scores:
            assert any(t.contains(s) for t in rule.thresholds), (rule.id, s)


# --- matching --------------------------------------------------------------------------------


def _ids(text):
    return {r.id for r in cr.rules_for(text)}


def test_rules_for_korean_chief_complaints():
    assert {"heart", "wells_pe", "perc", "add_rs"} <= _ids("55세 남성, 2시간 전부터 가슴이 아파요")
    assert {"wells_pe", "perc"} <= _ids("숨이 차요")
    # CURB-65 needs a pneumonia signal (cough/sputum) on top of fever/dyspnea; the old expectation that bare
    # "숨이 차요" matches CURB-65 encoded the over-matching bug
    assert "curb65" not in _ids("숨이 차요")
    assert "curb65" in _ids("3일 전부터 열이 나고 기침, 가래가 있어요")
    assert "ottawa_sah" in _ids("갑자기 머리가 깨질 듯이 아파요")
    assert "qsofa" in _ids("열이 나고 오한이 있어요")
    assert "alvarado" in _ids("오른쪽 아랫배가 아파요")
    assert "abcd2" in _ids("한쪽 팔에 힘이 빠졌다가 돌아왔어요")
    assert "centor" in _ids("인후통과 열")
    assert "cchr" in _ids("넘어지면서 머리를 부딪혔어요")
    assert "heart" in _ids("Chest pain for 2 hours")
    assert _ids("무릎에 멍이 들었어요") == set()


def test_render_for_prompt_contains_items_thresholds_citation():
    text = cr.render_for_prompt(cr.rules_for("흉통"))
    assert "Wells PE (Wells 2000)" in text and "HEART (Six 2008)" in text
    assert "심박수 100회/분 초과 +1.5" in text and "해석:" in text
    assert cr.render_for_prompt([]) == ""


# --- citations -------------------------------------------------------------------------------


def _check_citation(c):
    assert c.authors and c.title and c.journal and c.year > 1900 and c.volume_pages
    assert c.doi or c.pmid
    assert c.verified
    assert c.url.startswith("https://")


def test_every_rule_has_citation():
    assert len(cr.RULES) >= 12
    for r in cr.RULES:
        _check_citation(r.citation)
        assert r.verification in {"primary", "secondary", "unverified"}
        assert r.name_ko and r.name_en and r.items and r.thresholds


def test_every_protocol_check_has_citation():
    assert set(pr.PROTOCOLS_BY_CATEGORY) == set(cr.CATEGORY_NAMES)
    for p in pr.PROTOCOLS:
        assert p.cant_miss and p.checks
        for c in p.checks:
            _check_citation(c.citation)
            assert c.keywords and all(k == k.lower() for k in c.keywords)
            assert all(k == k.lower() for k in c.triggers)
            assert c.verification in {"primary", "secondary", "unverified"}


# --- protocols / red flags -------------------------------------------------------------------


def test_protocol_checks_and_conditionals():
    ids = {c.id for c in pr.must_checks_for("가슴이 아파요")}
    assert {"ecg", "troponin", "vitals"} <= ids and "aorta_imaging" not in ids
    assert "aorta_imaging" in {c.id for c in pr.must_checks_for("가슴이 찢어지는 듯 아프고 등으로 뻗쳐요")}
    assert "brain_ct" in {c.id for c in pr.must_checks_for("갑자기 벼락 치듯 두통")}
    assert "pregnancy_test" in {c.id for c in pr.must_checks_for("28세 여성, 아랫배가 아파요")}
    assert "pregnancy_test" not in {c.id for c in pr.must_checks_for("28세 남성, 아랫배가 아파요")}
    pending = {c.id for c in pr.pending_checks("가슴이 아파요", ["12유도 ECG 검사", "혈압과 맥박 측정"])}
    assert "ecg" not in pending and "vitals" not in pending and "troponin" in pending
    ecg = next(c for c in pr.must_checks_for("흉통") if c.id == "ecg")
    assert "심전도" in ecg.keywords
    assert pr.cant_miss_for("흉통")


def test_cant_miss_for_chief_complaints():
    flags = cant_miss_for("가슴 통증과 호흡곤란")
    assert "대동맥 박리" in flags and "급성 심부전" in flags
    assert flags.count("폐색전증") == 1
    assert "지주막하 출혈" in cant_miss_for("Headache since this morning")
    assert cant_miss_for("무릎 찰과상") == []


def test_category_from_chief_complaint_only():
    # a pertinent negative about dyspnea must not pull in the dyspnea/chest protocols for an abdominal case
    ids = {c.id for c in pr.must_checks_for("45세 여성. 주호소: 오른쪽 윗배 통증", "아니요, 숨은 안 차요. 가슴은 괜찮아요.")}
    assert "ecg" not in ids and "troponin" not in ids
    # but conditional checks still trigger from later facts
    assert "brain_ct" in {c.id for c in pr.must_checks_for("47세 여성. 주호소: 두통", "갑자기 벼락 치듯 아팠어요")}


# --- applicability (validated populations) and false-trigger regressions ----------------------


def _checks(initial, facts=""):
    return {c.id for c in pr.must_checks_for(initial, facts) if c.kind != "treatment"}


def test_chronic_headache_hypertension_no_ottawa_sah():
    # Takayasu-like: 2-month headache + hypertension is outside Ottawa SAH (acute, peak within 1 h)
    cc = "45세 남성. 주호소: 2개월 전부터 시작된 두통과 고혈압"
    assert "ottawa_sah" not in _ids(cc)
    ids = _checks(cc, "2개월 전부터 두통과 고혈압이 계속됐어요. 최근에 두통이 갑자기 심해졌어요.")
    assert "brain_ct" not in ids and {"vitals", "neuro_exam"} <= ids
    assert "ottawa_sah" not in _ids("35세 여성. 주호소: 반복되는 두통")  # not a new headache
    assert "ottawa_sah" not in _ids("30세 남성. 주호소: 넘어지면서 머리를 부딪힌 후 두통")  # traumatic
    assert "ottawa_sah" not in _ids("10세 남아. 주호소: 갑자기 시작된 두통")  # age < 16


def test_acute_thunderclap_headache_keeps_ottawa_and_ct():
    cc = "47세 여성. 주호소: 오늘 오후 갑자기 시작된 심한 두통"
    assert "ottawa_sah" in _ids(cc)
    assert {"brain_ct", "neuro_exam", "vitals"} <= _checks(cc, "1분도 안 돼서 제일 심해졌어요. 태어나서 처음이에요.")


def test_anaphylaxis_not_scored_as_stroke():
    cc = "45세 남성. 주호소: 운동 중 갑자기 발생한 전신 두드러기와 어지러움"
    assert cr.detect_categories(cc) == ["allergy"]
    ids = _checks(cc, "운동 중 두드러기와 호흡곤란이 생기고 잠깐 실신했어요. 의식이 흐려졌어요.")
    assert not ids & {"glucose", "onset_time", "brain_imaging"}
    assert {"vitals", "airway_breathing"} <= ids
    assert "abcd2" not in _ids(cc)
    # syncope alone and isolated dizziness are not the stroke protocol either
    assert "neuro" not in cr.detect_categories("40대 여성. 주호소: 입원 중 발생한 실신")
    assert "neuro" not in cr.detect_categories("가슴 두근거림과 어지럼증")
    # the epinephrine treatment check still exists for prompts but is never scored
    assert "epinephrine" in {c.id for c in pr.must_checks_for(cc)}


def test_focal_deficits_still_trigger_stroke_protocol():
    for cc in ("65세 남성. 주호소: 1시간 전부터 오른쪽 팔다리에 힘이 빠짐", "말이 어눌해지고 얼굴이 비뚤어졌어요",
               "갑자기 어지럽고 걸을 때 비틀거려요", "2일 전부터 시작된 의식 변화", "갑작스러운 전신 경련"):
        assert "neuro" in cr.detect_categories(cc), cc
        assert {"glucose", "brain_imaging"} <= _checks(cc), cc
    assert "neuro" not in cr.detect_categories("1개월 동안의 발작적인 기침과 호흡곤란")  # "발작적" is not seizure
    assert "neuro" not in cr.detect_categories("최근 수개월간 진행된 손발 저림과 기억력 감퇴")  # chronic, bilateral
    assert "neuro" not in cr.detect_categories("55세 남성. 주호소: 가슴 통증과 왼팔 저림")  # radiation, not focal


def test_chronic_febrile_cough_no_sepsis_checks():
    # hypersensitivity pneumonitis: weeks of cough/dyspnea/low-grade fever without instability
    cc = "38세 남성. 주호소: 1개월 전부터 시작된 기침, 호흡곤란, 미열"
    ids = _checks(cc, "1개월 전부터 마른 기침과 호흡곤란, 미열이 나기 시작했어요.")
    assert not ids & {"blood_culture", "lactate"}
    assert not _ids(cc) & {"qsofa", "curb65", "wells_pe", "perc"}
    # ...but instability on top of a chronic course still brings them back
    assert {"blood_culture", "lactate"} <= _checks(cc, "어제부터 의식이 흐려지고 혈압이 떨어졌어요.")
    assert {"blood_culture", "lactate"} <= _checks(cc, "혈압 85/50 mmHg, 맥박 120회/분")


def test_acute_fever_with_systemic_signs_keeps_sepsis_checks():
    assert {"blood_culture", "lactate"} <= _checks("35세 여성. 주호소: 2일 전부터 시작된 옆구리 통증과 배뇨통, 고열")
    assert {"blood_culture", "lactate"} <= _checks("49세 남성. 주호소: 윗배 통증, 발열, 오한")
    assert {"blood_culture", "lactate"} <= _checks("24세 여성. 주호소: 갑자기 시작된 발열")
    assert not _checks("28세 여성. 주호소: 5일 전부터 시작된 발열, 기침, 인후통", "노란 가래가 나와요.") & {
        "blood_culture", "lactate"}
    assert "qsofa" in _ids("열이 나고 오한이 있어요")


def test_acute_chest_pain_keeps_acs_checks():
    cc = "58세 남성. 주호소: 2시간 전부터 시작된 가슴통증"
    assert {"vitals", "ecg", "troponin", "cxr"} <= _checks(cc)
    assert {"heart", "wells_pe", "perc", "add_rs"} <= _ids(cc)
    # 2 weeks of recurrent chest pain may be unstable angina: troponin + HEART still apply
    assert "troponin" in _checks("47세 남성. 주호소: 2주 전부터 반복되는 가슴 통증")
    # months-long chest pain: ECG yes, acute troponin / HEART no
    chronic = "50세 여성. 주호소: 6개월 전부터 반복되는 가슴 통증"
    assert "ecg" in _checks(chronic) and "troponin" not in _checks(chronic)
    assert not _ids(chronic) & {"heart", "wells_pe", "perc", "add_rs"}


def test_pertinent_negatives_do_not_trigger_conditionals():
    cc = "61세 남성. 주호소: 40분 전부터 시작된 가슴 통증"
    ids = _checks(cc, "왼쪽 팔이랑 턱 쪽까지 퍼져요. 등이 찢어지는 느낌은 아니에요. 다리 붓거나 최근에 오래 비행기 탄 적은 없어요.")
    assert "aorta_imaging" not in ids and "pe_workup" not in ids
    assert "aorta_imaging" in _checks(cc, "등으로 찢어지듯 뻗쳐요.")
    # "머리가 아프고 입맛도 없어요": the negation belongs to the second clause only
    assert "meningitis_workup" in _checks("24세 여성. 주호소: 갑자기 시작된 발열", "갑자기 열이 오르고 머리가 아프고 입맛도 없어요.")


def test_rule_applicability_populations():
    assert "alvarado" in _ids("28세 남성. 주호소: 어제부터 시작된 복통")
    assert "alvarado" in _ids("명치에서 시작해 오른쪽 아랫배로 옮겨간 통증")  # RLQ keyword overrides location
    assert "alvarado" not in _ids("53세 여성. 주호소: 어젯밤부터 계속되는 윗배 통증")
    assert "alvarado" not in _ids("38세 여성. 주호소: 6개월간 지속된 복통과 잦은 설사")
    assert "abcd2" not in _ids("22세 여성. 주호소: 갑작스러운 의식 소실 및 전신 경련")  # not a TIA
    assert "bisap" in _ids("급성 췌장염으로 입원") and "bisap" not in _ids("만성 췌장염 6개월")
    assert not _ids("신생아. 주호소: 태어난 지 3시간 만에 생긴 호흡곤란") & {"wells_pe", "perc", "curb65"}  # adult-only rules
    assert "cchr" not in _ids("3개월 전 머리를 부딪혔어요")


def test_duration_and_age_parsing():
    assert cr.duration_level("2개월 전부터 시작된 두통") == 2
    assert cr.duration_level("3주간 지속된 기침") == 1
    assert cr.duration_level("1주일 전부터 시작된 두통") == 0
    assert cr.duration_level("1시간 전부터 시작된 복통 (임신 38주)") == 0
    assert cr.duration_level("22세 임신 30주 여성. 주호소: 발열") == 0
    assert cr.duration_level("3개월 전부터 두통, 오늘 갑자기 심해짐") == 0  # acute-on-chronic
    assert cr.age_years("45세 남성") == 45 and cr.age_years("40대 후반 여성") == 40
    assert cr.age_years("신생아. 주호소: 호흡곤란") == 0 and cr.age_years("주호소: 두통") is None


def test_endocarditis_clues_bring_blood_cultures_any_duration():
    cc = "21세 여성. 주호소: 1주일 전부터 뒷머리 두통, 한 달 전부터 종아리 통증, 전신 피로감과 발열"
    # the IE clue (murmur) is an exam finding, not in the history
    assert "blood_culture_ie" not in _checks(cc, "앓는 병은 없고 최근 치과 치료도 없었어요. 마약은 안 해요.")
    assert "blood_culture_ie" in _checks(cc, "체온 39°C. 좌측 흉골 방 영역에서 확장기 잡음이 청진됨.")
    chronic = "45세 남성. 주호소: 3주 전부터 계속되는 미열"
    assert "blood_culture_ie" in _checks(chronic, "작년에 인공판막 수술을 받았어요.")
    assert "blood_culture_ie" in _checks(chronic, "2주 전에 치과에서 이를 뽑았어요.")
    assert "blood_culture_ie" in _checks(chronic, "손톱 밑에 선상 출혈이 보여요.")
    assert "blood_culture_ie" not in _checks(chronic, "심잡음은 없다고 들었어요.")
    assert not _checks(chronic, "기침이 조금 있어요.") & {"blood_culture", "blood_culture_ie"}
    # sepsis already covers blood cultures: no double-weighted second check
    ids = _checks("30세 남성. 주호소: 어제부터 고열과 오한", "주사 마약을 해요.")
    assert "blood_culture" in ids and "blood_culture_ie" not in ids
    ie = next(c for c in pr.must_checks_for(chronic, "인공판막") if c.id == "blood_culture_ie")
    assert ie.citation.pmid == "37622656" and ie.verification == "primary"  # ESC 2023 sec. 5.3.1 read (2026-09-29)
    assert "30분 간격 3세트" in ie.name


def test_acute_dizziness_posterior_circulation_screen():
    stroke = {"glucose", "onset_time", "neuro_exam", "brain_imaging"}
    assert stroke <= _checks("70세 여성. 주호소: 갑자기 시작된 어지럼증")  # age >= 60
    assert "neuro" in cr.detect_categories("65세 남성. 주호소: 오늘 아침부터 빙빙 도는 어지럼")
    young = "50세 남성. 주호소: 갑자기 시작된 어지럼증"
    assert not _checks(young) & stroke
    assert stroke <= _checks(young, "고혈압약을 먹고 있고 당뇨도 있어요.")  # vascular risk factor
    assert not _checks(young, "고혈압이나 당뇨는 없어요.") & stroke  # negated risk factors
    # BPPV-like (positional AND recurrent) and weeks-long dizziness are excluded
    assert not _checks("73세 여성. 주호소: 갑자기 생기는 어지럼증",
                       "자세를 바꿀 때마다 갑자기 빙빙 도는 느낌이 들어요.") & stroke
    assert not _checks("72세 남성. 주호소: 3주 전부터 계속되는 어지럼증") & stroke
    # young, no risk factors, allergy or cardiac context: unchanged
    assert "neuro" not in cr.detect_categories("가슴 두근거림과 어지럼증")
    assert "neuro" not in cr.detect_categories("75세 남성. 주호소: 운동 중 갑자기 발생한 전신 두드러기와 어지러움")


# --- 2026-09-26 categories: positive and negative examples -----------------------------------


def _cats(cc, context=""):
    return set(cr.detect_categories(cc, context))


def test_keyword_gaps_in_existing_categories():
    assert "dyspnea" in _cats("34세 여성. 주호소: 오늘 아침부터 숨이 참")
    assert "abdominal_pain" in _cats("34세 남성. 주호소: 오늘 새벽부터 시작된 배아픔과 설사")
    assert "chest_pain" in _cats("57세 남성. 주호소: 운동할 때 생기는 흉골 뒤쪽 통증")
    assert "fever" in _cats("29세 여성. 주호소: 이틀 전부터 시작된 열과 몸살")
    # "힘 빠짐" (no particle) is a one-sided motor deficit
    assert "neuro" in _cats("71세 남성. 주호소: 1시간 전 갑자기 생긴 오른쪽 팔다리 힘 빠짐")


def test_infant_age_is_not_a_duration():
    assert cr.duration_level("생후 9개월 여아. 주호소: 2일 전 시작된 발진") == 0
    assert cr.duration_level("생후 18개월 남아. 주호소: 3주 전부터 기침") == 1
    assert cr.age_days("생후 2주 여아") == 14 and cr.age_days("생후 5일 남아") == 5
    assert cr.age_days("신생아. 주호소: 황달") == 0 and cr.age_days("45세 남성") is None


def test_syncope_protocol():
    assert {"vitals", "ecg", "cardiac_history"} <= _checks("68세 남성. 주호소: 의식 소실")
    assert {"ecg", "cardiac_history"} <= _checks("40대 후반 여성. 주호소: 입원 중 발생한 실신")
    assert "syncope" not in _cats("46세 여성. 주호소: 오늘 생긴 의식변화")  # altered mental status is the stroke protocol
    assert "syncope" not in _cats("30세 남성. 주호소: 어지럼증")


def test_palpitations_protocol():
    assert {"vitals", "ecg"} <= _checks("23세 여성. 주호소: 30분 전 갑자기 시작된 두근거림")
    assert "tsh" not in _checks("23세 여성. 주호소: 30분 전 갑자기 시작된 두근거림")
    assert "tsh" in _checks("32세 여성. 주호소: 3개월 전부터 시작된 두근거림, 체중 감소, 열불내성")
    assert "tsh" not in _checks("32세 여성. 주호소: 두근거림", "체중 감소나 손 떨림은 없어요.")


def test_hemoptysis_and_chronic_cough():
    ids = _checks("58세 남성. 주호소: 3일 전부터 시작된 소량의 객혈")
    assert {"cxr", "chest_ct"} <= ids and "cxr_cough" not in ids
    assert "cxr_cough" in _checks("42세 남성. 주호소: 3주간 지속된 기침과 화농성 객담")
    # acute cough is not the protocol (no can't-miss hints, no checks)
    assert "hemoptysis_cough" not in _cats("28세 여성. 주호소: 5일 전부터 시작된 발열, 기침, 인후통")
    assert "hemoptysis_cough" not in _cats("40세 남성. 주호소: 기침")


def test_jaundice_adult_vs_neonate():
    adult = _checks("70세 남성. 주호소: 2주 전부터 피부가 노래지는 증상")
    assert {"liver_panel", "med_alcohol_history", "pt_inr", "abdominal_us"} <= adult
    assert not adult & {"neonatal_bilirubin", "neonatal_direct_bilirubin"}
    newborn = _checks("생후 4일 여아. 주호소: 어제부터 피부가 노래짐")
    assert newborn == {"neonatal_bilirubin"}
    assert {"neonatal_bilirubin", "neonatal_direct_bilirubin"} <= _checks("생후 2주 여아. 주호소: 눈과 피부의 노란 변색")
    assert "jaundice" not in _cats("28세 여성. 주호소: 노란 가래와 기침")  # yellow sputum is not jaundice


def test_joint_arthrocentesis_acute_hot_joint_only():
    assert "arthrocentesis" in _checks("32세 남성. 주호소: 2일 전부터 시작된 오른쪽 무릎 통증, 부종, 발적")
    assert "arthrocentesis" in _checks("45세 남성. 주호소: 갑자기 시작된 엄지발가락 통증", "빨갛게 붓고 뜨거워요.")
    assert "arthrocentesis" not in _checks("68세 남성. 주호소: 3개월 전부터 악화된 우측 무릎 통증")  # chronic
    assert "arthrocentesis" not in _checks("45세 남성. 주호소: 어제부터 무릎 통증", "붓거나 열감은 없어요.")
    assert "arthrocentesis" not in _checks("7세 남아. 주호소: 2일 전부터 무릎 통증과 부종")  # adult guideline
    assert "joint" not in _cats("25세 여성. 주호소: 사고 직후 발생한 양쪽 무릎 통증")  # trauma
    assert cant_miss_for("무릎 찰과상") == []


def test_back_pain_protocol():
    assert {"red_flags", "neuro_exam_legs"} <= _checks("26세 여성. 주호소: 몇 시간 전 발생한 다리로 방사되는 요통")
    assert "back_pain" not in _cats("55세 남성. 주호소: 가슴이 찢어지고 등으로 뻗치는 통증")
    assert not _checks("15세 남성. 주호소: 허리 통증") & {"red_flags", "neuro_exam_legs"}  # ACP guideline: adults


def test_rash_acute_adult_only():
    assert {"drug_history", "mucosal_exam"} <= _checks("47세 남성. 주호소: 등 상부의 발진")
    assert "rash" not in _cats("26세 여성. 주호소: 5개월째 계속되는 양팔의 가려운 발진")  # chronic
    assert "rash" not in _cats("34세 남성. 주호소: 우안 시력 저하 및 검은 반점")  # floaters, not skin
    assert "rash" not in _cats("45세 남성. 주호소: 전신 두드러기")  # the anaphylaxis protocol instead
    assert not _checks("생후 3일 여아. 주호소: 어제부터 온몸에 생긴 발진") & {"drug_history", "mucosal_exam"}


def test_chronic_generalized_pruritus():
    assert {"cbc", "liver_panel", "renal_function", "tsh"} <= _checks("62세 남성. 주호소: 3개월 전부터 시작된 전신 가려움증")
    assert "pruritus" not in _cats("47세 남성. 주호소: 며칠 전부터 시작된 우측 대퇴부의 심한 가려움증")  # local, acute
    assert "pruritus" not in _cats("생후 9개월 여아. 주호소: 2일 전 시작된 전신의 가렵지 않은 발진")  # negated, acute


def test_edema_unilateral_vs_bilateral():
    assert "urinalysis" in _checks("64세 남성. 주호소: 열흘 전부터 시작된 피곤함과 양쪽 다리 부종")
    assert "urinalysis" in _checks("8세 남아. 주호소: 얼굴 부종과 거품 소변")
    ids = _checks("55세 여성. 주호소: 3일 전부터 왼쪽 다리가 부었어요")
    assert "dvt_workup" in ids and "urinalysis" not in ids
    assert "edema" not in _cats("50세 남성. 주호소: 목 뒤쪽에 점점 커지는 부종")  # local mass
    assert "edema" not in _cats("22세 남성. 주호소: 오른손 약지의 통증과 부종")


def test_menstrual_protocol():
    assert {"pregnancy_test", "amenorrhea_labs"} <= _checks("35세 여성. 주호소: 2개월 전부터 시작된 무월경")
    pmb = _checks("63세 여성. 주호소: 두 달째 가끔 있는 폐경 후 질출혈")
    assert "pmb_evaluation" in pmb and "pregnancy_test" not in pmb
    assert "pregnancy_test" in _checks("25세 여성. 주호소: 어제부터 질출혈")
    assert "menstrual" not in _cats("40세 남성. 주호소: 복통")


def test_fatigue_needs_two_weeks():
    assert {"cbc", "tsh"} <= _checks("69세 남성. 주호소: 두 달 전부터 계속되는 전신 피로감")
    assert not _checks("30세 여성. 주호소: 3일 전부터 피곤해요") & {"cbc", "tsh"}


def test_cognitive_protocol():
    assert {"b12", "tsh", "brain_imaging", "depression_screen"} <= _checks("67세 남성. 주호소: 1년 전부터 시작된 기억력 저하")
    assert "cognitive" not in _cats("45세 여성. 주호소: 두통")


def test_psychiatric_protocol():
    assert {"suicide_risk", "substance_use"} <= _checks("27세 여성. 주호소: 4개월째 계속되는 우울감")
    assert not _checks("9세 남아. 주호소: 학교에서의 행동 문제") & {"suicide_risk", "substance_use"}  # adult guideline
    # organic altered mental status: the stroke/encephalopathy work-up, not the psychiatric one
    assert "psychiatric" not in _cats("42세 여성. 주호소: 2일 전부터 시작된 의식 변화와 이상행동")
    assert "psychiatric" not in _cats("31세 남성. 주호소: 1주일간 악화된 구역, 실조증")  # "실조증" is not mania
    assert "psychiatric" not in _cats("23세 남성. 주호소: 걸음걸이 불안정")


def test_new_protocol_citations_verified():
    new = {"syncope", "palpitations", "hemoptysis_cough", "jaundice", "joint", "back_pain", "rash", "pruritus",
           "edema", "menstrual", "fatigue", "cognitive", "psychiatric"}
    assert new <= set(pr.PROTOCOLS_BY_CATEGORY)
    for cat in new:
        for c in pr.PROTOCOLS_BY_CATEGORY[cat].checks:
            assert c.citation in pr.GUIDELINES and c.citation.verified and c.citation.pmid, (cat, c.id)
            assert c.note or c.verification == "primary", (cat, c.id)


# 2026-09-29 verification pass: (category, check id) -> level now recorded; each note says what was read
UPGRADED_2026_09_29 = {
    ("chest_pain", "aorta_imaging"): "primary", ("chest_pain", "pe_workup"): "primary",  # second pass: sec. 4.11
    ("dyspnea", "pe_workup"): "primary", ("dyspnea", "epinephrine"): "primary",
    ("allergy", "epinephrine"): "primary", ("allergy", "airway_breathing"): "primary",
    ("neuro", "glucose"): "secondary", ("neuro", "onset_time"): "secondary", ("neuro", "brain_imaging"): "secondary",
    ("fever", "cbc_neutropenia"): "primary", ("abdominal_pain", "pelvic_us"): "primary",
}


@pytest.mark.parametrize("key,level", sorted(UPGRADED_2026_09_29.items()))
def test_protocol_verification_pass_2026_09_29(key, level):
    cat, cid = key
    c = next(c for c in pr.PROTOCOLS_BY_CATEGORY[cat].checks if c.id == cid)
    assert c.verification == level
    assert "2026-09-29" in c.note
    assert c.citation in pr.GUIDELINES and c.citation.verified and c.citation.pmid
    if cid in ("epinephrine", "airway_breathing"):
        assert c.citation is pr.G_WAO_ANAPHYLAXIS  # the guideline whose text was read


def test_protocol_verification_counts():
    from collections import Counter
    n = Counter(c.verification for p in pr.PROTOCOLS for c in p.checks)
    assert n["unverified"] <= 8 and n["primary"] >= 61


# 2026-09-29 second pass: (category, check id) -> (level, citation the read text belongs to)
SECOND_PASS_2026_09_29 = {
    ("dyspnea", "vitals"): ("primary", pr.G_PE), ("dyspnea", "ecg"): ("secondary", pr.G_HF),
    ("dyspnea", "cxr"): ("secondary", pr.G_HF), ("dyspnea", "natriuretic_peptide"): ("secondary", pr.G_HF),
    ("headache", "neuro_exam"): ("primary", pr.G_HEADACHE), ("headache", "brain_ct"): ("primary", pr.G_HEADACHE),
    ("fever", "vitals"): ("primary", pr.G_SEPSIS), ("fever", "blood_culture_ie"): ("primary", pr.G_ENDOCARDITIS),
    ("abdominal_pain", "vitals"): ("primary", pr.G_AORTA),
    ("abdominal_pain", "pregnancy_test"): ("primary", pr.G_ACR_PELVIC_PAIN),
    ("abdominal_pain", "aaa_imaging"): ("primary", pr.G_AORTA),
    ("allergy", "vitals"): ("primary", pr.G_WAO_ANAPHYLAXIS),
    ("syncope", "vitals"): ("primary", pr.G_ESC_SYNCOPE), ("syncope", "ecg"): ("primary", pr.G_ESC_SYNCOPE),
    ("syncope", "cardiac_history"): ("primary", pr.G_ESC_SYNCOPE),
    ("palpitations", "vitals"): ("primary", pr.G_PALPITATIONS), ("palpitations", "tsh"): ("primary", pr.G_PALPITATIONS),
    ("hemoptysis_cough", "cxr_cough"): ("primary", pr.G_COUGH),
    ("jaundice", "abdominal_us"): ("primary", pr.G_JAUNDICE_IMAGING),
    ("edema", "urinalysis"): ("primary", pr.G_GLOMERULAR),
    ("bilious_vomiting", "urgent_abd_imaging"): ("primary", pr.G_INFANT_VOMITING),
}


@pytest.mark.parametrize("key,expected", sorted(SECOND_PASS_2026_09_29.items(), key=lambda kv: kv[0]))
def test_protocol_verification_second_pass_2026_09_29(key, expected):
    level, citation = expected
    cat, cid = key
    c = next(c for c in pr.PROTOCOLS_BY_CATEGORY[cat].checks if c.id == cid)
    assert (c.verification, c.citation) == (level, citation)
    assert "2026-09-29" in c.note and c.citation in pr.GUIDELINES and c.citation.verified and c.citation.pmid
    assert not c.note.startswith("Tension pneumothorax")  # the old claim is not in BTS 2023 (only quoted as removed)


def test_retrocochlear_imaging_second_pass():
    c = next(c for c in pr.PROTOCOLS_BY_CATEGORY["hearing_loss"].checks
             if c.id == "retrocochlear_imaging" and c.citation is pr.G_HEARING_IMAGING)
    assert c.verification == "primary" and "조영증강" not in c.name  # ACR: with or without contrast


def test_still_unverified_checks_say_why():
    left = {(p.category, c.id) for p in pr.PROTOCOLS for c in p.checks if c.verification == "unverified"}
    assert left == {("chest_pain", "vitals"), ("chest_pain", "cxr"), ("headache", "vitals"), ("neuro", "vitals"),
                    ("neuro", "neuro_exam"), ("abdominal_pain", "abdominal_exam"), ("chronic_weakness", "neuro_exam"),
                    ("chronic_weakness", "ck")}


def test_aaa_imaging_age_floor_and_aortopathy_exception():
    """min_age=50 kept (Howard 2015: no acute AAA below 55); below 50 only with an aortopathy clue."""
    back = "허리 통증도 있어요."
    assert "aaa_imaging" in _checks("55세 남성. 주호소: 갑자기 시작된 복통", back)
    assert "aaa_imaging" in _checks("주호소: 갑자기 시작된 복통", back)  # unstated age
    assert "aaa_imaging" not in _checks("35세 남성. 주호소: 어제부터 시작된 복통", back)
    assert "aaa_imaging" in _checks("35세 남성. 주호소: 어제부터 시작된 복통", back + " 마르판 증후군이 있어요.")
    assert "aaa_imaging" in _checks("42세 여성. 주호소: 오늘 시작된 복통", "옆구리가 아프고 엘러스-단로스 증후군 진단을 받았어요.")
    assert "aaa_imaging" not in _checks("35세 남성. 주호소: 어제부터 시작된 복통", back + " 대동맥류는 없다고 들었어요.")
    assert "aaa_imaging" not in _checks("35세 남성. 주호소: 어제부터 시작된 복통", "마르판 증후군이 있어요.")  # no AAA clue
    c = next(c for c in pr.PROTOCOLS_BY_CATEGORY["abdominal_pain"].checks if c.id == "aaa_imaging")
    assert c.min_age is None and c.predicate == "aaa" and "Howard 2015" in c.note


# --- 2026-09-27: false-trigger fixes from the per-case audit of data/cases_aug ---------------------------


def test_periarticular_source_is_not_a_hot_joint():
    # cqa_236 de Quervain: wrist swelling/redness, but Finkelstein positive and passive joint motion preserved
    cc = "35세 여성. 주호소: 2일 전부터 시작된 좌측 손목 통증과 부종"
    assert "arthrocentesis" not in _checks(cc, "손목에 열감이랑 발적이 있어요. 손목 관절의 수동적 움직임에는 제한이 "
                                               "없음. Finkelstein 검사 양성")
    assert "arthrocentesis" not in _checks("40세 남성. 주호소: 어제부터 팔꿈치 부종", "주두 점액낭염 소견")
    # a negative Finkelstein test does not exclude the joint; DGI-like hot wrist keeps aspiration
    assert "arthrocentesis" in _checks(cc, "손목이 붓고 뜨거워요. Finkelstein 검사 음성")
    assert "arthrocentesis" in _checks("18세 남성. 주호소: 사흘 전부터 왼쪽 손목 통증",
                                       "왼쪽 손목이 붓고 붉으며, 만지거나 움직이면 심한 압통이 있음")


def test_edema_as_part_of_acute_illness_is_not_primary_edema():
    # leg swelling listed with 4 days of cough, chills, dyspnea (pneumonia/bacteremia, known cardiomyopathy)
    cc = "58세 여성. 주호소: 나흘 전부터 기침, 오한, 숨참, 다리 부종"
    assert "edema" not in _cats(cc) and "urinalysis" not in _checks(cc)
    assert "natriuretic_peptide" in _checks(cc)  # the heart-failure angle stays with the dyspnea protocol
    # primary edema, foamy urine or facial edema keep the glomerular work-up even with dyspnea
    assert "urinalysis" in _checks("45세 남성. 주호소: 3주 전부터 시작된 몸의 부종")
    assert "urinalysis" in _checks("60세 여성. 주호소: 호흡곤란과 거품 소변, 다리 부종")
    assert "urinalysis" in _checks("64세 남성. 주호소: 열흘 전부터 시작된 피곤함과 양쪽 다리 부종")


def test_chronic_urticaria_is_not_chronic_pruritus():
    # cqa_503: 6 months of itchy wheals -> EAACI limited work-up (CBC with differential + CRP/ESR), not the CPUO panel
    cc = "40세 여성. 주호소: 6개월 전부터 발생한 전신 가려움과 두드러기"
    assert _cats(cc) == {"urticaria_chronic"}
    ids = _checks(cc)
    assert ids == {"cbc_diff", "crp_esr"} and not ids & {"liver_panel", "renal_function", "tsh"}
    assert "urticaria_chronic" not in _cats("30세 남성. 주호소: 3주 전부터 반복되는 두드러기")  # < 6 weeks
    assert "urticaria_chronic" not in _cats("45세 남성. 주호소: 오늘 갑자기 생긴 전신 두드러기")  # acute: anaphylaxis
    assert "pruritus" in _cats("62세 남성. 주호소: 3개월 전부터 시작된 전신 가려움증")  # CPUO unchanged
    assert cr.duration_weeks("6개월 전부터") > 6 and cr.duration_weeks("3주 전부터") == 3
    assert cr.duration_weeks("1개월 전부터") < 6 and cr.duration_weeks("오늘 갑자기") == 0


def test_abdominal_pregnancy_and_aaa_populations():
    # known pregnancy (38 weeks) -> no hCG test, but pelvic ultrasound applies
    ids = _checks("30세 여성. 주호소: 한 시간 전부터 질출혈과 심한 복통 (임신 38주)", "첫 임신 때 아기 심박이 불안정했다고 들었어요")
    assert "pregnancy_test" not in ids and "pelvic_us" in ids and "aaa_imaging" not in ids
    # cqa_285: women over 55 are outside the early-pregnancy population
    assert "pregnancy_test" not in _checks("60세 여성. 주호소: 하루 전부터 시작된 심한 배 통증")
    assert "pregnancy_test" in _checks("27세 여성. 주호소: 오늘 아침부터 시작된 아랫배 통증")
    # obstetric history ("임신은 2번 했고") is not a current pregnancy: no pelvic US by itself
    ids = _checks("35세 여성. 주호소: 2일 전부터 시작된 우측 옆구리 통증과 배뇨통, 고열", "임신은 2번 했고 출산도 2번 했어요.")
    assert "pelvic_us" not in ids and "pregnancy_test" in ids and "aaa_imaging" not in ids  # 35 < 50
    assert "pelvic_us" in _checks("27세 여성. 주호소: 아랫배 통증", "생리가 늦게 시작하나 보다 했어요.")
    assert "pelvic_us" not in _checks("생후 8개월 남아. 주호소: 복통", "임신 39주에 자연분만으로 태어났어요.")
    # AAA imaging needs age >= 50 (or unstated age)
    assert "aaa_imaging" not in _checks("20세 남성. 주호소: 어제부터 시작된 복통과 구토", "등 통증도 있어요.")
    assert "aaa_imaging" in _checks("72세 남성. 주호소: 갑자기 시작된 복통", "허리 통증이 있고 실신했어요.")


def test_abdominal_location_word_needs_pain():
    assert "abdominal_pain" not in _cats("41세 여성. 주호소: 왼쪽 아랫배 피부 밑의 덩이")
    assert "abdominal_pain" in _cats("53세 여성. 주호소: 어젯밤부터 계속되는 윗배 통증")
    assert "abdominal_pain" in _cats("42세 여성. 주호소: 3일 전부터 시작된 명치부 통증과 가슴 쓰림")


def test_negation_and_trigger_word_fixes():
    # "경부 강직 음성" / "경부 강직(-)" are pertinent negatives (cqa_388)
    cc = "66세 남성. 주호소: 6개월째 계속되는 기침과 발열"
    assert "meningitis_workup" not in _checks(cc, "경부 강직(-), 커닉 징후(-)")
    assert "meningitis_workup" not in _checks(cc, "경정맥 확장 없으며 경부 강직 음성.")
    assert "meningitis_workup" in _checks(cc, "목이 뻣뻣하고 머리가 아파요.")
    # a long negated list: "심잡음이나 마찰음(friction rub)은 들리지 않음"
    ie = "45세 남성. 주호소: 3주 전부터 계속되는 미열"
    assert "blood_culture_ie" not in _checks(ie, "심잡음이나 마찰음(friction rub)은 들리지 않음.")
    # "자세떨림" (tremor) is not a rigor
    assert not _checks("33세 남성. 주호소: 일주일 전부터 열과 두통", "양팔을 뻗으면 자세떨림(postural tremor)이 보임") & {
        "blood_culture", "lactate"}
    # a drug-allergy history is not an allergic reaction (no epinephrine prompt)
    assert "epinephrine" not in {c.id for c in pr.must_checks_for("27세 여성. 주호소: 갑작스러운 호흡곤란",
                                                                  "페니실린 알레르기가 있습니다.")}
    assert "epinephrine" in {c.id for c in pr.must_checks_for("27세 여성. 주호소: 갑작스러운 호흡곤란",
                                                              "새우를 먹고 나서 입술이 붓고 두드러기가 났어요.")}
    # weeks-long febrile illness: a single borderline vital sign does not make it suspected sepsis (cqa_388)
    chronic = "38세 남성. 주호소: 1개월 전부터 시작된 기침, 호흡곤란, 미열"
    assert not _checks(chronic, "혈압 120/70 mmHg, 맥박 92회/분, 호흡수 22회/분") & {"blood_culture", "lactate"}
    assert {"blood_culture", "lactate"} <= _checks("22세 여성. 주호소: 2일 전부터 발열", "맥박 110회/분")  # acute: one sign


def test_adult_only_dyspnea_checks_not_for_newborns():
    ids = _checks("신생아. 주호소: 태어난 지 3시간 만에 생긴 호흡곤란")
    assert "cxr" in ids and not ids & {"ecg", "natriuretic_peptide", "pe_workup"}
    assert {"ecg", "natriuretic_peptide"} <= _checks("76세 여성. 주호소: 움직이면 숨이 많이 참")


def test_transient_altered_consciousness_is_syncope():
    # intermittent lowering of consciousness with falls (e.g. paroxysmal AV block) -> ECG, not stroke protocol
    cc = "80대 후반 환자. 주호소: 반복되는 일시적 의식저하와 낙상"
    assert "syncope" in _cats(cc) and "neuro" not in _cats(cc)
    assert "ecg" in _checks(cc)
    assert "neuro" in _cats("46세 여성. 주호소: 오늘 생긴 의식변화")
    assert "neuro" in _cats("22세 여성. 주호소: 반복되는 전신 경련")  # seizure stays neuro


def test_primary_amenorrhea_wording():
    assert {"pregnancy_test", "amenorrhea_labs"} <= _checks("17세 여성. 주호소: 초경이 없음")
    assert "amenorrhea_labs" in _checks("16세 여성. 주호소: 초경 지연")


def test_keyword_gaps_2026_09_27():
    assert "fever" in _cats("20세 남성. 주호소: 어제부터 시작된 열과 두통")
    assert "neuro" in _cats("46세 남성. 주호소: 일주일 전부터 오른쪽 다리 힘이 약해지고 걸음이 불안정함")


# --- 2026-09-27 categories: positive and negative examples ---------------------------------------------


def test_hearing_loss_protocol():
    # cqa_404: gradual unilateral hearing loss -> otoscopy, audiometry, retrocochlear MRI
    assert _checks("45세 남성. 주호소: 3개월 전부터 서서히 진행되는 우측 귀 난청") == {
        "ear_exam", "audiometry", "retrocochlear_imaging"}
    bilateral = _checks("58세 남성. 주호소: 2년 전부터 시작된 양측성 점진적 난청")
    assert {"ear_exam", "audiometry"} <= bilateral and "retrocochlear_imaging" not in bilateral
    sudden = pr.must_checks_for("50세 여성. 주호소: 오늘 아침 갑자기 왼쪽 귀가 잘 안 들려요")
    by_id = {c.id: c for c in sudden}
    assert {"ear_exam", "audiometry", "retrocochlear_imaging"} <= set(by_id)
    assert all(by_id[i].citation.pmid == "31369359" for i in ("ear_exam", "audiometry", "retrocochlear_imaging"))
    assert not _checks("5세 남아. 주호소: 양측 난청")  # adult guidelines
    assert "hearing_loss" not in _cats("30세 여성. 주호소: 귀 통증")


def test_neck_mass_protocol():
    assert {"malignancy_history", "neck_imaging"} <= _checks("22세 남성. 주호소: 좌측 경부의 무통성 종괴")
    assert not _checks("17세 남성. 주호소: 1개월 전부터 시작된 좌측 경부의 무통성 종괴")  # guideline: adults only
    # short, infectious story: not yet at increased risk, history still applies
    ids = _checks("30세 여성. 주호소: 3일 전부터 목에 멍울", "감기 기운과 인후통이 있어요.")
    assert "malignancy_history" in ids and "neck_imaging" not in ids
    assert "neck_mass" not in _cats("35세 여성. 주호소: 손목에 혹")  # wrist ganglion
    assert "neck_mass" not in _cats("50세 남성. 주호소: 목 뒤쪽에 점점 커지는 부종")


def test_bleeding_tendency_protocol():
    for cc in ("4세 남아. 주호소: 빈번한 코피", "19세 남성. 주호소: 사랑니를 뺀 뒤 멈추지 않는 잇몸 출혈",
               "55세 남성. 주호소: 치과 치료 후 지혈되지 않는 출혈", "34세 여성. 주호소: 멍이 쉽게 드는 증상"):
        assert {"platelet_count", "pt_aptt"} <= _checks(cc), cc
    assert "bleeding" not in _cats("성인 남성. 주호소: 한 달 전 다친 뒤 코를 비비면 나는 가벼운 코피")  # traumatic
    assert "bleeding" not in _cats("43세 여성. 주호소: 2주 전부터 대변 볼 때마다 나는 직장 출혈")


def test_chronic_weakness_protocol():
    assert {"neuro_exam", "ck"} <= _checks("48세 남성. 주호소: 3개월 전부터 서서히 진행된 근력 저하")
    ids = _checks("40대 남성. 주호소: 4개월 전부터 점점 심해지는 다리 감각이상과 다리 근력약화")
    assert "neuro_exam" in ids and "ck" not in ids  # sensory symptoms: not a myopathy pattern
    assert "ck" not in _checks("57세 남성. 주호소: 몇 주 전부터 오른팔 힘이 약해짐")  # one-sided
    assert "chronic_weakness" not in _cats("48세 남성. 주호소: 2일 전부터 양쪽 다리 힘이 빠짐")  # acute (e.g. GBS)
    assert "chronic_weakness" not in _cats("33세 여성. 주호소: 2개월 전부터 피로감 및 무력감")  # fatigue
    assert "chronic_weakness" not in _cats("20세 여성. 주호소: 2개월 전부터 시력 저하")


def test_infant_bilious_vomiting_protocol():
    for cc in ("생후 2일 남아. 주호소: 담즙성 구토", "생후 6시간 남아. 주호소: 담즙성 구토",
               "생후 3주 여아. 주호소: 어제부터 초록색 구토"):
        assert _checks(cc) == {"urgent_abd_imaging"}, cc
    assert "bilious_vomiting" not in _cats("45세 남성. 주호소: 담즙성 구토")  # ACR population: infants <= 3 months
    assert "bilious_vomiting" not in _cats("생후 2일 남아. 주호소: 수유 후 구토")  # not bilious


def test_2026_09_27_protocol_citations_verified():
    new = {"urticaria_chronic", "hearing_loss", "neck_mass", "bleeding", "chronic_weakness", "bilious_vomiting"}
    assert new <= set(pr.PROTOCOLS_BY_CATEGORY) and new <= set(cr.CATEGORY_NAMES)
    for cat in new:
        for c in pr.PROTOCOLS_BY_CATEGORY[cat].checks:
            assert c.citation in pr.GUIDELINES and c.citation.verified and c.citation.pmid, (cat, c.id)
            assert c.note or c.verification == "primary", (cat, c.id)
    # classification criteria are not a diagnostic guideline: the weakness checks stay "unverified"
    assert {c.verification for c in pr.PROTOCOLS_BY_CATEGORY["chronic_weakness"].checks} == {"unverified"}


# --- 2026-09-27: 12 more decision rules --------------------------------------------------------

NEW_RULES_2026_09_27 = ("pecarn_head_lt2", "pecarn_head_ge2", "nexus", "ccsr", "sfsr", "csrs", "gbs", "kocher", "pas",
                        "spesi", "pecarn_febrile_infant", "mcisaac")


def _all_false(rule_id):
    return {i.key: (0 if i.options else False) for i in cr.RULES_BY_ID[rule_id].items}


def test_new_rules_registered_with_verified_citations():
    assert len(cr.RULES) == 24 and len(cr.RULES_BY_ID) == 24
    pmids = {"pecarn_head_lt2": "19758692", "pecarn_head_ge2": "19758692", "nexus": "10891516", "ccsr": "11597285",
             "sfsr": "14747812", "csrs": "27378464", "gbs": "11073021", "kocher": "10608376", "pas": "12037754",
             "spesi": "20696966", "pecarn_febrile_infant": "30776077", "mcisaac": "9475915"}
    levels = {}
    for rid in NEW_RULES_2026_09_27:
        r = cr.RULES_BY_ID[rid]
        _check_citation(r.citation)
        assert r.citation.pmid == pmids[rid]
        assert r.applicability and r.note
        levels[rid] = r.verification
    assert levels == {"pecarn_head_lt2": "secondary", "pecarn_head_ge2": "secondary", "nexus": "primary",
                      "ccsr": "primary", "sfsr": "primary", "csrs": "primary", "gbs": "secondary",
                      "kocher": "primary", "pas": "primary", "spesi": "secondary", "pecarn_febrile_infant": "primary",
                      "mcisaac": "secondary"}
    assert cr.RULES_BY_ID["pecarn_febrile_infant"].citation.short == "Kuppermann 2019"


def test_pecarn_head_tiers():
    for rid in ("pecarn_head_lt2", "pecarn_head_ge2"):
        assert cr.score(rid, _all_false(rid)).label == "매우 저위험"
        assert cr.score(rid, {"severe_mechanism": True}).label == "중간 위험"
        partial = cr.score(rid, {"ams_gcs": False})
        assert "배제 불가" in partial.label and not partial.complete  # very-low-risk needs every item
    assert cr.score("pecarn_head_lt2", {"palpable_fx": True, "loc_ge5s": True}).label == "고위험"
    assert cr.score("pecarn_head_ge2", {"basilar_fx_signs": True}).label == "고위험"
    assert cr.score("pecarn_head_ge2", {"vomiting": True, "severe_headache": True}).score == 1  # tier, not count


def test_nexus_and_canadian_c_spine():
    assert cr.score("nexus", _all_false("nexus")).label == "저위험"
    assert cr.score("nexus", {"midline_tenderness": True, "intoxication": True}).label == "배제 불가"
    assert cr.score("ccsr", _all_false("ccsr")).label == "영상 불필요"
    assert cr.score("ccsr", {"age_ge_65": True}).label == "영상 필요"


def test_syncope_rules():
    assert cr.score("sfsr", _all_false("sfsr")).label == "저위험"
    assert cr.score("sfsr", {"hct_lt_30": True}).label == "고위험"
    # CSRS: negative points, graded ED diagnosis, range -3..11
    low = cr.score("csrs", {"vasovagal_predisposition": True, "ed_diagnosis": "미주신경성 실신"})
    assert low.score == -3 and low.label == "매우 저위험"
    assert cr.score("csrs", {"ed_diagnosis": -2}).label == "매우 저위험"
    assert cr.score("csrs", {"heart_disease": True}).label == "중간 위험"
    top = {"heart_disease": True, "sbp_abnormal": True, "troponin_high": True, "qrs_axis": True, "qrs_gt_130": True,
           "qtc_gt_480": True, "ed_diagnosis": "심장성 실신"}
    assert cr.score("csrs", top).score == 11 and cr.score("csrs", top).label == "매우 고위험"
    assert cr.RULES_BY_ID["csrs"].max_score == 11  # the -1 item does not raise the maximum
    assert cr.score("csrs", {"troponin_high": True, "qtc_gt_480": True}).label == "고위험"
    for s in range(-3, 12):
        assert any(t.contains(s) for t in cr.RULES_BY_ID["csrs"].thresholds), s


def test_glasgow_blatchford():
    assert cr.score("gbs", _all_false("gbs")).label == "저위험"
    r = cr.score("gbs", {"urea": "10–24.9(28–70)", "hemoglobin": 6, "sbp": 3, "pulse_ge_100": True, "melena": True})
    assert r.score == 15 and r.label == "치료 필요 가능"
    assert cr.RULES_BY_ID["gbs"].max_score == 23
    assert "배제 불가" in cr.score("gbs", {"melena": False}).label
    with pytest.raises(ValueError):
        cr.score("gbs", {"urea": 5})  # not an option value


def test_kocher_pas_mcisaac_spesi_febrile_infant():
    assert cr.score("kocher", {"fever_history": True, "non_weight_bearing": True}).meaning.startswith(
        "화농성 관절염 확률 40%")
    assert cr.score("kocher", {k: True for k in ("fever_history", "non_weight_bearing", "esr_ge_40",
                                                 "wbc_gt_12000")}).label == "매우 높음"
    assert cr.score("pas", {"anorexia": True, "fever": True}).label == "가능성 낮음"
    high = {"cough_hop_tenderness": True, "rlq_tenderness": True, "anorexia": True, "fever": True, "migration": True}
    assert cr.score("pas", high).score == 7 and cr.score("pas", high).label == "가능성 높음"
    assert cr.score("pas", {"cough_hop_tenderness": True, "rlq_tenderness": True, "anorexia": True,
                            "fever": True}).label == "불확실"  # 6
    assert cr.score("pas", {"rlq_tenderness": True, "nausea_vomiting": True}).label == "불확실"
    assert cr.RULES_BY_ID["pas"].max_score == 10
    assert cr.score("mcisaac", {"no_cough": True, "age": "45세 이상"}).label == "매우 낮음"
    kid = cr.score("mcisaac", {"fever_gt_38": True, "no_cough": True, "tender_anterior_nodes": True,
                               "tonsil_swelling_exudate": True, "age": "3–14세"})
    assert kid.score == 5 and kid.label == "매우 높음"
    assert cr.score("spesi", _all_false("spesi")).label == "저위험"
    assert cr.score("spesi", {"hr_ge_110": True}).meaning.startswith("30일 사망률 10.9%")
    assert cr.score("pecarn_febrile_infant", _all_false("pecarn_febrile_infant")).label == "저위험"
    assert cr.score("pecarn_febrile_infant", {"pct_gt_1_71": True}).label == "저위험 아님"


def test_rule_age_parsing():
    assert cr.rule_age_years("18개월 남아가 소파에서 떨어짐") == 1.5
    assert cr.rule_age_years("3살 아이") == 3
    assert cr.rule_age_years("45세 남성") == 45
    assert cr.rule_age_years("생후 6주 영아") == pytest.approx(42 / 365.25)
    assert cr.rule_age_years("3개월 전부터 두통") is None  # a duration, not an age
    # safety-side parsing is unchanged
    assert cr.age_years("3살 아이") is None and cr.duration_level("18개월 남아") == 2


def test_pecarn_head_applicability():
    assert _ids("18개월 남아가 소파에서 떨어져 머리를 부딪혔어요") == {"pecarn_head_lt2"}
    assert _ids("생후 10개월 아기, 침대에서 떨어져 머리를 부딪힘") == {"pecarn_head_lt2"}
    assert _ids("7세 남아. 주호소: 놀이터에서 넘어지면서 머리를 부딪혔어요") == {"pecarn_head_ge2"}
    assert _ids("3살 아이가 머리를 부딪혔어요") == {"pecarn_head_ge2"}
    assert _ids("아이가 머리를 부딪혔어요") == {"pecarn_head_lt2", "pecarn_head_ge2"}  # age unknown: both variants
    # adults: Canadian CT Head only; children never get the adult rule
    assert _ids("45세 남성, 넘어지면서 머리를 부딪혔어요") == {"cchr"}
    assert _ids("넘어지면서 머리를 부딪혔어요") == {"cchr"}
    assert "cchr" not in _ids("5세 여아가 머리를 부딪혔어요")
    assert not _ids("3개월 전 머리를 부딪혔어요") & {"pecarn_head_lt2", "pecarn_head_ge2", "cchr"}
    assert not _ids("7세 남아. 주호소: 어제부터 두통") & {"pecarn_head_lt2", "pecarn_head_ge2"}  # no trauma


def test_c_spine_applicability():
    assert {"nexus", "ccsr"} <= _ids("35세 여성. 교통사고 후 뒷목이 아파요")
    child = _ids("10세 남아. 교통사고 후 뒷목이 아파요")
    assert "ccsr" not in child and "nexus" in child
    assert not _ids("목이 아파요") & {"nexus", "ccsr"}  # sore throat in Korean
    assert not _ids("넘어져서 손목이 아파요") & {"nexus", "ccsr"}
    assert not _ids("40세 남성. 2년 전부터 뒷목이 뻐근하고 아파요") & {"nexus", "ccsr"}  # chronic, no trauma


def test_syncope_rules_applicability():
    assert _ids("62세 남성. 주호소: 화장실에서 갑자기 실신") == {"sfsr", "csrs"}
    assert "csrs" not in _ids("12세 여아. 주호소: 조회 시간에 쓰러졌어요")  # CSRS is >= 16 y
    assert not _ids("22세 여성. 주호소: 갑작스러운 의식 소실 및 전신 경련") & {"sfsr", "csrs"}  # seizure
    assert not _ids("가슴이 아파요") & {"sfsr", "csrs"}


def test_gbs_applicability():
    assert "gbs" in _ids("60세 남성. 주호소: 오늘 아침 피를 토하고 검은 변을 봤어요")
    assert "gbs" in _ids("55세 남성. 주호소: 3일 전부터 짜장면 같은 변")
    assert "gbs" not in _ids("기침하다 피를 토했어요")  # hemoptysis, not hematemesis
    assert "gbs" not in _ids("40세 여성. 주호소: 선홍색 혈변")  # lower GI bleeding
    assert "gbs" not in _ids("복통")


def test_pediatric_rules_applicability():
    assert "kocher" in _ids("5세 남아. 주호소: 어제부터 열이 나고 오른쪽 다리를 절뚝거려요")
    assert "kocher" not in _ids("45세 남성. 고관절 통증")
    assert "kocher" not in _ids("6세 남아. 넘어진 뒤 다리를 절뚝거려요")  # trauma
    assert "kocher" in _ids("6세 남아. 넘어진 적은 없는데 다리를 절뚝거려요")  # negated trauma
    assert {"alvarado", "pas"} <= _ids("10세 남아, 오른쪽 아랫배가 아파요")
    assert "pas" in _ids("6세 여아. 주호소: 어제부터 시작된 복통과 구토")
    assert "pas" not in _ids("28세 남성. 주호소: 어제부터 시작된 복통")
    assert "pas" not in _ids("8세 남아. 주호소: 6개월간 반복되는 복통")


def test_febrile_infant_applicability():
    assert "pecarn_febrile_infant" in _ids("생후 5주 영아. 주호소: 발열")
    assert "pecarn_febrile_infant" in _ids("신생아. 주호소: 열이 나요")
    assert "pecarn_febrile_infant" not in _ids("생후 3개월 영아. 주호소: 발열")  # > 60 days
    assert "pecarn_febrile_infant" not in _ids("영아. 주호소: 발열")  # age in days unknown
    assert "pecarn_febrile_infant" not in _ids("생후 5주 영아. 주호소: 황달")  # no fever
    assert not _ids("생후 5주 영아. 주호소: 발열") & {"qsofa", "curb65"}  # adult rules


def test_sore_throat_and_pe_severity_applicability():
    assert "mcisaac" in _ids("8세 아이. 주호소: 인후통과 열") and "centor" not in _ids("8세 아이. 주호소: 인후통과 열")
    assert {"centor", "mcisaac"} <= _ids("30세 여성. 주호소: 인후통과 열")
    assert "mcisaac" not in _ids("2세 남아. 인후통")
    # confirmed PE: severity (sPESI) replaces the diagnostic Wells/PERC
    ids = _ids("60세 여성. 폐색전증 진단 후 숨이 차요")
    assert "spesi" in ids and not ids & {"wells_pe", "perc"}
    assert "spesi" not in _ids("숨이 차요") and "wells_pe" in _ids("숨이 차요")


def test_existing_chief_complaints_keep_their_first_two_rules():
    # new rules are appended, so the policy's top-2 for existing adult presentations is unchanged
    assert [r.id for r in cr.rules_for("55세 남성, 2시간 전부터 가슴이 아파요")][:2] == ["wells_pe", "perc"]
    assert [r.id for r in cr.rules_for("28세 남성. 주호소: 어제부터 시작된 복통")] == ["alvarado"]
    assert _ids("무릎에 멍이 들었어요") == set()


def test_render_new_rules_short_and_signed():
    for text in ("아이가 머리를 부딪혔어요", "62세 남성. 주호소: 화장실에서 갑자기 실신",
                 "35세 여성. 교통사고 후 뒷목이 아파요", "60세 남성. 피를 토하고 검은 변"):
        out = cr.render_for_prompt(cr.rules_for(text)[:2])
        assert out and len(out) < 1000, (text, len(out))
    csrs = cr.render_for_prompt([cr.RULES_BY_ID["csrs"]])
    assert "-1" in csrs and "+-" not in csrs
    assert "진단 도구 아님" in cr.render_for_prompt([cr.RULES_BY_ID["spesi"]])
    assert "Kuppermann 2009" in cr.render_for_prompt([cr.RULES_BY_ID["pecarn_head_lt2"]])


# --- negation / subject / hedge via the normalisation layer (2026-09-27 migration) ---------------------------
@pytest.mark.parametrize("text, kw, affirmed, denied", [
    ("열이 나는 것 같아요", "열이", True, False),  # hedged present still triggers
    ("열이 나는지는 잘 모르겠어요", "열이", True, False),  # uncertain: triggers, never counts as denied
    ("기침이나 가래, 열은 없어요", "기침", False, True),  # sentence-final negation covers the list
    ("복시나 안구 돌출, 시야 결손은 보이지 않음", "복시", False, True),
    ("임신 32주, 오늘 아침부터 통증 없는 질 출혈", "임신", True, False),  # adnominal 없는 is not a list negation
    ("동성빈맥 (심박수 108회/분), 정상 축, st-t 변화 없음", "빈맥", True, False),  # "정상 축" is not a list negation
    ("열이 안 떨어져요", "열이", True, False),  # persistence
    ("어머니가 유방암으로 돌아가셨어요", "암", False, False),  # relative: neither the patient's nor denied
    ("아버지가 와파린을 드세요", "와파린", False, False),  # relative, keyword without a lexicon finding
    ("가족력: 고혈압", "고혈압", False, False),
    ("가족력은 없고, 고혈압이 있어요", "고혈압", True, False),
    ("저는 와파린 먹고 아버지는 아스피린", "와파린", True, False),
    ("혈청 b-hcg 양성\nhbsag 음성", "hcg 양성", True, False),  # a line break ends the sentence
    ("대동맥 박리 가족력이 있어요", "대동맥 박리 가족력", True, False),  # keyword about relatives: legacy reading
    ("(남편) 오늘 아침부터 아내가 헛소리를 해요", "헛소리", True, False),  # proxy speaker: the wife is the patient
    ("정상 심음, 박동 규칙적이며 심잡음 없음", "박동", False, True),  # "regular beat" is not a pulsatile mass
    ("아버지가 부정맥으로 치료받으신 적이 있어요", "부정맥", False, False),
    ("경부 강직 미약하게 의심되나 뚜렷하지 않음", "경부 강직", True, False),  # hedged exam finding still counts
])
def test_keyword_polarity_via_nlp_layer(text, kw, affirmed, denied):
    assert cr.contains_affirmed(text, (kw,)) is affirmed
    assert cr.negated(text, kw) is denied


def test_read_text_parses_once_and_matches_str():
    rt = cr.ReadText("두통은 없고 열이 나요")
    assert rt == "두통은 없고 열이 나요" and isinstance(rt, str)
    assert cr.contains_affirmed(rt, ("열이",)) and not cr.contains_affirmed(rt, ("두통",))
    assert rt.parsed() is rt.parsed()


@pytest.mark.parametrize("cc, present, absent", [
    ("45세 남성. 주호소: 열은 없고 기침만 3일째", set(), {"fever"}),  # denied by the layer: not a fever complaint
    ("60세 남성. 주호소: 두통은 없고 어지러움", {"neuro"}, {"headache"}),
    ("50세 남성. 주호소: 심와부 통증", {"abdominal_pain"}, set()),  # layer concept, no keyword (cqa_550)
    ("37세 여성. 주호소: 걸을 때 비틀거림", {"neuro"}, set()),  # colloquial gait disturbance
    ("36세 남성. 주호소: 이틀 전부터 오른눈이 침침하고 검은 점이 보임", set(), {"neuro", "rash"}),  # not from the layer
    ("50세 남성. 주호소: 목 뒤쪽에 점점 커지는 부종", set(), {"edema"}),
    ("55세 남성. 주호소: 치과 치료 후 지혈되지 않는 출혈", {"bleeding"}, set()),  # "지혈되지 않" is the complaint
])
def test_detect_categories_with_layer(cc, present, absent):
    cats = set(cr.detect_categories(cc))
    assert present <= cats and not (absent & cats), cats
