"""Pre-test safety/precondition checker (safety/preconditions.py)."""

import pytest

from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.safety import preconditions as P
from doctor_agent.safety.preconditions import check, check_all
from perf import assert_fast


def st(initial: str, *turns: tuple[str, str, str]) -> CaseState:
    s = CaseState(initial_info=initial)
    for typ, content, resp in turns:
        s.turns.append(Turn(Action(ActionType(typ), content), resp))
    return s


MENINGITIS = "34세 남성. 주호소: 어제부터 시작된 고열과 두통"


# --- rule table integrity ------------------------------------------------------------------------------------
def test_rule_table_integrity():
    ids = [r.id for r in P.RULES]
    assert len(ids) == len(set(ids)) >= 10
    for r in P.RULES:
        assert r.verification in ("primary", "secondary")
        assert r.hazard and r.precondition and r.note
        assert r.citation.short
        assert set(r.action_types) <= {"TEST", "EXAM"}
    for c in P.CITATIONS:
        if c.verified:
            assert c.pmid, c.title
        else:
            assert c.short_author in P.SOURCE_URLS, c.title


def test_ok_shape_and_non_matching_action():
    r = check("TEST", "일반혈액검사(CBC)", st(MENINGITIS))
    assert r == {"ok": True, "severity": None, "rule": None, "prerequisite": None, "why": "", "citation": ""}
    assert check("ASK", "요추천자 받아본 적 있나요?", st(MENINGITIS, ("ASK", "x", "의식이 흐려요")))["ok"]


def test_accepts_enum_action_type_and_none_state():
    assert check(ActionType.TEST, "요추천자", None)["ok"]
    r = check(ActionType.TEST, "요추천자", st(MENINGITIS, ("EXAM", "신경학적 검사", "좌측 편마비")))
    assert r["severity"] == "block"


# --- 1. LP: CT first -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("finding", [
    "우측 편마비와 구음장애가 있음",
    "안저 검사에서 양측 유두부종",
    "GCS 12, 지남력 저하",
    "오늘 아침 전신 경련이 한 번 있었어요",
    "신장 이식 후 면역억제제를 복용 중이에요",
    "3년 전에 뇌경색을 앓았어요",
])
def test_lp_ct_first_blocks_on_each_idsa_criterion(finding):
    r = check("TEST", "요추천자(뇌척수액 검사)", st(MENINGITIS, ("ASK", "과거력", finding)))
    assert r["ok"] is False and r["severity"] == "block" and r["rule"] == "lp_ct_first"
    assert r["prerequisite"] == ("TEST", "뇌 CT(비조영)")
    assert "항생제" in r["why"] and r["citation"] == "IDSA 세균성 수막염 지침 2004"


def test_lp_negated_findings_do_not_block():
    s = st(MENINGITIS,
           ("EXAM", "신경학적 검사", "의식은 명료하고 지남력 정상. 국소 신경학적 결손 없음. 유두부종 없음."),
           ("ASK", "경련", "경련은 없었어요."),
           ("ASK", "과거력", "특별히 앓던 병은 없어요. 아버지가 뇌졸중이 있었어요."))
    assert check("TEST", "요추천자", s)["ok"]
    assert check("TEST", "lumbar puncture", s)["severity"] is None


def test_lp_ct_already_done_or_requested():
    base = ("EXAM", "신경학적 검사", "좌측 상하지 근력 저하(편측 위약)")
    assert check("TEST", "요추천자", st(MENINGITIS, base, ("TEST", "뇌 CT", "종괴나 뇌부종 없음")))["ok"]
    # anti-loop: CT was requested but not available → no block
    assert check("TEST", "요추천자", st(MENINGITIS, base, ("TEST", "두부 CT", "해당 검사 결과는 제공되지 않습니다.")))["ok"]


def test_lp_seizure_false_friends():
    s = st(MENINGITIS, ("ASK", "동반 증상", "위경련처럼 배가 아팠고 발작성 기침이 있었어요"))
    assert check("TEST", "요추천자", s)["ok"]


def test_lp_child_seizure_warns_but_focal_sign_blocks():
    kid = "3세 남아. 주호소: 어제부터 시작된 발열"
    r = check("TEST", "요추천자", st(kid, ("ASK", "경련", "열이 나면서 1분 정도 전신 경련을 했어요")))
    assert r["ok"] and r["severity"] == "warn"
    assert check("TEST", "요추천자", st(kid, ("EXAM", "신경", "불러도 반응이 없음")))["severity"] == "block"


@pytest.mark.parametrize("text", [
    "발작이나 두통, 예전에 운동장애를 겪은 적은 없습니다.",  # comma list negated by the sentence-final verb
    "하루에도 몇 번씩 발작처럼 통증이 반복돼요.",  # simile
    "수술 후 서맥 발작이 와서 심폐소생술을 받았어요.",  # cardiac, not a seizure
    "어머니는 55세에 신장이식을 받으셨어요.",  # relative (honorific), not the patient
])
def test_lp_false_friends_do_not_block(text):
    assert check("TEST", "요추천자", st(MENINGITIS, ("ASK", "과거력", text)))["ok"]


def test_list_negation_does_not_swallow_an_affirmed_finding():
    s = st(MENINGITIS, ("EXAM", "신경학적 검사", "경련 있음, 유두부종 없음"))
    assert check("TEST", "요추천자", s)["severity"] == "block"


def test_lp_treatment_nonresponse_is_not_altered_consciousness():
    s = st(MENINGITIS, ("ASK", "과거력", "크론병 표준 치료에 반응이 없어서 약을 바꿨어요"))
    assert check("TEST", "요추천자", s)["ok"]


# --- 2. LP: bleeding risk -----------------------------------------------------------------------------------
def test_lp_anticoagulant_requires_coag_labs():
    s = st("72세 남성. 주호소: 발열과 두통", ("ASK", "약", "심방세동으로 와파린을 먹고 있어요"))
    r = check("TEST", "요추천자", s)
    assert r["severity"] == "block" and r["rule"] == "lp_coagulation"
    assert r["prerequisite"][0] == "TEST" and "INR" in r["prerequisite"][1]


def test_lp_anticoagulant_labs_known_normal():
    s = st("72세 남성. 주호소: 발열과 두통", ("ASK", "약", "와파린을 먹고 있어요"),
           ("TEST", "혈액검사, PT/INR", "혈소판 210,000/μL, INR 1.1"))
    r = check("TEST", "요추천자", s)
    assert r["ok"] and r["severity"] == "warn"  # remind about timing of the last dose only


def test_lp_low_platelets_or_high_inr_block_without_prerequisite():
    low = st(MENINGITIS, ("TEST", "CBC", "백혈구 2,100/μL, 혈소판 23,000/μL"))
    r = check("TEST", "요추천자", low)
    assert r["severity"] == "block" and r["prerequisite"] is None and "23,000" in r["why"]
    high = st(MENINGITIS, ("TEST", "응고검사", "PT INR 2.8"))
    assert check("TEST", "요추천자", high)["severity"] == "block"


def test_lp_aspirin_alone_and_denied_anticoagulant_ok():
    assert check("TEST", "요추천자", st(MENINGITIS, ("ASK", "약", "아스피린만 먹어요")))["ok"]
    assert check("TEST", "요추천자", st(MENINGITIS, ("ASK", "약", "항응고제는 안 먹어요")))["ok"]


# --- 3. Abdominal/pelvic ionising imaging: pregnancy -------------------------------------------------------
WOMAN = "26세 여성. 주호소: 어제부터 시작된 우하복부 통증"


def test_pregnancy_block_for_abdominal_ct_in_woman_of_childbearing_age():
    r = check("TEST", "복부 CT", st(WOMAN))
    assert r["severity"] == "block" and r["rule"] == "pregnancy_ionising_abd_pelvis"
    assert r["prerequisite"] == ("TEST", "소변 임신 반응 검사(β-hCG)")
    assert check("TEST", "KUB 단순 촬영", st(WOMAN))["severity"] == "block"
    assert check("TEST", "CT", st(WOMAN))["severity"] == "block"  # bare CT in an abdominal-pain case


def test_pregnancy_not_required_for_chest_or_head_imaging():
    assert check("TEST", "흉부 X-ray", st(WOMAN))["ok"]
    assert check("TEST", "뇌 CT", st("30세 여성. 주호소: 갑자기 시작된 두통"))["ok"]


def test_pregnancy_rule_excludes_men_children_and_older_women():
    assert check("TEST", "복부 CT", st("26세 남성. 주호소: 우하복부 통증"))["ok"]
    assert check("TEST", "복부 CT", st("7세 여아. 주호소: 복통"))["ok"]
    assert check("TEST", "복부 CT", st("68세 여성. 주호소: 복통"))["ok"]


def test_pregnancy_already_settled():
    neg = st(WOMAN, ("TEST", "소변 β-hCG", "음성"))
    assert check("TEST", "복부 CT", neg)["ok"]
    denied = st(WOMAN, ("ASK", "임신 가능성", "임신 가능성은 없어요. 생리는 1주 전에 끝났어요."))
    assert check("TEST", "복부 골반 CT", denied)["ok"]
    hyst = st("45세 여성. 주호소: 옆구리 통증", ("ASK", "수술력", "5년 전에 자궁 절제술을 받았어요"))
    assert check("TEST", "신장 CT", hyst)["ok"]


def test_pregnancy_known_pregnant_and_emergency_warn():
    preg = st(WOMAN, ("ASK", "임신", "지금 임신 12주예요"))
    r = check("TEST", "복부 CT", preg)
    assert r["ok"] and r["severity"] == "warn" and "초음파" in r["why"]
    shock = st(WOMAN, ("EXAM", "활력징후", "혈압 78/40 mmHg, 맥박 132회/분"))
    r = check("TEST", "복부 CT", shock)
    assert r["ok"] and r["severity"] == "warn" and r["prerequisite"][1].startswith("소변 임신")


def test_pregnancy_trauma_is_emergency_warn():
    s = st("25세 여성. 주호소: 교통사고 직후 발생한 우측 사타구니 통증")
    r = check("TEST", "골반 X-ray", s)
    assert r["ok"] and r["severity"] == "warn"


def test_pregnancy_age_unknown_woman_warns():
    r = check("TEST", "복부 CT", st("젊은 여성. 주호소: 복통"))
    assert r["ok"] and r["severity"] == "warn"


def test_pregnancy_block_downgraded_after_hcg_requested_but_unavailable():
    s = st(WOMAN, ("TEST", "소변 임신 반응 검사(β-hCG)", "해당 검사 결과는 제공되지 않습니다."))
    assert check("TEST", "복부 CT", s)["ok"]


# --- 4/5/6. Iodinated contrast --------------------------------------------------------------------------------
MAN_CKD = "67세 남성. 주호소: 갑자기 시작된 찢어지는 흉통"


def test_contrast_renal_warn_with_risk_factor_and_no_creatinine():
    s = st(MAN_CKD, ("ASK", "과거력", "당뇨가 있고 만성 콩팥병이 있어요. 알레르기는 없어요."))
    r = check("TEST", "흉부 CT 혈관조영술", s)
    assert r["ok"] and r["severity"] == "warn" and r["rule"] == "contrast_renal"
    assert r["prerequisite"] == ("TEST", "혈청 크레아티닌·eGFR")


def test_contrast_renal_satisfied_when_creatinine_known():
    s = st(MAN_CKD, ("ASK", "과거력", "당뇨가 있어요. 알레르기는 없어요."), ("TEST", "혈액검사", "크레아티닌 1.0 mg/dL"))
    assert check("TEST", "조영증강 흉부 CT", s)["ok"] and check("TEST", "조영증강 흉부 CT", s)["severity"] is None


def test_contrast_renal_no_risk_factor_and_noncontrast_ct():
    s = st("40세 남성. 주호소: 기침", ("ASK", "과거력", "특별한 병 없어요. 알레르기 없어요."))
    assert check("TEST", "조영 흉부 CT", s)["severity"] is None
    risky = st(MAN_CKD, ("ASK", "과거력", "투석 중이에요"))
    assert check("TEST", "비조영 흉부 CT", risky)["severity"] is None


def test_contrast_severe_renal_warns_but_never_blocks():
    s = st(MAN_CKD, ("TEST", "신기능", "eGFR 22 mL/min/1.73m²"), ("ASK", "알레르기", "없어요"))
    r = check("TEST", "CTA", s)
    assert r["ok"] and r["severity"] == "warn" and "보류하지 않습니다" in r["why"]


def test_contrast_prior_reaction_warns():
    s = st(MAN_CKD, ("ASK", "알레르기", "예전에 CT 찍을 때 조영제 알레르기로 두드러기가 났어요"),
           ("TEST", "혈액", "크레아티닌 0.9"))
    hits = check_all("TEST", "대동맥 CT 혈관조영", s)
    assert [h["rule"] for h in hits] == ["contrast_allergy"] and hits[0]["severity"] == "warn"
    # shellfish allergy is not a contrast risk factor, and allergy history is known → silent
    ok = st(MAN_CKD, ("ASK", "알레르기", "새우 알레르기가 있어요"), ("TEST", "혈액", "크레아티닌 0.9"))
    assert check("TEST", "대동맥 CT 혈관조영", ok)["ok"] and check("TEST", "대동맥 CT 혈관조영", ok)["severity"] is None


# --- 7/8. MRI / gadolinium ------------------------------------------------------------------------------------
def test_mri_orbital_metal_blocks_until_orbit_xray():
    s = st("45세 남성. 주호소: 허리 통증", ("ASK", "직업", "용접 일을 하는데 작년에 눈에 쇳가루가 튀어서 치료받았어요"))
    r = check("TEST", "요추 MRI", s)
    assert r["severity"] == "block" and r["rule"] == "mri_safety" and "안와" in r["prerequisite"][1]
    done = st("45세 남성. 주호소: 허리 통증", ("ASK", "직업", "눈에 쇳가루가 튀었어요"),
              ("TEST", "안와 X-ray", "금속 이물 없음"))
    assert check("TEST", "요추 MRI", done)["severity"] != "block"


def test_mri_orbit_simile_is_not_a_foreign_body():
    s = st("28세 남성. 주호소: 극심한 우측 안구 주변 통증",
           ("ASK", "양상", "뜨거운 쇠막대가 눈을 찌르는 것 같아요. 눈에 이물감이 있어요."), ("ASK", "금속", "없어요"))
    assert check("TEST", "뇌 MRI", s)["severity"] != "block"


def test_mri_pacemaker_warns_and_screening_prompt():
    s = st("80세 남성. 주호소: 어지럼", ("ASK", "과거력", "심박동기를 넣었어요"))
    r = check("TEST", "뇌 MRI", s)
    assert r["ok"] and r["severity"] == "warn" and r["prerequisite"][0] == "ASK"
    r = check("TEST", "뇌 MRI", st("50세 남성. 주호소: 두통"))
    assert r["ok"] and r["severity"] == "warn" and "금속" in r["prerequisite"][1]
    screened = st("50세 남성. 주호소: 두통", ("ASK", "몸에 금속이나 심박동기가 있나요?", "없어요"))
    assert check("TEST", "뇌 MRI", screened)["severity"] is None


def test_gadolinium_severe_renal_and_pregnancy():
    s = st("60세 남성. 주호소: 복통", ("ASK", "과거력", "혈액투석 중이에요. 몸에 금속은 없어요."))
    r = check("TEST", "조영 증강 복부 MRI", s)
    assert r["severity"] == "warn" and r["rule"] == "gadolinium"
    w = st("29세 여성. 주호소: 두통", ("ASK", "금속", "금속 없어요"))
    r = check("TEST", "가돌리늄 조영 뇌 MRI", w)
    assert r["severity"] == "warn" and "hCG" in r["prerequisite"][1]
    assert check("TEST", "비조영 뇌 MRI", w)["severity"] is None


# --- 9. Stress testing ---------------------------------------------------------------------------------------
CHEST = "58세 남성. 주호소: 2시간 전부터 시작된 가슴 통증"


def test_stress_test_blocked_until_ecg_and_troponin():
    r = check("TEST", "운동부하 검사(트레드밀)", st(CHEST))
    assert r["severity"] == "block" and r["prerequisite"] == ("TEST", "12유도 심전도")
    s = st(CHEST, ("TEST", "심전도", "정상 동성리듬, ST 변화 없음"))
    assert check("TEST", "트레드밀 검사", s)["prerequisite"] == ("TEST", "고감도 트로포닌")
    s2 = st(CHEST, ("TEST", "심전도", "정상 동성리듬, ST 변화 없음"),
            ("TEST", "고감도 트로포닌", "고감도 트로포닌 I 4 ng/L (참고치 <34)"))
    assert check("TEST", "트레드밀 검사", s2)["ok"]


def test_stress_test_contraindicated_with_positive_troponin_or_st_elevation():
    s = st(CHEST, ("TEST", "심전도", "V1-V4 ST분절 상승"))
    r = check("TEST", "운동부하검사", s)
    assert r["severity"] == "block" and r["prerequisite"] is None
    t = st(CHEST, ("TEST", "심전도", "비특이적"), ("TEST", "트로포닌", "고감도 트로포닌 I 1,240 ng/L (참고치 <34)"))
    assert check("TEST", "운동부하검사", t)["severity"] == "block"


def test_stress_test_chronic_chest_pain_not_blocked():
    assert check("TEST", "운동부하검사", st("55세 남성. 주호소: 6개월 전부터 계단 오를 때 가슴이 답답함"))["ok"]


# --- 10. Endoscopy -------------------------------------------------------------------------------------------
def test_endoscopy_blocked_with_free_air_or_peritonitis():
    s = st("70세 남성. 주호소: 갑자기 시작된 심한 복통", ("TEST", "흉부 X-ray", "횡격막 아래 공기(free air) 보임"))
    r = check("TEST", "위내시경", s)
    assert r["severity"] == "block" and r["prerequisite"] is None and r["rule"] == "endoscopy_perforation"
    p = st("70세 남성. 주호소: 갑자기 시작된 심한 복통", ("EXAM", "복부 진찰", "복벽 강직(판상 경직), 반발통"))
    r = check("TEST", "상부위장관 내시경", p)
    assert r["severity"] == "block" and r["prerequisite"][0] == "TEST"
    soft = st("50세 남성. 주호소: 명치 쓰림", ("EXAM", "복부 진찰", "복부 부드러움, 압통 없음, 천공 소견 없음"))
    assert check("TEST", "위내시경", soft)["ok"]


# --- 11. ABG --------------------------------------------------------------------------------------------------
def test_abg_anticoagulated_warns_only():
    s = st("75세 남성. 주호소: 호흡곤란", ("ASK", "약", "리바록사반을 먹고 있어요"))
    r = check("TEST", "동맥혈 가스 분석(ABG)", s)
    assert r["ok"] and r["severity"] == "warn" and r["rule"] == "abg_anticoagulation"
    assert check("TEST", "ABG", st("75세 남성. 주호소: 호흡곤란"))["severity"] is None


# --- 12. Digital vaginal examination -----------------------------------------------------------------------
def test_dve_blocked_in_antepartum_bleeding_until_ultrasound():
    s = st("31세 여성. 주호소: 임신 32주, 오늘 아침부터 통증 없는 질 출혈")
    r = check("EXAM", "내진(질 수지 검사)", s)
    assert r["severity"] == "block" and r["prerequisite"] == ("TEST", "산과 초음파(태반 위치 확인)")
    assert check("EXAM", "질경 검사", s)["ok"]  # speculum examination is allowed
    after = st("31세 여성. 주호소: 임신 32주, 오늘 아침부터 통증 없는 질 출혈",
               ("TEST", "산과 초음파", "태반은 자궁 저부 후벽에 위치, 전치태반 아님"))
    assert check("EXAM", "내진", after)["ok"]
    assert check("EXAM", "내진", st("31세 여성. 주호소: 임신 8주, 질 출혈"))["ok"]  # early pregnancy: not APH


# --- API behaviour -------------------------------------------------------------------------------------------
def test_block_sorted_before_warn_in_check_all():
    s = st(MENINGITIS, ("ASK", "약", "와파린 복용 중"), ("EXAM", "신경", "좌측 편마비"))
    hits = check_all("TEST", "요추천자", s)
    assert {h["rule"] for h in hits} == {"lp_ct_first", "lp_coagulation"}
    assert all(h["severity"] == "block" for h in hits)
    assert check("TEST", "요추천자", s)["rule"] == "lp_ct_first"


@pytest.mark.perf
def test_fast():
    s = st(WOMAN, *[("ASK", f"질문 {i}", "잘 모르겠어요. 특별한 건 없어요.") for i in range(30)])

    def eight():
        for c in ("복부 CT", "요추천자", "뇌 MRI", "운동부하검사", "위내시경", "ABG", "내진", "CBC"):
            check("TEST", c, s)
    assert_fast(eight, 0.5, what="8 precondition checks")  # strict budget: < 500 ms for the 8 checks


# --- 2026-09-27: negation / subject read through the normalisation layer -----------------------------------
@pytest.mark.parametrize("text", [
    "복시나 안구 돌출, 시야 결손은 보이지 않음.",  # list negation across a comma (was a false block)
    "형이 뇌전증이 있어서 경련을 자주 했어요.",  # relative with a subject particle
    "가족력: 뇌경색, 고혈압",  # family-history heading
    "청진: 증상 발작 때 쌕쌕거림 들림, 수포음 없음.",  # "증상 발작" = an episode, not a seizure
])
def test_lp_layer_negation_and_subject_do_not_block(text):
    assert check("TEST", "요추천자", st(MENINGITIS, ("EXAM", "진찰", text)))["ok"]


@pytest.mark.parametrize("text", [
    "3년 전에 뇌경색 진단을 받으셨어요.",  # the patient, spoken of with an honorific (was dropped with the "셨" cut)
    "경련을 한 것 같아요.",  # hedged: still a risk
    "오늘 아침 경련을 했는지는 잘 모르겠어요.",  # uncertain: never counts as ruled out
])
def test_lp_layer_hedged_or_honorific_findings_still_block(text):
    assert check("TEST", "요추천자", st(MENINGITIS, ("ASK", "과거력", text)))["severity"] == "block"
