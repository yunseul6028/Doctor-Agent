"""Can't-miss rule-out gate (safety/danger_gate.py): fake cases with environment responses, no LLM."""
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.safety import danger_gate as g
from doctor_agent.safety.danger_gate import gate, next_rule_out_action, polarity, status, unresolved_dangers

A, E, T = ActionType.ASK, ActionType.EXAM, ActionType.TEST


def case(initial: str, *turns: tuple[ActionType, str, str], ledger: list[dict] | None = None) -> CaseState:
    s = CaseState(initial_info=initial)
    for typ, content, response in turns:
        s.turns.append(Turn(Action(typ, content), response))
    if ledger:
        s.ddx_ledger.update(ledger)
    return s


CHEST = "58세 남성. 주호소: 5시간 전부터 시작된 가슴 통증"
ECG_OK = (T, "12유도 심전도", "정상 동율동, 심박수 78회/분, ST-T 변화 없음.")
TROP_OK = (T, "고감도 심장 트로포닌", "Troponin I < 0.01 ng/mL")


def names(state, **kw) -> list[str]:
    return [d["dx"] for d in unresolved_dangers(state, **kw)]


# --- negation-aware reading --------------------------------------------------------------------------------------

def test_polarity_trailing_negation_covers_a_list():
    t = "급성 출혈이나 급성 경색, 종괴 효과, 정중선 이동 소견 없음."
    assert polarity(t, "출혈") == "neg"
    assert polarity(t, "경색") == "neg"


def test_polarity_affirmation_before_comma_stays_affirmed():
    assert polarity("ST 분절 상승, T파 역전 없음", "st 분절") == "pos"
    assert polarity("ST 분절 상승 소견 없으나 T파 역전 관찰됨", "st 분절") == "neg"
    assert polarity("ST 분절 상승 소견 없으나 T파 역전 관찰됨", "t파 역전") == "pos"


def test_polarity_negated_verbs_and_english():
    assert polarity("대동맥 박리 소견은 관찰되지 않음", "박리") == "neg"
    assert polarity("No acute intracranial hemorrhage.", "hemorrhage") == "neg"
    assert polarity("뇌출혈 의심 소견 없음", "출혈") == "neg"
    assert polarity("왼쪽 고환으로 가는 혈류 감소 소견을 보이며", "혈류 감소") == "pos"


# --- chest pain --------------------------------------------------------------------------------------------------

def test_chest_pain_acs_unresolved_first_action_is_ecg():
    s = case(CHEST)
    assert "급성 관상동맥 증후군" in names(s, exclude=None)
    typ, content, reason = next_rule_out_action("급성 관상동맥 증후군", s)
    assert typ == T and "심전도" in content and "AHA/ACC" in reason


def test_acs_ruled_out_by_normal_ecg_and_troponin_after_3h():
    s = case(CHEST, ECG_OK, TROP_OK)
    st, ev = status("급성 관상동맥 증후군", s)
    assert st == "ruled_out" and any("심전도" in e for e in ev) and any("troponin" in e for e in ev)


def test_acs_early_presentation_needs_serial_troponin():
    s = case("50세 남성. 주호소: 1시간 전부터 시작된 흉통", ECG_OK, TROP_OK)
    assert status("급성 관상동맥 증후군", s)[0] == "unresolved"
    typ, content, _ = next_rule_out_action("급성 관상동맥 증후군", s)
    assert "재검" in content
    s2 = case("50세 남성. 주호소: 1시간 전부터 시작된 흉통", ECG_OK, TROP_OK,
              (T, "고감도 심장 트로포닌 재검(첫 검사 1~3시간 뒤)", "Troponin I < 0.01 ng/mL"))
    assert status("급성 관상동맥 증후군", s2)[0] == "ruled_out"


def test_acs_confirmed_by_st_elevation():
    s = case(CHEST, (T, "심전도", "II, III, aVF 유도에서 ST 분절 상승 관찰됨."))
    assert status("급성 관상동맥 증후군", s)[0] == "confirmed"
    assert status("급성 하벽 심근경색", s)[0] == "confirmed"  # more specific name maps to the entry


def test_acs_negated_st_elevation_is_normal_and_rising_troponin_wins():
    s = case(CHEST, (T, "심전도", "ST 분절 상승 소견 없음. T파 이상 없음."),
             (T, "트로포닌", "Troponin-I 0.08 ng/mL (참고치 < 0.04 ng/mL)."))
    assert g.PROBES["ecg"].read(g._Ctx(s)).result == "normal"
    assert status("급성 관상동맥 증후군", s)[0] == "unresolved"  # troponin raised: not ruled out, not STEMI


def test_dissection_not_live_without_triggers_and_negated_trigger():
    assert "대동맥 박리" not in names(case(CHEST, ECG_OK, TROP_OK), exclude=None)
    s = case(CHEST, (A, "통증 양상은?", "찢어지는 느낌은 아니에요. 누르는 것 같아요."))
    assert "대동맥 박리" not in names(s, exclude=None)


def test_dissection_low_add_rs_d_dimer_path():
    s = case(CHEST, (A, "통증 양상은?", "등으로 찢어지는 듯이 아파요."))
    assert "대동맥 박리" in names(s, exclude=None)
    typ, content, _ = next_rule_out_action("대동맥 박리", s)
    assert content == "D-dimer"
    s2 = case(CHEST, (A, "통증 양상은?", "등으로 찢어지는 듯이 아파요."), (T, "D-dimer", "D-dimer 0.3 μg/mL."))
    st, ev = status("대동맥 박리", s2)
    assert st == "ruled_out" and "ADD-RS ≤ 1" in ev


def test_dissection_high_add_rs_goes_straight_to_cta():
    s = case("67세 남성, 마르판 증후군. 주호소: 2시간 전 갑자기 찢어지는 흉통",
             (E, "활력징후", "혈압 82/50 mmHg, 맥박 118회/분"))
    assert g.add_rs(g._Ctx(s)) >= 2
    assert "CT 혈관조영" in next_rule_out_action("대동맥 박리", s)[1]
    s2 = case("67세 남성, 마르판 증후군. 주호소: 2시간 전 갑자기 찢어지는 흉통",
              (T, "흉부 대동맥 CT 혈관조영", "상행 대동맥에서 내막 피판(intimal flap)과 가성 내강 관찰됨."))
    assert status("대동맥 박리", s2)[0] == "confirmed"


def test_pe_wells_low_plus_d_dimer_negative_rules_out_and_age_adjusted():
    init = "72세 여성. 주호소: 어제부터 숨이 차고 가슴이 아픔. 2주 전 무릎 수술"
    s = case(init, (T, "D-dimer", "D-dimer 0.62 μg/mL"))  # below 720 ng/mL age-adjusted cut-off
    st, ev = status("폐색전증", s)
    assert st == "ruled_out" and any("Wells" in e for e in ev)


def test_pe_positive_d_dimer_then_ctpa():
    init = "45세 남성. 주호소: 어제부터 가슴 통증, 비행기 12시간 탄 뒤 다리가 부음"
    s = case(init, (T, "D-dimer", "D-dimer 1.8 μg/mL (참고치 < 0.5)"))
    assert "폐색전증" in names(s, exclude=None)
    assert "CTPA" in next_rule_out_action("폐색전증", s)[1]
    s_neg = case(init, (T, "CT 폐동맥 조영술(CTPA)", "폐동맥 내 충전 결손 없음. 특이 소견 없음."))
    assert status("폐색전증", s_neg)[0] == "ruled_out"
    s_pos = case(init, (T, "CTPA", "우하엽 분절 폐동맥에 충전 결손(filling defect) 관찰됨."))
    assert status("폐색전증", s_pos)[0] == "confirmed"


def test_tension_pneumothorax_ruled_out_by_breath_sounds_and_ordered_first():
    init = "30세 남성. 주호소: 2시간 전 갑자기 시작된 흉통과 호흡곤란"
    s = case(init)
    ds = names(s, exclude=None)
    assert ds[0] == "긴장성 기흉"  # tier 1 before ACS / PE
    s2 = case(init, (E, "흉부 청진", "양측 호흡음 대칭적이며 정상. 수포음 없음."))
    assert status("긴장성 기흉", s2)[0] == "ruled_out"
    assert "긴장성 기흉" not in names(s2, exclude=None)


# --- headache ----------------------------------------------------------------------------------------------------

HA = "42세 여성. 주호소: 3시간 전 갑자기 시작된 인생 최악의 두통"
CT_OK = (T, "비조영 뇌 CT", "급성 출혈이나 급성 경색, 종괴 효과, 정중선 이동 소견 없음.")


def test_sah_ct_within_6h_alert_rules_out():
    s = case(HA, (E, "신경학적 진찰", "의식 명료, 지남력 유지. 경부 강직 없음. 국소 신경학적 결손 없음."), CT_OK)
    st, ev = status("지주막하 출혈", s)
    assert st == "ruled_out" and any("6시간" in e for e in ev)


def test_sah_late_presentation_needs_lp():
    init = "42세 여성. 주호소: 어제 갑자기 시작된 벼락두통"
    s = case(init, CT_OK)
    assert status("지주막하 출혈", s)[0] == "unresolved"
    assert "요추천자" in next_rule_out_action("지주막하 출혈", s)[1]
    s2 = case(init, CT_OK, (T, "요추천자", "뇌척수액 맑음, 적혈구 2/μL, 백혈구 1/μL, 황색변색 없음."))
    assert status("지주막하 출혈", s2)[0] == "ruled_out"


def test_sah_confirmed_and_gate_allows_that_diagnosis():
    s = case(HA, (T, "뇌 CT", "기저 수조에 지주막하 출혈 소견 관찰됨."))
    assert status("지주막하 출혈", s)[0] == "confirmed"
    d = gate(s, "지주막하 출혈", remaining_turns=40)
    assert d["allow"] and d["danger"] == "지주막하 출혈"
    d2 = gate(s, "편두통", remaining_turns=40)  # never a block: only a hint for the caller
    assert d2["allow"] and d2["kind"] == "confirmed_other" and d2["danger"] == "지주막하 출혈" and d2["evidence"]


def test_meningitis_triad_absent_rules_out():
    init = "35세 남성. 주호소: 2일 전부터 열이 나고 머리가 아픔"
    s = case(init, (E, "활력징후", "체온 37.2℃, 혈압 124/80 mmHg, 맥박 80회/분"))
    assert status("뇌수막염", s)[0] == "unresolved"  # fever history affirmed in the chief complaint
    init2 = "35세 남성. 주호소: 2일 전부터 머리가 아프고 목이 뻣뻣한 느낌"
    s2 = case(init2, (A, "열이 났나요?", "아니요, 열은 없었어요."),
              (E, "활력징후", "체온 36.8℃, 혈압 120/80 mmHg"),
              (E, "신경학적 진찰", "의식 명료, 지남력 유지. 경부 강직 없음, Kernig 징후 음성."))
    st, ev = status("뇌수막염", s2)
    assert st == "ruled_out"


def test_meningitis_signs_present_leads_to_lp_and_csf_confirms():
    init = "25세 여성. 주호소: 어제부터 고열과 두통"
    s = case(init, (E, "활력징후", "체온 39.1℃, 혈압 118/70 mmHg"),
             (E, "신경학적 진찰", "의식 명료. 경부 강직 있음. Kernig 징후 양성."))
    assert "요추천자" in next_rule_out_action("뇌수막염", s)[1]
    s2 = case(init, (T, "요추천자 뇌척수액 검사", "뇌척수액 혼탁, 백혈구 1,250/μL(호중구 90%), 단백 180 mg/dL, 당 20 mg/dL"))
    assert status("뇌수막염", s2)[0] == "confirmed"


# --- abdominal pain ----------------------------------------------------------------------------------------------

def test_ectopic_hcg_negative_rules_out_and_male_not_listed():
    init = "26세 여성. 주호소: 오늘 아침부터 시작된 하복부 통증"
    s = case(init)
    assert "자궁외 임신" in names(s, exclude=None)
    assert "hCG" in next_rule_out_action("자궁외 임신", s)[1]
    s2 = case(init, (T, "소변 임신 검사", "소변 β-hCG: 음성"))
    assert status("자궁외 임신", s2)[0] == "ruled_out"
    assert "자궁외 임신" not in names(case("26세 남성. 주호소: 오늘 아침부터 시작된 하복부 통증"), exclude=None)


def test_ectopic_positive_hcg_then_intrauterine_pregnancy():
    init = "29세 여성. 주호소: 어제부터 아랫배 통증"
    s = case(init, (T, "혈청 β-hCG", "혈청 β-hCG 2,400 mIU/mL (양성)"))
    assert "질식" in next_rule_out_action("자궁외 임신", s)[1]
    s2 = case(init, (T, "혈청 β-hCG", "혈청 β-hCG 2,400 mIU/mL (양성)"),
              (T, "질식 초음파", "자궁 내 임신낭과 난황낭 확인됨. 부속기 종괴 없음."))
    assert status("자궁외 임신", s2)[0] == "ruled_out"


def test_aaa_rupture_live_in_older_patient_and_ruled_out_by_us():
    init = "72세 남성. 주호소: 3시간 전부터 복통과 허리 통증"
    s = case(init)
    assert "복부 대동맥류 파열" in names(s, exclude=None)
    s2 = case(init, (T, "복부 대동맥 초음파", "복부 대동맥 직경 1.9 cm로 정상, 대동맥류 없음."))
    assert status("복부 대동맥류 파열", s2)[0] == "ruled_out"


def test_perforation_live_only_with_peritoneal_signs():
    init = "28세 남성. 주호소: 어제부터 시작된 복통"
    assert "장 천공" not in names(case(init), exclude=None)
    s = case(init, (E, "복부 진찰", "우하복부 압통, 반발통 있음"))
    assert "장 천공" in names(s, exclude=None)
    s2 = case(init, (E, "복부 진찰", "우하복부 압통, 반발통 있음"),
              (T, "복부 CT", "충수 직경 11mm로 확장, 주위 지방층 침윤. 자유 공기나 농양 소견은 없음."))
    assert status("장 천공", s2)[0] == "ruled_out"


# --- fever / sepsis ----------------------------------------------------------------------------------------------

def test_sepsis_ruled_out_by_qsofa_zero_and_normal_lactate():
    init = "60세 여성. 주호소: 어제부터 고열과 오한"
    s = case(init, (E, "활력징후", "체온 38.9℃, 혈압 128/76 mmHg, 맥박 96회/분, 호흡수 18회/분"),
             (T, "혈중 젖산", "혈청 젖산(Lactate) 1.1 mmol/L"))
    assert status("패혈증", s)[0] == "ruled_out"


def test_lactate_not_confused_with_ldh_and_high_lactate_confirms():
    init = "60세 여성. 주호소: 어제부터 고열과 오한"
    s = case(init, (E, "활력징후", "체온 38.9℃, 혈압 128/76 mmHg, 호흡수 18회/분"),
             (T, "생화학 검사", "젖산탈수소효소(LDH) 285 U/L (경도 상승)"))
    assert g.PROBES["lactate"].read(g._Ctx(s)).result is None
    assert status("패혈증", s)[0] == "unresolved"
    s2 = case(init, (T, "혈중 젖산", "혈청 젖산(Lactate) 4.6 mmol/L (상승 소견)."))
    assert status("패혈증", s2)[0] == "confirmed"


def test_neutropenic_fever_from_wbc_and_percent():
    init = "55세 여성, 유방암 항암치료 중. 주호소: 오늘 발열"
    s = case(init)
    assert "호중구감소성 발열" in names(s, exclude=None)
    s2 = case(init, (T, "일반혈액검사", "백혈구 6,800/μL, 호중구 70%, 혈색소 11.2 g/dL"))
    assert status("호중구감소성 발열", s2)[0] == "ruled_out"


# --- ledger-derived dangers, working diagnosis, ordering ---------------------------------------------------------

def test_ledger_danger_without_protocol_is_included_and_ruled_out_by_ketones():
    init = "19세 남성. 주호소: 이틀 전부터 구토와 복통"
    ledger = [{"dx": "급성 위장염", "p": 0.5, "status": "유력"},
              {"dx": "당뇨병성 케톤산증", "p": 0.2, "status": "위험"}]
    s = case(init, ledger=ledger)
    ds = unresolved_dangers(s)
    dka = next(d for d in ds if d["dx"] == "당뇨병성 케톤산증")
    assert dka["source"] == "ddx_ledger"
    s2 = case(init, (T, "소변검사", "비중 1.018, pH 6.0, 단백 음성, 당 음성, 케톤 음성, 잠혈 음성"), ledger=ledger)
    assert status("DKA", s2)[0] == "ruled_out"


def test_working_diagnosis_is_excluded_by_default():
    ledger = [{"dx": "폐색전증", "p": 0.6, "status": "위험"}, {"dx": "폐렴", "p": 0.3, "status": "유력"}]
    s = case("45세 남성. 주호소: 어제부터 가슴 통증, 비행기 12시간 탄 뒤 다리가 부음", ledger=ledger)
    assert "폐색전증" not in names(s)
    assert "폐색전증" in names(s, exclude=None)


def test_unknown_diagnosis_status():
    assert status("편두통", case(HA)) == ("unresolved", [])
    assert next_rule_out_action("편두통", case(HA)) is None


# --- gate --------------------------------------------------------------------------------------------------------

def test_gate_blocks_with_rule_out_action():
    d = gate(case(CHEST), "위식도 역류질환", remaining_turns=40)
    assert not d["allow"] and d["kind"] == "rule_out" and d["danger"] == "급성 관상동맥 증후군"
    typ, content, reason = d["action"]
    assert typ == T and "심전도" in content


def test_gate_allows_after_rule_out_complete():
    d = gate(case(CHEST, ECG_OK, TROP_OK), "위식도 역류질환", remaining_turns=40)
    assert d["allow"] and d["kind"] == "none"


def test_gate_limits():
    s = case(CHEST)
    assert gate(s, "위식도 역류질환", remaining_turns=2)["allow"]
    assert gate(s, "위식도 역류질환", remaining_turns=40, max_gate_turns=3, gate_turns_used=3)["allow"]
    assert not gate(s, "위식도 역류질환", remaining_turns=40, max_gate_turns=3, gate_turns_used=2)["allow"]


def test_gate_proposed_danger_is_not_demanded_but_others_are():
    s = case(CHEST)
    d = gate(s, "급성 심근경색", remaining_turns=40)
    assert d["danger"] != "급성 관상동맥 증후군"


def test_gate_skips_done_and_unavailable_actions():
    s = case(CHEST, (T, "12유도 심전도", "해당 검사 결과는 제공되지 않습니다."))
    d = gate(s, "위식도 역류질환", remaining_turns=40)
    assert not d["allow"] and "트로포닌" in d["action"][1]


def test_gate_allows_when_no_rule_out_action_left():
    s = case(CHEST, (T, "12유도 심전도", "해당 검사 결과는 제공되지 않습니다."),
             (T, "고감도 심장 트로포닌", "해당 검사 결과는 제공되지 않습니다."),
             (T, "고감도 심장 트로포닌 재검(첫 검사 1~3시간 뒤)", "해당 검사 결과는 제공되지 않습니다."))
    d = gate(s, "위식도 역류질환", remaining_turns=40)
    assert d["allow"] and "배제 행동 없음" in d["why"]


def test_multi_result_response_is_segmented():
    s = case(CHEST, (T, "심전도와 흉부 X선", "심전도: 정상 동율동, ST-T 변화 없음. 흉부 X선: 좌측 기흉 관찰됨."))
    c = g._Ctx(s)
    assert c.probe("ecg").result == "normal"
    assert c.probe("cxr_ptx").result == "abnormal"


def test_table_entries_have_citations_steps_and_valid_probes():
    for r in g.RULE_OUT_TABLE:
        assert r.citations and all(x.verified for x in r.citations), r.name
        assert r.verification in ("primary", "secondary", "unverified")
        assert r.steps and all(s.probe in g.PROBES for s in r.steps), r.name
        assert g.lookup(r.name) is r


# --- regressions from the data/cases_aug audit (2026-09-27) ------------------------------------------------------

def test_negated_intrauterine_sac_is_not_an_intrauterine_pregnancy():
    init = "27세 여성. 주호소: 오늘 아침부터 시작된 아랫배 통증"
    s = case(init, (T, "소변 임신 검사", "소변 임신반응검사 양성, 혈청 β-hCG 3,850 mIU/mL"),
             (T, "질식 초음파", "자궁 내 임신낭 없음, 좌측 부속기에 3.2cm 불균질 종괴, 다량의 복강 내 액체(혈복강 의심)"))
    assert status("자궁외 임신", s)[0] == "confirmed"
    assert gate(s, "자궁외임신 파열", remaining_turns=40)["allow"]


def test_ectopic_not_live_in_established_pregnancy():
    s = case("30세 여성. 주호소: 한 시간 전부터 질출혈과 심한 복통 (임신 38주)")
    assert "자궁외 임신" not in names(s, exclude=None)


def test_pericarditis_st_elevation_is_not_stemi():
    s = case(CHEST, (T, "심전도", "다수의 유도에서 오목한(concave upward) ST 분절 상승과 PR 분절 하강이 관찰됨"))
    assert status("급성 관상동맥 증후군", s)[0] == "unresolved"


def test_serial_troponin_in_one_report_and_onset_not_taken_from_unrelated_answers():
    init = "47세 남성. 주호소: 2주 전부터 반복되는 가슴 통증"
    s = case(init, (A, "식사와 관련 있나요?", "밥 먹고 1시간 뒤에 심해져요."), ECG_OK,
             (T, "트로포닌", "고감도 트로포닌 I 3 ng/L, 3시간 후 재검 4 ng/L (참고치 <34)"))
    c = g._Ctx(s)
    assert c.onset_h == 336 and c.probe("troponin").n_normal == 2
    assert status("급성 관상동맥 증후군", s)[0] == "ruled_out"


def test_ischemic_stroke_needs_focal_deficit_to_be_live():
    assert "급성 허혈성 뇌졸중" not in names(case("48세 여성. 주호소: 오늘 아침부터 의식변화"), exclude=None)
    assert "뇌출혈" in names(case("48세 여성. 주호소: 오늘 아침부터 의식변화"), exclude=None)
    s = case("71세 남성. 주호소: 1시간 전 갑자기 생긴 오른쪽 팔다리 힘 빠짐")
    assert "급성 허혈성 뇌졸중" in names(s, exclude=None)


# --- 2026-09-27: normalisation-layer reading (conservative union; relatives dropped; measured vitals) ----------
def test_polarity_layer_adds_affirmations_and_drops_relatives():
    assert polarity("열이 안 떨어져요", "열이") == "pos"  # persistence
    assert polarity("경련을 했는지는 잘 모르겠어요", "경련") == "pos"  # uncertain never reads as negative
    assert polarity("어머니가 폐색전증을 앓으셨어요", "폐색전증") is None  # a relative's disease is not the patient's
    assert polarity("폐색전증은 없었어요. 어머니가 폐색전증을 앓으셨어요", "폐색전증") == "neg"


def test_ctx_affirmed_ignores_relatives_for_wells():
    s = case(CHEST, (A, "과거력", "어머니가 유방암 치료를 받으셨어요. 저는 특별한 병은 없어요."))
    assert not g._Ctx(s).affirmed(g._CANCER)


def test_vitals_read_from_measured_findings():
    c = g._Ctx(case("4세 여아. 주호소: 발열",
                    (E, "활력징후", "체온 38.4°C, 맥박 118회/분, 호흡수 28회/분"),
                    (E, "혈압", "혈압 우측 팔 170/100 mmHg")))
    assert (c.hr, c.rr, c.temp, c.sbp) == (118, 28, 38.4, 170)


def test_vitals_worst_within_a_response_and_catheter_saturation_is_not_spo2():
    c = g._Ctx(case(CHEST,
                    (E, "기립 검사", "누운 자세 혈압 126/80 mmHg, 기립 3분 후 혈압 106/70 mmHg"),
                    (E, "산소포화도", "산소포화도 96%"),
                    (T, "심도자술", "폐동맥 포화도 65%에서 45%로 감소")))
    assert c.sbp == 106 and c.spo2 == 96


# --- 2026-09-30: over-firing fixes (eval/offline/eval_danger_gate.py) -------------------------------------------
DYSPNEA_1M = "61세 여성. 주호소: 한 달째 발작처럼 오는 기침과 숨참"
DYSPNEA_ACUTE = "70세 남성. 주호소: 3일 전부터 시작된 호흡곤란"
NO_CUES = (E, "심장 진찰", "심음 정상(S1, S2), S3/S4 없음. 경정맥 확장 없음. 양측 하지 부종 없음.")


def test_heart_failure_needs_a_clue_to_be_live_from_dyspnea():
    assert "급성 심부전" not in names(case(DYSPNEA_ACUTE), exclude=None)
    assert "급성 심부전" not in names(case(DYSPNEA_ACUTE, NO_CUES), exclude=None)
    assert "급성 심부전" in names(case(DYSPNEA_ACUTE, (A, "누우면 어떤가요", "누우면 숨이 더 차서 베개를 여러 개 베고 자요.")),
                                exclude=None)
    assert "급성 심부전" in names(case(DYSPNEA_ACUTE, (E, "하지", "양측 하지에 함요 부종(2+) 관찰됨.")), exclude=None)
    assert "급성 심부전" in names(case("70세 남성. 주호소: 숨이 차고 다리 부종"), exclude=None)
    # the model's own ledger flag still makes it live without a clue
    assert "급성 심부전" in names(case(DYSPNEA_ACUTE, ledger=[{"dx": "급성 심부전", "p": 0.2, "status": "위험"}]),
                                exclude=None)


def test_heart_failure_clue_negated_inside_parentheses_does_not_count():
    s = case(DYSPNEA_ACUTE, (E, "폐 청진", "양쪽 호흡음 깨끗하고 부속음(수포음, 천명음)은 들리지 않음"))
    assert "급성 심부전" not in names(s, exclude=None)
    s2 = case(DYSPNEA_ACUTE, (E, "폐 청진", "양측 폐하부에서 미세한 수포음 청진됨"))
    assert "급성 심부전" in names(s2, exclude=None)


def test_heart_failure_ruled_out_by_normal_echo_not_by_low_ef():
    cue = (E, "하지", "양측 하지 함요 부종 있음")
    assert status("급성 심부전", case(DYSPNEA_ACUTE, cue, (T, "심초음파", "좌심실 박출률 62%로 정상. 판막 질환 없음.")))[0] \
        == "ruled_out"
    assert status("급성 심부전", case(DYSPNEA_ACUTE, cue, (T, "심초음파", "정상 범위 내 결과")))[0] == "ruled_out"
    assert status("급성 심부전", case(DYSPNEA_ACUTE, cue, (T, "심초음파", "좌심실 구출률 35%, 나머지 정상")))[0] \
        == "unresolved"
    assert status("급성 심부전", case(DYSPNEA_ACUTE, cue, (T, "심초음파", "좌심실 구출률 60%, 이완기 기능 장애 소견")))[0] \
        == "unresolved"


def test_tension_pneumothorax_not_live_for_longstanding_dyspnea_alone():
    assert "긴장성 기흉" not in names(case(DYSPNEA_1M), exclude=None)
    assert "긴장성 기흉" in names(case(DYSPNEA_ACUTE), exclude=None)
    assert "긴장성 기흉" in names(case(DYSPNEA_1M, (E, "활력징후", "혈압 120/80 mmHg, 산소포화도 89%")), exclude=None)
    assert "긴장성 기흉" in names(case("25세 남성. 주호소: 갑작스러운 좌측 가슴통증과 호흡곤란"), exclude=None)


def test_breath_sounds_normal_statement_not_negated_by_a_later_finding():
    s = case(DYSPNEA_ACUTE, (E, "흉부 청진", "양측 폐야 호흡음 명료함, 수포음이나 천명음 없음, 호흡 보조근 사용 없음."))
    assert status("긴장성 기흉", s)[0] == "ruled_out"
    # "no adventitious sounds" is not "absent breath sounds"
    s2 = case(DYSPNEA_ACUTE, (E, "흉부 청진", "청진상 이상 호흡음 없음. 양측 폐야에서 호흡음 청명함."))
    assert status("긴장성 기흉", s2)[0] == "ruled_out"
    s3 = case(DYSPNEA_ACUTE, (E, "흉부 청진", "좌측 호흡음 감소, 우측 호흡음 명료"))
    assert status("긴장성 기흉", s3)[0] != "ruled_out"


FEVER = "35세 여성. 주호소: 2일 전부터 시작된 우측 옆구리 통증과 배뇨통, 고열"
SEPSIS_VITALS = (E, "활력징후", "체온 39.2°C, 혈압 110/70 mmHg, 맥박 102회/분, 호흡수 20회/분")
SEPSIS_NEURO = (E, "신경학적", "의식 명료(GCS 15점), 지남력 유지됨.")
SEPSIS_LABS = (T, "혈액검사", "WBC 18,500/μL, Platelet 230,000/μL, CRP 156 mg/L, BUN 22 mg/dL, Cr 1.1 mg/dL")
SEPSIS_LFT = (T, "간기능", "AST 24 U/L, ALT 22 U/L, 총빌리루빈 0.8 mg/dL")


def test_sepsis_ruled_out_without_lactate_when_no_sofa_organ_dysfunction():
    s = case(FEVER, SEPSIS_VITALS, SEPSIS_NEURO, SEPSIS_LABS, SEPSIS_LFT)
    st, ev = status("패혈증", s)
    assert st == "ruled_out" and any("SOFA" in e for e in ev)


def test_sepsis_sofa_path_needs_every_organ_measured_and_normal():
    assert status("패혈증", case(FEVER, SEPSIS_VITALS, SEPSIS_NEURO, SEPSIS_LABS))[0] == "unresolved"  # no bilirubin
    low_plt = (T, "혈액검사", "WBC 18,500/μL, 혈소판 90,000/μL, Cr 1.1 mg/dL")
    assert status("패혈증", case(FEVER, SEPSIS_VITALS, SEPSIS_NEURO, low_plt, SEPSIS_LFT))[0] == "unresolved"
    high_cr = (T, "혈액검사", "WBC 18,500/μL, Platelet 230,000/μL, Cr 1.6 mg/dL")
    assert status("패혈증", case(FEVER, SEPSIS_VITALS, SEPSIS_NEURO, high_cr, SEPSIS_LFT))[0] == "unresolved"
    tachypnea = (E, "활력징후", "체온 39.2°C, 혈압 110/70 mmHg, 맥박 102회/분, 호흡수 24회/분")
    assert status("패혈증", case(FEVER, tachypnea, SEPSIS_NEURO, SEPSIS_LABS, SEPSIS_LFT))[0] == "unresolved"
    low_map = (E, "활력징후", "체온 39.2°C, 혈압 105/45 mmHg, 맥박 102회/분, 호흡수 20회/분")  # MAP 65
    assert status("패혈증", case(FEVER, low_map, SEPSIS_NEURO, SEPSIS_LABS, SEPSIS_LFT))[0] == "unresolved"
    assert status("패혈증", case(FEVER, SEPSIS_VITALS, SEPSIS_LABS, SEPSIS_LFT))[0] == "unresolved"  # mental unknown


def test_sepsis_sofa_readers_skip_urine_and_read_si_units():
    c = g._Ctx(case(FEVER, (T, "소변검사", "크레아티닌 0.5 mg/dL"), (T, "혈액검사", "혈소판 110 × 10⁹/L")))
    assert c.probe("creatinine").result != "normal"
    assert c.probe("platelets").result == "abnormal"
    c2 = g._Ctx(case(FEVER, (T, "간기능", "직접 빌리루빈 0.2 mg/dL")))
    assert c2.probe("bilirubin").result != "normal"


def test_ectopic_not_live_after_stated_menopause_or_hysterectomy():
    s = case("53세 여성. 주호소: 어젯밤부터 계속되는 윗배 통증", (A, "생리", "3년 전에 폐경됐어요."))
    assert "자궁외 임신" not in names(s, exclude=None)
    s2 = case("48세 여성. 주호소: 어젯밤부터 계속되는 아랫배 통증", (A, "생리", "폐경은 아직 안 됐어요."))
    assert "자궁외 임신" in names(s2, exclude=None)
    s3 = case("40세 여성. 주호소: 아랫배 통증", (A, "수술", "자궁 적출술을 받았어요."))
    assert "자궁외 임신" not in names(s3, exclude=None)


def test_ectopic_confirmed_from_a_generically_requested_ultrasound():
    s = case("27세 여성. 주호소: 오늘 아침부터 시작된 아랫배 통증",
             (T, "임신 검사", "소변 임신반응검사 양성, 혈청 β-hCG 3,850 mIU/mL"),
             (T, "초음파", "질식 초음파: 자궁 내 임신낭 없음, 좌측 부속기에 3.2cm 불균질 종괴, 다량의 복강 내 액체(혈복강 의심)"))
    assert status("자궁외 임신", s)[0] == "confirmed"
    assert gate(s, "자궁외임신 파열", remaining_turns=20)["allow"]


def test_dka_spelling_variant_maps_to_the_entry():
    assert g.lookup("정상혈당 당뇨병성 케토산증").name == "당뇨병성 케톤산증"
    assert g.lookup("당뇨병성 케토산증").name == "당뇨병성 케톤산증"


def test_bilateral_leg_edema_is_not_a_wells_dvt_sign():
    base = "56세 여성. 주호소: 4일 전부터 시작된 기침과 호흡곤란"
    assert g.wells_partial(g._Ctx(case(base, (E, "사지", "양측 하지 부종(+1) 관찰됨.")))) == 0
    assert g.wells_partial(g._Ctx(case(base, (A, "다리", "오른쪽 다리가 부었어요.")))) == 3
    assert g.wells_partial(g._Ctx(case(base, (A, "다리", "한쪽 다리에 힘이 빠져요.")))) == 0
    assert g.wells_partial(g._Ctx(case(base, (E, "사지", "좌측 종아리 압통 있음")))) == 3


def test_ischemic_stroke_not_demanded_once_hemorrhage_is_confirmed():
    s = case("71세 남성. 주호소: 1시간 전 갑자기 생긴 오른쪽 팔다리 힘 빠짐",
             (T, "뇌 CT", "좌측 기저핵에 3cm 크기의 급성 뇌내출혈(혈종) 관찰됨."))
    assert status("뇌출혈", s)[0] == "confirmed"
    assert "급성 허혈성 뇌졸중" not in names(s, exclude=None)
    s2 = case("71세 남성. 주호소: 1시간 전 갑자기 생긴 오른쪽 팔다리 힘 빠짐",
              (T, "뇌 CT", "급성 출혈 소견 없음."))
    assert "급성 허혈성 뇌졸중" in names(s2, exclude=None)


def test_raised_dangers_lists_ruled_out_ones_too():
    s = case(CHEST, ECG_OK, TROP_OK)
    raised = g.raised_dangers(s)
    assert raised.get("급성 관상동맥 증후군") == "chief_complaint"
    assert "급성 관상동맥 증후군" not in names(s, exclude=None)


def test_offline_danger_gate_script_runs_on_a_case_dir(tmp_path):
    import importlib.util
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("eval_danger_gate",
                                                  os.path.join(root, "eval", "offline", "eval_danger_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "c.json").write_text(json.dumps({
        "initial": "58세 남성. 주호소: 5시간 전부터 시작된 가슴 통증", "diagnosis": "위식도 역류질환",
        "tests": {"심전도|ecg": "정상 동율동, ST-T 변화 없음.", "트로포닌|troponin": "Troponin I < 0.01 ng/mL"}},
        ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "o.json"
    assert mod.main(["--cases", str(tmp_path), "--no-interp", "--json", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    row = res["cases"][0]
    assert "급성 관상동맥 증후군" in row["raised_initial"] and row["blocked"] is None
    assert res["summary"]["gold_blocked"] == 0
