"""Unstable-patient triage (safety/triage.py): fake cases with environment responses, no LLM."""
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.safety.triage import assess, priority_actions, render_for_prompt

A, E, T = ActionType.ASK, ActionType.EXAM, ActionType.TEST


def case(initial: str, *turns: tuple[ActionType, str, str]) -> CaseState:
    s = CaseState(initial_info=initial)
    for typ, content, response in turns:
        s.turns.append(Turn(Action(typ, content), response))
    return s


def keys(res: dict) -> set[str]:
    return {s["key"] for s in res["signals"]}


def contents(state, res=None) -> list[str]:
    return [a[1] for a in priority_actions(state, res or assess(state))]


def vitals(text: str) -> tuple[ActionType, str, str]:
    return (E, "활력징후", text)


NORMAL = vitals("혈압 124/78 mmHg, 맥박 78회/분, 호흡수 16회/분, 체온 36.8℃, 산소포화도 98%")


# --- missing vitals are not "stable" --------------------------------------------------------------------------

def test_no_vitals_is_unknown_and_asks_for_vitals_first():
    s = case("28세 남성. 주호소: 어제부터 시작된 복통")
    r = assess(s)
    assert r["level"] == "unknown"
    assert {"sbp", "hr", "rr", "temp"} <= set(r["missing"])
    acts = priority_actions(s, r)
    assert acts and acts[0][0] == E and "활력징후" in acts[0][1]
    text = render_for_prompt(s, r)
    assert "미확인" in text and len(text) <= 250


def test_normal_vitals_are_stable_with_no_actions():
    s = case("28세 남성. 주호소: 어제부터 시작된 복통", NORMAL)
    r = assess(s)
    assert r["level"] == "stable" and r["news2"] == 0
    assert priority_actions(s, r) == [] and render_for_prompt(s, r) == ""


def test_unavailable_vitals_after_asking_do_not_block_forever():
    s = case("40세 남성. 주호소: 2주 전부터 시작된 발진", (E, "활력징후", "해당 검사 결과는 제공되지 않습니다."))
    r = assess(s)
    assert r["level"] == "stable" and not r["missing"] and "sbp" in r["unavailable"]


def test_spo2_asked_separately_only_when_breathing_matters():
    routine = case("28세 남성. 주호소: 어제부터 시작된 복통",
                   vitals("체온 38.1℃, 맥박 98회/분, 혈압 124/78 mmHg, 호흡수 16회/분"))
    assert assess(routine)["level"] == "stable"
    chest = case("58세 남성. 주호소: 2시간 전부터 시작된 가슴 통증",
                 vitals("혈압 150/90 mmHg, 맥박 88회/분, 호흡수 18회/분, 체온 36.8℃"))
    r = assess(chest)
    assert r["level"] == "unknown" and r["missing"] == ["spo2"]
    assert contents(chest, r) == ["산소포화도(SpO2) 측정"]


# --- adult physiology --------------------------------------------------------------------------------------------

def test_septic_shock_is_unstable_and_asks_lactate_and_cultures():
    s = case("70세 여성. 주호소: 오늘 아침부터 헛소리를 해요",
             vitals("체온 38.9℃, 맥박 118회/분, 혈압 84/50 mmHg, 호흡수 26회/분, 산소포화도 95%"))
    r = assess(s)
    assert r["level"] == "unstable"
    assert {"sbp_low", "map_low", "news2_high", "qsofa", "ams"} <= keys(r)
    assert r["flags"]["sepsis_suspected"]
    acts = contents(s, r)
    assert "혈중 젖산(lactate)" in acts and "혈액 배양 2쌍(항생제 투여 전)" in acts and "혈당 측정" in acts


def test_done_actions_are_excluded():
    s = case("70세 여성. 주호소: 오늘 아침부터 헛소리를 해요",
             vitals("체온 38.9℃, 맥박 118회/분, 혈압 84/50 mmHg, 호흡수 26회/분, 산소포화도 95%"),
             (T, "혈중 젖산", "Lactate 4.2 mmol/L"), (T, "혈당", "혈당 112 mg/dL"))
    acts = contents(s)
    assert not any("젖산" in a or "혈당" in a for a in acts)
    assert "혈액 배양 2쌍(항생제 투여 전)" in acts


def test_chest_pain_with_hypotension_gets_ecg():
    s = case("62세 남성. 주호소: 1시간 전부터 시작된 가슴 통증",
             vitals("혈압 84/52 mmHg, 맥박 112회/분, 호흡수 22회/분, 체온 36.6℃, 산소포화도 94%"))
    r = assess(s)
    assert r["level"] == "unstable" and "chest_unstable" in keys(r)
    assert "12유도 심전도" in contents(s, r)


def test_stable_svt_is_concerning_not_unstable():
    s = case("23세 여성. 주호소: 약 30분 전 갑자기 시작된 심한 두근거림",
             vitals("혈압 110/70 mmHg, 맥박 180회/분, 호흡수 20회/분, 체온 36.7℃, 산소포화도 98%"))
    r = assess(s)
    assert r["level"] == "concerning" and r["shock_index"] >= 1.4
    acts = contents(s, r)
    assert "12유도 심전도" in acts and not any("hCG" in a for a in acts)


def test_hypoxaemia_below_90_is_unstable():
    s = case("34세 여성. 주호소: 오늘 아침부터 숨이 참",
             vitals("체온 37.4℃, 맥박 104회/분, 호흡수 24회/분, 혈압 116/72 mmHg, 산소포화도 88%(실내공기)"))
    r = assess(s)
    assert r["level"] == "unstable" and "spo2_low" in keys(r)
    assert contents(s, r)[0].startswith("호흡 평가")


def test_ruptured_ectopic_pattern_is_unstable_with_hcg_and_ultrasound():
    s = case("27세 여성. 주호소: 오늘 아침부터 시작된 아랫배 통증",
             vitals("혈압 92/58 mmHg, 맥박 118회/분, 호흡 22회/분, 체온 36.8℃, 산소포화도 98%"))
    r = assess(s)
    assert r["level"] == "unstable" and "occult_hemorrhage" in keys(r)
    acts = contents(s, r)
    assert "임신 검사(β-hCG)" in acts and any("초음파" in a for a in acts)


# --- mental status ------------------------------------------------------------------------------------------------

def test_gcs_8_is_critical_and_asks_airway_and_glucose():
    s = case("60세 남성. 주호소: 쓰러진 채 발견됨", (E, "의식 수준", "GCS 7점(E1V2M4), 통증 자극에만 반응함."), NORMAL)
    r = assess(s)
    assert r["level"] == "unstable" and "ams_severe" in keys(r) and r["gcs"] == 7
    acts = contents(s, r)
    assert acts[0].startswith("기도 평가") and "혈당 측정" in acts


def test_coma_scale_name_and_mild_stupor_are_not_avpu_p():
    s = case("4세 남아. 주호소: 오늘 발생한 쳐짐", (E, "활력징후", "글래스고 혼수 척도 14점, 심박수 분당 90회, 혈압 100/60 mmHg"))
    assert "ams_severe" not in keys(assess(s))
    s2 = case("46세 여성. 주호소: 오늘 생긴 의식변화", (E, "전신 상태", "의식은 약간 혼미하나 자극에 반응함."), NORMAL)
    r2 = assess(s2)
    assert "ams_severe" not in keys(r2) and "ams" in keys(r2)


def test_resolved_postictal_state_is_not_severe():
    s = case("22세 여성. 주호소: 갑작스러운 의식 소실 및 전신 경련",
             (E, "의식", "응급실 도착 당시 의식 혼미, 지남력 저하. 1시간 후 의식 완전히 회복됨."), NORMAL)
    assert "ams_severe" not in keys(assess(s))


def test_exam_alert_overrides_history_misreading():
    s = case("37세 여성. 주호소: 걸을 때 비틀거림",
             (A, "과거력", "크론병 표준 치료에 반응이 없어서 나탈리주맙을 치료받아 왔습니다."),
             (E, "신경학적", "의식 명료. 지남력 유지됨."), NORMAL)
    assert assess(s)["level"] == "stable"


# --- children -------------------------------------------------------------------------------------------------

def test_pediatric_hypotension_uses_pals_threshold():
    s = case("3세 남아. 주호소: 오늘 아침부터 열", vitals("체온 39.5℃, 맥박 120회/분, 호흡수 28회/분, 혈압 74/40 mmHg"))
    r = assess(s)
    assert r["pediatric"] and "sbp_low" in keys(r) and r["level"] == "unstable"  # 70 + 2*3 = 76
    adult = case("30세 남성. 주호소: 오늘 아침부터 열",
                 vitals("체온 37.5℃, 맥박 80회/분, 호흡수 16회/분, 혈압 95/60 mmHg, 산소포화도 98%"))
    assert "sbp_low" not in keys(assess(adult))


def test_pediatric_heart_rate_uses_age_centiles():
    infant = case("생후 2개월 남아. 주호소: 오늘부터 보챔",
                  vitals("맥박 150회/분, 호흡수 40회/분, 체온 37.0℃, 혈압 85/50 mmHg, 산소포화도 99%"))
    assert "hr_high" not in keys(assess(infant))
    child = case("10세 남아. 주호소: 오늘부터 보챔",
                 vitals("맥박 150회/분, 호흡수 20회/분, 체온 37.0℃, 혈압 105/65 mmHg, 산소포화도 99%"))
    assert "hr_high" in keys(assess(child))


def test_febrile_young_infant_is_concerning():
    s = case("생후 5주 여아. 주호소: 오늘부터 열", vitals("체온 38.3℃, 맥박 160회/분, 호흡수 44회/분"))
    assert "febrile_infant" in keys(assess(s))


def test_pregnancy_weeks_are_not_the_age():
    r = assess(case("24세 임신 30주 여성. 주호소: 갑자기 시작된 발열", NORMAL))
    assert not r["pediatric"] and r["age_years"] == 24


# --- specific syndromes -----------------------------------------------------------------------------------------

def test_anaphylaxis_is_unstable_and_airway_comes_first():
    s = case("25세 여성. 주호소: 30분 전 새우를 먹고 두드러기와 호흡곤란",
             (A, "증상", "온몸에 두드러기가 나고 숨이 차요."))
    r = assess(s)
    assert r["level"] == "unstable" and "anaphylaxis" in keys(r)
    acts = contents(s, r)
    assert acts[0].startswith("기도 평가") and any("활력징후" in a for a in acts)


def test_gi_bleeding_with_instability_is_critical():
    s = case("65세 남성. 주호소: 오늘 아침 피를 토했어요",
             vitals("혈압 96/60 mmHg, 맥박 112회/분, 호흡수 20회/분, 체온 36.5℃, 산소포화도 97%"))
    r = assess(s)
    assert "bleeding_unstable" in keys(r)
    assert "혈액 검사(CBC·혈색소)" in contents(s, r)


def test_small_hemoptysis_and_fatigue_vomiting_are_not_bleeding():
    s = case("60세 남성. 주호소: 간헐적으로 발생하는 객혈", (A, "양", "피가 살짝 섞인 가래가 가끔 나와요."), NORMAL)
    assert not assess(s)["flags"]["bleeding"]
    s2 = case("68세 남성. 주호소: 2일 전부터 시작된 소변량 감소",
              (E, "전신 상태", "의식은 명료하나 전신 피로감과 구토 소견 관찰됨."), NORMAL)
    assert not assess(s2)["flags"]["bleeding"]


def test_past_or_test_cardiac_arrest_is_not_now():
    s = case("55세 남성. 주호소: 한 시간 전 의식을 잃음",
             (E, "경동맥 자극", "경동맥 신경절 압박 시 심정지 시간 3초 미만으로 정상 반응임."), NORMAL)
    assert "arrest" not in keys(assess(s))
    assert "arrest" in keys(assess(case("60세 남성. 주호소: 심정지 후 자발순환 회복")))


def test_stridor_with_hypoxaemia_is_airway_threat():
    s = case("2세 남아. 주호소: 어젯밤부터 컹컹거리는 기침",
             (E, "호흡", "안정 시 흡기성 협착음 들림, 흉벽 함몰 있음."),
             vitals("맥박 150회/분, 호흡수 44회/분, 체온 38.0℃, 산소포화도 89%"))
    r = assess(s)
    assert r["level"] == "unstable" and "airway" in keys(r)


def test_render_is_capped():
    s = case("93세 남성. 주호소: 의식 저하, 토혈",
             vitals("혈압 70/40 mmHg, 맥박 140회/분, 호흡수 34회/분, 체온 35.0℃, 산소포화도 82%"),
             (E, "의식", "GCS 6점, 통증에만 반응."))
    text = render_for_prompt(s)
    assert text.startswith("[중증도: 불안정]") and len(text) <= 250
    assert len(render_for_prompt(s, max_chars=60)) <= 60
