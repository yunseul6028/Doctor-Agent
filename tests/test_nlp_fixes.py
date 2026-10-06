"""Regression tests for the normalisation-layer issues reported by the migration agents (2026-09-28).

One test per issue; each failed before its fix (see docs/nlp.md, "Fixes 2026-09-28")."""
from doctor_agent.nlp import LEXICON, parse
from doctor_agent.nlp import findings as F


def kp(text, source="patient", context=None):
    return {(f.concept, f.polarity) for f in parse(text, source, context)}


def by(text, concept, source="patient", context=None):
    return [f for f in parse(text, source, context) if f.concept == concept]


# --- parser ---------------------------------------------------------------------------------------------------
def test_01_newline_ends_a_sentence():
    # "b-hcg 양성\nhbsag 음성" is two lines of a report, not one list negated by the last word
    assert ("LAB:hcg_pos", "present") in kp("b-hcg 양성\nhbsag 음성", "test")
    assert ("SYM:headache", "present") in kp("두통\n발열 없음", "exam")
    fs = parse("두통\n발열 없음", "exam")
    t = F.normalize("두통\n발열 없음")
    assert all(t[f.start:f.end] == f.span for f in fs)  # offsets still index normalize(text)


def test_02_attributive_negation_and_normal_noun_phrase_do_not_negate_a_list():
    assert ("HX:pregnancy", "present") in kp("임신 32주, 통증 없는 질 출혈이 있어요.")
    got = kp("임신 32주, 통증 없는 질 출혈이 있어요.")
    assert {("SYM:pain", "absent"), ("SYM:vaginal_bleeding", "present")} <= got
    # "정상 축" is its own item (normal axis), not the predicate of the rhythm before it
    assert ("SIGN:tachycardia", "present") in kp("동성빈맥(118회/분), 정상 축, ST-T 변화 없음.", "test")
    # predicative / adverbial negations still cover the list
    assert kp("수포음, 천명음 없음.", "exam") >= {("SIGN:crackles", "absent"), ("SIGN:wheeze", "absent")}
    assert kp("수포음이나 천명음 없이 양호한 호흡음", "exam") >= {("SIGN:crackles", "absent"), ("SIGN:wheeze", "absent")}
    assert ("SYM:rash", "absent") in kp("특이 발진, 황달 없는 상태", "exam")


def test_03_list_item_with_words_of_its_own_keeps_its_reading():
    got = kp("좌심실비대 소견과 V5-V6 비특이적 ST분절 하강, ST분절 상승 없음", "test")
    assert ("ECG:ecg_st_depression", "present") in got and ("ECG:ecg_stemi", "absent") in got
    # the same head stated with another state word: the negation has its own subject
    got = kp("ST분절 하강, ST분절 상승 없음", "test")
    assert ("ECG:ecg_st_depression", "present") in got and ("ECG:ecg_stemi", "absent") in got
    assert ("SIGN:rebound_tenderness", "absent") in kp("명치 부위 경미한 압통, 반발통 없음", "exam")
    assert ("SIGN:epigastric_tenderness", "present") in kp("명치 부위 경미한 압통, 반발통 없음", "exam")
    # a bare list still inherits
    assert kp("경부 강직, 케르니그 징후 없음", "exam") >= {("SIGN:neck_stiffness", "absent"), ("SIGN:kernig", "absent")}
    # words after a coordinator belong to the next list item, not to the mention (safety snapshot regressions)
    assert ("SYM:headache", "absent") in kp("두통이나 호흡기, 소화기, 비뇨기 쪽 증상은 하나도 없습니다.")
    assert ("SYM:fever", "absent") in kp("요즘 감염된 적은 없고, 열이나 뚜렷한 체중 감소, 밤에 땀 나는 것도 없습니다.")
    assert ("SYM:syncope", "present") in kp("입원 중 발생한 실신 구조적 심장 질환이나 고혈압, 심근염을 진단받은 적은 전혀 없어요.")


def test_04_measured_value_wins_over_a_contradicting_verdict():
    assert by("심박수 108회/분, 빈맥 소견 없음.", "SIGN:tachycardia", "exam")[0].polarity == "present"
    assert {f.polarity for f in by("맥박 108회/분. 빈맥은 없음.", "SIGN:tachycardia", "exam")} == {"present"}
    assert {f.polarity for f in by("체온 36.5도, 발열 있음", "SYM:fever", "exam")} == {"absent"}
    # an unlabelled rate right after the rhythm word is a heart rate
    f = by("동성빈맥(118회/분)", "SIGN:tachycardia", "test")[0]
    assert (f.polarity, f.value) == ("present", 118.0)


def test_05_unstoppable_bleeding_is_bleeding():
    assert ("SYM:prolonged_bleeding", "present") in kp("지혈되지 않는 출혈이 있어요.")
    assert ("SYM:prolonged_bleeding", "present") in kp("코피가 지혈되지 않아요.")
    assert ("SYM:epistaxis", "present") in kp("코피가 지혈되지 않아요.")
    assert ("SYM:prolonged_bleeding", "absent") in kp("잘 지혈돼요.") | kp("출혈이 오래가지는 않아요.")


def test_06_symptom_attack_is_not_a_seizure():
    for t in ("증상 발작 시 가슴이 두근거려요.", "호흡곤란 발작이 있었어요.", "기침 발작이 심해요.", "공황 발작이 왔어요."):
        assert not any(c == "SYM:seizure" for c, _ in kp(t)), t
    assert ("SYM:seizure", "present") in kp("발작이 있었어요.")
    assert ("SYM:seizure", "present") in kp("경련 발작을 했어요.")


def test_07_vessel_saturation_is_not_spo2():
    assert not by("폐동맥 포화도 66%, 우심방 포화도 70%", "SIGN:hypoxemia", "test")
    assert not by("혼합정맥혈 산소포화도 55%", "SIGN:hypoxemia", "test")
    assert by("산소포화도 88%", "SIGN:hypoxemia", "exam")[0].polarity == "present"
    assert by("SpO2 97% (실내 공기)", "SIGN:hypoxemia", "exam")[0].polarity == "absent"
    assert by("포화도 90%", "SIGN:hypoxemia", "exam")[0].polarity == "present"


def test_08_family_history_heading_without_particle():
    for t in ("가족력: 고혈압, 당뇨", "가족력 - 고혈압", "가족력 고혈압"):
        fs = [f for f in parse(t, "patient", {"key": "과거력"}) if f.concept == "HX:hypertension"]
        assert fs and fs[0].subject == "family", t
    # the heading ends at the sentence; the patient's own history after it is the patient's
    fs = parse("가족력: 고혈압. 본인은 당뇨가 있어요.")
    assert {(f.concept, f.subject) for f in fs} >= {("HX:hypertension", "family"), ("HX:diabetes", "patient")}
    assert parse("과거력: 고혈압")[0].subject == "patient"


def test_09_proxy_speaker_names_the_patient():
    f = parse("(남편) 오늘 아침부터 아내가 헛소리를 해요.")[0]
    assert (f.concept, f.subject) == ("SYM:altered_mental_status", "patient")
    assert parse("(보호자) 아이가 열이 나요.")[0].subject == "patient"
    # a third person other than the proxy's own relation stays other / family
    f = [f for f in parse("(남편) 아내가 열이 나요. 저는 괜찮아요. 시어머니도 열이 났어요.") if f.concept == "SYM:fever"]
    assert [x.subject for x in f] == ["patient", "family"]
    # without a proxy tag the old reading holds
    assert parse("아내가 헛소리를 해요.")[0].subject == "other"


def test_10_double_negation_idioms_are_present():
    assert ("SYM:pain", "present") in kp("아프지 않은 곳이 없어요.")
    assert ("SYM:pain", "present") in kp("안 아픈 데가 없어요.")
    assert ("SYM:pain", "absent") in kp("아프지 않아요.")


def test_11_dedupe_keeps_distinct_mentions():
    fs = by("승모근 압통 있음, 측두동맥 압통 없음", "SIGN:tenderness", "exam")
    assert sorted(f.polarity for f in fs) == ["absent", "present"]
    fs = by("급성 뇌출혈, 급성 뇌경색 소견 없음.", "HX:stroke", "test")
    assert fs and all(f.polarity == "absent" for f in fs)
    spans = {s for f in fs for s in F.spans_of(f)}
    assert {"뇌출혈", "뇌경색"} <= {F.normalize("급성 뇌출혈, 급성 뇌경색 소견 없음.")[a:b] for a, b in spans}
    vals = sorted(f.value for f in by("혈압 150/90 mmHg, 재측정 혈압 90/60 mmHg", "SIGN:elevated_bp", "exam"))
    assert vals == [90.0, 150.0]


def test_12_lists_with_or_and_adverbial_negation():
    got = kp("S3, S4 또는 심잡음은 청진되지 않음", "exam")
    assert {("SIGN:s3", "absent"), ("SIGN:murmur", "absent")} <= got and not any(p == "present" for _c, p in got)
    s3 = by("S3, S4 또는 심잡음은 청진되지 않음", "SIGN:s3", "exam")
    t = F.normalize("S3, S4 또는 심잡음은 청진되지 않음")
    assert {"s3", "s4"} <= {t[a:b] for f in s3 for a, b in F.spans_of(f)}
    assert kp("수포음 및 천명음 없이 양호한 호흡음", "exam") >= {("SIGN:crackles", "absent"), ("SIGN:wheeze", "absent")}


def test_13_noun_ending_in_go_is_not_a_clause_end():
    assert kp("자살 사고 없음", "claim") == {("SYM:suicidal_ideation", "absent")}
    assert ("SYM:suicidal_ideation", "absent") in kp("자살 사고나 자해 생각은 없어요.")
    assert ("HX:trauma", "absent") in kp("교통사고 없음", "claim")
    assert kp("열이 나고 기침은 없어요") == {("SYM:fever", "present"), ("SYM:cough", "absent")}


def test_14_when_clause_with_a_change_verb_is_a_condition():
    assert ("SYM:cough", "present") not in kp("기침할 때 심해지지도 않아요.")
    assert ("SYM:cough", "present") not in kp("기침할 때 더 아프진 않아요.")
    assert ("SYM:cough", "present") in kp("기침할 때 가래는 안 나와요.")  # the presupposition rule still holds


def test_15_pediatric_vital_sign_ranges():
    got = kp("맥박 150회/분, 호흡수 40회/분", "exam", {"age_years": 0.25})
    assert {("SIGN:tachycardia", "absent"), ("SIGN:tachypnea", "absent")} <= got
    # the age can come from the text itself (the initial information)
    got = kp("생후 3개월 남아. 맥박 150회/분, 호흡수 40회/분", "exam")
    assert {("SIGN:tachycardia", "absent"), ("SIGN:tachypnea", "absent")} <= got
    assert ("SIGN:tachycardia", "present") in kp("맥박 190회/분", "exam", {"age_years": 0.25})
    assert ("SIGN:bradycardia", "present") in kp("맥박 80회/분", "exam", {"age_years": 0.05})
    assert ("SIGN:tachycardia", "present") in kp("맥박 130회/분", "exam", {"age_years": 8})
    assert ("SIGN:tachycardia", "present") in kp("맥박 130회/분", "exam")  # adults unchanged
    assert F.age_from_text("7세 여아. 주호소: 발열") == 7.0
    assert abs(F.age_from_text("생후 10일 된 남아") - 10 / 365.25) < 1e-6


# --- lexicon ----------------------------------------------------------------------------------------------------
def test_lexicon_somnolence_word_giemyeon_is_altered_consciousness():
    assert {c for c, _ in kp("의식 수준 기면 상태", "exam")} == {"SYM:altered_mental_status"}
    assert ("SYM:somnolence", "present") in kp("하루 종일 졸려요.")


def test_lexicon_past_pain_is_not_past_medical_history():
    assert "HX:pmh" not in {c for c, _ in kp("배가 아팠던 적은 없어요.")}
    assert "HX:pmh" not in {c for c, _ in kp("머리가 아픈 적이 있어요.")}
    assert "HX:pmh" not in {c for c, _ in kp("이렇게 아팠던 적은 없어요.")}  # about this pain, not past illness
    assert "HX:pmh" not in {c for c, _ in kp("전에도 이렇게 아팠던 적이 있어요.")}
    assert ("HX:pmh", "absent") in kp("특별히 아팠던 적은 없어요.")
    assert ("HX:pmh", "absent") in kp("지병은 없어요.")


def test_lexicon_cold_feeling_is_not_the_whole_respiratory_group():
    cs = {c for c, _ in kp("감기 기운이 있어요.")}
    assert "GRP:respiratory" not in cs and "SYM:uri_symptoms" in cs
    assert "GRP:respiratory" in LEXICON.ancestors("SYM:uri_symptoms")
    assert not LEXICON.concept("SYM:uri_symptoms").group
    assert ("GRP:respiratory", "absent") in kp("호흡기 증상은 없어요.")


def test_lexicon_kernig_and_brudzinski_are_separate():
    got = kp("Kernig 징후 음성, Brudzinski 징후 양성", "exam")
    assert {("SIGN:kernig", "absent"), ("SIGN:brudzinski", "present")} <= got


def test_lexicon_self_harm_is_not_suicidal_ideation():
    cs = {c for c, _ in kp("자해한 적이 있어요.")}
    assert "SYM:self_harm" in cs and "SYM:suicidal_ideation" not in cs
    assert ("SYM:suicidal_ideation", "present") in kp("죽고 싶다는 생각이 들어요.")


def test_lexicon_fracture_and_dislocation_are_not_trauma_history():
    cs = {c for c, _ in kp("작년에 손목 골절이 있었어요.")}
    assert "SIGN:fracture" in cs and "HX:trauma" not in cs
    assert ("SIGN:fracture", "absent") in kp("골절 소견 없음.", "test")
    assert ("SIGN:dislocation", "present") in kp("어깨 탈구 소견", "exam")
    assert ("HX:trauma", "present") in kp("넘어져서 다쳤어요.")


# --- public API ---------------------------------------------------------------------------------------------------
def test_public_helper_names_and_private_aliases():
    pairs = {"supports": "_supports", "first_cue": "_first_cue", "CUE": "_CUE", "IDIOM_NOT_NEG": "_IDIOM_NOT_NEG",
             "PRE_NEG_EN": "_PRE_NEG_EN", "PRE_NEG_KO": "_PRE_NEG_KO", "BARE_FILLER": "_BARE_FILLER", "SITE": "_SITE",
             "LAT": "_LAT", "LAB_RX": "_LAB_RX", "ref_direction": "_ref_direction", "sentences": "_sentences",
             "clauses": "_clauses", "UNAVAILABLE": "_UNAVAILABLE"}
    for pub, priv in pairs.items():
        assert getattr(F, pub) is getattr(F, priv), pub
        assert pub in F.__all__


def test_gestational_weeks_are_not_the_patients_age():
    from doctor_agent.nlp.findings import age_from_text

    assert age_from_text("22세 임신 30주 여성") == 22
    assert age_from_text("재태 34주 여성") is None
    assert abs(age_from_text("생후 30주 된 영아") - 30 * 7 / 365.25) < 1e-6
    assert abs(age_from_text("3주 된 신생아") - 3 * 7 / 365.25) < 1e-6
    assert age_from_text("45세 남성. 3주 전부터 기침") == 45


def test_triage_reported_false_positives():
    from doctor_agent.nlp.findings import parse

    def ids(text):
        return {f.concept for f in parse(text)}

    assert "SYM:altered_mental_status" not in ids("치료에 반응이 없어서 왔어요")
    assert "SYM:altered_mental_status" in ids("불러도 반응이 없어요")
    assert "SYM:hematemesis" not in ids("피로감과 구토가 있어요")
    assert "SYM:hematemesis" in ids("피를 토했어요")
    assert "SYM:urticaria" not in ids("피부 팽진도 감소")
    assert "SYM:urticaria" in ids("팽진이 올라왔어요")
