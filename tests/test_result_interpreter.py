"""Result interpreter (agent/result_interpreter.py): report text -> structured items, no LLM."""
import json

from doctor_agent.agent import prompts
from doctor_agent.agent.result_interpreter import interpret, llm_reasons, needs_llm, render_for_prompt
from doctor_agent.agent.result_interpreter import kind_of_test
from doctor_agent.nlp.lexicon import LEXICON


def pol(interp, concept):
    """Polarities read for one concept."""
    return {i.polarity for i in interp.items if i.concept == concept}


def item(interp, concept):
    return next(i for i in interp.items if i.concept == concept)


# ---------------------------------------------------------------- test kind
def test_kind_of_test():
    assert kind_of_test("흉부 X선") == "imaging"
    assert kind_of_test("Chest CT angiography") == "imaging"
    assert kind_of_test("심초음파") == "imaging"
    assert kind_of_test("심전도") == "ecg"
    assert kind_of_test("ECG") == "ecg"
    assert kind_of_test("혈액검사") == "lab"
    assert kind_of_test("신체진찰") == "exam"
    assert kind_of_test("활력징후") == "exam"
    assert kind_of_test("", "WBC 15,000/μL") == "lab"


# ---------------------------------------------------------------- imaging polarity
def test_korean_present_and_negated():
    it = interpret("흉부 X선", "우하엽 경화 소견. 기흉 없음.")
    c = item(it, "IMG:cxr_consolidation")
    assert c.polarity == "present" and c.laterality == "right" and c.kind == "imaging"
    assert pol(it, "IMG:cxr_ptx") == {"absent"}


def test_bare_organ_word_uses_the_test_name():
    assert pol(interpret("흉부 X선", "경화 소견 없음."), "IMG:cxr_consolidation") == {"absent"}
    assert pol(interpret("뇌 CT", "출혈 없음. 정상."), "IMG:ct_ich") == {"absent"}
    # the same word on another organ is not a lung finding
    assert "IMG:cxr_consolidation" not in interpret("복부 CT", "충수 주위 지방층 침윤").concepts(None)


def test_english_negation_lists():
    it = interpret("Chest X-ray", "Findings: No consolidation, effusion, or pneumothorax.")
    for c in ("IMG:cxr_consolidation", "IMG:pleural_effusion", "IMG:cxr_ptx"):
        assert pol(it, c) == {"absent"}, c
    it = interpret("Abdominal ultrasound", "Cholelithiasis without sonographic evidence of acute cholecystitis.")
    assert pol(it, "IMG:gallstone") == {"present"}
    assert pol(it, "IMG:us_cholecystitis") == {"absent"}


def test_hedges_make_uncertain():
    assert pol(interpret("흉부 X선", "좌측 소량의 흉수. 폐렴 가능성 있음."), "IMG:cxr_consolidation") == {"uncertain"}
    assert pol(interpret("뇌 CT", "뇌출혈 의심, MRI 추가 검사 권고."), "IMG:ct_ich") == {"uncertain"}
    assert pol(interpret("CT head", "Hypodensity in the right MCA territory concerning for acute infarct."),
               "IMG:stroke_imaging") == {"uncertain"}
    assert pol(interpret("심전도", "r/o 급성 심근경색, V1-V4 ST 분절 상승 의심."), "ECG:ecg_stemi") == {"uncertain"}


def test_cannot_exclude_is_uncertain_and_excluded_is_absent():
    assert pol(interpret("흉부 X선", "폐렴을 배제할 수 없음."), "IMG:cxr_consolidation") == {"uncertain"}
    assert pol(interpret("CT", "Pulmonary embolism cannot be excluded."), "IMG:ctpa_pe") == {"uncertain"}
    assert pol(interpret("CT 폐동맥 조영술", "폐동맥 내 충만 결손 없음. 폐색전증 배제됨."), "IMG:ctpa_pe") == {"absent"}


def test_normal_study():
    it = interpret("흉부 X선", "특이 소견 없음.")
    assert it.normal and pol(it, "IMG:normal_study") == {"present"}
    it = interpret("Chest X-ray", "Findings: Heart size normal. Lungs clear. Impression: Normal chest radiograph.")
    assert it.normal and pol(it, "IMG:cxr_hf") == {"absent"}
    # "X 외 특이 소견 없음" is not a normal study
    it = interpret("흉부 X선", "경미한 심비대 외 특이 소견 없음")
    assert not it.normal and pol(it, "IMG:cxr_hf") == {"present"}
    # an organ that is normal is not a normal study
    assert not interpret("복부 CT", "충수는 정상. 급성 충수염 소견 없음.").normal
    assert interpret("심전도", "정상 동리듬, 특이 소견 없음.").concepts() == {"ECG:normal_ecg"}


def test_sections_and_recommendations_are_ignored():
    it = interpret("Chest X-ray", "Indication: r/o pneumonia. Findings: Lungs are clear. Impression: No acute "
                                  "cardiopulmonary process.")
    assert it.normal and "IMG:cxr_consolidation" not in it.concepts("uncertain")
    assert any("indication" in x for x in it.ignored)
    it = interpret("흉부 CT", "우하엽 5mm 결절, 이전 CT와 비교하여 변화 없음. 12개월 후 추적 CT 권고.")
    n = item(it, "IMG:lung_nodule")
    assert n.polarity == "present" and n.comparison == "stable" and n.laterality == "right"
    assert any("권고" in x for x in it.ignored)


def test_comparison_phrases():
    it = interpret("Chest X-ray", "No change in the right pleural effusion.")
    e = item(it, "IMG:pleural_effusion")
    assert e.polarity == "present" and e.comparison == "stable"
    it = interpret("흉부 X선", "이전 검사와 비교하여 우하엽 경화는 소실됨.")
    c = item(it, "IMG:cxr_consolidation")
    assert c.polarity == "absent" and c.comparison == "resolved"
    it = interpret("CXR", "Small left pleural effusion, improved compared to prior.")
    assert item(it, "IMG:pleural_effusion").comparison == "improved"
    # "ST 분절 변화 없음" (no ST change) is not a comparison
    it = interpret("심전도", "ST 분절 변화 없음. QTc 520ms로 연장.")
    assert pol(it, "ECG:ecg_st_depression") == {"absent"} and pol(it, "ECG:ecg_long_qt") == {"present"}


def test_laterality_and_site():
    it = interpret("Brain CT", "Left subdural hematoma measuring 1.2 cm with 7 mm midline shift.")
    assert item(it, "IMG:subdural_hematoma").laterality == "left"
    assert pol(it, "IMG:mass_effect") == {"present"}
    it = interpret("흉부 CT", "양측 폐에 간유리 음영.")
    assert item(it, "IMG:ggo").laterality == "bilateral"
    it = interpret("Lower extremity venous duplex", "No evidence of deep venous thrombosis in the right lower extremity.")
    d = item(it, "IMG:doppler_dvt")
    assert d.polarity == "absent" and d.laterality == "right"
    # a Korean side word names the next list item, not the one before it
    it = interpret("복부 초음파", "복수 및 좌측 흉수 소견")
    assert item(it, "IMG:pleural_effusion").laterality == "left"
    assert all(i.laterality != "left" for i in it.items if i.concept != "IMG:pleural_effusion")


def test_organ_specific_words():
    it = interpret("하지 정맥 도플러", "좌측 대퇴정맥과 슬와정맥이 압박되지 않으며 내부에 혈전 관찰됨.")
    assert pol(it, "IMG:doppler_dvt") == {"present"}
    assert pol(interpret("고환 초음파", "좌측 고환 혈류 소실."), "IMG:testis_no_flow") == {"present"}
    assert pol(interpret("고환 초음파", "양측 고환 혈류 정상."), "IMG:testis_no_flow") == {"absent"}
    it = interpret("복부 초음파", "담낭 내 다수의 담석, 담낭벽 비후 5mm. 총담관 확장 없음.")
    assert pol(it, "IMG:gallstone") == {"present"} and pol(it, "IMG:us_cholecystitis") == {"present"}
    assert pol(it, "IMG:cbd_dilation") == {"absent"}
    # "확장기 허탈" (diastolic collapse) is not RV dilatation
    it = interpret("심초음파", "다량의 심낭 삼출과 우심실 확장기 허탈.")
    assert "IMG:echo_rv_strain" not in it.concepts(None) and pol(it, "IMG:echo_tamponade") == {"present"}
    # skin / auscultation words inside an imaging report are not patient findings
    it = interpret("흉부 X선", "양측 하엽에 반점상 경화, 폐야 깨끗한 부분 없음.")
    assert not any(i.concept.startswith(("SYM:", "QUAL:")) for i in it.items)


# ---------------------------------------------------------------- labs / exam / vitals
def test_lab_values_direction_and_critical():
    it = interpret("전해질", "Na 128 mmol/L (135-145), K 6.8 mmol/L")
    na, k = item(it, "LAB:na_low"), item(it, "LAB:k_high")
    assert na.polarity == "present" and na.direction == "low" and na.value == 128
    assert k.polarity == "present" and k.critical
    assert not na.critical  # 128 is low but above the critical limit
    g = item(interpret("혈액검사", "혈당 42 mg/dL"), "LAB:glucose_low")
    assert g.critical and g.kind == "lab"
    it = interpret("혈액검사", "WBC 15,000/μL, Hb 13.5 g/dL")
    assert pol(it, "LAB:wbc_high") == {"present"} and pol(it, "LAB:hb_low") == {"absent"}


def test_lab_value_above_printed_range():
    it = interpret("혈액검사", "D-dimer 750 ng/mL (정상: <500 ng/mL)")
    d = item(it, "LAB:ddimer_high")
    assert d.polarity == "present" and d.value == 750


def test_kb_lab_polarity_is_kb_tests_own():
    """kb_tests reads values against a printed range in its unit and knows absolute cut-offs; the interpreter only
    adds the value for display (review 2026-09-30: its own range re-read overrode kb_tests)."""
    # Korean "미만" range phrasing: above the range (was read ABSENT: a missed PE work-up)
    d = item(interpret("혈액검사", "D-dimer 750 ng/mL (정상 500 미만)"), "LAB:ddimer_high")
    assert d.polarity == "present" and d.value == 750 and d.unit == "ng/ml" and d.direction == "high"
    assert pol(interpret("혈액검사", "D-dimer 300 ng/mL (정상 500 미만)"), "LAB:ddimer_high") == {"absent"}
    # ESR above its reference range is not ESR > 50 (a cut-off finding); the plain ESR concept stays elevated
    it = interpret("혈액검사", "ESR 38 mm/hr (정상 <20)")
    assert pol(it, "LAB:esr_very_high") == {"absent"} and pol(it, "LAB:esr_high") == {"present"}
    assert item(it, "LAB:esr_very_high").value == 38
    it = interpret("염증수치", "CRP 4.2 mg/dL (정상 <0.5), ESR 48 mm/hr (정상 <15)")  # data/cases_clinicalqa_aug/cqa_388
    assert pol(it, "LAB:esr_very_high") == {"absent"}
    assert pol(interpret("혈액검사", "CRP 0.3 mg/dL (정상), ESR 24 mm/hr (약간 상승)"), "LAB:esr_very_high") == {"absent"}
    assert pol(interpret("혈액검사", "ESR 80 mm/hr (정상 20 미만)"), "LAB:esr_very_high") == {"present"}
    # AST 150 is above the range, not AST > 1000
    it = interpret("간기능", "AST 150 U/L (정상: <40 U/L)")
    assert pol(it, "LAB:ast_alt_very_high") == {"absent"} and pol(it, "LAB:ast_alt_high") == {"present"}
    assert pol(interpret("간기능", "AST 150 U/L (정상 40 미만)"), "LAB:ast_alt_very_high") == {"absent"}
    assert pol(interpret("간기능", "AST 150 U/L"), "LAB:ast_alt_very_high") == {"absent"}
    assert pol(interpret("간기능", "AST 1500 U/L (정상 40 미만)"), "LAB:ast_alt_very_high") == {"present"}
    # "(경미한 상승)" is not a reference keyword to kb_tests: kb_tests' own word-range rule is applied to it
    assert pol(interpret("종양표지자", "CEA 6.5 ng/mL (경미한 상승)."), "LAB:cea_high") == {"present"}
    # a low-side finding's direction is "low"
    f = item(interpret("혈액검사", "Hb 10.8 g/dL, MCV 75 fL, 페리틴 10 ng/mL"), "LAB:ferritin_low")
    assert f.polarity == "present" and f.direction == "low" and f.value == 10
    # a clause with a pending part keeps being skipped for kb_tests readings
    it = interpret("혈액배양 및 젖산", "혈액배양: 세균 배양 검사 진행 중 (Gram 음성 쌍구균 또는 Gram 양성 알균 가능성)이며, "
                                     "혈청 젖산은 3.8 mmol/L입니다.")
    assert "LAB:lactate_high" not in {i.concept for i in it.items}
    # no value is shown for ranges, titres, grades or a number of another name
    assert item(interpret("요검사", "백혈구 0-5 /HPF, 아질산염 음성"), "LAB:pyuria").value is None
    assert item(interpret("요검사", "단백뇨 3+, 요비중 1.010"), "LAB:proteinuria_nephrotic").value is None
    assert item(interpret("조직검사", "면역형광검사에서 기저막을 따라 IgG와 C3가 과립상으로 침착됨."), "LAB:igg_high").value is None


def test_pending_results_are_not_results():
    it = interpret("요검사", "성상: 대기 중, 아질산염: 대기 중, 백혈구 에스테라아제: 대기 중")
    assert not it.present() and it.pending
    assert "대기" in render_for_prompt(it)


def test_unavailable_is_not_normal():
    it = interpret("흉부 CT", "결과가 제공되지 않습니다.")
    assert it.unavailable and not it.normal and not it.items
    assert "정상으로 보지 말 것" in render_for_prompt(it)


def test_exam_and_vitals():
    it = interpret("신체진찰", "우하복부 압통 및 반발통 있음, 머피 징후 음성.")
    assert pol(it, "SIGN:rlq_tenderness") == {"present"} and pol(it, "SIGN:murphy_sign") == {"absent"}
    assert item(it, "SIGN:rlq_tenderness").kind == "exam"
    it = interpret("활력징후", "혈압 82/50 mmHg, 맥박 128회/분, 산소포화도 88%")
    assert item(it, "SIGN:hypotension").critical and item(it, "SIGN:hypoxemia").critical
    # children: heart rate limits are age-dependent (no adult critical HR flag for an infant)
    it = interpret("활력징후", "맥박 150회/분", {"age_years": 0.25})
    assert not any(i.critical for i in it.items)


# ---------------------------------------------------------------- critical results, disease links, rendering
def test_critical_imaging_and_supportive_links():
    it = interpret("흉부 X선", "우측 기흉, 약 30%.")
    p = item(it, "IMG:cxr_ptx")
    assert p.critical and p.polarity == "present"
    assert any(dx == "pneumothorax" for dx, _ko, _w in p.supports)
    line = render_for_prompt(it)
    assert "⚠" in line and "확진 아님" in line
    # absent findings carry no disease links
    assert not item(interpret("흉부 X선", "기흉 없음."), "IMG:cxr_ptx").supports


def test_render_for_prompt_length_and_order():
    it = interpret("흉부 X선", "우하엽 경화. 좌측 소량 흉수. 기흉 없음. 종격동 확장 없음. 폐렴 가능성.")
    line = render_for_prompt(it, max_chars=80)
    assert len(line) <= 80 and line.startswith("[흉부 X선]")
    full = render_for_prompt(it, max_chars=1000)
    assert full.index("있음:") < full.index("없음:")
    assert all(i.summary_ko for i in it.items)


def test_concept_ids_exist_in_the_lexicon():
    texts = [("흉부 X선", "우하엽 경화, 좌측 흉수, 무기폐, 심비대, 폐 결절"), ("뇌 CT", "경막하 혈종과 정중선 편위"),
             ("골반 초음파", "자궁 내 임신낭 없음, 우측 부속기 종괴, 골반 내 자유 액체"), ("심초음파", "국소 벽운동 저하, 승모판 역류")]
    for name, text in texts:
        for i in interpret(name, text).items:
            assert not i.concept or LEXICON.concept(i.concept), i.concept


# ---------------------------------------------------------------- needs_llm / robustness / prompt
def test_needs_llm():
    assert not needs_llm(interpret("흉부 X선", "우하엽 경화 소견. 기흉 없음."))
    serial = ("노출 3일 후 OCT: 바깥 망막층 고반사 병변. 노출 5일 후: 저반사 공간. 노출 19일 후: 부분 회복. "
              "노출 33일 후: 추가 회복.")
    it = interpret("OCT 영상", serial)
    assert needs_llm(it) and "serial_results" in llm_reasons(it)
    it = interpret("복부 X선", "십이지장 폐쇄를 시사하는 이중 기포 징후, 좌상복부 석회화 병변.")
    assert needs_llm(it) and "unmapped_findings" in llm_reasons(it)
    long_text = "우하엽 경화 소견. " * 60
    assert "long_text" in llm_reasons(interpret("흉부 X선", long_text))


def test_never_raises_and_is_serialisable():
    for name, text in [("", ""), (None, None), ("흉부 X선", "!!!"), ("x", "a" * 5000)]:
        it = interpret(name, text)
        json.dumps(it.as_dict(), ensure_ascii=False)
        render_for_prompt(it)


def test_no_state_between_calls():
    a = interpret("흉부 X선", "우측 기흉.").as_dict()
    interpret("뇌 CT", "지주막하 출혈.")
    assert interpret("흉부 X선", "우측 기흉.").as_dict() == a


def test_prompt_template():
    msgs = prompts.build_result_interpreter_messages("흉부 X선", "우하엽 경화", "[흉부 X선] 있음: 폐 경화·침윤(우측)")
    assert msgs[0]["content"] == prompts.RESULT_INTERPRETER_PROMPT
    assert "JSON" in prompts.RESULT_INTERPRETER_PROMPT and "우하엽 경화" in msgs[1]["content"]


def test_gold_sets_score_at_least_the_recorded_floor():
    """Offline gold sets (eval/offline/eval_result_interp.py): guard against regressions."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_ri", root / "eval" / "offline" / "eval_result_interp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, path in mod.GOLDS.items():
        res = mod.run(path)
        assert res["overall"]["f1"] >= 0.95, (name, res["overall"])
        assert res["normal_flag_accuracy"] >= 0.95, name
