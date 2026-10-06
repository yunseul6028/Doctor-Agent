"""Regression tests for the reference-range / look-alike reading fixes of 2026-09-30 (docs/nlp.md, "Fixes 2026-09-30").

One reader (knowledge/refrange.py) now serves nlp/findings.py and knowledge/kb_tests.py; each test failed before."""
import pytest

from doctor_agent.agent.result_interpreter import interpret
from doctor_agent.knowledge import kb_tests, refrange
from doctor_agent.nlp import findings as F
from doctor_agent.nlp import parse


def kp(text, source="test"):
    return {(f.concept, f.polarity) for f in parse(text, source)}


def ri(test, text):
    return {(i.concept, i.polarity) for i in interpret(test, text).items}


# --- the shared reader ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body,lo,hi,hi_strict,lo_strict", [
    ("정상 500 미만", None, 500, True, False),
    ("참고치 40 이하", None, 40, False, False),
    ("500미만", None, 500, True, False),
    ("정상 <500", None, 500, True, False),
    ("≤ 40", None, 40, False, False),
    ("less than 40", None, 40, True, False),
    ("up to 40", None, 40, False, False),
    ("정상 상한 60", None, 60, False, False),
    ("ULN 500", None, 500, False, False),
    ("ast 66 u/l, 상한치: 45 u/l", None, 45, False, False),
    ("정상 12 이상", 12, None, False, False),
    ("60 초과", 60, None, False, True),
    ("above 12", 12, None, False, True),
    ("≥12", 12, None, False, False),
    ("정상 하한 12", 12, None, False, False),
    ("LLN 12", 12, None, False, False),
    ("정상 0.4-4.0", 0.4, 4.0, False, False),
    ("정상 150,000-450,000", 150000, 450000, False, False),
])
def test_reader_limits(body, lo, hi, hi_strict, lo_strict):
    r = refrange.read(body)
    assert (r.lo, r.hi, r.hi_strict, r.lo_strict) == (lo, hi, hi_strict, lo_strict)


@pytest.mark.parametrize("body,says", [
    ("경미한 상승", "high"), ("mildly elevated", "high"), ("감소", "low"), ("H", "high"), ("정상", "normal"),
    ("정상 범위 초과", "high"), ("상승 없음", "normal"),
    ("정상 상한의 5배", ""),      # a multiple of the limit is not a normal statement
    ("3백분위수 미만", ""),       # a number that is not a limit: the words say nothing
    ("1394 nmol/l", ""),          # a value in another unit, not a flag ("l" is not "L = low")
])
def test_reader_words(body, says):
    r = refrange.read(body)
    assert not r.bounded and r.says == says


def test_reader_bare_limit_side_comes_from_the_analyte():
    r = refrange.read("정상치 500")
    assert r.limit == 500 and not r.bounded
    assert refrange.direction(r, 750, "high") == "high"
    assert refrange.direction(r, 300, "high") == "normal"
    assert refrange.direction(r, 750) == ""  # side unknown: no reading


# --- 1. nlp/findings: Korean / English comparator words ---------------------------------------------------------
def test_nlp_korean_upper_limit_words():
    assert ("LAB:ast_alt_high", "present") in kp("AST 150 U/L (정상 40 미만)")
    assert ("LAB:ast_alt_high", "present") in kp("AST 150 (참고치 40 이하)")
    assert ("LAB:ast_alt_high", "absent") in kp("AST 30 (정상 40 미만)")


def test_nlp_korean_lower_limit_words():
    assert ("LAB:hb_low", "present") in kp("Hb 9.1 g/dL (정상 12 이상)")
    assert ("LAB:hb_low", "absent") in kp("Hb 13.5 (정상 12 이상)")


def test_nlp_strict_and_inclusive_limits():
    assert F.ref_direction("(정상 40 미만)", 40)[0] == "high"
    assert F.ref_direction("(정상 40 이하)", 40)[0] == "normal"
    assert F.ref_direction("(정상 범위 초과)", 7)[0] == "high"  # was "normal" (the word 정상 read first)


def test_nlp_value_in_another_unit_is_not_a_flag():
    # "(1394 nmol/L)" restates the value in SI units; the "l" of the unit was read as an "L" (low) flag
    assert ("LAB:crp_high", "present") in kp("CRP 11.2 mg/dL (1067 nmol/L), ESR 64 mm/hr")
    assert ("LAB:cr_high", "present") in kp("Cr 1.86 mg/dL (164 μmol/L)")


def test_nlp_range_in_another_unit_falls_back_to_the_default_threshold():
    assert ("LAB:wbc_low", "present") in kp("WBC 3,200 (정상 4.0-10.0)")


def test_nlp_kb_value_direction_follows_the_finding_side():
    got = [f for f in parse("ferritin 12 ng/mL", "test") if f.concept == "LAB:ferritin_low"]
    assert got and got[0].polarity == "present" and got[0].direction == "low"


# --- 2. kb_tests: bare limits, unit doubt, direction words -------------------------------------------------------
def test_kb_bare_reference_number_is_an_upper_limit_for_a_high_finding():
    assert kb_tests.detect("D-dimer 750 ng/ml (정상치 500)")["ddimer_high"][0] == 1
    assert kb_tests.detect("lipase 400 u/l (정상 상한 60)")["lipase_high"][0] == 1
    assert kb_tests.detect("D-dimer 750 ng/mL (ULN 500)")["ddimer_high"][0] == 1
    assert kb_tests.detect("D-dimer 300 ng/mL (upper limit 500)")["ddimer_high"][0] == -1


def test_kb_unitless_value_against_a_unit_range_is_not_called_normal():
    # 1.2 is μg/mL (= 1200 ng/mL): comparing it with 500 as printed said "normal"
    assert "ddimer_high" not in kb_tests.detect("d-dimer 1.2 (정상 <500 ng/ml)")
    # the same unit shape with agreeing readings still reads
    assert kb_tests.detect("D-dimer 750 ng/mL (정상 <500)")["ddimer_high"][0] == 1


def test_kb_direction_word_parenthesis():
    assert kb_tests.detect("CEA 6.5 ng/mL (경미한 상승)")["cea_high"] == (1, True)
    assert kb_tests.detect("CEA 6.5 ng/mL (mildly elevated)")["cea_high"][0] == 1
    assert ("LAB:cea_high", "present") in ri("CEA", "CEA 6.5 ng/mL (mildly elevated)")


def test_kb_limit_parenthesis_needs_to_be_only_a_limit():
    # "(흉수/혈청 단백 비율 > 0.5)" is a criterion, not the reference range of the value before it
    text = "단백질 3.8 g/dL (흉수/혈청 단백 비율 > 0.5)"
    seg, refs = kb_tests._ref_ranges(text.lower())
    assert not refs and "흉수" in seg


# --- 3. look-alikes ---------------------------------------------------------------------------------------------
def test_ag_alone_is_an_anion_gap_only_among_electrolytes():
    assert "anion_gap_high" not in kb_tests.detect("폰빌레브란트 인자 항원(vWF:Ag) 95%, vWF:RCo 90% (정상).")
    assert kb_tests.detect("Na 135, Cl 100, HCO3 10, AG 25")["anion_gap_high"][0] == 1
    assert kb_tests.detect("음이온차 25")["anion_gap_high"][0] == 1
    assert ("LAB:anion_gap_high", "present") in ri("전해질", "Na 135, Cl 100, HCO3 10, AG 25")


def test_bare_saturation_is_transferrin_only_in_an_iron_panel():
    cath = "폐동맥 산소포화도가 65%에서 45%로 떨어짐"
    assert "tsat_high" not in kb_tests.detect(cath)
    assert ("LAB:tsat_high", "present") not in kp(cath)
    iron = "철 200, TIBC 250, 포화도 80%"
    assert kb_tests.detect(iron)["tsat_high"][0] == 1
    assert ("SIGN:hypoxemia", "present") not in kp(iron)  # nor is it an oxygen saturation


def test_perforation_is_free_air_only_in_the_gut():
    tee = "승모판 앞쪽 판엽 천공과 함께 중등도 승모판 역류가 보임."
    got = ri("경식도 심초음파 TEE", tee)
    assert not any(c == "IMG:free_air" for c, _ in got)
    assert ("", "present") in got  # still shown as an unmapped abnormal finding
    assert not any(c == "IMG:free_air" for c, _ in ri("이경검사", "양측 고막은 투명하며 천공이나 함몰 없음."))
    assert ("IMG:free_air", "present") in ri("복부 CT", "십이지장 궤양 천공 소견")
    assert ("IMG:free_air", "present") in ri("상부위장관 내시경", "십이지장 궤양 천공 소견")
    assert any(c == "IMG:free_air" for c, _ in ri("복부 CT", "위 천공 의심, 복강 내 유리 공기"))
