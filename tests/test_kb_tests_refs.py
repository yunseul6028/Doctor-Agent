"""Regression tests for kb_tests.detect(): printed reference ranges vs unit-scaled thresholds, cut-off findings, and
echo keyword patterns that must not reach across words ("우심실 확장기 허탈", "승모판 역류 및 대동맥판 협착")."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.knowledge import kb_tests  # noqa: E402


def pol(text: str, fid: str, context: int = 1):
    return kb_tests.detect(text, context).get(fid, (None,))[0]


# --- 1. value vs printed reference range, in the printed unit ----------------------------------------------------
@pytest.mark.parametrize("text,fid,want", [
    # the bug: 750 ng/mL was scaled to 0.75 (mg/L threshold unit) and compared with an unscaled "<500"
    ("D-dimer 750 ng/mL (정상 <500)", "ddimer_high", 1),
    ("D-dimer 750 ng/mL (정상: <500 ng/mL)", "ddimer_high", 1),
    ("WBC 10,500/μL, D-dimer 750 ng/mL (정상: <500 ng/mL)", "ddimer_high", 1),   # data/cases_aug cqa_521
    ("D-dimer 750 ng/mL (정상 ≤500)", "ddimer_high", 1),
    ("D-dimer 750 ng/mL (참고치 0-500)", "ddimer_high", 1),
    ("D-dimer: 750 ng/mL FEU (참고치 <500 ng/mL FEU)", "ddimer_high", 1),
    ("D-dimer 750 ng/mL (reference <500)", "ddimer_high", 1),
    ("D-dimer 750 ng/mL (<500)", "ddimer_high", 1),                  # bare range right after the value
    ("D-dimer 750 ng/mL (정상 500 미만)", "ddimer_high", 1),
    ("D-dimer 450 ng/mL (정상 <500)", "ddimer_high", -1),
    ("D-dimer 0.75 mg/L (정상 <0.5)", "ddimer_high", 1),
    ("D-dimer 0.3 (정상 <0.5 μg/mL)", "ddimer_high", -1),            # no unit on the value: the range's unit
    ("D-dimer 300 (정상 <500 ng/mL)", "ddimer_high", -1),
    ("D-dimer 0.75 mg/L FEU (정상 <500 ng/mL)", "ddimer_high", 1),   # two units printed: both converted
    ("D-dimer 0.4 mg/L (정상 <500 ng/mL)", "ddimer_high", -1),
    ("D-dimer 750 ng/mL", "ddimer_high", 1),                         # no range: default threshold, unit-scaled
    ("D-dimer 300 ng/mL", "ddimer_high", -1),
    # troponin in ng/L / pg/mL against a range printed in the same unit
    ("고감도 트로포닌 I 1,240 ng/L (참고치 <34)", "troponin_high", 1),  # data/sample_cases synthetic_002
    ("hs-트로포닌 T 52 ng/L (정상 <14)", "troponin_high", 1),
    ("hs-cTnI 30 ng/L (정상 <26 ng/L)", "troponin_high", 1),
    ("hs-cTnI 20 ng/L (정상 <26 ng/L)", "troponin_high", -1),
    ("hs-트로포닌 I 45 pg/mL (참고치 <34)", "troponin_high", 1),
    ("트로포닌 T 0.01 ng/mL (정상 <0.014)", "troponin_high", -1),
    ("Troponin-I < 0.01 ng/mL", "troponin_high", -1),
    ("트로포닌 (<0.01 ng/mL)", "troponin_high", -1),                  # a bare "(<x)" with no value before it is the value
    # HbA1c in IFCC mmol/mol
    ("HbA1c 53 mmol/mol (정상 <42)", "hba1c_high", 1),
    ("HbA1c 38 mmol/mol (정상 <42)", "hba1c_high", -1),
    ("HbA1c 53 mmol/mol", "hba1c_high", 1),
    # alternative SI units
    ("세룰로플라스민 120 mg/L", "ceruloplasmin_low", 1),
    ("세룰로플라스민 0.12 g/L", "ceruloplasmin_low", 1),
    ("세룰로플라스민 300 mg/L", "ceruloplasmin_low", -1),
    ("칼슘 3.1 mmol/L", "calcium_high", 1),
    ("코르티솔 50 nmol/L", "cortisol_low", 1),
    ("코르티솔 500 nmol/L", "cortisol_low", -1),
    ("비타민 B12 100 pmol/L", "b12_low", 1),
    ("피브리노겐 1.0 g/L", "fibrinogen_low", 1),
    ("피브리노겐 3.0 g/L", "fibrinogen_low", -1),
    ("젖산 54 mg/dL", "lactate_high", 1),
    ("요산 540 μmol/L", "uric_high", 1),
    ("요산 300 umol/L", "uric_high", -1),
    # qualitative-with-titre analytes read against the printed range too
    ("anti-CCP 15 U/mL (정상 <5)", "ccp_pos", 1),
    ("anti-CCP 3 U/mL (정상 <5)", "ccp_pos", -1),
])
def test_value_against_printed_range(text, fid, want):
    assert pol(text, fid) == want


@pytest.mark.parametrize("text,fid,want", [
    ("D-dimer 750 ng/mL (정상 범위 초과)", "ddimer_high", 1),       # "(정상 ...)" that says above the range
    ("D-dimer 상승 (정상 범위 초과)", "ddimer_high", 1),
    ("페리틴 5 ng/mL (정상 범위 미만)", "ferritin_low", 1),
    ("D-dimer 400 ng/mL (정상 범위 내)", "ddimer_high", -1),
    ("D-dimer 400 ng/mL (정상)", "ddimer_high", -1),
    ("리파아제 32 U/L (정상)", "lipase_high", -1),
])
def test_worded_reference(text, fid, want):
    assert pol(text, fid) == want


@pytest.mark.parametrize("text,fid,want", [
    # cut-off findings: a value above the reference range is not above the cut-off
    ("AST 150 U/L (정상 <40)", "ast_alt_very_high", -1),
    ("AST 1,850 U/L (정상 <40)", "ast_alt_very_high", 1),
    ("ESR 38 mm/hr (정상 <20 mm/hr)", "esr_very_high", -1),
    ("ESR 88 mm/hr (정상 <20 mm/hr)", "esr_very_high", 1),
    ("페리틴 600 ng/mL (정상 30-400)", "ferritin_very_high", -1),
    ("페리틴 3,500 ng/mL (정상 30-400)", "ferritin_very_high", 1),
    ("혈당 180 mg/dL (정상 70-100)", "glucose_very_high", -1),
    ("HbA1c 6.2% (정상 4.0-6.0)", "hba1c_high", -1),
    ("HbA1c 7.2% (정상 4.0-6.0)", "hba1c_high", 1),
    ("ADAMTS13 활성도 30% (정상 40-130)", "adamts13_low", -1),
    ("ADAMTS13 활성도 3% (정상 40-130)", "adamts13_low", 1),
    ("AST 150 U/L (정상 범위 초과)", "ast_alt_very_high", -1),     # worded "above range" is not "> 1000": the value decides
    # relative-to-ULN findings keep their multiplier
    ("리파아제 500 U/L (정상 13-60)", "lipase_high", 1),
    ("리파아제 150 U/L (정상 13-60)", "lipase_high", -1),
])
def test_cutoff_findings(text, fid, want):
    assert pol(text, fid) == want


# --- 2. echo: "우심실 확장기 허탈" is diastolic collapse (tamponade), not RV dilatation --------------------------
@pytest.mark.parametrize("text,rv,tamp", [
    ("우심실 확장기 허탈 소견", None, 1),
    ("우심방 및 우심실 확장기 허탈 동반", None, 1),
    ("우심실 확장기 허탈 관찰됨, 심낭 삼출 다량", None, 1),
    ("우심실 확장 소견", 1, None),
    ("우심실 확장 및 기능 저하", 1, None),
    ("우심실 기능 저하", 1, None),
    ("우심실 확장기말 직경 증가", 1, None),
    ("우심실 확장 없음", -1, None),
    ("RV dilatation with McConnell sign", 1, None),
    ("right ventricle dilated", 1, None),
    ("right ventricular dilatation", 1, None),
    ("우심실 크기 정상이며 하대정맥 허탈", None, None),        # IVC collapse is not tamponade
])
def test_rv_strain_vs_diastolic_collapse(text, rv, tamp):
    got = kb_tests.detect(text)
    assert got.get("echo_rv_strain", (None,))[0] == rv
    assert got.get("echo_tamponade", (None,))[0] == tamp


# --- 3. valve patterns stay on their own valve ------------------------------------------------------------------
@pytest.mark.parametrize("text,ms,as_", [
    ("승모판 역류 및 대동맥판막 협착", None, 1),
    ("중증 승모판 역류와 중등도 대동맥판막 협착(평균 압력차: 24 mm Hg)", None, 1),
    ("승모판 역류 및 삼첨판막 협착", None, None),
    ("대동맥판 역류 및 승모판막 협착", 1, None),
    ("대동맥 판막 경화, 승모판 역류 및 삼첨판 협착", None, None),
    ("대동맥판막 석회화 및 승모판 협착", 1, None),
    ("승모판 정상, 대동맥판 협착", None, 1),
    ("중등도 승모판 협착", 1, None),
    ("승모판막 개구 면적 1.0 cm2", 1, None),
    ("승모판 역류 및 협착", 1, None),                                  # both lesions on the mitral valve
    ("승모판 협착 없음", -1, None),
    ("대동맥판막 협착", None, 1),
    ("aortic stenosis", None, 1),
    ("mitral regurgitation and tricuspid stenosis", None, None),
])
def test_valve_patterns_do_not_cross_valves(text, ms, as_):
    got = kb_tests.detect(text)
    assert got.get("echo_ms", (None,))[0] == ms
    assert got.get("echo_as", (None,))[0] == as_
