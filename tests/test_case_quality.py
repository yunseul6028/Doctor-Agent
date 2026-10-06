"""Quality gate for the LLM-augmented evaluation cases (data/cases_aug). Rule-based, no LLM calls.

scripts/check_cases.py flags answer leaks, contradictions, implausible values and keyword-key problems; hard issues must
be zero after the recorded fixes. The detector tests below make sure the gate is not vacuous.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import check_cases as cc  # noqa: E402

pytestmark = pytest.mark.skipif(not cc.CASE_ROOT.exists(), reason="data/cases_aug not present")


@pytest.fixture(scope="module")
def results():
    return cc.check_all()


def _hard(results, types=None):
    return [(cid, i["type"], i.get("where", ""), i["detail"]) for cid, iss in results.items() for i in iss
            if i["severity"] == "hard" and (types is None or i["type"] in types)]


def test_all_cases_checked(results):
    assert len(results) >= 100  # 111 public cases (clinicalqa + sample); was >= 250 on the 267-case private set


def test_no_answer_leaks(results):
    assert _hard(results, {"leak", "hidden_key_missing", "meta_reference"}) == []


def test_no_hard_contradictions(results):
    assert _hard(results, {"vital_conflict", "lab_conflict", "fever_conflict", "demographic_test"}) == []


def test_no_hard_issues_at_all(results):
    assert _hard(results) == []


def test_fixes_already_applied():
    """--fix is idempotent: on the committed files there is nothing left to edit."""
    for p in cc.case_paths():
        case = json.loads(p.read_text(encoding="utf-8"))
        assert cc.apply_fixes(case, cc.case_id(p)) == [], cc.case_id(p)


def test_hidden_keys_cover_non_visible_fields():
    for p in cc.case_paths():
        case = json.loads(p.read_text(encoding="utf-8"))
        extra = set(case) - set(cc.VISIBLE) - set(cc.HIDDEN_KEYS)
        assert not extra, (cc.case_id(p), extra)


# ───────────────────────────── detectors fire on injected problems ─────────────────────────────
BASE = {
    "initial": "45세 남성. 주호소: 2일 전부터 시작된 발열과 기침",
    "history": {"열|발열": "이틀 전부터 열이 났어요."},
    "exam": {"활력징후|vital": "체온 38.6℃, 혈압 124/78 mmHg, 맥박 102회/분, 호흡수 22회/분, 산소포화도 94%"},
    "tests": {"일반혈액검사|cbc": "WBC 14,200/μL, Hb 13.8 g/dL, 혈소판 240,000/μL"},
    "diagnosis": "지역사회획득 폐렴", "aliases": ["폐렴", "community-acquired pneumonia", "CAP"],
    "must_check": [], "augmented": {"exam": [], "tests": []}, "augmented_full": {"panel": {}, "extra": []},
}


def _with(section: str, key: str, value: str, base=BASE) -> dict:
    case = copy.deepcopy(base)
    case[section][key] = value
    case["augmented"][section].append(key)
    return case


def _types(case: dict, severity: str = "hard") -> set[str]:
    return {i["type"] for i in cc.check_case(case) if i["severity"] == severity}


def test_clean_base_case_has_no_hard_issue():
    assert _types(BASE) == set()


def test_detects_leak_korean_english_and_abbreviation():
    assert "leak" in _types(_with("tests", "흉부 x선|cxr", "우하엽 경화 소견으로 폐렴에 합당"))
    assert "leak" in _types(_with("tests", "흉부 ct|chest ct", "findings of community-acquired pneumonia"))
    assert "leak" in _types(_with("tests", "흉부 ct|chest ct", "RLL consolidation, CAP pattern"))
    # a unit or a lower-case word that only looks like an abbreviation is not a leak
    assert "leak" not in _types(_with("tests", "혈당|glucose", "혈당 98 mg/dL, cap 없음"))


def test_detects_vital_and_lab_conflicts():
    assert "vital_conflict" in _types(_with("tests", "심전도|ecg", "동성리듬, 심박수 64회/분"))
    assert "vital_conflict" in _types(_with("exam", "체온 재측정|temp", "체온 36.5℃"))  # fever vs afebrile
    assert "lab_conflict" in _types(_with("tests", "빈혈 검사|anemia", "Hb 8.1 g/dL, MCV 70 fL"))
    # the same values, a reported range, or a later time point are fine
    assert _types(_with("tests", "심전도|ecg", "동성빈맥, 심박수 100회/분")) == set()
    assert _types(_with("tests", "빈혈 검사|anemia", "치료 3일 후 Hb 9.0 g/dL")) == set()


def test_detects_sex_and_age_inconsistent_tests():
    assert "demographic_test" in _types(_with("tests", "임신 검사|hcg", "소변 β-hCG 음성"))
    neonate = copy.deepcopy(BASE)
    neonate["initial"] = "생후 3일 남아. 주호소: 호흡곤란"
    assert "demographic_test" in _types(_with("tests", "트로포닌|troponin", "Troponin I < 0.01 ng/mL", neonate))
    assert "demographic_test" in _types(_with("tests", "복부 초음파|abdominal us", "간, 담낭, 자궁 및 난소 정상"))


def test_detects_implausible_values_and_bad_keys():
    assert "implausible_value" in _types(_with("tests", "전해질|electrolyte", "Na 214 mEq/L, K 4.0 mEq/L"))
    assert "key_empty_token" in _types(_with("tests", "갑상선 기능|tft|", "TSH 2.1"))
    assert "key_generic_token" in _types(_with("tests", "ct|흉부 ct", "정상"))


def test_mechanical_key_fix_keeps_specific_terms():
    case = _with("tests", "ct|흉부 ct|chest ct|", "정상")
    edits = cc.apply_fixes(case, "none")
    assert edits and "흉부 ct|chest ct" in case["tests"] and "흉부 ct|chest ct" in case["augmented"]["tests"]
    case = _with("exam", "안압 측정|iop", "안압 15 mmHg")
    cc.apply_fixes(case, "none")
    assert "안압 측정|iop 검사|iop test" in case["exam"]  # 'iop' alone would answer every 'biopsy' request
