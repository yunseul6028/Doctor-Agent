"""Finding → term matching (curated tables, lab parsing), candidate filters, diagnosis normalisation, and a small
offline-benchmark regression (scripts/eval_kb.py) so ranking quality cannot silently drop."""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import kb, kb_curated  # noqa: E402

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built (python scripts/build_kb.py)")


def _en(finding: str) -> set[str]:
    k = kb.get_kb()
    return {k.terms[t]["en"] for t in k.match_terms(finding)}


# --- curated Korean wording ------------------------------------------------------------------

@pytest.mark.parametrize("finding,expect", [
    ("어젯밤부터 열이 좀 났어요", "fever"),
    ("지난주에 소변 볼 때 좀 찌릿하", "dysuria"),
    ("계속 메스껍고", "nausea"),
    ("오늘 아침부터 숨이 차요", "dyspnea"),
    ("특히 배랑 팔다리가 심하게 가려워요", "itch"),
    ("소변을 보면 거품이 많아요", "proteinuria"),
    ("최근에 체중이 5kg 증가했어요", "weight gain"),
    ("뺨의 나비 모양 홍반", "malar rash"),
    ("우하복부 압통", "abdominal tenderness"),
])
def test_curated_phrasings(finding, expect):
    assert expect in _en(finding), _en(finding)


@pytest.mark.parametrize("finding,wrong", [
    ("지난주에 소변 볼 때 좀 찌릿하", "paresthesia"),   # the regex consumes the span
    ("우측 상부 흉부에서 미세한 수포음이 청진됨", "vesicle"),  # 수포음 = crackles
    ("양측 폐야에서 호흡음 명료", "slowed breathing"),   # 호흡음 ≠ 서호흡
    ("상부 종격동 확장", "edema"),                    # 상부 종격동 ≠ 부종
    ("다음 날부터 기침", "polydipsia"),               # 다음 날 ≠ 다음(다음증)
    ("왼쪽 무릎의 발적, 열감, 부종", "fever"),          # local warmth is not fever
    ("anti-GAD 항체 양성", "generalized anxiety disorder"),  # no short English abbreviations
    ("40분 전부터 시작된 가슴 통증", "mastodynia"),
])
def test_no_spurious_matches(finding, wrong):
    assert wrong not in _en(finding)


def test_generic_terms_never_match():
    k = kb.get_kb()
    for f in ("벽 두께 증가 소견 관찰됨", "혈당 98 mg/dL", "통증"):
        assert not [t for t in k.match_terms(f) if t in k.post], f


# --- vitals / lab values ---------------------------------------------------------------------

@pytest.mark.parametrize("finding,expect", [
    ("백혈구 14,200/μL(호중구 85%)", {"leukocytosis"}),
    ("WBC 3.2 x10^3/uL", {"leukopenia"}),
    ("헤모글로빈 8.2 g/dL", {"anemia"}),
    ("혈소판 45,000/μL", {"thrombocytopenia"}),
    ("Na 128 mEq/L", {"hyponatremia"}),
    ("K 6.1 mEq/L", {"hyperkalemia"}),
    ("혈당 512 mg/dL", {"hyperglycemia"}),
    ("체온 39.4℃, 맥박 118회/분, 혈압 84/50 mmHg", {"fever", "high fever", "tachycardia", "hypotension"}),
    ("호흡 30회/분", {"tachypnea"}),
    ("산소포화도 88%", {"hypoxia"}),
])
def test_lab_values(finding, expect):
    assert set(kb_curated.lab_terms(finding)) >= expect


@pytest.mark.parametrize("finding", ["체온 36.7℃", "WBC 6,200/μL, Hb 13.5 g/dL, 혈소판 240,000/μL",
                                     "Na 140 mEq/L, K 4.1 mEq/L", "맥박 76회/분", "혈압 124/78 mmHg"])
def test_normal_values_are_silent(finding):
    assert kb_curated.lab_terms(finding) == []


def test_negation_forms():
    assert kb.candidates(["케톤 (-)"]) == []
    assert kb.candidates(["발열 없음"]) == []


# --- candidates: filters and synonym grouping --------------------------------------------------

def test_patient_profile():
    assert kb.patient_profile("35세 여성. 주호소: 복시") == ("여성", 35.0)
    assert kb.patient_profile("7개월 남아") == ("남성", 7 / 12)
    assert kb.patient_profile("주호소: 두통") == ("", None)


def test_sex_filter_drops_other_sex_profiles():
    k = kb.get_kb()
    f = ["하복부 통증", "질 출혈", "무월경"]
    sexes = k._sex_table()
    male = kb.candidates(f, k=30, sex="남성")
    assert male and all(sexes[k.by_id[c["id"]]] != "여성" for c in male)
    female = kb.candidates(f, k=30, sex="여성")
    assert any(sexes[k.by_id[c["id"]]] == "여성" for c in female)


def test_synonymous_terms_count_once():
    """One finding matching several near-synonym terms (restlessness, agitation, ...) is one concept."""
    k = kb.get_kb()
    groups = k._groups("통증으로 인한 안절부절못함")
    assert len(groups) == 1 and len(groups[0]) >= 2


def test_backward_compatible_signature():
    res = kb.candidates(["발열", "기침", "흉통"], 5, ["콧물"])
    assert len(res) == 5
    assert {"id", "name_ko", "name_en", "score", "kcd", "matched", "sources"} <= set(res[0])


# --- diagnosis normalisation -------------------------------------------------------------------

@pytest.mark.parametrize("text,code", [
    ("급성 ST분절상승 심근경색(전벽)", "I21"),   # not I21.4 (NSTEMI) by fuzzy overlap
    ("원발성 자연 기흉(우측)", "J93.1"),
    ("자연 기흉", "J93.1"),
    ("지주막하 출혈", "I60"),                     # 지주막하 ↔ 거미막하
    ("급성 담석성 담낭염", "K81"),
    ("일차성 자발성 기흉", "J93.1"),
    ("acute calculous cholecystitis", "K81"),
    ("중추성 요붕증", "E23.2"),                   # not nephrogenic (N25.1)
])
def test_normalize_codes(text, code):
    n = kb.normalize_diagnosis(text)
    assert n and n["code"] == code, n


@pytest.mark.parametrize("text", ["stemi", "anterior stemi", "PBC", "CPPD", "tth", "Inferior MI"])
def test_normalize_no_wild_abbreviation_guesses(text):
    n = kb.normalize_diagnosis(text)
    bad = {"systemic mycosis", "localized anterior staphyloma", "색소피부건조증", "상세불명의 연충증"}
    assert n is None or n["name"] not in bad


def test_normalize_does_not_cross_word_boundaries():
    n = kb.normalize_diagnosis("고환 염전")  # torsion, not orchitis ("고환염")
    assert n is None or "고환염" not in n["name"]
    n = kb.normalize_diagnosis("글란츠만 혈소판무력증")  # not asthenia (R53)
    assert n is None or not n["code"].startswith("R53")


def test_normalize_deterministic_and_fast():
    t0 = time.perf_counter()
    first = [kb.normalize_diagnosis(x) for x in ("다카야수 동맥염", "마이그스 증후군", "Reactive arthritis")]
    again = [kb.normalize_diagnosis(x) for x in ("다카야수 동맥염", "마이그스 증후군", "Reactive arthritis")]
    assert first == again
    assert (time.perf_counter() - t0) / 6 < 0.2


# --- offline benchmark regression -----------------------------------------------------------------

def test_benchmark_regression_dev_subset():
    """Fixed dev subset (all 16 sample cases + first 34 clinicalqa cases by id) of scripts/eval_kb.py.
    Values at the time of writing (2026-09-26): top-10 16/50, top-50 23/50; margin of 2 cases."""
    from scripts import eval_kb as E
    cases = E.load_cases(("sample", "clinicalqa"))
    sub = [c for c in cases if c["_set"] == "sample"]
    sub += sorted((c for c in cases if c["_set"] == "clinicalqa"), key=lambda c: c["_id"])[:34]
    rows = [E.rank_case(c) for c in sub]
    top10 = sum(1 for r in rows if r["rank"] and r["rank"] <= 10)
    top50 = sum(1 for r in rows if r["rank"])
    assert len(sub) == 50
    assert top10 >= 14, top10
    assert top50 >= 21, top50
    assert max(r["ms"] for r in rows) < 500
