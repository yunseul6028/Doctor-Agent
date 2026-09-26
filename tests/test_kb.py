import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import kb  # noqa: E402

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built (python scripts/build_kb.py)")

KNOWN_SOURCES = {"DO", "WD", "KCD", "DDXPlus", "MedlinePlus", "MedlinePlus+LLM", "LLM"}


def _top_names(findings, k=5):
    return [c["name_ko"] for c in kb.candidates(findings, k=k)] + [c["name_en"].lower() for c in kb.candidates(findings, k=k)]


# --- lookup ------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,expect_en", [
    ("폐렴", "pneumonia"), ("당뇨병", "diabetes"), ("급성 충수염", "appendicitis"),
    ("pneumonia", "pneumonia"), ("Pulmonary embolism", "pulmonary embolism"), ("asthma", "asthma"),
])
def test_lookup_korean_and_english(q, expect_en):
    p = kb.lookup(q)
    assert p is not None
    assert any(expect_en in n.lower() for n, _ in p["names_en"])
    assert p["name_ko"]


def test_lookup_fuzzy_and_miss():
    assert kb.lookup("급성충수염") is not None           # spacing variant
    p = kb.lookup("폐렴 (지역사회획득)")                   # parenthetical qualifier ignored
    assert p is not None and "pneumonia" in p["name_en"].lower()
    assert kb.lookup("zzqx 없는질환") is None
    assert kb.lookup("") is None


def test_profile_has_symptoms_and_questions():
    p = kb.lookup("pneumonia")
    kos = {s["ko"] for s in p["symptoms"]}
    assert "기침" in kos and "발열" in kos
    assert p["questions"], "DDXPlus question wording should be attached"


# --- candidates --------------------------------------------------------------------------------

def test_candidates_pneumonia():
    # no symptom frequencies in the sources, so TB/influenza tie with pneumonia; top 10 is the honest bar
    names = _top_names(["발열", "기침", "흉통"], k=10)
    assert any("폐렴" in n or "pneumonia" in n for n in names), names


def test_candidates_diabetes():
    names = _top_names(["다음", "다뇨", "체중 감소"], k=5)
    assert any("당뇨" in n or "diabetes" in n for n in names), names


def test_candidates_english_and_matched_terms():
    res = kb.candidates(["fever", "cough", "shortness of breath"], k=10)
    assert res and all(r["matched"] for r in res)
    assert any("pneumonia" in r["name_en"].lower() for r in res)


def test_negated_findings_are_ignored():
    assert kb.candidates(["발열 없음"]) == []


# --- discriminators / normalisation / rendering ------------------------------------------------

def test_discriminators():
    d = kb.discriminators("폐렴", "인플루엔자")
    assert d is not None
    assert d["symptoms_a_only"] or d["symptoms_b_only"]
    assert kb.discriminators("폐렴", "zzqx 없는질환") is None


def test_normalize_diagnosis():
    n = kb.normalize_diagnosis("급성 충수염")
    assert n and n["code"].startswith("K35") and n["name"] == "급성 충수염"
    n = kb.normalize_diagnosis("pneumonia")
    assert n and n["code"].startswith("J1") and "폐렴" in n["name"]
    n = kb.normalize_diagnosis("J18.9")
    assert n and n["code"] == "J18.9"
    assert kb.normalize_diagnosis("zzqx") is None
    assert kb.normalize_diagnosis("자궁외임신")["sex"] == "여성"


def test_render_for_prompt_short_korean_with_sources():
    text = kb.render_for_prompt(["발열", "기침", "흉통"])
    assert text and len(text) <= 800
    assert "일치: " in text and "[" in text and "감별" in text
    dx_text = kb.render_for_prompt(dx=["폐렴", "인플루엔자"])
    assert "폐렴 (J18" in dx_text and "인플루엔자" in dx_text and len(dx_text) <= 800
    assert kb.render_for_prompt([]) == ""


# --- provenance ----------------------------------------------------------------------------------

def test_every_field_has_a_source():
    k = kb.get_kb()
    assert set(k.meta["sources"]) >= {"DDXPlus", "KCD", "DO", "WD", "MedlinePlus"}
    for d in k.diseases:
        for lst in ("names_ko", "names_en"):
            for name, src in d[lst]:
                assert name and src in KNOWN_SOURCES
        for kind, vals in d["codes"].items():
            for v, src in vals:
                assert v and src in KNOWN_SOURCES
        for f in ("symptoms", "risk", "tests"):
            for tid, srcs in d.get(f, []):
                assert tid in k.terms and srcs and set(srcs) <= KNOWN_SOURCES
        for f in ("def", "summary_ko", "severity", "medlineplus"):
            if f in d:
                assert d[f][1] in KNOWN_SOURCES
        for x in d.get("findings_from_tests", []):  # curated test-result links: [term, ["curated"], w, ref, R]
            assert x[0] in k.terms and x[1] == ["curated"] and x[2] in (1, 2, 3) and x[3]
    for tid, t in k.terms.items():
        if tid.startswith("TF:"):
            assert t["src"] == {"en": "curated", "ko": "curated"}
            continue
        assert t["src"].get("en") in KNOWN_SOURCES
        if t["ko"]:
            assert t["src"].get("ko") in KNOWN_SOURCES


def test_latency():
    k = kb.get_kb()
    t0 = time.perf_counter()
    for _ in range(5):
        k.candidates(["발열", "기침", "흉통", "호흡곤란"])
        k.lookup("급성 췌장염")
        k.normalize_diagnosis("급성 심근경색")
    assert (time.perf_counter() - t0) / 5 < 0.5
