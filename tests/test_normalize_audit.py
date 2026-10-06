"""Diagnosis-name normalisation audit (scripts/audit_normalize.py, 2026-09-29).

1. Regression tests for the mappings the audit found wrong (wrong chapter / opposite concept / homograph / tie-break).
2. The audit's cheap checks on every gold name (data/cases_aug diagnoses + aliases, specialty_gold_v*.jsonl, the
   can't-miss / rule-out names of safety/danger_gate.py, safety/protocols.py and the consult specs): any flag outside
   the hand-reviewed allow-list (data/labels/normalize_audit_allow.json) fails.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from doctor_agent.knowledge import kb, specialty  # noqa: E402

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built (python scripts/build_kb.py)")


def _code(text):
    n = kb.normalize_diagnosis(text)
    return n["code"] if n else None


@pytest.mark.parametrize("text,code", [
    # tie-break between cited ICD-10 categories (was I80 phlebitis; routed vasculitis to cardio)
    ("혈관염", "I77.6"), ("vasculitis", "I77.6"),
    # title agreement between tied categories (DO cites ICD-10-CM K55.3 = angiodysplasia in KCD; I01.0 is rheumatic)
    ("Necrotizing enterocolitis", "P77"), ("Acute pericarditis", "I30"), ("심낭염", "I30"),
    # poisoning is T36-T65, not substance use (F11.1) or a fuzzy neighbour (T58 carbon monoxide)
    ("Opioid overdose", "T40.2"), ("opioid poisoning", "T40.2"), ("acetaminophen overdose", "T39.1"),
    ("digoxin toxicity", "T46.0"), ("일산화탄소 중독", "T58"),
    # traumatic context of a longer KCD name (was S06.8 intracranial injury)
    ("두개내 출혈", "I62.9"),
    # parenthesised KCD title "헤노흐(-쇤라인)자반" (was D69.2 purpura)
    ("헤노흐-쇤라인 자반증", "D69.0"),
    # Korean fuzzy no longer crosses organs (was N28.0 renal artery obstruction)
    ("폐동맥 색전증", "I26"),
    # abbreviations the KB's synonym index sent to rare diseases
    ("acs", "I24.9"), ("ACS", "I24.9"), ("SJS", "L51.1"), ("SLE", "M32"), ("급성 관상동맥 증후군", "I24.9"),
    # spelling pairs
    ("다카야수 동맥염", "M31.4"), ("라이터 증후군", "M02.3"), ("신생아 중독성 홍반", "P83.1"),
    ("세균성 뇌수막염", "G00"), ("결핵성 림프절염", "A18.2"), ("양성 발작성 체위 현훈", "H81.1"),
    # English superstring must end on a word boundary (was I47.1 supraventricular)
    ("ventricular tachycardia", "I47.2"),
    # explicit codes still work
    ("급성 심근경색 (I21)", "I21"), ("I21.9", "I21.9"), ("KCD I21", "I21"), ("J18", "J18"),
    # curated one-offs
    ("저칼슘혈증", "E83.5"), ("당뇨병성 신증", "E14.2"), ("면역성 혈소판 감소증", "D69.3"),
])
def test_audit_fixed_codes(text, code):
    assert _code(text) == code, kb.normalize_diagnosis(text)


@pytest.mark.parametrize("text,bad", [
    ("흉강 비장 이식증", "F50"),        # 이식증 = pica homograph (thoracic splenosis)
    ("저칼슘혈증", None),               # never hypercalcaemia (checked by name below)
    ("심장 혈관육종", "I51"),           # fuzzy dropped the head noun (heart disease)
    ("파르보바이러스 B19 감염증", "B19"),  # a bare 3-character code inside a name is not a code
    ("Parvovirus B19 infection", "B19"),
    ("HLA-B27 관련 척추관절염", "B27"),
    ("심장성 실신", "I50"),             # "심장성, 심장 또는 심근부전 NOS" comma fragment
    ("급성 용혈", "O14"),               # "용혈, 간효소상승 ..." comma fragment of HELLP
    ("임신(자궁외 임신 포함)", "G25"),   # comma fragment "임신, 출산 및 산후기에 합병된 ..."
    ("Transient left 6th nerve palsy", "G57"),  # generic "nerve palsy" of peroneal palsy
    ("발열성 호중구감소증", "L98"),     # not Sweet syndrome
    ("Osteoclastoma", None),
    ("Epidermoid cyst", "K09"),
    ("폐-신장 증후군", "N04"),          # hyphenated compound is not "신장 증후군"
])
def test_audit_wrong_mappings_gone(text, bad):
    n = kb.normalize_diagnosis(text)
    if bad:
        assert n is None or not n["code"].startswith(bad), n
    if text == "저칼슘혈증":
        assert "고칼슘" not in (n or {}).get("name", "")
    if text == "Osteoclastoma":
        assert n is None or "osteoblastoma" not in n.get("name_en", "").lower(), n


@pytest.mark.parametrize("abbr", ["HD", "PD", "MS", "AS", "AD", "PV", "CDI"])
def test_ambiguous_abbreviations_not_normalised(abbr):
    assert kb.normalize_diagnosis(abbr) is None


def test_fuzzy_guards():
    k = kb.get_kb()
    assert kb._antonym_swap("저칼슘혈증", "고칼슘혈증")
    assert kb._antonym_swap("osteoclastoma", "osteoblastoma")
    assert not kb._antonym_swap("갑상선기능저하증", "갑상샘기능저하증")
    assert kb._same_head("대동맥축착증", "대동맥의축착") and not kb._same_head("심장혈관육종", "심장혈관질환")
    # word-order variant still fuzzes; a spelling variant still resolves
    assert _code("신경이완제 악성증후군") == "G21.0"
    assert _code("본태성 혈소판 과다증") == "D47.3"
    assert k._fuzzy("opioidpoisoning", ["opioid", "poisoning"]) is None or \
        "monoxide" not in k.diseases[k._fuzzy("opioidpoisoning", ["opioid", "poisoning"])]["names_en"][0][0]


def test_specialty_routing_of_fixed_names():
    assert specialty.specialty_of("혈관염") == "rheum_immune"
    assert specialty.specialty_of("흉강 비장 이식증") != "psych"
    assert specialty.specialty_of("두개내 출혈") == "neuro"
    assert specialty.specialty_of("폐동맥 색전증") in ("cardio", "resp_id")


def test_audit_checks_on_gold_sets_within_allow_list():
    import audit_normalize as au
    rows = au.run(au.all_groups())
    assert len(rows) > 1000  # 1083 on the public cases; was > 1500 with the private case sets
    bad = au.unallowed(rows, au.load_allow())
    msg = "\n".join(f"[{f}] {r['src']}:{r['id']} {r['input']!r} -> {r['code']} {r['name']} ({r['match']})"
                    for f, r in bad[:30])
    assert not bad, f"{len(bad)} new normalisation flags (review, fix or add to " \
                    f"data/labels/normalize_audit_allow.json):\n{msg}"
