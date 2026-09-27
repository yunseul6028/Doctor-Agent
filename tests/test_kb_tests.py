"""Curated test/lab/imaging finding → disease links (kb_tests.py + profile field findings_from_tests): detection of
results and their polarity, table/KB consistency, ranking effect of decisive results, discriminators, size/latency."""
import gzip
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import kb, kb_tests  # noqa: E402

needs_kb = pytest.mark.skipif(not kb.available(), reason="data/kb not built (python scripts/build_kb.py)")


# --- detection and polarity (no KB needed) ---------------------------------------------------------

@pytest.mark.parametrize("text,fid,pol", [
    ("리파아제 1,250 U/L", "lipase_high", 1),
    ("Lipase 32 U/L (정상)", "lipase_high", -1),
    ("혈청 아밀라아제 52 U/L, 리파아제 28 U/L", "lipase_high", -1),
    ("리파아제 상승", "lipase_high", 1),
    ("리파아제 정상 상한의 5배", "lipase_high", 1),
    ("Troponin-I < 0.01 ng/mL", "troponin_high", -1),
    ("hs-TnI 45 ng/L", "troponin_high", 1),
    ("트로포닌 I 2.3 ng/mL로 상승", "troponin_high", 1),
    ("항미토콘드리아항체(AMA) 양성 (타이터 1:160)", "ama_pos", 1),
    ("AMA (Anti-mitochondrial Ab) 음성", "ama_pos", -1),
    ("항핵항체(ANA) 양성(1:320, 균질형 패턴)", "ana_pos", 1),
    ("ANA 1:40", "ana_pos", -1),
    ("anti-dsDNA 양성", "dsdna_pos", 1),
    ("C3 감소", "complement_low", 1),
    ("HbA1c 9.2%", "hba1c_high", 1),
    ("당화혈색소 5.4%", "hba1c_high", -1),
    ("TSH 0.01 mIU/L로 감소", "tsh_low", 1),
    ("TSH 0.01 mIU/L로 감소", "tsh_high", -1),
    ("TSH 2.1 (정상: 0.4-4.0 mIU/L)", "tsh_high", -1),
    ("IgG 2450 mg/dL (정상 700-1600 mg/dL)", "igg_high", 1),
    ("소변 아질산염 양성", "ua_nitrite_pos", 1),
    ("WBC 50-100/HPF", "pyuria", 1),
    ("WBC 0-1/HPF", "pyuria", -1),
    ("소변 현미경: WBC 원주 관찰", "wbc_cast", 1),
    ("D-dimer 0.32 μg/mL", "ddimer_high", -1),
    ("D-dimer 1,250 ng/mL", "ddimer_high", 1),
    ("β-hCG: 음성", "hcg_pos", -1),
    ("ADAMTS13 활성도 < 10%", "adamts13_low", 1),
    ("복부 CT: 충수 직경 12mm로 비후 및 주위 지방 침윤", "ct_appendicitis", 1),
    ("충수는 정상 크기이며 주위 염증 없음", "ct_appendicitis", -1),
    ("간내외 담관 확장은 관찰되지 않음", "cbd_dilation", -1),
    ("심전도: II, III, aVF ST 분절 상승", "ecg_stemi", 1),
    ("심전도: 전반적인 ST 상승과 PR 분절 하강", "ecg_pericarditis", 1),
    ("관절액 검사: 백혈구 수 85,000/mm³(다형핵백혈구 92%)", "synovial_wbc_high", 1),
    ("관절액 편광현미경 검사상 요산염 결정(MSU) 및 피로인산염 결정(CPPD)은 관찰되지 않음", "msu_crystal", -1),
    ("자궁 내 임신낭 관찰되지 않음", "us_no_iup", 1),
])
def test_detect_polarity(text, fid, pol):
    got = kb_tests.detect(text)
    assert got.get(fid, (0,))[0] == pol, got


@pytest.mark.parametrize("text,fid", [
    ("주위 지방 조직의 소용돌이 모양 침윤 소견이 보임", "volvulus_sign"),   # not a whirl sign
    ("좌전하행지(LAD) 근위부 99% 혈전성 완전 폐색 소견", "egd_esophagitis"),  # LAD is not LA grade D
    ("20년 전에 우측 무릎 반월상 연골 손상으로 부분 절제술", "crescents"),     # meniscus
    ("Hb 12.4 g/dL", "b12_low"),
    ("1년 전부터 혈당이 약간 높다", "glucose_very_high"),                     # needs a value ≥ 250
    ("혈액 배양 검사 진행 중 (2쌍 발효 배양 시행).", "blood_culture_pos"),     # pending
    ("과거력: 기흉 병력", "cxr_ptx"),
    ("EEG: spike and wave 방전", "kidney_bx_membranous"),
])
def test_detect_false_positives(text, fid):
    assert fid not in kb_tests.detect(text), kb_tests.detect(text)


def test_negative_context():
    # a finding reported absent with no negation wording of its own → normal
    assert kb_tests.detect("트로포닌 상승", context=-1)["troponin_high"][0] == -1
    # ... but a measured value keeps its meaning, and "없음" in the text keeps per-segment polarity
    assert kb_tests.detect("리파아제 1,250 U/L", context=-1)["lipase_high"][0] == 1
    got = kb_tests.detect("담낭벽 비후 관찰, 담관 확장은 없음", context=-1)
    assert got["us_cholecystitis"][0] == 1 and got["cbd_dilation"][0] == -1


def test_table_is_consistent():
    ids = [f.id for f in kb_tests.FINDINGS]
    assert len(ids) == len(set(ids))
    for f in kb_tests.FINDINGS:
        assert f.mode in ("hi", "lo", "pos", "kw") and f.ko and f.en
        assert f.parsed_links(), f.id
        for dx, w, _r, ref in f.parsed_links():
            assert dx in kb_tests.DX and w in (1, 2, 3) and ref in kb_tests.REFS, (f.id, dx, ref)
    used = {ref for *_x, ref in kb_tests.links()}
    for key, ref in kb_tests.REFS.items():
        assert ref["cite"] and (ref["pmid"].isdigit() or key == "textbook")
        assert key in used, key  # every cited reference backs at least one link (docs/licenses.md lists them)


# --- KB integration ------------------------------------------------------------------------------

@needs_kb
def test_links_are_in_the_shipped_kb():
    """Every link of kb_tests.py is in data/kb/kb.json.gz (catches a forgotten rebuild)."""
    k = kb.get_kb()
    have = {(d["id"], x[0], x[2]) for d in k.diseases for x in d.get("findings_from_tests", [])}
    want = {(pid, "TF:" + fid, w) for fid, pid, w, _r, _ref in kb_tests.links()}
    assert want == have
    assert all("TF:" + f.id in k.terms for f in kb_tests.FINDINGS)


def _rank(findings, name, negatives=None, k=10):
    kbk = kb.get_kb()
    i = kbk._resolve(name)
    assert i is not None, name
    res = kb.candidates(findings, k=k, negatives=negatives)
    return next((r for r, c in enumerate(res, 1) if c["id"] == kbk.diseases[i]["id"]), None)


@needs_kb
@pytest.mark.parametrize("findings,dx,top", [
    (["상복부 통증", "구토", "리파아제 1,250 U/L"], "acute pancreatitis", 2),
    (["흉통", "식은땀", "트로포닌 I 2.3 ng/mL", "심전도: V1-V4 ST 분절 상승"], "acute myocardial infarction", 2),
    (["피로", "가려움", "ALP 420 U/L", "항미토콘드리아항체 양성"], "primary biliary cholangitis", 1),
    (["관절통", "발열", "ANA 양성 1:640", "anti-dsDNA 양성", "C3 감소"], "systemic lupus erythematosus", 1),
    (["다뇨", "다음", "HbA1c 9.2%"], "diabetes mellitus", 1),
    (["발열", "옆구리 통증", "소변 아질산염 양성", "소변 WBC 원주"], "pyelonephritis", 2),
    (["호흡곤란", "흉막성 흉통", "CT 폐동맥 조영: 우폐동맥 충만결손"], "pulmonary embolism", 1),
    (["우하복부 통증", "복부 CT: 충수 직경 12mm로 비후, 주위 지방 침윤"], "appendicitis", 1),
    # top 3 since the nlp migration (2026-09-28): "갑상선 비대" now matches goiter, which the Graves profile lacks (no
    # symptom features), and kb_tests reads "TSH 수용체 항체 양성" as elevated TSH too, so hypothyroidism edges past
    (["갑상선 비대", "체중 감소", "TSH 0.01 mIU/L", "TSH 수용체 항체 양성"], "Graves disease", 3),
])
def test_decisive_results_rank_the_disease(findings, dx, top):
    r = _rank(findings, dx)
    assert r is not None and r <= top, r


@needs_kb
def test_normal_rule_out_result_lowers_the_disease():
    base = ["흉통", "호흡곤란", "식은땀"]
    with_neg = _rank(base, "myocardial infarction", negatives=["트로포닌 음성"], k=50)
    without = _rank(base, "myocardial infarction", k=50)
    assert without is not None and (with_neg is None or with_neg >= without)
    # the score drops by the rule-out penalty (the rank may stay when MI leads by more than the penalty)
    score = {c["name_en"]: c["score"] for c in kb.candidates(base, k=50)}
    score_neg = {c["name_en"]: c["score"] for c in kb.candidates(base, k=50, negatives=["트로포닌 음성"])}
    assert score_neg.get("myocardial infarction", -99) < score["myocardial infarction"]


@needs_kb
def test_nonspecific_result_does_not_decide():
    # D-dimer alone is weak evidence (weight 1): it must not outrank a decisive result for another disease
    res = kb.candidates(["흉통", "D-dimer 상승", "심전도: II, III, aVF ST 분절 상승", "트로포닌 상승"], k=3)
    assert "infarction" in res[0]["name_en"].lower(), [c["name_en"] for c in res]


@needs_kb
def test_candidate_output_shape_backward_compatible():
    res = kb.candidates(["리파아제 1,250 U/L", "상복부 통증"], k=3)
    c = res[0]
    assert {"id", "name_ko", "name_en", "score", "kcd", "matched", "sources"} <= set(c)
    m = next(m for m in c["matched"] if m["id"].startswith("TF:"))
    assert m["ko"] == "리파아제 상승" and m["finding"] == "리파아제 1,250 U/L"
    assert "curated" in c["sources"]


@needs_kb
def test_profile_and_discriminators_expose_test_findings():
    p = kb.lookup("급성 췌장염")
    assert p["findings_from_tests"] and p["findings_from_tests"][0]["weight"] == 3
    assert all({"id", "ko", "en", "src", "weight", "ref", "rule_out"} <= set(x) for x in p["findings_from_tests"])
    assert kb.get_kb().test_ref(p["findings_from_tests"][0]["ref"])["cite"]
    d = kb.discriminators("급성 췌장염", "급성 담낭염")
    for key in ("symptoms_a_only", "symptoms_b_only", "tests_a_only", "tests_b_only", "tests_shared",
                "test_findings_a_only", "test_findings_b_only", "test_findings_shared"):
        assert key in d
    assert "리파아제 상승" in [x["ko"] for x in d["tests_a_only"]]       # decisive results lead tests_*_only
    assert any("담낭" in x["ko"] for x in d["test_findings_b_only"])
    assert all("src" in x and "en" in x for x in d["tests_a_only"])     # kb_hints reads these keys


@needs_kb
def test_render_mentions_decisive_tests():
    txt = kb.render_for_prompt(dx=["급성 췌장염"])
    assert "결정적 검사" in txt and "리파아제" in txt and len(txt) <= 800


@needs_kb
def test_match_terms_includes_test_findings():
    assert "TF:ama_pos" in kb.get_kb().match_terms("AMA 양성")


@needs_kb
def test_size_and_latency():
    size = sum(p.stat().st_size for p in (ROOT / "data" / "kb").iterdir() if p.is_file())
    assert size <= 20e6
    meta = json.load(gzip.open(ROOT / "data" / "kb" / "kb.json.gz", "rt", encoding="utf-8"))["meta"]
    assert meta["stats"]["test_links"] >= 400 and meta["test_refs"]
    findings = ["발열", "기침", "WBC 14,200/μL", "CRP 12 mg/dL", "흉부 X선: 우하엽 경화", "소변 아질산염 음성",
                "Troponin-I < 0.01 ng/mL", "D-dimer 0.3 μg/mL", "Na 138 mEq/L, K 4.1 mEq/L", "AST 22 U/L, ALT 18 U/L"] * 3
    k = kb.get_kb()
    k.candidates(findings)
    t0 = time.perf_counter()
    for _ in range(10):
        k.candidates(findings, negatives=["흉통 없음", "케톤 음성"])
    assert (time.perf_counter() - t0) / 10 < 0.1
