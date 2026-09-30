"""Orphanet phenotype frequencies (Orphadata, CC BY 4.0) and the HIRA prevalence prior (KOGL type 1) in the KB."""
import gzip
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import kb  # noqa: E402
from perf import assert_fast

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built (python scripts/build_kb.py)")

FREQ = {"O", "VF", "F", "OC", "VR", "EX"}


def _rank(findings, did, **kw):
    ids = [c["id"] for c in kb.candidates(findings, k=50, **kw)]
    return ids.index(did) + 1 if did in ids else None


# --- provenance / licences ----------------------------------------------------------------------

def test_sources_and_licenses_recorded():
    k = kb.get_kb()
    src = k.meta["sources"]
    assert src["Orphanet"]["license"] == "CC BY 4.0"
    assert "orphadata.com" in src["Orphanet"]["url"]
    assert "제1유형" in src["HIRA-stats"]["license"] and "15118806" in src["HIRA-stats"]["url"]
    assert k.meta["versions"]["Orphanet"]
    ledger = (ROOT / "docs/licenses.md").read_text(encoding="utf-8")
    assert "en_product4.xml" in ledger and "data.go.kr/data/15118806" in ledger
    # the HPO ontology file itself is not part of the build
    assert "hp.obo" not in (ROOT / "scripts/build_kb.py").read_text(encoding="utf-8").replace(
        "Human Phenotype Ontology file (hp.obo)", "")


def test_orpha_freq_structure():
    k = kb.get_kb()
    n_prof = n_links = n_ex = 0
    for d in k.diseases:
        of = d.get("orpha_freq")
        if not of:
            continue
        n_prof += 1
        own = {t: s for t, s in d["symptoms"]}
        assert d["codes"].get("orpha"), d["id"]
        for t, f in of:
            n_links += 1
            assert t in k.terms and f in FREQ
            if f == "EX":
                n_ex += 1
                assert "Orphanet" not in own.get(t, [])
            else:
                assert "Orphanet" in own[t]
    assert n_prof > 4000 and n_links > 100_000 and n_ex > 500
    orpha_only = [d for d in k.diseases if d["id"].startswith("ORPHA:")]
    assert len(orpha_only) > 1000
    assert all(d["names_en"] and d["names_en"][0][1] == "Orphanet" for d in orpha_only)


def test_new_phenotype_terms_have_no_llm_labels():
    k = kb.get_kb()
    new = {tid: t for tid, t in k.terms.items() if tid.startswith("HP:")}
    assert len(new) > 1000
    for t in new.values():
        assert t["src"]["en"] in ("Orphanet", "WD")
        if t["ko"]:
            assert t["src"]["ko"] in ("WD", "KCD"), t  # never "LLM"


# --- ranking ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("did,findings", [
    ("DOID:0050782", ["소화성 궤양", "설사", "식도염"]),               # Zollinger-Ellison: no symptoms before Orphanet
    ("DOID:0050777", ["소뇌실조증", "근긴장저하", "안구운동 실행증"]),     # Joubert syndrome
    ("DOID:0050633", ["안구진탕증", "눈부심", "난시"]),                   # albinism
    ("DOID:14323", ["세장지증", "수정체 이탈", "대동맥 박리"]),            # Marfan
    ("DOID:14499", ["혈관각화종", "각막 윤상 각화증", "신부전"]),          # Fabry
])
def test_rare_disease_from_orphanet_phenotypes(did, findings):
    r = _rank(findings, did)
    assert r is not None and r <= 3, (did, r)


def test_frequency_weights_postings():
    k = kb.get_kb()
    i = k.by_id["DOID:14323"]  # Marfan
    freq = dict(k.diseases[i]["orpha_freq"])
    own = {t: s for t, s in k.diseases[i]["symptoms"]}
    only = {f: t for t, f in freq.items() if own.get(t) == ["Orphanet"]}
    assert "VF" in only and "OC" in only
    w = {t: x for t, lst in k.post.items() for j, x in lst if j == i}
    assert w[only["VF"]] > w[only["OC"]] > 0
    p = kb.lookup("마르팡 증후군")
    assert p["symptoms"][0].get("freq") and any(s.get("freq") == "매우 흔함(80-99%)" for s in p["symptoms"])


def test_excluded_finding_penalises():
    k = kb.get_kb()
    import re
    case = None
    for t, dis in sorted(k.xpost.items()):
        if not re.search("[가-힣]", k.terms[t]["ko"]) or not k._match_spans(k.terms[t]["ko"]):
            continue
        for i in dis:
            d = k.diseases[i]
            freq = dict(d["orpha_freq"])
            pos = [k.terms[x]["ko"] for x, f in sorted(freq.items()) if f in ("O", "VF", "F")
                   and re.search("[가-힣]", k.terms[x]["ko"]) and k._match_spans(k.terms[x]["ko"])][:3]
            if len(pos) >= 2:
                case = (d["id"], pos, k.terms[t]["ko"])
                break
        if case:
            break
    assert case, "no disease with a Korean-labelled excluded finding"
    did, pos, ex = case

    def score(excl_w):
        old = k.EXCL_W
        k.EXCL_W = excl_w
        try:
            return next((c["score"] for c in k.candidates(pos + [ex], k=20000) if c["id"] == did), None)
        finally:
            k.EXCL_W = old
    with_pen, without = score(k.EXCL_W), score(0.0)
    assert with_pen is not None and without is not None and with_pen < without, case
    p = kb.lookup(did)
    assert any(x["ko"] == ex for x in p.get("excluded", []))


# --- prevalence prior ---------------------------------------------------------------------------------

def test_prevalence_table_and_demographics():
    k = kb.get_kb()
    with gzip.open(k.kb_dir / "kcd3_prev.tsv.gz", "rt", encoding="utf-8") as f:
        assert f.readline().startswith("code\t")
    assert len(k.prev) > 1500 and all(len(v) == 36 for v in k.prev.values())
    mi = k._resolve("급성 심근경색")
    assert k.prevalence(mi, "남성", 62) > 20 * k.prevalence(mi, "여성", 7)
    assert k.prevalence(mi) >= k.prevalence(mi, "남성", 62)
    no_code = next(i for i, d in enumerate(k.diseases) if not d["codes"].get("kcd") and not d["codes"].get("kcd_broad"))
    assert k.prevalence(no_code) is None


def test_prevalence_factor_is_capped():
    k = kb.get_kb()
    assert 0 < k.PREV_W <= 0.3
    fs = [k._prev_factor(i, s, a) for i in range(0, len(k.diseases), 97) for s, a in (("남성", 40), (None, None))]
    assert min(fs) >= 1 - k.PREV_W - 1e-9 and max(fs) <= 1 + k.PREV_W + 1e-9


def test_decisive_test_result_still_wins_for_rare_disease():
    # the prior scales only the symptom part: a decisive curated test result keeps its full weight
    k = kb.get_kb()
    assert k.PREV_TESTS is False
    res = kb.candidates(["피로", "관절통", "AMA 양성", "ALP 상승"], k=5, sex="여성", age=50)
    assert any("담관" in c["name_ko"] or "cholangitis" in c["name_en"].lower() for c in res[:3]), res


# --- size / latency -----------------------------------------------------------------------------------

@pytest.mark.perf
def test_kb_size_and_load_time():
    total = sum(p.stat().st_size for p in (ROOT / "data/kb").iterdir() if p.is_file())
    assert total < 20e6, total
    assert_fast(kb.KnowledgeBase, 5.0, tries=3, what="KnowledgeBase() load")  # strict budget: < 5 s (~1 s idle)
