"""Clinical-finding normalisation layer (src/doctor_agent/nlp): lexicon, parser phenomena, match, performance."""
import json
from pathlib import Path

import pytest

from doctor_agent.nlp import LEXICON, affirmed, concepts_in, denied, match, parse
from doctor_agent.nlp.lexicon import LEXICON_PATH, load
from perf import assert_fast

ROOT = Path(__file__).resolve().parents[1]


def kp(text, **kw):
    return {(f.concept, f.polarity) for f in parse(text, **kw)}


@pytest.mark.perf
def test_lexicon_loads_fast_and_has_categories():
    loaded = []
    assert_fast(lambda: loaded.append(load(LEXICON_PATH)), 0.3, tries=3, what="lexicon load")  # strict: < 300 ms
    lex = loaded[-1]
    cats = lex.stats()["by_category"]
    for c in ("SYM", "SIGN", "LAB", "IMG", "ECG", "HX", "QUAL", "GRP"):
        assert cats.get(c, 0) > 0
    assert LEXICON.concept("SYM:fever").kb  # linked to KB term ids
    assert "LAB:lipase_high" in LEXICON.by_kb("TF:lipase_high")
    assert "SIGN:abdominal_tenderness" in LEXICON.ancestors("SIGN:rlq_tenderness")


@pytest.mark.parametrize("text,expect", [
    ("열이 나고 기침도 해요", {("SYM:fever", "present"), ("SYM:cough", "present")}),
    ("열은 없고 기침만 해요", {("SYM:fever", "absent"), ("SYM:cough", "present")}),
    ("몸이 불덩이 같아요", {("SYM:fever", "present")}),
    ("으슬으슬 추워요", {("SYM:chills", "present")}),
    ("입맛이 없어요", {("SYM:anorexia", "present")}),
    ("잘 먹어요", {("SYM:anorexia", "absent")}),
    ("통증이 가라앉지 않아요", {("SYM:pain", "present")}),
    ("속이 좀 메스껍긴 한데 토하지는 않았어요.", {("SYM:nausea", "present"), ("SYM:vomiting", "absent")}),
    ("배는 안 아파요", {("SYM:abdominal_pain", "absent")}),
])
def test_polarity_and_idioms(text, expect):
    assert expect <= kp(text)


def test_list_negation_scope():
    got = kp("숨참이나 콧물, 코막힘은 없고, 두근거림이나 가슴 통증, 소화기 증상도 없어요.")
    for c in ("SYM:dyspnea", "SYM:rhinorrhea", "SYM:nasal_congestion", "SYM:palpitation", "SYM:chest_pain", "GRP:gi"):
        assert (c, "absent") in got
    assert kp("수포음, 천명음, 흉막마찰음 없음.", source="exam") >= {
        ("SIGN:crackles", "absent"), ("SIGN:wheeze", "absent"), ("SIGN:friction_rub", "absent")}
    assert kp("우하복부에 압통 있음, 반발통 없음", source="exam") >= {
        ("SIGN:rlq_tenderness", "present"), ("SIGN:rebound_tenderness", "absent")}


def test_hedge_uncertain_hypothetical_and_simile():
    f = parse("열이 좀 있는 것 같아요")[0]
    assert (f.concept, f.polarity, f.hedged, f.severity) == ("SYM:fever", "present", True, "mild")
    assert kp("잘 모르겠어요", context={"question": "열이 났나요?"}) == {("SYM:fever", "uncertain")}
    assert not any(f.concept == "SYM:seizure" for f in parse("발작처럼 몸이 떨려요"))
    assert all(f.hypothetical for f in parse("혹시 심근경색일까 봐 걱정돼요"))


def test_subject_and_temporality():
    f = parse("어머니가 유방암이 있으셨어요")[0]
    assert (f.concept, f.subject) == ("HX:cancer", "family")
    assert kp("가족 중에 심장병 있는 사람은 없어요") == {("HX:cardiovascular_disease", "absent")}
    f = parse("예전에 결핵을 앓은 적이 있어요")[0]
    assert (f.concept, f.temporality) == ("HX:tuberculosis", "past")
    f = parse("3일 전부터 열이 나요")[0]
    assert f.onset.startswith("3일 전") and f.temporality == "current"
    assert parse("가끔 두통이 있어요")[0].temporality == "intermittent"
    assert not affirmed("아버지가 당뇨가 있으세요", "HX:diabetes")


def test_yes_no_answers():
    assert kp("네, 좀 해요", context={"question": "기침 하세요?"}) == {("SYM:cough", "present")}
    assert kp("아니요", context={"question": "열이나 오한 있으세요?"}) == {("SYM:fever", "absent"), ("SYM:chills", "absent")}
    assert kp("네, 없었어요", context={"question": "열이나 오한은 없으셨어요?"}) == {
        ("SYM:fever", "absent"), ("SYM:chills", "absent")}


def test_numbers_and_labs():
    fs = {f.concept: f for f in parse("체온 38.5°C, 혈압 125/80 mmHg, 맥박 112회/분", source="exam")}
    assert fs["SYM:fever"].polarity == "present" and fs["SYM:fever"].value == 38.5
    assert fs["SIGN:tachycardia"].polarity == "present"
    assert fs["SIGN:hypotension"].polarity == "absent"
    assert kp("체온 36.7도") == {("SYM:fever", "absent")}
    got = kp("WBC 14,200/μL, Hb 10.5 g/dL, 혈소판 240,000/μL", source="test")
    assert {("LAB:wbc_high", "present"), ("LAB:hb_low", "present"), ("LAB:plt_low", "absent")} <= got
    assert ("LAB:lipase_high", "present") in kp("리파아제 1,250 U/L (참고치 <60)", source="test")


def test_match_and_concepts_in():
    ev = parse("열은 없고 오른쪽 아랫배가 너무 아파요. 체온 37.0도")
    assert match("발열 없음", ev)[0]
    assert not match("발열", ev)[0]
    assert match("우하복부 통증", ev)[0] and match("복통", ev)[0]  # child supports parent
    assert match("WBC 14200", parse("WBC 14.2 x10^3/μL", source="test"))[0]
    assert match("CRP 상승", parse("CRP 5.2 mg/dL", source="test"))[0]
    assert not match("무언가 이상한 것", ev)[0]
    assert "SIGN:tenderness" in concepts_in("우하복부 압통 있음", source="exam")
    assert denied("열은 없어요", "SYM:fever") and not denied("열이 나요", "SYM:fever")


@pytest.mark.perf
def test_parse_speed():
    rows = [json.loads(line) for line in (ROOT / "data/labels/findings_gold_v1.jsonl").read_text("utf-8").splitlines()]

    def all_rows():
        for r in rows:
            parse(r["text"], r["source"], r.get("context"))
    assert_fast(all_rows, 0.002, per=len(rows), what="parse() per gold row")  # strict budget: < 2 ms per text


def test_gold_set_quality():
    """Regression floor on the hand-verified gold set (see data/labels/findings_gold_v1_metrics.json)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("label_findings", ROOT / "scripts/label_findings.py")
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)
    gold = [json.loads(line) for line in lf.GOLD.read_text("utf-8").splitlines()]
    assert len(gold) >= 170  # 173 public items; was >= 300 on the 397-item private gold set
    for r in gold:
        r["pred"] = [lf._as_gold(p) for p in lf._pred(r["text"], r["source"], r.get("context"))]
    assert lf._items_prf(gold, "pred")["f1"] >= 0.93


def test_assess_spans_reads_other_vocabularies_with_the_same_rules():
    from doctor_agent.nlp import normalize
    from doctor_agent.nlp.findings import assess_spans

    def one(text, word):
        t = normalize(text)
        s = t.find(word)
        return assess_spans(text, [(s, s + len(word))])[0]
    f = one("기침은 없고 가래가 있어요", "기침")
    assert (f.polarity, f.subject) == ("absent", "patient")
    assert one("기침은 없고 가래가 있어요", "가래").polarity == "present"
    assert one("어머니가 당뇨가 있어요", "당뇨").subject == "family"
    assert one("no fever", "fever").polarity == "absent"
    assert one("근력 약화 관찰됨", "근력 약화").polarity == "present"
    assert assess_spans("", [(0, 1)]) == [None]
