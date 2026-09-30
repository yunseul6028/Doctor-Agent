"""Gold test set for the clinical-finding parser (src/doctor_agent/nlp), built WITHOUT any LLM.

    python scripts/label_findings.py extract   # every history/exam/test sentence of data/cases_aug, auto-labelled
                                               #   -> data/labels/findings_gold_candidates.jsonl
    python scripts/label_findings.py freq      # frequent history n-grams that no lexicon form covers (lexicon review)
    python scripts/label_findings.py sample    # stratified draft for hand review (+ self-written yes/no answers)
                                               #   -> data/labels/findings_gold_draft.jsonl
    python scripts/label_findings.py freeze    # draft + hand corrections -> data/labels/findings_gold_v1.jsonl
    python scripts/label_findings.py metrics   # precision/recall/F1 by phenomenon, and the old scattered logic on
                                               #   the same sentences -> data/labels/findings_gold_v1_metrics.json

Labelling protocol (hand review by the knowledge-rag agent, model claude-opus-5-5, 2026-09-27; no LLM was called by
any script): each sampled sentence was read with its auto-labels and the full list of findings was corrected by hand
in data/labels/findings_gold_corrections.json (id -> gold list; "ok" = draft accepted as is). A gold finding is
[concept, polarity, subject, temporality] (+ value for measured findings). Scope: every lexicon concept the sentence
asserts, denies or leaves uncertain, including measured vital-sign/lab concepts (체온 36.7 -> SYM:fever absent) and
"absence" expressions (잘 먹어요 -> SYM:anorexia absent). Findings from knowledge/kb_tests.detect() (test-result
concepts, their own logic) are out of scope and excluded from both sides. Concepts the lexicon does not have are not
labelled (listed in the item's "oov" note when they matter).

The draft is sampled with a fixed seed, stratified by surface phenomena detected with regexes on the raw text (not
by the parser output): list negation, negation, hedge, family, past history, numbers, idioms, similes,
hypotheticals, plain statements, exam lists; plus self-written yes/no answers to doctor questions (source
"synthetic": the case files contain no question/answer pairs).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.nlp import findings as nlpf  # noqa: E402
from doctor_agent.nlp.lexicon import LEXICON, compact, normalize  # noqa: E402


def _ground_qa(claim, qa: list[tuple[str, str]], negative: bool) -> tuple[bool, float, str]:
    """Old grounding against the answers of (question, answer) pairs read with their question (a yes/no answer
    resolves the question's findings). Offline comparison only; moved here from agent/grounding.py on 2026-09-30 (the
    agent reads ASK answers with their question inside the case evidence itself)."""
    from doctor_agent.agent import grounding
    if not qa:
        return False, 0.0, ""
    ev = grounding.Evidence.build("\n".join(re.sub(r"\s*\n\s*", "; ", a or "") for _q, a in qa),
                                  ["patient"] * len(qa), [q or "" for q, _a in qa])
    return grounding._ground(claim, ev, negative)

CASES = ROOT / "data" / "cases_aug"
LABELS = ROOT / "data" / "labels"
CANDIDATES = LABELS / "findings_gold_candidates.jsonl"
DRAFT = LABELS / "findings_gold_draft.jsonl"
CORRECTIONS = LABELS / "findings_gold_corrections.json"
GOLD = LABELS / "findings_gold_v1.jsonl"
METRICS = LABELS / "findings_gold_v1_metrics.json"
DEV_IDS = LABELS / "findings_dev_ids.json"
REVIEW = LABELS / "findings_gold_review.txt"
SEED = 20260927

SECTION_SOURCE = {"history": "patient", "exam": "exam", "tests": "test"}
_SPLIT = re.compile(r"(?<=[.?!])\s+(?=\S)|\n+")

# surface phenomena (regexes over the raw sentence; used only for stratification and per-phenomenon metrics)
PHENOMENA: dict[str, re.Pattern] = {
    "negation_list": re.compile(r"(?:,|이나|거나|및|또는|혹은)[^.]*(?:없|않|음성|아니)"),
    "negation": re.compile(r"없|않|아니|안 [가-힣]|음성|정상|\(-\)|부인"),
    "hedge": re.compile(r"것 같|거 같|듯|아마|모르|글쎄|의심|시사|가능성|애매|기억이"),
    "family": re.compile(r"어머니|아버지|엄마|아빠|부모|형제|자매|누나|언니|오빠|동생|할머니|할아버지|가족|삼촌|이모|고모|친척|남편|아내|동료"),
    "past": re.compile(r"예전|과거|전에|적이|적은|적도|앓았|진단받|수술|입원|어렸을|년 전"),
    "numbers": re.compile(r"\d+(?:\.\d+)?\s*(?:°|℃|도|mmhg|회/분|%|mg|g/dl|/μl|/ul|iu|u/l|meq|mmol|ng|pg|mm/hr)", re.I),
    "idiom": re.compile(r"입맛|밥맛|식욕|힘이 없|기운이 없|수 없|가라앉지|멈추지|잘 먹|잠을 못|잠이 안|잘 자|괜찮|멀쩡|좋아지지|낫지"),
    "simile": re.compile(r"처럼|같이|듯한|듯이|마냥"),
    "hypothetical": re.compile(r"까 봐|걱정|혹시|\?|아닐까|인지"),
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text or "") if len(s.strip()) >= 2]


def _phenomena(text: str, section: str) -> list[str]:
    tags = [k for k, rx in PHENOMENA.items() if rx.search(text)]
    if section == "exam" and "," in text and "negation" in tags:
        tags.append("exam_list")
    if not tags or tags == ["numbers"] and section == "history":
        tags.append("plain")
    return tags


def _key_label(key: str) -> str:
    return key.split("|")[0]


def _pred(text: str, source: str, context: dict | None) -> list[dict]:
    out = []
    for f in nlpf.parse(text, source, context):
        if f.cue.startswith("kb_tests"):
            continue
        d = {"concept": f.concept, "polarity": f.polarity, "subject": f.subject, "temporality": f.temporality,
             "span": f.span}
        if f.value is not None:
            d["value"] = f.value
        if f.hedged:
            d["hedged"] = True
        if f.hypothetical:
            d["hypothetical"] = True
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ extract
def extract() -> list[dict]:
    rows, seen = [], {}
    for f in sorted(CASES.glob("*/*.json")):
        case = json.loads(f.read_text(encoding="utf-8"))
        cid = f"{f.parent.name}/{f.stem}"
        for section in ("history", "exam", "tests"):
            for ki, (key, val) in enumerate((case.get(section) or {}).items()):
                for si, sent in enumerate(_sentences(str(val))):
                    norm = normalize(sent)
                    if norm in seen:
                        seen[norm]["dup"] += 1
                        continue
                    ctx = {"key": key} if section == "history" else None
                    row = {"id": f"{cid}:{section[0]}{ki}.{si}", "case": cid, "section": section,
                           "source": SECTION_SOURCE[section], "key": _key_label(key), "text": sent,
                           "context": ctx, "phenomena": _phenomena(sent, section), "dup": 0}
                    seen[norm] = row
                    rows.append(row)
    t0 = time.perf_counter()
    for r in rows:
        r["auto"] = _pred(r["text"], r["source"], r["context"])
    dt = time.perf_counter() - t0
    with open(CANDIDATES, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_sec = Counter(r["section"] for r in rows)
    print(f"{len(rows)} unique sentences ({dict(by_sec)}), parse {1000 * dt / max(1, len(rows)):.3f} ms/sentence avg"
          f" -> {CANDIDATES.relative_to(ROOT)}")
    return rows


# ------------------------------------------------------------------------------------------------ freq
def freq(top: int = 150) -> None:
    """Eojeol bigrams/trigrams of history answers not covered by any lexicon mention (candidates for casefreq.tsv)."""
    c: Counter = Counter()
    for f in sorted(CASES.glob("*/*.json")):
        case = json.loads(f.read_text(encoding="utf-8"))
        for val in (case.get("history") or {}).values():
            for sent in _sentences(str(val)):
                t = normalize(sent)
                covered = [(m.start, m.end) for m in LEXICON.scan(t)]
                words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[가-힣]+", t)]
                for n in (2, 3):
                    for i in range(len(words) - n + 1):
                        s, e = words[i][0], words[i + n - 1][1]
                        if any(a < e and s < b for a, b in covered):
                            continue
                        c[" ".join(w[2] for w in words[i:i + n])] += 1
    for gram, k in c.most_common(top):
        if k >= 3:
            print(f"{k:4d}  {gram}")


# ------------------------------------------------------------------------------------------------ synthetic yes/no
# Self-written doctor questions and short patient answers (Korean colloquial), with hand gold labels.
YESNO: list[tuple[str, str, list[list[str]]]] = [
    ("기침 하세요?", "네, 좀 해요.", [["SYM:cough", "present", "patient", "current"]]),
    ("기침 하세요?", "아니요, 안 해요.", [["SYM:cough", "absent", "patient", "current"]]),
    ("열이 나세요?", "네.", [["SYM:fever", "present", "patient", "current"]]),
    ("열이 나세요?", "아뇨.", [["SYM:fever", "absent", "patient", "current"]]),
    ("열이나 오한은 없으셨어요?", "네, 없었어요.", [["SYM:fever", "absent", "patient", "current"], ["SYM:chills", "absent", "patient", "current"]]),
    ("숨이 차세요?", "계단 오를 때만 좀 차요.", [["SYM:exertional_dyspnea", "present", "patient", "current"]]),
    ("숨이 차세요?", "아니요, 숨은 괜찮아요.", [["SYM:dyspnea", "absent", "patient", "current"]]),
    ("가슴이 아프세요?", "네, 가슴 한가운데가 쥐어짜듯이 아파요.", [["SYM:chest_pain", "present", "patient", "current"], ["QUAL:squeezing", "present", "patient", "current"]]),
    ("구토는 하셨어요?", "두 번 토했어요.", [["SYM:vomiting", "present", "patient", "current"]]),
    ("구토는 하셨어요?", "토하지는 않았는데 속이 메스꺼워요.", [["SYM:vomiting", "absent", "patient", "current"], ["SYM:nausea", "present", "patient", "current"]]),
    ("설사는 없으세요?", "네, 설사는 없어요.", [["SYM:diarrhea", "absent", "patient", "current"]]),
    ("어지러우세요?", "잘 모르겠어요.", [["SYM:dizziness", "uncertain", "patient", "current"]]),
    ("머리가 아프세요?", "글쎄요, 약간 무거운 느낌은 있어요.", [["SYM:headache", "uncertain", "patient", "current"]]),
    ("소변 볼 때 아프세요?", "네, 따끔거려요.", [["SYM:dysuria", "present", "patient", "current"]]),
    ("소변 볼 때 아프세요?", "아니요.", [["SYM:dysuria", "absent", "patient", "current"]]),
    ("가족 중에 당뇨 있는 분 계세요?", "네, 아버지가 당뇨가 있으세요.", [["HX:diabetes", "present", "family", "current"]]),
    ("가족 중에 당뇨 있는 분 계세요?", "아니요, 없어요.", [["HX:diabetes", "absent", "family", "current"]]),
    ("담배 피우세요?", "예전에 피웠는데 5년 전에 끊었어요.", [["HX:former_smoking", "present", "patient", "past"]]),
    ("담배 피우세요?", "아니요.", [["HX:smoking", "absent", "patient", "current"]]),
    ("술은 드세요?", "가끔 한두 잔 해요.", [["HX:alcohol", "present", "patient", "intermittent"]]),
    ("피가 섞인 가래가 나오나요?", "아니요, 그런 건 없어요.", [["SYM:hemoptysis", "absent", "patient", "current"]]),
    ("체중이 줄었나요?", "네, 한 달에 5킬로 정도 빠졌어요.", [["SYM:weight_loss", "present", "patient", "current"]]),
    ("밤에 식은땀 나세요?", "가끔요.", [["SYM:night_sweats", "present", "patient", "intermittent"]]),
    ("다리가 붓나요?", "아니요, 붓진 않아요.", [["SIGN:leg_edema", "absent", "patient", "current"]]),
    ("갑자기 시작됐나요?", "네, 갑자기 그랬어요.", [["QUAL:sudden_onset", "present", "patient", "current"]]),
    ("목이 뻣뻣하세요?", "아뇨, 목은 괜찮아요.", [["SIGN:neck_stiffness", "absent", "patient", "current"]]),
    ("입맛은 어떠세요?", "입맛이 하나도 없어요.", [["SYM:anorexia", "present", "patient", "current"]]),
    ("입맛은 어떠세요?", "잘 먹어요.", [["SYM:anorexia", "absent", "patient", "current"]]),
    ("잠은 잘 주무세요?", "아니요, 통 못 자요.", [["SYM:insomnia", "present", "patient", "current"]]),
    ("피부에 발진 같은 게 있나요?", "기억이 잘 안 나요.", [["SYM:rash", "uncertain", "patient", "current"]]),
    ("황달은 없으셨어요?", "아니요, 눈이 좀 노랬어요.", [["SYM:jaundice", "present", "patient", "current"]]),
    ("복통이 있으세요?", "배는 안 아파요.", [["SYM:abdominal_pain", "absent", "patient", "current"]]),
    ("두근거림이 있나요?", "네, 가끔 심장이 쿵쾅거려요.", [["SYM:palpitation", "present", "patient", "intermittent"]]),
    ("열은 몇 도까지 올랐나요?", "38.5도까지 올라갔어요.", [["SYM:fever", "present", "patient", "current", 38.5]]),
    ("실신한 적 있으세요?", "한 번도 없어요.", [["SYM:syncope", "absent", "patient", "past"]]),
    ("출혈 경향이 있나요?", "멍이 잘 들어요.", [["SYM:easy_bruising", "present", "patient", "current"]]),
    ("어머니도 비슷한 증상이 있으셨나요?", "네, 어머니도 편두통이 있으셨어요.", [["SYM:headache", "present", "family", "current"]]),
    ("복용 중인 약 있으세요?", "아니요, 먹는 약은 없어요.", [["HX:medication", "absent", "patient", "current"]]),
    ("알레르기 있으세요?", "페니실린에 알레르기가 있어요.", [["HX:allergy", "present", "patient", "current"]]),
    ("최근에 수술 받으셨어요?", "2주 전에 무릎 수술을 받았어요.", [["HX:recent_surgery", "present", "patient", "current"]]),
    ("오한도 있으세요?", "네, 으슬으슬 추워요.", [["SYM:chills", "present", "patient", "current"]]),
    ("열이 나세요?", "몸이 불덩이 같아요.", [["SYM:fever", "present", "patient", "current"]]),
    ("경련하셨어요?", "경련은 아니고 발작처럼 몸이 떨렸어요.", [["SYM:seizure", "absent", "patient", "current"], ["SYM:chills", "present", "patient", "current"]]),
    ("가래는요?", "누런 가래가 나와요.", [["SYM:sputum", "present", "patient", "current"]]),
]


def dev(n: int = 80, section: str | None = None, seed: int = 7) -> None:
    """Show a development sample (seed != SEED) and record its ids in data/labels/findings_dev_ids.json; sample()
    never puts these sentences in the gold draft, so the rules tuned on them are evaluated on held-out sentences."""
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines()]
    if section:
        rows = [r for r in rows if r["section"] == section]
    rng = random.Random(seed)
    rng.shuffle(rows)
    ids = set(json.loads(DEV_IDS.read_text(encoding="utf-8"))) if DEV_IDS.exists() else set()
    for r in rows[:n]:
        ids.add(r["id"])
        print(f"[{r['id']}] ({r['source']}) {r['text']}")
        for p in _pred(r["text"], r["source"], r["context"]):
            extra = f" ={p['value']}" if "value" in p else ""
            flags = ("H" if p.get("hedged") else "") + ("?" if p.get("hypothetical") else "")
            print(f"     {p['concept']} {p['polarity']} {p['subject']} {p['temporality']}{extra} {flags} «{p['span']}»")
    DEV_IDS.write_text(json.dumps(sorted(ids), ensure_ascii=False) + "\n", encoding="utf-8")


def sample() -> list[dict]:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines()]
    dev_ids = set(json.loads(DEV_IDS.read_text(encoding="utf-8"))) if DEV_IDS.exists() else set()
    rows = [r for r in rows if r["id"] not in dev_ids]
    rng = random.Random(SEED)
    quota = {"negation_list": 50, "negation": 35, "hedge": 35, "family": 35, "past": 35, "numbers": 40, "idiom": 30,
             "simile": 20, "hypothetical": 15, "exam_list": 30, "plain": 30}
    chosen: dict[str, dict] = {}
    for tag, n in quota.items():
        pool = [r for r in rows if tag in r["phenomena"] and r["id"] not in chosen and len(r["text"]) <= 220]
        rng.shuffle(pool)
        # numbers: keep a mix of patient speech and exam/test values
        if tag == "numbers":
            pool.sort(key=lambda r: r["section"] != "history")
            pool = pool[:12] + [r for r in pool[12:] if r["section"] != "history"]
            rng.shuffle(pool)
        for r in pool[:n]:
            r = dict(r)
            r["stratum"] = tag
            chosen[r["id"]] = r
    items = list(chosen.values())
    for i, (q, a, gold) in enumerate(YESNO):
        items.append({"id": f"synthetic:yn{i:02d}", "case": "synthetic", "section": "history", "source": "patient",
                      "key": "", "text": a, "context": {"question": q}, "phenomena": ["yesno"], "stratum": "yesno",
                      "auto": _pred(a, "patient", {"question": q}), "gold_written": gold})
    with open(DRAFT, "w", encoding="utf-8") as fh:
        for r in items:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(items)} draft items -> {DRAFT.relative_to(ROOT)}  ({dict(Counter(r['stratum'] for r in items))})")
    return items


def show(start: int = 0, n: int = 50, stratum: str | None = None) -> None:
    rows = [json.loads(line) for line in DRAFT.read_text(encoding="utf-8").splitlines()]
    if stratum:
        rows = [r for r in rows if r["stratum"] == stratum]
    for r in rows[start:start + n]:
        q = f" Q:{r['context']['question']}" if r.get("context") and r["context"].get("question") else ""
        print(f"[{r['id']}] ({r['source']},{r['stratum']}){q} {r['text']}")
        for p in r["auto"]:
            extra = f" ={p['value']}" if "value" in p else ""
            flags = ("H" if p.get("hedged") else "") + ("?" if p.get("hypothetical") else "")
            print(f"     {p['concept']} {p['polarity']} {p['subject']} {p['temporality']}{extra} {flags} «{p['span']}»")


# ------------------------------------------------------------------------------------------------ freeze
def _as_gold(p: dict) -> list:
    g = [p["concept"], p["polarity"], p["subject"], p["temporality"]]
    if "value" in p:
        g.append(p["value"])
    return g


def _review_to_corrections() -> dict:
    """data/labels/findings_gold_review.txt (hand review, shorthand) -> corrections dict (id -> "ok" | gold list)."""
    pol = {"P": "present", "A": "absent", "U": "uncertain"}
    items: dict = {}
    for line in REVIEW.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rid, _, val = line.partition("\t")
        val = val.strip()
        if val == "ok":
            items[rid] = "ok"
            continue
        gold = []
        for tok in [t.strip() for t in val.split(";") if t.strip()]:
            parts = tok.split("/")
            if LEXICON.concept(parts[0]) is None:
                raise SystemExit(f"review {rid}: unknown concept {parts[0]}")
            gold.append([parts[0], pol[parts[1]], parts[2] if len(parts) > 2 else "patient",
                         parts[3] if len(parts) > 3 else "current"])
        items[rid] = gold
    return {"reviewer": "knowledge-rag agent (claude-opus-5-5), hand review, 2026-09-27, no LLM call",
            "source": "data/labels/findings_gold_review.txt", "items": items}


def freeze() -> None:
    draft = [json.loads(line) for line in DRAFT.read_text(encoding="utf-8").splitlines()]
    corr = _review_to_corrections()
    CORRECTIONS.write_text(json.dumps(corr, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    items = corr.get("items", {})
    out, n_corr, missing = [], 0, 0
    for r in draft:
        auto = [_as_gold(p) for p in r["auto"]]
        if "gold_written" in r:
            gold = r["gold_written"]
            corrected = sorted(map(str, gold)) != sorted(map(str, auto))
        elif r["id"] in items:
            c = items[r["id"]]
            if c == "ok":
                gold, corrected = auto, False
            else:
                gold, corrected = c, True
        else:
            missing += 1
            continue
        n_corr += corrected
        spans = {p["concept"]: p["span"] for p in r["auto"]}
        out.append({"id": r["id"], "case": r["case"], "section": r["section"], "source": r["source"], "key": r["key"],
                    "text": r["text"], "context": r.get("context"), "phenomena": r["phenomena"], "stratum": r["stratum"],
                    "gold": gold, "spans": {g[0]: spans.get(g[0], "") for g in gold}, "auto_at_draft": auto,
                    "corrected": corrected})
    with open(GOLD, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} gold items ({n_corr} corrected by hand, {missing} draft items without a review entry)"
          f" -> {GOLD.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------------ eval
def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _items_prf(items: list[dict], pred_key: str, full: bool = False) -> dict:
    tp = fp = fn = 0
    for r in items:
        gold = Counter((g[0], g[1]) + ((g[2], g[3]) if full else ()) for g in r["gold"])
        pred = Counter((p[0], p[1]) + ((p[2], p[3]) if full else ()) for p in r[pred_key])
        tp += sum((gold & pred).values())
        fp += sum((pred - gold).values())
        fn += sum((gold - pred).values())
    return _prf(tp, fp, fn)


def _old_polarity(text: str, kw: str) -> dict:
    """What the scattered negation helpers say for one keyword occurrence: True = negated."""
    from doctor_agent.knowledge import clinical_rules
    from doctor_agent.safety import danger_gate, preconditions
    t = text.lower()
    k = kw.lower()
    if k not in t:
        return {}
    return {"clinical_rules.negated": clinical_rules.negated(t, k),
            "preconditions._neg": preconditions._neg(t, k),
            "danger_gate.polarity": danger_gate.polarity(text, kw) == "neg"}


def evaluate() -> dict:
    from doctor_agent.agent import grounding
    from doctor_agent.safety import preconditions
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]
    t0 = time.perf_counter()
    for r in gold:
        r["pred"] = [_as_gold(p) for p in _pred(r["text"], r["source"], r.get("context"))]
    dt = time.perf_counter() - t0
    res: dict = {"n_items": len(gold), "n_gold_findings": sum(len(r["gold"]) for r in gold),
                 "parse_ms_per_item": round(1000 * dt / max(1, len(gold)), 3)}
    res["overall"] = {"concept+polarity": _items_prf(gold, "pred"),
                      "concept+polarity+subject+temporality": _items_prf(gold, "pred", full=True),
                      "draft_auto_labels (before hand fixes)": _items_prf(gold, "auto_at_draft")}
    by = {}
    for tag in sorted({t for r in gold for t in r["phenomena"]} | {r["stratum"] for r in gold}):
        items = [r for r in gold if tag in r["phenomena"]]
        if items:
            by[tag] = {"n_items": len(items), **_items_prf(items, "pred"),
                       "full": _items_prf(items, "pred", full=True)["f1"]}
    res["by_phenomenon"] = by
    by_src = {}
    for src in ("patient", "exam", "test"):
        items = [r for r in gold if r["source"] == src]
        if items:
            by_src[src] = {"n_items": len(items), **_items_prf(items, "pred")}
    res["by_source"] = by_src

    # --- old logic vs new on the same gold findings -------------------------------------------------------
    pol = defaultdict(lambda: [0, 0])  # method -> [correct, total]
    fam = defaultdict(lambda: [0, 0])
    grd = defaultdict(lambda: [0, 0])
    for r in gold:
        pred = r["pred"]
        text = r["text"]
        cues = {f.concept: (f.cue, f.span) for f in nlpf.parse(text, r["source"], r.get("context"))}
        for g in r["gold"]:
            cid, gp, gs = g[0], g[1], g[2]
            if gp not in ("present", "absent"):
                continue
            kw = r["spans"].get(cid) or ""
            cue, span = cues.get(cid, ("", ""))
            # polarity: explicit mentions only (a surface keyword in the raw sentence; not measured values, "absence"
            # expressions or yes/no answers, which the old helpers never read), so every method sees the same items
            explicit = not cue.startswith(("value", "absence-form", "normal-word", "urine", "yes/no", "kb_tests"))
            if explicit and kw and kw.lower() in text.lower() and gs == "patient" and not r.get("context", {}) \
                    .get("question") if r.get("context") else explicit and kw and kw.lower() in text.lower() and gs == "patient":
                old = _old_polarity(text, kw)
                for name, neg in old.items():
                    pol[name][0] += (neg == (gp == "absent"))
                    pol[name][1] += 1
                newp = next((p[1] for p in pred if p[0] == cid), None)
                pol["nlp.parse"][0] += newp == gp
                pol["nlp.parse"][1] += 1
            # family/other attribution: a relative's finding must not count as the patient's
            if gs in ("family", "other") and gp == "present" and kw and kw.lower() in text.lower():
                from doctor_agent.knowledge.clinical_rules import contains_affirmed
                fam["clinical_rules.contains_affirmed (no subject handling)"][0] += not contains_affirmed(text, (kw.lower(),))
                fam["clinical_rules.contains_affirmed (no subject handling)"][1] += 1
                fam["preconditions._self_only + _affirmed"][0] += not preconditions._affirmed(preconditions._self_only(text.lower()), (kw.lower(),))
                fam["preconditions._self_only + _affirmed"][1] += 1
                fam["nlp.affirmed (patient only)"][0] += not nlpf.affirmed(text, cid, source=r["source"])
                fam["nlp.affirmed (patient only)"][1] += 1
            # grounding: the correct claim must ground, the contradicting claim must not
            if gs != "patient":
                continue
            c = LEXICON.concept(cid)
            if c is None or cid.startswith(("GRP:", "QUAL:")) or cid == "SYM:pain":
                continue
            label = c.ko
            right = label if gp == "present" else f"{label} 없음"
            wrong = f"{label} 없음" if gp == "present" else label
            ctx = r.get("context")
            evidence_text = text
            if ctx and ctx.get("question"):  # old grounding reads Q/A pairs; give it the pair
                ok_old_r = _ground_qa(grounding._parse_claim(right), [(ctx["question"], text)], gp == "absent")[0] \
                    or grounding.is_grounded(right, evidence_text)[0]
                ok_old_w = _ground_qa(grounding._parse_claim(wrong), [(ctx["question"], text)], gp != "absent")[0] \
                    or grounding.is_grounded(wrong, evidence_text)[0]
            else:
                ok_old_r = grounding.is_grounded(right, evidence_text)[0]
                ok_old_w = grounding.is_grounded(wrong, evidence_text)[0]
            ev = nlpf.parse(text, r["source"], ctx)
            ok_new_r = nlpf.match(right, ev)[0]
            ok_new_w = nlpf.match(wrong, ev)[0]
            grd["grounding.is_grounded: correct claim grounded"][0] += ok_old_r
            grd["grounding.is_grounded: correct claim grounded"][1] += 1
            grd["grounding.is_grounded: contradicting claim rejected"][0] += not ok_old_w
            grd["grounding.is_grounded: contradicting claim rejected"][1] += 1
            grd["nlp.match: correct claim grounded"][0] += ok_new_r
            grd["nlp.match: correct claim grounded"][1] += 1
            grd["nlp.match: contradicting claim rejected"][0] += not ok_new_w
            grd["nlp.match: contradicting claim rejected"][1] += 1
    acc = lambda d: {k: {"accuracy": round(v[0] / v[1], 3) if v[1] else None, "n": v[1]} for k, v in d.items()}
    res["vs_old"] = {"polarity_of_gold_findings": acc(pol), "family_not_attributed_to_patient": acc(fam),
                     "grounding_claims": acc(grd)}
    METRICS.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


def errors(n: int = 60, stratum: str | None = None) -> None:
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]
    k = 0
    for r in gold:
        if stratum and stratum not in r["phenomena"]:
            continue
        pred = [_as_gold(p) for p in _pred(r["text"], r["source"], r.get("context"))]
        g = Counter((x[0], x[1]) for x in r["gold"])
        p = Counter((x[0], x[1]) for x in pred)
        if g != p:
            print(f"[{r['id']}] {r['text']}")
            print("    missing:", sorted((g - p).elements()), " extra:", sorted((p - g).elements()))
            k += 1
            if k >= n:
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "freq", "dev", "sample", "show", "freeze", "metrics", "errors"])
    ap.add_argument("--section")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stratum")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract()
    elif args.cmd == "freq":
        freq()
    elif args.cmd == "dev":
        dev(args.n, args.section, args.seed)
    elif args.cmd == "sample":
        sample()
    elif args.cmd == "show":
        show(args.start, args.n, args.stratum)
    elif args.cmd == "freeze":
        freeze()
    elif args.cmd == "metrics":
        print(json.dumps(evaluate(), ensure_ascii=False, indent=1))
    elif args.cmd == "errors":
        errors(args.n, args.stratum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
