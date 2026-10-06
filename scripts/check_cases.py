"""Rule-based quality check for the augmented evaluation cases (data/cases_aug). No LLM or network calls.

The ~20 entries per case added by scripts/augment_cases.py --mode full were written by an LLM and are not
clinician-reviewed. This script flags, per case:

- leaks: the diagnosis / an alias (incl. English names and abbreviations) in any visible field (initial, history,
  exam, tests; keys and values); top-level fields that are neither visible nor in eval.llm_patient.HIDDEN_KEYS
- contradictions: vital signs / lab values in augmented entries that disagree with the original case (or with other
  augmented entries); measured fever vs "afebrile"; a finding present in one entry and denied in another;
  sex/age-inconsistent tests (pregnancy test in men, PSA in women, troponin in neonates, ...)
- implausible values: vitals/labs outside physiologic bounds (age-adjusted for infants), unit mismatches
- keyword-key problems (keyword simulator, eval/simulator.py): empty or overly generic search terms, short ASCII
  terms that are substrings of unrelated words, common requests hitting several entries, duplicate terms,
  entries that a normal request would hardly reach

Severity: "hard" = must be zero (asserted by tests/test_case_quality.py); "soft" = review list. Issues that only
involve original (non-augmented) facts are never hard, because those facts are not ours to change.

    python scripts/check_cases.py                         # check, print summary
    python scripts/check_cases.py --json out.json         # full report
    python scripts/check_cases.py --fix --report data/labels/case_quality_2026-09-27.json
        # apply the recorded fixes (FIXES + mechanical key clean-up) to data/cases_aug, write before/after report

Fixes only touch augmented entries: remove a contradictory/inconsistent augmented entry or clause, align an
augmented number with the original value, or drop a bad search term from an augmented key. Every edit is recorded.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "data/cases_aug"
VISIBLE = ("initial", "history", "exam", "tests")

try:  # keep in sync with the LLM patient (what it hides from the patient model)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from eval.llm_patient import HIDDEN_KEYS  # noqa: E402
except Exception:  # noqa: BLE001 — the checker must also run without the eval package
    HIDDEN_KEYS = {"diagnosis", "aliases", "must_check", "_note", "teaching_point", "category", "difficulty",
                   "source", "augmented", "augmented_full"}


# ───────────────────────────────────────────── case model ─────────────────────────────────────────────
class Entry:
    __slots__ = ("section", "key", "value", "aug")

    def __init__(self, section: str, key: str, value: str, aug: bool):
        self.section, self.key, self.value, self.aug = section, key, value, aug

    @property
    def where(self) -> str:
        return self.section if self.section == "initial" else f"{self.section}['{self.key}']"


def entries(case: dict) -> list[Entry]:
    aug = case.get("augmented") or {}
    out = [Entry("initial", "", str(case.get("initial", "")), False)]
    for sec in ("history", "exam", "tests"):
        added = set(aug.get(sec, []))
        for k, v in (case.get(sec) or {}).items():
            out.append(Entry(sec, k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), k in added))
    return out


def demographics(case: dict) -> tuple[str | None, float | None]:
    """(sex 'M'/'F'/None, age in years or None) from the initial line."""
    s = str(case.get("initial", ""))
    s = re.split(r"주호소|[.:]\s|\n", s)[0][:40]  # the demographic phrase only ('4개월 전부터' is not an age)
    sex = None
    if re.search(r"남성|남아|남자|소년|\bmale\b|\bman\b|\bboy\b", s, re.I):
        sex = "M"
    if re.search(r"여성|여아|여자|소녀|임신|산부|\bfemale\b|\bwoman\b|\bgirl\b", s, re.I):
        sex = "F"
    age = None
    if m := re.search(r"생후\s*(\d+)\s*(시간|일|주|개월)", s):
        n, u = int(m.group(1)), m.group(2)
        age = n / {"시간": 24 * 365, "일": 365, "주": 52, "개월": 12}[u]
    elif re.search(r"신생아|newborn|neonate", s, re.I):
        age = 0.01
    elif m := re.search(r"(\d+)\s*개월", s):
        age = int(m.group(1)) / 12
    elif m := re.search(r"(\d+)\s*세", s):
        age = float(m.group(1))
    elif m := re.search(r"(\d+)\s*대", s):
        age = int(m.group(1)) + 5
    elif re.search(r"십대", s):
        age = 16
    return sex, age


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _toks(key: str) -> list[str]:
    return [t.lower() for t in key.split("|")]


# ─────────────────────────────────────────────── leaks ───────────────────────────────────────────────
# generic words that may appear in diagnosis names without giving the answer away (augment_cases._GENERIC + more)
_GENERIC = {"급성", "만성", "원발성", "이차성", "증후군", "질환", "장애", "동맥염", "간염", "결핍", "결핍증", "감염", "현재",
            "삽화", "acute", "chronic", "primary", "secondary", "syndrome", "disease", "disorder", "deficiency", "type",
            "with", "the", "and", "due", "of", "without", "induced", "related", "associated", "infection", "benign",
            "malignant", "severe", "mild", "left", "right", "비정형", "atypical", "재발성", "선천성", "congenital",
            "후천성", "특발성", "idiopathic", "양성", "악성", "중증", "경증", "좌측", "우측", "단순", "복합", "유형", "형",
            "hereditary", "유전성", "가족성", "familial", "juvenile", "소아", "성인", "adult", "neonatal", "신생아"}


def leak_names(case: dict) -> tuple[list[str], list[str]]:
    """(full names, distinctive tokens) of the answer. Full names: diagnosis + aliases without parentheses."""
    raw = [str(case.get("diagnosis", "")), *map(str, case.get("aliases", []) or [])]
    full = set()
    for n in raw:
        paren = [x for x in re.findall(r"\((.*?)\)", n) if re.fullmatch(r"[A-Za-z][A-Za-z '\-]+", x.strip())]
        for variant in (re.sub(r"\(.*?\)", "", n), *paren):
            v = variant.strip()
            if len(_norm(v)) >= 3 or (len(_norm(v)) == 2 and not v.isascii()) or (len(v) == 2 and v.isupper()):
                full.add(v)
    toks = {t for n in raw for t in re.split(r"[\s,()/·\-']+", n)}
    toks = {t for t in toks if len(t) >= 3 and t.lower() not in _GENERIC and not t.isdigit()}
    return sorted(full), sorted(toks - full)


def _find_name(name: str, text: str) -> bool:
    if name.isascii():
        n = name.strip()
        if len(n) <= 3:  # abbreviations (MG, PE, MS) case-sensitively as whole words: 'mg/dL', '90 ms' are not leaks
            return re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", text) is not None
        if len(n) <= 5 or " " not in n:
            return re.search(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])", text.lower()) is not None
        return _norm(n) in _norm(text)
    return _norm(name) in _norm(text)


# diagnostic interpretation inside a result (the augmentation prompt forbade it; the patient must report findings)
_INTERPRET = re.compile(r"[가-힣A-Za-z]+\s*(?:에|과|와)\s*합당|합당한 소견|의심(?:됨|된다|되는|함)|시사(?:함|하는|됨)|"
                        r"consistent with|suggestive of|compatible with|diagnostic of", re.I)
_META = re.compile(r"기존\s*(진찰|소견|증례|검사|결과|기록)|증례에\s*(있|기술)|정답|진단명")


def check_leaks(case: dict, ents: list[Entry]) -> list[dict]:
    issues = []
    full, toks = leak_names(case)
    orig_text = " ".join(e.key + " " + e.value for e in ents if not e.aug)
    panel_keys = set(((case.get("augmented_full") or {}).get("panel") or {}).values())
    for e in ents:
        for part, text in (("key", e.key), ("value", e.value)):
            if part == "key" and e.key in panel_keys:  # curated search terms (augment_cases.PANEL), not generated
                continue
            hit = next((n for n in full if _find_name(n, text)), None)
            if hit:
                issues.append(_issue("leak", "hard" if e.aug else "soft", e, f"{part} names the diagnosis: '{hit}'",
                                     origin="aug" if e.aug else "orig"))
                continue
            if e.aug:
                t = next((t for t in toks if _find_name(t, text) and not _find_name(t, orig_text)), None)
                if t:
                    issues.append(_issue("leak_token", "soft", e, f"{part} contains answer token '{t}'"))
        for c in (_clauses(e.value) if e.aug else []):
            if (m := _INTERPRET.search(c)) and not re.search(r"없|않|음성|negative|\bno\b", c[m.end():], re.I):
                issues.append(_issue("interpretive", "soft", e, f"diagnostic interpretation: '{c.strip()[:80]}'"))
        if e.aug and (m := _META.search(e.value)):
            issues.append(_issue("meta_reference", "hard", e, f"refers to the case file: '{m.group(0)}'"))
    for k in case:
        if k not in VISIBLE and k not in HIDDEN_KEYS:
            issues.append({"type": "hidden_key_missing", "severity": "hard", "where": k,
                           "detail": f"top-level field '{k}' is shown to the LLM patient (not in HIDDEN_KEYS)"})
    return issues


def _issue(typ: str, sev: str, e: Entry | None, detail: str, **kw) -> dict:
    d = {"type": typ, "severity": sev, "detail": detail}
    if e is not None:
        d.update(where=e.where, section=e.section, key=e.key, aug=e.aug)
    d.update(kw)
    return d


# ─────────────────────────────────────────── structure ───────────────────────────────────────────
def check_structure(case: dict) -> list[dict]:
    issues = []
    aug = case.get("augmented") or {}
    for sec in ("exam", "tests"):
        for k in aug.get(sec, []):
            if k not in (case.get(sec) or {}):
                issues.append({"type": "augmented_list_mismatch", "severity": "hard", "where": f"augmented.{sec}",
                               "detail": f"listed key not in {sec}: '{k[:60]}'"})
        for k, v in (case.get(sec) or {}).items():
            if not isinstance(v, str) or not v.strip() or v.strip().lower() in ("null", "none", "n/a"):
                issues.append({"type": "empty_value", "severity": "hard", "where": f"{sec}['{k}']",
                               "detail": f"empty/null value: {v!r}"})
    return issues


# ─────────────────────────────────────── numbers: vitals and labs ───────────────────────────────────────
NUM = r"(\d[\d,]*(?:\.\d+)?)"

# excluded contexts for blood lab values (other specimens, other time points)
_OTHER_SPECIMEN = re.compile(r"소변|urine|요검사|요\s*(?:나트륨|칼륨|단백|크레아티닌)|뇌척수액|csf|요추\s*천자|lumbar|흉수|늑막액|pleural|"
                             r"복수\s*(?:검사|천자|분석)|복수액|ascit|관절액|관절\s*천자|synovial|객담|sputum|대변|stool|기관지|bal\b|"
                             r"심낭액|pericardial fluid|제대혈|양수|24시간|청소율|clearance|/hpf|배양|도자|catheter|혼합\s*정맥|정맥혈|venous|"
                             r"제대|umbilical|태아|fetal|모체|산모|maternal", re.I)
# positional / limb-specific blood pressure (four-limb BP, orthostatic) is a different measurement
_POSITIONAL = re.compile(r"하지|다리|발목|대퇴|leg|ankle|lower extremit|사지|팔|arm|좌측|우측|left|right|기립|누운|앉은|"
                         r"standing|supine|sitting|orthostatic|체위", re.I)
_TIMEPOINT = re.compile(r"이후|후에|후\s|추적|재검|입원\s*\d|\d+\s*(?:일|시간|주)\s*(?:후|째|뒤)|치료\s*후|수액\s*후|투여\s*후|"
                        r"이전|과거|전에|전\s|내원\s*전|\d+\s*(?:년|개월)\s*전|기저|평소|baseline|previous|prior|after|"
                        r"follow|repeat|식후|부하|ogtt|당부하|최저|최고|nadir|peak|변화|상승\s*추세|감소\s*추세|→|->", re.I)


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"(?<!\d)[.;\n](?!\d)|,\s|\s/\s|\(|\)", text) if c.strip()]


# vital signs: name -> (regex, hard tolerance, soft tolerance)
VITALS = {
    "sbp": (re.compile(r"(\d{2,3})\s*/\s*\d{2,3}\s*mm\s*hg", re.I), 20, 6),
    "dbp": (re.compile(r"\d{2,3}\s*/\s*(\d{2,3})\s*mm\s*hg", re.I), 15, 6),
    "hr": (re.compile(r"(?:맥박|심박수|심박|pulse|heart rate|\bhr\b)\s*(?:수)?\s*[:：]?\s*(?:약\s*)?(\d{2,3})\s*(?:회|bpm|/\s*min|/\s*분)", re.I), 25, 10),
    "rr": (re.compile(r"(?:호흡수|호흡\s*횟수|respiratory rate|\brr\b)\s*[:：]?\s*(?:약\s*)?(\d{1,3})\s*(?:회|/\s*min|/\s*분|breaths)", re.I), 8, 3),
    "temp": (re.compile(r"(\d{2}(?:\.\d{1,2})?)\s*(?:°\s*c\b|℃|도\s*(?:c\b|씨)?)", re.I), 0.8, 0.3),
    "spo2": (re.compile(r"(?:산소\s*포화도|spo2|sao2|sp02|o2\s*sat\w*)\s*(?:는|은)?\s*[:：]?\s*(?:약\s*)?(\d{2,3})\s*%", re.I), 4, 2),
}
_FAHRENHEIT = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d{1,2})?)\s*°\s*f\b", re.I)

# labs: name -> (regex capturing number [+ optional unit], normaliser(value, unit) -> canonical, rel. tolerance hard/soft,
#                absolute epsilon, plausible range in canonical units, allowed unit pattern)
def _thousands(v: float, _u: str) -> float:
    return v * 1000 if v < 2000 else v


def _crp(v: float, u: str) -> float | None:
    return v / 10 if "l" == u.lower()[-1:] and "dl" not in u.lower() else (v if u else None)


LABS = {
    "hb": (r"(?<!당화)(?<!당화 )(?:(?<![a-z])hb(?![a-z0-9])|(?<![a-z])hgb|헤모글로빈|혈색소|hemoglobin)(?!\s*a1c)\s*(?:수치)?\s*[:：]?\s*(?:약\s*)?"
           + NUM + r"\s*(g/dl|g/l|mg/dl)?",
           lambda v, u: v / 10 if u.lower() == "g/l" else v, 0.15, 0.08, 0.6, (2, 25), r"g/dl|g/l"),
    "wbc": (r"(?:(?<![a-z])wbc|백혈구)(?:\s*수)?\s*[:：]?\s*(?:약\s*)?" + NUM + r"(?!\s*[-~]\s*\d)(?!\s*/\s*hpf)",
            _thousands, 0.25, 0.1, 600, (100, 600000), None),
    "plt": (r"(?:혈소판|(?<![a-z])plt|platelets?)(?:\s*수)?\s*[:：]?\s*(?:약\s*)?" + NUM, _thousands, 0.25, 0.1, 15000,
            (1000, 2500000), None),
    "na": (r"(?:(?<![a-z])na(?![a-z])|나트륨|sodium)\s*[:：]?\s*" + NUM + r"\s*(meq/l|mmol/l|mg/dl)?",
           lambda v, u: v, 0.03, 0.015, 2.5, (100, 190), r"meq/l|mmol/l"),
    "k": (r"(?:(?<![a-z])k(?![a-z+])|칼륨|potassium)\s*[:：]?\s*" + NUM + r"\s*(meq/l|mmol/l|mg/dl)?",
          lambda v, u: v, 0.12, 0.05, 0.35, (1.0, 10), r"meq/l|mmol/l"),
    "cr": (r"(?:(?<![a-z])cr(?![a-z])|크레아티닌|creatinine)(?!\s*(?:청소율|clearance|kinase))\s*[:：]?\s*" + NUM
           + r"\s*(mg/dl|μmol/l|umol/l|µmol/l)?",
           lambda v, u: v / 88.4 if ("mol" in u.lower() or (not u and v > 25)) else v, 0.2, 0.1, 0.25, (0.1, 30), None),
    "bun": (r"(?:(?<![a-z])bun|요소질소|혈중요소질소)\s*[:：]?\s*" + NUM, lambda v, u: v, 0.25, 0.1, 5, (1, 300), None),
    "glucose": (r"(?<!식후 )(?<!2시간 )(?:공복\s*혈당|혈당|혈청\s*포도당|포도당|(?<![a-z])glucose|(?<![a-z])fbs)\s*(?:수치)?\s*[:：]?\s*" + NUM
                + r"\s*(mg/dl|mmol/l)?",
                lambda v, u: v * 18 if ("mmol" in u.lower() or (not u and v < 35)) else v, 0.35, 0.12, 25, (10, 2500), None),
    "ast": (r"(?<![a-z])(?:ast|sgot)(?![a-z/])\s*[:：]?\s*" + NUM, lambda v, u: v, 0.3, 0.12, 12, (1, 30000), None),
    "alt": (r"(?<![a-z])(?:alt|sgpt)(?![a-z])\s*[:：]?\s*" + NUM, lambda v, u: v, 0.3, 0.12, 12, (1, 30000), None),
    "alp": (r"(?<![a-z])(?:alp|알칼리성?\s*인산분해효소|alkaline phosphatase)(?![a-z])\s*[:：]?\s*" + NUM,
            lambda v, u: v, 0.3, 0.12, 20, (5, 6000), None),
    "tbil": (r"(?<!직접 )(?<!간접 )(?<!직접)(?<!간접)(?:총\s*빌리루빈|total bilirubin|t-?bil|(?<![a-z])빌리루빈)\s*[:：]?\s*" + NUM,
             lambda v, u: v, 0.3, 0.12, 0.5, (0, 70), None),
    "crp": (r"(?<![a-z])(?:hs-?)?crp\s*[:：]?\s*" + NUM + r"\s*(mg/dl|mg/l)?", _crp, 0.35, 0.15, 0.5, (0, 60), None),
    "esr": (r"(?:(?<![a-z])esr|적혈구\s*침강\s*속도|혈침)\s*[:：]?\s*" + NUM, lambda v, u: v, 0.35, 0.15, 8, (0, 160), None),
    "ca": (r"(?<!이온화 )(?<!이온화)(?:(?<![a-z])ca(?![a-z0-9])|칼슘|calcium)(?!\s*(?:19|125|15|72|242)\s*-?\s*\d*)\s*[:：]?\s*"
           + NUM + r"\s*(mg/dl|mmol/l)?", lambda v, u: v * 4 if "mmol" in u.lower() else v, 0.1, 0.05, 0.5, (3, 20), None),
    "albumin": (r"(?<!미세)(?<!미세 )(?:알부민|albumin)\s*[:：]?\s*" + NUM + r"\s*(g/dl|g/l)?",
                lambda v, u: v / 10 if u.lower() == "g/l" else v, 0.15, 0.08, 0.3, (0.5, 7), None),
    "tsh": (r"(?<![a-z])tsh\s*[:：]?\s*" + NUM, lambda v, u: v, 0.4, 0.2, 0.5, (0, 1000), None),
    "hba1c": (r"(?:hba1c|당화\s*혈색소|a1c)\s*[:：]?\s*" + NUM, lambda v, u: v, 0.08, 0.04, 0.3, (3, 20), None),
    "inr": (r"(?<![a-z])inr\s*[:：]?\s*" + NUM, lambda v, u: v, 0.2, 0.1, 0.15, (0.5, 15), None),
    "lipase": (r"(?:리파아제|리파제|(?<![a-z])lipase)\s*[:：]?\s*" + NUM, lambda v, u: v, 0.4, 0.2, 20, (0, 100000), None),
    "amylase": (r"(?:아밀라아제|아밀라제|(?<![a-z])amylase)\s*[:：]?\s*" + NUM, lambda v, u: v, 0.4, 0.2, 20, (0, 100000), None),
    "ph": (r"(?<![a-z])ph\s*[:：]?\s*(7\.\d+|6\.\d+)", lambda v, u: v, 0.012, 0.004, 0.03, (6.6, 7.9), None),
}
LABS_RE = {k: re.compile(v[0], re.I) for k, v in LABS.items()}


def _values(e: Entry, age: float | None) -> list[tuple[str, float, str, str]]:
    """[(analyte, canonical value, raw match, clause)] from an entry's value text (blood labs + vitals)."""
    out = []
    key_other = bool(_OTHER_SPECIMEN.search(e.key)) and e.section == "tests"
    for clause in _clauses(e.value):
        other = key_other or bool(_OTHER_SPECIMEN.search(clause))
        timed = bool(_TIMEPOINT.search(clause))
        for name, (rx, *_rest) in VITALS.items():
            if name == "spo2" and re.search(r"도자|catheter|정맥|venous|심방|심실|대동맥|폐동맥", e.key + clause, re.I):
                continue
            for m in rx.finditer(clause):
                v = _f(m.group(1))
                if name == "temp" and not 30 <= v <= 45:
                    continue
                pos = name in ("sbp", "dbp") and bool(_POSITIONAL.search(clause + " " + e.key))
                out.append((name, v, v, m.group(0), clause, timed or pos))
        if other:
            continue
        for name, rx in LABS_RE.items():
            if name == "ph" and re.search(r"소변|urine|요", clause):
                continue
            for m in rx.finditer(clause):
                try:
                    v = _f(m.group(1))
                except ValueError:
                    continue
                unit = (m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else "") or ""
                cv = LABS[name][1](v, unit)
                if cv is None:
                    continue
                hi = cv
                if r := re.match(r"\s*[-~–]\s*(\d[\d,]*(?:\.\d+)?)", clause[m.end(1):]):  # reported range 131-137
                    hi = LABS[name][1](_f(r.group(1)), unit) or cv
                out.append((name, cv, hi, m.group(0), clause, timed))
    return out


def check_numbers(case: dict, ents: list[Entry]) -> list[dict]:
    issues = []
    _sex, age = demographics(case)
    vals = {id(e): _values(e, age) for e in ents}
    # plausibility and units
    for e in ents:
        sev = "hard" if e.aug else "soft"
        for name, v, hi, raw, clause, _t in vals[id(e)]:
            rng = LABS[name][5] if name in LABS else None
            if name in VITALS:
                rng = {"sbp": (40, 300), "dbp": (10, 200), "hr": (20, 300), "rr": (4, 100), "temp": (30, 44),
                       "spo2": (40, 100)}[name]
            if rng and not (rng[0] <= v <= rng[1] and rng[0] <= hi <= rng[1]):
                issues.append(_issue("implausible_value", sev, e, f"{name}={v:g} outside {rng} in '{raw}'"))
            if name in LABS and LABS[name][6]:
                unit = re.search(r"(meq/l|mmol/l|mg/dl|g/dl|g/l)\s*$", raw, re.I)
                if unit and not re.fullmatch(LABS[name][6], unit.group(1), re.I):
                    issues.append(_issue("unit_mismatch", sev, e, f"{name} with unit '{unit.group(1)}' in '{raw}'"))
        for m in re.finditer(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg", e.value, re.I):
            if int(m.group(2)) >= int(m.group(1)):
                issues.append(_issue("implausible_value", sev, e, f"diastolic >= systolic in '{m.group(0)}'"))
        cels = [v for n, v, _h, _r, _c, _t in vals[id(e)] if n == "temp"]
        for m in _FAHRENHEIT.finditer(e.value):
            c = (float(m.group(1)) - 32) * 5 / 9
            if not cels:
                issues.append(_issue("unit_mismatch", "soft", e, f"temperature only in Fahrenheit: '{m.group(0)}'"))
            elif min(abs(c - x) for x in cels) > 0.4:
                issues.append(_issue("vital_conflict", sev, e, f"temp: '{m.group(0)}' = {c:.1f}°C but Celsius value(s) "
                                                               f"{cels} in the same entry", analyte="temp"))
        if e.aug and age is not None and e.key and re.search(r"활력|vital", e.key, re.I):
            for name, v, _hi, raw, _c, _t in vals[id(e)]:
                bad = None
                if age < 1 and name == "hr" and not 80 <= v <= 220:
                    bad = "infant heart rate"
                if age < 1 and name == "rr" and not 20 <= v <= 80:
                    bad = "infant respiratory rate"
                if age < 1 and name == "sbp" and v > 115:
                    bad = "infant systolic BP"
                if 1 <= age < 6 and name == "hr" and not 60 <= v <= 190:
                    bad = "child heart rate"
                if bad:
                    issues.append(_issue("implausible_value", "hard", e, f"{bad} {v:g} ('{raw}') for age {age:.2f}y"))
    # conflicts: every augmented value must agree with at least one original value (or reported range) of the same
    # quantity; with no original value, augmented entries must agree with each other. Time-qualified clauses skipped.
    orig = collections.defaultdict(list)
    augv = collections.defaultdict(list)
    for e in ents:
        for name, v, hi, raw, clause, timed in vals[id(e)]:
            if timed:
                continue
            (augv if e.aug else orig)[name].append((v, hi, raw, e))

    def tol(name: str, a: float, lo: float, hi: float) -> str | None:
        b = min(max(a, lo), hi)  # nearest point of the reference value/range
        if name in VITALS:
            hard, soft = VITALS[name][1], VITALS[name][2]
            d = abs(a - b)
            if name == "temp" and ((a >= 38.0 and b < 37.5) or (b >= 38.0 and a < 37.5)):
                return "hard"
            return "hard" if d > hard else "soft" if d > soft else None
        _rx, _n, hard, soft, eps, *_ = LABS[name]
        d = abs(a - b)
        if d <= eps:
            return None
        rel = d / max(abs(a), abs(b), 1e-9)
        return "hard" if rel > hard else "soft" if rel > soft else None

    rank = {None: 0, "soft": 1, "hard": 2}
    for name, avals in augv.items():
        ref = orig.get(name)
        typ = "vital_conflict" if name in VITALS else "lab_conflict"
        for v, _hi, raw, e in avals:
            if ref:
                sev = min((tol(name, v, lo, hi) for lo, hi, _r, _e in ref), key=lambda s: rank[s])
                if sev:
                    other = min(ref, key=lambda r: abs(r[0] - v))
                    issues.append(_issue(typ, sev, e, f"{name}: augmented '{raw}' vs original '{other[2]}' in "
                                                      f"{other[3].where}", analyte=name, orig_value=other[0], aug_value=v))
            else:
                for i2, (v2, hi2, raw2, e2) in enumerate(avals):
                    if e2 is e or i2 <= next(i for i, x in enumerate(avals) if x[3] is e and x[2] == raw):
                        continue
                    sev = tol(name, v, v2, hi2)
                    if sev:
                        issues.append(_issue(typ, sev, e, f"{name}: '{raw}' vs '{raw2}' in {e2.where} (both augmented)",
                                             analyte=name))
    return issues


# ────────────────────────────────────── findings: present vs denied ──────────────────────────────────────
FINDINGS = {
    "jaundice": r"황달|icterus|jaundice|황染",
    "hepatomegaly": r"간\s*비대|간\s*종대|hepatomegaly",
    "splenomegaly": r"비장\s*비대|비장\s*종대|비종대|비\s*종대|splenomegaly",
    "leg_edema": r"(?:하지|다리|발목|정강이|양측 하지|양하지)\s*(?:에\s*)?(?:함요\s*|요흔\s*|요함\s*)?부종|pitting edema|peripheral edema|leg edema",
    "murmur": r"심잡음|murmur",
    "crackles": r"수포음|crackle|rales",
    "wheeze": r"천명음|wheez",
    "neck_stiffness": r"경부\s*강직|목\s*경직|항부\s*강직|nuchal rigidity|neck stiffness",
    "rebound": r"반발\s*압통|반발통|rebound",
    "ascites": r"복수\s*(?:가|는|소견|저류|없|있|소량|중등도|다량|관찰|동반)|ascites|복강\s*내\s*액체",
    "clubbing": r"곤봉지|clubbing",
    "papilledema": r"유두\s*부종|papilledema",
    "cardiomegaly": r"심비대|심장\s*비대|심장\s*크기\s*(?:증가|확대)|cardiomegaly|심장\s*음영\s*확대",
    "pleural_effusion": r"흉수|늑막\s*삼출|흉막\s*삼출|pleural effusion",
    "st_elevation": r"st\s*분절\s*상승|st\s*상승|st[- ]elevation",
    "pallor": r"결막\s*(?:이\s*|은\s*)?창백|conjunctival pallor",
    "lymphadenopathy": r"림프절\s*(?:종대|비대|병증)|lymphadenopathy",
    "blood_culture_growth": r"혈액\s*배양|blood culture",
}
_NEG = re.compile(r"(?:관찰|보이|보|촉지|촉진|청진|들리|확인|발견|동반|나타나|감지|측정)\S{0,3}\s*(?:않|못)|없|음성|않|아니|정상|무\b|negative|absent|\bno\b|none|not\s|unremarkable|배제|관찰\s*안\s*됨|미검출|no growth", re.I)
_POS = re.compile(r"소견\s*$|외에|외\s|있|관찰|보임|보이|촉지|양성|증가|상승|확인|동반|들림|청진됨|positive|present|발생|나타|심함|현저|경미|\d\+|"
                  r"자라|자람|동정|검출|성장|grew|growth of", re.I)
_PRE_NEG = re.compile(r"(?:\bno|without|negative for|absence of|무)\s*$", re.I)


def _polarity(clause: str, m: re.Match) -> str | None:
    if _PRE_NEG.search(clause[max(0, m.start() - 20):m.start()]):
        return "neg"
    tail = clause[m.end():]
    n, p = _NEG.search(tail), _POS.search(tail)
    if n and (not p or n.start() <= p.start()):
        return "neg"
    if p:
        return "pos"
    return None


def finding_polarities(e: Entry) -> list[tuple[str, str, str]]:
    """[(finding, 'pos'|'neg', clause)] — a short clause without a predicate takes the polarity of the next clause in
    the same sentence ('간비대, 비장비대 없음' → both denied)."""
    out = []
    text = re.sub(r"그람\s*[음양]성|gram[- ]?(?:negative|positive)", "그람균", e.value, flags=re.I)
    text = re.sub(r"\((?:[^()]{0,40})\)", " ", text)  # parenthesised translations / reference ranges
    for sent in re.split(r"(?<!\d)[.;\n](?!\d)", text):
        parts = [c for c in re.split(r",\s*|\s*/\s*|·|및|그리고|이나\s|으나\s|지만\s|이며\s|으며\s|하며\s|고\s", sent) if c.strip()]
        pending, nxt = [], None
        for c in reversed(parts):
            found = []
            for f, rx in FINDINGS.items():
                for m in re.finditer(rx, c, re.I):
                    found.append((f, _polarity(c, m)))
            own = None
            if _NEG.search(c) or _POS.search(c):
                n, p = _NEG.search(c), _POS.search(c)
                own = "neg" if n and (not p or n.start() <= p.start()) else "pos"
            for f, pol in found:
                pol = pol or (nxt if len(c.strip()) <= 14 else None) or (None if nxt else "pos")
                if pol:
                    out.append((f, pol, c.strip()))
            nxt = own or nxt
    return out


# patient-language findings in the history (the patient's own report counts as "present")
HISTORY_FINDINGS = {
    "leg_edema": r"(?:다리|발목|발등|종아리|정강이)\S{0,2}\s*(?:가\s*|이\s*|도\s*)?(안\s*|전혀\s*)?(?:붓|부었|부어|부종)",
    "jaundice": r"(?:눈|흰자|피부|얼굴|몸)\S{0,2}\s*(?:이\s*|가\s*|도\s*)?(안\s*)?(?:노랗|노래)|황달",
}


def history_polarities(e: Entry) -> list[tuple[str, str, str]]:
    out = []
    for sent in re.split(r"[.?!\n]", e.value):
        for f, rx in HISTORY_FINDINGS.items():
            for m in re.finditer(rx, sent):
                tail = sent[m.end():]
                window = tail if re.match(r"\S{0,2}거나", tail) else tail[:12]  # '다리 붓거나 ... 적은 없어요'
                neg = (m.lastindex and m.group(1)) or re.search(r"않|없|아니|안\s", window)
                out.append((f, "neg" if neg else "pos", sent.strip()))
    return out


def check_findings(ents: list[Entry]) -> list[dict]:
    issues = []
    pol = {id(e): finding_polarities(e) for e in ents if e.section in ("exam", "tests", "initial")}
    pol.update({id(e): history_polarities(e) for e in ents if e.section == "history"})
    by = collections.defaultdict(list)
    for e in ents:
        for f, p, c in pol.get(id(e), []):
            if not _TIMEPOINT.search(c):
                by[f].append((p, c, e))
    for f, items in by.items():
        pos = [x for x in items if x[0] == "pos"]
        neg = [x for x in items if x[0] == "neg"]
        seen = set()
        for pp, pc, pe in pos:
            for np_, nc, ne in neg:
                if pe is ne or not (pe.aug or ne.aug) or (id(pe), id(ne)) in seen:
                    continue
                seen.add((id(pe), id(ne)))
                aug_e = ne if ne.aug else pe
                issues.append(_issue("finding_conflict", "soft", aug_e, f"{f}: present in {pe.where} ('{pc[:60]}') "
                                     f"but denied in {ne.where} ('{nc[:60]}')", finding=f))
    return issues


# ─────────────────────────────────────────────── fever ───────────────────────────────────────────────
_FEVER_NEG = re.compile(r"(?:(?<![가-힣])열|발열|고열|미열)\s*(?:은|이|도|감)?\s*(?:전혀\s*)?(?:없|안\s*(?:났|나|남)|나지\s*않|안\s*올)|"
                        r"afebrile|무열|no fever|발열\s*(?:소견\s*)?(?:은\s*)?없", re.I)
_FEVER_POS = re.compile(r"(?:(?<![가-힣])열|발열|고열|미열)\s*(?:이|도|은)?\s*(?:났|나(?:요|고|서)|있|올라|오르|심|계속)|"
                        r"\bfever\b|\bfebrile\b|고열", re.I)


def check_fever(case: dict, ents: list[Entry]) -> list[dict]:
    issues = []
    temps = {id(e): [(v, raw, c) for n, v, _hi, raw, c, t in _values(e, None) if n == "temp" and not t] for e in ents}
    o_hot = [(v, raw, e) for e in ents if not e.aug for v, raw, c in temps[id(e)] if v >= 38.0]
    o_cold = [(v, raw, e) for e in ents if not e.aug for v, raw, c in temps[id(e)] if v < 37.5]
    o_denied = [(m.group(0), e) for e in ents if not e.aug and e.section in ("initial", "history")
                for m in [_FEVER_NEG.search(e.value)] if m]
    for e in ents:
        if not e.aug:
            continue
        m = _FEVER_NEG.search(e.value)
        if m and o_hot and not temps[id(e)]:
            v, raw, oe = o_hot[0]
            issues.append(_issue("fever_conflict", "hard", e, f"says '{m.group(0)}' but {oe.where} measured '{raw}'"))
        for v, raw, _c in temps[id(e)]:
            if v >= 38.0 and o_denied and not o_hot:
                issues.append(_issue("fever_conflict", "soft", e, f"measured '{raw}' but patient denies fever "
                                     f"('{o_denied[0][0]}' in {o_denied[0][1].where})"))
            if v >= 38.0 and o_cold and not o_hot:
                pass  # covered by vital_conflict (temp crossing the fever threshold is hard there)
    return issues


# ──────────────────────────────────────── sex/age-inconsistent tests ────────────────────────────────────────
_PREG = re.compile(r"임신\s*(?:반응\s*)?검사|hcg|pregnancy", re.I)
_MALE_ONLY = re.compile(r"psa|전립선|prostat|고환|음낭|testic|scrot", re.I)
_FEMALE_ONLY = re.compile(r"자궁|난소|질경|질\s*분비물|골반\s*(?:내진|진찰)|내진|pap\s*(?:smear|test)|자궁경부|transvaginal|"
                          r"질식\s*초음파|경질\s*초음파|유방\s*촬영|mammogra", re.I)
_FEMALE_ORGAN = re.compile(r"자궁|난소|질\s*분비물|nabothian|uterus|uterine|ovar", re.I)
_MALE_ORGAN = re.compile(r"전립선|고환|음낭|prostat|testic|testis|scrot", re.I)
_TROPONIN = re.compile(r"트로포닌|troponin|심근\s*효소|ck-mb", re.I)


def check_demographics(case: dict, ents: list[Entry]) -> list[dict]:
    sex, age = demographics(case)
    dx = str(case.get("diagnosis", "")) + " " + " ".join(map(str, case.get("aliases", []) or []))
    issues = []
    for e in ents:
        if e.section != "tests" and e.section != "exam":
            continue
        sev = "hard" if e.aug else "soft"
        k = e.key
        if _PREG.search(k):
            if sex == "M":
                issues.append(_issue("demographic_test", sev, e, "pregnancy test in a male patient"))
            elif age is not None and age < 10:
                issues.append(_issue("demographic_test", sev, e, f"pregnancy test at age {age:.2f}y"))
            elif age is not None and age >= 60:
                issues.append(_issue("demographic_test", sev, e, f"pregnancy test at age {age:.0f}y"))
        if sex == "F" and _MALE_ONLY.search(k):
            issues.append(_issue("demographic_test", sev, e, "male-only exam/test in a female patient"))
        if sex == "M" and _FEMALE_ONLY.search(k) and not re.search(r"유방", k):
            issues.append(_issue("demographic_test", sev, e, "female-only exam/test in a male patient"))
        if _TROPONIN.search(k) and age is not None and age < 1 and not re.search(r"심|cardi|myocard", dx, re.I):
            # neonates / young infants (< ~5 weeks): not part of a routine work-up for a non-cardiac problem
            issues.append(_issue("demographic_test", "hard" if (e.aug and age < 0.1) else "soft", e,
                                 f"troponin/cardiac enzymes at age {age * 365:.0f} days (non-cardiac diagnosis)"))
        # organs of the other sex described in an augmented result (e.g. '자궁 및 난소 정상' for a man)
        if e.aug and sex == "M" and (m := _FEMALE_ORGAN.search(e.value)):
            issues.append(_issue("demographic_test", "hard", e, f"female organ '{m.group(0)}' in a male patient's result"))
        if e.aug and sex == "F" and (m := _MALE_ORGAN.search(e.value)):
            issues.append(_issue("demographic_test", "hard", e, f"male organ '{m.group(0)}' in a female patient's result"))
    return issues


# ─────────────────────────────────────────────── keys ───────────────────────────────────────────────
# search terms that must not stand alone (augment_cases._GENERIC_TOKENS)
GENERIC_TOKENS = {"검사", "진찰", "신체진찰", "신체 진찰", "test", "tests", "exam", "결과", "소견", "평가", "확인", "혈액", "blood",
                  "lab", "labs", "수치", "level", "기능", "function", "ct", "씨티", "mri", "x-ray", "xray", "x선", "엑스레이",
                  "초음파", "ultrasound", "영상", "영상검사", "imaging", "촬영", "scan", "배양", "culture", "배양 검사", "항체",
                  "antibody", "단층", "방사선", "physical", "눈", "귀", "코", "입", "목", "배", "몸", "청진", "촉진", "타진", "시진",
                  "징후", "sign", "signs", "진찰 소견", "검사 결과", "혈액검사", "혈액 검사", "피검사", "피 검사", "증상",
                  "통증", "병력", "과거력", "진단", "영상 검사", "특수 검사", "추가 검사", "신체 검사", "신체검사", "검진"}
# words in common requests that a short ASCII search term must not be a substring of
_ENGLISH_WORDS = ("analysis", "anaerobic", "anemia", "anaemia", "gamma", "panel", "banana", "mammography", "amylase",
                  "ammonia", "lactate", "cortisol", "culture", "sputum", "urine", "urinalysis", "plasma", "sodium",
                  "potassium", "calcium", "magnesium", "phosphate", "creatinine", "glucose", "hemoglobin", "platelet",
                  "ferritin", "troponin", "procalcitonin", "electrolyte", "echocardiography", "echocardiogram",
                  "endoscopy", "colonoscopy", "bronchoscopy", "biopsy", "cytology", "pathology", "serology", "antigen",
                  "antibody", "immunoglobulin", "complement", "coagulation", "fibrinogen", "angiography", "arterial",
                  "venous", "doppler", "ultrasound", "abdominal", "pelvic", "thoracic", "cervical", "lumbar", "spine",
                  "chest", "brain", "head", "cardiac", "renal", "hepatic", "thyroid", "vitamin", "blood gas", "lipid",
                  "cholesterol", "triglyceride", "stool", "occult", "rapid", "screen", "toxicology", "drug", "alcohol",
                  "ethanol", "osmolality", "ketone", "lipase", "uric acid", "hba1c", "ldh", "creatine kinase",
                  "esr", "crp", "cbc", "bmp", "cmp", "lft", "tft", "abga", "abg", "ecg", "ekg", "eeg", "emg", "ncs",
                  "mri", "cta", "mra", "pet", "d-dimer", "bnp", "nt-probnp", "inr", "aptt", "pt/inr", "hcg", "psa",
                  "tsh", "free t4", "ana", "rf", "anti-ccp", "anca", "c3", "c4", "ige", "igg", "igm", "iga", "hiv",
                  "hbsag", "anti-hcv", "vdrl", "rpr", "tb", "igra", "quantiferon", "afb", "gram stain", "csf",
                  "lumbar puncture", "spirometry", "pft", "holter", "stress test", "treadmill", "tilt", "x-ray", "xray")

# literal placeholder words that ended up as search terms (scripts/convert_english_cases.py artifact)
PLACEHOLDER_TOKENS = {"english", "korean", "영어", "한국어", "검색어", "keyword", "keywords"}

# (token, host word) pairs that name the same test ('inr' in 'pt/inr'), not a hazard
_SAME_TEST = {("echo", "echocardiography"), ("echo", "echocardiogram"), ("inr", "pt/inr"), ("bnp", "nt-probnp"),
              ("ccp", "anti-ccp"), ("hcv", "anti-hcv"), ("abg", "abga"), ("a1c", "hba1c"), ("ptt", "aptt"),
              ("coag", "coagulation"), ("pt/inr", "pt/inr")}
# hazardous short tokens in augmented keys that are replaced by '<tok> 검사|<tok> test' (host = a common request)
HAZARD_FIX = {"iop", "cta", "ife", "ica", "mma", "ige", "rom", "sma", "aso", "act"}

# common doctor requests (Korean + English); used to find entries that several unrelated requests hit
COMMON_REQUESTS = {
    "exam": ["활력징후 측정", "혈압 측정", "체온 측정", "전신 상태 관찰", "두경부 진찰", "인두 진찰", "결막 확인", "경부 림프절 촉진",
             "갑상선 촉진", "심장 청진", "폐 청진", "흉부 진찰", "복부 진찰", "복부 촉진", "장음 청진", "하지 부종 확인", "피부 진찰",
             "신경학적 진찰", "의식 상태 평가", "뇌신경 검사", "근력 검사", "감각 검사", "심부건반사", "보행 관찰", "직장수지검사",
             "안저 검사", "관절 진찰", "척추 진찰", "유방 진찰", "골반 진찰", "눈 진찰", "귀 진찰", "구강 진찰", "vital signs",
             "general appearance", "heent exam", "cardiac exam", "lung exam", "abdominal exam", "neurologic exam",
             "skin exam", "extremity exam", "musculoskeletal exam"],
    "tests": ["일반혈액검사", "CBC", "전해질 검사", "신기능 검사 (BUN/Cr)", "혈당 검사", "간기능 검사", "CRP", "ESR", "소변검사",
              "흉부 X선", "심전도", "뇌 CT", "뇌 MRI", "복부 초음파", "복부 CT", "흉부 CT", "임신 검사", "트로포닌", "D-dimer",
              "혈액 배양", "소변 배양", "객담 배양", "리파아제", "갑상선 기능 검사", "응고 검사 (PT/INR)", "동맥혈 가스 분석",
              "심장 초음파", "요추 천자", "뇌척수액 검사", "지질 검사", "HbA1c", "철분 검사", "비타민 B12", "ANA", "류마티스 인자",
              "간염 바이러스 검사", "HIV 검사", "혈액 도말", "뇌파", "근전도", "위내시경", "대장내시경", "복부 X선", "척추 MRI",
              "골밀도", "BNP", "암모니아", "CK", "젖산", "프로칼시토닌", "blood gas analysis", "urinalysis",
              "chest x-ray", "head ct", "abdominal ultrasound", "echocardiography", "blood culture", "lumbar puncture",
              "thyroid function test", "coagulation panel", "liver function test", "renal function"],
}


def _hits(section_table: dict, request: str) -> list[str]:
    q = request.lower()  # CaseFileEnvironment.step
    return [k for k in section_table if any(tok in q for tok in k.lower().split("|"))]


def check_keys(case: dict, ents: list[Entry]) -> list[dict]:
    issues = []
    aug = case.get("augmented") or {}
    for e in ents:
        if e.section not in ("history", "exam", "tests"):
            continue
        toks = _toks(e.key)
        sev = "hard" if e.aug else "soft"
        if any(t.strip() == "" for t in toks):
            issues.append(_issue("key_empty_token", sev, e, "empty search term (matches every request)"))
        for t in toks:
            if t.strip() in PLACEHOLDER_TOKENS:
                issues.append(_issue("key_placeholder_token", "hard", e, f"placeholder search term '{t.strip()}' "
                                     "(conversion artifact; value unaffected)", token=t))
            elif t.strip() in ("other", "기타", "table"):
                issues.append(_issue("key_placeholder_token", "soft", e, f"vague search term '{t.strip()}'", token=t))
        if e.section == "history":
            continue
        for t in toks:
            ts = t.strip()
            if not ts:
                continue
            if ts in GENERIC_TOKENS:
                issues.append(_issue("key_generic_token", sev, e, f"generic search term '{ts}' matches unrelated requests",
                                     token=t))
            elif ts.isascii() and len(ts) < 3:
                issues.append(_issue("key_generic_token", sev, e, f"short ASCII search term '{ts}' (substring of many words)",
                                     token=t))
            elif ts.isascii() and len(ts) <= 4 and re.fullmatch(r"[a-z0-9\-/]+", ts):
                hosts = [w for w in _ENGLISH_WORDS if ts in w and w != ts and not w.startswith(ts + " ")
                         and (ts, w) not in _SAME_TEST]
                if hosts:
                    issues.append(_issue("key_substring_hazard", "soft", e,
                                         f"'{ts}' is a substring of unrelated request words: {hosts[:4]}", token=t))
        # reachability: at least one search term short enough to occur inside a normal request
        stoks = [t.strip() for t in toks if t.strip()]
        if stoks and all(len(t.replace(" ", "")) > 10 for t in stoks):
            issues.append(_issue("key_unreachable", "soft", e, "every search term is a long phrase (>10 chars); "
                                                               "short requests miss it"))
    # duplicate search terms within a section, and across exam/tests
    for sec in ("exam", "tests"):
        owner = collections.defaultdict(list)
        for k in (case.get(sec) or {}):
            for t in set(t.strip() for t in _toks(k) if t.strip()):
                owner[t].append(k)
        for t, ks in owner.items():
            if len(ks) > 1:
                added = set(aug.get(sec, []))
                issues.append({"type": "key_duplicate_token", "severity": "soft", "where": sec, "token": t,
                               "aug": any(k in added for k in ks),
                               "detail": f"'{t}' in {len(ks)} {sec} keys: " + " || ".join(k[:40] for k in ks)})
    ex = {t.strip() for k in (case.get("exam") or {}) for t in _toks(k) if t.strip()}
    te = {t.strip() for k in (case.get("tests") or {}) for t in _toks(k) if t.strip()}
    for t in sorted(ex & te):
        issues.append({"type": "key_cross_section", "severity": "soft", "where": "exam+tests", "token": t,
                       "detail": f"'{t}' is a search term in both exam and tests"})
    # common requests that hit several entries including an augmented one (simulator concatenates all hits)
    for sec in ("exam", "tests"):
        table = case.get(sec) or {}
        added = set(aug.get(sec, []))
        for r in COMMON_REQUESTS[sec]:
            hits = _hits(table, r)
            if len(hits) < 2:
                continue
            n_aug = sum(h in added for h in hits)
            # augmented-only collisions are ours; collisions through a broad original key ('진찰', 'CT') are not
            typ = "key_collision" if n_aug == len(hits) else "key_collision_orig"
            issues.append({"type": typ, "severity": "soft", "where": sec, "request": r, "aug": n_aug > 0,
                           "detail": f"request '{r}' hits {len(hits)} entries: " + " || ".join(h[:40] for h in hits)})
    return issues


# ─────────────────────────────────────────────── driver ───────────────────────────────────────────────
def check_case(case: dict) -> list[dict]:
    ents = entries(case)
    return (check_structure(case) + check_leaks(case, ents) + check_numbers(case, ents) + check_findings(ents)
            + check_fever(case, ents) + check_demographics(case, ents) + check_keys(case, ents))


def case_paths(root: Path = CASE_ROOT) -> list[Path]:
    return sorted(root.glob("*/*.json"))


def case_id(p: Path) -> str:
    return f"{p.parent.name}/{p.stem}"


def check_all(root: Path = CASE_ROOT) -> dict[str, list[dict]]:
    return {case_id(p): check_case(json.loads(p.read_text(encoding="utf-8"))) for p in case_paths(root)}


def summarize(results: dict[str, list[dict]]) -> dict:
    by = collections.Counter()
    cases = collections.defaultdict(set)
    for cid, issues in results.items():
        for i in issues:
            k = f"{i['type']}:{i['severity']}"
            by[k] += 1
            cases[k].add(cid)
    return {"n_cases": len(results),
            "issues": {k: {"count": by[k], "cases": len(cases[k])} for k in sorted(by)},
            "hard_total": sum(v for k, v in by.items() if k.endswith(":hard")),
            "cases_with_hard": len({c for k, s in cases.items() if k.endswith(":hard") for c in s})}


# ─────────────────────────────────────────────── fixes ───────────────────────────────────────────────
# Manual fixes after reviewing the flags (rule-based, no LLM). Each: case id, section, a unique substring of the
# augmented key, action, arguments, reason. Actions:
#   remove            — delete the augmented entry
#   replace           — in the augmented value, replace `old` by `new` (align with the original value / drop a clause)
# Fixes are idempotent: skipped when the entry/text is no longer there.
# The fixes recorded on 2026-09-27 (see data/labels/case_quality_2026-09-27.json) were all for the private
# AgentClinic-derived cases, which are not part of the public repository; the committed public cases need none.
FIXES: list[dict] = [
]


def _find_key(case: dict, sec: str, sub: str) -> str | None:
    added = set((case.get("augmented") or {}).get(sec, []))
    ks = [k for k in (case.get(sec) or {}) if k in added and (sub in k)]
    return ks[0] if len(ks) == 1 else None


def _rename_key(case: dict, sec: str, old: str, new: str) -> None:
    case[sec] = {(new if k == old else k): v for k, v in case[sec].items()}
    aug = case.get("augmented") or {}
    if sec in aug:
        aug[sec] = [new if k == old else k for k in aug[sec]]
    af = case.get("augmented_full") or {}
    for pid, k in list((af.get("panel") or {}).items()):
        if k == old:
            af["panel"][pid] = new
    if "extra" in af:
        af["extra"] = [new if k == old else k for k in af["extra"]]


def _remove_key(case: dict, sec: str, key: str) -> None:
    del case[sec][key]
    aug = case.get("augmented") or {}
    aug[sec] = [k for k in aug.get(sec, []) if k != key]
    af = case.get("augmented_full") or {}
    for pid, k in list((af.get("panel") or {}).items()):
        if k == key:
            del af["panel"][pid]
    if "extra" in af:
        af["extra"] = [k for k in af["extra"] if k != key]


def mechanical_key_fixes(case: dict) -> list[dict]:
    """Drop empty / generic / <3-char ASCII search terms from augmented keys (the simulator matches substrings)."""
    edits = []
    aug = case.get("augmented") or {}
    for sec in ("exam", "tests"):
        for key in list(aug.get(sec, [])):
            if key not in case.get(sec, {}):
                continue
            toks = key.split("|")
            keep = [t for t in toks if t.strip() and t.strip().lower() not in GENERIC_TOKENS
                    and not (t.strip().isascii() and len(t.strip()) < 3)]
            for t in [t for t in keep if t.strip().lower() in HAZARD_FIX]:
                i = keep.index(t)
                variants = [v for v in (f"{t.strip()} 검사", f"{t.strip()} test") if v not in keep]
                keep[i:i + 1] = variants
            if keep and keep != toks:
                new = "|".join(keep)
                if new in case[sec]:
                    continue
                _rename_key(case, sec, key, new)
                edits.append({"action": "rename_key", "section": sec, "old": key, "new": new,
                              "reason": "search terms matching unrelated requests (empty/generic/short, or a short "
                                        "abbreviation inside a common request word such as 'iop' in 'biopsy') "
                                        "removed or made specific: " + str([t for t in toks if t not in keep])})
    return edits


def placeholder_key_fixes(case: dict) -> list[dict]:
    """Remove placeholder search terms ('english') from any key, original ones included (the value is untouched)."""
    edits = []
    for sec in ("history", "exam", "tests"):
        for key in list(case.get(sec, {})):
            toks = key.split("|")
            keep = [t for t in toks if t.strip().lower() not in PLACEHOLDER_TOKENS]
            if keep and keep != toks and "|".join(keep) not in case[sec]:
                new = "|".join(keep)
                _rename_key(case, sec, key, new)
                edits.append({"action": "rename_key", "section": sec, "old": key, "new": new,
                              "reason": "placeholder search term 'english' (conversion artifact) removed; "
                                        "value unchanged"})
    return edits


def apply_fixes(case: dict, cid: str) -> list[dict]:
    edits = []
    for fx in FIXES:
        if fx["case"] != cid:
            continue
        sec = fx["section"]
        key = _find_key(case, sec, fx["key"])
        if key is None:
            continue
        rec = {"action": fx["action"], "section": sec, "key": key, "reason": fx["reason"]}
        if fx["action"] == "remove":
            rec["old"] = case[sec][key]
            _remove_key(case, sec, key)
        elif fx["action"] == "replace":
            val = case[sec][key]
            if fx["old"] not in val:
                continue
            case[sec][key] = re.sub(r"\s{2,}", " ", val.replace(fx["old"], fx["new"])).strip(" ,")
            rec.update(old=fx["old"], new=fx["new"])
        edits.append(rec)
    return edits + mechanical_key_fixes(case) + placeholder_key_fixes(case)


_SUMMARISED = {"key_collision_orig", "key_generic_token", "key_duplicate_token", "key_cross_section"}


def _key_summary(results: dict[str, list[dict]]) -> dict:
    tok, req = collections.Counter(), collections.Counter()
    for iss in results.values():
        for i in iss:
            if i["type"] == "key_generic_token" and not i.get("aug"):
                tok[i["token"].strip()] += 1
            if i["type"] == "key_collision_orig":
                req[i["request"]] += 1
    return {"generic_tokens_in_original_keys": dict(tok.most_common()),
            "common_requests_hitting_several_entries_via_original_keys": dict(req.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(CASE_ROOT))
    ap.add_argument("--json", help="write the full issue list to this file")
    ap.add_argument("--fix", action="store_true", help="apply FIXES + mechanical key clean-up to the case files")
    ap.add_argument("--report", help="with --fix: write the before/after report (+ edits) here")
    ap.add_argument("--show", nargs="*", help="print issues of these types (e.g. leak:hard vital_conflict)")
    args = ap.parse_args()
    root = Path(args.root)

    before = check_all(root)
    edits = {}
    if args.fix:
        for p in case_paths(root):
            case = json.loads(p.read_text(encoding="utf-8"))
            ed = apply_fixes(case, case_id(p))
            if ed:
                p.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
                edits[case_id(p)] = ed
    after = check_all(root) if args.fix else before

    s_before, s_after = summarize(before), summarize(after)
    print(f"{s_after['n_cases']} cases")
    keys = sorted(set(s_before["issues"]) | set(s_after["issues"]))
    for k in keys:
        b = s_before["issues"].get(k, {"count": 0, "cases": 0})
        a = s_after["issues"].get(k, {"count": 0, "cases": 0})
        print(f"  {k:32s} {b['count']:5d} ({b['cases']:3d} cases)" + (f"  →  {a['count']:5d} ({a['cases']:3d} cases)"
                                                                      if args.fix else ""))
    print(f"hard issues: {s_before['hard_total']}" + (f" → {s_after['hard_total']}" if args.fix else ""))
    if args.fix:
        print(f"edited cases: {len(edits)}, edits: {sum(len(v) for v in edits.values())}")
    if args.show is not None:
        for cid, issues in after.items():
            for i in issues:
                if not args.show or i["type"] in args.show or f"{i['type']}:{i['severity']}" in args.show:
                    print(cid, i["type"], i["severity"], i.get("where", ""), "|", i["detail"])
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": s_after, "issues": after}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    if args.report:
        report = {
            "date": dt.date.today().isoformat(),
            "script": "scripts/check_cases.py --fix",
            "note": "Rule-based check of data/cases_aug (no LLM). Hard issues must be zero "
                    "(tests/test_case_quality.py); soft issues are a review list (ambiguous clinical judgments are "
                    "flagged, not fixed). Edits change LLM-augmented entries only, except removing the placeholder "
                    "search term 'english' from original keys (values untouched); original case facts never change.",
            "summary_before": s_before, "summary_after": s_after,
            "edits": edits,
            # broad search terms in ORIGINAL keys are summarised (not ours to change; hundreds of rows)
            "original_key_breadth": _key_summary(after),
            "issues_after": {cid: [i for i in iss if i["type"] not in _SUMMARISED or i.get("aug")]
                             for cid, iss in after.items()
                             if any(i["type"] not in _SUMMARISED or i.get("aug") for i in iss)},
        }
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
