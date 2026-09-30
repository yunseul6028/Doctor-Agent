"""Published diagnostic / classification criteria, checked against the findings of one case.

Owned by clinical-strategist (+ knowledge-rag for the licence ledger). Used by the pre-diagnosis review
(`agent/policy.py`) to (1) show the reviewer which criteria items are met / not met / unknown and (2) decide a subtype
("양극성 장애" → I형/II형, DKA vs HHS, complete vs incomplete Kawasaki, ...) only when the findings explicitly satisfy
the subtype rule.

Copyright: criteria content (items, points, thresholds, logic) is factual and is implemented here in our own Korean
wording. No table, sentence or figure of any source is copied. ICHD-3 and DSM-5-TR are used for content only.
Korean KDCA notifiable-disease case definitions (법정감염병 진단·신고 기준) are NOT implemented: the KDCA page marks the
document 공공누리 제4유형 (출처표시+상업적이용금지+변경금지), which forbids modification (checked 2026-09-27).

Verification fields
- Citation.verified: bibliographic data checked against PubMed E-utilities (esummary) or Crossref on 2026-09-27.
- Criteria.verification: how the items/thresholds/logic were checked.
  "primary" = against the original (abstract or full text); "secondary" = core in the original, some detail from
  reviewer knowledge; "unverified" = original not accessible, from reviewer knowledge. Details in Criteria.note.
- 2026-09-29 second pass: dka_hhs_2024 (Fig. 2B read: HHS HCO3 >= 15 and effective osmolality > 300 now used),
  ra_2010 (Table 3 of the Ann Rheum Dis co-publication) and jones_2015 (Table 7 of the statement; PR interval no longer
  counted as minor with carditis) -> primary; kawasaki_aha2017 -> secondary (supplemental criteria read in an
  open-access paper citing the statement; "secondary" also covers that case). light_1972 and bipolar_dsm5tr stay
  unverified (original / manual not accessible), mcdonald_2017 stays secondary.

Extraction (evaluate) is deliberately conservative: an item is "met" only when a pattern for it appears in the case text
and is not negated in its clause, or a number with the right label/unit passes the threshold. Doctor questions
("...있나요?") and "결과없음" ledger lines are ignored; "- 음성:" ledger lines count as negative. Anything else is
"unknown". Stdlib only, CPU only, deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from doctor_agent.knowledge.clinical_rules import Citation, ReadText, after_family_heading, age_years

MET, NOT_MET, UNKNOWN = "met", "not_met", "unknown"
_STATE_KO = {MET: "충족", NOT_MET: "불충족", UNKNOWN: "미확인"}

# --------------------------------------------------------------------------------------------
# Text preparation and extraction
# --------------------------------------------------------------------------------------------

_ACTION_LINE = re.compile(r"^\s*\d+\.\s*(ask|exam|test|diagnose)\s*:.*$", re.M)
_QUESTION = re.compile(r"[^.\n?!]*\?")
_KO_DAYS = (("일주일", "7일"), ("한 주", "7일"), ("열흘", "10일"), ("보름", "15일"), ("이틀", "2일"), ("사흘", "3일"),
            ("나흘", "4일"), ("닷새", "5일"), ("엿새", "6일"), ("이레", "7일"), ("한 달", "30일"), ("한달", "30일"),
            ("하루", "1일"), ("두 주", "14일"), ("2주일", "14일"), ("3주일", "21일"))
_NEGATIONS = ("없", "않", "아니", "기보다", "안 됨", "음성", "(-)", "정상", "no ", "not ", "denies", "denied", "without", "never",
              "negative", "absent", "normal")
_PRE_NEGATIONS = ("no ", "not ", "denies ", "denied ", "without ", "negative for ", "absence of ")
_CLAUSE_BREAKS = (".", "!", "\n", ",", ";", "고 ", "며 ", "면서", "는데", "지만", " but ", " and ", "→")
_NEG_WINDOW = 25
# layer reading (Doc._reading) only for keyword-like matches without their own result word
_LAYER_MAX_MATCH = 20
_OWN_RESULT = re.compile(r"없|않|아니|음성|양성|정상|negative|positive|normal|\(-\)|\(\+\)|\+")

_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(시간|일|주|개월|달|년|hours?|hrs?|days?|weeks?|months?|years?)(?![a-z])")
_DUR_DAYS = {"시간": 1 / 24, "일": 1, "주": 7, "개월": 30, "달": 30, "년": 365, "hour": 1 / 24, "hours": 1 / 24,
             "hr": 1 / 24, "hrs": 1 / 24, "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30,
             "year": 365, "years": 365}


def A(tok: str) -> str:
    """Regex for a short ASCII token that must not be part of a longer word ("ana" not in "anaphylaxis")."""
    return rf"(?<![a-z]){tok}(?![a-z])"


def RX(pattern: str) -> str:
    """Mark a diagnosis-name synonym as a regex (plain synonyms are matched as substrings, ignoring spaces)."""
    return "re:" + pattern


def prepare(text: str) -> str:
    """Lowercase, drop the doctor's own actions/questions, normalise Korean duration words and thousands commas."""
    t = (text or "").lower()
    t = _ACTION_LINE.sub("", t)
    t = _QUESTION.sub("", t)
    for a, b in _KO_DAYS:
        t = t.replace(a, b)
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    return t


class Doc:
    """Prepared case text + extraction helpers."""

    def __init__(self, text: str):
        self.raw = text or ""
        self.text = prepare(text)
        self._rt: ReadText | None = None  # parsed lazily by the normalisation layer, once per Doc

    def _line(self, pos: int) -> str:
        start = self.text.rfind("\n", 0, pos) + 1
        return self.text[start:pos].lstrip()

    def _skipped(self, pos: int) -> bool:
        return self._line(pos).startswith("- 결과없음")

    def _negated(self, start: int, end: int) -> bool:
        if self._line(start).startswith("- 음성"):
            return True
        stop = min(len(self.text), end + _NEG_WINDOW)
        for sep in _CLAUSE_BREAKS:
            j = self.text.find(sep, end)
            if 0 <= j < stop:
                stop = j
        tail = self.text[end:stop]
        if any(n in tail for n in _NEGATIONS):
            return True
        head = self.text[max(0, start - 15):start]
        return any(n in head for n in _PRE_NEGATIONS)

    def _reading(self, start: int, end: int) -> str:
        """One match: "aff", "neg" or "skip" (uncertain, hypothetical, or about a relative).

        2026-09-27: a short match without its own negation/result word is read by the normalisation layer when the
        layer has a finding there (list negation across commas, idioms, persistence, hedges keep "affirmed";
        "모르겠어요" and questions are "skip", never "neg"; relatives are "skip"). Ledger "- 음성" lines, long
        matches ("혈액 배양 ... 양성") and matches that carry their own result word ("ana 음성") keep the window rule."""
        if self._line(start).startswith("- 음성"):
            return "neg"
        span = self.text[start:end]
        if end - start <= _LAYER_MAX_MATCH and not _OWN_RESULT.search(span):
            hits = self._read.findings_at(start, end)
            if hits:
                if after_family_heading(self.text, start):  # "가족력: 루푸스"
                    return "skip"
                mine = [f for f in hits if f.subject == "patient"]
                if any(f.polarity == "present" and not f.hypothetical for f in mine):
                    return "aff"
                if not mine or any(f.polarity == "uncertain" or f.hypothetical for f in mine):
                    return "skip"
                return "neg"
        return "neg" if self._negated(start, end) else "aff"

    @property
    def _read(self) -> ReadText:
        if self._rt is None:
            self._rt = ReadText(self.text)
        return self._rt

    def state(self, patterns: tuple[str, ...]) -> tuple[str, str]:
        """MET if any match is affirmed; NOT_MET if matches exist and all are negated (none uncertain); else
        UNKNOWN."""
        negated, unsure = "", False
        for p in patterns:
            for m in re.finditer(p, self.text):
                if self._skipped(m.start()):
                    continue
                r = self._reading(m.start(), m.end())
                if r == "aff":
                    return MET, m.group(0).strip()
                if r == "neg":
                    negated = negated or m.group(0).strip()
                else:
                    unsure = True
        return (NOT_MET, negated) if negated and not unsure else (UNKNOWN, "")

    def values(self, num: "Num") -> list[float]:
        out = []
        pats = (num.pattern,) if num.pattern else tuple(
            rf"(?:{lab})[^\d\n]{{0,{num.window}}}?(?P<v>\d+(?:\.\d+)?)\s*(?P<u>[a-zμµ/%°℃·0-9^]*)" for lab in num.labels)
        for p in pats:
            for m in re.finditer(p, self.text):
                if self._skipped(m.start()):
                    continue
                v = float(m.group("v"))
                unit = (m.groupdict().get("u") or "").replace("μ", "u").replace("µ", "u")
                factor = None
                for u, f in num.units:
                    if unit.startswith(u):
                        factor = f
                        break
                if factor is None:
                    if num.unit_required:
                        continue
                    factor = 1.0
                v *= factor
                if num.small_scale and v < num.small_scale[0]:
                    v *= num.small_scale[1]
                if num.lo <= v <= num.hi:
                    out.append(v)
        return out

    def _clause(self, start: int, end: int) -> tuple[int, int]:
        lo = max(self.text.rfind(sep, 0, start) + len(sep) if self.text.rfind(sep, 0, start) >= 0 else 0
                 for sep in _CLAUSE_BREAKS)
        his = [j for j in (self.text.find(sep, end) for sep in _CLAUSE_BREAKS) if j >= 0]
        return lo, (min(his) if his else len(self.text))

    def durations_days(self, anchors: tuple[str, ...], window: int = 25) -> list[float]:
        """Durations (in days) written within `window` chars of an anchor pattern in the same clause."""
        out = []
        for m in _DUR.finditer(self.text):
            if self._skipped(m.start()):
                continue
            ls, le = self._clause(m.start(), m.end())
            around = self.text[max(ls, m.start() - window):min(le, m.end() + window)]
            if any(re.search(a, around) for a in anchors):
                out.append(float(m.group(1)) * _DUR_DAYS[m.group(2)])
        return out

    @property
    def age(self) -> float | None:
        return age_years(self.text)

    @property
    def female(self) -> str:
        f = re.search(r"여성|여자|여환|\d+\s*세\s*여|(?<![a-z])(female|woman)(?![a-z])|\d{1,3}\s?/\s?f(?![a-z])|"
                      r"(?<![a-z])f\s?/\s?\d{1,3}", self.text)
        m = re.search(r"남성|남자|남환|\d+\s*세\s*남|(?<![a-z])(male|man)(?![a-z])|\d{1,3}\s?/\s?m(?![a-z])|"
                      r"(?<![a-z])m\s?/\s?\d{1,3}", self.text)
        if f and not m:
            return MET
        if m and not f:
            return NOT_MET
        return UNKNOWN


@dataclass(frozen=True)
class Num:
    """A labelled number: label regexes, comparison, threshold in the canonical unit, unit conversions."""
    labels: tuple[str, ...] = ()
    op: str = ">="
    value: float = 0.0
    units: tuple[tuple[str, float], ...] = ()  # (unit prefix, factor to canonical)
    unit_required: bool = False
    lo: float = float("-inf")  # plausible range in canonical unit (other numbers are ignored)
    hi: float = float("inf")
    window: int = 12
    pattern: str = ""  # full regex override (named groups: v = number, optional u = unit)
    small_scale: tuple[float, float] = ()  # (below, multiply) e.g. platelets "85" (x10^3) → 85000
    agg: str = "last"  # last | max | min

    def check(self, v: float) -> bool:
        return {">=": v >= self.value, ">": v > self.value, "<": v < self.value, "<=": v <= self.value}[self.op]


@dataclass(frozen=True)
class CItem:
    key: str
    text: str  # Korean, our own wording
    points: float = 0.0
    domain: str = ""
    patterns: tuple[str, ...] = ()
    num: Num | None = None
    derive: Callable[[Doc], tuple[str, str]] | None = field(default=None, compare=False)
    whole_text: bool = False  # read the whole case even when the set has a `focus`


def item_state(item: CItem, doc: Doc) -> tuple[str, str]:
    if item.derive:
        return item.derive(doc)
    if item.num:
        vals = doc.values(item.num)
        if vals:
            v = {"last": vals[-1], "max": max(vals), "min": min(vals)}[item.num.agg]
            return (MET if item.num.check(v) else NOT_MET), f"{v:g}"
    if item.patterns:
        return doc.state(item.patterns)
    return UNKNOWN, ""


@dataclass(frozen=True)
class Outcome:
    band: str  # Korean result band
    decision: str = ""  # diagnosis/subtype name the findings explicitly satisfy ("" = none)
    score: float | None = None


@dataclass(frozen=True)
class Subtype:
    name: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Criteria:
    id: str
    short: str
    name_ko: str
    name_en: str
    kind: str  # classification | diagnostic | staging | definition
    synonyms: tuple[str, ...]  # diagnosis names that select this set (Korean/English)
    items: tuple[CItem, ...]
    logic: Callable[[dict, Doc], Outcome] = field(compare=False)
    rule: str  # Korean: the decision rule, our wording
    citation: Citation
    verification: str  # primary | secondary | unverified
    note: str = ""
    subtypes: tuple[Subtype, ...] = ()
    subtype_rule: str = ""  # Korean: how to pick the subtype
    related: tuple[str, ...] = ()  # regexes: other diagnosis names for which this set is a useful cross-check
    focus: tuple[str, ...] = ()  # regexes: if the chief complaint is not about these, items (unless whole_text) read
    # only the sentences that mention them (headache features are not taken from an abdominal-pain story)
    exclude: tuple[str, ...] = ()  # regexes: names that look similar but are not this disease
    copyright: str = "Content (items/thresholds) only, Korean wording ours; no source text copied."

    @property
    def cite(self) -> str:
        c = self.citation
        ref = f"PMID {c.pmid}" if c.pmid else (f"doi:{c.doi}" if c.doi else "")
        return f"{c.short}{', ' + ref if ref else ''}"


@dataclass(frozen=True)
class CriteriaResult:
    criteria_id: str
    band: str
    decision: str
    score: float | None
    states: dict = field(default_factory=dict, compare=False)  # key -> (state, evidence)
    met: tuple[str, ...] = ()
    not_met: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------------
# Shared helpers for the logic functions
# --------------------------------------------------------------------------------------------

def _m(s: dict, k: str) -> bool:
    return s[k][0] == MET


def _n(s: dict, k: str) -> bool:
    return s[k][0] == NOT_MET


def _u(s: dict, k: str) -> bool:
    return s[k][0] == UNKNOWN


def _pts(c: Criteria, s: dict) -> float:
    """Sum of points of met items; within a domain only the highest met item counts."""
    best: dict[str, float] = {}
    total = 0.0
    for it in c.items:
        if it.points and s[it.key][0] == MET:
            if it.domain:
                best[it.domain] = max(best.get(it.domain, float("-inf")), it.points)
            else:
                total += it.points
    return total + sum(best.values())


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _age_state(op: str, limit: float) -> Callable[[Doc], tuple[str, str]]:
    def f(doc: Doc) -> tuple[str, str]:
        a = doc.age
        if a is None:
            return UNKNOWN, ""
        ok = a <= limit if op == "<=" else a >= limit
        return (MET if ok else NOT_MET), f"{_fmt(a)}세"
    return f


def _female(doc: Doc) -> tuple[str, str]:
    st = doc.female
    return st, ("여성" if st == MET else "남성" if st == NOT_MET else "")


# Common numeric labels (canonical units noted)
_T_TEMP = r"(?P<v>\d{2}(?:\.\d+)?)\s*(?P<u>°c|℃|도|c(?![a-z]))"  # body temperature, °C
_CRP_UNITS = (("mg/dl", 10.0), ("mg/l", 1.0))  # canonical mg/L
_CRP = (r"crp", r"c-?반응성?\s?단백")
_ESR = (r"esr", r"적혈구\s?침강\s?속도", r"혈침")
_WBC = (r"wbc", r"백혈구(?!\s?(에스테라제|감소|증가|뇨|원주))")
_PLT = (r"혈소판(?!\s?(감소|증가))", r"(?<![a-z])plt(?![a-z])", r"platelets?")


def _temp(op: str, value: float) -> Num:
    return Num(op=op, value=value, pattern=_T_TEMP, lo=34, hi=43, agg="max")


# --------------------------------------------------------------------------------------------
# Citations (bibliographic data checked against PubMed esummary / Crossref, 2026-09-27)
# --------------------------------------------------------------------------------------------

C_SLE = Citation("Aringer M, Costenbader K, Daikh D, et al.",
                 "2019 European League Against Rheumatism/American College of Rheumatology Classification Criteria "
                 "for Systemic Lupus Erythematosus", "Arthritis Rheumatol", 2019, "71(9):1400-1412",
                 doi="10.1002/art.40930", pmid="31385462", verified=True)
C_RA = Citation("Aletaha D, Neogi T, Silman AJ, et al.",
                "2010 Rheumatoid arthritis classification criteria: an American College of Rheumatology/European "
                "League Against Rheumatism collaborative initiative", "Arthritis Rheum", 2010, "62(9):2569-2581",
                doi="10.1002/art.27584", pmid="20872595", verified=True)
C_TAK = Citation("Grayson PC, Ponte C, Suppiah R, et al.",
                 "2022 American College of Rheumatology/EULAR classification criteria for Takayasu arteritis",
                 "Ann Rheum Dis", 2022, "81(12):1654-1660", doi="10.1136/ard-2022-223482", pmid="36351705",
                 verified=True)
C_GCA = Citation("Ponte C, Grayson PC, Robson JC, et al.",
                 "2022 American College of Rheumatology/EULAR classification criteria for giant cell arteritis",
                 "Ann Rheum Dis", 2022, "81(12):1647-1653", doi="10.1136/ard-2022-223480", pmid="36351706",
                 verified=True)
C_KD = Citation("McCrindle BW, Rowley AH, Newburger JW, et al.",
                "Diagnosis, Treatment, and Long-Term Management of Kawasaki Disease: A Scientific Statement for "
                "Health Professionals From the American Heart Association", "Circulation", 2017,
                "135(17):e927-e999", doi="10.1161/CIR.0000000000000484", pmid="28356445", verified=True)
C_IE = Citation("Fowler VG, Durack DT, Selton-Suty C, et al.",
                "The 2023 Duke-International Society for Cardiovascular Infectious Diseases Criteria for Infective "
                "Endocarditis: Updating the Modified Duke Criteria", "Clin Infect Dis", 2023, "77(4):518-526",
                doi="10.1093/cid/ciad271", pmid="37138445", verified=True)
C_AKI = Citation("Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group.",
                 "KDIGO Clinical Practice Guideline for Acute Kidney Injury", "Kidney Int Suppl", 2012, "2(1):1-138",
                 doi="10.1038/kisup.2012.1", pmid="22890468", verified=True, short_author="KDIGO")
C_DKA = Citation("Umpierrez GE, Davis GM, ElSayed NA, et al.",
                 "Hyperglycemic Crises in Adults With Diabetes: A Consensus Report", "Diabetes Care", 2024,
                 "47(8):1257-1275", doi="10.2337/dci24-0032", pmid="39052901", verified=True)
C_LIGHT = Citation("Light RW, Macgregor MI, Luchsinger PC, Ball WC Jr.",
                   "Pleural effusions: the diagnostic separation of transudates and exudates", "Ann Intern Med",
                   1972, "77(4):507-513", doi="10.7326/0003-4819-77-4-507", pmid="4642731", verified=True)
C_SEPSIS = Citation("Singer M, Deutschman CS, Seymour CW, et al.",
                    "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)", "JAMA",
                    2016, "315(8):801-810", doi="10.1001/jama.2016.0287", pmid="26903338", verified=True)
C_JONES = Citation("Gewitz MH, Baltimore RS, Tani LY, et al.",
                   "Revision of the Jones Criteria for the diagnosis of acute rheumatic fever in the era of Doppler "
                   "echocardiography: a scientific statement from the American Heart Association", "Circulation",
                   2015, "131(20):1806-1818", doi="10.1161/CIR.0000000000000205", pmid="25908771", verified=True)
C_MS = Citation("Thompson AJ, Banwell BL, Barkhof F, et al.",
                "Diagnosis of multiple sclerosis: 2017 revisions of the McDonald criteria", "Lancet Neurol", 2018,
                "17(2):162-173", doi="10.1016/S1474-4422(17)30470-2", pmid="29275977", verified=True)
C_ICHD = Citation("Headache Classification Committee of the International Headache Society (IHS).",
                  "The International Classification of Headache Disorders, 3rd edition", "Cephalalgia", 2018,
                  "38(1):1-211", doi="10.1177/0333102417738202", pmid="29368949", verified=True,
                  short_author="ICHD-3")
C_DSM = Citation("American Psychiatric Association.",
                 "Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition, Text Revision (DSM-5-TR)",
                 "American Psychiatric Association Publishing", 2022, "book",
                 doi="10.1176/appi.books.9780890425787", verified=True, short_author="DSM-5-TR")
C_GOUT = Citation("Neogi T, Jansen TL, Dalbeth N, et al.",
                  "2015 Gout Classification Criteria: an American College of Rheumatology/European League Against "
                  "Rheumatism collaborative initiative", "Arthritis Rheumatol", 2015, "67(10):2557-2568",
                  doi="10.1002/art.39254", pmid="26352873", verified=True)


# --------------------------------------------------------------------------------------------
# 1. SLE — 2019 EULAR/ACR
# --------------------------------------------------------------------------------------------

def _low_c(which: str) -> tuple[str, ...]:
    return (rf"(?<![a-z]){which}(?!\d)[^\n\d]{{0,6}}(감소|저하|낮|low|decreased)", rf"low {which}(?!\d)",
            rf"저\s?{which}(?!\d)")


def _complement_both(doc: Doc) -> tuple[str, str]:
    c3, e3 = doc.state(_low_c("c3"))
    c4, e4 = doc.state(_low_c("c4"))
    both, eb = doc.state((r"c3[^\n\d]{0,3}(와|과|,|및|/|and)\s?c4[^\n\d]{0,6}(모두\s?)?(감소|저하|낮|low)",
                          r"low c3 and c4"))
    if both == MET or (c3 == MET and c4 == MET):
        return MET, eb or f"{e3}, {e4}"
    if NOT_MET in (c3, c4, both):
        return NOT_MET, ""
    return UNKNOWN, ""


def _sle(s: dict, doc: Doc) -> Outcome:
    c = CRITERIA_BY_ID["sle_2019"]
    score = _pts(c, s)
    clinical = any(_m(s, it.key) for it in c.items if it.domain in _SLE_CLINICAL)
    if _n(s, "ana"):
        return Outcome("분류 불가: 진입 기준(ANA ≥1:80) 불충족", "", score)
    if score >= 10 and clinical and _m(s, "ana"):
        return Outcome(f"분류 기준 충족 ({_fmt(score)}점 ≥10, ANA 양성, 임상 항목 ≥1)", "전신홍반루푸스", score)
    if _u(s, "ana"):
        return Outcome(f"ANA 미확인 (현재 {_fmt(score)}점, 기준 ≥10점)", "", score)
    return Outcome(f"아직 미충족 (현재 {_fmt(score)}점, 기준 ≥10점과 임상 항목 ≥1)", "", score)


_SLE_CLINICAL = ("constitutional", "hematologic", "neuropsychiatric", "mucocutaneous", "serosal", "musculoskeletal",
                 "renal")
SLE = Criteria(
    id="sle_2019", short="2019 EULAR/ACR SLE", name_ko="전신홍반루푸스 분류 기준", name_en="2019 EULAR/ACR SLE criteria",
    kind="classification",
    synonyms=("전신홍반루푸스", "전신성 홍반성 루푸스", "홍반루푸스", "홍반성 루푸스", "루푸스", "lupus", RX(A("sle"))),
    exclude=(r"약물\s?유발", r"drug.induced", r"원판상 루푸스만", r"신생아 루푸스"),
    items=(
        CItem("ana", "진입: ANA 양성(HEp-2 ≥1:80 또는 동등 검사)", derive=lambda d: _ana(d)),
        CItem("fever", "발열 >38.3°C", 2, "constitutional", num=_temp(">", 38.3)),
        CItem("leukopenia", "백혈구 <4,000/μL", 3, "hematologic",
              num=Num(_WBC, "<", 4000, lo=100, hi=200000, small_scale=(100, 1000))),
        CItem("thrombocytopenia", "혈소판 <100,000/μL", 4, "hematologic",
              num=Num(_PLT, "<", 100000, lo=1000, hi=2000000, small_scale=(1000, 1000))),
        CItem("hemolysis", "자가면역 용혈", 4, "hematologic",
              patterns=(r"자가면역\s?(성\s?)?용혈", r"쿰스\s?(검사\s?)?양성", r"coombs[^\n]{0,8}(양성|positive)",
                        r"autoimmune hemoly")),
        CItem("delirium", "섬망", 2, "neuropsychiatric", patterns=(r"섬망", r"delirium")),
        CItem("psychosis", "정신병 증상(망상·환각)", 3, "neuropsychiatric",
              patterns=(r"정신병", r"환청", r"환시", r"환각", r"망상", r"psychosis")),
        CItem("seizure", "경련 발작", 5, "neuropsychiatric", patterns=(r"경련", r"seizure", r"convuls")),
        CItem("alopecia", "비반흔성 탈모", 2, "mucocutaneous", patterns=(r"탈모", r"머리카락이\s?(많이\s?)?빠", r"alopecia")),
        CItem("oral_ulcer", "구강 궤양", 2, "mucocutaneous",
              patterns=(r"구강\s?궤양", r"구내염", r"입\s?안[^\n]{0,6}(궤양|헐)", r"입안이\s?헐", r"oral ulcer")),
        CItem("scle_dle", "아급성 피부 루푸스 또는 원판상 루푸스", 4, "mucocutaneous",
              patterns=(r"원판상", r"디스코이드", r"discoid", r"아급성\s?피부\s?루푸스", r"subacute cutaneous")),
        CItem("acle", "급성 피부 루푸스(나비 모양 뺨 발진 등)", 6, "mucocutaneous",
              patterns=(r"나비\s?모양", r"나비\s?형", r"협부\s?발진", r"뺨[^\n]{0,8}(발진|홍반)", r"malar", r"butterfly")),
        CItem("effusion", "흉막 또는 심낭 삼출", 5, "serosal",
              patterns=(r"흉수", r"흉막\s?삼출", r"늑막\s?삼출", r"심낭\s?(삼출|액)", r"pleural effusion",
                        r"pericardial effusion")),
        CItem("pericarditis", "급성 심낭염", 6, "serosal", patterns=(r"심낭염", r"심막염", r"pericarditis")),
        CItem("joint", "관절 침범(활막염 등)", 6, "musculoskeletal",
              patterns=(r"(?<!골)관절염", r"활막염", r"관절[^\n]{0,4}(부종|부어|붓)", r"synovitis", r"(?<!osteo)arthritis")),
        CItem("proteinuria", "단백뇨 >0.5 g/24시간", 4, "renal",
              num=Num((r"단백뇨", r"24시간\s?소변\s?단백", r"proteinuria"), ">", 0.5,
                      units=(("g", 1.0), ("mg", 0.001)), unit_required=True, lo=0, hi=50)),
        CItem("ln_2_5", "신장 생검 II형 또는 V형 루푸스 신염", 8, "renal",
              patterns=(r"루푸스\s?신염[^\n]{0,12}(class\s?|클래스\s?|제\s?)?(ii|v|2|5)(?![iv\d])\s?(형|급|class)?",
                        r"lupus nephritis[^\n]{0,10}class\s?(ii|v)(?![iv])")),
        CItem("ln_3_4", "신장 생검 III형 또는 IV형 루푸스 신염", 10, "renal",
              patterns=(r"루푸스\s?신염[^\n]{0,12}(class\s?|클래스\s?|제\s?)?(iii|iv|3|4)(?![iv\d])",
                        r"lupus nephritis[^\n]{0,10}class\s?(iii|iv)(?![iv])")),
        CItem("apl", "항인지질 항체 양성", 2, "apl",
              patterns=(r"항인지질", r"루푸스\s?항응고", r"lupus anticoagulant", r"(항\s?)?카디오리핀", r"anticardiolipin",
                        r"(β|베타|beta)\s?2\s?(-|\s)?(당단백|glycoprotein|gp)")),
        CItem("low_c3_or_c4", "C3 또는 C4 감소", 3, "complement",
              patterns=_low_c("c3") + _low_c("c4") + (r"보체[^\n]{0,4}(감소|저하|낮)", r"저보체",
                                                       r"hypocomplement")),
        CItem("low_c3_and_c4", "C3와 C4 모두 감소", 4, "complement", derive=_complement_both),
        CItem("sle_ab", "항dsDNA 또는 항Sm 항체 양성", 6, "specific_ab",
              patterns=(r"ds\s?-?dna", r"anti-?\s?sm(?![a-z])", r"항\s?-?sm(?![a-z])", r"항\s?스미스")),
    ),
    logic=lambda s, d: _sle(s, d),
    rule="ANA 양성이 진입 조건. 10개 영역에서 영역별 최고점 1개만 합산, 임상 항목 ≥1개와 합계 ≥10점이면 분류. "
         "다른 질환으로 더 잘 설명되는 항목은 세지 않음.",
    citation=C_SLE, verification="primary",
    note="Entry criterion, 10 domains, weights 2-10, >=10 points from the PubMed abstract; the full weight table "
         "(fever >38.3, leukopenia <4000, platelets <100k, proteinuria >0.5 g/24h, LN classes, complement, anti-dsDNA/Sm) "
         "checked against the full text at PMC6827566 (2026-09-27). Joint item is a proxy (synovitis/arthritis word; "
         "the '>=2 joints' detail is not parsed).",
)


def _ana(doc: Doc) -> tuple[str, str]:
    titer = _grab(doc, r"(" + A("ana") + r"|항핵\s?항체)[^\n\d]{0,15}1\s?:\s?(\d+)")
    if titer:
        return (MET if titer[-1] >= 80 else NOT_MET), f"1:{_fmt(titer[-1])}"
    pos, ev = doc.state((A("ana") + r"[^\n]{0,8}(양성|positive|\+)", r"항핵\s?항체[^\n]{0,8}(양성|positive)"))
    if pos == MET:
        return MET, ev
    neg, ev = doc.state((A("ana") + r"\s?(검사\s?)?(음성|negative)", r"항핵\s?항체\s?(검사\s?)?음성"))
    # "ANA 음성" is itself a negative word; the match being "negated" means the negative result is stated
    if neg != UNKNOWN:
        return NOT_MET, ev
    return UNKNOWN, ""


# --------------------------------------------------------------------------------------------
# 2. RA — 2010 ACR/EULAR
# --------------------------------------------------------------------------------------------

_SMALL_JOINTS = (r"손가락\s?(마디|관절)?", r"손목", r"중수지", r"근위\s?지(절|간)", r"발가락", r"중족지", r"손\s?관절",
                 A("mcp"), A("pip"), A("mtp"), r"wrist", r"knuckle")


def _ra_count(doc: Doc) -> tuple[str, str]:
    vals = doc.values(Num(pattern=r"(?P<v>\d+)\s?(개|곳)[^\n\d]{0,4}관절", lo=1, hi=80, agg="max"))
    vals += doc.values(Num(pattern=r"관절[^\n\d]{0,6}(?P<v>\d+)\s?(개|곳)", lo=1, hi=80))
    if vals:
        return MET, _fmt(max(vals))
    return UNKNOWN, ""


_SERO = r"(류마티스\s?인자|" + A("rf") + r"|acpa|anti-?\s?ccp|항\s?ccp|ccp\s?항체)"


def _sero(doc: Doc) -> tuple[str, str]:
    st, ev = doc.state((_SERO + r"[^\n]{0,8}(양성|상승|높|positive|\+)",))
    if st == MET:
        return MET, ev
    neg, ev = doc.state((_SERO + r"\s?(검사\s?)?(음성|negative|정상)",))
    return (NOT_MET, ev) if neg != UNKNOWN else (UNKNOWN, "")


def _ra(s: dict, doc: Doc) -> Outcome:
    n = float(s["joint_count"][1]) if _m(s, "joint_count") else None
    small = _m(s, "small_joints")
    bilateral = _m(s, "polyarticular")
    if n is not None:
        joint = (5 if n > 10 and small else 3 if small and n >= 4 else 2 if small else 1 if n >= 2 else 0)
    else:  # lower bound only
        joint = 2 if small else (1 if bilateral else 0)
    sero = 3 if _m(s, "sero_high") else 2 if _m(s, "sero_pos") else 0
    apr = 1 if _m(s, "apr") else 0
    dur = 1 if _m(s, "duration") else 0
    score = joint + sero + apr + dur
    detail = f"관절 {joint}+혈청 {sero}+염증 {apr}+기간 {dur}={score}점"
    if _n(s, "synovitis"):
        return Outcome("적용 불가: 활막염(관절 부종) 없음", "", score)
    if score >= 6 and _m(s, "synovitis"):
        return Outcome(f"분류 기준 충족 ({detail} ≥6)", "류마티스 관절염", score)
    return Outcome(f"아직 미충족 또는 미확인 ({detail}, 기준 ≥6; 관절 수 미상이면 하한값)", "", score)


RA = Criteria(
    id="ra_2010", short="2010 ACR/EULAR RA", name_ko="류마티스 관절염 분류 기준", name_en="2010 ACR/EULAR RA criteria",
    kind="classification",
    synonyms=("류마티스 관절염", "류마티스관절염", "류머티즘 관절염", "류마티스성 관절염", "rheumatoid arthritis"),
    items=(
        CItem("synovitis", "진입: 1개 이상 관절의 활막염(부종)",
              patterns=(r"관절[^\n]{0,6}(부종|부어|붓)", r"활막염", r"synovitis", r"joint swelling", r"swollen joint")),
        CItem("small_joints", "소관절 침범(손가락·손목·발가락 관절)",
              patterns=tuple(p + r"[^\n]{0,10}(부종|부어|붓|통증|아프|압통|관절염|강직|뻣뻣)" for p in _SMALL_JOINTS)),
        CItem("polyarticular", "여러 관절(양측·대칭) 침범",
              patterns=(r"양측[^\n]{0,8}(관절|손)", r"대칭", r"여러\s?(개\s?)?관절", r"다발성\s?관절", r"symmetric",
                        r"polyarthritis")),
        CItem("joint_count", "침범 관절 수", derive=_ra_count),
        CItem("sero_pos", "RF 또는 ACPA(항CCP) 양성", derive=lambda d: _sero(d)),
        CItem("sero_high", "RF 또는 ACPA 고역가(정상 상한 3배 초과)",
              patterns=(r"(류마티스\s?인자|" + A("rf") + r"|acpa|anti-?\s?ccp|항\s?ccp)[^\n]{0,12}(고역가|강양성|high)",)),
        CItem("apr", "CRP 또는 ESR 비정상",
              patterns=(r"(crp|esr|적혈구\s?침강|염증\s?수치|c-?반응성?\s?단백)[^\n]{0,8}(상승|증가|높|elevated)",)),
        CItem("duration", "증상 6주 이상",
              derive=lambda d: _dur_state(d, (r"관절", r"손가락", r"손목", r"아프", r"붓", r"통증", r"뻣뻣", r"증상"),
                                          ">=", 42)),
    ),
    logic=lambda s, d: _ra(s, d),
    rule="1개 이상 관절 활막염 + 더 나은 다른 진단 없음. 관절 분포 0–5, 혈청(RF/ACPA) 0–3, 급성기 반응 0–1, "
         "기간 ≥6주 0–1 합산 ≥6/10점이면 확정 RA.",
    citation=C_RA, verification="primary",
    note="Entry condition, 4 domains with ranges (0-5, 0-3, 0-1, 0-1) and >=6/10 from the PubMed abstract. 2026-09-29: "
         "Table 3 read in the simultaneous Ann Rheum Dis publication of the same criteria (Aletaha 2010, "
         "ard.bmj.com, PMID 20699241): 1 large joint 0; 2-10 large 1; 1-3 small 2; 4-10 small 3; >10 joints incl. >=1 "
         "small 5; low-positive RF/ACPA (> ULN, <= 3x ULN) 2, high-positive (> 3x ULN) 3, qualitative positive RF "
         "scored as low-positive; abnormal CRP or ESR 1; symptoms >= 6 weeks 1. Our joint count uses the stated total "
         "with 'small joints involved' as a proxy for the small-joint categories.",
)


def _dur_state(doc: Doc, anchors: tuple[str, ...], op: str, days: float, window: int = 25) -> tuple[str, str]:
    vals = doc.durations_days(anchors, window)
    if not vals:
        return UNKNOWN, ""
    v = max(vals)
    ok = v >= days if op == ">=" else v > days if op == ">" else v <= days if op == "<=" else v < days
    return (MET if ok else NOT_MET), f"{_fmt(round(v, 1))}일"


# --------------------------------------------------------------------------------------------
# 3. Takayasu arteritis and 4. GCA — 2022 ACR/EULAR
# --------------------------------------------------------------------------------------------

_LESION = (r"[^\n.]{0,35}(협착|폐색|동맥류|벽\s?비후|벽\s?두께|두께\s?증가|혈관염|좁아|막혀|stenosis|occlusion|aneurysm|"
           r"wall thicken)")
_TERRITORIES = {
    "좌 쇄골하": (r"(좌측|왼쪽|좌)\s?쇄골\s?하\s?동맥", r"left subclavian"),
    "우 쇄골하": (r"(우측|오른쪽|우)\s?쇄골\s?하\s?동맥", r"right subclavian"),
    "좌 경동맥": (r"(좌측|왼쪽|좌)\s?(총|공통)?\s?경동맥", r"left (common )?carotid"),
    "우 경동맥": (r"(우측|오른쪽|우)\s?(총|공통)?\s?경동맥", r"right (common )?carotid"),
    "흉부 대동맥": (r"(흉부|상행|하행)\s?대동맥|대동맥\s?궁", r"thoracic aorta|aortic arch|ascending aorta"),
    "복부 대동맥": (r"복부\s?대동맥", r"abdominal aorta"),
    "장간막동맥": (r"장간막\s?동맥|복강\s?동맥", r"mesenteric|celiac"),
    "신동맥": (r"신\s?동맥|콩팥\s?동맥", r"renal arter"),
    "무명동맥": (r"무명\s?동맥|완두\s?동맥", r"innominate|brachiocephalic"),
}
_BILATERAL = {"쇄골하": r"양측\s?쇄골\s?하\s?동맥|bilateral subclavian",
              "경동맥": r"양측\s?(총|공통)?\s?경동맥|bilateral (common )?carotid",
              "신동맥": r"양측\s?신\s?동맥|bilateral renal"}


def _territories(doc: Doc) -> set[str]:
    hit = set()
    for name, pats in _TERRITORIES.items():
        if doc.state(tuple(p + _LESION for p in pats))[0] == MET:
            hit.add(name)
    for name, pat in _BILATERAL.items():
        if doc.state((r"(" + pat + r")" + _LESION,))[0] == MET:
            if name == "신동맥":
                hit.add("신동맥")
            else:
                hit.update({f"좌 {name}", f"우 {name}"})
    return hit


def _terr_n(doc: Doc) -> tuple[str, str]:
    n = len(_territories(doc))
    return (MET, f"{n}개 영역") if n else (UNKNOWN, "")


def _paired(doc: Doc) -> tuple[str, str]:
    t = _territories(doc)
    for a in ("쇄골하", "경동맥"):
        if f"좌 {a}" in t and f"우 {a}" in t:
            return MET, f"양측 {a}"
    if doc.state((r"(" + _BILATERAL["신동맥"] + r")" + _LESION,))[0] == MET:
        return MET, "양측 신동맥"
    return UNKNOWN, ""


def _abd_aorta_branch(doc: Doc) -> tuple[str, str]:
    t = _territories(doc)
    if "복부 대동맥" in t and ({"신동맥", "장간막동맥"} & t):
        return MET, "복부 대동맥 + 신/장간막동맥"
    return UNKNOWN, ""


_IMAGING_VASCULITIS = (
    r"(대동맥|쇄골\s?하\s?동맥|경동맥|신\s?동맥|장간막\s?동맥|대혈관|분지)" + _LESION,
    r"(혈관|동맥|대동맥)\s?벽[^\n]{0,8}(비후|두꺼|조영\s?증강|부종)", r"대혈관\s?혈관염", r"large.vessel vasculitis",
    r"(subclavian|carotid|aort)[^\n]{0,15}(stenosis|occlusion|wall thicken)",
)


_ARM = r"(상지|팔|상완|arm)"


def _bp_diff(doc: Doc) -> tuple[str, str]:
    """Inter-arm systolic difference: an explicit 'difference' number or right/left arm readings."""
    d = doc.values(Num((r"(양팔|양측\s?팔|양측\s?상지|좌우|팔)[^\n\d]{0,8}혈압[^\n\d]{0,4}차이", r"혈압\s?차이",
                        r"arm[^\n\d]{0,20}difference"), ">=", 20, lo=0, hi=150))
    if d:
        return (MET if d[-1] >= 20 else NOT_MET), f"차이 {_fmt(d[-1])}"
    right = _grab(doc, r"(우측|오른쪽|오른|우|right)\s?" + _ARM + r"[^\n\d]{0,10}(\d{2,3})\s?/\s?\d{2,3}")
    left = _grab(doc, r"(좌측|왼쪽|왼|좌|left)\s?" + _ARM + r"[^\n\d]{0,10}(\d{2,3})\s?/\s?\d{2,3}")
    if right and left:
        diff = abs(right[-1] - left[-1])
        return (MET if diff >= 20 else NOT_MET), f"우 {_fmt(right[-1])}, 좌 {_fmt(left[-1])}"
    return UNKNOWN, ""


def _tak(s: dict, doc: Doc) -> Outcome:
    c = CRITERIA_BY_ID["takayasu_2022"]
    score = _pts(c, s) + (min(3, int(s["territories"][1].split("개")[0])) if _m(s, "territories") else 0)
    if _n(s, "age"):
        return Outcome("분류 불가: 진단 시 나이 >60세 (거대세포동맥염 기준 확인)", "", score)
    if _n(s, "imaging"):
        return Outcome("분류 불가: 대혈관 혈관염 영상 소견 없음", "", score)
    if score >= 5 and _m(s, "age") and _m(s, "imaging"):
        return Outcome(f"분류 기준 충족 ({_fmt(score)}점 ≥5, 60세 이하, 영상 소견)", "타카야수 동맥염", score)
    return Outcome(f"아직 미충족 또는 미확인 (현재 {_fmt(score)}점, 기준 ≥5 + 필수 조건 2개)", "", score)


TAK = Criteria(
    id="takayasu_2022", short="2022 ACR/EULAR TAK", name_ko="타카야수 동맥염 분류 기준",
    name_en="2022 ACR/EULAR Takayasu arteritis criteria", kind="classification",
    synonyms=("타카야수", "다카야스", "takayasu", "대혈관 혈관염", "large-vessel vasculitis", "large vessel vasculitis"),
    related=(r"(쇄골\s?하|경\s?|신\s?|장간막\s?|무명\s?)동맥\s?(협착|폐색)", r"대동맥\s?(궁\s?)?(협착|증후군|염)",
             r"무맥", r"pulseless", r"subclavian (artery )?stenosis", r"신혈관성\s?고혈압", r"renovascular"),
    items=(
        CItem("age", "필수: 진단 시 60세 이하", derive=_age_state("<=", 60)),
        CItem("imaging", "필수: 영상에서 대동맥·분지 혈관염 소견(협착·폐색·동맥류·벽 비후)", patterns=_IMAGING_VASCULITIS),
        CItem("female", "여성", 1, derive=_female),
        CItem("angina", "협심증(허혈성 흉통)", 2, patterns=(r"협심", r"angina")),
        CItem("claudication", "팔다리 파행(사용 시 통증·피로)", 2,
              patterns=(r"파행", r"claudication", r"(팔|다리)[^\n]{0,6}(쓰면|사용하면|들면|걸으면)[^\n]{0,8}(아프|저리|피로|힘)")),
        CItem("bruit", "동맥 잡음", 2,
              patterns=(r"(경동맥|쇄골\s?하|쇄골\s?위|복부|대동맥|혈관|동맥)\S{0,3}\s?잡음", r"bruit")),
        CItem("arm_pulse", "팔 맥박 감소", 2,
              patterns=(r"(요골|상완|상지|팔|손목)\S{0,3}\s?맥박[^\n]{0,18}(약|감소|촉지되지|안 잡|잡히지|없)", r"무맥",
                        r"pulseless", r"(radial|brachial) pulse[^\n]{0,10}(diminish|absent|weak|reduced)")),
        CItem("carotid", "경동맥 맥박 감소 또는 압통", 2,
              patterns=(r"경동맥[^\n]{0,6}(압통|맥박[^\n]{0,6}(약|감소))", r"carotidynia", r"carotid[^\n]{0,8}tender")),
        CItem("bp_diff", "양팔 수축기 혈압 차이 ≥20 mmHg", 1, derive=lambda d: _bp_diff(d)),
        CItem("territories", "침범 동맥 영역 수(1개 +1, 2개 +2, ≥3개 +3)", derive=_terr_n),
        CItem("paired", "좌우 쌍 동맥 침범", 1, derive=_paired),
        CItem("abd_branch", "복부 대동맥 + 신동맥 또는 장간막동맥 침범", 3, derive=_abd_aorta_branch),
    ),
    logic=lambda s, d: _tak(s, d),
    rule="필수: 진단 시 ≤60세 + 영상상 대혈관 혈관염. 여성1, 협심증2, 파행2, 동맥 잡음2, 팔 맥박 감소2, 경동맥 이상2, "
         "양팔 혈압차 ≥20 1, 침범 영역 수 1–3, 쌍 동맥1, 복부 대동맥+신/장간막3 합산 ≥5점이면 분류.",
    subtype_rule="혈관 협착이 동맥경화·섬유근이형성 같은 모방 질환이 아니고 ≤60세·혈관염 영상 소견·≥5점이면 "
                 "타카야수 동맥염으로 명명. >60세면 거대세포동맥염 기준을 봄.",
    citation=C_TAK, verification="primary",
    note="Absolute requirements, all item weights and the >=5 threshold stated in the PubMed abstract. Territory "
         "counting from imaging words is our heuristic (named arteries with stenosis/occlusion/aneurysm/wall thickening).",
)


def _gca(s: dict, doc: Doc) -> Outcome:
    c = CRITERIA_BY_ID["gca_2022"]
    score = _pts(c, s)
    if _n(s, "age"):
        return Outcome("분류 불가: 진단 시 나이 <50세 (타카야수 동맥염 기준 확인)", "", score)
    if score >= 6 and _m(s, "age"):
        return Outcome(f"분류 기준 충족 ({_fmt(score)}점 ≥6, 50세 이상)", "거대세포동맥염", score)
    return Outcome(f"아직 미충족 또는 미확인 (현재 {_fmt(score)}점, 기준 ≥6 + 50세 이상)", "", score)


def _esr_crp(doc: Doc) -> tuple[str, str]:
    esr = doc.values(Num(_ESR, ">=", 50, lo=0, hi=200))
    crp = doc.values(Num(_CRP, ">=", 10, units=_CRP_UNITS, unit_required=True, lo=0, hi=1000))
    if (esr and esr[-1] >= 50) or (crp and crp[-1] >= 10):
        return MET, (f"ESR {_fmt(esr[-1])}" if esr and esr[-1] >= 50 else f"CRP {_fmt(crp[-1])} mg/L")
    if esr and crp:
        return NOT_MET, f"ESR {_fmt(esr[-1])}, CRP {_fmt(crp[-1])} mg/L"
    return UNKNOWN, ""


GCA = Criteria(
    id="gca_2022", short="2022 ACR/EULAR GCA", name_ko="거대세포동맥염 분류 기준",
    name_en="2022 ACR/EULAR giant cell arteritis criteria", kind="classification",
    synonyms=("거대세포 동맥염", "거대세포동맥염", "측두동맥염", "측두 동맥염", "giant cell arteritis", "temporal arteritis",
              RX(A("gca")), "대혈관 혈관염", "large-vessel vasculitis", "large vessel vasculitis"),
    related=(r"류마티스성?\s?다발(성)?\s?근육?통", r"polymyalgia"),
    items=(
        CItem("age", "필수: 진단 시 50세 이상", derive=_age_state(">=", 50)),
        CItem("biopsy_halo", "측두동맥 생검 양성 또는 초음파 halo 징후", 5,
              patterns=(r"측두\s?동맥\s?(생검|조직\s?검사)[^\n]{0,15}(양성|거대세포|혈관염|육아종)", r"halo", r"헤일로",
                        r"temporal artery biopsy[^\n]{0,15}(positive|giant cell|arteritis)")),
        CItem("esr_crp", "ESR ≥50 mm/h 또는 CRP ≥10 mg/L", 3, derive=_esr_crp),
        CItem("visual_loss", "갑작스러운 시력 소실", 3,
              patterns=(r"갑자기[^\n]{0,8}(안\s?보|시력)", r"(갑작스러운|급성|급격한)\s?시력\s?(소실|상실|저하)", r"흑암시",
                        r"sudden (visual|vision) loss", r"amaurosis")),
        CItem("shoulder_neck", "어깨·목 아침 강직", 2,
              patterns=(r"(어깨|목)[^\n]{0,12}(아침|조조)[^\n]{0,8}(뻣뻣|강직|굳)", r"(아침|조조)\s?(강직|뻣뻣)[^\n]{0,12}(어깨|목)",
                        r"morning stiffness[^\n]{0,15}(shoulder|neck)")),
        CItem("jaw", "턱 또는 혀 파행", 2,
              patterns=(r"(턱|혀)[^\n]{0,6}파행", r"(씹을|씹으면|씹을 때)[^\n]{0,10}(턱)?[^\n]{0,4}(아프|통증|피로|힘들)",
                        r"jaw claudication")),
        CItem("temporal_headache", "새로 생긴 관자놀이 두통", 2,
              patterns=(r"(관자놀이|측두부?)[^\n]{0,8}(두통|통증|아프|아파)", r"temporal headache")),
        CItem("scalp", "두피 압통", 2, patterns=(r"두피[^\n]{0,6}(압통|아프|아파|통증|민감|쓰라)", r"scalp tender")),
        CItem("ta_exam", "측두동맥 진찰 이상(압통·박동 감소·결절)", 2,
              patterns=(r"측두\s?동맥[^\n]{0,10}(압통|박동[^\n]{0,4}(감소|약|없)|결절|비후|굵|딱딱)",
                        r"temporal artery[^\n]{0,10}(tender|pulseless|nodular|thicken)")),
        CItem("axillary", "영상에서 양측 겨드랑동맥 침범", 2, patterns=(r"양측\s?(액와|겨드랑)\s?동맥", r"bilateral axillary")),
        CItem("pet_aorta", "FDG-PET에서 대동맥 전반 섭취 증가", 2,
              patterns=(r"(pet|양전자)[^\n]{0,25}대동맥[^\n]{0,12}(섭취|활성|증가)", r"fdg[^\n]{0,25}aort")),
    ),
    logic=lambda s, d: _gca(s, d),
    rule="필수: 진단 시 ≥50세. 측두동맥 생검 양성/초음파 halo 5, ESR≥50 또는 CRP≥10 mg/L 3, 갑작스러운 시력 소실 3, "
         "어깨·목 아침 강직/턱·혀 파행/새 관자놀이 두통/두피 압통/측두동맥 이상/양측 겨드랑동맥/대동맥 PET 각 2 → ≥6점이면 분류.",
    subtype_rule="대혈관 혈관염: ≥50세·≥6점이면 거대세포동맥염, ≤60세·영상 소견·≥5점이면 타카야수 동맥염.",
    citation=C_GCA, verification="primary",
    note="Absolute requirement, every item weight and the >=6 threshold stated in the PubMed abstract.",
)


# --------------------------------------------------------------------------------------------
# 5. Kawasaki disease — AHA 2017
# --------------------------------------------------------------------------------------------

_FEVER_ANCHORS = (r"열", r"발열", r"fever", r"febrile")


def _kd(s: dict, doc: Doc) -> Outcome:
    feats = sum(_m(s, k) for k in ("conj", "oral", "rash", "extrem", "lymph"))
    known_absent = sum(_n(s, k) for k in ("conj", "oral", "rash", "extrem", "lymph"))
    if _n(s, "fever5"):
        return Outcome(f"발열 5일 미만 (주요 증상 {feats}/5; 4개 이상이면 4일째 진단 가능)", "", float(feats))
    if _m(s, "fever5") and feats >= 4:
        return Outcome(f"전형(완전) 가와사키병 기준 충족 (발열 ≥5일 + 주요 증상 {feats}/5)", "가와사키병", float(feats))
    if _m(s, "fever5") and feats in (2, 3):
        supp = sum(_m(s, k) for k in ("albumin", "wbc", "plt", "alt", "pyuria", "anemia"))
        if _m(s, "inflam") and (supp >= 3 or _m(s, "echo")):
            return Outcome(f"불완전 가와사키병 기준 충족 (주요 {feats}/5, CRP/ESR 상승, 보조 검사 {supp}개"
                           f"{' 또는 심초음파 이상' if _m(s, 'echo') else ''})", "불완전 가와사키병", float(feats))
        return Outcome(f"불완전 가와사키병 평가 대상 (주요 {feats}/5): CRP ≥3.0 mg/dL 또는 ESR ≥40이면 보조 검사 "
                       f"≥3개 또는 심초음파 확인 (현재 보조 {supp}개)", "", float(feats))
    if known_absent >= 4:
        return Outcome(f"주요 증상 부족 ({feats}/5)", "", float(feats))
    return Outcome(f"미확인 항목 있음 (발열 기간 {_STATE_KO[s['fever5'][0]]}, 주요 증상 {feats}/5)", "", float(feats))


def _kd_inflam(doc: Doc) -> tuple[str, str]:
    crp = doc.values(Num(_CRP, ">=", 30, units=_CRP_UNITS, unit_required=True, lo=0, hi=1000))
    esr = doc.values(Num(_ESR, ">=", 40, lo=0, hi=200))
    if (crp and crp[-1] >= 30) or (esr and esr[-1] >= 40):
        return MET, f"CRP {_fmt(crp[-1] / 10)} mg/dL" if crp and crp[-1] >= 30 else f"ESR {_fmt(esr[-1])}"
    if crp and esr:
        return NOT_MET, ""
    return UNKNOWN, ""


KD = Criteria(
    id="kawasaki_aha2017", short="AHA 2017 Kawasaki", name_ko="가와사키병 진단 기준", name_en="AHA 2017 Kawasaki disease",
    kind="diagnostic", synonyms=("가와사키", "kawasaki", "점막피부 림프절 증후군", "mucocutaneous lymph node"),
    subtypes=(Subtype("불완전 가와사키병", ("불완전 가와사키", "비전형 가와사키", "incomplete kawasaki", "atypical kawasaki")),
              Subtype("가와사키병", ("가와사키병", "전형적 가와사키", "완전 가와사키", "kawasaki disease"))),
    items=(
        CItem("fever5", "발열 5일 이상",
              derive=lambda d: _dur_state(d, _FEVER_ANCHORS, ">=", 5, window=12)),
        CItem("conj", "양측 안구 결막 충혈(삼출물 없음)",
              patterns=(r"결막[^\n]{0,4}(충혈|주사)", r"눈이?[^\n]{0,4}(충혈|빨갛|붉)", r"conjunctival (injection|hyperemia)",
                        r"red eyes")),
        CItem("oral", "입술·구강 변화(붉고 갈라진 입술, 딸기혀, 구강 점막 발적)",
              patterns=(r"딸기\s?(모양\s?)?혀", r"입술[^\n]{0,6}(갈라|붉|빨갛|균열|터)", r"구강\s?점막[^\n]{0,4}(발적|충혈)",
                        r"strawberry tongue", r"(cracked|red) lips")),
        CItem("rash", "발진", patterns=(r"발진", r"rash")),
        CItem("extrem", "손발 변화(급성기 홍반·부종, 아급성기 손발끝 낙설)",
              patterns=(r"(손|발|손발|손바닥|발바닥)\S{0,3}\s?(부종|붓|부었|홍반|붉|빨갛)", r"(손끝|발끝|손톱 주위)[^\n]{0,6}(껍질|낙설|벗겨)",
                        r"(periungual )?desquamation", r"(hand|feet|foot)[^\n]{0,10}(edema|swelling|erythema)")),
        CItem("lymph", "경부 림프절 비대(대개 한쪽, ≥1.5 cm)",
              patterns=(r"(목|경부)[^\n]{0,8}림프절[^\n]{0,6}(비대|종대|커|붓|만져|촉지)", r"목에[^\n]{0,4}(멍울|혹)",
                        r"cervical lymphadenopathy")),
        CItem("inflam", "CRP ≥3.0 mg/dL 또는 ESR ≥40 mm/h", derive=_kd_inflam),
        CItem("albumin", "보조: 알부민 ≤3.0 g/dL",
              num=Num((r"알부민", r"albumin"), "<=", 3.0, units=(("g/dl", 1.0), ("g/l", 0.1)), lo=0.5, hi=6)),
        CItem("wbc", "보조: 백혈구 ≥15,000/μL",
              num=Num(_WBC, ">=", 15000, lo=100, hi=200000, small_scale=(100, 1000))),
        CItem("plt", "보조: 발병 7일 이후 혈소판 ≥450,000/μL",
              num=Num(_PLT, ">=", 450000, lo=1000, hi=3000000, small_scale=(3000, 1000))),
        CItem("alt", "보조: ALT 상승", patterns=(A("alt") + r"[^\n]{0,8}(상승|증가|높|elevated)", r"간\s?수치[^\n]{0,4}(상승|증가|높)")),
        CItem("pyuria", "보조: 소변 백혈구 ≥10/HPF(무균 농뇨)", patterns=(r"농뇨", r"pyuria", r"소변[^\n]{0,8}백혈구[^\n]{0,6}(증가|많|다수)")),
        CItem("anemia", "보조: 연령 대비 빈혈", patterns=(r"빈혈", r"anemia")),
        CItem("echo", "심초음파 관상동맥 이상(확장·동맥류)",
              patterns=(r"관상\s?동맥[^\n]{0,12}(확장|동맥류|이상|z\s?-?점수)", r"coronary[^\n]{0,12}(aneurysm|dilat|ectasia)")),
    ),
    logic=lambda s, d: _kd(s, d),
    rule="전형: 발열 ≥5일 + 주요 증상 5개 중 ≥4개(4개 이상이면 4일째도 가능). 불완전: 발열 ≥5일 + 2–3개 → "
         "CRP ≥3.0 mg/dL 또는 ESR ≥40이면 보조 검사 ≥3개 또는 심초음파 이상 시 치료 대상.",
    subtype_rule="주요 증상 ≥4개면 가와사키병(전형), 2–3개 + 보조 기준 충족이면 불완전 가와사키병.",
    citation=C_KD, verification="secondary",
    note="Bibliographic data verified; the abstract only says an updated algorithm exists; the full text "
         "(ahajournals) returned 403. 2026-09-29: an open-access paper citing the 2017 statement (Agha 2017, "
         "PMC5856965, Table 1 and text) gives fever >= 5 days with principal features (rash, bilateral conjunctival "
         "injection, extremity changes, lymphadenopathy, oropharyngeal changes), incomplete KD with 2-3 features, and "
         "the supplemental labs (albumin <= 3.0 g/dL, anemia for age, ALT elevation, platelets >= 450,000 after day 7, "
         "WBC >= 15,000, urine WBC >= 10/hpf). The CRP >= 3.0 mg/dL / ESR >= 40 entry threshold and the '>= 3 labs' "
         "count are still reviewer knowledge. Note: AHA published a 2024 update (not consulted).",
)


# --------------------------------------------------------------------------------------------
# 6. Infective endocarditis — 2023 Duke-ISCVID
# --------------------------------------------------------------------------------------------

_TYPICAL_ORG = (r"황색\s?포도", r"s\.?\s?aureus", A("mrsa"), A("mssa"), r"사슬알균", r"연쇄상?\s?구균", r"streptococc",
                r"viridans", r"장구균", r"장알균", r"enterococc", A("hacek"), r"gallolyticus", r"bovis",
                r"lugdunensis")
_MULTI_SETS = r"(2|3|4|두|세|네|여러|모든|all|multiple|repeated)\s?(개\s?)?(세트|set|병|bottle|회|번|쌍)"


def _micro(doc: Doc) -> tuple[str, str]:
    bc_pos, ev = doc.state((r"(혈액\s?배양|blood culture)[^\n]{0,40}(양성|자라|자람|자랐|검출|동정|positive|grew|grow)",))
    if bc_pos != MET:
        if bc_pos == NOT_MET:
            return NOT_MET, "혈액 배양 음성"
        return UNKNOWN, ""
    typical = doc.state(_TYPICAL_ORG)[0] == MET
    multi = re.search(r"(혈액\s?배양|blood culture)[^\n]{0,40}" + _MULTI_SETS + "|" + _MULTI_SETS
                      + r"[^\n]{0,20}(혈액\s?배양|blood culture)", doc.text) is not None
    if typical and multi:
        return MET, "전형적 균, 여러 세트 양성"
    return NOT_MET, "주 기준 미달(균 종류·세트 수 불명)"


def _micro_minor(doc: Doc) -> tuple[str, str]:
    major = _micro(doc)
    bc_pos, ev = doc.state((r"(혈액\s?배양|blood culture)[^\n]{0,40}(양성|자라|자람|자랐|검출|동정|positive|grew|grow)",))
    if bc_pos == MET and major[0] != MET:
        return MET, "혈액 배양 양성(주 기준 미달)"
    return (NOT_MET, "") if major[0] == MET or bc_pos == NOT_MET else (UNKNOWN, "")


def _ie(s: dict, doc: Doc) -> Outcome:
    major = sum(_m(s, k) for k in ("micro_major", "imaging_major", "surgery_major"))
    minor = sum(_m(s, k) for k in ("predisposition", "fever", "vascular", "immunologic", "micro_minor"))
    tag = f"주 기준 {major}, 부 기준 {minor}"
    if major >= 2 or (major == 1 and minor >= 3) or minor >= 5:
        return Outcome(f"확정 IE ({tag})", "감염성 심내막염", float(major * 10 + minor))
    if (major == 1 and minor >= 1) or minor >= 3:
        return Outcome(f"가능 IE ({tag}): 확정에는 주 기준 추가(심초음파·혈액 배양) 필요", "", float(major * 10 + minor))
    return Outcome(f"기준 미달 또는 미확인 ({tag})", "", float(major * 10 + minor))


IE = Criteria(
    id="duke_iscvid_2023", short="2023 Duke-ISCVID", name_ko="감염성 심내막염 진단 기준",
    name_en="2023 Duke-ISCVID infective endocarditis criteria", kind="diagnostic",
    synonyms=("심내막염", "endocarditis"), exclude=(r"비세균성", r"비감염성", r"marantic", r"libman", r"nonbacterial"),
    items=(
        CItem("micro_major", "주: 전형적 원인균이 여러 혈액 배양 세트에서 자람", derive=_micro),
        CItem("imaging_major", "주: 영상(심초음파·심장 CT·PET)에서 증식물·농양·천공·새 판막 역류 등",
              patterns=(r"증식물", r"우종", r"vegetation", r"판막\s?(주위\s?)?(농양|천공|누공|가성\s?동맥류)",
                        r"(새로운|새로 생긴|신규)\s?(판막\s?)?역류", r"valv[^\n]{0,15}(abscess|perforation)", r"dehiscence",
                        r"판막\s?이개")),
        CItem("surgery_major", "주: 수술 중 직접 확인", patterns=(r"수술[^\n]{0,15}(심내막염|증식물)[^\n]{0,6}(확인|소견|관찰)",)),
        CItem("predisposition", "부: 소인(인공 판막, 판막 질환, 선천성 심질환, IE 과거력, 심장 기기, 주사 약물 사용 등)",
              patterns=(r"인공\s?(심장\s?)?판막", r"판막\s?(질환|치환|수술)", r"선천성\s?심", r"심내막염[^\n]{0,4}(과거력|병력|앓)",
                        r"(정맥\s?)?주사\s?(마약|약물)", r"마약[^\n]{0,6}주사", r"필로폰[^\n]{0,6}주사", r"injection drug", A("ivdu"),
                        r"(심박동기|제세동기|이식형\s?(심장\s?)?(기기|장치))", r"비후성\s?심근", r"(승모판|대동맥판)\s?(역류|협착|탈출)",
                        r"류마티스\s?심장")),
        CItem("fever", "부: 발열 >38.0°C", num=_temp(">", 38.0)),
        CItem("vascular", "부: 혈관 현상(색전, 패혈성 경색, 진균성 동맥류, 두개내 출혈, 결막 출혈, Janeway 병변)",
              patterns=(r"색전", r"(뇌|비장|신장)\s?경색", r"패혈성\s?(폐\s?)?(색전|경색)", r"진균성\s?동맥류", r"두개\s?내\s?출혈",
                        r"결막\s?출혈", r"janeway", r"제인웨이", r"emboli", r"mycotic aneurysm")),
        CItem("immunologic", "부: 면역 현상(Osler 결절, Roth 반점, 사구체신염, 류마티스 인자 양성)",
              patterns=(r"osler", r"오슬러", r"roth", r"로스\s?반점", r"사구체\s?신염", r"glomerulonephritis",
                        r"(류마티스\s?인자|" + A("rf") + r")[^\n]{0,6}(양성|positive)")),
        CItem("micro_minor", "부: 주 기준에 못 미치는 혈액 배양 양성", derive=_micro_minor),
    ),
    logic=lambda s, d: _ie(s, d),
    rule="확정: 주 2개, 또는 주 1 + 부 3, 또는 부 5. 가능: 주 1 + 부 1, 또는 부 3.",
    citation=C_IE, verification="primary",
    note="Major/minor categories, fever >38.0 C and definite/possible logic checked against the full text "
         "(PMC10681650, 2026-09-27). Our extraction simplifies the microbiology major criterion (typical organism + "
         "several positive sets); Coxiella/Bartonella/PCR paths and the 'rejected' rules are not parsed.",
)


# --------------------------------------------------------------------------------------------
# 7. AKI — KDIGO 2012
# --------------------------------------------------------------------------------------------

_CR_LAB = r"(크레아티닌|creatinine|(?<![a-z])s?cr(?![a-z]))"
_CR_UNIT = r"\s?(mg/dl|umol/l|μmol/l|µmol/l)?"


def _cr_values(doc: Doc) -> tuple[float | None, float | None]:
    """(baseline, current) serum creatinine in mg/dL, if the text makes both explicit."""
    def conv(v: str, unit: str | None) -> float:
        x = float(v)
        return x / 88.4 if unit and "mol" in unit else x

    m = re.search(_CR_LAB + r"[^\n\d]{0,20}(\d+(?:\.\d+)?)" + _CR_UNIT + r"\s?(에서|→|->|->|from)[^\n\d]{0,6}(\d+(?:\.\d+)?)"
                  + _CR_UNIT, doc.text)
    if m:
        return conv(m.group(2), m.group(3)), conv(m.group(5), m.group(6))
    base = re.search(r"(기저|평소|이전|과거|baseline|previous)[^\n\d]{0,12}" + _CR_LAB + r"[^\n\d]{0,8}(\d+(?:\.\d+)?)"
                     + _CR_UNIT, doc.text)
    vals = [(mm.start(), conv(mm.group(2), mm.group(3)))
            for mm in re.finditer(_CR_LAB + r"[^\n\d]{0,10}(\d+(?:\.\d+)?)" + _CR_UNIT, doc.text)]
    vals = [(p, v) for p, v in vals if 0.1 <= v <= 30]
    if base:
        b = conv(base.group(3), base.group(4))
        cur = [v for p, v in vals if not (base.start() <= p <= base.end())]
        return b, (cur[-1] if cur else None)
    return None, (vals[-1][1] if vals else None)


def _aki_stage(doc: Doc) -> tuple[int | None, str]:
    base, cur = _cr_values(doc)
    rrt = doc.state((r"(투석|crrt|신대체\s?요법)[^\n]{0,8}(시작|시행|필요|받)", r"renal replacement", r"dialysis (was )?(started|initiated)"))[0] == MET
    anuria = doc.state((r"무뇨", r"anuria"))[0] == MET
    uo = doc.values(Num(pattern=r"(\d+(?:\.\d+)?)\s?(ml/kg/(h|hr|시간))", lo=0, hi=10))
    if rrt:
        return 3, "신대체요법 시작"
    stage, why = None, ""
    if base and cur and base > 0:
        r, d = cur / base, cur - base
        if r >= 3 or (cur >= 4.0 and d >= 0.3):
            stage, why = 3, f"Cr {base:g}→{cur:g} ({r:.1f}배)"
        elif r >= 2:
            stage, why = 2, f"Cr {base:g}→{cur:g} ({r:.1f}배)"
        elif r >= 1.5 or d >= 0.3:
            stage, why = 1, f"Cr {base:g}→{cur:g} ({r:.1f}배, +{d:.2f})"
        else:
            stage, why = 0, f"Cr {base:g}→{cur:g} (기준 미달)"
    if anuria:
        return 3, (why + ", " if why else "") + "무뇨"
    if uo and uo[-1] < 0.3 and stage in (None, 0, 1, 2):
        hours = doc.durations_days((r"소변", r"ml/kg"))
        if hours and max(hours) * 24 >= 24:
            return 3, f"소변량 {uo[-1]:g} mL/kg/h ≥24시간"
    return stage, why


def _aki(s: dict, doc: Doc) -> Outcome:
    stage, why = _aki_stage(doc)
    if stage is None:
        return Outcome("병기 미확인: 기저 크레아티닌과 현재 값(또는 시간당 소변량)이 필요", "", None)
    if stage == 0:
        return Outcome(f"AKI 크레아티닌 기준 미달 ({why})", "", 0.0)
    return Outcome(f"AKI {stage}단계 ({why})", "", float(stage))


def _aki_item(n: int) -> Callable[[Doc], tuple[str, str]]:
    def f(doc: Doc) -> tuple[str, str]:
        st, why = _aki_stage(doc)
        if st is None:
            return UNKNOWN, ""
        return (MET if st >= n else NOT_MET), why
    return f


AKI = Criteria(
    id="kdigo_aki_2012", short="KDIGO AKI", name_ko="급성 신손상 정의·병기", name_en="KDIGO 2012 AKI definition/staging",
    kind="staging",
    synonyms=("급성 신손상", "급성신손상", "급성 신부전", "급성신부전", "급성 콩팥 손상", "acute kidney injury",
              "acute renal failure", RX(A("aki"))),
    related=(r"급성\s?세뇨관\s?괴사", r"acute tubular necrosis", r"신전성\s?(질소혈증|신부전)", r"prerenal"),
    items=(
        CItem("stage1", "정의/1단계: 48시간 내 Cr ≥0.3 mg/dL 상승 또는 7일 내 기저치 1.5–1.9배, 또는 소변 <0.5 mL/kg/h 6–12시간",
              derive=_aki_item(1)),
        CItem("stage2", "2단계: 기저치 2.0–2.9배 또는 소변 <0.5 mL/kg/h ≥12시간", derive=_aki_item(2)),
        CItem("stage3", "3단계: 기저치 ≥3배, Cr ≥4.0 mg/dL로 상승, 신대체요법 시작, 소변 <0.3 mL/kg/h ≥24시간 또는 "
                        "무뇨 ≥12시간", derive=_aki_item(3)),
    ),
    logic=lambda s, d: _aki(s, d),
    rule="AKI: 48시간 내 Cr ≥0.3 mg/dL 상승, 또는 7일 내 기저치 ≥1.5배, 또는 소변 <0.5 mL/kg/h 6시간. "
         "병기 1(1.5–1.9배), 2(2.0–2.9배), 3(≥3배, Cr ≥4.0, 투석 시작) 중 가장 높은 것.",
    citation=C_AKI, verification="primary",
    note="Definition 2.1.1 and Table 2 staging checked in the KDIGO 2012 guideline PDF (kdigo.org, 2026-09-27). "
         "PMID 22890468 is Khwaja's summary of the same guideline (Nephron Clin Pract 2012). Stage-by-urine-output "
         "is parsed only for anuria and <0.3 mL/kg/h >=24 h; the eGFR <35 pediatric rule is not parsed. "
         "No chronic (CKD) vs acute judgement is made without a stated baseline.",
)


# --------------------------------------------------------------------------------------------
# 8. DKA / HHS — ADA/EASD/JBDS/AACE/DTS 2024 consensus
# --------------------------------------------------------------------------------------------

_GLU = Num((r"혈당", r"포도당", r"glucose", A("bst"), r"blood sugar", r"당\s?수치"), units=(("mg/dl", 1.0), ("mmol/l", 18.0)),
           lo=20, hi=3000)
_PH = Num((A("ph"), r"산도"), lo=6.5, hi=7.8, window=8)
_HCO3 = Num((r"중탄산", r"hco3-?", r"bicarbonate", r"(?<![a-z])t?co2(?![a-z])", r"총\s?이산화탄소"), lo=1, hi=45, window=8)
_OSM = Num((r"(유효\s?)?(혈청\s?)?삼투압", r"osmolality", r"osmolarity", A("osm")), lo=240, hi=480)
# effective osmolality (2Na + glucose) for HHS, ADA 2024 Fig. 2B: > 300 mOsm/kg (total osmolality > 320)
_OSM_EFF = Num((r"유효\s?(혈청\s?)?삼투압", r"effective\s?(serum\s?)?osmolality", r"effective\s?osmolarity"), ">", 300,
               lo=240, hi=480)
_BOHB = Num((r"(베타|β|b)\s?-?(히드록시|하이드록시|hydroxy)\s?부티르", r"β-?ohb", A("bhb"), A("bohb"), r"혈중\s?케톤"),
            ">=", 3.0, units=(("mmol/l", 1.0), ("mm", 1.0)), lo=0, hi=20)


def _with(num: Num, op: str, value: float) -> Num:
    return Num(num.labels, op, value, num.units, num.unit_required, num.lo, num.hi, num.window, num.pattern,
               num.small_scale, num.agg)


def _ketones(doc: Doc) -> tuple[str, str]:
    b = doc.values(_BOHB)
    if b:
        return (MET if b[-1] >= 3.0 else NOT_MET), f"β-OHB {b[-1]:g}"
    st, ev = doc.state((r"(소변|요)\s?케톤[^\n]{0,8}([2-4]\s?\+|\+\s?\+|강양성|양성\s?\(?[2-4]\+)", r"(urine )?ketones?[^\n]{0,8}[2-4]\+",
                        r"케톤뇨", r"ketonuria", r"ketonemia", r"케톤혈증"))
    if st == MET:
        return MET, ev
    neg, ev2 = doc.state((r"(소변|요|혈중)\s?케톤[^\n]{0,4}(음성|없|negative|\(-\)|trace|미량|±|1\+)",))
    if neg != UNKNOWN or st == NOT_MET:
        return NOT_MET, ev2 or ev
    return UNKNOWN, ""


def _osm_state(doc: Doc) -> tuple[str, str]:
    """HHS hyperosmolality (ADA 2024 Fig. 2B): effective osmolality > 300 or total serum osmolality > 320 mOsm/kg."""
    eff = doc.values(_OSM_EFF)
    tot = doc.values(_with(_OSM, ">", 320))
    if (eff and eff[-1] > 300) or (tot and tot[-1] > 320):
        return MET, (f"유효 {eff[-1]:g}" if eff and eff[-1] > 300 else f"{tot[-1]:g}")
    if eff or tot:
        return NOT_MET, f"{(eff or tot)[-1]:g}"
    return UNKNOWN, ""


def _dka_hhs(s: dict, doc: Doc) -> Outcome:
    hyper = _m(s, "glu200") or _m(s, "diabetes")
    acid = _m(s, "ph") or _m(s, "hco3")
    # HHS "absence of acidosis" (ADA 2024 Fig. 2B): pH >= 7.3 and bicarbonate >= 15 mmol/L
    hco3 = doc.values(_HCO3)
    no_acid = _n(s, "ph") and bool(hco3) and hco3[-1] >= 15
    dka = hyper and _m(s, "ketones") and acid
    hhs = _m(s, "glu600") and _m(s, "osm") and no_acid and _n(s, "ketones")
    if dka and _m(s, "glu600") and _m(s, "osm"):
        return Outcome("DKA 기준 충족 + 고혈당·고삼투압 동반 (DKA-HHS 혼합 가능)", "당뇨병성 케톤산증")
    if dka:
        return Outcome("DKA 기준 충족 (고혈당/당뇨 + 케톤 ≥3.0 mmol/L 또는 요케톤 ≥2+ + pH <7.3 또는 HCO3 <18)",
                       "당뇨병성 케톤산증")
    if hhs:
        return Outcome("HHS 기준 충족 (혈당 ≥600, 유효 삼투압 >300 또는 총 삼투압 >320, pH ≥7.3·HCO3 ≥15, 유의한 케톤 없음)",
                       "고혈당성 고삼투압 상태")
    miss = [CRITERIA_BY_ID["dka_hhs_2024"].items[i].text.split(":")[0]
            for i, k in enumerate(("glu200", "glu600", "diabetes", "ketones", "ph", "hco3", "osm")) if _u(s, k)]
    return Outcome("DKA/HHS 판정 불가 (미확인: " + ", ".join(miss[:4]) + ")" if miss else "DKA/HHS 기준 미충족", "")


DKA = Criteria(
    id="dka_hhs_2024", short="ADA 2024 DKA/HHS", name_ko="당뇨병성 케톤산증/고혈당성 고삼투압 상태 진단 기준",
    name_en="2024 hyperglycemic crises consensus (DKA/HHS)", kind="diagnostic",
    synonyms=("케톤산증", "당뇨병성 케톤", RX(A("dka")), "ketoacidosis", "고삼투압", "고삼투성", RX(A("hhs")), RX(A("hhns")),
              "hyperosmolar", "고혈당 위기", "hyperglycemic crisis"),
    exclude=(r"알코올성\s?케톤", r"기아\s?케톤", r"alcoholic ketoacidosis", r"starvation ketosis"),
    subtypes=(Subtype("당뇨병성 케톤산증", ("케톤산증", RX(A("dka")), "ketoacidosis")),
              Subtype("고혈당성 고삼투압 상태", ("고삼투압", "고삼투성", RX(A("hhs")), RX(A("hhns")), "hyperosmolar"))),
    items=(
        CItem("glu200", "혈당 ≥200 mg/dL", num=_with(_GLU, ">=", 200)),
        CItem("glu600", "혈당 ≥600 mg/dL (HHS)", num=_with(_GLU, ">=", 600)),
        CItem("diabetes", "당뇨병 병력", patterns=(r"당뇨", r"diabet", r"인슐린\s?(주사|치료|투여|사용)", r"insulin")),
        CItem("ketones", "케톤: β-OHB ≥3.0 mmol/L 또는 요케톤 ≥2+", derive=_ketones),
        CItem("ph", "pH <7.30", num=_with(_PH, "<", 7.30)),
        CItem("hco3", "HCO3 <18 mmol/L", num=_with(_HCO3, "<", 18)),
        CItem("osm", "유효 삼투압 >300 또는 총 삼투압 >320 mOsm/kg (HHS)", derive=_osm_state),
    ),
    logic=lambda s, d: _dka_hhs(s, d),
    rule="DKA: 혈당 ≥200 mg/dL(또는 당뇨병 병력) + 케톤(β-OHB ≥3.0 mmol/L 또는 요케톤 ≥2+) + pH <7.3 또는 HCO3 <18. "
         "HHS: 혈당 ≥600, 유효 삼투압 >300 또는 총 삼투압 >320, 유의한 케톤 없음(β-OHB <3.0), pH ≥7.3·HCO3 ≥15.",
    subtype_rule="산증+케톤이면 당뇨병성 케톤산증, 산증·케톤 없이 혈당 ≥600·고삼투압이면 고혈당성 고삼투압 상태; 둘 다면 혼합.",
    citation=C_DKA, verification="primary",
    note="DKA components (glucose >=200 mg/dL or known diabetes; beta-OHB >=3.0 mmol/L or urine ketones >=2+; pH <7.30 "
         "or HCO3 <18) checked in the full text (PMC11272983). 2026-09-29: HHS criteria read in Fig. 2B (figure image): "
         "glucose >= 600 mg/dL; calculated effective osmolality > 300 mOsm/kg (2Na + glucose) or total > 320; beta-OHB "
         "< 3.0 mmol/L or urine ketones < 2+; pH >= 7.3 and bicarbonate >= 15 mmol/L. The former stricter 'HCO3 >= 18' "
         "for HHS was relaxed to the source value.",
)


# --------------------------------------------------------------------------------------------
# 9. Pleural effusion — Light's criteria
# --------------------------------------------------------------------------------------------

_PL = r"(흉수|흉막\s?액|늑막\s?액|흉막\s?천자|pleural(\s?fluid)?)"
_SE = r"(혈청|serum|혈액)"


def _ratio(doc: Doc, analyte: str, ratio_label: str, cut: float) -> tuple[str, str]:
    r = [x for x in _grab(doc, ratio_label + r"[^\n\d]{0,10}(\d(?:\.\d+)?)") if x <= 10]
    if r:
        return (MET if r[-1] > cut else NOT_MET), f"비 {r[-1]:g}"
    pl = _grab(doc, _PL + r"[^\n\d]{0,15}" + analyte + r"[^\n\d]{0,8}(\d+(?:\.\d+)?)")
    se = _grab(doc, _SE + r"[^\n\d]{0,10}" + analyte + r"[^\n\d]{0,8}(\d+(?:\.\d+)?)")
    if pl and se and se[-1] > 0:
        v = pl[-1] / se[-1]
        return (MET if v > cut else NOT_MET), f"{pl[-1]:g}/{se[-1]:g}={v:.2f}"
    return UNKNOWN, ""


def _grab(doc: Doc, pattern: str) -> list[float]:
    out = []
    for m in re.finditer(pattern, doc.text):
        if not doc._skipped(m.start()):
            out.append(float(m.groups()[-1]))
    return out


def _pl_ldh_uln(doc: Doc) -> tuple[str, str]:
    pl = _grab(doc, _PL + r"[^\n\d]{0,15}(ldh|젖산\s?탈수소)[^\n\d]{0,8}(\d+(?:\.\d+)?)")
    uln = _grab(doc, r"(ldh|젖산\s?탈수소)[^\n\d]{0,25}(정상\s?상한|uln|upper limit)[^\n\d]{0,8}(\d+(?:\.\d+)?)")
    if pl and uln and uln[-1] > 0:
        return (MET if pl[-1] > uln[-1] * 2 / 3 else NOT_MET), f"{pl[-1]:g} vs 2/3×{uln[-1]:g}"
    return UNKNOWN, ""


def _light(s: dict, doc: Doc) -> Outcome:
    keys = ("protein_ratio", "ldh_ratio", "ldh_uln")
    if any(_m(s, k) for k in keys):
        return Outcome("삼출액 (Light 기준 1개 이상 충족)", "삼출성 흉수")
    if all(_n(s, k) for k in keys):
        return Outcome("누출액 (Light 기준 3개 모두 불충족)", "누출성 흉수")
    return Outcome("판정 불가: 흉수·혈청 단백/LDH가 필요 (하나라도 충족하면 삼출액)", "")


LIGHT = Criteria(
    id="light_1972", short="Light 기준", name_ko="흉수 삼출액/누출액 구분(Light 기준)", name_en="Light's criteria",
    kind="diagnostic",
    synonyms=("흉수", "흉막 삼출", "흉막삼출", "늑막 삼출", "늑막삼출", "늑막염", "흉막염", "pleural effusion", "삼출성 흉수",
              "누출성 흉수"),
    subtypes=(Subtype("삼출성 흉수", ("삼출성", "삼출액", "exudat")), Subtype("누출성 흉수", ("누출성", "누출액", "transudat"))),
    items=(
        CItem("protein_ratio", "흉수/혈청 단백 비 >0.5",
              derive=lambda d: _ratio(d, r"(총\s?)?(단백|protein)", r"(단백|protein)\s?(비|ratio)", 0.5)),
        CItem("ldh_ratio", "흉수/혈청 LDH 비 >0.6",
              derive=lambda d: _ratio(d, r"(ldh|젖산\s?탈수소\S*)", r"(ldh|젖산\s?탈수소\S*)\s?(비|ratio)", 0.6)),
        CItem("ldh_uln", "흉수 LDH > 혈청 LDH 정상 상한의 2/3", derive=_pl_ldh_uln),
    ),
    logic=lambda s, d: _light(s, d),
    rule="흉수/혈청 단백 >0.5, 흉수/혈청 LDH >0.6, 흉수 LDH > 혈청 정상 상한의 2/3 중 하나라도 있으면 삼출액.",
    subtype_rule="하나라도 충족하면 삼출성, 셋 다 불충족이면 누출성 흉수.",
    citation=C_LIGHT, verification="unverified",
    note="Bibliographic data verified (PubMed has no abstract for this 1972 paper; original not accessible). The three "
         "cut-offs are the standard textbook statement of Light's criteria (reviewer knowledge). Diuretic-treated heart "
         "failure can be misclassified as exudate (not handled).",
)


# --------------------------------------------------------------------------------------------
# 10. Sepsis-3
# --------------------------------------------------------------------------------------------

_INFECTION = (r"감염", r"폐렴", r"신우신염", r"요로\s?감염", r"봉와직염", r"담관염", r"복막염", r"농양", r"균혈증", r"수막염",
              r"infection", r"pneumonia", r"pyelonephritis", r"cellulitis", r"cholangitis", r"bacteremia")


def _sbp(doc: Doc) -> tuple[str, str]:
    v = _grab(doc, r"(혈압|수축기|(?<![a-z])s?bp(?![a-z]))[^\n\d]{0,8}(\d{2,3})\s?/\s?\d{2,3}")
    v += _grab(doc, r"수축기\s?혈압[^\n\d]{0,6}(\d{2,3})")
    if v:
        return (MET if v[-1] <= 100 else NOT_MET), f"수축기 {v[-1]:g}"
    return UNKNOWN, ""


def _sepsis(s: dict, doc: Doc) -> Outcome:
    q = sum(_m(s, k) for k in ("rr", "sbp", "mentation"))
    if _m(s, "vaso") and _m(s, "lactate") and _m(s, "infection"):
        fluid = "" if _m(s, "fluids") else " (적절한 수액 후에도 지속인지 확인)"
        return Outcome(f"패혈성 쇼크 기준 충족: 승압제 필요 + 젖산 >2 mmol/L{fluid}",
                       "패혈성 쇼크" if _m(s, "fluids") else "")
    if _m(s, "infection") and _m(s, "sofa"):
        return Outcome("패혈증 기준 충족 (감염 + SOFA ≥2점 급상승)", "패혈증")
    tag = f"qSOFA {q}/3" + (" → 패혈증 의심, SOFA(장기부전) 확인" if q >= 2 else "")
    return Outcome(f"패혈증 판정 불가: 감염 {_STATE_KO[s['infection'][0]]}, SOFA 변화 미확인 ({tag})", "")


SEPSIS = Criteria(
    id="sepsis3_2016", short="Sepsis-3", name_ko="패혈증·패혈성 쇼크 정의(Sepsis-3)", name_en="Sepsis-3 definitions",
    kind="definition", synonyms=("패혈증", "패혈성 쇼크", "sepsis", "septic shock"),
    exclude=(r"신생아\s?패혈", r"neonatal sepsis"),
    subtypes=(Subtype("패혈성 쇼크", ("패혈성 쇼크", "septic shock")), Subtype("패혈증", ("패혈증", "sepsis"))),
    items=(
        CItem("infection", "감염(의심 또는 확인)", patterns=_INFECTION),
        CItem("sofa", "SOFA 점수 ≥2점 급상승", num=Num((A("sofa") + r"[^\n\d]{0,10}(점수)?[^\n\d]{0,6}",), ">=", 2, lo=0, hi=24)),
        CItem("rr", "qSOFA: 호흡수 ≥22회/분", num=Num((r"호흡\s?(수|횟수|빈도)", A("rr"), r"respiratory rate"), ">=", 22,
                                                   lo=4, hi=80, window=6)),
        CItem("sbp", "qSOFA: 수축기 혈압 ≤100 mmHg", derive=_sbp),
        CItem("mentation", "qSOFA: 의식 변화",
              patterns=(r"의식\s?(저하|변화|혼탁|이 흐)", r"혼돈", r"착란", r"기면", r"altered mental", r"confus")),
        CItem("vaso", "쇼크: 평균 동맥압 ≥65 유지에 승압제 필요",
              patterns=(r"승압제", r"노르\s?에피네프린", r"norepinephrine", r"vasopressor", r"바소프레신", r"vasopressin",
                        r"도파민\s?(투여|주입|시작)")),
        CItem("lactate", "쇼크: 젖산 >2 mmol/L",
              num=Num((r"젖산", r"lactate", r"lactic acid"), ">", 2.0, units=(("mmol/l", 1.0), ("mg/dl", 1 / 9.0)),
                      lo=0.1, hi=30)),
        CItem("fluids", "쇼크: 적절한 수액 소생 후에도 저혈압 지속",
              patterns=(r"수액[^\n]{0,15}(후에도|에도|불구|반응\s?없|반응하지)", r"despite (adequate )?fluid",
                        r"fluid.refractory")),
    ),
    logic=lambda s, d: _sepsis(s, d),
    rule="패혈증: 감염 + SOFA ≥2점 급상승. 패혈성 쇼크: 적절한 수액에도 MAP ≥65 유지에 승압제 필요 + 젖산 >2 mmol/L. "
         "qSOFA(호흡수 ≥22, 의식 변화, 수축기 ≤100) ≥2는 선별용.",
    subtype_rule="승압제 + 젖산 >2 + 수액 후 지속이면 패혈성 쇼크, 아니면 패혈증(감염+SOFA ≥2).",
    citation=C_SEPSIS, verification="primary",
    note="Sepsis (SOFA >=2), septic shock (vasopressor for MAP >=65 + lactate >2 mmol/L despite adequate fluids) and "
         "qSOFA items checked in the full text (PMC4968574, 2026-09-27). SOFA itself is not computed from organ values.",
)


# --------------------------------------------------------------------------------------------
# 11. Acute rheumatic fever — 2015 revised Jones (low-risk population)
# --------------------------------------------------------------------------------------------

def _jones(s: dict, doc: Doc) -> Outcome:
    major = sum(_m(s, k) for k in ("carditis", "polyarthritis", "chorea", "marginatum", "nodules"))
    # Table 7: PR prolongation counts only when carditis is not a major criterion; polyarthralgia only without arthritis
    minor = (sum(_m(s, k) for k in ("fever", "inflam")) + (1 if _m(s, "pr") and not _m(s, "carditis") else 0)
             + (1 if _m(s, "arthralgia") and not _m(s, "polyarthritis") else 0))
    tag = f"주 {major}, 부 {minor}"
    recurrent = _m(s, "prior_arf")
    ok = major >= 2 or (major >= 1 and minor >= 2) or (recurrent and minor >= 3)
    if ok and _m(s, "gas"):
        return Outcome(f"{'재발 ' if recurrent else ''}급성 류마티스열 기준 충족 ({tag}, 선행 A군 연쇄상구균 감염 증거)",
                       "급성 류마티스열")
    if ok:
        return Outcome(f"주·부 기준 충족({tag})이나 선행 A군 연쇄상구균 감염 증거(ASO 등) 미확인", "")
    return Outcome(f"기준 미충족 또는 미확인 ({tag}; 초발: 주 2 또는 주 1+부 2 + 선행 감염 증거)", "")


JONES = Criteria(
    id="jones_2015", short="2015 Jones", name_ko="급성 류마티스열 Jones 기준(2015 개정)",
    name_en="2015 revised Jones criteria", kind="diagnostic",
    synonyms=("류마티스열", "류마티스 열", "류머티스열", "rheumatic fever"),
    items=(
        CItem("gas", "필수: 선행 A군 연쇄상구균 감염 증거(ASO 상승, 인두 배양/신속 항원 양성 등)",
              patterns=(A("aso") + r"[^\n]{0,8}(상승|증가|높|양성|elevated)", r"항\s?스트렙토리신[^\n]{0,8}(상승|증가|높)",
                        r"anti-?dnase", r"(a군|group a)[^\n]{0,15}(양성|검출|배양|positive)", r"성홍열",
                        r"(신속|rapid)[^\n]{0,10}(항원|strep)[^\n]{0,6}(양성|positive)", r"인두\s?배양[^\n]{0,10}(양성|연쇄)")),
        CItem("carditis", "주: 심장염(임상 또는 심초음파상 판막염)",
              patterns=(r"심장염", r"판막염", r"carditis", r"(새로운|새로 생긴)\s?심잡음", r"승모판\s?역류")),
        CItem("polyarthritis", "주: 다발성 관절염", patterns=(r"다발성?\s?관절염", r"(여러|이동성)[^\n]{0,6}관절[^\n]{0,8}(붓|부종|관절염)",
                                                   r"polyarthritis", r"migratory arthritis")),
        CItem("chorea", "주: 무도병", patterns=(r"무도병", r"시든햄", r"chorea")),
        CItem("marginatum", "주: 변연 홍반", patterns=(r"변연\s?(성\s?)?홍반", r"erythema marginatum")),
        CItem("nodules", "주: 피하 결절", patterns=(r"피하\s?결절", r"subcutaneous nodule")),
        CItem("arthralgia", "부: 다발성 관절통", patterns=(r"(여러|다발성?)[^\n]{0,6}관절\s?(통|이 아프)", r"polyarthralgia")),
        CItem("fever", "부: 발열 ≥38.5°C", num=_temp(">=", 38.5)),
        CItem("inflam", "부: ESR ≥60 mm/h 또는 CRP ≥3.0 mg/dL",
              derive=lambda d: _either(d, Num(_ESR, ">=", 60, lo=0, hi=200),
                                       Num(_CRP, ">=", 30, units=_CRP_UNITS, unit_required=True, lo=0, hi=1000))),
        CItem("pr", "부: 나이 보정 PR 간격 연장(심장염을 주 기준으로 쓰지 않을 때)",
              patterns=(r"pr\s?(간격\s?)?(연장|길어)", r"1도\s?방실\s?(차단|블록)", r"first.degree (av|atrioventricular) block")),
        CItem("prior_arf", "과거 류마티스열/류마티스 심장병(재발 판정용)",
              patterns=(r"류마티스\s?열[^\n]{0,6}(과거력|병력|앓)", r"류마티스\s?(성\s?)?심장\s?(병|질환)", r"rheumatic heart")),
    ),
    logic=lambda s, d: _jones(s, d),
    rule="저위험 인구 기준. 선행 A군 연쇄상구균 감염 증거 + (주 2개 또는 주 1 + 부 2). 재발은 부 3개로도 가능. "
         "중·고위험 인구는 단관절염·다발성 관절통도 주 기준, 발열 ≥38.0, ESR ≥30.",
    citation=C_JONES, verification="primary",
    note="2026-09-29: full text read (Circulation statement PDF hosted by heartuniversity.org), Table 7: evidence of "
         "preceding GAS infection; initial ARF = 2 major or 1 major + 2 minor, recurrent = 2 major, 1 major + 2 minor "
         "or 3 minor; low-risk major = carditis (clinical and/or subclinical), polyarthritis only, chorea, erythema "
         "marginatum, subcutaneous nodules; low-risk minor = polyarthralgia, fever >= 38.5 C, ESR >= 60 mm in the "
         "first hour and/or CRP >= 3.0 mg/dL, prolonged PR interval (unless carditis is a major criterion). Only the "
         "low-risk rule is implemented (Korea is low incidence).",
)


def _either(doc: Doc, a: Num, b: Num) -> tuple[str, str]:
    va, vb = doc.values(a), doc.values(b)
    if (va and a.check(va[-1])) or (vb and b.check(vb[-1])):
        return MET, f"{va[-1]:g}" if va and a.check(va[-1]) else f"{vb[-1]:g}"
    if va and vb:
        return NOT_MET, ""
    return UNKNOWN, ""


# --------------------------------------------------------------------------------------------
# 12. Multiple sclerosis — 2017 McDonald
# --------------------------------------------------------------------------------------------

_LES = r"[^\n.]{0,15}(병변|고신호|lesion|plaque|판)"
_AREAS = {
    "뇌실 주위": (r"(측)?뇌실\s?주위" + _LES, r"periventricular" + _LES),
    "피질/피질 인접": (r"(피질|피질\s?하|피질\s?인접|피질\s?근처|피질\s?직하)" + _LES, r"(juxta)?cortical" + _LES),
    "천막하": (r"(천막\s?하|뇌간|뇌줄기|소뇌|교뇌|연수)" + _LES, r"(infratentorial|brainstem|cerebellar)" + _LES),
    "척수": (r"척수" + _LES, r"spinal cord" + _LES),
}


def _dis(doc: Doc) -> tuple[str, str]:
    hit = [a for a, pats in _AREAS.items() if doc.state(pats)[0] == MET]
    if len(hit) >= 2:
        return MET, ", ".join(hit)
    return UNKNOWN, ", ".join(hit)


def _ms(s: dict, doc: Doc) -> Outcome:
    if _m(s, "attacks2") and _m(s, "dis"):
        return Outcome("2회 이상 발작 + 공간 파종 → MS 기준 충족 (다른 설명 배제 전제)", "다발성 경화증")
    if _m(s, "dis") and (_m(s, "dit") or _m(s, "ocb")):
        why = "시간 파종" if _m(s, "dit") else "뇌척수액 올리고클론띠(시간 파종 대체)"
        return Outcome(f"공간 파종 + {why} → MS 기준 충족 (다른 설명 배제 전제)", "다발성 경화증")
    return Outcome("MS 기준 미확인: 공간 파종(4개 부위 중 ≥2) + 시간 파종 또는 올리고클론띠 필요", "")


MS = Criteria(
    id="mcdonald_2017", short="2017 McDonald", name_ko="다발성 경화증 McDonald 기준(2017)",
    name_en="2017 McDonald criteria", kind="diagnostic",
    synonyms=("다발성 경화증", "다발경화증", "multiple sclerosis", RX(A("ms")), "임상적 독립 증후군", "clinically isolated syndrome"),
    related=(r"시신경염", r"optic neuritis", r"탈수초", r"demyelinat", r"횡단\s?척수염", r"transverse myelitis"),
    exclude=(r"승모판", r"mitral", r"시신경\s?척수염", r"neuromyelitis", r"nmosd", r"결절성\s?경화", r"tuberous", r"계통\s?경화", r"측삭"),
    items=(
        CItem("attacks2", "2회 이상의 임상 발작(재발)",
              patterns=(r"(두|2|세|3)\s?(번|차례|회)[^\n]{0,4}(째\s?)?(발작|재발|삽화|에피소드|증상)", r"재발(?!\s?없)",
                        r"(이전|예전|전에)[^\n]{0,10}(비슷한|같은)\s?(증상|발작)", r"relaps")),
        CItem("dis", "공간 파종: 뇌실 주위·피질/인접·천막하·척수 중 ≥2곳 T2 병변", derive=_dis),
        CItem("dit", "시간 파종: 조영 증강·비증강 병변 동시 존재 또는 추적 MRI 새 병변",
              patterns=(r"(조영\s?증강|gd\+?|가돌리늄)[^\n]{0,25}(비증강|증강되지 않는|오래된)", r"enhancing and non-?enhancing",
                        r"(추적|follow-?up)[^\n]{0,15}(새로운|새|new)\s?(t2\s?)?(병변|lesion)")),
        CItem("ocb", "뇌척수액 특이 올리고클론띠", patterns=(r"올리고\s?클론", r"oligoclonal", A("ocb"))),
    ),
    logic=lambda s, d: _ms(s, d),
    rule="전형적 탈수초 발작에서 공간 파종(4개 부위 중 ≥2곳) + 시간 파종(동시 증강·비증강 병변, 새 병변, 또는 뇌척수액 "
         "올리고클론띠로 대체). 증상 병변도 인정. 다른 설명(NMOSD 등)이 없어야 함.",
    citation=C_MS, verification="secondary",
    note="From the PubMed abstract: OCB can substitute for dissemination in time, symptomatic and cortical lesions count. "
         "The four DIS areas and the DIT definitions are from reviewer knowledge (Lancet Neurol full text not open).",
)


# --------------------------------------------------------------------------------------------
# 13. Migraine vs tension-type headache — ICHD-3 (content only)
# --------------------------------------------------------------------------------------------

def _headache(s: dict, doc: Doc) -> Outcome:
    mig_feat = sum(_m(s, k) for k in ("unilateral", "pulsating", "mod_severe", "activity"))
    assoc = _m(s, "nausea") or (_m(s, "photo") and _m(s, "phono"))
    tth_feat = sum(_m(s, k) for k in ("bilateral", "pressing", "mild")) + (1 if _n(s, "activity") else 0)
    tth_assoc = _n(s, "nausea") and not (_m(s, "photo") and _m(s, "phono"))
    migraine = mig_feat >= 2 and assoc
    tth = tth_feat >= 2 and tth_assoc
    if _m(s, "red_flag"):  # ICHD-3 criterion "not better accounted for": secondary headache first
        return Outcome(f"판정 보류: 이차성 두통 위험 신호({s['red_flag'][1]}) → 지주막하출혈·수막염 등 먼저 배제", "")
    recurrent = _m(s, "attacks")
    if migraine and not tth:
        aura = " (전조 있음 → 조짐편두통 고려)" if _m(s, "aura") else ""
        if not recurrent:
            return Outcome(f"편두통 양상(통증 특징 {mig_feat}/4 + 동반 증상)이나 반복 발작(≥5회) 미확인 → 개연 편두통까지만", "")
        return Outcome(f"편두통 특징 충족 (통증 특징 {mig_feat}/4 + 동반 증상, 반복 발작){aura}", "편두통")
    if tth and not migraine:
        if not recurrent:
            return Outcome(f"긴장형 양상(특징 {tth_feat}/4)이나 반복 발작(≥10회) 미확인", "")
        return Outcome(f"긴장형 두통 특징 충족 (특징 {tth_feat}/4, 구역 없음, 빛·소리 과민 중 1개 이하, 반복)", "긴장형 두통")
    return Outcome(f"구분 불가 (편두통 특징 {mig_feat}/4, 긴장형 특징 {tth_feat}/4; 동반 증상 확인 필요)", "")


HEADACHE = Criteria(
    id="ichd3_migraine_tth", short="ICHD-3 편두통/긴장형", name_ko="편두통 vs 긴장형 두통(ICHD-3)",
    name_en="ICHD-3 migraine without aura vs tension-type headache", kind="diagnostic",
    synonyms=("편두통", "migraine", "긴장형 두통", "긴장성 두통", "긴장형두통", "긴장성두통", "tension-type", "tension type",
              "tension headache", "일차성 두통", "primary headache"),
    exclude=(r"전정\s?편두통", r"vestibular migraine", r"편마비\s?편두통", r"hemiplegic"),
    focus=(r"두통", r"머리", r"관자", r"이마", r"뒷목", r"편두", r"headache", r"(?<![a-z])head"),
    subtypes=(Subtype("편두통", ("편두통", "migraine")), Subtype("긴장형 두통", ("긴장형", "긴장성", "tension"))),
    items=(
        CItem("attacks", "편두통 ≥5회 / 긴장형 ≥10회 반복",
              patterns=(r"반복", r"자주", r"(거의\s?)?매일", r"날마다", r"(한\s?달|매달|매주|일주일|7일)[^\n]{0,6}(\d+|여러|몇)\s?(번|회)",
                        r"(\d{1,2})\s?(번|회)\s?이상", r"recurrent", r"episodes", r"daily")),
        CItem("unilateral", "편측", patterns=(r"한쪽", r"편측", r"(왼쪽|오른쪽|좌측|우측)\s?(머리|관자|이마|눈)", r"unilateral",
                                            r"one side")),
        CItem("pulsating", "박동성(욱신·지끈)", patterns=(r"욱신", r"지끈", r"박동", r"쿵쿵", r"맥박\s?뛰", r"pulsat", r"throbb")),
        CItem("mod_severe", "중등도 이상 강도(일상 방해)",
              patterns=(r"심한\s?(두통|통증)", r"극심", r"(일상|일|업무)[^\n]{0,6}(못|어려|지장)", r"누워\s?있어야",
                        r"(?<!\d)([6-9]|10)\s?/\s?10", r"moderate", r"severe")),
        CItem("activity", "일상 활동으로 악화",
              patterns=(r"(움직이면|움직여도|움직일\s?때|걸으면|걸어도|계단|활동|몸을 숙이면)[^\n]{0,10}(심해|악화|더 아프)",
                        r"(worse|aggravated) (with|by) (activity|movement)")),
        CItem("nausea", "구역·구토", patterns=(r"메스꺼", r"메스껍", r"속이?\s?울렁", r"오심", r"구역", r"구토", r"토했", r"토하", r"토한",
                                          r"토할", r"nausea", r"vomit"), whole_text=True),
        CItem("photo", "빛 과민", patterns=(r"빛[^\n]{0,6}(눈부|싫|괴롭|민감|예민|힘들)", r"눈부심", r"광\s?과민", r"photophob"), whole_text=True),
        CItem("phono", "소리 과민", patterns=(r"소리[^\n]{0,6}(싫|괴롭|민감|예민|거슬|힘들)", r"소음[^\n]{0,6}(싫|괴롭|민감|예민)",
                                         r"phonophob"), whole_text=True),
        CItem("bilateral", "양측", patterns=(r"양쪽", r"양측", r"머리\s?전체", r"띠를?\s?두른", r"bilateral", r"both sides")),
        CItem("pressing", "조이거나 누르는 양상", patterns=(r"조이", r"조여", r"누르는", r"압박", r"띠", r"묵직", r"pressing",
                                                   r"tight")),
        CItem("mild", "경도~중등도", patterns=(r"(약한|가벼운|심하지 않)", r"참을\s?만", r"(?<!\d)[1-4]\s?/\s?10", r"mild")),
        CItem("red_flag", "이차성 두통 위험 신호(벼락두통, 발열·목 강직, 의식 변화, 신경 결손, 유두 부종)",
              patterns=(r"벼락", r"갑자기[^\n]{0,10}(심한|극심|터지|망치)", r"(생애|평생|살면서)[^\n]{0,6}(최악|처음|가장 심한)",
                        r"thunderclap", r"worst headache", r"(목|경부|항부)[^\n]{0,3}(뻣뻣|강직)", r"neck stiff", r"nuchal rigid",
                        r"발열", r"열이\s?(나|있|났)", r"fever", r"의식\s?(저하|변화|혼탁)", r"혼돈", r"착란", r"편마비", r"마비",
                        r"구음\s?장애", r"유두\s?부종", r"papilledema"), whole_text=True),
        CItem("aura", "전조(시각 섬광·지그재그·암점 등)",
              patterns=(r"전조", r"섬광", r"번쩍", r"지그재그", r"암점", r"aura", r"scintillat")),
    ),
    logic=lambda s, d: _headache(s, d),
    rule="이차성 두통 위험 신호가 있으면 판정 보류. 편두통(무조짐): 4–72시간, 편측·박동성·중등도 이상·활동 시 악화 중 ≥2 + (구역/구토 또는 빛·소리 과민 모두), ≥5회. "
         "긴장형: 30분–7일, 양측·압박감·경도~중등도·활동에 악화 안 됨 중 ≥2 + 구역 없음 + 빛·소리 과민 중 1개 이하.",
    subtype_rule="위험 신호 없고 반복 발작일 때만: 편두통 조건만 충족하면 편두통, 긴장형 조건만 충족하면 긴장형 두통.",
    citation=C_ICHD, verification="primary",
    note="Criteria 1.1 and 2.1/2.2 read on ichd-3.org (2026-09-27). ICHD-3 is copyrighted by the IHS; only the content "
         "is implemented, in our own Korean wording. Attack duration (4-72 h / 30 min-7 d) is shown but not parsed.",
    copyright="ICHD-3 (c) International Headache Society: content only, Korean wording ours, no text copied.",
)


# --------------------------------------------------------------------------------------------
# 14. Bipolar I vs II — DSM-5-TR content (no text copied)
# --------------------------------------------------------------------------------------------

_ELEVATED = (r"(?<!경)조증", r"경조증", r"들뜬", r"들떠", r"기분이?\s?(너무\s?)?(좋|들뜨|고양)", r"잠을?\s?(거의\s?)?안\s?자",
             r"잠이\s?(거의\s?)?필요\s?없", r"(에너지|의욕|활력)이?\s?넘", r"과대", r"말이\s?많아", r"(hypo)?mani", r"elevated mood")


def _elev_days(doc: Doc) -> tuple[str, str]:
    d = doc.durations_days(_ELEVATED, window=30)
    d = [x for x in d if x < 400]
    if not d:
        return UNKNOWN, ""
    return MET, f"{_fmt(max(d))}일"


def _near(pats_a: tuple[str, ...], pats_b: tuple[str, ...], span: int = 30) -> tuple[str, ...]:
    out = []
    for a in pats_a:
        for b in pats_b:
            out += [rf"{a}[^\n]{{0,{span}}}{b}", rf"{b}[^\n]{{0,{span}}}{a}"]
    return tuple(out)


_HOSP = (r"입원", r"hospitali", r"admitted")
_PSYCH = (r"망상", r"환청", r"환각", r"정신병적", r"psychotic", r"delusion", r"hallucinat")
_IMPAIR = (r"(직장|회사)[^\n]{0,10}(잘리|해고|그만두|잃)", r"해고", r"체포", r"경찰", r"파산", r"큰\s?빚", r"카드\s?빚",
           r"(사회|직업)\s?(생활|기능)[^\n]{0,6}(불가|못|장애)", r"marked impairment", r"arrested", r"fired")


_DEPRESSED = (r"우울", r"(흥미|의욕)[^\n]{0,6}(잃|없어|상실)", r"depress")


def _mde(doc: Doc) -> tuple[str, str]:
    """Explicit major-depression label, or depressed mood lasting >= 14 days in the same clause."""
    st, ev = doc.state((r"주요\s?우울", r"우울\s?(삽화|장애)", r"우울증\s?(진단|치료|으로)", r"major depressi",
                        r"depressive episode"))
    if st == MET:
        return MET, ev
    d = doc.durations_days(_DEPRESSED)
    if d and max(d) >= 14:
        return MET, f"우울 {_fmt(max(d))}일"
    neg = doc.state(_DEPRESSED)[0]
    return (NOT_MET, "") if neg == NOT_MET or st == NOT_MET else (UNKNOWN, "")


def _bipolar(s: dict, doc: Doc) -> Outcome:
    days = float(s["elev_days"][1][:-1]) if _m(s, "elev_days") else None
    severe = _m(s, "hosp") or _m(s, "psychotic")
    mania_by_dur = days is not None and days >= 7 and (_m(s, "impair") or _m(s, "mania_label"))
    if _m(s, "elevated") and (severe or mania_by_dur):
        why = "입원" if _m(s, "hosp") else "정신병적 증상" if _m(s, "psychotic") else f"{_fmt(days)}일 + 뚜렷한 기능 손상"
        return Outcome(f"조증 삽화 근거({why}) → 양극성 I형", "양극성 I형 장애")
    no_mania = (days is not None and 4 <= days < 7 and not severe and not _m(s, "hosp") and not _m(s, "psychotic")
                and _n(s, "hosp"))
    if no_mania and _m(s, "mde"):
        return Outcome(f"경조증({_fmt(days)}일, 입원·정신병적 증상 없음) + 주요우울삽화 → 양극성 II형", "양극성 II형 장애")
    if no_mania:
        return Outcome(f"경조증 양상({_fmt(days)}일) — II형 판정에는 주요우울삽화 확인 필요", "")
    return Outcome("I/II형 구분 불가: 들뜬 시기의 기간, 입원 여부, 정신병적 증상, 기능 손상, 우울 삽화를 확인", "")


BIPOLAR = Criteria(
    id="bipolar_dsm5tr", short="DSM-5-TR 양극성 I/II", name_ko="양극성 I형 vs II형 구분", name_en="Bipolar I vs II (DSM-5-TR)",
    kind="diagnostic",
    synonyms=("양극성", "조울", "bipolar", "조증", "manic"),
    subtypes=(Subtype("양극성 I형 장애", (RX(r"(?<![a-z0-9ⅰⅱ])(i|1|ⅰ)\s?형"), RX(r"bipolar\s?(type\s?)?(i|1)(?![i0-9])"),
                                          RX(r"(?<![a-z0-9])type\s?(i|1)(?![i0-9])"))),
              Subtype("양극성 II형 장애", (RX(r"(?<![a-z0-9])(ii|2|ⅱ)\s?형"), RX(r"bipolar\s?(type\s?)?(ii|2)(?![i0-9])"),
                                           RX(r"(?<![a-z0-9])type\s?(ii|2)(?![i0-9])")))),
    items=(
        CItem("elevated", "들뜬(고양·과민) 기분 + 활력 증가 시기", patterns=_ELEVATED),
        CItem("elev_days", "들뜬 시기 지속 기간", derive=_elev_days),
        CItem("hosp", "들뜬 시기로 입원", patterns=_near(_ELEVATED, _HOSP)),
        CItem("psychotic", "들뜬 시기의 정신병적 증상", patterns=_near(_ELEVATED, _PSYCH)),
        CItem("impair", "뚜렷한 사회·직업 기능 손상", patterns=_IMPAIR),
        CItem("mania_label", "조증 삽화로 진단받은 적", patterns=(r"(?<!경)조증\s?(삽화|진단|으로)", r"(?<!hypo)manic episode")),
        CItem("mde", "주요우울삽화(2주 이상 우울·흥미 상실)", derive=lambda d: _mde(d)),
    ),
    logic=lambda s, d: _bipolar(s, d),
    rule="I형: 조증 삽화(≥7일 또는 입원 필요, 뚜렷한 기능 손상 또는 정신병적 증상) 1회 이상. II형: 경조증(≥4일, "
         "입원·정신병적 증상 없음) + 주요우울삽화, 조증은 한 번도 없음.",
    subtype_rule="입원·정신병적 증상·(≥7일+기능 손상)이면 I형; 4–6일·입원 없음 + 우울 삽화면 II형.",
    citation=C_DSM, verification="unverified",
    note="DSM-5-TR is cited as a reference only; the manual was not consulted. Content (manic episode >=1 week or any "
         "duration if hospitalised, marked impairment or psychotic features; hypomanic episode >=4 days without them; "
         "bipolar II = hypomania + major depression, never mania) is from reviewer knowledge.",
    copyright="DSM-5-TR (c) American Psychiatric Association: cited, content only, no text copied.",
)


# --------------------------------------------------------------------------------------------
# 15. Gout — 2015 ACR/EULAR
# --------------------------------------------------------------------------------------------

def _urate(doc: Doc) -> tuple[str, str]:
    v = doc.values(Num((r"요산(?!\s?(결정|염))", r"uric acid", r"(?<![a-z])urate(?! crystal)"), lo=0.5, hi=25,
                       units=(("mg/dl", 1.0), ("umol/l", 1 / 59.48), ("mmol/l", 16.81))))
    if not v:
        return UNKNOWN, ""
    return MET, f"{v[-1]:g}"


def _urate_pts(v: float) -> int:
    return -4 if v < 4 else 0 if v < 6 else 2 if v < 8 else 3 if v < 10 else 4


def _episode_feats(s: dict) -> int:
    return sum(_m(s, k) for k in ("erythema", "touch", "walking"))


def _gout(s: dict, doc: Doc) -> Outcome:
    if _m(s, "msu"):
        return Outcome("충분 기준 충족: 관절액/결절에서 요산 결정 확인", "통풍")
    if _n(s, "entry"):
        return Outcome("적용 불가: 말초 관절·점액낭의 통증/부종 삽화 없음", "")
    joint = 2 if _m(s, "mtp1") else 1 if _m(s, "ankle_midfoot") else 0
    feats = _episode_feats(s)
    typical = sum(_m(s, k) for k in ("rapid", "resolve14", "complete")) >= 2
    course = (2 if _m(s, "recurrent") else 1) if typical else 0
    urate = _urate_pts(float(s["urate"][1])) if _m(s, "urate") else 0
    score = (joint + feats + course + (4 if _m(s, "tophus") else 0) + urate + (-2 if _m(s, "msu_neg") else 0)
             + (4 if _m(s, "dect") else 0) + (4 if _m(s, "erosion") else 0))
    detail = f"{score}점 (관절 {joint}, 양상 {feats}, 경과 {course}, 요산 {urate})"
    if score >= 8 and _m(s, "entry"):
        return Outcome(f"분류 기준 충족: {detail} ≥8", "통풍", float(score))
    return Outcome(f"아직 미충족 또는 미확인: {detail}, 기준 ≥8 (관절액 요산 결정이면 바로 충족)", "", float(score))


GOUT = Criteria(
    id="gout_2015", short="2015 ACR/EULAR 통풍", name_ko="통풍 분류 기준", name_en="2015 ACR/EULAR gout criteria",
    kind="classification", synonyms=("통풍", "gout", "podagra"),
    exclude=(r"가성\s?통풍", r"pseudogout", r"cppd", r"피로인산"),
    focus=(r"관절", r"발가락", r"발목", r"무릎", r"발등", r"중족", r"손목", r"손가락", r"팔꿈치", r"점액낭", r"통풍", r"요산",
           r"joint", r"toe", r"ankle", r"knee", r"podagra"),
    items=(
        CItem("entry", "진입: 말초 관절·점액낭의 통증·부종·압통 삽화 ≥1회",
              patterns=(r"(관절|엄지\s?발가락|발가락|발목|무릎|발등|점액낭)[^\n]{0,8}(붓|부종|부어|통증|아프|아파|압통|열감)",
                        r"(joint|toe|ankle)[^\n]{0,10}(pain|swelling|swollen)")),
        CItem("msu", "충분: 증상 관절·점액낭·결절에서 요산 결정(MSU) 확인",
              patterns=(r"요산\s?(나트륨\s?)?결정(?![^\n]{0,4}(없|음성|관찰되지))", r"msu\s?(결정|crystal)", r"음성\s?복굴절",
                        r"바늘\s?모양[^\n]{0,6}결정", r"monosodium urate", r"negatively birefringent"), whole_text=True),
        CItem("mtp1", "제1 중족지 관절(엄지발가락) 침범 +2", patterns=(r"엄지\s?발가락", r"첫째\s?발가락", r"제\s?1\s?중족",
                                                            r"족무지", A("mtp1?"), r"big toe", r"great toe", r"podagra")),
        CItem("ankle_midfoot", "발목·중족부 침범 +1", patterns=(r"발목[^\n]{0,8}(붓|부종|통증|아프)", r"발등", r"중족부", r"ankle", r"midfoot")),
        CItem("erythema", "삽화 양상: 발적 +1", patterns=(r"붉", r"빨갛", r"빨개", r"발적", r"홍반", r"erythema", r"redness")),
        CItem("touch", "삽화 양상: 만지거나 스치기만 해도 극심한 통증 +1",
              patterns=(r"(스치|이불|닿기만|닿아도|살짝\s?만져도|만지지도)[^\n]{0,10}(아프|아파|통증|못)", r"can.t bear (the )?touch",
                        r"exquisite(ly)? tender")),
        CItem("walking", "삽화 양상: 걷기·관절 사용 매우 어려움 +1",
              patterns=(r"(걷기|걸을\s?수|걷지|걸음)[^\n]{0,6}(힘들|어렵|못|없)", r"절뚝", r"difficulty walking", r"unable to walk")),
        CItem("rapid", "전형적 삽화: 24시간 안에 최고 통증",
              patterns=(r"(밤사이|자다가|하룻밤|자고\s?일어나|몇\s?시간\s?(만에|사이|안에)|수\s?시간)",
                        r"within (hours|24 hours)", r"overnight")),
        CItem("resolve14", "전형적 삽화: 14일 이내 호전",
              derive=lambda d: _dur_state(d, (r"좋아|나아|나았|괜찮아|사라|호전|가라앉|resolv|subsid",), "<=", 14)),
        CItem("complete", "전형적 삽화: 삽화 사이 완전 회복",
              patterns=(r"(완전히|말끔히|씻은\s?듯)[^\n]{0,6}(좋아|나아|나았|괜찮|사라)", r"(사이|중간)에는?[^\n]{0,8}(괜찮|멀쩡|증상\s?없)",
                        r"complete(ly)? resol")),
        CItem("recurrent", "경과: 전형적 삽화 재발(≥2회)",
              patterns=(r"재발", r"반복", r"(전에도|예전에도|이전에도)[^\n]{0,12}(같은|비슷한)", r"(두|2|세|3|여러)\s?(번|차례)",
                        r"recurrent", r"previous (similar )?episode")),
        CItem("tophus", "통풍 결절(토푸스) +4", patterns=(r"통풍\s?결절", r"토푸스", r"tophus", r"tophi", r"결절종[^\n]{0,6}백색"), whole_text=True),
        CItem("urate", "혈청 요산(<4: −4, 6–8: +2, 8–10: +3, ≥10: +4 mg/dL)", derive=_urate, whole_text=True),
        CItem("msu_neg", "증상 관절 관절액 요산 결정 음성 −2",
              patterns=(r"(관절액|활액)[^\n]{0,20}결정\s?(이\s?|은\s?|은\s?전혀\s?)?(없|음성|관찰되지|보이지)",), whole_text=True),
        CItem("dect", "영상: 초음파 이중 윤곽 징후 또는 DECT 요산 침착 +4",
              patterns=(r"이중\s?윤곽", r"double.contour", r"(이중\s?에너지|dect|dual.energy)[^\n]{0,20}(요산|urate)"), whole_text=True),
        CItem("erosion", "영상: X선 통풍성 골미란 +4", patterns=(r"골\s?미란", r"펀치\s?아웃", r"punched.out", r"erosion"), whole_text=True),
    ),
    logic=lambda s, d: _gout(s, d),
    rule="진입: 말초 관절·점액낭 증상 삽화. 충분: 요산 결정 확인. 점수: 관절 부위(발목·중족부1, 엄지발가락2), 양상(발적·"
         "압통·보행곤란 각1), 경과(전형 삽화1, 재발2), 결절4, 혈청 요산(−4~+4), 관절액 결정 음성 −2, 영상 각4 → ≥8점.",
    citation=C_GOUT, verification="primary",
    note="Entry, sufficient criterion, all domain points, the typical-episode definition (>=2 of: max pain <24 h, "
         "resolution <=14 d, complete resolution between episodes) and >=8 threshold checked in the full text "
         "(PMC4566153, open access). CPPD (pseudogout) is excluded by name.",
)


# --------------------------------------------------------------------------------------------
# Registry and API
# --------------------------------------------------------------------------------------------

CRITERIA: tuple[Criteria, ...] = (SLE, RA, TAK, GCA, KD, IE, AKI, DKA, LIGHT, SEPSIS, JONES, MS, HEADACHE, BIPOLAR, GOUT)
CRITERIA_BY_ID: dict[str, Criteria] = {c.id: c for c in CRITERIA}


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").lower())


def _match_len(name: str, syn: str) -> int:
    """Length of the match of one synonym in a diagnosis name (0 = no match). Plain synonyms match as substrings
    ignoring spaces and case; RX() synonyms are regexes on the lowercase name."""
    low = (name or "").lower()
    if syn.startswith("re:"):
        m = re.search(syn[3:], low)
        return len(m.group(0)) if m else 0
    n = _norm(syn)
    return len(n) if n and n in _norm(low) else 0


def name_matches(name: str, synonyms: tuple[str, ...]) -> bool:
    return any(_match_len(name, s) for s in synonyms)


def which_subtype(c: Criteria, name: str) -> Subtype | None:
    """The subtype of `c` that `name` names (most specific synonym wins; a tie = ambiguous = None)."""
    best: list[tuple[int, Subtype]] = []
    for st in c.subtypes:
        n = max((_match_len(name, s) for s in st.synonyms), default=0)
        if n:
            best.append((n, st))
    if not best:
        return None
    best.sort(key=lambda x: -x[0])
    if len(best) > 1 and best[0][0] == best[1][0]:
        return None
    return best[0][1]


def _excluded(c: Criteria, name: str) -> bool:
    low = (name or "").lower()
    return any(re.search(p, low) for p in c.exclude)


def criteria_for(dx_name: str, related: bool = True) -> list[Criteria]:
    """Criteria sets for a diagnosis name (Korean/English, ignores spaces and case). Direct name/synonym/subtype matches
    come first; with `related`, sets whose cross-check patterns match (e.g. an arterial stenosis → Takayasu) follow."""
    if not (dx_name or "").strip():
        return []
    direct, rel = [], []
    for c in CRITERIA:
        if _excluded(c, dx_name):
            continue
        # subtype names count, their short synonyms ("삼출성", "2형") do not: they are only used to tell subtypes apart
        names = c.synonyms + tuple(st.name for st in c.subtypes)
        if name_matches(dx_name, names):
            direct.append(c)
        elif related and any(re.search(p, dx_name.lower()) for p in c.related):
            rel.append(c)
    return direct + rel


def evaluate(criteria_id: str, findings_text: str) -> CriteriaResult:
    """Check one criteria set against the case findings (free text). Conservative: see module docstring."""
    c = CRITERIA_BY_ID[criteria_id]
    doc = Doc(findings_text)
    focused = doc
    if c.focus and not _about(findings_text, c.focus):
        focused = Doc(_focus_text(findings_text, c.focus))
    states = {it.key: item_state(it, doc if it.whole_text else focused) for it in c.items}
    out = c.logic(states, doc)

    def label(it: CItem) -> str:
        ev = states[it.key][1]
        return it.text.split("(")[0].strip() + (f"[{ev}]" if ev and len(ev) <= 24 else "")

    return CriteriaResult(
        c.id, out.band, out.decision, out.score, states,
        met=tuple(label(it) for it in c.items if states[it.key][0] == MET),
        not_met=tuple(label(it) for it in c.items if states[it.key][0] == NOT_MET),
        unknown=tuple(it.text.split("(")[0].strip() for it in c.items if states[it.key][0] == UNKNOWN),
    )


def _about(text: str, focus: tuple[str, ...]) -> bool:
    """The first non-empty line (the initial info / chief complaint) is about the focus: then every answer of the case
    is read ("욱신거려요" answers a question about the headache), otherwise only sentences naming the focus."""
    first = next((ln for ln in (text or "").lower().split("\n") if ln.strip()), "")
    return any(re.search(p, first) for p in focus)


def _focus_text(text: str, focus: tuple[str, ...]) -> str:
    """Only the sentences that mention the focus (e.g. the head for headache features), one per line; a findings-
    ledger line keeps its "- 음성:" / "- 결과없음:" prefix on every kept sentence."""
    out = []
    for line in (text or "").split("\n"):
        m = re.match(r"\s*-\s*(양성|음성|결과없음)\s*:", line)
        prefix = m.group(0).strip() + " " if m else ""
        body = line[m.end():] if m else line
        for sent in re.split(r"(?<=[.!?])\s+|;\s*", body):
            if any(re.search(p, sent.lower()) for p in focus):
                out.append(prefix + sent.strip())
    return "\n".join(out)


def decision_matches(c: Criteria, decision: str, name: str) -> bool:
    """True if `name` names the diagnosis/subtype `decision` of criteria set `c`."""
    if not decision:
        return False
    if any(st.name == decision for st in c.subtypes):
        st = which_subtype(c, name)
        return st is not None and st.name == decision
    return _norm(decision) in _norm(name) or name_matches(name, c.synonyms)


def conflicting_subtype(c: Criteria, decision: str, name: str) -> str:
    """Another subtype of `c` that `name` claims although the findings selected `decision` ("" if none)."""
    if not decision or not any(st.name == decision for st in c.subtypes):
        return ""
    st = which_subtype(c, name)
    return st.name if st is not None and st.name != decision else ""


MAX_RENDER = 500


def render_for_review(dx_name: str, findings_text: str, max_sets: int = 2) -> str:
    """Short Korean summary for the pre-diagnosis reviewer ("" when no criteria set matches). At most `max_sets` sets
    and MAX_RENDER characters; a second set is added only if it fits."""
    out = ""
    for c in criteria_for(dx_name)[:max_sets]:
        r = evaluate(c.id, findings_text)
        lines = [f"[진단 기준 대조: {c.short} ({c.cite})] 판정: {r.band}"]
        if r.met:
            lines.append("충족: " + ", ".join(r.met))
        if r.not_met:
            lines.append("불충족: " + ", ".join(r.not_met))
        if r.unknown:
            lines.append("미확인: " + ", ".join(r.unknown[:5]) + (" 등" if len(r.unknown) > 5 else ""))
        lines.append("규칙: " + (c.subtype_rule or c.rule))
        block = "\n".join(lines)
        if not out:
            out = block if len(block) <= MAX_RENDER else block[:MAX_RENDER - 1].rstrip() + "…"
        elif len(out) + 1 + len(block) <= MAX_RENDER:
            out += "\n" + block
    return out
