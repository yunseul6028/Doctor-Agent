"""Can't-miss rule-out gate. Content is owned by clinical-strategist.

Pure functions over a CaseState (no LLM, no I/O). For each can't-miss diagnosis:
- RULE_OUT[name].rule_out / .confirm read the environment's responses (negation-aware keyword and number
  patterns) and return the evidence spans that rule the danger out or confirm it;
- RULE_OUT[name].steps list the next rule-out actions, least invasive adequate option first.

API
- status(dx_name, state) -> (status, evidence)             status: "ruled_out" | "confirmed" | "unresolved"
- unresolved_dangers(state, exclude=None) -> list[dict]     chief-complaint + DDx-ledger(위험) dangers, by urgency
- next_rule_out_action(danger, state) -> (ActionType, content, reason) | None
- gate(state, proposed_dx, remaining_turns, max_gate_turns=3, gate_turns_used=0) -> dict

Critical results: a critical imaging / ECG finding the policy's result interpreter read as present
(state.result_criticals, see CRITICAL_RESULT_DANGER) confirms the matching danger and puts it on the checked list;
a precursor finding (CRITICAL_RESULT_REQUIRES, e.g. a pneumothorax without tension signs) only puts it on the list.

Evidence is read only from the environment's text (initial info + responses), never from the doctor's own questions
("객혈이 있나요?" is not a hemoptysis finding). A result counts for a rule-out only when it is read as normal; an
unavailable or unreadable result marks the step as done (so it is not repeated) but rules nothing out.

Verification (2026-09-27): citations reused from protocols.py / clinical_rules.py are already PubMed-verified. New
citations below were checked against PubMed E-utilities on 2026-09-27 (esummary); what was read in each abstract is
in RuleOut.note. RuleOut.verification uses the protocols.py scale: "primary" = the rule-out statement was read in
the source text (abstract or full text); "secondary" = confirmed via secondary summaries; "unverified" = reviewer
knowledge / our operationalisation. Numeric cut-offs marked "ours" are conservative operationalisations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from doctor_agent.agent.text import dx_keys, same_dx
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge.clinical_rules import (
    C_ADD_RS,
    C_PERC,
    C_WELLS_PE,
    OTHER,
    POS,
    UNC,
    Citation,
    ReadText,
    age_years,
    contains_affirmed,
    duration_level,
    keyword_statuses,
)
from doctor_agent.safety.protocols import (
    G_AAA,
    G_AMI,
    G_ANAPHYLAXIS,
    G_CHEST_PAIN,
    G_AORTA,
    G_ECTOPIC,
    G_EARLY_PREGNANCY,
    G_HEADACHE,
    G_HF,
    G_LOW_BACK_PAIN,
    G_MENINGITIS,
    G_NEUTROPENIA,
    G_PE,
    G_PLEURAL,
    G_SEPSIS,
    G_STROKE,
    cant_miss_for,
    protocols_for,
)

# --------------------------------------------------------------------------------------------
# New citations (bibliographic data checked against PubMed E-utilities esummary, 2026-09-27)
# --------------------------------------------------------------------------------------------

C_ADVISED = Citation(
    "Nazerian P, Mueller C, Soeiro AM, et al.",
    "Diagnostic Accuracy of the Aortic Dissection Detection Risk Score Plus D-Dimer for Acute Aortic Syndromes: "
    "The ADvISED Prospective Multicenter Study",
    "Circulation", 2018, "137(3):250-258", doi="10.1161/CIRCULATIONAHA.117.029457", pmid="29030346",
    verified=True, short_author="ADvISED(Nazerian)",
)
C_CT_6H_SAH = Citation(
    "Perry JJ, Stiell IG, Sivilotti ML, et al.",
    "Sensitivity of computed tomography performed within six hours of onset of headache for diagnosis of "
    "subarachnoid haemorrhage: prospective cohort study",
    "BMJ", 2011, "343:d4277", doi="10.1136/bmj.d4277", pmid="21768192", verified=True,
)
C_MENINGITIS_EXAM = Citation(
    "Attia J, Hatala R, Cook DJ, Wong JG.",
    "The rational clinical examination. Does this adult patient have acute meningitis?",
    "JAMA", 1999, "282(2):175-181", doi="10.1001/jama.282.2.175", pmid="10411200", verified=True,
)
C_HYPERGLYCEMIC_CRISES = Citation(
    "Umpierrez GE, Davis GM, ElSayed NA, et al.",
    "Hyperglycemic Crises in Adults With Diabetes: A Consensus Report",
    "Diabetes Care", 2024, "47(8):1257-1275", doi="10.2337/dci24-0032", pmid="39052901", verified=True,
    short_author="ADA/EASD 고혈당 위기 합의",
)
C_PERFORATED_ULCER = Citation(
    "Tarasconi A, Coccolini F, Biffl WL, et al.",
    "Perforated and bleeding peptic ulcer: WSES guidelines",
    "World J Emerg Surg", 2020, "15:3", doi="10.1186/s13017-019-0283-9", pmid="31921329", verified=True,
    short_author="WSES 궤양 천공 지침",
)
C_ESC_HF = Citation(
    "McDonagh TA, Metra M, Adamo M, et al.",
    "2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure",
    "Eur Heart J", 2021, "42(36):3599-3726", doi="10.1093/eurheartj/ehab368", pmid="34447992", verified=True,
    short_author="ESC 심부전 지침",
)
C_SCROTAL = Citation(
    "Expert Panel on Urologic Imaging; Wang CL, Aryal B, et al.",
    "ACR Appropriateness Criteria Acute Onset of Scrotal Pain-Without Trauma, Without Antecedent Mass",
    "J Am Coll Radiol", 2019, "16(5S):S38-S43", doi="10.1016/j.jacr.2019.02.016", pmid="31054757", verified=True,
    short_author="ACR 급성 음낭 통증 적정성 기준",
)
C_ANAPHYLAXIS_CRITERIA = Citation(
    "Sampson HA, Muñoz-Furlong A, Campbell RL, et al.",
    "Second symposium on the definition and management of anaphylaxis: summary report",
    "J Allergy Clin Immunol", 2006, "117(2):391-397", doi="10.1016/j.jaci.2005.12.1303", pmid="16461139",
    verified=True, short_author="NIAID/FAAN 아나필락시스 기준",
)

NEW_CITATIONS: tuple[Citation, ...] = (
    C_ADVISED, C_CT_6H_SAH, C_MENINGITIS_EXAM, C_HYPERGLYCEMIC_CRISES, C_PERFORATED_ULCER, C_ESC_HF, C_SCROTAL,
    C_ANAPHYLAXIS_CRITERIA,
)

# --------------------------------------------------------------------------------------------
# Text normalisation and negation-aware reading of findings
# --------------------------------------------------------------------------------------------

_UNAVAILABLE = ("제공되지 않", "결과가 없", "시행되지 않", "not available")


def _norm(text: str) -> str:
    t = (text or "").lower()
    for a, b in (("µ", "u"), ("μ", "u"), ("β", "b"), ("−", "-"), ("–", "-"), ("，", ",")):
        t = t.replace(a, b)
    t = re.sub(r"st\s*-?\s*(구간|분절|segment)", "st 분절", t)
    # "보이지 않음", "관찰되지 않음", "동반되지 않음" → "없음" so the verb stem is not read as an affirmation
    t = re.sub(r"(보이|관찰되|확인되|시사되|동반되|나타나|발견되|들리|청진되|촉지되|만져지|있)지\s*않", "없", t)
    t = re.sub(r"(관찰|확인|발견)\s*(안\s*됨|안\s*되)", "없", t)
    return t.replace("비정상", "이상함")  # keep "정상" inside "비정상" from reading as normal


_SENT_END = re.compile(r"[.?!\n;]")
_NEG_AFTER = ("없", "않", "아니", "음성", "배제", "negative", "absent", "not seen", "none", "no evidence")
_NEG_NEAR = ("정상", "normal", "(-)")  # only right after the finding ("t파 정상"), not further down a list
_NEG_BEFORE = ("no ", "without ", "negative for", "absence of", "free of", "not ")
_AFFIRM = ("관찰", "있음", "있고", "있으", "있어", "있었", "있는", "보임", "보이", "확인", "시사", "의심", "양성", "동반",
           "상승", "하강", "증가", "감소", "역전", "도치", "소실", "positive", "seen", "present", "noted", "elevat",
           "consistent", "suggest")
_CLAUSE = ("으나", "지만", "는데", "그러나", " but ", "하나 ", "되나 ")


def _first(tail: str, tokens: tuple[str, ...], limit: int | None = None) -> int | None:
    pos = [i for i in (tail.find(k) for k in tokens) if i >= 0 and (limit is None or i < limit)]
    return min(pos) if pos else None


def _affirmed_at(t: str, idx: int, kw: str, near: bool = True) -> bool:
    """One occurrence of kw at idx in normalised text t: affirmed (True) or negated (False).

    Negated when, in the same sentence and clause, a negation follows ("출혈 없음"; a trailing negation covers a list:
    "급성 출혈, 뇌경색, 종괴 소견 없음") or an English negation precedes ("no acute hemorrhage"). An affirmation that
    comes first ("ST 분절 상승, T파 역전 없음") keeps it affirmed, unless the affirmation is itself negated
    ("상승 소견 없음"). near=False ignores "정상" right after kw (for normal patterns: "호흡음 대칭적이며 정상")."""
    s = max(t.rfind(c, 0, idx) for c in ".?!\n;") + 1
    m = _SENT_END.search(t, idx + len(kw))
    e = m.start() if m else len(t)
    if any(n in t[max(s, idx - 30):idx] for n in _NEG_BEFORE):
        return False
    tail = t[idx + len(kw):e]
    cut = _first(tail, _CLAUSE)
    if cut is not None:
        tail = tail[:cut]
    neg = [p for p in (_first(tail, _NEG_AFTER), _first(tail, _NEG_NEAR, limit=10) if near else None)
           if p is not None]
    if not neg:
        return True
    n = min(neg)
    for a in _AFFIRM:
        p = tail.find(a)
        if 0 <= p < n and _first(tail[p:p + len(a) + 8], _NEG_AFTER) is None:
            return True
    return False


def polarity(text: str, kw: str) -> str | None:
    """'pos' if any occurrence of kw is affirmed, 'neg' if all are negated, None if absent (or only about relatives).
    Text is normalised.

    2026-09-27: each occurrence is also read by the normalisation layer (clinical_rules.keyword_statuses). The
    reading stays conservative for a rule-out gate: an occurrence is affirmed when either reader affirms it (the
    layer adds hedged / uncertain / persisting findings: "열이 안 떨어져요", "경련했는지 모르겠어요"), negated only when
    both negate it, and dropped when the layer attributes it to a relative ("어머니가 폐색전증")."""
    # a ReadText is taken as already normalised (a _Turn text or _Ctx.facts): parsed once for many keywords
    t = text if isinstance(text, ReadText) else ReadText(_norm(text))
    k = kw.lower()
    idxs = [m.start() for m in re.finditer(re.escape(k), t)]
    if not idxs:
        return None
    # occurrences the layer has no finding for keep this module's own reading only
    layer = [st if by_layer or st == OTHER else None for st, by_layer in keyword_statuses(t, k, detail=True)]
    if len(layer) != len(idxs):  # normalisation changed the occurrences: legacy reading only
        layer = [None] * len(idxs)
    res = [_affirmed_at(t, i, k) or s in (POS, UNC) for i, s in zip(idxs, layer) if s != OTHER]
    if not res:
        return None
    return "pos" if any(res) else "neg"


_GENERIC_NORMAL = re.compile(
    r"정상|특이\s*(소견|사항|이상)?\s*(은|는|이|가)?\s*없|이상\s*소견\s*(은|이)?\s*없|이상\s*없|unremarkable|\bnormal\b"
    r"|no acute|within normal|청명|폐야\s*(가|는)?\s*(명확|깨끗)")


# --------------------------------------------------------------------------------------------
# Case context: responses, vitals, onset
# --------------------------------------------------------------------------------------------


@dataclass
class _Turn:
    i: int  # 0 = initial info
    type: str  # "" for the initial info
    content: str  # normalised action content
    text: str  # normalised response (or initial info)
    raw_content: str = ""

    @property
    def unavailable(self) -> bool:
        return any(u in self.text for u in _UNAVAILABLE)

    def cite(self, snippet: str) -> str:
        where = "처음 정보" if self.i == 0 else f"#{self.i} {self.type} '{self.raw_content[:30]}'"
        return f"{where} → {snippet.strip()[:90]}"


_RE_SBP = re.compile(r"(?:혈압|blood pressure|bp)\s*[:은는]?\s*(\d{2,3})\s*/\s*(\d{2,3})")
_RE_HR = re.compile(r"(?:맥박|심박수?|heart rate|pulse|hr)\s*[:은는]?\s*(?:분당\s*)?(\d{2,3})")
_RE_RR = re.compile(r"(?:호흡수|호흡|respiratory rate|rr)\s*[:은는]?\s*(?:분당\s*)?(\d{1,2})\s*(?:회|/|breaths|bpm)")
_RE_SPO2 = re.compile(r"(?:산소\s*포화도?|spo2|sao2|o2 sat\w*)[^0-9%]{0,20}(\d{2,3})\s*%")
_RE_TEMP = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(°\s*c|℃|°\s*f|℉|도)")
_RE_ARM_BP = re.compile(r"(오른|우측|right|왼|좌측|left)[^0-9/]{0,12}(\d{2,3})\s*/\s*\d{2,3}")
_RE_ONSET = (
    (re.compile(r"(\d+(?:\.\d+)?)\s*분\s*(?:전|째|정도 전)"), 1 / 60),
    (re.compile(r"(\d+(?:\.\d+)?)\s*시간\s*(?:전|째|정도 전|반 전)"), 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*ago"), 1.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s*ago"), 1 / 60),
    (re.compile(r"(\d+)\s*일\s*(?:전|째)"), 24.0),
    (re.compile(r"(\d+)\s*days?\s*ago"), 24.0),
    (re.compile(r"(\d+)\s*주\s*(?:전|째|간|동안)"), 168.0),
    (re.compile(r"(\d+)\s*(?:개월|달)\s*(?:전|째|간|동안)"), 720.0),
    (re.compile(r"(\d+)\s*년\s*(?:전|째|간|동안)"), 8760.0),
)
# measured findings of the normalisation layer -> vital sign (the finding's value is the measured number)
_VITAL_OF = {"SIGN:hypotension": "sbp", "SIGN:elevated_bp": "sbp", "SIGN:tachycardia": "hr", "SIGN:bradycardia": "hr",
             "SIGN:tachypnea": "rr", "SIGN:bradypnea": "rr", "SIGN:hypoxemia": "spo2", "SYM:fever": "temp",
             "SYM:high_fever": "temp", "SIGN:hypothermia": "temp"}
_RE_SPO2_LABEL = re.compile(r"산소|spo2|sao2|sp02|o2")
_WORST = {"sbp": min, "spo2": min, "hr": max, "rr": max}
_ONSET_Q = ("언제", "시작", "몇 시", "발병", "처음", "얼마나 됐", "얼마나 되", "onset", "when")
_KO_HOURS = (("한 시간", 1), ("두 시간", 2), ("세 시간", 3), ("네 시간", 4), ("다섯 시간", 5), ("반나절", 6),
             ("그저께", 48), ("어제", 24), ("어젯밤", 12), ("며칠", 72), ("지난주", 168))


class _Ctx:
    """Everything the rules read from one case. Built fresh per call (no cross-case state)."""

    def __init__(self, state):
        self.state = state
        self.cc = state.initial_info or ""
        self.turns: list[_Turn] = [_Turn(0, "", "", _norm(self.cc), "")]
        for i, t in enumerate(state.turns, 1):
            typ = getattr(t.action.type, "value", str(t.action.type))
            self.turns.append(_Turn(i, typ, _norm(t.action.content), _norm(t.response), t.action.content))
        # environment text only (no doctor questions); parsed once by the normalisation layer for all lookups
        self.facts = ReadText(". ".join(t.text for t in self.turns))
        self.age = age_years(self.cc)
        self._probe_cache: dict[str, _Outcome] = {}
        self._measured_cache: dict[int, dict[str, list[float]]] = {}
        self.sbp = self._latest(_RE_SBP, int, "sbp")
        self.hr = self._latest(_RE_HR, int, "hr")
        self.rr = self._latest(_RE_RR, int, "rr")
        self.spo2 = self._latest(_RE_SPO2, int, "spo2")
        self.temp = self._latest_temp()
        self.onset_h = self._onset_hours()
        # result interpreter (see CRITICAL_RESULT_DANGER): confirming findings, and findings that only raise a danger
        self.critical, self.critical_raised = _critical_results(state)

    # -- readers ----------------------------------------------------------------------------
    def _measured(self, t: _Turn) -> dict[str, list[float]]:
        """Vital-sign values the normalisation layer measured in one response."""
        if t.i not in self._measured_cache:
            from doctor_agent.nlp import parse
            out: dict[str, list[float]] = {}
            for f in parse(t.text, "patient"):
                name = _VITAL_OF.get(f.concept)
                if name and f.value is not None and f.subject == "patient":
                    if name == "spo2" and not _RE_SPO2_LABEL.search(f.span):
                        continue  # "폐동맥 포화도 66%" (cardiac catheterisation) is not the arterial SpO2
                    out.setdefault(name, []).append(f.value)
            self._measured_cache[t.i] = out
        return self._measured_cache[t.i]

    def _latest(self, rx: re.Pattern, cast, name: str = "") -> float | None:
        """Value from the latest response that has one. Within a response, the layer's measured values and this
        module's regex values are pooled and the worst one is kept (_WORST: lowest SBP/SpO2, highest HR/RR/T), so
        "누운 자세 128/78, 기립 후 108/68" reads 108 (conservative for shock, qSOFA, PERC and ADD-RS)."""
        val = None
        for t in self.turns:
            vs = list(self._measured(t).get(name, [])) if name else []
            vs += [float(m.group(1)) for m in rx.finditer(t.text)]
            if vs:
                val = cast(_WORST.get(name, lambda x: x[-1])(vs))
        return val

    def _latest_temp(self) -> float | None:
        val = None
        for t in self.turns:
            vs = list(self._measured(t).get("temp", []))
            for m in _RE_TEMP.finditer(t.text):
                x, unit = float(m.group(1)), m.group(2).replace(" ", "")
                if unit == "도" and not re.search(r"(체온|열|temp|bt)", t.text[max(0, m.start() - 12):m.start()]):
                    continue
                if unit in ("°f", "℉"):
                    x = (x - 32) * 5 / 9
                if 34 <= x <= 43:
                    vs.append(x)
            if vs:
                val = max(vs)
        return val

    def _onset_hours(self) -> float | None:
        # chief complaint first; later only answers to onset questions (other answers mention unrelated durations)
        texts = [self.turns[0].text] + [t.text for t in self.turns[1:]
                                        if t.type == "ASK" and any(q in t.content for q in _ONSET_Q)]
        for text in texts:
            best = None
            for rx, mult in _RE_ONSET:
                if m := rx.search(text):
                    best = (m.start(), float(m.group(1)) * mult) if best is None or m.start() < best[0] else best
            for word, h in _KO_HOURS:
                if (j := text.find(word)) >= 0 and (best is None or j < best[0]):
                    best = (j, float(h))
            if best:
                return best[1]
        return None

    def affirmed(self, kws: tuple[str, ...]) -> bool:
        return contains_affirmed(self.facts, kws)

    def denied(self, kws: tuple[str, ...]) -> bool:
        return any(k in self.facts and polarity(self.facts, k) == "neg" for k in kws)

    def chronic(self, level: int) -> bool:
        return duration_level(self.cc) >= level

    def probe(self, pid: str) -> "_Outcome":
        if pid not in self._probe_cache:
            self._probe_cache[pid] = PROBES[pid].read(self)
        return self._probe_cache[pid]

    def ledger_p(self, names: tuple[str, ...]) -> float | None:
        for e in getattr(getattr(self.state, "ddx_ledger", None), "entries", []):
            if any(same_dx(e.dx, n) for n in names):
                return e.p
        return None


# --------------------------------------------------------------------------------------------
# Probes: one finding read from one kind of action (e.g. "hemorrhage on head CT", "troponin value")
# --------------------------------------------------------------------------------------------


@dataclass
class _Outcome:
    result: str | None = None  # "normal" | "abnormal" | "done" (done but unreadable/unavailable) | None (not done)
    spans: list[str] = field(default_factory=list)
    kws: set[str] = field(default_factory=set)
    n_normal: int = 0
    values: list[float] = field(default_factory=list)


Reader = Callable[[_Turn, "_Ctx"], list[tuple[str, str, str, float | None]]]  # (result, snippet, kw, value)


@dataclass(frozen=True)
class Probe:
    id: str
    kind: ActionType
    action: str  # Korean content for the next action
    action_kw: tuple[str, ...]  # substrings of the doctor's action content marking this probe as done
    family: str = ""  # result family, to cut this result out of a multi-result response
    target: tuple[str, ...] = ()  # abnormal finding keywords (negation-aware)
    normal_re: str = ""  # explicit normal pattern
    generic_normal: bool = True  # "정상", "특이 소견 없음" counts as normal
    silent_normal: bool = False  # a returned result that does not mention the target counts as normal
    reader: Reader | None = None  # numeric reader, applied to every response (bundled lab panels)
    any_turn: bool = False  # qualitative probe read from every EXAM/TEST response (e.g. mental status)

    def matches(self, t: _Turn) -> bool:
        return t.type in ("ASK", "EXAM", "TEST") and any(k in t.content for k in self.action_kw)

    def read(self, c: _Ctx) -> _Outcome:
        out = _Outcome()
        rows: list[tuple[_Turn, str, str, str, float | None]] = []
        for t in c.turns:
            hit = self.matches(t)
            if self.reader is not None:
                for res, snip, kw, val in self.reader(t, c):
                    rows.append((t, res, snip, kw, val))
                if hit and out.result is None:
                    out.result = "done"
                continue
            if not (hit or (self.any_turn and (t.type in ("EXAM", "TEST") or t.i == 0))):
                continue
            if hit and out.result is None:
                out.result = "done"
            if t.unavailable:
                continue
            seg = _segment(t, self.family) if hit else t.text
            if seg is None:
                continue
            res, kws = self._classify(seg, t, hit)
            if res:
                rows.append((t, res, seg, kws, None))
        for t, res, snip, kw, val in rows:
            out.spans.append(t.cite(snip))
            if kw:
                out.kws.update(kw if isinstance(kw, tuple) else (kw,))
            if val is not None:
                out.values.append(val)
            if res == "normal":
                out.n_normal += 1
        results = {r[1] for r in rows}
        if "abnormal" in results:
            out.result = "abnormal"
            out.spans = [t.cite(s) for t, r, s, _, _ in rows if r == "abnormal"]
        elif "normal" in results:
            out.result = "normal"
        return out

    def _classify(self, seg: str, t: _Turn, hit: bool) -> tuple[str | None, tuple[str, ...]]:
        seg = ReadText(seg)  # already normalised; parsed once for all targets
        pos = tuple(k for k in self.target if k in seg and polarity(seg, k) == "pos")
        if pos:
            return "abnormal", pos
        if any(k in seg for k in self.target):
            return "normal", ()
        if self.normal_re and any(_affirmed_at(seg, m.start(), m.group(0), near=False)
                                  for m in re.finditer(self.normal_re, seg)):
            return "normal", ()
        if not hit:  # any_turn probes: only explicit statements count
            return None, ()
        if self.generic_normal and _GENERIC_NORMAL.search(seg):
            return "normal", ()
        if self.silent_normal and seg.strip():
            return "normal", ()
        return None, ()


FAMILY_LABELS: dict[str, tuple[str, ...]] = {
    "ecg": ("심전도", "ecg", "ekg"),
    "cxr": ("흉부 x", "흉부x", "흉부 엑스", "흉부 방사선", "흉부 사진", "흉부 단순", "가슴 x", "chest x", "cxr"),
    "ct_head": ("뇌 ct", "뇌ct", "두부 ct", "머리 ct", "brain ct", "head ct", "ct brain", "ct head", "비조영 ct"),
    "mri_brain": ("뇌 mri", "뇌mri", "머리 mri", "brain mri", "mri brain", "확산강조", "dwi"),
    "ctpa": ("ct 폐동맥", "폐동맥 ct", "ctpa", "ct pulmonary", "폐동맥 조영", "폐동맥 혈관조영"),
    "aorta_ct": ("대동맥 ct", "흉부 대동맥", "대동맥 조영", "대동맥 혈관조영", "aortic ct", "ct aorta"),
    "chest_ct": ("흉부 ct", "흉부ct", "chest ct", "ct chest"),
    "abd_ct": ("복부 ct", "복부ct", "복부-골반 ct", "복부 골반 ct", "복부골반 ct", "abdominal ct", "ct abdomen",
               "abdominopelvic"),
    "mesenteric_cta": ("장간막", "mesenteric", "복부 혈관조영", "복부 ct 혈관조영", "복부 cta"),
    "aorta_us": ("대동맥 초음파", "복부 대동맥", "aortic ultrasound", "aorta ultrasound"),
    "pelvic_us": ("질식 초음파", "경질 초음파", "골반 초음파", "transvaginal", "pelvic ultrasound", "pelvic us",
                  "산부인과 초음파"),
    "scrotal_us": ("음낭 초음파", "고환 초음파", "음낭 도플러", "scrotal"),
    "csf": ("뇌척수액", "csf", "요추천자", "요추 천자", "lumbar puncture"),
    "hcg": ("hcg", "임신 검사", "임신검사", "임신 반응", "pregnancy test"),
    "spine_mri": ("요추 mri", "척추 mri", "요천추 mri", "lumbar mri", "spine mri", "l-spine mri"),
    "coronary": ("관상동맥 조영", "관상동맥조영", "coronary angiogra", "심도자"),
}
_ALL_LABELS = [(f, lab) for f, labs in FAMILY_LABELS.items() for lab in labs]


def _segment(t: _Turn, family: str) -> str | None:
    """The part of a response that belongs to this result family. A multi-test request ("심전도와 흉부 X선") returns
    several results: cut from this family's label to the next other family's label. No label in the response:
    the whole response when only this family was requested, else None (cannot attribute)."""
    if not family:
        return t.text
    own = [t.text.find(lab) for lab in FAMILY_LABELS[family] if lab in t.text]
    if own:
        start = min(own)
        others = [i for f, lab in _ALL_LABELS if f != family for i in [t.text.find(lab, start + 1)] if i > start]
        return t.text[start:min(others) if others else len(t.text)]
    matched = [(f, lab) for f, lab in _ALL_LABELS if lab in t.content]
    requested = {f for f, lab in matched if not any(f2 != f and lab != l2 and lab in l2 for f2, l2 in matched)}
    return t.text if requested <= {family} else None


# --- numeric readers --------------------------------------------------------------------------------------------
_NUMBER = r"([<>≤≥]\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_WORD_NORMAL = ("정상", "음성", "normal", "negative", "미검출", "검출 안", "undetect")
_WORD_HIGH = ("상승", "증가", "양성", "elevat", "high", "positive", "높")


def _values(text: str, name_re: str, unit_re: str, window: int = 25):
    """(value, unit, below_limit, snippet, name_end) for each 'name ... number unit' in text."""
    for m in re.finditer(name_re, text):
        sub = text[m.end(): m.end() + window + 30]
        n = re.search(_NUMBER + r"\s*(" + unit_re + r")?", sub)
        if not n or n.start() > window or "참고" in sub[:n.start()] or "기준" in sub[:n.start()]:
            yield None, "", False, text[m.start(): m.end() + 30], m.end()
            continue
        val = float(n.group(2).replace(",", ""))
        lt = bool(n.group(1) and n.group(1).strip() in "<≤") or \
            sub[n.end():n.end() + 5].strip().startswith(("미만", "이하"))
        yield val, (n.group(3) or ""), lt, text[m.start(): m.end() + n.end() + 14], m.end()


def _word_result(text: str, end: int) -> str | None:
    near = text[end:end + 18]
    if any(w in near for w in _WORD_HIGH) and not any(w in near for w in ("없", "않")):
        return "abnormal"
    if any(w in near for w in _WORD_NORMAL):
        return "normal"
    return None


def _numeric_reader(name_re: str, unit_re: str, judge: Callable[[float, str, bool, _Ctx], str | None],
                    skip_families: tuple[str, ...] = (), serial: bool = False) -> Reader:
    """serial: also read later values with a unit in the same sentence ("3 ng/L, 3시간 후 재검 4 ng/L")."""
    def read(t: _Turn, c: _Ctx):
        if skip_families and any(lab in t.content for f in skip_families for lab in FAMILY_LABELS[f]):
            return []
        out = []
        for val, unit, lt, snip, end in _values(t.text, name_re, unit_re):
            res = judge(val, unit, lt, c) if val is not None else None
            if res is None:
                res = _word_result(t.text, end)
            if res:
                out.append((res, snip, "", val))
            if serial and val is not None:
                m = _SENT_END.search(t.text, end)
                rest = t.text[end:m.start() if m else len(t.text)].split("참고")[0].split("기준")[0]
                for n in list(re.finditer(_NUMBER + r"\s*(" + unit_re + r")", rest))[1:]:
                    v2 = float(n.group(2).replace(",", ""))
                    r2 = judge(v2, n.group(3), bool(n.group(1) and n.group(1).strip() in "<≤"), c)
                    if r2:
                        out.append((r2, rest[max(0, n.start() - 12): n.end()], "", v2))
        return out
    return read


def _troponin_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    """Conventional ng/mL: normal < 0.03 (or below the detection limit), abnormal > 0.04; high-sensitivity ng/L:
    normal < 14, abnormal >= 35 (between: indeterminate → repeat). Cut-offs are ours, from common assay 99th
    percentiles; the report's own flag words (상승/정상) are used when there is no number."""
    ngml = unit == "ng/ml" or (not unit and v < 1)
    if ngml:
        return "normal" if lt or v < 0.03 else "abnormal" if v > 0.04 else None
    return "normal" if lt or v < 14 else "abnormal" if v >= 35 else None


def _ddimer_ng(v: float, unit: str) -> float:
    if unit in ("ug/ml", "mg/l", "ug/ml feu", "mg/l feu") or (not unit and v < 10):
        return v * 1000
    return v  # ng/ml, ug/l


def _ddimer_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    """Normal < 500 ng/mL FEU (fixed cut-off; the age-adjusted cut-off for PE is applied in _ddimer_below)."""
    return "normal" if lt or _ddimer_ng(v, unit) < 500 else "abnormal"


def _lactate_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    mmol = v / 9.0 if unit == "mg/dl" else v
    return "normal" if lt or mmol < 2.0 else "abnormal"


def _glucose_mgdl(v: float, unit: str) -> float:
    return v * 18 if unit == "mmol/l" or (not unit and v < 35) else v


def _glucose_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    g = _glucose_mgdl(v, unit)
    return "normal" if g >= 70 else "abnormal"


def _bnp_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    return "normal" if lt or v < 100 else "abnormal"


def _ntbnp_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    return "normal" if lt or v < 300 else "abnormal"


def _bohb_judge(v: float, unit: str, lt: bool, c: _Ctx) -> str | None:
    return "normal" if lt or v < 1.5 else "abnormal" if v >= 3.0 else None


def _anc_reader(t: _Turn, c: _Ctx):
    """Absolute neutrophil count per uL: explicit ANC, or WBC x neutrophil %. >= 1000 normal, < 500 abnormal."""
    if any(lab in t.content for lab in FAMILY_LABELS["csf"]) or any(lab in t.text[:20] for lab in ("뇌척수액", "csf")):
        return []
    out = []

    def judge(anc: float, snip: str):
        if anc >= 1000:
            out.append(("normal", snip, "", anc))
        elif anc < 500:
            out.append(("abnormal", snip, "", anc))

    for val, unit, lt, snip, end in _values(t.text, r"(절대\s*호중구\s*수|호중구\s*절대\s*수|\banc\b|absolute neutrophil)",
                                            r"/ul|/mm3|x\s*10\^?3/ul|/μl"):
        if val is not None:
            judge(val * 1000 if val < 50 else val, snip)
    if out:
        return out
    wbc = re.search(r"(백혈구|wbc)[^0-9%]{0,12}" + _NUMBER, t.text)
    pct = re.search(r"(호중구|neutrophil|분엽핵)[^0-9%]{0,10}(\d+(?:\.\d+)?)\s*%", t.text)
    if wbc and pct:
        w = float(wbc.group(3).replace(",", ""))
        w = w * 1000 if w < 200 else w
        judge(w * float(pct.group(2)) / 100, t.text[wbc.start(): pct.end()])
    return out


def _csf_wbc_reader(t: _Turn, c: _Ctx):
    """CSF white cells (only from CSF results): <= 5/uL normal, > 5 abnormal (pleocytosis)."""
    if not (any(lab in t.content for lab in FAMILY_LABELS["csf"]) or "뇌척수액" in t.text or "csf" in t.text):
        return []
    out = []
    for val, unit, lt, snip, end in _values(t.text, r"(백혈구|wbc|세포\s*수|cell count)", r"/ul|/mm3|/mm³|개"):
        if val is not None:
            out.append(("normal" if val <= 5 else "abnormal", snip, "", val))
    return out


def _abga_reader(t: _Turn, c: _Ctx):
    """pH >= 7.30 and HCO3 >= 18 → no metabolic acidosis (normal); pH < 7.30 or HCO3 < 18 → abnormal."""
    ph = re.search(r"\bph\s*:?\s*(\d\.\d+)", t.text)
    hco3 = re.search(r"(hco3-?|중탄산\w*|bicarbonate)\s*:?\s*(\d+(?:\.\d+)?)", t.text)
    if not ph or not 6.5 < float(ph.group(1)) < 7.8:
        return []
    if "소변" in t.content or "urin" in t.content or "비중" in t.text[:ph.start()]:  # urine pH is not blood pH
        return []
    p = float(ph.group(1))
    h = float(hco3.group(2)) if hco3 else None
    snip = t.text[ph.start(): (hco3.end() if hco3 else ph.end()) + 8]
    if p < 7.30 or (h is not None and h < 18):
        return [("abnormal", snip, "", p)]
    if h is not None:
        return [("normal", snip, "", p)]
    return []


_RE_HCG = re.compile(r"(b-?hcg|hcg|임신\s*반응\s*(?:검사)?|임신\s*검사|pregnancy test)")


def _hcg_reader(t: _Turn, c: _Ctx):
    """Qualitative (양성/음성) or quantitative hCG: >= 25 mIU/mL positive, < 5 negative (5-24 indeterminate)."""
    out = []
    for m in _RE_HCG.finditer(t.text):
        near = t.text[m.end(): m.end() + 30]
        snip = t.text[m.start(): m.end() + 30]
        n = re.match(r"[^0-9.]{0,8}?([<>≤≥]\s*)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(miu/ml|iu/l)", near)
        if n:
            v = float(n.group(2).replace(",", ""))
            lt = bool(n.group(1) and n.group(1).strip() in "<≤")
            res = "normal" if lt or v < 5 else "abnormal" if v >= 25 else None
        elif re.match(r"[^.,]{0,12}?(양성|positive|\(\+\))", near):
            res = "abnormal"
        elif re.match(r"[^.,]{0,12}?(음성|negative|\(-\))", near):
            res = "normal"
        else:
            res = None
        if res:
            out.append((res, snip, "", None))
    return out


def _ketone_reader(t: _Turn, c: _Ctx):
    out = []
    for val, unit, lt, snip, end in _values(t.text, r"(베타\s*-?\s*하이드록시부티\w*|b-?hydroxybutyrate|bhb|bohb)",
                                            r"mmol/l"):
        res = _bohb_judge(val, unit, lt, c) if val is not None else _word_result(t.text, end)
        if res:
            out.append((res, snip, "", val))
    for m in re.finditer(r"(케톤|ketone)", t.text):
        near = t.text[m.end(): m.end() + 14]
        snip = t.text[m.start(): m.end() + 14]
        if re.search(r"[23]\+|\+\+|양성|positive|강양성", near):
            out.append(("abnormal", snip, "", None))
        elif re.search(r"음성|negative|\(-\)|없|trace|미량", near):
            out.append(("normal", snip, "", None))
    return out


# --- probe table ------------------------------------------------------------------------------------------------
_KW_ECG = FAMILY_LABELS["ecg"]
_ECG_ISCHEMIA = ("st 분절", "st 상승", "st 하강", "st elevation", "st depression", "st-t", "t파 역전", "t파 도치",
                 "t파 이상", "t wave inversion", "병적 q", "q파", "좌각 차단", "좌각차단", "lbbb", "stemi", "허혈")
STEMI_KW = ("st 분절", "st 상승", "st elevation", "stemi")
_AMS = ("의식 저하", "의식저하", "의식 변화", "의식변화", "의식이 흐", "의식이 없", "의식을 잃", "혼돈", "착란", "헛소리",
        "횡설수설", "섬망", "기면", "혼미", "혼수", "지남력 장애", "지남력 저하", "지남력 상실", "confus", "delir",
        "altered mental", "obtund", "letharg", "stupor", "coma", "disorient")
_NECK = ("경부 강직", "목 강직", "항부 강직", "nuchal rigidity", "neck stiffness", "kernig", "brudzinski",
         "뇌막 자극", "수막 자극", "meningeal sign", "meningismus")  # exam findings only (not "목이 뻣뻣" history)
_KW_NEURO = ("신경학적", "신경 검진", "신경학 검사", "신경 검사", "신경계 진찰", "의식", "경부 강직", "목 강직", "뇌막 자극",
             "neuro", "kernig", "brudzinski", "nuchal", "gcs", "지남력")

PROBES: dict[str, Probe] = {p.id: p for p in (
    Probe("ecg", ActionType.TEST, "12유도 심전도", _KW_ECG, "ecg", target=_ECG_ISCHEMIA, silent_normal=True),
    Probe("troponin", ActionType.TEST, "고감도 심장 트로포닌", ("트로포닌", "troponin", "심근효소", "심장효소", "ctn"),
          reader=_numeric_reader(r"(?:hs-?)?(?:c?tn[it]\b|트로포닌\s*-?\s*[it]?|troponin\s*-?\s*[it]?)",
                                 r"ng/ml|ng/l|pg/ml|ug/l", _troponin_judge, serial=True)),
    Probe("coronary", ActionType.TEST, "관상동맥 조영술", FAMILY_LABELS["coronary"], "coronary",
          target=("협착", "폐색", "occlusion", "stenosis", "혈전")),
    Probe("ddimer", ActionType.TEST, "D-dimer", ("d-dimer", "d dimer", "디다이머", "d-이합체", "d-다이머", "ddimer"),
          reader=_numeric_reader(r"(d\s*-?\s*dimer|디다이머|d-이합체|d-다이머)",
                                 r"ug/ml(?:\s*feu)?|mg/l(?:\s*feu)?|ng/ml|ug/l", _ddimer_judge)),
    Probe("aorta_cta", ActionType.TEST, "흉부 대동맥 CT 혈관조영",
          FAMILY_LABELS["aorta_ct"] + ("대동맥 ct 혈관", "경식도 초음파", "transesophageal"), "aorta_ct",
          target=("박리", "dissection", "내막 피판", "intimal flap", "벽내 혈종", "intramural", "가성 내강", "false lumen",
                  "이중 내강"), silent_normal=True),
    Probe("chest_ct_aorta", ActionType.TEST, "흉부 CT", FAMILY_LABELS["chest_ct"], "chest_ct",
          target=("박리", "dissection", "내막 피판", "intimal flap", "벽내 혈종", "intramural", "가성 내강"),
          normal_re=r"대동맥[^.]{0,20}(정상|이상 없|특이 소견 없)", generic_normal=False),
    Probe("ctpa", ActionType.TEST, "CT 폐동맥 조영술(CTPA)", FAMILY_LABELS["ctpa"], "ctpa",
          target=("충전 결손", "충만 결손", "충전결손", "충만결손", "filling defect", "폐색전", "폐동맥 색전", "색전", "혈전", "embol"),
          silent_normal=True),
    Probe("cxr_ptx", ActionType.TEST, "흉부 X선", FAMILY_LABELS["cxr"], "cxr",
          target=("기흉", "pneumothorax"), silent_normal=True),
    Probe("chest_ct_ptx", ActionType.TEST, "흉부 CT", FAMILY_LABELS["chest_ct"] + FAMILY_LABELS["ctpa"], "chest_ct",
          target=("기흉", "pneumothorax"), silent_normal=True),
    Probe("breath_sounds", ActionType.EXAM, "흉부 청진(양측 호흡음 대칭 여부)과 기관 위치 확인",
          ("청진", "호흡음", "폐음", "auscult", "흉부 진찰", "폐 진찰", "흉부 검진", "lung exam", "chest exam", "기관 위치"),
          target=("호흡음 감소", "호흡음이 감소", "호흡음 소실", "호흡음이 소실", "호흡음이 들리지", "호흡음 없", "absent breath",
                  "decreased breath", "diminished breath", "기관 편위", "기관이 한쪽", "tracheal deviation"),
          normal_re=r"(호흡음|폐음)[^.]{0,15}(정상|대칭|양호|깨끗|clear|equal)|양측\s*(호흡음|폐음)[^.]{0,6}(동일|대칭)"
                    r"|clear to auscultation|equal breath", generic_normal=False),
    Probe("ct_head", ActionType.TEST, "비조영 뇌 CT", FAMILY_LABELS["ct_head"], "ct_head",
          target=("지주막하", "subarachnoid", "sah", "출혈", "hemorrhage", "haemorrhage", "혈종", "hematoma"),
          silent_normal=True),
    Probe("mri_dwi", ActionType.TEST, "뇌 MRI(확산강조영상 포함)", FAMILY_LABELS["mri_brain"], "mri_brain",
          target=("급성 뇌경색", "뇌경색", "경색", "infarct", "확산 제한", "확산제한", "diffusion restriction",
                  "restricted diffusion"), silent_normal=True),
    Probe("mri_bleed", ActionType.TEST, "뇌 MRI", FAMILY_LABELS["mri_brain"], "mri_brain",
          target=("출혈", "hemorrhage", "혈종", "hematoma"), silent_normal=True),
    Probe("lp_sah", ActionType.TEST, "요추천자(뇌척수액 적혈구 수·황색변색 확인)", FAMILY_LABELS["csf"], "csf",
          target=("황색변색", "황변", "xanthochrom", "혈성", "bloody", "적혈구 다수", "적혈구가 다수"), silent_normal=True),
    Probe("lp_mening", ActionType.TEST, "요추천자 뇌척수액 검사(세포 수·단백·당·그람 염색·배양)", FAMILY_LABELS["csf"],
          reader=_csf_wbc_reader),
    Probe("csf_organism", ActionType.TEST, "뇌척수액 그람 염색·배양", FAMILY_LABELS["csf"], "csf",
          target=("그람 양성", "그람 음성", "gram-positive", "gram-negative", "gram positive", "gram negative", "쌍구균",
                  "diplococc", "배양 양성", "균이 동정", "균 동정"), generic_normal=False),
    Probe("vitals", ActionType.EXAM, "활력징후(혈압·맥박·호흡수·체온·산소포화도)",
          ("활력", "혈압", "맥박", "심박", "호흡수", "체온", "산소포화", "spo2", "vital", "blood pressure", "heart rate",
           "pulse", "respiratory rate", "temperature"), generic_normal=False, silent_normal=True),
    Probe("neck", ActionType.EXAM, "신경학적 진찰: 의식 수준·지남력, 경부 강직, Kernig·Brudzinski 징후",
          ("신경학적", "신경 검진", "신경학 검사", "신경 검사", "신경계 진찰", "경부", "목 강직", "목 진찰", "뇌막 자극",
           "수막 자극", "neuro", "kernig", "brudzinski", "nuchal", "meninge"),
          target=_NECK, generic_normal=False, any_turn=True),
    Probe("mental", ActionType.EXAM, "의식 수준·지남력(GCS) 평가", _KW_NEURO, target=_AMS,
          normal_re=r"의식\s*(은|이|상태|수준)?\s*:?\s*(명료|명확|정상|alert|clear)|지남력\s*(은|이)?\s*:?\s*(유지|정상|온전)"
                    r"|\balert\b|gcs\s*:?\s*15|명료", generic_normal=False, any_turn=True),
    Probe("glucose", ActionType.TEST, "혈당(모세혈관 혈당) 측정", ("혈당", "포도당", "glucose", "bst", "blood sugar"),
          reader=_numeric_reader(r"(혈당|혈중\s*포도당|혈청\s*포도당|glucose|bst|blood sugar)", r"mg/dl|mmol/l",
                                 _glucose_judge, skip_families=("csf",))),
    Probe("lactate", ActionType.TEST, "혈중 젖산", ("젖산", "락테이트", "lactate", "lactic"),
          reader=_numeric_reader(r"(젖산|lactate|lactic acid)(?!\s*(탈수소|수소|dehydrogenase|\)?\s*탈수소))",
                                 r"mmol/l|mg/dl", _lactate_judge)),
    Probe("anc", ActionType.TEST, "일반혈액검사(백혈구 감별계산, 절대 호중구 수)",
          ("일반혈액", "혈구", "cbc", "백혈구", "호중구", "complete blood count", "neutrophil"), reader=_anc_reader),
    Probe("hcg", ActionType.TEST, "소변 또는 혈청 β-hCG 임신 검사", FAMILY_LABELS["hcg"] + ("임신반응",),
          reader=_hcg_reader),
    Probe("pelvic_us", ActionType.TEST, "질식(골반) 초음파", FAMILY_LABELS["pelvic_us"], "pelvic_us",
          target=("자궁외 임신", "자궁 외 임신", "자궁외임신", "이소성 임신", "난관 임신", "ectopic", "tubal pregnancy",
                  "부속기 종괴", "부속기에 종괴", "adnexal mass", "혈복강", "hemoperitoneum"),
          normal_re=r"자궁\s*(강\s*)?(내|안)\s*[^.]{0,10}(임신낭|태아|난황낭)|intrauterine (pregnancy|gestation)"
                    r"|자궁내\s*임신|난황낭|yolk sac|태아\s*심박",
          generic_normal=False),
    Probe("aaa", ActionType.TEST, "복부 대동맥 초음파", FAMILY_LABELS["aorta_us"] + FAMILY_LABELS["abd_ct"],
          target=("대동맥류", "aneurysm", "대동맥 파열", "후복막 혈종", "후복막강 출혈", "retroperitoneal hem",
                  "retroperitoneal hematoma"),
          normal_re=r"대동맥[^.]{0,20}(정상|직경\s*[12]\.?\d?\s*cm|이상 없|특이 소견 없)|aorta[^.]{0,20}normal",
          generic_normal=False),
    Probe("abd_ct", ActionType.TEST, "복부 CT(조영증강)", FAMILY_LABELS["abd_ct"], "abd_ct",
          target=("대동맥류", "aneurysm", "후복막 혈종", "유리 공기", "자유 공기", "free air", "복강 내 공기",
                  "pneumoperitoneum", "천공", "perforat", "상장간막", "장간막 동맥 폐색", "장벽 내 공기", "pneumatosis",
                  "문맥 가스", "장 괴사"), silent_normal=True),
    Probe("free_air_xr", ActionType.TEST, "흉부 X선(직립)·복부 X선", FAMILY_LABELS["cxr"] + ("복부 x", "복부x", "복부 단순",
                                                                                     "abdominal x", "kub"),
          target=("유리 공기", "자유 공기", "free air", "횡격막 하 공기", "횡격막하 공기", "pneumoperitoneum"),
          generic_normal=False),
    Probe("mesenteric_cta", ActionType.TEST, "복부 CT 혈관조영(장간막 동맥)", FAMILY_LABELS["mesenteric_cta"],
          "mesenteric_cta",
          target=("상장간막", "sma", "장간막 동맥 폐색", "장간막동맥 폐색", "혈전", "색전", "embol", "thromb", "occlusion",
                  "장벽 조영증강 소실", "장벽 내 공기", "pneumatosis", "문맥 가스", "장 괴사", "허혈"),
          silent_normal=True),
    Probe("ketone", ActionType.TEST, "혈청 베타-하이드록시부티르산(또는 소변 케톤)",
          ("케톤", "ketone", "하이드록시부티", "hydroxybutyrate", "bhb", "소변 검사", "소변검사", "요검사", "urinalysis"),
          reader=_ketone_reader),
    Probe("abga", ActionType.TEST, "정맥혈 또는 동맥혈 가스분석(pH, 중탄산염)",
          ("가스분석", "가스 분석", "abga", "vbg", "blood gas", "혈액가스", "혈액 가스"), reader=_abga_reader),
    Probe("airway", ActionType.EXAM, "기도·호흡 평가(입술·혀·인두 부종, 쉰 목소리, 천명음, 양측 호흡음 청진)",
          ("기도", "천명", "쌕쌕", "호흡음", "청진", "쉰 목소리", "목소리", "인두", "구강", "stridor", "wheez", "airway",
           "auscult"),
          target=("천명", "stridor", "쌕쌕", "wheez", "혀 부종", "혀가 붓", "인두 부종", "후두 부종", "구인두 부종",
                  "목젖 부종", "쉰 목소리", "hoarse", "호흡음 감소", "호흡 곤란", "호흡곤란"),
          normal_re=r"(호흡음|폐음)[^.]{0,15}(정상|깨끗|clear)|기도\s*(는|가)?\s*(개방|유지|patent)|clear to auscultation",
          generic_normal=False),
    Probe("bnp", ActionType.TEST, "BNP 또는 NT-proBNP", ("bnp", "나트륨이뇨"),
          reader=lambda t, c: _numeric_reader(r"nt-?\s*pro\s*-?bnp", r"pg/ml", _ntbnp_judge)(t, c)
          or _numeric_reader(r"(?<!pro)(?<!pro-)\bbnp", r"pg/ml", _bnp_judge)(t, c)),
    Probe("cxr_edema", ActionType.TEST, "흉부 X선", FAMILY_LABELS["cxr"], "cxr",
          target=("폐부종", "폐 부종", "pulmonary edema", "울혈", "congestion", "커얼리", "kerley"), silent_normal=True),
    Probe("scrotal_us", ActionType.TEST, "음낭 도플러 초음파(양측 고환 혈류)", FAMILY_LABELS["scrotal_us"], "scrotal_us",
          target=("혈류 감소", "혈류가 감소", "혈류 소실", "혈류가 없", "혈류 부재", "염전", "torsion", "whirlpool",
                  "소용돌이"),
          normal_re=r"혈류[^.]{0,12}(정상|대칭|양호|유지)|normal (blood )?flow|symmetric flow"),
    Probe("cauda_history", ActionType.ASK,
          "소변이 안 나오거나 새는 증상, 대변 실금, 항문·회음부(안장 부위) 감각 저하가 있나요?",
          ("소변", "대변", "배변", "안장", "회음", "항문", "실금", "bladder", "bowel", "saddle"),
          target=("소변이 안 나", "소변을 못", "소변이 새", "요실금", "요저류", "대변 실금", "변실금", "안장", "회음부 감각",
                  "saddle", "retention", "incontinence"),
          normal_re=r"^(아니|없어|없었|괜찮|그런 (건|적은?) 없|no\b)|문제 없|이상 없", generic_normal=False),
    Probe("leg_neuro", ActionType.EXAM, "하지 신경학적 진찰(근력·감각·반사, 회음부 감각, 항문 괄약근 긴장도)",
          ("하지", "다리 근력", "근력", "신경학적", "반사", "괄약근", "회음부 감각", "neuro", "reflex", "rectal tone"),
          target=("근력 저하", "근력이 저하", "위약", "마비", "감각 저하", "감각 소실", "반사 소실", "반사 감소",
                  "괄약근 긴장도 저하", "괄약근 긴장도 감소", "안장 감각", "weakness", "saddle anesthesia"),
          silent_normal=False),
    Probe("spine_mri", ActionType.TEST, "요추 MRI", FAMILY_LABELS["spine_mri"], "spine_mri",
          target=("마미", "cauda equina", "척수 압박", "심한 협착", "경막외 농양", "epidural abscess", "전이"),
          silent_normal=True),
)}


# --------------------------------------------------------------------------------------------
# Composite clinical readings (Wells, PERC, ADD-RS, meningitis triad, qSOFA)
# --------------------------------------------------------------------------------------------

_DVT_SIGNS = ("다리가 붓", "다리 부종", "종아리 압통", "종아리가 붓", "종아리 부종", "하지 부종", "한쪽 다리", "calf swelling",
              "leg swelling", "심부정맥혈전", "dvt")
_IMMOBILE = ("수술", "침상", "부동", "깁스", "석고 붕대", "immobil", "surgery", "bedridden", "bed rest")
_PRIOR_VTE = ("혈전 병력", "혈전증 병력", "폐색전증 병력", "폐색전증을 앓", "혈전이 생긴 적", "혈전증을 앓", "previous dvt",
              "prior dvt", "history of dvt", "history of pe", "previous pe")
_HEMOPTYSIS = ("객혈", "피가 섞인 가래", "피 섞인 가래", "혈담", "hemoptysis")
_CANCER = ("암 치료", "암 진단", "암을 앓", "항암", "전이", "cancer", "malignan", "종양 치료")
_ESTROGEN = ("피임약", "호르몬", "에스트로겐", "estrogen", "contracepti", "hormone")
_TRAUMA = ("외상", "다쳤", "다친", "교통사고", "trauma", "injur")


def wells_partial(c: _Ctx) -> float:
    """Wells PE score from the case text, without 'PE is the most likely diagnosis' (3 points; not knowable from
    text, and the gate runs when another diagnosis is proposed). > 4 = 'PE likely' (two-level Wells, C_WELLS_PE)."""
    s = 0.0
    s += 3 if c.affirmed(_DVT_SIGNS) else 0
    s += 1.5 if c.hr is not None and c.hr > 100 else 0
    s += 1.5 if c.affirmed(_IMMOBILE) else 0
    s += 1.5 if c.affirmed(_PRIOR_VTE) else 0
    s += 1 if c.affirmed(_HEMOPTYSIS) else 0
    s += 1 if c.affirmed(_CANCER) else 0
    return s


def perc_negative(c: _Ctx) -> bool:
    """All 8 PERC items negative (C_PERC): age < 50, HR < 100, SpO2 >= 95% (measured), and no hemoptysis, estrogen
    use, prior VTE, unilateral leg swelling or recent surgery/trauma. We also require that the patient explicitly
    denied leg swelling or recent surgery (evidence the history was taken)."""
    if c.age is None or c.age >= 50 or c.hr is None or c.hr >= 100 or c.spo2 is None or c.spo2 < 95:
        return False
    if any(c.affirmed(k) for k in (_HEMOPTYSIS, _ESTROGEN, _PRIOR_VTE, _DVT_SIGNS, _IMMOBILE, _TRAUMA)):
        return False
    return c.denied(_DVT_SIGNS + _IMMOBILE)


def _ddimer_below(c: _Ctx, age_adjusted: bool) -> list[str] | None:
    """D-dimer below 500 ng/mL FEU, or below age x 10 ng/mL above 50 years (ESC 2019, age-adjusted)."""
    o = c.probe("ddimer")
    if o.result != "normal" and not (age_adjusted and o.result == "abnormal"):
        return None
    cutoff = max(500.0, c.age * 10) if age_adjusted and c.age and c.age > 50 else 500.0
    vals = []
    for t in c.turns:
        for val, unit, lt, snip, end in _values(t.text, r"(d\s*-?\s*dimer|디다이머|d-이합체|d-다이머)",
                                                r"ug/ml(?:\s*feu)?|mg/l(?:\s*feu)?|ng/ml|ug/l"):
            if val is not None:
                vals.append((lt or _ddimer_ng(val, unit) < cutoff, t.cite(snip)))
            elif (w := _word_result(t.text, end)) is not None:
                vals.append((w == "normal", t.cite(snip)))
    if vals and all(ok for ok, _ in vals):
        return [s for _, s in vals]
    return None


def add_rs(c: _Ctx) -> int:
    """Aortic Dissection Detection Risk Score (C_ADD_RS): 1 point per category with any feature present."""
    cond = c.affirmed(("마르판", "marfan", "결합조직 질환", "대동맥 질환 가족력", "대동맥 박리 가족력", "대동맥판 질환",
                       "대동맥판막", "이엽성 대동맥판", "대동맥 수술", "대동맥 시술", "흉부 대동맥류", "대동맥류 병력",
                       "aortic valve disease", "bicuspid", "thoracic aortic aneurysm"))
    pain = c.affirmed(("갑자기", "갑작스", "벼락", "찢어지", "찢기는", "뜯기는", "찢는", "칼로 베", "tearing", "ripping",
                       "sudden", "abrupt", "극심한", "인생 최악", "severe"))
    arms = [(m.group(1), int(m.group(2))) for t in c.turns for m in _RE_ARM_BP.finditer(t.text)]
    right = [v for s, v in arms if s in ("오른", "우측", "right")]
    left = [v for s, v in arms if s in ("왼", "좌측", "left")]
    arm_diff = bool(right and left and abs(right[-1] - left[-1]) > 20)
    exam = arm_diff or (c.sbp is not None and c.sbp < 90) or c.affirmed((
        "맥박 차이", "맥박이 약", "맥박 소실", "맥박이 만져지지", "pulse deficit", "양팔 혈압 차", "혈압 차이", "확장기 잡음",
        "대동맥판 역류", "diastolic murmur", "aortic regurg", "쇼크", "shock", "저혈압", "hypotens", "편마비", "한쪽 팔다리",
        "focal deficit"))
    return int(cond) + int(pain) + int(exam)


def _no_fever(c: _Ctx) -> list[str] | None:
    if c.temp is None or c.temp >= 38.0:
        return None
    if c.affirmed(("발열", "열이 나", "열이 났", "열이 있", "고열", "미열", "오한", "fever", "febrile", "chills")):
        return None
    return [f"체온 {c.temp:.1f}℃, 발열 병력 없음"]


def _mental_normal(c: _Ctx) -> list[str] | None:
    o = c.probe("mental")
    return o.spans if o.result == "normal" and not c.affirmed(_AMS) else None


def _qsofa(c: _Ctx) -> int | None:
    """qSOFA from measured vitals (SBP <= 100, RR >= 22) and altered mentation; None when SBP or RR is unknown."""
    if c.sbp is None or c.rr is None:
        return None
    return int(c.sbp <= 100) + int(c.rr >= 22) + int(c.affirmed(_AMS) or c.probe("mental").result == "abnormal")


# --------------------------------------------------------------------------------------------
# Rule-out table
# --------------------------------------------------------------------------------------------

Finder = Callable[[_Ctx], "list[str] | None"]


@dataclass(frozen=True)
class Step:
    probe: str
    when: Callable[[_Ctx], bool] | None = None  # step applies only when this is true
    reason: str = ""  # Korean
    repeat: int = 0  # >0: a repeat of the probe, due while fewer than this many results exist
    content: str = ""  # overrides the probe's action text


@dataclass(frozen=True)
class RuleOut:
    name: str  # Korean display name (as in protocols.cant_miss where it appears)
    aliases: tuple[str, ...]
    tier: int  # 1 = minutes (immediately life-threatening), 2 = hours, 3 = urgent but slower
    criteria: str  # Korean: what counts as ruled out
    rule_out: Finder
    confirm: Finder
    steps: tuple[Step, ...]
    citations: tuple[Citation, ...]
    verification: str
    note: str = ""
    check_ids: tuple[str, ...] = ()  # protocols.py checks whose applicability makes this CC danger live
    applies: Callable[[_Ctx], bool] | None = None  # custom applicability for CC-derived dangers


def _all(*finders: Finder) -> Finder:
    def f(c: _Ctx):
        out: list[str] = []
        for g in finders:
            r = g(c)
            if r is None:
                return None
            out += r
        return out
    return f


def _any(*finders: Finder) -> Finder:
    def f(c: _Ctx):
        for g in finders:
            if (r := g(c)) is not None:
                return r
        return None
    return f


def _is(pid: str, result: str, kws: tuple[str, ...] = ()) -> Finder:
    def f(c: _Ctx):
        o = c.probe(pid)
        if o.result != result or (kws and not (o.kws & set(kws))):
            return None
        return o.spans or [f"{pid}: {result}"]
    return f


def _when(pred: Callable[[_Ctx], bool], label: str) -> Finder:
    return lambda c: [label] if pred(c) else None


# --- per-danger findings ----------------------------------------------------------------------------------------
def _acs_out(c: _Ctx):
    ecg = _is("ecg", "normal")(c)
    if ecg is None:
        return None
    if c.chronic(2):  # months-long chest pain: troponin not indicated (protocols acute_only=2)
        return ecg + ["수개월 이상 지속된 흉통(급성 아님)"]
    trop = c.probe("troponin")
    if trop.result != "normal":
        return None
    need = 2 if c.onset_h is not None and c.onset_h < 3 else 1
    return ecg + trop.spans if trop.n_normal >= need else None


_PERICARDITIS_ECG = ("오목", "concave", "pr 분절 하강", "pr 하강", "pr depression", "광범위", "미만성", "diffuse",
                     "다수의 유도", "대부분의 유도", "전 유도")


def _stemi(c: _Ctx):
    """ST elevation read on the ECG, unless the report describes the diffuse concave / PR-depression pattern of
    pericarditis."""
    ev = _is("ecg", "abnormal", STEMI_KW)(c)
    return ev if ev is not None and not any(k in " ".join(ev) for k in _PERICARDITIS_ECG) else None


def _pe_out(c: _Ctx):
    if (r := _is("ctpa", "normal")(c)) is not None:
        return r
    if wells_partial(c) <= 4 and (d := _ddimer_below(c, age_adjusted=True)) is not None:
        return [f"Wells(부분) {wells_partial(c):g}점 ≤ 4"] + d
    p = c.ledger_p(("폐색전증",))
    if perc_negative(c) and (p is None or p < 0.15):
        return ["PERC 8항목 모두 음성(낮은 사전확률)"]
    return None


def _sah_out(c: _Ctx):
    ct = _is("ct_head", "normal")(c)
    if ct is None:
        return None
    alert = not c.affirmed(_AMS) and c.probe("mental").result != "abnormal"
    if c.onset_h is not None and c.onset_h <= 6 and alert:
        return ct + [f"두통 발생 {c.onset_h:g}시간 뒤(6시간 이내) CT, 의식 명료"]
    lp = _is("lp_sah", "normal")(c)
    return ct + lp if lp is not None else None


def _applies_checks(check_ids: tuple[str, ...], c: _Ctx) -> bool:
    """True when one of these protocols.py checks applies to the chief complaint (+ learned facts)."""
    text = ReadText(f"{c.cc}. {c.facts}")  # parsed once for all checks
    return any(ch.id in check_ids and ch.applies(text, cc=c.cc) for p in protocols_for(c.cc) for ch in p.checks)


def _gestation_12w(c: _Ctx) -> bool:
    """A stated gestation of >= 12 weeks: the pregnancy is established (ectopic is no longer the question)."""
    return any(int(m.group(1)) >= 12 for m in re.finditer(r"(?:임신|재태)\s*(\d+)\s*주", c.facts))


_FOCAL = ("편마비", "반신", "한쪽 팔", "한쪽 다리", "한쪽 팔다리", "팔다리 힘", "팔다리에 힘", "힘 빠짐", "힘이 빠", "마비",
          "구음장애", "구음 장애", "발음이 어눌", "말이 어눌", "실어", "말을 못", "안면 마비", "얼굴이 비뚤", "입이 돌아",
          "시야 결손", "한쪽 눈", "복시", "실조", "hemipare", "hemipleg", "aphasia", "dysarthria", "facial droop",
          "focal", "weakness")


def _ptx_applies(c: _Ctx) -> bool:
    return c.affirmed(("숨이 차", "숨차", "숨쉬기", "숨을 쉬", "호흡곤란", "호흡 곤란", "dyspnea", "short of breath", "기흉",
                       "pneumothorax")) or c.affirmed(_TRAUMA) or (c.spo2 is not None and c.spo2 < 94) \
        or (c.sbp is not None and c.sbp < 90)


def _perforation_applies(c: _Ctx) -> bool:
    return c.affirmed(("반발압통", "반발 압통", "반발통", "반동 압통", "근성 방어", "근육 방어", "복막 자극", "복막자극",
                       "판자", "복부 강직", "rebound", "guarding", "rigid", "천공", "free air", "peritonitis"))


def _dka_confirm(c: _Ctx):
    k, a = c.probe("ketone"), c.probe("abga")
    return k.spans + a.spans if k.result == "abnormal" and a.result == "abnormal" else None


def _dka_out(c: _Ctx):
    if (r := _is("ketone", "normal")(c)) is not None:
        return r
    if (r := _is("abga", "normal")(c)) is not None:
        return r
    g = c.probe("glucose")
    sglt2 = c.affirmed(("sglt2", "글리플로진", "gliflozin"))
    if g.values and max(_glucose_mgdl(v, "") for v in g.values) < 200 and not c.affirmed(("당뇨", "diabet")) \
            and not sglt2:
        return g.spans + ["혈당 < 200 mg/dL, 당뇨병 병력 없음"]
    return None


def _sepsis_out(c: _Ctx):
    q = _qsofa(c)
    lac = _is("lactate", "normal")(c)
    if q == 0 and lac is not None and (c.sbp or 0) > 100:
        return [f"수축기 혈압 {c.sbp}, 호흡수 {c.rr}, 의식 변화 없음(qSOFA 0)"] + lac
    return None


def _sepsis_confirm(c: _Ctx):
    q = _qsofa(c)
    lac = c.probe("lactate")
    if lac.result == "abnormal" and any(v >= 4 for v in lac.values):
        return lac.spans
    if q is not None and q >= 2 and (c.temp is not None and c.temp >= 38 or c.affirmed(("발열", "고열", "fever"))):
        return [f"qSOFA {q}점 + 발열"]
    return None


def _anaphylaxis_out(c: _Ctx):
    if c.sbp is None or c.sbp < 90 or c.affirmed(("저혈압", "쇼크", "hypotens", "shock", "실신", "기절")):
        return None
    if c.spo2 is not None and c.spo2 < 94:
        return None
    air = _is("airway", "normal")(c)
    return [f"수축기 혈압 {c.sbp}"] + air if air is not None else None


def _anaphylaxis_confirm(c: _Ctx):
    if (r := _is("airway", "abnormal")(c)) is not None:
        return r
    if c.sbp is not None and c.sbp < 90:
        return [f"수축기 혈압 {c.sbp} mmHg"]
    return None


def _cauda_out(c: _Ctx):
    if (r := _is("spine_mri", "normal")(c)) is not None:
        return r
    h = _is("cauda_history", "normal")(c)
    e = _is("leg_neuro", "normal")(c)
    return h + e if h is not None and e is not None else None


# Words that make a pneumothorax a tension pneumothorax (report or findings). Shared with the critical-result reader.
TENSION_PTX_MARKERS = ("긴장성", "종격동 이동", "종격동 전위", "종격동 편위", "기관 편위", "기관 전위", "mediastinal shift",
                       "tracheal deviation", "tension")


def _ptx_confirm(c: _Ctx):
    tension = _affirmed_any_language(c.facts, TENSION_PTX_MARKERS) or (c.sbp is not None and c.sbp < 90)
    if not tension:
        return None
    for pid in ("cxr_ptx", "chest_ct_ptx"):
        o = c.probe(pid)
        if o.result == "abnormal":
            return o.spans
    # a pneumothorax the result interpreter read on imaging (raised, not confirmed by the reading itself)
    return c.critical_raised.get("긴장성 기흉")


def _ptx_out(c: _Ctx):
    """Normal breath sounds, or imaging without pneumothorax. Once imaging showed a pneumothorax (critical result),
    only the bedside check can rule tension out."""
    if c.critical_raised.get("긴장성 기흉"):
        return _is("breath_sounds", "normal")(c)
    return _any(_is("breath_sounds", "normal"), _is("cxr_ptx", "normal"), _is("chest_ct_ptx", "normal"))(c)


_AD_ALIASES = ("대동맥 박리", "급성 대동맥 증후군", "대동맥박리", "aortic dissection", "acute aortic syndrome")

RULE_OUT_TABLE: tuple[RuleOut, ...] = (
    # ---------------- tier 1: minutes ----------------
    RuleOut(
        "긴장성 기흉", ("기흉", "tension pneumothorax", "pneumothorax"), 1,
        "양측 호흡음 대칭(청진) 또는 흉부 X선/CT에 기흉 없음",
        rule_out=_ptx_out,
        confirm=_ptx_confirm,
        steps=(Step("breath_sounds", reason="긴장성 기흉은 임상 진단: 청진이 가장 빠름"),
               Step("cxr_ptx", reason="기흉 확인(불안정하면 영상 전에 감압)")),
        citations=(G_PLEURAL,), verification="unverified", applies=_ptx_applies,
        note="BTS 2023 (verified citation): tension pneumothorax is a clinical diagnosis; decompression must not wait "
             "for imaging (reviewer knowledge, as in protocols.py). Normal bilateral breath sounds / a CXR without "
             "pneumothorax as rule-out evidence is our operationalisation. Live when dyspnea, trauma, SpO2 < 94% or "
             "SBP < 90. 2026-09-29: a pneumothorax on imaging confirms only with tension words in the report "
             "(mediastinal/tracheal shift, 긴장성, tension; negation-aware) or SBP < 90; otherwise the danger stays "
             "unresolved and only normal breath sounds rule it out (CRITICAL_RESULT_REQUIRES, _ptx_out)."),
    RuleOut(
        "아나필락시스", ("anaphylaxis", "아나필락시스 쇼크", "anaphylactic shock"), 1,
        "수축기 혈압 ≥ 90, 산소포화도 ≥ 94%, 기도·호흡 평가 정상(천명·협착음·인두/후두 부종 없음)",
        rule_out=_anaphylaxis_out, confirm=_anaphylaxis_confirm,
        steps=(Step("vitals", when=lambda c: c.sbp is None or c.spo2 is None, reason="저혈압·저산소 확인"),
               Step("airway", reason="호흡기 침범(천명·협착음·부종) 확인")),
        citations=(C_ANAPHYLAXIS_CRITERIA, G_ANAPHYLAXIS), verification="unverified", check_ids=("epinephrine",),
        note="NIAID/FAAN 2006 criteria (Sampson; PubMed 16461139 verified, abstract has no criteria text): skin/mucosal "
             "involvement plus respiratory compromise or reduced BP. Absence of respiratory and circulatory "
             "involvement as rule-out is our operationalisation (reviewer knowledge). GI involvement is not read."),
    RuleOut(
        "상기도 부종(혈관부종)", ("혈관부종", "상기도 부종", "angioedema", "후두 부종", "laryngeal edema"), 1,
        "기도 평가 정상(혀·인두·후두 부종, 쉰 목소리, 협착음 없음)",
        rule_out=_is("airway", "normal"), confirm=_is("airway", "abnormal"),
        steps=(Step("airway", reason="기도 침범 여부"),),
        citations=(G_ANAPHYLAXIS,), verification="unverified", check_ids=("airway_breathing",),
        note="Airway exam as rule-out of airway involvement: reviewer knowledge."),
    RuleOut(
        "대동맥 박리", _AD_ALIASES, 1,
        "대동맥 CT 혈관조영 음성, 또는 ADD-RS ≤ 1 + D-dimer < 500 ng/mL",
        rule_out=_any(_is("aorta_cta", "normal"), _is("chest_ct_aorta", "normal"),
                      _all(_when(lambda c: add_rs(c) <= 1, "ADD-RS ≤ 1"),
                           lambda c: _ddimer_below(c, age_adjusted=False))),
        confirm=_any(_is("aorta_cta", "abnormal"), _is("chest_ct_aorta", "abnormal")),
        steps=(Step("ddimer", when=lambda c: add_rs(c) <= 1, reason="ADD-RS ≤ 1: D-dimer 음성이면 배제(ADvISED)"),
               Step("aorta_cta", reason="ADD-RS ≥ 2 또는 D-dimer 양성: 대동맥 CT 혈관조영")),
        citations=(C_ADVISED, C_ADD_RS, G_AORTA), verification="primary", check_ids=("aorta_imaging",),
        note="ADvISED (abstract read 2026-09-27): ADD-RS <= 1 with D-dimer < 500 ng/mL had a 0.3% failure rate "
             "(3/924). ADD-RS items per C_ADD_RS, read from the case text (unstated = absent). A plain chest CT counts "
             "only when it states the aorta is normal. Live in chest pain when protocols' aorta_imaging applies."),
    RuleOut(
        "복부 대동맥류 파열", ("복부 대동맥류", "대동맥류 파열", "ruptured aaa", "abdominal aortic aneurysm", "aaa"), 1,
        "복부 대동맥 초음파/CT에서 대동맥류 없음(대동맥 정상)",
        rule_out=_any(_is("aaa", "normal"), _is("abd_ct", "normal")),
        confirm=_any(_is("aaa", "abnormal"), _is("abd_ct", "abnormal", ("대동맥류", "aneurysm", "후복막 혈종"))),
        steps=(Step("aaa", reason="대동맥류 유무(초음파가 가장 빠름)"),),
        citations=(G_AAA,), verification="unverified", check_ids=("aaa_imaging",),
        note="SVS 2018 (verified citation). Ultrasound/CT showing a normal aorta as rule-out is reviewer knowledge; "
             "an ultrasound counts only when it mentions the aorta, a whole-abdomen CT reported normal counts."),
    RuleOut(
        "급성 관상동맥 증후군", ("급성 심근경색", "심근경색", "불안정 협심증", "stemi", "nstemi", "acs", "acute coronary syndrome",
                         "myocardial infarction", "급성 관동맥 증후군"), 1,
        "심전도 허혈 변화 없음 + 트로포닌 정상(증상 3시간 이내면 1~3시간 뒤 재검까지 정상)",
        rule_out=_acs_out,
        confirm=_any(_stemi, _all(_is("ecg", "abnormal"), _is("troponin", "abnormal")), _is("coronary", "abnormal")),
        steps=(Step("ecg", reason="도착 10분 이내 심전도(STEMI 확인)"),
               Step("troponin", when=lambda c: not c.chronic(2), reason="심근 손상 확인"),
               Step("troponin", repeat=2, content="고감도 심장 트로포닌 재검(첫 검사 1~3시간 뒤)",
                    when=lambda c: not c.chronic(2) and (c.probe("troponin").result is None
                                                         or c.probe("troponin").result == "done"
                                                         or (c.onset_h is not None and c.onset_h < 3)),
                    reason="증상 3시간 이내이거나 첫 값이 애매하면 연속 측정")),
        citations=(G_CHEST_PAIN,), verification="secondary", check_ids=("ecg", "troponin"),
        note="AHA/ACC 2021 chest pain (secondary summaries): ECG within 10 min; hs-cTn preferred; a single hs-cTn below "
             "the limit of detection can rule out when symptoms began > 3 h earlier, otherwise serial 0/1-3 h "
             "sampling. Troponin cut-offs are ours (see _troponin_judge)."),
    RuleOut(
        "지주막하 출혈", ("지주막하출혈", "sah", "subarachnoid hemorrhage", "subarachnoid haemorrhage",
                     "뇌동맥류 파열"), 1,
        "두통 6시간 이내·의식 명료 환자의 비조영 뇌 CT 음성, 또는 CT 음성 + 요추천자 정상(황색변색 없음)",
        rule_out=_sah_out,
        confirm=_any(_is("ct_head", "abnormal", ("지주막하", "subarachnoid", "sah")), _is("lp_sah", "abnormal", ("황색변색", "황변", "xanthochrom"))),
        steps=(Step("ct_head", reason="벼락두통: 비조영 뇌 CT"),
               Step("lp_sah", when=lambda c: c.probe("ct_head").result == "normal",
                    reason="발병 6시간 넘었거나 시각 불명·의식 변화: CT 음성이어도 요추천자")),
        citations=(C_CT_6H_SAH, G_HEADACHE), verification="primary", check_ids=("brain_ct",),
        note="Perry 2011 BMJ (abstract read 2026-09-27): CT within 6 h of onset in neurologically intact adults found "
             "all 121 SAH (sensitivity 100%). Onset time is read from the text ('3시간 전'); unknown onset needs LP. "
             "ACEP 2019 (G_HEADACHE) Level B, secondary."),
    # ---------------- tier 2: hours ----------------
    RuleOut(
        "폐색전증", ("폐동맥 색전증", "pulmonary embolism", "pe", "폐색전"), 2,
        "CTPA 음성, 또는 Wells ≤ 4 + D-dimer 음성(50세 초과는 나이×10), 또는 낮은 사전확률에서 PERC 8항목 음성",
        rule_out=_pe_out, confirm=_is("ctpa", "abnormal"),
        steps=(Step("ddimer", when=lambda c: wells_partial(c) <= 4, reason="Wells ≤ 4: D-dimer 음성이면 배제"),
               Step("ctpa", reason="Wells > 4 또는 D-dimer 양성: CT 폐동맥 조영")),
        citations=(G_PE, C_WELLS_PE, C_PERC), verification="unverified", check_ids=("pe_workup",),
        note="ESC 2019 (verified citation; recommendation text not re-read here): D-dimer in non-high clinical "
             "probability, age-adjusted cut-off > 50 y, CTPA otherwise. Wells computed without the 'PE most likely' "
             "item (ours). PERC only when the ledger probability for PE is < 15%."),
    RuleOut(
        "뇌수막염", ("세균성 뇌수막염", "수막염", "세균성 수막염", "바이러스성 수막염", "meningitis", "bacterial meningitis"), 2,
        "발열·경부 강직·의식 변화가 모두 없음(진찰로 확인), 또는 뇌척수액 백혈구 ≤ 5/μL",
        rule_out=_any(_all(_no_fever, _is("neck", "normal"), _mental_normal), _is("lp_mening", "normal")),
        confirm=_any(_is("lp_mening", "abnormal"), _is("csf_organism", "abnormal")),
        steps=(Step("vitals", when=lambda c: c.temp is None, reason="발열 확인"),
               Step("neck", reason="경부 강직·의식 변화 확인(셋 다 없으면 배제)"),
               Step("ct_head", when=lambda c: c.affirmed(_AMS) or c.probe("mental").result == "abnormal"
                    or c.affirmed(("경련", "발작", "seizure", "편마비", "유두부종", "papilledema", "면역저하", "항암")),
                    reason="의식 변화·국소 결손·경련·면역저하: 요추천자 전 CT"),
               Step("lp_mening", reason="수막염 징후가 하나라도 있으면 요추천자(항생제 지연 금지)")),
        citations=(C_MENINGITIS_EXAM, G_MENINGITIS), verification="primary", check_ids=("meningitis_workup",),
        note="Attia 1999 JAMA (abstract read 2026-09-27): absence of fever, neck stiffness and altered mental status "
             "effectively eliminates meningitis (sensitivity 99-100%). Each must be documented (temperature < 38 "
             "measured and no fever history; neck exam negative; mental status explicitly normal). CSF WBC <= 5/uL "
             "normal (reviewer knowledge). CT-before-LP indications from IDSA 2004 (secondary)."),
    RuleOut(
        "패혈증", ("패혈성 쇼크", "sepsis", "septic shock", "균혈증"), 2,
        "측정한 활력징후로 qSOFA 0(수축기 혈압 > 100, 호흡수 < 22, 의식 정상) + 젖산 < 2 mmol/L",
        rule_out=_sepsis_out, confirm=_sepsis_confirm,
        steps=(Step("vitals", when=lambda c: c.sbp is None or c.rr is None, reason="혈압·호흡수·의식"),
               Step("lactate", reason="조직 저관류(젖산) 확인")),
        citations=(G_SEPSIS,), verification="unverified", check_ids=("blood_culture",),
        note="SSC 2021 recommends against qSOFA alone for screening and suggests measuring lactate (read, see "
             "protocols.py). Requiring qSOFA 0 AND lactate < 2 for rule-out is our operationalisation. Confirm: "
             "lactate >= 4 or qSOFA >= 2 with fever (ours)."),
    RuleOut(
        "자궁외 임신", ("임신(자궁외 임신 포함)", "이소성 임신", "난관 임신", "ectopic pregnancy", "자궁 외 임신"), 2,
        "β-hCG 음성, 또는 초음파에서 자궁 내 임신 확인",
        rule_out=_any(_is("hcg", "normal"), _is("pelvic_us", "normal")),
        confirm=_any(_is("pelvic_us", "abnormal", ("자궁외 임신", "자궁 외 임신", "자궁외임신", "이소성 임신", "난관 임신",
                                                   "ectopic", "tubal pregnancy")),
                     _all(_is("hcg", "abnormal"), _is("pelvic_us", "abnormal"))),
        steps=(Step("hcg", reason="가임기 여성 복통: 임신 여부"),
               Step("pelvic_us", when=lambda c: c.probe("hcg").result in ("abnormal", "done"),
                    reason="hCG 양성: 자궁 내 임신 확인")),
        citations=(G_ECTOPIC, G_EARLY_PREGNANCY), verification="unverified", check_ids=("pregnancy_test", "pelvic_us"),
        applies=lambda c: _applies_checks(("pregnancy_test", "pelvic_us"), c) and not _gestation_12w(c),
        note="ACOG PB 193 / ACEP 2017 (verified citations; text not re-read): hCG + transvaginal US. Heterotopic "
             "pregnancy (IVF) is not handled."),
    RuleOut(
        "장간막 허혈", ("급성 장간막 허혈", "mesenteric ischemia", "acute mesenteric ischemia", "장간막 동맥 폐색"), 2,
        "CT 혈관조영에서 장간막 혈관 개통·장 허혈 소견 없음(젖산·D-dimer로는 배제 불가)",
        rule_out=_is("mesenteric_cta", "normal"), confirm=_is("mesenteric_cta", "abnormal"),
        steps=(Step("mesenteric_cta", reason="의심되면 지체 없이 CT 혈관조영"),),
        citations=(G_AMI,), verification="primary", check_ids=("mesenteric_cta",),
        note="WSES 2022 rec. 4-5 (read, see protocols.py): CTA without delay; lactate/D-dimer cannot exclude."),
    RuleOut(
        "장 천공", ("위장관 천공", "소화성 궤양 천공", "궤양 천공", "장천공", "perforation", "perforated viscus",
                 "perforated peptic ulcer"), 2,
        "복부 CT에서 유리 공기·천공 소견 없음",
        rule_out=_is("abd_ct", "normal"),
        confirm=_any(_is("free_air_xr", "abnormal"),
                     _is("abd_ct", "abnormal", ("유리 공기", "자유 공기", "free air", "복강 내 공기", "pneumoperitoneum",
                                                "천공", "perforat"))),
        steps=(Step("abd_ct", reason="복막 자극 징후: 유리 공기 확인(단순 X선 음성으로는 배제 불가)"),),
        citations=(C_PERFORATED_ULCER,), verification="unverified", applies=_perforation_applies,
        note="WSES 2020 (verified citation; abstract has no imaging statement): CT as the rule-out test is reviewer "
             "knowledge (plain films miss free air). Live when peritoneal signs (rebound, guarding, rigidity) appear."),
    RuleOut(
        "뇌출혈", ("두개내 출혈", "뇌내출혈", "출혈성 뇌졸중", "intracerebral hemorrhage", "intracranial hemorrhage",
                "hemorrhagic stroke"), 2,
        "뇌 CT(또는 MRI)에서 출혈 없음",
        rule_out=_any(_is("ct_head", "normal"), _is("mri_bleed", "normal")),
        confirm=_any(_is("ct_head", "abnormal", ("출혈", "hemorrhage", "haemorrhage", "혈종", "hematoma")),
                     _is("mri_bleed", "abnormal")),
        steps=(Step("ct_head", reason="급성 신경학적 결손: 즉시 뇌 영상"),),
        citations=(G_STROKE,), verification="unverified", check_ids=("brain_imaging",),
        note="AHA/ASA 2019 (verified citation): emergent brain imaging before any reperfusion therapy; CT excluding "
             "hemorrhage is reviewer knowledge."),
    RuleOut(
        "급성 허혈성 뇌졸중", ("뇌경색", "허혈성 뇌졸중", "뇌졸중", "ischemic stroke", "acute ischemic stroke", "stroke",
                      "일과성 허혈 발작 후 조기 뇌졸중"), 2,
        "뇌 MRI 확산강조영상에서 급성 경색 없음(CT 음성만으로는 배제 불가)",
        rule_out=_is("mri_dwi", "normal"), confirm=_is("mri_dwi", "abnormal"),
        steps=(Step("ct_head", reason="출혈 먼저 배제"), Step("mri_dwi", reason="급성 경색 확인(확산강조영상)")),
        citations=(G_STROKE,), verification="unverified", check_ids=("brain_imaging",),
        applies=lambda c: _applies_checks(("brain_imaging",), c) and c.affirmed(_FOCAL),
        note="Live from the chief complaint only with a focal deficit (non-focal altered mental status: CT for "
             "hemorrhage, not a DWI-MRI demand; ours). Reviewer knowledge: early CT is insensitive for ischemia; DWI-MRI is the rule-out test (small/posterior "
             "strokes can still be DWI-negative)."),
    RuleOut(
        "저혈당", ("hypoglycemia", "hypoglycaemia", "저혈당증"), 2,
        "혈당 ≥ 70 mg/dL",
        rule_out=_is("glucose", "normal"),
        confirm=lambda c: c.probe("glucose").spans if c.probe("glucose").result == "abnormal"
        and any(_glucose_mgdl(v, "") < 54 for v in c.probe("glucose").values) else None,
        steps=(Step("glucose", reason="뇌졸중 유사 증상의 흔한 원인, 즉시 교정 가능"),),
        citations=(G_STROKE,), verification="unverified", check_ids=("glucose",),
        note="AHA/ASA 2019: glucose is the only lab required before IV alteplase (see protocols.py). 70 / 54 mg/dL are "
             "the usual level-1/level-2 hypoglycaemia thresholds (reviewer knowledge)."),
    RuleOut(
        "당뇨병성 케톤산증", ("dka", "diabetic ketoacidosis", "케톤산증", "당뇨병 케톤산증"), 2,
        "혈청/소변 케톤 음성, 또는 pH ≥ 7.30 + 중탄산 ≥ 18, 또는 혈당 < 200 + 당뇨병·SGLT2 억제제 없음",
        rule_out=_dka_out, confirm=_dka_confirm,
        steps=(Step("glucose", reason="고혈당 확인"), Step("ketone", reason="케톤(베타-하이드록시부티르산)"),
               Step("abga", reason="대사성 산증 확인")),
        citations=(C_HYPERGLYCEMIC_CRISES,), verification="secondary",
        note="ADA/EASD 2024 consensus (PubMed 39052901 verified; full text HTTP 403): BOHB >= 3.0 mmol/L supports DKA "
             "(secondary summary). Glucose >= 200 or known diabetes, pH < 7.3 and/or HCO3 < 18: reviewer knowledge. "
             "Only live from the DDx ledger (no chief-complaint protocol)."),
    RuleOut(
        "고환 염전", ("testicular torsion", "정삭 염전", "고환염전"), 2,
        "음낭 도플러 초음파에서 양측 고환 혈류 정상",
        rule_out=_is("scrotal_us", "normal"), confirm=_is("scrotal_us", "abnormal"),
        steps=(Step("scrotal_us", reason="급성 음낭 통증: 혈류 확인(지연 시 고환 괴사)"),),
        citations=(C_SCROTAL,), verification="primary",
        note="ACR AC 2019 (abstract read 2026-09-27): duplex Doppler US of the scrotum is usually appropriate as the "
             "initial imaging for acute scrotal pain. Only live from the DDx ledger."),
    RuleOut(
        "호중구감소성 발열", ("febrile neutropenia", "neutropenic fever", "발열성 호중구감소증"), 2,
        "절대 호중구 수 ≥ 1,000/μL",
        rule_out=_is("anc", "normal"), confirm=_is("anc", "abnormal"),
        steps=(Step("anc", reason="항암치료·면역저하 발열: 호중구 수"),),
        citations=(G_NEUTROPENIA,), verification="unverified", check_ids=("cbc_neutropenia",),
        note="IDSA 2010 defines neutropenia as ANC < 500 or expected to fall < 500 within 48 h (reviewer knowledge); "
             "our rule-out margin is >= 1000."),
    # ---------------- tier 3 ----------------
    RuleOut(
        "급성 심부전", ("심부전", "울혈성 심부전", "heart failure", "acute heart failure", "급성 심부전 악화"), 3,
        "BNP < 100 pg/mL 또는 NT-proBNP < 300 pg/mL",
        rule_out=_is("bnp", "normal"), confirm=_all(_is("bnp", "abnormal"), _is("cxr_edema", "abnormal")),
        steps=(Step("bnp", reason="나트륨이뇨펩티드 정상이면 급성 심부전 가능성 낮음"),),
        citations=(C_ESC_HF, G_HF), verification="unverified", check_ids=("natriuretic_peptide",),
        note="AHA/ACC/HFSA 2022: natriuretic peptides to support or exclude HF in dyspnea (see protocols.py). The acute "
             "rule-out thresholds (BNP < 100, NT-proBNP < 300 pg/mL) are ESC 2021 per reviewer knowledge (abstract "
             "has no thresholds)."),
    RuleOut(
        "마미 증후군", ("cauda equina syndrome", "마미총 증후군"), 3,
        "요추 MRI 정상, 또는 배뇨·배변 장애와 안장 감각 저하 부정 + 하지 신경학적 진찰 정상",
        rule_out=_cauda_out, confirm=_is("spine_mri", "abnormal", ("마미", "cauda equina")),
        steps=(Step("cauda_history", reason="배뇨·배변 장애, 안장 감각 저하"),
               Step("leg_neuro", reason="하지 근력·감각·반사, 항문 괄약근"),
               Step("spine_mri", when=lambda c: c.probe("cauda_history").result == "abnormal"
                    or c.probe("leg_neuro").result == "abnormal", reason="위험 신호 있으면 즉시 MRI")),
        citations=(G_LOW_BACK_PAIN,), verification="unverified", check_ids=("red_flags",),
        note="ACP/APS 2007 rec. 1 and 3 (read, see protocols.py): focused history/exam, imaging for severe or "
             "progressive deficits. The clinical rule-out pathway is our operationalisation."),
)

RULE_OUT: dict[str, RuleOut] = {r.name: r for r in RULE_OUT_TABLE}

# Critical imaging / ECG results read by agent/result_interpreter.py (present, not hedged) that confirm a can't-miss
# diagnosis. The policy keeps them in state.result_criticals (only when AGENT_USE_RESULT_INTERPRETER is on); the gate
# treats them as confirming evidence next to its own readers, and adds the danger to the checked list even when the
# chief complaint / DDx ledger did not raise it (so the one-time "confirmed_other" hint can fire). Only findings that
# are the disease itself are listed (widened mediastinum, no intrauterine pregnancy, positive blood culture are not).
CRITICAL_RESULT_DANGER: dict[str, str] = {
    "IMG:cxr_ptx": "긴장성 기흉", "IMG:ct_dissection": "대동맥 박리", "IMG:aaa_imaging": "복부 대동맥류 파열",
    "ECG:ecg_stemi": "급성 관상동맥 증후군", "IMG:ct_sah": "지주막하 출혈", "IMG:ctpa_pe": "폐색전증",
    "IMG:free_air": "장 천공", "IMG:ct_ich": "뇌출혈", "IMG:subdural_hematoma": "뇌출혈",
    "IMG:stroke_imaging": "급성 허혈성 뇌졸중", "IMG:testis_no_flow": "고환 염전",
}


# Concepts that are only a precursor of the mapped danger: they confirm it only when the reading or the report of that
# turn affirms one of these words (negation-aware). Otherwise the danger is raised (put on the checked list, so the gate
# asks for the bedside check) but not confirmed; the finding itself still reaches the doctor through the policy's
# one-time critical-result alert. A pneumothorax on imaging is not a tension pneumothorax (2026-09-29): tension is
# confirmed by the report (mediastinal / tracheal shift, "긴장성", "tension") or, as in _ptx_confirm, by the gate's
# own readers (an abnormal chest image with SBP < 90 or tension words anywhere in the environment's text).
CRITICAL_RESULT_REQUIRES: dict[str, tuple[str, ...]] = {"IMG:cxr_ptx": TENSION_PTX_MARKERS}


_RE_EN_NEG_BEFORE = re.compile(r"\b(?:no|without|not|absent|negative for|nor)\b[^.;]{0,25}$")


_RE_SHIFT_KO = re.compile(r"(?:종격동|기관)[이가은는]?\s?(?:[가-힣]{1,5}\s?){0,2}?(?:이동|전위|편위|밀려|치우)[가-힣]*")


def _affirmed_any_language(text: str, kws: tuple[str, ...]) -> bool:
    """contains_affirmed (Korean negation) plus a guard for English negation before the word ("without mediastinal
    shift", "no tension"). With the tension markers, free Korean word order ("종격동이 좌측으로 이동함") also counts."""
    text = str(text)
    if kws is TENSION_PTX_MARKERS:
        for m in _RE_SHIFT_KO.finditer(text):
            if contains_affirmed(text[max(0, m.start() - 20):m.end() + 20], (m.group(0),)):
                return True
    if not contains_affirmed(text, kws):
        return False
    for k in kws:
        for m in re.finditer(re.escape(k), text):
            if not _RE_EN_NEG_BEFORE.search(text[max(0, m.start() - 40):m.start()]) and contains_affirmed(
                    text[max(0, m.start() - 40):m.end() + 40], (k,)):
                return True
    return False


def _critical_text(state, c: dict) -> str:
    """Reading summary/label plus the environment's response of the turn the critical result came from."""
    parts = [str(c.get("summary") or ""), str(c.get("label") or "")]
    turns = getattr(state, "turns", None) or []
    i = c.get("turn")
    if isinstance(i, int) and 1 <= i <= len(turns):
        parts.append(getattr(turns[i - 1], "response", "") or "")
    return ". ".join(p for p in parts if p)


def _critical_results(state) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(confirmed, raised_only): {danger name: evidence} from the present (not hedged) critical results."""
    confirmed: dict[str, list[str]] = {}
    raised: dict[str, list[str]] = {}
    for c in getattr(state, "result_criticals", None) or []:
        concept = str(c.get("concept", ""))
        name = CRITICAL_RESULT_DANGER.get(concept)
        if not name or c.get("polarity") != "present":
            continue
        ev = f"{c.get('test', '')}: {c.get('summary') or c.get('label', '')}"[:120]
        need = CRITICAL_RESULT_REQUIRES.get(concept)
        if need and not _affirmed_any_language(_critical_text(state, c).lower(), need):
            raised.setdefault(name, []).append(ev)
        else:
            confirmed.setdefault(name, []).append(ev)
    return confirmed, {k: v for k, v in raised.items() if k not in confirmed}


def critical_result_dangers(state) -> dict[str, list[str]]:
    """{danger name: evidence} from the critical results the policy's result interpreter read in this case that
    confirm the danger (CRITICAL_RESULT_REQUIRES: findings that confirm only with extra words in the reading)."""
    return _critical_results(state)[0]


def critical_result_raised(state) -> dict[str, list[str]]:
    """{danger name: evidence} for critical results that raise a danger without confirming it (a pneumothorax on
    imaging without tension signs raises 긴장성 기흉, so the gate still asks for the bedside check)."""
    return _critical_results(state)[1]


# --------------------------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------------------------


def lookup(name: str) -> RuleOut | None:
    """Table entry for a diagnosis name (Korean/English, aliases, hedges like '의심', more specific names that contain
    an alias of >= 3 characters, e.g. '급성 하벽 심근경색' → 급성 관상동맥 증후군)."""
    if not (name or "").strip():
        return None
    for r in RULE_OUT_TABLE:
        if any(same_dx(name, a) for a in (r.name,) + r.aliases):
            return r
    core = dx_keys(name)[0]
    for r in RULE_OUT_TABLE:
        for a in (r.name,) + r.aliases:
            k = dx_keys(a)[0]
            if len(k) >= 3 and k in core and not re.search(r"(없는|아닌|배제|제외)", name):
                return r
    return None


def _applies(r: RuleOut, c: _Ctx) -> bool:
    if r.applies is not None:
        return r.applies(c)
    return _applies_checks(r.check_ids, c) if r.check_ids else True


def _status(r: RuleOut, c: _Ctx) -> tuple[str, list[str]]:
    if ev := c.critical.get(r.name):
        return "confirmed", ev
    if (ev := r.confirm(c)) is not None:
        return "confirmed", ev
    if (ev := r.rule_out(c)) is not None:
        return "ruled_out", ev
    return "unresolved", []


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def status(dx_name: str, state) -> tuple[str, list[str]]:
    """("ruled_out" | "confirmed" | "unresolved", evidence spans). Unknown diagnoses are ("unresolved", [])."""
    r = lookup(dx_name)
    if r is None:
        return "unresolved", []
    return _status(r, _Ctx(state))


def _working_dx(state) -> str | None:
    ledger = getattr(state, "ddx_ledger", None)
    live = [e for e in (ledger.ranked() if ledger else []) if e.status != "배제"]
    if live:
        return live[0].dx
    ddx = getattr(state, "ddx", None) or []
    return str(ddx[0].get("dx", "")) if ddx and isinstance(ddx[0], dict) else None


def _dangers(state, c: _Ctx, exclude: str | None) -> list[dict]:
    found: dict[str, dict] = {}
    for name in cant_miss_for(c.cc):
        r = lookup(name)
        if r and r.name not in found and _applies(r, c):
            found[r.name] = {"source": "chief_complaint", "p": 0.0}
    ledger = getattr(state, "ddx_ledger", None)
    for e in (ledger.entries if ledger else []):
        if e.status != "위험":
            continue
        r = lookup(e.dx)
        if r is None:
            continue
        if r.name in found:
            found[r.name]["source"] = "both"
            found[r.name]["p"] = max(found[r.name]["p"], e.p)
        else:
            found[r.name] = {"source": "ddx_ledger", "p": e.p}
    for name in list(c.critical) + list(c.critical_raised):
        if name in RULE_OUT and name not in found:
            found[name] = {"source": "critical_result", "p": 0.0}
    excluded = lookup(exclude) if exclude else None
    out = []
    for order, r in enumerate(RULE_OUT_TABLE):
        if r.name not in found or (excluded is not None and excluded.name == r.name):
            continue
        st, ev = _status(r, c)
        if st == "ruled_out":
            continue
        info = found[r.name]
        out.append({"dx": r.name, "tier": r.tier, "status": st, "evidence": ev, "source": info["source"],
                    "p": info["p"], "criteria": r.criteria, "citation": ", ".join(x.short for x in r.citations),
                    "_order": order})
    # urgency tier first; within a tier, dangers the model itself flagged (ledger 위험) first, then table order
    out.sort(key=lambda d: (d["tier"], d["source"] == "chief_complaint", -d["p"], d["_order"]))
    for d in out:
        del d["_order"]
    return out


def unresolved_dangers(state, exclude: str | None = "") -> list[dict]:
    """Can't-miss diagnoses not yet ruled out: from the chief complaint (protocols.cant_miss_for, when the matching
    protocol check applies) and DDx-ledger entries with status 위험; minus ruled-out ones and the working diagnosis
    (`exclude`; default "" = top live ledger entry; None = exclude nothing). Ordered by urgency.
    Each item: {dx, tier, status ('unresolved'|'confirmed'), evidence, source, p, criteria, citation}."""
    c = _Ctx(state)
    if exclude == "":
        exclude = _working_dx(state)
    return _dangers(state, c, exclude)


def _next_action(r: RuleOut, c: _Ctx, state) -> tuple[ActionType, str, str] | None:
    for s in r.steps:
        p = PROBES[s.probe]
        if s.when is not None and not s.when(c):
            continue
        o = c.probe(s.probe)
        content = s.content or p.action
        if s.repeat:
            n_results = max(sum(1 for t in c.turns if p.matches(t)), o.n_normal)
            if n_results == 0 or n_results >= s.repeat or o.result == "abnormal":
                continue
        elif o.result is not None:
            continue
        if hasattr(state, "asked") and state.asked(Action(p.kind, content)):
            continue
        return p.kind, content, f"{r.name} 배제: {s.reason} ({r.citations[0].short})"
    return None


def next_rule_out_action(danger, state) -> tuple[ActionType, str, str] | None:
    """Next rule-out action for a danger (dict from unresolved_dangers, or a name): (ActionType, Korean content,
    reason). Steps already done (by action text, or a readable result anywhere) are skipped; None when nothing is
    left."""
    name = danger.get("dx") if isinstance(danger, dict) else danger
    r = lookup(name or "")
    if r is None:
        return None
    return _next_action(r, _Ctx(state), state)


def gate(state, proposed_dx: str, remaining_turns: int, max_gate_turns: int = 3, gate_turns_used: int = 0,
         min_remaining: int = 2) -> dict:
    """Decide whether a DIAGNOSE of `proposed_dx` may go ahead.

    Returns {"allow", "danger", "action", "why", "kind", "evidence"}; kind is "none" (allowed), "rule_out" (blocked:
    do `action` first) or "confirmed_other" (allowed, but a different can't-miss diagnosis has confirming findings;
    `why` is a hint the caller may show the model once; never a block, because the confirmed danger is usually a
    manifestation of the proposed cause, e.g. Listeria -> meningitis).
    Allowed when: the gate already forced `max_gate_turns` actions in this case (`gate_turns_used`, counted by the
    caller); `remaining_turns` <= `min_remaining`; the proposed diagnosis is a confirmed can't-miss diagnosis; no
    unresolved danger has a rule-out action left. A proposed diagnosis that is itself a danger (confirmed or not) is
    never demanded to be ruled out."""
    def result(allow: bool, why: str, kind: str = "none", danger: str | None = None, action=None, evidence=None):
        return {"allow": allow, "danger": danger, "action": action, "why": why, "kind": kind,
                "evidence": evidence or []}

    if gate_turns_used >= max_gate_turns:
        return result(True, f"게이트 한도({max_gate_turns}회) 소진")
    if remaining_turns <= min_remaining:
        return result(True, f"남은 턴 {remaining_turns}개: 진단 우선")
    c = _Ctx(state)
    proposed = lookup(proposed_dx)
    if proposed is not None:
        st, ev = _status(proposed, c)
        if st == "confirmed":
            return result(True, f"제안 진단이 확진된 위험 질환({proposed.name})", danger=proposed.name, evidence=ev)
    dangers = _dangers(state, c, proposed_dx)
    warnings = [d for d in dangers if d["status"] == "confirmed"]
    for d in dangers:
        if d["status"] == "confirmed":
            continue
        r = RULE_OUT[d["dx"]]
        if (act := _next_action(r, c, state)) is not None:
            return result(False, f"{d['dx']} 미배제(배제 기준: {r.criteria})", "rule_out", d["dx"], act)
    if warnings:  # not a block: a confirmed danger is often a manifestation of the proposed, more specific cause
        w = warnings[0]
        return result(True, f"{w['dx']} 소견이 확인됨: 제안 진단({proposed_dx})이 이를 설명하는지 확인",
                      "confirmed_other", w["dx"], None, w["evidence"])
    if dangers:
        return result(True, "남은 위험 질환에 할 수 있는 배제 행동 없음: " + ", ".join(d["dx"] for d in dangers))
    return result(True, "미배제 위험 질환 없음")
