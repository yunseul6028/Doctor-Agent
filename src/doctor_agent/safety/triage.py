"""Unstable-patient triage: how sick is the patient right now, and what primary-survey steps are still missing.

Content is owned by clinical-strategist. Deterministic, stdlib + the normalisation layer (doctor_agent.nlp), CPU only,
no network, nothing kept between calls (every call reads one CaseState).

    assess(state) -> {"level", "signals", "why_ko", "citation", "news2", "news2_missing", "shock_index", "qsofa",
                      "vitals", "missing", "flags", "pediatric"}
    priority_actions(state, level_or_assessment) -> [(ActionType, content_ko, reason_ko), ...]   (ABCDE order)
    render_for_prompt(state, assessment=None, actions=None) -> str (<= 250 chars)

Levels
- "unstable":   at least one critical signal (below) -> primary survey (ABCDE) before anything else.
- "concerning": at least one warning signal (NEWS2 red score / aggregate 5-6, qSOFA >= 2, shock index >= 1.0, ...).
- "stable":     no warning signal AND the core vitals are known (or were requested and are not available).
- "unknown":    no warning signal yet, but core vitals are missing -> measure them first. Missing vitals never mean
                "stable" (a 4th level on purpose; callers that only know 3 levels should treat it as "concerning"
                for prompting and ask for vitals first).

Signals come from the initial information and every environment response so far (not the doctor's questions).
Vital-sign numbers are the normalisation layer's measured values (nlp.parse with the patient's age, so children get
Fleming 2011 centiles); within the case the WORST value of each vital is used (conservative: orthostatic or repeated
readings). Past / "평소" (chronic) values and other people's values are ignored.

Thresholds and their sources (verification scale as in protocols.py: "primary" = read in the source text or
abstract on 2026-09-28; "secondary" = confirmed via secondary sources only; "ours" = our operationalisation)
- NEWS2 bands and triggers (adults >= 16 y): RCP 2017 Chart 1 (bands) and Chart 2/4 (aggregate 5-6 = urgent,
  >= 7 = emergency, 3 in one parameter = urgent ward review), read in the RCP PDFs  -> primary.
  Supplemental oxygen (+2) is added when the text says the patient is on oxygen; SpO2 Scale 2 is not used.
- qSOFA >= 2 (RR >= 22, SBP <= 100, altered mentation): Seymour 2016 abstract -> primary.
- Shock index HR/SBP strata <0.6 / 0.6-1.0 / 1.0-1.4 / >= 1.4 with rising transfusion need and mortality:
  Mutschler 2013 abstract (adult trauma) -> primary; using >= 1.0 as "warning" and >= 1.4 as "critical" outside
  trauma is ours.
- Children: HR/RR above the 99th / below the 1st centile for age (Fleming 2011, via the parser) -> primary (table);
  SIPA shock index cut-offs >1.22 (4-6 y), >1.0 (7-12 y), >0.9 (13-16 y): Acker 2015 abstract -> primary;
  hypotension SBP < 60 (0-28 d), < 70 (1-12 mo), < 70 + 2 x age (1-10 y), < 90 (> 10 y): PALS definition,
  confirmed via Sarganas 2019 (abstract discusses the PALS/ATLS cut-offs) -> secondary.
- MAP < 65 mmHg: Sepsis-3 septic-shock definition (vasopressors to keep MAP >= 65; Singer 2016 abstract) -> primary.
- SBP < 90, SpO2 < 90 %, RR >= 30 or <= 8, GCS <= 8 / responds only to pain or not at all as "critical": ours
  (NEWS2 gives these 3 points each; RR >= 30 is the CURB-65 severity item; GCS <= 8 is the usual ATLS airway
  threshold -> unverified).
- Anaphylaxis: acute skin/mucosal involvement + respiratory compromise or reduced BP/syncope = NIAID/FAAN criterion 1
  (Sampson 2006) -> primary (criteria re-used from danger_gate.py).
- Airway compromise (stridor with hypoxaemia/retractions/cyanosis, drooling with sore throat/stridor, tongue or
  throat swelling), active bleeding with instability, chest pain with instability, ongoing seizure: ours.
"""
from __future__ import annotations

import re

from doctor_agent.env.interface import ActionType
from doctor_agent.knowledge.clinical_rules import (
    C_CURB65,
    C_GBS,
    C_PECARN_FI,
    C_QSOFA,
    Citation,
    ReadText,
    contains_affirmed,
    duration_level,
    rule_age_years,
)
from doctor_agent.knowledge.diagnostic_criteria import C_SEPSIS
from doctor_agent.nlp import normalize, parse
from doctor_agent.nlp.findings import _BP, _PEDS_HR, _PEDS_RR, _peds_row, age_from_text
from doctor_agent.safety.danger_gate import C_ANAPHYLAXIS_CRITERIA
from doctor_agent.safety.protocols import G_CHEST_PAIN, G_ECTOPIC, G_SEPSIS, G_STROKE

# --------------------------------------------------------------------------------------------
# Citations (bibliographic data checked against PubMed E-utilities on 2026-09-28 unless noted)
# --------------------------------------------------------------------------------------------

# Grey literature, not in PubMed: checked on rcp.ac.uk (Chart 1 scoring system, Chart 2 thresholds and triggers,
# Chart 4 clinical response; "no copyright restriction on NEWS2", acknowledgement required).
C_NEWS2 = Citation(
    "Royal College of Physicians.",
    "National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS. "
    "Updated report of a working party",
    "London: RCP", 2017, "", verified=False, short_author="RCP NEWS2",
)
C_NEWS2_VALIDATION = Citation(
    "Pimentel MAF, Redfern OC, Gerry S, et al.",
    "A comparison of the ability of the National Early Warning Score and the National Early Warning Score 2 to "
    "identify patients at risk of in-hospital mortality: A multi-centre database study",
    "Resuscitation", 2019, "134:147-156", doi="10.1016/j.resuscitation.2018.09.026", pmid="30287355", verified=True,
)
C_SHOCK_INDEX = Citation(
    "Mutschler M, Nienaber U, Münzberg M, et al.",
    "The Shock Index revisited - a fast guide to transfusion requirement? A retrospective analysis on 21,853 "
    "patients derived from the TraumaRegister DGU",
    "Crit Care", 2013, "17(4):R172", doi="10.1186/cc12851", pmid="23938104", verified=True,
)
C_SIPA = Citation(
    "Acker SN, Ross JT, Partrick DA, Tong S, Bensard DD.",
    "Pediatric specific shock index accurately identifies severely injured children",
    "J Pediatr Surg", 2015, "50(2):331-334", doi="10.1016/j.jpedsurg.2014.08.009", pmid="25638631", verified=True,
)
C_PEDS_HYPOTENSION = Citation(
    "Sarganas G, Schaffrath Rosario A, Berger S, Neuhauser HK.",
    "An unambiguous definition of pediatric hypotension is still lacking: Gaps between two percentile-based "
    "definitions and Pediatric Advanced Life Support/Advanced Trauma Life Support guidelines",
    "J Trauma Acute Care Surg", 2019, "86(3):448-453", doi="10.1097/TA.0000000000002139", pmid="30489506",
    verified=True, short_author="PALS 저혈압 기준(Sarganas)",
)
C_FLEMING = Citation(
    "Fleming S, Thompson M, Stevens R, et al.",
    "Normal ranges of heart rate and respiratory rate in children from birth to 18 years of age: a systematic "
    "review of observational studies",
    "Lancet", 2011, "377(9770):1011-1018", doi="10.1016/S0140-6736(10)62226-X", pmid="21411136", verified=True,
)
C_GCS = Citation(
    "Teasdale G, Jennett B.",
    "Assessment of coma and impaired consciousness. A practical scale",
    "Lancet", 1974, "2(7872):81-84", doi="10.1016/s0140-6736(74)91639-0", pmid="4136544", verified=True,
)
C_BTS_OXYGEN = Citation(
    "O'Driscoll BR, Howard LS, Earis J, Mak V.",
    "BTS guideline for oxygen use in adults in healthcare and emergency settings",
    "Thorax", 2017, "72(Suppl 1):ii1-ii90", doi="10.1136/thoraxjnl-2016-209729", pmid="28507176", verified=True,
    short_author="BTS 산소 지침",
)
C_ACEP_US = Citation(
    "American College of Emergency Physicians.",
    "Ultrasound Guidelines: Emergency, Point-of-Care and Clinical Ultrasound Guidelines in Medicine",
    "Ann Emerg Med", 2017, "69(5):e27-e54", doi="10.1016/j.annemergmed.2016.08.457", pmid="28442101",
    verified=True, short_author="ACEP 초음파 지침",
)

# --------------------------------------------------------------------------------------------
# Vocabulary (lowercase, negation-aware via contains_affirmed)
# --------------------------------------------------------------------------------------------

_VITAL_OF = {"SIGN:hypotension": "sbp", "SIGN:elevated_bp": "sbp", "SIGN:tachycardia": "hr",
             "SIGN:bradycardia": "hr", "SIGN:tachypnea": "rr", "SIGN:bradypnea": "rr", "SIGN:hypoxemia": "spo2",
             "SYM:fever": "temp", "SYM:high_fever": "temp", "SIGN:hypothermia": "temp"}
_RE_SPO2_LABEL = re.compile(r"산소|spo2|sao2|sp02|o2|포화도")
_RE_GCS = re.compile(r"(?:gcs|glasgow(?:\s?coma\s?scale)?|글래스고(?:\s?혼수\s?척도)?)[^\d\n]{0,12}(\d{1,2})(?!\d)")
_RE_EVM = re.compile(r"(?<![a-z])e\s?([1-4])\s?v\s?([1-5]|t)\s?m\s?([1-6])(?![0-9])")
_RE_AVPU = re.compile(r"avpu[^a-z\n]{0,6}([avpu])(?![a-z])")
# responds only to pain / not at all (AVPU P or U); read in exam/test texts and the initial information only
_RE_PU = re.compile(r"통증\s?(?:자극)?에만\s?(?:겨우\s?)?반응|(?:모든\s?)?자극에\s?(?:전혀\s?)?반응(?:이|하지)\s?(?:없|않)"
                    r"|무반응|반응이\s?없는\s?상태|혼수(?!\s?척도)(?:\s?상태)?|혼미(?:\s?상태)?|의식(?:이)?\s?없는\s?상태"
                    r"|unresponsive|comatose|\bcoma\b|stupor|responds? only to pain")
_RE_V = re.compile(r"(?:언어|목소리|부르면|말\s?걸면|음성)\s?(?:자극)?에(?:만)?\s?(?:겨우\s?)?반응|기면|obtund|letharg")
# "약간 혼미" (translated "slightly stuporous") and a resolved episode are not AVPU P now
_RE_MILD_BEFORE = re.compile(r"(?:약간|조금|다소|살짝|경도의?)\s?$")
_RE_RESOLVED_AFTER = re.compile(r"^[^.]{0,30}?(?:였으나|었으나|했으나|이었|였다가|회복|명료해)")
_RE_RECOVERED = re.compile(r"의식(?:이|은)?\s?(?:완전히\s?)?(?:회복|돌아왔|돌아옴|명료해)|(?:명료|정상)\s?(?:으로|로)\s?회복")
_RE_RESPONDS_AFTER = re.compile(r"^[^.]{0,12}?(?:언어|말|부르면|자극)에\s?반응(?:함|하|을\s?보)")
_RE_NEG_AFTER = re.compile(r"^\s?(?:은|는|이|가)?\s?(?:없|아님|아니|않|음성|\(-\))")
_RE_ON_O2 = re.compile(r"산소\s?(?:\d+\s?l|마스크|투여|공급|치료\s?중|를?\s?(?:달고|하고|받고)|가\s?필요)|\d+\s?l(?:/min|/분)?\s?(?:의\s?)?산소|비강\s?캐뉼라|nasal cannula"
                       r"|on\s(?:\d+\s?l\s)?(?:oxygen|o2)|non-?rebreather|고유량 산소")
_RE_SEIZURE_ONGOING = re.compile(r"경련(?:이|을)?\s?(?:계속|멈추지|안\s?멈|지속|5분\s?이상|반복)|경련\s?중(?!단)|경련\s?발작\s?지속"
                                 r"|status epilepticus|간질\s?지속|뇌전증\s?지속")
_RE_FEMALE = re.compile(r"여성|여자|여아|소녀|female|woman|girl|(?<![가-힣])여(?=[\s,.)]|$)")
_RE_PREGNANT = re.compile(r"임신\s?\d+\s?주|\d+\s?주\s?(?:차\s?)?임신|임산부|pregnan")

# not "목이 붓" (neck swelling) or "목이 막히" (globus / choking feeling): not airway oedema by themselves
_KW_AIRWAY_SWELL = ("혀가 붓", "혀가 부", "혀 부종", "설 부종", "목구멍이 붓", "목구멍이 부", "인두 부종", "후두 부종",
                    "목이 조이", "숨길이 막", "tongue swelling", "throat swelling", "laryngeal edema", "throat tightness")
_KW_DROOL = ("침을 흘", "침 흘림", "침을 삼키지 못", "침을 못 삼", "drool")
_KW_MUFFLED = ("목소리가 변", "목소리가 이상", "뜨거운 감자", "hot potato", "muffled voice")
_KW_SKIN_ALLERGY = ("두드러기", "입술이 붓", "입술이 부", "눈이 붓", "얼굴이 붓", "홍조", "온몸이 가렵", "전신 가려", "hives",
                    "urticaria", "flushing", "angioedema", "혈관부종")
_KW_GI_BLEED = ("토혈", "피를 토", "피가 섞인 구토", "커피 찌꺼기", "커피색 구토", "혈변", "피가 섞인 변", "피똥", "흑색변",
                "검은 변", "짜장면 같은 변", "타르", "hematemesis", "melena", "hematochezia", "coffee-ground")
# small / intermittent haemoptysis is not instability: only large-volume wording counts
_KW_MASSIVE_HEMOPTYSIS = ("대량 객혈", "객혈을 많이", "피를 한 컵", "피를 많이 뱉", "massive hemoptysis")
_KW_HEAVY_BLEED = ("출혈이 많", "피가 많이", "피가 쏟", "출혈이 멈추지", "피가 멈추지", "지혈이 안", "패드를 흠뻑", "massive bleeding",
                   "heavy bleeding", "uncontrolled bleeding")
# a source or systemic sign of infection (not bare cough: PE / pneumothorax cough is not sepsis)
_KW_INFECTION = ("오한", "고름", "배뇨통", "소변 볼 때 아", "상처가 빨갛", "봉와직염", "폐렴", "chills", "rigor", "pus",
                 "cellulitis", "pneumonia", "dysuria")
_KW_TRAUMA = ("교통사고", "추락", "떨어졌", "넘어졌", "부딪", "맞았", "외상", "trauma", "fall", "collision", "accident")
_KW_ABD_PELVIC = ("복통", "배가 아", "아랫배", "하복부", "골반", "배꼽", "옆구리", "abdominal pain", "pelvic pain")

# action "already done" keywords (lowercase substrings of earlier EXAM/TEST contents, or results already in the texts)
_DONE = {
    "airway": ("기도", "airway", "구강", "인두", "후두", "협착음", "stridor", "heent", "두경부"),
    "vitals": ("활력", "vital", "생체 징후", "생체징후"),
    "spo2": ("산소포화", "산소 포화", "spo2", "포화도", "pulse ox", "맥박 산소"),
    "breathing": ("폐 청진", "폐청진", "흉부 청진", "흉부청진", "호흡음", "폐 진찰", "흉부 진찰", "흉부진찰", "lung", "chest exam",
                  "호흡 평가", "호흡 상태"),
    "abga": ("동맥혈", "abga", "abg", "blood gas", "가스 분석", "혈액 가스"),
    "ecg": ("심전도", "ecg", "ekg", "electrocardiogram"),
    "glucose": ("혈당", "포도당", "glucose", "bst", "blood sugar"),
    "lactate": ("젖산", "락테이트", "lactate", "lactic"),
    "blood_culture": ("혈액배양", "혈액 배양", "blood culture"),
    "cbc": ("cbc", "혈색소", "헤모글로빈", "hemoglobin", "혈구", "혈액 검사", "피검사", "blood count"),
    "hcg": ("hcg", "임신 검사", "임신검사", "임신 반응", "임신반응", "pregnancy test"),
    "pocus": ("초음파", "fast", "pocus", "ultrasound", "echo", "에코"),
    "neuro": ("신경학", "신경 검", "신경계", "의식", "gcs", "동공", "pupil", "neuro", "지남력", "mental status"),
    "skin": ("피부", "발진", "skin", "rash", "점상출혈", "자반", "전신 관찰"),
}
# result words that mean the value is already in the case text (initial information or an earlier response)
_KNOWN = {
    "glucose": re.compile(r"(?:혈당|포도당|glucose|bst)\s?[:은는이가]?\s?\d"),
    "ecg": re.compile(r"심전도|ecg|ekg|동성\s?(?:리듬|율동|빈맥|서맥)|st\s?(?:분절|상승|하강)"),
    "lactate": re.compile(r"(?:젖산|락테이트|lactate)\s?[:은는이가]?\s?\d"),
    "blood_culture": re.compile(r"혈액\s?배양|blood culture"),
    "hcg": re.compile(r"hcg|임신\s?(?:검사|반응)"),
}


# --------------------------------------------------------------------------------------------
# Reading the case
# --------------------------------------------------------------------------------------------

def _texts(state) -> list[tuple[str, str, str]]:
    """(where, text, parser source) for the initial information and every usable environment response."""
    out = [("처음 정보", state.initial_info or "", "patient")]
    for i, t in enumerate(getattr(state, "turns", []), 1):
        typ = getattr(t.action.type, "value", str(t.action.type))
        if typ == "DIAGNOSE" or not t.response or "제공되지 않습니다" in t.response:
            continue
        src = {"EXAM": "exam", "TEST": "test"}.get(typ, "patient")
        out.append((f"#{i} {typ} '{t.action.content[:20]}'", t.response, src))
    return out


def _requested(state, kws: tuple[str, ...]) -> bool:
    """An EXAM/TEST whose content names one of `kws` was already done (whatever the answer was)."""
    for t in getattr(state, "turns", []):
        typ = getattr(t.action.type, "value", str(t.action.type))
        if typ in ("EXAM", "TEST") and any(k in normalize(t.action.content) for k in kws):
            return True
    return False


def _age(initial: str) -> tuple[float | None, bool]:
    """(age in years or None, pediatric?) from the initial information."""
    m = re.match(r"\s*(\d{1,3})\s?(?:세|살)(?![가-힣])", initial or "")
    age = float(m.group(1)) if m else age_from_text(initial)
    if age is None:
        age = rule_age_years(initial)
    child_word = bool(re.search(r"아기|아이|영아|유아|신생아|소아|남아|여아|환아|어린이|infant|child|newborn|baby", initial or ""))
    return age, (age is not None and age < 16) or (age is None and child_word)


def _peds_sbp_floor(age: float | None) -> float:
    """PALS hypotension threshold (SBP below this is hypotension); unknown pediatric age -> 70 (infant floor)."""
    if age is None:
        return 70.0
    if age < 28 / 365.25:
        return 60.0
    if age < 1:
        return 70.0
    if age <= 10:
        return 70.0 + 2 * int(age)
    return 90.0


def _sipa_cut(age: float | None) -> float | None:
    if age is None or age < 4 or age >= 17:
        return None
    return 1.22 if age < 7 else 1.0 if age < 13 else 0.9


def _news2_points(name: str, v: float) -> int:
    """NEWS2 Chart 1 (RCP 2017), SpO2 Scale 1."""
    if name == "rr":
        return 3 if v <= 8 else 1 if v <= 11 else 0 if v <= 20 else 2 if v <= 24 else 3
    if name == "spo2":
        return 3 if v <= 91 else 2 if v <= 93 else 1 if v <= 95 else 0
    if name == "sbp":
        return 3 if v <= 90 else 2 if v <= 100 else 1 if v <= 110 else 0 if v <= 219 else 3
    if name == "hr":
        return 3 if v <= 40 else 1 if v <= 50 else 0 if v <= 90 else 1 if v <= 110 else 2 if v <= 130 else 3
    if name == "temp":
        return 3 if v <= 35.0 else 1 if v <= 36.0 else 0 if v <= 38.0 else 1 if v <= 39.0 else 2
    return 0


class _Case:
    """Everything triage reads from one case (built per call)."""

    def __init__(self, state):
        self.state = state
        self.initial = state.initial_info or ""
        self.age, self.peds = _age(self.initial)
        self.texts = _texts(state)
        self.all = ReadText("\n".join(t for _, t, _ in self.texts))
        self.all_low = normalize(str(self.all))
        self.clinical = ReadText("\n".join(t for w, t, s in self.texts if s in ("exam", "test") or w == "처음 정보"))
        self.acute = duration_level(self.initial) == 0
        self.vals: dict[str, list[tuple[float, str]]] = {}
        self.dbp_for: dict[float, float] = {}
        self.concepts: dict[str, list[tuple[str, str]]] = {}  # concept -> [(polarity, where)] (current, patient)
        ctx = {"age_years": self.age} if self.age is not None else None
        for where, text, src in self.texts:
            for f in parse(text, src, context=ctx):
                if f.subject != "patient" or f.temporality in ("past", "chronic"):
                    continue
                if f.concept == "SYM:hematemesis" and "피로" in f.span:
                    continue  # lexicon regex spans "피로감과 구토" (fatigue and vomiting) -> not haematemesis
                name = _VITAL_OF.get(f.concept)
                if name and f.value is not None:
                    if name == "spo2" and not _RE_SPO2_LABEL.search(f.span.lower()):
                        continue
                    self.vals.setdefault(name, []).append((f.value, where))
                    continue
                if not f.hedged and not f.hypothetical:
                    self.concepts.setdefault(f.concept, []).append((f.polarity, where))
            n = normalize(text)
            for m in _BP.finditer(n):
                self.dbp_for.setdefault(float(m.group(1)), float(m.group(2)))

    # -- values
    def worst(self, name: str) -> tuple[float, str] | None:
        vs = self.vals.get(name)
        if not vs:
            return None
        pick = min if name in ("sbp", "spo2") else max
        return pick(vs, key=lambda x: x[0])

    def lowest(self, name: str) -> tuple[float, str] | None:
        vs = self.vals.get(name)
        return min(vs, key=lambda x: x[0]) if vs else None

    def present(self, cid: str) -> bool:
        return any(p == "present" for p, _ in self.concepts.get(cid, []))

    def clinical_polarity(self, cid: str) -> set[str]:
        """Polarities of a concept in exam/test texts and the initial information (the clinician's observation)."""
        return {p for p, w in self.concepts.get(cid, []) if w == "처음 정보" or " EXAM " in w or " TEST " in w}

    def observed(self, cid: str) -> bool:
        """Present in an observation, or present in the history and not contradicted by an observation (an exam
        saying "의식 명료" overrides a history sentence the lexicon misread, e.g. "치료에 반응이 없어서")."""
        clin = self.clinical_polarity(cid)
        return "present" in clin or (self.present(cid) and "absent" not in clin)

    def absent(self, cid: str) -> bool:
        return any(p == "absent" for p, _ in self.concepts.get(cid, [])) and not self.present(cid)

    def affirmed(self, kws: tuple[str, ...], clinical_only: bool = False) -> bool:
        return contains_affirmed(self.clinical if clinical_only else self.all, kws)

    def gcs(self) -> int | None:
        best = None
        for _, text, src in self.texts:
            n = normalize(text)
            for m in _RE_GCS.finditer(n):
                v = int(m.group(1))
                if 3 <= v <= 15:
                    best = v if best is None else min(best, v)
            for m in _RE_EVM.finditer(n):
                v = int(m.group(1)) + (1 if m.group(2) == "t" else int(m.group(2))) + int(m.group(3))
                best = v if best is None else min(best, v)
        return best

    def avpu(self) -> str | None:
        """Worst AVPU level read from exam/test texts and the initial information ("P"/"U" merged as "P")."""
        worst = None
        for where, text, src in self.texts:
            if src == "patient" and where != "처음 정보":
                continue
            n = normalize(text)
            m = _RE_AVPU.search(n)
            if m and m.group(1) in "pu":
                return "P"
            for rx, lvl in ((_RE_PU, "P"), (_RE_V, "V")):
                for m in rx.finditer(n):
                    if _RE_NEG_AFTER.match(n[m.end():m.end() + 6]):
                        continue
                    after = n[m.end():m.end() + 40]
                    if lvl == "P" and (_RE_RESOLVED_AFTER.search(after) or _RE_RECOVERED.search(n, m.end())):
                        continue  # "혼미했으나 1시간 후 의식 회복": not now
                    if lvl == "P" and (_RE_MILD_BEFORE.search(n[max(0, m.start() - 8):m.start()])
                                       or _RE_RESPONDS_AFTER.search(after)):
                        worst = "V"  # "약간 혼미하나 자극에 반응함": drowsy, not AVPU P
                        continue
                    if lvl == "P":
                        return "P"
                    worst = "V"
            if m := _RE_AVPU.search(n):
                worst = worst or ("V" if m.group(1) == "v" else None)
        return worst


# --------------------------------------------------------------------------------------------
# assess()
# --------------------------------------------------------------------------------------------

def _sig(out: list, key: str, ko: str, severity: str, cite: Citation | None, where: str = "") -> None:
    if any(s["key"] == key for s in out):
        return
    out.append({"key": key, "ko": ko, "severity": severity, "where": where,
                "cite": cite.short if cite else "ours"})


def assess(state) -> dict:
    """Triage level and the signals behind it (see the module docstring for thresholds and sources)."""
    c = _Case(state)
    sig: list[dict] = []
    sbp, hr, rr = c.worst("sbp"), c.worst("hr"), c.worst("rr")
    spo2, tmax, tmin = c.worst("spo2"), c.worst("temp"), c.lowest("temp")
    dbp = c.dbp_for.get(sbp[0]) if sbp else None
    map_ = round((sbp[0] + 2 * dbp) / 3) if sbp and dbp else None
    si = round(hr[0] / sbp[0], 2) if hr and sbp and sbp[0] > 0 else None
    gcs, avpu = c.gcs(), c.avpu()

    # mental status: severe (GCS <= 8, P/U) is critical; new confusion (V, GCS 9-14, AMS finding) is a NEWS2 red score
    ams_finding = c.observed("SYM:altered_mental_status") and (c.acute or c.affirmed(("의식 저하", "혼돈", "기면"), True))
    severe_ams = (gcs is not None and gcs <= 8) or avpu == "P"
    # a documented GCS 15 is the current reading (postictal confusion that has cleared is not "new confusion")
    new_confusion = severe_ams or (gcs is not None and gcs < 15) or (gcs is None and (ams_finding or avpu == "V"))
    alert_known = c.absent("SYM:altered_mental_status") or (gcs == 15)
    if severe_ams:
        _sig(sig, "ams_severe", f"의식 저하 심함({'GCS ' + str(gcs) if gcs is not None and gcs <= 8 else '통증에만 반응/무반응'})",
             "critical", C_GCS)
    elif new_confusion:
        _sig(sig, "ams", "의식 변화/혼돈" + (f"(GCS {gcs})" if gcs is not None else ""), "warning", C_NEWS2)

    if c.peds:
        floor = _peds_sbp_floor(c.age)
        if sbp and sbp[0] < floor:
            _sig(sig, "sbp_low", f"저혈압 SBP {sbp[0]:.0f}(<{floor:.0f}, 소아 기준)", "critical", C_PEDS_HYPOTENSION, sbp[1])
        band_hr, band_rr = _peds_row(_PEDS_HR, c.age), _peds_row(_PEDS_RR, c.age)
        if hr and band_hr:
            if hr[0] > band_hr[3]:
                _sig(sig, "hr_high", f"빈맥 {hr[0]:.0f}회/분(연령 99백분위 초과)", "warning", C_FLEMING, hr[1])
            elif hr[0] < band_hr[0]:
                _sig(sig, "hr_low", f"서맥 {hr[0]:.0f}회/분(연령 1백분위 미만)", "warning", C_FLEMING, hr[1])
        if rr and band_rr:
            if rr[0] > band_rr[3]:
                _sig(sig, "rr_high", f"빈호흡 {rr[0]:.0f}회/분(연령 99백분위 초과)", "warning", C_FLEMING, rr[1])
            elif rr[0] < band_rr[0]:
                _sig(sig, "rr_low", f"서호흡 {rr[0]:.0f}회/분(연령 1백분위 미만)", "critical", C_FLEMING, rr[1])
        cut = _sipa_cut(c.age)
        if si is not None and cut is not None and si > cut:
            _sig(sig, "si_high", f"소아 쇼크지수 {si}(>{cut})", "warning", C_SIPA)
        if c.age is not None and c.age < 0.25 and tmax and tmax[0] >= 38.0:
            _sig(sig, "febrile_infant", f"생후 3개월 미만 발열 {tmax[0]:.1f}℃", "warning", C_PECARN_FI, tmax[1])
    else:
        if sbp and sbp[0] < 90:
            _sig(sig, "sbp_low", f"저혈압 SBP {sbp[0]:.0f}(<90)", "critical", C_NEWS2, sbp[1])
        if map_ is not None and map_ < 65:
            _sig(sig, "map_low", f"평균동맥압 {map_}(<65)", "critical", C_SEPSIS, sbp[1] if sbp else "")
        # SI >= 1.4 is critical with a low-normal SBP; a tachyarrhythmia with SBP > 100 (SVT 180/110) is a warning
        if si is not None and si >= 1.4 and sbp[0] <= 100:
            _sig(sig, "si_high", f"쇼크지수 {si}(≥1.4)", "critical", C_SHOCK_INDEX)
        elif si is not None and si >= 1.0:
            _sig(sig, "si_high", f"쇼크지수 {si}(≥1.0)", "warning", C_SHOCK_INDEX)
        if rr and (rr[0] >= 30 or rr[0] <= 8):
            _sig(sig, "rr_extreme", f"호흡수 {rr[0]:.0f}회/분", "critical", C_CURB65 if rr[0] >= 30 else C_NEWS2, rr[1])
    if spo2 and spo2[0] < 90:
        _sig(sig, "spo2_low", f"산소포화도 {spo2[0]:.0f}%(<90)", "critical", C_BTS_OXYGEN, spo2[1])

    # NEWS2 (adults): aggregate and single-parameter red scores
    news2, news2_missing = None, []
    if not c.peds:
        pts: dict[str, int] = {}
        for name, v in (("rr", rr), ("spo2", spo2), ("sbp", sbp), ("hr", hr)):
            if v:
                pts[name] = _news2_points(name, v[0])
            else:
                news2_missing.append(name)
        if tmax or tmin:
            pts["temp"] = max(_news2_points("temp", x[0]) for x in (tmax, tmin) if x)
        else:
            news2_missing.append("temp")
        if new_confusion:
            pts["avpu"] = 3
        elif not alert_known:
            news2_missing.append("consciousness")
        if _RE_ON_O2.search(c.all_low):
            pts["o2"] = 2
        news2 = sum(pts.values()) if len(news2_missing) < 5 else None
        names = {"rr": "호흡수", "spo2": "산소포화도", "sbp": "수축기 혈압", "hr": "맥박", "temp": "체온"}
        vals = {"rr": rr, "spo2": spo2, "sbp": sbp, "hr": hr, "temp": tmax if tmax and pts.get("temp") and
                _news2_points("temp", tmax[0]) == pts["temp"] else tmin}
        for name, p in pts.items():
            if p == 3 and name in names and vals.get(name):
                v = vals[name]
                _sig(sig, f"news2_red_{name}", f"{names[name]} {v[0]:g}(NEWS2 3점)", "warning", C_NEWS2, v[1])
        if news2 is not None and news2 >= 7:
            _sig(sig, "news2_high", f"NEWS2 {news2}점(≥7)", "critical", C_NEWS2)
        elif news2 is not None and news2 >= 5:
            _sig(sig, "news2_medium", f"NEWS2 {news2}점(5–6)", "warning", C_NEWS2)
    elif tmin and tmin[0] < 35.0:
        _sig(sig, "hypothermia", f"저체온 {tmin[0]:.1f}℃", "warning", C_NEWS2, tmin[1])

    # qSOFA (adults, suspected infection or not: reported as a warning)
    qsofa = None
    if not c.peds and sbp and rr:
        qsofa = int(sbp[0] <= 100) + int(rr[0] >= 22) + int(new_confusion)
        if qsofa >= 2:
            _sig(sig, "qsofa", f"qSOFA {qsofa}점", "warning", C_QSOFA)

    fever = bool(tmax and tmax[0] >= 38.0) or c.present("SYM:fever")
    infection = fever or bool(tmin and tmin[0] < 36.0) or c.affirmed(_KW_INFECTION)
    # "저혈압이었으며" without a number (exam/initial text): hypotension described by the clinician
    if not sbp and "present" in c.clinical_polarity("SIGN:hypotension"):
        _sig(sig, "sbp_low", "저혈압(수치 없이 기술됨)", "critical", C_NEWS2)
    if _RE_ON_O2.search(c.all_low):
        _sig(sig, "on_oxygen", "산소 투여 필요", "warning", C_NEWS2)
    circ_bad = any(s["key"] in ("sbp_low", "map_low") or (s["key"] == "si_high" and s["severity"] == "critical")
                   for s in sig) or c.present("SIGN:shock")
    if c.peds:  # children: a low-normal adult SBP is normal; SIPA / PALS hypotension only
        circ_warn = circ_bad or any(s["key"] == "si_high" for s in sig)
    else:
        circ_warn = circ_bad or (si is not None and si >= 1.0) or (sbp is not None and sbp[0] <= 100)
    resp_bad = bool(spo2 and spo2[0] <= 91) or c.present("SIGN:cyanosis") or c.present("SIGN:retractions") \
        or any(s["key"] in ("rr_extreme", "rr_low") for s in sig)
    syncope = c.present("SYM:syncope") and c.acute

    # airway
    stridor = c.present("SIGN:stridor")
    swell = c.affirmed(_KW_AIRWAY_SWELL) and c.acute
    drool = c.affirmed(_KW_DROOL) and (stridor or c.present("SYM:sore_throat") or c.affirmed(_KW_MUFFLED))
    if swell or drool or (stridor and (resp_bad or severe_ams)):
        _sig(sig, "airway", "기도 위협(" + ("혀·인두 부종" if swell else "침 흘림" if drool else "협착음+호흡부전") + ")",
             "critical", None)
    elif stridor:
        _sig(sig, "stridor", "협착음(상기도 폐쇄 가능)", "warning", None)
    if c.present("SIGN:cyanosis"):
        distress = c.present("SIGN:retractions") or any(s["key"] in ("rr_high", "rr_extreme") for s in sig) \
            or bool(rr and rr[0] >= 25)
        _sig(sig, "cyanosis", "청색증" + (" + 호흡곤란 징후" if distress else ""), "critical" if distress else "warning",
             C_BTS_OXYGEN)
    if c.present("SIGN:retractions"):
        _sig(sig, "retractions", "흉벽 함몰(호흡 일 증가)", "warning", None)

    # anaphylaxis (NIAID/FAAN criterion 1)
    # explicit wording only: the lexicon's urticaria concept also fires on translated "피부 팽진" (skin turgor)
    skin = c.acute and c.affirmed(_KW_SKIN_ALLERGY)
    resp_allergy = c.present("SYM:dyspnea") or c.present("SIGN:wheeze") or stridor or bool(spo2 and spo2[0] < 94)
    anaphylaxis = skin and (resp_allergy or circ_warn or syncope)
    if anaphylaxis:
        _sig(sig, "anaphylaxis", "아나필락시스 의심(피부·점막 + 호흡/순환 이상)", "critical", C_ANAPHYLAXIS_CRITERIA)

    # bleeding
    bleeding = c.acute and (c.affirmed(_KW_GI_BLEED + _KW_MASSIVE_HEMOPTYSIS + _KW_HEAVY_BLEED)
                            or any(c.present(x) for x in ("SYM:hematemesis", "SYM:melena", "SYM:hematochezia")))
    # vaginal bleeding counts only with instability (pregnancy bleeding with shock index >= 1: rupture / abruption)
    if c.acute and c.present("SYM:vaginal_bleeding") and (circ_warn or syncope):
        bleeding = True
    if bleeding:
        if circ_warn or syncope:
            _sig(sig, "bleeding_unstable", "활동성 출혈 + 순환 불안정", "critical", C_SHOCK_INDEX)
        else:
            _sig(sig, "bleeding", "급성 출혈(토혈·혈변·객혈 등)", "warning", C_GBS)

    # chest pain with instability
    chest = c.acute and (c.present("SYM:chest_pain") or c.present("SYM:chest_tightness"))
    if chest and (circ_bad or (si is not None and si >= 1.0 and sbp is not None and sbp[0] <= 100) or resp_bad
                  or syncope or severe_ams):
        _sig(sig, "chest_unstable", "흉통 + 혈역학/호흡 불안정", "critical", G_CHEST_PAIN)

    # occult haemorrhage: reproductive-age woman with acute abdominal/pelvic pain and SI >= 1.0 with SBP <= 100
    # (ruptured ectopic pregnancy until proven otherwise; ours, ACOG PB 193 for the ectopic work-up)
    if (not c.peds and c.acute and _RE_FEMALE.search(normalize(c.initial)) and (c.age is None or 12 <= c.age <= 50)
            and (c.affirmed(_KW_ABD_PELVIC) or c.present("SYM:abdominal_pain") or c.present("SYM:pelvic_pain"))
            and si is not None and si >= 1.0 and sbp is not None and sbp[0] <= 100):
        _sig(sig, "occult_hemorrhage", "가임기 여성 복통 + 쇼크지수 상승(복강 내 출혈 의심)", "critical", G_ECTOPIC)

    if _RE_SEIZURE_ONGOING.search(c.all_low) and c.affirmed(("경련", "seizure", "status epilepticus", "간질", "뇌전증")):
        _sig(sig, "seizure_ongoing", "경련 지속", "critical", None)
    # cardiac arrest: only when the initial information says so (a Holter "asystole" or a past arrest is not now)
    if re.search(r"심정지|심폐소생술\s?중|cpr\s?중|cardiac arrest", normalize(c.initial)):
        _sig(sig, "arrest", "심정지", "critical", None)
    if c.present("SIGN:shock"):
        _sig(sig, "shock", "쇼크 소견", "critical", C_SEPSIS)

    # completeness of the core vitals
    # SpO2 is asked for separately only when breathing/circulation matters here (a vitals request already asked
    # for it otherwise: an extra "SpO2" turn on a routine case costs efficiency)
    need_spo2 = any(s["severity"] in ("critical", "warning") for s in sig) or resp_allergy or chest \
        or new_confusion or c.present("SYM:hemoptysis") or syncope
    vitals_asked = _requested(state, _DONE["vitals"])
    core = {"sbp": sbp, "hr": hr, "rr": rr, "temp": tmax or tmin}
    missing = [k for k, v in core.items() if not v and not (vitals_asked or _requested(state, _VITAL_KW[k]))]
    if need_spo2 and not spo2 and not _requested(state, _DONE["spo2"]):
        missing.append("spo2")
    unavailable = [k for k, v in list(core.items()) + [("spo2", spo2)] if not v and k not in missing]
    for k in missing:
        _sig(sig, f"unknown_{k}", f"{_VITAL_KO[k]} 미측정", "unknown", None)

    crit = [s for s in sig if s["severity"] == "critical"]
    warn = [s for s in sig if s["severity"] == "warning"]
    if crit:
        level = "unstable"
    elif warn:
        level = "concerning"
    elif missing:
        level = "unknown"
    else:
        level = "stable"

    why = _why(level, crit or warn, missing, news2)
    cites = []
    for s in crit + warn:
        if s["cite"] != "ours" and s["cite"] not in cites:
            cites.append(s["cite"])
    # reproductive age 12-50 y (ours); an unstated adult age counts
    female_repro = bool(_RE_FEMALE.search(normalize(c.initial))) and (
        (c.age is not None and 12 <= c.age <= 50) or (c.age is None and not c.peds))
    return {
        "level": level,
        "signals": sig,
        "why_ko": why,
        "citation": "; ".join(cites[:4]),
        "news2": news2,
        "news2_missing": news2_missing,
        "shock_index": si,
        "map": map_,
        "qsofa": qsofa,
        "gcs": gcs,
        "vitals": {k: (v[0] if v else None) for k, v in (("sbp", sbp), ("hr", hr), ("rr", rr), ("spo2", spo2),
                                                          ("temp_max", tmax), ("temp_min", tmin))},
        "missing": missing,
        "unavailable": unavailable,
        "pediatric": c.peds,
        "age_years": c.age,
        "flags": {
            "ams": bool(new_confusion), "severe_ams": bool(severe_ams), "chest_pain": bool(chest),
            "syncope": bool(syncope), "fever": bool(fever),
            "sepsis_suspected": bool(infection and (any(s["key"] in ("qsofa", "news2_high", "news2_medium", "sbp_low",
                                                                      "map_low", "si_high", "ams", "ams_severe",
                                                                      "shock", "febrile_infant") for s in sig)
                                                    or circ_bad)),
            "bleeding": bool(bleeding), "anaphylaxis": bool(anaphylaxis),
            "airway": any(s["key"] in ("airway", "stridor") for s in sig) or bool(anaphylaxis),
            "respiratory": bool(resp_bad or resp_allergy or stridor),
            "circulatory": bool(circ_bad), "circulatory_warning": bool(circ_warn),
            "trauma": c.affirmed(_KW_TRAUMA) and c.acute,
            "female_reproductive": female_repro and not _RE_PREGNANT.search(c.all_low),
            "pregnant": bool(_RE_PREGNANT.search(c.all_low)),
            "abd_pelvic_pain": c.affirmed(_KW_ABD_PELVIC) or c.present("SYM:abdominal_pain")
            or c.present("SYM:pelvic_pain") or c.present("SYM:vaginal_bleeding"),
            "seizure": c.present("SYM:seizure") and c.acute,
        },
    }


_VITAL_KO = {"sbp": "혈압", "hr": "맥박", "rr": "호흡수", "temp": "체온", "spo2": "산소포화도"}
_VITAL_KW = {"sbp": ("혈압", "blood pressure", "bp"), "hr": ("맥박", "심박", "pulse", "heart rate"),
             "rr": ("호흡수", "respiratory rate"), "temp": ("체온", "temperature", "열 측정")}


def _why(level: str, main: list[dict], missing: list[str], news2: int | None) -> str:
    if level == "unknown":
        return "미측정: " + "·".join(_VITAL_KO[k] for k in missing) + " → 안정 여부 판단 불가"
    if level == "stable":
        return "측정된 활력징후·의식 상태에 경고 신호 없음" + (f"(NEWS2 {news2}점)" if news2 is not None else "")
    head = "불안정" if level == "unstable" else "주의"
    return f"{head}: " + ", ".join(s["ko"] for s in main[:3])


# --------------------------------------------------------------------------------------------
# priority_actions()
# --------------------------------------------------------------------------------------------

def _done(state, c_low: str, key: str) -> bool:
    if _requested(state, _DONE[key]):
        return True
    rx = _KNOWN.get(key)
    return bool(rx and rx.search(c_low))


def priority_actions(state, level=None) -> list[tuple[ActionType, str, str]]:
    """Primary-survey next steps (ABCDE order) not yet done, for the given level (or an assess() result).

    "stable" -> []. "unknown" -> vitals (and SpO2 when relevant). "concerning"/"unstable" -> vitals first, then the
    A/B/C/D/E steps the signals call for. Each item: (ActionType, Korean action content, Korean reason with source)."""
    res = level if isinstance(level, dict) else assess(state)
    lvl = res["level"] if isinstance(level, dict) or level is None else level
    if lvl == "stable":
        return []
    f = res["flags"]
    low = normalize("\n".join(t for _, t, _ in _texts(state)))
    out: list[tuple[ActionType, str, str]] = []

    def add(key: str, typ: ActionType, content: str, reason: str) -> None:
        if not _done(state, low, key) and all(o[1] != content for o in out):
            out.append((typ, content, reason))

    missing = res["missing"]
    if lvl == "unknown":
        if any(k in missing for k in ("sbp", "hr", "rr", "temp")):
            add("vitals", ActionType.EXAM, "활력징후(혈압·맥박·호흡수·체온·산소포화도)",
                "안정 여부 판단 전 활력징후 필요(NEWS2 항목)")
        elif "spo2" in missing:
            add("spo2", ActionType.EXAM, "산소포화도(SpO2) 측정", "호흡·순환 평가에 산소포화도 필요(NEWS2 항목)")
        return out

    # A: airway
    if f["airway"] or f["severe_ams"]:
        add("airway", ActionType.EXAM, "기도 평가(발성·협착음·구강/인두 부종·침 흘림)",
            "기도 위협 신호 → 기도 개방성 먼저 확인(ABCDE)")
    # vitals first when any core value is missing
    if any(k in missing for k in ("sbp", "hr", "rr", "temp")):
        add("vitals", ActionType.EXAM, "활력징후(혈압·맥박·호흡수·체온·산소포화도)", "중증도 판단에 활력징후 필요(RCP NEWS2)")
    elif "spo2" in missing:
        add("spo2", ActionType.EXAM, "산소포화도(SpO2) 측정", "호흡 평가에 산소포화도 필요(RCP NEWS2)")
    # B: breathing
    if f["respiratory"] or f["anaphylaxis"]:
        add("breathing", ActionType.EXAM, "호흡 평가·폐 청진(호흡음·천명·보조근 사용)", "호흡 이상 신호 → 호흡 평가(ABCDE)")
        if lvl == "unstable" and (res["vitals"]["spo2"] is not None and res["vitals"]["spo2"] < 90
                                  or any(s["key"] in ("rr_extreme", "rr_low") for s in res["signals"])):
            add("abga", ActionType.TEST, "동맥혈 가스 분석(ABGA)", "저산소/호흡부전 → 혈액가스로 환기·산증 확인(BTS 산소 지침)")
    # C: circulation
    if f["chest_pain"] or f["syncope"] or (f["circulatory"] and not res["pediatric"]) or any(
            s["key"] in ("news2_red_hr", "hr_high", "hr_low", "chest_unstable") for s in res["signals"]):
        add("ecg", ActionType.TEST, "12유도 심전도", "흉통/실신/순환 이상 → 즉시 심전도(AHA/ACC 흉통 지침: 10분 이내)")
    if f["sepsis_suspected"]:
        add("lactate", ActionType.TEST, "혈중 젖산(lactate)", "패혈증 의심 + 불안정 신호 → 젖산 측정(SSC 2021)")
        add("blood_culture", ActionType.TEST, "혈액 배양 2쌍(항생제 투여 전)", "패혈증 의심 → 항생제 전 혈액배양(SSC 2021)")
    if f["bleeding"]:
        add("cbc", ActionType.TEST, "혈액 검사(CBC·혈색소)", "급성 출혈 → 혈색소로 출혈량 평가(Blatchford 2000)")
    if f["female_reproductive"] and (f["abd_pelvic_pain"] or f["syncope"] or f["circulatory"]):
        add("hcg", ActionType.TEST, "임신 검사(β-hCG)", "가임기 여성 + 복통/실신/순환 이상 → 자궁외 임신 배제(ACOG 자궁외 임신 지침)")
    if not f["anaphylaxis"] and (f["circulatory"] or (f["circulatory_warning"]
                                                       and (f["trauma"] or f["abd_pelvic_pain"] or f["bleeding"]))):
        add("pocus", ActionType.TEST, "응급 초음파(FAST·심장 초음파)",
            "저혈압/쇼크(또는 복통·출혈 + 순환 이상) → 복강 내 출혈·심낭압전·심기능 확인(ACEP 초음파 지침)")
    # D: disability
    if f["ams"] or f["seizure"]:
        add("glucose", ActionType.TEST, "혈당 측정", "의식 변화/경련 → 저혈당 먼저 배제(AHA/ASA 뇌졸중 지침)")
        if res.get("gcs") is None:
            add("neuro", ActionType.EXAM, "의식 수준(GCS)·동공·신경학적 검진", "의식 변화 정도를 GCS로 기록(Teasdale 1974)")
    # E: exposure
    if f["fever"] and (f["ams"] or f["circulatory"]) or f["anaphylaxis"]:
        add("skin", ActionType.EXAM, "전신 피부 관찰(점상출혈·자반·두드러기)",
            "발열+의식/순환 이상 또는 아나필락시스 → 피부 소견 확인")
    return out


# --------------------------------------------------------------------------------------------
# render_for_prompt()
# --------------------------------------------------------------------------------------------

_LEVEL_KO = {"unstable": "불안정", "concerning": "주의", "stable": "안정", "unknown": "미확인"}


def render_for_prompt(state, assessment: dict | None = None, actions: list | None = None, max_chars: int = 250) -> str:
    """One short Korean line for the doctor prompt (<= max_chars). Empty string when stable with nothing to add."""
    res = assessment or assess(state)
    lvl = res["level"]
    acts = actions if actions is not None else priority_actions(state, res)
    if lvl == "stable":
        return ""
    head = f"[중증도: {_LEVEL_KO[lvl]}] "
    sigs = [s["ko"] for s in res["signals"] if s["severity"] in ("critical", "warning")][:3]
    body = ", ".join(sigs) if sigs else res["why_ko"]
    tail = (" → 우선: " + ", ".join(a[1].split("(")[0] for a in acts[:4])) if acts else ""
    if lvl == "unstable":
        tail += " (ABCDE 먼저)"
    text = head + body + tail
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text
