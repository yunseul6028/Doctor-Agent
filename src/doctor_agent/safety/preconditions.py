"""Pre-test safety/precondition checker. Content is owned by clinical-strategist.

Called before a TEST (or an invasive EXAM) is sent to the environment. Each rule says which requested actions it
matches, the hazard, the precondition that must be known/done first, how to tell from the case so far whether the
precondition is already satisfied (environment responses, negation-aware), the prerequisite action to do instead, the
severity and a citation.

    check(action_type, content, state) -> {"ok", "severity", "rule", "prerequisite", "why", "citation"}

- "block": clear risk context in the case AND the precondition is not yet met → do `prerequisite` first
  (prerequisite None = the requested action is contraindicated as things stand; do not send it).
- "warn": annotate only (send the action, but mention the caveat / consider the prerequisite).
- Conservative by design: blocks need an affirmed risk factor from the case text; unknown facts only warn.
- Anti-loop: a block whose prerequisite was already requested (whatever the response) is downgraded to "warn".

Case text = initial info + environment/patient responses (never the doctor's own question text, which contains
the very keywords it asks about). Clauses about relatives ("아버지가 뇌졸중") are dropped before matching risk factors.
Stdlib only, CPU only, deterministic, no network.

Verification (2026-09-27)
- Citation.verified: bibliographic data checked against PubMed E-utilities (False = not indexed in PubMed; data
  taken from the issuing body's own PDF, URL in SOURCE_URLS).
- PreconditionRule.verification: "primary" = recommendation read in the source text; "secondary" = confirmed via
  secondary summaries only (full text not accessible); what was read is stated in each rule's note.
- Considered and deliberately NOT included: carotid sinus massage after recent stroke/bruit (the 2018 ESC syncope
  guideline removed this contraindication); pregnancy screening before iodinated contrast itself (ACR Manual 2024:
  not recommended; only the pelvic radiation matters); age/hypertension alone as eGFR-screening triggers (ACR Manual
  2024: large false-positive rate).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from doctor_agent.agent.text import similarity
from doctor_agent.knowledge.clinical_rules import (
    NEG,
    POS,
    UNC,
    Citation,
    ReadText,
    age_years,
    detect_categories,
    duration_level,
    keyword_statuses,
)
from doctor_agent.safety.protocols import G_CHEST_PAIN, G_MENINGITIS

# --------------------------------------------------------------------------------------------
# Citations (bibliographic data checked against PubMed E-utilities on 2026-09-27 unless verified=False)
# --------------------------------------------------------------------------------------------

P_ABN_LP = Citation(
    "Dodd KC, Emsley HCA, Desborough MJR, Chhetri SK.",
    "Periprocedural antithrombotic management for lumbar puncture: Association of British Neurologists clinical "
    "guideline",
    "Pract Neurol", 2018, "18(6):436-446", doi="10.1136/practneurol-2017-001820", pmid="30154234",
    verified=True, short_author="ABN 요추천자 항혈전제 지침",
)
P_AABB_PLT = Citation(
    "Kaufman RM, Djulbegovic B, Gernsheimer T, et al.",
    "Platelet transfusion: a clinical practice guideline from the AABB",
    "Ann Intern Med", 2015, "162(3):205-213", doi="10.7326/M14-1589", pmid="25383671",
    verified=True, short_author="AABB 혈소판 수혈 지침",
)
P_ACR_SPR_PREG = Citation(
    "American College of Radiology; Society for Pediatric Radiology.",
    "ACR-SPR Practice Parameter for Imaging Pregnant or Potentially Pregnant Patients with Ionizing Radiation "
    "(Revised 2023, Resolution 31)",
    "Reston, VA: ACR", 2023, "practice parameter",
    verified=False, short_author="ACR-SPR 임신 가능 환자 영상 지침",
)
P_ACOG_723 = Citation(
    "American College of Obstetricians and Gynecologists' Committee on Obstetric Practice.",
    "Committee Opinion No. 723: Guidelines for Diagnostic Imaging During Pregnancy and Lactation",
    "Obstet Gynecol", 2017, "130(4):e210-e216", doi="10.1097/AOG.0000000000002355", pmid="28937575",
    verified=True, short_author="ACOG 임신 중 영상검사",
)
P_ACR_CONTRAST = Citation(
    "ACR Committee on Drugs and Contrast Media.",
    "ACR Manual on Contrast Media 2024",
    "Reston, VA: ACR", 2024, "manual",
    verified=False, short_author="ACR 조영제 매뉴얼",
)
P_ACR_NKF_ICM = Citation(
    "Davenport MS, Perazella MA, Yee J, et al.",
    "Use of Intravenous Iodinated Contrast Media in Patients with Kidney Disease: Consensus Statements from the "
    "American College of Radiology and the National Kidney Foundation",
    "Radiology", 2020, "294(3):660-668", doi="10.1148/radiol.2019192094", pmid="31961246",
    verified=True, short_author="ACR-NKF 요오드 조영제 합의",
)
P_ACR_NKF_GBCM = Citation(
    "Weinreb JC, Rodby RA, Yee J, et al.",
    "Use of Intravenous Gadolinium-based Contrast Media in Patients with Kidney Disease: Consensus Statements from "
    "the American College of Radiology and the National Kidney Foundation",
    "Radiology", 2021, "298(1):28-35", doi="10.1148/radiol.2020202903", pmid="33170103",
    verified=True, short_author="ACR-NKF 가돌리늄 합의",
)
P_MR_SAFE = Citation(
    "Greenberg TD, Hoff MN, Gilk TB, et al.",
    "ACR guidance document on MR safe practices: Updates and critical information 2019",
    "J Magn Reson Imaging", 2020, "51(2):331-338", doi="10.1002/jmri.26880", pmid="31355502",
    verified=True, short_author="ACR MR 안전 지침",
)
P_HRS_CIED = Citation(
    "Indik JH, Gimbel JR, Abe H, et al.",
    "2017 HRS expert consensus statement on magnetic resonance imaging and radiation exposure in patients with "
    "cardiovascular implantable electronic devices",
    "Heart Rhythm", 2017, "14(7):e97-e153", doi="10.1016/j.hrthm.2017.04.025", pmid="28502708",
    verified=True, short_author="HRS 이식형 심장기기 MRI 합의",
)
P_EXERCISE = Citation(
    "Fletcher GF, Ades PA, Kligfield P, et al.",
    "Exercise standards for testing and training: a scientific statement from the American Heart Association",
    "Circulation", 2013, "128(8):873-934", doi="10.1161/CIR.0b013e31829b5b44", pmid="23877260",
    verified=True, short_author="AHA 운동부하검사 표준",
)
P_ASGE_APPROPRIATE = Citation(
    "ASGE Standards of Practice Committee; Early DS, Ben-Menachem T, Decker GA, et al.",
    "Appropriate use of GI endoscopy",
    "Gastrointest Endosc", 2012, "75(6):1127-1131", doi="10.1016/j.gie.2012.01.011", pmid="22624807",
    verified=True, short_author="ASGE 내시경 적정 사용",
)
P_AARC_ABG = Citation(
    "American Association for Respiratory Care.",
    "AARC clinical practice guideline. Sampling for arterial blood gas analysis",
    "Respir Care", 1992, "37(8):913-917", pmid="10145784",
    verified=True, short_author="AARC 동맥혈 채취 지침",
)
P_RCOG_APH = Citation(
    "Royal College of Obstetricians and Gynaecologists.",
    "Antepartum Haemorrhage (Green-top Guideline No. 63)",
    "London: RCOG", 2011, "GTG 63",
    verified=False, short_author="RCOG 산전 출혈 지침",
)

# Sources not indexed in PubMed: where the text was read (full URLs in docs/licenses.md; no URLs in shipped code)
SOURCE_URLS: dict[str, str] = {
    "ACR-SPR 임신 가능 환자 영상 지침": "gravitas.acr.org PPTS docId=23",
    "ACR 조영제 매뉴얼": "acr.org Clinical-Resources Contrast-Manual (2024 edition PDF)",
    "RCOG 산전 출혈 지침": "rcog.org.uk gtg_63.pdf",
}

CITATIONS: tuple[Citation, ...] = (
    G_MENINGITIS, P_ABN_LP, P_AABB_PLT, P_ACR_SPR_PREG, P_ACOG_723, P_ACR_CONTRAST, P_ACR_NKF_ICM, P_ACR_NKF_GBCM,
    P_MR_SAFE, P_HRS_CIED, G_CHEST_PAIN, P_EXERCISE, P_ASGE_APPROPRIATE, P_AARC_ABG, P_RCOG_APH,
)

# --------------------------------------------------------------------------------------------
# Case context
# --------------------------------------------------------------------------------------------

_UNAVAILABLE = "제공되지 않습니다"
# negations that clinical_rules.negated() does not know ("항응고제는 안 먹어요", "전치태반 아님")
_RE_EXTRA_NEG = re.compile(r"(?<![가-힣])안\s*(먹|드|복용|받|맞|써|쓰|피우|마시)|아님")


_NEG_TOKENS = ("없", "않", "아니", "음성", "(-)", "no ", "not ", "denies", "denied", "without", "never", "negative",
               "absent")
_AFFIRM_BREAKS = ("고 ", "며 ", "지만", "는데", "있", "보임", "보이", "관찰", "양성", "+", " but ", " and ")
_RE_SENTENCE_END = re.compile(r"[.?!\n]")


def _list_negated(t: str, kw: str) -> bool:
    """Comma lists negated by the sentence-final verb ("발작이나 두통, 운동장애를 앓은 적은 없습니다"), which
    clinical_rules.negated() misses because it stops at the comma. Only when the negation ends the sentence
    ("없습니다", "없음"; not "통증 없는 질 출혈") and no affirmation or new clause comes first ("발작 있음, 두통 없음")."""
    idx = t.find(kw)
    if idx < 0:
        return False
    while idx >= 0:
        start = idx + len(kw)
        m = _RE_SENTENCE_END.search(t, start)
        tail = t[start:m.start() if m else len(t)]
        pos = [tail.find(n) for n in _NEG_TOKENS if n in tail]
        if not pos or len(tail) > 80:
            return False
        p = min(pos)
        rest = tail[p:].strip()
        if len(rest) > 8 or " " in rest or any(b in tail[:p] for b in _AFFIRM_BREAKS):
            return False
        idx = t.find(kw, idx + 1)
    return True


def _statuses(text: str, kw: str) -> list[str]:
    """Occurrence statuses of kw (clinical_rules.keyword_statuses: the normalisation layer, legacy fallback). A
    keyword the layer has no finding for also gets the sentence-final list negation above."""
    st = keyword_statuses(text, kw, detail=True)
    if st and not any(layer for _, layer in st) and _list_negated(str(text or "").lower(), kw):
        return [NEG if s == POS else s for s, _ in st]
    return [s for s, _ in st]


def _neg(text: str, kw: str) -> bool:
    """Denied for the patient (at least one denial, no affirmed or uncertain mention; relatives ignored)."""
    st = _statuses(text, kw)
    return NEG in st and not (POS in st or UNC in st)


def _affirmed(text: str, keywords: tuple[str, ...]) -> bool:
    """Affirmed for the patient: present (hedged too) or uncertain. Relatives' mentions never count."""
    rt = text if isinstance(text, ReadText) else ReadText(text or "")
    t = rt.lower()
    return any(k in t and any(s in (POS, UNC) for s in _statuses(rt, k)) for k in keywords)


def _self_only(text: str) -> ReadText:
    """The case text for risk-factor matching: extra negations normalised ("안 먹어요" → "아니 먹어요") for the
    legacy fallback. Relatives are no longer cut out here: the normalisation layer attributes each finding to its
    subject ("아버지가 뇌졸중" → family), and keywords without a lexicon finding get the relative-before check of
    clinical_rules.keyword_statuses (2026-09-27; the old clause cut also dropped the patient's own findings in any
    clause with an honorific "셨")."""
    return ReadText(_RE_EXTRA_NEG.sub(lambda m: "아니 " + (m.group(1) or ""), text or ""))


_RE_FEMALE = re.compile(r"여성|여자|여아|소녀|female|woman|girl|\d+\s*세\s*여|\d+\s*[/ ]?\s*f\b|\bf\s*/\s*\d+")
_RE_MALE = re.compile(r"남성|남자|남아|소년|(?<!fe)male|\bman\b|\bboy\b|\d+\s*세\s*남|\d+\s*[/ ]?\s*m\b|\bm\s*/\s*\d+")


class Ctx:
    """What the case has revealed so far (built once per check call)."""

    def __init__(self, state) -> None:
        initial = (getattr(state, "initial_info", "") or "") if state is not None else ""
        turns = list(getattr(state, "turns", []) or []) if state is not None else []
        self.initial = initial.lower()
        self.turns: list[tuple[str, str, str]] = []
        for t in turns:
            a = t.action
            typ = getattr(a.type, "value", a.type)
            self.turns.append((str(typ).upper(), (a.content or "").lower(), (t.response or "").lower()))
        responses = [r for _, _, r in self.turns if _UNAVAILABLE not in r]
        self.raw = "\n".join([self.initial, *responses])
        self.facts = _self_only(self.raw)
        self.age = age_years(initial)
        m_f, m_m = _RE_FEMALE.search(self.initial), _RE_MALE.search(self.initial)
        self.sex = "F" if m_f and (not m_m or m_f.start() <= m_m.start()) else ("M" if m_m else None)

    def affirmed(self, keywords: tuple[str, ...]) -> bool:
        return _affirmed(self.facts, keywords)

    def mentioned(self, keywords: tuple[str, ...]) -> bool:
        """Present at all in the case text (results: '크레아티닌 1.0', '혈소판 248,000')."""
        return any(k in self.raw for k in keywords)

    def requested(self, keywords: tuple[str, ...] | re.Pattern, types: tuple[str, ...] = ("TEST", "EXAM")) -> bool:
        """An earlier action of these types whose content matches (done or at least requested)."""
        for typ, content, _ in self.turns:
            if typ not in types:
                continue
            if isinstance(keywords, re.Pattern):
                if keywords.search(content):
                    return True
            elif any(k in content for k in keywords):
                return True
        return False

    def asked(self, keywords: tuple[str, ...]) -> bool:
        return self.requested(keywords, ("ASK",))


# --------------------------------------------------------------------------------------------
# Shared detectors
# --------------------------------------------------------------------------------------------

_ANTICOAG = (
    "와파린", "warfarin", "쿠마딘", "coumadin", "항응고", "anticoag", "리바록사반", "rivaroxaban", "자렐토", "xarelto",
    "아픽사반", "apixaban", "엘리퀴스", "eliquis", "에독사반", "edoxaban", "릭시아나", "lixiana", "다비가트란",
    "dabigatran", "프라닥사", "pradaxa", "헤파린", "heparin", "에녹사파린", "enoxaparin", "클렉산", "clexane", "noac",
    "doac", "혈액 희석", "피를 묽게", "피 묽게", "혈전 용해", "thrombolys",
)
_ANTIPLATELET_P2Y12 = ("클로피도그렐", "clopidogrel", "플라빅스", "plavix", "티카그렐러", "ticagrelor", "브릴린타",
                       "brilinta", "프라수그렐", "prasugrel", "에피언트")
_COAGULOPATHY = ("혈소판 감소", "혈소판감소", "thrombocytopen", "혈우병", "hemophilia", "haemophilia", "응고 장애",
                 "응고장애", "응고병", "coagulopathy", "파종성 혈관내 응고", "간경변", "간경화", "cirrhosis",
                 "폰빌레브란트", "von willebrand", "출혈 경향", "출혈경향", "bleeding disorder")
_RE_PLATELET = re.compile(r"(혈소판|platelet|plt)[^\d.;\n]{0,15}?(\d[\d,]*(?:\.\d+)?)\s*(만|k|×|x|/|천)?")
_RE_INR = re.compile(r"(?<![a-z])inr\s*[:=]?\s*(\d+(?:\.\d+)?)")
_KW_COAG_LABS = ("inr", "pt ", "pt:", "pt(", "aptt", "ptt", "프로트롬빈", "prothrombin", "응고 검사", "응고검사",
                 "coagulation")


def _bleeding_risk(ctx: Ctx) -> bool:
    return ctx.affirmed(_ANTICOAG + _ANTIPLATELET_P2Y12 + _COAGULOPATHY)


def _platelets(ctx: Ctx) -> float | None:
    vals = []
    for m in _RE_PLATELET.finditer(ctx.raw):
        n = float(m.group(2).replace(",", ""))
        unit = m.group(3) or ""
        if unit == "만":
            n *= 10000
        elif unit == "천":
            n *= 1000
        elif n < 1000:  # K/µL or ×10^9/L
            n *= 1000
        vals.append(n)
    return min(vals) if vals else None


def _inr(ctx: Ctx) -> float | None:
    vals = [float(m.group(1)) for m in _RE_INR.finditer(ctx.raw)]
    return max(vals) if vals else None


_RE_SBP = re.compile(r"(?:혈압|blood pressure|bp)\s*:?\s*(\d{2,3})\s*/")
_EMERGENCY = ("쇼크", "shock", "심정지", "cardiac arrest", "무반응", "unresponsive", "대량 출혈", "massive bleeding",
              "다발성 외상", "multiple trauma", "major trauma", "교통사고", "추락", "사고 직후", "사고로", "사고 후",
              "외상 후", "trauma")


def _emergency(ctx: Ctx) -> bool:
    """Critically urgent (ACR-SPR: imaging proceeds without determining pregnancy status)."""
    if any(int(m.group(1)) < 90 for m in _RE_SBP.finditer(ctx.raw)):
        return True
    return ctx.affirmed(_EMERGENCY)


# --- pregnancy status --------------------------------------------------------------------------------------------
_RE_PREG_WEEKS = re.compile(r"(?:임신|재태)\s*(\d+)\s*주|(\d+)\s*주\s*(?:차\s*)?(?:임신|임산)"
                            r"|(\d+)\s*weeks?\s*(?:pregnant|gestation)|gestational age\s*(?:of\s*)?(\d+)")
_KNOWN_PREGNANT = ("임신 중", "임신중", "임신했", "임산부", "산모", "pregnant", "임신 반응 양성", "임신반응 양성",
                   "임신 검사 양성", "임신검사 양성", "hcg 양성", "positive pregnancy", "positive hcg")
_PREG_TEST_KW = ("hcg", "임신 반응", "임신반응", "임신 검사", "임신검사", "임신 테스트", "pregnancy test")
_NOT_POSSIBLE = ("폐경", "postmenopaus", "menopaus", "자궁 절제", "자궁절제", "자궁 적출", "자궁적출", "hysterectomy",
                 "난관 결찰", "난관결찰", "tubal ligation", "불임 수술")
_PREMENOPAUSE = ("폐경 전", "폐경전", "폐경은 아직", "폐경이 아직", "폐경 안", "폐경이 안", "premenopaus", "perimenopaus")


def _preg_weeks(ctx: Ctx) -> int | None:
    for m in _RE_PREG_WEEKS.finditer(ctx.facts):
        if not _affirmed(ctx.facts, (m.group(0),)):
            continue
        return int(next(g for g in m.groups() if g))
    return None


def _known_pregnant(ctx: Ctx) -> bool:
    return _preg_weeks(ctx) is not None or ctx.affirmed(_KNOWN_PREGNANT)


def _pregnancy_excluded(ctx: Ctx) -> bool:
    """Pregnancy status already settled as 'not pregnant': test done/requested, not possible, or denied."""
    if ctx.requested(_PREG_TEST_KW, ("TEST",)) or ctx.mentioned(_PREG_TEST_KW):
        return True
    if ctx.affirmed(_NOT_POSSIBLE) and not any(k in ctx.facts for k in _PREMENOPAUSE):
        return True
    t = ctx.facts
    # patient denies ("임신 가능성은 없어요", "임신은 아니에요", "성관계는 없었어요"): ACR-SPR accepts a reasonable attestation
    for kw in ("임신 가능성", "임신 가능", "임신", "pregnan", "성관계", "성경험", "sexually active"):
        if kw in t and _neg(t, kw):
            return True
    return False


def _childbearing(ctx: Ctx) -> bool | None:
    """True = female 12-50; False = male / child / older; None = female with age unknown or sex unknown."""
    if ctx.sex == "M":
        return False
    if ctx.age is not None and not 12 <= ctx.age <= 50:
        return False
    if ctx.sex == "F" and ctx.age is not None:
        return True
    return None


# --------------------------------------------------------------------------------------------
# Action matchers (on lowercase action content)
# --------------------------------------------------------------------------------------------

_RE_LP = re.compile(r"요추\s*천자|척수\s*천자|허리\s*천자|뇌\s*척수액|척수액|(?<![a-z])csf(?![a-z])|lumbar puncture"
                    r"|spinal tap")
_RE_CT = re.compile(r"(?<![a-z])(ct|cta|ctpa|ccta|mdct|hrct|ldct|ctu)(?![a-z])|씨티|시티|전산화|단층\s*촬영"
                    r"|computed tomograph")
_RE_NONCONTRAST = re.compile(r"비\s*조영|무\s*조영|조영제?\s*없이|조영\s*전\b|non[- ]?contrast|without contrast"
                             r"|non[- ]?enhanc|unenhanc")
_RE_CONTRAST = re.compile(r"조영|contrast|enhanc|(?<![a-z])(cta|ctpa|ccta|ctu)(?![a-z])|혈관\s*촬영|angiogra"
                          r"|관상\s*동맥\s*ct|심장\s*ct|폐동맥\s*ct|혈관\s*ct|ct\s*혈관|cardiac ct|coronary ct")
_RE_CATH_ANGIO = re.compile(r"관상\s*동맥\s*조영|심도자|coronary angiogra|혈관\s*조영술|catheter angiogra"
                            r"|(?<![a-z])cag(?![a-z])|배설성\s*요로|(?<![a-z])ivp(?![a-z])|정맥\s*신우|요로\s*조영")
_RE_MRI = re.compile(r"(?<![a-z])(mri|mra|mrcp|mrv|mr)(?![a-z])|자기\s*공명|엠알|magnetic resonance")
_RE_XRAY = re.compile(r"x-?ray|엑스\s*레이|엑스선|x\s*선|단순\s*촬영|radiograph|(?<![a-z])(kub|xr|cxr)(?![a-z])"
                      r"|방사선\s*촬영")
_RE_FLUORO = re.compile(r"바륨|barium|대장\s*조영|위장\s*조영|위장관\s*조영|식도\s*조영|(?<![a-z])(ugi|hsg)(?![a-z])"
                        r"|enema|자궁\s*난관\s*조영|투시|fluoro")
_RE_NUCLEAR = re.compile(r"핵의학|뼈\s*스캔|골\s*스캔|bone scan|(?<![a-z])(pet|spect)(?![a-z])|신티|scintig"
                         r"|v/q|환기\s*[-·]?\s*관류|관류\s*스캔")
_RE_ABD_PELVIS = re.compile(
    r"복부|뱃|배\s|골반|하복부|상복부|신장|콩팥|요관|방광|충수|맹장|간\s|췌장|담낭|담도|대장|결장|직장|자궁|난소|요로|요추"
    r"|허리|고관절|천골|장간막|abdom|pelvi|renal|kidney|appendi|a/p|liver|pancrea|lumbar|(?<![a-z])hip|sacr|mesenter"
    r"|(?<![a-z])(kub|ctu|ivp|hsg)(?![a-z])|urogra")
_RE_OTHER_REGION = re.compile(
    r"흉부|가슴|폐|chest|thora|머리|두부|뇌|head|brain|cran|경추|목|neck|팔|다리|손|발|무릎|어깨|extrem|부비동|sinus"
    r"|안와|orbit|관상|coronary|심장|cardiac|폐동맥|pulmonary|(?<![a-z])(cxr|ctpa|ccta)(?![a-z])")
_ABD_COMPLAINT = ("복통", "복부", "배가 아", "배 아", "아랫배", "윗배", "옆구리", "골반", "abdominal", "pelvic", "flank")


def _ionising(c: str) -> bool:
    return bool(_RE_CT.search(c) or _RE_XRAY.search(c) or _RE_FLUORO.search(c) or _RE_CATH_ANGIO.search(c))


def _abd_pelvic_ionising(c: str, ctx: Ctx) -> bool:
    if _RE_MRI.search(c) and not _RE_CT.search(c):
        return False
    if _RE_FLUORO.search(c) and not _RE_OTHER_REGION.search(c):
        return True  # barium enema, HSG, UGI: the beam is on the abdomen/pelvis
    if not _ionising(c):
        return False
    if _RE_ABD_PELVIS.search(c):
        return True
    # bare "CT" / "X-ray" in an abdominal-pain case
    return not _RE_OTHER_REGION.search(c) and any(k in ctx.initial for k in _ABD_COMPLAINT)


def _iodinated(c: str) -> bool:
    if _RE_MRI.search(c) and not _RE_CT.search(c):
        return False
    if _RE_CATH_ANGIO.search(c):
        return True
    return bool(_RE_CT.search(c) and _RE_CONTRAST.search(c) and not _RE_NONCONTRAST.search(c))


def _gadolinium(c: str) -> bool:
    if not _RE_MRI.search(c) or _RE_CT.search(c) or _RE_NONCONTRAST.search(c):
        return False
    return bool(re.search(r"조영|contrast|enhanc|gadolin|가돌리늄|(?<![a-z])gd(?![a-z])", c))


# --------------------------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str  # "block" | "warn"
    why: str  # Korean
    prerequisite: tuple[str, str] | None = None


@dataclass(frozen=True)
class PreconditionRule:
    id: str
    name: str  # Korean
    action_types: tuple[str, ...]
    matches: Callable[[str, Ctx], bool]  # (lowercase content, ctx)
    evaluate: Callable[[Ctx, str], Finding | None]
    hazard: str  # Korean
    precondition: str  # Korean
    citation: Citation
    verification: str  # "primary" | "secondary"
    note: str = ""


# --- 1. LP: CT first ----------------------------------------------------------------------------------------------
_FOCAL = ("국소 신경학적 결손", "국소 신경 결손", "국소신경학적 결손", "국소 신경학적 이상", "국소 신경 징후", "편마비",
          "반신마비", "반신 마비", "한쪽 마비", "편측 마비", "편측 위약", "편측 근력", "안면 마비", "안면마비", "얼굴 마비",
          "구음장애", "구음 장애", "실어", "언어 장애", "언어장애", "시야 결손", "시야결손", "반맹", "복시", "안구 운동 장애",
          "외전 장애", "외전 마비", "주시 마비", "동공 부동", "동공부동", "동공 크기가 다", "팔 떨어짐", "회내 표류",
          "hemipar", "hemipleg", "focal neuro", "focal deficit", "facial palsy", "facial droop", "aphasia",
          "dysarthria", "visual field", "diplopia", "gaze palsy", "cranial nerve palsy", "nerve palsy", "anisocoria",
          "pronator drift", "arm drift", "leg drift")
_PAPILLEDEMA = ("유두부종", "유두 부종", "시신경유두 부종", "시신경 유두 부종", "시신경유두부종", "울혈유두", "울혈 유두",
                "papilledema", "papilloedema", "optic disc swelling", "optic disc edema", "disc edema")
_AMS = ("의식 저하", "의식저하", "의식 변화", "의식변화", "의식이 흐", "의식 혼탁", "의식이 떨어", "혼미", "기면", "혼수",
        "혼돈", "착란", "지남력 저하", "지남력 장애", "지남력이 떨어", "지남력 상실", "헛소리", "횡설수설", "불러도 반응이 없",
        "자극에 반응이 없", "통증 자극에만 반응",
        "깨우기 어려", "drowsy", "stupor", "obtund", "letharg", "confus", "disorient", "altered mental",
        "decreased consciousness", "decreased level of consciousness", "comatose")
_SEIZURE = ("경련", "발작", "seizure", "convuls")
_IMMUNO = ("hiv", "에이즈", "면역억제", "면역 억제", "면역저하", "면역 저하", "장기이식", "장기 이식", "신장 이식",
           "간 이식", "골수 이식", "이식 후", "이식을 받", "항암", "화학요법", "항암치료", "chemotherap", "transplant",
           "immunosuppress", "immunocompromis", "프레드니솔론", "prednis", "스테로이드를 장기", "장기간 스테로이드",
           "고용량 스테로이드", "호중구 감소", "neutropen")
_CNS_HX = ("뇌종양", "뇌 종양", "뇌수술", "뇌 수술", "뇌졸중", "뇌경색", "뇌출혈", "중풍", "뇌농양", "뇌 농양", "수두증",
           "뇌실 션트", "vp 션트", "뇌전이", "뇌 전이", "brain tumor", "brain mass", "stroke", "brain abscess", "hydrocephalus",
           "vp shunt", "ventriculoperitoneal", "brain metasta", "craniotomy", "mass lesion")
_RE_GCS = re.compile(r"gcs\s*[:=]?\s*(\d{1,2})")
_RE_HEAD_IMAGING = re.compile(r"(뇌|두부|머리|두개|head|brain|cran)[^.;\n]{0,8}?(ct|씨티|단층|mri|자기\s*공명|mr\b)"
                              r"|(ct|mri)[^.;\n]{0,6}?(뇌|두부|머리|head|brain)")


def _lp_ct_risks(ctx: Ctx) -> list[str]:
    found = []
    if _affirmed(ctx.facts.replace("반복시", "반복 시"), _FOCAL):
        found.append("국소 신경학적 결손")
    if ctx.affirmed(_PAPILLEDEMA):
        found.append("유두부종")
    gcs = [int(m.group(1)) for m in _RE_GCS.finditer(ctx.raw)]
    if ctx.affirmed(_AMS) or any(3 <= g < 15 for g in gcs):
        found.append("의식 변화")
    seizure_text = re.sub(r"발작성|발작적|(발작|경련)\s*(처럼|같이|같은|듯)|(근육|복부|위|배|장|다리|종아리|안면|얼굴|눈꺼풀)\s*경련"
                          r"|(서맥|빈맥|부정맥|심방세동|기침|천식|공황|불안|통증|증상|호흡곤란|쌕쌕거림)\s*발작", "", ctx.facts)
    if _affirmed(seizure_text, _SEIZURE):
        found.append("새로 생긴 경련")
    if ctx.affirmed(_IMMUNO):
        found.append("면역저하")
    if ctx.affirmed(_CNS_HX):
        found.append("중추신경계 질환 병력")
    return found


_MASS_EFFECT_SIGNS = ("국소 신경학적 결손", "유두부종", "의식 변화")


def _eval_lp_ct(ctx: Ctx, c: str) -> Finding | None:
    risks = _lp_ct_risks(ctx)
    if not risks:
        return None
    if ctx.requested(_RE_HEAD_IMAGING) or _RE_HEAD_IMAGING.search(ctx.raw):
        return None
    # IDSA Table 2 is for adults: in children only the direct signs of raised pressure/mass effect block
    child = ctx.age is not None and ctx.age < 18
    severity = "block" if not child or any(r in _MASS_EFFECT_SIGNS for r in risks) else "warn"
    return Finding(
        severity,
        f"{', '.join(risks)}이(가) 있어 요추천자 전에 뇌 CT로 종괴·뇌부종(뇌탈출 위험)을 먼저 배제해야 합니다. "
        "세균성 수막염이 의심되면 혈액배양 후 경험적 항생제를 CT보다 먼저 시작합니다.",
        ("TEST", "뇌 CT(비조영)"),
    )


# --- 2. LP: bleeding risk -----------------------------------------------------------------------------------------
def _eval_lp_coag(ctx: Ctx, c: str) -> Finding | None:
    plt, inr = _platelets(ctx), _inr(ctx)
    if plt is not None and plt < 40000:
        return Finding("block", f"혈소판 {int(plt):,}/μL로 요추천자 후 척추 혈종 위험이 큽니다(4만 미만). 혈소판 교정 전에는 "
                                "요추천자를 미루고 경험적 치료를 유지하세요.")
    if inr is not None and inr > 1.4:
        return Finding("block", f"INR {inr}로 요추천자 기준(1.4 이하)을 넘습니다. 항응고 교정 전에는 요추천자를 미루고 "
                                "경험적 치료를 유지하세요.")
    if not _bleeding_risk(ctx):
        return None
    on_drug = ctx.affirmed(_ANTICOAG + _ANTIPLATELET_P2Y12)
    coag_known = inr is not None or ctx.mentioned(_KW_COAG_LABS) or ctx.requested(_KW_COAG_LABS, ("TEST",))
    if plt is not None and (coag_known or not on_drug):
        if on_drug:
            return Finding("warn", "항응고제/P2Y12 억제제 복용 중: 마지막 복용 시각을 확인하세요(DOAC 24-48시간, "
                                   "클로피도그렐 7일 중단 권고).")
        return None
    return Finding(
        "block",
        "항응고제·항혈소판제(아스피린 단독 제외) 복용 또는 출혈 경향이 있어, 요추천자 전에 혈소판 수와 PT/INR·aPTT를 "
        "확인해야 합니다(척추 혈종 위험).",
        ("TEST", "혈소판 수 포함 일반혈액검사 및 PT/INR, aPTT"),
    )


# --- 3. Abdominal/pelvic ionising imaging: pregnancy status -------------------------------------------------------
def _eval_pregnancy_ionising(ctx: Ctx, c: str) -> Finding | None:
    cb = _childbearing(ctx)
    if cb is False:
        return None
    if _known_pregnant(ctx):
        return Finding("warn", "임신 중: 가능하면 초음파나 MRI를 먼저 고려하되, 진단에 꼭 필요한 CT/X-ray는 보류하지 "
                               "않습니다(태아 선량은 대부분 위해 역치보다 훨씬 낮음).")
    if _pregnancy_excluded(ctx):
        return None
    prereq = ("TEST", "소변 임신 반응 검사(β-hCG)")
    if cb is None:
        if ctx.sex == "F":
            return Finding("warn", "가임기 여성일 수 있습니다(나이 미상): 복부·골반 방사선 검사 전 임신 여부를 확인하세요.",
                           prereq)
        return None
    if _emergency(ctx):
        return Finding("warn", "응급 상황이므로 임신 여부 확인 없이 진행할 수 있으나, 가능하면 소변 β-hCG를 함께 "
                               "시행하고 기록하세요.", prereq)
    return Finding("block", "가임기 여성의 복부·골반 방사선 검사(CT, X-ray, 투시)는 임신 여부를 먼저 확인해야 합니다.",
                   prereq)


# --- 4. Nuclear medicine: pregnancy (history suffices) -------------------------------------------------------------
def _eval_pregnancy_nuclear(ctx: Ctx, c: str) -> Finding | None:
    if _childbearing(ctx) is False or _pregnancy_excluded(ctx) or ctx.sex != "F":
        return None
    if _known_pregnant(ctx):
        return Finding("warn", "임신 중 핵의학 검사: 대부분 태아 선량은 낮지만 방사선과와 필요성을 상의하세요.")
    return Finding("warn", "가임기 여성의 핵의학 검사: 임신 가능성(마지막 월경)을 먼저 문진하세요.",
                   ("ASK", "임신 가능성이 있나요? 마지막 월경은 언제였나요?"))


# --- 5. Iodinated contrast: prior reaction / allergy history --------------------------------------------------------
_CONTRAST_REACTION = ("조영제 알레르기", "조영제 과민", "조영제 부작용", "조영제에 알레르기", "조영제에 두드러기",
                      "조영제 맞고", "조영제를 맞고", "조영제 반응", "조영제로 인한", "contrast allergy",
                      "contrast reaction", "allergic to contrast", "reaction to contrast", "contrast-induced anaphyl")
_ALLERGY_ASK = ("알레르기", "알러지", "allerg", "약물 부작용", "약 부작용", "과민반응", "과민 반응")


def _eval_contrast_allergy(ctx: Ctx, c: str) -> Finding | None:
    if ctx.affirmed(_CONTRAST_REACTION):
        return Finding("warn", "이전 요오드 조영제 알레르기 반응: 재발 위험이 가장 큰 인자입니다. 비조영 검사나 다른 "
                               "영상으로 대체 가능한지 검토하고, 꼭 필요하면 전처치(스테로이드·항히스타민)와 응급 대비 후 "
                               "시행하세요. (갑각류·포비돈 요오드 알레르기는 위험 인자가 아님)")
    if ctx.asked(_ALLERGY_ASK) or ctx.mentioned(_ALLERGY_ASK):
        return None
    return Finding("warn", "조영제 투여 전 이전 조영제·약물 알레르기 반응 여부가 아직 확인되지 않았습니다.",
                   ("ASK", "이전에 조영제나 약물에 알레르기 반응이나 부작용이 있었던 적이 있나요?"))


# --- 6. Iodinated contrast: kidney function ------------------------------------------------------------------------
_RENAL_RF = ("만성 신부전", "만성신부전", "만성 콩팥병", "만성콩팥병", "만성 신장병", "신부전", "신장병", "콩팥병",
             "신장 질환", "신장질환", "신기능 저하", "신장 기능 저하", "ckd", "투석", "dialysis",
             "신장 이식", "kidney transplant", "renal transplant", "신장 절제", "신절제", "nephrectomy", "단백뇨",
             "albuminuria", "proteinuria", "급성 신손상", "급성 신부전", "acute kidney injury", "당뇨", "diabetes",
             "메트포르민", "metformin", "다이아벡스", "글루코파지")
_RENAL_LAB = ("크레아티닌", "creatinine", "egfr", "사구체", "신기능", "신장 기능", "콩팥 기능", "bun", "신장기능")
_RENAL_PANEL = _RENAL_LAB + ("생화학", "화학 검사", "chemistry", "bmp", "cmp", "대사 패널", "종합 대사")
_RE_EGFR = re.compile(r"(egfr|사구체\s*여과율|gfr)[^\d.;\n]{0,12}?(\d+(?:\.\d+)?)")
_SEVERE_RENAL = ("투석", "dialysis", "말기 신부전", "말기 신장", "esrd", "ckd 4", "ckd 5", "ckd stage 4", "ckd stage 5",
                 "콩팥병 4기", "콩팥병 5기", "신부전 4기", "신부전 5기", "급성 신손상", "급성 신부전", "acute kidney injury")


def _severe_renal(ctx: Ctx) -> bool:
    if any(float(m.group(2)) < 30 for m in _RE_EGFR.finditer(ctx.raw)):
        return True
    return ctx.affirmed(_SEVERE_RENAL) or bool(re.search(r"(?<![a-z])aki(?![a-z])", ctx.facts))


def _eval_contrast_renal(ctx: Ctx, c: str) -> Finding | None:
    if _severe_renal(ctx):
        return Finding("warn", "eGFR 30 미만·급성 신손상·투석: 조영제 신독성 위험. 비조영/대체 영상을 검토하고, "
                               "필요하면 생리식염수 수액 예방 후 시행하세요. 생명을 위협하는 진단에 꼭 필요하면 신기능 때문에 "
                               "보류하지 않습니다.")
    if ctx.requested(_RENAL_PANEL, ("TEST",)) or ctx.mentioned(_RENAL_LAB):
        return None
    if not ctx.affirmed(_RENAL_RF):
        return None
    return Finding("warn", "신장 질환·당뇨·메트포르민 등 위험 인자가 있어 조영 CT 전에 신기능(eGFR)을 확인하는 것이 "
                           "권고됩니다. 응급이면 결과를 기다리느라 검사를 늦추지 마세요.",
                   ("TEST", "혈청 크레아티닌·eGFR"))


# --- 7. MRI: implants / metallic foreign bodies --------------------------------------------------------------------
_CIED = ("심박동기", "심장 박동기", "인공 심박동기", "페이스메이커", "pacemaker", "제세동기", "defibrillator",
         "(?<![a-z])icd", "crt-d", "crt-p", "이식형 심장")
_IMPLANT = ("인공와우", "인공 와우", "cochlear", "신경자극기", "신경 자극기", "neurostimulat", "인슐린 펌프",
            "insulin pump", "동맥류 클립", "aneurysm clip", "금속 파편", "금속파편", "쇳조각", "쇳가루", "shrapnel",
            "총상", "bullet", "metal fragment", "metallic foreign", "금속 이물", "약물 주입 펌프")
_MRI_SCREEN_ASK = ("금속", "심박동기", "박동기", "임플란트", "implant", "pacemaker", "보형물", "인공", "수술 받", "수술한",
                   "기계", "metal", "device")
_METAL = (r"(쇳가루|철가루|쇳조각|쇠붙이|금속\s*(조각|파편|이물|가루)|파편"
          r"|metal(lic)?\s*(fragment|foreign|shaving|splinter|particle))")
_EYE = r"(눈|안구|각막|eye|orbit|안와)"
# metal actually entering the eye (not a simile like "쇠막대가 눈을 찌르는 것 같아요"; not "이물감")
_RE_ORBIT_FB = re.compile(_METAL + r"[^.;\n]{0,20}?" + _EYE + r"|" + _EYE + r"[^.;\n]{0,20}?" + _METAL
                          + r"|(용접|연마|그라인더|weld|grind)[^.;\n]{0,25}?(눈|eye)")
_RE_ORBIT_XRAY = re.compile(r"(안와|눈|orbit)[^.;\n]{0,8}?(x-?ray|엑스|x\s*선|ct|단순\s*촬영|radiograph)")


def _eval_mri_safety(ctx: Ctx, c: str) -> Finding | None:
    fb = next((m.group(0) for m in _RE_ORBIT_FB.finditer(ctx.facts) if _affirmed(ctx.facts, (m.group(0),))), None)
    if fb and not (ctx.requested(_RE_ORBIT_XRAY) or _RE_ORBIT_XRAY.search(ctx.raw)):
        return Finding("block", "눈(안와)에 금속 이물이 들어갔을 가능성이 있어, MRI 전에 안와 X-ray로 금속 이물을 먼저 "
                                "배제해야 합니다(자기장에 의한 이물 이동·실명 위험).",
                       ("TEST", "안와 X-ray(금속 이물 확인)"))
    t = ctx.facts
    if any(re.search(k, t) and _affirmed(t, (re.search(k, t).group(0),)) for k in _CIED):
        return Finding("warn", "심장 박동기/제세동기: MRI 호환(MR-conditional) 여부와 기기·리드 정보를 확인하고, "
                               "심장 기기 담당팀과 프로토콜을 정한 뒤 시행하세요.",
                       ("ASK", "심장 박동기(또는 제세동기)의 종류와 MRI 호환 여부, 삽입 시기를 알고 계신가요?"))
    if ctx.affirmed(_IMPLANT):
        return Finding("warn", "체내 금속·전자 기기(인공와우, 신경자극기, 동맥류 클립, 금속 파편 등): MRI 안전 여부를 "
                               "먼저 확인하세요.")
    if ctx.asked(_MRI_SCREEN_ASK) or ctx.mentioned(("심박동기", "pacemaker", "금속", "임플란트", "implant")):
        return None
    return Finding("warn", "MRI 전 금속·전자 기기 선별 문진이 아직 없습니다.",
                   ("ASK", "몸에 심장 박동기, 인공와우, 금속 핀·클립, 금속 파편 같은 금속이나 전자 기기가 있나요?"))


# --- 8. Gadolinium: severe kidney disease / pregnancy --------------------------------------------------------------
def _eval_gadolinium(ctx: Ctx, c: str) -> Finding | None:
    if _severe_renal(ctx):
        return Finding("warn", "eGFR 30 미만·급성 신손상·투석 환자의 가돌리늄 조영: 신원성 전신 섬유증 위험이 낮은 그룹 II "
                               "약제를 최소 용량으로 사용하고, 꼭 필요한 검사면 보류하지 않습니다.")
    if _childbearing(ctx) is not False and ctx.sex == "F":
        if _known_pregnant(ctx):
            return Finding("warn", "임신 중 가돌리늄 조영은 일상적으로 피합니다. 비조영 MRI나 초음파로 대체 가능한지 "
                                   "검토하세요.")
        if not _pregnancy_excluded(ctx) and _childbearing(ctx):
            return Finding("warn", "가임기 여성의 가돌리늄 조영 MRI: 임신 가능성을 먼저 확인하세요.",
                           ("TEST", "소변 임신 반응 검사(β-hCG)"))
    return None


# --- 9. Exercise/stress testing in acute chest pain ----------------------------------------------------------------
_RE_STRESS = re.compile(r"운동\s*부하|트레드밀|treadmill|exercise\s*(stress|test|ecg|tolerance)"
                        r"|stress\s*(test|echo|ecg|mpi|imaging)|부하\s*(검사|심전도|심초음파|심근|영상)"
                        r"|스트레스\s*(검사|심초음파)|도부타민|dobutamine|심근\s*관류|(?<![a-z])spect(?![a-z])"
                        r"|thallium|탈륨")
_ECG = ("심전도", "ecg", "ekg")
_TROPONIN = ("트로포닌", "troponin", "심근효소", "심근 효소", "심장 효소", "cardiac enzyme", "ck-mb", "tni", "tnt")
_ACS_ECG = ("st분절 상승", "st 분절 상승", "st 상승", "st상승", "st elevation", "st분절 하강", "st 분절 하강", "st 하강",
            "st하강", "st depression", "새로운 좌각차단", "new lbbb", "new left bundle")
_ABS_CONTRA = ("급성 심근경색", "acute myocardial infarction", "stemi", "불안정 협심증", "불안정형 협심증",
               "unstable angina", "대동맥 박리", "aortic dissection", "폐색전", "pulmonary embol", "심근염",
               "myocarditis", "심낭염", "pericarditis", "심내막염", "endocarditis", "중증 대동맥판 협착",
               "심한 대동맥판 협착", "severe aortic stenosis", "급성 심부전", "비대상성 심부전", "decompensated")
_RE_TROP_REF = re.compile(r"(?:참고치|참고 범위|정상|기준치?|ref\w*|uln|99)[^\d<]{0,10}<\s*(\d[\d,]*\.?\d*)"
                          r"|\(\s*<\s*(\d[\d,]*\.?\d*)")
_RE_NUM = re.compile(r"(\d[\d,]*\.?\d*)")


def _troponin_elevated(ctx: Ctx) -> bool:
    for seg in re.split(r"[;\n]|(?<!\d)\.(?!\d)", ctx.raw):
        k = next((k for k in _TROPONIN if k in seg), None)
        if not k:
            continue
        tail = seg[seg.find(k) + len(k):]
        if _affirmed(tail, ("상승", "증가", "높", "elevated", "raised", "양성", "positive", "↑")):
            return True
        ref = _RE_TROP_REF.search(tail)
        nums = _RE_NUM.findall(tail)
        if ref and nums:
            ref_v = float((ref.group(1) or ref.group(2)).replace(",", ""))
            val = float(nums[0].replace(",", ""))
            if val > ref_v and not tail.lstrip().startswith("<"):
                return True
    return False


def _acute_chest_pain(ctx: Ctx) -> bool:
    return "chest_pain" in detect_categories(ctx.initial) and duration_level(ctx.initial) < 1


def _eval_stress(ctx: Ctx, c: str) -> Finding | None:
    if ctx.affirmed(_ACS_ECG) or _troponin_elevated(ctx):
        return Finding("block", "심전도 허혈 변화 또는 트로포닌 상승이 있어 급성 관상동맥증후군이 의심됩니다. "
                                "운동·약물 부하검사는 금기이며, 심장내과 협진과 관상동맥 평가가 우선입니다.")
    if ctx.affirmed(_ABS_CONTRA):
        return Finding("block", "부하검사의 절대 금기(급성 심근경색, 불안정 협심증, 대동맥 박리, 급성 폐색전증, 급성 "
                                "심근염·심낭염, 활동성 심내막염, 중증 대동맥판 협착, 비대상성 심부전)에 해당하는 소견이 "
                                "있습니다.")
    if not _acute_chest_pain(ctx):
        return None
    if not (ctx.requested(_ECG, ("TEST",)) or ctx.mentioned(_ECG)):
        return Finding("block", "급성 흉통에서는 부하검사 전에 12유도 심전도(10분 이내)로 급성 관상동맥증후군을 먼저 "
                                "평가해야 합니다.", ("TEST", "12유도 심전도"))
    if not (ctx.requested(_TROPONIN, ("TEST",)) or ctx.mentioned(_TROPONIN)):
        return Finding("block", "급성 흉통에서는 부하검사 전에 고감도 트로포닌으로 급성 관상동맥증후군을 배제해야 합니다.",
                       ("TEST", "고감도 트로포닌"))
    return None


# --- 10. Endoscopy with suspected perforation ------------------------------------------------------------------
_RE_ENDOSCOPY = re.compile(r"내시경|endoscop|(?<![a-z])(egd|ercp)(?![a-z])|colonoscop|sigmoidoscop|결장경|식도위십이지장")
_PERFORATION = ("천공", "perforat", "free air", "유리 공기", "자유 공기", "복강 내 공기", "복강내 공기", "횡격막 아래 공기",
                "횡격막하 공기", "횡격막 하 공기", "기복증", "pneumoperitoneum", "free gas", "subdiaphragmatic air")
_PERITONITIS = ("판상 경직", "판자처럼", "board-like", "boardlike", "rigid abdomen", "복부 강직", "복벽 강직", "근성 방어",
                "guarding", "전반적 복막", "범발성 복막", "generalized peritonitis", "diffuse peritonitis", "복막 자극 징후",
                "복막자극징후", "복막염")
_RE_ABD_IMAGING = re.compile(r"(ct|씨티|x-?ray|엑스|x\s*선|단순\s*촬영|radiograph|kub)")


def _eval_endoscopy(ctx: Ctx, c: str) -> Finding | None:
    if ctx.affirmed(_PERFORATION):
        return Finding("block", "위장관 천공이 확인되었거나 의심됩니다. 내시경은 금기이며(송기로 천공 악화), 외과 협진이 "
                                "우선입니다.")
    if ctx.affirmed(_PERITONITIS) and not ctx.requested(_RE_ABD_IMAGING, ("TEST",)):
        return Finding("block", "복막염 징후(복벽 강직·근성 방어)가 있어 내시경 전에 영상으로 천공(유리 공기)을 먼저 "
                                "배제해야 합니다.", ("TEST", "복부 CT(또는 입위 흉부 X-ray)로 유리 공기 확인"))
    return None


# --- 11. Arterial puncture / ABG on anticoagulation --------------------------------------------------------------
_RE_ABG = re.compile(r"동맥혈\s*가스|동맥혈가스|(?<![a-z])abg(?![a-z])|arterial blood gas|동맥혈\s*검사|동맥\s*천자"
                     r"|arterial puncture|동맥혈\s*채취")


def _eval_abg(ctx: Ctx, c: str) -> Finding | None:
    if not _bleeding_risk(ctx):
        return None
    return Finding("warn", "항응고제 복용/응고 장애: 동맥 천자는 상대적 금기입니다. 산소포화도나 정맥혈 가스로 대신할 수 "
                           "있는지 검토하고, 시행하면 요골동맥에서 하고 충분히 압박하세요.")


# --- 12. Digital vaginal examination in antepartum bleeding ------------------------------------------------------
_RE_DVE = re.compile(r"내진|질\s*수지|수지\s*검사|digital vaginal|vaginal exam|pelvic exam|골반\s*(진찰|검사|내진)"
                     r"|자궁\s*경부\s*(촉진|내진|개대)|bimanual|양손\s*진찰|쌍합진")
_RE_SPECULUM = re.compile(r"질경|speculum|초음파|ultrasound|sono|(?<![a-z])usg?(?![a-z])")
_VAG_BLEEDING = ("질 출혈", "질출혈", "하혈", "피가 비치", "질에서 피", "밑으로 피", "아래로 피", "vaginal bleeding",
                 "bleeding per vagina", "antepartum h", "산전 출혈")
_LATE_PREG = ("임신 후기", "임신 3기", "3분기", "third trimester", "만삭")
_PREVIA = ("전치태반", "전치 태반", "placenta previa", "placenta praevia", "저위태반", "저위 태반", "low-lying placenta")
_RE_US = re.compile(r"초음파|ultrasound|sono|(?<![a-z])usg?(?![a-z])")
_PLACENTA_NORMAL = re.compile(r"태반[^.;\n]{0,20}?(정상|후벽|전벽|저부|기저부|위쪽|상부|fundal|posterior|anterior)"
                              r"|placenta[^.;\n]{0,20}?(normal|fundal|posterior|anterior|not low)")


def _eval_dve(ctx: Ctx, c: str) -> Finding | None:
    wk = _preg_weeks(ctx)
    late = (wk is not None and wk >= 20) or ctx.affirmed(_LATE_PREG)
    if not late or not ctx.affirmed(_VAG_BLEEDING):
        return None
    if ctx.affirmed(_PREVIA):
        return Finding("block", "전치태반/저위태반에서 수지 내진은 대량 출혈을 일으킬 수 있어 금기입니다. 질경 검사와 "
                                "산과 협진으로 진행하세요.")
    if _PLACENTA_NORMAL.search(ctx.raw) or ctx.requested(_RE_US):
        return None
    return Finding("block", "임신 20주 이후 질 출혈: 초음파로 전치태반을 배제하기 전에는 수지 내진을 하지 않습니다"
                            "(질경 검사는 가능).", ("TEST", "산과 초음파(태반 위치 확인)"))


RULES: tuple[PreconditionRule, ...] = (
    PreconditionRule(
        "lp_ct_first", "요추천자 전 뇌 CT", ("TEST", "EXAM"),
        lambda c, ctx: bool(_RE_LP.search(c)), _eval_lp_ct,
        hazard="두개내 종괴·뇌부종이 있을 때 요추천자 후 뇌탈출",
        precondition="국소 신경학적 결손, 유두부종, 의식 변화, 새로 생긴 경련(1주 이내), 면역저하, 중추신경계 질환 병력 중 "
                     "하나라도 있으면 요추천자 전에 뇌 CT",
        citation=G_MENINGITIS, verification="secondary",
        note="IDSA 2004 Table 2 (B-II). Full text not accessible; list confirmed via CCJM 2017 review and search "
             "summaries. The CCJM rendering also lists age >= 60 (Hasbun 2001 criteria); deliberately not a block "
             "trigger here (conservative). CT should not delay blood cultures + empiric antibiotics. Table 2 is for "
             "adults: under 18 only focal deficit / papilledema / altered consciousness block (seizure, "
             "immunocompromise, CNS history → warn), our conservative extrapolation.",
    ),
    PreconditionRule(
        "lp_coagulation", "요추천자 전 응고·혈소판 확인", ("TEST", "EXAM"),
        lambda c, ctx: bool(_RE_LP.search(c)), _eval_lp_coag,
        hazard="척추 경막외/경막하 혈종",
        precondition="항응고제·P2Y12 억제제 복용 또는 출혈 경향이 있으면 혈소판·PT/INR·aPTT 확인; 혈소판 4만 미만 또는 "
                     "INR 1.4 초과면 교정 전 요추천자 보류",
        citation=P_ABN_LP, verification="secondary",
        note="ABN 2018 journal page read through a summarising fetch tool (not the PDF itself): platelets "
             "> 40x10^9/L considered safe (20-40 individual), warfarin INR <= 1.4, DOAC 24-48 h, "
             "clopidogrel/ticagrelor/prasugrel 7 days, low-dose aspirin need not be withheld. AABB 2015 "
             "(PMID 25383671, abstract read): prophylactic platelets for elective diagnostic LP if < 50x10^9/L "
             "(weak, very-low-quality).",
    ),
    PreconditionRule(
        "pregnancy_ionising_abd_pelvis", "복부·골반 방사선 검사 전 임신 확인", ("TEST",),
        _abd_pelvic_ionising, _eval_pregnancy_ionising,
        hazard="인지하지 못한 임신에서 태아 방사선 노출",
        precondition="가임기(12-50세) 여성은 복부·골반 CT/X-ray/투시/복부 혈관조영 전에 임신 여부 확인(문진상 가능성 "
                     "없음 또는 β-hCG). 치명적 응급은 예외",
        citation=P_ACR_SPR_PREG, verification="primary",
        note="ACR-SPR 2023 PDF read: III.A.1 chest/extremity/head-neck/CT outside abdomen-pelvis need no verification; "
             "III.A.2 standard-dose CT abdomen/pelvis, abdominal/pelvic angiography and fluoroscopy may require it; "
             "history (hysterectomy, tubal ligation) may suffice; III.E patient attesting not pregnant → proceed; "
             "critically urgent → proceed without determining status. Appendix D: screening from age 12, hCG if the "
             "answer is not a clear 'no'. ACOG CO 723 (abstract): needed CT/X-ray should not be withheld in pregnancy.",
    ),
    PreconditionRule(
        "pregnancy_nuclear", "핵의학 검사 전 임신 가능성 문진", ("TEST",),
        lambda c, ctx: bool(_RE_NUCLEAR.search(c)), _eval_pregnancy_nuclear,
        hazard="태아 방사선 노출(대부분 저선량)",
        precondition="가임기 여성은 임신 가능성 문진(검사는 병력이 불충분할 때만)",
        citation=P_ACR_SPR_PREG, verification="primary",
        note="ACR-SPR 2023 III.A.2: for diagnostic nuclear medicine a clinical history that the patient cannot "
             "reasonably be pregnant is sufficient; pregnancy tests not routinely required (except I-131). Warn only.",
    ),
    PreconditionRule(
        "contrast_allergy", "요오드 조영제 전 알레르기 확인", ("TEST",),
        lambda c, ctx: _iodinated(c), _eval_contrast_allergy,
        hazard="조영제 알레르기양 반응(아나필락시스)",
        precondition="이전 조영제 반응 여부 확인; 있으면 대체 검사 또는 전처치",
        citation=P_ACR_CONTRAST, verification="primary",
        note="ACR Manual on Contrast Media 2024 PDF read ('Risk Factors for Adverse Reactions'): prior "
             "allergic-like or unknown-type reaction to the same contrast class is the greatest risk factor "
             "(~5-fold); unrelated allergies, asthma, shellfish or povidone-iodine allergy are not grounds to "
             "restrict or premedicate. "
             "Warn only (never withhold an emergency study on this basis).",
    ),
    PreconditionRule(
        "contrast_renal", "요오드 조영제 전 신기능 확인", ("TEST",),
        lambda c, ctx: _iodinated(c), _eval_contrast_renal,
        hazard="조영제 연관 급성 신손상",
        precondition="신장 질환 병력(CKD, AKI 과거력, 투석, 신장 수술, 단백뇨)·당뇨·메트포르민이 있으면 eGFR 확인; "
                     "eGFR < 30 또는 AKI면 수액 예방",
        citation=P_ACR_NKF_ICM, verification="primary",
        note="ACR-NKF 2020 (Kidney Med open-access twin PMC7525144 read): eGFR screening should be used; prophylaxis "
             "with IV saline for AKI or eGFR < 30 not on dialysis; if contrast is required for a life-threatening "
             "diagnosis it should not be withheld because of kidney function → warn only. Risk-factor list from ACR "
             "Manual 2024 (age and hypertension alone deliberately excluded: large false-positive rate).",
    ),
    PreconditionRule(
        "mri_safety", "MRI 전 금속·기기 선별", ("TEST",),
        lambda c, ctx: bool(_RE_MRI.search(c)) and not _RE_CT.search(c), _eval_mri_safety,
        hazard="강자성 이물 이동(안구 손상), 이식형 기기 오작동·발열",
        precondition="모든 환자 MRI 안전 선별 문진; 안와 금속 이물 병력이면 안와 X-ray(1-2방향); 심장 이식 기기는 HRS "
                     "합의에 따라 MR-conditional 여부 확인",
        citation=P_MR_SAFE, verification="primary",
        note="ACR MR safe practices 2019 (author PDF read): orbit radiographs recommended for all patients who sought "
             "care for orbital trauma by a metallic foreign body (1 or 2 views); CIED management deferred to HRS 2017 "
             "(Indik, PMID 28502708, bibliographic only). Block only for the orbital foreign body; the rest warn.",
    ),
    PreconditionRule(
        "gadolinium", "가돌리늄 조영 MRI 주의", ("TEST",),
        lambda c, ctx: _gadolinium(c), _eval_gadolinium,
        hazard="신원성 전신 섬유증(NSF), 태아 노출",
        precondition="eGFR < 30·AKI·투석이면 그룹 II 약제 최소 용량; 가임기 여성은 임신 선별",
        citation=P_ACR_NKF_GBCM, verification="primary",
        note="ACR-NKF 2021 (Kidney Med twin PMC7873723 read): kidney-function screening optional for group II GBCM; "
             "group II should not be withheld if harm would result. ACR Manual 2024 'Gadolinium Pregnancy Screening "
             "Statement' read: avoid routine GBCA in pregnancy; screen for unsuspected pregnancy. Warn only.",
    ),
    PreconditionRule(
        "stress_test_acs", "부하검사 전 급성 관상동맥증후군 배제", ("TEST",),
        lambda c, ctx: bool(_RE_STRESS.search(c)), _eval_stress,
        hazard="급성 관상동맥증후군·대동맥 박리 등에서 부하검사 중 심정지·사망",
        precondition="급성 흉통이면 심전도와 고감도 트로포닌으로 ACS 먼저 배제; 절대 금기(급성 MI 2일 이내, 불안정 협심증, "
                     "대동맥 박리, 급성 PE, 급성 심근염·심낭염, 심내막염, 중증 AS, 비대상성 심부전)면 시행하지 않음",
        citation=P_EXERCISE, verification="secondary",
        note="Absolute contraindications from Fletcher 2013 Table as reproduced in AAFP 2017 (Am Fam Physician "
             "96(5):293). ECG within 10 min + hs-troponin before further testing, stress testing for intermediate "
             "risk after ACS is excluded: AHA/ACC 2021 chest pain guideline (G_CHEST_PAIN, summaries).",
    ),
    PreconditionRule(
        "endoscopy_perforation", "천공 의심 시 내시경 금기", ("TEST", "EXAM"),
        lambda c, ctx: bool(_RE_ENDOSCOPY.search(c)), _eval_endoscopy,
        hazard="송기에 의한 천공 악화·긴장성 기복증",
        precondition="천공이 알려졌거나 의심되면 내시경 금기; 복막염 징후가 있으면 영상으로 유리 공기 먼저 확인",
        citation=P_ASGE_APPROPRIATE, verification="secondary",
        note="ASGE 2012 'Appropriate use of GI endoscopy': endoscopy generally contraindicated when a perforated "
             "viscus is known or suspected. Full text not accessible (403); confirmed via search summaries of the "
             "ASGE PDF.",
    ),
    PreconditionRule(
        "abg_anticoagulation", "항응고 환자의 동맥 천자", ("TEST",),
        lambda c, ctx: bool(_RE_ABG.search(c)), _eval_abg,
        hazard="천자 부위 출혈·혈종",
        precondition="응고 장애 또는 중등도 이상 항응고(아스피린 제외)는 상대적 금기: 대체 가능성 검토, 요골동맥·압박",
        citation=P_AARC_ABG, verification="primary",
        note="AARC 1992 CPG PDF read (item 5.4): coagulopathy or medium-to-high-dose anticoagulation (heparin, "
             "coumadin, thrombolytics; not necessarily aspirin) may be a relative contraindication. PubMed pages "
             "913-7; the AARC reprint says 891-897. Warn only.",
    ),
    PreconditionRule(
        "dve_antepartum_bleeding", "산전 출혈에서 수지 내진 전 초음파", ("EXAM", "TEST"),
        lambda c, ctx: bool(_RE_DVE.search(c)) and not _RE_SPECULUM.search(c), _eval_dve,
        hazard="전치태반에서 수지 내진으로 인한 대량 출혈",
        precondition="임신 20주 이후 질 출혈이면 초음파로 전치태반을 배제한 뒤 수지 내진(질경 검사는 가능)",
        citation=P_RCOG_APH, verification="primary",
        note="RCOG GTG 63 (2011) PDF read, section 7.3: if placenta praevia is possible, digital vaginal examination "
             "should not be performed until an ultrasound has excluded placenta praevia. 20 weeks is our threshold "
             "(the guideline's APH definition starts at 24 weeks; earlier cut-off is the conservative side).",
    ),
)

RULES_BY_ID: dict[str, PreconditionRule] = {r.id: r for r in RULES}

# --------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------

_OK = {"ok": True, "severity": None, "rule": None, "prerequisite": None, "why": "", "citation": ""}


def _already_requested(ctx: Ctx, prereq: tuple[str, str]) -> bool:
    """Same-type earlier action with (nearly) the same content, whatever the response (anti-loop)."""
    typ, content = prereq
    return any(t == typ and similarity(c, content) >= 0.5 for t, c, _ in ctx.turns)


def check_all(action_type, content: str, state) -> list[dict]:
    """Every rule that fires for this action (blocks first)."""
    typ = str(getattr(action_type, "value", action_type) or "").upper()
    c = (content or "").lower()
    if not c:
        return []
    ctx = Ctx(state)
    out = []
    for rule in RULES:
        if typ not in rule.action_types or not rule.matches(c, ctx):
            continue
        f = rule.evaluate(ctx, c)
        if f is None:
            continue
        severity, why = f.severity, f.why
        if severity == "block" and f.prerequisite and _already_requested(ctx, f.prerequisite):
            severity, why = "warn", why + " (선행 조치는 이미 요청됨)"
        out.append({"ok": severity != "block", "severity": severity, "rule": rule.id, "prerequisite": f.prerequisite,
                    "why": why, "citation": rule.citation.short})
    out.sort(key=lambda r: r["severity"] != "block")
    return out


def check(action_type, content: str, state) -> dict:
    """Pre-send safety check for one action. Returns the most severe finding, or an ok result.

    {"ok": bool, "severity": "block"|"warn"|None, "rule": id|None, "prerequisite": (type, content)|None,
     "why": Korean text, "citation": short citation}
    """
    hits = check_all(action_type, content, state)
    return hits[0] if hits else dict(_OK)
