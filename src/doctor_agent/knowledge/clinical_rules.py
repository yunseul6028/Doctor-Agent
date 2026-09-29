"""Published clinical decision rules. Content is owned by clinical-strategist + knowledge-rag.

Each rule's formula (items, points, thresholds) comes from the original paper cited on the rule.
Formulas are facts and are not copyrightable; the Korean wording is our own (nothing copied from MDCalc etc.).
Stdlib only, CPU only, deterministic.

Verification fields
- Citation.verified: bibliographic data (authors, title, journal, year, volume, pages, DOI) checked
  against PubMed (E-utilities) or Crossref on 2026-09-25.
- Rule.verification: how the items/points/thresholds were checked.
  "primary" = against the original abstract or full text; "secondary" = core checked in the original,
  some detail only in secondary sources; "unverified" = from reviewer knowledge, original not accessible.
  Details are in Rule.note.

Applicability (2026-09-26)
- rules_for() only returns a rule when the chief complaint is inside the rule's validated population
  (Rule.applies_to): category/keyword match + requires_any + excludes_any + chronic_cutoff + min_age.
  Rule.applicability states where each condition comes from: the population sentence of the original abstract
  (PubMed, re-read 2026-09-26) or, when marked, our own proxy / reviewer knowledge.
- Duration (duration_level) and age (age_years) are parsed from the chief complaint only.
- detect_categories(): "neuro" means the acute stroke protocol (focal deficit, altered mental status, seizure),
  not syncope or isolated dizziness; "allergy" is the anaphylaxis protocol; neuro/allergy/rash are dropped for
  complaints lasting >= 2 weeks. 13 more categories were added on 2026-09-26 (syncope, palpitations,
  hemoptysis/chronic cough, jaundice, joint, back pain, rash, chronic pruritus, edema, amenorrhea/abnormal
  vaginal bleeding, fatigue, cognitive decline, psychiatric); their checks live in safety/protocols.py.

2026-09-27: 12 more rules (PECARN head <2 / >=2 y, NEXUS, Canadian C-spine, SF Syncope, Canadian Syncope Risk
Score, Glasgow-Blatchford, Kocher, Pediatric Appendicitis Score, sPESI, PECARN febrile infant, McIsaac).
Applicability gained max_age / max_age_days (pediatric rules need a stated pediatric age or a child word),
veto_any (always excludes, even over a rule keyword) and rule_age_years() ("3살", "18개월 된 아기").
Adult-only rules (min_age >= 15) are skipped for child words. detect_categories() is unchanged.
Not encoded: Ranson / Glasgow-Imrie (48-h scores; BISAP covers early severity), NIHSS (15-item exam scale,
too long for the prompt), CHA2DS2-VASc (not diagnostic), Rochester / Step-by-Step (criteria not verifiable
in accessible abstracts; PECARN febrile infant rule used instead).

2026-09-29 (verification pass): csrs -> primary (original CMAJ full text read, points derived by the paper's own
method); alvarado unverified -> secondary (points, cut-offs and bands cross-checked in three open-access papers citing
the original). The other secondary rules (add_rs, cchr, pecarn_head x2, gbs, spesi, mcisaac) were re-tried: originals
paywalled or scanned PDF behind a download check, so they stay secondary.

2026-09-27 (later): negated() / contains_affirmed() read each keyword occurrence through the clinical-finding
normalisation layer (doctor_agent.nlp; keyword_statuses(), ReadText); detect_categories() drops a category whose
keywords are all denied by the layer and adds categories from curated layer concepts. See docs/nlp.md (migration).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# Chief-complaint categories (shared with safety/protocols.py)
# --------------------------------------------------------------------------------------------

CATEGORY_NAMES: dict[str, str] = {
    "chest_pain": "흉통",
    "dyspnea": "호흡곤란",
    "headache": "두통",
    "neuro": "신경 증상",
    "fever": "발열",
    "abdominal_pain": "복통",
    "allergy": "알레르기 반응",
    # 2026-09-26 additions (most common uncovered chief complaints in data/cases_aug)
    "syncope": "실신/일과성 의식 소실",
    "palpitations": "두근거림",
    "hemoptysis_cough": "객혈/만성 기침",
    "jaundice": "황달",
    "joint": "관절 통증/부기",
    "back_pain": "요통",
    "rash": "급성 피부 발진",
    "pruritus": "만성 전신 가려움",
    "edema": "부종/거품뇨",
    "menstrual": "무월경/비정상 질출혈",
    "fatigue": "피로",
    "cognitive": "기억력·인지 저하",
    "psychiatric": "정신과적 증상(기분·행동·지각)",
    # 2026-09-27 additions
    "urticaria_chronic": "만성 두드러기(6주 이상)",
    "hearing_loss": "난청",
    "neck_mass": "경부 종괴(성인)",
    "bleeding": "출혈 경향(멍·코피·지혈 안 됨)",
    "chronic_weakness": "만성 사지 근력 저하(2주 이상)",
    "bilious_vomiting": "영아 담즙성 구토",
}

# Lowercase substrings matched against the case text (Korean + English variants).
# "neuro" = the acute stroke protocol, so it lists focal deficits, altered mental status and seizure only.
# Syncope alone, isolated dizziness, bilateral/generalized weakness or numbness do NOT open it
# (see _neuro_cooccurrence for "one side + weakness" and "dizziness + gait disturbance").
_NEURO_FOCAL: tuple[str, ...] = (
    "마비", "편마비", "반신", "구음장애", "구음 장애", "말이 어눌", "발음이 어눌", "말이 꼬", "실어", "언어장애",
    "언어 장애", "말을 못", "얼굴이 비뚤", "입이 돌아", "입꼬리가 처", "안면 마비", "안면마비", "시야 결손",
    "시야결손", "시야 장애", "시야가 좁", "시야가 가려", "반맹", "한쪽 눈이 안 보", "복시", "둘로 보", "실조",
    "보행 장애", "보행장애", "뇌졸중", "중풍", "일과성 허혈", "slurred", "aphasia", "dysarthria", "facial droop",
    "facial palsy", "hemipar", "hemipleg", "visual field", "diplopia", "ataxi", "stroke", "transient ischemic",
)
_NEURO_AMS_SEIZURE: tuple[str, ...] = (
    "의식 저하", "의식저하", "의식 변화", "의식변화", "의식이 흐", "의식 혼탁", "혼돈", "착란", "헛소리", "경련", "발작",
    "뇌전증", "seizure", "convuls", "confusion", "altered mental",
)
# Anaphylaxis/allergic reaction (acute). Deliberately not bare "알레르기" (e.g. allergic rhinitis history).
_ALLERGY: tuple[str, ...] = (
    "두드러기", "아나필락시스", "알레르기 반응", "알레르기반응", "알레르기성 반응", "혈관부종", "혈관 부종",
    "입술이 붓", "입술이 부", "혀가 붓", "혀가 부", "얼굴이 붓", "벌에 쏘", "anaphyla", "urticaria", "hives",
    "angioedema", "allergic reaction",
)
_HEMOPTYSIS: tuple[str, ...] = (
    "객혈", "피가 섞인 가래", "피 섞인 가래", "피섞인 가래", "혈담", "피가래", "기침할 때 피", "기침하면 피",
    "hemoptysis", "coughing up blood", "blood-streaked sputum",
)
_COUGH: tuple[str, ...] = ("기침", "cough")
_ITCH: tuple[str, ...] = ("가려", "가렵", "소양", "itch", "prurit")
_GENERALIZED: tuple[str, ...] = ("전신", "온몸", "몸 전체", "몸전체", "generalized", "whole body", "all over")
# Bilateral/generalized or unilateral-limb swelling (not local swelling of a joint, the neck or the vulva)
_RE_EDEMA = re.compile(
    r"(다리|발목|발등|종아리|하지|전신|몸|눈꺼풀|눈 주위|눈두덩)\S{0,3}\s?(부종|붓|부었|부어|부기)")
# 2026-09-27: keyword sets for the new categories (combinations are checked in detect_categories)
_URTICARIA: tuple[str, ...] = ("두드러기", "팽진", "urticaria", "hives", "wheal")
_HEARING: tuple[str, ...] = (
    "난청", "청력 저하", "청력저하", "청력 감소", "청력이 떨어", "청력이 나빠", "청력 소실", "잘 안 들", "잘 들리지",
    "안 들려", "안 들리", "귀가 먹", "귀가 멍", "hearing loss", "deafness", "hard of hearing",
)
# neck lump (not "손목"/"발목"; not a thyroid nodule, which the neck-mass guideline excludes)
_RE_NECK_MASS = re.compile(
    r"((?<![손발])목|경부|턱밑|턱 밑|쇄골 위|쇄골위)[^.,;]{0,12}?(종괴|혹|멍울|덩이|덩어리|림프절 비대|림프절 종대)"
    r"|neck (mass|lump|swelling)|cervical lymphadenopathy")
_BLEEDING: tuple[str, ...] = (
    "쉽게 멍", "멍이 쉽게", "멍이 잘", "멍이 자주", "멍이 많이", "코피", "잇몸 출혈", "잇몸에서 피", "지혈되지", "지혈이 안",
    "지혈이 잘 안", "지혈이 되지", "피가 멈추지", "피가 안 멈", "출혈이 멈추지", "출혈이 멎지", "출혈 경향", "출혈경향",
    "점상출혈", "점상 출혈", "자반", "easy bruising", "bruises easily", "bruising", "epistaxis", "nosebleed",
    "petechia", "purpura", "prolonged bleeding", "bleeding tendency",
)
# limb/muscle weakness (not "시력 약화", not generalized "무력감", which is fatigue)
_RE_LIMB_WEAKNESS = re.compile(
    r"(근력|근육|팔|다리|손|하지|상지|사지|어깨|허벅지|하체)\S{0,4}\s?(의\s)?(약화|저하|위약|힘이 빠|힘이 없|힘 빠)"
    r"|근위약|근력\s?(약화|저하)|(muscle|limb|leg|arm|proximal) weakness|weakness of the (arm|leg|limb)")
_BILIOUS: tuple[str, ...] = ("담즙성 구토", "담즙 구토", "담즙이 섞인 구토", "초록색 구토", "녹색 구토", "초록색 토",
                             "bilious vomit", "bilious emesis", "green vomit")
# transient/recurrent altered consciousness without focal signs: the syncope protocol (ECG), not stroke
_TRANSIENT: tuple[str, ...] = ("간헐", "일시적", "반복", "잠깐", "잠시", "episod", "intermittent", "transient")
_SEIZURE: tuple[str, ...] = ("경련", "발작", "뇌전증", "seizure", "convuls")
# leg edema as one of several acute symptoms (e.g. pneumonia, heart failure) is not the primary-edema work-up
_EDEMA_PRIMARY: tuple[str, ...] = ("거품", "단백뇨", "foamy", "proteinuria", "얼굴", "눈꺼풀", "눈 주위", "눈두덩", "전신",
                                   "몸의 부종", "몸 부종", "generalized")
_ABD_PAIN_WORDS: tuple[str, ...] = ("통증", "아프", "아파", "아픔", "아픈", "쓰림", "쓰려", "불편", "쥐어", "pain", "ache",
                                    "tender", "cramp")
_ABD_LOCATION_ONLY: tuple[str, ...] = ("명치", "윗배", "아랫배", "옆구리", "상복부", "하복부", "epigastric")
_JOINT_TRAUMA: tuple[str, ...] = ("사고", "외상", "넘어", "부딪", "골절", "탈구", "다친", "다쳤", "삐", "찰과상", "열상",
                             "타박", "상처", "trauma", "injur", "fall", "fracture", "sprain")

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chest_pain": (
        "흉통", "가슴 통증", "가슴통증", "가슴이 아", "가슴 아", "가슴이 답답", "가슴 답답", "가슴이 조이",
        "가슴을 쥐어", "가슴이 쥐어", "가슴이 뻐근", "가슴이 찢", "가슴이 짓눌", "가슴이 타", "흉골 뒤", "흉골하",
        "chest pain", "chest tightness", "chest discomfort", "retrosternal",
    ),
    "dyspnea": (
        "호흡곤란", "숨이 차", "숨차", "숨이 참", "숨찬", "숨참", "숨쉬기", "숨이 막", "숨 가쁨", "숨가쁨", "숨이 가쁘",
        "dyspnea", "dyspnoea", "shortness of breath", "short of breath", "breathless",
    ),
    "headache": ("두통", "머리가 아", "머리 아", "머리가 깨질", "머리가 터질", "headache"),
    "neuro": _NEURO_FOCAL + _NEURO_AMS_SEIZURE,
    "fever": (
        "발열", "열이", "열나", "열감", "열과", "오한", "고열", "미열", "fever", "febrile", "chills",
    ),
    "abdominal_pain": (
        "복통", "배가 아", "배 아", "배아픔", "배 통증", "배가 쥐어", "명치", "윗배", "아랫배", "옆구리", "상복부",
        "하복부", "abdominal pain", "belly pain", "stomach ache", "stomachache", "epigastric",
    ),
    "allergy": _ALLERGY,
    # --- 2026-09-26 additions. Some categories need more than a keyword; see detect_categories(). ---
    "syncope": (
        "실신", "기절", "의식 소실", "의식소실", "의식을 잃", "의식 잃", "의식 상실", "정신을 잃", "쓰러졌", "쓰러짐",
        "syncope", "faint", "passed out", "loss of consciousness",
    ),
    "palpitations": ("두근", "심계항진", "가슴이 뛰", "심장이 뛰", "심장이 빨리", "palpitation", "racing heart"),
    "hemoptysis_cough": _HEMOPTYSIS,  # + chronic cough (>= 2 weeks), added in detect_categories
    "jaundice": ("황달", "노랗", "노래졌", "노래지", "노란 변색", "jaundice", "icter", "yellowing"),
    "joint": (
        "관절", "무릎", "손목", "발목", "어깨", "팔꿈치", "고관절", "엄지발가락", "중족지", "통풍", "joint", "knee",
        "ankle", "wrist", "shoulder", "elbow", "arthr", "gout",
    ),
    "back_pain": ("요통", "허리 통증", "허리통증", "허리가 아", "허리 아", "천골", "low back pain", "lumbago"),
    "rash": ("발진", "물집", "수포", "홍반", "rash", "blister", "bullous", "exanthem"),
    "pruritus": _ITCH,  # kept only when generalized and >= 2 weeks, see detect_categories
    "edema": ("거품 소변", "거품뇨", "단백뇨", "foamy urine", "proteinuria", "leg swelling", "edema", "oedema"),
    "menstrual": (
        "무월경", "월경이 없", "생리가 없", "생리를 안", "생리가 안 ", "생리를 하지", "월경을 하지", "생리가 늦",
        "초경", "질출혈", "질 출혈", "부정출혈", "부정 출혈", "폐경 후 출혈", "amenorrh", "missed period",
        "vaginal bleeding", "postmenopausal bleeding",
    ),
    "fatigue": ("피로", "피곤", "무력감", "기운이 없", "쇠약감", "fatigue", "tiredness", "exhaustion"),
    "cognitive": (
        "기억력", "기억 장애", "기억장애", "건망", "치매", "인지 저하", "인지기능", "인지 기능", "깜빡", "memory",
        "dementia", "cognitive decline", "forgetful",
    ),
    "psychiatric": (
        "우울", "기분", "이상행동", "이상 행동", "행동 문제", "공격적", "환청", "환시", "환각", "망상", "초조", "경조증",
        "자살", "자해", "불안감", "불안해", "공황", "depress", "suicid", "psychos", "hallucinat", "delusion",
        "mania", "anxiety", "agitation",
    ),
    # 2026-09-27 additions: all need more than a keyword, see detect_categories()
    "urticaria_chronic": (),
    "hearing_loss": _HEARING,
    "neck_mass": (),
    "bleeding": _BLEEDING,
    "chronic_weakness": (),
    "bilious_vomiting": (),
}

# Phrases blanked out before keyword matching (false friends of a keyword).
_MASKS: tuple[str, ...] = ("발작적", "발작성", "paroxysmal")

# Categories whose protocol is acute-only: dropped for a chief complaint lasting >= 2 weeks (duration_level >= 1)
# unless acute markers are present. Chronic focal deficits need imaging too, but not the stroke/reperfusion
# protocol (onset time, emergent CT) that this category scores.
ACUTE_ONLY_CATEGORIES: frozenset[str] = frozenset({"neuro", "allergy", "rash"})

_SIDE = r"(한쪽|한 쪽|편측|반쪽|왼쪽|오른쪽|좌측|우측|왼|오른|one side|left|right|unilateral)"
_MOTOR = r"(힘이 빠|힘 빠|힘이 없|힘이 안|마비|위약|약화|근력 저하|근력저하|weak)"
_SENSORY = r"(저림|저리|저려|감각|numb)"
_RE_SIDE_MOTOR = re.compile(_SIDE + r".{0,15}?" + _MOTOR)
_RE_SIDE_SENSORY = re.compile(_SIDE + r".{0,15}?" + _SENSORY)
_DIZZY = ("어지럼", "어지러", "어지럽", "현훈", "빙빙", "vertigo", "dizz")
_GAIT = ("보행", "걷기", "걸을 때 비틀", "비틀거", "균형", "실조", "ataxi", "gait", "unsteady")


def _mask(t: str) -> str:
    for m in _MASKS:
        t = t.replace(m, " ")
    return t


def _neuro_cooccurrence(t: str, chest: bool) -> bool:
    """Focal deficit expressed as a combination: one side + weakness (or numbness, except with chest pain where
    arm numbness is usually radiation), or dizziness + gait disturbance."""
    if _RE_SIDE_MOTOR.search(t):
        return True
    if not chest and _RE_SIDE_SENSORY.search(t):
        return True
    return any(k in t for k in _DIZZY) and any(k in t for k in _GAIT)


# --------------------------------------------------------------------------------------------
# Duration / age parsing (chief complaint text)
# --------------------------------------------------------------------------------------------

_ACUTE_MARKERS: tuple[str, ...] = (
    "갑자기", "갑작스", "급작스", "급격", "벼락", "순간", "방금", "분 전", "시간 전", "시간째", "오늘", "어젯밤",
    "sudden", "abrupt", "thunderclap", "minutes ago", "hours ago", "this morning", "today",
)
_CHRONIC_WORDS: tuple[str, ...] = ("만성", "오래전부터", "오래 전부터", "chronic", "long-standing", "longstanding")
_RE_INFANT_AGE_DUR = re.compile(r"생후\s*\d+\s*(시간|일|주|개월|달)")
_RE_PREG_WEEKS = re.compile(r"(임신|재태)\s*\d+\s*주|\d+\s*주\s*(차\s*)?임신|gestation")
_RE_MONTHS_YEARS = re.compile(
    r"(\d+|몇|수|여러|반)\s*(개월|달|년)(?!생)|(?<![0-9])(한|두|세)\s달(?!리)"
    r"|\b(\d+|several|few|many)\s*(months?|years?)\b")
_RE_WEEKS = re.compile(r"(\d+|몇|수|여러|두|세)\s*주(?!\s*(임신|수))|\b(\d+|several|few)\s*weeks?\b")


def has_acute_marker(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _ACUTE_MARKERS)


def duration_level(text: str) -> int:
    """Chief-complaint duration: 0 = acute/unknown, 1 = >= 2 weeks, 2 = months/years/"만성".

    Acute markers ("갑자기", "시간 전", "오늘", ...) override to 0 (acute-on-chronic is treated as acute).
    Pregnancy weeks ("임신 38주") are ignored. Our own heuristic, not from a paper.
    """
    t = (text or "").lower()
    if has_acute_marker(t):
        return 0
    t = _RE_PREG_WEEKS.sub(" ", t)
    t = _RE_INFANT_AGE_DUR.sub(" ", t)  # "생후 9개월" is the age, not how long the complaint has lasted
    if any(k in t for k in _CHRONIC_WORDS) or _RE_MONTHS_YEARS.search(t):
        return 2
    for m in _RE_WEEKS.finditer(t):
        n = m.group(1) or m.group(3) or ""
        if not n.isdigit() or int(n) >= 2:
            return 1
    return 0


_RE_DUR_UNITS = re.compile(r"(\d+|몇|수|여러|반|한|두|세)\s*(개월|달|년|주)(?!생|\s*(임신|수))"
                           r"|\b(\d+|several|few|many)\s*(months?|years?|weeks?)\b")
_KO_NUM = {"몇": 3, "수": 3, "여러": 3, "반": 0.5, "한": 1, "두": 2, "세": 3, "several": 3, "few": 3, "many": 3}
_UNIT_WEEKS = {"개월": 4.3, "달": 4.3, "년": 52, "주": 1, "month": 4.3, "year": 52, "week": 1}


def duration_weeks(text: str) -> float:
    """Longest stated chief-complaint duration in weeks (0 = acute/unknown; "만성" = 52). Acute markers and
    pregnancy weeks / infant ages are ignored the same way as in duration_level(). Our heuristic."""
    t = (text or "").lower()
    if has_acute_marker(t):
        return 0.0
    t = _RE_INFANT_AGE_DUR.sub(" ", _RE_PREG_WEEKS.sub(" ", t))
    best = 52.0 if any(k in t for k in _CHRONIC_WORDS) else 0.0
    for m in _RE_DUR_UNITS.finditer(t):
        n, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(5))
        num = float(n) if n.isdigit() else _KO_NUM.get(n, 1)
        best = max(best, num * _UNIT_WEEKS[unit.rstrip("s")])
    return best


_RE_AGE = re.compile(r"(\d+)\s*세|(\d+)\s*대|(\d+)[- ]year[- ]old")


def age_years(text: str) -> float | None:
    """Age from the chief complaint ("45세", "40대 후반", "신생아"); None when not stated."""
    t = (text or "").lower()
    if any(k in t for k in ("신생아", "생후", "영아", "newborn", "neonat", "infant")):
        return 0.0
    m = _RE_AGE.search(t)
    if m:
        return float(next(g for g in m.groups() if g))
    return None


_RE_AGE_INFANT = re.compile(r"생후\s*(\d+)\s*(시간|일|주|개월|달)")
_INFANT_UNIT_DAYS = {"시간": 1 / 24, "일": 1, "주": 7, "개월": 30, "달": 30}


def age_days(text: str) -> float | None:
    """Infant age in days from "생후 5일", "생후 2주", "생후 3개월"; 0 for "신생아"/"newborn"; None otherwise."""
    t = (text or "").lower()
    m = _RE_AGE_INFANT.search(t)
    if m:
        return float(m.group(1)) * _INFANT_UNIT_DAYS[m.group(2)]
    if any(k in t for k in ("신생아", "newborn", "neonat")):
        return 0.0
    return None


# Rule applicability only (not used by detect_categories/safety): ages in months ("18개월 된 아기") and "N살".
# A bare "3개월" is a duration ("3개월 전 머리를 부딪혔어요"), so months need an age marker after them.
_RE_AGE_MONTHS = re.compile(
    r"(\d+)\s*개월\s*(된|짜리|째 되는|\s?(남아|여아|아기|아이|영아|유아|환아|아들|딸|남자아이|여자아이))"
    r"|(\d+)[- ]months?[- ]old")
_RE_AGE_SAL = re.compile(r"(?<![0-9])(\d{1,2})\s*살")
# Child words: a pediatric patient when no age is stated (the parent is usually the one speaking)
_CHILD_WORDS: tuple[str, ...] = (
    "소아", "남아", "여아", "환아", "아기", "어린이", "유아", "영아", "신생아", "우리 아이", "아이가", "아이는", "아이의",
    "남자아이", "여자아이", "초등학생", "유치원", "child", "toddler", "infant", "newborn", "baby", "kid ",
)
ADULT_RULE_MIN_AGE = 15  # rules with min_age >= this are adult-only: skipped for "아기", "소아" etc.


def rule_age_years(text: str) -> float | None:
    """Age in years for rule applicability: "생후 N일/주/개월", "N개월 된/남아", "N살", then age_years()."""
    t = (text or "").lower()
    d = age_days(t)
    if d is not None and d > 0:
        return d / 365.25
    m = _RE_AGE_MONTHS.search(t)
    if m:
        return float(m.group(1) or m.group(4)) / 12
    m = _RE_AGE_SAL.search(t)
    if m:
        return float(m.group(1))
    return age_years(t)


# Posterior-circulation stroke screen for acute dizziness (our heuristic, not from a cited rule): age >= 60 or a
# vascular risk factor, unless the story is clearly BPPV-like (positional AND recurrent/weeks-long).
_VASCULAR_RF = ("고혈압", "혈압약", "당뇨", "심방세동", "뇌졸중", "뇌경색", "뇌출혈", "중풍", "hypertension",
                "diabetes", "atrial fibrillation", "afib", "prior stroke", "history of stroke")
_POSITIONAL = ("자세를 바꿀", "자세를 바꾸", "자세 변화", "고개를 돌", "고개를 젖", "고개를 숙", "머리를 돌",
               "누울 때", "누우면", "누웠다", "돌아누", "돌아 누", "일어날 때", "일어나면", "체위", "positional",
               "roll over", "rolling over", "turning the head", "turn my head", "lying down")
_RECURRENT_EPISODES = ("때마다", "반복", "자주", "여러 번", "매번", "recurrent", "every time", "episodes")
DIZZY_STROKE_MIN_AGE = 60


def _dizzy_stroke_risk(t: str, context: str) -> bool:
    """Acute dizziness/vertigo that warrants the stroke checks (posterior circulation can't be excluded)."""
    if not any(k in t for k in _DIZZY) or duration_level(t) >= 1:
        return False
    both = f"{t}. {(context or '').lower()}"
    age = age_years(t)
    if not ((age is not None and age >= DIZZY_STROKE_MIN_AGE) or contains_affirmed(both, _VASCULAR_RF)):
        return False
    bppv_like = contains_affirmed(both, _POSITIONAL) and contains_affirmed(both, _RECURRENT_EPISODES)
    return not bppv_like


# Chief-complaint category -> lexicon concepts (data/lexicon protocol links "category:<name>"). Only categories whose
# linked concepts name exactly that complaint add categories from the layer (colloquial forms the keyword lists
# miss); the keyword lists stay the primary detector. Not from the layer: categories whose links or concepts are
# broader than the protocol population (rash <- "반점" floaters, edema <- local swelling, neuro <- visual symptoms,
# joint <- leg edema / surgery, allergy <- facial edema, bleeding <- any bleeding, psychiatric <- "불안" as a
# feeling, pruritus / menstrual / cognitive with their own rules) and heartburn for chest pain.
_LAYER_CATEGORIES = ("chest_pain", "dyspnea", "headache", "fever", "abdominal_pain", "syncope", "palpitations",
                     "hemoptysis_cough", "jaundice", "back_pain", "hearing_loss")
_CATEGORY_CONCEPT_SKIP = {"chest_pain": {"SYM:heartburn"}}
# neuro concepts that are not a focal deficit for the stroke protocol (monocular "시력 저하" and drowsiness were never
# keywords; a stroke history is not a current deficit, and "뇌졸중" in the complaint is already a keyword)
_NON_FOCAL_NEURO = {"SYM:seizure", "SYM:altered_mental_status", "SYM:vision_loss", "SYM:somnolence", "HX:stroke"}


def _category_concepts() -> dict[str, frozenset[str]]:
    from doctor_agent.nlp import LEXICON
    out = {}
    for c in (*_LAYER_CATEGORIES, "neuro"):  # neuro: only for the focal-deficit concept set below
        ids = set(LEXICON.by_protocol(f"category:{c}")) - _CATEGORY_CONCEPT_SKIP.get(c, set())
        for i in list(ids):
            ids.update(LEXICON.descendants(i))
        if ids:
            out[c] = frozenset(ids)
    return out


def _layer_concepts(rt: ReadText) -> set[str]:
    """Concepts affirmed (present, hedged or uncertain) for the patient in the text."""
    return {f.concept for _s, _e, f in rt.parsed()[1] if f.subject == "patient" and f.polarity != "absent"}


def _all_denied(rt: ReadText, kws: tuple[str, ...]) -> bool:
    """Every keyword occurrence of this category is denied by the layer ("열은 없고 기침만 해요": not a fever complaint).
    Occurrences the layer has no finding for count as not denied (a keyword window rule is too weak to drop a
    category: "지혈되지 않는 출혈")."""
    st = [x for k in kws if k in rt for x in keyword_statuses(rt, k, detail=True)]
    return bool(st) and all(x == NEG and layer for x, layer in st)


def detect_categories(text: str, context: str = "") -> list[str]:
    """Categories of a chief complaint, in CATEGORY_KEYWORDS order.

    - neuro (stroke protocol): focal deficit keywords or co-occurrences, altered mental status, seizure. In an
      allergic context (hives, anaphylaxis) only a focal deficit counts, so "두드러기 + 의식 저하" is not stroke.
      Acute dizziness in the chief complaint also opens it when age >= 60 or a vascular risk factor is known
      (`context` = facts learned later, used only for risk factors / BPPV-like pattern), unless positional AND
      recurrent (BPPV-like).
    - acute-only categories (neuro, allergy, rash) are dropped when the complaint has lasted >= 2 weeks.
    - hemoptysis_cough: hemoptysis at any duration, or cough lasting >= 2 weeks. pruritus: generalized itch
      lasting >= 2 weeks. edema: limb/generalized/facial swelling or foamy urine (not a swollen joint or neck).
      rash/edema are dropped in an allergic context; joint is dropped after trauma.
    - 2026-09-27: edema is dropped when it is one symptom of an acute dyspnea/fever/chest-pain complaint (unless
      foamy urine, facial or generalized edema); abdominal location words need a pain word; urticaria removes
      pruritus and opens urticaria_chronic at >= 6 weeks; hearing_loss, neck_mass (adult checks), bleeding
      (not after trauma), chronic_weakness (limb weakness >= 2 weeks), bilious_vomiting (infants <= 3 months);
      transient/recurrent altered consciousness without focal signs or seizure is syncope, not stroke.
    """
    t = _mask((text or "").lower())
    rt = ReadText(t)
    found = {c for c, kws in CATEGORY_KEYWORDS.items() if any(k in t for k in kws) and not _all_denied(rt, kws)}
    layer = _layer_concepts(rt)
    found |= {c for c in _LAYER_CATEGORIES if layer & _CATEGORY_CONCEPTS.get(c, frozenset())}
    focal = any(k in t for k in _NEURO_FOCAL) or _neuro_cooccurrence(t, "chest_pain" in found) \
        or bool(layer & _FOCAL_CONCEPTS)
    if focal:
        found.add("neuro")
    elif "allergy" in found:
        found.discard("neuro")
    elif _dizzy_stroke_risk(t, context):
        found.add("neuro")
    dur = duration_level(t)
    # cough alone is only the chronic-cough protocol (>= 2 weeks, our proxy for ACCP's subacute/chronic cough)
    if dur >= 1 and any(k in t for k in _COUGH):
        found.add("hemoptysis_cough")
    # chronic pruritus = generalized itch lasting >= 2 weeks (proxy; the European guideline defines >= 6 weeks)
    if not (dur >= 1 and any(k in t for k in _GENERALIZED) and contains_affirmed(t, _ITCH)):
        found.discard("pruritus")
    if _RE_EDEMA.search(t):
        found.add("edema")
    # 2026-09-27: leg edema listed among acute dyspnea/fever/chest-pain symptoms (pneumonia, heart failure) is covered
    # by those protocols; the glomerular (urinalysis) work-up is for primary edema, foamy urine or facial edema
    if "edema" in found and found & {"dyspnea", "fever", "chest_pain"} and not any(k in t for k in _EDEMA_PRIMARY):
        found.discard("edema")
    # abdominal location words alone ("좌하복부의 피하 덩이") need a pain word
    if "abdominal_pain" in found and not any(k in t for k in CATEGORY_KEYWORDS["abdominal_pain"]
                                             if k not in _ABD_LOCATION_ONLY) \
            and not any(k in t for k in _ABD_PAIN_WORDS):
        found.discard("abdominal_pain")
    # urticaria: itch from wheals is not chronic pruritus of unknown origin; >= 6 weeks = chronic urticaria (EAACI)
    if any(k in t for k in _URTICARIA):
        found.discard("pruritus")
        if duration_weeks(t) >= 6:
            found.add("urticaria_chronic")
    if _RE_NECK_MASS.search(t):
        found.add("neck_mass")
    if "bleeding" in found and any(k in t for k in _JOINT_TRAUMA):  # post-traumatic nosebleed etc.
        found.discard("bleeding")
    if dur >= 1 and _RE_LIMB_WEAKNESS.search(t):
        found.add("chronic_weakness")
    if any(k in t for k in _BILIOUS):
        d = age_days(t)
        if (d is not None and d <= 90) or any(k in t for k in ("영아", "infant")):
            found.add("bilious_vomiting")
    # transient/recurrent loss or lowering of consciousness without focal signs or seizure: syncope, not stroke
    if "neuro" in found and not focal and any(k in t for k in _TRANSIENT) and not any(k in t for k in _SEIZURE) \
            and any(k in t for k in _NEURO_AMS_SEIZURE):
        found.discard("neuro")
        found.add("syncope")
    if "allergy" in found:  # hives/angioedema: the anaphylaxis protocol, not rash or edema work-up
        found -= {"rash", "edema"}
    if "neuro" in found:  # acute altered mental status / focal deficit: organic work-up first
        found.discard("psychiatric")
    if "joint" in found and any(k in t for k in _JOINT_TRAUMA):  # injuries: fracture/dislocation, not arthritis
        found.discard("joint")
    if dur >= 1:
        found -= ACUTE_ONLY_CATEGORIES
    return [c for c in CATEGORY_KEYWORDS if c in found]


def _legacy_negated_at(t: str, idx: int, keyword: str) -> bool:
    """Pre-2026-09-27 reading of one occurrence (fallback for keywords the normalisation layer does not cover):
    negated when a negation word follows within the clause (40 chars, stops at commas and connectives)."""
    start = idx + len(keyword)
    end = min(len(t), start + _NEG_WINDOW)
    for sep in _CLAUSE_BREAKS:
        j = t.find(sep, start)
        if 0 <= j < end:
            end = j
    tail = t[start:end]
    return any(n in tail for n in _NEGATIONS)


# chars after a keyword searched for a negation within the same clause (40: "심잡음이나 굴러가는 소리(friction
# rub) 청진되지 않음" negates the murmur)
_NEG_WINDOW = 40
_NEGATIONS: tuple[str, ...] = ("없", "않", "아니", "기보다", "안 해", "안 했", "음성", "(-)", "no ", "not ", "denies",
                               "denied", "without", "never", "negative", "absent")
# Clause ends: punctuation and connectives that start a new clause ("머리가 아프고 입맛도 없어요" → 2 clauses).
# "거나" is deliberately not a break: "저리거나 힘이 빠진 적 없고" negates both.
_CLAUSE_BREAKS: tuple[str, ...] = (".", "?", "!", "\n", ",", ";", "고 ", "며 ", "면서", "는데", "지만", " but ", " and ")

# Keywords read by the legacy rule even where the layer has a finding: the keyword carries its own negation
# ("의식이 없", "수동적 움직임에는 제한이 없") or is itself about relatives ("대동맥 박리 가족력").
_KW_OWN_NEGATION = re.compile(r"없|않|아니|(?<![가-힣])안\s|못\s|음성|정상|\(-\)|\bno\b|\bnot\b|without|denie|negative"
                              r"|absent|(?<=[가-힣])지$|멈$|멎$")  # "지혈되지", "피가 안 멈": the negation that follows is part of the phrase
_KW_ABOUT_RELATIVES = re.compile(r"가족|family")
# Fallback subject rule for keywords the layer has no finding for: the latest person word before the keyword in its
# sentence is a relative (with a particle: "어머니가", "형은", "가족 중", "가족력") and no self word follows it.
# Not 아들/딸/자녀: a parent often speaks for a child patient.
_RE_RELATIVE = re.compile(
    r"가족력|가족\s?중|가족분|집안에|family history"
    r"|(?:(?:외|친)?(?:할머니|할아버지)|어머니|어머님|엄마|아버지|아버님|아빠|부모님?|오빠|누나|언니|남동생|여동생|동생"
    r"|삼촌|외삼촌|이모|고모|숙부|큰아버지|작은아버지|형제|자매|사촌|(?<![가-힣])형)(?:께서|가|이|는|은|도|\s?중|\s?쪽)"
    r"|\b(?:mother|father|brother|sister|sibling|parents?)\b")
_RE_SELF = re.compile(r"(?<![가-힣])저(?:는|도|만)|(?<![가-힣])제가|본인(?:은|이|도)|(?<![가-힣])나(?:는|도)\s|(?<![가-힣])내가"
                      r"|환자(?:는|가|분은)|\bi\b|\bthe patient\b")


_RE_FAMILY_HISTORY = re.compile(r"가족력\s?(?::|상|으로|에서?\s|이\s?있)|가족\s?중|family history\s?(?::|of)")


def _relative_before(t: str, idx: int, rx: re.Pattern = _RE_RELATIVE) -> bool:
    s = max(t.rfind(c, 0, idx) for c in ".?!;\n") + 1
    head = t[s:idx]
    rel = [m.end() for m in rx.finditer(head)]
    if not rel:
        return False
    return not any(m.start() >= rel[-1] for m in _RE_SELF.finditer(head))


def after_family_heading(text: str, pos: int) -> bool:
    """True when a family-history heading ("가족력:", "가족 중", "family history of") precedes pos in its sentence
    and no self word ("저는", "제가") comes after it."""
    return _relative_before((text or "").lower(), pos, _RE_FAMILY_HISTORY)


# Occurrence status: "pos" (affirmed, hedged or not), "unc" (uncertain / hypothetical: never counts as denied),
# "neg" (denied for the patient), "other" (about a relative or another person: neither the patient's nor denied)
POS, UNC, NEG, OTHER = "pos", "unc", "neg", "other"


class ReadText(str):
    """A text plus its findings from the normalisation layer (doctor_agent.nlp), parsed lazily once per object.

    Used by negated() / contains_affirmed() so that one case text is parsed once for many keyword lookups (pass a
    ReadText instead of a str). Created per call/case by the caller; nothing is kept at module level."""

    def __new__(cls, text: str = "", source: str = "patient"):
        obj = super().__new__(cls, text or "")
        obj.source = source
        obj._parsed = None
        return obj

    def parsed(self):
        """(normalised text, [(start, end, finding)]) with offsets into the normalised text. Each line is parsed on
        its own (a line break ends a sentence in case files; parsing per line keeps offsets simple: "b-hcg 양성
        \\n hbsag 음성" must not be one list)."""
        if self._parsed is None:
            from doctor_agent.nlp import normalize, parse
            from doctor_agent.nlp.findings import spans_of
            parts, spans, off = [], [], 0
            self._line_off = []
            for line in str(self).split("\n"):
                n = normalize(line)
                spans += [(off + a, off + b, f) for f in parse(line, self.source) for a, b in spans_of(f)]
                parts.append(n)
                self._line_off.append(off)
                off += len(n) + 1
            self._parsed = ("\n".join(parts), spans)
        return self._parsed

    def findings_at(self, start: int, end: int) -> list:
        """Layer findings overlapping [start, end) of this (unnormalised) text; a list-scope negation that does not
        end its sentence is left out (see _usable)."""
        from doctor_agent.nlp import normalize
        norm, spans = self.parsed()
        raw = str(self)

        def to_norm(pos: int) -> int:
            ls = raw.rfind("\n", 0, pos) + 1
            return self._line_off[raw.count("\n", 0, ls)] + len(normalize(raw[ls:pos] + "x")) - 1

        s, e = to_norm(start), to_norm(end)
        return [f for fs, fe, f in spans if fs < e and s < fe and _usable(norm, fe, f)]


def _as_read(text, source: str = "patient") -> ReadText:
    return text if isinstance(text, ReadText) else ReadText(text or "", source)


_RE_SENT_END = re.compile(r"[.?!;\n]")


def _usable(norm: str, end: int, f) -> bool:
    """A list-scope negation ("기침이나 가래, 열은 없어요") is trusted only when the negation ends the sentence;
    an adnominal one ("임신 32주, 통증 없는 질 출혈") falls back to the per-keyword rule."""
    if not (f.polarity == "absent" and f.cue.startswith("list:")):
        return True
    m = _RE_SENT_END.search(norm, end)
    tail = norm[end:m.start() if m else len(norm)]
    cue = f.cue.split(":", 1)[1].strip()  # the negation word the layer used ("list:정상")
    p = tail.find(cue) if cue else -1
    if p < 0:
        return False
    rest = tail[p:].strip()
    return len(rest) <= 8 and " " not in rest


# A relative or partner speaks for the patient ("(남편) 오늘 아침부터 아내가 헛소리를 해요"): person words then name the
# patient, so subject attribution is not trusted (conservative: the finding counts as the patient's).
_RE_PROXY = re.compile(r"\((?:남편|아내|부인|배우자|보호자|엄마|아빠|어머니|아버지|딸|아들|며느리|사위|가족|동생|형|언니|누나|오빠)"
                       r"(?:\s?[가-힣]{0,4})?\)|보호자\s?(?::|진술|에 따르면|가 대신|분이 대신)|대신\s?(?:대답|말씀|설명)")


def _legacy_status(t: str, idx: int, kw: str, family_kw: bool, proxy: bool = False) -> str:
    if not family_kw and not proxy and _relative_before(t, idx):
        return OTHER
    return NEG if _legacy_negated_at(t, idx, kw) else POS


def keyword_statuses(text, keyword: str, source: str = "patient", detail: bool = False) -> list:
    """Status of every occurrence of keyword (lowercase substring) in text: POS / UNC / NEG / OTHER.

    An occurrence overlapping a finding of the normalisation layer takes that finding's reading (polarity with
    list scope, idioms, persistence and double negation; subject; hedges keep "present"; uncertain and hypothetical
    answers are UNC). An occurrence the layer has no finding for, or a keyword that carries its own negation, keeps
    the legacy clause-window rule (NEG or POS), plus a relative-before-it check (OTHER). detail=True returns
    (status, read_by_layer) pairs."""
    rt = _as_read(text, source)
    kw = (keyword or "").lower()
    if not kw:
        return []
    norm, findings = rt.parsed()
    fam_kw = bool(_KW_ABOUT_RELATIVES.search(kw))
    proxy = bool(_RE_PROXY.search(norm))
    out: list = []

    def add(st: str, layer: bool) -> None:
        out.append((st, layer) if detail else st)

    if kw not in norm:  # normalisation changed the keyword's surface: legacy reading on the raw text
        raw = str(rt).lower()
        for m in re.finditer(re.escape(kw), raw):
            add(_legacy_status(raw, m.start(), kw, fam_kw, proxy), False)
        return out
    own = fam_kw or bool(_KW_OWN_NEGATION.search(kw))
    for m in re.finditer(re.escape(kw), norm):
        s, e = m.start(), m.end()
        hits = [] if own else [f for fs, fe, f in findings if fs < e and s < fe and _usable(norm, fe, f)]
        if not hits:
            add(_legacy_status(norm, s, kw, fam_kw, proxy), False)
            continue
        # "가족력: 고혈압" (a family-history heading without a subject particle) is not the patient's
        if _relative_before(norm, s, _RE_FAMILY_HISTORY):
            mine = []
        else:
            mine = [f for f in hits if f.subject == "patient" or proxy]
        if any(f.polarity == "present" and not f.hypothetical for f in mine):
            add(POS, True)
        elif any(f.polarity == "uncertain" or f.hypothetical for f in mine):
            add(UNC, True)
        elif mine:
            add(NEG, True)
        else:
            add(OTHER, True)
    return out


def negated(text: str, keyword: str) -> bool:
    """True when the keyword occurs, is denied for the patient at least once, and no occurrence is affirmed or
    uncertain ("다리 붓거나 비행기 탄 적은 없어요", "등이 찢어지는 느낌은 아니에요"; relatives' mentions are ignored).
    Read through the normalisation layer (keyword_statuses); `text` may be a ReadText."""
    st = keyword_statuses(text, keyword)
    return NEG in st and not (POS in st or UNC in st)


def contains_affirmed(text: str, keywords: tuple[str, ...]) -> bool:
    """Any keyword affirmed for the patient: present (hedged too) or uncertain ("열이 나는지 모르겠어요" still
    triggers). Denied mentions and mentions about relatives ("어머니가 유방암") do not count."""
    rt = _as_read(text)
    low = str(rt).lower()
    for k in keywords:
        if k and k.lower() in low:  # cheap pre-filter: the text is parsed only when a keyword occurs
            st = keyword_statuses(rt, k)
            if POS in st or UNC in st:
                return True
    return False


# --------------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    authors: str
    title: str
    journal: str
    year: int
    volume_pages: str
    doi: str | None = None
    pmid: str | None = None
    verified: bool = False  # bibliographic data checked against PubMed/Crossref
    short_author: str = ""  # override for group authors (e.g. "ACOG")

    @property
    def short(self) -> str:
        """Short form for prompts, e.g. 'Wells 2000'."""
        name = self.short_author or self.authors.split(",")[0].split()[0]
        return f"{name} {self.year}"

    @property
    def url(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return ""

    def full(self) -> str:
        ref = f"{self.authors} {self.title}. {self.journal}. {self.year};{self.volume_pages}."
        if self.doi:
            ref += f" doi:{self.doi}"
        if self.pmid:
            ref += f" PMID:{self.pmid}"
        return ref


@dataclass(frozen=True)
class Item:
    key: str
    text: str  # Korean, our own wording
    points: float = 1.0  # points when present (binary item)
    options: tuple[tuple[str, float], ...] = ()  # graded item: (Korean label, points)
    group: str = ""  # used by "groups" / "tiers" rules

    @property
    def max_points(self) -> float:
        # a negative binary item (e.g. CSRS vasovagal predisposition -1) contributes at most 0
        return max(p for _, p in self.options) if self.options else max(self.points, 0.0)


@dataclass(frozen=True)
class Threshold:
    lo: float  # inclusive
    hi: float  # inclusive
    range_text: str
    label: str
    meaning: str
    rule_out: bool = False  # this band means "ruled out" (only valid when every item was checked)

    def contains(self, s: float) -> bool:
        return self.lo <= s <= self.hi


@dataclass(frozen=True)
class Rule:
    id: str
    short: str  # e.g. "Wells PE"
    name_ko: str
    name_en: str
    purpose: str  # Korean: what the rule answers
    population: str  # Korean: who it applies to
    categories: tuple[str, ...]
    keywords: tuple[str, ...]  # extra lowercase triggers beyond the categories
    items: tuple[Item, ...]
    thresholds: tuple[Threshold, ...]
    citation: Citation
    verification: str  # "primary" | "secondary" | "unverified"
    note: str = ""
    method: str = "sum"  # "sum" | "groups" (count of groups with any positive) | "tiers" (highest positive tier)
    groups: tuple[tuple[str, str], ...] = ()  # ordered (key, Korean label); for "tiers" low → high
    secondary_thresholds: tuple[Threshold, ...] = ()
    # Applicability (validated population of the original paper). See `applies_to`.
    requires_any: tuple[str, ...] = ()  # the chief complaint must also contain one of these
    excludes_any: tuple[str, ...] = ()  # outside the population (a rule keyword overrides these)
    chronic_cutoff: int = 0  # exclude when duration_level(text) >= this (1 = >=2 weeks, 2 = months+); 0 = never
    min_age: float | None = None  # exclude when a stated age is below this
    applicability: str = ""  # source of the conditions above
    # 2026-09-27 additions
    max_age: float | None = None  # pediatric rule: age must be < max_age (years); needs a stated age or child word
    max_age_days: float | None = None  # infant rule: age in days must be <= this (from "생후 N일/주/개월", "신생아")
    veto_any: tuple[str, ...] = ()  # always excluded when affirmed, even when a rule keyword matched

    def applies_to(self, text: str) -> bool:
        """Chief complaint is inside the rule's validated population.

        Match = (category or rule keyword) AND requires_any AND NOT excludes_any (unless a rule keyword is present,
        e.g. "명치에서 시작해 오른쪽 아랫배로" keeps Alvarado) AND NOT veto_any (affirmed) AND duration below
        chronic_cutoff AND age inside [min_age, max_age) / <= max_age_days.
        Age comes from rule_age_years() ("45세", "3살", "18개월 된 아기", "생후 5주"). When no age is stated,
        a child word ("소아", "아기", "남아", ...) counts as pediatric: rules with min_age >= 15 are skipped, and
        pediatric rules (max_age) need either a stated pediatric age or such a word.
        """
        t = (text or "").lower()
        kw = any(k in t for k in self.keywords)
        if not (kw or set(detect_categories(t)).intersection(self.categories)):
            return False
        if self.requires_any and not any(k in t for k in self.requires_any):
            return False
        if not kw and any(k in t for k in self.excludes_any):
            return False
        if self.veto_any and contains_affirmed(t, self.veto_any):
            return False
        # "18개월 된 아기" is an age, not a months-long complaint
        if self.chronic_cutoff and duration_level(_RE_AGE_MONTHS.sub(" ", t)) >= self.chronic_cutoff:
            return False
        age = rule_age_years(t)
        child = age is None and any(k in t for k in _CHILD_WORDS)
        if self.min_age is not None:
            if age is not None and age < self.min_age:
                return False
            if child and self.min_age >= ADULT_RULE_MIN_AGE:
                return False
        if self.max_age is not None:
            if age is None and not child:
                return False
            if age is not None and age >= self.max_age:
                return False
        if self.max_age_days is not None:
            d = age_days(t)
            if d is None or d > self.max_age_days:
                return False
        return True

    @property
    def max_score(self) -> float:
        if self.method in ("groups", "tiers"):
            return float(len(self.groups))
        return float(sum(i.max_points for i in self.items))

    @property
    def cite(self) -> str:
        return f"{self.short} ({self.citation.short})"


@dataclass(frozen=True)
class ScoreResult:
    rule_id: str
    score: float
    max_score: float
    label: str
    meaning: str
    complete: bool
    missing: tuple[str, ...] = ()
    secondary_label: str = ""
    citation: str = ""
    positives: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        s = f"{self.citation}: {_fmt(self.score)}/{_fmt(self.max_score)}점 → {self.label} ({self.meaning})"
        if self.secondary_label:
            s += f"; {self.secondary_label}"
        if self.missing:
            s += f" [미확인 항목 {len(self.missing)}개]"
        return s


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# --------------------------------------------------------------------------------------------
# Citations (all bibliographic data checked against PubMed E-utilities / Crossref, 2026-09-25)
# --------------------------------------------------------------------------------------------

C_WELLS_PE = Citation(
    "Wells PS, Anderson DR, Rodger M, et al.",
    "Derivation of a simple clinical model to categorize patients probability of pulmonary embolism: "
    "increasing the models utility with the SimpliRED D-dimer",
    "Thromb Haemost", 2000, "83(3):416-420", doi="10.1055/s-0037-1613830", pmid="10744147", verified=True,
)
C_PERC = Citation(
    "Kline JA, Mitchell AM, Kabrhel C, Richman PB, Courtney DM.",
    "Clinical criteria to prevent unnecessary diagnostic testing in emergency department patients with "
    "suspected pulmonary embolism",
    "J Thromb Haemost", 2004, "2(8):1247-1255", doi="10.1111/j.1538-7836.2004.00790.x", pmid="15304025",
    verified=True,
)
C_HEART = Citation(
    "Six AJ, Backus BE, Kelder JC.",
    "Chest pain in the emergency room: value of the HEART score",
    "Neth Heart J", 2008, "16(6):191-196", doi="10.1007/BF03086144", pmid="18665203", verified=True,
)
C_ADD_RS = Citation(
    "Rogers AM, Hermann LK, Booher AM, et al.",
    "Sensitivity of the aortic dissection detection risk score, a novel guideline-based tool for "
    "identification of acute aortic dissection at initial presentation: results from the International "
    "Registry of Acute Aortic Dissection",
    "Circulation", 2011, "123(20):2213-2218", doi="10.1161/CIRCULATIONAHA.110.988568", pmid="21555704",
    verified=True,
)
C_QSOFA = Citation(
    "Seymour CW, Liu VX, Iwashyna TJ, et al.",
    "Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for "
    "Sepsis and Septic Shock (Sepsis-3)",
    "JAMA", 2016, "315(8):762-774", doi="10.1001/jama.2016.0288", pmid="26903335", verified=True,
)
C_CURB65 = Citation(
    "Lim WS, van der Eerden MM, Laing R, et al.",
    "Defining community acquired pneumonia severity on presentation to hospital: an international "
    "derivation and validation study",
    "Thorax", 2003, "58(5):377-382", doi="10.1136/thorax.58.5.377", pmid="12728155", verified=True,
)
C_CENTOR = Citation(
    "Centor RM, Witherspoon JM, Dalton HP, Brody CE, Link K.",
    "The diagnosis of strep throat in adults in the emergency room",
    "Med Decis Making", 1981, "1(3):239-246", doi="10.1177/0272989X8100100304", pmid="6763125", verified=True,
)
C_OTTAWA_SAH = Citation(
    "Perry JJ, Stiell IG, Sivilotti ML, et al.",
    "Clinical decision rules to rule out subarachnoid hemorrhage for acute headache",
    "JAMA", 2013, "310(12):1248-1255", doi="10.1001/jama.2013.278018", pmid="24065011", verified=True,
)
C_CCHR = Citation(
    "Stiell IG, Wells GA, Vandemheen K, et al.",
    "The Canadian CT Head Rule for patients with minor head injury",
    "Lancet", 2001, "357(9266):1391-1396", doi="10.1016/S0140-6736(00)04561-X", pmid="11356436", verified=True,
)
C_ABCD2 = Citation(
    "Johnston SC, Rothwell PM, Nguyen-Huynh MN, et al.",
    "Validation and refinement of scores to predict very early stroke risk after transient ischaemic attack",
    "Lancet", 2007, "369(9558):283-292", doi="10.1016/S0140-6736(07)60150-0", pmid="17258668", verified=True,
)
C_ALVARADO = Citation(
    "Alvarado A.",
    "A practical score for the early diagnosis of acute appendicitis",
    "Ann Emerg Med", 1986, "15(5):557-564", doi="10.1016/S0196-0644(86)80993-3", pmid="3963537", verified=True,
)
C_BISAP = Citation(
    "Wu BU, Johannes RS, Sun X, Tabak Y, Conwell DL, Banks PA.",
    "The early prediction of mortality in acute pancreatitis: a large population-based study",
    "Gut", 2008, "57(12):1698-1703", doi="10.1136/gut.2008.152702", pmid="18519429", verified=True,
)
# 2026-09-27 additions (bibliographic data checked against PubMed E-utilities efetch, 2026-09-27)
C_PECARN_HEAD = Citation(
    "Kuppermann N, Holmes JF, Dayan PS, et al.",
    "Identification of children at very low risk of clinically-important brain injuries after head trauma: "
    "a prospective cohort study",
    "Lancet", 2009, "374(9696):1160-1170", doi="10.1016/S0140-6736(09)61558-0", pmid="19758692", verified=True,
)
C_NEXUS = Citation(
    "Hoffman JR, Mower WR, Wolfson AB, Todd KH, Zucker MI.",
    "Validity of a set of clinical criteria to rule out injury to the cervical spine in patients with blunt "
    "trauma",
    "N Engl J Med", 2000, "343(2):94-99", doi="10.1056/NEJM200007133430203", pmid="10891516", verified=True,
)
C_CCSR = Citation(
    "Stiell IG, Wells GA, Vandemheen KL, et al.",
    "The Canadian C-spine rule for radiography in alert and stable trauma patients",
    "JAMA", 2001, "286(15):1841-1848", doi="10.1001/jama.286.15.1841", pmid="11597285", verified=True,
)
C_SFSR = Citation(
    "Quinn JV, Stiell IG, McDermott DA, Sellers KL, Kohn MA, Wells GA.",
    "Derivation of the San Francisco Syncope Rule to predict patients with short-term serious outcomes",
    "Ann Emerg Med", 2004, "43(2):224-232", doi="10.1016/s0196-0644(03)00823-0", pmid="14747812", verified=True,
)
C_CSRS = Citation(
    "Thiruganasambandamoorthy V, Kwong K, Wells GA, et al.",
    "Development of the Canadian Syncope Risk Score to predict serious adverse events after emergency "
    "department assessment of syncope",
    "CMAJ", 2016, "188(12):E289-E298", doi="10.1503/cmaj.151469", pmid="27378464", verified=True,
)
C_GBS = Citation(
    "Blatchford O, Murray WR, Blatchford M.",
    "A risk score to predict need for treatment for upper-gastrointestinal haemorrhage",
    "Lancet", 2000, "356(9238):1318-1321", doi="10.1016/S0140-6736(00)02816-6", pmid="11073021", verified=True,
)
C_KOCHER = Citation(
    "Kocher MS, Zurakowski D, Kasser JR.",
    "Differentiating between septic arthritis and transient synovitis of the hip in children: an "
    "evidence-based clinical prediction algorithm",
    "J Bone Joint Surg Am", 1999, "81(12):1662-1670", doi="10.2106/00004623-199912000-00002", pmid="10608376",
    verified=True,
)
C_PAS = Citation(
    "Samuel M.",
    "Pediatric appendicitis score",
    "J Pediatr Surg", 2002, "37(6):877-881", doi="10.1053/jpsu.2002.32893", pmid="12037754", verified=True,
)
C_SPESI = Citation(
    "Jiménez D, Aujesky D, Moores L, et al.",
    "Simplification of the pulmonary embolism severity index for prognostication in patients with acute "
    "symptomatic pulmonary embolism",
    "Arch Intern Med", 2010, "170(15):1383-1389", doi="10.1001/archinternmed.2010.199", pmid="20696966",
    verified=True,
)
C_PECARN_FI = Citation(
    "Kuppermann N, Dayan PS, Levine DA, et al.",
    "A Clinical Prediction Rule to Identify Febrile Infants 60 Days and Younger at Low Risk for Serious "
    "Bacterial Infections",
    "JAMA Pediatr", 2019, "173(4):342-351", doi="10.1001/jamapediatrics.2018.5501", pmid="30776077", verified=True,
)
C_MCISAAC = Citation(
    "McIsaac WJ, White D, Tannenbaum D, Low DE.",
    "A clinical score to reduce unnecessary antibiotic use in patients with sore throat",
    "CMAJ", 1998, "158(1):75-83", pmid="9475915", verified=True,
)

# --------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------

# Shared applicability vocabularies (lowercase substrings)
_TRAUMA = ("외상", "다친", "다쳤", "부딪", "넘어지", "넘어져", "교통사고", "맞았", "trauma", "injur", "hit his",
           "hit her", "hit my")
_RECURRENT = ("반복", "재발", "평소", "자주", "recurrent", "usual headache")
_NON_RLQ = ("명치", "윗배", "상복부", "심와부", "옆구리", "왼쪽", "좌측", "좌하복부", "epigastr", "flank", "left",
            "upper")
# 2026-09-27 vocabularies
_HEAD_INJURY = ("머리를 부딪", "머리를 박", "머리를 다쳤", "머리를 다친", "머리 부상", "두부 외상", "머리 외상",
                "넘어지면서 머리", "떨어져서 머리", "떨어지면서 머리", "머리부터 떨어", "머리를 찧", "head injury",
                "head trauma", "hit his head", "hit her head", "hit my head", "hit their head")
# neck pain/injury words (" 목이" with a space so "손목이 아파요" does not match; "목이 아파요" alone is a sore throat
# in Korean, so the neck rules also require a trauma word)
_NECK = ("뒷목", " 목이 아", " 목이 뻐근", " 목이 안 돌", "목 통증", "목을 다", "목을 삐", "목 부상", "목뼈", "경추",
         "whiplash", "neck pain", "neck injury", "neck trauma", "c-spine", "cervical spine")
_BLUNT_TRAUMA = ("교통사고", "추돌", "충돌", "사고", "넘어", "떨어", "추락", "낙상", "부딪", "다쳤", "다친", "외상", "맞았",
                 "trauma", "injur", "collision", "crash", "fall", "fell")
_SEIZURE_INTOX = ("경련", "뇌전증", "seizure", "convuls", "만취", "intoxicat")
_UGIB = ("토혈", "피를 토", "피가 섞인 구토", "피 섞인 구토", "구토에 피", "토한 것에 피", "커피 찌꺼기", "커피색 구토",
         "흑색변", "흑변", "검은 변", "검은색 변", "변이 검", "짜장면 같은 변", "짜장 같은 변", "타르 같은 변", "타르변", "hematemesis",
         "haematemesis", "melena", "melaena", "coffee-ground", "coffee ground", "black stool", "tarry stool",
         "upper gi bleed", "위장관 출혈", "상부위장관 출혈")
_HIP_LIMP = ("고관절", "엉덩이 관절", "사타구니", "절뚝", "다리를 절", "다리를 저", "걷지 않으려", "걸으려 하지 않",
             "걷기를 거부", "걷지 못", "딛지 못", "딛지 않", "체중을 싣지", "hip", "limp", "refuses to walk",
             "won't walk", "not bearing weight", "non-weight-bearing")
_SORE_THROAT = ("인후통", "목이 따", "목 따가", "목구멍", "편도", "삼키기", "sore throat", "pharyngitis", "tonsil")
# PE already confirmed: severity (sPESI), not the diagnostic Wells/PERC
_PE_CONFIRMED = ("폐색전증 진단", "폐색전증으로 진단", "폐색전증 확진", "폐동맥 색전증 진단", "폐색전증이 확인",
                 "diagnosed with pulmonary embol", "confirmed pulmonary embol", "pulmonary embolism was confirmed")

_INF = float("inf")

RULES: tuple[Rule, ...] = (
    Rule(
        id="wells_pe", short="Wells PE", name_ko="Wells 폐색전증 점수",
        name_en="Wells score for pulmonary embolism",
        purpose="폐색전증의 임상적 사전확률 분류",
        population="폐색전증이 의심되는 환자",
        categories=("chest_pain", "dyspnea"),
        keywords=("객혈", "hemoptysis", "폐색전", "pulmonary embol"),
        items=(
            Item("dvt_signs", "심부정맥혈전증 의심 증상·징후(한쪽 다리 부종, 다리 정맥 주행 부위 압통)", 3.0),
            Item("pe_most_likely", "폐색전증보다 더 가능성 높은 다른 진단이 없음", 3.0),
            Item("hr_gt_100", "심박수 100회/분 초과", 1.5),
            Item("immobilization_surgery", "최근 4주 이내 부동(침상 안정) 또는 수술", 1.5),
            Item("prior_vte", "심부정맥혈전증·폐색전증 과거력", 1.5),
            Item("hemoptysis", "객혈", 1.0),
            Item("malignancy", "악성종양(치료 중 또는 최근 치료)", 1.0),
        ),
        thresholds=(
            Threshold(0, 1.5, "<2", "저확률", "폐색전증 사전확률 낮음"),
            Threshold(2, 6, "2–6", "중간 확률", "폐색전증 사전확률 중간"),
            Threshold(6.5, _INF, ">6", "고확률", "폐색전증 사전확률 높음"),
        ),
        secondary_thresholds=(
            Threshold(0, 4, "≤4", "PE 가능성 낮음", "D-dimer 음성이면 폐색전증 배제 가능"),
            Threshold(4.5, _INF, ">4", "PE 가능성 높음", "D-dimer만으로 배제하지 말고 영상검사로 확인"),
        ),
        citation=C_WELLS_PE, verification="primary",
        chronic_cutoff=2, min_age=18, veto_any=_PE_CONFIRMED,
        applicability="Wells 2000 abstract: patients with suspected PE (prospective cohort). Excluding "
        "months-long complaints and children is our proxy for 'acute suspicion' in the adult derivation cohort.",
        note="7 items, points, 3-tier (<2, 2-6, >6) and 2-tier (<=4 / >4) cut points all stated in the "
        "PubMed abstract. Malignancy/immobilization wording beyond the abstract is ours.",
    ),
    Rule(
        id="perc", short="PERC", name_ko="폐색전증 배제 기준", name_en="Pulmonary Embolism Rule-out Criteria",
        purpose="사전확률이 낮은 환자에서 D-dimer 검사 없이 폐색전증 배제",
        population="의사가 임상적으로 폐색전증 사전확률이 낮다고 판단한 환자에만 적용",
        categories=("chest_pain", "dyspnea"),
        keywords=(),
        items=(
            Item("age_ge_50", "나이 50세 이상"),
            Item("hr_ge_100", "맥박 100회/분 이상"),
            Item("sao2_le_94", "산소포화도 94% 이하"),
            Item("unilateral_leg_swelling", "한쪽 다리 부종"),
            Item("hemoptysis", "객혈"),
            Item("recent_surgery_trauma", "최근 수술 또는 외상"),
            Item("prior_vte", "폐색전증·심부정맥혈전증 과거력"),
            Item("hormone_use", "호르몬제 사용(경구피임약, 호르몬 치료)"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "PERC 음성",
                      "저위험군이면 D-dimer 없이 배제 가능(원 연구 저위험군 유병률 1.4%)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "PERC 양성", "배제 불가 → D-dimer 또는 Wells 평가"),
        ),
        citation=C_PERC, verification="primary",
        chronic_cutoff=2, min_age=18, veto_any=_PE_CONFIRMED,
        applicability="Kline 2004 abstract: ED patients evaluated for suspected PE, applied to low-risk "
        "(gestalt) patients only. Months-long complaints and children excluded (our proxy; adult ED cohort).",
        note="8 criteria from the PubMed abstract (age <50, pulse <100, SaO2 >94%, no unilateral leg swelling, "
        "no hemoptysis, no recent trauma/surgery, no prior PE/DVT, no hormone use). Items are phrased as "
        "risk-present, so score = number of failed criteria.",
    ),
    Rule(
        id="heart", short="HEART", name_ko="HEART 점수", name_en="HEART score",
        purpose="흉통 환자의 주요 심장 사건 위험 분류",
        population="응급실에 온 흉통 환자(급성 관상동맥 증후군 감별)",
        categories=("chest_pain",),
        keywords=(),
        items=(
            Item("history", "병력(급성 관상동맥 증후군 의심 정도)",
                 options=(("낮음", 0), ("중간", 1), ("높음", 2))),
            Item("ecg", "심전도",
                 options=(("정상", 0), ("비특이적 재분극 이상", 1), ("의미 있는 ST 분절 하강", 2))),
            Item("age", "나이", options=(("45세 미만", 0), ("45–64세", 1), ("65세 이상", 2))),
            Item("risk_factors",
                 "위험인자(치료 중인 당뇨, 현재·최근 흡연, 고혈압, 고콜레스테롤혈증, 관상동맥질환 가족력, 비만)",
                 options=(("없음", 0), ("1–2개", 1), ("3개 이상 또는 죽상경화성 질환 병력", 2))),
            Item("troponin", "트로포닌",
                 options=(("정상 상한 이하", 0), ("정상 상한의 1–2배", 1), ("정상 상한의 2배 초과", 2))),
        ),
        thresholds=(
            Threshold(0, 3, "0–3", "저위험", "주요 심장 사건 2.5% → 조기 퇴원 고려"),
            Threshold(4, 6, "4–6", "중간 위험", "주요 심장 사건 20.3% → 입원 관찰"),
            Threshold(7, 10, "7–10", "고위험", "주요 심장 사건 72.7% → 조기 침습적 치료 고려"),
        ),
        citation=C_HEART, verification="primary",
        excludes_any=_TRAUMA, chronic_cutoff=2, min_age=18,
        applicability="Six 2008 abstract: patients referred to the ER for chest pain (NSTE-ACS question). "
        "Traumatic, months-long and pediatric chest pain excluded (our proxy; adult ED cohort).",
        note="Score bands from the abstract; item grades from Table 1 of the PMC full text (PMC2442661). "
        "This is the 2008 original: troponin 1-2x / >2x normal limit (later versions use 1-3x / >3x). "
        "Table 1 prints the age-2 band as '≤65' (apparent typo); we use >=65.",
    ),
    Rule(
        id="add_rs", short="ADD-RS", name_ko="대동맥 박리 탐지 위험 점수",
        name_en="Aortic Dissection Detection Risk Score",
        purpose="급성 대동맥 박리의 임상적 위험 분류",
        population="흉통·등 통증·복통 등으로 대동맥 박리가 의심되는 환자",
        categories=("chest_pain",),
        keywords=("찢어지", "찢기는", "뜯기는", "등으로 뻗", "등까지", "tearing", "ripping", "aortic dissection"),
        method="groups",
        groups=(("condition", "고위험 기저질환"), ("pain", "고위험 통증 양상"), ("exam", "고위험 진찰 소견")),
        items=(
            Item("marfan_ctd", "마르판 증후군 등 결합조직 질환", group="condition"),
            Item("family_history", "대동맥 질환 가족력", group="condition"),
            Item("aortic_valve", "알려진 대동맥판막 질환", group="condition"),
            Item("aortic_manipulation", "최근 대동맥 시술·수술", group="condition"),
            Item("known_taa", "알려진 흉부 대동맥류", group="condition"),
            Item("abrupt_onset", "통증이 갑자기 시작됨", group="pain"),
            Item("severe_pain", "통증 강도가 매우 심함", group="pain"),
            Item("tearing_pain", "찢어지는·뜯기는 듯한 통증", group="pain"),
            Item("pulse_deficit", "맥박 결손 또는 양팔 수축기 혈압 차이", group="exam"),
            Item("focal_neuro", "통증과 동반된 국소 신경학적 결손", group="exam"),
            Item("new_ar_murmur", "통증과 동반된 새로운 대동맥판 역류 잡음", group="exam"),
            Item("hypotension_shock", "저혈압 또는 쇼크", group="exam"),
        ),
        thresholds=(
            Threshold(0, 0, "0", "저위험", "대동맥 박리 가능성 낮음(단, 0점도 완전 배제는 아님)"),
            Threshold(1, 1, "1", "중간 위험", "추가 평가 필요"),
            Threshold(2, 3, "2–3", "고위험", "즉시 대동맥 영상검사(CT 혈관조영 등)"),
        ),
        citation=C_ADD_RS, verification="secondary",
        chronic_cutoff=2,
        applicability="Rogers 2011 abstract: acute aortic dissection at initial presentation. Months-long "
        "complaints excluded (not an acute presentation).",
        note="Score definition (0-3 = number of categories met) and 0 / 1 / 2-3 bands are in the PubMed "
        "abstract (4.3% of confirmed dissections scored 0). The 12 marker names were cross-checked in a "
        "secondary source (LITFL) because the full text returned HTTP 403.",
    ),
    Rule(
        id="qsofa", short="qSOFA", name_ko="빠른 순차 장기부전 평가", name_en="quick SOFA",
        purpose="감염 의심 환자의 병원 내 사망 위험(패혈증 가능성) 선별",
        population="중환자실 밖의 감염 의심 성인",
        categories=("fever",),
        keywords=("패혈", "sepsis", "감염"),
        items=(
            Item("rr_ge_22", "호흡수 22회/분 이상"),
            Item("altered_mentation", "의식 변화"),
            Item("sbp_le_100", "수축기 혈압 100 mmHg 이하"),
        ),
        thresholds=(
            Threshold(0, 1, "0–1", "qSOFA 낮음", "패혈증을 배제하지 말 것(단독 선별도구로 쓰지 않음)"),
            Threshold(2, 3, "2–3", "qSOFA 양성", "사망 위험 3–14배 → 패혈증 평가(젖산, 혈액배양, 장기 기능)"),
        ),
        citation=C_QSOFA, verification="primary",
        chronic_cutoff=1, min_age=18,
        applicability="Seymour 2016 abstract: adult encounters with suspected infection outside the ICU. "
        "Complaints lasting >=2 weeks without acute markers are not the acute suspected-infection encounter "
        "(our proxy).",
        note="Items and >=2 cut point from the PubMed abstract. Surviving Sepsis Campaign 2021 recommends "
        "AGAINST qSOFA as a single screening tool (see safety/protocols.py), so a low score never rules out sepsis.",
    ),
    Rule(
        id="curb65", short="CURB-65", name_ko="CURB-65 폐렴 중증도 점수", name_en="CURB-65",
        purpose="지역사회획득 폐렴의 30일 사망 위험 분류",
        population="지역사회획득 폐렴으로 진단되었거나 의심되는 성인",
        categories=("fever", "dyspnea"),
        keywords=("폐렴", "pneumonia"),
        items=(
            Item("confusion", "새로 생긴 의식 혼란"),
            Item("urea_gt_7", "혈중 요소 7 mmol/L 초과(BUN 약 19 mg/dL 초과)"),
            Item("rr_ge_30", "호흡수 30회/분 이상"),
            Item("low_bp", "수축기 혈압 90 mmHg 미만 또는 이완기 혈압 60 mmHg 이하"),
            Item("age_ge_65", "나이 65세 이상"),
        ),
        thresholds=(
            Threshold(0, 1, "0–1", "저위험", "30일 사망률 0.7–3.2%"),
            Threshold(2, 2, "2", "중간 위험", "30일 사망률 3%"),
            Threshold(3, 5, "3–5", "고위험", "30일 사망률 17–57% → 중증 폐렴으로 관리"),
        ),
        citation=C_CURB65, verification="primary",
        requires_any=("폐렴", "pneumonia", "기침", "가래", "객담", "cough", "sputum", "phlegm"),
        chronic_cutoff=1, min_age=18,
        applicability="Lim 2003 abstract: adults hospitalised with community-acquired pneumonia. Requires a "
        "respiratory infection signal (cough/sputum/pneumonia) on top of fever or dyspnea; >=2-week courses "
        "(TB, bronchiectasis, ILD) and children excluded (our proxy).",
        note="Items and per-score 30-day mortality (0:0.7, 1:3.2, 2:3, 3:17, 4:41.5, 5:57%) from the abstract. "
        "The low/intermediate/high labels are our grouping of those numbers. BUN conversion is ours (urea x 2.8).",
    ),
    Rule(
        id="centor", short="Centor", name_ko="Centor 인두염 점수", name_en="Centor criteria",
        purpose="성인 인후통에서 A군 사슬알균 인두염 확률 추정",
        population="인후통으로 온 성인",
        categories=(),
        keywords=("인후통", "목이 따", "목 따가", "목구멍", "편도", "삼키기", "sore throat", "pharyngitis", "tonsil"),
        items=(
            Item("tonsillar_exudate", "편도 삼출물"),
            Item("tender_anterior_nodes", "앞목 림프절 비대·압통"),
            Item("no_cough", "기침 없음"),
            Item("fever_history", "발열 병력"),
        ),
        thresholds=(
            Threshold(0, 0, "0", "매우 낮음", "배양 양성 확률 2.5%"),
            Threshold(1, 1, "1", "낮음", "배양 양성 확률 6.5%"),
            Threshold(2, 2, "2", "중간", "배양 양성 확률 15%"),
            Threshold(3, 3, "3", "높음", "배양 양성 확률 32%"),
            Threshold(4, 4, "4", "매우 높음", "배양 양성 확률 56%"),
        ),
        citation=C_CENTOR, verification="primary",
        chronic_cutoff=1, min_age=15,
        applicability="Centor 1981 abstract: adult ER patients complaining of sore throat. Age cut 15 and the "
        ">=2-week exclusion are ours.",
        note="4 variables and per-count culture-positive probabilities from the PubMed abstract. The McIsaac "
        "age modification was not implemented because its criteria could not be read in the primary source.",
    ),
    Rule(
        id="ottawa_sah", short="Ottawa SAH", name_ko="오타와 지주막하 출혈 규칙", name_en="Ottawa SAH Rule",
        purpose="급성 두통 환자에서 지주막하 출혈 배제",
        population="의식 명료한 성인, 새로 생긴 심한 비외상성 두통이 1시간 이내 최고조, 신경학적 결손 없음",
        categories=("headache",),
        keywords=("벼락", "thunderclap", "지주막하", "subarachnoid"),
        items=(
            Item("age_ge_40", "나이 40세 이상"),
            Item("neck_pain_stiffness", "목 통증 또는 목 뻣뻣함"),
            Item("witnessed_loc", "목격된 의식 소실"),
            Item("onset_exertion", "힘쓰는 중(운동, 배변, 성관계 등) 발생"),
            Item("thunderclap", "벼락두통(순간적으로 최고 강도 도달)"),
            Item("limited_neck_flexion", "진찰상 목 굴곡 제한"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "규칙 음성", "지주막하 출혈 배제 가능(적용 대상군에 한함)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "규칙 양성", "배제 불가 → 비조영 뇌 CT 등 추가 검사"),
        ),
        citation=C_OTTAWA_SAH, verification="primary",
        excludes_any=_TRAUMA + _RECURRENT, chronic_cutoff=1, min_age=16,
        applicability="Perry 2013 abstract: adults with acute non-traumatic headache peaking within 1 h and a "
        "normal neurologic exam; findings 'apply only to patients with these specific characteristics'. "
        "Excluded here: trauma, recurrent/usual headaches (not a new headache), complaints lasting >=2 weeks, "
        "age <16 ('older than 15' per the full text; reviewer knowledge, not in the abstract).",
        note="6 criteria, population, and 100% sensitivity / 15.3% specificity from the PubMed abstract.",
    ),
    Rule(
        id="cchr", short="Canadian CT Head", name_ko="캐나다 두부 CT 규칙", name_en="Canadian CT Head Rule",
        purpose="경미한 두부 외상에서 뇌 CT 필요 여부 판단",
        population="GCS 13–15의 경미한 두부 외상 성인(목격된 의식 소실, 기억 상실 또는 지남력 장애)",
        categories=(),
        keywords=("머리를 부딪", "머리를 박", "머리를 다쳤", "머리 부상", "두부 외상", "머리 외상", "넘어지면서 머리",
                  "head injury", "head trauma", "hit his head", "hit her head", "hit my head"),
        method="tiers",
        groups=(("medium", "중등도 위험 인자"), ("high", "고위험 인자")),
        items=(
            Item("gcs_lt15_2h", "외상 2시간 후에도 GCS 15 미만", group="high"),
            Item("open_depressed_fx", "개방성 또는 함몰 두개골 골절 의심", group="high"),
            Item("basal_fx_signs", "두개저 골절 징후(고막 뒤 혈종, 너구리 눈, 뇌척수액 이루·비루, 귀 뒤 멍)",
                 group="high"),
            Item("vomiting_ge2", "2회 이상 구토", group="high"),
            Item("age_ge_65", "나이 65세 이상", group="high"),
            Item("amnesia_gt30", "충격 전 30분 이상의 기억 상실", group="medium"),
            Item("dangerous_mechanism", "위험한 손상 기전(보행 중 차량 충돌, 차량 밖으로 튕겨남, 높은 곳에서 추락)",
                 group="medium"),
        ),
        thresholds=(
            Threshold(0, 0, "해당 없음", "저위험", "규칙상 CT 불필요(적용 대상군에 한함)", rule_out=True),
            Threshold(1, 1, "중등도 인자", "중등도 위험", "임상적으로 중요한 뇌손상 가능 → 뇌 CT"),
            Threshold(2, 2, "고위험 인자", "고위험", "신경외과적 중재 필요 가능 → 뇌 CT"),
        ),
        citation=C_CCHR, verification="secondary",
        chronic_cutoff=1, min_age=16,
        applicability="Stiell 2001 abstract: adults with GCS 13-15 after (minor) head injury presenting to the "
        "ED. Injuries >=2 weeks old excluded; age 16 cut-off is reviewer knowledge (abstract: 'adults').",
        note="5 high-risk + 2 medium-risk factors from the PubMed abstract. The abstract text renders "
        "vomiting as '>2 episodes' and age '>65'; the rule is widely published as >=2 and >=65, which we use. "
        "Basal-fracture signs and the dangerous-mechanism definition are not in the abstract.",
    ),
    Rule(
        id="abcd2", short="ABCD2", name_ko="ABCD2 점수", name_en="ABCD2 score",
        purpose="일과성 허혈 발작(TIA) 후 조기 뇌졸중 위험 분류",
        population="일과성 허혈 발작이 의심되는 환자(증상 소실)",
        categories=("neuro",),
        keywords=("일과성", "tia "),
        items=(
            Item("age_ge_60", "나이 60세 이상"),
            Item("bp_ge_140_90", "첫 혈압 140/90 mmHg 이상"),
            Item("clinical", "임상 양상",
                 options=(("해당 없음", 0), ("위약 없는 언어장애", 1), ("한쪽 위약", 2))),
            Item("duration", "증상 지속 시간", options=(("10분 미만", 0), ("10–59분", 1), ("60분 이상", 2))),
            Item("diabetes", "당뇨병"),
        ),
        thresholds=(
            Threshold(0, 3, "0–3", "저위험", "2일 내 뇌졸중 위험 1.0%"),
            Threshold(4, 5, "4–5", "중간 위험", "2일 내 뇌졸중 위험 4.1%"),
            Threshold(6, 7, "6–7", "고위험", "2일 내 뇌졸중 위험 8.1% → 즉시 평가"),
        ),
        citation=C_ABCD2, verification="primary",
        requires_any=("일과성", "tia", "돌아왔", "돌아옴", "회복", "사라졌", "없어졌", "풀렸", "좋아졌", "호전",
                      "transient", "resolved"),
        chronic_cutoff=1,
        applicability="Johnston 2007 abstract: patients diagnosed with TIA (symptoms resolved). Requires a "
        "focal-deficit complaint (neuro category) that resolved or is called TIA; >=2 weeks excluded.",
        note="All items, points and bands (2-day risk 1.0 / 4.1 / 8.1%) from the PubMed abstract.",
    ),
    Rule(
        id="alvarado", short="Alvarado", name_ko="Alvarado 충수염 점수", name_en="Alvarado score",
        purpose="급성 충수염 가능성 추정",
        population="충수염이 의심되는 복통 환자",
        categories=("abdominal_pain",),
        keywords=("충수", "맹장", "appendic", "오른쪽 아랫배", "오른쪽 하복부", "우하복부", "우측 하복부",
                  "right lower", "rlq"),
        items=(
            Item("migration", "통증이 우하복부로 이동"),
            Item("anorexia", "식욕부진(또는 소변 케톤)"),
            Item("nausea_vomiting", "오심·구토"),
            Item("rlq_tenderness", "우하복부 압통", 2.0),
            Item("rebound", "반발 압통"),
            Item("fever", "체온 상승(37.3°C 초과)"),
            Item("leukocytosis", "백혈구 증가(10,000/µL 초과)", 2.0),
            Item("left_shift", "호중구 좌방이동"),
        ),
        thresholds=(
            Threshold(0, 4, "0–4", "가능성 낮음", "충수염 가능성 낮음(다른 원인 평가)"),
            Threshold(5, 6, "5–6", "부합", "충수염과 부합 → 관찰·영상검사"),
            Threshold(7, 8, "7–8", "가능성 높음", "충수염 가능성 높음 → 외과 협진"),
            Threshold(9, 10, "9–10", "가능성 매우 높음", "충수염 가능성 매우 높음 → 외과 협진"),
        ),
        citation=C_ALVARADO, verification="secondary",
        excludes_any=_NON_RLQ, chronic_cutoff=1,
        applicability="Alvarado 1986 abstract: abdominal pain suggestive of acute appendicitis. Generic or "
        "right-lower-quadrant pain only: upper/left/flank locations are excluded unless RLQ/appendix words are "
        "also present; >=2-week pain excluded (not acute).",
        note="The abstract confirms the 8 factors and their weight order (RLQ tenderness and leukocytosis "
        "highest). Original full text paywalled. 2026-09-29, open-access full texts citing the original: items "
        "with anorexia or urine ketones and temperature > 37.3 C, and 4-6 / 7-8 / 9-10 = compatible / probable / "
        "very probable (Favara 2022, PMC9524677); 2 points for RLQ tenderness and for leukocytosis > 10,000, 1 for "
        "the others (Nasiri 2012 Table 1, PMC3410771, modified score without left shift); risk strata 1-4 / 5-6 / "
        "7-10 and cut-points 5 (rule out) and 7 (rule in, poor specificity) (Ohle 2011 systematic review, "
        "PMC3299622). Our bands (0-4 low, 5-6, 7-8, 9-10) follow these; secondary.",
    ),
    Rule(
        id="bisap", short="BISAP", name_ko="BISAP 급성 췌장염 중증도 점수",
        name_en="Bedside Index for Severity in Acute Pancreatitis",
        purpose="급성 췌장염의 병원 내 사망 위험 조기 예측",
        population="급성 췌장염으로 진단된 환자(첫 24시간)",
        categories=(),
        keywords=("췌장염", "pancreatitis", "리파아제", "lipase", "아밀라아제", "amylase"),
        items=(
            Item("bun_gt_25", "BUN 25 mg/dL 초과"),
            Item("impaired_mental", "의식 장애"),
            Item("sirs", "전신염증반응증후군(SIRS) 해당"),
            Item("age_gt_60", "나이 60세 초과"),
            Item("pleural_effusion", "흉수"),
        ),
        thresholds=(
            Threshold(0, 0, "0", "최저 위험", "병원 내 사망률 1% 미만"),
            Threshold(1, 4, "1–4", "중간", "점수가 높을수록 사망 위험 증가"),
            Threshold(5, 5, "5", "최고 위험", "병원 내 사망률 20% 초과"),
        ),
        citation=C_BISAP, verification="primary",
        chronic_cutoff=2,
        applicability="Wu 2008 abstract: acute pancreatitis, first 24 h. Chronic pancreatitis / months-long "
        "complaints excluded.",
        note="5 variables and the <1% / >20% range from the abstract. The commonly used '>=3 = severe' cut "
        "point is from later validation papers and is deliberately not encoded here.",
    ),
    # ---------------------------------------------------------------------------------------- 2026-09-27
    Rule(
        id="pecarn_head_lt2", short="PECARN 두부(2세 미만)", name_ko="PECARN 소아 두부 외상 규칙(2세 미만)",
        name_en="PECARN pediatric head trauma rule, age <2 years",
        purpose="2세 미만 두부 외상에서 뇌 CT 필요 여부 판단",
        population="외상 24시간 이내, GCS 14–15인 2세 미만 소아",
        categories=(),
        keywords=_HEAD_INJURY,
        method="tiers",
        groups=(("intermediate", "중간 위험 인자"), ("high", "고위험 인자")),
        items=(
            Item("ams_gcs", "GCS 14 이하 또는 의식 변화(보챔, 처짐, 반응 느림)", group="high"),
            Item("palpable_fx", "만져지는 두개골 골절", group="high"),
            Item("nonfrontal_hematoma", "이마 외 부위(뒤통수·정수리·옆머리) 두피 혈종", group="intermediate"),
            Item("loc_ge5s", "5초 이상 의식 소실", group="intermediate"),
            Item("severe_mechanism", "심한 손상 기전(차량 사고 중 튕겨나감·전복·동승자 사망, 헬멧 없이 차량에 치임, "
                 "0.9 m 초과 추락, 빠른 물체에 머리 맞음)", group="intermediate"),
            Item("not_acting_normally", "보호자가 보기에 평소와 다르게 행동", group="intermediate"),
        ),
        thresholds=(
            Threshold(0, 0, "해당 없음", "매우 저위험", "CT 불필요(적용 대상군에 한함)", rule_out=True),
            Threshold(1, 1, "중간 위험 인자", "중간 위험", "관찰 또는 뇌 CT 중 임상 판단"),
            Threshold(2, 2, "고위험 인자", "고위험", "뇌 CT 권고"),
        ),
        citation=C_PECARN_HEAD, verification="secondary",
        max_age=2, chronic_cutoff=1,
        applicability="Kuppermann 2009 abstract: children younger than 18 years within 24 h of head trauma with "
        "GCS 14-15; separate rule for <2 years. Needs a head-injury phrase and a stated age <2 years or an "
        "infant/child word; >=2-week-old injuries excluded (our proxy for 'within 24 h').",
        note="The 6 very-low-risk criteria for <2 years are in the PubMed abstract (NPV 100%, sensitivity 100% in "
        "validation). The split into high-risk (CT) vs intermediate (observation vs CT) factors and the severe-"
        "mechanism definition (fall >0.9 m) come from the paper's figure, cross-checked in open-access reviews "
        "(PMC13328899, PMC13335041) because the full text is paywalled.",
    ),
    Rule(
        id="pecarn_head_ge2", short="PECARN 두부(2세 이상)", name_ko="PECARN 소아 두부 외상 규칙(2–17세)",
        name_en="PECARN pediatric head trauma rule, age >=2 years",
        purpose="2–17세 두부 외상에서 뇌 CT 필요 여부 판단",
        population="외상 24시간 이내, GCS 14–15인 2–17세 소아",
        categories=(),
        keywords=_HEAD_INJURY,
        method="tiers",
        groups=(("intermediate", "중간 위험 인자"), ("high", "고위험 인자")),
        items=(
            Item("ams_gcs", "GCS 14 이하 또는 의식 변화(처짐, 같은 질문 반복, 반응 느림)", group="high"),
            Item("basilar_fx_signs", "두개저 골절 징후(고막 뒤 혈종, 너구리 눈, 귀 뒤 멍, 뇌척수액 이루·비루)",
                 group="high"),
            Item("any_loc", "의식 소실(시간 무관)", group="intermediate"),
            Item("vomiting", "구토", group="intermediate"),
            Item("severe_mechanism", "심한 손상 기전(차량 사고 중 튕겨나감·전복·동승자 사망, 헬멧 없이 차량에 치임, "
                 "1.5 m 초과 추락, 빠른 물체에 머리 맞음)", group="intermediate"),
            Item("severe_headache", "심한 두통", group="intermediate"),
        ),
        thresholds=(
            Threshold(0, 0, "해당 없음", "매우 저위험", "CT 불필요(적용 대상군에 한함)", rule_out=True),
            Threshold(1, 1, "중간 위험 인자", "중간 위험", "관찰 또는 뇌 CT 중 임상 판단"),
            Threshold(2, 2, "고위험 인자", "고위험", "뇌 CT 권고"),
        ),
        citation=C_PECARN_HEAD, verification="secondary",
        min_age=2, max_age=18, chronic_cutoff=1,
        applicability="Kuppermann 2009 abstract: children younger than 18 years within 24 h of head trauma with "
        "GCS 14-15; separate rule for >=2 years. Needs a head-injury phrase and a stated age 2-17 or a child "
        "word (with no age, both PECARN variants apply); >=2-week-old injuries excluded (our proxy).",
        note="The 6 very-low-risk criteria for >=2 years are in the PubMed abstract (NPV 99.95%, sensitivity 96.8% "
        "in validation). High vs intermediate split and the severe-mechanism definition (fall >1.5 m) from the "
        "paper's figure via open-access reviews (PMC13328899, PMC13335041).",
    ),
    Rule(
        id="nexus", short="NEXUS", name_ko="NEXUS 경추 영상 기준", name_en="NEXUS low-risk criteria (C-spine)",
        purpose="둔상 환자에서 경추 영상검사 없이 경추 손상 배제",
        population="둔상 후 경추 손상이 의심되는 환자",
        categories=(),
        keywords=_NECK,
        items=(
            Item("midline_tenderness", "경추 정중선(뒷목 가운데 뼈) 압통"),
            Item("focal_neuro", "국소 신경학적 결손"),
            Item("altered_alertness", "의식·각성 저하"),
            Item("intoxication", "음주·약물 중독 상태"),
            Item("distracting_injury", "주의를 분산시키는 심한 통증의 다른 손상"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "저위험", "경추 영상검사 불필요(적용 대상군에 한함)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "배제 불가", "경추 영상검사(CT 등)"),
        ),
        citation=C_NEXUS, verification="primary",
        requires_any=_BLUNT_TRAUMA, chronic_cutoff=1,
        applicability="Hoffman 2000 abstract: 34,069 patients who underwent C-spine radiography after blunt "
        "trauma. Needs a neck word and a trauma word (Korean '목이 아파요' alone means sore throat); >=2-week "
        "complaints excluded (our proxy for the acute ED visit).",
        note="All 5 criteria (no midline tenderness, no focal deficit, normal alertness, no intoxication, no "
        "painful distracting injury) and sensitivity 99.0% / NPV 99.8% from the PubMed abstract. Items are "
        "phrased as risk-present, so score = number of failed criteria.",
    ),
    Rule(
        id="ccsr", short="Canadian C-spine", name_ko="캐나다 경추 규칙", name_en="Canadian C-Spine Rule",
        purpose="의식 명료하고 활력징후가 안정된 외상 성인에서 경추 영상검사 필요 여부 판단",
        population="머리·목 둔상 후 GCS 15, 활력징후 안정된 16세 이상",
        categories=(),
        keywords=_NECK,
        items=(
            Item("age_ge_65", "나이 65세 이상(고위험)"),
            Item("dangerous_mechanism", "위험한 손상 기전(고위험: 높은 곳 추락, 고속 차량 사고·전복·튕겨나감 등)"),
            Item("paresthesia", "팔다리 감각 이상(고위험)"),
            Item("no_low_risk_factor", "저위험 인자가 하나도 없음(단순 후방 추돌, 응급실에서 앉아 있음, 수상 후 걸어 다님, "
                 "목 통증이 나중에 시작, 경추 정중선 압통 없음)"),
            Item("cannot_rotate", "목을 좌우 45도 능동 회전 불가(저위험 인자가 있을 때만 확인)"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "영상 불필요", "경추 영상검사 불필요(적용 대상군에 한함)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "영상 필요", "경추 영상검사"),
        ),
        citation=C_CCSR, verification="primary",
        requires_any=_BLUNT_TRAUMA, chronic_cutoff=1, min_age=16,
        applicability="Stiell 2001 abstract: adults with blunt trauma to the head/neck, stable vital signs and "
        "GCS 15. Needs a neck word and a trauma word; age <16 / child words and >=2-week complaints excluded "
        "(16 is reviewer knowledge; abstract says 'adults').",
        note="The 3 questions (high-risk factors age >=65 / dangerous mechanism / paresthesias; low-risk "
        "factors allowing range-of-motion testing; active 45-degree rotation) and 100% sensitivity from the "
        "PubMed abstract. Flattened into 5 risk-present items: any one = imaging (equivalent to the 3-step "
        "flow). Dangerous-mechanism examples are reviewer knowledge.",
    ),
    Rule(
        id="sfsr", short="SF Syncope", name_ko="샌프란시스코 실신 규칙", name_en="San Francisco Syncope Rule",
        purpose="실신 환자의 7일 내 중대한 결과(사망, 부정맥, 심근경색, 폐색전증, 출혈 등) 위험 선별",
        population="응급실에 온 실신·실신 전 증상 환자",
        categories=("syncope",),
        keywords=(),
        items=(
            Item("chf_history", "울혈성 심부전 병력"),
            Item("hct_lt_30", "헤마토크릿 30% 미만"),
            Item("abnormal_ecg", "심전도 이상"),
            Item("dyspnea", "숨참 호소"),
            Item("sbp_lt_90", "수축기 혈압 90 mmHg 미만"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "저위험", "단기 중대한 결과 위험 낮음(단독 배제 도구로 쓰지 말 것)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "고위험", "중대한 원인 평가·입원 고려"),
        ),
        citation=C_SFSR, verification="primary",
        excludes_any=_SEIZURE_INTOX, chronic_cutoff=1,
        applicability="Quinn 2004 abstract: ED patients presenting with syncope or near syncope. Seizure, "
        "intoxication and >=2-week recurrent courses excluded (our proxy for the index ED visit).",
        note="5 predictors (CHESS) and 96% sensitivity / 62% specificity from the PubMed abstract (derivation "
        "study; later external validations reported lower sensitivity, hence the caution in the low band).",
    ),
    Rule(
        id="csrs", short="Canadian Syncope", name_ko="캐나다 실신 위험 점수", name_en="Canadian Syncope Risk Score",
        purpose="실신 후 30일 내 중대한 사건(부정맥, 심근경색, 구조적 심질환, 폐색전증, 출혈, 사망) 위험 분류",
        population="실신 24시간 이내 응급실에 온 16세 이상",
        categories=("syncope",),
        keywords=(),
        items=(
            Item("vasovagal_predisposition", "미주신경성 소인(덥고 붐비는 곳, 오래 서 있기, 공포·감정·통증으로 유발)",
                 -1.0),
            Item("heart_disease", "심장질환 병력(관상동맥질환, 심방세동·조동, 심부전, 판막질환)"),
            Item("sbp_abnormal", "응급실 수축기 혈압 90 미만 또는 180 mmHg 초과", 2.0),
            Item("troponin_high", "트로포닌 상승(정상 99백분위수 초과)", 2.0),
            Item("qrs_axis", "QRS 축 이상(-30도 미만 또는 100도 초과)"),
            Item("qrs_gt_130", "QRS 폭 130 ms 초과"),
            Item("qtc_gt_480", "QTc 480 ms 초과", 2.0),
            Item("ed_diagnosis", "응급실 판단", options=(("판단 보류", 0), ("미주신경성 실신", -2), ("심장성 실신", 2))),
        ),
        thresholds=(
            Threshold(-3, -2, "−3~−2", "매우 저위험", "30일 중대한 사건 위험 매우 낮음"),
            Threshold(-1, 0, "−1~0", "저위험", "대개 귀가 가능"),
            Threshold(1, 3, "1–3", "중간 위험", "추가 평가·관찰"),
            Threshold(4, 5, "4–5", "고위험", "입원 관찰 고려"),
            Threshold(6, 11, "6–11", "매우 고위험", "입원·심장 감시"),
        ),
        citation=C_CSRS, verification="primary",
        excludes_any=_SEIZURE_INTOX, chronic_cutoff=1, min_age=16,
        applicability="Thiruganasambandamoorthy 2016 abstract: adults (>=16 y) with syncope presenting within "
        "24 h. Seizure/intoxication and >=2-week courses excluded (our proxy for 'within 24 h').",
        note="9 predictors and the -3..11 range (0.4% to 83.6% 30-day risk) from the PubMed abstract. "
        "2026-09-29: original full text read (CMAJ, PMC5008955): points = shrinkage-corrected coefficients (Table 4) "
        "divided by the smallest and rounded, which gives vasovagal predisposition -1, heart disease +1, SBP <90 or "
        ">180 +2, troponin +2, abnormal QRS axis +1, QRS >130 ms +1, QTc >480 ms +2 (0.90/0.48 = 1.9), ED "
        "vasovagal -2, ED cardiac +2 and the published -3..11 range; the paper groups <= -2 very low (<1%), -1..3 "
        "low-medium (1-8%), >= 4 high/very high (>12%). The 5-band split used here is from the 2020 validation "
        "(via PMC12591636 Table 1). QRS axis: abstract '> 100°', Table 4 '> 110°'; the abstract value is kept.",
    ),
    Rule(
        id="gbs", short="Glasgow-Blatchford", name_ko="Glasgow-Blatchford 상부위장관 출혈 점수",
        name_en="Glasgow-Blatchford bleeding score",
        purpose="상부위장관 출혈에서 치료(수혈·내시경 지혈·수술) 필요 위험 분류",
        population="토혈·흑색변 등 상부위장관 출혈로 온 성인(내시경 전)",
        categories=(),
        keywords=_UGIB,
        items=(
            Item("urea", "혈중 요소(mmol/L; 괄호는 BUN mg/dL)",
                 options=(("6.5 미만(<18)", 0), ("6.5–7.9(18–22)", 2), ("8–9.9(22–28)", 3), ("10–24.9(28–70)", 4),
                          ("25 이상(≥70)", 6))),
            Item("hemoglobin", "헤모글로빈(g/dL)",
                 options=(("남 13 이상·여 12 이상", 0), ("남 12–12.9·여 10–11.9", 1), ("남 10–11.9", 3),
                          ("10 미만", 6))),
            Item("sbp", "수축기 혈압(mmHg)", options=(("110 이상", 0), ("100–109", 1), ("90–99", 2), ("90 미만", 3))),
            Item("pulse_ge_100", "맥박 100회/분 이상"),
            Item("melena", "흑색변"),
            Item("syncope", "실신", 2.0),
            Item("hepatic_disease", "간질환", 2.0),
            Item("cardiac_failure", "심부전", 2.0),
        ),
        thresholds=(
            Threshold(0, 0, "0", "저위험", "외래 관리 고려(Stanley 2009)", rule_out=True),
            Threshold(1, 23, "1 이상", "치료 필요 가능", "입원·조기 내시경 평가(점수 높을수록 위험)"),
        ),
        citation=C_GBS, verification="secondary",
        veto_any=("객혈", "기침할 때", "기침하면서", "기침하다", "hemoptysis", "coughing up"), chronic_cutoff=1, min_age=16,
        applicability="Blatchford 2000 abstract: patients admitted for upper-GI haemorrhage (UK adults). Needs "
        "hematemesis/melena words; hemoptysis vetoed; age <16 (our proxy for the adult cohort) and >=2-week "
        "courses excluded.",
        note="The 8 variables are in the Blatchford 2000 abstract; point values are not (full text paywalled) and "
        "were cross-checked in two open-access tables (PMC13544648, PMC12522279). The score-0 low-risk band is "
        "from Stanley 2009 Lancet abstract (PMID 19091393). Later guidelines use <=1; not encoded.",
    ),
    Rule(
        id="kocher", short="Kocher", name_ko="Kocher 기준(소아 고관절 화농성 관절염)",
        name_en="Kocher criteria",
        purpose="소아 급성 고관절 통증에서 화농성 관절염과 일과성 활막염 구별",
        population="급성으로 고관절을 아파하거나 절뚝이는 소아",
        categories=(),
        keywords=_HIP_LIMP,
        items=(
            Item("fever_history", "발열 병력"),
            Item("non_weight_bearing", "아픈 다리에 체중을 싣지 못함"),
            Item("esr_ge_40", "적혈구 침강 속도 40 mm/h 이상"),
            Item("wbc_gt_12000", "백혈구 12,000/µL 초과"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "매우 낮음", "화농성 관절염 확률 0.2% 미만"),
            Threshold(1, 1, "1개", "낮음", "화농성 관절염 확률 3%"),
            Threshold(2, 2, "2개", "중간", "화농성 관절염 확률 40% → 관절 초음파·천자 고려"),
            Threshold(3, 3, "3개", "높음", "화농성 관절염 확률 93% → 관절 천자"),
            Threshold(4, 4, "4개", "매우 높음", "화농성 관절염 확률 99.6% → 관절 천자"),
        ),
        citation=C_KOCHER, verification="primary",
        veto_any=_TRAUMA + ("넘어진", "넘어졌", "골절", "삐었", "fracture", "sprain"), chronic_cutoff=1, max_age=18,
        applicability="Kocher 1999 abstract: children with an acutely irritable hip. Needs a hip/limp word and a "
        "stated age <18 or a child word; trauma (affirmed) and >=2-week courses excluded (our proxy for 'acute').",
        note="4 predictors and per-count probabilities (<0.2 / 3.0 / 40.0 / 93.1 / 99.6%) from the PubMed "
        "abstract. The original uses a history of fever; the later >38.5 C modification (Caird 2006) is not "
        "encoded.",
    ),
    Rule(
        id="pas", short="PAS", name_ko="소아 충수염 점수", name_en="Pediatric Appendicitis Score",
        purpose="소아 급성 복통에서 충수염 가능성 추정",
        population="1–17세, 7일 미만 급성 복통 소아",
        categories=("abdominal_pain",),
        keywords=("충수", "맹장", "appendic", "오른쪽 아랫배", "우하복부", "right lower", "rlq"),
        items=(
            Item("cough_hop_tenderness", "기침·타진·뛰기 시 우하복부 통증", 2.0),
            Item("anorexia", "식욕부진"),
            Item("fever", "발열(38°C 초과)"),
            Item("nausea_vomiting", "오심·구토"),
            Item("rlq_tenderness", "우하복부 압통", 2.0),
            Item("leukocytosis", "백혈구 10,000/µL 초과"),
            Item("neutrophilia", "호중구 7,500/µL 초과"),
            Item("migration", "통증이 우하복부로 이동"),
        ),
        thresholds=(
            Threshold(0, 2, "0–2", "가능성 낮음", "충수염 가능성 낮음(검증 연구에서 충수염의 2.4%만 해당)"),
            Threshold(3, 6, "3–6", "불확실", "관찰·초음파 등 추가 검사"),
            Threshold(7, 10, "7–10", "가능성 높음", "충수염 가능성 높음 → 외과 협진"),
        ),
        citation=C_PAS, verification="primary",
        chronic_cutoff=1, min_age=1, max_age=18,
        applicability="Samuel 2002 abstract: children 4-15 y with pain suggestive of appendicitis; Goldman 2008 "
        "(PMID 18534219) validated it in unselected children 1-17 y with abdominal pain <7 days, which we follow. "
        "Needs a stated age 1-17 or a child word; >=2-week pain excluded (our proxy for <7 days).",
        note="8 variables and points (2 for the two physical signs, total 10) from the Samuel abstract; the fever "
        ">38 C, WBC >10,000 and neutrophil >7,500 cut-offs and the <=2 / 3-6 / >=7 bands from the Goldman 2008 "
        "abstract.",
    ),
    Rule(
        id="spesi", short="sPESI", name_ko="간이 폐색전증 중증도 지수", name_en="simplified PESI",
        purpose="확진된 급성 폐색전증의 30일 사망 위험 분류(진단 도구 아님)",
        population="급성 증상성 폐색전증으로 확진된 환자",
        categories=(),
        keywords=_PE_CONFIRMED,
        items=(
            Item("age_gt_80", "나이 80세 초과"),
            Item("cancer", "암"),
            Item("cardiopulmonary", "만성 심폐질환(심부전 또는 만성 폐질환)"),
            Item("hr_ge_110", "심박수 110회/분 이상"),
            Item("sbp_lt_100", "수축기 혈압 100 mmHg 미만"),
            Item("sao2_lt_90", "산소포화도 90% 미만"),
        ),
        thresholds=(
            Threshold(0, 0, "0", "저위험", "30일 사망률 1.0%", rule_out=True),
            Threshold(1, 6, "1 이상", "고위험", "30일 사망률 10.9% → 입원·우심실 평가"),
        ),
        citation=C_SPESI, verification="secondary",
        min_age=18,
        applicability="Jimenez 2010 abstract: patients with acute symptomatic PE (derivation outpatients, RIETE "
        "validation). Only when PE is stated as diagnosed/confirmed, so it never competes with Wells/PERC "
        "(which are vetoed by the same phrases). Adults only (our proxy).",
        note="The 6 variables and 30-day mortality 1.0% (score 0) vs 10.9% (>=1) are in the PubMed abstract; the "
        "cut-offs (age >80, HR >=110, SBP <100, SaO2 <90%) are not and were cross-checked in open-access papers "
        "(PMC13302099, PMC11681475). Full text paywalled.",
    ),
    Rule(
        id="pecarn_febrile_infant", short="PECARN 발열 영아", name_ko="PECARN 발열 영아 저위험 규칙",
        name_en="PECARN febrile infant rule",
        purpose="생후 60일 이하 발열 영아에서 중증 세균 감염(요로감염, 균혈증, 세균성 뇌수막염) 저위험군 식별",
        population="이전에 건강했던 생후 60일 이하 발열 영아",
        categories=("fever",),
        keywords=(),
        items=(
            Item("ua_positive", "소변검사 이상(백혈구 에스테라제·아질산염 양성 또는 소변 백혈구 증가)"),
            Item("anc_gt_4090", "절대 호중구 수 4,090/µL 초과"),
            Item("pct_gt_1_71", "프로칼시토닌 1.71 ng/mL 초과"),
        ),
        thresholds=(
            Threshold(0, 0, "0개", "저위험", "중증 세균 감염 가능성 낮음(음성 예측도 99.6%)", rule_out=True),
            Threshold(1, _INF, "1개 이상", "저위험 아님", "혈액·소변 배양, 뇌척수액 검사 고려, 입원·항생제"),
        ),
        citation=C_PECARN_FI, verification="primary",
        max_age_days=60,
        applicability="Kuppermann 2019 abstract: previously healthy febrile infants 60 days and younger. Needs "
        "fever and an infant age <=60 days ('생후 N일/주/개월' or '신생아'); '영아' alone (up to 1 year) is not enough.",
        note="3 criteria (negative urinalysis, ANC <=4090/uL, procalcitonin <=1.71 ng/mL) and NPV 99.6% / "
        "sensitivity 97.7% from the PubMed abstract. Items phrased as risk-present. The urinalysis definition "
        "is reviewer knowledge. Rochester / Step-by-Step were not encoded (criteria not in accessible abstracts).",
    ),
    Rule(
        id="mcisaac", short="McIsaac", name_ko="McIsaac 점수(수정 Centor)", name_en="McIsaac (modified Centor) score",
        purpose="인후통에서 A군 사슬알균 인두염 확률 추정(나이 보정)",
        population="새로 생긴 상기도 감염·인후통으로 온 3세 이상",
        categories=(),
        keywords=_SORE_THROAT,
        items=(
            Item("fever_gt_38", "체온 38°C 초과"),
            Item("no_cough", "기침 없음"),
            Item("tender_anterior_nodes", "앞목 림프절 비대·압통"),
            Item("tonsil_swelling_exudate", "편도 부종 또는 삼출물"),
            Item("age", "나이", options=(("3–14세", 1), ("15–44세", 0), ("45세 이상", -1))),
        ),
        thresholds=(
            Threshold(-1, 0, "0 이하", "매우 낮음", "A군 사슬알균 양성 8%"),
            Threshold(1, 1, "1", "낮음", "양성 14%"),
            Threshold(2, 2, "2", "중간", "양성 23% → 신속항원검사·배양"),
            Threshold(3, 3, "3", "높음", "양성 37% → 신속항원검사·배양"),
            Threshold(4, 5, "4 이상", "매우 높음", "양성 55%"),
        ),
        citation=C_MCISAAC, verification="secondary",
        chronic_cutoff=1, min_age=3,
        applicability="McIsaac 1998 abstract: patients aged 3 to 76 with a new upper respiratory infection "
        "(sore throat). Age <3 and >=2-week complaints excluded.",
        note="Abstract confirms the 0-4 score and the age-appropriate design but not the items. Items and age "
        "points (3-14 +1, 15-44 0, >=45 -1) cross-checked in open-access sources (PMC12731270 Table 1); per-score "
        "GAS-positive rates (<=0: 8, 1: 14, 2: 23, 3: 37, >=4: 55%) from the Fine 2012 validation abstract "
        "(PMID 22566485, 206,870 patients >=3 y). Original full text (scanned PDF) not accessible.",
    ),
)

RULES_BY_ID: dict[str, Rule] = {r.id: r for r in RULES}


# --------------------------------------------------------------------------------------------
# Lookup / rendering / scoring
# --------------------------------------------------------------------------------------------


def rules_for(text: str) -> list[Rule]:
    """Rules whose validated population matches the chief complaint (see Rule.applies_to)."""
    return [r for r in RULES if r.applies_to(text)]


def _item_text(item: Item) -> str:
    if item.options:
        return f"{item.text}(" + "/".join(f"{lab} {_fmt(p)}" for lab, p in item.options) + ")"
    return f"{item.text} {'+' if item.points >= 0 else ''}{_fmt(item.points)}"


def render_for_prompt(rules: list[Rule]) -> str:
    """Compact Korean text for a small LLM: items, thresholds, and short citation per rule."""
    if not rules:
        return ""
    lines = ["[임상 결정 규칙: 해당 항목을 문진·진찰·검사로 확인하고, 판단 근거로 규칙 이름과 출처를 인용]"]
    for r in rules:
        lines.append(f"■ {r.cite}: {r.purpose}. 대상: {r.population}")
        if r.method == "sum":
            lines.append("  항목: " + " / ".join(_item_text(i) for i in r.items))
        else:
            for gkey, glabel in r.groups:
                texts = [i.text for i in r.items if i.group == gkey]
                lines.append(f"  {glabel}: " + " / ".join(texts))
            if r.method == "groups":
                lines.append("  점수 = 하나라도 해당되는 범주 수(0–3)")
        bands = " · ".join(f"{t.range_text} {t.label}({t.meaning})" for t in r.thresholds)
        lines.append("  해석: " + bands)
        if r.secondary_thresholds:
            lines.append("  2단계 해석: " + " · ".join(
                f"{t.range_text} {t.label}({t.meaning})" for t in r.secondary_thresholds))
    return "\n".join(lines)


def _item_value(item: Item, value) -> float:
    if item.options:
        if isinstance(value, str):
            for lab, p in item.options:
                if lab == value:
                    return p
            raise ValueError(f"{item.key}: unknown option {value!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{item.key}: graded item needs option points or label, got {value!r}")
        if float(value) not in {float(p) for _, p in item.options}:
            raise ValueError(f"{item.key}: {value!r} is not one of the option points")
        return float(value)
    return item.points if value else 0.0


def _band(thresholds: tuple[Threshold, ...], s: float) -> Threshold | None:
    return next((t for t in thresholds if t.contains(s)), None)


def score(rule_id: str, answers: dict) -> ScoreResult:
    """Score a rule. answers: item key → bool (binary) or option points/label (graded).

    Unanswered items count as absent and are listed in `missing`. A rule-out band is only reported as
    a rule-out when every item was answered (complete=True); otherwise the label says it cannot rule out.
    """
    rule = RULES_BY_ID.get(rule_id)
    if rule is None:
        raise KeyError(f"unknown rule: {rule_id}")
    keys = {i.key for i in rule.items}
    unknown = set(answers) - keys
    if unknown:
        raise ValueError(f"{rule_id}: unknown item keys {sorted(unknown)}")

    values = {i.key: _item_value(i, answers[i.key]) for i in rule.items if i.key in answers}
    missing = tuple(i.key for i in rule.items if i.key not in answers)
    positives = tuple(k for k, v in values.items() if v > 0)

    if rule.method == "sum":
        s = float(sum(values.values()))
    else:
        hit = {i.group for i in rule.items if values.get(i.key, 0) > 0}
        if rule.method == "groups":
            s = float(len(hit))
        else:  # tiers: 1-based index of the highest positive tier
            s = float(max((idx + 1 for idx, (g, _) in enumerate(rule.groups) if g in hit), default=0))

    band = _band(rule.thresholds, s)
    label, meaning = (band.label, band.meaning) if band else ("범위 밖", "")
    complete = not missing
    if band is not None and band.rule_out and not complete:
        label = f"{band.label}(미확인 항목 있음: 배제 불가)"
        meaning = "확인하지 않은 항목이 있어 규칙으로 배제할 수 없음"
    sec = _band(rule.secondary_thresholds, s) if rule.secondary_thresholds else None
    return ScoreResult(
        rule_id=rule.id, score=s, max_score=rule.max_score, label=label, meaning=meaning, complete=complete,
        missing=missing, secondary_label=f"{sec.label}({sec.meaning})" if sec else "", citation=rule.cite,
        positives=positives,
    )


# Read-only, derived from the import-time lexicon (never modified): category concept sets for detect_categories()
_CATEGORY_CONCEPTS: dict[str, frozenset[str]] = _category_concepts()
_FOCAL_CONCEPTS: frozenset[str] = _CATEGORY_CONCEPTS.get("neuro", frozenset()) - _NON_FOCAL_NEURO
