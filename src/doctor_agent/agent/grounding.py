"""Grounding checker: is a finding the model wrote down actually something the environment said?

The small LLM sometimes records findings, or cites DDx evidence, that never appeared in the environment's responses.
This module checks claims against the case's own evidence only (initial info + environment responses of this case;
never the doctor's questions except as the context of a yes/no answer, never other cases). Pure code, CPU only, no
LLM calls.

Reading findings is delegated to the clinical-finding normalisation layer (`doctor_agent.nlp`, docs/nlp.md): every
environment response is parsed once into findings (concept, polarity, subject, value, ...), with the doctor's question
as context for answers to ASK turns (so "네, 좀 그래요" resolves the question's concepts), and the claim is parsed the
same way. Precision first — a claim is grounded only with clear support:
- Concept claims ("발열 없음", "우하복부 압통", "WBC 14,200"): every concept the claim names must be supported by an
  evidence finding with the same polarity and subject (a present child supports a present parent, an absent parent
  an absent child; `nlp.findings._supports`), numbers equal up to rounding. The claim's remaining words ("우측 상지"
  in "우측 상지 혈압 182/98") and numbers must appear in the supporting clause. A pain-quality/modifier concept
  (찢어지는, 등으로 뻗치는) must be supported in the same response as the symptom it qualifies (or the chief complaint).
- Claims naming no concept ("좌측 요골동맥 맥박 약함", "64세 남성", "트로포닌 21"): the literal words (Korean particles
  stripped; lab analyte names via the layer's analyte table) and numbers must sit in one comma part of the evidence,
  with the same polarity (the layer's cue rules) and a consistent direction (reference ranges in parentheses give
  the value's direction; reference numbers never ground).
- Sentences saying a result is not available ("결과가 제공되지 않습니다") never ground anything (dropped by the layer and
  by the literal reader alike). Uncertain or hypothetical evidence never supports.

What the layer lacks is kept here: the literal-word reader for concept-free claims, the strict number comparison, and
two guards on the layer's output (list negation only for bare list items; a present parent grounds a more specific
claim only when the claim's qualifier word is in the same comma part).
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field, replace

from doctor_agent.agent.text import similarity
from doctor_agent.nlp import LEXICON, Finding, normalize, parse
from doctor_agent.nlp import findings as _nlp  # the layer's rules are reused, never copied

# --- legacy synonym table: lexicon build input only -------------------------------------------------------------
# Not used by the checker any more (the lexicon holds these forms, provenance "grounding"). Kept only because
# scripts/build_lexicon.py:merge_grounding reads it; delete once the build takes it from data/lexicon/seed.tsv.
_PAIN = r"(?:아프|아파|아픈|아팠|통증|쑤시|쓰리|쓰려|결리|pain)"
_GROUPS: list[tuple[str, list[str], dict]] = [
    ("fever", ["발열", "고열", "미열", "열감", "열이", "열은", "열도", "열나", "열난", "열날", "fever", "febrile", "pyrexia",
               r"claim:(?<![가-힣])열(?![가-힣])"], {"measure": "temp", "cc": True}),
    ("chills", ["오한", "으슬", "chills", "rigor"], {}),
    ("dyspnea", ["호흡곤란", "숨참", "숨이차", "숨차", "숨이가쁘", "숨가쁘", "숨가쁨", "숨쉬기가힘", "숨쉬기힘", "숨이막",
                 "dyspnea", "shortness of breath", "sob", "re:숨이? ?[^\\n.,]{0,6}?(?:차|가빠|가쁘)"], {"cc": True}),
    ("chest_pain", ["흉통", "chest pain", "re:가슴[^\\n.,]{0,12}?" + _PAIN], {"cc": True}),
    ("chest_tight", ["흉부불편감", "가슴이답답", "가슴답답", "가슴이조이", "압박감", "chest tightness"], {"cc": True}),
    ("headache", ["두통", "headache", "re:머리[^\\n.,]{0,10}?(?:" + _PAIN[3:-1] + "|지끈)"], {"cc": True}),
    ("abd_pain", ["복통", "abdominal pain",
                  "re:(?:(?<![가-힣])배(?:가|는|도|를|쪽이?| )|윗배|아랫배|명치|복부)[^\\n.,]{0,10}?" + _PAIN], {"cc": True}),
    ("rlq", ["우하복부", "오른쪽아랫배", "오른쪽하복부", "rlq", "right lower quadrant"], {}),
    ("ruq", ["우상복부", "오른쪽윗배", "오른쪽상복부", "ruq", "right upper quadrant"], {}),
    ("llq", ["좌하복부", "왼쪽아랫배", "왼쪽하복부", "llq"], {}),
    ("luq", ["좌상복부", "왼쪽윗배", "왼쪽상복부", "luq"], {}),
    ("epigastric", ["상복부", "명치", "윗배", "epigastri"], {}),
    ("hypogastric", ["하복부", "아랫배", "hypogastri", "lower abdomen"], {}),
    ("right", ["우측", "오른쪽", "오른", "right", "rt"], {}),
    ("left", ["좌측", "왼쪽", "왼", "left", "lt"], {}),
    ("pain", ["통증", "아프", "아파", "아픈", "아팠", "쑤시", "쓰리", "쓰려", "결리", "pain"], {"optional": True, "cc": True}),
    ("tearing", ["찢어", "찢는", "찢기", "찢듯", "찢어지", "뜯기", "tearing", "ripping"], {}),
    ("squeezing", ["쥐어짜", "짓누르", "짓눌", "조이는", "squeez", "pressure-like"], {}),
    ("stabbing", ["찌르", "찌릿", "찌른", "stabbing", "sharp"], {}),
    ("burning", ["화끈", "타는듯", "타는것", "burning"], {}),
    ("sudden", ["갑자기", "급성발병", "갑작스", "sudden", "abrupt"], {}),
    ("radiation", ["방사", "뻗치", "뻗쳐", "뻗어", "퍼지", "radiat"], {}),
    ("nausea", ["구역", "메스꺼", "메슥", "울렁", "nausea"], {}),
    ("vomiting", ["구토", "토했", "토하", "토를", "토함", "게워", "vomit", "emesis"], {}),
    ("diarrhea", ["설사", "묽은변", "diarrhea"], {}),
    ("constipation", ["변비", "constipation"], {}),
    ("cough", ["기침", "cough"], {"cc": True}),
    ("sputum", ["가래", "객담", "sputum"], {}),
    ("hemoptysis", ["객혈", "각혈", "피가래", "피섞인가래", "hemoptysis"], {}),
    ("syncope", ["실신", "기절", "의식을잃", "정신을잃", "쓰러졌", "syncope"], {}),
    ("dizziness", ["어지러", "어지럼", "현기증", "dizz", "vertigo"], {"cc": True}),
    ("sweating", ["식은땀", "발한", "땀이", "땀을", "땀나", "diaphoresis", "sweat"], {}),
    ("edema", ["부종", "붓", "부었", "부어", "edema", "swelling"], {}),
    ("weakness", ["위약", "근력저하", "힘이빠", "힘빠", "힘이없", "weakness"], {}),
    ("fatigue", ["피로", "피곤", "기운이없", "기운없", "무기력", "쇠약감", "fatigue", "malaise"], {}),
    ("anorexia", ["식욕부진", "식욕저하", "입맛이없", "입맛없", "밥맛이없", "anorexia"], {}),
    ("weight_loss", ["체중감소", "살이빠", "몸무게가줄", "체중이줄", "weight loss"], {}),
    ("palpitation", ["두근", "심계항진", "palpitation"], {}),
    ("dysuria", ["배뇨통", "소변볼때아프", "소변볼때아파", "소변볼때통증", "dysuria"], {}),
    ("hematuria", ["혈뇨", "소변에피", "hematuria"], {}),
    ("melena", ["흑색변", "짜장", "검은변", "까만변", "melena"], {}),
    ("hematochezia", ["혈변", "피가섞인변", "선혈변", "hematochezia"], {}),
    ("jaundice", ["황달", "눈이노랗", "피부가노랗", "jaundice", "icterus"], {}),
    ("rash", ["발진", "rash"], {}),
    ("confusion", ["의식저하", "혼돈", "착란", "혼미", "confusion", "altered mental"], {}),
    ("neck_stiff", ["경부강직", "항부강직", "목이뻣뻣", "목뻣뻣", "re:목[이은도가]? ?뻣뻣", "neck stiffness", "nuchal rigidity"], {}),
    ("tenderness", ["압통", "tenderness", "re:누르(?:면|니|자|는데)[^\\n.,]{0,8}?(?:아프|아파|아픈|통증)"], {}),
    ("rebound", ["반발통", "반동압통", "rebound"], {}),
    ("murphy", ["머피", "murphy"], {}),
    ("murmur", ["심잡음", "잡음", "murmur"], {}),
    ("crackles", ["수포음", "악설음", "crackle", "rale"], {}),
    ("wheeze", ["천명", "쌕쌕", "wheez"], {}),
    ("tachycardia", ["빈맥", "맥이빠르", "심박이빠르", "tachycardi"], {"measure": "tachy"}),
    ("bradycardia", ["서맥", "bradycardi"], {"measure": "brady"}),
    ("tachypnea", ["빈호흡", "tachypnea"], {"measure": "tachypnea"}),
    ("hypotension", ["저혈압", "혈압저하", "혈압이낮", "hypotension"], {"measure": "hypotension"}),
    ("hypertension", ["고혈압", "혈압이높", "hypertension"], {}),
    ("hypoxemia", ["저산소", "산소포화도저하", "포화도저하", "hypoxemi", "hypoxia", "desaturation"], {"measure": "hypoxemia"}),
    ("leukocytosis", ["백혈구증가", "백혈구상승", "백혈구수증가", "leukocytosis"], {"measure": "leukocytosis"}),
    ("anemia", ["빈혈", "anemia"], {"measure": "anemia"}),
    ("thrombocytopenia", ["혈소판감소", "thrombocytopenia"], {"measure": "thrombocytopenia"}),
    ("wbc", ["백혈구", "wbc", "white blood cell"], {}),
    ("hb", ["혈색소", "헤모글로빈", "hb", "hgb", "hemoglobin"], {}),
    ("plt", ["혈소판", "plt", "platelet"], {}),
    ("crp", ["crp", "c반응단백", "c-반응"], {}),
    ("troponin", ["트로포닌", "troponin", "tni", "tnt"], {}),
    ("ddimer", ["d-dimer", "d dimer", "ddimer", "디다이머", "d다이머"], {}),
    ("st_elev", ["st분절상승", "st상승", "st elevation", "ste"], {}),
    ("st_dep", ["st분절하강", "st하강", "st저하", "st depression"], {}),
    ("bp", ["혈압", "bp", "blood pressure"], {}),
    ("pulse", ["맥박", "심박수", "심박", "pulse", "heart rate", "hr"], {}),
    ("temp", ["체온", "temperature", "bt"], {}),
    ("spo2", ["산소포화도", "spo2", "sao2"], {}),
    ("ecg", ["심전도", "ecg", "ekg"], {"optional": True}),
    ("cxr", ["흉부x선", "흉부엑스레이", "가슴x선", "흉부방사선", "chest x-ray", "chest xray", "cxr"], {"optional": True}),
    ("ct", ["ct", "씨티", "전산화단층"], {"optional": True}),
    ("mri", ["mri", "자기공명"], {"optional": True}),
    ("us", ["초음파", "ultrasound", "sonograph", "echo"], {"optional": True}),
    ("labs", ["혈액검사", "피검사", "일반혈액", "cbc"], {"optional": True}),
    ("physical", ["신체검사", "신체진찰", "진찰", "청진", "촉진", "시진", "타진", "physical exam"], {"optional": True}),
]

# --- literal words (what the lexicon does not cover) -------------------------------------------------------------
_NUM = re.compile(r"(?<![a-z\d.])\d+(?:\.\d+)?")  # digits glued to letters are names (spo2, v5, b12)
_REF_PAREN = re.compile(r"\(([^)]*(?:참고|정상|기준|normal|ref)[^)]*|[<>≤≥][^)]*)\)")
_PAREN = re.compile(r"\([^)]*\)")
_PART_END = re.compile(r",|;|\n| / |(?<!\d)\.|\.(?!\d)|[?!]")
_HIGH = re.compile(r"증가|상승|높|항진|elevat|\bhigh\b|increas|hyperactive|↑|\(h\)")
_LOW = re.compile(r"감소|저하|낮|decreas|\blow\b|↓|\(l\)")
# test / exam names: context, not content ("심전도상 ST 하강" is grounded by "ST분절 하강"); used as tokens only when a
# claim has nothing else ("CT 정상")
_TEST_NAME = re.compile(r"^(?:심전도|ecg|ekg|흉부|x선|엑스레이|xray|cxr|ct|씨티|mri|초음파|ultrasound|echo|혈액검사|피검사|cbc"
                        r"|신체검사|신체진찰|진찰|청진|촉진|시진|타진|검사|영상|촬영|혈관조영)")
_STOP = {"있음", "있다", "있어", "있고", "있는", "양성", "음성", "이상", "소견", "검사", "결과", "환자", "증상", "호소", "관찰", "확인",
         "동반", "약간", "심함", "심한", "심하게", "지속", "지속적", "상태", "수치", "정도", "부위", "경우", "최근", "현재", "과거", "병력",
         "과거력", "기왕력", "없음", "없다", "없어", "없고", "없는", "아님", "부정", "정상", "않음", "부인", "negative", "positive", "normal",
         "absent", "present", "no", "not", "denies", "without", "mmhg", "mg", "dl", "ml", "ul", "μl", "ng", "bpm", "회", "분", "도",
         "증가", "상승", "높음", "높은", "감소", "저하", "낮음", "낮은", "elevated", "high", "low", "increased", "decreased", "and", "the",
         "of", "with", "in", "on", "for", "관련", "의한", "인한", "시사", "의심", "가능성", "및", "또는", "전", "후", "때", "쪽",
         "적은", "적이", "적도", "것", "같아요", "그런", "특별한", "특별히", "조금", "많이", "계속", "자주", "가끔", "요즘",
         "있어요", "있었어요", "했어요", "해요", "돼요", "됐어요", "나요", "났어요", "양상", "느낌", "특이", "특이사항", "특별",
         "외", "그외", "기타", "별다른", "소견상",
         "mm", "cm", "kg", "mmol", "meq", "iu", "hpf", "cmh2o", "fl", "pg", "sec", "min", "mcg", "μg"}
_PARTICLES = sorted(["에서는", "에서", "으로", "에게", "까지", "부터", "이나", "이며", "이고", "하고", "처럼", "보다", "에는", "은", "는",
                     "이", "가", "을", "를", "도", "의", "에", "로", "과", "와", "만", "함", "됨", "임"], key=len, reverse=True)
# last syllables of a conjugated verb/adjective ("뻗치는", "심한") — dropped so other endings still match
_VERB_END = set("는은한된던게히고서며어아여해지")
# fillers of a list item: a part holding only concept mentions and these takes the negation of the next part
_MODIFIERS = re.compile(r"심한|심하게|약간|조금|좀|가벼운|경미한|미약한|간헐적|지속적|급성|만성|새로운|새로|전혀|별로|아무"
                        r"|(?<!비)특이적인?|명확한|뚜렷한")


def _variant_regex(v: str) -> str:
    if re.fullmatch(r"[a-z0-9 \-]+", v):  # ASCII: word boundaries
        body = re.escape(v).replace(r"\ ", " ?").replace(r"\-", "[- ]?")
        return rf"(?<![a-z0-9]){body}" + ("(?![a-z])" if len(v) <= 3 else "")
    chars = [re.escape(ch) for ch in v if ch != " "]
    if len(chars) == 1:
        return rf"(?<![가-힣]){chars[0]}"
    # spaces are allowed between the characters only when the match starts a word: "숨이 차" yes, "대동맥 박리" is
    # not "맥박"; a negating prefix makes another word ("비특이적" is not "특이")
    return f"(?:(?<![비무미]){''.join(chars)}|(?<![가-힣]){'[ ]?'.join(chars)})"


def _stem(word: str) -> str:
    for _ in range(2):  # stacked particles: "등으로의", "복부에서는"
        # a one-syllable particle must leave ≥ 2 syllables ("식도" stays; "등으로" → "등")
        p = next((p for p in _PARTICLES if word.endswith(p) and len(word) - len(p) >= (1 if len(p) > 1 else 2)), None)
        if p is None:
            break
        word = word[:-len(p)]
    return word


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(_REF_PAREN.sub(" ", text))]


def _num_match(a: float, b: float) -> bool:
    """Same value up to rounding: 38.5 = 38.50, SpO2 96 ≠ 98, 36.8 ≠ 36.9; counts ≥ 1000 within 0.2 %;
    "WBC 14.2" (×10³/μL) = 14,200. Stricter than the layer's 3 % tolerance on purpose (precision first)."""
    small, big = sorted((abs(a), abs(b)))
    if abs(a - b) <= (0.002 * big if big >= 1000 else 0.051):
        return True
    return big >= 1000 and abs(small * 1000 - big) <= 0.004 * big


def _direction(text: str) -> str | None:
    """high | low | normal from a reference range in parentheses after the last value, else from flag words."""
    m = _REF_PAREN.search(text)
    if m:
        before = _NUM.findall(text[:m.start()])
        if before:
            d, _ref = _nlp._ref_direction(text[m.start():], float(before[-1]))
            if d:
                return d
    rest = _REF_PAREN.sub(" ", text)
    flag = _HIGH.search(rest) or _LOW.search(rest)
    if flag and _nlp._first_cue(rest[flag.end():flag.end() + 6])[0] != "neg":
        return "high" if _HIGH.search(rest) else "low"
    return None


# --- evidence ---------------------------------------------------------------------------------------------------
@dataclass
class _Part:
    """One comma part of an evidence clause (the unit of the literal reader)."""

    text: str  # reference ranges removed
    clause: int  # clause id inside its segment
    direction: str | None
    numbers: list[float]


@dataclass
class _Seg:
    """One environment response (or the initial info), parsed once."""

    raw: str
    text: str  # normalize(raw)
    source: str
    question: str
    findings: list[Finding]
    parts: list[_Part]


_PATIENT_SPEECH = re.compile(r"(?:요|죠|까)\s*(?:[.?!~,]|$)")


def _guess_source(line: str) -> str:
    """Source of a response line when the caller did not say: patient speech ends in 요/죠/까, reports do not."""
    return "patient" if _PATIENT_SPEECH.search(line) else "test"


def _span_window(t: str, start: int, end: int) -> str:
    """The comma part of t around [start, end)."""
    a = max((m.end() for m in _PART_END.finditer(t, 0, start)), default=0)
    m = _PART_END.search(t, end)
    return t[a:m.start() if m else len(t)]


_SEVERITY = re.compile(r"경미|경한|약간|심한|심함|중등도|미약|현저|뚜렷")
# an item stated with its own state word is a complete finding, not a list head ("장음 항진, 반발통 없음")
_STATE_END = re.compile(r"(?:하강|상승|증가|감소|항진|저하|확장|비대|팽만|건조)$")
_REF_DIR_CONCEPT = re.compile(r"_(high|low)$")


def _guard_findings(t: str, fs: list[Finding], source: str) -> list[Finding]:
    """Precision guards on the layer's reading of one response.

    - List negation. The layer lets a comma part without its own cue take the next part's negation ("수포음, 천명음
      없음"). Kept for a one-word item only. A part with words of its own ("좌심실비대 소견과 V5-V6 비특이적 ST분절 하강,
      ST분절 상승 없음") or a severity ("명치 부위 경미한 압통, 반발통 없음") keeps its default (present) reading; a
      multi-word item ("장음 항진, 반발통 없음" / "경부 강직, Kernig 징후 없음") is ambiguous -> uncertain.
    - Test values. A kb_tests reading that contradicts the reference range printed next to the value
      ("트로포닌 I 1240 ng/L (참고치 <34)" read as not elevated) -> uncertain.
    - Conflicts. One concept read both present and absent in the same clause -> both uncertain."""
    out = []
    for f in fs:
        if f.polarity == "absent" and f.cue.startswith("list:"):
            a = max((m.end() for m in _PART_END.finditer(t, 0, f.start)), default=0)
            m = _PART_END.search(t, f.end)
            b = m.start() if m else len(t)
            chars = list(t[a:b])
            for o in fs:
                for i in range(max(o.start, a), min(o.end, b)):
                    chars[i - a] = " "
            rest = _nlp._SITE.sub(" ", _nlp._BARE_FILLER.sub(" ", _MODIFIERS.sub(" ", _PAREN.sub(" ", "".join(chars)))))
            state = _STATE_END.search(f.span.strip())
            head = f.span.strip()[:state.start()].strip() if state else ""
            after = t[b:b + 40]
            if _SEVERITY.search(t[a:b]) or (head and re.search(re.escape(head) + r"\s?(?:하강|상승|증가|감소|항진|저하)", after)):
                # "명치 부위 경미한 압통, 반발통 없음"; "ST분절 하강, ST분절 상승 없음" (the negation has its own subject)
                f = replace(f, polarity="present", cue="grounding:stated-item", confidence=0.7)
            elif len(re.sub(r"[^가-힣a-z0-9]", "", rest)) > 1 \
                    or (state and " " in f.span.strip() and source != "patient"):
                # an item with words of its own, or a report item stated with its state word ("장음 항진, 반발통 및
                # 근육 강직 없음"): the list reading is not reliable either way
                f = replace(f, polarity="uncertain", cue="grounding:ambiguous-list", confidence=0.5)
        elif f.cue == "when-clause":  # "기침할 때 심해지지도 않아요" does not say the patient coughs
            f = replace(f, polarity="uncertain", cue="grounding:when-clause", confidence=0.5)
        elif f.cue.startswith("kb_tests-value") and (dm := _REF_DIR_CONCEPT.search(f.concept)):
            refs = _REF_PAREN.findall(f.clause)
            d = _direction(f.clause) if len(refs) == 1 and len(_numbers(f.clause)) == 1 else None
            if d in ("high", "low", "normal") and (d == dm.group(1)) != (f.polarity == "present"):
                f = replace(f, polarity="uncertain", cue="grounding:ref-conflict", confidence=0.5)
        out.append(f)
    seen: dict[tuple, set] = {}
    for f in out:
        seen.setdefault((f.concept, f.subject, f.temporality), set()).add(f.polarity)
    return [replace(f, polarity="uncertain", cue="grounding:conflict")
            if {"present", "absent"} <= seen[(f.concept, f.subject, f.temporality)] else f for f in out]


def _parts(t: str) -> list[_Part]:
    out: list[_Part] = []
    ci = 0
    for a, b, _q in _nlp._sentences(t):
        sent = t[a:b]
        if _nlp._UNAVAILABLE.search(sent):
            continue  # "결과가 제공되지 않습니다" grounds nothing, not even the test named in it
        for c in _nlp._clauses(sent):
            for pa, pb in c.parts:
                raw = sent[pa:pb].strip()
                if raw:
                    text = _REF_PAREN.sub(" ", raw).strip()
                    out.append(_Part(text, ci, _direction(raw), _numbers(raw)))
            ci += 1
    return out


def _parse_seg(raw: str, source: str, question: str, cache: dict | None) -> _Seg:
    key = (raw, source, question)
    if cache is not None and key in cache:
        return cache[key]
    t = normalize(raw)
    fs = parse(raw, source, {"question": question} if question else None)
    seg = _Seg(raw, t, source, question, _guard_findings(t, fs, source), _parts(t))
    if cache is not None:
        cache[key] = seg
    return seg


@dataclass
class Evidence:
    """Parsed evidence of one case (build once per check; not shared across cases)."""

    raw: str
    segs: list[_Seg] = field(default_factory=list)
    part_seg: list[int] = field(default_factory=list)  # flat parts of all segments, with their segment index
    parts: list[_Part] = field(default_factory=list)
    joined: str = ""
    compact: str = ""
    starts: list[int] = field(default_factory=list)
    cache: dict | None = None  # per-case parse cache (segments and claims), owned by the case state
    by_concept: dict[str, list[tuple[int, Finding]]] = field(default_factory=dict)
    _hits: dict = field(default_factory=dict)
    _claims: dict = field(default_factory=dict)  # parsed claims, per evidence object (i.e. per check of one case)
    _qa: dict = field(default_factory=dict)  # qa list -> Evidence of the yes/no answers

    @classmethod
    def build(cls, text: str, sources: list[str] | None = None, questions: list[str] | None = None,
              cache: dict | None = None) -> "Evidence":
        """One segment per line. sources/questions (optional, per line): "patient" | "exam" | "test" and the question
        a patient answer replies to; guessed from the wording when not given. cache: per-case parse cache."""
        lines = (text or "").split("\n")
        ev = cls(text or "", cache=cache)
        for i, line in enumerate(lines):
            src = sources[i] if sources and i < len(sources) and sources[i] else _guess_source(line)
            q = questions[i] if questions and i < len(questions) and questions[i] else ""
            ev.segs.append(_parse_seg(line, src, q if src == "patient" else "", cache))
        for si, seg in enumerate(ev.segs):
            for f in seg.findings:
                ev.by_concept.setdefault(f.concept, []).append((si, f))
            for p in seg.parts:
                ev.parts.append(p)
                ev.part_seg.append(si)
        pos = 0
        for p in ev.parts:
            ev.starts.append(pos)
            pos += len(p.text) + 1
        ev.joined = "\n".join(p.text for p in ev.parts)
        ev.compact = ev.joined.replace(" ", "")
        return ev

    def claim(self, text: str) -> "_Claim":
        store = self._claims if self.cache is None else self.cache.setdefault("\x00claims", {})
        if text not in store:
            store[text] = _parse_claim(text)
        return store[text]

    def qa_evidence(self, qa: list[tuple[str, str]]) -> "Evidence":
        """The answers of (question, answer) pairs, parsed with their question (yes/no answers resolve it)."""
        key = tuple(qa)
        if key not in self._qa:
            self._qa[key] = Evidence.build("\n".join(re.sub(r"\s*\n\s*", "; ", a or "") for _q, a in qa),
                                           ["patient"] * len(qa), [q or "" for q, _a in qa])
        return self._qa[key]


def _as_evidence(evidence: "str | Evidence") -> Evidence:
    return evidence if isinstance(evidence, Evidence) else Evidence.build(evidence)


# --- claims -----------------------------------------------------------------------------------------------------
@dataclass
class _Claim:
    text: str  # normalised claim
    concepts: list[Finding]  # what the layer reads in the claim
    tokens: list[tuple]  # literal words left after the concepts: ("lit", regex, core) | ("lab", key, key)
    optional: list[tuple]  # test / exam names
    all_tokens: list[tuple]  # literal words of the whole claim (for value claims read literally)
    numbers: list[float]
    free_numbers: list[float]  # numbers no concept value accounts for
    negative: bool
    direction: str | None
    finding_like: bool
    numeric: bool  # every concept comes from a value -> may also be checked literally (label words + numbers)


_LAB_TOKENS = [(key, rx) for key, rx, *_ in _nlp._LAB_RX]
_LAB_RX = dict(_LAB_TOKENS)
_FINDING_MARK = re.compile(r"징후|sign|청취|촉지|압통|양성|음성|상승|하강|증가|감소|저하|비대|음영|결절|종괴|병변|출혈|협착|폐색|부종"
                           r"|잡음|확장|침윤|경화|삼출|기흉|골절|궤양|천공|강직|마비|결손|저림|증후")


def _from_value(c: Finding) -> bool:
    return c.cue.startswith(("value", "kb_tests", "normal-word"))


def _tokens(text: str) -> tuple[list[tuple], list[tuple]]:
    """(required, optional) literal tokens of a claim text with numbers and concept words already blanked."""
    labs, required, optional = [], [], []
    for key, rx in _LAB_TOKENS:  # lab analyte names read with the layer's analyte table ("WBC" = "백혈구")
        if rx.search(text):
            labs.append(("lab", key, key))
            text = rx.sub(" ", text)
    for w in re.findall(r"[a-z][a-z0-9]*|[가-힣]+", text):
        w = _stem(w)
        if w in _STOP or _nlp._CUE.match(w):  # "없어요", "정상", "명료", "양성" carry polarity, not content
            continue
        if _TEST_NAME.match(w):
            optional.append(("lit", _variant_regex(w), w))
        elif len(w) < 2:
            continue
        elif re.fullmatch(r"[가-힣]+", w):
            core = w[:-1] if len(w) >= 3 and w[-1] in _VERB_END else w  # "좌심실비대한" / "좌심실비대"
            required.append(("lit", _variant_regex(core), core))
        else:
            required.append(("lit", rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", w))
    return labs + required, optional


def _claim_negated(text: str) -> bool:
    """A concept-free claim states an absence when any negation cue of the layer is in it ("CT 이상 없음", "호흡음
    정상"), after masking the idioms that only look negative."""
    masked = _nlp._IDIOM_NOT_NEG.sub(" ", text)
    return any(m.lastgroup == "neg" for m in _nlp._CUE.finditer(masked))


def _parse_claim(claim: str) -> _Claim:
    t = normalize(claim)
    concepts = parse(claim, "claim")
    specific = {c.concept for c in concepts}
    # generic pain next to a specific pain concept is redundant (as in nlp.match)
    concepts = [c for c in concepts if not (c.concept == "SYM:pain" and any("SYM:pain" in LEXICON.ancestors(s)
                                                                             for s in specific))]
    numbers = _numbers(t)
    chars = list(t)
    for c in concepts:
        if c.cue.startswith("kb_tests"):
            continue  # a test-result reading spans the whole clause: its words stay literal ("II·III·aVF ST 하강")
        if c.start > 0 and re.match(r"[가-힣]", t[c.start - 1]):
            continue  # a concept inside a longer word ("동성빈맥") keeps the whole word as a literal too
        end = c.end + len(re.match(r"[가-힣a-z]*", t[c.end:]).group())  # the rest of the word goes with the concept
        for i in range(c.start, min(end, len(chars))):
            chars[i] = " "
    rest = _NUM.sub(" ", _REF_PAREN.sub(" ", "".join(chars)))
    tokens, optional = _tokens(rest)
    all_tokens, all_optional = _tokens(_NUM.sub(" ", _REF_PAREN.sub(" ", t)))
    if not tokens and not concepts and numbers:  # "당 62 mg/dL": a one-syllable lab name counts when it has a value
        tokens = [("lit", rf"(?<![가-힣]){re.escape(w)}(?![가-힣])", w)
                  for w in re.findall(r"(?<![가-힣])[가-힣](?![가-힣])", rest)]
    values = [c.value for c in concepts if c.value is not None]
    free = [x for x in numbers if not any(_num_match(x, v) for v in values)]
    worded = [c for c in concepts if not _from_value(c)]
    if worded:
        negative = any(c.polarity == "absent" for c in worded)
    else:  # value concepts carry the value's reading, not the claim's polarity ("체온 36.7" is not a negation)
        negative = _claim_negated(rest)
    direction = "high" if _HIGH.search(rest) else "low" if _LOW.search(rest) else None
    finding_like = bool([c for c in concepts if not c.cue.startswith("kb_tests")] or numbers or _FINDING_MARK.search(t))
    numeric = bool(numbers) and not worded
    return _Claim(t, concepts, tokens, optional, all_tokens or all_optional, numbers, free, negative, direction,
                  finding_like, numeric)


def _needed(n: int) -> int:
    return n if n <= 3 else max(n - 1, -(-3 * n // 4))  # all of 1–3 tokens; all but one (≥ 3/4) of more


# --- concept path -----------------------------------------------------------------------------------------------
def _qualified_parent(c: Finding, e: Finding, seg: _Seg) -> bool:
    """A present parent grounds a more specific present claim when the claim's qualifier word sits in the same comma
    part: "등으로 뻗치는" (QUAL:radiation_back) by "등 쪽 날개뼈 사이로 뻗치더니" (QUAL:radiation + 등)."""
    if e.polarity != "present" or e.hypothetical or e.subject != c.subject or e.concept.startswith("GRP:"):
        return False
    espan = e.span.replace(" ", "")
    qual = []
    for w in re.findall(r"[가-힣]+|[a-z]+", c.span):
        w = _stem(w)
        if w and w not in _STOP and w not in espan and not _nlp._CUE.match(w):
            qual.append(w)
    if not qual:
        return False
    window = _span_window(seg.text, e.start, e.end)
    return all(re.search(_variant_regex(w), window) for w in qual)


def _unqualified(e: Finding, seg: _Seg) -> bool:
    """An absent parent speaks for its children only when it is not itself narrowed by a site word in front of it:
    "열은 없어요" covers 고열, but "흉벽 압통 없음" says nothing about the McBurney point."""
    a = max((m.end() for m in _PART_END.finditer(seg.text, 0, e.start)), default=0)
    head = _nlp._BARE_FILLER.sub(" ", _MODIFIERS.sub(" ", seg.text[a:e.start]))
    return not re.search(r"[가-힣a-z0-9]{2,}", head)


def _qualifiers(c: Finding) -> list[str]:
    """Korean words of a claim mention that are not themselves the concept (or a relative): "전반" in "하복부 전반의
    압통"."""
    family = {c.concept, *LEXICON.ancestors(c.concept), *LEXICON.descendants(c.concept)}
    out = []
    for w in re.findall(r"[가-힣]{2,}", c.span):
        w = _stem(w)
        if len(w) < 2 or w in _STOP or _MODIFIERS.fullmatch(w) or _nlp._CUE.match(w):
            continue
        if not any(m.cid in family for m in LEXICON.scan(w)):
            out.append(w)
    return out


def _window(seg: _Seg, e: Finding) -> str:
    """Where a claim's other words must be found: the comma part of a report, the whole answer of a patient."""
    text = seg.text if seg.source == "patient" else _span_window(seg.text, e.start, e.end)
    return text + (" " + normalize(seg.question) if e.cue.startswith("yes/no") else "")


def _support_ok(c: Finding, e: Finding, seg: _Seg) -> bool:
    """Precision guards on top of the layer's support rule."""
    if e.concept != c.concept:
        if c.polarity == "absent" and not _unqualified(e, seg):
            return False
        if e.concept.startswith("GRP:") and not re.search(r"증상|증세|symptom", e.span):
            return False  # "감기 기운 없음" is not "no respiratory symptom of any kind"
    # the claim's own qualifier words: always for a more general evidence concept, and within a report comma part
    # also for the same concept (reports list several findings of one coarse concept)
    if c.polarity == "present" and (e.concept != c.concept or seg.source != "patient") \
            and not all(re.search(_variant_regex(w), _window(seg, e)) for w in _qualifiers(c)):
        return False  # "하복부 전반의 압통" is not grounded by "우하복부 압통", "피부 긴장도 저하" not by "구강 점막 건조"
    current_use = c.concept in _CURRENT_USE and not re.search(r"력|과거|예전|이전", c.span)
    if c.polarity == "present" and e.temporality == "past" and c.temporality != "past" \
            and (not c.concept.startswith("HX:") or current_use):
        return False  # a symptom that went away does not ground a current one; "끊었어요" does not ground "피워요"
    return True


# history concepts that describe a current habit or treatment ("담배는 10년 전에 끊었어요" is not current smoking)
_CURRENT_USE = {"HX:smoking", "HX:alcohol", "HX:drug_use", "HX:anticoagulant", "HX:antiplatelet", "HX:medication",
                "HX:hormone_use", "HX:pregnancy"}


def _supported_by(c: Finding, ev: Evidence) -> list[tuple[int, Finding]]:
    """Evidence findings (segment index, finding) that support one claim finding."""
    cands = list(ev.by_concept.get(c.concept, []))
    for other in (LEXICON.descendants(c.concept) if c.polarity == "present" else LEXICON.ancestors(c.concept)):
        cands += ev.by_concept.get(other, [])
    out = [(si, e) for si, e in cands if _nlp._supports(c, e, LEXICON)
           and (c.value is None or (e.value is not None and _num_match(c.value, e.value)))
           and _support_ok(c, e, ev.segs[si])]
    if not out and c.polarity == "present" and c.value is None:
        out = [(si, e) for a in LEXICON.ancestors(c.concept) for si, e in ev.by_concept.get(a, [])
               if _qualified_parent(c, e, ev.segs[si])]
    return out


def _seg_numbers(seg: _Seg) -> list[float]:
    return [n for p in seg.parts for n in p.numbers]


def _span_of(ev: Evidence, si: int, e: Finding) -> str:
    seg = ev.segs[si]
    if e.cue.startswith("yes/no") and seg.question:
        return f"Q: {seg.question[:60]} → A: {seg.raw[:60]}"
    return (e.clause or e.span)[:120]


def _token_in(tok: tuple, text: str) -> bool:
    rx = _LAB_RX[tok[1]] if tok[0] == "lab" else re.compile(tok[1])
    return bool(rx.search(text))


def _ground_concepts(claim: _Claim, ev: Evidence, negative: bool | None,
                     extra_numbers: tuple[float, ...]) -> tuple[bool, float, str]:
    concepts = claim.concepts
    if negative is not None and negative != claim.negative:  # the ledger status decides ("발열" filed under 음성)
        pol = "absent" if negative else "present"
        concepts = [c if _from_value(c) else replace(c, polarity=pol) for c in concepts]
    sup = [_supported_by(c, ev) for c in concepts]
    got = sum(1 for s in sup if s)
    n_tok = len(concepts) + len(claim.tokens)
    best = (False, round(got / n_tok, 3), "")
    if got < len(concepts):
        return best
    # a modifier (quality, radiation, onset) belongs to the symptom it qualifies: same response, or the chief complaint
    homes = sorted({si for s in sup for si, _ in s})
    bound = len(concepts) > 1 and any(c.concept.startswith("QUAL:") for c in concepts)
    if bound:
        homes = [h for h in homes if all(any(si in (h, 0) for si, _ in s) for s in sup)]
    need = _needed(n_tok) - len(concepts)
    for home in homes:
        allowed = [[(si, e) for si, e in s if si in (home, 0)] for s in sup] if bound else sup
        for si, e in [(si, e) for s in allowed for si, e in s if si == home]:
            # the claim's other words must sit in the same answer / report part, its numbers in the same comma part
            seg = ev.segs[si]
            if sum(1 for tok in claim.tokens if _token_in(tok, _window(seg, e))) < need:
                continue
            pool = _numbers(_span_window(seg.text, e.start, e.end))
            if not all(any(_num_match(x, y) for y in pool) for x in claim.free_numbers):
                continue
            if extra_numbers and not all(any(_num_match(x, y) for y in _seg_numbers(seg)) for x in extra_numbers):
                continue
            spans = [_span_of(ev, si, e)] + [_span_of(ev, *s[0]) for s in allowed if (si, e) not in s]
            return True, 1.0, " / ".join(dict.fromkeys(spans))[:120]
    return best


# --- literal path (claims the lexicon has no concept for) --------------------------------------------------------
_BARE_NEG = re.compile(r"[가-힣a-z0-9 ]{1,14}? ?(?:없음|없다|없어요|없습니다|음성|안 보임|보이지 않음)")
def _same_clause_next(ev: Evidence, pi: int) -> int | None:
    j = pi + 1
    if j < len(ev.parts) and ev.part_seg[j] == ev.part_seg[pi] and ev.parts[j].clause == ev.parts[pi].clause:
        return j
    return None


def _literal_polarity(ev: Evidence, pi: int, s: int, e: int) -> bool | None:
    """True = negated, False = affirmed, None = uncertain; the layer's cue rules, applied to the comma part."""
    p = ev.parts[pi]
    tail = _PAREN.sub(" ", p.text[e:])  # "혈압 152/94 (양측 차이 없음)": the remark is not about 혈압
    j = _same_clause_next(ev, pi)
    # a bare list item ("수포음, 천명음 없음") takes the negation of a short negated part right after it; an item with
    # its own words ("우상복부 압통, 반발통 없음") does not
    if j is not None and not re.search(r"[가-힣a-z0-9]", p.text[:s] + " " + p.text[e:]) \
            and _BARE_NEG.fullmatch(ev.parts[j].text):
        tail += " " + ev.parts[j].text
    kind = _nlp._first_cue(tail)[0]
    if kind == "unc":
        return None
    head = p.text[:s]
    return kind == "neg" or bool(_nlp._PRE_NEG_EN.search(head) or _nlp._PRE_NEG_KO.search(head))


def _token_hits(ev: Evidence, tok: tuple) -> dict[int, bool]:
    """part index -> negated, for the first usable occurrence of a literal token in that part."""
    key = tok[:2]
    if key in ev._hits:
        return ev._hits[key]
    hits: dict[int, bool] = {}
    if tok[0] == "lit" and tok[2] not in ev.compact:  # cheap pre-check before the regex
        ev._hits[key] = hits
        return hits
    rx = _LAB_RX[tok[1]] if tok[0] == "lab" else re.compile(tok[1])
    for m in rx.finditer(ev.joined):
        pi = bisect_right(ev.starts, m.start()) - 1
        if hits.get(pi) is False:
            continue
        pol = _literal_polarity(ev, pi, m.start() - ev.starts[pi], m.end() - ev.starts[pi])
        if pol is not None:
            hits[pi] = pol
    ev._hits[key] = hits
    return hits


def _number_near(ev: Evidence, pi: int, x: float) -> bool:
    pool = ev.parts[pi].numbers
    if not pool and (j := _same_clause_next(ev, pi)) is not None:
        pool = ev.parts[j].numbers
    return any(_num_match(x, y) for y in pool)


def _ground_literal(claim: _Claim, ev: Evidence, tokens: list[tuple], neg: bool,
                    extra_numbers: tuple[float, ...]) -> tuple[bool, float, str]:
    tokens = tokens or claim.optional
    if not tokens or all(tok[0] == "lit" and (_nlp._SITE.fullmatch(tok[2]) or _nlp._LAT.fullmatch(tok[2]))
                         for tok in tokens):
        return False, 0.0, ""  # a side or a body part alone states no finding ("오른쪽은 괜찮아요")
    hits = [_token_hits(ev, tok) for tok in tokens]
    need = _needed(len(tokens))
    best = (False, 0.0, "")
    for pi in sorted(set().union(*hits)):
        present = [h[pi] for h in hits if pi in h]
        score = round(len(present) / len(tokens), 3)
        if len(present) < need:
            best = max(best, (False, score, ""), key=lambda x: x[1])
            continue
        p = ev.parts[pi]
        if claim.numbers and not all(_number_near(ev, pi, x) for x in claim.numbers):
            continue
        if extra_numbers and not all(any(_num_match(x, y) for y in _seg_numbers(ev.segs[ev.part_seg[pi]]))
                                     for x in extra_numbers):
            continue
        part_neg = any(present)
        if not part_neg and p.direction == "normal" and not claim.numbers:
            part_neg = True  # a lab value inside its reference range reads as "no abnormality"
        if claim.direction and not neg:
            if part_neg or (p.direction and p.direction != claim.direction):
                continue
        elif neg != part_neg:
            continue
        return True, score, p.text[:120]
    return best


def _ground(claim: _Claim, ev: Evidence, negative: bool | None = None,
            extra_numbers: tuple[float, ...] = ()) -> tuple[bool, float, str]:
    neg = claim.negative if negative is None else negative
    if claim.concepts:
        res = _ground_concepts(claim, ev, negative, extra_numbers)
        if res[0] or not claim.numeric:
            return res
        # a value claim the layer could not pair ("좌측 상지 혈압 138/78", "트로포닌 21"): label words + numbers
        return _ground_literal(claim, ev, claim.all_tokens, neg, extra_numbers)
    return _ground_literal(claim, ev, claim.tokens, neg, extra_numbers)


def _ground_qa(claim: _Claim, qa: list[tuple[str, str]], negative: bool, ev: Evidence) -> tuple[bool, float, str]:
    """Grounding against the answers of (question, answer) pairs read with their question: a yes/no answer resolves
    the question's findings ("가슴이 두근거리나요?" → "네, 좀 그래요")."""
    if not qa:
        return False, 0.0, ""
    return _ground(claim, ev.qa_evidence(qa), negative)


# --- public API -------------------------------------------------------------------------------------------------
_SOURCE = {"ASK": "patient", "EXAM": "exam", "TEST": "test"}
_TAG = " (미확인)"
# numbers in a finding's detail that are durations, counts or ages ("3일 전부터") are not checked: the patient says
# "사흘 전" while the model writes "3일"
_TIME_NUM = re.compile(r"\d+(?:\.\d+)? ?(?:일|주|개월|달|년|시간|분|초|세|살|번|차례|시|층|갑|병|잔|kg|cm)")
_REASON_SPLIT = re.compile(r"[,;/·+]|\n|(?<!\d)\.|\.(?!\d)| (?:및|그리고|또는|and|with) |(?<=[가-힣])(?:와|과|고|며) "
                           r"|(?<=[가-힣0-9A-Za-z%)])(?:으로|로|에서) |때문|이므로|므로|시사|의심|고려|가능성|감별|진단|배제|필요|보아|미루어|소견상")


def evidence_text(state) -> str:
    """Initial info + every environment response, one line each (never the doctor's own questions)."""
    parts = [state.initial_info] + [t.response for t in state.turns]
    return "\n".join(re.sub(r"\s*\n\s*", "; ", p or "") for p in parts)


def _state_evidence(state) -> Evidence:
    """Evidence of one case with each response's source and, for ASK turns, the question it answers. Parses are
    cached on the case state itself (a new state per case, so nothing crosses cases)."""
    cache = state.__dict__.setdefault("_grounding_cache", {})
    kinds = [getattr(t.action.type, "value", "") for t in state.turns]
    sources = [""] + [_SOURCE.get(k, "") for k in kinds]
    questions = [""] + [t.action.content if k == "ASK" else "" for t, k in zip(state.turns, kinds)]
    return Evidence.build(evidence_text(state), sources, questions, cache)


def is_grounded(claim: str, evidence: "str | Evidence") -> tuple[bool, float, str]:
    """(grounded, score 0..1 = share of the claim's concepts/words found, matched evidence text)."""
    ev = _as_evidence(evidence)
    return _ground(ev.claim(claim), ev)


def check_findings(ledger, evidence: "str | Evidence", qa: list[tuple[str, str]] | None = None,
                   unavailable: list[str] | None = None) -> list:
    """Set `verified` and `span` on every finding of a FindingsLedger; returns the findings found unverified.
    양성/음성 findings are checked against the evidence (item text + numbers in the detail); 결과없음 findings against the
    list of requests the environment said it had no result for (left unchecked when that list is not given)."""
    ev = _as_evidence(evidence)
    bad = []
    for f in ledger.items:
        if f.status == "결과없음":
            if unavailable is None:
                f.verified, f.span = None, ""
                continue
            claim = ev.claim(f.item)
            match = next((u for u in unavailable if similarity(u, f.item) >= 0.3
                          or _ground_literal(claim, Evidence.build(u, ["test"]), claim.all_tokens, False, ())[0]), None)
            f.verified, f.span = match is not None, (match or "")[:120]
        else:
            claim = ev.claim(f.item)
            detail = ev.claim(f.detail) if f.detail else None
            if detail and not claim.direction and detail.direction and not detail.negative:
                claim = replace(claim, direction=detail.direction)  # item "CRP", detail "상승"
            negative = f.status == "음성" or claim.negative
            extra = tuple(_numbers(_TIME_NUM.sub(" ", normalize(f.detail)))) if f.detail else ()
            ok, _, span = _ground(claim, ev, negative, extra)
            if not ok and qa and not extra:
                ok, _, span = _ground_qa(claim, qa, negative, ev)
            f.verified, f.span = ok, span
        if f.verified is False:
            bad.append(f)
    return bad


def check_ddx_support(ddx_ledger, evidence: "str | Evidence", qa: list[tuple[str, str]] | None = None,
                      mode: str = "drop") -> list[dict]:
    """Check each DDx support/against item that states a finding (abstract reasoning such as "전형적 양상" is left
    alone). Ungrounded ones are removed (mode="drop") or tagged "(미확인)" (mode="flag"). Returns what was removed or
    flagged: [{"dx", "side": "for"|"against", "item"}]."""
    ev = _as_evidence(evidence)
    out = []
    for e in ddx_ledger.entries:
        for side, attr in (("for", "support"), ("against", "against")):
            kept = []
            for item in getattr(e, attr):
                if item.endswith(_TAG) or _item_ok(item, ev, qa):
                    kept.append(item)
                    continue
                out.append({"dx": e.dx, "side": side, "item": item})
                if mode == "flag":
                    kept.append(item + _TAG)
            setattr(e, attr, kept)
    return out


def _item_ok(item: str, ev: Evidence, qa: list[tuple[str, str]] | None) -> bool:
    claim = ev.claim(item)
    if not claim.finding_like:
        return True  # reasoning, not a stated finding
    if _ground(claim, ev)[0]:
        return True
    return bool(qa) and _ground_qa(claim, qa, claim.negative, ev)[0]


def ungrounded_in_text(reason: str, evidence: "str | Evidence", qa: list[tuple[str, str]] | None = None) -> list[str]:
    """Finding-like phrases cited in a free-text reason (named symptoms/signs, lab values, numbers) that are not
    grounded in the evidence. Heuristic, meant for the reviewer and for logging."""
    ev = _as_evidence(evidence)
    out: list[str] = []
    for chunk in _REASON_SPLIT.split(reason or ""):
        chunk = chunk.strip(" '\"()[]")
        if len(chunk) < 2 or chunk in out:
            continue
        if not _item_ok(chunk, ev, qa):
            out.append(chunk)
    return out


def apply(state, ddx_mode: str = "drop") -> dict:
    """Verify the findings ledger, clean the DDx ledger's support/against, check the latest action's reason.
    Updates the ledgers in place (state.ddx is re-synced when DDx evidence was removed) and returns a report.
    Answers to ASK turns are read with their question, so yes/no answers are part of the evidence itself."""
    ev = _state_evidence(state)
    bad = check_findings(state.findings, ev, None, state.unavailable())
    removed = check_ddx_support(state.ddx_ledger, ev, None, ddx_mode)
    if removed and ddx_mode == "drop":
        state.ddx = state.ddx_ledger.as_list()
    reason = state.turns[-1].action.reason if state.turns else ""
    return {
        "findings_checked": sum(1 for f in state.findings.items if f.verified is not None),
        "findings_unverified": len(bad),
        "unverified_examples": [f"{f.item} [{f.status}]" for f in bad[:5]],
        "ddx_removed": len(removed),
        "ddx_removed_examples": [f"{r['dx']}/{r['side']}: {r['item']}" for r in removed[:5]],
        "reason_ungrounded": ungrounded_in_text(reason, ev, None)[:5],
    }
