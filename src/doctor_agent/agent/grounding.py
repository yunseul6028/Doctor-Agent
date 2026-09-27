"""Grounding checker: is a finding the model wrote down actually something the environment said?

The small LLM sometimes records findings, or cites DDx evidence, that never appeared in the environment's responses.
This module checks claims against the case's own evidence only (initial info + environment responses of this case;
never the doctor's questions, never other cases). Pure code, CPU only, no LLM calls.

Matching, in short (precision first: a claim is grounded only with clear support):
- Text is normalised (NFKC, lower case, thousands separators removed, "(-)"/"(+)" → 음성/양성) and split into
  clauses (sentences, commas, Korean contrast connectives such as ~고/~는데/~지만/~으나). Sentences that say a
  result is not available ("결과가 제공되지 않습니다") are removed, so they never ground anything.
- A claim is turned into concept tokens: synonym groups (열/발열/fever, 호흡곤란/숨이 차요, WBC/백혈구, ...),
  plus the remaining content words (Korean particles stripped). Test names (심전도, CT, ...) are optional tokens,
  because a result text usually does not repeat the test name.
- One clause must contain the claim's tokens (all of them for 1–2 tokens, all but one and ≥ 2/3 for more), every
  number in the claim (±3 %, or ×1000 for "14.2" vs "14,200"), and the same polarity: a negated claim ("발열 없음")
  needs a negated clause ("열은 없어요"), and a positive claim is never grounded by a negated clause.
- Some concepts are also read from measurements: 발열 from 체온, 빈맥 from 맥박, 백혈구 증가 from WBC, etc.
  Lab reference ranges in parentheses ("(참고치 <34)") give the value's direction; a claimed "상승" is refused when
  the value is within range.
- For findings, a yes/no answer to a question that names the finding ("기침 하세요?" → "네, 좀 해요") also grounds it.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field, replace

from doctor_agent.agent.text import similarity

# --- synonym groups ---------------------------------------------------------------------------------------------
# Variants: plain strings (spaces optional between Korean characters; ASCII words need word boundaries),
# "re:<regex>" raw regex (negation is also looked for inside the match), "claim:<regex>" used only to read claims.
# Flags: optional = a test/exam name (not required when other tokens exist).
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

# measurement concepts: label regex, predicate(value) -> True (present) / False (absent) / None (unclear)
def _wbc(v: float) -> float:
    return v * 1000 if v < 200 else v


_MEASURES: dict[str, tuple[re.Pattern, object]] = {
    "temp": (re.compile(r"체온|temperature|\bbt\b"), lambda v: True if v >= 37.8 else False if v < 37.5 else None),
    "tachy": (re.compile(r"맥박|심박수?|\bhr\b|pulse|heart rate"), lambda v: True if v > 100 else False if 55 <= v <= 100 else None),
    "brady": (re.compile(r"맥박|심박수?|\bhr\b|pulse|heart rate"), lambda v: True if v < 60 else False if v >= 60 else None),
    "tachypnea": (re.compile(r"호흡수|\brr\b|respiratory rate"), lambda v: True if v > 20 else False if 10 <= v <= 20 else None),
    "hypotension": (re.compile(r"혈압|\bbp\b"), lambda v: True if v < 90 else False if v >= 100 else None),
    "hypoxemia": (re.compile(r"산소 ?포화도|spo2|sao2|포화도"), lambda v: True if v < 92 else False if v >= 95 else None),
    "leukocytosis": (re.compile(r"백혈구|\bwbc\b"), lambda v: True if _wbc(v) > 11000 else False if 4000 <= _wbc(v) <= 11000 else None),
    "anemia": (re.compile(r"혈색소|헤모글로빈|\bhg?b\b"), lambda v: True if v < 12 else False if 13.5 <= v < 25 else None),
    "thrombocytopenia": (re.compile(r"혈소판|\bplt\b"),
                         lambda v: True if _wbc(v) < 150000 else False if _wbc(v) >= 150000 else None),
}


def _variant_regex(v: str) -> str:
    if v.startswith("re:"):
        return v[3:]
    if re.fullmatch(r"[a-z0-9 \-]+", v):  # ASCII: word boundaries; prefixes like "tachycardi" match word starts
        body = re.escape(v).replace(r"\ ", " ?").replace(r"\-", "[- ]?")
        return rf"(?<![a-z0-9]){body}" + ("(?![a-z])" if len(v) <= 3 else "")
    chars = [re.escape(ch) for ch in v if ch != " "]
    if len(chars) == 1:
        return chars[0]
    # spaces are allowed between the characters only when the match starts a word: "숨이 차" yes, "대동맥 박리" is
    # not "맥박"
    return f"(?:{''.join(chars)}|(?<![가-힣]){'[ ]?'.join(chars)})"


_GROUP_FLAGS = {g: flags for g, _, flags in _GROUPS}
_EVIDENCE_RE = {g: re.compile("|".join(_variant_regex(v) for v in vs if not v.startswith("claim:"))) for g, vs, _ in _GROUPS}
_RAW_VARIANT_GROUPS = {g for g, vs, _ in _GROUPS if any(v.startswith("re:") for v in vs)}
# Claim reading: variants indexed by first character (raw regexes are tried everywhere); at each position the
# longest match wins, and the rest of the word goes with the concept ("두근거림", "통증이", "발열과").
_CLAIM_BY_CHAR: dict[str, list[tuple[re.Pattern, str]]] = {}
_CLAIM_ANYWHERE: list[tuple[re.Pattern, str]] = []
for _g, _vs, _ in _GROUPS:
    for _v in _vs:
        if _v.startswith(("re:", "claim:")):
            _CLAIM_ANYWHERE.append((re.compile(_v.split(":", 1)[1]), _g))
        else:
            _CLAIM_BY_CHAR.setdefault(_v[0], []).append((re.compile(_variant_regex(_v)), _g))
_WORD_REST = re.compile(r"[가-힣]*|[a-z]*")


def _claim_concepts(text: str) -> tuple[list[str], str, bool]:
    """(groups named in the text, text with those words blanked out, a negation sits inside a matched phrase —
    "배는 안 아파요")."""
    groups: list[str] = []
    out, i, neg_inside = [], 0, False
    while i < len(text):
        best = None
        for rx, grp in _CLAIM_BY_CHAR.get(text[i], []) + _CLAIM_ANYWHERE:
            m = rx.match(text, i)
            if m and m.end() > i and (best is None or m.end() > best[0]):
                best = (m.end(), grp)
        if best is None:
            out.append(text[i])
            i += 1
            continue
        end = _WORD_REST.match(text, best[0]).end()
        if best[1] in _RAW_VARIANT_GROUPS and _negated_tail(text[i:best[0]]):
            neg_inside = True
        if best[1] not in groups:
            groups.append(best[1])
        # a concept inside a longer word ("폐부종", "동성빈맥") keeps the whole word as a literal too, so "폐부종"
        # is not grounded by "유두부종"
        mid_word = out and re.fullmatch(r"[가-힣]", out[-1])
        out.append(text[i:end] if mid_word else " ")
        i = end
    return groups, "".join(out), neg_inside

# --- normalisation and cue words --------------------------------------------------------------------------------
_NUM = re.compile(r"(?<![a-z\d.])\d+(?:\.\d+)?")  # digits glued to letters are names (spo2, v5, b12)
_UNAVAILABLE = re.compile(r"제공 ?되지 ?않|제공하지 ?않|제공 ?불가|결과가? ?없습니다|확인할 ?수 ?없|시행되지 ?않|정보가 ?없")
_SENTENCE = re.compile(r"(?<!\d)\.|\.(?!\d)|[?!;\n]")
_CLAUSE_HARD = re.compile(r"(?<=[가-힣])(?:고|는데|지만|으나|면서|며|(?<!에)서) ")
_REF_PAREN = re.compile(r"\(([^)]*(?:참고|정상|기준|normal|ref)[^)]*|[<>≤≥][^)]*)\)")
_NEG = re.compile(r"없|않|아니|아뇨|(?:^| )안 ?(?:해|했|하|나|났|아|먹|피|마|들|느|보|되|됐|와|왔|좋|붓|부)|못 |음성|(?<!비)정상|깨끗"
                  r"|부인|미관찰|negative|normal|unremarkable|absent|\bnone\b|\bno\b|\bnot\b|denie|without")
_NEG_IDIOM = re.compile(r"(?:힘|입맛|밥맛|기운|의욕|정신|기력)[이도은가]? ?없|어쩔 ?수 ?없|수 ?없|상관 ?없|관계 ?없|틀림 ?없|끊임 ?없|없을 ?정도|아니라 ?[가-힣]*(?:있|나|해)")
_PRE_NEG = re.compile(r"\b(?:no|denies|denied|without|negative for|absence of|not)\b[^,;]{0,25}$")
_PRE_NEG_KO = re.compile(r"(?:^| )(?:안|못|전혀) ?$|(?<!비)정상(?:적인?)? ?$")  # "배는 안 아파요", "정상 동리듬"
_UNSURE =re.compile(r"모르|글쎄|기억이 ?안|확실하지|애매|잘 몰")
_PAREN = re.compile(r"\([^)]*\)")
_WORDY = re.compile(r"[가-힣a-z0-9]")
_BARE_NEG = re.compile(r"[가-힣a-z0-9 ]{1,14}? ?(?:없음|없다|없어요|없습니다|음성|안 보임|보이지 않음)")
_KO_NUM = {"한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5", "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9", "열": "10"}
_KO_COUNT = re.compile(r"(?<![가-힣])(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열) ?(번|차례|개|잔|갑|병|알|시간|달)")
_HIGH = re.compile(r"증가|상승|높|항진|elevat|\bhigh\b|increas|hyperactive|↑|\(h\)")
_LOW = re.compile(r"감소|저하|낮|decreas|\blow\b|↓|\(l\)")
_STOP = {"있음", "있다", "있어", "있고", "있는", "양성", "음성", "이상", "소견", "검사", "결과", "환자", "증상", "호소", "관찰", "확인",
         "동반", "약간", "심함", "심한", "심하게", "지속", "지속적", "상태", "수치", "정도", "부위", "경우", "최근", "현재", "과거", "병력",
         "과거력", "기왕력", "없음", "없다", "없어", "없고", "없는", "아님", "부정", "정상", "않음", "부인", "negative", "positive", "normal",
         "absent", "present", "no", "not", "denies", "without", "mmhg", "mg", "dl", "ml", "ul", "μl", "ng", "bpm", "회", "분", "도",
         "증가", "상승", "높음", "높은", "감소", "저하", "낮음", "낮은", "elevated", "high", "low", "increased", "decreased", "and", "the",
         "of", "with", "in", "on", "for", "관련", "의한", "인한", "시사", "의심", "가능성", "및", "또는", "전", "후", "때", "쪽",
         "적은", "적이", "적도", "것", "같아요", "그런", "특별한", "특별히", "조금", "많이", "계속", "자주", "가끔", "요즘",
         "있어요", "있었어요", "했어요", "해요", "돼요", "됐어요", "나요", "났어요",
         "mm", "cm", "kg", "mmol", "meq", "iu", "hpf", "cmh2o", "fl", "pg", "sec", "min", "ng", "dl", "ul", "mcg", "μg"}
_PARTICLES = sorted(["에서는", "에서", "으로", "에게", "까지", "부터", "이나", "이며", "이고", "하고", "처럼", "보다", "에는", "은", "는",
                     "이", "가", "을", "를", "도", "의", "에", "로", "과", "와", "만", "함", "됨", "임"], key=len, reverse=True)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("(-)", " 음성 ").replace("(+)", " 양성 ").replace("µ", "μ")
    t = re.sub(r"[–—−]", "-", t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    t = _KO_COUNT.sub(lambda m: _KO_NUM[m.group(1)] + m.group(2), t)  # "두 번 토했어요" → "2번"
    return re.sub(r"\s+", " ", t).strip()


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text)]


def _num_match(a: float, b: float) -> bool:
    """Same value up to rounding: 38.5 = 38.50, SpO2 96 ≠ 98, 36.8 ≠ 36.9; counts ≥ 1000 within 0.2 %;
    "WBC 14.2" (×10³/μL) = 14,200."""
    small, big = sorted((abs(a), abs(b)))
    if abs(a - b) <= (0.002 * big if big >= 1000 else 0.051):
        return True
    return big >= 1000 and abs(small * 1000 - big) <= 0.004 * big


def _negated_tail(tail: str) -> bool:
    return bool(_NEG.search(_NEG_IDIOM.sub(" ", tail)))


# --- evidence ---------------------------------------------------------------------------------------------------
@dataclass
class _Clause:
    text: str
    seg: int
    soft: bool  # ended by a comma (a list: "수포음, 천명음 없음" negates both)
    direction: str | None  # high | low | normal (lab value vs reference range / flag words)
    numbers: list[float]


@dataclass
class Evidence:
    """Parsed evidence of one case (build once per check; not shared across cases)."""

    raw: str
    clauses: list[_Clause] = field(default_factory=list)
    joined: str = ""
    compact: str = ""
    starts: list[int] = field(default_factory=list)
    _hits: dict = field(default_factory=dict)
    _claims: dict = field(default_factory=dict)  # parsed claims, per evidence object (i.e. per check of one case)
    _qa: dict = field(default_factory=dict)  # (question, answer) -> parsed question and answer

    def claim(self, text: str) -> "_Claim":
        if text not in self._claims:
            self._claims[text] = _parse_claim(text)
        return self._claims[text]

    def qa_pair(self, q: str, a: str) -> tuple["Evidence", "Evidence"]:
        if (q, a) not in self._qa:
            self._qa[(q, a)] = (Evidence.build(re.sub(r"[.,;?!]", " ", q)), Evidence.build(a))
        return self._qa[(q, a)]

    @classmethod
    def build(cls, text: str) -> "Evidence":
        ev = cls(text or "")
        for seg_i, seg in enumerate((text or "").split("\n")):
            seg = _norm(seg)
            for sent in _SENTENCE.split(seg):
                if not sent.strip() or _UNAVAILABLE.search(sent):
                    continue  # "결과가 제공되지 않습니다" grounds nothing, not even the test named in it
                pieces = [p for p in _CLAUSE_HARD.sub(lambda m: m.group(0) + "\x00", sent).split("\x00")]
                for piece in pieces:
                    parts = piece.split(",")
                    for k, part in enumerate(parts):
                        if part.strip():
                            ev.clauses.append(_clause(part.strip(), seg_i, k < len(parts) - 1))
        pos = 0
        for c in ev.clauses:
            ev.starts.append(pos)
            pos += len(c.text) + 1
        ev.joined = "\n".join(c.text for c in ev.clauses)
        ev.compact = ev.joined.replace(" ", "")
        return ev

    def token_hits(self, token: tuple[str, str]) -> dict[int, tuple[bool, bool]]:
        """clause index -> (negated, measured) for the first usable occurrence of a token in that clause."""
        if token in self._hits:
            return self._hits[token]
        kind, key = token[0], token[1]
        hits: dict[int, tuple[bool, bool]] = {}
        if kind == "lit" and token[2] not in self.compact:  # cheap pre-check before the regex
            self._hits[token] = hits
            return hits
        rx = _EVIDENCE_RE[key] if kind == "grp" else re.compile(key)
        for m in rx.finditer(self.joined):
            ci = bisect_right(self.starts, m.start()) - 1
            if ci in hits and not hits[ci][0]:
                continue
            c = self.clauses[ci]
            s, e = m.start() - self.starts[ci], m.end() - self.starts[ci]
            pol = self._polarity(ci, s, e, inside=kind == "grp" and key in _RAW_VARIANT_GROUPS)
            if pol is None:
                continue
            hits[ci] = (pol, False)
        if kind == "grp" and (meas := _GROUP_FLAGS[key].get("measure")):
            label, pred = _MEASURES[meas]
            for ci, c in enumerate(self.clauses):
                if ci in hits or not (lm := label.search(c.text)):
                    continue
                num = _NUM.search(c.text[lm.end():lm.end() + 25])
                if num and (res := pred(float(num.group()))) is not None:
                    hits[ci] = (not res, True)
        self._hits[token] = hits
        return hits

    def _polarity(self, ci: int, s: int, e: int, inside: bool) -> bool | None:
        """True = negated, False = affirmed, None = uncertain answer."""
        c = self.clauses[ci]
        tail = _PAREN.sub(" ", c.text[e:e + 30])  # "혈압 152/94 (양측 차이 없음)": the remark is not about 혈압
        # a bare list item ("수포음, 천명음 없음") takes the negation of a bare negated item right after it;
        # an item with its own words ("우상복부 압통, 반발통 없음", "의식 명료, ... 결손 없음") does not
        if c.soft and ci + 1 < len(self.clauses) and self.clauses[ci + 1].seg == c.seg \
                and not _WORDY.search(c.text[:s] + " " + c.text[e:]) and _BARE_NEG.fullmatch(self.clauses[ci + 1].text):
            tail += " " + self.clauses[ci + 1].text
        if _UNSURE.search(tail):
            return None
        span = c.text[s:e] if inside else ""
        head = c.text[:s]
        return _negated_tail(span + " " + tail) or bool(_PRE_NEG.search(head) or _PRE_NEG_KO.search(head))


def _clause(text: str, seg: int, soft: bool) -> _Clause:
    direction = None
    m = _REF_PAREN.search(text)
    if m:
        before = _numbers(text[:m.start()])
        ref = m.group(1)
        if before:
            v = before[-1]
            up = re.search(r"[<≤]=? ?(\d+(?:\.\d+)?)", ref)
            lo = re.search(r"[>≥]=? ?(\d+(?:\.\d+)?)", ref)
            rng = re.search(r"(\d+(?:\.\d+)?) ?[-~] ?(\d+(?:\.\d+)?)", ref)
            if rng:
                a, b = float(rng.group(1)), float(rng.group(2))
                direction = "high" if v > b else "low" if v < a else "normal"
            elif up:
                direction = "high" if v >= float(up.group(1)) else "normal"
            elif lo:
                direction = "low" if v <= float(lo.group(1)) else "normal"
        text = (text[:m.start()] + " " + text[m.end():]).strip()  # reference numbers never ground a claim
    if direction is None:
        flag = _HIGH.search(text) or _LOW.search(text)
        if flag and not _negated_tail(text[flag.end():flag.end() + 6]):
            direction = "high" if _HIGH.search(text) else "low"
    return _Clause(text, seg, soft, direction, _numbers(text))


def _as_evidence(evidence: "str | Evidence") -> Evidence:
    return evidence if isinstance(evidence, Evidence) else Evidence.build(evidence)


# --- claims -----------------------------------------------------------------------------------------------------
@dataclass
class _Claim:
    tokens: list[tuple[str, str]]  # ("grp", group) | ("lit", regex)
    optional: list[tuple[str, str]]
    numbers: list[float]
    negative: bool
    direction: str | None
    finding_like: bool


def _stem(word: str) -> str:
    for _ in range(2):  # stacked particles: "등으로의", "복부에서는"
        # a one-syllable particle must leave ≥ 2 syllables ("식도" stays; "등으로" → "등")
        p = next((p for p in _PARTICLES if word.endswith(p) and len(word) - len(p) >= (1 if len(p) > 1 else 2)), None)
        if p is None:
            break
        word = word[:-len(p)]
    return word


# last syllables of a conjugated verb/adjective ("뻗치는", "심한") — dropped so other endings still match
_VERB_END = set("는은한된던게히고서며어아여해지")


def _parse_claim(claim: str) -> _Claim:
    t = _norm(claim)
    t = _REF_PAREN.sub(" ", t)
    numbers = _numbers(t)
    t_nonum = _NUM.sub(" ", t)
    groups, rest, neg_inside = _claim_concepts(t_nonum)
    negative = neg_inside or bool(_NEG.search(_NEG_IDIOM.sub(" ", rest)))
    direction = "high" if _HIGH.search(rest) else "low" if _LOW.search(rest) else None
    lits, short = [], []
    for w in re.findall(r"[a-z][a-z0-9]*|[가-힣]+", rest):
        w = _stem(w)
        if w in _STOP or _NEG.match(w):  # "없어요", "않았어요", "정상" carry polarity, not content
            continue
        if len(w) < 2:
            short.append(w)
        elif re.fullmatch(r"[가-힣]+", w):
            core = w[:-1] if len(w) >= 3 and w[-1] in _VERB_END else w  # "좌심실비대한" / "좌심실비대"
            lits.append(("lit", _variant_regex(core), core))
        else:
            lits.append(("lit", rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", w))
    if not lits and not groups and numbers:  # "당 62 mg/dL": a one-syllable lab name counts when it has a value
        lits = [("lit", rf"(?<![가-힣]){re.escape(w)}(?![가-힣])", w) for w in short if re.fullmatch(r"[가-힣]", w)]
    required = [("grp", g) for g in groups if not _GROUP_FLAGS[g].get("optional")] + lits
    optional = [("grp", g) for g in groups if _GROUP_FLAGS[g].get("optional")]
    finding_like = bool([g for g in groups if not _GROUP_FLAGS[g].get("optional")] or numbers or _FINDING_MARK.search(t))
    return _Claim(required, optional, numbers, negative, direction, finding_like)


_FINDING_MARK = re.compile(r"징후|sign|청취|촉지|압통|양성|음성|상승|하강|증가|감소|저하|비대|음영|결절|종괴|병변|출혈|협착|폐색|부종"
                           r"|잡음|확장|침윤|경화|삼출|기흉|골절|궤양|천공|강직|마비|결손|저림|증후")


def _needed(n: int) -> int:
    return n if n <= 3 else max(n - 1, -(-3 * n // 4))  # all of 1–3 tokens; all but one (≥ 3/4) of more


def _ground(claim: _Claim, ev: Evidence, negative: bool | None = None,
            extra_numbers: tuple[float, ...] = ()) -> tuple[bool, float, str]:
    neg = claim.negative if negative is None else negative
    tokens = claim.tokens or claim.optional
    if not tokens:
        return False, 0.0, ""
    hits = [ev.token_hits(tok) for tok in tokens]
    need = _needed(len(tokens))
    best = (False, 0.0, "")
    for ci in sorted(set().union(*hits)):
        present = [h[ci] for h in hits if ci in h]
        score = len(present) / len(tokens)
        if len(present) < need:
            best = max(best, (False, score, ""), key=lambda x: x[1])
            continue
        c = ev.clauses[ci]
        if claim.numbers and not all(_number_near(ev, ci, x, same_clause=True) for x in claim.numbers):
            continue
        if extra_numbers and not all(_number_near(ev, ci, x, same_clause=False) for x in extra_numbers):
            continue
        clause_neg = any(p[0] for p in present)
        measured = any(p[1] for p in present)
        if not measured and not clause_neg and c.direction == "normal" and not claim.numbers:
            clause_neg = True  # a lab value inside its reference range reads as "no abnormality"
        if claim.direction and not neg:
            if clause_neg or (c.direction and c.direction not in (claim.direction,) and not measured):
                continue
        elif neg != clause_neg:
            continue
        return True, round(score, 3), c.text[:120]
    if not neg and not claim.direction and len(tokens) >= 2:
        return _ground_segment(claim, ev, tokens, hits, need, extra_numbers) or best
    return best


def _ground_segment(claim: _Claim, ev: Evidence, tokens: list, hits: list[dict], need: int,
                    extra_numbers: tuple[float, ...]) -> tuple[bool, float, str] | None:
    """Positive multi-token claims: the tokens may sit in different clauses of one response ("칼로 찢는 것처럼
    아파요" answers a question about the chest pain of the chief complaint). All tokens must be affirmed in that
    response; complaint-type symptoms (흉통, 복통, ...) may come from the initial info instead."""
    affirmed = [{ci for ci, (negd, _) in h.items() if not negd} for h in hits]
    borrow = [tok[0] == "grp" and _GROUP_FLAGS[tok[1]].get("cc", False) for tok in tokens]
    for seg in sorted({ev.clauses[ci].seg for a in affirmed for ci in a}):
        own = [{ci for ci in a if ev.clauses[ci].seg == seg} for a in affirmed]
        init = [b and any(ev.clauses[ci].seg == 0 for ci in a) for a, b in zip(affirmed, borrow)]
        # a token negated anywhere in this response disqualifies it
        if any(negd and ev.clauses[ci].seg == seg for h in hits for ci, (negd, _) in h.items()):
            continue
        got = sum(1 for o, i in zip(own, init) if o or i)
        if got < need or not any(own):
            continue
        pool = [n for c in ev.clauses if c.seg == seg for n in c.numbers]
        if not all(any(_num_match(x, y) for y in pool) for x in (*claim.numbers, *extra_numbers)):
            continue
        used = sorted(set().union(*own))
        return True, round(got / len(tokens), 3), " / ".join(ev.clauses[ci].text for ci in used)[:120]
    return None


def _number_near(ev: Evidence, ci: int, x: float, same_clause: bool) -> bool:
    c = ev.clauses[ci]
    if same_clause:
        pool = c.numbers or (ev.clauses[ci + 1].numbers if ci + 1 < len(ev.clauses) and ev.clauses[ci + 1].seg == c.seg else [])
    else:
        pool = [n for k in ev.clauses if k.seg == c.seg for n in k.numbers]
    return any(_num_match(x, y) for y in pool)


# --- yes/no answers to questions --------------------------------------------------------------------------------
_YES = re.compile(r"^\s*(?:네|예|응|맞아|맞습니다|있어|있습니다|그래|좀|약간|yes)")
_NO = re.compile(r"^\s*(?:아니|아뇨|없어|없습니다|안 |않|전혀|no\b)")


def _ground_qa(claim: _Claim, qa: list[tuple[str, str]], negative: bool, ev: Evidence) -> tuple[bool, float, str]:
    """A short yes/no answer to a question that names the claim: "yes" grounds a positive claim, "no" a negative one."""
    tokens = claim.tokens or claim.optional
    if not tokens or claim.numbers:
        return False, 0.0, ""
    for q, a in qa:
        a_n = _norm(a)
        if len(a_n) > 80 or _UNSURE.search(a_n):
            continue
        answer = False if _NO.search(a_n) else True if _YES.search(a_n) else None
        if answer is None or answer == negative:
            continue
        qev, aev = ev.qa_pair(q, a)
        hit = sum(1 for tok in tokens if qev.token_hits(tok))
        if not qev.clauses or hit < _needed(len(tokens)):
            continue
        # "네, 기침은 없어요": the answer itself names the finding with the other polarity
        if any(negd != negative for tok in tokens for negd, _ in aev.token_hits(tok).values()):
            continue
        return True, round(hit / len(tokens), 3), f"Q: {q[:60]} → A: {a[:60]}"
    return False, 0.0, ""


# --- public API -------------------------------------------------------------------------------------------------
def evidence_text(state) -> str:
    """Initial info + every environment response, one line each (never the doctor's own questions)."""
    parts = [state.initial_info] + [t.response for t in state.turns]
    return "\n".join(re.sub(r"\s*\n\s*", "; ", p or "") for p in parts)


def is_grounded(claim: str, evidence: "str | Evidence") -> tuple[bool, float, str]:
    """(grounded, score 0..1 = share of the claim's concept tokens found in the best clause, matched clause text)."""
    return _ground(_parse_claim(claim), _as_evidence(evidence))


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
            match = next((u for u in unavailable if similarity(u, f.item) >= 0.3
                          or _ground(ev.claim(f.item), Evidence.build(u), negative=False)[0]), None)
            f.verified, f.span = match is not None, (match or "")[:120]
        else:
            claim = ev.claim(f.item)
            detail = ev.claim(f.detail) if f.detail else None
            if detail and not claim.direction and detail.direction and not detail.negative:
                claim = replace(claim, direction=detail.direction)  # item "CRP", detail "상승"
            negative = f.status == "음성" or claim.negative
            extra = tuple(_numbers(_TIME_NUM.sub(" ", _norm(f.detail)))) if f.detail else ()
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


_TAG = " (미확인)"
# numbers in a finding's detail that are durations, counts or ages ("3일 전부터") are not checked: the patient says
# "사흘 전" while the model writes "3일"
_TIME_NUM = re.compile(r"\d+(?:\.\d+)? ?(?:일|주|개월|달|년|시간|분|초|세|살|번|차례|시|층|갑|병|잔|kg|cm)")


def _item_ok(item: str, ev: Evidence, qa: list[tuple[str, str]] | None) -> bool:
    claim = ev.claim(item)
    if not claim.finding_like:
        return True  # reasoning, not a stated finding
    if _ground(claim, ev)[0]:
        return True
    return bool(qa) and _ground_qa(claim, qa, claim.negative, ev)[0]


_REASON_SPLIT = re.compile(r"[,;/·+]|\n|(?<!\d)\.|\.(?!\d)| (?:및|그리고|또는|and|with) |(?<=[가-힣])(?:와|과|고|며) "
                           r"|(?<=[가-힣0-9A-Za-z%)])(?:으로|로|에서) |때문|이므로|므로|시사|의심|고려|가능성|감별|진단|배제|필요|보아|미루어|소견상")


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
    Updates the ledgers in place (state.ddx is re-synced when DDx evidence was removed) and returns a report."""
    ev = Evidence.build(evidence_text(state))
    qa = [(t.action.content, t.response) for t in state.turns if getattr(t.action.type, "value", "") == "ASK"]
    bad = check_findings(state.findings, ev, qa, state.unavailable())
    removed = check_ddx_support(state.ddx_ledger, ev, qa, ddx_mode)
    if removed and ddx_mode == "drop":
        state.ddx = state.ddx_ledger.as_list()
    reason = state.turns[-1].action.reason if state.turns else ""
    return {
        "findings_checked": sum(1 for f in state.findings.items if f.verified is not None),
        "findings_unverified": len(bad),
        "unverified_examples": [f"{f.item} [{f.status}]" for f in bad[:5]],
        "ddx_removed": len(removed),
        "ddx_removed_examples": [f"{r['dx']}/{r['side']}: {r['item']}" for r in removed[:5]],
        "reason_ungrounded": ungrounded_in_text(reason, ev, qa)[:5],
    }
