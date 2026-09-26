import re

_NON_WORD = re.compile(r"[^0-9a-z가-힣]")
SIMILAR = 0.7  # char-bigram Jaccard above which two actions count as the same request


def _bigrams(text: str) -> set[str]:
    t = _NON_WORD.sub("", text.lower())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(a: str, b: str) -> float:
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x and y else float(a.strip().lower() == b.strip().lower())


# --- history-taking intent inside an EXAM/TEST request ----------------------------------------------------------
# Cues that the content is really a question for the patient (only the patient can answer it).
_HISTORY_CUES = [
    re.compile(r"환자(에게|분께|분에게)|물어|여쭤|여쭈|문진|병력"),  # explicit history-taking
    re.compile(r"(였|았|었|했|됐|났|졌|왔|셨)는지|적이?\s*(있|없)|경험|복용|드시|먹"),  # past experience / medication
    re.compile(r"언제|얼마나|며칠|몇\s*(시간|일|주|달|개월|년|번)|처음\s*(시작|발생)"),  # onset / timing
    re.compile(r"(통증|아프|아픈|아파|뻣뻣|저리|저림|어지러|메스꺼|구역|불편|느끼|느낌|숨이\s*차|가렵|두근)[^.]*?는지"),  # subjective symptom
    re.compile(r"(으?시|하시)는지|(으?신|하신|계신)지|신\s*적"),  # honorific verb addressed to the patient
]
# Nouns naming a physical-exam technique or a test: if present, the request is a real exam/test (do not convert).
_EXAM_TECHNIQUE = re.compile(
    r"촉진|청진|타진|시진|측정|징후|반사|검사|검진|신경학적|직장\s*수지|내진|안저|촬영|초음파|심전도|혈액|소변|배양|수치"
    r"|sign|test|maneuver|reflex|ct|mri|ecg|ekg|x-?ray", re.IGNORECASE)


def looks_like_history_question(text: str) -> bool:
    """True when an EXAM/TEST request is really history-taking (a question for the patient) and names no exam
    technique or test. Mixed requests (question + real exam) stay as they are."""
    return any(p.search(text) for p in _HISTORY_CUES) and not _EXAM_TECHNIQUE.search(text)


# --- diagnosis-name normalisation for the DDx ledger ------------------------------------------------------------
_PARENS = re.compile(r"[(\[（][^)\]）]*[)\]）]")
_PAREN_INNER = re.compile(r"[(\[（]([^)\]）]*)[)\]）]")
# spelling / wording variants seen in model output, applied after lowercasing and removing whitespace/punctuation
_DX_VARIANTS = [
    ("타카야수", "다카야수"),  # Takayasu transliteration
    ("롱qt", "긴qt"), ("longqt", "긴qt"), ("qt연장", "긴qt"),  # long QT
    ("길랭바레", "길랑바레"), ("귈랭바레", "길랑바레"),  # Guillain-Barré transliteration
    ("쉐그렌", "쇼그렌"), ("쇼겐", "쇼그렌"),  # Sjögren
    ("베세트", "베체트"),  # Behçet
    ("쿠씽", "쿠싱"),  # Cushing
]
_DX_HEDGE = re.compile(r"(의증|의심|가능성|추정)$")  # "폐색전증 의심" is the same candidate as "폐색전증"


def _dx_norm(text: str) -> str:
    t = _NON_WORD.sub("", text.lower())
    for a, b in _DX_VARIANTS:
        t = t.replace(a, b)
    return _DX_HEDGE.sub("", t)


def dx_keys(name: str) -> tuple[str, str]:
    """(normalised name without parenthesised text, normalised text inside the parentheses — usually English)."""
    core = _dx_norm(_PARENS.sub(" ", name))
    inner = _dx_norm(" ".join(_PAREN_INNER.findall(name)))
    return core or inner, inner


def same_dx(a: str, b: str, threshold: float = 0.75) -> bool:
    (ca, ea), (cb, eb) = dx_keys(a), dx_keys(b)
    keys_a, keys_b = {k for k in (ca, ea) if k}, {k for k in (cb, eb) if k}
    if keys_a & keys_b:  # same name, same English name, or one side's English matches the other's plain name
        return True
    return similarity(ca, cb) >= threshold
