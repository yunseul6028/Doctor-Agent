"""Specialty routing and evidence slicing for runtime specialist consults. Owned by knowledge-rag.

Six specialty ids (fixed): cardio, resp_id, gi_liver, neuro, rheum_immune, peds_obgyn.

API
    specialty_of(dx_name)         -> one of the six ids | None   (diagnosis name, Korean/English free text)
    specialty_detail(dx_name)     -> {"specialty", "group", "how", "code", "name"}   (group also names buckets outside
                                     the six: endo_metab, renal_uro, heme_onc, psych, derm, ent_eye, msk_ortho,
                                     tox_trauma, symptom, other)
    route(state)                  -> (specialty | None, share, [Korean reasons])
    resources(specialty, state)   -> {"criteria", "rules", "protocols", "kb_candidates"} (each bounded; {} on error)
    render_resources(res, max_chars=700) -> short Korean text for a consult prompt

How a name is mapped (specialty_detail)
    1. OVERRIDES: a short curated regex list for names whose ICD chapter misleads (candidal endocarditis is B37 but a
       heart problem, pregnancy words, neonatal words, ...). First match wins.
    2. knowledge/kb.py normalize_diagnosis(): KCD code → KCD_TABLE (chapter/block ranges, most specific first).
    3. No KCD code but a Disease Ontology id: nearest mapped DO ancestor (DO_MAP; profile field "parents", DO CC0).
    4. FALLBACK: organ keywords in the name (only when the KB has nothing).
    Anything else → specialty None, group "other".
    peds_obgyn by name covers pregnancy (KCD O), perinatal (P), most congenital (Q) and female genital (N70-N98)
    conditions; children's and pregnant patients' other diseases are routed there by patient context (route).

route(state) — the exact rule
    Candidates: live entries of state.ddx_ledger (status != 배제) in ranked order, else state.ddx; top TOP_K.
    Mass: the candidate's probability p when any p > 0 (entries without p get 0), else rank weights 1/(rank+1).
    share(s) = mass of candidates with specialty_of == s / total mass (unmapped candidates stay in the total).
    Patient-context override (checked first):
      infant   age < 1 year (nlp.findings.age_from_text on the initial info, then clinical_rules.rule_age_years)
               → peds_obgyn, share 1.0 (every infant diagnosis is a pediatric problem).
      child    1 <= age < 18, and pregnant = safety.protocols current-pregnancy predicate on initial info + responses
               (female, not < 10 y, pregnancy affirmed or "N주 임신"/amenorrhoea wording not negated).
               A candidate is relevant when specialty_of == peds_obgyn, or (child) its KCD code is in chapter P/Q, the
               KB profile is pediatric-named, or it is on PEDIATRIC_DX; (pregnant) its code is in chapter O or its
               name has a pregnancy word.
               → peds_obgyn with share = relevant mass share, when the top-1 candidate is relevant or that share
               >= CONTEXT_SHARE (0.3). Otherwise the organ routing below applies (reason says why).
    Organ routing: best = argmax share over the six (ties: fixed SPECIALTIES order); None when nothing maps.
    The caller decides when to consult (suggested: MIN_TURNS = 3 and share >= MIN_SHARE = 0.6).

Stdlib only, CPU only, no network, deterministic. Per-case: nothing is stored between calls except the KB's own
read-only indexes (knowledge/kb.py). Every public function catches all errors. Call warm() once at start-up so that
the first call does not pay for the KB's lazy name indexes (~1 s load + ~0.3 s index build).
"""
from __future__ import annotations

import re

SPECIALTIES: tuple[str, ...] = ("cardio", "resp_id", "gi_liver", "neuro", "rheum_immune", "peds_obgyn")
SPECIALTY_KO: dict[str, str] = {"cardio": "심장·혈관", "resp_id": "호흡기·감염", "gi_liver": "소화기·간담췌",
                                "neuro": "신경", "rheum_immune": "류마티스·면역·알레르기", "peds_obgyn": "소아·산부인과"}
TOP_K = 5
MIN_TURNS = 3        # suggested caller gate: consult only after this many turns ...
MIN_SHARE = 0.6      # ... and when the routed specialty holds at least this share of the top-DDx mass
CONTEXT_SHARE = 0.3  # child/pregnant override: relevant mass share needed when the top-1 candidate is not relevant
# KB fuzzy name matching (char-bigram Dice) costs 40-100 ms on long English names: at runtime it runs only for short
# Korean names (FUZZY_MAX_KEY compact characters, ~1 ms). FUZZY = True turns it on for every name (offline A/B).
FUZZY = False
FUZZY_MAX_KEY = 14
MAX_RESPONSES = 8  # recent responses read by the criteria evaluation
MAX_FINDINGS = 8   # positive (and negative) findings passed to kb.candidates
MAX_ITEMS = {"criteria": 3, "rules": 4, "protocols": 3, "kb_candidates": 4}

# --------------------------------------------------------------------------------------------------------------------
# Name → specialty tables
# --------------------------------------------------------------------------------------------------------------------

# (regex on the lowercased name, specialty or None, group). Checked before the KB. Keep this list short.
OVERRIDES: tuple[tuple[str, str | None, str], ...] = (
    (r"심내막염|endocarditis", "cardio", "cardio"),                       # candidal/Q fever codes sit in A/B
    (r"임신|자간|태반|산후|산욕|분만|(?<![가-힣])유산(?!균)|양수|태아|자궁\s?외|난관\s?임신|pregnan|eclampsia|placent"
     r"|postpartum|ectopic|miscarriage|abortion|hyperemesis", "peds_obgyn", "peds_obgyn"),
    (r"신생아|미숙아|모유\s?황달|newborn|neonat|breast ?milk jaundice", "peds_obgyn", "peds_obgyn"),
    (r"수막염|뇌수막|뇌염|meningit|encephalit", "neuro", "neuro"),       # infectious codes (A39, A87, B37.5) → neuro
    (r"폐렴|늑막염|흉막염|폐섬유증|간질성\s?폐|pneumonia|pleurisy|pleuritis|pulmonary fibrosis|interstitial lung",
     "resp_id", "resp_id"),  # DO files interstitial lung disease under connective tissue disease
    (r"간염|혈색소증|혈색소침착|윌슨|장간막\s?허혈|hepatitis|hemochromatosis|haemochromatosis|wilson|mesenteric isch",
     "gi_liver", "gi_liver"),
    (r"패혈증|균혈증|sepsis|septic|bacter[ae]mia", "resp_id", "resp_id"),
    (r"아나필락시스|혈청병|anaphyla|serum sickness", "rheum_immune", "rheum_immune"),
)

# KCD/ICD-10 code ranges → (specialty, group). A code matches an entry when its 3-character category lies in
# [lo, hi] or the code starts with a 4-character prefix given as lo == hi. Most specific entries first.
_O = None
KCD_TABLE: tuple[tuple[str, str, str | None, str], ...] = (
    # 4-character exceptions
    ("E830", "E830", "gi_liver", "gi_liver"), ("E831", "E831", "gi_liver", "gi_liver"),  # Wilson, haemochromatosis
    ("M797", "M797", "rheum_immune", "rheum_immune"),  # fibromyalgia
    ("I776", "I776", "rheum_immune", "rheum_immune"),  # arteritis, unspecified
    ("R091", "R091", "resp_id", "resp_id"),            # pleurisy
    ("D693", "D693", _O, "heme_onc"),
    ("E28", "E28", "peds_obgyn", "peds_obgyn"),        # ovarian dysfunction (PCOS)
    # infectious
    ("A00", "A09", "gi_liver", "gi_liver"), ("A80", "A89", "neuro", "neuro"), ("B15", "B19", "gi_liver", "gi_liver"),
    ("A00", "B99", "resp_id", "resp_id"),
    # neoplasms by site
    ("C15", "C26", "gi_liver", "gi_liver"), ("C30", "C39", "resp_id", "resp_id"), ("C45", "C45", "resp_id", "resp_id"),
    ("C51", "C58", "peds_obgyn", "peds_obgyn"), ("C69", "C69", _O, "ent_eye"), ("C70", "C72", "neuro", "neuro"),
    ("D01", "D01", "gi_liver", "gi_liver"), ("D02", "D02", "resp_id", "resp_id"), ("D06", "D06", "peds_obgyn", "peds_obgyn"),
    ("D12", "D13", "gi_liver", "gi_liver"), ("D25", "D28", "peds_obgyn", "peds_obgyn"), ("D32", "D33", "neuro", "neuro"),
    ("D37", "D37", "gi_liver", "gi_liver"), ("D38", "D38", "resp_id", "resp_id"), ("D39", "D39", "peds_obgyn", "peds_obgyn"),
    ("D42", "D43", "neuro", "neuro"),
    ("C00", "D49", _O, "heme_onc"),
    # blood / immune
    ("D80", "D84", "rheum_immune", "rheum_immune"), ("D86", "D86", "resp_id", "resp_id"),
    ("D89", "D89", "rheum_immune", "rheum_immune"), ("D50", "D89", _O, "heme_onc"),
    # endocrine / metabolic
    ("E84", "E84", "resp_id", "resp_id"), ("E00", "E90", _O, "endo_metab"),
    ("F00", "F99", _O, "psych"),
    ("G00", "G99", "neuro", "neuro"),
    ("H81", "H82", "neuro", "neuro"), ("H00", "H95", _O, "ent_eye"),
    # circulatory
    ("I60", "I69", "neuro", "neuro"), ("I85", "I85", "gi_liver", "gi_liver"), ("I88", "I89", _O, "heme_onc"),
    ("I00", "I99", "cardio", "cardio"),
    ("J00", "J99", "resp_id", "resp_id"),
    ("K00", "K14", _O, "ent_eye"), ("K20", "K93", "gi_liver", "gi_liver"),
    ("L50", "L50", "rheum_immune", "rheum_immune"), ("L00", "L99", _O, "derm"),
    # musculoskeletal: inflammatory / systemic → rheum_immune, the rest → msk_ortho
    ("M00", "M19", "rheum_immune", "rheum_immune"), ("M30", "M36", "rheum_immune", "rheum_immune"),
    ("M45", "M46", "rheum_immune", "rheum_immune"), ("M60", "M60", "rheum_immune", "rheum_immune"),
    ("M00", "M99", _O, "msk_ortho"),
    ("N70", "N98", "peds_obgyn", "peds_obgyn"), ("N00", "N99", _O, "renal_uro"),
    ("O00", "O99", "peds_obgyn", "peds_obgyn"), ("P00", "P96", "peds_obgyn", "peds_obgyn"),
    # congenital by organ
    ("Q00", "Q07", "neuro", "neuro"), ("Q20", "Q28", "cardio", "cardio"), ("Q30", "Q34", "resp_id", "resp_id"),
    ("Q38", "Q45", "gi_liver", "gi_liver"), ("Q60", "Q64", _O, "renal_uro"), ("Q65", "Q79", _O, "msk_ortho"),
    ("Q85", "Q85", "neuro", "neuro"), ("Q00", "Q99", "peds_obgyn", "peds_obgyn"),
    ("R00", "R99", _O, "symptom"),
    ("T78", "T78", "rheum_immune", "rheum_immune"), ("S00", "T98", _O, "tox_trauma"), ("V01", "Y98", _O, "tox_trauma"),
    ("U07", "U07", "resp_id", "resp_id"),
)

# Disease Ontology class → (specialty, group); nearest mapped ancestor wins, ties by this order.
DO_MAP: dict[str, tuple[str | None, str]] = {
    "DOID:6713": ("neuro", "neuro"),                 # cerebrovascular disease
    "DOID:865": ("rheum_immune", "rheum_immune"),    # vasculitis
    "DOID:3118": ("gi_liver", "gi_liver"),           # hepatobiliary disease
    "DOID:409": ("gi_liver", "gi_liver"),            # liver disease
    "DOID:9741": ("gi_liver", "gi_liver"),           # biliary tract disease
    "DOID:26": ("gi_liver", "gi_liver"),             # pancreas disease
    "DOID:77": ("gi_liver", "gi_liver"),             # gastrointestinal system disease
    "DOID:1287": ("cardio", "cardio"),               # cardiovascular system disease
    "DOID:1579": ("resp_id", "resp_id"),             # respiratory system disease
    "DOID:863": ("neuro", "neuro"),                  # nervous system disease
    "DOID:229": ("peds_obgyn", "peds_obgyn"),        # female reproductive system disease
    "DOID:848": ("rheum_immune", "rheum_immune"),    # arthritis
    "DOID:1575": ("rheum_immune", "rheum_immune"),   # rheumatic disease
    "DOID:65": ("rheum_immune", "rheum_immune"),     # connective tissue disease
    # system-wide classes last: an organ class at the same depth wins (IPF: lung + autoimmune → resp_id)
    "DOID:417": ("rheum_immune", "rheum_immune"),    # autoimmune disease
    "DOID:0060056": ("rheum_immune", "rheum_immune"),  # hypersensitivity reaction disease
    "DOID:2914": ("rheum_immune", "rheum_immune"),   # immune system disease
    "DOID:0050117": ("resp_id", "resp_id"),          # disease by infectious agent
    "DOID:18": (None, "renal_uro"), "DOID:48": (None, "renal_uro"),
    "DOID:28": (None, "endo_metab"), "DOID:0014667": (None, "endo_metab"),
    "DOID:74": (None, "heme_onc"), "DOID:162": (None, "heme_onc"),
    "DOID:150": (None, "psych"), "DOID:16": (None, "derm"),
    "DOID:5614": (None, "ent_eye"), "DOID:2742": (None, "ent_eye"),
    "DOID:17": (None, "msk_ortho"),
    "DOID:0080015": ("peds_obgyn", "peds_obgyn"),    # physical disorder (congenital)
}
_DO_ORDER = {k: i for i, k in enumerate(DO_MAP)}

# organ keywords, used only when the KB cannot resolve the name
FALLBACK: tuple[tuple[str, str | None, str], ...] = (
    (r"골절|탈구|염좌|추간판|디스크|힘줄|건\s?파열|fracture|dislocation|sprain|disc herniation|tendon", None, "msk_ortho"),
    (r"심장|심근|협심|부정맥|판막|심낭|심실|심방|대동맥|방실|빈맥|서맥|cardi|heart|coronary|aort|arrhythm|tachycard",
     "cardio", "cardio"),
    (r"뇌졸중|뇌경색|뇌출혈|뇌|신경|척수|치매|두통|경련|발작|stroke|cerebr|neur|brain|spinal|dementia|seizure", "neuro",
     "neuro"),
    (r"폐|기관지|기관|흉막|늑막|호흡|결핵|copd|pulmon|lung|bronch|pleur|respirat", "resp_id", "resp_id"),
    (r"간경변|담관|담낭|담도|췌장|위장|식도|대장|소장|십이지장|위염|장염|hepat|biliar|pancrea|gastr|esophag|colon|bowel"
     r"|intestin", "gi_liver", "gi_liver"),
    (r"관절|류마|루푸스|혈관염|근염|lupus|arthrit|vasculit|myosit|rheumat", "rheum_immune", "rheum_immune"),
    (r"자궁|난소|난관|질염|소아|영아|uter|ovar|vagin|pediatric|infant", "peds_obgyn", "peds_obgyn"),
    # organ words first: an infection of a named organ goes to that organ ("감염성 장염" → gi_liver)
    (r"감염|균|바이러스|진균|infect|viral|fung", "resp_id", "resp_id"),
)

# pediatric-predominant diseases (child override relevance); regex on the lowercased name
PEDIATRIC_DX = re.compile(
    r"가와사키|kawasaki|크룹|croup|장중첩|intussuscep|유문\s?협착|pyloric|세기관지염|bronchiolitis|rsv|열성\s?경련|febrile seiz"
    r"|히르슈?슈프룽|hirschsprung|선천성?\s?거대\s?결장|괴사성\s?장염|necrotizing enterocol|수족구|hand,? foot|돌발진|장미진|roseola"
    r"|전염성\s?홍반|erythema infectiosum|레그|perthes|대퇴골두\s?골단|slipped capital|윌름스|wilms|신경모세포종|neuroblastoma"
    r"|림프모구|lymphoblastic|담도\s?폐쇄|biliary atresia|iga\s?혈관염|헤노흐|henoch|소아|juvenile|childhood|infantile|성홍열|scarlet"
    r"|홍역|measles|풍진|rubella|볼거리|유행성\s?이하선염|mumps|백일해|pertussis|급성\s?중이염|otitis media|선천")
_PREG_WORDS = re.compile(r"임신|자간|태반|산후|산욕|분만|유산|양수|태아|자궁\s?외|pregnan|eclampsia|placent|postpartum|ectopic"
                         r"|gestation")

# --------------------------------------------------------------------------------------------------------------------
# Evidence slicing tables (ids of existing modules → specialties)
# --------------------------------------------------------------------------------------------------------------------

CRITERIA_SPECIALTY: dict[str, tuple[str, ...]] = {
    "sle_2019": ("rheum_immune",), "ra_2010": ("rheum_immune",), "takayasu_2022": ("rheum_immune", "cardio"),
    "gca_2022": ("rheum_immune", "neuro"), "kawasaki_aha2017": ("peds_obgyn", "rheum_immune", "cardio"),
    "duke_iscvid_2023": ("cardio", "resp_id"), "kdigo_aki_2012": (), "dka_hhs_2024": (), "light_1972": ("resp_id",),
    "sepsis3_2016": ("resp_id",), "jones_2015": ("rheum_immune", "cardio", "peds_obgyn"), "mcdonald_2017": ("neuro",),
    "ichd3_migraine_tth": ("neuro",), "bipolar_dsm5tr": (), "gout_2015": ("rheum_immune",),
}
RULE_SPECIALTY: dict[str, tuple[str, ...]] = {
    "wells_pe": ("cardio", "resp_id"), "perc": ("cardio", "resp_id"), "heart": ("cardio",), "add_rs": ("cardio",),
    "qsofa": ("resp_id",), "curb65": ("resp_id",), "centor": ("resp_id",), "mcisaac": ("resp_id", "peds_obgyn"),
    "ottawa_sah": ("neuro",), "cchr": ("neuro",), "abcd2": ("neuro",), "alvarado": ("gi_liver",),
    "bisap": ("gi_liver",), "gbs": ("gi_liver",), "pecarn_head_lt2": ("peds_obgyn", "neuro"),
    "pecarn_head_ge2": ("peds_obgyn", "neuro"), "nexus": (), "ccsr": (), "sfsr": ("cardio",), "csrs": ("cardio",),
    "kocher": ("peds_obgyn", "rheum_immune"), "pas": ("peds_obgyn", "gi_liver"), "spesi": ("cardio", "resp_id"),
    "pecarn_febrile_infant": ("peds_obgyn", "resp_id"),
}
# chief-complaint categories of clinical_rules / safety.protocols → specialties
CATEGORY_SPECIALTY: dict[str, tuple[str, ...]] = {
    "chest_pain": ("cardio",), "dyspnea": ("resp_id", "cardio"), "headache": ("neuro",), "neuro": ("neuro",),
    "fever": ("resp_id",), "abdominal_pain": ("gi_liver", "peds_obgyn"), "allergy": ("rheum_immune",),
    "syncope": ("cardio", "neuro"), "palpitations": ("cardio",), "hemoptysis_cough": ("resp_id",),
    "jaundice": ("gi_liver", "peds_obgyn"), "joint": ("rheum_immune",), "back_pain": (), "rash": ("rheum_immune",),
    "pruritus": ("gi_liver",), "edema": ("cardio",), "menstrual": ("peds_obgyn",), "fatigue": (),
    "cognitive": ("neuro",), "psychiatric": (), "urticaria_chronic": ("rheum_immune",), "hearing_loss": (),
    "neck_mass": (), "bleeding": (), "chronic_weakness": ("neuro", "rheum_immune"),
    "bilious_vomiting": ("peds_obgyn", "gi_liver"),
}

# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------

_OVR = [(re.compile(p), s, g) for p, s, g in OVERRIDES]
_FB = [(re.compile(p), s, g) for p, s, g in FALLBACK]
_CODE_CLEAN = re.compile(r"[^0-9A-Z]")


def _kb():
    """The process-wide read-only KB singleton of knowledge/kb.py (None when the data file is missing)."""
    from doctor_agent.knowledge import kb
    if kb._KB is not None:  # already loaded: skip the file-exists check (a stat per call)
        return kb._KB
    return kb.get_kb() if kb.available() else None


def warm() -> bool:
    """Load the KB and build its lazy name indexes once (process start-up), so later calls stay fast."""
    try:
        k = _kb()
        if k is None:
            return False
        # build the KCD table and the lazy name/bigram indexes on the Korean, English and mixed paths
        for probe in ("가나다라마바 증후군", "Qwerty zxcvb disorder due to left occlusion", "기타 zxcvb 증후군"):
            k.normalize_diagnosis(probe, fuzzy=True)  # fuzzy on: builds the bigram index once (~0.3 s)
        k.candidates(["발열", "기침"], k=3, sex="여성", age=30.0)  # sex table, prevalence and test postings
        return True
    except Exception:
        return False


def _fuzzy_ok(name: str) -> bool:
    key = re.sub(r"[^0-9a-z가-힣]", "", (name or "").lower())
    return FUZZY or (len(key) <= FUZZY_MAX_KEY and len(re.findall(r"[가-힣]", key)) >= 0.8 * len(key))


def _code_entry(code: str) -> tuple[str | None, str] | None:
    c = _CODE_CLEAN.sub("", (code or "").upper())
    if len(c) < 3:
        return None
    c3 = c[:3]
    for lo, hi, spec, grp in KCD_TABLE:
        if len(lo) == 4:
            if c.startswith(lo):
                return spec, grp
        elif lo <= c3 <= hi:
            return spec, grp
    return None


def _do_entry(k, doid: str) -> tuple[str | None, str] | None:
    level, seen = [doid], {doid}
    for _ in range(12):
        hits = [x for x in level if x in DO_MAP and x != doid]
        if hits:
            return DO_MAP[min(hits, key=lambda x: _DO_ORDER[x])]
        nxt = []
        for x in level:
            i = k.by_id.get(x)
            if i is None:
                continue
            for p, _src in k.diseases[i].get("parents", ()):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        if not nxt:
            return None
        level = nxt
    return None


def _detail(name: str) -> dict:
    out = {"specialty": None, "group": "other", "how": "none", "code": "", "name": ""}
    t = (name or "").strip()
    if not t:
        return out
    low = t.lower()
    for rx, spec, grp in _OVR:
        if rx.search(low):
            return {**out, "specialty": spec, "group": grp, "how": "override"}
    k = _kb()
    if k is not None:
        n = k.normalize_diagnosis(t, fuzzy=_fuzzy_ok(t))
        if n:
            out.update(code=n.get("code", ""), name=n.get("name", ""))
            e = _code_entry(n.get("code", ""))
            if e:
                return {**out, "specialty": e[0], "group": e[1], "how": "kcd"}
            if n.get("id", "").startswith("DOID:"):
                e = _do_entry(k, n["id"])
                if e:
                    return {**out, "specialty": e[0], "group": e[1], "how": "do"}
    for rx, spec, grp in _FB:
        if rx.search(low):
            return {**out, "specialty": spec, "group": grp, "how": "keyword"}
    return out


def specialty_detail(dx_name: str) -> dict:
    """{"specialty": id | None, "group": specialty or out-of-six bucket, "how": override|kcd|do|keyword|none,
    "code": KCD code or "", "name": KB standard name or ""}."""
    try:
        return _detail(dx_name)
    except Exception:
        return {"specialty": None, "group": "other", "how": "error", "code": "", "name": ""}


def specialty_of(dx_name: str) -> str | None:
    """One of SPECIALTIES for a diagnosis name, or None (outside the six / unknown). Never raises."""
    return specialty_detail(dx_name)["specialty"]


# --------------------------------------------------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------------------------------------------------

def _candidates(state) -> list[tuple[str, float]]:
    """[(name, weight)] of the top live DDx candidates (see module docstring)."""
    names: list[tuple[str, float]] = []
    led = getattr(state, "ddx_ledger", None)
    if led is not None and getattr(led, "entries", None):
        names = [(e.dx, float(e.p or 0.0)) for e in led.ranked() if e.status != "배제"]
    if not names:
        for d in getattr(state, "ddx", None) or []:
            if isinstance(d, dict) and str(d.get("dx", "")).strip() and d.get("status") != "배제":
                try:
                    p = float(d.get("p", 0.0) or 0.0)
                except (TypeError, ValueError):
                    p = 0.0
                names.append((str(d["dx"]).strip(), p))
        names.sort(key=lambda x: -x[1])
    names = names[:TOP_K]
    if not names:
        return []
    if sum(p for _, p in names) > 0:
        return [(n, max(p, 0.0)) for n, p in names]
    return [(n, 1.0 / (r + 1)) for r, (n, _) in enumerate(names)]


def _context(state) -> tuple[float | None, bool]:
    """(patient age in years or None, currently pregnant)."""
    from doctor_agent.knowledge.clinical_rules import rule_age_years
    from doctor_agent.nlp.findings import age_from_text
    from doctor_agent.safety.protocols import PREDICATES
    init = getattr(state, "initial_info", "") or ""
    age = age_from_text(init)
    if age is None:
        age = rule_age_years(init)
    text = " ".join([init] + [getattr(t, "response", "") or "" for t in getattr(state, "turns", [])])
    return age, bool(PREDICATES["current_pregnancy"](init, text))


def _relevant(name: str, det: dict, child: bool, pregnant: bool) -> bool:
    if det["specialty"] == "peds_obgyn":
        return True
    low = name.lower()
    code = _CODE_CLEAN.sub("", det.get("code", "").upper())
    if child:
        if code[:1] in ("P", "Q") or PEDIATRIC_DX.search(low):
            return True
        k = _kb()
        if k is not None:
            i = k._resolve(name, fuzzy=_fuzzy_ok(name))
            if i is not None and k._age[i] == "child":
                return True
    if pregnant and (code[:1] == "O" or _PREG_WORDS.search(low)):
        return True
    return False


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def route(state) -> tuple[str | None, float, list[str]]:
    """(specialty | None, share of the top-DDx mass, Korean reasons). Never raises: (None, 0.0, [reason])."""
    try:
        cands = _candidates(state)
        if not cands:
            return None, 0.0, ["감별 진단 후보가 없어 분과를 정할 수 없음"]
        total = sum(w for _, w in cands) or 1.0
        dets = [(n, w, specialty_detail(n)) for n, w in cands]
        mass: dict[str, float] = {s: 0.0 for s in SPECIALTIES}
        members: dict[str, list[str]] = {s: [] for s in SPECIALTIES}
        unmapped: list[str] = []
        for n, w, d in dets:
            if d["specialty"] in mass:
                mass[d["specialty"]] += w
                members[d["specialty"]].append(f"{n} {_pct(w / total)}")
            else:
                unmapped.append(f"{n}({d['group']})")
        age, pregnant = _context(state)
        child = age is not None and age < 18
        reasons: list[str] = []
        if age is not None and age < 1:
            return "peds_obgyn", 1.0, [f"영아(나이 {age:.2f}세): 모든 감별 진단을 소아과 관점에서 봐야 함"]
        if child or pregnant:
            ctx = "소아(나이 %d세)" % int(age) if child else "임신 중"
            rel = [(n, w) for n, w, d in dets if _relevant(n, d, child, pregnant)]
            share = sum(w for _, w in rel) / total
            top_rel = bool(rel) and rel[0][0] == dets[0][0]
            if top_rel or share >= CONTEXT_SHARE:
                why = "1순위 후보가 %s 관련 질환" % ctx if top_rel else "%s 관련 후보 비중 %s" % (ctx, _pct(share))
                return "peds_obgyn", round(share, 3), [
                    f"환자 맥락 {ctx} + {why}: " + ", ".join(f"{n} {_pct(w / total)}" for n, w in rel[:3])]
            reasons.append(f"{ctx}이지만 선두 후보가 연령·임신 특이 질환이 아니라 장기별 분과로 배정")
        best = max(SPECIALTIES, key=lambda s: (mass[s], -SPECIALTIES.index(s)))
        if mass[best] <= 0:
            return None, 0.0, reasons + ["상위 감별 후보가 6개 분과 어디에도 속하지 않음: " + ", ".join(unmapped[:3])]
        share = mass[best] / total
        reasons.append(f"상위 감별 {len(cands)}개 중 {SPECIALTY_KO[best]}({best}) 비중 {_pct(share)}: "
                       + ", ".join(members[best][:3]))
        others = [f"{SPECIALTY_KO[s]} {_pct(mass[s] / total)}" for s in SPECIALTIES if s != best and mass[s] > 0]
        if others:
            reasons.append("다른 분과: " + ", ".join(others))
        if unmapped:
            reasons.append("6개 분과 밖 후보: " + ", ".join(unmapped[:3]))
        return best, round(share, 3), reasons
    except Exception as e:  # never break the agent loop
        return None, 0.0, [f"분과 배정 실패({type(e).__name__})"]


# --------------------------------------------------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------------------------------------------------

def _case_text(state) -> str:
    """The policy's criteria-check composition (initial info, responses, verified findings ledger), with only the last
    MAX_RESPONSES responses: the ledger already carries older findings, and evaluate() time grows with the text."""
    findings = getattr(state, "findings", None)
    ledger = findings.render(exclude_unverified=True) if findings is not None else ""
    turns = list(getattr(state, "turns", []))[-MAX_RESPONSES:]
    return "\n".join([getattr(state, "initial_info", "") or ""] + [getattr(t, "response", "") or "" for t in turns]
                     + [ledger])


def _one_line(s: str, n: int = 90) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;") + "…"


def _criteria(specialty: str, state, dx_names: list[str]) -> list[dict]:
    from doctor_agent.knowledge import diagnostic_criteria as dc
    direct: dict[str, list[str]] = {}
    for n in dx_names:
        for c in dc.criteria_for(n, related=False):
            direct.setdefault(c.id, []).append(n)
    out = []
    for c in dc.CRITERIA:
        if specialty not in CRITERIA_SPECIALTY.get(c.id, ()) and c.id not in direct:
            continue
        out.append({"id": c.id, "name": c.name_ko, "cite": c.cite, "summary": _one_line(c.subtype_rule or c.rule),
                    "for_dx": direct.get(c.id, [])})
    out.sort(key=lambda r: (not r["for_dx"], specialty not in CRITERIA_SPECIALTY.get(r["id"], ())))
    out = out[:MAX_ITEMS["criteria"]]
    # the first set that names a current candidate is also evaluated on the case text (one evaluation: time budget)
    row = next((r for r in out if r["for_dx"]), None)
    if row is not None:
        r = dc.evaluate(row["id"], _case_text(state))
        row.update(band=r.band, met=len(r.met), not_met=len(r.not_met), unknown=len(r.unknown))
    return out


def _rules(specialty: str, state) -> list[dict]:
    from doctor_agent.knowledge import clinical_rules as cr
    cc = getattr(state, "initial_info", "") or ""
    low = cc.lower()
    cats = set(cr.detect_categories(low))  # once; Rule.applies_to repeats it, so it only runs for rules that can match
    out = []
    for r in cr.RULES:
        if specialty not in RULE_SPECIALTY.get(r.id, ()):
            continue
        maybe = any(k in low for k in r.keywords) or bool(cats.intersection(r.categories))
        out.append({"id": r.id, "name": r.short, "cite": r.citation.short, "purpose": _one_line(r.purpose, 60),
                    "applies": bool(maybe and r.applies_to(cc))})
    out.sort(key=lambda x: not x["applies"])
    return [x for x in out if x["applies"]][:MAX_ITEMS["rules"]] or out[:2]


def _protocols(specialty: str, state) -> list[dict]:
    from doctor_agent.safety import protocols as pr
    cc = getattr(state, "initial_info", "") or ""
    learned = " ".join(getattr(t, "response", "") or "" for t in getattr(state, "turns", []))
    actions = [getattr(getattr(t, "action", None), "content", "") or "" for t in getattr(state, "turns", [])]
    pending = {c.id for c in pr.pending_checks(cc, actions, learned)}
    out = []
    for p in pr.protocols_for(cc, learned):
        out.append({"category": p.category, "name": p.name_ko, "cant_miss": list(p.cant_miss[:4]),
                    "pending": [c.name for c in p.checks if c.id in pending and c.kind != "treatment"][:3],
                    "in_specialty": specialty in CATEGORY_SPECIALTY.get(p.category, ())})
    out.sort(key=lambda x: not x["in_specialty"])
    return out[:MAX_ITEMS["protocols"]]


def _kb_candidates(specialty: str, state, dx_names: list[str]) -> list[dict]:
    from doctor_agent.knowledge import kb as kbm
    k = _kb()
    if k is None:
        return []
    items = list(getattr(getattr(state, "findings", None), "items", []) or [])
    # most recent findings only: kb.candidates parses every finding (~1 ms each), keep the call inside the budget
    pos = [f.item for f in items if f.status == "양성"][-MAX_FINDINGS:]
    neg = [f.item for f in items if f.status == "음성"][-MAX_FINDINGS // 2:]
    sex, age = kbm.patient_profile(getattr(state, "initial_info", "") or "")
    rows: list[tuple[int, str]] = []  # (disease index, why)
    seen: set[int] = set()
    for n in dx_names:  # the DDx ledger's own candidates in this specialty first
        if specialty_of(n) != specialty:
            continue
        i = k._resolve(n, fuzzy=_fuzzy_ok(n))
        if i is not None and i not in seen:
            seen.add(i)
            rows.append((i, "감별 목록"))
    if pos:
        for c in k.candidates(pos, k=10, negatives=neg, sex=sex or None, age=age):
            i = k.by_id[c["id"]]
            if i in seen or len(rows) >= MAX_ITEMS["kb_candidates"]:
                continue
            e = _code_entry(c["kcd"][0]) if c["kcd"] else None
            spec = e[0] if e else specialty_of(c["name_ko"] or c["name_en"])
            if spec == specialty:
                seen.add(i)
                rows.append((i, "소견 일치: " + ", ".join(dict.fromkeys(m["ko"] for m in c["matched"][:3]))))
    out = []
    for i, why in rows[:MAX_ITEMS["kb_candidates"]]:
        p = k.profile(i)
        tests = [x["ko"] for x in p["findings_from_tests"] if x["weight"] >= 2][:2]
        if not tests:
            tests = [x["ko"] for x in p["tests"] if x["en"].lower() not in ("medical history", "physical examination",
                                                                           "history", "physical exam")][:2]
        sx = [x["ko"] for x in p["symptoms"] if re.search(r"[가-힣]", x["ko"])][:3]
        out.append({"name": p["name_ko"] or p["name_en"], "code": (p["codes"].get("kcd") or [[""]])[0][0],
                    "why": why, "key_findings": sx, "key_tests": tests})
    return out


def resources(specialty: str, state) -> dict:
    """Evidence slice for one specialty (see module docstring). {} on error or an unknown specialty."""
    try:
        if specialty not in SPECIALTIES:
            return {}
        dx_names = [n for n, _ in _candidates(state)]
        return {"specialty": specialty,
                "criteria": _criteria(specialty, state, dx_names),
                "rules": _rules(specialty, state),
                "protocols": _protocols(specialty, state),
                "kb_candidates": _kb_candidates(specialty, state, dx_names)}
    except Exception:
        return {}


def render_resources(res: dict, max_chars: int = 700) -> str:
    """Short Korean text for a consult prompt (<= max_chars; "" when empty). Never raises."""
    try:
        if not res:
            return ""
        sp = res.get("specialty", "")
        lines = [f"[{SPECIALTY_KO.get(sp, sp)} 분과 참고 자료: 확진 근거가 아니라 감별·검사 계획용]"]
        for c in res.get("criteria", []):
            band = f" 현재 판정: {c['band']}" if c.get("band") else ""
            lines.append(f"- 진단 기준 {c['name']} ({c['cite']}): {c['summary']}{band}")
        for r in res.get("rules", []):
            lines.append(f"- 결정 규칙 {r['name']} ({r['cite']}): {r['purpose']}" + ("" if r["applies"] else " [주호소상 비해당]"))
        for p in res.get("protocols", []):
            s = f"- 안전 프로토콜 {p['name']}: 배제할 위험 질환 {', '.join(p['cant_miss'])}"
            if p.get("pending"):
                s += f"; 아직 안 한 확인 {', '.join(p['pending'])}"
            lines.append(s)
        for c in res.get("kb_candidates", []):
            s = f"- 후보 {c['name']}" + (f" ({c['code']})" if c.get("code") else "") + f" [{c['why']}]"
            if c.get("key_findings"):
                s += f" 전형: {', '.join(c['key_findings'])}"
            if c.get("key_tests"):
                s += f" / 결정적 검사: {', '.join(c['key_tests'])}"
            lines.append(s)
        if len(lines) == 1:
            return ""
        out = ""
        for line in lines:
            if len(line) > max_chars // 2:
                line = line[: max_chars // 2 - 1] + "…"
            if len(out) + len(line) + 1 > max_chars:
                break
            out += ("\n" if out else "") + line
        return out
    except Exception:
        return ""
