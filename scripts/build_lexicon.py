"""Build data/lexicon/concepts.json: one clinical-finding lexicon merged from every dictionary in the code base.

    python scripts/build_lexicon.py            # rebuild + print a summary
    python scripts/build_lexicon.py --check    # rebuild in memory and fail if concepts.json is stale

Sources, in priority order (a surface form already owned by an earlier concept is not re-assigned; the conflict is
counted in meta.conflicts). Every form/regex keeps its provenance tags.
    seed            data/lexicon/seed.tsv (hand-authored concepts, Korean lay/medical/English/abbreviations)
    casefreq        data/lexicon/casefreq.tsv (short generic expressions added after a frequency review of
                    data/cases_aug history answers; scripts/label_findings.py --freq lists the uncovered n-grams)
    curated         knowledge/kb_curated.py SYNONYMS + REGEX (finding -> KB term tables)
    grounding       _GROUPS below (the legacy synonym groups of agent/grounding.py, moved here 2026-09-30)
    kb_tests        knowledge/kb_tests.py FINDINGS (test/lab/imaging result concepts -> LAB:/IMG:/ECG: ids, KB TF: ids)
    clinical_rules  knowledge/clinical_rules.py CATEGORY_KEYWORDS and keyword tuples (protocol category links)
    protocols       safety/protocols.py _TRIG_* / _KW_* tuples (trigger links)
    danger_gate     safety/danger_gate.py keyword tuples
    preconditions   safety/preconditions.py keyword tuples
    kb              data/kb/kb.json.gz term ids whose English/Korean label equals a concept label or synonym
Keyword tuples are assigned to a concept by scanning each keyword with the lexicon built so far: a keyword whose
text is exactly one concept's mention becomes a form of that concept (and links the tuple/category name); a keyword
that is only a prefix of a known form links the tuple but adds no form; the rest are counted as unmapped.

No LLM, no network. Deterministic output (sorted keys, no timestamps except meta.built from --date).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.nlp.lexicon import Lexicon, compact_key  # noqa: E402

LEX_DIR = ROOT / "data" / "lexicon"
OUT = LEX_DIR / "concepts.json"
VERSION = "1.0"

# words that contain a shorter form but mean something else: they consume their span at scan time
BLOCK = [
    "발작성", "발작적", "paroxysmal", "불안정", "무통성", "무통", "방사선", "간질성", "부인과", "산부인과", "경기도", "경기 관람",
    "축구 경기", "야구 경기", "종양 표지", "종양표지", "양성 종양", "양성종양", "강직성 척추염", "과민성 대장", "과민성대장",
    "다음 날", "다음날", "다음 주", "다음번", "다음에", "열심히", "열흘", "열어", "열린", "가래떡", "자궁경부", "황달 구역", "종창성", "객담 도말", "객담 배양", "객담 검사", "가래 검사", "객담 그람", "말초 맥박", "원위부 맥박", "족배동맥 맥박", "후경골동맥 맥박", "대퇴 맥박", "방사형", "구역의", "구역에서", "구역 내", "경동맥 잡음", "경동맥잡음", "혈관 잡음", "복부 잡음",
    # "기억이 안 나요" answers a question (uncertainty), it is not memory loss
    "기억이 안", "기억이 잘 안", "기억 안", "기억은 안",
    # skin turgor ("피부 팽진도 감소"), not wheals
    "팽진도",
]

# ambiguous in free text; not taken from the merged tables ("병변 등 통증" = "lesions etc.", "방사통" also in the wrist)
DROP_FORMS = {"등 통증", "손상", "질환", "잠에서 깨", "방사통", "잡음", "흡입기", "산모", "부인", "피부 병변"}

# grounding.py group -> concept id (None: a site, a test name or an analyte name, not a finding concept)
GROUNDING_MAP = {
    "fever": "SYM:fever", "chills": "SYM:chills", "dyspnea": "SYM:dyspnea", "chest_pain": "SYM:chest_pain",
    "chest_tight": "SYM:chest_tightness", "headache": "SYM:headache", "abd_pain": "SYM:abdominal_pain",
    "pain": "SYM:pain", "tearing": "QUAL:tearing", "squeezing": "QUAL:squeezing", "stabbing": "QUAL:stabbing",
    "burning": "QUAL:burning", "sudden": "QUAL:sudden_onset", "radiation": "QUAL:radiation", "nausea": "SYM:nausea",
    "vomiting": "SYM:vomiting", "diarrhea": "SYM:diarrhea", "constipation": "SYM:constipation", "cough": "SYM:cough",
    "sputum": "SYM:sputum", "hemoptysis": "SYM:hemoptysis", "syncope": "SYM:syncope", "dizziness": "SYM:dizziness",
    "sweating": "SYM:sweating", "edema": "SIGN:edema", "weakness": "SYM:weakness", "fatigue": "SYM:fatigue",
    "anorexia": "SYM:anorexia", "weight_loss": "SYM:weight_loss", "palpitation": "SYM:palpitation",
    "dysuria": "SYM:dysuria", "hematuria": "SYM:hematuria", "melena": "SYM:melena", "hematochezia": "SYM:hematochezia",
    "jaundice": "SYM:jaundice", "rash": "SYM:rash", "confusion": "SYM:altered_mental_status",
    "neck_stiff": "SIGN:neck_stiffness", "tenderness": "SIGN:tenderness", "rebound": "SIGN:rebound_tenderness",
    "murphy": "SIGN:murphy_sign", "murmur": "SIGN:murmur", "crackles": "SIGN:crackles", "wheeze": "SIGN:wheeze",
    "tachycardia": "SIGN:tachycardia", "bradycardia": "SIGN:bradycardia", "tachypnea": "SIGN:tachypnea",
    "hypotension": "SIGN:hypotension", "hypertension": "HX:hypertension", "hypoxemia": "SIGN:hypoxemia",
    "leukocytosis": "LAB:wbc_high", "anemia": "LAB:hb_low", "thrombocytopenia": "LAB:plt_low",
    "st_elev": "ECG:ecg_stemi", "st_dep": "ECG:ecg_st_depression",
}
# kb_curated.SYNONYMS keys (English KB term labels) that do not equal a seed label/synonym
CURATED_MAP = {
    "high fever": "SYM:high_fever", "sweaty": "SYM:sweating", "decreased appetite": "SYM:anorexia",
    "dehydration": "SIGN:dehydration", "severe headache": "SYM:headache", "mental confusion":
    "SYM:altered_mental_status", "coma": "SYM:altered_mental_status", "hypoesthesia": "SYM:numbness",
    "loss of speech": "SYM:aphasia", "face drooping": "SYM:facial_droop", "trouble walking": "SYM:gait_disturbance",
    "cerebellar ataxia": "SIGN:ataxia", "sadness": "SYM:depressed_mood", "productive cough with abnormal sputum":
    "SYM:sputum", "chest tightness": "SYM:chest_tightness", "heart arrhythmia": "SIGN:irregular_pulse",
    "arterial hypertension": "HX:hypertension", "nail clubbing": "SIGN:clubbing", "abdominal mass": "SIGN:abdominal_mass",
    "indigestion": "SYM:indigestion", "frequent urination": "SYM:frequency", "urinary urgency": "SYM:urgency",
    "arthritis": "SIGN:joint_swelling", "exanthem": "SYM:rash", "itch": "SYM:pruritus", "petechia": "SIGN:petechiae",
    "blisters": "SYM:blister", "xerostomia": "SYM:dry_mouth", "dry eye": "SYM:dry_eye", "mouth ulcer": "SYM:mouth_ulcer",
    "runny nose": "SYM:rhinorrhea", "sneeze": "SYM:sneezing", "otalgia": "SYM:ear_pain", "ocular redness": "SYM:red_eye",
    "dysphonia": "SYM:hoarseness", "obesity": None, "tobacco smoking history": "HX:smoking",
    "alcohol abuse or addiction": "HX:alcohol_use_disorder", "sensitivity to light": "SYM:photophobia",
    "sensitivity to the sun": "SYM:photosensitivity", "cloudy urine": "SYM:cloudy_urine",
    "foul-smelling urine": "SYM:cloudy_urine", "painful swallowing": "SYM:odynophagia", "thirst": "SYM:polydipsia",
    "bruise": "SYM:easy_bruising", "intestinal bleeding": "SYM:hematochezia", "gastrointestinal bleeding":
    "SYM:hematochezia", "nosebleed": "SYM:epistaxis", "bleeding gums": "SYM:gum_bleeding", "eye pain": "SYM:eye_pain",
    "paroxysmal nocturnal dyspnea": "SYM:pnd", "intermittent claudication": "SYM:claudication", "jaw pain": "SYM:jaw_pain",
    "suicidal ideation": "SYM:suicidal_ideation", "urinary incontinence": "SYM:urinary_incontinence",
    "fecal incontinence": "SYM:fecal_incontinence", "urinary retention": "SYM:urinary_retention",
    "testicular pain": "SYM:scrotal_pain", "breast lump": "SYM:breast_lump", "nipple discharge": "SYM:nipple_discharge",
    "muscle rigidity": "SIGN:rigidity", "strawberry tongue": "SIGN:strawberry_tongue", "swollen joints":
    "SIGN:joint_swelling", "joint effusion": "SIGN:joint_swelling", "silvery scales": None, "red eye": "SYM:red_eye",
    "watery eyes": "SYM:lacrimation", "change in bowel habits": "GRP:bowel", "clay-colored stools": "SYM:pale_stool",
    "oliguria": "SYM:oliguria", "nocturia": "SYM:nocturia", "hyporeflexia": "SIGN:hyporeflexia",
    "hyperreflexia": "SIGN:hyperreflexia", "xeroderma": "SYM:dry_skin", "altered level of consciousness":
    "SYM:altered_mental_status", "ketonuria": "LAB:ketone_pos", "hypoxia": "SIGN:hypoxemia", "elevated transaminases":
    "LAB:ast_alt_high", "kidney failure": "LAB:cr_high", "leukopenia": "LAB:wbc_low", "hyponatremia": "LAB:na_low",
    "hypokalemia": "LAB:k_low", "hyperkalemia": "LAB:k_high", "hypoglycemia": "LAB:glucose_low",
    "hyperglycemia": "LAB:glucose_high", "eosinophilia": "LAB:eos_high", "hypoalbuminemia": "LAB:albumin_low",
    "tenderness": "SIGN:tenderness", "abdominal tenderness": "SIGN:abdominal_tenderness", "hepatosplenomegaly":
    "SIGN:hepatosplenomegaly", "hematemesis": "SYM:hematemesis", "tremor": "SYM:tremor", "somnolence": "SYM:somnolence",
    "memory loss": "SYM:memory_loss", "hemiparesis": "SYM:hemiparesis", "paralysis": "SYM:paralysis",
    "paresthesia": "SYM:paresthesia", "dysarthria": "SYM:dysarthria", "diplopia": "SYM:diplopia", "ptosis": "SIGN:ptosis",
    "blurred vision": "SYM:blurred_vision", "vision loss": "SYM:vision_loss", "photophobia": "SYM:photophobia",
    "neck stiffness": "SIGN:neck_stiffness", "deafness": "SYM:hearing_loss", "tinnitus": "SYM:tinnitus",
    "insomnia": "SYM:insomnia", "anxiety": "SYM:anxiety", "hallucination": "SYM:hallucination", "delusion":
    "SYM:delusion", "irritability": "SYM:irritability", "wheeze": "SIGN:wheeze", "stridor": "SIGN:stridor",
    "pleuritic chest pain": "SYM:pleuritic_pain", "palpitation": "SYM:palpitation", "tachycardia": "SIGN:tachycardia",
    "bradycardia": "SIGN:bradycardia", "hypotension": "SIGN:hypotension", "cyanosis": "SIGN:cyanosis",
    "edema": "SIGN:edema", "tachypnea": "SIGN:tachypnea", "melena": "SYM:melena", "bloating": "SYM:bloating",
    "heartburn": "SYM:heartburn", "dysphagia": "SYM:dysphagia", "jaundice": "SYM:jaundice", "ascites": "SIGN:ascites",
    "hepatomegaly": "SIGN:hepatomegaly", "splenomegaly": "SIGN:splenomegaly", "dark urine": "SYM:dark_urine",
    "dysuria": "SYM:dysuria", "polyuria": "SYM:polyuria", "polydipsia": "SYM:polydipsia", "hematuria": "SYM:hematuria",
    "proteinuria": "LAB:proteinuria", "vaginal bleeding": "SYM:vaginal_bleeding", "vaginal discharge":
    "SYM:vaginal_discharge", "pelvic pain": "SYM:pelvic_pain", "amenorrhea": "SYM:amenorrhea", "menorrhagia":
    "SYM:menorrhagia", "erectile dysfunction": "SYM:erectile_dysfunction", "arthralgia": "SYM:arthralgia",
    "joint stiffness": "SYM:joint_stiffness", "morning stiffness": "SYM:morning_stiffness", "myalgia": "SYM:myalgia",
    "back pain": "SYM:back_pain", "neck pain": "SYM:neck_pain", "bone pain": "SYM:bone_pain", "limb pain":
    "SYM:limb_pain", "muscle atrophy": "SIGN:muscle_atrophy", "maculopapular rash": "SYM:rash", "erythema":
    "SIGN:erythema", "urticaria": "SYM:urticaria", "easy bruising": "SYM:easy_bruising", "hair loss": "SYM:hair_loss",
    "pallor": "SIGN:pallor", "raynaud phenomenon": "SYM:raynaud", "sore throat": "SYM:sore_throat",
    "nasal congestion": "SYM:nasal_congestion", "lymphadenopathy": "SIGN:lymphadenopathy", "exophthalmos":
    "SIGN:exophthalmos", "malar rash": "SIGN:malar_rash", "flank pain": "SYM:flank_pain", "orthopnea": "SYM:orthopnea",
    "fever": "SYM:fever", "chills": "SYM:chills", "night sweats": "SYM:night_sweats", "fatigue": "SYM:fatigue",
    "weight loss": "SYM:weight_loss", "weight gain": "SYM:weight_gain", "malaise": "SYM:malaise",
    "headache": "SYM:headache", "dizziness": "SYM:dizziness", "vertigo": "SYM:vertigo", "syncope": "SYM:syncope",
    "seizure": "SYM:seizure", "weakness": "SYM:weakness", "cough": "SYM:cough", "hemoptysis": "SYM:hemoptysis",
    "dyspnea": "SYM:dyspnea", "chest pain": "SYM:chest_pain", "nausea": "SYM:nausea", "vomiting": "SYM:vomiting",
    "abdominal pain": "SYM:abdominal_pain", "diarrhea": "SYM:diarrhea", "constipation": "SYM:constipation",
    "hematochezia": "SYM:hematochezia", "leukocytosis": "LAB:wbc_high", "anemia": "LAB:hb_low",
    "thrombocytopenia": "LAB:plt_low", "high blood pressure": "HX:hypertension",
}
# kb_curated.REGEX patterns not merged: their free gaps swallow unrelated words or negations ("변화 없이 검사" is not
# melena, "계속 쓰려" is not heartburn, "아토피 피부염" is not hematemesis); the seed has tighter patterns
CURATED_RE_SKIP = {"decreased appetite", "insomnia", "memory loss", "deafness", "amenorrhea", "hematemesis", "melena",
                   "hematochezia", "abdominal pain", "heartburn", "proteinuria",
                   "altered level of consciousness"}
# clinical_rules.CATEGORY_KEYWORDS category -> the concept its keywords name (None: mixed set, scan each keyword)
CATEGORY_MAP = {
    "chest_pain": "SYM:chest_pain", "dyspnea": "SYM:dyspnea", "headache": "SYM:headache", "neuro": None,
    "fever": "SYM:fever", "abdominal_pain": "SYM:abdominal_pain", "allergy": "SYM:allergic_reaction",
    "syncope": "SYM:syncope", "palpitations": "SYM:palpitation", "hemoptysis_cough": "SYM:hemoptysis",
    "jaundice": "SYM:jaundice", "joint": "SYM:arthralgia", "back_pain": "SYM:back_pain", "rash": "SYM:rash",
    "pruritus": "SYM:pruritus", "edema": "SIGN:edema", "menstrual": None, "fatigue": "SYM:fatigue",
    "cognitive": "SYM:memory_loss", "psychiatric": None, "hearing_loss": "SYM:hearing_loss", "bleeding": "GRP:bleeding",
}


# ------------------------------------------------------------------ seed
def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split("|") if x.strip()]


def read_seed(path: Path) -> "OrderedDict[str, dict]":
    concepts: OrderedDict[str, dict] = OrderedDict()
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"{path.name}:{ln}: expected 3 tab-separated fields, got {len(parts)}")
        cid, fld, val = parts
        c = concepts.setdefault(cid, _new(cid))
        if fld in ("ko", "en"):
            c[fld] = val.strip()
        elif fld in ("lay", "med", "syn", "abbr"):
            kind = {"syn": "en"}.get(fld, fld)
            for form in _split(val):
                fig = form.endswith("~")
                _add_form(c, form.rstrip("~").strip(), kind, "seed", fig)
        elif fld == "neg":
            for form in _split(val):
                _add_form(c, form, "neg", "seed", False)
        elif fld in ("re", "negre"):
            c[fld].append([val.strip(), "seed"])
        elif fld == "parents":
            c["parents"] += [p for p in _split(val) if p not in c["parents"]]
        elif fld == "flags":
            c["flags"] += [p for p in re.split(r"[,|]", val) if p.strip()]
        else:
            raise SystemExit(f"{path.name}:{ln}: unknown field {fld!r}")
    return concepts


def _new(cid: str) -> dict:
    return {"id": cid, "cat": cid.split(":")[0], "ko": "", "en": "", "forms": [], "re": [], "negre": [], "parents": [],
            "flags": [], "kb": [], "protocol": []}


def _add_form(c: dict, form: str, kind: str, prov: str, fig: bool = False) -> bool:
    form = form.strip()
    if not form:
        return False
    key = compact_key(form)
    for f in c["forms"]:
        if compact_key(f[0]) == key and (f[1] == "neg") == (kind == "neg"):
            if prov not in f[2].split("+"):
                f[2] += "+" + prov
            return False
    c["forms"].append([form, kind, prov, 1 if fig else 0])
    return True


# ------------------------------------------------------------------ merge helpers
class Builder:
    def __init__(self, concepts: "OrderedDict[str, dict]"):
        self.c = concepts
        self.owner: dict[str, str] = {}  # compact form key -> concept id (first owner wins)
        self.conflicts: list[tuple[str, str, str, str]] = []
        self.unmapped: dict[str, list[str]] = {}
        self.added: dict[str, int] = {}
        for cid, c in concepts.items():
            for f in c["forms"]:
                if f[1] != "neg":
                    self.owner.setdefault(compact_key(f[0]), cid)

    def form(self, cid: str, form: str, kind: str, prov: str) -> None:
        if cid not in self.c or form.strip() in DROP_FORMS:
            return
        key = compact_key(form)
        if len(key) < 2:
            return
        own = self.owner.get(key)
        if own and own != cid:
            self.conflicts.append((form, prov, cid, own))
            return
        if _add_form(self.c[cid], form, kind, prov):
            self.added[prov] = self.added.get(prov, 0) + 1
        self.owner.setdefault(key, cid)

    def regex(self, cid: str, pattern: str, prov: str) -> None:
        if cid not in self.c:
            return
        for r in self.c[cid]["re"]:
            if r[0] == pattern:
                if prov not in r[1].split("+"):
                    r[1] += "+" + prov
                return
        self.c[cid]["re"].append([pattern, prov])
        self.added[prov] = self.added.get(prov, 0) + 1

    def link(self, cid: str, name: str) -> None:
        if cid in self.c and name not in self.c[cid]["protocol"]:
            self.c[cid]["protocol"].append(name)

    def lexicon(self) -> Lexicon:
        return Lexicon.from_data({"concepts": list(self.c.values()), "block": BLOCK})


def _en_index(concepts) -> dict[str, str]:
    ix: dict[str, str] = {}
    for cid, c in concepts.items():
        for s in [c["en"]] + [f[0] for f in c["forms"] if f[1] == "en"]:
            ix.setdefault(s.strip().lower(), cid)
    return ix


def merge_curated(b: Builder) -> None:
    from doctor_agent.knowledge import kb_curated
    ix = _en_index(b.c)
    for en, forms in kb_curated.SYNONYMS.items():
        cid = CURATED_MAP.get(en, ix.get(en.lower()))
        if cid is None:
            b.unmapped.setdefault("curated", []).append(en)
            continue
        for f in forms:
            b.form(cid, f, "lay", "curated")
    for pat, en in kb_curated.REGEX:
        if en in CURATED_RE_SKIP:
            continue
        cid = CURATED_MAP.get(en, ix.get(en.lower()))
        if cid:
            b.regex(cid, pat, "curated")
        else:
            b.unmapped.setdefault("curated", []).append("re:" + en)


# --- legacy synonym groups of the grounding checker (agent/grounding.py until 2026-09-30) ---------------------------
# The checker reads the lexicon now (these forms carry provenance "grounding"); the table is kept here only as a
# build input of merge_grounding.
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



def merge_grounding(b: Builder) -> None:
    for grp, variants, _flags in _GROUPS:
        cid = GROUNDING_MAP.get(grp)
        if cid is None:
            continue
        b.link(cid, "grounding:" + grp)
        for v in variants:
            if v.startswith("claim:"):
                continue
            if v.startswith("re:"):
                b.regex(cid, v[3:], "grounding")
            else:
                b.form(cid, v, "en" if re.fullmatch(r"[a-z0-9 \-]+", v) else "lay", "grounding")


# kb_tests finding ids that are imaging / endoscopy / physiology results (the rest are laboratory, microbiology or
# pathology results -> LAB:)
_IMAGING = re.compile(
    r"^(ct_|cxr|echo_|us_|mri|doppler|ctpa|egd_|colon_|eeg|pft|ph_|thyroid_uptake|sbo_|free_air|volvulus|intussusception"
    r"|gallstone|cbd_|stone_imaging|hydronephrosis|small_kidneys|pancreas_|aaa_|coronary_aneurysm|ggo|honeycombing"
    r"|pleural_effusion|cavity|upper_lobe|bhl|psc_imaging|hcc_imaging|takayasu_imaging|sacroiliitis|chondrocalcinosis"
    r"|pituitary_mass|pcos_us|lung_mass|gastric_mass|biliary_mass|stroke_imaging|cvst_imaging|temporal_lobe_mri"
    r"|wernicke_mri|nph_imaging|osteomyelitis_mri|varices|obstructive_pft|bd_reversible|lvef|kf_ring|testis_no_flow"
    r"|colon_mass|myopathic_emg|als_emg|rns_|ncs_|fev1fvc)")


def merge_kb_tests(b: Builder) -> None:
    from doctor_agent.knowledge import kb_tests
    for f in kb_tests.FINDINGS:
        prefix = "ECG" if f.id.startswith("ecg_") else "IMG" if _IMAGING.match(f.id) else "LAB"
        cid = f"{prefix}:{f.id}"
        c = b.c.setdefault(cid, _new(cid))
        c["ko"] = c["ko"] or f.ko
        c["en"] = c["en"] or f.en
        if "TF:" + f.id not in c["kb"]:
            c["kb"].append("TF:" + f.id)
        if "kb_tests" not in c["flags"]:
            c["flags"].append("kb_tests")
        label = f.ko.split("·")[0].strip()
        if ":" not in f.ko and len(compact_key(label)) >= 3:
            b.form(cid, label, "med", "kb_tests")
        en = f.en.split(":", 1)[-1].split("/")[0].strip()
        if len(en) >= 6:
            b.form(cid, en, "en", "kb_tests")


def _string_tuples(module) -> dict[str, tuple[str, ...]]:
    out = {}
    for name, val in vars(module).items():
        if name.isupper() or (name.startswith("_") and name[1:].replace("_", "").isupper()):
            if isinstance(val, tuple) and val and all(isinstance(x, str) for x in val):
                out[name] = val
    return out


def _assign(lex: Lexicon, kw: str) -> tuple[str | None, bool]:
    """(concept id, the keyword is itself a full mention of it). Prefix keywords ("가슴이 아") resolve through the
    forms that start with them when all such forms belong to one concept."""
    from doctor_agent.nlp.lexicon import normalize
    t = normalize(kw)
    ms = [m for m in lex.scan(t) if not m.is_absence]
    specific = [m for m in ms if not m.cid.startswith(("SYM:pain", "GRP:", "QUAL:"))] or ms
    if len(specific) == 1 and len({m.cid for m in specific}) == 1:
        m = specific[0]
        full = m.start <= 1 and m.end >= len(t) - 1
        return m.cid, full
    if ms:
        return None, False
    key = compact_key(kw)
    if len(key) >= 2:
        owners = {cid for k, cid in ((k, v[0]) for k, v in lex._forms.items()) if k.startswith(key) and len(k) > len(key)}
        if len(owners) == 1:
            return owners.pop(), False
    return None, False


def merge_keywords(b: Builder) -> None:
    from doctor_agent.knowledge import clinical_rules
    from doctor_agent.safety import danger_gate, preconditions, protocols
    lex = b.lexicon()
    # category keyword sets: forms of the mapped concept, plus a protocol link on every concept they name
    for cat, kws in clinical_rules.CATEGORY_KEYWORDS.items():
        target = CATEGORY_MAP.get(cat)
        for kw in kws:
            cid, full = _assign(lex, kw)
            if target and cid in (None, target) or (target and cid and target in lex.ancestors(cid)):
                cid = cid or target
            if cid is None:
                b.unmapped.setdefault("clinical_rules", []).append(f"{cat}:{kw}")
                continue
            b.link(cid, "category:" + cat)
            if full:
                b.form(cid, kw, "en" if re.fullmatch(r"[a-z0-9 \-']+", kw) else "lay", "clinical_rules")
    for prov, module in (("clinical_rules", clinical_rules), ("protocols", protocols), ("danger_gate", danger_gate),
                         ("preconditions", preconditions)):
        for name, kws in _string_tuples(module).items():
            if name == "CATEGORY_NAMES" or name.startswith(("_KW_", "_MASKS", "_UNAVAILABLE", "_NEG", "_AFFIRM",
                                                            "_CLAUSE", "_ONSET_Q", "_PARTICLES", "_STOP", "_WORD_")):
                continue
            for kw in kws:
                cid, full = _assign(lex, kw)
                if cid is None:
                    b.unmapped.setdefault(prov, []).append(f"{name}:{kw}")
                    continue
                b.link(cid, f"{prov}:{name}")
                if full:
                    b.form(cid, kw, "en" if re.fullmatch(r"[a-z0-9 \-']+", kw) else "lay", prov)


def merge_casefreq(b: Builder, path: Path) -> None:
    if not path.exists():
        return
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"{path.name}:{ln}: expected id<TAB>kind<TAB>forms")
        cid, kind, forms = parts
        if cid not in b.c:
            raise SystemExit(f"{path.name}:{ln}: unknown concept {cid}")
        for form in _split(forms):
            if kind == "neg":
                _add_form(b.c[cid], form, "neg", "casefreq")
            else:
                b.form(cid, form, kind, "casefreq")


def link_kb(b: Builder, kb_path: Path) -> int:
    if not kb_path.exists():
        return 0
    with gzip.open(kb_path, "rt", encoding="utf-8") as f:
        terms = json.load(f)["terms"]
    ix: dict[str, list[str]] = {}
    for tid, t in terms.items():
        for s in [t.get("en", ""), t.get("ko", "")] + list(t.get("syn", [])):
            k = compact_key(s)
            if len(k) >= 3:
                ix.setdefault(k, []).append(tid)
    n = 0
    for c in b.c.values():
        keys = [compact_key(c["en"]), compact_key(c["ko"])]
        keys += [compact_key(f[0]) for f in c["forms"] if f[1] in ("med", "en")]
        found: list[str] = []
        for k in keys:
            for tid in ix.get(k, ()):
                if tid not in found and tid not in c["kb"]:
                    found.append(tid)
        found.sort(key=lambda t: (not t.startswith(("TF:", "DDX:")), t))
        c["kb"] += found[:8]
        n += bool(found)
    return n


def build(date: str = "") -> dict:
    concepts = read_seed(LEX_DIR / "seed.tsv")
    b = Builder(concepts)
    merge_casefreq(b, LEX_DIR / "casefreq.tsv")
    merge_curated(b)
    merge_grounding(b)
    merge_kb_tests(b)
    merge_keywords(b)
    n_kb = link_kb(b, ROOT / "data" / "kb" / "kb.json.gz")
    # drop parents that do not exist; sort nothing else (seed order is kept for readability)
    for c in b.c.values():
        c["parents"] = [p for p in c["parents"] if p in b.c]
    out_concepts = []
    for c in b.c.values():
        d = {k: c[k] for k in ("id", "cat", "ko", "en") if c[k]}
        for k in ("parents", "flags", "kb", "protocol", "forms", "re", "negre"):
            if c[k]:
                d[k] = c[k]
        out_concepts.append(d)
    by_cat: dict[str, int] = {}
    for c in out_concepts:
        by_cat[c["cat"]] = by_cat.get(c["cat"], 0) + 1
    prov: dict[str, int] = {}
    for c in out_concepts:
        for f in c.get("forms", []) + [[r[0], "re", r[1]] for r in c.get("re", []) + c.get("negre", [])]:
            for p in f[2].split("+"):
                prov[p] = prov.get(p, 0) + 1
    meta = {
        "version": VERSION, "built": date, "builder": "scripts/build_lexicon.py",
        "counts": {"concepts": len(out_concepts), "by_category": dict(sorted(by_cat.items())),
                   "forms": sum(len(c.get("forms", [])) for c in out_concepts),
                   "regexes": sum(len(c.get("re", [])) + len(c.get("negre", [])) for c in out_concepts),
                   "kb_linked_concepts": n_kb, "entries_by_provenance": dict(sorted(prov.items()))},
        "conflicts": len(b.conflicts),
        "unmapped_keywords": {k: len(v) for k, v in sorted(b.unmapped.items())},
        "note": "Self-authored lexicon (no text copied from any source). Surface forms are short generic Korean/English"
                " phrases; provenance tags name the in-repo table each entry came from.",
    }
    return {"meta": meta, "block": BLOCK, "concepts": out_concepts, "_debug": {"conflicts": b.conflicts,
                                                                             "unmapped": b.unmapped}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if concepts.json differs from a fresh build")
    ap.add_argument("--date", default="2026-09-27")
    ap.add_argument("--show-unmapped", action="store_true")
    ap.add_argument("--show-conflicts", action="store_true")
    args = ap.parse_args()
    data = build(args.date)
    debug = data.pop("_debug")
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old.strip() != text.strip():
            print("concepts.json is stale: run python scripts/build_lexicon.py")
            return 1
        print("concepts.json is up to date")
        return 0
    OUT.write_text(text + "\n", encoding="utf-8")
    m = data["meta"]
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(json.dumps(m["counts"], ensure_ascii=False))
    print("conflicts:", m["conflicts"], "unmapped:", m["unmapped_keywords"])
    if args.show_unmapped:
        for k, v in debug["unmapped"].items():
            print(f"--- {k} ({len(v)})")
            print("  " + " | ".join(v))
    if args.show_conflicts:
        for form, prov, cid, own in debug["conflicts"]:
            print(f"  {form!r} ({prov}) wanted {cid}, owned by {own}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
