"""Next-question planner: ranks candidate ASK / EXAM / TEST actions by how well they separate the live DDx.

    suggest(state, k=3, include_safety=True) -> list[Suggestion]
    render_for_prompt(suggestions, max_chars=300) -> str      "추천 다음 행동 (참고): ..." (Korean, <= 300 chars)

How (stdlib only, CPU only, deterministic, no network, no state between calls):
1. Hypotheses: the top live diagnoses of the DDx ledger (<= MAX_DX, "배제" dropped; state.ddx or KB candidates when the
   ledger is empty), resolved to KB profiles, plus one "other" hypothesis (P_OTHER) so a single leading diagnosis still
   values confirmatory tests. Can't-miss ("위험") entries get a prior boost (DANGER_BOOST).
2. Features and P(feature | dx) from the KB profile: symptoms / risk factors (Orphanet frequency class when known,
   else by source consensus), and curated test-result links (findings_from_tests; link weight 3/2/1 -> P_TEST).
   Features the profile does not list get a small leak probability (larger when the profile has almost no features,
   so an empty profile is not "evidence of absence").
3. Candidate actions: one ASK per symptom / history feature (DDXPlus Korean question, else a lexicon-concept template),
   one EXAM per examination sign (body-area template), one TEST per test *request* (curated results mapped to the
   test that reveals them by TEST_RULES; several results of one test, e.g. the ECG, form one multi-outcome action).
4. Value = expected information gain (bits) about the hypotheses over the action's joint outcomes x cost-tier weight
   (ask > exam > lab > imaging > invasive).
5. Exclusions: actions already done (state.asked, earlier TEST/EXAM text naming the test), features already known from
   the case text or the findings ledger (normalisation layer concepts -> KB terms, kb_tests.detect results), and
   TEST/EXAM requests that safety/preconditions.check() would block.
6. Pending minimum safety checks (safety/protocols.py) are appended with safety=True (they are shown elsewhere in the
   prompt; render_for_prompt leaves them out). A ranked action that also completes a pending check is marked too.

Suggestions are hints for the LLM, never evidence; nothing here calls an LLM.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import product

from doctor_agent.agent.text import SIMILAR, similarity
from doctor_agent.env.interface import Action, ActionType

MAX_DX = 4
P_OTHER = 0.2        # prior mass of the "some other disease" hypothesis
DANGER_BOOST = 1.5   # prior multiplier of can't-miss ("위험") ledger entries
MAX_FEATS_PER_TEST = 4
# P(feature present | disease)
P_ORPHA = {"O": 0.95, "VF": 0.85, "F": 0.5, "OC": 0.15, "VR": 0.03, "EX": 0.01}
P_SYM_MULTI = 0.6    # listed by DDXPlus or by >= 2 independent sources
P_SYM_SINGLE = 0.45
P_IMPLIED = 0.3      # general term implied by a listed specific one (backoff)
P_TEST = {3: 0.9, 2: 0.65, 1: 0.35}
LEAK_SYM, LEAK_SYM_SPARSE, LEAK_SYM_OTHER = 0.06, 0.2, 0.1
LEAK_TEST, LEAK_TEST_SPARSE, LEAK_TEST_OTHER = 0.03, 0.08, 0.05
# value multiplier per cost tier. Offline sweep (cases_aug, N=2, gold in DDx): flat 1.0 -> decisive-test@3 0.84,
# these values 0.74, (1/.9/.8/.6/.4) 0.62, (1/.8/.6/.4/.2) 0.48; mild values keep the cost order without hiding tests
TIER_W = {"ask": 1.0, "exam": 0.95, "lab": 0.9, "imaging": 0.75, "invasive": 0.5}
TYPE_KO = {"ASK": "문진", "EXAM": "진찰", "TEST": "검사"}
MAX_CHARS = 300

_HANGUL = re.compile(r"[가-힣]")
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^0-9a-z가-힣]")
_SIGN_LABEL = re.compile(r"압통|비대|청진|잡음|반사|징후|종대|촉지|수포음|천명음|호흡음|반발통|근성 방어|안검하수|경부 강직")
_GENERIC_TERMS = {"통증", "고통", "증상", "징후", "불편감", "이상", "pain", "symptom", "symptoms", "sign", "aches", "ache"}


@dataclass
class Suggestion:
    type: str                      # "ASK" | "EXAM" | "TEST"
    content_ko: str
    targets: list[str] = field(default_factory=list)  # DDx names the action mainly argues for
    expected_value: float = 0.0    # information gain (bits) x cost-tier weight
    cost_tier: str = "ask"         # ask | exam | lab | imaging | invasive
    source: str = ""               # "DDXPlus" | "lexicon" | "KB" | "curated" | "protocol:<id>"
    citation: str = ""
    safety: bool = False           # a pending minimum safety check (source "protocol:<id>", shown by the protocols
    #                                hint) or a ranked action that also completes one (note says which)
    features: list[str] = field(default_factory=list)  # KB term ids the action observes
    note: str = ""                 # e.g. a precondition warning
    done_keywords: tuple[str, ...] = ()  # safety checks: action text that completes the check

    def action(self) -> Action:
        return Action(ActionType(self.type), self.content_ko, reason="question_planner")


# ------------------------------------------------------------------ Korean wording tables
def _josa(word: str, pair: str = "이가") -> str:
    """Subject particle after `word`: 이 after a final consonant, 가 otherwise (이/가 for non-Hangul endings)."""
    w = word.rstrip(" )")
    ch = w[-1] if w else ""
    if not ("가" <= ch <= "힣"):
        return f"{pair[0]}({pair[1]})" if ch.isalpha() else pair[0]
    return pair[0] if (ord(ch) - 0xAC00) % 28 else pair[1]


# lexicon HX concept -> natural question (other HX concepts: "{ko} 병력이 있으신가요?")
HX_QUESTIONS = {
    "HX:smoking": "담배를 피우시나요? 피우신다면 하루 몇 개비, 몇 년 정도인가요?",
    "HX:former_smoking": "예전에 담배를 피우신 적이 있나요?",
    "HX:alcohol": "술은 얼마나 자주, 얼마나 드시나요?",
    "HX:alcohol_use_disorder": "술을 매일 많이 드시거나 끊기 어려운 편인가요?",
    "HX:drug_use": "마약이나 주사 약물을 사용하신 적이 있나요?",
    "HX:medication": "현재 복용 중인 약이 있나요?",
    "HX:hormone_use": "피임약이나 호르몬제를 복용하시나요?",
    "HX:anticoagulant": "피를 묽게 하는 약(항응고제)을 드시나요?",
    "HX:antiplatelet": "아스피린 같은 항혈소판제를 드시나요?",
    "HX:recent_surgery": "최근에 수술을 받으신 적이 있나요?",
    "HX:immobilization": "최근 오래 누워 지내거나 거동이 불편하셨나요?",
    "HX:long_travel": "최근 장거리 비행이나 오래 앉아 있던 여행이 있었나요?",
    "HX:travel": "최근 해외나 다른 지역에 여행을 다녀오셨나요?",
    "HX:sick_contact": "주변에 비슷한 증상이 있는 사람이 있나요?",
    "HX:tick_bite": "최근 야외 활동을 하셨거나 진드기에 물린 적이 있나요?",
    "HX:raw_food": "최근 날음식이나 덜 익힌 음식을 드신 적이 있나요?",
    "HX:pregnancy": "임신 가능성이 있나요? 마지막 월경은 언제였나요?",
    "HX:menopause": "폐경이 되셨나요?",
    "HX:sexual_activity": "최근 성관계가 있었나요?",
    "HX:unprotected_sex": "최근 콘돔 없이 성관계를 하신 적이 있나요?",
    "HX:family_history_generic": "가족 중에 비슷한 병을 앓은 분이 있나요?",
    "HX:vaccination": "예방접종은 제때 맞으셨나요?",
    "HX:occupational_exposure": "일하시면서 먼지·화학물질 등에 노출되시나요?",
    "HX:trauma": "최근에 다치신 적이 있나요?",
    "HX:head_trauma": "최근에 머리를 부딪히신 적이 있나요?",
    "HX:dental_procedure": "최근 치과 치료를 받으신 적이 있나요?",
    "HX:allergy": "알레르기가 있으신가요?",
    "HX:immunosuppression": "면역력을 떨어뜨리는 약을 드시거나 면역 질환이 있나요?",
    "HX:chemotherapy": "최근 항암치료를 받으셨나요?",
}

# body area of an examination sign (first match of concept id / Korean label)
EXAM_AREAS = [
    (re.compile(r"cva|늑골척추각"), "복부·옆구리 진찰"),
    (re.compile(r"rectal|직장"), "직장수지검사"),
    (re.compile(r"neck_stiff|kernig|brudzinski|경부 강직|뇌막"), "신경학적 진찰(뇌막자극징후)"),
    (re.compile(r"tachy|brady|hypotens|elevated_bp|hypoxem|hypotherm|irregular_pulse|shock|빈맥|서맥|저혈압|혈압|빈호흡"
                r"|저산소|저체온|쇼크|맥박|산소포화"), "활력징후 측정"),
    (re.compile(r"murmur|s3|friction|심잡음|심음|마찰음"), "심장 청진"),
    (re.compile(r"wheeze|stridor|crackles|breath|dullness|retraction|천명|협착음|수포음|호흡음|탁음|흉벽"), "폐 청진"),
    (re.compile(r"jvd|lymph|thyroid|goiter|경정맥|림프절|갑상선"), "경부 진찰"),
    (re.compile(r"joint|관절"), "관절 진찰"),
    (re.compile(r"abdom|tender|rebound|guard|murphy|psoas|bowel|hepato|spleno|ascites|복부|압통|반발통|방어|머피|요근|장음"
                r"|간비대|비장비대|비대|복수"), "복부 진찰"),
    (re.compile(r"edema|clubbing|pulse_deficit|cyanosis|부종|곤봉지|청색증"), "사지 진찰"),
    (re.compile(r"rash|purpura|petech|jaundice|icter|skin|발진|자반|점상|황달|홍반|피부"), "피부 진찰"),
    (re.compile(r"eye|conjunc|ptosis|결막|공막|안검|동공|시야|안저"), "눈 진찰"),
    (re.compile(r"reflex|atrophy|fascic|rigid|ataxia|tremor|weakness|gait|babinski|반사|근위축|근섬유|강직|실조|떨림|근력"
                r"|보행|감각|뇌신경|바빈스키"), "신경학적 진찰"),
]

# curated test result (kb_tests id regex) -> (Korean test request, cost tier, done-keywords (lower case, no spaces)).
# First matching rule wins; results without a rule fall back to _generic_test().
TEST_RULES: list[tuple[str, str, str, tuple[str, ...], str]] = [
    # (id regex, request, tier, done keywords, action type)
    (r"^ecg_", "심전도", "lab", ("심전도", "ecg", "ekg"), "TEST"),
    (r"^(echo_|lvef_low|coronary_aneurysm)", "심초음파", "imaging", ("심초음파", "echo"), "TEST"),
    (r"^(troponin_high|ckmb_high)", "심근효소(트로포닌, CK-MB)", "lab", ("트로포닌", "troponin", "심근효소", "ck-mb"), "TEST"),
    (r"^(bnp_high|ntprobnp_high)", "BNP(NT-proBNP)", "lab", ("bnp",), "TEST"),
    (r"^ddimer_high", "D-dimer", "lab", ("d-dimer", "ddimer", "디다이머"), "TEST"),
    (r"^ctpa_pe", "CT 폐동맥 조영술", "imaging", ("폐동맥", "ctpa"), "TEST"),
    (r"^doppler_dvt", "하지 정맥 도플러 초음파", "imaging", ("도플러", "하지정맥", "하지초음파"), "TEST"),
    (r"^(ct_dissection|takayasu_imaging)", "흉부 CT 혈관조영(대동맥)", "imaging", ("대동맥ct", "ct혈관", "cta", "혈관조영"),
     "TEST"),
    (r"^(ggo|honeycombing)", "흉부 CT", "imaging", ("흉부ct", "chestct", "폐ct", "고해상도ct", "hrct"), "TEST"),
    (r"^(cxr_|cavity|upper_lobe_tb|bhl|pleural_effusion|lung_mass)", "흉부 X선", "imaging",
     ("흉부x", "흉부엑스", "흉부사진", "흉부방사선", "가슴x", "chestx", "cxr", "x-ray"), "TEST"),
    (r"^(fev1fvc_low|obstructive_pft|bd_reversible)", "폐기능 검사(기관지확장제 반응 포함)", "lab",
     ("폐기능", "pft", "spirometry"), "TEST"),
    (r"^(aaa_imaging)", "복부 초음파(복부대동맥)", "imaging", ("복부초음파", "복부대동맥", "복부ct"), "TEST"),
    (r"^(us_cholecystitis|gallstone|cbd_stone|cbd_dilation)", "복부 초음파(간담도)", "imaging",
     ("복부초음파", "담낭", "간담도", "복부ct"), "TEST"),
    (r"^intussusception_sign", "복부 초음파", "imaging", ("복부초음파",), "TEST"),
    (r"^(ct_pancreatitis|pancreas_|hcc_imaging|ct_appendicitis|ct_diverticulitis|sbo_imaging|free_air|volvulus_sign"
     r"|biliary_mass)", "복부 CT", "imaging", ("복부ct", "abdominalct", "복부골반ct", "복부단층"), "TEST"),
    (r"^(ct_pyelonephritis|stone_imaging|hydronephrosis)", "복부 CT(신장·요로)", "imaging",
     ("복부ct", "신장ct", "요로ct", "신장초음파"), "TEST"),
    (r"^small_kidneys", "신장 초음파", "imaging", ("신장초음파", "콩팥초음파"), "TEST"),
    (r"^psc_imaging", "MRCP", "imaging", ("mrcp", "자기공명담췌관"), "TEST"),
    (r"^(egd_|varices|gastric_mass|gastric_ca_bx|barrett_bx)", "위내시경(필요 시 생검)", "invasive",
     ("위내시경", "상부내시경", "egd", "내시경"), "TEST"),
    (r"^villous_atrophy", "위내시경 십이지장 생검", "invasive", ("위내시경", "십이지장생검", "내시경"), "TEST"),
    (r"^ph_monitoring", "24시간 식도 pH 검사", "invasive", ("ph검사", "ph모니터"), "TEST"),
    (r"^(colon_)", "대장내시경", "invasive", ("대장내시경", "colonoscopy"), "TEST"),
    (r"^(ua_nitrite_pos|ua_le_pos|pyuria|wbc_cast|rbc_cast|atn_casts)", "소변검사(현미경 포함)", "lab",
     ("소변검사", "요검사", "소변분석", "urinalysis", "소변현미경"), "TEST"),
    (r"^urine_culture_pos", "소변배양", "lab", ("소변배양", "urineculture"), "TEST"),
    (r"^proteinuria_nephrotic", "24시간 소변 단백", "lab", ("소변단백", "단백뇨", "24시간소변"), "TEST"),
    (r"^(kidney_bx_|crescents)", "신장 생검", "invasive", ("신장생검", "신생검"), "TEST"),
    (r"^liver_bx_", "간 생검", "invasive", ("간생검",), "TEST"),
    (r"^(caseating_granuloma|noncaseating_granuloma)", "조직 생검", "invasive", ("생검", "조직검사"), "TEST"),
    (r"^temporal_bx", "측두동맥 생검", "invasive", ("측두동맥",), "TEST"),
    (r"^perifascicular", "근육 생검", "invasive", ("근육생검", "근생검"), "TEST"),
    (r"^reed_sternberg", "림프절 생검", "invasive", ("림프절생검", "림프절조직"), "TEST"),
    (r"^lung_ca_bx", "기관지내시경 조직 생검", "invasive", ("기관지내시경", "폐생검", "폐조직"), "TEST"),
    (r"^(csf_|xanthochromia|hsv_pcr_pos|oligoclonal|crypto_pos)", "요추천자(뇌척수액 검사)", "invasive",
     ("요추천자", "뇌척수액", "csf", "lumbar"), "TEST"),
    (r"^(ct_sah|ct_ich)", "뇌 CT", "imaging", ("뇌ct", "두부ct", "머리ct", "brainct", "headct"), "TEST"),
    (r"^(stroke_imaging|cvst_imaging|mri_ms|temporal_lobe_mri|wernicke_mri|nph_imaging)", "뇌 MRI", "imaging",
     ("뇌mri", "머리mri", "brainmri", "두부mri"), "TEST"),
    (r"^pituitary_mass", "뇌하수체 MRI", "imaging", ("뇌하수체", "sellamri"), "TEST"),
    (r"^osteomyelitis_mri", "병변 부위 MRI", "imaging", ("mri",), "TEST"),
    (r"^sacroiliitis", "천장관절 X선·MRI", "imaging", ("천장관절", "sacroil"), "TEST"),
    (r"^chondrocalcinosis", "관절 X선", "imaging", ("관절x", "관절엑스", "관절방사선"), "TEST"),
    (r"^(myopathic_emg|als_emg|ncs_demyelination|rns_decrement)", "근전도·신경전도 검사", "imaging",
     ("근전도", "emg", "신경전도", "반복신경자극", "반복자극"), "TEST"),
    (r"^eeg_", "뇌파", "imaging", ("뇌파", "eeg"), "TEST"),
    (r"^thyroid_uptake", "갑상선 섭취율 스캔", "imaging", ("섭취율", "갑상선스캔"), "TEST"),
    (r"^(msu_crystal|cppd_crystal|synovial_)", "관절천자(관절액 검사)", "invasive", ("관절천자", "관절액", "활액", "synovial"),
     "TEST"),
    (r"^(pcos_us|us_no_iup)", "골반 초음파(질식)", "imaging", ("골반초음파", "질식초음파", "경질초음파", "자궁초음파"), "TEST"),
    (r"^testis_no_flow", "음낭(고환) 도플러 초음파", "imaging", ("고환초음파", "음낭초음파", "도플러"), "TEST"),
    (r"^blood_culture_pos", "혈액배양", "lab", ("혈액배양", "bloodculture"), "TEST"),
    (r"^(atypical_lymph|schistocytes|spherocytes|blasts|auer)", "말초혈액 도말", "lab", ("말초혈액", "도말", "smear"), "TEST"),
    (r"^(mcv_high|mcv_low)", "일반혈액검사(CBC)", "lab", ("일반혈액", "cbc", "전혈구", "혈구검사"), "TEST"),
    (r"^(ast_alt_very_high|alp_high)", "간기능 검사", "lab", ("간기능", "간수치", "lft", "ast", "alt", "alp"), "TEST"),
    (r"^(tsh_|ft4_)", "갑상선 기능 검사(TSH, 유리T4)", "lab", ("갑상선기능", "tsh", "t4", "tft"), "TEST"),
    (r"^(glucose_very_high)", "혈당", "lab", ("혈당", "glucose"), "TEST"),
    (r"^hba1c_high", "당화혈색소(HbA1c)", "lab", ("당화혈색소", "hba1c"), "TEST"),
    (r"^ketone_pos", "혈청·소변 케톤", "lab", ("케톤", "ketone"), "TEST"),
    (r"^(anion_gap_high|metabolic_acidosis)", "동맥혈가스·전해질(음이온차)", "lab", ("동맥혈", "abg", "음이온차", "혈액가스"),
     "TEST"),
    (r"^(ferritin_|tsat_high)", "철 검사(페리틴, 트랜스페린 포화도)", "lab", ("페리틴", "ferritin", "철검사", "철분"), "TEST"),
    (r"^(b12_low|mma_high)", "비타민 B12·메틸말론산", "lab", ("b12", "비타민b", "메틸말론산"), "TEST"),
    (r"^(factor8_low|factor9_low|vwf_low|aptt_prolonged)", "응고검사(aPTT, 응고인자, vWF)", "lab",
     ("응고", "aptt", "ptt", "인자"), "TEST"),
    (r"^(fibrinogen_low|fdp_high)", "DIC 검사(피브리노겐, FDP)", "lab", ("피브리노겐", "fdp", "dic"), "TEST"),
    (r"^complement_low", "보체(C3, C4)", "lab", ("보체", "c3", "c4", "complement"), "TEST"),
    (r"^(anca_)", "ANCA(PR3, MPO)", "lab", ("anca",), "TEST"),
    (r"^m_protein", "혈청 단백전기영동", "lab", ("전기영동", "spep", "m단백"), "TEST"),
    (r"^flow_pnh", "유세포분석(CD55/CD59)", "lab", ("유세포", "cd55", "cd59"), "TEST"),
    (r"^acth_stim_fail", "ACTH 자극검사", "lab", ("acth자극", "신속acth"), "TEST"),
    (r"^cushing_tests", "덱사메타손 억제검사", "lab", ("덱사메타손", "심야코르티솔"), "TEST"),
    (r"^aldo_renin_high", "알도스테론/레닌 비", "lab", ("알도스테론", "레닌"), "TEST"),
    (r"^gh_not_suppressed", "경구당부하 후 성장호르몬 억제검사", "lab", ("성장호르몬",), "TEST"),
    (r"^(siadh_urine)", "소변 삼투압·나트륨", "lab", ("소변삼투압", "소변나트륨"), "TEST"),
    (r"^(dilute_urine|desmopressin_)", "물제한·데스모프레신 검사", "lab", ("물제한", "수분제한", "데스모프레신"), "TEST"),
    (r"^kf_ring", "세극등 안과 검사", "exam", ("세극등", "카이저", "안과"), "EXAM"),
    (r"^schirmer_low", "쉬르머 검사", "exam", ("쉬르머", "schirmer"), "EXAM"),
    (r"^(scrub_pos)", "쯔쯔가무시 항체 검사", "lab", ("쯔쯔가무시", "scrub"), "TEST"),
    (r"^legionella_pos", "레지오넬라 소변항원 검사", "lab", ("레지오넬라", "legionella"), "TEST"),
    (r"^afb_pos", "객담 항산균 도말·배양(결핵 PCR)", "lab", ("항산균", "afb", "결핵"), "TEST"),
    (r"^calcium_high", "혈청 칼슘", "lab", ("칼슘", "calcium"), "TEST"),
    (r"^adamts13_low", "ADAMTS13 활성도", "lab", ("adamts13",), "TEST"),
    (r"^(clue_cells|trich_pos)", "질 분비물 검사(습식 도말, pH)", "lab", ("질분비물", "습식도말", "wetmount", "트리코모나스"),
     "TEST"),
]
_TEST_RULES = [(re.compile(p), req, tier, kws, typ) for p, req, tier, kws, typ in TEST_RULES]
_RESULT_SUFFIX = re.compile(r"\s*(현저한\s*)?(상승|감소|저하|양성|음성|증가|억제|검출|연장)(\([^)]*\))?(·[^·]*)?$")
_PAREN_TAIL = re.compile(r"\s*\([^)]*\)\s*$")


def _generic_test(fid: str, ko: str) -> tuple[str, str, tuple[str, ...], str]:
    """Request wording for a curated result without a rule: the analyte name ("리파아제 상승" -> "리파아제 검사")."""
    label = ko.split(":", 1)[-1].strip() if ":" in ko else ko
    base = _PAREN_TAIL.sub("", _RESULT_SUFFIX.sub("", label)).strip(" ·") or label
    req = base if base.endswith("검사") else base + " 검사"
    kws = tuple(k for k in (_norm(x) for x in re.split(r"[·/,]", base)) if len(k) >= 2)
    return req, "lab", kws or (_norm(base),), "TEST"


def _norm(s: str) -> str:
    return _SPACE.sub("", (s or "").lower())


def request_for_result(fid: str, ko: str) -> tuple[str, str, tuple[str, ...], str]:
    """(request text, cost tier, done keywords, action type) for a curated test-result id (without the "TF:" prefix)."""
    for rx, req, tier, kws, typ in _TEST_RULES:
        if rx.search(fid):
            return req, tier, kws, typ
    return _generic_test(fid, ko)


# ------------------------------------------------------------------ helpers
def _entropy(ps) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def _info_gain(prior: list[float], likes: list[list[float]]) -> float:
    """Expected entropy reduction over the joint binary outcomes of the features (likes[f][h] = P(f present | h))."""
    h0 = _entropy(prior)
    exp_h = 0.0
    for outcome in product((True, False), repeat=len(likes)):
        joint = []
        for h, ph in enumerate(prior):
            v = ph
            for f, present in enumerate(outcome):
                q = likes[f][h]
                v *= q if present else 1.0 - q
            joint.append(v)
        z = sum(joint)
        if z > 0:
            exp_h += z * _entropy([x / z for x in joint])
    return max(0.0, h0 - exp_h)


class _Hyp:
    __slots__ = ("name", "idx", "prior", "danger", "sym", "tst", "sparse_sym", "sparse_tst")

    def __init__(self, name: str, idx: int | None, prior: float, danger: bool = False):
        self.name, self.idx, self.prior, self.danger = name, idx, prior, danger
        self.sym: dict[str, float] = {}
        self.tst: dict[str, float] = {}
        self.sparse_sym = self.sparse_tst = True

    def p(self, tid: str) -> float:
        if tid.startswith("TF:"):
            if self.idx is None:
                return LEAK_TEST_OTHER
            return self.tst.get(tid, LEAK_TEST_SPARSE if self.sparse_tst else LEAK_TEST)
        if self.idx is None:
            return LEAK_SYM_OTHER
        return self.sym.get(tid, LEAK_SYM_SPARSE if self.sparse_sym else LEAK_SYM)


def _profile_features(k, h: _Hyp) -> None:
    d = k.diseases[h.idx]
    freq = dict(d.get("orpha_freq", ()))
    for t, srcs in d.get("symptoms", []) + d.get("risk", []):
        if t in k.stop:
            continue
        if t in freq and freq[t] in P_ORPHA:
            p = P_ORPHA[freq[t]]
            if any(s != "Orphanet" for s in srcs):
                p = max(p, P_SYM_SINGLE)
        else:
            p = P_SYM_MULTI if ("DDXPlus" in srcs or len(set(srcs)) >= 2) else P_SYM_SINGLE
        h.sym[t] = max(h.sym.get(t, 0.0), p)
    for t, f in freq.items():
        if f == "EX" and t not in h.sym:
            h.sym[t] = P_ORPHA["EX"]
    for t, p in list(h.sym.items()):
        for g in k.implies.get(t, ()):
            if g not in h.sym:
                h.sym[g] = P_IMPLIED
    for x in d.get("findings_from_tests", []):
        h.tst[x[0]] = max(h.tst.get(x[0], 0.0), P_TEST.get(int(x[2]), 0.3))
    h.sparse_sym = len(d.get("symptoms", [])) < 5
    h.sparse_tst = not h.tst


def _case_text(state) -> str:
    parts = [getattr(state, "initial_info", "") or ""]
    parts += [t.response for t in getattr(state, "turns", []) if "제공되지 않습니다" not in (t.response or "")]
    return "\n".join(parts)


def _known_terms(k, state) -> set[str]:
    """KB term ids already known (present or absent) from the case text + the findings ledger."""
    from doctor_agent.knowledge import kb_tests
    from doctor_agent.nlp import parse
    from doctor_agent.nlp.lexicon import LEXICON
    known: set[str] = set()
    texts = [_case_text(state)] + [f"{f.item} {f.detail}".strip() for f in state.findings.items]
    for i, text in enumerate(texts):
        if not text:
            continue
        for f in parse(text, "exam" if i == 0 else "claim"):
            if f.subject != "patient" or f.hypothetical:
                continue
            c = LEXICON.concept(f.concept)
            if c:
                known.update(c.kb)
            known.update(k.cmap.get(f.concept, {}))
        for fid, (pol, _v) in kb_tests.detect(text).items():
            if pol:
                known.add("TF:" + fid)
    return known


def _hypotheses(k, state) -> list[_Hyp]:
    from doctor_agent.knowledge import kb
    live = [e for e in state.ddx_ledger.ranked() if e.status != "배제"][:MAX_DX]
    rows = [(e.dx, e.p, e.status == "위험") for e in live]
    if not rows:
        rows = [(str(d.get("dx", "")), float(d.get("p", 0) or 0), False) for d in (state.ddx or [])[:MAX_DX]
                if isinstance(d, dict) and d.get("dx")]
    if not rows:  # no DDx yet: KB candidates of the case text
        sex, age = kb.patient_profile(state.initial_info or "")
        pos = [state.initial_info or ""] + [f.item for f in state.findings.items if f.status == "양성"]
        neg = [f.item for f in state.findings.items if f.status == "음성"]
        cands = k.candidates(pos, k=MAX_DX, negatives=neg, sex=sex or None, age=age)
        tot = sum(max(c["score"], 0.0) for c in cands) or 1.0
        rows = [(c["name_ko"], max(c["score"], 0.0) / tot, False) for c in cands]
    hyps: list[_Hyp] = []
    seen: set[int] = set()
    for name, p, danger in rows:
        i = k._resolve(name)
        if i is not None and i in seen:
            continue
        if i is not None:
            seen.add(i)
        hyps.append(_Hyp(name, i, max(p, 0.02) * (DANGER_BOOST if danger else 1.0), danger))
    if not hyps:
        return []
    tot = sum(h.prior for h in hyps)
    for h in hyps:
        h.prior = (1.0 - P_OTHER) * h.prior / tot
        if h.idx is not None:
            _profile_features(k, h)
    hyps.append(_Hyp("기타 질환", None, P_OTHER))
    return hyps


def _disease_hits(k, label: str) -> set[int]:
    """KB profiles whose name is exactly this label (the feature is itself a disease: "동맥염", "뇌성마비")."""
    from doctor_agent.knowledge.kb import _n
    return {i for i, _ in k.names.get(_n(label), ())} if label else set()


def _ask_text(k, tid: str, hyp_idx: frozenset = frozenset(), hyp_names: tuple[str, ...] = ()
              ) -> tuple[str, str, str] | None:
    """(action type, Korean content, source) for a symptom / history KB term; None when it cannot be worded or would
    ask about one of the hypotheses themselves ("동맥염" while 다카야스 동맥염 is on the list)."""
    from doctor_agent.nlp.lexicon import LEXICON
    t = k.terms[tid]
    dis = _disease_hits(k, (t.get("ko") or "").strip()) | _disease_hits(k, (t.get("en") or "").strip())
    if dis & hyp_idx:
        return None
    key = _NON_WORD.sub("", (t.get("ko") or "").lower())
    if dis and len(key) >= 2 and any(key in n for n in hyp_names):
        return None
    concepts = [LEXICON.concept(c) for c in LEXICON.by_kb(tid)]
    concepts = [c for c in concepts if c and not c.group]
    sign = next((c for c in concepts if c.cat == "SIGN"), None)
    ko = (t.get("ko") or "").strip()
    q = (t.get("q") or {}).get("ko")
    if q:  # DDXPlus asks the patient (also about signs the patient can notice, e.g. a rash)
        return "ASK", q, "DDXPlus"
    if sign or (not concepts and _SIGN_LABEL.search(ko)):
        label = sign.ko if sign else ko
        key = f"{sign.id if sign else ''} {label}".lower()
        area = next((a for rx, a in EXAM_AREAS if rx.search(key)), "신체 진찰")
        return "EXAM", f"{area}({label} 확인)", "lexicon" if sign else "KB"
    hx = next((c for c in concepts if c.cat == "HX"), None)
    if hx:
        return "ASK", HX_QUESTIONS.get(hx.id, f"{hx.ko} 병력이 있으신가요?"), "lexicon"
    sym = next((c for c in concepts if c.cat in ("SYM", "QUAL")), None)
    label = sym.ko if sym else ko
    if not label or not _HANGUL.search(label) or label in _GENERIC_TERMS or len(label) > 20:
        return None
    if dis and not sym:  # a disease as a risk factor
        return "ASK", f"{label} 진단을 받으신 적이 있나요?", "KB"
    return "ASK", f"{label}{_josa(label)} 있으신가요?", "lexicon" if sym else "KB"


def _done_test(state, kws: tuple[str, ...]) -> bool:
    for t in state.turns:
        if t.action.type in (ActionType.TEST, ActionType.EXAM):
            c = _norm(t.action.content)
            if any(kw and kw in c for kw in kws):
                return True
    return False


# ------------------------------------------------------------------ public API
def suggest(state, k: int = 3, include_safety: bool = True) -> list[Suggestion]:
    """Top-k discriminating next actions (+ pending safety checks marked safety=True when include_safety). Fail-safe:
    any problem returns what could be computed (possibly [])."""
    try:
        return _suggest(state, k, include_safety)
    except Exception:
        return []


def _suggest(state, k: int, include_safety: bool) -> list[Suggestion]:
    from doctor_agent.knowledge import kb
    if not kb.available():
        return _safety(state) if include_safety else []
    kbase = kb.get_kb()
    hyps = _hypotheses(kbase, state)
    ranked: list[Suggestion] = []
    if len(hyps) >= 2:
        known = _known_terms(kbase, state)
        prior = [h.prior for h in hyps]
        # candidate features: everything the named hypotheses list
        feats: set[str] = set()
        for h in hyps:
            feats.update(h.sym)
            feats.update(h.tst)
        feats -= known
        scored: dict[str, float] = {}
        for tid in feats:
            if tid not in kbase.terms or tid in kbase.stop:
                continue
            likes = [h.p(tid) for h in hyps]
            if max(likes) - min(likes) < 0.05:
                continue
            scored[tid] = _info_gain(prior, [likes])
        # group into actions
        hyp_idx = frozenset(h.idx for h in hyps if h.idx is not None)
        hyp_names = tuple(_NON_WORD.sub("", h.name.lower()) for h in hyps if h.idx is not None)
        actions: dict[tuple[str, str], dict] = {}
        for tid, ig in sorted(scored.items(), key=lambda x: (-x[1], x[0])):
            if ig <= 1e-4:
                continue
            if tid.startswith("TF:"):
                req, tier, kws, typ = request_for_result(tid[3:], kbase.terms[tid]["ko"])
                src = "curated"
            else:
                r = _ask_text(kbase, tid, hyp_idx, hyp_names)
                if r is None:
                    continue
                typ, req, src = r
                tier, kws = ("exam" if typ == "EXAM" else "ask"), ()
            a = actions.setdefault((typ, req), {"tier": tier, "kws": kws, "src": src, "feats": []})
            if len(a["feats"]) < (MAX_FEATS_PER_TEST if typ == "TEST" else 3):
                a["feats"].append(tid)
        cands: list[Suggestion] = []
        for (typ, req), a in actions.items():
            likes = [[h.p(t) for h in hyps] for t in a["feats"]]
            ig = _info_gain(prior, likes)
            mean = [sum(p * q for p, q in zip(prior, ls)) for ls in likes]
            targets = []
            for j, h in enumerate(hyps):
                if h.idx is None:
                    continue
                if any(ls[j] >= max(0.3, m + 0.15) for ls, m in zip(likes, mean)):
                    targets.append(h.name)
            if not targets:  # the action argues *against* someone: name the most affected hypothesis
                j = max(range(len(hyps) - 1), key=lambda j: max(abs(ls[j] - m) for ls, m in zip(likes, mean)))
                targets = [hyps[j].name]
            cite = ""
            if typ == "TEST" and a["src"] == "curated":
                refs = []
                for t in a["feats"]:
                    for h in hyps:
                        if h.idx is None:
                            continue
                        for x in kbase.diseases[h.idx].get("findings_from_tests", []):
                            if x[0] == t and len(x) > 3 and x[3] and x[3] not in refs:
                                refs.append(x[3])
                cite = "; ".join(filter(None, (kbase.test_ref(r).get("cite", "").split(".")[0] for r in refs[:2])))
            cands.append(Suggestion(typ, req, targets, round(ig * TIER_W.get(a["tier"], 0.5), 4), a["tier"], a["src"],
                                    cite, features=list(a["feats"])))
        cands.sort(key=lambda s: (-s.expected_value, s.type, s.content_ko))
        ranked = _filter(state, cands, k, actions)
    pending = _safety(state)
    for s in ranked:
        for p in pending:
            if _matches_check(p, s.content_ko):
                s.safety = True
                s.note = (s.note + " " if s.note else "") + f"안전 확인 항목({p.content_ko})"
    return ranked + ([p for p in pending if not any(_matches_check(p, s.content_ko) for s in ranked)]
                     if include_safety else [])


def _filter(state, cands: list[Suggestion], k: int, actions: dict) -> list[Suggestion]:
    from doctor_agent.safety import preconditions
    out: list[Suggestion] = []
    for s in cands:
        if len(out) >= k:
            break
        act = s.action()
        if state.asked(act):
            continue
        if s.type in ("TEST", "EXAM") and _done_test(state, actions[(s.type, s.content_ko)]["kws"]):
            continue
        if any(o.type == s.type and similarity(o.content_ko, s.content_ko) >= SIMILAR for o in out):
            continue
        if s.type in ("TEST", "EXAM"):
            chk = preconditions.check(act.type, s.content_ko, state)
            if chk.get("severity") == "block":
                continue
            if chk.get("severity") == "warn":
                s.note = f"주의: {chk.get('why', '')[:60]}"
        out.append(s)
    return out


def _matches_check(p: Suggestion, content: str) -> bool:
    c = (content or "").lower()
    return any(kw in c for kw in p.done_keywords)


def _safety(state) -> list[Suggestion]:
    """Pending minimum safety checks as suggestions (safety=True, expected_value 0)."""
    try:
        from doctor_agent.safety import protocols
        actions = [t.action.content for t in state.turns]
        context = "\n".join(t.response for t in state.turns if "제공되지 않습니다" not in (t.response or ""))
        out = []
        for c in protocols.pending_checks(state.initial_info or "", actions, context):
            if c.kind not in ("ask", "exam", "test"):
                continue
            tier = {"ask": "ask", "exam": "exam"}.get(c.kind, "lab")
            out.append(Suggestion(c.kind.upper(), c.name, [], 0.0, tier, f"protocol:{c.id}", c.citation.short,
                                  safety=True, done_keywords=tuple(c.keywords)))
        return out
    except Exception:
        return []


def render_for_prompt(suggestions: list[Suggestion], max_chars: int = MAX_CHARS, include_safety: bool = False) -> str:
    """Short Korean hint: "추천 다음 행동 (참고): 1) [문진] ... (감별: A·B) ..."; "" when there is nothing to show."""
    rows = [s for s in suggestions if include_safety or not s.source.startswith("protocol:")]
    if not rows:
        return ""
    head = "추천 다음 행동 (참고):"
    parts = []
    for i, s in enumerate(rows, 1):
        tgt = "·".join(t[:12] for t in s.targets[:2])
        parts.append(f" {i}) [{TYPE_KO.get(s.type, s.type)}] {s.content_ko}" + (f" (감별: {tgt})" if tgt else ""))
    text = head
    for p in parts:
        if len(text) + len(p) > max_chars:
            break
        text += p
    if text == head:  # the first item alone is too long: cut it
        text = (head + parts[0])[: max_chars - 1] + "…"
    return text
