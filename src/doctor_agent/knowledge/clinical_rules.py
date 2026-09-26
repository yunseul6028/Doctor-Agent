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
"""
from __future__ import annotations

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
}

# Lowercase substrings matched against the case text (Korean + English variants)
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chest_pain": (
        "흉통", "가슴 통증", "가슴통증", "가슴이 아", "가슴 아", "가슴이 답답", "가슴 답답", "가슴이 조이",
        "가슴을 쥐어", "가슴이 쥐어", "가슴이 뻐근", "가슴이 찢", "가슴이 짓눌", "가슴이 타", "chest pain", "chest tightness", "chest discomfort",
    ),
    "dyspnea": (
        "호흡곤란", "숨이 차", "숨차", "숨쉬기", "숨이 막", "숨 가쁨", "숨가쁨", "숨이 가쁘",
        "dyspnea", "dyspnoea", "shortness of breath", "short of breath", "breathless",
    ),
    "headache": ("두통", "머리가 아", "머리 아", "머리가 깨질", "머리가 터질", "headache"),
    "neuro": (
        "마비", "힘이 빠", "힘이 없", "저림", "저리", "감각 저하", "감각이 없", "말이 어눌", "발음이 어눌",
        "구음장애", "실어", "어지럼", "어지러", "현훈", "의식 저하", "의식저하", "의식 변화", "의식이 흐",
        "혼돈", "헛소리", "경련", "발작", "실신", "기절", "얼굴이 비뚤", "입이 돌아", "시야", "복시",
        "weakness", "numbness", "slurred", "aphasia", "dizz", "vertigo", "seizure", "syncope",
        "confusion", "altered mental", "facial droop", "stroke", "transient ischemic",
    ),
    "fever": (
        "발열", "열이", "열나", "열감", "오한", "고열", "미열", "fever", "febrile", "chills",
    ),
    "abdominal_pain": (
        "복통", "배가 아", "배 아", "배 통증", "배가 쥐어", "명치", "윗배", "아랫배", "옆구리", "상복부",
        "하복부", "abdominal pain", "belly pain", "stomach ache", "stomachache", "epigastric",
    ),
}


def detect_categories(text: str) -> list[str]:
    """Categories whose keywords appear in the text, in CATEGORY_KEYWORDS order."""
    t = (text or "").lower()
    return [c for c, kws in CATEGORY_KEYWORDS.items() if any(k in t for k in kws)]


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
        return max(p for _, p in self.options) if self.options else self.points


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

# --------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------

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
        note="Items and >=2 cut point from the PubMed abstract. Surviving Sepsis Campaign 2021 recommends "
        "AGAINST qSOFA as a single screening tool (see safety/protocols.py), so a low score never rules out sepsis.",
    ),
    Rule(
        id="curb65", short="CURB-65", name_ko="CURB-65 폐렴 중증도 점수", name_en="CURB-65",
        purpose="지역사회획득 폐렴의 30일 사망 위험 분류",
        population="지역사회획득 폐렴으로 진단되었거나 의심되는 성인",
        categories=("fever", "dyspnea"),
        keywords=("폐렴", "기침", "가래", "pneumonia", "cough", "sputum"),
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
        note="All items, points and bands (2-day risk 1.0 / 4.1 / 8.1%) from the PubMed abstract.",
    ),
    Rule(
        id="alvarado", short="Alvarado", name_ko="Alvarado 충수염 점수", name_en="Alvarado score",
        purpose="급성 충수염 가능성 추정",
        population="충수염이 의심되는 복통 환자",
        categories=("abdominal_pain",),
        keywords=("충수", "맹장", "appendic"),
        items=(
            Item("migration", "통증이 우하복부로 이동"),
            Item("anorexia", "식욕부진(또는 소변 케톤)"),
            Item("nausea_vomiting", "오심·구토"),
            Item("rlq_tenderness", "우하복부 압통", 2.0),
            Item("rebound", "반발 압통"),
            Item("fever", "체온 상승(37.3°C 이상)"),
            Item("leukocytosis", "백혈구 증가(10,000/µL 초과)", 2.0),
            Item("left_shift", "호중구 좌방이동"),
        ),
        thresholds=(
            Threshold(0, 4, "0–4", "가능성 낮음", "충수염 가능성 낮음(다른 원인 평가)"),
            Threshold(5, 6, "5–6", "부합", "충수염과 부합 → 관찰·영상검사"),
            Threshold(7, 8, "7–8", "가능성 높음", "충수염 가능성 높음 → 외과 협진"),
            Threshold(9, 10, "9–10", "가능성 매우 높음", "충수염 가능성 매우 높음 → 외과 협진"),
        ),
        citation=C_ALVARADO, verification="unverified",
        note="The abstract confirms the 8 factors and their weight order (RLQ tenderness and leukocytosis "
        "highest) but not the points, the temperature/WBC cut-offs, or the 5-6 / 7-8 / 9-10 bands. Full text "
        "is paywalled; these come from reviewer knowledge. Re-check before relying on the bands.",
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
        note="5 variables and the <1% / >20% range from the abstract. The commonly used '>=3 = severe' cut "
        "point is from later validation papers and is deliberately not encoded here.",
    ),
)

RULES_BY_ID: dict[str, Rule] = {r.id: r for r in RULES}


# --------------------------------------------------------------------------------------------
# Lookup / rendering / scoring
# --------------------------------------------------------------------------------------------


def rules_for(text: str) -> list[Rule]:
    """Rules relevant to a chief complaint (matched by category keywords or rule-specific keywords)."""
    t = (text or "").lower()
    cats = set(detect_categories(t))
    return [r for r in RULES if cats.intersection(r.categories) or any(k in t for k in r.keywords)]


def _item_text(item: Item) -> str:
    if item.options:
        return f"{item.text}(" + "/".join(f"{lab} {_fmt(p)}" for lab, p in item.options) + ")"
    return f"{item.text} +{_fmt(item.points)}"


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
