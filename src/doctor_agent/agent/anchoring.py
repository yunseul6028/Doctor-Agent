"""Broad starting differential and premature-closure (anchoring) check. Content is owned by clinical-strategist.

Two pure functions for the policy (the lead wires them in; nothing here keeps state across calls or cases):

    initial_differential(initial_info) -> list[dict]    <= 8 starting candidates from the chief complaint + age/sex:
        can't-miss diagnoses (safety/protocols.py, tag "위험"), common causes per chief-complaint category (COMMON
        below, tag "흔함") and knowledge-base candidates (knowledge/kb.py, tag "KB"); deduplicated.
    render_for_prompt(ddx, max_chars=300) -> str       short Korean text, shown once at turn 1
    anchoring_check(state, min_turns=None) -> dict | None
                                                       devil's-advocate prompt when the top live DDx looks
                                                       prematurely closed (only after min_turns turns, default
                                                       MIN_TURNS; the caller fires it at most once per case)

COMMON table: every category cites a review/guideline whose differential covers these causes. Citation.verified =
bibliographic data checked against PubMed E-utilities on 2026-09-28. The row *content* was written from reviewer
knowledge and matched to the article topic, not line-checked against the full text (CONTENT_VERIFICATION =
"unverified"); pediatric rows (peds=True) and categories with ref None are standard textbook knowledge without a
specific source (source "미검증"). Caveat: the table was written after looking at the chief complaints of
data/cases_aug, so its offline recall there is optimistic.
The extra categories (EXTRA_CATEGORIES) cover frequent chief complaints that clinical_rules.detect_categories()
does not open a safety protocol for (dizziness, diarrhea, vision loss, ...); they add common causes only.

Stdlib + project modules only, CPU only, no network. Any KB problem degrades to no KB rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from doctor_agent.agent.text import dx_keys, same_dx, similarity
from doctor_agent.knowledge.clinical_rules import CATEGORY_NAMES, Citation, detect_categories, rule_age_years
from doctor_agent.safety import protocols as P

MAX_DDX = 8
MAX_RENDER = 300
# anchoring_check never fires before this many turns. 5 (was 3): from turn 3 it fired on 59% of the replayed
# trajectories that ended right and 46% of those that ended wrong; from turn 5 on 26% / 35% (eval/offline/
# eval_triggers.py, runs of non-competition dev models; re-check on gpt-oss-20b). Same default as
# AgentConfig.anchoring_min_turns (AGENT_ANCHORING_MIN_TURNS), which the policy passes in.
MIN_TURNS = 5
EARLY_TURN = 2         # "since early turns": the top candidate was already top at turn <= EARLY_TURN
MIN_P = 0.4            # a top candidate below this probability is not "closed"
WEAK_SUPPORT_P = 0.6   # weak-support trigger: top p >= this with <= 1 grounded supporting finding
MIN_AGAINST = 2        # contradiction trigger: >= this many concrete findings listed against the top candidate
CONTENT_VERIFICATION = "unverified"


def _aafp(authors: str, title: str, year: int, vp: str, pmid: str) -> Citation:
    return Citation(authors, title, "Am Fam Physician", year, vp, pmid=pmid, verified=True,
                    short_author="AAFP " + authors.split()[0])


# review articles checked against PubMed (esummary) on 2026-09-28
R_CHEST = _aafp("McConaghy JR, Sharma M, Patel H", "Acute Chest Pain in Adults: Outpatient Evaluation", 2020,
                "102(12):721-727", "33320506")
R_DYSPNEA = _aafp("Budhwar N, Syed Z", "Chronic Dyspnea: Diagnosis and Evaluation", 2020, "101(9):542-548", "32352727")
R_HEADACHE = _aafp("Viera AJ, Antono B", "Acute Headache in Adults: A Diagnostic Approach", 2022, "106(3):260-268",
                   "36126007")
R_ABDOMEN = _aafp("Cartwright SL, Knudson MP", "Evaluation of acute abdominal pain in adults", 2008, "77(7):971-978",
                  "18441863")
R_JAUNDICE = _aafp("Fargo MV, Grogan SP, Saguil A", "Evaluation of Jaundice in Adults", 2017, "95(3):164-168",
                   "28145671")
R_JOINT = _aafp("Foster ZJ, Voss TT, Hatch J, Frimodig A", "Polyarticular Joint Pain in Adults: Evaluation and "
                "Differential Diagnosis", 2023, "107(1):42-51", "36689970")
R_RASH = _aafp("Ely JW, Seabury Stone M", "The generalized rash: part I. Differential diagnosis", 2010,
               "81(6):726-734", "20229971")
R_EDEMA = _aafp("Trayes KP, Studdiford JS, Pickle S, Tully AS", "Edema: diagnosis and management", 2013,
                "88(2):102-110", "23939641")
R_FATIGUE = _aafp("Rosenthal TC, Majeroni BA, Pretorius R, Malik K", "Fatigue: an overview", 2008,
                  "78(10):1173-1179", "19035066")
R_BLEEDING = _aafp("Ballas M, Kraut EH", "Bleeding and bruising: a diagnostic work-up", 2008, "77(8):1117-1124",
                   "18481559")
R_WEAKNESS = _aafp("Saguil A", "Evaluation of the patient with muscle weakness", 2005, "71(7):1327-1336", "15832536")
R_PALPITATIONS = _aafp("Wexler RK, Pleister A, Raman SV", "Palpitations: Evaluation in the Primary Care Setting", 2017,
                       "96(12):784-789", "29431371")
R_HEMOPTYSIS = _aafp("Earwood JS, Thompson TD", "Hemoptysis: evaluation and management", 2015, "91(4):243-249",
                     "25955625")
R_HEARING = _aafp("Michels TC, Duffy MT, Rogers DJ", "Hearing Loss in Adults: Differential Diagnosis and Treatment",
                  2019, "100(2):98-108", "31305044")
R_AMENORRHEA = _aafp("Klein DA, Paradise SL, Reeder RM", "Amenorrhea: A Systematic Approach to Diagnosis and "
                     "Management", 2019, "100(1):39-48", "31259490")
R_DIZZINESS = _aafp("Muncie HL, Sirmans SM, James E", "Dizziness: Approach to Evaluation and Management", 2017,
                    "95(3):154-162", "28145669")
R_DIARRHEA = _aafp("Burgers K, Lindberg B, Bevis ZJ", "Chronic Diarrhea in Adults: Evaluation and Differential "
                   "Diagnosis", 2020, "101(8):472-480", "32293842")
R_VISION = _aafp("Pelletier AL, Rojas-Roldan L, Coffin J", "Vision Loss in Older Adults", 2016, "94(3):219-226",
                 "27479624")
R_TREMOR = _aafp("Crawford P, Zimmerman EE", "Tremor: Sorting Through the Differential Diagnosis", 2018,
                 "97(3):180-186", "29431985")
R_BREAST = _aafp("Klein S", "Evaluation of palpable breast masses", 2005, "71(9):1731-1738", "15887452")
R_VOMITING = _aafp("Scorza K, Williams A, Phillips JD, Shaw J", "Evaluation of nausea and vomiting", 2007,
                   "76(1):76-84", "17668843")
R_CONSTIPATION = _aafp("Sadler K, Arnold F, Dean S", "Chronic Constipation in Adults", 2022, "106(3):299-306",
                       "36126011")
R_LGIB = _aafp("Hawks MK, Svarverud JE", "Acute Lower Gastrointestinal Bleeding: Evaluation and Management", 2020,
               "101(4):206-212", "32053333")
R_LUTS = _aafp("Arnold MJ, Gaillardetz A, Ohiokpehai J", "Benign Prostatic Hyperplasia: Rapid Evidence Review", 2023,
               "107(6):613-622", "37327163")


@dataclass(frozen=True)
class Common:
    dx: str
    sex: str = ""                 # "여성" / "남성": only for that sex (unstated sex passes)
    min_age: float | None = None  # years; unstated age passes
    max_age: float | None = None
    peds: bool = False            # pediatric row: standard pediatric teaching, not from the category's (adult) source


def _c(*names: str) -> tuple[Common, ...]:
    return tuple(Common(n) for n in names)


def _kid(name: str, max_age: float, min_age: float | None = None) -> Common:
    return Common(name, min_age=min_age, max_age=max_age, peds=True)


F, M = "여성", "남성"
# category -> (reference, common causes). Pediatric rows (peds=True, source "미검증") come first and apply only to
# children; adult-only rows carry min_age. Within an age group the order is roughly most common first.
COMMON: dict[str, tuple[Citation | None, tuple[Common, ...]]] = {
    "chest_pain": (R_CHEST, (Common("근골격계 흉벽 통증"), Common("위식도 역류질환"), Common("안정형 협심증", min_age=30),
                             Common("공황장애"), Common("폐렴"), Common("급성 심낭염"))),
    "dyspnea": (R_DYSPNEA, (_kid("신생아 일과성 빈호흡", 0.1), _kid("신생아 호흡곤란증후군", 0.1), _kid("태변 흡인 증후군", 0.1),
                            _kid("선천성 심질환", 1), _kid("세기관지염", 2, 0.08), _kid("크룹", 6, 0.5),
                            _kid("기도 이물 흡인", 5, 0.5), Common("천식", min_age=1), Common("만성 폐쇄성 폐질환", min_age=40),
                            Common("심부전", min_age=15), Common("폐렴", min_age=0.1), Common("빈혈", min_age=1),
                            Common("간질성 폐질환", min_age=30))),
    "headache": (R_HEADACHE, _c("긴장형 두통", "편두통", "약물과용 두통", "군발두통", "부비동염")),
    "neuro": (P.G_STROKE, _c("급성 허혈성 뇌졸중", "경련 후 마비(토드 마비)", "편두통 조짐", "뇌전증 발작", "대사성 뇌병증")),
    "fever": (None, (_kid("급성 중이염", 12, 0.25), _kid("돌발진", 3, 0.25), _kid("수족구병", 10, 0.25),
                     Common("바이러스성 상기도 감염"), Common("요로감염/신우신염"), Common("폐렴"), Common("급성 위장관염"),
                     Common("연조직염"))),
    "abdominal_pain": (R_ABDOMEN, (_kid("장중첩증", 5, 0.25), _kid("장간막 림프절염", 15, 2), _kid("변비", 15, 1),
                                   Common("급성 위장관염"), Common("급성 충수염", min_age=2),
                                   Common("담석증/급성 담낭염", min_age=12), Common("소화성 궤양", min_age=12),
                                   Common("요로결석", min_age=12), Common("과민성 장증후군", min_age=12),
                                   Common("급성 췌장염", min_age=12), Common("골반염증성 질환", sex=F, min_age=12))),
    "allergy": (P.G_ANAPHYLAXIS, _c("급성 두드러기", "혈관부종", "음식 알레르기", "약물 알레르기")),
    "syncope": (P.G_SYNCOPE, _c("미주신경성 실신", "기립성 저혈압", "상황성 실신", "부정맥성 실신")),
    "palpitations": (R_PALPITATIONS, _c("심방 또는 심실 조기수축", "발작성 상심실성 빈맥", "심방세동", "불안/공황장애",
                                        "갑상선 기능 항진증")),
    "hemoptysis_cough": (R_HEMOPTYSIS, _c("급성 기관지염", "기관지확장증", "폐결핵", "폐암", "천식", "위식도 역류질환")),
    "jaundice": (R_JAUNDICE, (_kid("신생아 생리적 황달", 0.1), _kid("모유 황달", 0.25),
                              _kid("신생아 용혈성 질환(혈액형 부적합)", 0.1), _kid("담도 폐쇄증", 0.5),
                              Common("총담관 결석", min_age=12), Common("바이러스성 간염", min_age=1),
                              Common("알코올성 간질환", min_age=15), Common("약물 유발 간손상", min_age=1),
                              Common("췌장암/담관암", min_age=40), Common("길버트 증후군", min_age=1))),
    "joint": (R_JOINT, (_kid("일과성 활막염", 10, 2), _kid("레그-칼베-페르테스병", 12, 3), _kid("대퇴골두 골단 분리증", 16, 9),
                        _kid("소아 특발성 관절염", 16, 1), _kid("성장통", 12, 3), Common("골관절염", min_age=30),
                        Common("통풍", min_age=18), Common("류마티스 관절염", min_age=16), Common("가성통풍", min_age=40),
                        Common("반응성 관절염", min_age=10), Common("점액낭염/건염", min_age=12))),
    "back_pain": (P.G_LOW_BACK_PAIN, _c("비특이적 요통(요추 염좌)", "요추 추간판 탈출증", "척추관 협착증", "강직성 척추염",
                                         "골다공증성 압박골절")),
    "rash": (R_RASH, (_kid("신생아 중독성 홍반", 0.1), _kid("영아 지루 피부염", 1), _kid("돌발진", 3, 0.25),
                      _kid("수족구병", 10, 0.25), _kid("전염성 홍반", 15, 2), _kid("수두", 15, 1), _kid("아토피 피부염", 15),
                      Common("바이러스 발진"), Common("약물 발진"), Common("접촉 피부염"), Common("두드러기"),
                      Common("대상포진", min_age=15), Common("성홍열", min_age=2, max_age=15))),
    "pruritus": (P.G_PRURITUS, _c("피부 건조증", "담즙 정체성 간질환", "만성 콩팥병(요독성 소양증)", "갑상선 질환",
                                  "철결핍", "림프종")),
    "edema": (R_EDEMA, _c("만성 정맥 기능부전", "심부전", "신증후군", "간경변", "약물 유발 부종(칼슘통로차단제)",
                          "급성 사구체신염")),
    "menstrual": (R_AMENORRHEA, (Common("임신", sex=F), Common("다낭성 난소 증후군", sex=F),
                                 Common("시상하부성 무월경", sex=F), Common("고프로락틴혈증"), Common("갑상선 기능 이상"),
                                 Common("조기 난소 부전", sex=F), Common("자궁근종", sex=F, min_age=18))),
    "fatigue": (R_FATIGUE, _c("우울증", "빈혈", "갑상선 기능 저하증", "수면 장애(수면무호흡)", "당뇨병", "만성 간질환")),
    "cognitive": (P.G_DEMENTIA, _c("알츠하이머병", "혈관성 치매", "경도인지장애", "우울증(가성 치매)", "비타민 B12 결핍",
                                   "갑상선 기능 저하증")),
    "psychiatric": (P.G_PSYCH_EVAL, _c("주요우울장애", "양극성 장애", "조현병", "불안장애", "물질 관련 장애", "섬망")),
    "urticaria_chronic": (P.G_URTICARIA, _c("만성 자발성 두드러기", "물리적 두드러기(유발성)",
                                            "자가면역 갑상선염 동반 두드러기")),
    "hearing_loss": (R_HEARING, _c("귀지 막힘", "노인성 난청", "소음성 난청", "삼출성 중이염", "이경화증", "청신경초종")),
    "neck_mass": (P.G_NECK_MASS, _c("반응성 림프절염", "갑상설관 낭종", "새열 낭종", "두경부암 전이", "림프종")),
    "bleeding": (R_BLEEDING, _c("면역 혈소판 감소증", "폰빌레브란트병", "혈우병", "약물(항응고제·항혈소판제)", "간질환",
                                "백혈병")),
    "chronic_weakness": (R_WEAKNESS, _c("염증성 근병증(다발성 근염)", "갑상선 근병증", "약물 유발 근병증(스테로이드·스타틴)",
                                        "근위축성 측삭 경화증", "경추 척수병증", "중증 근무력증")),
    "bilious_vomiting": (P.G_INFANT_VOMITING, _c("장회전 이상과 중장 염전", "십이지장 폐쇄", "공장·회장 폐쇄",
                                                 "히르슈슈프룽병", "괴사성 장염")),
    # --- extra complaint categories (not safety protocols; common causes only) ---
    "x_dizziness": (R_DIZZINESS, _c("양성 돌발성 체위성 현훈", "전정신경염", "메니에르병", "전정 편두통", "기립성 저혈압",
                                    "후순환 뇌졸중")),
    "x_diarrhea": (R_DIARRHEA, _c("감염성 장염", "과민성 장증후군", "염증성 장질환", "클로스트리디오이데스 디피실 감염",
                                  "셀리악병", "약물 유발 설사")),
    "x_vision": (R_VISION, _c("백내장", "연령 관련 황반변성", "녹내장", "당뇨망막병증", "망막 박리", "망막 중심동맥 폐쇄")),
    "x_tremor": (R_TREMOR, _c("본태성 진전", "파킨슨병", "생리적 진전 증강(카페인·불안)", "약물 유발 진전",
                              "갑상선 기능 항진증")),
    "x_breast_mass": (R_BREAST, (Common("섬유선종", sex=F), Common("유방 낭종", sex=F), Common("섬유낭성 변화", sex=F),
                                 Common("유방암"), Common("여성형 유방", sex=M))),
    "x_vomiting": (R_VOMITING, (_kid("비후성 유문 협착증", 0.25), _kid("영아 위식도 역류", 1), Common("급성 위장관염"),
                                Common("약물 유발 구토"), Common("임신 오조", sex=F, min_age=12), Common("장폐색"),
                                Common("위마비", min_age=15), Common("당뇨병성 케톤산증"))),
    "x_constipation": (R_CONSTIPATION, _c("기능성 변비", "변비형 과민성 장증후군", "약물 유발 변비(아편유사제 등)",
                                          "갑상선 기능 저하증", "대장암")),
    "x_rectal_bleeding": (R_LGIB, _c("치핵", "치열", "게실 출혈", "대장암", "염증성 장질환", "허혈성 대장염")),
    "x_luts": (R_LUTS, (Common("양성 전립선 비대증", sex=M, min_age=40), Common("요로감염"), Common("과민성 방광"),
                        Common("전립선염", sex=M), Common("방광암", min_age=40))),
    "x_diplopia": (None, _c("중증 근무력증", "당뇨병성 뇌신경 마비", "갑상선 안병증", "뇌동맥류(동안신경 마비)",
                            "다발성 경화증")),
    "x_polyuria": (None, _c("당뇨병", "중추성 요붕증", "신성 요붕증", "고칼슘혈증", "심인성 다음증")),
    "x_skin_lesion": (None, _c("지루각화증", "광선각화증", "기저세포암", "편평세포암", "흑색종")),
}

EXTRA_CATEGORIES: dict[str, tuple[str, ...]] = {
    "x_dizziness": ("어지럼", "어지러", "어지럽", "현훈", "빙빙", "방이 도", "vertigo", "dizz"),
    "x_diarrhea": ("설사", "묽은 변", "diarrhea"),
    "x_vision": ("시력", "보이지 않", "안 보", "흐려 보", "흐릿", "암점", "vision"),
    "x_tremor": ("떨림", "떨리", "진전", "tremor"),
    "x_breast_mass": ("유방", "breast"),
    "x_vomiting": ("구토", "토해", "토를", "구역", "vomit", "nausea"),
    "x_constipation": ("변비", "constipation"),
    "x_rectal_bleeding": ("혈변", "직장 출혈", "항문 출혈", "대변에 피", "rectal bleeding", "hematochezia"),
    "x_luts": ("배뇨", "소변보기", "빈뇨", "잔뇨", "소변 줄기", "dysuria"),
    "x_diplopia": ("복시", "둘로 보", "겹쳐 보", "diplopia"),
    "x_polyuria": ("다뇨", "갈증", "소변을 자주", "소변량 증가", "polyuria", "polydipsia"),
    "x_skin_lesion": ("피부 병변", "반점", "딱지", "비늘", "사마귀", "점이", "skin lesion"),
}

_CC = re.compile(r"주\s*호소\s*[:：]\s*(.*)", re.S)
_FEMALE_DX = re.compile(r"임신|자궁|난소|질출혈|폐경|월경|골반염")
_MALE_DX = re.compile(r"고환|전립선")
_NEONATE_DX = re.compile(r"신생아|영아|장회전|중장 염전|히르슈슈프룽|괴사성 장염|십이지장 폐쇄|공장·회장")
_OLD_DX = re.compile(r"대동맥류|치매|노인성")
_ADULT_DX = re.compile(r"관상동맥|대동맥 박리|폐색전|만성 폐쇄성|협심증|폐경|자궁내막암|고형암|폐암|췌담도|임균|결정성")
_VAGUE_KB = re.compile(r"상세불명|기타|달리 분류")


def _chief_complaint(initial_info: str) -> str:
    m = _CC.search(initial_info or "")
    return (m.group(1) if m else initial_info or "").strip()


def _profile(initial_info: str) -> tuple[str, float | None]:
    """(sex, age in years) from the demographics before "주호소" ("생후 5일 남아", "35세 여성", "신생아")."""
    from doctor_agent.knowledge.kb import patient_profile
    text = initial_info or ""
    demo = text[: _CC.search(text).start()] if _CC.search(text) else text
    sex, age = patient_profile(demo)
    rage = rule_age_years(demo)
    return sex, (rage if rage is not None else age)


def _fits(name: str, sex: str, age: float | None) -> bool:
    """Drops names that cannot apply to the stated sex/age (unstated sex/age keeps everything)."""
    if sex == "남성" and _FEMALE_DX.search(name):
        return False
    if sex == "여성" and _MALE_DX.search(name):
        return False
    if age is not None and age >= 1 and _NEONATE_DX.search(name):
        return False
    if age is not None and age < 40 and _OLD_DX.search(name):
        return False
    if age is not None and age < 12 and _ADULT_DX.search(name):
        return False
    return True


def _row_fits(c: Common, sex: str, age: float | None) -> bool:
    if c.sex and sex and c.sex != sex:
        return False
    if age is not None and c.min_age is not None and age < c.min_age:
        return False
    if age is not None and c.max_age is not None and age > c.max_age:
        return False
    return _fits(c.dx, sex, age)


def _row_source(cat: str, row: Common) -> str:
    ref = COMMON[cat][0]
    return "미검증" if row.peds or ref is None else ref.short


def extra_categories(text: str) -> list[str]:
    t = (text or "").lower()
    return [k for k, kws in EXTRA_CATEGORIES.items() if any(w in t for w in kws)]


def _kb_rows(cc: str, sex: str, age: float | None, k: int) -> list[dict]:
    try:
        from doctor_agent.knowledge import kb
        if not kb.available() or not cc:
            return []
        return [{"dx": c["name_ko"], "tag": "KB", "source": "KB", "kcd": (c.get("kcd") or [""])[0]}
                for c in kb.candidates([cc], k=k, sex=sex or None, age=age)
                if c.get("name_ko") and not _VAGUE_KB.search(c["name_ko"])]
    except Exception:  # noqa: BLE001 - the KB is optional; never break the agent
        return []


def _round_robin(groups: list[list[dict]]) -> list[dict]:
    out, i = [], 0
    while any(i < len(g) for g in groups):
        out += [g[i] for g in groups if i < len(g)]
        i += 1
    return out


def initial_differential(initial_info: str, max_n: int = MAX_DDX, n_danger: int = 3, n_common: int = 4,
                         n_kb: int = 1) -> list[dict]:
    """Broad starting DDx: [{"dx", "tag": 위험|흔함|KB, "source", "category"?, "kcd"?}], <= max_n, deduplicated.

    Order: up to n_danger can't-miss diagnoses, n_common common causes, n_kb KB candidates, then the remaining
    can't-miss / common rows alternately. Several categories are interleaved (round-robin)."""
    text = initial_info or ""
    cc = _chief_complaint(text)
    sex, age = _profile(text)
    cats = [c for c in detect_categories(text) if c in COMMON]
    cats += [c for c in extra_categories(cc) if c not in cats]
    infant = age is not None and age < 1
    danger = _round_robin([sorted(({"dx": dx, "tag": "위험", "source": "안전 프로토콜: " + CATEGORY_NAMES.get(c, c),
                                    "category": c}
                                   for dx in P.PROTOCOLS_BY_CATEGORY[c].cant_miss if _fits(dx, sex, age)),
                                  key=lambda r: infant and not _NEONATE_DX.search(r["dx"]))  # infants: neonatal first
                           for c in cats if c in P.PROTOCOLS_BY_CATEGORY])
    common = _round_robin([[{"dx": r.dx, "tag": "흔함", "source": _row_source(c, r),
                             "category": c} for r in COMMON[c][1] if _row_fits(r, sex, age)] for c in cats])
    kb_rows = _kb_rows(cc, sex, age, k=n_kb + 4)
    out: list[dict] = []

    def add(rows: list[dict], n: int) -> None:
        for r in rows:
            if n <= 0 or len(out) >= max_n:
                return
            if any(same_dx(r["dx"], o["dx"]) for o in out):
                continue
            out.append(r)
            n -= 1
    add(danger, n_danger)
    add(common, n_common)
    add(kb_rows, n_kb)
    add(_round_robin([danger, common]), max_n)
    return out[:max_n]


def render_for_prompt(ddx: list[dict], max_chars: int = MAX_RENDER) -> str:
    """One short Korean block for turn 1: the starting list grouped by tag."""
    if not ddx:
        return ""
    head = "[초기 감별 목록: 넓게 시작, 문진·검사로 좁힐 것]"
    parts = []
    for tag, label in (("위험", "반드시 배제"), ("흔함", "흔한 원인"), ("KB", "지식베이스 후보")):
        names = [d["dx"] for d in ddx if d.get("tag") == tag]
        if names:
            parts.append(f"{label}: " + ", ".join(names))
    text = head + "\n" + "\n".join(parts)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


# ------------------------------------------------------------------------------------------------ anchoring check
# "against" entries that are missing information, not a finding ("결과 없음", "아직 미확인", "언급 없음", "다소 비전형적")
_PENDING = re.compile(r"결과\s*(없|미|대기)|확인\s*(필요|안|못)|미시행|아직|필요함?$|부재|평가\s*(안|못|필요)|모름|제한|불명|"
                      r"대기|미확인|가능성|검사\s*없|언급|미상|알\s*수\s*없|명확하?지|명확히|다소|전형적이지|비전형|"
                      r"설명\s*(어렵|못|안)|어려움|부족|않음$")
# demographic tokens do not ground a finding
_DEMO = re.compile(r"^(\d+세|\d+대|남성|여성|남자|여자|고령|노인|젊은|가임기|환자|소아|성인)$")
_STATUS_WORDS = re.compile(r"양성|음성|상승|저하|증가|감소|이상|소견|정상|비정상|확장|결손|존재|없음|있음|진단|확인")
_STOP = {"흉부", "복부", "혈액", "소변", "영상", "검사", "촬영", "결과", "ct", "mri", "x선", "초음파", "급성", "만성",
         "원발성", "이차성", "증후군", "질환", "장애", "감염", "질병", "상세불명", "기타", "의한", "동반", "혹은", "또는",
         "medical", "test", "blood", "scan", "imaging", "disease", "syndrome", "acute", "chronic", "the", "and", "of"}
_TOKEN = re.compile(r"[0-9a-z가-힣\-]+")
_QUALIFIER = re.compile(r"(급성|만성|아급성|재발성|우측|좌측|양측|중증|경증)+")
_TARGET_TYPES = ("EXAM", "TEST")


def _tokens(text: str) -> set[str]:
    t = _STATUS_WORDS.sub(" ", (text or "").lower())
    out = set()
    for w in _TOKEN.findall(t):
        w = w.replace("-", "")
        if w in _STOP:
            continue
        if (re.search(r"[가-힣]", w) and len(w) >= 2) or len(w) >= 3:
            out.add(w)
    return out


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", (text or "").lower())


def _dx_tokens(dx: str) -> set[str]:
    core, inner = dx_keys(dx)
    toks = _tokens(re.sub(r"[(\[（][^)\]）]*[)\]）]", " ", dx)) | _tokens(inner)
    return toks | ({core} if len(core) >= 2 else set())


def _kb_profile(dx: str) -> dict | None:
    """KB profile only when it is really the same disease (kb.lookup is fuzzy: '원발성 담즙성 담관염' -> '담관염')."""
    try:
        from doctor_agent.knowledge import kb
        if not kb.available():
            return None
        p = kb.lookup(dx)
    except Exception:  # noqa: BLE001
        return None
    if not p:
        return None
    names = [n for n, _ in p.get("names_ko", [])] + [n for n, _ in p.get("names_en", [])]
    if any(same_dx(dx, n) for n in names if n):
        return p
    core = dx_keys(dx)[0]  # "급성 충수염" -> profile "충수염": only a generic qualifier may differ
    for n in names:
        k = dx_keys(n)[0] if n else ""
        if k and core.endswith(k) and _QUALIFIER.fullmatch(core[: -len(k)]):
            return p
    return None


def _kb_discriminators(a: str, b: str) -> dict | None:
    try:
        from doctor_agent.knowledge import kb
        return kb.discriminators(a, b) if kb.available() else None
    except Exception:  # noqa: BLE001
        return None


def _test_labels(profile: dict | None, disc: dict | None) -> list[str]:
    """Test names / decisive test results that confirm or refute the disease (KB tests + curated results)."""
    labels: list[str] = []
    if profile:
        labels += [x["ko"] for x in profile.get("findings_from_tests", []) if x.get("weight", 0) >= 2]
        labels += [x["ko"] for x in profile.get("tests", []) if x.get("en", "").lower() not in
                   {"medical history", "physical examination", "blood test", "medical diagnosis"}]
    if disc:
        labels += [x["ko"] for x in disc.get("test_findings_a_only", []) + disc.get("tests_a_only", [])]
    return list(dict.fromkeys(labels))


def _targeted(dx: str, turns, labels: list[str]) -> bool:
    """True when an EXAM/TEST action (content or stated reason) names the disease or one of its tests/results."""
    name_toks = _dx_tokens(dx)
    label_toks = set().union(*(_tokens(x) for x in labels)) if labels else set()
    for t in turns:
        typ = getattr(t.action.type, "value", t.action.type)
        if typ not in _TARGET_TYPES:
            continue
        text = f"{t.action.content} {getattr(t.action, 'reason', '')}"
        comp = _compact(text)
        if any(k in comp for k in name_toks):
            return True
        if label_toks and any(k in comp for k in label_toks):
            return True
    return False


def _live_sorted(ddx: list[dict]) -> list[dict]:
    return sorted((d for d in ddx or [] if str(d.get("dx", "")).strip() and d.get("status") != "배제"),
                  key=lambda d: -float(d.get("p", 0) or 0))


def _current(state) -> list[dict]:
    led = getattr(state, "ddx_ledger", None)
    if led is not None and getattr(led, "entries", None):
        return [x for x in led.as_list() if x.get("status") != "배제"]
    return _live_sorted(getattr(state, "ddx", []))


def _env_text(state) -> str:
    """What the environment actually said: the initial info and every response (compact form)."""
    parts = [getattr(state, "initial_info", "") or ""]
    parts += [getattr(t, "response", "") or "" for t in getattr(state, "turns", []) or []]
    return _compact(" ".join(parts))


def _grounded(items: list[str], env: str, findings) -> list[str]:
    """Evidence strings (deduplicated) with a non-demographic token that occurs in the environment's text and that
    the grounding check did not mark unverified (a finding with verified False and the same item)."""
    unverified = [f.item for f in findings if f.verified is False]
    out: list[str] = []
    for s in items:
        toks = {t for t in _tokens(s) if not _DEMO.match(t)}
        if not toks or not any(t in env for t in toks):
            continue
        if any(similarity(s, u) >= 0.6 for u in unverified):
            continue
        if not any(similarity(s, o) >= 0.5 or toks & _tokens(o) for o in out):
            out.append(s)
    return out


def _concrete_against(items: list[str], env: str) -> list[str]:
    """Distinct "against" entries that state a finding the environment reported (not missing information)."""
    return _grounded([a for a in items if a and not _PENDING.search(a)], env, [])


def anchoring_check(state, min_turns: int | None = None) -> dict | None:
    """Premature-closure check on the current top live DDx. None when it does not apply.

    Fires (after `min_turns` turns, default MIN_TURNS; top p >= MIN_P, and the same top in the last two snapshots) when any of:
      stable_untested: top unchanged since turn <= EARLY_TURN and no EXAM/TEST so far named it or one of its KB
                       tests / decisive results / discriminating results vs. the 2nd candidate;
      weak_support:    p >= WEAK_SUPPORT_P but <= 1 distinct supporting item grounded in the environment's text
                       (initial info + responses; not marked unverified by the grounding check);
      contradicted:    >= MIN_AGAINST distinct, grounded findings listed against it (not "결과 없음"/"확인 필요").
    Returns {"dx", "p", "reasons", "why_ko", "prompt_ko", "suggested_actions"}; the caller shows it at most once."""
    turns = list(getattr(state, "turns", []) or [])
    if len(turns) < (MIN_TURNS if min_turns is None else min_turns):
        return None
    live = _current(state)
    if not live:
        return None
    top = live[0]
    dx, p = str(top["dx"]), float(top.get("p", 0) or 0)
    if p < MIN_P:
        return None
    tops = [(_live_sorted(t.ddx) or [{"dx": ""}])[0]["dx"] for t in turns]
    if not tops or not tops[-1] or not same_dx(tops[-1], dx):
        return None
    since = len(tops)  # 1-based turn from which every snapshot top is this dx
    while since > 1 and tops[since - 2] and same_dx(tops[since - 2], dx):
        since -= 1
    second = live[1]["dx"] if len(live) > 1 else ""
    profile = _kb_profile(dx)
    disc = _kb_discriminators(dx, second) if second and profile else None
    labels = _test_labels(profile, disc)
    reasons: list[str] = []
    why: list[str] = []
    if since <= EARLY_TURN and not _targeted(dx, turns, labels):
        reasons.append("stable_untested")
        why.append(f"{since}번째 행동부터 1순위였지만 이를 확인·배제할 진찰/검사를 아직 하지 않음")
    findings = list(getattr(getattr(state, "findings", None), "items", []) or [])
    env = _env_text(state)
    good = _grounded(list(top.get("for") or []), env, findings)
    if p >= WEAK_SUPPORT_P and len(good) <= 1:
        reasons.append("weak_support")
        why.append(f"확률 {round(p * 100)}%인데 확인된 양성 소견 근거가 {len(good)}개뿐")
    against = _concrete_against(list(top.get("against") or []), env)
    if len(against) >= MIN_AGAINST:
        reasons.append("contradicted")
        why.append("반대 소견: " + ", ".join(against[-3:]))
    if not reasons:
        return None
    suggested = _suggest(dx, second, profile, disc)
    why_ko = "; ".join(why)
    prompt = (f"[반론 점검] 1순위 '{dx}'({round(p * 100)}%)에 일찍 고정됐을 수 있습니다: {why_ko}. "
              f"확정 전에 스스로 반론하세요. (1) 지금까지의 소견을 똑같이 설명할 수 있는 다른 진단 2개는? "
              f"(2) '{dx}'가 틀렸다면 어떤 검사·진찰 결과가 나와야 하나? 그 결과를 확인하는 행동을 다음 행동으로 고르세요.")
    if suggested:
        prompt += " 후보 행동: " + "; ".join(suggested)
    return {"dx": dx, "p": p, "reasons": reasons, "why_ko": why_ko, "prompt_ko": prompt,
            "suggested_actions": suggested}


def _suggest(dx: str, second: str, profile: dict | None, disc: dict | None) -> list[str]:
    out: list[str] = []
    if disc:
        b = [x["ko"] for x in disc.get("test_findings_b_only", []) + disc.get("tests_b_only", [])]
        a = [x["ko"] for x in disc.get("test_findings_a_only", []) + disc.get("tests_a_only", [])]
        if a:
            out.append(f"'{dx}' 확인: {a[0]}")
        if b:
            out.append(f"'{second}' 감별: {b[0]}")
    if profile and len(out) < 2:
        tf = [x["ko"] for x in profile.get("findings_from_tests", []) if x.get("weight", 0) >= 2]
        if tf and not any(tf[0] in s for s in out):
            out.append(f"'{dx}' 결정적 검사: {tf[0]}")
    if second and not any(second in s for s in out):
        out.append(f"2순위 '{second}'와 가르는 진찰/검사 1개")
    if not out:
        out.append(f"'{dx}'를 확진하거나 배제할 결정적 검사 1개")
    return out[:3]
