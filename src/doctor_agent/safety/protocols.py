"""Chief-complaint safety protocols. Content is owned by clinical-strategist.

Per category: can't-miss diagnoses and the minimum safe checks, each with the guideline it comes from.
Check.keywords are lowercase substrings for matching the doctor's action text (usable as a scorer
`must_check`: "|".join(keywords)). Check.triggers, when non-empty, make a check conditional: it applies
only if the case text/transcript contains one of them.

Verification (2026-09-25)
- Citation.verified: bibliographic data checked against PubMed E-utilities.
- Check.verification: "primary" = recommendation read in the guideline text; "secondary" = confirmed via
  secondary summaries only (guideline full text paywalled); "unverified" = from reviewer knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass

from doctor_agent.knowledge.clinical_rules import (
    C_ALVARADO,
    CATEGORY_NAMES,
    Citation,
    detect_categories,
)

# --------------------------------------------------------------------------------------------
# Guideline citations (bibliographic data verified against PubMed)
# --------------------------------------------------------------------------------------------

G_CHEST_PAIN = Citation(
    "Gulati M, Levy PD, Mukherjee D, et al.",
    "2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain",
    "Circulation", 2021, "144(22):e368-e454", doi="10.1161/CIR.0000000000001029", pmid="34709879",
    verified=True, short_author="AHA/ACC 흉통 지침",
)
G_AORTA = Citation(
    "Isselbacher EM, Preventza O, Hamilton Black J 3rd, et al.",
    "2022 ACC/AHA Guideline for the Diagnosis and Management of Aortic Disease",
    "Circulation", 2022, "146(24):e334-e482", doi="10.1161/CIR.0000000000001106", pmid="36322642",
    verified=True, short_author="ACC/AHA 대동맥 지침",
)
G_PE = Citation(
    "Konstantinides SV, Meyer G, Becattini C, et al.",
    "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in "
    "collaboration with the European Respiratory Society (ERS)",
    "Eur Heart J", 2020, "41(4):543-603", doi="10.1093/eurheartj/ehz405", pmid="31504429",
    verified=True, short_author="ESC 폐색전증 지침",
)
G_HF = Citation(
    "Heidenreich PA, Bozkurt B, Aguilar D, et al.",
    "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure",
    "Circulation", 2022, "145(18):e895-e1032", doi="10.1161/CIR.0000000000001063", pmid="35363499",
    verified=True, short_author="AHA/ACC/HFSA 심부전 지침",
)
G_PLEURAL = Citation(
    "Roberts ME, Rahman NM, Maskell NA, et al.",
    "British Thoracic Society Guideline for pleural disease",
    "Thorax", 2023, "78(Suppl 3):s1-s42", doi="10.1136/thorax-2022-219784", pmid="37433578",
    verified=True, short_author="BTS 흉막질환 지침",
)
G_ANAPHYLAXIS = Citation(
    "Shaker MS, Wallace DV, Golden DBK, et al.",
    "Anaphylaxis-a 2020 practice parameter update, systematic review, and Grading of Recommendations, "
    "Assessment, Development and Evaluation (GRADE) analysis",
    "J Allergy Clin Immunol", 2020, "145(4):1082-1123", doi="10.1016/j.jaci.2020.01.017", pmid="32001253",
    verified=True, short_author="아나필락시스 진료지침",
)
G_HEADACHE = Citation(
    "Godwin SA, Cherkas DS, Panagos PD, et al.",
    "Clinical Policy: Critical Issues in the Evaluation and Management of Adult Patients Presenting to the "
    "Emergency Department With Acute Headache",
    "Ann Emerg Med", 2019, "74(4):e41-e74", doi="10.1016/j.annemergmed.2019.07.009", pmid="31543134",
    verified=True, short_author="ACEP 두통 정책",
)
G_MENINGITIS = Citation(
    "Tunkel AR, Hartman BJ, Kaplan SL, et al.",
    "Practice guidelines for the management of bacterial meningitis",
    "Clin Infect Dis", 2004, "39(9):1267-1284", doi="10.1086/425368", pmid="15494903",
    verified=True, short_author="IDSA 세균성 수막염 지침",
)
G_STROKE = Citation(
    "Powers WJ, Rabinstein AA, Ackerson T, et al.",
    "Guidelines for the Early Management of Patients With Acute Ischemic Stroke: 2019 Update to the 2018 "
    "Guidelines for the Early Management of Acute Ischemic Stroke",
    "Stroke", 2019, "50(12):e344-e418", doi="10.1161/STR.0000000000000211", pmid="31662037",
    verified=True, short_author="AHA/ASA 뇌졸중 지침",
)
G_SEPSIS = Citation(
    "Evans L, Rhodes A, Alhazzani W, et al.",
    "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021",
    "Crit Care Med", 2021, "49(11):e1063-e1143", doi="10.1097/CCM.0000000000005337", pmid="34605781",
    verified=True, short_author="SSC 2021",
)
G_NEUTROPENIA = Citation(
    "Freifeld AG, Bow EJ, Sepkowitz KA, et al.",
    "Clinical practice guideline for the use of antimicrobial agents in neutropenic patients with cancer: "
    "2010 update by the Infectious Diseases Society of America",
    "Clin Infect Dis", 2011, "52(4):e56-e93", doi="10.1093/cid/cir073", pmid="21258094",
    verified=True, short_author="IDSA 호중구감소성 발열 지침",
)
G_EARLY_PREGNANCY = Citation(
    "Hahn SA, Promes SB, Brown MD; ACEP Clinical Policies Subcommittee on Early Pregnancy.",
    "Clinical Policy: Critical Issues in the Initial Evaluation and Management of Patients Presenting to the "
    "Emergency Department in Early Pregnancy",
    "Ann Emerg Med", 2017, "69(2):241-250.e20", doi="10.1016/j.annemergmed.2016.11.002", pmid="28126120",
    verified=True, short_author="ACEP 초기 임신 정책",
)
G_ECTOPIC = Citation(
    "American College of Obstetricians and Gynecologists' Committee on Practice Bulletins-Gynecology.",
    "ACOG Practice Bulletin No. 193: Tubal Ectopic Pregnancy",
    "Obstet Gynecol", 2018, "131(3):e91-e103", doi="10.1097/AOG.0000000000002560", pmid="29470343",
    verified=True, short_author="ACOG 자궁외 임신 지침",
)
G_AAA = Citation(
    "Chaikof EL, Dalman RL, Eskandari MK, et al.",
    "The Society for Vascular Surgery practice guidelines on the care of patients with an abdominal aortic "
    "aneurysm",
    "J Vasc Surg", 2018, "67(1):2-77.e2", doi="10.1016/j.jvs.2017.10.044", pmid="29268916",
    verified=True, short_author="SVS 복부 대동맥류 지침",
)
G_AMI = Citation(
    "Bala M, Catena F, Kashuk J, et al.",
    "Acute mesenteric ischemia: updated guidelines of the World Society of Emergency Surgery",
    "World J Emerg Surg", 2022, "17(1):54", doi="10.1186/s13017-022-00443-x", pmid="36261857",
    verified=True, short_author="WSES 장간막 허혈 지침",
)

GUIDELINES: tuple[Citation, ...] = (
    G_CHEST_PAIN, G_AORTA, G_PE, G_HF, G_PLEURAL, G_ANAPHYLAXIS, G_HEADACHE, G_MENINGITIS, G_STROKE,
    G_SEPSIS, G_NEUTROPENIA, G_EARLY_PREGNANCY, G_ECTOPIC, G_AAA, G_AMI,
)

# --------------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    id: str
    name: str  # Korean
    kind: str  # "ask" | "exam" | "test" | "treatment"
    keywords: tuple[str, ...]  # lowercase substrings that mark the check as done in action text
    citation: Citation
    verification: str  # "primary" | "secondary" | "unverified"
    triggers: tuple[str, ...] = ()  # conditional check: applies only if the case text contains one
    when: str = ""  # Korean condition text for prompts
    note: str = ""

    def applies(self, text: str) -> bool:
        if not self.triggers:
            return True
        t = (text or "").lower()
        return any(k in t for k in self.triggers)

    def done_in(self, action_text: str) -> bool:
        t = (action_text or "").lower()
        return any(k in t for k in self.keywords)

    @property
    def must_check(self) -> str:
        """Scorer format: '|'-separated alternatives (eval/scorer.py)."""
        return "|".join(self.keywords)


@dataclass(frozen=True)
class Protocol:
    category: str
    name_ko: str
    cant_miss: tuple[str, ...]
    checks: tuple[Check, ...]


# --------------------------------------------------------------------------------------------
# Shared keyword sets
# --------------------------------------------------------------------------------------------

_KW_VITALS = ("활력", "혈압", "맥박", "심박", "호흡수", "체온", "산소포화", "spo2", "vital", "blood pressure",
              "heart rate", "pulse", "respiratory rate", "temperature", "oxygen saturation")
_KW_ECG = ("심전도", "ecg", "ekg", "electrocardiogram")
_KW_TROPONIN = ("트로포닌", "troponin", "심근효소", "심장효소", "심근 효소", "ck-mb")
_KW_CXR = ("흉부 x", "흉부x", "흉부 엑스", "가슴 x", "가슴 엑스", "흉부 방사선", "흉부 사진", "chest x", "cxr",
           "chest radiograph")
_KW_PE_TEST = ("d-dimer", "d dimer", "디다이머", "d-이합체", "ct 폐동맥", "폐동맥 ct", "폐동맥 조영", "ctpa",
               "ct pulmonary", "pulmonary angiogra")
_KW_AORTA = ("대동맥 ct", "대동맥 조영", "ct 혈관조영", "ct 혈관 조영", "cta", "ct angiogra", "양팔 혈압",
             "양측 팔 혈압", "양쪽 팔 혈압", "경식도 초음파", "transesophageal")
_KW_NEURO_EXAM = ("신경학적", "신경 검진", "신경학 검사", "신경 검사", "동공", "경부 강직", "목 강직", "뇌막 자극",
                  "neuro exam", "neurologic", "pupil", "nuchal", "kernig", "brudzinski", "nihss")
_KW_BRAIN_CT = ("뇌 ct", "머리 ct", "두부 ct", "두경부 ct", "brain ct", "head ct", "ct brain", "ct head",
                "비조영 ct", "noncontrast ct", "non-contrast ct")
_KW_BRAIN_IMAGING = _KW_BRAIN_CT + ("뇌 mri", "머리 mri", "brain mri", "mri brain", "확산강조", "dwi")
_KW_LP = ("요추천자", "요추 천자", "뇌척수액", "척수액", "lumbar puncture", "spinal tap", "csf")
_KW_BLOOD_CULTURE = ("혈액배양", "혈액 배양", "blood culture")
_KW_GLUCOSE = ("혈당", "포도당", "glucose", "bst", "blood sugar")
_KW_PREGNANCY = ("임신 검사", "임신검사", "임신 반응", "임신반응", "hcg", "pregnancy test", "소변 임신")

_TRIG_THUNDERCLAP = ("벼락", "갑자기", "갑작스", "순간", "최악의", "인생 최악", "태어나서", "처음 겪", "힘을 주다",
                     "운동 중", "성관계", "thunderclap", "sudden", "worst headache", "worst of")
_TRIG_MENINGITIS = ("열", "발열", "오한", "목이 뻣뻣", "목 뻣뻣", "경부 강직", "의식", "혼돈", "헛소리", "fever",
                    "stiff neck", "neck stiffness", "confus")
_TRIG_HEADACHE_MENINGISM = ("두통", "머리가 아", "목이 뻣뻣", "목 뻣뻣", "경부 강직", "의식", "혼돈", "헛소리",
                            "headache", "stiff neck", "neck stiffness", "confus", "altered mental")
_TRIG_AORTA = ("찢어지", "찢기는", "뜯기는", "등으로", "등까지", "등 통증", "양팔", "마르판", "tearing", "ripping",
               "radiat", "back pain", "marfan")
_TRIG_PE = ("객혈", "다리가 붓", "다리 부종", "종아리", "수술", "부동", "장거리", "비행", "피임약", "호르몬", "혈전",
            "hemoptysis", "leg swelling", "calf", "surgery", "immobil", "long flight", "contracepti", "thrombo")
_TRIG_ANAPHYLAXIS = ("두드러기", "입술이 붓", "혀가 붓", "얼굴이 붓", "알레르기", "벌에", "땅콩", "새우", "먹고 나서",
                     "주사 맞고", "hives", "urticaria", "allerg", "angioedema", "bee sting")
_TRIG_NEUTROPENIA = ("항암", "암 치료", "화학요법", "면역저하", "면역 저하", "골수", "chemotherapy", "neutropen",
                     "immunocompromis", "transplant", "이식")
_TRIG_FEMALE = ("여성", "여자", "female", "woman", "임신", "생리", "월경", "질출혈", "질 출혈", "pregnan", "menstru",
                "vaginal bleeding")
_TRIG_AAA = ("등 통증", "허리 통증", "옆구리", "박동", "실신", "기절", "쇼크", "저혈압", "back pain", "flank",
             "pulsatile", "syncope", "shock", "hypotens")
_TRIG_AMI = ("심방세동", "부정맥", "통증에 비해", "진찰 소견에 비해", "혈전", "atrial fibrillation", "afib",
             "out of proportion")


def _vitals(citation: Citation, when: str = "도착 즉시", note: str = "") -> Check:
    return Check("vitals", "활력징후(혈압, 맥박, 호흡수, 체온, 산소포화도)", "exam", _KW_VITALS, citation,
                 "unverified", when=when, note=note or "Basic initial assessment; not a specific numbered "
                 "recommendation checked in the guideline text.")


# --------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------

PROTOCOLS: tuple[Protocol, ...] = (
    Protocol(
        category="chest_pain", name_ko=CATEGORY_NAMES["chest_pain"],
        cant_miss=("급성 관상동맥 증후군", "대동맥 박리", "폐색전증", "긴장성 기흉"),
        checks=(
            _vitals(G_CHEST_PAIN),
            Check("ecg", "12유도 심전도(도착 10분 이내)", "test", _KW_ECG, G_CHEST_PAIN, "secondary",
                  when="모든 급성 흉통",
                  note="Class 1: acquire and review ECG for STEMI within 10 min of arrival. Full text paywalled "
                  "(HTTP 403); confirmed via ACC summaries."),
            Check("troponin", "심장 트로포닌(가능하면 고감도)", "test", _KW_TROPONIN, G_CHEST_PAIN, "secondary",
                  when="급성 관상동맥 증후군 의심",
                  note="hs-cTn is the preferred biomarker. Confirmed via ACC summaries."),
            Check("cxr", "흉부 X선", "test", _KW_CXR, G_CHEST_PAIN, "unverified",
                  when="폐·흉막·대동맥 등 다른 원인 평가",
                  note="Guideline suggests CXR to evaluate alternative cardiac/pulmonary/thoracic causes."),
            Check("aorta_imaging", "양팔 혈압 비교 + 대동맥 영상검사(CT 혈관조영 등)", "test", _KW_AORTA, G_AORTA,
                  "unverified", triggers=_TRIG_AORTA, when="찢어지는 통증, 등으로 뻗는 통증, 양팔 혈압 차이 등",
                  note="Use with ADD-RS (knowledge/clinical_rules.py)."),
            Check("pe_workup", "폐색전증 사전확률 평가 후 D-dimer 또는 CT 폐동맥조영", "test", _KW_PE_TEST, G_PE,
                  "unverified", triggers=_TRIG_PE, when="객혈, 한쪽 다리 부종, 최근 수술·부동 등 위험인자",
                  note="ESC 2019: clinical probability (Wells/Geneva), D-dimer if not high probability, CTPA."),
        ),
    ),
    Protocol(
        category="dyspnea", name_ko=CATEGORY_NAMES["dyspnea"],
        cant_miss=("폐색전증", "급성 심부전", "아나필락시스", "긴장성 기흉"),
        checks=(
            _vitals(G_PE, note="Hemodynamic status (shock/hypotension) defines high-risk PE in ESC 2019; "
                    "SpO2 is basic assessment. Not checked in the guideline text."),
            Check("ecg", "12유도 심전도", "test", _KW_ECG, G_HF, "unverified", when="모든 급성 호흡곤란",
                  note="HF guideline recommends ECG in initial evaluation of suspected HF."),
            Check("cxr", "흉부 X선", "test", _KW_CXR, G_PLEURAL, "unverified", when="기흉·폐렴·폐부종 평가",
                  note="Tension pneumothorax is a clinical diagnosis; do not delay decompression for imaging."),
            Check("natriuretic_peptide", "BNP 또는 NT-proBNP", "test", ("bnp", "nt-probnp", "나트륨이뇨"), G_HF,
                  "unverified", when="심부전 의심",
                  note="Class 1: natriuretic peptides to support/exclude HF in patients presenting with dyspnea."),
            Check("pe_workup", "폐색전증 사전확률 평가 후 D-dimer 또는 CT 폐동맥조영", "test", _KW_PE_TEST, G_PE,
                  "unverified", triggers=_TRIG_PE, when="객혈, 한쪽 다리 부종, 최근 수술·부동 등 위험인자"),
            Check("epinephrine", "아나필락시스 의심 시 즉시 에피네프린 근육주사", "treatment",
                  ("에피네프린", "epinephrine", "아드레날린", "adrenaline", "epipen"), G_ANAPHYLAXIS, "unverified",
                  triggers=_TRIG_ANAPHYLAXIS, when="두드러기·입술/혀 부종·알레르겐 노출 후 호흡곤란"),
        ),
    ),
    Protocol(
        category="headache", name_ko=CATEGORY_NAMES["headache"],
        cant_miss=("지주막하 출혈", "뇌수막염", "뇌종양/두개내압 상승"),
        checks=(
            _vitals(G_HEADACHE),
            Check("neuro_exam", "신경학적 진찰(의식, 동공, 국소 결손, 경부 강직)", "exam", _KW_NEURO_EXAM, G_HEADACHE,
                  "unverified", when="모든 급성 두통",
                  note="The ACEP recommendations are conditioned on a normal neurologic exam, so it must be done."),
            Check("brain_ct", "비조영 뇌 CT(발병 6시간 이내 음성이면 지주막하 출혈 배제에 유용)", "test",
                  _KW_BRAIN_CT, G_HEADACHE, "secondary", triggers=_TRIG_THUNDERCLAP,
                  when="벼락두통·갑자기 시작해 1시간 이내 최고조·인생 최악의 두통",
                  note="Abstract confirms critical question 3 (normal NCCT within 6 h). The Level B answer "
                  "(negative NCCT within 6 h in neurologically normal patients rules out SAH) is from "
                  "reviewer knowledge; use with the Ottawa SAH Rule."),
            Check("meningitis_workup", "혈액배양 + 요추천자(CT 선행 적응증 확인)", "test",
                  _KW_BLOOD_CULTURE + _KW_LP, G_MENINGITIS, "secondary", triggers=_TRIG_MENINGITIS,
                  when="두통과 발열·목 강직·의식 변화 동반",
                  note="IDSA 2004: blood cultures + LP; CT before LP if immunocompromised, CNS disease history, "
                  "new seizure, papilledema, altered consciousness, or focal deficit; do not delay antibiotics. "
                  "Confirmed via secondary summaries (AAFP 2005 review)."),
        ),
    ),
    Protocol(
        category="neuro", name_ko=CATEGORY_NAMES["neuro"],
        cant_miss=("급성 허혈성 뇌졸중", "뇌출혈", "저혈당", "일과성 허혈 발작 후 조기 뇌졸중"),
        checks=(
            _vitals(G_STROKE),
            Check("glucose", "혈당 측정", "test", _KW_GLUCOSE, G_STROKE, "unverified",
                  when="모든 급성 신경학적 결손·의식 변화",
                  note="Blood glucose is the only lab required before IV alteplase; hypoglycemia mimics stroke."),
            Check("onset_time", "증상 시작 시각(마지막으로 정상이었던 시각) 확인", "ask",
                  ("마지막으로 정상", "마지막 정상", "증상 시작", "발병 시각", "언제부터", "몇 시", "last known well",
                   "onset time", "when did"),
                  G_STROKE, "unverified", when="급성 신경학적 결손", note="Determines reperfusion eligibility."),
            Check("neuro_exam", "신경학적 진찰(국소 결손, 의식 수준)", "exam", _KW_NEURO_EXAM, G_STROKE, "unverified",
                  when="모든 급성 신경 증상"),
            Check("brain_imaging", "비조영 뇌 CT 또는 뇌 MRI", "test", _KW_BRAIN_IMAGING, G_STROKE, "unverified",
                  when="급성 뇌졸중 의심", note="Emergent brain imaging before any reperfusion therapy."),
        ),
    ),
    Protocol(
        category="fever", name_ko=CATEGORY_NAMES["fever"],
        cant_miss=("패혈증", "뇌수막염", "호중구감소성 발열"),
        checks=(
            _vitals(G_SEPSIS, note="SSC 2021 recommends against qSOFA alone vs SIRS/NEWS/MEWS for screening; "
                    "vital signs feed those scores."),
            Check("blood_culture", "항생제 투여 전 혈액배양", "test", _KW_BLOOD_CULTURE, G_SEPSIS, "primary",
                  when="패혈증 의심",
                  note="SSC 2021: obtain cultures including blood before antimicrobials if it does not delay "
                  "treatment (read in PMC8486643, co-published version)."),
            Check("lactate", "혈중 젖산", "test", ("젖산", "락테이트", "lactate", "lactic"), G_SEPSIS, "primary",
                  when="패혈증 의심", note="SSC 2021: suggest measuring blood lactate (weak recommendation)."),
            Check("cbc_neutropenia", "일반혈액검사(호중구 수)", "test",
                  ("일반혈액", "혈구", "cbc", "백혈구", "호중구", "complete blood count", "neutrophil count"), G_NEUTROPENIA,
                  "unverified", triggers=_TRIG_NEUTROPENIA, when="항암치료 중·면역저하 환자의 발열"),
            Check("meningitis_workup", "혈액배양 + 요추천자(CT 선행 적응증 확인)", "test",
                  _KW_BLOOD_CULTURE + _KW_LP, G_MENINGITIS, "secondary", triggers=_TRIG_HEADACHE_MENINGISM,
                  when="발열과 두통·목 강직·의식 변화 동반"),
        ),
    ),
    Protocol(
        category="abdominal_pain", name_ko=CATEGORY_NAMES["abdominal_pain"],
        cant_miss=("장 천공", "자궁외 임신", "복부 대동맥류 파열", "장간막 허혈"),
        checks=(
            _vitals(G_AAA),
            Check("abdominal_exam", "복부 진찰(압통 위치, 반발 압통, 근성 방어)", "exam",
                  ("복부 진찰", "복부 촉진", "배 진찰", "압통", "반발", "근성 방어", "복막 자극", "abdominal exam",
                   "palpat", "rebound", "guarding", "tenderness"),
                  C_ALVARADO, "unverified", when="모든 복통",
                  note="Tenderness and rebound are Alvarado components; cited as the source for exam findings, "
                  "not as a guideline."),
            Check("pregnancy_test", "임신 검사(β-hCG)", "test", _KW_PREGNANCY, G_ECTOPIC, "unverified",
                  triggers=_TRIG_FEMALE, when="가임기 여성의 복통",
                  note="ACOG PB 193 / ACEP 2017 cover ectopic pregnancy evaluation (hCG + transvaginal US). "
                  "Recommendation text not re-read."),
            Check("pelvic_us", "임신 양성이면 골반(질식) 초음파", "test",
                  ("질식 초음파", "골반 초음파", "경질 초음파", "transvaginal", "pelvic ultrasound", "pelvic us"),
                  G_EARLY_PREGNANCY, "unverified", triggers=("임신", "pregnan", "hcg 양성"),
                  when="임신 확인된 복통·질출혈"),
            Check("aaa_imaging", "복부 대동맥 초음파 또는 CT", "test",
                  ("복부 초음파", "복부 ct", "대동맥 초음파", "abdominal ultrasound", "abdominal ct", "ct abdomen",
                   "aortic ultrasound"),
                  G_AAA, "unverified", triggers=_TRIG_AAA, when="고령·등/옆구리 통증·박동성 종괴·실신·저혈압"),
            Check("mesenteric_cta", "장간막 허혈 의심 시 지체 없이 CT 혈관조영", "test",
                  ("ct 혈관조영", "ct 혈관 조영", "cta", "ct angiogra", "복부 혈관조영", "mesenteric"), G_AMI,
                  "primary", triggers=_TRIG_AMI, when="심방세동, 진찰 소견에 비해 심한 통증",
                  note="WSES 2022 recommendation 5 (1A): CTA without delay in any patient with suspected AMI; "
                  "lactate/D-dimer may assist but cannot exclude (rec. 4). Read in PMC9580452."),
        ),
    ),
)

PROTOCOLS_BY_CATEGORY: dict[str, Protocol] = {p.category: p for p in PROTOCOLS}


# --------------------------------------------------------------------------------------------
# Lookup / rendering
# --------------------------------------------------------------------------------------------


def protocols_for(text: str) -> list[Protocol]:
    return [PROTOCOLS_BY_CATEGORY[c] for c in detect_categories(text) if c in PROTOCOLS_BY_CATEGORY]


def cant_miss_for(text: str) -> list[str]:
    """Can't-miss diagnoses for the text, deduplicated in order."""
    out: list[str] = []
    for p in protocols_for(text):
        out += [dx for dx in p.cant_miss if dx not in out]
    return out


def must_checks_for(text: str, context: str = "") -> list[Check]:
    """Applicable minimum checks, deduplicated by check id.

    Categories come from `text` only (the chief complaint): matching whole conversations over-triggers, e.g. a
    pertinent negative like "숨은 안 차요" would add the dyspnea protocol. Conditional checks are triggered by
    `text` + `context` (facts learned later, e.g. pregnancy or a thunderclap onset).
    """
    trigger_text = f"{text} {context}"
    out: dict[str, Check] = {}
    for p in protocols_for(text):
        for c in p.checks:
            if c.id not in out and c.applies(trigger_text):
                out[c.id] = c
    return list(out.values())


def pending_checks(text: str, actions: list[str], context: str = "") -> list[Check]:
    """Applicable checks not yet matched by any of the doctor's action texts."""
    done = " ".join(actions).lower()
    return [c for c in must_checks_for(text, context) if not c.done_in(done)]


def render_for_prompt(text: str, actions: list[str] | None = None) -> str:
    """Compact Korean text: can't-miss diagnoses + (pending) minimum checks with short citations."""
    cant = cant_miss_for(text)
    checks = pending_checks(text, actions or []) if actions is not None else must_checks_for(text)
    if not cant and not checks:
        return ""
    lines = []
    if cant:
        lines.append("반드시 배제할 위험 질환: " + ", ".join(cant))
    if checks:
        lines.append("최소 안전 확인" + ("(아직 안 함)" if actions is not None else "") + ":")
        lines += [f"- {c.name} [{c.when}] ({c.citation.short})" if c.when else f"- {c.name} ({c.citation.short})"
                  for c in checks]
    return "\n".join(lines)
