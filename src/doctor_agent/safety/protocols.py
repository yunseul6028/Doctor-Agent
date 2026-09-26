"""Chief-complaint safety protocols. Content is owned by clinical-strategist.

Per category: can't-miss diagnoses and the minimum safe checks, each with the guideline it comes from.
Check.keywords are lowercase substrings for matching the doctor's action text (usable as a scorer
`must_check`: "|".join(keywords)). Check.triggers, when non-empty, make a check conditional: it applies
only if the case text/transcript contains one of them outside a negated clause ("…은 없어요", "…아니에요").
Check.acute_only drops a check for long-standing chief complaints (duration of the chief complaint only), and
Check.predicate replaces triggers with a function (e.g. "sepsis": fever with acute systemic-illness signals).

Verification (2026-09-25)
- Citation.verified: bibliographic data checked against PubMed E-utilities.
- Check.verification: "primary" = recommendation read in the guideline text; "secondary" = confirmed via
  secondary summaries only (guideline full text paywalled); "unverified" = from reviewer knowledge.
- 2026-09-26: 13 categories added (syncope ... psychiatric) with 19 more guidelines; same verification scale,
  what was read is stated in each Check.note. Check.min_duration / Check.min_age limit a check to the cited
  guideline's population (chronic presentations, adults).
- 2026-09-27: false-trigger fixes from a per-case audit of data/cases_aug (see tests) and 6 categories added
  (chronic urticaria, hearing loss, adult neck mass, bleeding tendency, chronic limb weakness, infant bilious
  vomiting) with 8 more citations checked against PubMed E-utilities on 2026-09-27.
"""
from __future__ import annotations

from dataclasses import dataclass

import re

from doctor_agent.knowledge.clinical_rules import (
    C_ALVARADO,
    CATEGORY_NAMES,
    Citation,
    age_days,
    age_years,
    contains_affirmed,
    detect_categories,
    duration_level,
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

G_ENDOCARDITIS = Citation(
    "Delgado V, Ajmone Marsan N, de Waha S, et al.",
    "2023 ESC Guidelines for the management of endocarditis",
    "Eur Heart J", 2023, "44(39):3948-4042", doi="10.1093/eurheartj/ehad193", pmid="37622656",
    verified=True, short_author="ESC 심내막염 지침",
)

# --- 2026-09-26 additions (bibliographic data checked against PubMed E-utilities on 2026-09-26) ---
G_SYNCOPE = Citation(
    "Shen WK, Sheldon RS, Benditt DG, et al.",
    "2017 ACC/AHA/HRS Guideline for the Evaluation and Management of Patients With Syncope",
    "Circulation", 2017, "136(5):e60-e122", doi="10.1161/CIR.0000000000000499", pmid="28280231",
    verified=True, short_author="ACC/AHA/HRS 실신 지침",
)
G_PALPITATIONS = Citation(
    "Raviele A, Giada F, Bergfeldt L, et al.",
    "Management of patients with palpitations: a position paper from the European Heart Rhythm Association",
    "Europace", 2011, "13(7):920-934", doi="10.1093/europace/eur130", pmid="21697315",
    verified=True, short_author="EHRA 두근거림 권고",
)
G_HEMOPTYSIS = Citation(
    "Expert Panel on Thoracic Imaging; Olsen KM, Manouchehr-Pour S, et al.",
    "ACR Appropriateness Criteria Hemoptysis",
    "J Am Coll Radiol", 2020, "17(5S):S148-S159", doi="10.1016/j.jacr.2020.01.043", pmid="32370959",
    verified=True, short_author="ACR 객혈 적정성 기준",
)
G_COUGH = Citation(
    "Irwin RS, Baumann MH, Bolser DC, et al.",
    "Diagnosis and management of cough executive summary: ACCP evidence-based clinical practice guidelines",
    "Chest", 2006, "129(1 Suppl):1S-23S", doi="10.1378/chest.129.1_suppl.1S", pmid="16428686",
    verified=True, short_author="ACCP 기침 지침",
)
G_LIVER = Citation(
    "Kwo PY, Cohen SM, Lim JK.",
    "ACG Clinical Guideline: Evaluation of Abnormal Liver Chemistries",
    "Am J Gastroenterol", 2017, "112(1):18-35", doi="10.1038/ajg.2016.517", pmid="27995906",
    verified=True, short_author="ACG 간기능 이상 지침",
)
G_JAUNDICE_IMAGING = Citation(
    "Expert Panel on Gastrointestinal Imaging; Hindman NM, Arif-Tiwari H, et al.",
    "ACR Appropriateness Criteria Jaundice",
    "J Am Coll Radiol", 2019, "16(5S):S126-S140", doi="10.1016/j.jacr.2019.02.012", pmid="31054739",
    verified=True, short_author="ACR 황달 적정성 기준",
)
G_NEONATAL_JAUNDICE = Citation(
    "Kemper AR, Newman TB, Slaughter JL, et al.",
    "Clinical Practice Guideline Revision: Management of Hyperbilirubinemia in the Newborn Infant 35 or More "
    "Weeks of Gestation",
    "Pediatrics", 2022, "150(3):e2022058859", doi="10.1542/peds.2022-058859", pmid="35927462",
    verified=True, short_author="AAP 신생아 황달 지침",
)
G_HOT_JOINT = Citation(
    "Coakley G, Mathews C, Field M, et al.",
    "BSR & BHPR, BOA, RCGP and BSAC guidelines for management of the hot swollen joint in adults",
    "Rheumatology (Oxford)", 2006, "45(8):1039-1041", doi="10.1093/rheumatology/kel163a", pmid="16829534",
    verified=True, short_author="BSR 급성 관절염(hot swollen joint) 지침",
)
G_GOUT_DX = Citation(
    "Richette P, Doherty M, Pascual E, et al.",
    "2018 updated European League Against Rheumatism evidence-based recommendations for the diagnosis of gout",
    "Ann Rheum Dis", 2020, "79(1):31-38", doi="10.1136/annrheumdis-2019-215315", pmid="31167758",
    verified=True, short_author="EULAR 통풍 진단 권고",
)
G_LOW_BACK_PAIN = Citation(
    "Chou R, Qaseem A, Snow V, et al.",
    "Diagnosis and treatment of low back pain: a joint clinical practice guideline from the American College of "
    "Physicians and the American Pain Society",
    "Ann Intern Med", 2007, "147(7):478-491", doi="10.7326/0003-4819-147-7-200710020-00006", pmid="17909209",
    verified=True, short_author="ACP/APS 요통 지침",
)
G_SJS_TEN = Citation(
    "Creamer D, Walsh SA, Dziewulski P, et al.",
    "U.K. guidelines for the management of Stevens-Johnson syndrome/toxic epidermal necrolysis in adults 2016",
    "Br J Dermatol", 2016, "174(6):1194-1227", doi="10.1111/bjd.14530", pmid="27317286",
    verified=True, short_author="BAD SJS/TEN 지침",
)
G_PRURITUS = Citation(
    "Weisshaar E, Szepietowski JC, Dalgard FJ, et al.",
    "European S2k Guideline on Chronic Pruritus",
    "Acta Derm Venereol", 2019, "99(5):469-506", doi="10.2340/00015555-3164", pmid="30931482",
    verified=True, short_author="유럽 만성 소양증 지침",
)
G_VTE_DX = Citation(
    "Lim W, Le Gal G, Bates SM, et al.",
    "American Society of Hematology 2018 guidelines for management of venous thromboembolism: diagnosis of "
    "venous thromboembolism",
    "Blood Adv", 2018, "2(22):3226-3256", doi="10.1182/bloodadvances.2018024828", pmid="30482764",
    verified=True, short_author="ASH 정맥혈전색전증 진단 지침",
)
G_GLOMERULAR = Citation(
    "Rovin BH, Adler SG, Barratt J, et al.",
    "Executive summary of the KDIGO 2021 Guideline for the Management of Glomerular Diseases",
    "Kidney Int", 2021, "100(4):753-779", doi="10.1016/j.kint.2021.05.015", pmid="34556300",
    verified=True, short_author="KDIGO 사구체질환 지침",
)
G_AMENORRHEA = Citation(
    "Practice Committee of the American Society for Reproductive Medicine.",
    "Current evaluation of amenorrhea: a committee opinion",
    "Fertil Steril", 2024, "122(1):52-61", doi="10.1016/j.fertnstert.2024.02.001", pmid="38456861",
    verified=True, short_author="ASRM 무월경 평가",
)
G_PMB = Citation(
    "American College of Obstetricians and Gynecologists.",
    "ACOG Committee Opinion No. 734: The Role of Transvaginal Ultrasonography in Evaluating the Endometrium of "
    "Women With Postmenopausal Bleeding",
    "Obstet Gynecol", 2018, "131(5):e124-e129", doi="10.1097/AOG.0000000000002631", pmid="29683909",
    verified=True, short_author="ACOG 폐경 후 출혈",
)
G_MECFS = Citation(
    "National Institute for Health and Care Excellence.",
    "Myalgic encephalomyelitis (or encephalopathy)/chronic fatigue syndrome: diagnosis and management "
    "(NICE guideline NG206)",
    "London: NICE", 2021, "NG206", pmid="35438859",
    verified=True, short_author="NICE NG206",
)
G_DEMENTIA = Citation(
    "Knopman DS, DeKosky ST, Cummings JL, et al.",
    "Practice parameter: diagnosis of dementia (an evidence-based review). Report of the Quality Standards "
    "Subcommittee of the American Academy of Neurology",
    "Neurology", 2001, "56(9):1143-1153", doi="10.1212/wnl.56.9.1143", pmid="11342678",
    verified=True, short_author="AAN 치매 진단 지침",
)
G_PSYCH_EVAL = Citation(
    "Silverman JJ, Galanter M, Jackson-Triche M, et al.",
    "The American Psychiatric Association Practice Guidelines for the Psychiatric Evaluation of Adults",
    "Am J Psychiatry", 2015, "172(8):798-802", doi="10.1176/appi.ajp.2015.1720501", pmid="26234607",
    verified=True, short_author="APA 정신과 평가 지침",
)

# --- 2026-09-27 additions (bibliographic data checked against PubMed E-utilities on 2026-09-27) ---
G_URTICARIA = Citation(
    "Zuberbier T, Abdul Latiff AH, Abuzakouk M, et al.",
    "The international EAACI/GA²LEN/EuroGuiDerm/APAAACI guideline for the definition, classification, diagnosis, "
    "and management of urticaria",
    "Allergy", 2022, "77(3):734-766", doi="10.1111/all.15090", pmid="34536239",
    verified=True, short_author="EAACI 두드러기 지침",
)
G_SSNHL = Citation(
    "Chandrasekhar SS, Tsai Do BS, Schwartz SR, et al.",
    "Clinical Practice Guideline: Sudden Hearing Loss (Update)",
    "Otolaryngol Head Neck Surg", 2019, "161(1_suppl):S1-S45", doi="10.1177/0194599819859885", pmid="31369359",
    verified=True, short_author="AAO-HNS 돌발성 난청 지침",
)
G_ARHL = Citation(
    "Tsai Do BS, Bush ML, Weinreich HM, et al.",
    "Clinical Practice Guideline: Age-Related Hearing Loss",
    "Otolaryngol Head Neck Surg", 2024, "170 Suppl 2:S1-S54", doi="10.1002/ohn.750", pmid="38687845",
    verified=True, short_author="AAO-HNS 노인성 난청 지침",
)
G_HEARING_IMAGING = Citation(
    "Expert Panel on Neurologic Imaging; Sharma A, Kirsch CFE, et al.",
    "ACR Appropriateness Criteria Hearing Loss and/or Vertigo",
    "J Am Coll Radiol", 2018, "15(11S):S321-S331", doi="10.1016/j.jacr.2018.09.020", pmid="30392601",
    verified=True, short_author="ACR 난청/어지럼 적정성 기준",
)
G_NECK_MASS = Citation(
    "Pynnonen MA, Gillespie MB, Roman B, et al.",
    "Clinical Practice Guideline: Evaluation of the Neck Mass in Adults",
    "Otolaryngol Head Neck Surg", 2017, "157(2_suppl):S1-S30", doi="10.1177/0194599817722550", pmid="28891406",
    verified=True, short_author="AAO-HNS 성인 경부 종괴 지침",
)
G_HEMOPHILIA = Citation(
    "Srivastava A, Santagostino E, Dougall A, et al.",
    "WFH Guidelines for the Management of Hemophilia, 3rd edition",
    "Haemophilia", 2020, "26 Suppl 6:1-158", doi="10.1111/hae.14046", pmid="32744769",
    verified=True, short_author="WFH 혈우병 지침",
)
G_IIM_CRITERIA = Citation(
    "Lundberg IE, Tjärnlund A, Bottai M, et al.",
    "2017 European League Against Rheumatism/American College of Rheumatology classification criteria for adult and "
    "juvenile idiopathic inflammatory myopathies and their major subgroups",
    "Ann Rheum Dis", 2017, "76(12):1955-1964", doi="10.1136/annrheumdis-2017-211468", pmid="29079590",
    verified=True, short_author="EULAR/ACR 염증성 근병증 분류기준",
)
G_INFANT_VOMITING = Citation(
    "Expert Panel on Pediatric Imaging; Alazraki AL, Rigsby CK, et al.",
    "ACR Appropriateness Criteria Vomiting in Infants",
    "J Am Coll Radiol", 2020, "17(11S):S505-S515", doi="10.1016/j.jacr.2020.09.002", pmid="33153561",
    verified=True, short_author="ACR 영아 구토 적정성 기준",
)

GUIDELINES: tuple[Citation, ...] = (
    G_CHEST_PAIN, G_AORTA, G_PE, G_HF, G_PLEURAL, G_ANAPHYLAXIS, G_HEADACHE, G_MENINGITIS, G_STROKE,
    G_SEPSIS, G_NEUTROPENIA, G_EARLY_PREGNANCY, G_ECTOPIC, G_AAA, G_AMI, G_ENDOCARDITIS,
    G_SYNCOPE, G_PALPITATIONS, G_HEMOPTYSIS, G_COUGH, G_LIVER, G_JAUNDICE_IMAGING, G_NEONATAL_JAUNDICE, G_HOT_JOINT,
    G_GOUT_DX, G_LOW_BACK_PAIN, G_SJS_TEN, G_PRURITUS, G_VTE_DX, G_GLOMERULAR, G_AMENORRHEA, G_PMB, G_MECFS,
    G_DEMENTIA, G_PSYCH_EVAL,
    G_URTICARIA, G_SSNHL, G_ARHL, G_HEARING_IMAGING, G_NECK_MASS, G_HEMOPHILIA, G_IIM_CRITERIA, G_INFANT_VOMITING,
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
    triggers: tuple[str, ...] = ()  # conditional check: applies only if the case text contains one (not negated)
    when: str = ""  # Korean condition text for prompts
    note: str = ""
    # acute-only check: not applicable when the chief complaint has lasted >= this duration_level
    # (1 = >=2 weeks, 2 = months/years/"만성"); 0 = always. Judged on the chief complaint only, because
    # history facts mention unrelated past durations ("6개월 전 건강검진").
    acute_only: int = 0
    predicate: str = ""  # name in PREDICATES; replaces `triggers` with a function of (chief complaint, all text)
    # population limits from the chief complaint: minimum duration_level (e.g. 1 = >= 2 weeks) and age in years
    # (an unstated age passes); used when the cited guideline covers only adults or chronic presentations
    min_duration: int = 0
    min_age: float | None = None

    def applies(self, text: str, cc: str | None = None) -> bool:
        """text = chief complaint + learned facts (triggers); cc = chief complaint (duration). cc defaults to text."""
        cc = text if cc is None else cc
        if self.acute_only and duration_level(cc) >= self.acute_only:
            return False
        if self.min_duration and duration_level(cc) < self.min_duration:
            return False
        if self.min_age is not None:
            age = age_years(cc)
            if age is not None and age < self.min_age:
                return False
        if self.predicate:
            return PREDICATES[self.predicate](cc, text)
        if not self.triggers:
            return True
        return contains_affirmed(text, self.triggers)

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
_TRIG_AMS = ("의식 저하", "의식저하", "의식 변화", "의식변화", "의식이 흐", "의식이 없", "의식을 잃", "혼돈", "착란",
             "헛소리", "횡설수설", "앞뒤가 맞지 않", "환각", "섬망", "기면", "축 처", "축 늘어", "confus", "delir",
             "altered mental", "obtund", "letharg")
_TRIG_FEVER = ("발열", "열이", "열과", "열도", "열나", "열감", "고열", "미열", "오한",
               "fever", "febrile", "chills")
_TRIG_MENINGITIS = _TRIG_FEVER + ("뻣뻣", "경부 강직", "stiff neck", "neck stiffness") \
    + _TRIG_AMS
_TRIG_HEADACHE_MENINGISM = ("두통", "머리가 아", "목이 뻣뻣", "목 뻣뻣", "경부 강직", "headache", "stiff neck",
                            "neck stiffness") + _TRIG_AMS
_TRIG_AORTA = ("찢어지", "찢기는", "뜯기는", "등으로", "등까지", "등 통증", "양팔", "마르판", "tearing", "ripping",
               "radiat", "back pain", "marfan")
_TRIG_PE = ("객혈", "다리가 붓", "다리 부종", "종아리", "수술", "부동", "장거리", "비행", "피임약", "호르몬", "혈전",
            "hemoptysis", "leg swelling", "calf", "surgery", "immobil", "long flight", "contracepti", "thrombo")
# not bare "알레르기": a drug-allergy history ("페니실린 알레르기가 있습니다") is not an allergic reaction (da_858)
_TRIG_ANAPHYLAXIS = ("두드러기", "입술이 붓", "혀가 붓", "얼굴이 붓", "알레르기 반응", "알레르기성 반응", "아나필락시스",
                     "벌에", "땅콩", "새우", "먹고 나서", "주사 맞고", "hives", "urticaria", "allergic reaction",
                     "anaphyla", "angioedema", "bee sting")
_TRIG_NEUTROPENIA = ("항암", "암 치료", "화학요법", "면역저하", "면역 저하", "골수", "chemotherapy", "neutropen",
                     "immunocompromis", "transplant", "이식")
_TRIG_FEMALE = ("여성", "여자", "female", "woman", "임신", "생리", "월경", "질출혈", "질 출혈", "pregnan", "menstru",
                "vaginal bleeding")
_TRIG_AAA = ("등 통증", "허리 통증", "옆구리", "박동", "실신", "기절", "쇼크", "저혈압", "back pain", "flank",
             "pulsatile", "syncope", "shock", "hypotens")
_TRIG_AMI = ("심방세동", "부정맥", "통증에 비해", "진찰 소견에 비해", "혈전", "atrial fibrillation", "afib",
             "out of proportion")


# Sepsis (SSC 2021 applies to *suspected sepsis*): systemic-instability signals, from any text
_TRIG_SEPSIS_INSTABILITY = _TRIG_AMS + (
    "저혈압", "혈압이 떨어", "혈압이 낮", "쇼크", "빈맥", "맥박이 빠르", "맥이 빠르", "심장이 빨리", "숨을 빨리",
    "호흡이 빠르", "소변량 감소", "소변량이 줄", "소변이 안 나", "소변이 나오지", "덜덜", "사시나무", "오한 떨림",
    "패혈", "균혈", "sepsis", "septic", "bacteremia", "hypotens", "tachycard", "shock", "rigor", "oliguria",
    "hiv", "에이즈", "면역억제", "aids",
) + _TRIG_NEUTROPENIA
# Acute febrile-illness signals, counted only when the chief complaint is not >= 2 weeks old
_TRIG_SEPSIS_ACUTE = ("갑자기", "갑작스", "고열", "오한", "high fever", "sudden")
_RE_HIGH_TEMP = re.compile(r"(39|40|41)(\.\d)?\s*(°|도|℃)")
_RE_SBP = re.compile(r"혈압\s*:?\s*(\d{2,3})\s*/|blood pressure\s*:?\s*(\d{2,3})\s*/|bp\s*:?\s*(\d{2,3})\s*/")
_RE_HR = re.compile(r"(?:맥박|심박수?|heart rate|pulse|hr)\s*:?\s*(\d{2,3})")
_RE_RR = re.compile(r"(?:호흡수|respiratory rate|rr)\s*:?\s*(\d{1,2})")


def _numeric_instability(t: str, need: int = 1) -> bool:
    """At least `need` of: SBP <= 100, RR >= 22 (qSOFA, Seymour 2016), HR > 100, when numbers appear in the text."""
    sbp = [int(g) for m in _RE_SBP.finditer(t) for g in m.groups() if g]
    hr = [int(m.group(1)) for m in _RE_HR.finditer(t)]
    rr = [int(m.group(1)) for m in _RE_RR.finditer(t)]
    n = any(x <= 100 for x in sbp) + any(x > 100 for x in hr) + any(x >= 22 for x in rr)
    return n >= need


def _sepsis_suspected(cc: str, text: str) -> bool:
    """Blood cultures / lactate apply to fever with acute systemic illness, not to weeks-long febrile illness
    without instability (e.g. hypersensitivity pneumonitis, TB). Our operationalisation of SSC 2021's
    'suspected sepsis'."""
    t = (text or "").lower()
    # weeks-long febrile illness: one borderline vital sign (e.g. RR 22 in hypersensitivity pneumonitis) is not enough
    need = 2 if duration_level(cc) >= 1 else 1
    if contains_affirmed(t, _TRIG_SEPSIS_INSTABILITY) or _numeric_instability(t, need):
        return True
    return duration_level(cc) == 0 and (contains_affirmed(t, _TRIG_SEPSIS_ACUTE) or bool(_RE_HIGH_TEMP.search(t)))


# Infective-endocarditis risk clues (negation-aware): fever + any of these → blood cultures, any duration
_TRIG_IE = (
    "심내막염", "심잡음", "심장 잡음", "심장잡음", "잡음이 들", "수축기 잡음", "확장기 잡음", "인공판막", "인공 판막",
    "기계판막", "판막 치환", "판막 수술", "판막 시술", "판막 질환", "판막질환", "판막증", "승모판", "대동맥판",
    "류마티스 심장", "류마티스열", "선천성 심장", "치과 치료", "치과 시술", "치과에서", "발치", "이를 뽑", "스케일링",
    "마약 주사", "주사 마약", "정맥 주사 약물", "약물 주사", "필로폰", "헤로인", "선상 출혈", "손톱 밑 출혈",
    "손톱 아래 출혈", "제인웨이", "오슬러", "결막 점상", "결막에 점", "균혈증", "혈액배양 양성", "혈액 배양 양성",
    "endocarditis", "murmur", "prosthetic valve", "valve replacement", "valvular", "valve disease", "dental",
    "tooth extraction", "injection drug", "ivdu", "pwid", "heroin", "splinter", "janeway", "osler",
    "conjunctival petechiae", "bacteremia", "positive blood culture",
)


def _ie_suspected(cc: str, text: str) -> bool:
    """Fever (category) + an IE risk clue, when the sepsis blood-culture check does not already apply
    (one blood-culture check per case, so it is not double-weighted in scoring)."""
    return contains_affirmed(text, _TRIG_IE) and not _sepsis_suspected(cc, text)



# --- predicates for the 2026-09-26 categories (our operationalisations, not from the cited papers) ---------
_TRIG_HEMOPTYSIS = ("객혈", "피가 섞인 가래", "피 섞인 가래", "피섞인 가래", "혈담", "피가래", "기침할 때 피",
                    "기침하면 피", "hemoptysis", "coughing up blood", "blood-streaked sputum")
_RE_UNILATERAL_LEG = re.compile(
    r"(한쪽|한 쪽|편측|왼쪽|오른쪽|좌측|우측|왼|오른|one|left|right|unilateral)\s?\S{0,2}\s?"
    r"(다리|종아리|하지|발목|허벅지|leg|calf)")
_TRIG_KNOWN_PREGNANCY = ("임신 중", "임신중", "임산부", "산모", "pregnant")
_TRIG_POSTMENOPAUSAL = ("폐경", "postmenopaus", "menopaus")
_TRIG_AMENORRHEA = ("무월경", "월경이 없", "생리가 없", "생리를 안", "생리가 안 ", "생리를 하지", "월경을 하지",
                    "생리가 늦", "초경", "초경이 없", "초경 없", "초경을 안", "초경을 하지", "초경이 아직",
                    "amenorrh", "missed period")
_RE_PREG_WEEKS_PR = re.compile(r"(임신|재태)\s*\d+\s*주|\d+\s*주\s*(차\s*)?임신")


def _neonate(cc: str, text: str) -> bool:
    """Newborn (<= 28 days) by the chief complaint: the AAP 2022 population (>= 35 weeks' gestation assumed)."""
    d = age_days(cc)
    return d is not None and d <= 28


def _not_neonate(cc: str, text: str) -> bool:
    d = age_days(cc)
    return d is None or d > 28


def _neonate_prolonged(cc: str, text: str) -> bool:
    """Jaundiced newborn >= 14 days old (AAP 2022: formula-fed still jaundiced at 2 weeks, breastfed at 3-4)."""
    d = age_days(cc)
    return d is not None and 14 <= d <= 60


def _hemoptysis(cc: str, text: str) -> bool:
    return contains_affirmed(cc, _TRIG_HEMOPTYSIS)


def _chronic_cough_only(cc: str, text: str) -> bool:
    """Cough >= 2 weeks without hemoptysis (the hemoptysis CXR check covers that case, no double weight)."""
    return duration_level(cc) >= 1 and not _hemoptysis(cc, text)


def _unilateral_leg(cc: str, text: str) -> bool:
    return bool(_RE_UNILATERAL_LEG.search((cc or "").lower()))


def _not_unilateral_leg(cc: str, text: str) -> bool:
    return not _unilateral_leg(cc, text)


def _amenorrhea(cc: str, text: str) -> bool:
    return contains_affirmed(cc, _TRIG_AMENORRHEA)


def _postmenopausal_bleeding(cc: str, text: str) -> bool:
    t = (cc or "").lower()
    return not _amenorrhea(cc, text) and any(k in t for k in _TRIG_POSTMENOPAUSAL)


def _pregnancy_possible(cc: str, text: str) -> bool:
    """Amenorrhea, or abnormal vaginal bleeding when neither pregnancy nor menopause is already stated."""
    if _amenorrhea(cc, text):
        return True
    t = (cc or "").lower()
    if _RE_PREG_WEEKS_PR.search(t) or any(k in t for k in _TRIG_KNOWN_PREGNANCY + _TRIG_POSTMENOPAUSAL):
        return False
    age = age_years(cc)
    return age is None or 10 <= age <= 55


# --- 2026-09-27 predicates (false-trigger fixes and new categories; our operationalisations) --------------
# current pregnancy (not obstetric history "임신은 2번 했고", not an infant's birth history "임신 39주에 태어났")
_TRIG_CURRENT_PREGNANCY = ("임신 중", "임신중", "임신했", "임신 반응 양성", "임신반응 양성", "임신 검사 양성",
                           "임신검사 양성", "hcg 양성", "hcg가 양성", "임산부", "pregnant", "positive pregnancy",
                           "positive hcg", "생리가 늦", "생리가 안", "생리를 안", "생리가 없", "월경이 없", "무월경",
                           "missed period", "amenorrh")
_RE_CURRENT_PREG_WEEKS = re.compile(r"(임신|재태)\s*\d+\s*주(?![^.]{0,15}(태어|출생|분만|조산|낳))"
                                    r"|\d+\s*주\s*(차\s*)?임신")


def _current_pregnancy(cc: str, text: str) -> bool:
    age = age_years(cc)
    if age is not None and age < 10:
        return False
    t = (text or "").lower()
    return contains_affirmed(t, _TRIG_CURRENT_PREGNANCY) or bool(_RE_CURRENT_PREG_WEEKS.search(t))


def _pregnancy_test_abdominal(cc: str, text: str) -> bool:
    """Woman of reproductive age (10-55 or unstated) whose pregnancy is not already known and who is not stated to be
    postmenopausal (ACOG PB 193 / ACEP 2017 populations)."""
    if not contains_affirmed(text, _TRIG_FEMALE):
        return False
    age = age_years(cc)
    if age is not None and not 10 <= age <= 55:
        return False
    if _RE_CURRENT_PREG_WEEKS.search((text or "").lower()) or contains_affirmed(text, _TRIG_KNOWN_PREGNANCY):
        return False
    return not any(k in (cc or "").lower() for k in _TRIG_POSTMENOPAUSAL)


_TRIG_HOT_JOINT = ("붓", "부었", "부어", "부종", "부기", "발적", "빨갛", "붉", "열감", "뜨겁", "뜨끈", "발열", "열이", "swell",
                   "effusion", "erythem", "fever")
# periarticular source: tenosynovitis (Finkelstein), bursitis, or passive joint motion preserved
_TRIG_PERIARTICULAR = ("finkelstein", "핑켈스타인", "건초염", "건막염", "힘줄염", "건염", "tenosynovitis", "tendinitis",
                       "tendonitis", "점액낭염", "bursitis", "수동적 움직임에는 제한이 없", "수동적 움직임은 정상",
                       "수동 운동 범위는 정상", "수동 관절 운동은 정상", "passive range of motion is normal",
                       "full passive range")


def _hot_joint(cc: str, text: str) -> bool:
    """BSR 2006 population: acute hot swollen joint. A periarticular source (positive Finkelstein test, bursitis,
    preserved passive joint motion) is not a joint effusion (cqa_236 de Quervain)."""
    return contains_affirmed(text, _TRIG_HOT_JOINT) and not contains_affirmed(text, _TRIG_PERIARTICULAR)


_RE_UNILATERAL_EAR = re.compile(r"(한쪽|한 쪽|편측|왼쪽|오른쪽|좌측|우측|one|left|right|unilateral)\s?\S{0,2}\s?"
                                r"(귀|난청|청력|ear|hearing)|asymmetric|비대칭")
_TRIG_SUDDEN = ("갑자기", "갑작스", "돌발", "자고 일어나", "아침에 일어나", "sudden", "abrupt")


def _sudden_hearing(cc: str, text: str) -> bool:
    return any(k in (cc or "").lower() for k in _TRIG_SUDDEN)


def _retrocochlear(cc: str, text: str) -> bool:
    """Sudden hearing loss (AAO-HNS 2019 KAS 6) or unilateral/asymmetric hearing loss (AAO-HNS 2024 KAS 5)."""
    return _sudden_hearing(cc, text) or bool(_RE_UNILATERAL_EAR.search((cc or "").lower()))


_NECK_INFECTION = ("인후통", "목이 아프", "목감기", "감기", "편도", "치통", "sore throat", "tonsil", "dental abscess")


def _neck_mass_risk(cc: str, text: str) -> bool:
    """AAO-HNS 2017 increased malignancy risk: no infectious history and present >= 2 weeks or of uncertain duration.
    Only a stated duration < 2 weeks together with an infectious story drops it (our reading)."""
    days_stated = bool(re.search(r"\d+\s*일|어제|오늘|며칠", (cc or "").lower()))
    short = duration_level(cc) == 0 and days_stated
    return not (short and contains_affirmed(text, _NECK_INFECTION))


_TRIG_SENSORY = ("저림", "저리", "저려", "감각", "타는 듯", "화끈", "numb", "tingl", "paresthes", "burning")


def _myopathy_pattern(cc: str, text: str) -> bool:
    """CK for symmetric weakness without sensory symptoms (a myopathy pattern), not a one-sided or sensory picture."""
    t = (cc or "").lower()
    one_side = bool(re.search(r"(한쪽|한 쪽|편측|왼|오른|좌측|우측|left|right|unilateral)", t))
    return not one_side and not any(k in t for k in _TRIG_SENSORY)


PREDICATES = {
    "sepsis": _sepsis_suspected, "ie": _ie_suspected, "neonate": _neonate, "not_neonate": _not_neonate,
    "neonate_prolonged": _neonate_prolonged, "hemoptysis": _hemoptysis, "chronic_cough": _chronic_cough_only,
    "unilateral_leg": _unilateral_leg, "not_unilateral_leg": _not_unilateral_leg, "amenorrhea": _amenorrhea,
    "postmenopausal_bleeding": _postmenopausal_bleeding, "pregnancy_possible": _pregnancy_possible,
    "current_pregnancy": _current_pregnancy, "pregnancy_test_abdominal": _pregnancy_test_abdominal,
    "hot_joint": _hot_joint, "sudden_hearing": _sudden_hearing, "retrocochlear": _retrocochlear,
    "neck_mass_risk": _neck_mass_risk, "myopathy_pattern": _myopathy_pattern,
}


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
                  when="급성 관상동맥 증후군 의심", acute_only=2,
                  note="hs-cTn is the preferred biomarker for *acute* chest pain. Confirmed via ACC summaries. "
                  "Not applied to months-long chest pain (acute_only=2; ECG still applies to stable chest pain)."),
            Check("cxr", "흉부 X선", "test", _KW_CXR, G_CHEST_PAIN, "unverified",
                  when="폐·흉막·대동맥 등 다른 원인 평가",
                  note="Guideline suggests CXR to evaluate alternative cardiac/pulmonary/thoracic causes."),
            Check("aorta_imaging", "양팔 혈압 비교 + 대동맥 영상검사(CT 혈관조영 등)", "test", _KW_AORTA, G_AORTA,
                  "unverified", triggers=_TRIG_AORTA, when="찢어지는 통증, 등으로 뻗는 통증, 양팔 혈압 차이 등",
                  acute_only=2,
                  note="Use with ADD-RS (knowledge/clinical_rules.py)."),
            Check("pe_workup", "폐색전증 사전확률 평가 후 D-dimer 또는 CT 폐동맥조영", "test", _KW_PE_TEST, G_PE,
                  "unverified", triggers=_TRIG_PE, when="객혈, 한쪽 다리 부종, 최근 수술·부동 등 위험인자",
                  acute_only=2, note="ESC 2019: clinical probability (Wells/Geneva), D-dimer if not high probability, CTPA."),
        ),
    ),
    Protocol(
        category="dyspnea", name_ko=CATEGORY_NAMES["dyspnea"],
        cant_miss=("폐색전증", "급성 심부전", "아나필락시스", "긴장성 기흉"),
        checks=(
            _vitals(G_PE, note="Hemodynamic status (shock/hypotension) defines high-risk PE in ESC 2019; "
                    "SpO2 is basic assessment. Not checked in the guideline text."),
            Check("ecg", "12유도 심전도", "test", _KW_ECG, G_HF, "unverified", when="모든 급성 호흡곤란(성인)",
                  min_age=18,
                  note="HF guideline recommends ECG in initial evaluation of suspected HF. Adult guideline: not "
                  "applied to newborns/children (2026-09-27)."),
            Check("cxr", "흉부 X선", "test", _KW_CXR, G_PLEURAL, "unverified", when="기흉·폐렴·폐부종 평가",
                  note="Tension pneumothorax is a clinical diagnosis; do not delay decompression for imaging."),
            Check("natriuretic_peptide", "BNP 또는 NT-proBNP", "test", ("bnp", "nt-probnp", "나트륨이뇨"), G_HF,
                  "unverified", when="심부전 의심(성인)", min_age=18,
                  note="Class 1: natriuretic peptides to support/exclude HF in patients presenting with dyspnea."),
            Check("pe_workup", "폐색전증 사전확률 평가 후 D-dimer 또는 CT 폐동맥조영", "test", _KW_PE_TEST, G_PE,
                  "unverified", triggers=_TRIG_PE, when="객혈, 한쪽 다리 부종, 최근 수술·부동 등 위험인자",
                  acute_only=2, min_age=18),
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
                  _KW_BRAIN_CT, G_HEADACHE, "secondary", triggers=_TRIG_THUNDERCLAP, acute_only=1,
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
                  when="패혈증 의심(급성 발열 + 오한·저혈압·빈맥·의식 변화 등)", predicate="sepsis",
                  note="SSC 2021: obtain cultures including blood before antimicrobials if it does not delay "
                  "treatment (read in PMC8486643, co-published version)."),
            Check("lactate", "혈중 젖산", "test", ("젖산", "락테이트", "lactate", "lactic"), G_SEPSIS, "primary",
                  when="패혈증 의심(급성 발열 + 오한·저혈압·빈맥·의식 변화 등)", predicate="sepsis", note="SSC 2021: suggest measuring blood lactate (weak recommendation)."),
            Check("blood_culture_ie", "감염성 심내막염 의심 시 항생제 전 혈액배양(여러 세트)", "test", _KW_BLOOD_CULTURE,
                  G_ENDOCARDITIS, "unverified", predicate="ie",
                  when="발열 + 심잡음·판막질환/인공판막·최근 치과/판막 시술·주사 약물·색전 징후·균혈증",
                  note="Citation bibliographically verified (PubMed 37622656). The recommendation (>=3 blood "
                  "culture sets before antibiotics in suspected IE) is from reviewer knowledge, not re-read in "
                  "the guideline text. The clue list is our operationalisation of IE risk / modified Duke minor "
                  "criteria."),
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
                  predicate="pregnancy_test_abdominal", when="가임기 여성의 복통(임신 사실이 이미 알려진 경우 제외)",
                  note="ACOG PB 193 / ACEP 2017 cover ectopic pregnancy evaluation (hCG + transvaginal US). "
                  "Recommendation text not re-read. 2026-09-27: not for known pregnancy (임신 38주), age > 55 or "
                  "stated menopause."),
            Check("pelvic_us", "임신 양성이면 골반(질식) 초음파", "test",
                  ("질식 초음파", "골반 초음파", "경질 초음파", "transvaginal", "pelvic ultrasound", "pelvic us"),
                  G_EARLY_PREGNANCY, "unverified", predicate="current_pregnancy",
                  when="임신 확인된(또는 월경 지연) 복통·질출혈",
                  note="2026-09-27: current pregnancy only, not an obstetric history ('임신은 2번 했고') or an "
                  "infant's birth history."),
            Check("aaa_imaging", "복부 대동맥 초음파 또는 CT", "test",
                  ("복부 초음파", "복부 ct", "대동맥 초음파", "abdominal ultrasound", "abdominal ct", "ct abdomen",
                   "aortic ultrasound"),
                  G_AAA, "unverified", triggers=_TRIG_AAA, min_age=50,
                  when="50세 이상(또는 나이 미상)·등/옆구리 통증·박동성 종괴·실신·저혈압",
                  note="min_age=50 is our operationalisation (AAA is rare below 50; SVS screening starts at 65, or "
                  "55 with risk); it stops flank pain in young adults (pyelonephritis, DKA) from requiring it."),
            Check("mesenteric_cta", "장간막 허혈 의심 시 지체 없이 CT 혈관조영", "test",
                  ("ct 혈관조영", "ct 혈관 조영", "cta", "ct angiogra", "복부 혈관조영", "mesenteric"), G_AMI,
                  "primary", triggers=_TRIG_AMI, when="심방세동, 진찰 소견에 비해 심한 통증",
                  note="WSES 2022 recommendation 5 (1A): CTA without delay in any patient with suspected AMI; "
                  "lactate/D-dimer may assist but cannot exclude (rec. 4). Read in PMC9580452."),
        ),
    ),
)

PROTOCOLS = PROTOCOLS + (
    Protocol(
        category="allergy", name_ko=CATEGORY_NAMES["allergy"],
        cant_miss=("아나필락시스", "상기도 부종(혈관부종)"),
        checks=(
            _vitals(G_ANAPHYLAXIS, note="Hypotension/hypoxia define severity in anaphylaxis; basic assessment, "
                    "not a numbered recommendation checked in the practice parameter text."),
            Check("airway_breathing", "기도·호흡 평가(입술·혀·목 부종, 쉰 목소리, 천명음·쌕쌕거림, 청진)", "exam",
                  ("기도", "천명", "쌕쌕", "호흡음", "청진", "쉰 목소리", "목소리", "삼키기", "stridor", "wheez",
                   "airway", "auscult", "hoarse", "혀 부종", "입술 부종", "인두 부종", "후두 부종"),
                  G_ANAPHYLAXIS, "unverified", when="모든 급성 알레르기 반응",
                  note="Airway/breathing involvement is a defining feature of anaphylaxis; from reviewer knowledge, "
                  "not a numbered recommendation read in the practice parameter."),
            Check("epinephrine", "아나필락시스 의심 시 즉시 에피네프린 근육주사", "treatment",
                  ("에피네프린", "epinephrine", "아드레날린", "adrenaline", "epipen"), G_ANAPHYLAXIS, "unverified",
                  when="호흡기·순환기 증상 동반 알레르기 반응"),
        ),
    ),
)

# --------------------------------------------------------------------------------------------
# 2026-09-26 additions: the most common chief complaints in data/cases_aug that had no protocol.
# Verification levels were set on 2026-09-26 from what was actually read (see each note).
# --------------------------------------------------------------------------------------------

_KW_CBC = ("일반혈액", "혈구", "cbc", "백혈구", "혈색소", "헤모글로빈", "혈소판", "말초혈액", "complete blood count",
           "blood count", "hemoglobin", "platelet")
_KW_LIVER = ("간기능", "간 기능", "간수치", "간 수치", "간효소", "간 효소", "빌리루빈", "bilirubin", "lft", "ast/alt",
             "ast, alt", "ast와 alt", "ast·alt", "transaminase", "아미노전이효소", "알칼리성 인산", "alkaline phosphatase",
             "liver function", "liver panel", "liver test", "liver chemistr")
_KW_RENAL = ("신장 기능", "신장기능", "신기능", "콩팥 기능", "크레아티닌", "creatinine", "bun", "egfr", "사구체여과",
             "renal function", "kidney function", "요소질소")
_KW_TSH = ("tsh", "갑상선", "갑상샘", "thyroid")
_KW_URINALYSIS = ("소변 검사", "소변검사", "요검사", "요 검사", "요분석", "단백뇨", "소변 단백", "알부민뇨", "urinalysis",
                  "urine protein", "proteinuria", "urine albumin", "albuminuria", "dipstick")
_KW_CHEST_CT = ("흉부 ct", "흉부ct", "가슴 ct", "폐 ct", "chest ct", "ct chest", "ct of the chest", "흉부 전산화",
                "ct 혈관조영", "ct angiogra", "cta")
_KW_MED_HISTORY = ("복용 중인 약", "복용중인 약", "복용하는 약", "복용하시는 약", "먹는 약", "드시는 약", "약을 먹",
                   "약을 드", "새로 시작한 약", "약물", "처방", "한약", "건강보조", "보조제", "영양제", "medication",
                   "drug", "medicine", "supplement", "herbal")
_KW_ALCOHOL = ("음주", "술을", "술은", "술 마", "술도", "alcohol", "drink")

PROTOCOLS = PROTOCOLS + (
    Protocol(
        category="syncope", name_ko=CATEGORY_NAMES["syncope"],
        cant_miss=("부정맥(서맥·빈맥, QT 연장)", "구조적 심질환(대동맥판 협착, 비후성 심근병증)", "폐색전증",
                   "대동맥 박리", "내출혈·저혈량"),
        checks=(
            Check("vitals", "활력징후(누운 자세·선 자세 혈압/맥박 포함)", "exam", _KW_VITALS + ("기립", "orthostatic"),
                  G_SYNCOPE, "unverified", when="모든 실신",
                  note="Orthostatic BP/HR is part of the initial evaluation in the guideline; the COR/LOE was not "
                  "read (full text returned HTTP 403)."),
            Check("ecg", "12유도 심전도", "test", _KW_ECG, G_SYNCOPE, "secondary", when="모든 실신 초기 평가",
                  note="Class I: resting 12-lead ECG in the initial evaluation. Full text paywalled (HTTP 403); "
                  "confirmed via ACC 'Ten points to remember' and ACEP Now summaries."),
            Check("cardiac_history", "심장성 실신 단서 문진(운동 중·누운 자세 실신, 전조 없는 실신, 두근거림, "
                  "심질환, 가족 돌연사)", "ask",
                  ("운동 중", "운동할 때", "누운 상태", "누워 있을 때", "전조", "두근", "심장병", "심장 질환", "심장질환",
                   "돌연사", "급사", "exertion", "palpitation", "heart disease", "sudden death", "prodrome"),
                  G_SYNCOPE, "secondary", when="모든 실신",
                  note="Class I: detailed history and physical examination. The specific cardiac-syncope clues are "
                  "the guideline's high-risk features as summarized by ACC; wording ours."),
        ),
    ),
    Protocol(
        category="palpitations", name_ko=CATEGORY_NAMES["palpitations"],
        cant_miss=("심실성 빈맥", "심방세동", "WPW 증후군", "QT 연장 증후군", "갑상선 기능 항진증"),
        checks=(
            _vitals(G_PALPITATIONS, when="모든 두근거림",
                    note="Part of the physical examination in the EHRA initial evaluation; vital signs as such are "
                    "not a separately worded recommendation."),
            Check("ecg", "12유도 심전도", "test", _KW_ECG, G_PALPITATIONS, "primary", when="모든 두근거림",
                  note="EHRA 2011: initial evaluation of all patients = history, physical examination and a "
                  "standard 12-lead ECG (read in the Europace full text)."),
            Check("tsh", "갑상선 기능 검사(TSH)", "test", _KW_TSH, G_PALPITATIONS, "unverified",
                  triggers=("체중 감소", "체중이 줄", "살이 빠", "더위", "열불내성", "땀이 많", "손 떨림", "손이 떨",
                            "떨림", "갑상선", "weight loss", "heat intolerance", "tremor", "thyroid"),
                  when="체중 감소·더위 못 견딤·떨림 등 갑상선 기능 항진 단서",
                  note="EHRA 2011 says specific laboratory tests when a systemic cause is suspected (read); choosing "
                  "TSH for these clues is our operationalisation."),
        ),
    ),
    Protocol(
        category="hemoptysis_cough", name_ko=CATEGORY_NAMES["hemoptysis_cough"],
        cant_miss=("대량 객혈", "폐암", "폐결핵", "폐색전증"),
        checks=(
            Check("cxr", "흉부 X선", "test", _KW_CXR, G_HEMOPTYSIS, "primary", predicate="hemoptysis",
                  when="모든 객혈",
                  note="ACR AC 2020 (abstract): chest radiograph and CT with contrast or CTA recommended for "
                  "massive and nonmassive hemoptysis."),
            Check("chest_ct", "흉부 CT(조영증강) 또는 CT 혈관조영", "test", _KW_CHEST_CT, G_HEMOPTYSIS, "primary",
                  predicate="hemoptysis", when="모든 객혈",
                  note="Same ACR AC 2020 statement (abstract)."),
            Check("cxr_cough", "흉부 X선", "test", _KW_CXR, G_COUGH, "unverified", predicate="chronic_cough",
                  when="2주 이상 지속되는 기침",
                  note="Read (PMC3345522): CXR for cough with lung-cancer risk factors (grade E/A) and bronchoscopy "
                  "for suspected airway malignancy. Applying CXR to every cough lasting >= 2 weeks is reviewer "
                  "knowledge of the ACCP chronic-cough algorithm, not re-read."),
        ),
    ),
    Protocol(
        category="jaundice", name_ko=CATEGORY_NAMES["jaundice"],
        cant_miss=("급성 간부전", "급성 담관염/담도 폐쇄", "췌담도 악성 종양", "용혈", "신생아 병적 황달·담도 폐쇄증"),
        checks=(
            Check("liver_panel", "간기능 검사(AST/ALT, ALP, 총·직접 빌리루빈 분획)", "test", _KW_LIVER, G_LIVER,
                  "primary", predicate="not_neonate", when="모든 황달",
                  note="ACG 2017 (abstract): elevated total bilirubin should be fractionated to direct/indirect."),
            Check("med_alcohol_history", "약물·한약·건강보조제 복용력과 음주력 문진", "ask",
                  _KW_MED_HISTORY + _KW_ALCOHOL, G_LIVER, "primary", predicate="not_neonate", when="모든 황달",
                  note="ACG 2017 recommendation 11 (strong) + clinical assessment statement 1 (read in the ACG "
                  "summary PDF)."),
            Check("pt_inr", "PT/INR(간 합성능)과 의식 상태(간성 뇌증) 확인", "test",
                  ("pt/inr", "inr", "프로트롬빈", "prothrombin", "응고 검사", "응고검사", "pt 검사", "coagulation"),
                  G_LIVER, "primary", predicate="not_neonate", when="모든 황달",
                  note="ACG 2017 recommendation 19: acute hepatitis with elevated PT and/or encephalopathy needs "
                  "immediate referral to a liver specialist. That PT must be measured to apply it is our reading."),
            Check("abdominal_us", "복부(간·담도) 초음파", "test",
                  ("복부 초음파", "간 초음파", "담도 초음파", "상복부 초음파", "우상복부 초음파", "abdominal ultrasound",
                   "abdominal us", "ruq ultrasound", "liver ultrasound", "복부 ct", "abdominal ct", "mrcp"),
                  G_JAUNDICE_IMAGING, "unverified", predicate="not_neonate", when="모든 황달(담도 폐쇄 감별)",
                  note="ACR AC Jaundice 2019 abstract lists US among the modalities; US as the usual first test is "
                  "reviewer knowledge (variant tables not read)."),
            Check("neonatal_bilirubin", "혈청 또는 경피 빌리루빈(TSB/TcB) 측정", "test",
                  ("빌리루빈", "bilirubin", "tsb", "tcb", "경피"), G_NEONATAL_JAUNDICE, "secondary",
                  predicate="neonate", when="황달이 있는 생후 28일 이내 신생아",
                  note="AAP 2022 (>= 35 weeks' gestation): measure TSB/TcB in jaundiced infants and use TSB to "
                  "guide treatment. Full text HTTP 403; confirmed via secondary summaries."),
            Check("neonatal_direct_bilirubin", "직접(결합) 빌리루빈 측정(담즙 정체 배제)", "test",
                  ("직접 빌리루빈", "직접빌리루빈", "결합 빌리루빈", "direct bilirubin", "conjugated", "direct-reacting"),
                  G_NEONATAL_JAUNDICE, "secondary", predicate="neonate_prolonged",
                  when="생후 2주 이후에도 지속되는 신생아 황달",
                  note="AAP 2022: measure total and direct/conjugated bilirubin in breastfed infants still jaundiced "
                  "at 3-4 weeks and formula-fed at 2 weeks (secondary summaries). We use >= 14 days for both."),
        ),
    ),
    Protocol(
        category="joint", name_ko=CATEGORY_NAMES["joint"],
        cant_miss=("세균성(화농성) 관절염", "파종성 임균 감염", "결정성 관절염(통풍·가성통풍)"),
        checks=(
            Check("arthrocentesis", "관절 천자(활액 그람 염색·배양·결정 검사) — 항생제 전", "test",
                  ("관절 천자", "관절천자", "관절액", "관절 액", "활액", "관절 흡인", "천자", "arthrocentesis",
                   "synovial fluid", "joint aspiration", "joint fluid", "aspirat"),
                  G_HOT_JOINT, "secondary", acute_only=1, min_age=16, predicate="hot_joint",
                  when="급성(2주 미만) 관절 부기·발적·열감(성인), 건초염·점액낭염 등 관절 주위 원인 제외",
                  note="BSR 2006: aspirate, Gram-stain and culture synovial fluid before antibiotics in a hot "
                  "swollen joint (confirmed via secondary summaries; abstract has no text). EULAR 2018 gout "
                  "(abstract, read): search for crystals in synovial fluid in every person with suspected gout. "
                  "2026-09-27: not when a periarticular source is found (positive Finkelstein, bursitis, preserved "
                  "passive joint motion) - our operationalisation of 'hot swollen joint'."),
        ),
    ),
    Protocol(
        category="back_pain", name_ko=CATEGORY_NAMES["back_pain"],
        cant_miss=("마미 증후군", "척추 전이암", "척추 감염(골수염·경막외 농양)", "압박 골절", "복부 대동맥류"),
        checks=(
            Check("red_flags", "위험 신호 문진(암 병력·체중 감소, 발열, 대소변 장애·안장 감각 저하, 외상, "
                  "진행하는 다리 약화)", "ask",
                  ("체중 감소", "체중이 줄", "살이 빠", "암 병력", "암 진단", "암을", "발열", "열이", "대소변", "소변",
                   "배변", "변실금", "안장", "외상", "넘어", "다리에 힘", "weight loss", "cancer", "fever", "bladder",
                   "bowel", "incontinence", "saddle", "trauma"),
                  G_LOW_BACK_PAIN, "primary", min_age=18, when="모든 요통",
                  note="ACP/APS 2007 recommendation 1 (focused history and exam to find specific spinal causes) and "
                  "3 (imaging when severe/progressive deficits or serious conditions are suspected), read in the "
                  "abstract. The red-flag list is the guideline's usual list, wording ours."),
            Check("neuro_exam_legs", "하지 신경학적 진찰(근력·감각·반사, 하지 직거상 검사)", "exam",
                  _KW_NEURO_EXAM + ("하지 근력", "다리 근력", "근력 검사", "직거상", "slr", "straight leg", "반사",
                                    "reflex", "감각 검사"),
                  G_LOW_BACK_PAIN, "primary", min_age=18, when="모든 요통",
                  note="ACP/APS 2007 recommendation 1 (physical examination to identify radiculopathy / "
                  "neurologic deficit), abstract."),
        ),
    ),
    Protocol(
        category="rash", name_ko=CATEGORY_NAMES["rash"],
        cant_miss=("스티븐스-존슨 증후군/독성 표피 괴사 용해", "DRESS", "수막구균혈증", "괴사성 근막염"),
        checks=(
            Check("drug_history", "최근(약 2개월) 새로 시작한 약물 문진", "ask", _KW_MED_HISTORY, G_SJS_TEN,
                  "secondary", acute_only=1, min_age=16, when="급성 발진(성인)",
                  note="BAD 2016: identify and withdraw the culprit drug (drug-induced in most SJS/TEN). Full text "
                  "not read; confirmed via secondary summaries. Applying it to every acute adult rash is ours."),
            Check("mucosal_exam", "점막 침범 진찰(눈·입안·생식기)", "exam",
                  ("점막", "입안", "입 안", "구강", "결막", "눈 충혈", "눈이 충혈", "생식기", "성기", "mucos", "oral",
                   "conjunctiv", "genital"),
                  G_SJS_TEN, "secondary", acute_only=1, min_age=16, when="급성 발진(성인)",
                  note="BAD 2016: history/exam for mucosal involvement (eyes, mouth, nose, genitalia) as an early "
                  "SJS/TEN feature; confirmed via secondary summaries."),
        ),
    ),
    Protocol(
        category="pruritus", name_ko=CATEGORY_NAMES["pruritus"],
        cant_miss=("담즙 정체성 간질환", "만성 콩팥병", "혈액 악성 종양(림프종, 진성 적혈구증가증)", "갑상선 질환",
                   "임신성 담즙 정체"),
        checks=(
            Check("cbc", "일반혈액검사(CBC)", "test", _KW_CBC, G_PRURITUS, "primary", when="2주 이상 전신 가려움",
                  note="European S2k 2019: baseline investigations for chronic pruritus of unknown origin include "
                  "blood count, electrolytes, liver and kidney function, TSH and glucose (read in the Acta DV "
                  "full text). The guideline defines chronic as >= 6 weeks; our category uses >= 2 weeks."),
            Check("liver_panel", "간기능 검사(빌리루빈, ALP 포함)", "test", _KW_LIVER, G_PRURITUS, "primary",
                  when="2주 이상 전신 가려움"),
            Check("renal_function", "신장 기능(크레아티닌)", "test", _KW_RENAL, G_PRURITUS, "primary",
                  when="2주 이상 전신 가려움"),
            Check("tsh", "갑상선 기능 검사(TSH)", "test", _KW_TSH, G_PRURITUS, "primary", when="2주 이상 전신 가려움"),
        ),
    ),
    Protocol(
        category="edema", name_ko=CATEGORY_NAMES["edema"],
        cant_miss=("심부정맥혈전증", "신증후군/사구체신염", "심부전", "간경변"),
        checks=(
            Check("urinalysis", "소변 검사(단백뇨·혈뇨)", "test", _KW_URINALYSIS, G_GLOMERULAR, "unverified",
                  predicate="not_unilateral_leg", when="양측·전신·얼굴 부종 또는 거품뇨",
                  note="KDIGO 2021 bibliographically verified; the urinalysis/proteinuria assessment recommendation "
                  "was not read (executive-summary abstract only). Applying it to bilateral edema is ours."),
            Check("dvt_workup", "심부정맥혈전증 평가(사전확률 → D-dimer 또는 하지 정맥 압박 초음파)", "test",
                  ("d-dimer", "d dimer", "디다이머", "d-이합체", "하지 정맥 초음파", "다리 초음파", "하지 초음파",
                   "정맥 초음파", "도플러", "압박 초음파", "compression ultrasound", "duplex", "venous ultrasound",
                   "leg ultrasound"),
                  G_VTE_DX, "primary", predicate="unilateral_leg", when="한쪽 다리 부종",
                  note="ASH 2018 recommendations 5a/6a/7a (read in PMC6258916): D-dimer first at low pretest "
                  "probability, proximal or whole-leg ultrasound at intermediate/high probability."),
        ),
    ),
    Protocol(
        category="menstrual", name_ko=CATEGORY_NAMES["menstrual"],
        cant_miss=("임신(자궁외 임신 포함)", "자궁내막암(폐경 후 출혈)", "뇌하수체 종양"),
        checks=(
            Check("pregnancy_test", "임신 검사(β-hCG)", "test", _KW_PREGNANCY, G_AMENORRHEA, "secondary",
                  predicate="pregnancy_possible", when="무월경 또는 가임기 비정상 질출혈",
                  note="ASRM 2024: exclude pregnancy first in amenorrhea (full text not read; confirmed via ASRM "
                  "page / secondary summaries). For bleeding in reproductive age, see ACEP 2017 / ACOG PB 193."),
            Check("amenorrhea_labs", "TSH·프로락틴·FSH 검사", "test",
                  ("tsh", "갑상선", "thyroid", "프로락틴", "prolactin", "fsh", "난포자극", "성선자극"), G_AMENORRHEA,
                  "secondary", predicate="amenorrhea", when="무월경(임신 배제 후)",
                  note="ASRM 2024: initial investigations = exclude pregnancy, TSH, prolactin, FSH (secondary)."),
            Check("pmb_evaluation", "질식 초음파(자궁내막 두께) 또는 자궁내막 조직검사", "test",
                  ("질식 초음파", "경질 초음파", "골반 초음파", "자궁내막", "자궁 내막", "transvaginal", "pelvic ultrasound",
                   "endometrial"),
                  G_PMB, "primary", predicate="postmenopausal_bleeding", when="폐경 후 질출혈",
                  note="ACOG CO 734 (abstract): prompt evaluation to exclude endometrial carcinoma; TVUS (<= 4 mm) "
                  "or endometrial sampling as the first approach."),
        ),
    ),
    Protocol(
        category="fatigue", name_ko=CATEGORY_NAMES["fatigue"],
        cant_miss=("혈액 악성 종양(백혈병·림프종)", "빈혈", "갑상선 기능 저하증", "당뇨병", "우울증", "고형암"),
        checks=(
            Check("cbc", "일반혈액검사(CBC)", "test", _KW_CBC, G_MECFS, "primary", min_duration=1,
                  when="2주 이상 지속되는 피로",
                  note="NICE NG206 1.2.3: investigations to exclude other diagnoses include urinalysis, full blood "
                  "count, U&E, liver function, thyroid function, ESR, CRP, ... (read in the guideline PDF). NICE "
                  "suspects ME/CFS after 6 weeks (adults) / 4 weeks (children); our cut-off is >= 2 weeks."),
            Check("tsh", "갑상선 기능 검사(TSH)", "test", _KW_TSH, G_MECFS, "primary", min_duration=1,
                  when="2주 이상 지속되는 피로"),
        ),
    ),
    Protocol(
        category="cognitive", name_ko=CATEGORY_NAMES["cognitive"],
        cant_miss=("비타민 B12 결핍", "갑상선 기능 저하증", "경막하 혈종·정상압 수두증·뇌종양", "우울증(가성 치매)",
                   "섬망"),
        checks=(
            Check("b12", "비타민 B12 검사", "test", ("b12", "비타민 b", "코발라민", "cobalamin"), G_DEMENTIA,
                  "primary", when="기억력·인지 저하",
                  note="AAN 2001 (abstract): screening for depression, B12 deficiency and hypothyroidism should be "
                  "performed (Guideline); structural neuroimaging (noncontrast CT or MRI) is appropriate."),
            Check("tsh", "갑상선 기능 검사(TSH)", "test", _KW_TSH, G_DEMENTIA, "primary", when="기억력·인지 저하"),
            Check("brain_imaging", "뇌 영상(비조영 CT 또는 MRI)", "test", _KW_BRAIN_IMAGING, G_DEMENTIA, "primary",
                  when="기억력·인지 저하"),
            Check("depression_screen", "우울증 선별 문진", "ask", ("우울", "기분", "흥미", "depress", "mood"),
                  G_DEMENTIA, "primary", when="기억력·인지 저하"),
        ),
    ),
    Protocol(
        category="psychiatric", name_ko=CATEGORY_NAMES["psychiatric"],
        cant_miss=("자살 위험", "물질 중독·금단", "기질적 원인(갑상선·약물·뇌 병변)", "섬망"),
        checks=(
            Check("suicide_risk", "자살 사고·계획·시도력 문진", "ask",
                  ("자살", "죽고 싶", "죽고싶", "자해", "스스로 해", "삶을 끝", "극단적", "suicid", "self-harm",
                   "kill yourself", "end your life"),
                  G_PSYCH_EVAL, "secondary", min_age=18, when="성인의 기분·행동·지각 증상",
                  note="APA 2015 guideline 3: assess current suicidal ideas, plans and intent and prior attempts. "
                  "Full text HTTP 403; confirmed via AAFP 2016 practice-guideline summary."),
            Check("substance_use", "음주·흡연·약물(마약, 처방약 오남용) 사용 문진", "ask",
                  _KW_ALCOHOL + ("흡연", "담배", "마약", "약물", "물질", "대마", "smok", "tobacco", "substance",
                                 "cannabis", "marijuana", "cocaine", "amphetamine"),
                  G_PSYCH_EVAL, "secondary", min_age=18, when="성인의 기분·행동·지각 증상",
                  note="APA 2015 guideline 2: assess use of tobacco, alcohol and other substances and misuse of "
                  "prescribed/OTC medications (AAFP 2016 summary)."),
        ),
    ),
)


# --------------------------------------------------------------------------------------------
# 2026-09-27 additions: uncovered chief complaints in data/cases_aug with a guideline we could check.
# --------------------------------------------------------------------------------------------

_KW_CRP_ESR = ("crp", "c-반응", "c 반응", "c반응", "esr", "적혈구 침강", "적혈구침강", "염증 수치", "염증수치", "sedimentation")
_KW_EAR_EXAM = ("이경", "고막", "외이도", "귀 진찰", "귀 안", "귀 속", "귀 내시경", "소리굽쇠", "음차", "웨버", "린네",
                "weber", "rinne", "tuning fork", "otoscop", "tympanic", "ear exam")
_KW_AUDIOMETRY = ("청력 검사", "청력검사", "순음", "청각 검사", "청력 측정", "어음", "audiometr", "audiogram", "pure tone",
                  "pure-tone", "hearing test")
_KW_RETROCOCHLEAR = ("mri", "자기공명", "내이도", "청신경", "소뇌교각", "abr", "청성 뇌간", "청성뇌간", "뇌간 유발",
                     "auditory brainstem", "internal auditory")
_KW_COAG = ("pt/inr", "pt 검사", "pt,", "pt와", "prothrombin", "프로트롬빈", "aptt", "ptt", "부분 트롬보플라스틴",
            "응고 검사", "응고검사", "혈액응고", "혈액 응고", "coagulation", "inr")
_KW_CK = ("ck 검사", "ck(", "(ck", "ck)", "ck 수치", "혈청 ck", "ck/", "/ck", "ck와", "ck,", "cpk", "creatine kinase",
          "크레아틴 키나", "크레아틴키나", "근육 효소", "근효소", "근육효소")
_KW_INFANT_ABD_IMAGING = ("복부 x", "복부x", "복부 엑스", "복부 단순", "복부 방사선", "복부 촬영", "abdominal x",
                          "abdominal radiograph", "abdominal film", "kub", "상부 위장관 조영", "상부위장관 조영",
                          "위장관 조영", "upper gi", "ugi", "위장관 투시", "fluoroscop")

PROTOCOLS = PROTOCOLS + (
    Protocol(
        category="urticaria_chronic", name_ko=CATEGORY_NAMES["urticaria_chronic"],
        cant_miss=("두드러기 혈관염", "자가염증 증후군(슈니츨러 증후군 등)", "혈관부종(ACE 억제제·유전성)",
                   "갑상선 자가면역 질환"),
        checks=(
            Check("cbc_diff", "일반혈액검사(백혈구 감별계산 포함)", "test", _KW_CBC, G_URTICARIA, "primary",
                  when="6주 이상 지속되는 두드러기",
                  note="EAACI/GA2LEN 2021 (Allergy 2022), section 4.2 and Table 8, read in the EDF-hosted PDF: 'We "
                  "recommend limited investigations. Basic tests include differential blood count, CRP and/or "
                  "ESR' (consensus, strong); total IgE and IgG anti-TPO only in specialist care; no routine tests "
                  "in acute urticaria. Chronic = > 6 weeks; the chronic-pruritus panel does not apply."),
            Check("crp_esr", "CRP 및/또는 적혈구침강속도(ESR)", "test", _KW_CRP_ESR, G_URTICARIA, "primary",
                  when="6주 이상 지속되는 두드러기"),
        ),
    ),
    Protocol(
        category="hearing_loss", name_ko=CATEGORY_NAMES["hearing_loss"],
        cant_miss=("돌발성 감각신경성 난청", "청신경초종(후미로 병변)", "뇌졸중(전정·청각 동반 증상)", "진주종"),
        checks=(
            Check("ear_exam", "이경 검사 + 소리굽쇠 검사(전음성 vs 감각신경성 구분)", "exam", _KW_EAR_EXAM, G_SSNHL,
                  "primary", min_age=18, predicate="sudden_hearing", when="갑자기 생긴 난청(성인)",
                  note="AAO-HNS 2019 KAS 1 (strong, abstract): distinguish sensorineural from conductive hearing "
                  "loss at first presentation. Otoscopy + tuning fork as the means is the guideline body per "
                  "reviewer knowledge (full text not read)."),
            Check("ear_exam", "이경 검사(귀지·감염·고막 이상) + 소리굽쇠 검사", "exam", _KW_EAR_EXAM, G_ARHL, "primary",
                  min_age=18, when="서서히 진행한 난청(성인)",
                  note="AAO-HNS 2024 ARHL KAS 2 (abstract): if hearing loss is suggested, examine the ear canal and "
                  "tympanic membrane with otoscopy. Guideline population is >= 50 years; applied to adults >= 18."),
            Check("audiometry", "청력 검사(순음 청력검사, 가능한 빨리)", "test", _KW_AUDIOMETRY, G_SSNHL, "primary",
                  min_age=18, predicate="sudden_hearing", when="갑자기 생긴 난청(성인, 발병 14일 이내 권고)",
                  note="AAO-HNS 2019 KAS 4 (abstract): obtain audiometry as soon as possible (within 14 days of "
                  "onset) to confirm SSNHL. KAS 3/5 (strong, against): no routine head CT, no routine labs."),
            Check("audiometry", "청력 검사(순음 청력검사)", "test", _KW_AUDIOMETRY, G_ARHL, "primary", min_age=18,
                  when="서서히 진행한 난청(성인)",
                  note="AAO-HNS 2024 ARHL KAS 4 (strong, abstract): if screening suggests hearing loss, obtain an "
                  "audiogram. Population >= 50 years; applied to adults >= 18 (our extension)."),
            Check("retrocochlear_imaging", "후미로 병변 평가(내이도 MRI 또는 청성뇌간반응)", "test", _KW_RETROCOCHLEAR,
                  G_SSNHL, "primary", min_age=18, predicate="sudden_hearing", when="갑자기 생긴 감각신경성 난청(성인)",
                  note="AAO-HNS 2019 KAS 6 (abstract): evaluate SSNHL for retrocochlear pathology with MRI or "
                  "auditory brainstem response."),
            Check("retrocochlear_imaging", "비대칭(한쪽) 난청의 후미로 병변 평가(내이도 조영증강 MRI)", "test",
                  _KW_RETROCOCHLEAR, G_HEARING_IMAGING, "unverified", min_age=18, predicate="retrocochlear",
                  when="한쪽 또는 비대칭 감각신경성 난청(성인)",
                  note="Citation verified (PubMed 30392601); the abstract has no variant ratings. MRI head/IAC for "
                  "asymmetric SNHL as 'usually appropriate' is reviewer knowledge. AAO-HNS 2024 ARHL KAS 5 (read): "
                  "evaluate or refer significant asymmetric hearing loss."),
        ),
    ),
    Protocol(
        category="neck_mass", name_ko=CATEGORY_NAMES["neck_mass"],
        cant_miss=("두경부 편평세포암 전이", "림프종", "갑상선암", "침샘 종양", "결핵성 림프절염"),
        checks=(
            Check("malignancy_history", "악성 위험 문진(지속 기간·크기 변화, 감염 증상, 흡연·음주, 체중 감소, 삼킴 곤란·"
                  "쉰 목소리·귀 통증)", "ask",
                  ("흡연", "담배", "음주", "술을", "술은", "체중 감소", "체중이 줄", "살이 빠", "삼킴", "삼키기", "쉰 목소리",
                   "목소리 변화", "목소리가", "귀 통증", "귀가 아", "크기 변화", "커졌", "smok", "tobacco", "alcohol",
                   "weight loss", "dysphagia", "hoarse", "otalgia"),
                  G_NECK_MASS, "primary", min_age=18, when="성인 경부 종괴",
                  note="AAO-HNS 2017 statements 1-3 (abstract): identify increased malignancy risk from history "
                  "(no infectious cause, >= 2 weeks without fluctuation or uncertain duration), exam "
                  "(fixed, firm, > 1.5 cm, ulcerated) and other suspicious findings. The listed history items are "
                  "the usual head-and-neck cancer clues, wording ours. Guideline excludes children."),
            Check("neck_imaging", "조영증강 경부 CT(또는 MRI)", "test",
                  ("경부 ct", "목 ct", "neck ct", "ct neck", "ct of the neck", "경부 mri", "목 mri", "neck mri",
                   "mri neck", "경부 전산화", "경부 조영"),
                  G_NECK_MASS, "primary", min_age=18, predicate="neck_mass_risk",
                  when="악성 위험 증가(2주 이상 지속 또는 기간 불명, 감염 원인 없음)",
                  note="AAO-HNS 2017 (strong, abstract): order neck CT (or MRI) with contrast for a neck mass at "
                  "increased risk for malignancy; FNA (not open biopsy) when the diagnosis remains uncertain; no "
                  "routine antibiotics without signs of bacterial infection."),
        ),
    ),
    Protocol(
        category="bleeding", name_ko=CATEGORY_NAMES["bleeding"],
        cant_miss=("혈우병·폰빌레브란트병", "백혈병·재생불량성 빈혈", "면역 혈소판 감소증", "혈전성 미세혈관병증(TTP/HUS)",
                   "파종성 혈관내 응고(DIC)", "항응고제 과다"),
        checks=(
            Check("platelet_count", "일반혈액검사(혈소판 수)", "test", _KW_CBC, G_HEMOPHILIA, "primary",
                  when="쉽게 드는 멍·잦은 코피·시술 후 지혈 안 됨 등 출혈 경향",
                  note="WFH 2020 chapter 3 'Screening tests' (read in the WFH PDF, p. 41): platelet count, PT and "
                  "APTT may be used to screen a patient suspected of having a bleeding disorder; Table 3-1 "
                  "interprets the pattern. Normal results do not exclude mild disorders (e.g. VWD)."),
            Check("pt_aptt", "PT·aPTT(응고 선별검사)", "test", _KW_COAG, G_HEMOPHILIA, "primary",
                  when="쉽게 드는 멍·잦은 코피·시술 후 지혈 안 됨 등 출혈 경향"),
        ),
    ),
    Protocol(
        category="chronic_weakness", name_ko=CATEGORY_NAMES["chronic_weakness"],
        cant_miss=("척수 압박·척수병증", "근위축성 측삭 경화증", "염증성 근병증(다발성 근염·피부근염)", "중증 근무력증",
                   "길랭-바레 증후군/CIDP", "전해질 이상(저칼륨혈증)·갑상선 질환"),
        checks=(
            Check("neuro_exam", "신경학적 진찰(근력 분포, 반사, 감각, 상위/하위 운동신경원 징후)", "exam",
                  _KW_NEURO_EXAM + ("근력 검사", "근력 평가", "도수 근력", "반사", "reflex", "muscle strength",
                                    "motor exam"),
                  G_IIM_CRITERIA, "unverified", when="2주 이상 지속되는 사지 근력 저하",
                  note="Basic assessment. The EULAR/ACR 2017 criteria (read in PMC5736307) need *objective* "
                  "symmetric proximal weakness, i.e. an examination; they are classification, not diagnostic, "
                  "criteria. No diagnostic guideline for chronic weakness was found; the check is reviewer "
                  "knowledge."),
            Check("ck", "혈청 크레아틴 키나아제(CK)", "test", _KW_CK, G_IIM_CRITERIA, "unverified",
                  predicate="myopathy_pattern", when="감각 증상 없는 대칭성(양측) 근력 저하",
                  note="EULAR/ACR 2017 criteria item (read): elevated serum CK, LDH, AST or ALT. Classification "
                  "criteria, not a diagnostic guideline; ordering CK for chronic symmetric weakness without "
                  "sensory symptoms is reviewer knowledge."),
        ),
    ),
    Protocol(
        category="bilious_vomiting", name_ko=CATEGORY_NAMES["bilious_vomiting"],
        cant_miss=("장회전 이상과 중장 염전", "십이지장 폐쇄·협착", "공장·회장 폐쇄", "히르슈슈프룽병", "괴사성 장염"),
        checks=(
            Check("urgent_abd_imaging", "즉시 복부 X선 + 상부위장관 조영술(장회전 이상 배제)", "test",
                  _KW_INFANT_ABD_IMAGING, G_INFANT_VOMITING, "secondary",
                  when="생후 3개월 이하 영아의 담즙성(초록색) 구토",
                  note="ACR AC Vomiting in Infants 2020 (narrative page read through an automated summary, "
                  "variant titles quoted): bilious vomiting in an infant > 2 days old (suspected malrotation) -> "
                  "fluoroscopy upper GI series usually appropriate; vomiting in the first 2 days -> abdominal "
                  "radiograph first, then UGI (double bubble) or contrast enema (distal obstruction). Population "
                  "< 3 months. Urgency of malrotation/volvulus is reviewer knowledge."),
        ),
    ),
)

PROTOCOLS_BY_CATEGORY: dict[str, Protocol] = {p.category: p for p in PROTOCOLS}


# --------------------------------------------------------------------------------------------
# Lookup / rendering
# --------------------------------------------------------------------------------------------


def protocols_for(text: str, context: str = "") -> list[Protocol]:
    return [PROTOCOLS_BY_CATEGORY[c] for c in detect_categories(text, context)
            if c in PROTOCOLS_BY_CATEGORY]


def cant_miss_for(text: str, context: str = "") -> list[str]:
    """Can't-miss diagnoses for the text, deduplicated in order."""
    out: list[str] = []
    for p in protocols_for(text, context):
        out += [dx for dx in p.cant_miss if dx not in out]
    return out


def must_checks_for(text: str, context: str = "") -> list[Check]:
    """Applicable minimum checks, deduplicated by check id.

    Categories come from `text` only (the chief complaint): matching whole conversations over-triggers, e.g. a
    pertinent negative like "숨은 안 차요" would add the dyspnea protocol. The one exception: acute dizziness in
    `text` opens the stroke protocol when `context` reveals a vascular risk factor. Conditional checks are triggered by
    `text` + `context` (facts learned later, e.g. pregnancy or a thunderclap onset); a trigger inside a negated
    clause ("등이 찢어지는 느낌은 아니에요") does not count. Acute-only checks use the duration of `text` only.
    """
    trigger_text = f"{text}. {context}"  # sentence break so a negation in context can't reach back into text
    out: dict[str, Check] = {}
    for p in protocols_for(text, context):
        for c in p.checks:
            if c.id not in out and c.applies(trigger_text, cc=text):
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
