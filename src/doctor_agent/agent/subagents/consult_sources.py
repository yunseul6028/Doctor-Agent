"""Sources for the specialist consult content (agent/subagents/consult.py) added in the 2026-09-29 verification pass.

Owned by clinical-strategist. Each citation's bibliographic data (authors, title, journal, volume/pages, DOI, PMID)
was generated from PubMed E-utilities esummary on 2026-09-29; what was read (abstract, or full text where stated) and
which claim it supports is in the matching SpecialtySpec.note and in docs/licenses.md. Facts only; no text copied.
"""
from __future__ import annotations

from doctor_agent.knowledge.clinical_rules import Citation

C_ACOG_THROMBOCYTOPENIA = Citation(
    "American College of Obstetricians and Gynecologists.",
    "ACOG Practice Bulletin No. 207: Thrombocytopenia in Pregnancy",
    "Obstet Gynecol", 2019, "133(3):e181-e193",
    doi="10.1097/AOG.0000000000003100", pmid="30801473", verified=True, short_author="ACOG 임신 중 혈소판감소 지침",
)

C_SIBAI_POSTPARTUM = Citation(
    "Sibai BM.",
    "Etiology and management of postpartum hypertension-preeclampsia",
    "Am J Obstet Gynecol", 2012, "206(6):470-475",
    doi="10.1016/j.ajog.2011.09.002", pmid="21963308", verified=True,
)

C_KAMEL_POSTPARTUM = Citation(
    "Kamel H, Navi BB, Sriram N, et al.",
    "Risk of a thrombotic event after the 6-week postpartum period",
    "N Engl J Med", 2014, "370(14):1307-1315",
    doi="10.1056/NEJMoa1311485", pmid="24524551", verified=True,
)

C_CANTO_NO_CHEST_PAIN = Citation(
    "Canto JG, Shlipak MG, Rogers WJ, et al.",
    "Prevalence, clinical characteristics, and mortality among patients with myocardial infarction "
    "presenting without chest pain",
    "JAMA", 2000, "283(24):3223-3229",
    doi="10.1001/jama.283.24.3223", pmid="10866870", verified=True,
)

C_HANSEN_AAS_MISDX = Citation(
    "Hansen MS, Nogareda GJ, Hutchison SJ.",
    "Frequency of and inappropriate treatment of misdiagnosis of acute aortic dissection",
    "Am J Cardiol", 2007, "99(6):852-856",
    doi="10.1016/j.amjcard.2006.10.055", pmid="17350381", verified=True,
)

C_IMAZIO_TAMPONADE = Citation(
    "Imazio M.",
    "[Ten questions about cardiac tamponade]",
    "G Ital Cardiol (Rome)", 2018, "19(9):471-478",
    doi="10.1714/2951.29665", pmid="30087507", verified=True,
)

C_SLIWA_PPCM = Citation(
    "Sliwa K, Hilfiker-Kleiner D, Petrie MC, et al.",
    "Current state of knowledge on aetiology, diagnosis, management, and therapy of peripartum "
    "cardiomyopathy: a position statement from the Heart Failure Association of the European Society of "
    "Cardiology Working Group on peripartum cardiomyopathy",
    "Eur J Heart Fail", 2010, "12(8):767-778",
    doi="10.1093/eurjhf/hfq120", pmid="20675664", verified=True,
)

C_FRIEDMAN_PEDS_CHEST_PAIN = Citation(
    "Friedman KG, Kane DA, Rathod RH, et al.",
    "Management of pediatric chest pain using a standardized assessment and management plan",
    "Pediatrics", 2011, "128(2):239-245",
    doi="10.1542/peds.2011-0141", pmid="21746719", verified=True,
)

C_NORMAN_FEVER_ELDERLY = Citation(
    "Norman DC.",
    "Fever in the elderly",
    "Clin Infect Dis", 2000, "31(1):148-151",
    doi="10.1086/313896", pmid="10913413", verified=True,
)

C_THOMPSON_MENINGOCOCCAL = Citation(
    "Thompson MJ, Ninis N, Perera R, et al.",
    "Clinical recognition of meningococcal disease in children and adolescents",
    "Lancet", 2006, "367(9508):397-403",
    doi="10.1016/S0140-6736(06)67932-4", pmid="16458763", verified=True,
)

C_ACOG_LISTERIA = Citation(
    "American College of Obstetricians and Gynecologists.",
    "Committee Opinion No. 614: Management of pregnant women with presumptive exposure to Listeria "
    "monocytogenes",
    "Obstet Gynecol", 2014, "124(6):1241-1244",
    doi="10.1097/01.AOG.0000457501.73326.6c", pmid="25411758", verified=True, short_author="ACOG 리스테리아 지침",
)

C_HILL_PYELONEPHRITIS = Citation(
    "Hill JB, Sheffield JS, McIntire DD, et al.",
    "Acute pyelonephritis in pregnancy",
    "Obstet Gynecol", 2005, "105(1):18-23",
    doi="10.1097/01.AOG.0000149154.96285.a0", pmid="15625136", verified=True,
)

C_LYON_ELDERLY_ABDOMEN = Citation(
    "Lyon C, Clark DC.",
    "Diagnosis of acute abdominal pain in older patients",
    "Am Fam Physician", 2006, "74(9):1537-1544",
    pmid="17111893", verified=True,
)

C_WEINER_STEROID_PERFORATION = Citation(
    "Weiner HL, Rezai AR, Cooper PR.",
    "Sigmoid diverticular perforation in neurosurgical patients receiving high-dose corticosteroids",
    "Neurosurgery", 1993, "33(1):40-43",
    doi="10.1227/00006123-199307000-00006", pmid="8355846", verified=True,
)

C_HOM_INTUSSUSCEPTION = Citation(
    "Hom J, Kaplan C, Fowler S, et al.",
    "Evidence-Based Diagnostic Test Accuracy of History, Physical Examination, and Imaging for "
    "Intussusception: A Systematic Review and Meta-analysis",
    "Pediatr Emerg Care", 2022, "38(1):e225-e230",
    doi="10.1097/PEC.0000000000002224", pmid="32941364", verified=True,
)

C_OKANO_STROKE_MIMICS = Citation(
    "Okano Y, Ishimatsu K, Kato Y, et al.",
    "Clinical features of stroke mimics in the emergency department",
    "Acute Med Surg", 2018, "5(3):241-248",
    doi="10.1002/ams2.338", pmid="29988676", verified=True,
)

C_AHA_CVT = Citation(
    "Saposnik G, Bushnell C, Coutinho JM, et al.",
    "Diagnosis and Management of Cerebral Venous Thrombosis: A Scientific Statement From the American "
    "Heart Association",
    "Stroke", 2024, "55(3):e77-e90",
    doi="10.1161/STR.0000000000000456", pmid="38284265", verified=True, short_author="AHA 뇌정맥동 혈전증 성명",
)

C_GCA_FAST_TRACK = Citation(
    "Monti S, Águeda AF, Luqmani RA, et al.",
    "Systematic literature review informing the 2018 update of the EULAR recommendation for the "
    "management of large vessel vasculitis: focus on giant cell arteritis",
    "RMD Open", 2019, "5(2):e001003",
    doi="10.1136/rmdopen-2019-001003", pmid="31673411", verified=True,
)

C_SEPTIC_CRYSTAL = Citation(
    "Duangkum K, Saengmongkonpipat P, Tantiwong P, et al.",
    "Prevalence and characteristics of concomitant septic and crystal-induced arthritis: A hospital "
    "database and literature review",
    "Rheumatol Immunol Res", 2025, "6(3):179-187",
    doi="10.1515/rir-2025-0021", pmid="41050478", verified=True,
)

C_ANA_HEALTHY = Citation(
    "Tan EM, Feltkamp TE, Smolen JS, et al.",
    "Range of antinuclear antibodies in \"healthy\" individuals",
    "Arthritis Rheum", 1997, "40(9):1601-1611",
    doi="10.1002/art.1780400909", pmid="9324014", verified=True,
)

C_PULMONARY_RENAL = Citation(
    "Boyle N, O'Callaghan M, Ataya A, et al.",
    "Pulmonary renal syndrome: a clinical review",
    "Breathe (Sheff)", 2022, "18(4):220208",
    doi="10.1183/20734735.0208-2022", pmid="36865943", verified=True,
)

C_LN_PREGNANCY = Citation(
    "Gholizadeh Ghozloujeh Z, Singh T, Jhaveri KD, et al.",
    "Lupus nephritis: management challenges during pregnancy",
    "Front Nephrol", 2024, "4:1390783",
    doi="10.3389/fneph.2024.1390783", pmid="38895665", verified=True,
)

C_TERATOGENS = Citation(
    "Dathe K, Schaefer C.",
    "The Use of Medication in Pregnancy",
    "Dtsch Arztebl Int", 2019, "116(46):783-790",
    doi="10.3238/arztebl.2019.0783", pmid="31920194", verified=True,
)

C_ACOG_ADNEXAL_TORSION = Citation(
    "American College of Obstetricians and Gynecologists.",
    "Adnexal Torsion in Adolescents: ACOG Committee Opinion No, 783",
    "Obstet Gynecol", 2019, "134(2):e56-e63",
    doi="10.1097/AOG.0000000000003373", pmid="31348225", verified=True, short_author="ACOG 난소 염전 의견",
)

C_ABUSE_REPORTING = Citation(
    "Flaherty EG, Sege RD, Griffith J, et al.",
    "From suspicion of physical child abuse to reporting: primary care clinician decision-making",
    "Pediatrics", 2008, "122(3):611-619",
    doi="10.1542/peds.2007-2311", pmid="18676507", verified=True,
)

C_LAWTON_MSCC = Citation(
    "Lawton AJ, Lee KA, Cheville AL, et al.",
    "Assessment and Management of Patients With Metastatic Spinal Cord Compression: A Multidisciplinary "
    "Review",
    "J Clin Oncol", 2019, "37(1):61-71",
    doi="10.1200/JCO.2018.78.1211", pmid="30395488", verified=True,
)

C_RICE_SVC = Citation(
    "Rice TW, Rodriguez RM, Light RW.",
    "The superior vena cava syndrome: clinical characteristics and evolving etiology",
    "Medicine (Baltimore)", 2006, "85(1):37-42",
    doi="10.1097/01.md.0000198474.99876.f0", pmid="16523051", verified=True,
)

C_HCM_GUIDELINE = Citation(
    "El-Hajj Fuleihan G, Clines GA, Hu MI, et al.",
    "Treatment of Hypercalcemia of Malignancy in Adults: An Endocrine Society Clinical Practice Guideline",
    "J Clin Endocrinol Metab", 2023, "108(3):507-528",
    doi="10.1210/clinem/dgac621", pmid="36545746", verified=True, short_author="Endocrine Society 악성 고칼슘혈증 지침",
)

C_CLARKE_LEUKAEMIA = Citation(
    "Clarke RT, Van den Bruel A, Bankhead C, et al.",
    "Clinical presentation of childhood leukaemia: a systematic review and meta-analysis",
    "Arch Dis Child", 2016, "101(10):894-901",
    doi="10.1136/archdischild-2016-311251", pmid="27647842", verified=True,
)

C_HUS_LANCET = Citation(
    "Fakhouri F, Zuber J, Frémeaux-Bacchi V, et al.",
    "Haemolytic uraemic syndrome",
    "Lancet", 2017, "390(10095):681-696",
    doi="10.1016/S0140-6736(17)30062-4", pmid="28242109", verified=True,
)

C_FLC_SCREENING = Citation(
    "Katzmann JA, Kyle RA, Benson J, et al.",
    "Screening panels for detection of monoclonal gammopathies",
    "Clin Chem", 2009, "55(8):1517-1522",
    doi="10.1373/clinchem.2009.126664", pmid="19520758", verified=True,
)

C_NASON_TORSION = Citation(
    "Nason GJ, Tareen F, McLoughlin D, et al.",
    "Scrotal exploration for acute scrotal pain: a 10-year experience in two tertiary referral paediatric "
    "units",
    "Scand J Urol", 2013, "47(5):418-422",
    doi="10.3109/00365599.2012.752403", pmid="23281617", verified=True,
)

C_MONTAGUE_HYPERK_ECG = Citation(
    "Montague BT, Ouellette JR, Buller GK.",
    "Retrospective review of the frequency of ECG changes in hyperkalemia",
    "Clin J Am Soc Nephrol", 2008, "3(2):324-330",
    doi="10.2215/CJN.04611007", pmid="18235147", verified=True,
)

C_PEARLE_DRAINAGE = Citation(
    "Pearle MS, Pierce HL, Miller GL, et al.",
    "Optimal method of urgent decompression of the collecting system for obstruction and infection due to "
    "ureteral calculi",
    "J Urol", 1998, "160(4):1260-1264",
    pmid="9751331", verified=True,
)

C_CHAVEZ_RHABDO = Citation(
    "Chavez LO, Leon M, Einav S, et al.",
    "Beyond muscle destruction: a systematic review of rhabdomyolysis for clinical practice",
    "Crit Care", 2016, "20(1):135",
    doi="10.1186/s13054-016-1314-5", pmid="27301374", verified=True,
)

C_MARSTON_RAAA = Citation(
    "Marston WA, Ahlquist R, Johnson G Jr, et al.",
    "Misdiagnosis of ruptured abdominal aortic aneurysms",
    "J Vasc Surg", 1992, "16(1):17-22",
    doi="10.1067/mva.1992.34344", pmid="1619721", verified=True,
)

C_FAUNDES_HYDRONEPHROSIS = Citation(
    "Faúndes A, Brícola-Filho M, Pinto e Silva JL.",
    "Dilatation of the urinary tract during pregnancy: proposal of a curve of maximal caliceal diameter "
    "by gestational age",
    "Am J Obstet Gynecol", 1998, "178(5):1082-1086",
    doi="10.1016/s0002-9378(98)70552-6", pmid="9609588", verified=True,
)

C_AAP_UTI = Citation(
    "Subcommittee on Urinary Tract Infection, Steering Committee on Quality Improvement and Management; Roberts KB.",
    "Urinary tract infection: clinical practice guideline for the diagnosis and management of the initial "
    "UTI in febrile infants and children 2 to 24 months",
    "Pediatrics", 2011, "128(3):595-610",
    doi="10.1542/peds.2011-1330", pmid="21873693", verified=True, short_author="AAP 영아 요로감염 지침",
)

C_EGRIS = Citation(
    "Walgaard C, Lingsma HF, Ruts L, et al.",
    "Prediction of respiratory insufficiency in Guillain-Barré syndrome",
    "Ann Neurol", 2010, "67(6):781-787",
    doi="10.1002/ana.21976", pmid="20517939", verified=True,
)

C_MG_CRISIS = Citation(
    "Takahashi S, Kinno R.",
    "Management of Myasthenic Crisis and Emerging Roles of Molecularly Targeted Therapies: A Narrative "
    "Review",
    "Neurol Int", 2025, "17(10):163",
    doi="10.3390/neurolint17100163", pmid="41149784", verified=True,
)

C_CURTIS_MENINGITIS = Citation(
    "Curtis S, Stobart K, Vandermeer B, et al.",
    "Clinical features suggestive of meningitis in children: a systematic review of prospective data",
    "Pediatrics", 2010, "126(5):952-960",
    doi="10.1542/peds.2010-0277", pmid="20974781", verified=True,
)

C_ACR_APPENDICITIS_CHILD = Citation(
    "Expert Panel on Pediatric Imaging; Koberlein GC, Trout AT, et al.",
    "ACR Appropriateness Criteria Suspected Appendicitis-Child",
    "J Am Coll Radiol", 2019, "16(5S):S252-S263",
    doi="10.1016/j.jacr.2019.02.022", pmid="31054752", verified=True,
)

# --- 2026-09-29 second pass: sources for the claims that were still reviewer knowledge in resp_id, endo_metab and psych
# (bibliographic data generated from PubMed E-utilities esummary on 2026-09-29; what was read is in each spec note).

C_TIBBALLS_EPIGLOTTITIS = Citation(
    "Tibballs J, Watson T.",
    "Symptoms and signs differentiating croup and epiglottitis",
    "J Paediatr Child Health", 2011, "47(3):77-82",
    doi="10.1111/j.1440-1754.2010.01892.x", pmid="21091577", verified=True,
)

C_EPIGLOTTITIS_STATPEARLS = Citation(
    "Sutton AE, Waseem M.",
    "Epiglottitis",
    "StatPearls [Internet]", 2024, "Treasure Island (FL): StatPearls Publishing",
    pmid="28613691", verified=True,
)

C_ACOG_IAI = Citation(
    "American College of Obstetricians and Gynecologists' Committee on Obstetric Practice.",
    "Committee Opinion No. 712: Intrapartum Management of Intraamniotic Infection",
    "Obstet Gynecol", 2017, "130(2):e95-e101",
    doi="10.1097/AOG.0000000000002236", pmid="28742677", verified=True,
)

C_GC_ADRENAL_INSUFFICIENCY = Citation(
    "Beuschlein F, Else T, Bancos I, et al.",
    "European Society of Endocrinology and Endocrine Society Joint Clinical Guideline: Diagnosis and therapy of "
    "glucocorticoid-induced adrenal insufficiency",
    "Eur J Endocrinol", 2024, "190(5):G25-G51",
    doi="10.1093/ejendo/lvae029", pmid="38714321", verified=True,
)

C_AKAMIZU_STORM = Citation(
    "Akamizu T.",
    "Thyroid Storm: A Japanese Perspective",
    "Thyroid", 2018, "28(1):32-40",
    doi="10.1089/thy.2017.0243", pmid="28899229", verified=True,
)

C_ATA_HYPOTHYROIDISM = Citation(
    "Jonklaas J, Bianco AC, Bauer AJ, et al.",
    "Guidelines for the treatment of hypothyroidism: prepared by the american thyroid association task force on "
    "thyroid hormone replacement",
    "Thyroid", 2014, "24(12):1670-751",
    doi="10.1089/thy.2014.0028", pmid="25266247", verified=True,
)

C_WALKER_HYPERCALCEMIA = Citation(
    "Walker MD, Shane E.",
    "Hypercalcemia: A Review",
    "JAMA", 2022, "328(16):1624-1636",
    doi="10.1001/jama.2022.18331", pmid="36282253", verified=True,
)

C_HOORN_HYPONATREMIA = Citation(
    "Hoorn EJ, Zietse R.",
    "Diagnosis and Treatment of Hyponatremia: Compilation of the Guidelines",
    "J Am Soc Nephrol", 2017, "28(5):1340-1349",
    doi="10.1681/ASN.2016101139", pmid="28174217", verified=True,
)

C_SEARCH_DKA_YOUTH = Citation(
    "Jensen ET, Stafford JM, Saydah S, et al.",
    "Increase in Prevalence of Diabetic Ketoacidosis at Diagnosis Among Youth With Type 1 Diabetes: The SEARCH "
    "for Diabetes in Youth Study",
    "Diabetes Care", 2021, "44(7):1573-1578",
    doi="10.2337/dc20-0389", pmid="34099516", verified=True,
)

C_GLASER_CEREBRAL_EDEMA = Citation(
    "Glaser N, Barnett P, McCaslin I, et al.",
    "Risk factors for cerebral edema in children with diabetic ketoacidosis. The Pediatric Emergency Medicine "
    "Collaborative Research Committee of the American Academy of Pediatrics",
    "N Engl J Med", 2001, "344(4):264-9",
    doi="10.1056/NEJM200101253440404", pmid="11172153", verified=True,
)

C_MUIR_CEREBRAL_EDEMA = Citation(
    "Muir AB, Quisling RG, Yang MC, et al.",
    "Cerebral edema in childhood diabetic ketoacidosis: natural history, radiographic findings, and early "
    "identification",
    "Diabetes Care", 2004, "27(7):1541-6",
    doi="10.2337/diacare.27.7.1541", pmid="15220225", verified=True,
)

C_EFNS_WERNICKE = Citation(
    "Galvin R, Bråthen G, Ivashynka A, et al.",
    "EFNS guidelines for diagnosis, therapy and prevention of Wernicke encephalopathy",
    "Eur J Neurol", 2010, "17(12):1408-18",
    doi="10.1111/j.1468-1331.2010.03153.x", pmid="20642790", verified=True,
)

C_BERGINK_POSTPARTUM = Citation(
    "Bergink V, Rasgon N, Wisner KL.",
    "Postpartum Psychosis: Madness, Mania, and Melancholia in Motherhood",
    "Am J Psychiatry", 2016, "173(12):1179-1188",
    doi="10.1176/appi.ajp.2016.16040454", pmid="27609245", verified=True,
)

C_AAP_ADOLESCENT_SUICIDE = Citation(
    "Shain B.",
    "Suicide and Suicide Attempts in Adolescents",
    "Pediatrics", 2016, "138(1):e20161420",
    doi="10.1542/peds.2016-1420", pmid="27354459", verified=True,
)

C_PHEO_REVIEW = Citation(
    "Aygun N, Uludag M.",
    "Pheochromocytoma and Paraganglioma: From Clinical Findings to Diagnosis",
    "Sisli Etfal Hastan Tip Bul", 2020, "54(3):271-280",
    doi="10.14744/SEMB.2020.14826", pmid="33312023", verified=True,
)

C_DAVIES_GRAVES = Citation(
    "Davies TF, Andersen S, Latif R, et al.",
    "Graves' disease",
    "Nat Rev Dis Primers", 2020, "6(1):52",
    doi="10.1038/s41572-020-0184-y", pmid="32616746", verified=True,
)

C_BRIET_APOPLEXY = Citation(
    "Briet C, Salenave S, Bonneville JF, et al.",
    "Pituitary Apoplexy",
    "Endocr Rev", 2015, "36(6):622-45",
    doi="10.1210/er.2015-1042", pmid="26414232", verified=True,
)

CONSULT_SOURCES: tuple[Citation, ...] = (
    C_ACR_APPENDICITIS_CHILD, C_EGRIS, C_MG_CRISIS, C_CURTIS_MENINGITIS, C_ACOG_THROMBOCYTOPENIA, C_SIBAI_POSTPARTUM, C_KAMEL_POSTPARTUM, C_CANTO_NO_CHEST_PAIN, C_HANSEN_AAS_MISDX,
    C_IMAZIO_TAMPONADE, C_SLIWA_PPCM, C_FRIEDMAN_PEDS_CHEST_PAIN, C_NORMAN_FEVER_ELDERLY, C_THOMPSON_MENINGOCOCCAL,
    C_ACOG_LISTERIA, C_HILL_PYELONEPHRITIS, C_LYON_ELDERLY_ABDOMEN, C_WEINER_STEROID_PERFORATION,
    C_HOM_INTUSSUSCEPTION, C_OKANO_STROKE_MIMICS, C_AHA_CVT, C_GCA_FAST_TRACK, C_SEPTIC_CRYSTAL, C_ANA_HEALTHY,
    C_PULMONARY_RENAL, C_LN_PREGNANCY, C_TERATOGENS, C_ACOG_ADNEXAL_TORSION, C_ABUSE_REPORTING, C_LAWTON_MSCC,
    C_RICE_SVC, C_HCM_GUIDELINE, C_CLARKE_LEUKAEMIA, C_HUS_LANCET, C_FLC_SCREENING, C_NASON_TORSION,
    C_MONTAGUE_HYPERK_ECG, C_PEARLE_DRAINAGE, C_CHAVEZ_RHABDO, C_MARSTON_RAAA, C_FAUNDES_HYDRONEPHROSIS, C_AAP_UTI,
    # 2026-09-29 second pass
    C_TIBBALLS_EPIGLOTTITIS, C_EPIGLOTTITIS_STATPEARLS, C_ACOG_IAI, C_GC_ADRENAL_INSUFFICIENCY, C_AKAMIZU_STORM,
    C_ATA_HYPOTHYROIDISM, C_WALKER_HYPERCALCEMIA, C_HOORN_HYPONATREMIA, C_SEARCH_DKA_YOUTH, C_GLASER_CEREBRAL_EDEMA,
    C_MUIR_CEREBRAL_EDEMA, C_EFNS_WERNICKE, C_BERGINK_POSTPARTUM, C_AAP_ADOLESCENT_SUICIDE, C_PHEO_REVIEW,
    C_DAVIES_GRAVES, C_BRIET_APOPLEXY,
)
