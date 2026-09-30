"""Curated test/lab/imaging finding → disease links (source tag "curated"; our own wording, stdlib only).

Why: the open sources in the KB (DDXPlus, Disease Ontology, Wikidata P780/P923, MedlinePlus) link diseases to symptoms
and to *names* of tests ("CT scan"), never to *results* ("lipase 3× ULN", "AMA positive"). Decisive results are what
separate diseases late in an interview, so this table adds them.

    FINDINGS   finding concepts: id, Korean/English label, how to detect them in a finding text, and their links
    DX         disease key → KB profile id (Disease Ontology / Wikidata / DDXPlus id; checked by scripts/build_kb.py)
    REFS       reference key → citation (+ PMID). Every link cites one; "textbook" marks a standard textbook fact that we
               did not tie to a specific guideline (listed as unverified in docs/data-sources.md)
    detect(text, context)  → {finding id: (polarity, numeric)}; polarity +1 abnormal/positive, -1 normal/negative

Link weights: 3 = diagnostic criterion / (near-)pathognomonic result, 2 = strong support, 1 = nonspecific support.
"R" = a normal/negative result argues against the disease (used as a penalty); results without R never penalise.
Reference ranges and cut-offs are common adult values (our choice, not from one lab); see each REFS entry.
A reference range printed next to a value ("750 ng/mL (정상 <500)", "(정상 40 미만)", "(정상치 500)", "(경미한 상승)"; read
by knowledge/refrange.py, the reader nlp/findings.py uses too) wins over the default threshold and is compared in
the printed unit; cut-off findings (Finding.cutoff, e.g. "AST > 1000") only accept "within the range" from it.
The facts are standard clinical knowledge; the wording, regexes and weights are ours (no text copied from any source).
Build: scripts/build_kb.py writes the links into data/kb/kb.json.gz (profile field "findings_from_tests").
Runtime: kb.py uses detect() on the agent's findings. No network, no LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from doctor_agent.knowledge import refrange

SOURCE = "curated"

# ------------------------------------------------------------------ references (verified via PubMed E-utilities)
REFS: dict[str, dict] = {
    "atlanta": {"cite": "Banks PA, et al. Classification of acute pancreatitis 2012: revision of the Atlanta classification. Gut 2013", "pmid": "23100216"},
    "udmi4": {"cite": "Thygesen K, et al. Fourth universal definition of myocardial infarction (2018). Eur Heart J 2019", "pmid": "30165617"},
    "easl_pbc": {"cite": "EASL Clinical Practice Guidelines: primary biliary cholangitis. J Hepatol 2017", "pmid": "28427765"},
    "aasld_pbc": {"cite": "Lindor KD, et al. Primary biliary cholangitis: 2018 AASLD practice guidance. Hepatology 2019", "pmid": "30070375"},
    "sle2019": {"cite": "Aringer M, et al. 2019 EULAR/ACR classification criteria for SLE. Arthritis Rheumatol 2019", "pmid": "31385462"},
    "ada2024": {"cite": "ADA. 2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes-2024. Diabetes Care 2024", "pmid": "38078589"},
    "dka": {"cite": "Kitabchi AE, et al. Hyperglycemic crises in adult patients with diabetes. Diabetes Care 2009", "pmid": "19564476"},
    "idsa_uti": {"cite": "Gupta K, et al. IDSA/ESCMID guidelines: acute uncomplicated cystitis and pyelonephritis in women. Clin Infect Dis 2011", "pmid": "21292654"},
    "esc_pe": {"cite": "Konstantinides SV, et al. 2019 ESC Guidelines for acute pulmonary embolism. Eur Heart J 2020", "pmid": "31504429"},
    "ash_vte": {"cite": "Lim W, et al. ASH 2018 guidelines for VTE: diagnosis of venous thromboembolism. Blood Adv 2018", "pmid": "30482764"},
    "esc_aorta": {"cite": "Erbel R, et al. 2014 ESC Guidelines on aortic diseases. Eur Heart J 2014", "pmid": "25173340"},
    "isth_dic": {"cite": "Taylor FB Jr, et al. ISTH criteria and scoring system for DIC. Thromb Haemost 2001", "pmid": "11816725"},
    "aih": {"cite": "Hennes EM, et al. Simplified criteria for the diagnosis of autoimmune hepatitis. Hepatology 2008", "pmid": "18537184"},
    "celiac": {"cite": "Rubio-Tapia A, et al. ACG guidelines update: celiac disease. Am J Gastroenterol 2023", "pmid": "36602836"},
    "ata_hyper": {"cite": "Ross DS, et al. 2016 ATA guidelines for hyperthyroidism and other causes of thyrotoxicosis. Thyroid 2016", "pmid": "27521067"},
    "ata_hypo": {"cite": "Jonklaas J, et al. ATA guidelines for the treatment of hypothyroidism. Thyroid 2014", "pmid": "25266247"},
    "ra2010": {"cite": "Aletaha D, et al. 2010 ACR/EULAR rheumatoid arthritis classification criteria. Arthritis Rheum 2010", "pmid": "20872595"},
    "gout2015": {"cite": "Neogi T, et al. 2015 ACR/EULAR gout classification criteria. Arthritis Rheumatol 2015", "pmid": "26352873"},
    "cppd": {"cite": "Zhang W, et al. EULAR recommendations for calcium pyrophosphate deposition, part I. Ann Rheum Dis 2011", "pmid": "21216817"},
    "septic_arthritis": {"cite": "Margaretten ME, et al. Does this adult patient have septic arthritis? JAMA 2007", "pmid": "17405973"},
    "aasld_hbv": {"cite": "Terrault NA, et al. AASLD 2018 hepatitis B guidance. Hepatology 2018", "pmid": "29405329"},
    "hcv": {"cite": "Bhattacharya D, et al. Hepatitis C Guidance 2023 update: AASLD-IDSA recommendations. Clin Infect Dis 2023", "pmid": "37229695"},
    "tg18_chole": {"cite": "Yokoe M, et al. Tokyo Guidelines 2018: diagnostic criteria of acute cholecystitis. J Hepatobiliary Pancreat Sci 2018", "pmid": "29032636"},
    "tg18_cholangitis": {"cite": "Kiriyama S, et al. Tokyo Guidelines 2018: diagnostic criteria of acute cholangitis. J Hepatobiliary Pancreat Sci 2018", "pmid": "29032610"},
    "psc": {"cite": "Bowlus CL, et al. AASLD practice guidance on primary sclerosing cholangitis and cholangiocarcinoma. Hepatology 2023", "pmid": "36083140"},
    "wses_app": {"cite": "Di Saverio S, et al. 2020 update of the WSES Jerusalem guidelines: acute appendicitis. World J Emerg Surg 2020", "pmid": "32295644"},
    "mening": {"cite": "van de Beek D, et al. ESCMID guideline: acute bacterial meningitis. Clin Microbiol Infect 2016", "pmid": "27062097"},
    "esc_hf": {"cite": "McDonagh TA, et al. 2021 ESC Guidelines for acute and chronic heart failure. Eur Heart J 2021", "pmid": "34447992"},
    "sepsis3": {"cite": "Singer M, et al. Sepsis-3 definitions. JAMA 2016", "pmid": "26903338"},
    "aki": {"cite": "Khwaja A. KDIGO clinical practice guidelines for acute kidney injury. Nephron Clin Pract 2012", "pmid": "22890468"},
    "aga_ida": {"cite": "Ko CW, et al. AGA guidelines: GI evaluation of iron deficiency anemia. Gastroenterology 2020", "pmid": "32810434"},
    "b12": {"cite": "Stabler SP. Vitamin B12 deficiency. N Engl J Med 2013", "pmid": "23301732"},
    "ttp": {"cite": "Zheng XL, et al. ISTH guidelines for the diagnosis of TTP. J Thromb Haemost 2020", "pmid": "32914582"},
    "aiha": {"cite": "Jäger U, et al. Diagnosis and treatment of autoimmune hemolytic anemia in adults: first international consensus. Blood Rev 2020", "pmid": "31839434"},
    "mg": {"cite": "Sanders DB, et al. International consensus guidance for myasthenia gravis. Neurology 2016", "pmid": "27358333"},
    "gbs": {"cite": "Leonhard SE, et al. Diagnosis and management of Guillain-Barré syndrome in ten steps. Nat Rev Neurol 2019", "pmid": "31541214"},
    "aha_stroke": {"cite": "Powers WJ, et al. Early management of acute ischemic stroke: 2019 update. Stroke 2019", "pmid": "31662037"},
    "ich": {"cite": "Greenberg SM, et al. 2022 AHA/ASA guideline: spontaneous intracerebral hemorrhage. Stroke 2022", "pmid": "35579034"},
    "sah": {"cite": "Hoh BL, et al. 2023 AHA/ASA guideline: aneurysmal subarachnoid hemorrhage. Stroke 2023", "pmid": "37212182"},
    "mcdonald": {"cite": "Thompson AJ, et al. Diagnosis of multiple sclerosis: 2017 revisions of the McDonald criteria. Lancet Neurol 2018", "pmid": "29275977"},
    "addison": {"cite": "Bornstein SR, et al. Diagnosis and treatment of primary adrenal insufficiency. J Clin Endocrinol Metab 2016", "pmid": "26760044"},
    "cushing": {"cite": "Nieman LK, et al. The diagnosis of Cushing's syndrome. J Clin Endocrinol Metab 2008", "pmid": "18334580"},
    "pheo": {"cite": "Lenders JW, et al. Pheochromocytoma and paraganglioma guideline. J Clin Endocrinol Metab 2014", "pmid": "24893135"},
    "aldo": {"cite": "Funder JW, et al. Management of primary aldosteronism. J Clin Endocrinol Metab 2016", "pmid": "26934393"},
    "acromegaly": {"cite": "Katznelson L, et al. Acromegaly guideline. J Clin Endocrinol Metab 2014", "pmid": "25356808"},
    "prolactin": {"cite": "Melmed S, et al. Diagnosis and treatment of hyperprolactinemia. J Clin Endocrinol Metab 2011", "pmid": "21296991"},
    "hpt": {"cite": "Bilezikian JP, et al. Evaluation and management of primary hyperparathyroidism (5th international workshop). J Bone Miner Res 2022", "pmid": "36245251"},
    "tb": {"cite": "Lewinsohn DM, et al. ATS/IDSA/CDC guidelines: diagnosis of tuberculosis. Clin Infect Dis 2017", "pmid": "28052967"},
    "esc_ie": {"cite": "Delgado V, et al. 2023 ESC Guidelines for the management of endocarditis. Eur Heart J 2023", "pmid": "37622656"},
    "cap": {"cite": "Metlay JP, et al. ATS/IDSA guideline: community-acquired pneumonia. Am J Respir Crit Care Med 2019", "pmid": "31573350"},
    "bts_pleural": {"cite": "Roberts ME, et al. British Thoracic Society guideline for pleural disease. Thorax 2023", "pmid": "37553157"},
    "acog_ectopic": {"cite": "ACOG Practice Bulletin No. 193: Tubal ectopic pregnancy. Obstet Gynecol 2018", "pmid": "29470343"},
    "kdigo_gn": {"cite": "KDIGO 2021 guideline for the management of glomerular diseases. Kidney Int 2021", "pmid": "34556256"},
    "sjogren": {"cite": "Shiboski CH, et al. 2016 ACR/EULAR classification criteria for primary Sjögren's syndrome. Arthritis Rheumatol 2017", "pmid": "27785888"},
    "ssc": {"cite": "van den Hoogen F, et al. 2013 classification criteria for systemic sclerosis. Arthritis Rheum 2013", "pmid": "24122180"},
    "asas": {"cite": "Rudwaleit M, et al. ASAS classification criteria for axial spondyloarthritis. Ann Rheum Dis 2009", "pmid": "19297344"},
    "gpa": {"cite": "Robson JC, et al. 2022 ACR/EULAR classification criteria for granulomatosis with polyangiitis. Ann Rheum Dis 2022", "pmid": "35110333"},
    "gca": {"cite": "Ponte C, et al. 2022 ACR/EULAR classification criteria for giant cell arteritis. Ann Rheum Dis 2022", "pmid": "36351706"},
    "myositis": {"cite": "Lundberg IE, et al. 2017 EULAR/ACR classification criteria for idiopathic inflammatory myopathies. Ann Rheum Dis 2017", "pmid": "29079590"},
    "hh": {"cite": "Kowdley KV, et al. ACG clinical guideline: hereditary hemochromatosis. Am J Gastroenterol 2019", "pmid": "31335359"},
    "cml": {"cite": "Hochhaus A, et al. European LeukemiaNet 2020 recommendations for CML. Leukemia 2020", "pmid": "32127639"},
    "aml": {"cite": "Döhner H, et al. Diagnosis and management of AML in adults: 2022 ELN recommendations. Blood 2022", "pmid": "35797463"},
    "imwg": {"cite": "Rajkumar SV, et al. IMWG updated criteria for the diagnosis of multiple myeloma. Lancet Oncol 2014", "pmid": "25439696"},
    "mono": {"cite": "Womack J, et al. Common questions about infectious mononucleosis. Am Fam Physician 2015", "pmid": "25822555"},
    "malaria": {"cite": "White NJ, et al. Malaria. Lancet 2014", "pmid": "23953767"},
    "cdi": {"cite": "McDonald LC, et al. IDSA/SHEA guidelines for Clostridium difficile infection: 2017 update. Clin Infect Dis 2018", "pmid": "29462280"},
    "hp": {"cite": "Chey WD, et al. ACG clinical guideline: Helicobacter pylori infection. Am J Gastroenterol 2017", "pmid": "28071659"},
    "varices": {"cite": "Garcia-Tsao G, et al. Portal hypertensive bleeding in cirrhosis: 2016 AASLD practice guidance. Hepatology 2017", "pmid": "27786365"},
    "gerd": {"cite": "Vakil N, et al. The Montreal definition and classification of GERD. Am J Gastroenterol 2006", "pmid": "16928254"},
    "he": {"cite": "Vilstrup H, et al. Hepatic encephalopathy in chronic liver disease: 2014 AASLD/EASL guideline. Hepatology 2014", "pmid": "25042402"},
    "sti": {"cite": "Workowski KA, et al. Sexually transmitted infections treatment guidelines, 2021. MMWR Recomm Rep 2021", "pmid": "34292926"},
    "hcc": {"cite": "Marrero JA, et al. Diagnosis, staging and management of HCC: 2018 AASLD guidance. Hepatology 2018", "pmid": "29624699"},
    "kawasaki": {"cite": "McCrindle BW, et al. Diagnosis, treatment and long-term management of Kawasaki disease (AHA). Circulation 2017", "pmid": "28356445"},
    "lyme": {"cite": "Lantos PM, et al. IDSA/AAN/ACR 2020 guidelines for Lyme disease. Clin Infect Dis 2021", "pmid": "33417672"},
    "pericard": {"cite": "Adler Y, et al. 2015 ESC Guidelines for pericardial diseases. Eur Heart J 2015", "pmid": "26320112"},
    "esc_af": {"cite": "Hindricks G, et al. 2020 ESC Guidelines for atrial fibrillation. Eur Heart J 2021", "pmid": "32860505"},
    "hyponat": {"cite": "Spasovski G, et al. Clinical practice guideline on diagnosis and treatment of hyponatraemia. Eur J Endocrinol 2014", "pmid": "24569125"},
    "divert": {"cite": "Hall J, et al. ASCRS clinical practice guidelines for left-sided colonic diverticulitis. Dis Colon Rectum 2020", "pmid": "32384404"},
    "sbo": {"cite": "Ten Broek RPG, et al. Bologna guidelines for adhesive small bowel obstruction (2017 update). World J Emerg Surg 2018", "pmid": "29946347"},
    "influenza": {"cite": "Uyeki TM, et al. IDSA 2018 update on seasonal influenza. Clin Infect Dis 2019", "pmid": "30566567"},
    "wilson": {"cite": "Schilsky ML, et al. 2022 AASLD practice guidance on Wilson disease. Hepatology 2023", "pmid": "36151586"},
    "co": {"cite": "Rose JJ, et al. Carbon monoxide poisoning: pathogenesis, management and future directions. Am J Respir Crit Care Med 2017", "pmid": "27753502"},
    "uc": {"cite": "Magro F, et al. Third European evidence-based consensus on ulcerative colitis, part 1. J Crohns Colitis 2017", "pmid": "28158501"},
    "crohn": {"cite": "Gomollón F, et al. 3rd European evidence-based consensus on Crohn's disease 2016, part 1. J Crohns Colitis 2017", "pmid": "27660341"},
    "pcos": {"cite": "Rotterdam ESHRE/ASRM PCOS Consensus Workshop Group. Revised 2003 consensus on PCOS. Fertil Steril 2004", "pmid": "14711538"},
    "ipf": {"cite": "Raghu G, et al. Diagnosis of idiopathic pulmonary fibrosis (ATS/ERS/JRS/ALAT). Am J Respir Crit Care Med 2018", "pmid": "30168753"},
    "sarcoid": {"cite": "Crouser ED, et al. Diagnosis and detection of sarcoidosis (ATS). Am J Respir Crit Care Med 2020", "pmid": "32293205"},
    "cf": {"cite": "Farrell PM, et al. Diagnosis of cystic fibrosis: consensus guidelines from the CF Foundation. J Pediatr 2017", "pmid": "28129811"},
    "epilepsy": {"cite": "Fisher RS, et al. ILAE official report: a practical clinical definition of epilepsy. Epilepsia 2014", "pmid": "24730690"},
    "di": {"cite": "Christ-Crain M, et al. Diabetes insipidus. Nat Rev Dis Primers 2019", "pmid": "31395885"},
    "gold": {"cite": "Agustí A, et al. GOLD 2023 report: executive summary. Eur Respir J 2023", "pmid": "36858443"},
    "gina": {"cite": "Reddel HK, et al. GINA strategy 2021: executive summary. Eur Respir J 2022", "pmid": "34667060"},
    "aspergillosis": {"cite": "Denning DW, et al. Chronic pulmonary aspergillosis: guidelines for diagnosis and management. Eur Respir J 2016", "pmid": "26699723"},
    "svt": {"cite": "Page RL, et al. 2015 ACC/AHA/HRS guideline for adult supraventricular tachycardia. J Am Coll Cardiol 2016", "pmid": "26409259"},
    "va": {"cite": "Priori SG, et al. 2015 ESC guidelines for ventricular arrhythmias and prevention of sudden cardiac death. Eur Heart J 2015", "pmid": "26320108"},
    "textbook": {"cite": "Standard textbook clinical knowledge; no specific guideline cited (unverified)", "pmid": ""},
}

# ------------------------------------------------------------------ disease key → KB profile id
DX: dict[str, str] = {
    "acute_pancreatitis": "DOID:2913", "pancreatitis": "DOID:4989", "chronic_pancreatitis": "DOID:0051065",
    "mi": "DOID:5844", "ami": "DOID:9408", "unstable_angina": "DOID:8805", "myocarditis": "DOID:820",
    "pbc": "DOID:12236", "sle": "DOID:9074", "dm": "DOID:9351", "t2dm": "DOID:9352", "t1dm": "DOID:9744",
    "dka": "DOID:1837", "pyelonephritis": "DOID:11400", "uti": "DOID:0080784", "cystitis": "DOID:1679",
    "pe": "DOID:9477", "thrombosis": "DOID:0060903", "cvst": "DOID:3572", "aortic_dissection": "DOID:0080685",
    "dic": "DOID:11247", "aih": "DOID:2048", "celiac": "DOID:10608", "graves": "DOID:12361",
    "hyperthyroidism": "DOID:7998", "hypothyroidism": "DOID:1459", "hashimoto": "DOID:7188",
    "subacute_thyroiditis": "DOID:7165", "ra": "DOID:7148", "gout": "DOID:13189", "cppd": "DOID:1156",
    "hbv": "DOID:2043", "hcv": "DOID:1883", "hav": "DOID:12549", "hev": "DOID:4411",
    "cholecystitis": "DOID:1949", "cholelithiasis": "DOID:10211", "choledocholithiasis": "DOID:11755",
    "cholangitis": "DOID:9446", "psc": "DOID:0060643", "cholangiocarcinoma": "DOID:4947",
    "appendicitis": "DOID:8337", "bacterial_meningitis": "DOID:9470", "viral_meningitis": "DOID:10310",
    "meningitis": "DOID:9471", "crypto_meningitis": "DOID:0080159", "encephalitis": "DOID:9588",
    "viral_encephalitis": "DOID:646", "hf": "WD:Q181754", "chf": "DOID:6000", "dcm": "DOID:12930",
    "hcm": "DOID:11984", "aortic_stenosis": "DOID:1712", "mitral_stenosis": "DOID:1754", "sepsis": "WD:Q183134",
    "aki": "DOID:3021", "atn": "DOID:12556", "ckd": "DOID:784", "ida": "DOID:11758", "hypochromic_anemia": "DOID:11759",
    "b12": "DOID:0050731", "pernicious": "DOID:13381", "megaloblastic": "DOID:13382", "ttp": "DOID:10772",
    "hus": "DOID:12554", "mg": "DOID:437", "gbs": "DOID:12842", "ischemic_stroke": "DOID:0051062",
    "cerebral_infarction": "DOID:3526", "hemorrhagic_stroke": "DOID:0051063", "sah": "WD:Q693442",
    "ms": "DOID:2377", "addison": "DOID:13774", "adrenal_insufficiency": "WD:Q2507454",
    # DOID:446 is "primary hyperaldosteronism" but in DO/KB it also carries the "Cushing's syndrome" synonym (and the
    # Korean name 쿠싱증후군), so both aldosterone and Cushing tests point at it; DOID:3946 = Cushing's disease
    "aldosteronism_cushing": "DOID:446", "cushing_disease": "DOID:3946", "conn": "DOID:12028",
    "pheo": "DOID:0050771", "acromegaly": "DOID:2449", "prolactinoma": "DOID:5394", "hyperprolactinemia": "DOID:12700",
    "pituitary_adenoma": "DOID:3829", "tb": "DOID:399", "ptb": "DOID:2957", "ie": "DOID:0060000",
    "endocarditis": "DOID:10314", "pneumonia": "DOID:552", "bacterial_pneumonia": "DOID:874",
    "viral_pneumonia": "DOID:10533", "pneumothorax": "DOID:1673", "nephrolithiasis": "DOID:585",
    "urolithiasis": "DOID:0080653", "hydronephrosis": "DOID:11111", "ectopic": "DOID:0060329",
    "preeclampsia": "DOID:10591", "nephrotic": "DOID:1184", "gn": "DOID:2921", "igan": "DOID:2986",
    "psgn": "DOID:14064", "membranous": "DOID:10976", "sjogren": "DOID:12894", "ssc": "DOID:418",
    "as": "DOID:7147", "reactive_arthritis": "DOID:6196", "mctd": "DOID:3492", "hemochromatosis": "DOID:2352",
    "cml": "DOID:8552", "mm": "DOID:9538", "mono": "DOID:8568", "malaria": "DOID:12365", "cdiff": "DOID:0060185",
    "pud": "DOID:750", "gastric_ulcer": "DOID:10808", "duodenal_ulcer": "DOID:1724", "perforation": "DOID:752",
    "he": "DOID:13413", "cirrhosis": "DOID:5082", "syphilis": "DOID:4166", "hcc": "DOID:684",
    "polymyositis": "DOID:0080745", "dermatomyositis": "DOID:10223", "kawasaki": "DOID:13378", "lyme": "DOID:11729",
    "pericarditis": "DOID:1787", "tamponade": "DOID:115", "af": "DOID:0060224", "siadh": "DOID:3401",
    "diverticulitis": "DOID:7475", "septic_arthritis": "DOID:813", "gpa": "DOID:12132", "gca": "DOID:13375",
    "takayasu": "DOID:2508", "pmr": "DOID:853", "goodpasture": "DOID:9808", "anti_gbm": "DOID:4780",
    "influenza": "DOID:8469", "covid": "DOID:0080600", "rsv": "DOID:1273", "strep_pharyngitis": "WD:Q840143",
    "rheumatic_fever": "DOID:1586", "hiv": "DOID:526", "aml": "DOID:9119", "all": "DOID:9952",
    "acute_leukemia": "DOID:12603", "itp": "DOID:8924", "aiha": "DOID:718", "hemolytic_anemia": "DOID:583",
    "hs": "DOID:12971", "beta_thal": "DOID:12241", "thalassemia": "DOID:10241", "pv": "DOID:8997",
    "et": "DOID:2224", "aplastic": "DOID:12449", "pnh": "DOID:0060284", "co_poisoning": "WD:Q125367",
    "apap": "WD:Q2572879", "methb": "DOID:10783", "wilson": "DOID:893", "primary_hpt": "DOID:11202",
    "hpt": "DOID:13543", "scrub_typhus": "DOID:13371", "hfrs": "DOID:11266", "sfts": "WD:Q3508725",
    "dengue": "DOID:12205", "leptospirosis": "DOID:2297", "brucellosis": "DOID:11077", "q_fever": "DOID:11100",
    "copd": "DOID:3083", "asthma": "DOID:2841", "uc": "DOID:8577", "crohn": "DOID:8778",
    "obstruction": "DOID:8437", "volvulus": "DOID:8445", "intussusception": "DOID:8446", "varices": "DOID:112",
    "gerd": "DOID:8534", "reflux_esophagitis": "DOID:13976", "barrett": "DOID:9206", "crc": "DOID:9256",
    "gastric_cancer": "DOID:10534", "lung_cancer": "DOID:1324", "pancreatic_cancer": "DOID:1793",
    "prostate_cancer": "DOID:10283", "ovarian_cancer": "DOID:2394", "hodgkin": "DOID:8567",
    "epilepsy": "DOID:1826", "sarcoidosis": "DOID:11335", "mycoplasma": "DOID:13276", "legionella": "DOID:10457",
    "lung_abscess": "DOID:0060317", "aspergilloma": "DOID:0050153", "aspergillosis": "DOID:13564",
    "pcp": "DOID:11339", "testicular_torsion": "DOID:11996", "pcos": "DOID:11612", "cf": "DOID:1485",
    "di": "DOID:9409", "central_di": "DOID:0081055", "nephrogenic_di": "DOID:12387", "typhoid": "DOID:13258",
    "shigellosis": "DOID:12385", "amebiasis": "DOID:9181", "cholera": "DOID:1498", "gonorrhea": "DOID:7551",
    "chlamydia": "DOID:11263", "trichomoniasis": "DOID:1947", "bv": "DOID:3385", "toxoplasmosis": "DOID:9965",
    "alcoholic_hepatitis": "DOID:12351", "aosd": "DOID:14256", "hlh": "DOID:0050120", "ipf": "DOID:0050156",
    "ild": "DOID:3082", "hp_pneumonitis": "DOID:841", "wernicke": "DOID:2384", "nph": "DOID:1572", "als": "DOID:332",
    "long_qt": "DOID:2843", "svt": "WD:Q1598909", "wpw": "DOID:384", "brugada": "DOID:0050451",
    "pulm_htn": "DOID:6432", "aaa": "DOID:7693", "osteomyelitis": "DOID:1019", "nms": "DOID:14464",
    "igg4": "DOID:0080356", "antiphospholipid": "DOID:2988", "ischemic_colitis": "DOID:0060181",
    "lactic_acidosis": "DOID:3650", "interstitial_nephritis": "DOID:1063", "pid": "DOID:1003",
    "hemophilia_a": "DOID:12134", "hemophilia_b": "DOID:12259", "vwd": "DOID:12531",
}


@dataclass
class Finding:
    id: str
    ko: str
    en: str
    mode: str              # "hi" / "lo" numeric analyte, "pos" qualitative result, "kw" report keyword
    pat: str               # analyte (hi/lo/pos) or keyword (kw) regex over the lower-cased text
    links: str             # "dxkey:weight[R]:refkey ..." (R = a normal/negative result argues against it)
    thr: float | None = None       # hi: abnormal above; lo: abnormal below (after unit scaling)
    ref_mult: float = 1.0          # hi: abnormal above ref_mult × the upper reference limit given in the text
    units: dict = field(default_factory=dict)  # unit prefix → multiplier to the threshold's unit
    titer: int = 80                # pos: titre 1:N at or above this counts as positive
    fixed: bool = False            # kw: the keyword itself carries the polarity (e.g. "혈류 소실"), no negation check
    not_if: str = ""               # skip the finding when this regex occurs in the same segment
    alias: str = ""                # a short name ("AG", "포화도") that names this analyte only in context:
    alias_ctx: str = ""            # ... it counts only when this regex occurs somewhere in the text
    need_value: bool = False       # hi/lo: only a measured value can make it abnormal ("혈당이 높다" ≠ glucose ≥ 250)
    cutoff: bool = False           # hi/lo: the finding is an absolute cut-off ("AST > 1000"), not "outside the reference
                                   # range": a printed range can only make the value normal, never abnormal by itself

    def parsed_links(self) -> list[tuple[str, int, bool, str]]:
        out = []
        for tok in self.links.split():
            dx, w, ref = tok.split(":")
            out.append((dx, int(w.rstrip("R")), w.endswith("R"), ref))
        return out


F = Finding

# contexts that give a short analyte name its meaning (see Finding.alias)
_ACID_BASE_CTX = (r"hco3|중탄산|bicarb|(?<![a-z])(?:cl|na)(?![a-z])|chloride|sodium|나트륨|염소|클로라이드|전해질|electrolyte"
                  r"|산증|acidosis|(?<![a-z])ph(?![a-z])|(?<![a-z])abga?(?![a-z])|동맥혈\s*가스|blood gas|젖산|lactate|케톤|ketone")
_IRON_CTX = (r"(?<![가-힣])철(?:분|\s*결합|\s*결핍)?(?![가-힣])|혈청\s*철|iron|(?<![a-z])fe(?![a-z])|페리틴|ferritin|tibc|uibc"
             r"|트랜스페린|transferrin")


def _gap(n: int, stop: str) -> str:
    """A lazy gap of up to n characters that does not cross a clause boundary or any word matching `stop` (another
    heart structure / valve): "승모판 역류 및 대동맥판 협착" is not mitral stenosis."""
    return r"(?:(?![.,;:\n]|" + stop + r").){0," + str(n) + "}?"


# words that end the subject of an echo finding (another chamber / valve / vessel, or a normal statement)
_NOT_MV = r"대동맥|아오타|삼첨|폐동맥|판막|심실|심방|정상|aortic|tricuspid|pulmon|normal"
_NOT_AV = r"승모|삼첨|폐동맥|판막|심실|심방|정상|mitral|tricuspid|pulmon|normal"
_NOT_RV = r"좌심|대동맥|판막|승모|삼첨|하대|정맥|(?<![a-z])ivc|정상|normal"
_M = {"ng/l": 0.001, "pg/ml": 0.001}  # troponin ng/L (hs) → ng/mL
FINDINGS: list[Finding] = [
    # ---------------- pancreas / heart / vessels
    F("lipase_high", "리파아제 상승", "elevated serum lipase", "hi", r"리파(?:아)?제|lipase",
      "acute_pancreatitis:3R:atlanta pancreatitis:2:atlanta", thr=180, ref_mult=3),
    F("amylase_high", "아밀라아제 상승", "elevated serum amylase", "hi", r"아밀라(?:아)?제|amylase",
      "acute_pancreatitis:2:atlanta pancreatitis:2:atlanta", thr=300, ref_mult=3),
    F("ct_pancreatitis", "영상: 췌장 부종·주위 염증", "imaging: pancreatic edema / peripancreatic inflammation", "kw",
      r"(췌장|이자).{0,20}(부종|종대|미만성\s*비대|주위\s*(액체|지방|침윤|삼출|염증)|괴사|염증)|peripancreatic|pancreatic (edema|necrosis)|interstitial edematous pancreatitis",
      "acute_pancreatitis:3:atlanta pancreatitis:2:atlanta"),
    F("pancreas_calcification", "영상: 췌장 석회화", "imaging: pancreatic calcification", "kw",
      r"(췌장|이자|췌관).{0,15}(석회|calcif|결석)|pancreatic calcif", "chronic_pancreatitis:3:textbook pancreatitis:2:textbook"),
    F("pancreas_mass", "영상: 췌장 종괴", "imaging: pancreatic mass", "kw",
      r"(췌장|이자).{0,15}(종괴|종양|mass|tumor|저음영\s*병변)|pancreatic (mass|tumor)|double duct|이중\s*관\s*징후",
      "pancreatic_cancer:3:textbook"),
    F("troponin_high", "트로포닌 상승", "elevated troponin", "hi",
      r"트로포닌|troponin|(?<![a-z])(?:hs-?)?c?tn[it](?![a-z])", "mi:3R:udmi4 ami:3R:udmi4 myocarditis:2:udmi4 pe:1:esc_pe",
      thr=0.04, units=_M),
    F("ckmb_high", "CK-MB 상승", "elevated CK-MB", "hi", r"ck-?mb", "mi:2:udmi4 ami:2:udmi4", thr=6),
    F("ecg_stemi", "심전도: ST 분절 상승", "ECG: ST-segment elevation", "kw",
      r"st\s*(분절|구간|segment)?\s*(상승|elevation)|stemi|st상승|q\s*파|병적\s*q|pathologic q|q wave",
      "mi:3:udmi4 ami:3:udmi4", not_if=r"(광범위|전반적|미만성|diffuse|widespread|모든\s*유도)|pr\s*(분절|구간|segment)?\s*(하강|저하|depression)"),
    F("ecg_st_depression", "심전도: ST 하강·T파 역전", "ECG: ST depression / T-wave inversion", "kw",
      r"st\s*(분절|구간|segment)?\s*(하강|저하|depression)|t\s*파\s*(역전|전도)|t-?wave inversion|inverted t",
      "mi:1:udmi4 unstable_angina:1:udmi4", not_if=r"pr\s*(분절|구간|segment)?\s*(하강|저하|depression)"),
    F("ecg_pericarditis", "심전도: 미만성 ST 상승·PR 하강", "ECG: diffuse ST elevation / PR depression", "kw",
      r"(광범위|전반적|미만성|diffuse|widespread|모든\s*유도).{0,25}st.{0,10}(상승|elevation)|pr\s*(분절|구간|segment)?\s*(하강|저하|depression)",
      "pericarditis:3:pericard"),
    F("ecg_af", "심전도: 심방세동", "ECG: atrial fibrillation", "kw",
      r"심방\s*세동|atrial fibrillation|(?<![a-z])a-?fib|불규칙적?으로\s*불규칙|irregularly irregular|p\s*파\s*(소실|없음|없이|없고)|absent p wave",
      "af:3:esc_af", fixed=False),
    F("ecg_svt", "심전도: 좁은 QRS 빈맥", "ECG: narrow-complex regular tachycardia", "kw",
      r"좁은\s*qrs.{0,10}빈맥|narrow[- ]complex tachycardia|방실\s*결절\s*회귀|avnrt|psvt|발작성\s*상심실성|상심실성\s*빈맥|pseudo-?r|이중\s*(전도\s*)?(통로|경로)|dual (av nodal )?pathway",
      "svt:3:svt"),
    F("ecg_delta", "심전도: 델타파", "ECG: delta wave / pre-excitation", "kw", r"델타\s*파|delta wave|pre-?excitation|조기\s*흥분",
      "wpw:3:svt"),
    F("ecg_long_qt", "심전도: QT 연장", "ECG: prolonged QTc", "hi", r"qtc|qt\s*간격|qt interval|(?<![a-z])qt(?![a-z])",
      "long_qt:3:va", thr=480),
    F("ecg_brugada", "심전도: 브루가다 양상", "ECG: Brugada pattern", "kw", r"브루가다|brugada|coved", "brugada:3:va"),
    F("bnp_high", "BNP 상승", "elevated BNP", "hi", r"(?<!pro)(?<!pro-)(?<![a-z])bnp|b형\s*나트륨\s*이뇨\s*펩티드",
      "hf:2R:esc_hf chf:2R:esc_hf", thr=100),
    F("ntprobnp_high", "NT-proBNP 상승", "elevated NT-proBNP", "hi", r"nt-?\s*pro-?\s*bnp|pro-?bnp",
      "hf:2R:esc_hf chf:2R:esc_hf", thr=300),
    F("lvef_low", "심초음파: 좌심실 구혈률 저하", "echo: reduced LVEF", "lo",
      r"lvef|(?<![a-z])ef(?![a-z])|좌심실\s*(?:구혈률|박출률)|구혈률|박출률|ejection fraction",
      "hf:2:esc_hf chf:2:esc_hf dcm:2:esc_hf", thr=40),
    F("cxr_hf", "흉부영상: 심비대·폐부종", "chest imaging: cardiomegaly / pulmonary edema", "kw",
      r"심비대|심장\s*(비대|확대)|cardiomegaly|심흉[곽]?비.{0,8}(증가|확대)|폐\s*부종|pulmonary edema|폐\s*울혈|kerley|박쥐\s*날개|bat-?wing|혈관\s*재분포|cephalization",
      "hf:2:esc_hf chf:2:esc_hf"),
    F("echo_vegetation", "심초음파: 판막 증식(우종)", "echo: valvular vegetation", "kw",
      r"우종|증식물|vegetation|판막.{0,10}(종괴|증식)", "ie:3:esc_ie endocarditis:3:esc_ie"),
    F("blood_culture_pos", "혈액배양 양성", "positive blood culture", "kw",
      r"(혈액\s*배양|blood culture).{0,40}(양성|성장|검출|자랐|positive|grew|growth|구균|간균|균\s*동정|staphylo|strepto|enteroco|e\.?\s*coli|대장균|포도알?구균|사슬알?구균|장알?구균)",
      "ie:2:esc_ie endocarditis:2:esc_ie sepsis:2:sepsis3 septic_arthritis:1:septic_arthritis osteomyelitis:1:textbook"),
    F("echo_pericardial_effusion", "심초음파: 심낭 삼출", "echo: pericardial effusion", "kw",
      r"심낭\s*(삼출|액\s*(저류|고임)|내\s*액체)|심장막\s*삼출|pericardial effusion", "pericarditis:2:pericard tamponade:2:pericard"),
    F("echo_tamponade", "심초음파: 심장눌림 소견", "echo: tamponade physiology", "kw",
      r"(우심실|우심방|right (?:atrial|ventricular))" + _gap(20, _NOT_RV) + r"(허탈|collapse)|심장\s*눌림|tamponade|전기적\s*교대|electrical alternans",
      "tamponade:3:pericard"),
    F("echo_hcm", "심초음파: 비대칭 중격 비후", "echo: asymmetric septal hypertrophy", "kw",
      r"비대칭(성|적)?\s*(심실\s*)?중격\s*비후|asymmetric septal hypertrophy|(?<![a-z])sam(?![a-z])|systolic anterior motion|수축기\s*전방\s*운동|좌심실\s*유출로\s*폐쇄|lvot obstruction",
      "hcm:3:textbook"),
    F("echo_as", "심초음파: 대동맥판 협착", "echo: aortic valve stenosis", "kw",
      r"(?:대동맥|아오타)\s*판막?" + _gap(25, _NOT_AV) + r"(협착|개구\s*면적)|aortic (?:valve )?stenosis|aortic valve area", "aortic_stenosis:3:textbook"),
    F("echo_ms", "심초음파: 승모판 협착", "echo: mitral stenosis", "kw",
      r"승모\s*판막?" + _gap(25, _NOT_MV) + r"(협착|개구\s*면적|하키\s*스틱)|mitral (?:valve )?stenosis|mitral valve area|hockey[- ]stick",
      "mitral_stenosis:3:textbook"),
    F("echo_rv_strain", "심초음파: 우심실 확장·부하", "echo: RV dilatation / strain", "kw",
      r"우심실" + _gap(10, _NOT_RV) + r"(확장(?!\s*기)|확장기\s*말\s*(직경|내경|크기|용적).{0,8}(증가|확대|커)|부전|과부하|긴장|기능\s*저하)"
      r"|rv (?:dilat|strain|dysfunction|enlarg)|right ventric\w*\s+(?:dilat|enlarg|strain|dysfunction)|mcconnell|d-?shaped|d자\s*모양",
      "pe:1:esc_pe pulm_htn:1:textbook"),
    F("ddimer_high", "D-dimer 상승", "elevated D-dimer", "hi", r"d-?\s*dimer|d-?\s*다이머|디다이머",
      "pe:1R:esc_pe thrombosis:1R:ash_vte aortic_dissection:1:esc_aorta dic:2:isth_dic cvst:1:textbook",
      thr=0.5, units={"ng/ml": 0.001, "μg/l": 0.001, "ug/l": 0.001, "µg/l": 0.001}),
    F("ctpa_pe", "CT 폐동맥 충만결손", "CT pulmonary angiography: filling defect", "kw",
      r"(폐동맥|pulmonary arter).{0,30}(충만\s*결손|충전\s*결손|혈전|색전|filling defect|thromb|embol)|폐\s*색전|pulmonary embol|saddle embol|안장\s*색전",
      "pe:3R:esc_pe"),
    F("doppler_dvt", "초음파: 심부정맥 혈전", "ultrasound: deep vein thrombosis", "kw",
      r"(심부\s*정맥|대퇴\s*정맥|총대퇴|슬와\s*정맥|오금\s*정맥|popliteal|femoral vein|deep vein|하지\s*정맥).{0,30}(혈전|압박되지\s*않|압박\s*안|비압박|thromb|non-?compress)|심부\s*정맥\s*혈전|(?<![a-z])dvt(?![a-z])",
      "thrombosis:3R:ash_vte pe:1:esc_pe"),
    F("ct_dissection", "CT: 대동맥 박리(내막 피판)", "CT: aortic dissection flap", "kw",
      r"내막\s*(피판|판)|intimal flap|가성\s*내강|거짓\s*내강|false lumen|이중\s*내강|double lumen|대동맥.{0,10}박리|aortic dissection|dissection flap",
      "aortic_dissection:3R:esc_aorta"),
    F("cxr_mediastinum", "흉부 X선: 종격동 확장", "chest X-ray: widened mediastinum", "kw",
      r"종격동.{0,8}(확장|확대|넓어|넓음|widening)|widened mediastinum|mediastinal widening", "aortic_dissection:2:esc_aorta"),
    F("aaa_imaging", "영상: 복부대동맥류", "imaging: abdominal aortic aneurysm", "kw",
      r"복부\s*대동맥.{0,15}(류|확장|직경\s*\d|aneurysm)|abdominal aortic aneurysm|(?<![a-z])aaa(?![a-z])", "aaa:3:textbook"),
    F("coronary_aneurysm", "심초음파: 관상동맥 확장·동맥류", "echo: coronary artery dilatation / aneurysm", "kw",
      r"관상\s*동맥.{0,15}(확장|동맥류|aneurysm|dilat)|coronary (artery )?(aneurysm|dilat|ectasia)", "kawasaki:3:kawasaki"),
    # ---------------- lungs / infection
    F("cxr_consolidation", "흉부영상: 폐 경화·침윤", "chest imaging: consolidation / infiltrate", "kw",
      r"(폐|폐야|폐엽|하엽|상엽|중엽|설엽|lung|lobe|lobar).{0,25}(경화|침윤|consolidat|infiltrat|opacit|음영\s*증가|폐렴)|consolidat|air bronchogram|공기\s*기관지\s*조영|엽성\s*폐렴|폐렴\s*소견",
      "pneumonia:2R:cap bacterial_pneumonia:1:cap"),
    F("ggo", "흉부 CT: 간유리 음영", "chest CT: ground-glass opacity", "kw", r"간\s*유리\s*(음영|혼탁|양)|ground[- ]glass",
      "viral_pneumonia:1:textbook pcp:2:textbook covid:1:textbook ild:1:ipf hp_pneumonitis:1:textbook"),
    F("honeycombing", "흉부 CT: 벌집모양·UIP", "chest CT: honeycombing / UIP pattern", "kw",
      r"벌집\s*(모양|양상|폐)|honeycomb|uip\s*(pattern|양상|형)|통상\s*간질성\s*폐렴", "ipf:3:ipf ild:2:ipf"),
    F("cxr_ptx", "흉부영상: 기흉", "chest imaging: pneumothorax", "kw",
      r"기흉|pneumothorax|(내장\s*)?흉막\s*선|pleural line|폐\s*(표지|혈관\s*음영).{0,10}(소실|없|관찰되지)|absent lung markings|폐\s*허탈|deep sulcus",
      "pneumothorax:3R:bts_pleural", fixed=False),
    F("pleural_effusion", "흉부영상: 흉막 삼출", "chest imaging: pleural effusion", "kw",
      r"흉막\s*삼출|늑막\s*삼출|흉수|pleural effusion|늑골\s*횡격막\s*각.{0,10}(둔화|소실|무뎌)|costophrenic.{0,12}blunt",
      "hf:1:bts_pleural pneumonia:1:bts_pleural tb:1:bts_pleural"),
    F("cavity", "흉부영상: 공동", "chest imaging: cavitation", "kw", r"공동(?!\s*생활|\s*주택|체)|cavit",
      "ptb:2:tb tb:1:tb lung_abscess:2:textbook gpa:1:gpa aspergilloma:1:aspergillosis"),
    F("upper_lobe_tb", "흉부영상: 상엽 침윤·결절", "chest imaging: upper-lobe infiltrate / nodules", "kw",
      r"(상엽|폐첨부|첨부|apical|upper lobe).{0,25}(침윤|공동|결절|infiltrat|cavit|음영|반흔|섬유)",
      "ptb:2:tb tb:1:tb"),
    F("afb_pos", "항산균·결핵균 검사 양성", "positive AFB smear / TB PCR / culture", "pos",
      r"항산균|(?<![a-z])afb(?![a-z])|acid-?fast|결핵균|(?<![a-z])mtb(?![a-z])|tb[- ]?pcr|xpert|결핵\s*(pcr|배양|균)|mycobacterium tuberculosis",
      "tb:3:tb ptb:3:tb"),
    F("igra_pos", "IGRA·투베르쿨린 양성", "positive IGRA / tuberculin test", "pos",
      r"(?<![a-z])igra(?![a-z])|quantiferon|퀀티페론|인터페론\s*감마\s*분비|t-?spot|투베르쿨린|tuberculin|(?<![a-z])ppd(?![a-z])|mantoux",
      "tb:1:tb ptb:1:tb"),
    F("caseating_granuloma", "조직: 건락성 육아종", "biopsy: caseating granuloma", "kw",
      r"(?<!비)건락\s*(성|화)?\s*(괴사|육아종)|(?<!non)(?<!non-)caseat", "tb:3:tb ptb:2:tb"),
    F("noncaseating_granuloma", "조직: 비건락성 육아종", "biopsy: non-caseating granuloma", "kw",
      r"비\s*건락\s*(성|화)?\s*육아종|non-?caseating|noncaseating", "sarcoidosis:2:sarcoid crohn:1:crohn"),
    F("bhl", "흉부영상: 양측 폐문 림프절 비대", "chest imaging: bilateral hilar lymphadenopathy", "kw",
      r"양측\s*(폐\s*)?문\s*(부\s*)?림프절.{0,6}(비대|종대|커짐)|bilateral hilar (lymph)?adenopathy|(?<![a-z])bhl(?![a-z])",
      "sarcoidosis:3:sarcoid"),
    F("legionella_pos", "레지오넬라 항원·검사 양성", "positive Legionella urinary antigen / PCR", "pos",
      r"레지오넬라|legionella", "legionella:3:cap"),
    F("pneumococcal_ag", "폐렴구균 소변항원 양성", "positive pneumococcal urinary antigen", "pos",
      r"(폐렴\s*구균|폐렴\s*알균|pneumococ|streptococcus pneumoniae|s\.\s*pneumoniae).{0,20}(항원|antigen)",
      "pneumonia:2:cap bacterial_pneumonia:2:cap"),
    F("mycoplasma_pos", "마이코플라스마 검사 양성", "positive Mycoplasma pneumoniae test", "pos",
      r"마이코플라[스즈]마|mycoplasma", "mycoplasma:3:cap"),
    F("cold_agglutinin", "한랭응집소 양성", "cold agglutinins", "pos", r"한랭\s*응집|cold agglutinin",
      "mycoplasma:1:textbook aiha:1:aiha"),
    F("influenza_pos", "인플루엔자 검사 양성", "positive influenza test", "pos",
      r"인플루엔자|influenza|독감\s*(검사|항원|키트)|flu\s*(a|b)?\s*(항원|antigen|pcr|test)", "influenza:3:influenza"),
    F("covid_pos", "SARS-CoV-2 검사 양성", "positive SARS-CoV-2 test", "pos", r"sars-?cov-?2|covid|코로나\s*(19|바이러스)?",
      "covid:3:textbook"),
    F("rsv_pos", "RSV 검사 양성", "positive RSV test", "pos", r"(?<![a-z])rsv(?![a-z])|호흡기\s*세포\s*융합",
      "rsv:3:textbook"),
    F("strep_pos", "A군 연쇄구균 검사 양성", "positive group A streptococcus test", "pos",
      r"(a군\s*)?(연쇄상?\s*구균|연쇄알균|streptococ|strep).{0,20}(신속|항원|rapid|배양|culture|검사|test)|rapid strep|(?<![a-z])radt(?![a-z])|신속\s*항원\s*검사",
      "strep_pharyngitis:3:textbook rheumatic_fever:1:textbook"),
    F("aso_high", "ASO 역가 상승", "elevated antistreptolysin O titre", "hi",
      r"(?<![a-z])as[lo]o?(?![a-z])|antistreptolysin|항\s*스트렙토리?신|항스트렙톨리신",
      "psgn:2:kdigo_gn rheumatic_fever:2:textbook gn:1:kdigo_gn", thr=250),
    F("mono_pos", "이종친화항체·EBV 검사 양성", "positive heterophile antibody / EBV serology", "pos",
      r"monospot|heterophil|이종\s*친화\s*(성\s*)?항체|이종\s*항체|폴-?번넬|paul-?bunnell|ebv.{0,15}(vca|igm|항체|pcr)|vca\s*igm|epstein",
      "mono:3:mono"),
    F("atypical_lymph", "말초혈액: 비정형 림프구", "blood smear: atypical lymphocytes", "kw",
      r"비\s*정형\s*림프구|이형\s*림프구|atypical lymphocyt|반응성\s*림프구|reactive lymphocyt", "mono:2:mono"),
    F("pcp_pos", "주폐포자충 검사 양성", "positive Pneumocystis test", "pos",
      r"pneumocystis|(주)?폐포자충|(?<![a-z])p[cj]p(?![a-z])|beta-?d-?glucan|베타\s*-?\s*d\s*-?\s*글루칸", "pcp:3:textbook"),
    F("aspergillus_pos", "아스페르길루스 검사 양성·진균구", "positive Aspergillus test / fungus ball", "kw",
      r"galactomannan|갈락토만난|aspergillus.{0,15}(양성|배양|igg|precipitin|항체)|아스페르길루스.{0,15}(양성|배양|항체|침강)|fungus ball|진균\s*구|공기\s*초승달|air crescent|monod",
      "aspergilloma:3:aspergillosis aspergillosis:2:aspergillosis"),
    F("fev1fvc_low", "폐기능: FEV1/FVC 감소", "spirometry: low FEV1/FVC", "lo", r"fev1\s*/\s*fvc|1\s*초율|fev1%",
      "copd:3R:gold asthma:1:gina", thr=70),
    F("obstructive_pft", "폐기능: 폐쇄성 환기장애", "spirometry: obstructive defect", "kw",
      r"폐쇄성\s*(환기\s*)?(장애|패턴|양상)|obstructive (pattern|defect|ventilatory)", "copd:2:gold asthma:2:gina"),
    F("bd_reversible", "기관지확장제 가역성·기도과민성 양성", "bronchodilator reversibility / airway hyperresponsiveness", "kw",
      r"(기관지\s*확장제|bronchodilator|벤토린|살부타몰|albuterol|salbutamol).{0,40}(후|반응|가역).{0,30}(\d{2}\s*%|증가|개선|가역|호전|significant|reversib)|가역성.{0,10}(양성|있|확인)|메타콜린.{0,25}(양성|과민|pc20)|methacholine.{0,25}(positive|hyperrespons|pc20)|기도\s*과민",
      "asthma:3:gina"),
    F("sweat_chloride_high", "땀 염화물 상승", "elevated sweat chloride", "hi", r"땀\s*(염소|염화물|클로라이드|chloride)|sweat chloride|sweat test|땀\s*검사",
      "cf:3:cf", thr=59),
    # ---------------- liver / biliary / GI
    F("ama_pos", "항미토콘드리아항체 양성", "anti-mitochondrial antibody positive", "pos",
      r"항\s*-?\s*미토콘드리아|항항미토콘드리아|anti-?\s*mitochondrial|(?<![a-z])ama(?:-?m2)?(?![a-z])",
      "pbc:3R:easl_pbc"),
    F("alp_high", "ALP 상승", "elevated alkaline phosphatase", "hi",
      r"(?<![a-z])alp(?![a-z])|alkaline phosphatase|알칼리\s*(성\s*)?(인산분해효소|포스파타[아]?제)",
      "pbc:1:aasld_pbc psc:1:psc choledocholithiasis:1:tg18_cholangitis cholangitis:1:tg18_cholangitis", thr=240, ref_mult=2),
    F("ast_alt_very_high", "AST/ALT 현저한 상승(>1000)", "aminotransferases > 1000 U/L", "hi",
      r"(?<![a-z])(?:ast|alt|sgot|sgpt|got|gpt)(?![a-z])",
      "hav:1:textbook hbv:1:textbook hev:1:textbook apap:1:textbook aih:1:aih", thr=1000, need_value=True),
    F("asma_pos", "항평활근항체·항LKM 양성", "anti-smooth muscle / anti-LKM antibody positive", "pos",
      r"항\s*-?\s*평활근|anti-?\s*smooth muscle|(?<![a-z])asma(?![a-z])|anti-?\s*lkm|항\s*-?\s*lkm|(?<![a-z])lkm-?1",
      "aih:3:aih"),
    F("igg_high", "IgG 상승", "elevated serum IgG", "hi", r"(?<![a-z])igg(?![0-9a-z])(?!\s*4)|면역\s*글로불린\s*g(?![a-z])",
      "aih:1:aih", thr=1600),
    F("liver_bx_aih", "간생검: 경계면 간염·형질세포 침윤", "liver biopsy: interface hepatitis / plasma cells", "kw",
      r"경계\s*면\s*간염|interface hepatitis|(간|문맥|portal|liver|hepat).{0,30}(형질\s*세포|plasma cell).{0,15}(침윤|풍부|infiltrat)|간세포\s*로제트|로제트\s*형성|rosett",
      "aih:3:aih"),
    F("liver_bx_pbc", "간생검: 소담관 손상·소실", "liver biopsy: florid duct lesion / ductopenia", "kw",
      r"(소\s*담관|간내\s*담관|interlobular bile duct).{0,20}(손상|파괴|소실|감소)|담관\s*소실|florid duct|ductopenia|non-?suppurative destructive cholangitis",
      "pbc:3:easl_pbc"),
    F("psc_imaging", "MRCP: 구슬 모양 담관 협착", "MRCP: beaded multifocal bile-duct strictures", "kw",
      r"(구슬|염주|beaded|beading)|다발성?\s*(담관\s*)?협착.{0,20}확장|multifocal strictur|양파\s*껍질|onion[- ]skin",
      "psc:3:psc"),
    F("hbsag_pos", "HBsAg·HBV 검사 양성", "HBsAg / HBV marker positive", "pos",
      r"hbsag|b형\s*간염\s*(표면\s*)?항원|hbv\s*dna|igm\s*anti-?hbc|anti-?hbc\s*igm|hbc\s*igm|hbeag", "hbv:3:aasld_hbv"),
    F("hcv_pos", "HCV 항체·RNA 양성", "HCV antibody / RNA positive", "pos",
      r"anti-?\s*hcv|hcv\s*(항체|ab|rna)|c형\s*간염\s*(항체|rna|바이러스)", "hcv:3:hcv"),
    F("hav_igm_pos", "IgM anti-HAV 양성", "IgM anti-HAV positive", "pos",
      r"igm\s*anti-?\s*hav|anti-?\s*hav\s*igm|hav\s*igm|a형\s*간염.{0,10}igm", "hav:3:textbook"),
    F("hev_igm_pos", "IgM anti-HEV 양성", "IgM anti-HEV positive", "pos",
      r"igm\s*anti-?\s*hev|anti-?\s*hev\s*igm|hev\s*(igm|rna)|e형\s*간염.{0,10}(igm|rna)", "hev:3:textbook"),
    F("ceruloplasmin_low", "세룰로플라스민 감소", "low ceruloplasmin", "lo", r"세룰로플라스민|ceruloplasmin",
      "wilson:3R:wilson", thr=20),
    F("kf_ring", "카이저-플라이셔 고리", "Kayser-Fleischer ring", "kw", r"카이저\s*-?\s*플라이셔|kayser[- ]?fleischer|k-?f\s*(고리|ring)",
      "wilson:3:wilson"),
    F("ammonia_high", "암모니아 상승", "elevated ammonia", "hi", r"암모니아|ammonia|(?<![a-z])nh3(?![a-z])",
      "he:2:he cirrhosis:1:he", thr=80),
    F("afp_high", "AFP 상승", "elevated alpha-fetoprotein", "hi", r"(?<![a-z])afp(?![a-z])|알파\s*-?\s*태아\s*단백|alpha-?\s*fetoprotein",
      "hcc:2:hcc", thr=200),
    F("hcc_imaging", "영상: 동맥기 조영증강 후 씻김", "imaging: arterial enhancement with washout", "kw",
      r"(동맥기|arterial phase).{0,40}(조영\s*증강|과증강|고음영|hyperenhanc).{0,50}(지연기|문맥기|씻김|washout|소실)|washout|li-?rads\s*-?\s*5|(?<![a-z])lr-?5",
      "hcc:3:hcc"),
    F("us_cholecystitis", "초음파: 담낭벽 비후·주위 액체", "ultrasound: gallbladder wall thickening / pericholecystic fluid", "kw",
      r"(담낭|쓸개).{0,20}(벽\s*(비후|두께\s*(증가|\d)|두꺼|부종)|주위\s*(액체|체액|삼출|액)|팽대|확장|긴장)|gallbladder wall thick|pericholecystic|초음파\s*머피|sonographic murphy",
      "cholecystitis:3R:tg18_chole"),
    F("gallstone", "영상: 담석", "imaging: gallstones", "kw",
      r"담석|(담낭|쓸개).{0,10}(내\s*)?결석|gallstone|cholelith|acoustic shadow|후방\s*음영", "cholelithiasis:3:tg18_chole cholecystitis:1:tg18_chole"),
    F("cbd_stone", "영상: 총담관 결석", "imaging: common bile duct stone", "kw",
      r"(총\s*담관|총수\s*담관|담관|담도|cbd|common bile duct).{0,20}(결석|돌|stone|calcul)|choledocholith",
      "choledocholithiasis:3:tg18_cholangitis cholangitis:1:tg18_cholangitis"),
    F("cbd_dilation", "영상: 담관 확장", "imaging: bile-duct dilatation", "kw",
      r"(담관|담도|총\s*담관|cbd|bile duct).{0,20}(확장|dilat|늘어)",
      "choledocholithiasis:2:tg18_cholangitis cholangitis:2:tg18_cholangitis pancreatic_cancer:1:textbook cholangiocarcinoma:1:psc"),
    F("ct_appendicitis", "영상: 충수 비후·주위 염증", "imaging: enlarged inflamed appendix", "kw",
      r"(충수|appendi).{0,25}(비후|확장|팽대|벽\s*(비후|조영)|직경|주위\s*(지방|염증|액체|농양)|지방\s*침윤|fat strand|dilat|thicken|enlarg|distend|inflam|결석|천공|농양)|appendicolith|충수\s*결석|충수염\s*(소견|에 합당)",
      "appendicitis:3R:wses_app"),
    F("ct_diverticulitis", "CT: 게실 주위 염증", "CT: pericolic inflammation around diverticula", "kw",
      r"게실.{0,25}(염|주위\s*(지방|염증|침윤|농양|액체)|벽\s*비후)|diverticulitis|diverticul.{0,30}(fat strand|inflam|thicken)",
      "diverticulitis:3R:divert"),
    F("sbo_imaging", "영상: 소장 확장·이행부위", "imaging: dilated bowel loops / transition point", "kw",
      r"(소장|장관|장고리|bowel|small bowel).{0,20}(확장|팽창|dilat|distend)|이행\s*부위|전이\s*부위|transition point|공기\s*-?\s*액체\s*(층|수평면)|air-?fluid level|계단\s*모양|step-?ladder",
      "obstruction:3R:sbo"),
    F("free_air", "영상: 복강 내 유리 공기", "imaging: free intraperitoneal air", "kw",
      r"(복강\s*내|횡격막\s*(하|아래)|free).{0,10}(자유\s*)?(공기|가스|air)|pneumoperitoneum|기복증|유리\s*공기",
      "perforation:3:textbook pud:1:textbook"),
    F("volvulus_sign", "영상: 커피콩·소용돌이 징후", "imaging: coffee-bean / whirl sign", "kw",
      r"커피\s*콩|coffee[- ]bean|whirl(pool)?\s*sign|소용돌이\s*(징후|sign)", "volvulus:3:textbook"),
    F("intussusception_sign", "초음파: 표적 징후", "ultrasound: target / pseudokidney sign", "kw",
      r"표적\s*(징후|모양)|target sign|도넛\s*(징후|모양)|doughnut|pseudo-?kidney|가성\s*신장", "intussusception:3:textbook"),
    F("egd_gastric_ulcer", "내시경: 위궤양", "endoscopy: gastric ulcer", "kw",
      r"(?<![가-힣])위\s*(각부|전정부|체부|소만|대만|저부|유문부)?\s*(의\s*)?궤양|위궤양|gastric ulcer",
      "gastric_ulcer:3:hp pud:3:hp"),
    F("egd_duodenal_ulcer", "내시경: 십이지장궤양", "endoscopy: duodenal ulcer", "kw",
      r"십이지장\s*(구부)?\s*(의\s*)?궤양|구부\s*궤양|duodenal ulcer", "duodenal_ulcer:3:hp pud:3:hp"),
    F("hpylori_pos", "헬리코박터 검사 양성", "positive H. pylori test", "pos",
      r"헬리코박터|h\.?\s*pylori|요소\s*호기|urea breath|clo\s*(test|검사)|신속\s*요소\s*분해\s*효소",
      "pud:2:hp gastric_ulcer:1:hp duodenal_ulcer:1:hp gastric_cancer:1:hp"),
    F("varices", "내시경: 식도·위 정맥류", "endoscopy: esophageal / gastric varices", "kw", r"정맥류|varice|varix",
      "varices:3:varices cirrhosis:2:varices", not_if=r"하지|다리|종아리|(?<![a-z])leg|lower extrem|음낭|정삭|scrot"),
    F("egd_esophagitis", "내시경: 역류성(미란성) 식도염", "endoscopy: erosive (reflux) esophagitis", "kw",
      r"역류성\s*식도염|미란성\s*식도염|(?<![a-z])la\s*(분류|grade|등급)\s*[abcd](?![a-z])|(?<![a-z])la\s*[abcd]\s*(등급|grade)|erosive esophagitis|reflux esophagitis|점막\s*결손.{0,20}(식도|위식도)",
      "gerd:3:gerd reflux_esophagitis:3:gerd"),
    F("ph_monitoring", "식도 pH 검사: 산 노출 증가", "esophageal pH monitoring: increased acid exposure", "kw",
      r"(24\s*시간\s*)?(식도\s*)?(ph|임피던스)\s*(검사|모니터링|측정).{0,40}(산\s*노출|양성|증가|비정상|drmeester|드미스터)|acid exposure time|(?<![a-z])aet(?![a-z])",
      "gerd:3:gerd"),
    F("barrett_bx", "생검: 바렛 식도(원주상피 화생)", "biopsy: Barrett's metaplasia", "kw",
      r"바렛|barrett|원주\s*상피.{0,10}(화생|대체)|장\s*상피\s*화생.{0,10}식도", "barrett:3:textbook gerd:1:gerd"),
    F("colon_uc", "대장내시경: 직장부터 연속적 염증", "colonoscopy: continuous inflammation from the rectum", "kw",
      r"직장.{0,40}(연속|미만성|diffuse|continuous).{0,40}(염증|궤양|발적|점막|미란)|(연속적|continuous).{0,20}(점막\s*염증|궤양|colitis)|음와\s*농양|crypt abscess|술잔\s*세포\s*(감소|소실)",
      "uc:3:uc"),
    F("colon_crohn", "내시경: 도약 병변·조약돌·회장말단 궤양", "endoscopy: skip lesions / cobblestoning / terminal ileitis", "kw",
      r"(도약|skip|건너뛰는|분절성).{0,10}(병변|lesion|궤양|분포)|조약돌|cobblestone|종주\s*(성\s*)?궤양|longitudinal ulcer|(회장\s*말단|말단\s*회장|terminal ileum).{0,20}(궤양|협착|염증|ulcer|strictur|inflam)|누공|fistula",
      "crohn:3:crohn"),
    F("calprotectin_high", "분변 칼프로텍틴 상승", "elevated fecal calprotectin", "hi", r"칼프로텍틴|calprotectin",
      "uc:1:uc crohn:1:crohn", thr=200),
    F("cdiff_pos", "C. difficile 독소·PCR 양성", "C. difficile toxin / PCR positive", "pos",
      r"(c\.?\s*diff|클로스트리(디)?(움|오이데스)\s*디피실|clostridi(um|oides) difficile|(?<![a-z])cdi(?![a-z]))|위막|pseudomembran",
      "cdiff:3:cdi"),
    F("stool_pathogen", "대변 배양·PCR 병원체 검출", "stool pathogen detected", "kw",
      r"(살모넬라|salmonella)\s*(typhi|티피|장티푸스)|장티푸스균|(시겔라|이질균|shigella)|콜레라균|vibrio cholerae|(아메바|entamoeba)",
      "typhoid:2:textbook shigellosis:2:textbook cholera:2:textbook amebiasis:2:textbook"),
    F("colon_mass", "대장내시경: 종괴·선암", "colonoscopy: mass / adenocarcinoma", "kw",
      r"(대장|결장|직장|colon|rect|s\s*상|구불).{0,20}(종괴|종양|mass|tumor|환상\s*(병변|협착)|선암|adenocarcinoma|암종)|사과\s*씨\s*모양|apple[- ]core",
      "crc:3:textbook"),
    F("gastric_ca_bx", "생검: 위선암·반지세포암", "biopsy: gastric adenocarcinoma / signet ring cell", "kw",
      r"(위|gastric).{0,20}(선암|adenocarcinoma|반지\s*세포|signet)|반지\s*세포|signet ring|보르만|borrmann",
      "gastric_cancer:3:textbook"),
    F("villous_atrophy", "생검: 융모 위축", "biopsy: villous atrophy", "kw", r"융모.{0,6}(위축|소실|편평|둔화)|villous atrophy|marsh\s*(3|iii)",
      "celiac:3:celiac"),
    F("ttg_pos", "항tTG·항근내막항체 양성", "anti-tTG / anti-endomysial antibody positive", "pos",
      r"(?<![a-z])ttg(?![a-z])|transglutaminase|트랜스글루타미나제|endomysial|근내막|ema-?\s*iga|iga\s*-?\s*ema|deamidated gliadin|글리아딘",
      "celiac:3R:celiac"),
    # ---------------- kidney / urine
    F("ua_nitrite_pos", "소변 아질산염 양성", "urine nitrite positive", "pos", r"아질산\s*(염)?|nitrite",
      "uti:2:idsa_uti cystitis:2:idsa_uti pyelonephritis:1:idsa_uti"),
    F("ua_le_pos", "소변 백혈구 에스테라아제 양성", "urine leukocyte esterase positive", "pos",
      r"백혈구\s*에스테라(아)?제|leuko(cyte)?\s*esterase|(?<![a-z])le(?![a-z])",
      "uti:2R:idsa_uti cystitis:1:idsa_uti pyelonephritis:1:idsa_uti"),
    F("pyuria", "농뇨(소변 백혈구 증가)", "pyuria", "hi", r"(?:wbc|백혈구)(?=[^,;]{0,20}/\s*hpf)|농뇨|pyuria",
      "uti:2R:idsa_uti cystitis:1:idsa_uti pyelonephritis:2:idsa_uti", thr=10),
    F("wbc_cast", "소변 백혈구 원주", "urine WBC casts", "kw", r"(백혈구|wbc)\s*원주|wbc cast|white (blood )?cell cast",
      "pyelonephritis:3:textbook interstitial_nephritis:1:textbook"),
    F("rbc_cast", "소변 적혈구 원주·변형 적혈구", "urine RBC casts / dysmorphic RBCs", "kw",
      r"(적혈구|rbc)\s*원주|rbc cast|red (blood )?cell cast|변형\s*적혈구|이형\s*적혈구|dysmorphic",
      "gn:3:kdigo_gn psgn:2:kdigo_gn igan:2:kdigo_gn"),
    F("urine_culture_pos", "소변배양 양성", "positive urine culture", "kw",
      r"(소변|요)\s*배양.{0,30}(양성|10\s*\^?\s*[45⁴⁵]|cfu|성장|자랐|검출|e\.?\s*coli|대장균|klebsiella|enteroco|proteus)|urine culture.{0,30}(positive|grew|cfu|e\.?\s*coli)",
      "uti:3:idsa_uti cystitis:2:idsa_uti pyelonephritis:2:idsa_uti"),
    F("ct_pyelonephritis", "CT: 신장 쐐기 모양 조영저하·신주위 염증", "CT: striated nephrogram / perinephric stranding", "kw",
      r"(신장|콩팥|신실질|kidney|renal).{0,25}(주위\s*(지방|침윤|염증)|쐐기|줄무늬|조영\s*(저하|감소|결손)|부종|종대)|perinephric strand|striated nephrogram|wedge-?shaped",
      "pyelonephritis:3:textbook"),
    F("stone_imaging", "영상: 요로결석", "imaging: urinary-tract stone", "kw",
      r"(요관|신장|신우|방광|요로|ureter|renal|kidney).{0,20}(결석|stone|calcul)|요로\s*결석|신\s*결석|요관\s*결석|ureterolith|nephrolith|urolith",
      "nephrolithiasis:3:textbook urolithiasis:3:textbook"),
    F("hydronephrosis", "영상: 수신증", "imaging: hydronephrosis", "kw", r"수신증|hydronephrosis|신우\s*(확장|확대)|신배\s*확장|hydroureter|수뇨관",
      "hydronephrosis:3:textbook nephrolithiasis:1:textbook"),
    F("proteinuria_nephrotic", "신증후군 범위 단백뇨(≥3.5 g/일)", "nephrotic-range proteinuria", "hi",
      r"(24\s*시간\s*)?(요\s*단백|소변\s*단백|단백뇨|urine protein|proteinuria|upcr|단백\s*/\s*크레아티닌)",
      "nephrotic:3:kdigo_gn membranous:1:kdigo_gn", thr=3.5, units={"mg": 0.001}, need_value=True),
    F("kidney_bx_iga", "신생검: 메산지움 IgA 침착", "kidney biopsy: mesangial IgA deposits", "kw",
      r"(메산지움|메산지얼|mesangi).{0,25}iga|iga\s*(침착|deposit)", "igan:3:kdigo_gn"),
    F("kidney_bx_linear", "신생검: 선형 IgG 침착", "kidney biopsy: linear IgG deposition", "kw", r"(선형|linear).{0,20}igg",
      "goodpasture:3:kdigo_gn anti_gbm:3:kdigo_gn"),
    F("crescents", "신생검: 반월체", "kidney biopsy: crescents", "kw", r"반월\s*체|초승달\s*(모양\s*)?(사구체|형성)|(?<!air )crescent(ic)?\s*(glomerul|formation|gn)|crescents",
      "gn:2:kdigo_gn gpa:1:gpa goodpasture:1:kdigo_gn"),
    F("kidney_bx_membranous", "신생검: 상피하 침착·스파이크", "kidney biopsy: subepithelial deposits / spikes", "kw",
      r"(상피\s*하|subepithelial).{0,20}(침착|deposit|hump|혹)|(기저막|gbm).{0,15}(스파이크|spike)|pla2r", "membranous:3:kdigo_gn psgn:1:kdigo_gn"),
    F("gbm_ab_pos", "항GBM 항체 양성", "anti-GBM antibody positive", "pos",
      r"anti-?\s*gbm|항\s*-?\s*gbm|사구체\s*기저막\s*(에\s*대한\s*)?항체|glomerular basement membrane",
      "goodpasture:3:kdigo_gn anti_gbm:3:kdigo_gn"),
    # ---------------- rheumatology / immunology
    F("ana_pos", "항핵항체(ANA) 양성", "antinuclear antibody positive", "pos",
      r"항\s*핵\s*항체|anti-?\s*nuclear|(?<![a-z])f?ana(?![a-z])",
      "sle:2R:sle2019 sjogren:1:sjogren ssc:1:ssc aih:1:aih mctd:1:textbook dermatomyositis:1:myositis"),
    F("dsdna_pos", "항dsDNA 항체 양성", "anti-dsDNA antibody positive", "pos",
      r"ds-?\s*dna|이중\s*가닥\s*dna|double[- ]stranded dna|이중\s*나선\s*dna", "sle:3:sle2019"),
    F("sm_pos", "항Sm 항체 양성", "anti-Smith antibody positive", "pos",
      r"anti-?\s*sm(?![a-z])|항\s*-?\s*sm(?![a-z])|smith\s*(항체|antibod)|anti-?\s*smith", "sle:3:sle2019"),
    F("complement_low", "보체(C3/C4) 감소", "low complement C3/C4", "lo",
      r"(?<![a-z])c[34](?![0-9a-z])|보체|complement|ch50",
      "sle:2:sle2019 psgn:2:kdigo_gn", thr=None),
    F("aps_ab_pos", "항인지질항체 양성", "antiphospholipid antibody positive", "pos",
      r"루푸스\s*항응고|lupus anticoagulant|항\s*카디오리핀|anticardiolipin|cardiolipin|beta-?2\s*gp|β2\s*-?\s*당단백|b2gpi",
      "antiphospholipid:3:textbook sle:1:sle2019"),
    F("ssa_pos", "항SSA/Ro·SSB/La 양성", "anti-SSA/Ro or SSB/La positive", "pos",
      r"(?<![a-z])ss-?\s*[ab](?![a-z])|anti-?\s*ro(?![a-z])|anti-?\s*la(?![a-z])|항\s*-?\s*ro(?![a-z가-힣])|항\s*-?\s*la(?![a-z가-힣])",
      "sjogren:3:sjogren sle:1:sle2019"),
    F("schirmer_low", "쉬르머 검사 감소", "reduced Schirmer test", "lo", r"쉬르머|schirmer", "sjogren:2:sjogren", thr=5.1),
    F("scl70_pos", "항Scl-70·항센트로미어 항체 양성", "anti-Scl-70 / anticentromere antibody positive", "pos",
      r"scl-?\s*70|topoisomerase|토포이소머라[아]?제|centromere|센트로미어|중심절|rna\s*중합효소\s*iii|rna polymerase iii",
      "ssc:3:ssc"),
    F("rnp_pos", "항U1-RNP 항체 양성", "anti-U1-RNP antibody positive", "pos", r"u1-?\s*rnp|anti-?\s*rnp|항\s*-?\s*rnp",
      "mctd:3:textbook sle:1:sle2019"),
    F("rf_pos", "류마티스인자 양성", "rheumatoid factor positive", "pos",
      r"류마(티스|토이드)\s*인자|(?<![a-z])rf(?![a-z])|rheumatoid factor", "ra:1:ra2010 sjogren:1:sjogren", thr=15),
    F("ccp_pos", "항CCP 항체 양성", "anti-CCP antibody positive", "pos",
      r"(?<![a-z])ccp(?![a-z])|citrullinated|(?<![a-z])acpa(?![a-z])|시트룰린", "ra:3:ra2010", thr=20),
    F("hlab27_pos", "HLA-B27 양성", "HLA-B27 positive", "pos", r"hla-?\s*b\s*27", "as:2:asas reactive_arthritis:1:asas"),
    F("sacroiliitis", "영상: 천장관절염", "imaging: sacroiliitis", "kw",
      r"천장\s*관절.{0,20}(염|미란|침식|경화|골수\s*부종|강직|erosion|sclerosis)|sacroiliitis|대나무\s*척추|bamboo spine|신데스모파이트|syndesmophyte",
      "as:3:asas"),
    F("anca_pr3_pos", "c-ANCA·PR3 양성", "c-ANCA / anti-PR3 positive", "pos", r"c-?\s*anca|(?<![a-z])pr-?3(?![a-z])|proteinase[- ]?3",
      "gpa:3:gpa"),
    F("anca_mpo_pos", "p-ANCA·MPO 양성", "p-ANCA / anti-MPO positive", "pos",
      r"p-?\s*anca|(?<![a-z])mpo(?![a-z])|myeloperoxidase", "gpa:1:gpa psc:1:psc uc:1:uc"),
    F("anca_pos", "ANCA 양성", "ANCA positive", "pos", r"(?<![a-z])(?<![cp]-)(?<![cp])anca(?![a-z])", "gpa:2:gpa"),
    F("msu_crystal", "관절액: 요산염 결정", "synovial fluid: monosodium urate crystals", "kw",
      r"(요산\s*(나트륨|염)?|monosodium urate|(?<![a-z])msu(?![a-z])|urate).{0,20}결정|(음성\s*복굴절|negative(ly)? birefring)|바늘\s*(모양|형).{0,10}결정|needle-?shaped|요산\s*결정",
      "gout:3:gout2015"),
    F("cppd_crystal", "관절액: 피로인산칼슘 결정", "synovial fluid: CPP crystals", "kw",
      r"(피로인산|칼슘\s*피로인산|(?<![a-z])cppd?(?![a-z])|calcium pyrophosphate).{0,20}결정|양성\s*복굴절|positive(ly)? birefring|(장사방형|마름모|rhomboid)",
      "cppd:3:cppd"),
    F("chondrocalcinosis", "X선: 연골석회화", "X-ray: chondrocalcinosis", "kw", r"연골\s*석회|chondrocalcinosis", "cppd:2:cppd"),
    F("uric_high", "요산 상승", "elevated serum uric acid", "hi", r"(?<!소변\s)요산(?!\s*(나트륨|염|결정))|uric acid",
      "gout:1:gout2015", thr=7.0),
    F("synovial_wbc_high", "관절액 백혈구 증가(>50,000)", "synovial fluid WBC > 50,000/µL", "hi",
      r"(관절액|활액|synovial).{0,30}(백혈구|wbc|세포\s*수|cell count)", "septic_arthritis:3:septic_arthritis", thr=50000, need_value=True),
    F("synovial_pus", "관절액: 혼탁·그람염색/배양 양성", "synovial fluid: purulent / Gram stain or culture positive", "kw",
      r"(관절액|활액|synovial).{0,40}(그람|gram|배양|culture).{0,25}(양성|구균|간균|검출|성장|positive|cocci)|(관절액|활액).{0,15}(농|고름|탁한|혼탁|purulent|turbid)",
      "septic_arthritis:3:septic_arthritis"),
    F("ck_high", "CK 현저한 상승", "markedly elevated creatine kinase", "hi",
      r"(?<![a-z])c[p]?k(?![a-z-])|크레아틴\s*(포스포)?\s*키나[아]?제|creatine (phospho)?kinase",
      "polymyositis:2:myositis dermatomyositis:2:myositis nms:1:textbook", thr=1000, need_value=True),
    F("myositis_ab_pos", "근염 특이 항체 양성", "myositis-specific antibody positive", "pos",
      r"jo-?\s*1|aminoacyl|mda-?\s*5|mi-?\s*2(?![0-9])|tif1|nxp-?\s*2|(?<![a-z])srp(?![a-z])|hmgcr",
      "dermatomyositis:2:myositis polymyositis:2:myositis"),
    F("myopathic_emg", "근전도: 근병증성 소견", "EMG: myopathic pattern", "kw", r"근병증\s*(성|적)?\s*(전위|양상|소견|변화)|myopathic",
      "polymyositis:2:myositis dermatomyositis:2:myositis"),
    F("perifascicular", "근생검: 속주위 위축", "muscle biopsy: perifascicular atrophy", "kw", r"(근)?속\s*주위\s*위축|perifascicular",
      "dermatomyositis:3:myositis"),
    F("esr_very_high", "ESR 현저한 상승(>50)", "ESR > 50 mm/h", "hi",
      r"(?<![a-z])esr(?![a-z])|적혈구\s*침강\s*속도|혈침", "gca:1:gca pmr:1:textbook", thr=50, need_value=True),
    F("temporal_bx", "측두동맥 생검: 거대세포 혈관염", "temporal artery biopsy: giant-cell vasculitis", "kw",
      r"(측두\s*동맥|temporal artery).{0,30}(생검|biopsy|초음파|ultrasound).{0,40}(거대\s*세포|육아종|혈관염|giant cell|granulom|vasculitis|양성|후광|halo)|halo sign|후광\s*징후",
      "gca:3:gca"),
    F("takayasu_imaging", "혈관영상: 대동맥·분지 협착", "angiography: aorta / branch stenosis", "kw",
      r"(쇄골하\s*동맥|대동맥\s*궁|대동맥\s*분지|subclavian|aortic arch).{0,25}(협착|폐색|벽\s*비후|stenosis|occlusion)",
      "takayasu:3:textbook"),
    # ---------------- endocrine
    F("hba1c_high", "당화혈색소 ≥6.5%", "HbA1c ≥ 6.5%", "hi",
      r"hb\s*a1c|당화\s*(혈색소|헤모글로빈)|(?<![a-z])a1c(?![a-z])|glycated h[ae]moglobin|glycosylated",
      "dm:3R:ada2024 t2dm:2:ada2024 t1dm:1:ada2024", thr=6.4),
    F("glucose_very_high", "혈당 현저한 상승(≥250)", "plasma glucose ≥ 250 mg/dL", "hi",
      r"혈당|glucose|(?<![a-z])fbs(?![a-z])|blood sugar|(?<![a-z])bst(?![a-z])|포도당",
      "dm:2:ada2024 dka:1:dka t1dm:1:ada2024", thr=249, units={"mmol": 18.0}, need_value=True),
    F("ketone_pos", "케톤 양성", "ketones positive (serum or urine)", "pos",
      r"케톤|ketone|하이드록시\s*부티르산|hydroxybutyrate|(?<![a-z])bhb(?![a-z])", "dka:3R:dka t1dm:1:dka"),
    F("anion_gap_high", "음이온차 증가", "high anion gap", "hi", r"음이온\s*차|anion gap",
      "dka:2:dka lactic_acidosis:1:textbook", thr=16,
      # "AG" alone is an anion gap only among electrolytes / blood gases ("vWF:Ag 95%", "HBsAg" are antigens)
      alias=r"(?<![a-z:/-])(?<![a-z] )ag(?![a-z])", alias_ctx=_ACID_BASE_CTX),
    F("metabolic_acidosis", "대사성 산증", "metabolic acidosis", "kw", r"대사성\s*산증|metabolic acidosis",
      "dka:1:dka aki:1:aki ckd:1:textbook lactic_acidosis:1:textbook"),
    F("tsh_low", "TSH 억제(저하)", "suppressed TSH", "lo",
      r"(?<![a-z])tsh(?![a-z])(?![\s-]*(?:수용체|receptor))|갑상[선샘]\s*자극\s*호르몬(?!\s*수용체)", "hyperthyroidism:2R:ata_hyper graves:2R:ata_hyper subacute_thyroiditis:1:ata_hyper", thr=0.1),
    F("tsh_high", "TSH 상승", "elevated TSH", "hi",
      r"(?<![a-z])tsh(?![a-z])(?![\s-]*(?:수용체|receptor))|갑상[선샘]\s*자극\s*호르몬(?!\s*수용체)", "hypothyroidism:3R:ata_hypo hashimoto:2:ata_hypo", thr=10),
    F("ft4_high", "유리T4 상승", "elevated free T4", "hi", r"free\s*t4|(?<![a-z])ft4(?![a-z])|유리\s*(t4|티록신)|free thyroxine",
      "hyperthyroidism:2:ata_hyper graves:1:ata_hyper", thr=1.8),
    F("ft4_low", "유리T4 저하", "low free T4", "lo", r"free\s*t4|(?<![a-z])ft4(?![a-z])|유리\s*(t4|티록신)|free thyroxine",
      "hypothyroidism:2:ata_hypo", thr=0.8),
    F("trab_pos", "TSH 수용체 항체 양성", "TSH-receptor antibody positive", "pos",
      r"(?<![a-z])trab(?![a-z])|(?<![a-z])tsi(?![a-z])|tbii|갑상[선샘]\s*자극\s*(면역\s*글로불린|항체)|tsh\s*수용체\s*항체|thyrotropin receptor|tsh receptor",
      "graves:3:ata_hyper hyperthyroidism:2:ata_hyper"),
    F("tpo_pos", "항TPO·항Tg 항체 양성", "anti-TPO / anti-thyroglobulin antibody positive", "pos",
      r"(?<![a-z])tpo|thyroid peroxidase|갑상[선샘]\s*과산화\s*효소|thyroglobulin antibod|anti-?\s*tg(?![a-z])|항\s*-?\s*티로글로불린|항\s*갑상[선샘]",
      "hashimoto:2:ata_hypo hypothyroidism:1:ata_hypo graves:1:ata_hyper"),
    F("thyroid_uptake_high", "갑상선 섭취율 미만성 증가", "diffusely increased thyroid uptake", "kw",
      r"(섭취율|uptake|스캔|scan|신티).{0,30}(미만성|diffuse|전반적).{0,15}(증가|상승|increase)|diffuse(ly)?\s*increased uptake|섭취율.{0,10}(증가|상승)|갑상[선샘].{0,20}(혈류\s*증가|inferno)",
      "graves:3:ata_hyper hyperthyroidism:2:ata_hyper"),
    F("thyroid_uptake_low", "갑상선 섭취율 감소", "low thyroid uptake", "kw",
      r"(섭취율|uptake|스캔|신티).{0,20}(감소|저하|낮|decreas|low|absent|거의\s*없)", "subacute_thyroiditis:2:ata_hyper", fixed=True),
    F("cortisol_low", "아침 코르티솔 저하", "low morning cortisol", "lo", r"코르티솔|cortisol",
      "addison:2:addison adrenal_insufficiency:2:addison", thr=5,
      not_if=r"덱사메타손|dexamethasone|심야|자정|late-?night|midnight|24\s*시간|urinary|소변|타액|saliva"),
    F("acth_stim_fail", "ACTH 자극검사 반응 부족", "inadequate cortisol response to ACTH stimulation", "kw",
      r"(acth|코신트로핀|cosyntropin|synacthen|신악텐|속효성).{0,40}(자극|stimulation|부하).{0,50}(반응\s*(없|저하|부족|미흡|불충분)|불충분|inadequate|blunted|fail|미달|증가\s*(없|않))",
      "addison:3:addison adrenal_insufficiency:3:addison", fixed=True),
    F("acth_high", "ACTH 상승", "elevated ACTH", "hi", r"(?<![a-z])acth(?![a-z])|부신\s*피질\s*자극\s*호르몬",
      "addison:1:addison cushing_disease:1:cushing", thr=100,
      not_if=r"자극|stimulation|cosyntropin|코신트로핀"),
    F("cushing_tests", "코르티솔 과다(덱사메타손 비억제·심야 코르티솔 상승)", "hypercortisolism: no dexamethasone suppression / high late-night cortisol", "kw",
      r"(덱사메타손|dexamethasone|(?<![a-z])dst(?![a-z])).{0,40}(억제\s*(되지\s*않|안\s*됨|실패|없)|비억제|not suppress|failure to suppress|non-?suppress|억제\s*불가)|(심야|자정|late-?night|midnight|야간).{0,15}(타액\s*)?(코르티솔|cortisol).{0,25}(상승|증가|높|elevat)|(24\s*시간\s*)?(소변|요)\s*(유리\s*)?코르티솔.{0,20}(상승|증가|높)|urinary free cortisol.{0,20}(elevat|high|increas)",
      "aldosteronism_cushing:3:cushing cushing_disease:2:cushing", fixed=True),
    F("metanephrine_high", "메타네프린·카테콜아민 상승", "elevated metanephrines / catecholamines", "hi",
      r"메타네프린|metanephrine|노르메타네프린|카테콜아민|catecholamine|(?<![a-z])vma(?![a-z])|바닐릴만델산", "pheo:3R:pheo", thr=None),
    F("aldo_renin_high", "알도스테론/레닌 비 상승", "high aldosterone-to-renin ratio", "kw",
      r"알도스테론\s*/?\s*레닌.{0,20}(비|ratio)?.{0,15}(상승|증가|높|\d{2,})|aldosterone.{0,5}renin ratio.{0,20}(elevat|high|\d{2,})|(?<![a-z])arr(?![a-z]).{0,15}(상승|증가|높|elevat|\d{2,})|레닌.{0,10}(억제|저하|감소|낮)|suppressed renin|알도스테론.{0,15}(상승|증가|높)",
      "aldosteronism_cushing:3:aldo conn:3:aldo", fixed=True),
    F("igf1_high", "IGF-1 상승", "elevated IGF-1", "hi", r"igf-?\s*1|인슐린\s*유사\s*성장\s*인자|somatomedin",
      "acromegaly:3R:acromegaly", thr=None),
    F("gh_not_suppressed", "경구당부하 후 성장호르몬 비억제", "GH not suppressed after oral glucose", "kw",
      r"(성장\s*호르몬|(?<![a-z])gh(?![a-z])|growth hormone).{0,40}(억제\s*(되지\s*않|안|실패)|비억제|not suppress|failure to suppress)",
      "acromegaly:3:acromegaly", fixed=True),
    F("prolactin_high", "프로락틴 상승", "elevated prolactin", "hi", r"프로락틴|prolactin|(?<![a-z])prl(?![a-z])",
      "hyperprolactinemia:3R:prolactin prolactinoma:2:prolactin pituitary_adenoma:1:prolactin", thr=25),
    F("pituitary_mass", "MRI: 뇌하수체 선종·종괴", "MRI: pituitary adenoma / sellar mass", "kw",
      r"뇌하수체.{0,20}(선종|샘종|종괴|종양|mass|adenoma|미세\s*선종|거대\s*선종)|pituitary (macro|micro)?adenoma|(macro|micro)adenoma|(터키\s*안장|안장|sella).{0,15}(종괴|mass|종양)",
      "pituitary_adenoma:3:prolactin prolactinoma:1:prolactin acromegaly:1:acromegaly"),
    F("pth_high", "PTH 상승", "elevated parathyroid hormone", "hi",
      r"(?<![a-z])i?pth(?![a-z])|부갑상[선샘]\s*호르몬|parathyroid hormone|파라토르몬", "primary_hpt:2:hpt hpt:2:hpt ckd:1:textbook", thr=65),
    F("calcium_high", "고칼슘혈증", "hypercalcemia", "hi",
      r"(?<![a-z])ca(?![a-z])(?!\s*-?\s*(19|125|15|72|242|50))|칼슘|calcium", "primary_hpt:1:hpt hpt:1:hpt mm:1:imwg", thr=10.5,
      not_if=r"피로인산|pyrophosphate|석회|calcif|옥살산|oxalate|결석"),
    F("siadh_urine", "소변 삼투압·나트륨 부적절 상승", "inappropriately concentrated urine / high urine sodium", "kw",
      r"(소변|요)\s*(삼투압|osm\w*)[^\d,]{0,12}([3-9]\d{2}|1\d{3})|(소변|요)\s*(삼투압|osm\w*).{0,12}(높|증가|상승)|urine osm\w*[^\d,]{0,15}([3-9]\d{2}|1\d{3})|urine osm\w*.{0,15}(high|inappropriat)|(소변|요)\s*나트륨.{0,15}(>\s*)?([4-9]\d|\d{3}|높|증가)|urine sodium.{0,15}(>\s*)?([4-9]\d|\d{3})",
      "siadh:2:hyponat", fixed=True),
    F("dilute_urine", "저장뇨·물제한 후 농축 실패", "dilute urine / failure to concentrate on water deprivation", "kw",
      r"(소변|요)\s*(삼투압|osm\w*).{0,20}(저하|낮|감소|<\s*[1-3]\d{2}|[1-2]\d{2}\s*mosm)|저장뇨|hyposthenuria|dilute urine|(물\s*제한|수분\s*제한|water deprivation).{0,40}(농축\s*(안|되지|실패)|증가\s*(없|않)|fail)",
      "di:2:di central_di:1:di nephrogenic_di:1:di", fixed=True),
    F("desmopressin_response", "데스모프레신 투여 후 소변 농축", "urine concentrates after desmopressin", "kw",
      r"(데스모프레신|desmopressin|ddavp|바소프레신|vasopressin).{0,40}(후|투여).{0,30}(증가|상승|농축|responds|increase|\d{2,3}\s*%)",
      "central_di:3:di di:2:di", fixed=True, not_if=r"(반응|변화|증가)\s*(없|않)|no response|unrespons"),
    F("desmopressin_no_response", "데스모프레신 투여에도 농축 없음", "no urine concentration after desmopressin", "kw",
      r"(데스모프레신|desmopressin|ddavp|바소프레신|vasopressin).{0,50}((반응|변화|증가)\s*(없|않)|no response|unrespons|no (change|increase))",
      "nephrogenic_di:3:di di:2:di", fixed=True),
    F("pcos_us", "초음파: 다낭성 난소", "ultrasound: polycystic ovaries", "kw",
      r"다낭성\s*난소|polycystic ovar|(난포|follicle).{0,15}(1[2-9]|[2-9]\d|다수|여러\s*개|string of pearls|염주|목걸이)|목걸이\s*모양",
      "pcos:2:pcos"),
    F("hcg_pos", "β-hCG 양성", "beta-hCG positive", "pos",
      r"(β|b|베타|beta)\s*-?\s*hcg|(?<![a-z])hcg(?![a-z])|임신\s*(반응|검사)|소변\s*임신", "ectopic:1R:acog_ectopic"),
    F("us_no_iup", "초음파: 자궁 내 임신낭 없음·부속기 종괴", "ultrasound: empty uterus / adnexal mass in pregnancy", "kw",
      r"자궁\s*(내|안).{0,15}(임신낭|태낭|gestational sac|임신).{0,15}(없|관찰되지\s*않|보이지\s*않|확인되지\s*않)|no intrauterine (pregnancy|gestational)|empty uterus|빈\s*자궁|(부속기|난관|자궁\s*외|adnex|tub).{0,20}(임신낭|태낭|gestational sac|ring|고리)",
      "ectopic:3:acog_ectopic", fixed=True),
    F("testis_no_flow", "초음파: 고환 혈류 감소·소실", "ultrasound: absent testicular blood flow", "kw",
      r"(고환|testic|정소).{0,25}(혈류.{0,6}(감소|소실|없|저하)|무혈류|decreased flow|absent flow|no flow)|whirlpool|정삭.{0,10}꼬임",
      "testicular_torsion:3:textbook", fixed=True,
      not_if=r"(감소|소실|저하)[^,]{0,8}(없|않)|정상\s*혈류|혈류.{0,6}(정상|유지|보존|증가)"),
    # ---------------- hematology / oncology
    F("ferritin_low", "페리틴 감소", "low ferritin", "lo", r"페리틴|ferritin", "ida:3R:aga_ida hypochromic_anemia:3R:aga_ida", thr=30),
    F("ferritin_very_high", "페리틴 현저한 상승", "markedly elevated ferritin", "hi", r"페리틴|ferritin",
      "aosd:2:textbook hlh:2:textbook hemochromatosis:1:hh", thr=1000, need_value=True),
    F("tsat_high", "트랜스페린 포화도 상승", "elevated transferrin saturation", "hi",
      r"트랜스페린\s*포화도|transferrin sat\w*|(?<![a-z])tsat(?![a-z])|철\s*포화도",
      "hemochromatosis:2:hh", thr=45,
      # "포화도" alone is a transferrin saturation only in an iron panel (else oxygen / venous saturation)
      alias=r"(?<!산소)(?<!산소 )(?<![a-z])포화도(?=\s*[:]?\s*\d)", alias_ctx=_IRON_CTX),
    F("mcv_high", "대적혈구(MCV 증가)", "macrocytosis (high MCV)", "hi", r"(?<![a-z])mcv(?![a-z])|평균\s*적혈구\s*(용적|부피)",
      "b12:2:b12 pernicious:1:b12 megaloblastic:2:b12", thr=100),
    F("mcv_low", "소적혈구(MCV 감소)", "microcytosis (low MCV)", "lo", r"(?<![a-z])mcv(?![a-z])|평균\s*적혈구\s*(용적|부피)",
      "ida:1:aga_ida hypochromic_anemia:1:aga_ida thalassemia:1:textbook beta_thal:1:textbook", thr=80),
    F("b12_low", "비타민 B12 저하", "low vitamin B12", "lo", r"비타민\s*b\s*-?\s*12|(?<![a-z])b\s*-?\s*12(?![\d.])|코발라민|cobalamin",
      "b12:3R:b12 pernicious:2:b12 megaloblastic:2:b12", thr=200, not_if=r"결핍증?\s*(병력|진단)"),
    F("mma_high", "메틸말론산·호모시스테인 상승", "elevated methylmalonic acid / homocysteine", "hi",
      r"메틸\s*말론산|methylmalonic|(?<![a-z])mma(?![a-z])|호모시스테인|homocysteine", "b12:2:b12", thr=None),
    F("if_ab_pos", "내인자·벽세포 항체 양성", "anti-intrinsic factor / parietal cell antibody positive", "pos",
      r"내인\s*(성\s*)?인자\s*항체|intrinsic factor|벽\s*세포\s*항체|parietal cell|항\s*벽세포|항\s*내인자", "pernicious:3:b12"),
    F("schistocytes", "말초혈액: 분열적혈구", "blood smear: schistocytes", "kw",
      r"분열\s*적혈구|파쇄\s*적혈구|조각\s*적혈구|schistocyt|헬멧\s*세포|helmet cell|fragmented (rbc|red)",
      "ttp:2:ttp hus:2:ttp dic:1:isth_dic"),
    F("adamts13_low", "ADAMTS13 활성도 <10%", "ADAMTS13 activity < 10%", "lo", r"adamts-?\s*13", "ttp:3R:ttp", thr=10),
    F("haptoglobin_low", "합토글로빈 감소", "low haptoglobin", "lo", r"합토글로빈|haptoglobin",
      "hemolytic_anemia:2:aiha aiha:1:aiha ttp:1:ttp hus:1:ttp hs:1:textbook pnh:1:textbook", thr=30),
    F("dat_pos", "직접항글로불린(쿰스) 검사 양성", "direct antiglobulin (Coombs) test positive", "pos",
      r"직접\s*(항\s*글로불린|쿰스|coombs)|direct (antiglobulin|coombs)|(?<![a-z])dat(?![a-z])|(?<!간접\s)(?<!간접)쿰스|(?<!indirect )coombs",
      "aiha:3:aiha hemolytic_anemia:2:aiha"),
    F("spherocytes", "구상적혈구·삼투압 취약성 증가", "spherocytes / increased osmotic fragility", "kw",
      r"구상\s*적혈구|spherocyt|osmotic fragility|삼투압?\s*취약성.{0,10}(증가|상승)|ema\s*결합", "hs:2:textbook aiha:1:aiha"),
    F("flow_pnh", "유세포분석: CD55/CD59 결핍", "flow cytometry: CD55/CD59 (GPI) deficiency", "kw",
      r"cd55|cd59|flaer|gpi\s*(결핍|anchor)", "pnh:3:textbook"),
    F("hba2_high", "HbA2 상승", "elevated HbA2", "hi", r"hb\s*a2|혈색소\s*a2", "beta_thal:3:textbook thalassemia:2:textbook", thr=3.5),
    F("blasts", "모세포 증가", "increased blasts", "kw",
      r"(모세포|아세포|(?<![a-z])blast).{0,20}(\d{2}\s*%|증가|다수|관찰|출현|>)|골수.{0,25}(모세포|아세포|blast)|아우어|auer",
      "acute_leukemia:3:aml aml:2:aml all:2:aml"),
    F("auer", "아우어 소체", "Auer rods", "kw", r"아우어|auer", "aml:3:aml"),
    F("bcr_abl", "BCR-ABL·필라델피아 염색체 양성", "BCR-ABL / Philadelphia chromosome positive", "pos",
      r"bcr-?\s*abl|필라델피아|philadelphia|t\s*\(\s*9\s*;\s*22", "cml:3:cml all:1:cml"),
    F("jak2_pos", "JAK2 V617F 양성", "JAK2 V617F positive", "pos", r"jak-?\s*2|v617f", "pv:3:textbook et:2:textbook"),
    F("m_protein", "M 단백·형질세포 증가·용골성 병변", "M protein / clonal plasma cells / lytic bone lesions", "kw",
      r"(?<![a-z])m\s*-?\s*(단백|protein|spike|peak|band)|단(일)?클론\s*(성\s*)?(단백|감마|면역\s*글로불린|띠)|monoclonal (protein|gammopathy|band|spike)|bence[- ]?jones|벤스\s*존스|유리\s*경쇄.{0,25}(비|ratio).{0,15}(증가|상승|비정상|이상)|free light chain.{0,25}(abnormal|elevat)|형질\s*세포.{0,25}(\d{2}\s*%|증가)|plasma cells?.{0,20}(\d{2}\s*%|increas)|천공\s*성?\s*(골\s*)?병변|punched[- ]?out|용골\s*성\s*병변|골\s*용해\s*성\s*병변|lytic (bone )?lesion",
      "mm:3:imwg"),
    F("fibrinogen_low", "피브리노겐 감소", "low fibrinogen", "lo", r"피브리노겐|fibrinogen", "dic:2:isth_dic", thr=150),
    F("fdp_high", "FDP 상승", "elevated fibrin degradation products", "hi", r"(?<![a-z])fdp(?![a-z])|섬유소\s*분해\s*산물|fibrin degradation",
      "dic:2:isth_dic", thr=None),
    F("reed_sternberg", "생검: 리드-스턴버그 세포", "biopsy: Reed-Sternberg cells", "kw",
      r"리드\s*-?\s*스턴버그|reed[- ]?sternberg|cd15.{0,15}cd30|cd30.{0,15}cd15", "hodgkin:3:textbook"),
    F("lung_ca_bx", "생검: 폐암(선암·편평상피암·소세포암)", "biopsy: lung carcinoma", "kw",
      r"(폐|기관지|lung|bronch).{0,25}(선암|편평\s*(상피\s*)?세포\s*암|소세포\s*암|adenocarcinoma|squamous cell carcinoma|small cell carcinoma|악성\s*세포)",
      "lung_cancer:3:textbook"),
    F("lung_mass", "흉부영상: 폐 종괴", "chest imaging: lung mass", "kw",
      r"(폐|폐야|폐엽|하엽|상엽|lung).{0,20}(종괴|mass|종양|spiculat|침상)|폐\s*결절.{0,15}(\d\s*cm|침상|spiculat)",
      "lung_cancer:2:textbook"),
    F("psa_high", "PSA 상승", "elevated PSA", "hi", r"(?<![a-z])psa(?![a-z])|전립[선샘]\s*특이\s*항원", "prostate_cancer:1:textbook", thr=10),
    F("ca125_high", "CA-125 상승", "elevated CA-125", "hi", r"ca\s*-?\s*125", "ovarian_cancer:1:textbook", thr=200),
    F("ca199_high", "CA 19-9 상승", "elevated CA 19-9", "hi", r"ca\s*-?\s*19\s*-?\s*9",
      "pancreatic_cancer:1:textbook cholangiocarcinoma:1:psc", thr=100),
    F("cea_high", "CEA 상승", "elevated CEA", "hi", r"(?<![a-z])cea(?![a-z])|암\s*배아\s*항원", "crc:1:textbook", thr=10),
    # ---------------- neurology
    F("achr_pos", "아세틸콜린수용체·MuSK 항체 양성", "anti-AChR / anti-MuSK antibody positive", "pos",
      r"아세틸\s*콜린\s*수용체|항\s*아세틸\s*콜린|(?<![a-z])achr|acetylcholine receptor|musk", "mg:3:mg"),
    F("rns_decrement", "반복신경자극: 감쇠 반응", "repetitive nerve stimulation: decrement", "kw",
      r"(반복\s*(신경\s*)?자극|repetitive (nerve )?stimulation|rns).{0,40}(감소|감쇠|점감|decrement)|감쇠\s*반응|점감\s*반응|decremental|single[- ]fiber.{0,20}(jitter|지터).{0,10}(증가|increas)|(텐실론|edrophonium|얼음\s*(팩\s*)?검사|ice pack).{0,20}(양성|호전|개선|positive)",
      "mg:3:mg"),
    F("csf_albcyto", "뇌척수액: 알부민세포해리", "CSF: albuminocytologic dissociation", "kw",
      r"(알부민|단백)\s*-?\s*세포\s*해리|albuminocytologic|cytoalbumin|(뇌척수액|csf).{0,40}단백.{0,10}(상승|증가|높).{0,40}(세포|백혈구).{0,10}(정상|\d\s*(개|/))",
      "gbs:3:gbs"),
    F("ncs_demyelination", "신경전도: 탈수초·전도차단", "nerve conduction: demyelination / conduction block", "kw",
      r"(신경\s*전도|ncs|nerve conduction).{0,50}(탈수초|전도\s*차단|전도\s*속도.{0,6}(저하|감소|지연)|원위\s*잠시.{0,6}연장|demyelinat|conduction block|slow)",
      "gbs:2:gbs"),
    F("csf_bacterial", "뇌척수액: 호중구 증가·당 감소(세균성)", "CSF: neutrophilic pleocytosis / low glucose", "kw",
      r"(뇌척수액|csf|요추\s*천자|척수액).{0,80}(호중구|다형핵|neutrophil|polymorph|혼탁|탁한|cloudy|turbid|포도당\s*(감소|저하|낮)|glucose\s*(low|decreas)|그람\s*(양성|음성)|gram[- ](positive|negative))",
      "bacterial_meningitis:3R:mening meningitis:2:mening"),
    F("csf_lymphocytic", "뇌척수액: 림프구 우세 세포 증가", "CSF: lymphocytic pleocytosis", "kw",
      r"(뇌척수액|csf|척수액).{0,80}(림프구|lymphocyt|단핵구|mononuclear)",
      "viral_meningitis:2:mening meningitis:1:mening encephalitis:1:textbook viral_encephalitis:1:textbook crypto_meningitis:1:textbook"),
    F("hsv_pcr_pos", "뇌척수액 HSV PCR 양성", "CSF HSV PCR positive", "pos",
      r"((?<![a-z])hsv|herpes simplex|단순\s*포진).{0,20}(pcr|dna)", "encephalitis:3:textbook viral_encephalitis:3:textbook"),
    F("crypto_pos", "크립토콕쿠스 항원·묵즙 양성", "cryptococcal antigen / India ink positive", "pos",
      r"묵즙|india ink|cryptococ|크립토콕[쿠커]스|(?<![a-z])crag(?![a-z])", "crypto_meningitis:3:textbook"),
    F("temporal_lobe_mri", "MRI: 측두엽 신호 증가", "MRI: temporal-lobe hyperintensity", "kw",
      r"(측두엽|temporal lobe).{0,25}(고신호|신호\s*(강도\s*)?증가|hyperintens|부종|edema)", "encephalitis:2:textbook viral_encephalitis:2:textbook"),
    F("xanthochromia", "뇌척수액 황색변색", "CSF xanthochromia", "kw", r"황색\s*변색|황변|xanthochrom", "sah:3:sah"),
    F("ct_sah", "CT: 지주막하 출혈", "CT: subarachnoid hemorrhage", "kw",
      r"(지주막\s*하|거미막\s*하|subarachnoid).{0,20}(출혈|고음영|hemorrhage|blood|혈액)|기저\s*수조.{0,15}(고음영|출혈)",
      "sah:3R:sah"),
    F("ct_ich", "CT: 뇌실질 내 출혈", "CT: intracerebral hemorrhage", "kw",
      r"(뇌실질\s*(내)?|뇌\s*내|intracerebral|intraparenchymal|기저핵|시상|피각|putamen|putaminal|thalam|소뇌|cerebellar|뇌엽|lobar).{0,20}(출혈|혈종|hemorrhage|hematoma)|뇌\s*출혈",
      "hemorrhagic_stroke:3R:ich"),
    F("stroke_imaging", "영상: 급성 뇌경색(확산제한)", "imaging: acute infarct (diffusion restriction)", "kw",
      r"확산\s*강조.{0,30}(고신호|제한|high)|diffusion.{0,20}restrict|dwi.{0,25}(고신호|high|restrict|제한)|급성\s*(뇌)?경색|acute infarct|(허혈성|급성).{0,10}뇌경색|뇌경색\s*소견|(중대뇌|mca|대혈관).{0,20}(폐색|occlusion)|large vessel occlusion|조기\s*허혈\s*변화|early ischemic change|고밀도\s*중대뇌동맥|hyperdense mca",
      "ischemic_stroke:3:aha_stroke cerebral_infarction:3:aha_stroke"),
    F("cvst_imaging", "영상: 정맥동 혈전", "imaging: cerebral venous sinus thrombosis", "kw",
      r"(정맥동|상시상|횡정맥동|구불정맥동|venous sinus|transverse sinus|sagittal sinus).{0,45}(혈전|충만\s*결손|폐색|고음영|thromb|filling defect|hyperdens)|empty delta|빈\s*델타|cord sign",
      "cvst:3:textbook"),
    F("mri_ms", "MRI: 탈수초 병변(뇌실주위·피질인접)", "MRI: periventricular / juxtacortical demyelinating lesions", "kw",
      r"(뇌실\s*주위|측뇌실\s*주위|periventricular|피질\s*(인접|근처)|juxtacortical|천막\s*하|infratentorial).{0,30}(병변|고신호|lesion|hyperintens|탈수초)|dawson|탈수초\s*(성\s*)?병변|demyelinating lesion|조영\s*증강.{0,10}(병변|lesion).{0,20}(동시|새로운)",
      "ms:3:mcdonald"),
    F("oligoclonal", "뇌척수액 올리고클론띠 양성", "CSF oligoclonal bands", "pos", r"올리고\s*클론|oligoclonal", "ms:2:mcdonald"),
    F("eeg_epileptiform", "뇌파: 간질양 방전", "EEG: epileptiform discharges", "kw",
      r"(eeg|뇌파).{0,40}(극파|예파|간질\s*양|뇌전증\s*양|epileptiform|spike|sharp wave|3\s*hz|극서파)",
      "epilepsy:2:epilepsy", not_if=r"(없|정상|normal|no\s)"),
    F("wernicke_mri", "MRI: 유두체·시상 내측 신호 증가", "MRI: mammillary / medial thalamic hyperintensity", "kw",
      r"(유두체|mammillary|시상\s*내측|medial thalam|수도관\s*주위|periaqueductal).{0,25}(고신호|신호\s*증가|hyperintens|위축)",
      "wernicke:3:textbook"),
    F("nph_imaging", "영상: 비례에 맞지 않는 뇌실 확장", "imaging: ventriculomegaly out of proportion", "kw",
      r"뇌실.{0,6}(확장|확대).{0,25}(비례|불균형|위축에\s*비해|out of proportion)|ventriculomegaly|에반스\s*지수|evans index|desh",
      "nph:2:textbook"),
    F("als_emg", "근전도: 광범위 탈신경", "EMG: widespread denervation", "kw",
      r"(광범위|다분절|여러\s*분절|widespread|diffuse).{0,25}(탈신경|denervation|섬유속\s*(성\s*)?전위|fasciculation)",
      "als:3:textbook"),
    F("gastric_mass", "내시경: 위 종괴·융기성 궤양 병변", "endoscopy: gastric mass / ulcerated elevated lesion", "kw",
      r"(?<![가-힣])위(\s*(체부|전정부|각부|저부|분문부|유문부|소만|대만|벽)|\s)(?!치).{0,25}(종괴|종양|융기성\s*병변|궤양을\s*동반한.{0,15}병변|mass|tumor|보르만|borrmann)",
      "gastric_cancer:2:textbook"),
    F("biliary_mass", "영상·생검: 담관 종괴·선암", "imaging/biopsy: bile-duct mass or adenocarcinoma", "kw",
      r"(담관|담도|간문부|bile duct|hilar).{0,30}(종괴|종양|mass|tumor)|클라스킨|klatskin|(담도|담관)\s*(생검|솔\s*세포).{0,30}(선암|adenocarcinoma|악성)",
      "cholangiocarcinoma:3:psc"),
    F("factor8_low", "제8인자 활성 감소", "low factor VIII activity", "lo", r"(제\s*)?(8|viii)\s*인자|factor\s*(8|viii)|(?<![a-z])fviii",
      "hemophilia_a:3:textbook vwd:1:textbook", thr=40),
    F("factor9_low", "제9인자 활성 감소", "low factor IX activity", "lo", r"(제\s*)?(9|ix)\s*인자|factor\s*(9|ix)(?![a-z])|(?<![a-z])fix(?=\s*(활성|activity))",
      "hemophilia_b:3:textbook", thr=40),
    F("vwf_low", "폰빌레브란트인자 감소", "low von Willebrand factor", "lo", r"폰\s*빌레브란트\s*인자|von willebrand factor|vwf|ristocetin|리스토세틴",
      "vwd:3:textbook", thr=40),
    F("aptt_prolonged", "aPTT 단독 연장", "prolonged aPTT", "hi", r"a?ptt|활성화\s*부분\s*트롬보플라스틴",
      "hemophilia_a:1:textbook hemophilia_b:1:textbook vwd:1:textbook", thr=45, need_value=True),
    F("small_kidneys", "초음파: 양측 신장 크기 감소·피질 에코 증가", "ultrasound: small echogenic kidneys", "kw",
      r"(양측\s*)?(신장|콩팥|kidney).{0,15}(크기\s*(가\s*)?(감소|작|위축)|위축|small|atroph)|(피질|cortical).{0,10}(에코|음영).{0,6}증가|increased cortical echogenicity",
      "ckd:3:textbook"),
    F("atn_casts", "소변: 과립원주(진흙색 원주)", "urine: muddy-brown granular casts", "kw",
      r"과립\s*원주|진흙\s*(갈색|색)\s*원주|muddy[- ]brown|granular cast|세뇨관\s*상피\s*세포\s*원주|renal tubular epithelial cell",
      "atn:3:aki aki:2:aki"),
    # ---------------- tropical / zoonotic (Korea-relevant) and others
    F("scrub_pos", "쯔쯔가무시 검사 양성·가피", "scrub typhus serology / PCR positive or eschar", "kw",
      r"(쯔쯔가무시|쓰쓰가무시|tsutsugamushi|orientia|scrub typhus).{0,25}(항체|igm|igg|ifa|pcr|양성|역가|간접\s*면역)|가피|eschar|검은\s*딱지",
      "scrub_typhus:3:textbook"),
    F("hantaan_pos", "한타바이러스 항체 양성", "hantavirus serology positive", "kw",
      r"(한탄|hantaan|hantavirus|한타\s*바이러스|유행성\s*출혈열|신증후군\s*출혈열).{0,25}(항체|igm|ifa|pcr|양성|역가)",
      "hfrs:3:textbook"),
    F("sfts_pos", "SFTS 바이러스 PCR 양성", "SFTS virus PCR positive", "kw",
      r"(sfts|중증\s*열성\s*혈소판\s*감소).{0,25}(pcr|rt-?pcr|양성|검출)", "sfts:3:textbook"),
    F("dengue_pos", "뎅기 NS1·IgM 양성", "dengue NS1 / IgM positive", "kw", r"(뎅기|dengue).{0,20}(ns1|igm|pcr|항원|양성)|ns1\s*(항원|antigen)",
      "dengue:3:textbook"),
    F("lepto_pos", "렙토스피라 검사 양성", "Leptospira test positive", "kw", r"(렙토스피라|leptospir).{0,25}(항체|mat|pcr|양성|igm|역가)",
      "leptospirosis:3:textbook"),
    F("malaria_pos", "말라리아 도말·항원 양성", "malaria smear / antigen positive", "kw",
      r"(말라리아|malaria|plasmodium|열원충).{0,25}(도말|smear|원충|rdt|신속|항원|양성|관찰|검출|pcr)|(plasmodium|말라리아)\s*(vivax|falciparum|삼일열|열대열)|삼일열\s*(말라리아\s*)?원충",
      "malaria:3:malaria"),
    F("brucella_pos", "브루셀라 검사 양성", "Brucella test positive", "kw", r"(브루셀라|brucella).{0,25}(항체|응집|배양|양성|pcr)",
      "brucellosis:3:textbook"),
    F("qfever_pos", "Q열(Coxiella) 항체 양성", "Coxiella burnetii serology positive", "kw", r"(큐\s*열|q\s*fever|coxiella).{0,25}(항체|igg|igm|양성|phase)",
      "q_fever:3:textbook"),
    F("lyme_pos", "라임병 항체 양성", "Lyme serology positive", "kw", r"(라임|lyme|borrelia|보렐리아).{0,25}(항체|elisa|western|웨스턴|immunoblot|양성)",
      "lyme:3:lyme"),
    F("toxo_pos", "톡소플라스마 IgM 양성", "Toxoplasma IgM positive", "kw", r"(톡소|toxoplasm).{0,20}(igm|양성|pcr)",
      "toxoplasmosis:3:textbook"),
    F("hiv_pos", "HIV 검사 양성", "HIV test positive", "pos", r"(?<![a-z])hiv(?![a-z])|에이즈\s*검사|인간\s*면역\s*결핍", "hiv:3:textbook"),
    F("syphilis_pos", "매독 혈청검사 양성", "syphilis serology positive", "pos",
      r"(?<![a-z])(vdrl|rpr|tpha|tppa|tp-pa|fta-?abs)(?![a-z])|매독", "syphilis:3:sti"),
    F("gonorrhea_pos", "임균 검사 양성", "N. gonorrhoeae test positive", "kw",
      r"(임균|임질|gonorrh|neisseria gonorrhoeae).{0,25}(pcr|naat|배양|양성|검출)|그람\s*음성\s*(세포\s*내\s*)?쌍구균|gram-?negative (intracellular )?diplococ",
      "gonorrhea:3:sti", not_if=r"뇌척수액|csf|수막"),
    F("chlamydia_pos", "클라미디아 검사 양성", "Chlamydia trachomatis test positive", "kw", r"(클라미디아|chlamydia).{0,25}(pcr|naat|양성|검출)",
      "chlamydia:3:sti pid:1:sti"),
    F("trich_pos", "질 트리코모나스 검출", "Trichomonas detected", "kw", r"(트리코모나스|trichomonas).{0,25}(관찰|양성|검출|pcr|naat)|운동성\s*편모충",
      "trichomoniasis:3:sti"),
    F("clue_cells", "단서세포·아민 냄새", "clue cells / positive whiff test", "kw", r"단서\s*세포|clue cell|whiff|아민\s*냄새|amine odor",
      "bv:3:sti"),
    F("co_hb_high", "일산화탄소헤모글로빈 상승", "elevated carboxyhemoglobin", "hi",
      r"일산화\s*탄소\s*헤모글로빈|카[르]?복시\s*헤모글로빈|carboxy\s*h[ae]moglobin|(?<![a-z])co-?\s*hb(?![a-z])|(?<![a-z])cohb",
      "co_poisoning:3:co", thr=9),
    F("methb_high", "메트헤모글로빈 상승", "elevated methemoglobin", "hi", r"메트\s*헤모글로빈|methemoglobin|met-?\s*hb",
      "methb:3:textbook", thr=5),
    F("apap_level_high", "아세트아미노펜 혈중농도 상승", "elevated acetaminophen level", "hi",
      r"(아세트아미노펜|acetaminophen|paracetamol|타이레놀)\s*(혈중\s*)?(농도|level|수치|concentration)", "apap:3:textbook", thr=20),
    F("lactate_high", "젖산 상승", "elevated lactate", "hi", r"젖산|락테이트|lactate|lactic acid",
      "sepsis:1:sepsis3 lactic_acidosis:2:textbook ischemic_colitis:1:textbook", thr=4),
    F("procalcitonin_high", "프로칼시토닌 상승", "elevated procalcitonin", "hi", r"프로칼시토닌|procalcitonin|(?<![a-z])pct(?![a-z])",
      "sepsis:1:sepsis3 bacterial_pneumonia:1:cap", thr=0.5),
    F("osteomyelitis_mri", "MRI: 골수 부종·골수염", "MRI: marrow edema / osteomyelitis", "kw",
      r"골수\s*(부종|염|신호\s*변화)|bone marrow edema|osteomyelitis|골\s*파괴.{0,10}(골막|periosteal)",
      "osteomyelitis:3:textbook"),
]

# Alternative units (unit prefix, lower-case, "μ" for micro → multiplier to the threshold's unit). Standard SI ↔
# conventional conversion factors (molar masses; textbook). Units not listed are taken to be the threshold's unit.
_MORE_UNITS: dict[str, dict[str, float]] = {
    "bnp_high": {"pmol": 3.47},                     # BNP 1 pmol/L = 3.47 pg/mL
    "ntprobnp_high": {"pmol": 8.46},                # NT-proBNP 1 pmol/L = 8.46 pg/mL
    "igg_high": {"g/l": 100.0},                     # g/L → mg/dL
    "ceruloplasmin_low": {"mg/l": 0.1, "g/l": 100.0},
    "afp_high": {"iu/ml": 1.21, "kiu/l": 1.21},     # AFP 1 IU/mL = 1.21 ng/mL
    "uric_high": {"μmol": 1 / 59.48},               # μmol/L → mg/dL
    "ft4_high": {"pmol": 0.0777}, "ft4_low": {"pmol": 0.0777},  # pmol/L → ng/dL
    "cortisol_low": {"nmol": 0.0362},               # nmol/L → μg/dL
    "acth_high": {"pmol": 4.54},                    # pmol/L → pg/mL
    "prolactin_high": {"miu/l": 0.0472, "mu/l": 0.0472, "μiu/ml": 0.0472},  # mIU/L → ng/mL (WHO 84/500)
    "pth_high": {"pmol": 9.43},                     # pmol/L → pg/mL
    "calcium_high": {"mmol": 4.008},                # mmol/L → mg/dL
    "b12_low": {"pmol": 1.355},                     # pmol/L → pg/mL
    "haptoglobin_low": {"g/l": 100.0},
    "fibrinogen_low": {"g/l": 100.0},
    "lactate_high": {"mg/dl": 0.111},               # mg/dL → mmol/L
    "apap_level_high": {"μmol": 0.151},             # μmol/L → μg/mL
    "glucose_very_high": {"mmol": 18.0},
}
# Findings defined by an absolute cut-off rather than by "above/below the reference range"
_CUTOFF = {"ast_alt_very_high", "synovial_wbc_high", "ck_high", "esr_very_high", "hba1c_high", "glucose_very_high",
           "ferritin_very_high", "adamts13_low", "proteinuria_nephrotic", "fev1fvc_low"}
for _f in FINDINGS:
    if _f.id in _MORE_UNITS:
        _f.units = {**_f.units, **_MORE_UNITS[_f.id]}
    if _f.id in _CUTOFF:
        _f.cutoff = True

# ------------------------------------------------------------------ detection
_NUM = re.compile(r"([<>≤≥]=?)?\s*(\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)\s*(%|[a-zμµ/.³^0-9]+)?")
_TITER = re.compile(r"1\s*:\s*(\d{2,5})")
_FOLD = re.compile(r"(\d+(?:\.\d+)?)\s*배")
# polarity words (lower-cased text); the earliest one after the analyte wins
_POS_W = r"양성|positive|\(\s*\+\s*\)|(?<![\d\w])\+{1,4}(?![\d\w])|검출(?!\s*(되지|안|않|없|불가))|detected|(?<!non)(?<!non-)reactive|상승|증가|높|항진|elevat|increas|high|raised|↑|초과|above|abnormal|(?<!특)이상(?!\s*(없|무))|의심|합당|부합|compatible|consistent|suggest"
_NEG_W = r"음성|negative|\(\s*-\s*\)|(?<!비)정상|normal|within|wnl|unremarkable|없|않|미검출|not detected|non-?reactive|absent|부재|(?<![a-z])no(?![a-z])|(?<![a-z])none(?![a-z])|아님|미만\s*\(정상|-\s*$"
_DOWN_W = r"감소|저하|낮|결핍|부족|decreas|(?<![a-z])low|reduced|deficien|↓|below|억제|suppress"
_POL = re.compile(f"(?P<neg>{_NEG_W})|(?P<pos>{_POS_W})|(?P<down>{_DOWN_W})")
_NEG_AFTER_POS = re.compile(r"^\s*(은|는|이|가|도)?\s*(없|않|아님|안\s|no\b|not\b|none)")
_NEG_BEFORE = re.compile(r"((?<![a-z])no|without|negative for|음성인|없는|(?<!비)정상)\s*$")
_POS_BEFORE = re.compile(r"(상승된|증가된|높은|양성인|elevated|high|raised|increased|positive)\s*$")
_DOWN_BEFORE = re.compile(r"(감소된|저하된|낮은|low|decreased|reduced)\s*$")
_PENDING = re.compile(r"진행\s*중|pending|대기|예정|결과\s*(미|안\s*나)|의뢰|보냄|(?<![a-z])sent(?![a-z])|시행하지|미시행|not done|not performed|검사\s*필요")
_RULEOUT = re.compile(r"배제|감별\s*(위해|을\s*위해)|r/o(?![a-z])|rule out|확인\s*위해|평가\s*위해|위해\s*시행|고려|권고|필요")
_HISTORY = re.compile(r"병력|과거력|과거에|history of|진단\s*받|앓았|치료\s*(중|받)|복용\s*중|수술\s*받|가족력")
# a parenthesis with a reference word is a reference range wherever it stands ("(정상 0.4-4.0)", "(참고치: <500 ng/mL)",
# "(정상 범위 초과)", "(ULN 60)"); right after a measured value any parenthesis that knowledge/refrange.py reads as a
# limit ("(<500)", "(13-60)", "(40미만)", "(≥12)") or as a direction word ("(경미한 상승)", "(mildly elevated)", "(감소)")
# is one too. A bare "(<0.01)" that follows no value is the value itself.
_PAREN = re.compile(r"\(([^()]*)\)")
_REF_WORD = re.compile(r"정상|참고|normal|(?<![a-z])ref|범위|기준|상한|하한|(?<![a-z])[ul]ln(?![a-z])|(?:upper|lower) limit")
_AFTER_VALUE = re.compile(r"\d\s*(?:%|[a-zμµ/.³^0-9]+)?\s*$")
_WORD_ONLY = re.compile(r"^[^\d]{1,20}$")  # a direction-word parenthesis: short, no numbers
# a parenthesis that holds nothing but a limit or a range ("(<500)", "(13-60 U/L)", "(40미만)", "(≥ 12)", "(up to 40)"),
# not a statement that merely contains one ("(흉수/혈청 단백 비율 > 0.5)")
_N = r"[\d.,]+\s*(?:%|[a-zμµ/.³^0-9]+)?\s*"
_LIMIT_ONLY = re.compile(r"^\s*(?:(?:[<>≤≥]|<=|>=|=<|=>)\s*|(?:less than|lower than|below|under|up to|above|over|greater than"
                         r"|more than|higher than|at least)\s+)?" + _N + r"(?:[-~–]\s*" + _N + r")?"
                         r"(?:미만|이하|이상|초과|or (?:less|more|below|above|lower|higher|greater))?\s*$")
_SEG = re.compile(r"[;\n]|,\s+|,(?=[^\d\s])|\s/\s|\s-\s|·")


def _unit_norm(u: str) -> str:
    u = (u or "").lower().replace("µ", "μ").replace("mcg", "μg").rstrip(".")
    if u[:1] == "u" and u[1:2] in ("g", "m", "i", "l"):  # ug/L, umol/L, uIU/mL, uL
        u = "μ" + u[1:]
    return u


@dataclass
class _Ref(refrange.Ref):
    pos: int = 0  # where the parenthesis starts in the segment


def _ref_ranges(seg: str) -> tuple[str, list[_Ref]]:
    """Blank out reference parentheses (see _PAREN) and read them with knowledge/refrange.py; return (text, [_Ref]).
    Bounds stay in the unit printed in the parenthesis (ref.unit, "" when none is printed): see _vs_ref."""
    refs = []
    spans = []
    for m in _PAREN.finditer(seg):
        body = m.group(1)
        r = refrange.read(body)
        if not _REF_WORD.search(body):
            if not _AFTER_VALUE.search(seg[:m.start()]):
                continue  # a bare "(<0.01)" is the value itself unless it follows one
            if not ((r.bounded and _LIMIT_ONLY.match(body)) or (r.says and _WORD_ONLY.match(body.strip()))):
                continue  # "(85%)", "(우측)": not a reference
        ref = _Ref(**vars(r), pos=m.start())
        ref.unit = _unit_norm(ref.unit)
        refs.append(ref)
        spans.append((m.start(), m.end()))
    for a, b in spans:
        seg = seg[:a] + " " * (b - a) + seg[b:]
    return seg, refs


def _scale(f: Finding, v: float, unit: str) -> float:
    """A value in `unit` → the threshold's unit (f.units; units not listed are taken to be the threshold's unit)."""
    if f.id == "hba1c_high" and (unit.startswith("mmol") or (not unit.startswith("%") and v > 20)):
        return v / 10.929 + 2.15  # IFCC mmol/mol → NGSP %
    for u, mult in f.units.items():
        if unit.startswith(_unit_norm(u)):
            return v * mult
    return v


def _number(after: str, f: Finding) -> tuple[float, str, float, str] | None:
    """First value in the text right after the analyte: (value in the threshold's unit, comparator, value as printed,
    printed unit)."""
    m = re.match(r"[^\d<>≤≥,]{0,14}?([<>≤≥]=?\s*)?(\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)\s*(%|[a-zμµ/.³^]+)?", after)
    if not m:
        return None
    lead = after[: m.start(2)]
    if re.search(r"1\s*:\s*$|x\s*$|×\s*$", lead):
        return None
    raw = float(m.group(2).replace(",", ""))
    unit = _unit_norm(m.group(3) or "")
    return _scale(f, raw, unit), (m.group(1) or "").strip(), raw, unit


def _vs_ref(f: Finding, num: tuple[float, str, float, str], bound: float, ref: _Ref) -> tuple[float, float] | None:
    """(value, bound) in one unit, or None when the unit is in doubt. The value and the range are compared as printed
    when they share a unit or one of them has none ("D-dimer 750 ng/mL (정상 <500)"); both are converted to the
    threshold's unit only when two different units are printed ("0.75 mg/L (정상 <500 ng/mL)"). When only one side
    carries a unit and the two are two orders of magnitude apart, the printed comparison is trusted only if the
    reading in the threshold's unit agrees with it: "D-dimer 1.2 (정상 <500 ng/mL)" is 1.2 μg/mL, not 1.2 ng/mL, so it
    is left unread (None) rather than called normal."""
    _v, _c, raw, unit = num
    if ref.unit and unit and ref.unit != unit:
        return _scale(f, raw, unit), _scale(f, bound, ref.unit)
    if ref.unit != unit and refrange.unit_gap(raw, bound):
        if (raw > bound) != (_scale(f, raw, unit) > _scale(f, bound, ref.unit)):
            return None
    return raw, bound


def _ref_limits(f: Finding, ref: _Ref) -> tuple[float | None, float | None]:
    """(upper, lower) limit of a printed reference for this finding: a bare limit ("정상치 500") is the limit on the
    finding's abnormal side."""
    hi, lo = ref.hi, ref.lo
    if ref.limit is not None and not ref.bounded:
        if f.mode in ("hi", "pos"):
            hi = ref.limit
        elif f.mode == "lo":
            lo = ref.limit
    return hi, lo


def _lazy(pat: str) -> str:
    """Gaps between words (".{0,N}") match as little as possible, so the analyte ends at its first mention."""
    return re.sub(r"\.\{0,(\d+)\}(?!\?)", r".{0,\1}?", pat)


_COMPILED: list[tuple[Finding, re.Pattern, re.Pattern | None]] = [
    (f, re.compile(_lazy(f.pat)), re.compile(f.not_if) if f.not_if else None) for f in FINDINGS]
BY_ID: dict[str, Finding] = {f.id: f for f in FINDINGS}
_ALIASES: dict[str, tuple[re.Pattern, re.Pattern]] = {
    f.id: (re.compile(f.alias), re.compile(f.alias_ctx)) for f in FINDINGS if f.alias}


def analyte_match(f: Finding, seg: str, text: str, pat: re.Pattern | None = None) -> re.Match | None:
    """The finding's first mention in `seg`: its own pattern, else its short alias when `text` (the whole finding
    text) holds the alias' context."""
    m = (pat or re.compile(_lazy(f.pat))).search(seg)
    if m is None and f.id in _ALIASES:
        alias, ctx = _ALIASES[f.id]
        if ctx.search(text):
            m = alias.search(seg)
    return m


def _polarity(f: Finding, seg: str, s: int, e: int, refs) -> tuple[int, bool]:
    """(+1 abnormal / -1 normal / 0 unknown, value-based?) for one analyte/keyword match at seg[s:e]."""
    after = seg[e:e + 45]
    before = seg[max(0, s - 16):s]
    if _PENDING.search(after[:40]):
        return 0, False
    if f.mode == "kw":
        if f.fixed:
            return 1, False
        if _HISTORY.search(seg) or _RULEOUT.search(seg):
            return 0, False
        if _NEG_BEFORE.search(before):
            return -1, False
        tail = seg[e:e + 40]
        if re.search(_NEG_W, tail):
            return -1, False
        return 1, False
    ref = next((r for r in refs if e <= r.pos <= e + 40), None)
    up_is_good = f.mode in ("hi", "pos")
    if ref and ref.says:
        # the words of the reference judge the value printed before them ("CEA 6.5 ng/mL (경미한 상승)"): a reading of
        # that value, like a numeric one
        valued = f.mode in ("hi", "lo") and _number(after[:ref.pos - e], f) is not None
        if ref.says == "normal":
            return -1, valued
        if not f.cutoff and not f.need_value:  # "(정상 범위 초과)" says above the range, not above a cut-off
            if ref.says == "abnormal":
                return 1, valued
            return (1 if (ref.says == "high") == up_is_good else -1), valued
    abnormal = 0 if f.need_value else 1
    fold = _FOLD.search(after[:30]) if f.mode == "hi" else None  # "정상 상한의 5배" (× upper limit)
    if fold:
        return (1 if float(fold.group(1)) >= f.ref_mult else -1), True
    # earliest polarity word right after the analyte
    m = _POL.search(after[:32])
    if m:
        kind = m.lastgroup
        if kind == "pos" and _NEG_AFTER_POS.match(after[m.end():m.end() + 8]):
            kind = "neg"
        # a number before the word decides for numeric analytes ("TSH 0.01 mIU/L로 감소")
        num = _number(after, f) if f.mode in ("hi", "lo") and f.thr is not None else None
        if num is None or m.start() < 3:
            if kind == "neg":
                return -1, False
            if kind == "pos":
                return (abnormal if up_is_good else -1), False
            return (-1 if up_is_good else abnormal), False
    if f.mode == "pos":
        t = _TITER.search(after[:30])
        if t:
            return (1 if int(t.group(1)) >= f.titer else -1), True
        num = _number(after, f)
        hi = _ref_limits(f, ref)[0] if ref else None
        if num and hi is not None:
            got = _vs_ref(f, num, hi, ref)
            if got is None:
                return 0, False
            return (1 if refrange.above(got[0], got[1], ref.hi_strict) else -1), True
        if num and f.thr is not None:
            return (1 if num[0] > f.thr else -1), True
        if _NEG_BEFORE.search(before):
            return -1, False
        if _POS_BEFORE.search(before):
            return 1, False
        return 0, False
    # numeric hi / lo
    if _POS_BEFORE.search(before):
        return (abnormal if f.mode == "hi" else -1), False
    if _DOWN_BEFORE.search(before):
        return (abnormal if f.mode == "lo" else -1), False
    num = _number(after, f)
    if num is None:
        return 0, False
    v, cmp_ = num[0], num[1]
    # a printed reference range wins over the default threshold; cut-off findings ("AST > 1000") only take "within
    # the range" from it
    hi, lo = _ref_limits(f, ref) if ref else (None, None)
    bound = hi if f.mode == "hi" else lo
    if bound is not None:
        got = _vs_ref(f, num, bound, ref)
        if got is None:
            if not f.cutoff:
                return 0, False  # the unit is in doubt: not read
        else:
            rv, b = got
            if f.mode == "hi" and (rv <= b or not f.cutoff):
                return (1 if refrange.above(rv, b, ref.hi_strict, f.ref_mult) else -1), True
            if f.mode == "lo" and (rv >= b or not f.cutoff):
                return (1 if refrange.below(rv, b, ref.lo_strict) else -1), True
    if f.thr is None:
        return 0, False
    if f.mode == "hi":
        if cmp_.startswith(("<", "≤")):
            return (-1 if v <= f.thr * 1.0001 else 0), True
        if cmp_.startswith((">", "≥")) and v >= f.thr:
            return 1, True
        return (1 if v > f.thr else -1), True
    if cmp_.startswith((">", "≥")):
        return (-1 if v >= f.thr else 0), True
    if cmp_.startswith(("<", "≤")) and v <= f.thr:
        return 1, True
    return (1 if v < f.thr else -1), True


def detect(text: str, context: int = 1) -> dict[str, tuple[int, bool]]:
    """Test findings in one finding text → {finding id: (polarity, value_based)}.

    context: +1 for a finding reported as present (a bare analyte name in a very short text counts as abnormal),
    -1 for one reported as absent/normal (every detected finding is normal unless a measured value says otherwise)."""
    low = (text or "").lower()
    if not low or len(low) > 600:
        low = low[:600]
    # a text reported absent that itself says what is normal ("케톤 음성", "담관 확장 없음") keeps its own polarity;
    # one that does not ("트로포닌 상승" marked absent) is normal as a whole
    flip = context < 0 and not re.search(_NEG_W, low)
    out: dict[str, tuple[int, bool]] = {}
    for seg in _SEG.split(low):
        seg = seg.strip()
        if len(seg) < 2:
            continue
        seg, refs = _ref_ranges(seg)
        for f, pat, not_if in _COMPILED:
            m = analyte_match(f, seg, low, pat)
            if not m or (not_if and not_if.search(seg)):
                continue
            pol, by_value = _polarity(f, seg, m.start(), m.end(), refs)
            if context < 0 and not by_value and (flip or pol == 0):
                pol = -1
            elif pol == 0 and context > 0 and f.mode != "kw" and not f.need_value \
                    and len(seg) <= (m.end() - m.start()) + 6:
                pol = 1
            if pol == 0:
                continue
            prev = out.get(f.id)
            if prev is None or (pol > 0 and prev[0] < 0):  # a positive mention wins over a normal one
                out[f.id] = (pol, by_value)
    return out


def links() -> list[tuple[str, str, int, bool, str]]:
    """All links: (finding id, profile id, weight, rule_out, ref key)."""
    out = []
    for f in FINDINGS:
        for dx, w, r, ref in f.parsed_links():
            out.append((f.id, DX[dx], w, r, ref))
    return out
