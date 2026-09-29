"""Specialist consult sub-agents: content (role prompts, compact specialty knowledge) + prompt builder + parser.

Owned by clinical-strategist. The framework (when to call, how to merge the result) lives elsewhere; this module only
builds one `SubagentCall` and parses the model's answer into a `SubagentResult` (interface in `base.py`).

Same fixed gpt-oss-20b model, different role: a consultant who reads the grounded findings ledger and the current DDx
and returns strict JSON (assessment, extra differentials, dangerous diagnoses not yet excluded, up to 3 next actions).

Content rules
- Each `SpecialtySpec` is hand-written, compact (rendered text ~0.5-0.9k chars) and points to our existing machinery by id
  instead of repeating it: clinical decision rules (`knowledge/clinical_rules.py` RULES ids), diagnostic/classification
  criteria (`knowledge/diagnostic_criteria.py` CRITERIA ids), chief-complaint safety protocols (`safety/protocols.py`
  categories) and can't-miss rule-out entries (`safety/danger_gate.py` RULE_OUT_TABLE names). Tests check every id.
- Only the rules that apply to this patient (`rules_for`) and the criteria matching the current DDx (`criteria_for`) are
  rendered, so the prompt carries no out-of-population score.
- Claims not already backed by a cited module are either cited below (`SpecialtySpec.citations`, bibliographic data
  checked against PubMed E-utilities esummary on 2026-09-28; ledger rows in docs/licenses.md) or marked as reviewer
  knowledge in `SpecialtySpec.note` (verification "unverified"), as in the other clinical modules.
- 2026-09-29 verification pass: every claim previously marked reviewer knowledge was checked against a source
  (`consult_sources.py`; abstract, or full text where the note says so), reworded to what the source supports, or
  dropped. `SpecialtySpec.verification` is the weakest level among the spec's own claims (protocols.py scale:
  "primary" = read in the source text, "secondary" = abstract / summary only, "unverified" = reviewer knowledge left).
  Claims still from reviewer knowledge are listed after "Reviewer knowledge:" in the note.
- The pediatric / pregnancy branch (`patient_profile`) reuses the existing readers: age (`safety/triage._age`),
  pediatric vital-sign bands (`nlp/findings` Fleming 2011 centiles, `safety/triage` PALS hypotension floor) and
  current pregnancy (`safety/protocols` predicate "current_pregnancy" / "pregnancy_test_abdominal").
Stdlib only, CPU only, deterministic; no LLM call here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from doctor_agent.agent.parser import _THOUGHT_RE, _json_objects
from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult
from doctor_agent.agent.subagents.consult_sources import (
    C_ACR_APPENDICITIS_CHILD,
    C_AAP_UTI,
    C_ABUSE_REPORTING,
    C_ACOG_ADNEXAL_TORSION,
    C_ACOG_LISTERIA,
    C_ACOG_THROMBOCYTOPENIA,
    C_AHA_CVT,
    C_ANA_HEALTHY,
    C_CANTO_NO_CHEST_PAIN,
    C_CHAVEZ_RHABDO,
    C_CLARKE_LEUKAEMIA,
    C_CURTIS_MENINGITIS,
    C_EGRIS,
    C_FAUNDES_HYDRONEPHROSIS,
    C_FLC_SCREENING,
    C_FRIEDMAN_PEDS_CHEST_PAIN,
    C_GCA_FAST_TRACK,
    C_HANSEN_AAS_MISDX,
    C_HCM_GUIDELINE,
    C_HILL_PYELONEPHRITIS,
    C_HOM_INTUSSUSCEPTION,
    C_HUS_LANCET,
    C_IMAZIO_TAMPONADE,
    C_KAMEL_POSTPARTUM,
    C_LAWTON_MSCC,
    C_LN_PREGNANCY,
    C_LYON_ELDERLY_ABDOMEN,
    C_MARSTON_RAAA,
    C_MG_CRISIS,
    C_MONTAGUE_HYPERK_ECG,
    C_NASON_TORSION,
    C_NORMAN_FEVER_ELDERLY,
    C_OKANO_STROKE_MIMICS,
    C_PEARLE_DRAINAGE,
    C_PULMONARY_RENAL,
    C_RICE_SVC,
    C_SEPTIC_CRYSTAL,
    C_SIBAI_POSTPARTUM,
    C_SLIWA_PPCM,
    C_TERATOGENS,
    C_THOMPSON_MENINGOCOCCAL,
    C_WEINER_STEROID_PERFORATION,
)
from doctor_agent.agent.text import same_dx, similarity
from doctor_agent.knowledge.clinical_rules import Citation, contains_affirmed, rules_for
from doctor_agent.knowledge.diagnostic_criteria import C_AKI, criteria_for
from doctor_agent.llm.harmony import split_harmony
from doctor_agent.nlp.findings import _PEDS_HR, _PEDS_RR, _peds_row
from doctor_agent.safety import protocols as _protocols
from doctor_agent.safety.danger_gate import C_HYPERGLYCEMIC_CRISES, C_SCROTAL
from doctor_agent.safety.preconditions import P_ACOG_723, P_RCOG_APH
from doctor_agent.safety.protocols import (
    G_AMI,
    G_CHEST_PAIN,
    G_ECTOPIC,
    G_EARLY_PREGNANCY,
    G_GLOMERULAR,
    G_GOUT_DX,
    G_HOT_JOINT,
    G_MECFS,
    G_NECK_MASS,
    G_NEUTROPENIA,
    G_PE,
    G_SEPSIS,
    protocols_for,
)
from doctor_agent.safety.triage import C_FLEMING, C_PEDS_HYPOTENSION, _age, _peds_sbp_floor

# --------------------------------------------------------------------------------------------
# New citations (claims not covered by an already-cited module). PubMed esummary checked 2026-09-28.
# --------------------------------------------------------------------------------------------

C_HINTS = Citation(
    "Kattah JC, Talkad AV, Wang DZ, Hsieh YH, Newman-Toker DE.",
    "HINTS to diagnose stroke in the acute vestibular syndrome: three-step bedside oculomotor examination more "
    "sensitive than early MRI diffusion-weighted imaging",
    "Stroke", 2009, "40(11):3504-3510", doi="10.1161/STROKEAHA.109.551234", pmid="19762709", verified=True,
)
C_PREECLAMPSIA = Citation(
    "American College of Obstetricians and Gynecologists.",
    "Gestational Hypertension and Preeclampsia: ACOG Practice Bulletin, Number 222",
    "Obstet Gynecol", 2020, "135(6):e237-e260", doi="10.1097/AOG.0000000000003891", pmid="32443079", verified=True,
    short_author="ACOG 전자간증 지침",
)
C_FEBRILE_INFANT_AAP = Citation(
    "Pantell RH, Roberts KB, Adams WG, et al.",
    "Evaluation and Management of Well-Appearing Febrile Infants 8 to 60 Days Old",
    "Pediatrics", 2021, "148(2):e2021052228", doi="10.1542/peds.2021-052228", pmid="34281996", verified=True,
    short_author="AAP 발열 영아 지침",
)
C_TLS = Citation(
    "Cairo MS, Bishop M.",
    "Tumour lysis syndrome: new therapeutic strategies and classification",
    "Br J Haematol", 2004, "127(1):3-11", doi="10.1111/j.1365-2141.2004.05094.x", pmid="15384972", verified=True,
)
C_MYELOMA_IMWG = Citation(
    "Rajkumar SV, Dimopoulos MA, Palumbo A, et al.",
    "International Myeloma Working Group updated criteria for the diagnosis of multiple myeloma",
    "Lancet Oncol", 2014, "15(12):e538-e548", doi="10.1016/S1470-2045(14)70442-5", pmid="25439696", verified=True,
    short_author="IMWG",
)
C_HIT_ASH = Citation(
    "Cuker A, Arepally GM, Chong BH, et al.",
    "American Society of Hematology 2018 guidelines for management of venous thromboembolism: heparin-induced "
    "thrombocytopenia",
    "Blood Adv", 2018, "2(22):3360-3392", doi="10.1182/bloodadvances.2018024489", pmid="30482768", verified=True,
    short_author="ASH HIT 지침",
)
C_TTP_ISTH = Citation(
    "Zheng XL, Vesely SK, Cataland SR, et al.",
    "ISTH guidelines for the diagnosis of thrombotic thrombocytopenic purpura",
    "J Thromb Haemost", 2020, "18(10):2486-2495", doi="10.1111/jth.15006", pmid="32914582", verified=True,
    short_author="ISTH TTP 지침",
)
NEW_CITATIONS: tuple[Citation, ...] = (C_HINTS, C_PREECLAMPSIA, C_FEBRILE_INFANT_AAP, C_TLS, C_MYELOMA_IMWG,
                                       C_HIT_ASH, C_TTP_ISTH)

# --------------------------------------------------------------------------------------------
# Specialty content
# --------------------------------------------------------------------------------------------

BRANCHES = ("peds", "pregnant", "female_repro", "adult")


@dataclass(frozen=True)
class SpecialtySpec:
    id: str
    name_ko: str
    role_ko: str  # one or two sentences: the consultant's viewpoint
    must_not_miss: tuple[str, ...]
    key_asks: tuple[str, ...]
    key_exams: tuple[str, ...]
    key_tests: tuple[str, ...]
    pitfalls: tuple[str, ...]
    # references into existing modules (ids / names, checked by tests; logic is not duplicated here)
    rule_ids: tuple[str, ...] = ()  # knowledge/clinical_rules.py RULES[*].id
    criteria_ids: tuple[str, ...] = ()  # knowledge/diagnostic_criteria.py CRITERIA[*].id
    protocol_categories: tuple[str, ...] = ()  # safety/protocols.py Protocol.category
    rule_out_names: tuple[str, ...] = ()  # safety/danger_gate.py RULE_OUT_TABLE[*].name
    # extra lines rendered only for a patient branch ("peds" | "pregnant" | "female_repro" | "adult")
    branch_notes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    citations: tuple[Citation, ...] = ()  # sources for the claims above not covered by the referenced modules
    verification: str = "unverified"  # primary | secondary | unverified (as in the other clinical modules)
    note: str = ""

    def render(self, branch: str = "adult") -> str:
        lines = [f"[분과 지식: {self.name_ko}]",
                 "반드시 배제: " + ", ".join(self.must_not_miss),
                 "판별 문진: " + "; ".join(self.key_asks),
                 "판별 진찰: " + "; ".join(self.key_exams),
                 "판별 검사: " + "; ".join(self.key_tests),
                 "흔한 함정: " + "; ".join(self.pitfalls)]
        lines += [f"- {x}" for x in self.branch_notes.get(branch, ())]
        return "\n".join(lines)


_PREG_IMAGING = "영상: 초음파·MRI(비조영) 우선. 꼭 필요한 CT·X선은 미루지 않음. 가돌리늄 조영제는 피함"
_PREG_DRUGS = "처방 전 임신 금기 약(이소트레티노인·와파린, 1분기 후 ACE 억제제·ARB, 16주부터 테트라사이클린 등) 확인"

SPECIALTIES: dict[str, SpecialtySpec] = {s.id: s for s in (
    SpecialtySpec(
        id="cardio", name_ko="심장·혈관",
        role_ko="흉통·호흡곤란·실신·두근거림·부종을 심장과 대혈관 관점에서 봅니다. 허혈, 대동맥, 폐색전, 부정맥, 심부전을 가립니다.",
        must_not_miss=("급성 관상동맥 증후군", "대동맥 박리", "폐색전증", "긴장성 기흉", "심장눌림증(심낭 압전)",
                       "악성 부정맥(심실빈맥·완전 방실차단·QT 연장)", "급성 심부전", "감염성 심내막염"),
        key_asks=("통증 시작 속도·성질(갑자기 찢어짐 vs 운동 시 조이는 느낌)", "운동 중·누운 채 실신, 전조 없는 실신",
                  "젊은 나이 돌연사 가족력", "최근 수술·장기 부동·편측 다리 부기", "QT 연장 약·이뇨제 복용"),
        key_exams=("양팔 혈압 차이·맥박 결손", "경정맥 확장·폐 수포음·하지 부종", "새 심잡음(특히 이완기)",
                   "양측 호흡음 대칭"),
        key_tests=("12유도 심전도(가장 먼저)", "고감도 트로포닌(반복 측정)", "D-dimer(사전확률 낮을 때만)",
                   "CT 폐동맥·대동맥 혈관조영", "심장 초음파", "BNP"),
        pitfalls=("정상 심전도·트로포닌 1회로 ACS 배제 금지", "여성·당뇨·고령은 흉통 없는 심근경색이 많음",
                  "폐색전 사전확률이 높으면 D-dimer 없이 바로 영상", "대동맥 박리를 ACS로 오인해 항혈전제 투여"),
        rule_ids=("heart", "add_rs", "wells_pe", "perc", "spesi", "sfsr", "csrs"),
        criteria_ids=("duke_iscvid_2023", "jones_2015", "takayasu_2022"),
        protocol_categories=("chest_pain", "dyspnea", "syncope", "palpitations", "edema"),
        rule_out_names=("급성 관상동맥 증후군", "대동맥 박리", "폐색전증", "긴장성 기흉", "급성 심부전", "복부 대동맥류 파열"),
        branch_notes={
            "pregnant": ("임신·산후 흉통/호흡곤란: 폐색전증, 주산기 심근병증(임신 마지막 달~산후 6개월), 전자간증(폐부종) 고려",),
            "peds": ("소아 흉통은 심장 원인이 드묾(약 1%). 병력·가족력·심장 진찰·심전도에 이상이 있을 때 추가 심장 검사",),
        },
        citations=(G_CHEST_PAIN, G_PE, C_PREECLAMPSIA, C_IMAZIO_TAMPONADE, C_CANTO_NO_CHEST_PAIN, C_HANSEN_AAS_MISDX,
                   C_SLIWA_PPCM, C_FRIEDMAN_PEDS_CHEST_PAIN),
        verification="secondary",
        note="Serial troponin / ECG within 10 min: G_CHEST_PAIN (as in protocols.py). D-dimer only at non-high "
             "pre-test probability: G_PE (as protocols.py). 2026-09-29: tamponade is life-threatening, a clinical "
             "diagnosis confirmed by echo (Imazio 2018 abstract). MI without chest pain in 33%, more often older, "
             "women, diabetes (Canto 2000 abstract; was 'atypical symptoms'). Acute aortic syndrome misdiagnosed in "
             "39%, ACS the commonest, antithrombotics -> bleeding (Hansen 2007 abstract). PPCM last month of "
             "pregnancy to 6 months postpartum (Sliwa 2010 abstract); pulmonary oedema is a preeclampsia severe "
             "feature (ACOG PB 222 full text, Box 3). Pediatric chest pain cardiac in 1.2%; testing guided by "
             "history, family history, exam and ECG (Friedman 2011 abstract; the former 'exertional syncope' "
             "trigger list was dropped as not in the abstract).",
    ),
    SpecialtySpec(
        id="resp_id", name_ko="호흡기·감염",
        role_ko="기침·호흡곤란·발열·객혈을 폐와 감염 관점에서 봅니다. 감염 부위, 중증도, 면역 상태를 정합니다.",
        must_not_miss=("패혈증·패혈성 쇼크", "세균성 뇌수막염", "호중구감소성 발열", "폐색전증", "긴장성 기흉",
                       "후두개염·심경부 감염", "활동성 폐결핵", "감염성 심내막염"),
        key_asks=("발열 기간·오한", "감염 부위 증상(가래, 배뇨, 복통, 두통·목 경직, 피부)",
                  "면역 저하(항암·스테로이드·HIV·비장 절제)", "최근 입원·항생제", "여행·동물·결핵 접촉", "체중 감소·야간 발한"),
        key_exams=("활력징후·의식(qSOFA)", "산소포화도", "폐 청진(국소 수포음·호흡음 감소)", "경부 강직", "점상출혈·자반",
                   "심잡음"),
        key_tests=("혈액배양 2세트(항생제 전)", "젖산", "CBC(호중구 수)", "CRP·프로칼시토닌", "흉부 X선", "소변검사·배양",
                   "객담 항산균 검사(결핵 의심 시)"),
        pitfalls=("고령자는 감염이 있어도 발열이 없거나 약할 수 있음", "바이러스 감염으로 조기 종결", "항생제 전 배양 누락",
                  "발열+점상출혈·자반은 수막구균 감염 의심(전형 소견은 늦게 나타남)"),
        rule_ids=("qsofa", "curb65", "centor", "mcisaac", "wells_pe", "perc", "spesi"),
        criteria_ids=("sepsis3_2016", "light_1972", "duke_iscvid_2023"),
        protocol_categories=("fever", "dyspnea", "hemoptysis_cough"),
        rule_out_names=("패혈증", "뇌수막염", "호중구감소성 발열", "폐색전증", "긴장성 기흉", "아나필락시스"),
        branch_notes={
            "peds": ("생후 60일 이하 발열은 겉보기와 무관하게 중증 세균 감염 평가(소변·혈액, 필요 시 뇌척수액)",
                     "침 흘림·삼킴 곤란·앉아서 숨쉬기 = 후두개염: 목 안 억지 진찰 금지"),
            "pregnant": ("임신 중 발열: 신우신염·리스테리아(임신 중 흔함)·융모양막염 고려, 필요한 흉부 X선은 미루지 않음",),
        },
        citations=(G_SEPSIS, C_FEBRILE_INFANT_AAP, C_NORMAN_FEVER_ELDERLY, C_THOMPSON_MENINGOCOCCAL, C_HILL_PYELONEPHRITIS,
                   C_ACOG_LISTERIA, P_ACOG_723),
        verification="unverified",
        note="Blood cultures before antibiotics and lactate: G_SEPSIS (protocols.py, primary). Febrile infant "
             "<= 60 days: AAP 2021 (population 8-60 days, well-appearing; our summary) + PECARN febrile infant rule. "
             "2026-09-29: fever absent or blunted in 20-30% of infections in the elderly (Norman 2000 abstract; "
             "'immunocompromised' dropped, not in the source). Haemorrhagic rash is a classic but late feature of "
             "meningococcal disease (Thompson 2006 abstract). Antepartum pyelonephritis 1.4%, septicaemia 17% (Hill "
             "2005 abstract); listeriosis 13x more common in pregnancy, flu-like fever (ACOG CO 614 abstract). "
             "Needed radiography is not withheld in pregnancy (ACOG CO 723 abstract; '(차폐)' shielding removed, "
             "not in the source). Premature closure on a viral diagnosis: cognitive-error literature (advocate.py). "
             "Reviewer knowledge: epiglottitis triad and 'no forced throat exam' (no guideline abstract found), "
             "chorioamnionitis in the pregnancy fever list.",
    ),
    SpecialtySpec(
        id="gi_liver", name_ko="소화기·간",
        role_ko="복통·구토·설사·위장관 출혈·황달을 소화기와 간·담도·췌장 관점에서 봅니다.",
        must_not_miss=("장 천공", "장간막 허혈", "장폐색·교액", "충수염", "급성 담관염", "중증 급성 췌장염",
                       "위장관 출혈", "급성 간부전", "복부 대동맥류 파열", "자궁외 임신", "당뇨병성 케톤산증",
                       "심근경색(흉통 없이 명치 통증)"),
        key_asks=("통증 위치 이동(명치→우하복부)", "시작 속도", "구토 내용(담즙·혈액)", "배변·방귀 중단", "흑색변·혈변",
                  "음주·NSAID·아세트아미노펜·한약", "마지막 월경", "복부 수술력"),
        key_exams=("압통 위치·반발 압통·근성 방어", "장음", "머피 징후", "직장 수지 검사(출혈 시)", "황달·복수·자세고정 떨림"),
        key_tests=("CBC", "간기능·빌리루빈 분획", "리파아제", "젖산", "β-hCG(가임기 여성)", "소변검사", "복부 초음파(담도)",
                   "복부 CT(조영)", "내시경(출혈)"),
        pitfalls=("진찰 소견보다 심한 통증은 장간막 허혈 의심", "고령·스테로이드 복용자는 진찰 소견이 가볍게 보일 수 있음",
                  "가임기 여성 임신 검사 누락", "명치 통증을 위염으로 결론 전 심전도(흉통 없는 심근경색)"),
        rule_ids=("alvarado", "bisap", "gbs", "pas"),
        criteria_ids=("dka_hhs_2024",),
        protocol_categories=("abdominal_pain", "jaundice", "bilious_vomiting"),
        rule_out_names=("장 천공", "장간막 허혈", "복부 대동맥류 파열", "자궁외 임신", "당뇨병성 케톤산증"),
        branch_notes={
            "peds": ("영아 담즙성 구토 = 장회전 이상·중장 염전 즉시 배제",
                     "간헐적 보챔·구토·혈변은 장중첩증 의심: 증상만으로 배제 불가, 초음파"),
            "pregnant": ("임신 중 우상복부·명치 통증은 HELLP·전자간증부터(혈압·혈소판·간효소)",),
            "female_repro": ("가임기 여성 하복부 통증: 영상·약 전에 임신 검사",),
        },
        citations=(G_AMI, G_ECTOPIC, C_PREECLAMPSIA, C_LYON_ELDERLY_ABDOMEN, C_WEINER_STEROID_PERFORATION,
                   C_CANTO_NO_CHEST_PAIN, C_HOM_INTUSSUSCEPTION),
        verification="secondary",
        note="Pain out of proportion in mesenteric ischaemia: G_AMI (WSES 2022). Pregnancy test in abdominal pain: "
             "G_ECTOPIC (as protocols.py). Bilious vomiting: protocols 'bilious_vomiting'. 2026-09-29: severe "
             "persistent RUQ/epigastric pain is a preeclampsia severe feature; HELLP = LDH, AST/ALT > 2x, platelets "
             "< 100,000 (ACOG PB 222 full text read, Boxes 2-3 and HELLP section). Exam misleadingly benign in older "
             "patients (Lyon 2006 abstract); corticosteroids mask inflammatory signs of perforation (Weiner 1993 "
             "abstract). MI without chest pain (Canto 2000 abstract; 'inferior MI' dropped, not in the source). "
             "Intussusception: history/exam and the classic symptoms have low accuracy, point-of-care US the best "
             "(Hom 2022 meta-analysis abstract; the 'triad' wording was replaced).",
    ),
    SpecialtySpec(
        id="neuro", name_ko="신경",
        role_ko="두통·국소 신경 결손·의식 변화·경련·어지럼·근력 저하를 병변 위치(중추 vs 말초, 뇌·척수·신경·근육) 관점에서 봅니다.",
        must_not_miss=("급성 허혈성 뇌졸중", "뇌출혈·지주막하 출혈", "뇌수막염·뇌염", "저혈당", "척수 압박·마미 증후군",
                       "길랭-바레 증후군(호흡 부전)", "중증 근무력증 위기", "뇌정맥동 혈전증", "거대세포동맥염(시력 소실)"),
        key_asks=("마지막 정상 시각", "벼락두통(1분 내 최고조)", "외상·항응고제", "발열·목 경직", "시야 이상·복시",
                  "상행성 근력 저하", "대소변 장애·안장 감각 저하", "경련 목격 양상"),
        key_exams=("의식(GCS)·동공", "편측 근력·감각·안면·구음", "소뇌 검사·보행", "경부 강직", "반사(상·하위 운동신경원)",
                   "안저(유두부종)", "급성 지속성 현훈이면 HINTS"),
        key_tests=("혈당(즉시)", "비조영 뇌 CT", "뇌 MRI(DWI)", "CT 혈관조영", "요추천자(CT 음성 SAH·수막염)", "전해질",
                   "ESR·CRP(50세 이상 새 두통)", "척추 MRI"),
        pitfalls=("어지럼을 말초성으로 단정(후순환 뇌졸중)", "초기 MRI DWI 음성으로 뇌졸중 배제 금지",
                  "저혈당이 뇌졸중을 흉내냄", "발병 6시간 후 CT 음성만으로 SAH 배제 금지"),
        rule_ids=("ottawa_sah", "cchr", "abcd2", "nexus", "ccsr", "pecarn_head_lt2", "pecarn_head_ge2"),
        criteria_ids=("mcdonald_2017", "ichd3_migraine_tth", "gca_2022"),
        protocol_categories=("headache", "neuro", "cognitive", "psychiatric", "hearing_loss", "chronic_weakness",
                             "back_pain"),
        rule_out_names=("지주막하 출혈", "뇌출혈", "급성 허혈성 뇌졸중", "뇌수막염", "저혈당", "마미 증후군"),
        branch_notes={
            "pregnant": ("임신 20주 이후·산후 새 두통·시야 이상·경련 = 전자간증·자간증(혈압), 임신·산후는 뇌정맥동 혈전증 위험",),
            "peds": ("소아 두부 외상은 PECARN(나이별), 영아 대천문 팽륭은 수막염 가능성을 높임",),
        },
        citations=(C_HINTS, C_PREECLAMPSIA, C_OKANO_STROKE_MIMICS, C_EGRIS, C_MG_CRISIS, C_AHA_CVT, C_KAMEL_POSTPARTUM,
                   C_GCA_FAST_TRACK, C_CURTIS_MENINGITIS),
        verification="secondary",
        note="HINTS more sensitive than early DWI MRI in acute vestibular syndrome: Kattah 2009 (title/abstract). SAH "
             "CT timing: danger_gate C_CT_6H_SAH. Protocol 'psychiatric' is shared with psych (organic cause first); "
             "bipolar_dsm5tr moved to psych on 2026-09-29. 2026-09-29: "
             "hypoglycaemia is among the commonest stroke mimics (Okano 2018 abstract; glucose before alteplase: "
             "protocols 'neuro'). GBS respiratory insufficiency, 22% ventilated in week 1 (Walgaard 2010 abstract). "
             "Myasthenic crisis = respiratory failure, life-threatening (Takahashi 2025 abstract). Postpartum "
             "eclampsia/new headache and visual disturbance as severe features (ACOG PB 222 full text). CVT: women of "
             "reproductive age and prothrombotic states vulnerable (AHA 2024 abstract) + postpartum thrombosis/stroke "
             "risk (Kamel 2014 abstract). GCA permanent visual loss, fast-track lowers it (Monti 2019 abstract). "
             "Bulging fontanel LR 3.5-8 for meningitis (Curtis 2010 abstract; 'sunken' dropped, not in the source).",
    ),
    SpecialtySpec(
        id="rheum_immune", name_ko="류마티스·면역",
        role_ko="관절·피부·다장기 염증과 알레르기 반응을 자가면역·혈관염·결정·과민반응 관점에서 봅니다.",
        must_not_miss=("화농성 관절염", "아나필락시스·혈관부종", "스티븐스-존슨/독성 표피 괴사 용해·DRESS",
                       "거대세포동맥염(시력 소실)", "폐-신장 증후군(ANCA 혈관염·항GBM)", "중증 루푸스(신염·중추신경)",
                       "괴사성 근막염"),
        key_asks=("관절 수·분포·대칭, 급성 단일 관절+발열", "아침 강직 시간", "발진·광과민·구강 궤양·레이노",
                  "눈 건조·충혈", "새 약물(2개월 이내)", "혈뇨·거품뇨·객혈", "두피 압통·턱 파행·시야 이상"),
        key_exams=("관절 부기·열감·운동 범위", "피부(촉지 자반, 점막 침범)", "측두동맥 압통", "양팔 혈압·맥박",
                   "입술·혀 부종, 천명음(알레르기)"),
        key_tests=("관절 천자(세포 수·그람 염색·배양·결정)", "ESR·CRP", "ANA→항dsDNA·보체", "RF·항CCP", "ANCA", "요산",
                   "소변검사(단백·적혈구 원주)", "크레아티닌", "CK"),
        pitfalls=("발작 중 요산 정상이어도 통풍 배제 불가", "통풍·가성통풍이 있어도 화농성 동반 가능: 천자",
                  "ANA 양성만으로 루푸스 진단 금지(분류 기준)", "열감 단일 관절에 스테로이드 전 감염 배제"),
        rule_ids=(),
        criteria_ids=("sle_2019", "ra_2010", "gout_2015", "gca_2022", "takayasu_2022", "jones_2015"),
        protocol_categories=("joint", "rash", "allergy", "urticaria_chronic", "edema", "bleeding", "pruritus",
                             "chronic_weakness"),
        rule_out_names=("아나필락시스", "상기도 부종(혈관부종)"),
        branch_notes={
            "peds": ("소아 5일 이상 발열+결막 충혈·입술·손발·발진·경부 림프절 = 가와사키병(관상동맥)",
                     "소아 고관절 통증+발열 = 화농성 고관절염(Kocher 기준)"),
            "pregnant": ("임신 중 루푸스·항인지질 증후군 악화와 전자간증 감별(혈압·단백뇨·보체)",),
        },
        citations=(G_HOT_JOINT, G_GOUT_DX, C_SEPTIC_CRYSTAL, C_ANA_HEALTHY, C_PULMONARY_RENAL, C_GCA_FAST_TRACK,
                   C_LN_PREGNANCY, C_PREECLAMPSIA),
        verification="secondary",
        note="Aspirate the hot joint before antibiotics: G_HOT_JOINT (protocols.py). Serum urate may be normal during a "
             "flare: G_GOUT_DX (EULAR 2018 diagnosis recommendations). Kawasaki and Kocher notes point to criteria "
             "kawasaki_aha2017 / rule kocher (owned by peds_obgyn). 2026-09-29: concomitant septic and crystal "
             "arthritis in 4% of crystal arthritis, mortality 15.6% (Duangkum 2025 abstract). ANA positive in 13.3% "
             "of healthy people at 1:80 (Tan 1997 abstract). Pulmonary-renal syndrome = alveolar haemorrhage + GN, "
             "mostly ANCA vasculitis / anti-GBM, rapid respiratory and renal failure (Boyle 2022 abstract). GCA "
             "visual loss (Monti 2019 abstract). Lupus nephritis flare vs pregnancy conditions incl. preeclampsia "
             "(Gholizadeh 2024 abstract); SLE and APS are preeclampsia risk factors (ACOG PB 222 full text, Box 1).",
    ),
    SpecialtySpec(
        id="peds_obgyn", name_ko="소아·산부인과",
        role_ko="나이와 임신 상태에 따라 판단 기준을 바꿉니다. 소아는 나이별 활력징후와 보호자 관찰을, 가임기·임신 환자는 임신 관련 응급과 약·영상 안전을 봅니다.",
        must_not_miss=("자궁외 임신", "전자간증·자간증·HELLP", "태반 조기 박리·전치태반", "난소 염전", "고환 염전",
                       "영아 중증 세균 감염", "장중첩증", "장회전 이상·중장 염전", "가와사키병", "소아 당뇨병성 케톤산증",
                       "아동 학대"),
        key_asks=("나이(영아는 개월·일 수)", "임신 가능성·마지막 월경·임신 주수", "질출혈·복통·태동",
                  "수유·섭취량·소변 횟수(기저귀)", "예방접종력·출생력", "보호자가 본 처짐·보챔"),
        key_exams=("나이별 기준의 활력징후", "혈압(임신 20주 이후)", "탈수 징후(모세혈관 재충전)", "복부·고환 진찰",
                   "골반 내진(출혈 원인 확인 전 전치태반 의심 시 금지)"),
        key_tests=("β-hCG", "질식·골반 초음파", "소변 단백·혈소판·간효소(전자간증)", "소변검사·배양(영아 발열)",
                   "혈당·케톤", "복부 초음파(장중첩증)"),
        pitfalls=("소아에 성인 활력징후 기준 적용", "가임기 여성 임신 검사 없이 CT·약 처방",
                  "임신 20주 이후 두통·복통을 혈압 확인 없이 넘김", "설명과 맞지 않는 손상을 사고로 기록",
                  "도플러 혈류 정상으로 난소 염전 배제 금지"),
        rule_ids=("pecarn_febrile_infant", "pecarn_head_lt2", "pecarn_head_ge2", "pas", "kocher", "mcisaac"),
        criteria_ids=("kawasaki_aha2017", "jones_2015"),
        protocol_categories=("menstrual", "abdominal_pain", "bilious_vomiting", "jaundice"),
        rule_out_names=("자궁외 임신", "고환 염전", "패혈증", "당뇨병성 케톤산증"),
        branch_notes={
            "peds": ("영아 발열·처짐은 겉보기가 괜찮아도 중증 감염 평가",
                     "영상: 장중첩증은 초음파, 충수염은 임상 위험도로 영상 여부 결정"),
            "pregnant": ("20주 전: 자궁외 임신·유산(β-hCG + 질식 초음파)",
                         "20주 이후·산후: 혈압 ≥140/90(4시간 간격 2회)+단백뇨 또는 중증 소견이면 전자간증. 중증: ≥160/110, "
                         "혈소판 <10만, 간효소 2배, 우상복부·명치 통증, 새 두통·시야 이상, 폐부종, 크레아티닌 >1.1",
                         "후기 질출혈: 초음파로 전치태반 배제 전 내진 금지, 통증+출혈은 태반 조기 박리",
                         _PREG_IMAGING, _PREG_DRUGS),
            "female_repro": ("임신 여부 미확인: 영상·약 전에 β-hCG, 양성이면 질식 초음파", _PREG_IMAGING),
            "adult": ("성인·비임신 환자: 이 분과의 특이 위험 없음. 일반 원칙만 적용",),
        },
        citations=(G_ECTOPIC, G_EARLY_PREGNANCY, C_PREECLAMPSIA, P_RCOG_APH, P_ACOG_723, C_FEBRILE_INFANT_AAP,
                   C_FLEMING, C_PEDS_HYPOTENSION, C_SIBAI_POSTPARTUM, C_TERATOGENS, C_ACOG_ADNEXAL_TORSION,
                   C_ABUSE_REPORTING, C_HOM_INTUSSUSCEPTION, C_ACR_APPENDICITIS_CHILD),
        verification="secondary",
        note="Ectopic work-up: G_ECTOPIC / G_EARLY_PREGNANCY (protocols.py, danger_gate). Placenta praevia before "
             "digital exam: P_RCOG_APH (preconditions.py). Imaging in pregnancy (ultrasound/MRI first, do not "
             "withhold needed CT, avoid gadolinium): P_ACOG_723. Pediatric vitals: Fleming 2011 centiles + PALS "
             "hypotension (triage.py). 2026-09-29: preeclampsia = BP >= 140/90 twice >= 4 h apart after 20 weeks + "
             "proteinuria or a new severe feature; severe = >= 160/110, platelets < 100,000, transaminases 2x, "
             "persistent RUQ/epigastric pain, new headache, visual disturbance, pulmonary oedema, creatinine > 1.1 "
             "(ACOG PB 222 full text read, Boxes 2-3; the old '>= 140/90 alone = preeclampsia' was corrected). "
             "Postpartum onset: ACOG PB 222 (HELLP first expressed postpartum in 30%) + Sibai 2012 abstract; the "
             "'6 weeks' limit is not in PB 222 and was dropped. Teratogens: retinoids and coumarins teratogenic, "
             "ACE-I/ARB contraindicated after the first trimester, tetracyclines from week 16 (Dathe & Schaefer 2019, "
             "full text PMC6935972, Tables 1-3). Adnexal torsion: timely laparoscopy, no imaging criterion is "
             "sufficient and Doppler flow alone must not guide (ACOG CO 783 abstract). Injury inconsistent with the "
             "history as an abuse indicator; suspected abuse under-reported (Flaherty 2008 abstract). Intussusception "
             "US (Hom 2022 abstract); 'ultrasound first for appendicitis / minimise radiation' reworded to risk-based "
             "imaging (ACR AC Suspected Appendicitis-Child 2019 abstract, PMID 31054752; variant tables not read).",
    ),
    SpecialtySpec(
        id="heme_onc", name_ko="혈액·종양",
        role_ko="빈혈·혈구 감소·출혈·림프절 비대·체중 감소·피로를 혈액 질환과 암(합병증 포함) 관점에서 봅니다. 혈구 수치와 말초혈액 도말부터 봅니다.",
        must_not_miss=("호중구감소성 발열", "급성 백혈병", "TTP/HUS", "DIC", "헤파린 유발 혈소판 감소증", "종양 용해 증후군",
                       "악성 척수 압박", "상대정맥 증후군", "악성 고칼슘혈증", "급성 용혈", "재생불량성 빈혈"),
        key_asks=("체중 감소·야간 발한·발열", "잇몸·코 출혈, 멍", "항암 치료와 마지막 날짜", "헤파린 5~10일 뒤 혈소판 감소",
                  "요통·뼈 통증+다리 힘 빠짐", "새 약·감염 후 진한 소변", "얼굴·목·팔 부기, 숨참"),
        key_exams=("림프절 위치·크기·단단함·고정", "간비종대", "점상출혈·자반", "창백·황달", "척추 압통·하지 근력·감각",
                   "목·가슴 정맥 확장"),
        key_tests=("CBC+백혈구 감별", "말초혈액 도말(모세포·분열적혈구)", "망상적혈구", "LDH·간접 빌리루빈·합토글로빈·직접 Coombs",
                   "PT/aPTT·피브리노겐·D-dimer", "ADAMTS13", "칼슘·요산·칼륨·인·크레아티닌", "SPEP·혈청 유리 경쇄",
                   "유세포 분석·골수/림프절 생검"),
        pitfalls=("항암 후 발열은 호중구 수 확인 전 안심 금지", "혈소판 감소+용혈은 도말로 TTP부터",
                  "단단하고 고정된 림프절·B 증상은 경과 관찰 말고 생검", "빈혈+신기능 저하+고칼슘+뼈 통증 = 골수종",
                  "암 환자의 새 요통은 신경 결손 전에 척수 압박 의심(전척추 MRI)"),
        rule_ids=(),
        criteria_ids=(),
        protocol_categories=("fatigue", "neck_mass", "bleeding", "jaundice", "pruritus", "fever", "back_pain"),
        rule_out_names=("호중구감소성 발열", "마미 증후군"),
        branch_notes={
            "peds": ("소아 창백·멍·발열·팔다리 통증·간비종대 = 급성 백혈병 의심(CBC·도말)",
                     "설사(장출혈성 대장균) 후 혈소판 감소+용혈+급성 신손상 = HUS"),
            "pregnant": ("임신 중 혈소판 감소: 양성 임신성 vs HELLP·전자간증, 20주 전이면 TTP·HUS(혈압·간효소·도말)",),
        },
        citations=(G_NEUTROPENIA, G_NECK_MASS, G_MECFS, C_TLS, C_MYELOMA_IMWG, C_HIT_ASH, C_TTP_ISTH, C_FLC_SCREENING,
                   C_LAWTON_MSCC, C_RICE_SVC, C_HCM_GUIDELINE, C_CLARKE_LEUKAEMIA, C_HUS_LANCET,
                   C_ACOG_THROMBOCYTOPENIA, C_PREECLAMPSIA),
        verification="secondary",
        note="Neutropenic fever: G_NEUTROPENIA (protocols.py, danger_gate). Neck mass work-up / biopsy of a suspicious "
             "persistent node: G_NECK_MASS (AAO-HNS 2017). Fatigue CBC/TSH: G_MECFS (as protocols.py 'fatigue'). TLS "
             "electrolytes (urate, K, phosphate, Ca, creatinine): C_TLS (abstract). Myeloma CRAB features + SPEP: "
             "C_MYELOMA_IMWG (abstract). HIT timing and 4Ts pretest probability: C_HIT_ASH (abstract). ADAMTS13 "
             "testing for TTP: C_TTP_ISTH (abstract). Owns protocol categories 'fatigue' (can't-miss led by "
             "haematological malignancy and anaemia, CBC first) and 'neck_mass' (lymphoma / metastatic node). "
             "2026-09-29: serum PEL + FLC screening panel; omitting FLC missed myeloma cases (Katzmann 2009 "
             "abstract). MSCC: suspect on new/worse back pain before deficits, whole-spine MRI (Lawton 2019 "
             "abstract). SVC syndrome: face/neck swelling 82%, arm 68%, dyspnoea 66%, chest veins 38%, malignancy 60% "
             "(Rice 2006 abstract; 'orthopnoea' dropped, not in the abstract). Hypercalcaemia of malignancy is the "
             "most common metabolic complication of cancer (Endocrine Society 2023 abstract). Childhood leukaemia: "
             "hepatosplenomegaly, pallor, fever, bruising > 50%, limb pain 43% (Clarke 2016 abstract). HUS triad, "
             "shiga toxin form (Fakhouri 2017 abstract). Pregnancy thrombocytopenia: gestational benign vs serious "
             "causes (ACOG PB 207 abstract); HELLP platelets < 100,000 and TTP/HUS before 20 weeks (ACOG PB 222 full "
             "text).",
    ),
    SpecialtySpec(
        id="renal_uro", name_ko="신장·비뇨",
        role_ko="소변량·크레아티닌 변화, 혈뇨·단백뇨, 옆구리·음낭 통증, 배뇨 장애를 콩팥과 요로 관점에서 봅니다. 급성 신손상은 신전성·신성·신후성으로 나눕니다.",
        must_not_miss=("고칼륨혈증", "요로 폐쇄 동반 감염·요로성 패혈증", "급속 진행성 사구체신염·폐-신장 증후군",
                       "횡문근융해증", "급성 요폐·양측 요로 폐쇄", "고환 염전", "복부 대동맥류(신산통 오인)"),
        key_asks=("소변량·마지막 소변", "구토·설사·섭취 감소", "NSAID·ACE 억제제/ARB·이뇨제·조영제", "혈뇨·거품뇨·부종",
                  "옆구리 통증+발열·오한", "배뇨 곤란·잔뇨감", "근육통·장시간 부동·과격한 운동", "갑작스러운 음낭 통증"),
        key_exams=("혈압·기립성 변화·체액 상태", "늑척추각 압통", "방광 팽만", "고환 위치·거고근 반사", "전립선 압통",
                   "부종·자반"),
        key_tests=("크레아티닌(기저치 비교)·BUN", "칼륨·심전도", "소변 현미경(적혈구 원주·백혈구)", "소변 단백/크레아티닌 비",
                   "CK", "신장·방광 초음파(수신증)·잔뇨량", "소변·혈액 배양", "ANCA·항GBM·보체", "음낭 도플러"),
        pitfalls=("급성 신손상은 크레아티닌과 소변량 둘 다로 판단", "심전도는 고칼륨에 둔감: 정상이어도 칼륨 수치로 판단",
                  "발열+폐쇄 결석은 응급 배액", "소변 잠혈 양성인데 적혈구 거의 없음 = 근색소뇨(또는 혈색소뇨): CK",
                  "고환 염전 임상 의심이 크면 영상 기다리지 말고 수술 탐색", "고령 복통·요통을 신산통으로 단정 전 대동맥류 배제"),
        rule_ids=("qsofa",),
        criteria_ids=("kdigo_aki_2012", "sepsis3_2016"),
        protocol_categories=("edema", "abdominal_pain", "back_pain"),
        rule_out_names=("패혈증", "고환 염전", "복부 대동맥류 파열"),
        branch_notes={
            "peds": ("원인 모를 발열 영아(2~24개월)는 요로 감염 평가: 소변검사+배양", "청소년 급성 음낭 통증은 고환 염전부터"),
            "pregnant": ("임신 중 신우신염은 패혈증 동반이 흔함. 생리적 수신증(우측에 많음)을 폐쇄로 오인 주의",),
        },
        citations=(C_AKI, G_GLOMERULAR, C_SCROTAL, G_SEPSIS, C_PULMONARY_RENAL, C_MONTAGUE_HYPERK_ECG, C_PEARLE_DRAINAGE,
                   C_CHAVEZ_RHABDO, C_NASON_TORSION, C_MARSTON_RAAA, C_AAP_UTI, C_HILL_PYELONEPHRITIS,
                   C_FAUNDES_HYDRONEPHROSIS),
        verification="secondary",
        note="AKI definition/staging by creatinine rise and urine output: C_AKI (criteria kdigo_aki_2012, owned here "
             "since 2026-09-28; moved from gi_liver / rheum_immune). Glomerular disease work-up (urine sediment, "
             "protein/creatinine ratio, ANCA/anti-GBM/complement): G_GLOMERULAR. Scrotal Doppler US as initial "
             "imaging for acute scrotal pain: C_SCROTAL (danger_gate); torsion is shared with peds_obgyn (peak in "
             "adolescents; peds_obgyn keeps it). Urosepsis: G_SEPSIS. 2026-09-29: 'creatinine rises late' reworded "
             "to the KDIGO definition (C_AKI). ECG insensitive for hyperkalaemia, manage by serial K (Montague 2008 "
             "abstract). Obstructing stone + infection: urgent decompression, nephrostomy or stent equally effective "
             "(Pearle 1998 RCT abstract). Heme-positive dipstick with few RBCs = myoglobinuria; dipstick cannot "
             "tell Hb from Mb (Chavez 2016, full text PMC4908773). Immediate exploration on clinical suspicion of "
             "torsion (Nason 2013 abstract) + damage after 6 h delay (C_SCROTAL abstract). Ruptured AAA misdiagnosed "
             "in 30%, renal colic the commonest (Marston 1992 abstract; 'first renal colic' reworded to abdominal / "
             "back pain in the elderly). Febrile infants 2-24 months: UA + culture (AAP 2011 abstract). Pregnancy "
             "pyelonephritis septicaemia 17% (Hill 2005 abstract; 'common' reworded); physiologic dilatation more "
             "often right (Faundes 1998 abstract). Pulmonary-renal syndrome (Boyle 2022 abstract).",
    ),
)}

# --------------------------------------------------------------------------------------------
# endo_metab / psych (2026-09-29, knowledge-rag). Appended, existing entries above unchanged.
# Bibliographic data checked against PubMed E-utilities esummary on 2026-09-29; "abstract" = the claim is in the
# PubMed abstract (efetch), otherwise bibliographic data only and the content is reviewer knowledge (see `note`).
# --------------------------------------------------------------------------------------------

C_DKA_ABDOMINAL_PAIN = Citation(
    "Umpierrez G, Freire AX.",
    "Abdominal pain in patients with hyperglycemic crises",
    "J Crit Care", 2002, "17(1):63-67", doi="10.1053/jcrc.2002.33030", pmid="12040551", verified=True,
)
C_ATA_THYROTOXICOSIS = Citation(
    "Ross DS, Burch HB, Cooper DS, et al.",
    "2016 American Thyroid Association Guidelines for Diagnosis and Management of Hyperthyroidism and Other Causes "
    "of Thyrotoxicosis",
    "Thyroid", 2016, "26(10):1343-1421", doi="10.1089/thy.2016.0229", pmid="27521067", verified=True,
    short_author="ATA 갑상선 중독증 지침",
)
C_ADRENAL_INSUFFICIENCY = Citation(
    "Bornstein SR, Allolio B, Arlt W, et al.",
    "Diagnosis and Treatment of Primary Adrenal Insufficiency: An Endocrine Society Clinical Practice Guideline",
    "J Clin Endocrinol Metab", 2016, "101(2):364-389", doi="10.1210/jc.2015-1710", pmid="26760044", verified=True,
    short_author="내분비학회 부신 기능 저하 지침",
)
C_HYPONATRAEMIA = Citation(
    "Spasovski G, Vanholder R, Allolio B, et al.",
    "Clinical practice guideline on diagnosis and treatment of hyponatraemia",
    "Eur J Endocrinol", 2014, "170(3):G1-G47", doi="10.1530/EJE-13-1020", pmid="24569125", verified=True,
    short_author="유럽 저나트륨혈증 지침",
)
C_PHEO = Citation(
    "Lenders JW, Duh QY, Eisenhofer G, et al.",
    "Pheochromocytoma and paraganglioma: an endocrine society clinical practice guideline",
    "J Clin Endocrinol Metab", 2014, "99(6):1915-1942", doi="10.1210/jc.2014-1498", pmid="24893135", verified=True,
)
C_HYPOGLYCAEMIA = Citation(
    "Cryer PE, Axelrod L, Grossman AB, et al.",
    "Evaluation and management of adult hypoglycemic disorders: an Endocrine Society Clinical Practice Guideline",
    "J Clin Endocrinol Metab", 2009, "94(3):709-728", doi="10.1210/jc.2008-1410", pmid="19088155", verified=True,
)
C_PITUITARY_APOPLEXY = Citation(
    "Rajasekaran S, Vanderpump M, Baldeweg S, et al.",
    "UK guidelines for the management of pituitary apoplexy",
    "Clin Endocrinol (Oxf)", 2011, "74(1):9-20", doi="10.1111/j.1365-2265.2010.03913.x", pmid="21044119",
    verified=True,
)
C_CAM = Citation(
    "Inouye SK, van Dyck CH, Alessi CA, et al.",
    "Clarifying confusion: the confusion assessment method. A new method for detection of delirium",
    "Ann Intern Med", 1990, "113(12):941-948", doi="10.7326/0003-4819-113-12-941", pmid="2240918", verified=True,
)
C_DELIRIUM = Citation(
    "Inouye SK, Westendorp RG, Saczynski JS.",
    "Delirium in elderly people",
    "Lancet", 2014, "383(9920):911-922", doi="10.1016/S0140-6736(13)60688-1", pmid="23992774", verified=True,
)
C_CSSRS = Citation(
    "Posner K, Brown GK, Stanley B, et al.",
    "The Columbia-Suicide Severity Rating Scale: initial validity and internal consistency findings from three "
    "multisite studies with adolescents and adults",
    "Am J Psychiatry", 2011, "168(12):1266-1277", doi="10.1176/appi.ajp.2011.10111704", pmid="22193671",
    verified=True,
)
C_NMS_DELPHI = Citation(
    "Gurrera RJ, Caroff SN, Cohen A, et al.",
    "An international consensus study of neuroleptic malignant syndrome diagnostic criteria using the Delphi method",
    "J Clin Psychiatry", 2011, "72(9):1222-1228", doi="10.4088/JCP.10m06438", pmid="21733489", verified=True,
)
C_HUNTER = Citation(
    "Dunkley EJ, Isbister GK, Sibbritt D, et al.",
    "The Hunter Serotonin Toxicity Criteria: simple and accurate diagnostic decision rules for serotonin toxicity",
    "QJM", 2003, "96(9):635-642", doi="10.1093/qjmed/hcg109", pmid="12925718", verified=True,
)
C_BFCRS = Citation(
    "Bush G, Fink M, Petrides G, et al.",
    "Catatonia. I. Rating scale and standardized examination",
    "Acta Psychiatr Scand", 1996, "93(2):129-136", doi="10.1111/j.1600-0447.1996.tb09814.x", pmid="8686483",
    verified=True,
)
C_CIWA_AR = Citation(
    "Sullivan JT, Sykora K, Schneiderman J, et al.",
    "Assessment of alcohol withdrawal: the revised clinical institute withdrawal assessment for alcohol scale "
    "(CIWA-Ar)",
    "Br J Addict", 1989, "84(11):1353-1357", doi="10.1111/j.1360-0443.1989.tb00737.x", pmid="2597811", verified=True,
)
C_AUTOIMMUNE_ENCEPHALITIS = Citation(
    "Graus F, Titulaer MJ, Balu R, et al.",
    "A clinical approach to diagnosis of autoimmune encephalitis",
    "Lancet Neurol", 2016, "15(4):391-404", doi="10.1016/S1474-4422(15)00401-9", pmid="26906964", verified=True,
)
ENDO_PSYCH_CITATIONS: tuple[Citation, ...] = (
    C_DKA_ABDOMINAL_PAIN, C_ATA_THYROTOXICOSIS, C_ADRENAL_INSUFFICIENCY, C_HYPONATRAEMIA, C_PHEO, C_HYPOGLYCAEMIA,
    C_PITUITARY_APOPLEXY, C_CAM, C_DELIRIUM, C_CSSRS, C_NMS_DELPHI, C_HUNTER, C_BFCRS, C_CIWA_AR,
    C_AUTOIMMUNE_ENCEPHALITIS)
NEW_CITATIONS = NEW_CITATIONS + ENDO_PSYCH_CITATIONS

SPECIALTIES.update({s.id: s for s in (
    SpecialtySpec(
        id="endo_metab", name_ko="내분비·대사",
        role_ko="혈당 이상, 칼슘·나트륨 이상, 설명 안 되는 피로·체중·체온·맥박 변화와 의식 변화를 호르몬과 대사 관점에서 봅니다. 혈당과 혈액가스부터 봅니다.",
        must_not_miss=("당뇨병성 케톤산증", "고삼투성 고혈당 상태", "저혈당", "부신 위기", "갑상선 중독 위기",
                       "점액수종 혼수", "중증 고칼슘혈증", "중증 저나트륨혈증", "갈색세포종 위기", "뇌하수체 졸중"),
        key_asks=("갈증·다뇨·체중 감소, 인슐린 중단·SGLT2 억제제", "스테로이드 장기 복용·최근 중단",
                  "더위/추위 못 견딤·두근거림·떨림", "발작성 두통·발한·두근거림", "갑작스러운 심한 두통+시야 이상",
                  "이뇨제·리튬·칼슘/비타민 D 복용"),
        key_exams=("의식·탈수·쿠스말 호흡·아세톤 냄새", "체온·맥박·기립성 혈압", "갑상선 종대·안구 돌출",
                   "피부 색소 침착·자주색 선조", "시야(양측 이측 반맹)·안구 운동"),
        key_tests=("혈당(즉시)", "케톤·동맥혈 가스·음이온 차", "혈장·소변 삼투압, 소변 Na", "전해질·보정 칼슘", "TSH·유리 T4",
                   "아침 코르티솔·ACTH", "PTH", "혈장 유리 메타네프린", "뇌하수체 MRI"),
        pitfalls=("DKA는 복통·구토로 와서 위장관 질환으로 오인: 복통+고혈당이면 케톤·혈액가스",
                  "SGLT2 억제제 복용 중에는 혈당이 높지 않아도 DKA", "부신 위기 의심 시 검사 결과를 기다리며 스테로이드 지연 금지",
                  "의식 저하·이상 행동은 혈당부터"),
        rule_ids=(),
        criteria_ids=("dka_hhs_2024",),
        protocol_categories=("fatigue", "palpitations", "menstrual", "chronic_weakness"),
        rule_out_names=("저혈당", "당뇨병성 케톤산증"),
        branch_notes={
            "peds": ("소아 첫 1형 당뇨는 DKA로 옴. 치료 중 두통·의식 저하는 뇌부종",),
            "pregnant": ("임신 중 DKA는 혈당이 조금만 높아도 생김. 입덧(임신 오조)과 갑상선 중독 구별",),
        },
        citations=(C_HYPERGLYCEMIC_CRISES, C_DKA_ABDOMINAL_PAIN, C_ATA_THYROTOXICOSIS, C_ADRENAL_INSUFFICIENCY,
                   C_HYPONATRAEMIA, C_PHEO, C_HYPOGLYCAEMIA, C_PITUITARY_APOPLEXY),
        verification="unverified",
        note="DKA/HHS definition (glucose >= 200 mg/dL or known diabetes, ketones, pH/bicarbonate, osmolality): "
             "criteria dka_hhs_2024 / C_HYPERGLYCEMIC_CRISES (ADA/EASD 2024, as danger_gate). Abdominal pain in 46% of "
             "DKA, associated with the metabolic acidosis: C_DKA_ABDOMINAL_PAIN (abstract) - past trajectories routed "
             "DKA to gi_liver, so the pitfall names it. Plasma free metanephrines first for pheochromocytoma: C_PHEO "
             "(abstract). Whipple's triad / glucose first: C_HYPOGLYCAEMIA (abstract). Rapid hydrocortisone in "
             "pituitary apoplexy: C_PITUITARY_APOPLEXY (abstract). Thyroid storm (TSH/fT4), adrenal crisis "
             "(cortisol/ACTH, steroid not delayed for tests), hyponatraemia work-up by serum/urine osmolality and "
             "urine sodium: C_ATA_THYROTOXICOSIS / C_ADRENAL_INSUFFICIENCY / C_HYPONATRAEMIA (bibliographic data "
             "verified, full text not read). Overlap: hyponatraemia/hyperkalaemia names (E87) route to renal_uro; "
             "SIADH, adrenal, thyroid causes are listed here. Protocol 'abdominal_pain' is not owned here (its "
             "can't-miss list is surgical); the DKA pitfall covers it. Reviewer knowledge: SGLT2-inhibitor link of "
             "DKA without marked hyperglycaemia; content of the thyroid storm / adrenal crisis / hyponatraemia "
             "work-up items; hypercalcaemia (Ca/PTH), myxoedema coma, physical signs, pediatric cerebral oedema and "
             "pregnancy notes.",
    ),
    SpecialtySpec(
        id="psych", name_ko="정신",
        role_ko="기분·행동·지각·사고 변화를 보되, 먼저 신체(기질적) 원인과 위험(자살, 약물 반응, 금단)을 가립니다. 새로 생긴 정신 증상은 기질적 원인이 배제될 때까지 정신과 질환으로 단정하지 않습니다.",
        must_not_miss=("섬망(기저 신체 질환)", "자살 위험", "신경이완제 악성 증후군", "세로토닌 증후군", "알코올 금단·진전 섬망",
                       "베르니케 뇌병증", "뇌염(자가면역·감염)", "저혈당·갑상선 중독", "물질 중독", "긴장증"),
        key_asks=("시작 시점·경과(급성, 하루 중 기복)", "자살 생각·계획·수단·이전 시도", "새 약·증량·중단(항정신병제, 세로토닌 약)",
                  "음주량·마지막 음주, 약물 사용", "발열·두통·경련", "과거 조증·우울 삽화, 가족력"),
        key_exams=("주의력(숫자 거꾸로 말하기)·의식 수준", "체온·맥박·혈압", "근강직 vs 클로누스·반사 항진", "떨림·발한·동공",
                   "안구 운동 이상·운동실조", "무언·경직·자세 유지(긴장증)"),
        key_tests=("혈당(즉시)", "전해질·Ca·신기능·간기능", "TSH", "CK", "소변 약물 검사·알코올 농도", "CBC·CRP",
                   "리튬 등 약물 농도", "뇌 MRI·뇌파·요추천자(발열·신경 징후 동반 시)"),
        pitfalls=("고령·입원 환자의 새 정신 증상을 우울·치매로 단정(저활동성 섬망 놓침)",
                  "첫 정신병 삽화는 기질적 원인 검사 전 확진 금지", "자살 생각은 직접 물음",
                  "강직·고열·CK 상승 = NMS, 클로누스·반사 항진 = 세로토닌 증후군"),
        rule_ids=(),
        criteria_ids=("bipolar_dsm5tr",),
        protocol_categories=("psychiatric", "cognitive"),
        rule_out_names=("저혈당", "뇌수막염"),
        branch_notes={
            "peds": ("소아·청소년: 자해·학대·괴롭힘 확인, 보호자와 따로 면담",),
            "pregnant": ("산후 급성 혼동·환각·기분 급변 = 산후 정신병(응급), 영아 안전 확인",),
        },
        citations=(_protocols.G_PSYCH_EVAL, C_CAM, C_DELIRIUM, C_CSSRS, C_NMS_DELPHI, C_HUNTER, C_BFCRS, C_CIWA_AR,
                   C_AUTOIMMUNE_ENCEPHALITIS),
        verification="unverified",
        note="Organic causes and suicide risk in the psychiatric evaluation: G_PSYCH_EVAL (APA 2015, as protocols "
             "'psychiatric'). Delirium = acute onset/fluctuation + inattention (CAM, C_CAM); under-recognised in "
             "older people: C_DELIRIUM (abstract). Suicide ideation/behaviour asked directly with a structured scale: "
             "C_CSSRS (instrument). NMS elements hyperthermia, rigidity, mental status change, CK elevation: "
             "C_NMS_DELPHI (abstract). Serotonin toxicity by clonus/hyperreflexia (Hunter): C_HUNTER (abstract). "
             "Catatonia signs: C_BFCRS (instrument). Alcohol withdrawal assessment: C_CIWA_AR (instrument). "
             "Autoimmune encephalitis in the organic differential: C_AUTOIMMUNE_ENCEPHALITIS (bibliographic data "
             "verified). bipolar_dsm5tr moved here from neuro (2026-09-29); protocol 'psychiatric' and 'cognitive' "
             "stay shared with neuro (organic cause first). Reviewer knowledge: psychiatric presentation of "
             "autoimmune encephalitis; first-episode psychosis work-up list, hypoactive delirium in the elderly, "
             "Wernicke, pediatric and postpartum psychosis notes.",
    ),
)})

SPECIALTY_IDS: tuple[str, ...] = tuple(SPECIALTIES)

# --------------------------------------------------------------------------------------------
# Patient branch (age / pregnancy), reusing the existing readers
# --------------------------------------------------------------------------------------------

_POSTPARTUM = ("출산 후", "출산한 지", "분만 후", "분만한 지", "산후", "제왕절개", "postpartum")
_RE_GA_WEEKS = re.compile(r"(?:임신|재태)\s*(\d{1,2})\s*주(?![^.]{0,15}(?:태어|출생|분만|조산|낳))|(\d{1,2})\s*주\s*(?:차\s*)?임신")


def _case_text(state) -> str:
    parts = [state.initial_info or ""]
    for t in getattr(state, "turns", []):
        typ = getattr(t.action.type, "value", str(t.action.type))
        if typ != "DIAGNOSE" and t.response and "제공되지 않습니다" not in t.response:
            parts.append(t.response)
    return ". ".join(parts)


def patient_profile(state) -> dict:
    """{"branch": peds | pregnant | female_repro | adult, "age": years or None, "weeks": gestational weeks or None,
    "postpartum": bool}. Age is read from the initial information only; pregnancy from everything the environment
    said (initial info + responses)."""
    initial = state.initial_info or ""
    age, peds = _age(initial)
    out = {"branch": "adult", "age": age, "weeks": None, "postpartum": False}
    if peds:
        out["branch"] = "peds"
        return out
    text = _case_text(state)
    preds = _protocols.PREDICATES
    m = _RE_GA_WEEKS.search(text.lower())
    weeks = int(m.group(1) or m.group(2)) if m else None
    postpartum = contains_affirmed(text, _POSTPARTUM)
    if preds["current_pregnancy"](initial, text) or postpartum:
        out.update(branch="pregnant", weeks=weeks, postpartum=postpartum)
    elif preds["pregnancy_test_abdominal"](initial, text):
        out["branch"] = "female_repro"
    return out


def _age_ko(age: float) -> str:
    if age < 1 / 12:
        return f"생후 {max(1, round(age * 365.25))}일"
    if age < 2:
        return f"{round(age * 12)}개월"
    return f"{int(age)}세"


def render_profile(profile: dict) -> str:
    b, age = profile.get("branch", "adult"), profile.get("age")
    if b == "peds":
        if age is None:
            return "[환자 구분] 소아(나이 미상): 나이(개월 수)부터 확인하고 나이별 활력징후 기준을 쓰세요."
        s = f"[환자 구분] 소아 {_age_ko(age)}: 성인 활력징후 기준을 쓰지 마세요."
        hr, rr = _peds_row(_PEDS_HR, age), _peds_row(_PEDS_RR, age)
        if hr and rr:
            s += f" 이 나이 정상(1~99 백분위, Fleming 2011): 맥박 {hr[0]}~{hr[3]}/분, 호흡수 {rr[0]}~{rr[3]}/분."
        if age < 17:
            s += f" 수축기 혈압 {int(_peds_sbp_floor(age))} mmHg 미만은 저혈압(PALS)."
        return s
    if b == "pregnant":
        if profile.get("postpartum") and profile.get("weeks") is None:
            # ACOG PB 222 / Sibai 2012 (postpartum preeclampsia-eclampsia), Kamel 2014 (thrombosis risk highest in the
            # first 6 weeks, still raised to 12 weeks), AHA 2024 CVT statement (see consult_sources.py)
            return ("[환자 구분] 산후 환자: 전자간증·자간증은 산후에도 생깁니다. 혈전(폐색전증·뇌졸중·뇌정맥동 혈전증) 위험은 "
                    "산후 6주에 가장 높고 12주까지 남습니다.")
        w = profile.get("weeks")
        stage = "" if w is None else (f" {w}주(20주 전)" if w < 20 else f" {w}주(20주 이후)")
        return f"[환자 구분] 임신 중{stage}: 임신 관련 응급을 먼저 생각하고 약·영상 안전을 지키세요."
    if b == "female_repro":
        return "[환자 구분] 가임기 여성(임신 여부 미확인): 방사선 영상·약 처방 전에 임신 검사가 필요합니다."
    return "[환자 구분] 성인" + (f" {int(age)}세" if age is not None else "") + "."


# --------------------------------------------------------------------------------------------
# Case context shared with advocate.py
# --------------------------------------------------------------------------------------------

FINDINGS_MAX = 1400  # chars of the findings ledger
FINDING_LINE_MAX = 600  # per status line (the newest part is kept)
DDX_MAX = 700
DONE_MAX = 450
RECENT_TURNS = 2
RECENT_RESP_MAX = 160
RESOURCES_MAX = 900
RESOURCE_ITEM_MAX = 300
INITIAL_MAX = 600
HINT_MAX_CHARS = 600  # SubagentCall.max_chars_out: cap of the Korean hint passed back to the doctor prompt


def _cap(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def _tail(s: str, n: int) -> str:
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _findings_block(state) -> str:
    f = state.findings.render(exclude_unverified=True) if getattr(state, "findings", None) else ""
    if not f:
        return "(아직 정리된 소견 없음)"
    lines = [_tail(x, FINDING_LINE_MAX) for x in f.split("\n")]
    return _tail("\n".join(lines), FINDINGS_MAX)


def ddx_names(state) -> list[str]:
    led = getattr(state, "ddx_ledger", None)
    if led is not None and led.entries:
        return [e.dx for e in led.ranked()]
    return [str(d.get("dx", "")).strip() for d in (getattr(state, "ddx", None) or []) if isinstance(d, dict)
            and str(d.get("dx", "")).strip()]


def _ddx_block(state) -> str:
    led = getattr(state, "ddx_ledger", None)
    text = led.render() if led is not None else ""
    if not text:
        text = "\n".join(f"{i}. {d.get('dx')} ({round(float(d.get('p', 0) or 0) * 100)}%)"
                         for i, d in enumerate(getattr(state, "ddx", None) or [], 1)
                         if isinstance(d, dict) and d.get("dx"))
    if not text:
        return "(아직 없음)"
    return _cap("\n".join(_cap(x, 200) for x in text.split("\n")), DDX_MAX)


def _done_block(state) -> str:
    items = []
    for t in reversed(getattr(state, "turns", [])):  # newest first, oldest dropped at the cap
        typ = getattr(t.action.type, "value", str(t.action.type))
        mark = " (결과 없음)" if "제공되지 않습니다" in (t.response or "") else ""
        item = f"{typ} {_cap(t.action.content, 30)}{mark}"
        if sum(len(x) + 2 for x in items) + len(item) > DONE_MAX:
            items.append("…")
            break
        items.append(item)
    return "; ".join(items) if items else "(없음)"


def _recent_block(state) -> str:
    turns = getattr(state, "turns", [])[-RECENT_TURNS:]
    return "\n".join(f"{getattr(t.action.type, 'value', t.action.type)}: {_cap(t.action.content, 60)} → "
                     f"{_cap(t.response, RECENT_RESP_MAX)}" for t in turns)


def case_context(state) -> str:
    """Grounded case summary: initial info, verified findings ledger, last exchanges, current DDx, actions done."""
    parts = [f"[처음 정보] {_cap(state.initial_info, INITIAL_MAX)}",
             "[소견 장부(환경 답변에서 확인된 것만)]\n" + _findings_block(state)]
    if recent := _recent_block(state):
        parts.append("[최근 대화]\n" + recent)
    parts += ["[현재 감별 진단]\n" + _ddx_block(state), "[이미 한 행동(최신순)] " + _done_block(state)]
    return "\n\n".join(parts)


def render_resources(resources: dict | None) -> str:
    """Optional evidence from knowledge/specialty.py: {label: str | list | dict}. Empty values are skipped."""
    if not isinstance(resources, dict) or not resources:
        return ""
    lines, total = [], 0
    for k, v in resources.items():
        if isinstance(v, (list, tuple)):
            v = "; ".join(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in v if x)
        elif isinstance(v, dict):
            v = "; ".join(f"{a}: {b}" for a, b in v.items() if b)
        v = str(v or "").strip()
        if not v:
            continue
        line = f"- {k}: {_cap(v, RESOURCE_ITEM_MAX)}"
        if total + len(line) > RESOURCES_MAX:
            break
        lines.append(line)
        total += len(line) + 1
    return "[참고 자료]\n" + "\n".join(lines) if lines else ""


# --------------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------------

CONSULT_SYSTEM = """당신은 {name_ko} 분과 자문의입니다. 주치의의 환자를 검토하고 조언만 합니다. 진단을 확정하지 않습니다.
{role_ko}

규칙:
1. [처음 정보], [소견 장부], [최근 대화]에 있는 소견만 근거로 쓰세요. 없는 검사 결과나 소견을 지어내지 마세요.
2. 결과가 아직 없으면 "미확인"입니다. "결과가 제공되지 않습니다"는 정상이 아닙니다.
3. next_actions: 1순위와 2순위 후보를 가장 잘 가르는 행동부터 최대 3개. 가장 판별력 높은 하나를 첫째로. 이미 한 행동과 결과 없는 검사는 다시 제안하지 마세요. type은 ASK(문진), EXAM(진찰), TEST(검사) 중 하나.
4. missed_dangers: 아직 확인·배제하지 않은 위험 질환만. 이미 배제된 것은 빼세요.
5. ddx_add: 현재 감별 진단에 없는데 소견을 잘 설명하는 질환만(없으면 빈 목록).
6. 한국어로 짧게 쓰고, JSON 한 줄만 출력하세요.
{{"assessment": "...", "ddx_add": [{{"name": "...", "why": "..."}}], "missed_dangers": ["..."], "next_actions": [{{"type": "ASK|EXAM|TEST", "content": "...", "why": "..."}}], "confidence_note": "..."}}"""

CONSULT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "ddx_add": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name"]}},
        "missed_dangers": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        "next_actions": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["ASK", "EXAM", "TEST"]}, "content": {"type": "string"},
            "why": {"type": "string"}}, "required": ["type", "content"]}},
        "confidence_note": {"type": "string"},
    },
    "required": ["assessment", "next_actions"],
}


def _knowledge_links(spec: SpecialtySpec, state) -> list[str]:
    """Only what applies to this patient: protocol can't-miss for the chief complaint, rules in their validated
    population, criteria matching a current DDx name."""
    initial = state.initial_info or ""
    out = []
    cant: list[str] = []
    for p in protocols_for(initial):
        if p.category in spec.protocol_categories:
            cant += [d for d in p.cant_miss if d not in cant]
    if cant:
        out.append("[이 주호소에서 배제할 위험 질환(안전 프로토콜)] " + ", ".join(cant))
    rules = [r.cite for r in rules_for(initial) if r.id in spec.rule_ids]
    if rules:
        out.append("[이 환자에 적용 가능한 임상 결정 규칙] " + ", ".join(rules))
    crit: list[str] = []
    for dx in ddx_names(state)[:5]:
        for c in criteria_for(dx, related=False):
            label = f"{c.short}({c.name_ko})"
            if c.id in spec.criteria_ids and label not in crit:
                crit.append(label)
    if crit:
        out.append("[감별 진단 관련 진단 기준] " + ", ".join(crit))
    return out


def build_consult(state, specialty: str, resources: dict | None = None) -> SubagentCall:
    """One consult call for `specialty` (an id in SPECIALTIES). Raises KeyError for an unknown id."""
    spec = SPECIALTIES[specialty]
    profile = patient_profile(state)
    system = CONSULT_SYSTEM.format(name_ko=spec.name_ko, role_ko=spec.role_ko)
    blocks = [spec.render(profile["branch"]), render_profile(profile)]
    blocks += _knowledge_links(spec, state)
    if res := render_resources(resources):
        blocks.append(res)
    blocks.append(case_context(state))
    blocks.append(f"위 환자에 대해 {spec.name_ko} 자문 JSON을 쓰세요.")
    user = "\n\n".join(blocks)
    return SubagentCall(name=f"consult:{specialty}",
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        json_schema=CONSULT_SCHEMA, max_chars_out=HINT_MAX_CHARS)


# --------------------------------------------------------------------------------------------
# Tolerant parsing (shared with advocate.py)
# --------------------------------------------------------------------------------------------

ACTION_TYPES = ("ASK", "EXAM", "TEST")
_TYPE_ALIASES = {"ASK": "ASK", "문진": "ASK", "질문": "ASK", "병력": "ASK", "HISTORY": "ASK",
                 "EXAM": "EXAM", "진찰": "EXAM", "신체진찰": "EXAM", "신체 진찰": "EXAM", "PHYSICAL": "EXAM",
                 "TEST": "TEST", "검사": "TEST", "LAB": "TEST", "IMAGING": "TEST"}
_FENCE = re.compile(r"```(?:json)?", re.I)
MAX_ITEMS = 3


def _close_json(frag: str) -> str | None:
    """Close a truncated JSON fragment (open string, arrays, objects). None if brackets are mismatched."""
    stack, in_str, esc = [], False, False
    for i, ch in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                return frag[: i + 1]
    return frag + ('"' if in_str else "") + "".join(reversed(stack))


def _repair(text: str, keys: frozenset[str]) -> dict | None:
    """Best effort for output cut off mid-JSON (max tokens): close it, cutting back to earlier commas if needed."""
    start = text.find("{")
    while start != -1:
        s = text[start:]
        cuts = [len(s)] + [i for i in range(len(s) - 1, 0, -1) if s[i] == ","][:40]
        for cut in cuts:
            closed = _close_json(s[:cut].rstrip().rstrip(","))
            if closed is None:
                continue
            try:
                obj = json.loads(closed)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and keys & set(obj):
                return obj
        start = text.find("{", start + 1)
    return None


def extract_json(text: str | None, keys: frozenset[str]) -> dict | None:
    """The last JSON object with at least one of `keys` at top level; harmony leftovers, reasoning tags, code fences
    and surrounding prose are ignored; a truncated object is closed as a last resort. Never raises."""
    try:
        raw = text or ""
        final, analysis = split_harmony(raw)
        cands = [_FENCE.sub("", _THOUGHT_RE.sub("", final)), final] + ([analysis] if analysis else []) + [raw]
        for c in cands:
            objs = [o for o in _json_objects(c) if keys & set(o)]
            if objs:
                return objs[-1]
        for c in cands[:2]:
            if (obj := _repair(c, keys)) is not None:
                return obj
    except Exception:  # noqa: BLE001 - parser must never raise
        return None
    return None


def _s(x, n: int) -> str:
    if isinstance(x, (dict, list)):
        return ""
    return _cap(str(x if x is not None else "").replace("\n", " "), n)


def as_list(x) -> list:
    if x is None or x == "":
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def norm_named(items, known: list[str] | None = None, n: int = MAX_ITEMS) -> list[dict]:
    """[{"name","why"}] from dicts (name/dx/diagnosis) or plain strings; drops empties, duplicates and names that are
    already in `known` (same_dx)."""
    out: list[dict] = []
    for it in as_list(items):
        if isinstance(it, dict):
            name = _s(it.get("name") or it.get("dx") or it.get("diagnosis"), 40)
            why = _s(it.get("why") or it.get("reason") or it.get("evidence"), 90)
        else:
            name, why = _s(it, 40), ""
        if not name or any(same_dx(name, o["name"]) for o in out) or any(same_dx(name, k) for k in known or []):
            continue
        out.append({"name": name, "why": why})
        if len(out) >= n:
            break
    return out


def norm_strings(items, n: int = MAX_ITEMS, width: int = 50) -> list[str]:
    out: list[str] = []
    for it in as_list(items):
        s = _s(it.get("name") or it.get("dx") or "", width) if isinstance(it, dict) else _s(it, width)
        if s and not any(similarity(s, o) >= 0.8 for o in out):
            out.append(s)
        if len(out) >= n:
            break
    return out


def norm_actions(items, n: int = MAX_ITEMS) -> list[dict]:
    """[{"type": ASK|EXAM|TEST, "content", "why"}]; anything else (DIAGNOSE, unknown types, empty content) is dropped."""
    out: list[dict] = []
    for it in as_list(items):
        if not isinstance(it, dict):
            continue
        raw_t = str(it.get("type") or it.get("action") or "").strip()
        typ = _TYPE_ALIASES.get(raw_t.upper(), _TYPE_ALIASES.get(raw_t, ""))
        content = _s(it.get("content") or it.get("action_content") or it.get("what"), 80)
        if typ not in ACTION_TYPES or not content:
            continue
        if any(o["type"] == typ and similarity(o["content"], content) >= 0.7 for o in out):
            continue
        out.append({"type": typ, "content": content, "why": _s(it.get("why") or it.get("reason"), 90)})
        if len(out) >= n:
            break
    return out


def fit_lines(lines: list[str], max_chars: int) -> str:
    """Join lines, dropping trailing lines (then cutting the last) to stay within max_chars."""
    out: list[str] = []
    for ln in lines:
        cur, sep = len("\n".join(out)), 1 if out else 0
        if cur + sep + len(ln) > max_chars:
            if (room := max_chars - cur - sep) > 20:
                out.append(_cap(ln, room))
            break
        out.append(ln)
    return "\n".join(out)


def _action_text(a: dict) -> str:
    return f"{a['type']} {a['content']}" + (f" — {a['why']}" if a.get("why") else "")


def parse_consult(text: str | None, specialty: str = "", known_ddx: list[str] | None = None,
                  max_chars: int = HINT_MAX_CHARS) -> SubagentResult:
    """Parse a consult answer. ok=False (empty hint) when no JSON object is found or it has no usable content.
    The hint only restates what the JSON said (no content is added here). `known_ddx` drops ddx_add entries that
    are already on the DDx. Never raises."""
    name = f"consult:{specialty}" if specialty else "consult"
    try:
        obj = extract_json(text, frozenset(CONSULT_SCHEMA["properties"]))
        if obj is None:
            return SubagentResult(name, False, "", [], [], [], {})
        assessment = _s(obj.get("assessment"), 160)
        ddx_add = norm_named(obj.get("ddx_add"), known_ddx)
        dangers = norm_strings(obj.get("missed_dangers"))
        actions = norm_actions(obj.get("next_actions"))
        note = _s(obj.get("confidence_note"), 100)
        if not (assessment or ddx_add or dangers or actions):
            return SubagentResult(name, False, "", [], [], [], obj)
        label = SPECIALTIES[specialty].name_ko if specialty in SPECIALTIES else "분과"
        lines = [f"[{label} 자문]" + (f" {assessment}" if assessment else "")]
        if dangers:
            lines.append("- 아직 배제 안 된 위험: " + ", ".join(dangers))
        if actions:
            lines.append("- 권하는 다음 행동: " + " / ".join(_action_text(a) for a in actions))
        if ddx_add:
            lines.append("- 추가 감별: " + ", ".join(d["name"] + (f"({d['why']})" if d["why"] else "") for d in ddx_add))
        if note:
            lines.append("- 확신도: " + note)
        return SubagentResult(name, True, fit_lines(lines, max_chars), ddx_add, actions, dangers, obj)
    except Exception:  # noqa: BLE001 - parser must never raise
        return SubagentResult(name, False, "", [], [], [], {})
