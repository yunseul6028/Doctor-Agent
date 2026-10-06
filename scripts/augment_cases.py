"""Augment converted cases with results for tests/exams a physician would reasonably order (offline, dev only).

Converted exam questions deliberately omit the decisive test (e.g. vessel imaging for Takayasu arteritis), so a
correct request returns "not provided". This fills in plausible results **consistent with the answer and with the
existing findings**, including normal/negative results for common differential tests, and marks every added entry.

    source .venv/bin/activate
    python scripts/augment_cases.py                       # all cases in data/cases_clinicalqa
    python scripts/augment_cases.py --ids 181 521         # only these
    python scripts/augment_cases.py --overwrite

LLM: AUGMENT_LLM_* in .env (falls back to LLM_*). Input cases are never modified.
Output: data/cases_clinicalqa_aug/cqa_<id>.json. Reproducibility record: data/labels/augmentation_meta.json.

Full mode (routine panel + decisive/differential tests, for every evaluation set):

    python scripts/augment_cases.py --mode full                              # all sets → data/cases_aug/<set>/
    python scripts/augment_cases.py --mode full --in data/cases_clinicalqa --out data/cases_aug/clinicalqa --ids 181
    python scripts/augment_cases.py --mode measure                           # keyword-match rate of common requests, before/after

Full mode adds a standard exam/lab panel (vitals, HEENT, heart, lungs, abdomen, neuro, CBC, BMP, LFT, CRP/ESR, UA, CXR,
ECG, and presentation-dependent brain CT / abdominal US·CT / hCG / troponin / D-dimer / blood culture / lipase) for items
not already covered by an existing key, plus the decisive and differential tests for the chief complaint. Panel keys
are curated here (broad Korean + English search terms); the LLM only writes the results. Record:
data/labels/augmentation_full_meta.json.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.parser import _json_objects  # noqa: E402
from doctor_agent.config import LLMConfig  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Same as eval.run_local.load_dotenv (inlined so this script does not import the agent loop)."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

IN_DIR = ROOT / "data/cases_clinicalqa"
OUT_DIR = ROOT / "data/cases_clinicalqa_aug"
META = ROOT / "data/labels/augmentation_meta.json"

SYSTEM_PROMPT = """당신은 의학 교육용 진단 연습 증례를 보강하는 도구입니다.
증례에는 원래 시험 문제에 있던 소견만 들어 있어서, 의사가 합리적으로 요청할 만한 검사·진찰 결과가 빠져 있습니다.
빠진 결과를 정답 진단과 기존 소견에 **모순 없이** 채워 넣으세요.

규칙:
1. 이 주호소와 감별 진단에서 의사가 흔히 요청할 검사와 진찰을 떠올리세요. 확진 검사와, 다른 감별 진단을 배제하는 검사를 모두 포함하세요.
2. 이미 증례에 있는 항목은 다시 만들지 말고, 기존 수치와 소견을 바꾸지 마세요.
3. 정답 질환에서 실제로 나올 법한 현실적인 결과를 쓰세요. 수치에는 단위를 붙이세요. 감별 진단 배제용 검사는 대부분 정상 또는 음성으로 쓰세요.
4. 결과 문장에 최종 진단명을 직접 쓰지 마세요 (예: "타카야수 동맥염에 합당" 금지). 판독 소견만 쓰세요.
5. 추가 진찰 6개 이하, 추가 검사 10개 이하.
6. 키는 "|"로 구분한 검색어 (한국어 + 영어 소문자 약어)로 쓰세요. 예: "ct 혈관조영|cta|혈관 ct|ct angiography"

출력: JSON 객체 하나만 (설명 금지).
{"exam": {"<키>": "<소견>", ...}, "tests": {"<키>": "<결과>", ...}}"""


def build_user(case: dict) -> str:
    visible = {k: case[k] for k in ("initial", "history", "exam", "tests") if k in case}
    return (f"[정답 진단] {case['diagnosis']}\n[동의어] {', '.join(case.get('aliases', []))}\n\n"
            f"[현재 증례]\n{json.dumps(visible, ensure_ascii=False, indent=1)}\n\n빠진 진찰·검사 결과를 JSON으로 추가하세요.")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


# generic words that may appear in diagnosis names without giving the answer away
_GENERIC = {"급성", "만성", "원발성", "이차성", "증후군", "질환", "장애", "동맥염", "간염", "결핍", "결핍증", "감염", "현재",
            "삽화", "acute", "chronic", "primary", "syndrome", "disease", "disorder", "deficiency"}


def _distinctive_tokens(names: list[str]) -> set[str]:
    toks = {t.lower() for n in names for t in re.split(r"[\s,()/·\-]+", n)}
    return {t for t in toks if len(t) >= 3 and t not in _GENERIC}


def validate(case: dict, add: dict) -> list[str]:
    errors = []
    raw = [case["diagnosis"], *case.get("aliases", [])]
    names = [_norm(re.sub(r"\(.*?\)", "", n)) for n in raw if len(n) >= 3] + sorted(_distinctive_tokens(raw))
    for section in ("exam", "tests"):
        entries = add.get(section)
        if not isinstance(entries, dict):
            errors.append(f"{section} missing")
            continue
        existing = {tok.strip().lower() for k in case.get(section, {}) for tok in k.split("|")}
        for key, val in entries.items():
            if not isinstance(val, str) or not val.strip():
                errors.append(f"empty value for {key}")
            if {tok.strip().lower() for tok in key.split("|")} & existing:
                errors.append(f"key overlaps existing entry: {key}")
            if any(n and n in _norm(val) for n in names):
                errors.append(f"diagnosis named in result: {key}")
    if len(add.get("exam", {})) > 6 or len(add.get("tests", {})) > 10:
        errors.append("too many entries")
    return errors


def augment_one(path: Path, client: OpenAICompatClient) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user(case)}]
    log = {"id": path.stem, "attempts": 0, "errors": []}
    for _ in range(3):
        log["attempts"] += 1
        objs = [o for o in _json_objects(client.chat(messages)) if "tests" in o or "exam" in o]
        add = objs[-1] if objs else {}
        errors = validate(case, add)
        if not errors:
            break
        log["errors"].append(errors)
        messages += [{"role": "assistant", "content": json.dumps(add, ensure_ascii=False)},
                     {"role": "user", "content": "다음 문제를 고쳐 다시 출력하세요: " + "; ".join(errors)}]
    else:
        # keep only the entries that pass on their own (drop overlapping keys / entries naming the diagnosis)
        add = {sec: {k: v for k, v in (add.get(sec) or {}).items() if not validate(case, {"exam": {}, "tests": {}, sec: {k: v}})}
               for sec in ("exam", "tests")}
        if not add["exam"] and not add["tests"]:
            log["status"] = "skipped"
            return log
        log["partial"] = True

    out = dict(case)
    out["exam"] = {**case.get("exam", {}), **add["exam"]}
    out["tests"] = {**case.get("tests", {}), **add["tests"]}
    out["augmented"] = {"exam": list(add["exam"]), "tests": list(add["tests"])}
    out["_note"] = (case.get("_note", "") + " | Entries listed in 'augmented' were generated by an LLM "
                    "(scripts/augment_cases.py), not in the source item, not clinician-reviewed.").strip(" |")
    (OUT_DIR / path.name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.update(status="ok", added_exam=len(add["exam"]), added_tests=len(add["tests"]))
    return log


# ─────────────────────────────────────────────── full mode ───────────────────────────────────────────────
FULL_ROOT = ROOT / "data/cases_aug"
# agentclinic / diagnosisarena: private evaluation sets (not redistributable, not in the public repository); they are
# only used when present locally (scripts/convert_english_cases.py).
FULL_SETS = {name: path for name, path in {
    "sample": ROOT / "data/sample_cases", "clinicalqa": ROOT / "data/cases_clinicalqa",
    "agentclinic": ROOT / "data/cases_agentclinic", "diagnosisarena": ROOT / "data/cases_diagnosisarena",
}.items() if path.is_dir()}
FULL_META = ROOT / "data/labels/augmentation_full_meta.json"
FULL_MAX_WORKERS = 3  # the API quota is shared with other runs
FULL_EXTRA_LIMIT = {"exam": 4, "tests": 8}

# (id, section, curated search key, what to report, coverage probes, conditional)
# probes: groups of synonymous doctor requests; the item counts as already covered when every group hits an existing key
# (generic modality tokens such as a bare "ct" are ignored for coverage).
PANEL = [
    ("vitals", "exam", "활력징후|활력 징후|바이탈|vital|혈압|맥박|체온|호흡수|산소포화도|spo2|blood pressure|heart rate",
     "활력징후: 혈압, 맥박, 호흡수, 체온, 산소포화도", [["활력징후 측정", "vital signs", "혈압 측정"]], False),
    ("general", "exam", "전신 상태|전신상태|전반적 상태|일반 상태|외관|general appearance|appearance|병색",
     "전신 상태: 의식 수준, 병색(급성/만성), 영양·수화 상태", [["전신 상태 관찰", "general appearance"]], False),
    ("heent", "exam", "heent|두경부|두부 진찰|결막|공막|인두|편도|구강|고막|이경|안구 진찰|눈 진찰|목구멍",
     "두경부(HEENT): 결막(창백), 공막(황달), 구강 점막, 인두·편도, 고막", [["heent 진찰", "인두 진찰", "결막 확인"]], False),
    ("neck", "exam", "경부|목 진찰|목 촉진|림프절|lymph|갑상선 촉진|갑상샘 촉진|갑상선 진찰|경정맥|jvp|neck",
     "경부: 림프절, 갑상선, 경정맥 확장, 경부 강직", [["경부 진찰", "경부진찰", "목 진찰", "림프절 촉진"]], False),
    ("heart", "exam", "심장 진찰|심장 청진|심장진찰|심장청진|심음|심잡음|cardiac exam|heart exam|heart sound|murmur",
     "심장: 리듬, 심음(S1/S2, S3/S4), 심잡음, 마찰음", [["심장 청진", "심장 진찰"]], False),
    ("lungs", "exam", "폐 청진|폐청진|흉부 청진|흉부청진|호흡음|폐 진찰|흉부 진찰|흉부진찰|lung exam|chest exam|수포음|천명음",
     "폐/흉부: 호흡음, 수포음·천명음, 타진음, 호흡 노력", [["폐 청진", "흉부 청진", "흉부 진찰"]], False),
    ("abdomen", "exam", "복부|배 진찰|배 촉진|abdom|장음|간비대|비장비대|간비장",
     "복부: 팽만, 장음, 압통·반발통, 간·비장 종대, 종괴, 늑골척추각 압통", [["복부 진찰", "복부진찰", "abdominal exam"]], False),
    ("extremities", "exam", "사지|팔다리|부종|하지 진찰|extremit|edema|말초 맥박|곤봉지|관절 진찰",
     "사지: 부종, 말초 맥박, 곤봉지, 관절 이상, 종아리 압통", [["사지 진찰", "하지 부종 확인"]], False),
    ("skin", "exam", "피부|발진|skin|rash|점상출혈|자반",
     "피부: 발진, 창백, 황달, 출혈반, 병변", [["피부 진찰", "피부진찰", "skin exam"]], False),
    ("neuro", "exam", "신경학적|신경학|신경 진찰|신경진찰|신경계|neuro|의식 수준|의식 상태|의식|지남력|mental status|뇌신경|cranial nerve|"
     "동공|근력|motor|감각 검사|감각|sensory|반사|reflex|바빈스키|babinski|소뇌|보행|gait|협조 운동|롬버그|romberg",
     "신경학적 진찰: 의식/지남력, 뇌신경, 근력, 감각, 심부건반사, 소뇌 기능·보행 (하위 항목별로 짧게)",
     [["신경학적 진찰", "신경학적진찰", "신경학 검사", "neurologic exam"]], False),
    ("cbc", "tests", "일반혈액검사|일반혈액|일반 혈액|전혈구|cbc|complete blood count|백혈구|헤모글로빈|혈소판|wbc|hemoglobin|platelet",
     "CBC: WBC(감별 포함), Hb, Hct, MCV, 혈소판", [["일반혈액검사", "cbc", "전혈구 검사"]], False),
    ("bmp", "tests", "전해질|electrolyte|나트륨|칼륨|sodium|potassium|bun|크레아티닌|creatinine|신기능|신장 기능|신장기능|"
     "콩팥 기능|renal function|혈당|glucose|bmp|기초 대사|생화학|chemistry",
     "기초 화학: Na, K, Cl, HCO3, BUN, Cr, 혈당, Ca",
     [["전해질 검사", "전해질검사", "electrolytes"], ["신기능 검사", "신장기능검사", "크레아티닌", "bun/cr"], ["혈당 검사", "glucose"]], False),
    ("lft", "tests", "간기능|간 기능|간수치|간 수치|간효소|lft|liver function|ast/alt|빌리루빈|bilirubin|알부민",
     "간기능: AST, ALT, ALP, GGT, 총빌리루빈, 알부민", [["간기능 검사", "간기능검사", "lft", "liver function"]], False),
    ("crp_esr", "tests", "crp|c-반응|c반응|염증 수치|염증수치|염증 표지|esr|적혈구침강|적혈구 침강|혈침|inflammatory marker",
     "CRP, ESR", [["crp 검사", "염증 수치", "esr"]], False),
    ("ua", "tests", "소변검사|소변 검사|요검사|소변 분석|요분석|urinalysis|u/a|소변 현미경",
     "소변검사: 비중, pH, 단백, 당, 케톤, 잠혈, 백혈구 에스테라제, 아질산염, 현미경(RBC/WBC)", [["소변검사", "urinalysis", "요검사"]], False),
    ("cxr", "tests", "흉부 x선|흉부x선|흉부 x-ray|흉부 엑스레이|흉부 사진|흉부 방사선|흉부 단순|가슴 x선|가슴 엑스레이|chest x-ray|chest xray|cxr",
     "흉부 X선 판독", [["흉부 x선", "chest x-ray", "흉부 x-ray"]], False),
    ("ecg", "tests", "심전도|ecg|ekg|electrocardiogram", "12유도 심전도: 리듬, 심박수, 축, 간격, ST-T",
     [["심전도", "ecg", "ekg"]], False),
    ("brain_ct", "tests", "뇌 ct|뇌ct|두부 ct|두부ct|머리 ct|head ct|brain ct|뇌 단층",
     "뇌 CT (두통, 신경학적 증상, 의식 변화, 실신, 경련, 어지럼, 두부 외상 등에서)", [["뇌 ct", "두부 ct", "brain ct"]], True),
    ("abd_us", "tests", "복부 초음파|복부초음파|abdominal ultrasound|abdominal us|간 초음파|배 초음파|상복부 초음파",
     "복부 초음파 (복통, 간담도·신장 문제, 황달, 복부 종괴 등에서)", [["복부 초음파", "abdominal ultrasound"]], True),
    ("abd_ct", "tests", "복부 ct|복부ct|복부-골반 ct|복부 골반 ct|abdominal ct|abdomen ct|abdominopelvic ct|배 ct",
     "복부·골반 CT (복통, 복부 종괴, 위장관 출혈, 원인 불명 발열·체중감소 등에서)", [["복부 ct", "abdominal ct", "복부 골반 ct"]], True),
    ("pregnancy", "tests", "임신 검사|임신검사|임신 반응|hcg|pregnancy",
     "임신 검사 (소변/혈청 β-hCG; 가임기 여성이면 작성)", [["임신 검사", "hcg", "pregnancy test"]], True),
    ("troponin", "tests", "트로포닌|troponin|심근효소|심근 효소|심장효소|심장 효소|ck-mb|cardiac enzyme",
     "트로포닌/심근효소 (흉통, 호흡곤란, 실신, 심계항진, 상복부 통증 등에서)", [["트로포닌", "troponin", "심근효소"]], True),
    ("ddimer", "tests", "d-dimer|d dimer|ddimer|디다이머|d-다이머",
     "D-dimer (호흡곤란, 흉통, 하지 부종, 실신 등 혈전색전증 감별 시)", [["d-dimer", "디다이머"]], True),
    ("blood_culture", "tests", "혈액 배양|혈액배양|blood culture|혈액 균배양",
     "혈액 배양 (발열, 패혈증 의심, 감염 감별 시)", [["혈액 배양", "blood culture"]], True),
    ("lipase", "tests", "리파아제|리파제|lipase|아밀라아제|아밀라제|amylase|췌장 효소",
     "리파아제/아밀라아제 (상복부 통증, 구토, 복통 등에서)", [["리파아제", "lipase", "아밀라아제"]], True),
]
# search terms that must not stand alone in a new key (they would match unrelated requests)
_GENERIC_TOKENS = {"검사", "진찰", "신체진찰", "신체 진찰", "test", "tests", "exam", "결과", "소견", "평가", "확인", "혈액", "blood",
                   "lab", "labs", "수치", "level", "기능", "function", "ct", "씨티", "mri", "x-ray", "xray", "x선", "엑스레이",
                   "초음파", "ultrasound", "영상", "영상검사", "imaging", "촬영", "scan", "배양", "culture", "배양 검사", "항체",
                   "antibody", "단층", "방사선", "physical", "눈", "귀", "코", "입", "목", "배", "몸", "청진", "촉진", "타진", "시진",
                   "징후", "sign", "signs", "진찰 소견", "검사 결과"}

FULL_SYSTEM_PROMPT = """당신은 의학 교육용 진단 연습 증례를 보강하는 도구입니다.
증례는 시험 문제에서 변환되어 정답과 관련된 소견만 들어 있습니다. 실제 진료 기록이라면 기본 진찰과 기본 검사 결과가 (대부분 정상이거나 흔한 우연 소견과 함께) 있었을 것입니다.
다음 두 가지를 채우세요.

A. 기본 패널: [채울 기본 항목]에 나열된 각 항목의 결과.
B. 확진·감별 검사: 이 주호소에서 의사가 요청할 확진 검사, 주요 감별 진단을 배제하는 검사, 필요한 특수 진찰 중 증례와 기본 패널에 아직 없는 것.

규칙:
1. 정답 진단과 기존 소견에 모순이 없어야 합니다. 정답 질환이 실제로 만드는 이상 소견은 기본 패널에도 반영하세요 (예: 빈혈을 일으키는 질환이면 CBC에 Hb 저하, 폐 질환이면 폐 청진·흉부 X선 이상). 관련 없는 계통은 정상으로 쓰되, 나이·기저질환에 맞는 흔한 경미한 우연 소견은 가끔 넣어도 됩니다.
2. 기존 증례의 수치·소견을 바꾸지 마세요. 기본 항목 결과에 기존 증례와 겹치는 부분(예: 초기 정보·병력의 활력징후, 기존 진찰의 일부 소견)이 있으면 기존 값과 똑같이 맞추세요.
3. 현실적인 수치에 단위를 붙이세요 (예: "Hb 13.8 g/dL"). 판독문은 보고서처럼 소견만 간결하게 쓰세요 (항목당 1~3문장, 300자 이하). "기존 소견과 같이" 같은 증례 자체에 대한 언급은 쓰지 마세요.
4. 결과와 키에 정답 진단명이나 그 동의어를 절대 쓰지 마세요. "~에 합당", "~ 의심", "~ 시사" 같은 진단적 해석도 쓰지 마세요. 관찰된 소견만 쓰세요.
5. [조건부] 항목은 이 환자의 나이·성별·주호소에서 의사가 실제로 시행할 법할 때만 결과를 쓰고, 아니면 null로 두세요. 임신 검사는 가임기 여성(약 12~50세)이면 결과를 쓰고, 남성·소아·폐경 여성이면 null입니다.
6. B 항목은 진찰 4개 이하, 검사 8개 이하입니다. 키는 "|"로 구분한 검색어로, 한국어 표현(띄어쓴 형태와 붙여쓴 형태)과 영어 소문자 약어를 3~8개 넣으세요.
   예: "갑상선 기능 검사|갑상선기능|tft|tsh|free t4", "경추 mri|경추mri|c-spine mri|cervical mri".
   "검사", "ct", "mri", "초음파", "x선", "혈액", "배양", "항체", "청진", "촉진", "징후" 같은 일반어를 단독 검색어로 쓰지 말고 부위나 검사명을 붙이세요 (예: "장음 청진"). 기존 키나 기본 패널과 같은 검사는 만들지 마세요.
   "exam"에는 의사가 직접 하는 진찰·수기(예: 직장수지검사, 특수 징후, 안저 검사)를, "tests"에는 검체·영상·생리 검사를 넣으세요.

출력: JSON 객체 하나만 (설명 금지).
{"panel": {"<항목 id>": "<결과>" 또는 null, ...}, "exam": {"<키>": "<소견>", ...}, "tests": {"<키>": "<결과>", ...}}"""


def _toks(key: str) -> list[str]:
    return [t.strip().lower() for t in key.split("|") if t.strip()]


# the matched existing entry must also contain these results, otherwise a broad key like "혈액검사" whose value only
# holds liver tests would wrongly count as a CBC
_EVIDENCE = {
    "vitals": r"mmhg|/분|℃|°c|bpm|혈압\s*\d",
    "cbc": r"wbc|백혈구|혈색소|헤모글로빈|hemoglobin|hb\s*\d|혈소판|platelet|plt",
    "bmp": r"\bna\b|나트륨|sodium|\bk\b|칼륨|potassium|크레아티닌|creatinine|\bcr\b|\bbun\b|혈당|glucose",
    "lft": r"\bast\b|\balt\b|빌리루빈|bilirubin|\balp\b",
    "crp_esr": r"crp|esr|침강",
    "ua": r"비중|잠혈|요단백|단백뇨|아질산|urine|소변|요검사|/hpf",
}


def _covered(case: dict, section: str, probe_groups: list[list[str]], pid: str = "") -> bool:
    """Would the keyword simulator already answer this panel item from an existing key? (CaseFileEnvironment logic)"""
    ev = _EVIDENCE.get(pid)
    entries = [(_toks(k), v) for k, v in case.get(section, {}).items()
               if not ev or re.search(ev, str(v), re.I)]
    toks = [t for ts, _v in entries for t in ts if t not in _GENERIC_TOKENS]
    return all(any(t in probe.lower() for probe in group for t in toks) for group in probe_groups)


def _clean_key(key: str, taken: set[str], curated: bool = False) -> str:
    """Drops search terms that duplicate existing ones, are too generic, or are short ASCII fragments ('pt' ⊂ 'sept')."""
    out = []
    for t in _toks(key):
        if t in taken or t in out:
            continue
        if not curated and (t in _GENERIC_TOKENS or (t.isascii() and len(t) < 3) or len(t) < 2):
            continue
        out.append(t)
    return "|".join(out)


def _leak_names(case: dict) -> list[str]:
    """Normalized diagnosis names + distinctive tokens that must not appear in added text.

    Stricter than validate(): 2-character Korean names (폐렴, 통풍) are checked too. Distinctive tokens already present in
    the visible case text (e.g. 'b12' when the case already reports a B12 level) are not new leaks and are allowed."""
    raw = [case["diagnosis"], *case.get("aliases", [])]
    visible = _norm(json.dumps({k: case.get(k) for k in ("initial", "history", "exam", "tests")}, ensure_ascii=False))
    full = {_norm(re.sub(r"\(.*?\)", "", n)) for n in raw}
    full = {n for n in full if len(n) >= 3 or (len(n) == 2 and not n.isascii())}
    toks = {_norm(t) for t in _distinctive_tokens(raw)} - {""}
    return sorted(full | {t for t in toks if t not in visible})


def _text_errors(names: list[str], text: str, where: str) -> list[str]:
    n, low = _norm(text), text.lower()

    def found(x: str) -> bool:  # short ASCII names (rds, hsv) only as whole words, so 'ballard score' is not 'rds'
        if x.isascii() and len(x) <= 4:
            return re.search(rf"(?<![a-z0-9]){re.escape(x)}(?![a-z0-9])", low) is not None
        return x in n

    hit = next((x for x in names if found(x)), None)
    return [f"{where}: names the diagnosis ('{hit}') — describe findings only"] if hit else []


_META_WORDS = re.compile(r"기존\s*(진찰|소견|증례|검사|결과|기록)|증례에\s*(있|기술)|정답|진단명")


def _meta_errors(text: str, where: str) -> list[str]:
    """Results must read like a chart entry, not refer to the case file or the answer."""
    m = _META_WORDS.search(text)
    return [f"{where}: do not refer to '{m.group(0)}' — write the finding itself"] if m else []


def panel_todo(case: dict) -> list[tuple]:
    return [p for p in PANEL if not _covered(case, p[1], p[4], p[0])]


def build_full_user(case: dict, todo: list[tuple]) -> str:
    visible = {k: case[k] for k in ("initial", "history", "exam", "tests") if k in case}
    done = [f"{p[0]} ({p[3].split(':')[0].split(' (')[0]})" for p in PANEL if p not in todo]
    lines = [f"- {p[0]} ({'진찰' if p[1] == 'exam' else '검사'}){' [조건부]' if p[5] else ''}: {p[3]}" for p in todo]
    return (f"[정답 진단] {case['diagnosis']}\n[동의어] {', '.join(case.get('aliases', []))}\n\n"
            f"[현재 증례]\n{json.dumps(visible, ensure_ascii=False, indent=1)}\n\n"
            f"[이미 증례에 있는 기본 항목 — 만들지 말 것] {', '.join(done) or '없음'}\n\n"
            f"[채울 기본 항목]\n" + "\n".join(lines) + "\n\nJSON으로 출력하세요.")


def check_full(case: dict, todo: list[tuple], add: dict) -> tuple[list[str], list[tuple], list[str]]:
    """Returns (errors to send back, accepted entries [(section, key, value, panel_id|None)], dropped notes)."""
    names = _leak_names(case)
    errors, entries, dropped = [], [], []
    taken = {s: {t for k in case.get(s, {}) for t in _toks(k)} for s in ("exam", "tests")}
    panel = add.get("panel")
    if not isinstance(panel, dict):
        errors.append("'panel' object missing")
        panel = {}
    for pid, sec, key, _desc, _probes, cond in todo:
        val = panel.get(pid)
        if val is None or (isinstance(val, str) and val.strip().lower() in ("", "null", "none")):
            if not cond:
                errors.append(f"panel.{pid}: result missing (required)")
            continue
        if not isinstance(val, str):
            errors.append(f"panel.{pid}: must be a string")
            continue
        errs = _text_errors(names, val, f"panel.{pid}") + ([f"panel.{pid}: too long"] if len(val) > 500 else []) \
            + _meta_errors(val, f"panel.{pid}")
        if errs:
            errors += errs
            continue
        k = _clean_key(key, taken[sec], curated=True)
        if not k:
            dropped.append(f"panel.{pid}: all search terms already used")
            continue
        entries.append((sec, k, val.strip(), pid))
        taken[sec] |= set(_toks(k))
    for sec in ("exam", "tests"):
        extra = add.get(sec) or {}
        if not isinstance(extra, dict):
            errors.append(f"'{sec}' must be an object")
            continue
        for i, (key, val) in enumerate(extra.items()):
            if i >= FULL_EXTRA_LIMIT[sec]:
                dropped.append(f"{sec}: over limit, dropped '{key}'")
                continue
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{sec} '{key}': empty value")
                continue
            errs = _text_errors(names, key + " " + val, f"{sec} '{key}'") + ([f"{sec} '{key}': too long"] if len(val) > 500 else []) \
                + _meta_errors(val, f"{sec} '{key}'")
            if errs:
                errors += errs
                continue
            k = _clean_key(key, taken[sec])
            if not k:
                dropped.append(f"{sec}: '{key}' duplicates existing/panel search terms or is too generic")
                continue
            entries.append((sec, k, val.strip(), None))
            taken[sec] |= set(_toks(k))
    return errors, entries, dropped


def _chat_with_backoff(client: OpenAICompatClient, messages: list[dict]) -> str:
    """The client already retries twice; on a persistent 429 wait longer (other runs share the quota)."""
    for k in range(12):
        try:
            return client.chat(messages)
        except RuntimeError as e:
            limited = re.search(r"429|RESOURCE_EXHAUSTED|quota|rate.?limit", str(e), re.I)
            if k >= (11 if limited else 2):
                raise
            time.sleep(min(60 * (k + 1), 300) if limited else 10)
    raise RuntimeError("unreachable")


def augment_full_one(path: Path, out_dir: Path, set_name: str, client: OpenAICompatClient) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    todo = panel_todo(case)
    log = {"set": set_name, "id": path.stem, "attempts": 0, "errors": [], "panel_todo": [p[0] for p in todo]}
    messages = [{"role": "system", "content": FULL_SYSTEM_PROMPT}, {"role": "user", "content": build_full_user(case, todo)}]
    try:
        for _ in range(3):
            log["attempts"] += 1
            objs = [o for o in _json_objects(_chat_with_backoff(client, messages)) if "panel" in o]
            add = objs[-1] if objs else {}
            errors, entries, dropped = check_full(case, todo, add)
            if not errors:
                break
            log["errors"].append(errors)
            messages += [{"role": "assistant", "content": json.dumps(add, ensure_ascii=False)},
                         {"role": "user", "content": "다음 문제를 고쳐 JSON 전체를 다시 출력하세요: " + "; ".join(errors)}]
    except Exception as e:  # noqa: BLE001 — one failing case must not stop the batch
        log.update(status="error", error=str(e)[:300])
        return log
    log["dropped"] = dropped
    if not entries:
        log["status"] = "skipped"
        return log
    log["status"] = "partial" if errors else "ok"

    new = {"exam": {}, "tests": {}}
    for sec, key, val, _pid in entries:
        if key not in case.get(sec, {}):
            new[sec][key] = val
    out = dict(case)
    out["exam"] = {**case.get("exam", {}), **new["exam"]}
    out["tests"] = {**case.get("tests", {}), **new["tests"]}
    prev = case.get("augmented") or {}
    out["augmented"] = {"exam": [*prev.get("exam", []), *new["exam"]], "tests": [*prev.get("tests", []), *new["tests"]]}
    out["augmented_full"] = {"panel": {pid: key for sec, key, _v, pid in entries if pid},
                             "extra": [key for sec, key, _v, pid in entries if not pid],
                             "script": "scripts/augment_cases.py --mode full", "date": dt.date.today().isoformat()}
    out["_note"] = (case.get("_note", "") + " | Entries listed in 'augmented' were generated by an LLM "
                    "(scripts/augment_cases.py --mode full: routine panel + decisive/differential tests), not in the "
                    "source item, not clinician-reviewed.").strip(" |")
    (out_dir / path.name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.update(added_exam=len(new["exam"]), added_tests=len(new["tests"]),
               added_panel=sum(1 for e in entries if e[3]), added_extra=sum(1 for e in entries if not e[3]))
    return log


def _match_id(path: Path, ids: set[str]) -> bool:
    return path.stem in ids or path.stem.split("_", 1)[-1] in ids


def main_full(args) -> None:
    load_dotenv(ROOT / ".env")
    cfg = LLMConfig.from_env("AUGMENT_LLM")
    cfg.temperature = 0.0
    cfg.max_tokens = max(cfg.max_tokens, 8192)
    cfg.timeout_s = max(cfg.timeout_s, 180.0)
    client = OpenAICompatClient(cfg)

    if args.in_dir:
        if not args.out_dir:
            sys.exit("--out is required with --in")
        src = (ROOT / args.in_dir).resolve()
        name = next((n for n, d in FULL_SETS.items() if d.resolve() == src), src.name)
        pairs = [(name, src, Path(args.out_dir))]
    else:
        pairs = [(name, src, FULL_ROOT / name) for name, src in FULL_SETS.items()]
    jobs = []
    for name, src, dst in pairs:
        src, dst = (p if p.is_absolute() else ROOT / p for p in (src, dst))
        dst.mkdir(parents=True, exist_ok=True)
        paths = sorted(src.glob("*.json"))
        if args.ids:
            paths = [p for p in paths if _match_id(p, set(args.ids))]
        if not args.overwrite:
            paths = [p for p in paths if not (dst / p.name).exists()]
        jobs += [(p, dst, name) for p in paths]
    print(f"{len(jobs)} cases, model {cfg.model}, workers {min(args.workers, FULL_MAX_WORKERS)}", flush=True)

    def run(job):
        log = augment_full_one(job[0], job[1], job[2], client)
        print(log["set"], log["id"], log["status"], f"+exam {log.get('added_exam', 0)} +tests {log.get('added_tests', 0)}",
              (log["errors"][-1] if log["errors"] and log["status"] != "ok" else "") or log.get("error", ""), flush=True)
        return log

    with ThreadPoolExecutor(min(args.workers, FULL_MAX_WORKERS)) as ex:
        logs = list(ex.map(run, jobs))

    meta = json.loads(FULL_META.read_text(encoding="utf-8")) if FULL_META.exists() else {"runs": []}
    meta["runs"].append({"date": dt.date.today().isoformat(), "model": cfg.model, "temperature": cfg.temperature,
                         "reasoning_effort": cfg.reasoning_effort, "system_prompt": FULL_SYSTEM_PROMPT,
                         "user_template": "see build_full_user() in scripts/augment_cases.py",
                         "panel": [{"id": p[0], "section": p[1], "key": p[2], "conditional": p[5]} for p in PANEL],
                         "items": logs})
    FULL_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    by_set: dict[str, dict] = {}
    for log in logs:
        by_set.setdefault(log["set"], {}).setdefault(log["status"], 0)
        by_set[log["set"]][log["status"]] += 1
    print(json.dumps(by_set, ensure_ascii=False))


# common doctor requests used as a proxy for "would the environment answer this?" (keyword simulator)
MEASURE_REQUESTS = [
    ("EXAM", r) for r in ["활력징후 측정", "혈압 측정", "체온 측정", "전신 상태 관찰", "두경부 진찰", "인두 진찰", "결막 확인",
                          "경부 림프절 촉진", "갑상선 촉진", "심장 청진", "폐 청진", "흉부 진찰", "복부 진찰", "하지 부종 확인",
                          "피부 진찰", "신경학적 진찰", "의식 상태 평가", "뇌신경 검사", "근력 검사", "감각 검사",
                          "심부건반사", "보행 관찰"]
] + [
    ("TEST", r) for r in ["일반혈액검사", "CBC", "전해질 검사", "신기능 검사 (BUN/Cr)", "혈당 검사", "간기능 검사", "CRP",
                          "ESR", "소변검사", "흉부 X선", "심전도", "뇌 CT", "복부 초음파", "복부 CT", "임신 검사",
                          "트로포닌", "D-dimer", "혈액 배양", "리파아제", "갑상선 기능 검사", "응고 검사 (PT/INR)",
                          "동맥혈 가스 분석"]
]


def measure() -> None:
    from doctor_agent.env.interface import Action, ActionType
    from eval.simulator import CaseFileEnvironment

    miss = {"이 진찰 결과는 제공되지 않습니다.", "이 검사 결과는 제공되지 않습니다."}

    def rates(paths: list[Path]) -> tuple[float, dict[str, float]]:
        per = {r: 0 for _, r in MEASURE_REQUESTS}
        for p in paths:
            env = CaseFileEnvironment.from_file(p)
            for t, r in MEASURE_REQUESTS:
                per[r] += env.step(Action(ActionType(t), r)).text not in miss
        n = max(len(paths), 1)
        return sum(per.values()) / (n * len(MEASURE_REQUESTS)), {r: v / n for r, v in per.items()}

    rows = {}
    for name, src in FULL_SETS.items():
        before = sorted(src.glob("*.json"))
        after = [FULL_ROOT / name / p.name if (FULL_ROOT / name / p.name).exists() else p for p in before]
        (b, pb), (a, pa) = rates(before), rates(after)
        rows[name] = (len(before), sum(1 for p in after if FULL_ROOT in p.parents), b, a, pb, pa)
        print(f"{name:15s} n={len(before):3d} augmented={rows[name][1]:3d}  match {b:6.1%} → {a:6.1%}")
    print("\nper request (all sets pooled, before → after):")
    n_tot = sum(r[0] for r in rows.values())
    for _, r in MEASURE_REQUESTS:
        b = sum(v[4][r] * v[0] for v in rows.values()) / n_tot
        a = sum(v[5][r] * v[0] for v in rows.values()) / n_tot
        print(f"  {r:22s} {b:6.1%} → {a:6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decisive", "full", "measure"], default="decisive",
                    help="decisive: original mode (clinicalqa → cases_clinicalqa_aug); full: routine panel + decisive; "
                         "measure: keyword-match rate of common requests before/after full augmentation")
    ap.add_argument("--in", dest="in_dir", help="full mode: input case dir (default: all sets in FULL_SETS)")
    ap.add_argument("--out", dest="out_dir", help="full mode: output case dir (required with --in)")
    ap.add_argument("--ids", nargs="*", help="case ids, e.g. 181 521 (default: all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "measure":
        measure()
        return
    if args.mode == "full":
        main_full(args)
        return

    load_dotenv(ROOT / ".env")
    cfg = LLMConfig.from_env("AUGMENT_LLM")
    cfg.temperature = 0.0
    client = OpenAICompatClient(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(IN_DIR.glob("cqa_*.json"))
    if args.ids:
        paths = [p for p in paths if p.stem.removeprefix("cqa_") in set(args.ids)]
    if not args.overwrite:
        paths = [p for p in paths if not (OUT_DIR / p.name).exists()]

    with ThreadPoolExecutor(args.workers) as ex:
        logs = list(ex.map(lambda p: augment_one(p, client), paths))
    for log in logs:
        print(log["id"], log["status"], f"+exam {log.get('added_exam', 0)} +tests {log.get('added_tests', 0)}",
              log["errors"][-1] if log["status"] != "ok" and log["errors"] else "")

    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {"runs": []}
    meta["runs"].append({"date": dt.date.today().isoformat(), "model": cfg.model, "temperature": cfg.temperature,
                         "reasoning_effort": cfg.reasoning_effort, "system_prompt": SYSTEM_PROMPT,
                         "user_template": build_user.__doc__ or "see build_user() in scripts/augment_cases.py",
                         "items": logs})
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
