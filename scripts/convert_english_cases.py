"""Convert English OSCE-style / case-report datasets into Korean interactive cases (offline, dev only).

Sources (raw downloads live in data/external/, git-ignored, never packaged):
  agentclinic     AgentClinic-MedQA(-Extended), MIT, https://github.com/SamuelSchmidgall/AgentClinic
                  file agentclinic_medqa_extended.jsonl (lines 0-106 are exactly agentclinic_medqa.jsonl)
  diagnosisarena  DiagnosisArena, MIT, https://huggingface.co/datasets/shzyk/DiagnosisArena

    source .venv/bin/activate
    python scripts/convert_english_cases.py agentclinic                    # AgentClinic-MedQA base set (ids 0-106)
    python scripts/convert_english_cases.py agentclinic --set extended     # the 107 cases added in _extended
    python scripts/convert_english_cases.py diagnosisarena                 # seeded subset (see select_da)
    python scripts/convert_english_cases.py diagnosisarena --ids 1 42 --overwrite

Translation is done by the LLM in CONVERT_LLM_* (.env; falls back to LLM_*), only facts present in the source.
The English original is never copied into the case; the case keeps only source {dataset, id, license, url}.
Output: data/cases_agentclinic/ac_<id>.json, data/cases_diagnosisarena/da_<id>.json.
Reproducibility record: data/labels/<source>_conversion_meta.json. Needs pyarrow for DiagnosisArena.
"""
import argparse
import datetime as dt
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from convert_clinicalqa import norm, numbers, parse_json  # noqa: E402
from doctor_agent.config import LLMConfig  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402
from eval.run_local import load_dotenv  # noqa: E402

EXT = ROOT / "data/external"

SOURCES = {
    "agentclinic": {
        "key": "agentclinic",
        "dataset": "AgentClinic-MedQA",
        "license": "MIT",
        "url": "https://github.com/SamuelSchmidgall/AgentClinic",
        "revision": "b6570edefb940857a7c334350656b29f9d984f24",
        "raw": EXT / "agentclinic/agentclinic_medqa_extended.jsonl",
        "out": ROOT / "data/cases_agentclinic",
        "prefix": "ac",
        "meta": ROOT / "data/labels/agentclinic_conversion_meta.json",
    },
    "diagnosisarena": {
        "key": "diagnosisarena",
        "dataset": "DiagnosisArena",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/shzyk/DiagnosisArena",
        "revision": "7163704e580ba643647483307f64d426eef33ef0",
        "raw": EXT / "diagnosisarena/test.parquet",
        "out": ROOT / "data/cases_diagnosisarena",
        "prefix": "da",
        "meta": ROOT / "data/labels/diagnosisarena_conversion_meta.json",
    },
}

# Chief-complaint vocabulary of snuh/ClinicalQA (so categories line up across case sets).
CATEGORIES = [
    "가려움증", "가슴통증/가슴불쾌감", "객혈", "거품 소변", "고혈압", "골절/탈구", "관절통/관절부기", "구역/구토",
    "급성 복통", "기분장애", "기억력 저하", "기침", "난청", "다뇨", "두근거림", "두드러기/혈관부종", "두통",
    "만성복통/소화불량/속쓰림", "목덩이", "무뇨증/핍뇨증", "무월경", "물질남용", "미숙아/저체중 출생아", "발달지연",
    "발열", "발작/간질", "배뇨곤란/배뇨통", "배벽과 서혜부 덩이/탈장", "변비", "복부덩이/골반덩이", "복부팽만/복수",
    "부종", "분만/산후 관리", "불임", "산전관리", "삼킴곤란", "설사", "성매개감염", "시력장애", "실신", "심박이상",
    "알레르기반응", "어지럼/현기증", "외상", "요실금", "월경통", "유방덩이/통증", "의식변화/혼수", "저혈압/쇼크",
    "질분비물", "질출혈", "청색증", "체중감소/식욕부진", "체중증가/비만", "출혈경향", "토혈", "팔다리 근력약화/마비",
    "피로", "피부발진", "허리통증", "혈뇨", "혈변/흑색변", "호흡곤란", "황달",
]

SYSTEM_PROMPT = """당신은 영어 의학 증례를 한국어 대화형 진단 연습 증례(JSON)로 옮기는 번역·변환기입니다.
원문에 적힌 사실만 한국어로 옮기세요. 원문에 없는 증상, 병력, 진찰 소견, 검사 결과, 수치, 음성 소견을 절대 만들어 내지 마세요.
요약하거나 빠뜨리지도 마세요. 원문의 모든 병력·진찰·검사 사실이 어딘가에 들어가야 합니다.

출력 형식: 아래 키를 가진 JSON 객체 하나만 출력하세요 (코드블록, 설명 문장 금지).
{
  "initial": "<나이>세 <남성|여성>. 주호소: <주호소와 기간>",
  "category": "<주호소 분류: 아래 목록 중 하나>",
  "history": {"<키워드1|키워드2|english>": "<환자 말투 답변>", ...},
  "exam": {"<키워드|english>": "<진찰 소견>", ...},
  "tests": {"<키워드|english>": "<검사 결과>", ...},
  "diagnosis": "<정답 진단명의 표준 한국어 의학 용어>",
  "aliases": ["<원문 영어 정답 진단명 그대로>", "<한국어 동의어>", "<영어 동의어·약어>", ...],
  "difficulty": "<쉬움|보통|어려움>",
  "teaching_point": "<한국어 1~2문장>"
}

규칙:
1. initial: 나이, 성별, 주호소(원문에 기간이 있으면 기간 포함)만 쓰세요. 다른 증상, 병력, 소견, 진단 힌트는 넣지 마세요.
   나이가 "early 70s"처럼 범위면 "70대 초반 여성"처럼, 신생아·영아면 "생후 3일 남아"처럼 원문 그대로 옮기세요.
   형식 예: "35세 여성. 주호소: 1개월 전부터 시작된 물체가 둘로 보이는 증상".
2. category: 다음 목록에서 주호소에 가장 맞는 것 하나를 고르세요: {categories}
   맞는 것이 없을 때만 짧은 한국어 주호소를 새로 쓰세요.
3. history: 원문의 병력 정보(발병 양상, 동반 증상, 과거력, 약물, 가족력, 사회력, 계통 문진 등)를 주제별 항목으로 나누세요.
   - 값은 환자 본인이 말하는 쉬운 한국어 구어체(1~2문장)로 쓰세요. 환자가 영유아·소아이거나 의식이 없으면 보호자가 말하는 말투로 쓰세요
     ("아이가 ~해요"). 의학 용어나 진단명은 쓰지 마세요. 과거에 진단받은 병은 환자가 들은 이름으로 짧게 말해도 됩니다.
   - 이전에 받은 검사 수치(LDL, HbA1c 등)나 심전도·영상·조직검사 결과는 history에 넣지 말고 tests로 옮기세요.
   - 원문에 명시된 음성 정보("denies ...", "no ...")만 음성으로 쓰세요. 원문에 없는 음성 정보를 추가하지 마세요.
   - 원문에 없는 세부 묘사(통증 강도, 시각, 상황, 감정 등)나 원문에 없는 인과관계("~하려고", "~때문에")를 덧붙이지 마세요.
4. exam: 원문의 신체진찰 소견과 활력징후를 부위/종류별로 나누세요. 의학 용어로, 원문 수치 그대로 쓰세요.
   영어 의학 용어는 표준 한국어 의학 용어로 옮기세요(예: tibia → 경골, ovary → 난소, lymphocyte → 림프구). 영어 음차는 쓰지 마세요.
   심전도, 영상, 혈액·소변 검사, 조직검사, 내시경, 피부경처럼 장비나 검체가 필요한 것은 tests에 넣으세요.
5. tests: 원문의 검사 결과를 검사 종류별로 나누세요. 수치, 단위, 참고치(정상 범위)는 원문 그대로 모두 옮기고 빠뜨리지 마세요.
   원문이 "normal"이라고 한 검사는 "정상"으로만 쓰세요.
6. 숫자 (initial, history, exam, tests 모두): 모든 숫자는 원문과 같은 아라비아 숫자와 단위로 쓰세요. 단위를 바꾸거나(°F→°C, mg/dL→mmol/L), 계산하거나,
   반올림하거나, 한국식 단위(만, 천)로 바꾸지 마세요. 환자 말투에서도 환산값을 덧붙이지 마세요
   (예: "11 pound" → "11파운드"로만 쓰고 "약 5kg"을 붙이지 않음). "12,000/mm^3"은 "12,000/mm^3" 그대로 씁니다.
   원문이 숫자를 영어 단어로 썼으면("two days") 같은 값의 아라비아 숫자로 써도 됩니다("2일").
   원문에 단위가 없는 수치에는 단위를 붙이지 마세요(예: "body mass index was 26" → "체질량지수 26").
   원문 단위가 틀려 보여도 고치지 말고 그대로 쓰세요. 원문이 참고치(reference range, normal)라고 밝힌 범위만 "참고치"라고 쓰고,
   여러 번 잰 값의 범위(예: 표의 "131-137")는 "측정값 131-137"처럼 원문 의미 그대로 쓰세요.
7. 키: "|"로 구분한 검색 키워드 3~8개 (한국어 일상어, 의학 용어, 영어 약어를 섞어서). 의사가 그 항목을 물을 때 쓸 법한 단어로 하세요.
   키에 정답 진단명이나 그 약어를 넣지 마세요.
8. 원문의 "Objective", 정답 진단, 치료·경과 정보는 증례 본문에 넣지 마세요.
   원문 검사 결과 자체가 진단명을 말하는 경우(예: 조직검사 "consistent with ...")에만 그 문장을 그대로 번역해 넣으세요.
9. diagnosis: 원문 정답 진단을 표준 한국어 의학 용어로 번역하세요(대한의사협회 용어 수준, 뜻을 넓히거나 좁히지 말 것).
   aliases의 첫 항목은 원문 영어 정답 진단명을 그대로 쓰고, 이어서 한국어 동의어·다른 표기와 영어 동의어, 흔한 약어를 쓰세요.
10. difficulty: 문진·진찰·검사를 순서대로 해 나가는 대화형 진단 기준으로 매기세요.
   "쉬움": 흔한 질환의 전형적 양상이고, 결정적 검사 한 가지로 바로 확진된다.
   "보통": 병력·진찰·검사 소견을 여러 개 종합해야 하거나, 결정적 단서가 특정 질문(과거력, 약물, 가족력, 노출력)으로만 드러난다.
   "어려움": 드문 질환이거나, 비전형적 양상이거나, 다른 진단을 시사하는 혼란 소견이 있다.
11. teaching_point: 이 증례에서 어떤 단서로 어떤 감별진단과 구별해 진단하는지 한국어 1~2문장.
""".replace("{categories}", ", ".join(CATEGORIES))

USER_TEMPLATE = """[원문 증례 — {dataset} id={id}]

{body}

정답 진단 (diagnosis/aliases 작성용, 본문에 넣지 말 것): {answer}
"""

# Reviewer fixes (eval-simulator, 2026-09-26): mistranslated Korean terms found by reviewing every
# (source answer, Korean diagnosis) pair and by the hand spot-check. Applied on write and by --apply-overrides (no LLM
# call). "fields" replaces top-level fields; "text" does literal (old, new) replacements inside history/exam/tests
# values. The LLM's own diagnosis is kept in the meta log as diagnosis_ko.
OVERRIDES: dict[str, dict[int, dict]] = {
    "agentclinic": {
        3: {"fields": {"diagnosis": "미만성 거대 B세포 림프종"}},        # was "거대 B세포 심층 유포성 림프종"
        5: {"text": [("다낭성 오반 증후군", "다낭성 난소 증후군")]},
        17: {"fields": {"diagnosis": "괴저농피증"}},                     # pyoderma gangrenosum, was "괴사성 화농여드름"
        53: {"fields": {"category": "어지럼/현기증"}},                   # typo "어지점/현기증"
        60: {"text": [("다낭성 오바리 증후군(다낭성 난소 증후군)", "다낭성 난소 증후군")]},
        97: {"fields": {"diagnosis": "심상성 천포창"}},                  # pemphigus vulgaris, was "보통천식창"
        # stray Chinese characters (hanja) in the LLM output
        6: {"text": [("喘명음", "천명음")]},
        16: {"text": [("鼓音", "고음")]},
        29: {"text": [("播散性 淋菌 感染", "")]},
        87: {"text": [("쿠姆斯검사", "쿰스검사")]},
        89: {"text": [("遊離 T4", "유리 T4")]},
    },
    "diagnosisarena": {
        88: {"text": [("성별 미상 성인 남아/남성.", "성인 남성.")]},      # source: "A man"
        122: {"fields": {"diagnosis": "미주신경 매개성 발작성 방실 차단"}},  # was "미경골 신경 매개성 ..."
        138: {"fields": {"diagnosis": "할로포 연속성 말단피부염"}, "text": [("국소 국소 ", "국소 ")]},
        264: {"fields": {"diagnosis": "마요키 환상 모세혈관확장성 자반"}},  # was "마요키 마디형 혈관확장성 자반증"
        344: {"text": [("연고를 입에 대고 나쁜 후", "국소용 젤로 이를 닦은 후")]},  # source: used a topical gel to brush teeth
        367: {"text": [("皮質(겉질)", "겉질")]},
        573: {"fields": {"diagnosis": "림프모구성 급성기의 만성 골수성 백혈병"}},  # was "림프구성 급반전을 동반한 ..."
        661: {"fields": {"diagnosis": "난관 혈관근섬유모세포종"}},        # angiomyofibroblastoma, was "혈관근지방종"
        707: {"fields": {"diagnosis": "미만성 범세기관지염"}},            # was "미만성 완세기관지염"
        744: {"fields": {"diagnosis": "아교섬유산성단백질(GFAP) 성상세포병증"}},  # was "... 난소세포병증"
        797: {"fields": {"diagnosis": "파종성 마이코박테리움 콜롬비엔세 감염"}},  # was hanja "播種性"
        838: {"text": [(",嗅覺喪失", ", 후각 상실")]},
    },
}


def apply_fix(case: dict, fix: dict) -> dict:
    case.update(fix.get("fields", {}))
    for old, new in fix.get("text", []):
        case["initial"] = case["initial"].replace(old, new)
        for sec in ("history", "exam", "tests"):
            case[sec] = {k.replace(old, new): v.replace(old, new) for k, v in case[sec].items()}
        case["aliases"] = [a.replace(old, new) for a in case["aliases"] if a.replace(old, new).strip()]
    return case


REQUIRED = {"initial": str, "category": str, "history": dict, "exam": dict, "tests": dict, "diagnosis": str,
            "aliases": list, "difficulty": str, "teaching_point": str}

WORD_NUMS = {
    "once": 1, "one": 1, "single": 1, "twice": 2, "two": 2, "both": 2, "three": 3, "thrice": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


# ---------------------------------------------------------------- loading / selection
def load_agentclinic() -> dict[int, dict]:
    path = SOURCES["agentclinic"]["raw"]
    if not path.exists():
        sys.exit(f"missing {path}; see docs/data-sources.md for the download command")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = {}
    for i, r in enumerate(rows):
        osce = r["OSCE_Examination"]
        body_src = {
            "Patient information and history": osce.get("Patient_Actor", {}),
            "Physical examination": osce.get("Physical_Examination_Findings", {}),
            "Test results": osce.get("Test_Results", {}),
        }
        items[i] = {"id": i, "answer": osce["Correct_Diagnosis"].strip(),
                    "body": json.dumps(body_src, ensure_ascii=False, indent=1),
                    "subset": "medqa" if i < 107 else "medqa_extended"}
    return items


def load_diagnosisarena() -> dict[int, dict]:
    import pyarrow.parquet as pq

    path = SOURCES["diagnosisarena"]["raw"]
    if not path.exists():
        sys.exit(f"missing {path}; see docs/data-sources.md for the download command")
    items = {}
    for r in pq.read_table(path).to_pylist():
        body = (f"Case information:\n{r['Case Information']}\n\nPhysical examination:\n{r['Physical Examination']}"
                f"\n\nDiagnostic tests:\n{r['Diagnostic Tests']}")
        items[r["id"]] = {"id": r["id"], "answer": r["Final Diagnosis"].strip(), "body": body,
                          "right_option": r["Options"][r["Right Option"]]}
    return items


DA_SEED, DA_STRATA, DA_PER_STRATUM = 20260926, 10, 5
DA_COMPOUND = re.compile(r"[;/,:\d]|\b(and|with|due to|caused by|induced|secondary to|associated|setting|stage|grade|type)\b",
                         re.I)
DA_SKIN = re.compile(r"skin|lesion|rash|plaque|papul|nodul|erupt|prurit|itch|scalp|nail", re.I)


def select_da(items: dict[int, dict]) -> list[int]:
    """Seeded, stratified subset of DiagnosisArena (50 cases).

    Eligible: body 700-2200 chars; answer <= 60 chars and a single diagnosis (no digits, lists or compound etiology
    such as "X due to Y" / "stage IV"); the opening 350 chars do not describe a skin lesion (the set is dominated by
    dermatology reports that hinge on images/histology); the answer (minus parentheses) is not already named in the
    source text. Eligible ids are sorted, cut into 10 equal strata (ids are roughly grouped by journal, so this
    spreads specialties) and 5 are drawn per stratum with random.Random(DA_SEED)."""
    ok = []
    for i, it in items.items():
        ans = it["answer"]
        core = norm(re.sub(r"\(.*?\)", "", ans))
        if not (700 <= len(it["body"]) <= 2200) or len(ans) > 60 or DA_COMPOUND.search(ans):
            continue
        if DA_SKIN.search(it["body"][:350]) or (len(core) >= 4 and core in norm(it["body"])):
            continue
        ok.append(i)
    ok.sort()
    rng, sel = random.Random(DA_SEED), []
    for s in range(DA_STRATA):
        sel += rng.sample(ok[s * len(ok) // DA_STRATA:(s + 1) * len(ok) // DA_STRATA], DA_PER_STRATUM)
    return sorted(sel)


# ---------------------------------------------------------------- validation
def src_numbers(text: str) -> set[str]:
    out = numbers(text) | {"1"}
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORD_NUMS:
            out.add(str(WORD_NUMS[w]))
    # "early 70s" / "1990s" → 70 / 1990; "10^9/L" style exponents
    out |= {m for m in re.findall(r"(\d+)s\b", text)}
    return out


UNIT_RE = re.compile(
    r"°\s?[CF]\b|mm\s?Hg|(?:[mµμunpk]?(?:g|mol|eq|Eq|EQ|IU|U|L|Osm)|cells|copies)\s?/\s?"
    r"(?:[mµμdn]?L|mm\^?3|mm³|kg|m\^?2|m²|hr?|min|day|d|hpf|HPF|[mµμ]?l)\b")


def units(s: str) -> set[str]:
    out = set()
    for u in UNIT_RE.findall(s):
        u = re.sub(r"[\s^]", "", u).replace("μ", "µ").replace("³", "3").replace("²", "2").lower()
        out.add(u.replace("/hr", "/h"))
    return out


def answer_variants(ans: str) -> list[str]:
    out = [ans, re.sub(r"\(.*?\)", "", ans)] + re.findall(r"\((.*?)\)", ans)
    return [norm(v) for v in out if norm(v)]


def validate(case: dict, item: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for k, t in REQUIRED.items():
        if not isinstance(case.get(k), t):
            errors.append(f"missing or wrong type: {k}")
    if errors:
        return errors, warnings
    if case["difficulty"] not in {"쉬움", "보통", "어려움"}:
        errors.append(f"bad difficulty {case['difficulty']!r}")
    for sec in ("history", "exam", "tests"):
        for k, v in case[sec].items():
            if not isinstance(v, str) or not k.strip():
                errors.append(f"{sec}: non-string entry {k!r}")
    if not case["history"]:
        errors.append("empty history")
    if not re.search(r"[가-힣]", case["diagnosis"]):
        errors.append("diagnosis is not in Korean")
    hanja = re.findall(r"[\u4e00-\u9fff]+", json.dumps(case, ensure_ascii=False))
    if hanja:
        errors.append(f"Chinese characters in output (write Korean hangul): {hanja}")
    if case["category"] not in CATEGORIES:
        warnings.append(f"category {case['category']!r} not in the ClinicalQA vocabulary")

    # Diagnosis must match the source answer: the source English name has to be among the aliases.
    ans_vars = answer_variants(item["answer"])
    names = [norm(a) for a in case["aliases"] if isinstance(a, str)]
    if not any(v == n for v in ans_vars for n in names):
        errors.append(f"aliases must include the source answer verbatim: {item['answer']!r}")

    # The answer must not leak into the opening line or any lookup key.
    leak = {norm(t) for t in [case["diagnosis"], *case["aliases"], *ans_vars] if isinstance(t, str)}
    leak = {t for t in leak if len(t) >= 3}
    init = norm(case["initial"])
    for t in leak:
        if t in init:
            errors.append(f"answer term {t!r} leaks into initial")
    for sec in ("history", "exam", "tests"):
        for k in case[sec]:
            for tok in k.split("|"):
                if len(norm(tok)) >= 3 and norm(tok) in leak:
                    errors.append(f"answer term {tok!r} used as a {sec} key")
            v = norm(case[sec][k])
            for t in leak:
                if t in v:
                    warnings.append(f"answer term {t!r} appears in {sec} value (check it is a source fact)")
    if not re.match(r"^(생후\s*)?\d+(세|개월|일|주|대)", case["initial"].strip()):
        warnings.append("initial does not start with an age")

    # Numeric fidelity: no number in exam/tests that is not in the source; history extras are warnings.
    src = src_numbers(item["body"])
    for sec in ("exam", "tests"):
        extra = numbers(" ".join(case[sec].values())) - src
        if extra:
            errors.append(f"{sec} has numbers not in source: {sorted(extra)}")
    extra_units = units(" ".join([*case["exam"].values(), *case["tests"].values()])) - units(item["body"])
    if extra_units:
        errors.append(f"exam/tests have units not in source (do not add or change units): {sorted(extra_units)}")
    # Translation must not add numbers anywhere (e.g. "11 lb (약 5kg)"), so history/initial extras are errors here.
    extra = numbers(" ".join(case["history"].values()) + " " + case["initial"]) - src
    if extra:
        errors.append(f"history/initial have numbers not in source (no conversions or estimates): {sorted(extra)}")
    all_vals = " ".join([case["initial"], *case["history"].values(), *case["exam"].values(), *case["tests"].values()])
    missing = numbers(item["body"]) - numbers(all_vals)
    if missing:
        warnings.append(f"source numbers missing from case: {sorted(missing)}")
    return errors, warnings


# ---------------------------------------------------------------- conversion
def convert_one(item: dict, client: OpenAICompatClient, src: dict, fixed_difficulty: str | None) -> dict:
    cid = item["id"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(dataset=src["dataset"], id=cid, body=item["body"],
                                                                 answer=item["answer"])}]
    log = {"id": cid, "answer": item["answer"], "date": dt.date.today().isoformat(), "attempts": []}
    for _ in range(2):  # first try + one retry
        raw = ""
        try:
            raw = client.chat(messages)
            case = parse_json(raw)
            errors, warnings = validate(case, item)
        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            case, errors, warnings = None, [f"{type(e).__name__}: {e}"], []
        log["attempts"].append({"errors": errors, "warnings": warnings})
        if not errors:
            out = {
                "_note": f"Converted from {src['dataset']} (id={cid}) by scripts/convert_english_cases.py: "
                         "translated to Korean and split into history/exam/tests by an LLM, not clinician-reviewed. "
                         "Local evaluation only; never packaged.",
                "category": case["category"],
                "difficulty": fixed_difficulty or case["difficulty"],
                "teaching_point": case["teaching_point"],
                "initial": case["initial"],
                "history": case["history"],
                "exam": case["exam"],
                "tests": case["tests"],
                "diagnosis": case["diagnosis"],
                "aliases": case["aliases"],
                "must_check": [],
                "source": {"dataset": src["dataset"], "id": cid, "license": src["license"], "url": src["url"]},
            }
            out = apply_fix(out, OVERRIDES[src["key"]].get(cid, {}))
            (src["out"] / f"{src['prefix']}_{cid}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log.update(status="converted", diagnosis_ko=case["diagnosis"], warnings=warnings,
                       llm_difficulty=case["difficulty"])
            return log
        messages = messages[:2] + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "출력에 다음 문제가 있습니다. 규칙을 지켜 JSON 전체를 다시 출력하세요:\n- "
                                        + "\n- ".join(errors)},
        ]
    log["status"] = "skipped"
    print(f"[skip] {cid}: {log['attempts'][-1]['errors']}", file=sys.stderr)
    return log


def apply_overrides(src: dict) -> list[int]:
    """Apply OVERRIDES to existing outputs (no LLM). Returns the ids changed."""
    changed = []
    for cid, fix in OVERRIDES[src["key"]].items():
        path = src["out"] / f"{src['prefix']}_{cid}.json"
        if not path.exists():
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        new = apply_fix(json.loads(json.dumps(case)), fix)
        if new != case:
            path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed.append(cid)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--set", default="base", choices=["base", "extended", "all"],
                    help="agentclinic only: base = AgentClinic-MedQA (0-106), extended = the 107 added cases")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="explicit ids (overrides the default selection)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=2, help="parallel LLM calls (keep ≤ 2: shared API quota)")
    ap.add_argument("--apply-overrides", action="store_true",
                    help="only apply the reviewer fixes in OVERRIDES to existing outputs, then exit")
    args = ap.parse_args()

    src = SOURCES[args.source]
    if args.apply_overrides:
        changed = apply_overrides(src)
        if src["meta"].exists():
            meta = json.loads(src["meta"].read_text(encoding="utf-8"))
            meta.pop("diagnosis_override", None)  # key name used by an earlier version of this script
            meta["overrides"] = {str(k): v for k, v in OVERRIDES[src["key"]].items()}
            src["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"reviewer fixes applied to {len(changed)} files: {changed}")
        return
    if args.source == "agentclinic":
        items = load_agentclinic()
        sel = {"base": range(107), "extended": range(107, len(items)), "all": range(len(items))}[args.set]
        default_ids, selection = list(sel), f"AgentClinic-MedQA set={args.set}"
        fixed_difficulty = None
    else:
        items = load_diagnosisarena()
        default_ids = select_da(items)
        selection = " ".join(select_da.__doc__.split())
        # Journal case reports of rare / atypical presentations (SOTA LLMs score low on the benchmark): all 어려움.
        fixed_difficulty = "어려움"
    ids = args.ids if args.ids is not None else default_ids

    load_dotenv(ROOT / ".env")
    cfg = LLMConfig.from_env("CONVERT_LLM")
    cfg.temperature = 0.0
    cfg.max_retries = 6  # the client sleeps 30 s on 429 before each retry
    client = OpenAICompatClient(cfg)
    src["out"].mkdir(parents=True, exist_ok=True)
    src["meta"].parent.mkdir(parents=True, exist_ok=True)

    todo = [items[i] for i in ids if args.overwrite or not (src["out"] / f"{src['prefix']}_{i}.json").exists()]
    print(f"source={args.source} model={cfg.model} items={len(todo)}")
    with ThreadPoolExecutor(max_workers=min(args.workers, 2)) as pool:
        logs = list(pool.map(lambda it: convert_one(it, client, src, fixed_difficulty), todo))
    for lg in logs:
        print(f"{lg['id']:>5} {lg['status']:<9} attempts={len(lg['attempts'])} warnings={len(lg.get('warnings', []))}")

    prev = json.loads(src["meta"].read_text(encoding="utf-8")) if src["meta"].exists() else {}
    item_log = {str(k): v for k, v in prev.get("items", {}).items()}
    item_log.update({str(lg["id"]): lg for lg in logs})
    meta = {
        "script": "scripts/convert_english_cases.py",
        "dataset": src["dataset"],
        "dataset_url": src["url"],
        "dataset_revision": src["revision"],
        "dataset_license": src["license"],
        "raw_file": str(src["raw"].relative_to(ROOT)),
        "selection": selection,
        "selected_ids": sorted(set(prev.get("selected_ids", [])) | set(ids)),
        "fixed_difficulty": fixed_difficulty,
        "overrides": {str(k): v for k, v in OVERRIDES[src["key"]].items()},
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "reasoning_effort": cfg.reasoning_effort,
        "max_tokens": cfg.max_tokens,
        "date": dt.date.today().isoformat(),
        "workers": min(args.workers, 2),
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "retry_rule": "one retry with the validation errors appended as a user turn; skipped if it fails again",
        "items": dict(sorted(item_log.items(), key=lambda kv: int(kv[0]))),
    }
    src["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_ok = sum(v["status"] == "converted" for v in meta["items"].values())
    print(f"converted {n_ok}/{len(meta['items'])} → {src['out'].relative_to(ROOT)}; meta → "
          f"{src['meta'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
