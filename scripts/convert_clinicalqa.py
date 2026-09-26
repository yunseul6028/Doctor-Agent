"""Convert selected snuh/ClinicalQA items into our interactive case format (offline, dev only).

Source: https://huggingface.co/datasets/snuh/ClinicalQA (Apache-2.0), downloaded to
data/external/snuh_clinicalqa/ (git-ignored, never packaged).

    source .venv/bin/activate
    python scripts/convert_clinicalqa.py                 # convert all SELECTED_IDS
    python scripts/convert_clinicalqa.py --ids 1 42      # only these ids
    python scripts/convert_clinicalqa.py --overwrite     # redo existing outputs

LLM: CONVERT_LLM_* in .env (falls back to the shared LLM_*). Output: data/cases_clinicalqa/cqa_<id>.json.
Reproducibility record (prompt, model, date, per-item log): data/labels/clinicalqa_conversion_meta.json.
Needs pyarrow (requirements-dev.txt only).
"""
import argparse
import datetime as dt
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.config import LLMConfig  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402
from eval.run_local import load_dotenv  # noqa: E402

DATASET = "snuh/ClinicalQA"
DATASET_REVISION = "29bdc9043f3e693b7671f868e2038ec7ac288acd"
PARQUET = ROOT / "data/external/snuh_clinicalqa/data/train-00000-of-00001.parquet"
OUT_DIR = ROOT / "data/cases_clinicalqa"
META = ROOT / "data/labels/clinicalqa_conversion_meta.json"

# 40 diagnosis-answer items (question asks for the most likely diagnosis/cause and the correct option is a
# disease). Picked by hand from the 207 diagnosis-type stems, spread across chief complaints and specialties,
# favouring long vignettes with history + exam + labs. Treatment / next-test / mechanism items excluded, as were
# answers that are not a disease (e.g. "ACE 억제제 사용", "약물 유발성 변비", 피임 방법).
SELECTED_IDS = [
    1, 6, 42, 47, 52, 117, 121, 148, 157, 181,
    211, 225, 232, 238, 285, 328, 343, 383, 388, 404,
    437, 458, 466, 516, 521, 523, 550, 578, 595, 630,
    660, 734, 742, 762, 841, 940, 966, 996, 1024, 1029,
]

SYSTEM_PROMPT = """당신은 의학 교육용 객관식 문항을 대화형 진단 연습 증례(JSON)로 바꾸는 변환기입니다.
원문 문항에 적힌 사실만 옮기세요. 원문에 없는 증상, 병력, 진찰 소견, 검사 결과, 수치, 음성 소견을 절대 만들어 내지 마세요.

출력 형식: 아래 키를 가진 JSON 객체 하나만 출력하세요 (코드블록, 설명 문장 금지).
{
  "initial": "<나이>세 <남성|여성>. 주호소: <주호소와 기간>",
  "history": {"<키워드1|키워드2|english>": "<환자 말투 답변>", ...},
  "exam": {"<키워드|english>": "<진찰 소견>", ...},
  "tests": {"<키워드|english>": "<검사 결과>", ...},
  "diagnosis": "<정답 진단명(한국어)>",
  "aliases": ["<한국어 동의어>", "<영어 진단명>", ...],
  "difficulty": "<쉬움|보통|어려움>",
  "teaching_point": "<한국어 1~2문장>"
}

규칙:
1. initial: 나이, 성별, 주호소(원문에 기간이 있으면 기간 포함)만 쓰세요. 다른 증상, 병력, 소견, 진단 힌트는 넣지 마세요.
   형식 예: "58세 남성. 주호소: 2시간 전부터 시작된 가슴 통증".
2. history: 원문의 병력 정보(발병 양상, 동반 증상, 과거력, 약물, 가족력, 사회력 등)를 주제별 항목으로 나누세요.
   - 값은 환자 본인이 말하는 쉬운 한국어 구어체(1~2문장)로 쓰고, 의학 용어나 진단명은 쓰지 마세요
     (예: "고혈압" → "혈압이 높아서 약 먹고 있어요"). 과거에 진단받은 병은 환자가 들은 이름으로 짧게 말해도 되지만,
     검사 수치나 의학 약어(T-score, HbA1c 등)는 history에 넣지 말고 tests로 옮기세요.
   - 원문에 "~는 없다"처럼 명시된 음성 정보만 음성으로 쓰세요. 원문에 없는 음성 정보를 추가하지 마세요.
   - 수량·기간·횟수는 원문 숫자와 단위를 그대로 쓰고 다른 값으로 환산하거나 풀어 쓰지 마세요
     (예: "20갑년" → "하루 한 갑씩 20년"처럼 바꾸면 안 됨. "담배는 20갑년 정도 피웠다고 들었어요"처럼 쓰세요).
   - 원문에 없는 세부 묘사(통증 강도, 시각, 상황, 감정 등)를 덧붙이지 마세요.
3. exam: 원문의 신체진찰 소견과 활력징후를 부위/종류별로 나누세요. 의학 용어 그대로, 원문 수치 그대로 쓰세요.
   심전도, 영상, 혈액·소변 검사, 조직검사, 청력·폐기능 검사처럼 장비나 검체가 필요한 것은 원문에서 '신체 검사' 문장에
   있더라도 exam이 아니라 tests에 넣으세요.
4. tests: 원문의 검사 결과(혈액, 소변, 영상, 심전도, 조직검사 등)를 검사 종류별로 나누세요. 수치, 단위, 참고치(정상 범위)는
   원문 그대로 모두 옮기고 빠뜨리지 마세요.
   원문이 "정상"이라고 한 검사는 "정상"으로만 쓰세요.
5. 키: "|"로 구분한 검색 키워드 3~8개 (한국어 일상어, 의학 용어, 영어 약어를 섞어서). 의사가 그 항목을 물을 때 쓸 법한 단어로 하세요.
   키에 정답 진단명이나 그 약어를 넣지 마세요.
6. 원문 문항 끝의 질문 문장("가장 가능성 높은 진단은?")과 보기는 증례에 넣지 마세요. 해설은 teaching_point 작성에만 참고하고,
   해설에만 있고 문항 본문·검사 소견에 없는 사실은 history/exam/tests에 넣지 마세요.
7. diagnosis: 정답 보기의 진단명을 한국어로 쓰세요 (정답 보기 문구를 그대로 쓰는 것이 원칙). aliases: 한국어 동의어·다른 표기와 영어 진단명, 흔한 약어.
8. difficulty: 문진·진찰·검사를 순서대로 해 나가는 대화형 진단 기준으로 매기세요.
   "쉬움": 흔한 질환의 전형적 양상이고, 결정적 검사 한 가지로 바로 확진된다.
   "보통": 병력·진찰·검사 소견을 여러 개 종합해야 하거나, 결정적 단서가 특정 질문(과거력, 약물, 가족력, 노출력)으로만 드러난다.
   "어려움": 드문 질환이거나, 비전형적 양상이거나, 다른 진단을 시사하는 혼란 소견이 있거나, 보기 중 비슷한 감별진단과 구별이 어렵다.
9. teaching_point: 이 증례에서 어떤 단서로 어떤 감별진단과 구별해 진단하는지 한국어 1~2문장.
"""

USER_TEMPLATE = """[원문 문항 — {dataset} question_id={id}]
주호소 분류: {chief_complaint}
학습 목표: {purpose}

문항 본문:
{question}

검사 소견:
{exam}

보기:
{options}

정답: {answer_letter}. {answer_text}

해설 (teaching_point 작성용 참고, 사실 추출 금지):
{explanation}
"""

# Reviewer override (eval-simulator, 2026-09-25). The LLM rated 29/40 as 쉬움 because the MCQ vignettes hand over every
# finding. These items are rare diseases, carry misleading findings, or hinge on separating close options, so they are
# marked 어려움 to keep a difficulty mix. The LLM's own rating is kept in the meta log as llm_difficulty.
DIFFICULTY_OVERRIDE = {
    181: "어려움",   # Takayasu arteritis: rare cause of secondary hypertension
    238: "어려움",   # SLE with psoriasis-like scalp/nail findings as a confounder
    328: "어려움",   # bipolar II vs bipolar I / MDD / substance-induced
    437: "어려움",   # central vs nephrogenic DI vs primary polydipsia
    466: "어려움",   # congenital long QT: rare, syncope vs seizure
    521: "어려움",   # CVST: postpartum headache, easy to miss
    841: "어려움",   # exercise-induced anaphylaxis vs food/cholinergic urticaria
    1029: "어려움",  # autoimmune hepatitis vs viral/drug hepatitis
}

REQUIRED = {"initial": str, "history": dict, "exam": dict, "tests": dict, "diagnosis": str,
            "aliases": list, "difficulty": str, "teaching_point": str}


def load_items() -> dict[int, dict]:
    import pyarrow.parquet as pq

    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; download snuh/ClinicalQA first (see docs/licenses.md)")
    return {r["question_id"]: r for r in pq.read_table(PARQUET).to_pylist()}


def answer_text(item: dict) -> str:
    return item["options"][f"option_{item['answer']}"]


def build_user(item: dict) -> str:
    opts = "\n".join(f"{k[-1]}. {v}" for k, v in item["options"].items() if v)
    return USER_TEMPLATE.format(
        dataset=DATASET, id=item["question_id"], chief_complaint=item["chief_complaint"], purpose=item["purpose"],
        question=item["question"], exam=item["exam"] or "(없음)", options=opts, answer_letter=item["answer"],
        answer_text=answer_text(item), explanation=item["explanation"],
    )


def norm(s: str) -> str:
    return re.sub(r"[\s\-_·,/'’]", "", s.lower())


def answer_variants(ans: str) -> list[str]:
    """'원발성 담즙성 담관염(이전 명칭: ...)' → full text, text without parentheses, parenthetical text."""
    out = [ans, re.sub(r"\(.*?\)", "", ans)]
    out += [p.split(":")[-1] for p in re.findall(r"\((.*?)\)", ans)]
    return [norm(v) for v in out if norm(v)]


def numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d+(?:[.,]\d+)*", s)}


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def validate(case: dict, item: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors trigger a retry, then skip."""
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

    # Diagnosis must match the source answer.
    ans = answer_text(item)
    names = [norm(case["diagnosis"])] + [norm(a) for a in case["aliases"] if isinstance(a, str)]
    if not any(v == n or (len(v) >= 3 and (v in n or n in v)) for v in answer_variants(ans) for n in names if n):
        errors.append(f"diagnosis {case['diagnosis']!r} does not match source answer {ans!r}")

    # The answer must not leak into the opening line or the lookup keys.
    leak_terms = {t for t in [case["diagnosis"], *case["aliases"], *answer_variants(ans)] if isinstance(t, str)}
    leak_terms = {norm(t) for t in leak_terms if len(norm(t)) >= 3}
    init = norm(case["initial"])
    for t in leak_terms:
        if t in init:
            errors.append(f"answer term {t!r} leaks into initial")
    for sec in ("history", "exam", "tests"):
        for k in case[sec]:
            for tok in k.split("|"):
                if len(norm(tok)) >= 3 and norm(tok) in leak_terms:
                    errors.append(f"answer term {tok!r} used as a {sec} key")
            v = norm(case[sec][k])
            for t in leak_terms:
                if t in v:
                    warnings.append(f"answer term {t!r} appears in {sec} value (check it is a source fact)")
    if not re.match(r"^\d+(세|개월|일)", case["initial"].strip()):
        warnings.append("initial does not start with an age")

    # No invented numbers: every number in exam/tests must appear in the source vignette or lab text.
    src_nums = numbers(item["question"] + " " + (item["exam"] or ""))
    for sec in ("exam", "tests"):
        extra = numbers(" ".join(case[sec].values())) - src_nums
        if extra:
            errors.append(f"{sec} has numbers not in source: {sorted(extra)}")
    extra = numbers(" ".join(case["history"].values()) + " " + case["initial"]) - src_nums
    if extra:
        warnings.append(f"history/initial numbers not in source: {sorted(extra)}")
    # Omissions: every number in the source lab text should survive somewhere in the case.
    all_vals = " ".join([case["initial"], *case["history"].values(), *case["exam"].values(), *case["tests"].values()])
    missing = numbers(item["exam"] or "") - numbers(all_vals)
    if missing:
        warnings.append(f"source lab numbers missing from case: {sorted(missing)}")
    return errors, warnings


def convert_one(item: dict, client: OpenAICompatClient) -> dict:
    qid = item["question_id"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user(item)}]
    log = {"id": qid, "attempts": []}
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
                "_note": f"Converted from {DATASET} (question_id={qid}) by scripts/convert_clinicalqa.py. "
                         "Source item was LLM-drafted and clinician-reviewed per the dataset card; this conversion "
                         "is LLM-generated and not separately clinician-reviewed.",
                "category": item["chief_complaint"],
                "difficulty": DIFFICULTY_OVERRIDE.get(qid, case["difficulty"]),
                "teaching_point": case["teaching_point"],
                "initial": case["initial"],
                "history": case["history"],
                "exam": case["exam"],
                "tests": case["tests"],
                "diagnosis": case["diagnosis"],
                "aliases": case["aliases"],
                "must_check": [],
                "source": {"dataset": DATASET, "id": qid, "license": "Apache-2.0"},
            }
            (OUT_DIR / f"cqa_{qid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
            log["status"] = "converted"
            log["warnings"] = warnings
            log["llm_difficulty"] = case["difficulty"]
            return log
        messages = messages[:2] + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "출력에 다음 문제가 있습니다. 규칙을 지켜 JSON 전체를 다시 출력하세요:\n- "
                                        + "\n- ".join(errors)},
        ]
    log["status"] = "skipped"
    print(f"[skip] {qid}: {log['attempts'][-1]['errors']}", file=sys.stderr)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=SELECTED_IDS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = LLMConfig.from_env("CONVERT_LLM")
    cfg.temperature = 0.0
    client = OpenAICompatClient(cfg)
    items = load_items()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)

    todo = [items[i] for i in args.ids if args.overwrite or not (OUT_DIR / f"cqa_{i}.json").exists()]
    print(f"model={cfg.model} items={len(todo)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        logs = list(pool.map(lambda it: convert_one(it, client), todo))
    for lg in logs:
        print(f"{lg['id']:>5} {lg['status']:<9} attempts={len(lg['attempts'])} "
              f"warnings={len(lg.get('warnings', []))}")

    prev = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    item_log = {str(k): v for k, v in prev.get("items", {}).items()}
    item_log.update({str(lg["id"]): lg for lg in logs})
    meta = {
        "script": "scripts/convert_clinicalqa.py",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "Apache-2.0",
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "reasoning_effort": cfg.reasoning_effort,
        "max_tokens": cfg.max_tokens,
        "date": dt.date.today().isoformat(),
        "selected_ids": SELECTED_IDS,
        "difficulty_override": DIFFICULTY_OVERRIDE,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "retry_rule": "one retry with the validation errors appended as a user turn; skipped if it fails again",
        "items": dict(sorted(item_log.items(), key=lambda kv: int(kv[0]))),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_ok = sum(v["status"] == "converted" for v in meta["items"].values())
    print(f"converted {n_ok}/{len(meta['items'])} → {OUT_DIR.relative_to(ROOT)}; meta → {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
