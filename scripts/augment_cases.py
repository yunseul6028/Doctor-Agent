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

from doctor_agent.agent.parser import _json_objects  # noqa: E402
from doctor_agent.config import LLMConfig  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402
from eval.run_local import load_dotenv  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", help="case ids, e.g. 181 521 (default: all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

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
