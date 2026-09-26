"""Interactive test mode (not a submission artifact). Run in a real terminal because it reads keyboard input.

python eval/play.py                 # you are the doctor: ask the virtual patient (LLM) questions and diagnose
python eval/play.py --role patient  # you are the patient: answer the doctor agent's questions
python eval/play.py --case synthetic_003
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.loop import run_case  # noqa: E402
from doctor_agent.config import Config, LLMConfig  # noqa: E402
from doctor_agent.env.interface import Action, ActionType, Environment, Observation  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402
from eval.judge import judge_diagnosis  # noqa: E402
from eval.llm_patient import PERSONA_CHOICES, PERSONAS, LLMPatientEnvironment  # noqa: E402
from eval.run_local import load_dotenv  # noqa: E402
from eval.scorer import missed_checks, score_case  # noqa: E402

CASES = ROOT / "data/sample_cases"
PREFIX = {"질문": ActionType.ASK, "문진": ActionType.ASK, "진찰": ActionType.EXAM,
          "검사": ActionType.TEST, "진단": ActionType.DIAGNOSE}
LABEL = {ActionType.ASK: "문진", ActionType.EXAM: "진찰", ActionType.TEST: "검사", ActionType.DIAGNOSE: "진단"}
SCORE = {"accuracy": "정확도", "efficiency": "효율성", "safety": "안전성", "case_checks": "증례 체크"}


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        sys.exit(0)


def pick_case(name: str | None) -> tuple[str, dict]:
    paths = sorted(CASES.glob("*.json"))
    if name:
        p = CASES / f"{name.removesuffix('.json')}.json"
        return p.stem, json.loads(p.read_text(encoding="utf-8"))
    print("\n증례 목록 (정답은 숨김)")
    cases = [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    for i, (stem, c) in enumerate(cases, 1):
        meta = " · ".join(x for x in (c.get("category"), c.get("difficulty")) if x)
        print(f"  {i:>2}. {stem}  {meta}  |  {c['initial']}")
    while True:
        sel = ask(f"\n번호를 고르세요 (1-{len(cases)}): ")
        if sel.isdigit() and 1 <= int(sel) <= len(cases):
            return cases[int(sel) - 1]


def parse_input(text: str) -> Action:
    head, _, rest = text.partition(" ")
    if head in PREFIX and rest.strip():
        return Action(PREFIX[head], rest.strip())
    return Action(ActionType.ASK, text)  # no prefix = question


def save(stem: str, case: dict, result: dict, judged: dict | None, scores: dict, who: str, sec: float) -> None:
    out = ROOT / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    row = {"case": stem, "initial": case["initial"], "answer": case["diagnosis"], **result,
           "judge": judged, "scores": scores, "sec": round(sec, 1)}
    (out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps({"doctor_model": who, "avg": scores, "cases": [row]}, ensure_ascii=False, indent=2), encoding="utf-8")


def report(case: dict, result: dict, judge_llm) -> tuple[dict | None, dict]:
    judged = judge_diagnosis(judge_llm, case, result["diagnosis"])
    scores = score_case(case, result, 60, judged["score"])
    print("\n" + "=" * 50)
    print(f"  제출한 진단 : {result['diagnosis']}")
    print(f"  정답        : {case['diagnosis']}")
    print(f"  채점 이유   : {judged['reason']}")
    print(f"  사용한 턴   : {result['n_turns']}")
    print("  점수        : " + ", ".join(f"{SCORE[k]} {v:.2f}" if v is not None else f"{SCORE[k]} –" for k, v in scores.items()))
    missed = missed_checks(case, result)
    if missed:
        print(f"  빠뜨린 확인 : {', '.join(missed)}")
    if case.get("teaching_point"):
        print(f"  핵심 포인트 : {case['teaching_point']}")
    print("=" * 50)
    return judged, scores


def play_as_doctor(stem: str, case: dict, cfg: Config, persona: str) -> None:
    env = LLMPatientEnvironment(case, OpenAICompatClient(LLMConfig.from_env("PATIENT_LLM")), persona)
    print(f"\n[처음 정보] {env.reset().text}")
    print("[환자 유형] 무작위 (끝나고 공개)" if persona == "mixed" else f"[환자 유형] {PERSONAS[env.persona]['label']}")
    print("입력 방법: 그냥 쓰면 질문 · '진찰 복부' · '검사 CT' · '진단 급성 충수염' · Ctrl+C 종료\n")
    turns, diagnosis, t0 = [], None, time.time()
    while len(turns) < cfg.agent.max_turns:
        text = ask(f"[{len(turns) + 1}턴] 의사 > ")
        if not text:
            continue
        action = parse_input(text)
        obs = env.step(action)
        turns.append({**action.to_dict(), "response": obs.text})
        if action.type == ActionType.DIAGNOSE:
            diagnosis = action.content
            break
        print(f"   {'환자' if action.type == ActionType.ASK else '결과'} > {obs.text}\n")
    result = {"diagnosis": diagnosis, "turns": turns, "n_turns": len(turns), "ddx": [], "llm_calls": 0}
    judged, scores = report(case, result, OpenAICompatClient(LLMConfig.from_env("JUDGE_LLM")))
    print(f"  환자 유형   : {PERSONAS[env.persona]['label']}")
    save(stem, case, result, judged, scores, "사람 (직접 진료)", time.time() - t0)


class HumanPatientEnvironment(Environment):
    """You play the patient (and report exam/test results)."""

    def __init__(self, case: dict):
        self.case = case

    def reset(self) -> Observation:
        return Observation(self.case["initial"])

    def step(self, action: Action) -> Observation:
        if action.type == ActionType.DIAGNOSE:
            return Observation("진단이 제출되었습니다.", done=True)
        print(f"\n   의사 ({LABEL[action.type]}) > {action.content}")
        who = "환자" if action.type == ActionType.ASK else "결과"
        return Observation(ask(f"   {who} > ") or "잘 모르겠어요.")


def play_as_patient(stem: str, case: dict, cfg: Config, persona: str) -> None:
    print(f"\n[처음 정보] {case['initial']}")
    print("당신은 환자입니다. 에이전트의 질문에 답하세요. 진찰·검사 요청에는 결과를 직접 적거나, 빈칸으로 두면 '잘 모르겠어요'가 됩니다.")
    t0 = time.time()
    result = run_case(HumanPatientEnvironment(case), OpenAICompatClient(cfg.llm), cfg)
    judged, scores = report(case, result, OpenAICompatClient(LLMConfig.from_env("JUDGE_LLM")))
    save(stem, case, result, judged, scores, f"{cfg.llm.model} (사람이 환자 역할)", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["doctor", "patient"], default="doctor")
    ap.add_argument("--case", help="e.g. synthetic_001 (omit to choose from a list)")
    ap.add_argument("--persona", choices=PERSONA_CHOICES, default="standard", help="virtual patient type in doctor mode")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = Config()
    stem, case = pick_case(args.case)
    (play_as_doctor if args.role == "doctor" else play_as_patient)(stem, case, cfg, args.persona)
    print("\n결과는 뷰어에서도 볼 수 있어요: python eval/viewer.py")


if __name__ == "__main__":
    main()
