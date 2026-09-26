"""Local batch evaluation.

python eval/run_local.py                                  # all roles on LLM (reads .env)
python eval/run_local.py --doctor dummy --patient keyword --judge none   # smoke test without an LLM
python eval/viewer.py                                    # view the results viewer only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.loop import run_case  # noqa: E402
from doctor_agent.config import Config, LLMConfig  # noqa: E402
from doctor_agent.llm.client import DummyLLM, OpenAICompatClient  # noqa: E402
from eval.judge import judge_diagnosis  # noqa: E402
from eval.llm_patient import PERSONA_CHOICES, PERSONAS, LLMPatientEnvironment  # noqa: E402
from eval.scorer import missed_checks, score_case  # noqa: E402
from eval.simulator import CaseFileEnvironment  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor", choices=["dummy", "llm"], default="llm")
    ap.add_argument("--patient", choices=["keyword", "llm"], default="llm")
    ap.add_argument("--judge", choices=["none", "llm"], default="llm")
    ap.add_argument("--persona", choices=PERSONA_CHOICES, default="standard",
                    help="virtual patient type (mixed = one hard type per case)")
    ap.add_argument("--cases", default=str(ROOT / "data/sample_cases"))
    ap.add_argument("--out", default=str(ROOT / "eval/results"))
    ap.add_argument("--no-view", action="store_true", help="do not open the results viewer when done")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = Config()
    print(f"doctor={cfg.llm.model if args.doctor == 'llm' else 'dummy'} "
          f"patient={LLMConfig.from_env('PATIENT_LLM').model if args.patient == 'llm' else 'keyword'} "
          f"judge={LLMConfig.from_env('JUDGE_LLM').model if args.judge == 'llm' else 'none'} persona={args.persona}")
    patient_llm = OpenAICompatClient(LLMConfig.from_env("PATIENT_LLM")) if args.patient == "llm" else None
    judge_llm = OpenAICompatClient(LLMConfig.from_env("JUDGE_LLM")) if args.judge == "llm" else None

    rows = []
    for path in sorted(Path(args.cases).glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        doctor = DummyLLM() if args.doctor == "dummy" else OpenAICompatClient(cfg.llm)  # new instance per case
        env = LLMPatientEnvironment(case, patient_llm, args.persona) if patient_llm else CaseFileEnvironment(case)
        t0 = time.time()
        try:
            result = run_case(env, doctor, cfg)
        except Exception as e:  # noqa: BLE001 — one case failing must not stop the batch
            print(f"{path.stem}: FAILED {e}")
            continue
        judged = judge_diagnosis(judge_llm, case, result["diagnosis"]) if judge_llm else None
        scores = score_case(case, result, cfg.agent.max_turns, judged["score"] if judged else None)
        persona = PERSONAS[env.persona]["label"] if patient_llm else "keyword"
        rows.append({"case": path.stem, "persona": persona, "initial": case["initial"], "answer": case["diagnosis"], **result, "judge": judged,
                     "scores": scores, "missed_checks": missed_checks(case, result), "sec": round(time.time() - t0, 1)})
        print(f"{path.stem} [{persona}]: dx={result['diagnosis']!r} turns={result['n_turns']} {scores}")

    if rows:
        avg = {}
        for k in rows[0]["scores"]:
            vals = [r["scores"][k] for r in rows if r["scores"][k] is not None]
            avg[k] = round(sum(vals) / len(vals), 3) if vals else None
        print("AVG", avg)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps({"doctor_model": cfg.llm.model, "persona": args.persona, "avg": avg, "cases": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not args.no_view:
            from eval import viewer

            viewer.webbrowser.open(viewer.build().as_uri())


if __name__ == "__main__":
    main()
