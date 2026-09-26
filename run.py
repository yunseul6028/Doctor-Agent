"""Submission entry point (run by the competition server).

TODO(agent-engineer): replace the input/output protocol and environment adapter with the official participant guide's spec once published.
Currently a placeholder that runs local case files: python run.py --cases data/sample_cases
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.agent.loop import run_case  # noqa: E402
from doctor_agent.config import Config  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="predictions.json")
    args = ap.parse_args()

    # Temporary: local simulator. Replace with env/official.py once the official API is published.
    sys.path.insert(0, str(ROOT))
    from eval.simulator import CaseFileEnvironment

    cfg = Config()
    preds = {}
    for path in sorted(Path(args.cases).glob("*.json")):
        llm = OpenAICompatClient(cfg.llm)  # new instance per case
        try:
            preds[path.stem] = run_case(CaseFileEnvironment.from_file(path), llm, cfg)["diagnosis"]
        except Exception as e:  # noqa: BLE001 — one case failing must not stop the rest
            print(f"[{path.stem}] failed: {e}", file=sys.stderr)
            preds[path.stem] = None
    Path(args.out).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
