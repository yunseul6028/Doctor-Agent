"""Calibrate the code-computed confidence (src/doctor_agent/agent/confidence.py) on local result files. Dev only.

    python scripts/calibrate_confidence.py eval/results/run_*.json            # print the report
    python scripts/calibrate_confidence.py --write eval/results/run_*.json    # also write data/labels/confidence_params.json

The written file is not shipped; to use it at runtime set AGENT_CONFIDENCE_PARAMS=data/labels/confidence_params.json,
or copy the rounded params into DEFAULT_PARAMS.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.agent import confidence  # noqa: E402


def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    report = confidence.calibrate(paths)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if "--write" in argv:
        confidence.PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        confidence.PARAMS_PATH.write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
