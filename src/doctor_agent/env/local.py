"""Local case files (data/*_cases/*.json) through the keyword simulator. For dev runs of run.py only."""
from collections.abc import Iterator
from pathlib import Path

from doctor_agent.env.interface import Environment


def iter_cases(cases: str | Path | None = None, **_: object) -> Iterator[tuple[str, Environment]]:
    if not cases:
        raise ValueError("local environment needs --cases <dir or .json file>")
    from eval.simulator import CaseFileEnvironment  # ships in the ZIP; run.py puts the repo root on sys.path

    p = Path(cases)
    for path in ([p] if p.is_file() else sorted(p.glob("*.json"))):
        yield path.stem, CaseFileEnvironment.from_file(path)
