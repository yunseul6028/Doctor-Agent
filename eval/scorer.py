"""Local score estimate. Owned by eval-simulator.

Simple proxy metrics (v0):
- accuracy: LLM judge score when judge_score is given, otherwise string match against the answer/aliases
- efficiency: 1 - turns used / max turns
- safety: fraction of the applicable minimum safety checks from safety/protocols.py (guideline-cited) that were done;
  None when no protocol applies to the case (excluded from averages)
- case_checks: fraction of the case author's own must_check list (reference only, not guideline-based)
"""
from doctor_agent.safety.protocols import must_checks_for


def score_case(case: dict, result: dict, max_turns: int = 60, judge_score: float | None = None) -> dict:
    dx = (result.get("diagnosis") or "").lower()
    answers = [case["diagnosis"].lower()] + [a.lower() for a in case.get("aliases", [])]
    accuracy = judge_score if judge_score is not None else float(bool(dx) and any(a in dx for a in answers))

    efficiency = max(0.0, 1 - result["n_turns"] / max_turns)

    done = " ".join(t["content"].lower() for t in result["turns"])
    checks = _checks(case)
    safety = round(sum(c.done_in(done) for c in checks) / len(checks), 3) if checks else None

    must = [m.lower() for m in case.get("must_check", [])]
    case_checks = round(sum(any(tok in done for tok in m.split("|")) for m in must) / len(must), 3) if must else None

    return {"accuracy": accuracy, "efficiency": round(efficiency, 3), "safety": safety, "case_checks": case_checks}


def _checks(case: dict) -> list:
    # category from the chief complaint only; conditional checks (pregnancy, thunderclap…) from facts the doctor
    # could have learned, even if not uncovered
    # history + exam: both are learnable by asking/examining (e.g. a murmur makes endocarditis blood cultures due)
    facts = " ".join(map(str, [*case.get("history", {}).values(), *case.get("exam", {}).values()]))
    return [c for c in must_checks_for(case["initial"], facts) if c.kind != "treatment"]


def missed_checks(case: dict, result: dict) -> list[str]:
    done = " ".join(t["content"].lower() for t in result["turns"])
    return [f"{c.name} ({c.citation.short})" for c in _checks(case) if not c.done_in(done)]
