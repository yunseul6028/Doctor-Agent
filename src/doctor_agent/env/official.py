"""Adapter for the official N.O.V.A. 2026 environment API.

TODO(agent-engineer): implement once the participant guide publishes the API (expected before 10.12). Only this file
(and, if the I/O protocol requires it, the output writer in run.py) should change; the agent talks to
`doctor_agent.env.interface.Environment` only.

What has to be filled in:
  1. `iter_cases(**opts)`: enumerate the private cases the server hands us and yield `(case_id, OfficialEnvironment)`
     one at a time. Create a fresh environment per case (case independence).
  2. `OfficialEnvironment.reset()`: return the initial information (demographics, chief complaint) as
     `Observation(text=...)`.
  3. `OfficialEnvironment.step(action)`: translate `Action(type=ASK|EXAM|TEST|DIAGNOSE, content=...)` into the
     official call and return `Observation(text=<patient/exam/test response>, done=<case finished>)`. The official
     diagnosis format (free text vs. code) goes here too. Never send `action.reason`.
  4. Map official "no result"/"unknown" responses to text containing "제공되지 않습니다" if the wording differs — the
     policy uses that phrase to detect unavailable results (see CaseState.unavailable()).
  5. Errors: raise ordinary exceptions; run_case (submission mode) records them and still submits a diagnosis.
  6. If the server announces a per-case time limit, set AGENT_CASE_TIME_BUDGET_S (or cfg.agent.case_time_budget_s)
     a little below it.

Selection: `python run.py --env official` or `DOCTOR_ENV=official`.
"""
from collections.abc import Iterator

from doctor_agent.env.interface import Action, Environment, Observation

NOT_READY = "official environment API is not published yet — implement doctor_agent/env/official.py"


class OfficialEnvironment(Environment):
    def __init__(self, case_id: str, handle: object = None):
        self.case_id = case_id
        self.handle = handle  # TODO: official session/case handle

    def reset(self) -> Observation:
        raise NotImplementedError(NOT_READY)

    def step(self, action: Action) -> Observation:
        raise NotImplementedError(NOT_READY)


def iter_cases(**opts: object) -> Iterator[tuple[str, Environment]]:
    raise NotImplementedError(NOT_READY)
