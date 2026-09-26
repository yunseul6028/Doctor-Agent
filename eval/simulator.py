"""Local virtual patient environment. Owned by eval-simulator.

Answers only from the case file. v0 uses keyword matching; v1 will switch to an LLM-based patient.
"""
import json
from pathlib import Path

from doctor_agent.env.interface import Action, ActionType, Environment, Observation


class CaseFileEnvironment(Environment):
    def __init__(self, case: dict):
        self.case = case

    @classmethod
    def from_file(cls, path: str | Path) -> "CaseFileEnvironment":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def reset(self) -> Observation:
        return Observation(self.case["initial"])

    def step(self, action: Action) -> Observation:
        if action.type == ActionType.DIAGNOSE:
            return Observation("진단이 제출되었습니다.", done=True)
        table = {
            ActionType.ASK: self.case.get("history", {}),
            ActionType.EXAM: self.case.get("exam", {}),
            ActionType.TEST: self.case.get("tests", {}),
        }[action.type]
        q = action.content.lower()
        hits = [v for k, v in table.items() if any(tok in q for tok in k.lower().split("|"))]
        if hits:
            return Observation(" ".join(hits))
        return Observation({ActionType.ASK: "잘 모르겠어요.", ActionType.EXAM: "이 진찰 결과는 제공되지 않습니다.",
                            ActionType.TEST: "이 검사 결과는 제공되지 않습니다."}[action.type])
