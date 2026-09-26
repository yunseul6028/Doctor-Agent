from dataclasses import dataclass, field

from doctor_agent.agent.ledger import DdxLedger, FindingsLedger
from doctor_agent.agent.text import SIMILAR, similarity
from doctor_agent.env.interface import Action


@dataclass
class Turn:
    action: Action
    response: str
    ddx: list[dict] = field(default_factory=list)  # DDx snapshot at the time of this action


@dataclass
class CaseState:
    """State of a single case. Created fresh per case (no cross-case sharing)."""

    initial_info: str
    turns: list[Turn] = field(default_factory=list)
    ddx: list[dict] = field(default_factory=list)  # [{"dx": str, "p": float}]
    confidence: float = 0.0
    findings: FindingsLedger = field(default_factory=FindingsLedger)
    ddx_ledger: DdxLedger = field(default_factory=DdxLedger)
    reviews: list[dict] = field(default_factory=list)  # pre-diagnosis reviews: {turn, proposed, verdict, issues}
    safety_pushback: bool = False  # already asked once to finish safety checks before diagnosing

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def asked(self, action: Action) -> bool:
        """True if an action of the same type with (nearly) the same content was already done."""
        return any(t.action.type == action.type and similarity(t.action.content, action.content) >= SIMILAR
                   for t in self.turns)

    def unavailable(self) -> list[str]:
        """Requests the environment said it has no result for."""
        return [t.action.content for t in self.turns if "제공되지 않습니다" in t.response]

    def view(self, recent: int = 4) -> str:
        """Compact prompt view: ledgers + short list of older actions + the last few exchanges verbatim."""
        parts = [f"[처음 정보] {self.initial_info}"]
        if f := self.findings.render():
            parts.append("[소견 장부]\n" + f)
        if d := self.ddx_ledger.render():
            parts.append("[감별 진단 장부]\n" + d)
        older, last = self.turns[:-recent] if recent else self.turns, self.turns[-recent:] if recent else []
        if older:
            parts.append("[이전 행동]\n" + "\n".join(
                f"{i}. {t.action.type.value}: {t.action.content[:40]} → {t.response[:60]}" for i, t in enumerate(older, 1)))
        if last:
            start = len(older) + 1
            parts.append("[최근 대화]\n" + "\n".join(
                f"{i}. {t.action.type.value}: {t.action.content}\n   → {t.response}" for i, t in enumerate(last, start)))
        return "\n\n".join(parts)

    def transcript(self) -> str:
        lines = [f"[처음 정보] {self.initial_info}"]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"{i}. {t.action.type.value}: {t.action.content}\n   → {t.response}")
        return "\n".join(lines)
