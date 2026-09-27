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
    gate_turns: int = 0  # actions forced by the can't-miss rule-out gate in this case
    confirmed_other_shown: bool = False  # the gate's "another can't-miss diagnosis is confirmed" hint was shown
    safety_log: list[dict] = field(default_factory=list)  # grounding / gate / precondition events (for the viewer)
    view_max_chars: int = 0  # prompt-length cap for view(); set from AgentConfig.max_view_chars by the loop

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

    def view(self, recent: int = 4, max_chars: int | None = None) -> str:
        """Compact prompt view: ledgers + short list of older actions + the last few exchanges verbatim.

        Capped at `max_chars` (default: self.view_max_chars; 0 = unlimited). When over the cap, material is cut in this
        order: oldest compact actions, long verbatim responses, fewer verbatim exchanges, the oldest part of each ledger
        line, and finally a hard cut that keeps the initial info and the most recent text."""
        cap = self.view_max_chars if max_chars is None else max_chars
        text = self._render(recent)
        if not cap or len(text) <= cap:
            return text
        n_older = max(0, self.turn_count - recent) if recent else self.turn_count
        for skip in range(1, n_older + 1):  # 1) drop oldest compact actions
            if len(text := self._render(recent, skip_older=skip)) <= cap:
                return text
        for resp_cap in (400, 150):  # 2) shorten verbatim responses
            if len(text := self._render(recent, skip_older=n_older, resp_cap=resp_cap)) <= cap:
                return text
        for r in range(min(recent, self.turn_count) - 1, 0, -1):  # 3) fewer verbatim exchanges
            if len(text := self._render(r, skip_older=self.turn_count, resp_cap=150)) <= cap:
                return text
        r = 1 if self.turn_count else 0
        for line_cap in (400, 200, 100):  # 4) keep only the newest part of each ledger line
            if len(text := self._render(r, skip_older=self.turn_count, resp_cap=150, line_cap=line_cap)) <= cap:
                return text
        head = f"[처음 정보] {self.initial_info}"[: cap // 3]  # 5) hard cut
        return head + "\n…\n" + text[-max(0, cap - len(head) - 3):]

    def _render(self, recent: int, skip_older: int = 0, resp_cap: int | None = None, line_cap: int | None = None) -> str:
        def cut_line(line: str) -> str:
            return line if line_cap is None or len(line) <= line_cap else "…" + line[-line_cap:]

        parts = [f"[처음 정보] {self.initial_info}"]
        if f := self.findings.render():
            parts.append("[소견 장부]\n" + "\n".join(cut_line(x) for x in f.split("\n")))
        if d := self.ddx_ledger.render():
            parts.append("[감별 진단 장부]\n" + "\n".join(cut_line(x) for x in d.split("\n")))
        older, last = self.turns[:-recent] if recent else self.turns, self.turns[-recent:] if recent else []
        shown = older[skip_older:]
        if shown or (older and skip_older):
            lines = [f"{i}. {t.action.type.value}: {t.action.content[:40]} → {t.response[:60]}"
                     for i, t in enumerate(shown, min(skip_older, len(older)) + 1)]
            if skip_older and older:
                lines.insert(0, f"(앞선 행동 {min(skip_older, len(older))}개 생략)")
            parts.append("[이전 행동]\n" + "\n".join(lines))
        if last:
            start = len(older) + 1

            def resp(t: Turn) -> str:
                return t.response if resp_cap is None or len(t.response) <= resp_cap else t.response[:resp_cap] + "…"
            parts.append("[최근 대화]\n" + "\n".join(
                f"{i}. {t.action.type.value}: {t.action.content}\n   → {resp(t)}" for i, t in enumerate(last, start)))
        return "\n\n".join(parts)

    def transcript(self) -> str:
        lines = [f"[처음 정보] {self.initial_info}"]
        for i, t in enumerate(self.turns, 1):
            lines.append(f"{i}. {t.action.type.value}: {t.action.content}\n   → {t.response}")
        return "\n".join(lines)
