"""Structured memory the code keeps for the small LLM: a findings ledger and a differential-diagnosis ledger.

The model reports new findings and per-candidate evidence each turn; the code merges them across turns, so the
prompt can show a short, organised summary instead of the whole conversation.
"""
from dataclasses import dataclass, field

from doctor_agent.agent.text import same_dx, similarity

FINDING_STATUS = {"양성": "양성", "positive": "양성", "있음": "양성", "이상": "양성",
                  "음성": "음성", "negative": "음성", "없음": "음성", "정상": "음성",
                  "결과없음": "결과없음", "미제공": "결과없음", "제공안됨": "결과없음", "unknown": "결과없음", "모름": "결과없음"}
DX_STATUS = {"유력": "유력", "active": "유력", "배제": "배제", "ruled out": "배제", "ruled_out": "배제",
             "위험": "위험", "cant_miss": "위험", "must_rule_out": "위험"}
SAME_ITEM = 0.6
UNVERIFIED_TAG = " (미확인)"  # appended to findings not found in what the environment said
CODE_SOURCE = "result_interpreter"  # Finding.source of items read by agent/result_interpreter.py (grounded by construction)
LLM_RADIOLOGY_SOURCE = "llm_radiology"  # Finding.source of items read by the LLM radiology sub-agent (grounding-checked)
REF_STATUS = "참고"  # DDx candidates suggested by a sub-agent (DdxLedger.refs): reference only, never auto-confirmed
MAX_REFS = 4


@dataclass
class Finding:
    item: str
    status: str  # 양성 | 음성 | 결과없음
    detail: str = ""
    turn: int = 0
    verified: bool | None = None  # set by agent/grounding.py: found (True) / not found (False) in the environment's text
    span: str = ""  # evidence text that grounded it
    # "" = reported by the model; CODE_SOURCE = read from a result text by code (verified=True);
    # LLM_RADIOLOGY_SOURCE = read from a result text by the LLM radiology sub-agent (checked by grounding like "")
    source: str = ""


@dataclass
class FindingsLedger:
    items: list[Finding] = field(default_factory=list)

    def update(self, raw: object, turn: int) -> None:
        if not isinstance(raw, list):
            return
        for r in raw:
            if not isinstance(r, dict) or not str(r.get("item", "")).strip():
                continue
            status = FINDING_STATUS.get(str(r.get("status", "")).strip().lower().replace(" ", ""), "양성")
            self.add(Finding(str(r["item"]).strip(), status, str(r.get("detail", "")).strip(), turn))

    def add(self, new: Finding) -> None:
        """Merge one finding: later information about the same item wins, except that a model report (or an LLM
        radiology reading) never replaces a code reading of the same result (same or later turn): the code read the
        text itself."""
        for i, old in enumerate(self.items):
            if old.item == new.item or similarity(old.item, new.item) >= SAME_ITEM:
                if old.source == CODE_SOURCE and new.source != CODE_SOURCE and old.turn >= new.turn:
                    return
                self.items[i] = new
                return
        self.items.append(new)

    def render(self, exclude_unverified: bool = False) -> str:
        """Grouped by status. Findings the grounding check did not find in the environment's responses
        (verified is False) are tagged "(미확인)", or left out when exclude_unverified is set."""
        if not self.items:
            return ""
        lines = []
        for status in ("양성", "음성", "결과없음"):
            group = [f"{f.item}{f' ({f.detail})' if f.detail else ''}{UNVERIFIED_TAG if f.verified is False else ''}"
                     for f in self.items if f.status == status and not (exclude_unverified and f.verified is False)]
            if group:
                lines.append(f"- {status}: " + "; ".join(group))
        return "\n".join(lines)

    def as_list(self) -> list[dict]:
        return [f.__dict__ for f in self.items]


@dataclass
class DxEntry:
    dx: str
    p: float = 0.0
    status: str = "유력"  # 유력 | 배제 | 위험(반드시 배제해야 할 위험 질환)
    support: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)


def _merge(old: list[str], new: object) -> list[str]:
    if not isinstance(new, list):
        return old
    out = list(old)
    for x in new:
        x = str(x).strip()
        if x and not any(similarity(x, y) >= SAME_ITEM for y in out):
            out.append(x)
    return out[-6:]


@dataclass
class DdxLedger:
    entries: list[DxEntry] = field(default_factory=list)
    # "참고" candidates from specialist sub-agents: shown in the view, but kept out of `entries` so confidence,
    # anchoring, the can't-miss gate, the planner and the forced-diagnosis fallback never treat them as live DDx.
    # A ref leaves this list when the main model itself lists the same diagnosis in its ddx.
    refs: list[DxEntry] = field(default_factory=list)

    def add_ref(self, dx: str, why: str = "") -> bool:
        """Adds a sub-agent candidate as 참고 (p=0). False when it is already in the ledger or the refs are full."""
        dx = str(dx or "").strip()
        if not dx or self._find(dx) is not None or any(same_dx(r.dx, dx) for r in self.refs) \
                or len(self.refs) >= MAX_REFS:
            return False
        why = str(why or "").strip()
        self.refs.append(DxEntry(dx, 0.0, REF_STATUS, [why] if why else []))
        return True

    def refs_list(self) -> list[dict]:
        return [{"dx": r.dx, "status": r.status, "why": r.support[0] if r.support else ""} for r in self.refs]

    def _find(self, dx: str) -> DxEntry | None:
        return next((e for e in self.entries if same_dx(e.dx, dx)), None)  # keeps the first entry's display name

    def update(self, raw: object) -> None:
        if not isinstance(raw, list):
            return
        for r in raw:
            if not isinstance(r, dict) or not str(r.get("dx", "")).strip():
                continue
            dx = str(r["dx"]).strip()
            e = self._find(dx)
            if e is None:
                e = DxEntry(dx)
                self.entries.append(e)
                self.refs = [x for x in self.refs if not same_dx(x.dx, dx)]  # the model adopted a 참고 candidate
            try:
                e.p = max(0.0, min(1.0, float(r.get("p", e.p))))
            except (TypeError, ValueError):
                pass
            e.status = DX_STATUS.get(str(r.get("status", "")).strip().lower(), e.status)
            e.support = _merge(e.support, r.get("for"))
            e.against = _merge(e.against, r.get("against"))

    def ranked(self) -> list[DxEntry]:
        live = sorted((e for e in self.entries if e.status != "배제"), key=lambda e: -e.p)
        return live + [e for e in self.entries if e.status == "배제"]

    def render(self) -> str:
        if not self.entries and not self.refs:
            return ""
        lines = []
        for i, e in enumerate(x for x in self.ranked() if x.status != "배제"):
            tag = " ⚠위험" if e.status == "위험" else ""
            ev = []
            if e.support:
                ev.append("지지: " + ", ".join(e.support))
            if e.against:
                ev.append("반대: " + ", ".join(e.against))
            lines.append(f"{i + 1}. {e.dx} ({round(e.p * 100)}%){tag}" + (f" — {' / '.join(ev)}" if ev else ""))
        ruled = [f"{e.dx}({', '.join(e.against[-2:]) or '근거 미기재'})" for e in self.entries if e.status == "배제"]
        if ruled:
            lines.append("배제됨: " + "; ".join(ruled))
        if self.refs:
            lines.append("참고(자문, 미확인): " + "; ".join(f"{r.dx}({r.support[0][:40]})" if r.support else r.dx
                                                       for r in self.refs))
        return "\n".join(lines)

    def as_list(self) -> list[dict]:
        return [{"dx": e.dx, "p": e.p, "status": e.status, "for": e.support, "against": e.against} for e in self.ranked()]
