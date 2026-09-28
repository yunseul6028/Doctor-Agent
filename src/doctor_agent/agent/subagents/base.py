"""Interface contract between the sub-agent framework and the content modules (docs/architecture.md "Specialist
sub-agents"). Do not change the fields without updating the architecture doc first."""
from dataclasses import dataclass, field


@dataclass
class SubagentCall:
    name: str                 # e.g. "consult:cardio", "advocate", "radiology"
    messages: list[dict]      # chat messages for the fixed LLM
    json_schema: dict | None  # optional structured output
    max_chars_out: int = 600  # cap of the rendered hint


@dataclass
class SubagentResult:
    name: str
    ok: bool
    hint_ko: str = ""         # short Korean text injected into the NEXT main prompt (<= max_chars_out)
    # [{"name": str, "why": str}] candidates added to the DDx ledger as "참고" (never auto-confirmed)
    ddx_add: list[dict] = field(default_factory=list)
    suggested_actions: list[dict] = field(default_factory=list)  # [{"type": "ASK|EXAM|TEST", "content": str, "why": str}]
    red_flags: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)  # parsed JSON for logging


def failed(name: str, error: object) -> SubagentResult:
    """ok=False result carrying the reason in raw["error"]."""
    return SubagentResult(name, False, raw={"error": str(error)[:200]})
