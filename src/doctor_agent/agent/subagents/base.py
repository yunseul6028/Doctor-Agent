"""Runtime specialist sub-agent interface (same gpt-oss model, different role prompt + evidence).

Created by clinical-strategist only because it was absent; the framework agent owns this file."""
from dataclasses import dataclass


@dataclass
class SubagentCall:
    name: str
    messages: list[dict]
    json_schema: dict | None
    max_chars_out: int = 600


@dataclass
class SubagentResult:
    name: str
    ok: bool
    hint_ko: str
    ddx_add: list[dict]
    suggested_actions: list[dict]
    red_flags: list[str]
    raw: dict
