"""Output parsing: the one JSON extractor (find_json) behind every LLM answer the agent reads (step / final action,
pre-diagnosis review, sub-agents), plus the action parser."""
import json
import re
from typing import Callable

from doctor_agent.env.interface import Action, ActionType
from doctor_agent.llm.harmony import split_harmony

# Reasoning blocks some models put in the content (Gemma <thought>, <think>, gpt-oss analysis leaks)
_THOUGHT_RE = re.compile(r"<(thought|think|thinking|analysis)>.*?(</\1>|$)", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?", re.I)
_DECODER = json.JSONDecoder()

# JSON schema of one step output (structured output / vLLM guided decoding). Deliberately permissive: only the action
# fields are required so a server-side schema never blocks an otherwise usable answer.
ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "status": {"type": "string"}, "detail": {"type": "string"}}}},
        "ddx": {"type": "array", "items": {"type": "object", "properties": {
            "dx": {"type": "string"}, "p": {"type": "number"}, "status": {"type": "string"},
            "for": {"type": "array", "items": {"type": "string"}},
            "against": {"type": "array", "items": {"type": "string"}}}}},
        "type": {"type": "string", "enum": [t.value for t in ActionType]},
        "content": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["type", "content"],
}


def _json_objects(text: str) -> list[dict]:
    """All top-level JSON objects decodable from any '{' in the text, in order."""
    out, i = [], text.find("{")
    while i != -1:
        try:
            obj, end = _DECODER.raw_decode(text, i)
            if isinstance(obj, dict):
                out.append(obj)
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return out


def _close_json(frag: str) -> str | None:
    """Close a truncated JSON fragment (open string, arrays, objects). None if brackets are mismatched."""
    stack, in_str, esc = [], False, False
    for i, ch in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                return frag[: i + 1]
    return frag + ('"' if in_str else "") + "".join(reversed(stack))


def _repair(text: str, accept: Callable[[dict], bool] | None) -> dict | None:
    """Best effort for output cut off mid-JSON (max tokens): close it, cutting back to earlier commas if needed."""
    start = text.find("{")
    while start != -1:
        s = text[start:]
        cuts = [len(s)] + [i for i in range(len(s) - 1, 0, -1) if s[i] == ","][:40]
        for cut in cuts:
            closed = _close_json(s[:cut].rstrip().rstrip(","))
            if closed is None:
                continue
            try:
                obj = json.loads(closed)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and (accept is None or accept(obj)):
                return obj
        start = text.find("{", start + 1)
    return None


def find_json(text: str | None, accept: Callable[[dict], bool] | None = None, *, harmony: bool = True,
              strip_fences: bool = False, repair: bool = False) -> dict | None:
    """The last JSON object `accept`ed (default: any object) in the first candidate text that has one; None if none.

    Candidates (harmony=True): the final channel with reasoning blocks (<think>, <analysis>, ...) removed (and ```json
    fences with strip_fences), the final channel, the leaked analysis channel, the raw text; harmony=False: the raw text
    only. repair: a truncated object in one of the first two candidates is closed as a last resort (max tokens)."""
    text = text or ""
    if harmony:
        final, analysis = split_harmony(text)  # leaked gpt-oss channel markers
        first = _THOUGHT_RE.sub("", final)
        cands = [_FENCE.sub("", first) if strip_fences else first, final] + ([analysis] if analysis else []) + [text]
    else:
        cands = [text]
    for c in cands:
        objs = [o for o in _json_objects(c) if accept is None or accept(o)]
        if objs:
            return objs[-1]
    if repair:
        for c in cands[:2]:
            if (obj := _repair(c, accept)) is not None:
                return obj
    return None


def as_list(x: object) -> list:
    """A JSON field as a list: None / "" -> [], a list or tuple -> a list, anything else -> [x]."""
    if x is None or x == "":
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def extract_action_json(text: str) -> dict | None:
    """The last JSON object that looks like an action (has 'type' and 'content'); reasoning blocks are ignored first."""
    return find_json(text, lambda o: "type" in o and "content" in o)


def parse_action(text: str) -> tuple[Action, list[dict], float] | None:
    """Extracts (action, ddx, confidence) from the LLM output. Returns None on failure."""
    obj = extract_action_json(text)
    if obj is None:
        return None
    try:
        action = Action(ActionType(str(obj["type"]).upper()), str(obj["content"]).strip(), str(obj.get("reason", "")).strip())
    except ValueError:
        return None
    if not action.content:
        return None
    ddx = obj.get("ddx") if isinstance(obj.get("ddx"), list) else []
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return action, ddx, conf
