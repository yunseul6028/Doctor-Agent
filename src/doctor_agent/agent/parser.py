import json
import re

from doctor_agent.env.interface import Action, ActionType
from doctor_agent.llm.harmony import split_harmony

# Reasoning blocks some models put in the content (Gemma <thought>, <think>, gpt-oss analysis leaks)
_THOUGHT_RE = re.compile(r"<(thought|think|thinking|analysis)>.*?(</\1>|$)", re.S | re.I)
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


def extract_action_json(text: str) -> dict | None:
    """The last JSON object that looks like an action (has 'type' and 'content'); reasoning blocks are ignored first."""
    text = text or ""
    final, analysis = split_harmony(text)  # leaked gpt-oss channel markers
    candidates = (_THOUGHT_RE.sub("", final), final) + ((analysis,) if analysis else ()) + ((text,) if final != text else ())
    for candidate in candidates:
        objs = [o for o in _json_objects(candidate) if "type" in o and "content" in o]
        if objs:
            return objs[-1]
    return None


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
