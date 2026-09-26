import json
import re

from doctor_agent.env.interface import Action, ActionType

# Reasoning blocks some models put in the content (Gemma <thought>, <think>, gpt-oss analysis leaks)
_THOUGHT_RE = re.compile(r"<(thought|think|thinking|analysis)>.*?(</\1>|$)", re.S | re.I)
_DECODER = json.JSONDecoder()


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
    for candidate in (_THOUGHT_RE.sub("", text), text):
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
