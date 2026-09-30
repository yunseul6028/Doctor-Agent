"""agent/parser.find_json: the one JSON extractor behind the step / final action, the pre-diagnosis review and the
sub-agent readers (consult / advocate with fence stripping + truncation repair, radiology). One test per former
caller's edge cases."""
import json

from doctor_agent.agent import parser as P
from doctor_agent.agent.subagents import consult
from doctor_agent.agent.subagents.advocate import parse_advocate
from doctor_agent.agent.subagents.orchestrator import parse_radiology

ACT = '{"type": "ASK", "content": "열이 있나요?"}'
LEAK = "<|channel|>analysis<|message|>{A}<|end|><|start|>assistant<|channel|>final<|message|>{F}<|return|>"


def test_action_prefers_final_channel_and_skips_reasoning_blocks():
    other = '{"type": "TEST", "content": "CT"}'
    assert P.extract_action_json(LEAK.format(A=other, F=ACT))["content"] == "열이 있나요?"
    assert P.extract_action_json(f"<think>{other}</think>{ACT}")["content"] == "열이 있나요?"
    # only the analysis channel carries the action (cut off before the final channel)
    assert P.extract_action_json("<|channel|>analysis<|message|>결론: " + ACT)["type"] == "ASK"
    # the last action-shaped object wins; objects without type+content are ignored
    assert P.extract_action_json(f'{other} {ACT} {{"x": 1}}')["content"] == "열이 있나요?"
    assert P.extract_action_json("그냥 글") is None and P.extract_action_json(None) is None


def test_find_json_accept_and_harmony_off():
    text = LEAK.format(A='{"key_findings": []}', F='{"confirmation": "CT"}')
    assert P.find_json(text, lambda o: "key_findings" in o) == {"key_findings": []}  # falls back to the analysis
    # harmony=False reads the raw text only: the last matching object in it
    assert P.find_json(text, lambda o: "key_findings" in o or "confirmation" in o, harmony=False) == {"confirmation": "CT"}
    assert P.find_json("no json", harmony=False) is None
    assert P.find_json('{"a": 1} {"b": 2}') == {"b": 2}


def test_repair_closes_truncated_objects_only_when_asked():
    cut = '{"assessment": "심근경색 의심", "next_actions": [{"type": "TEST", "content": "트로포닌"}, {"type": "TE'
    keys = frozenset(consult.CONSULT_SCHEMA["properties"])
    assert P.find_json(cut, lambda o: bool(keys & set(o))) is None
    fixed = P.find_json(cut, lambda o: bool(keys & set(o)), repair=True)
    assert fixed["assessment"] == "심근경색 의심" and fixed["next_actions"][0]["content"] == "트로포닌"
    assert P._close_json('{"a": "x\\"y') == '{"a": "x\\"y"}' and P._close_json('{"a": ]') is None


def test_consult_extract_json_fences_repair_and_never_raises():
    keys = frozenset(consult.CONSULT_SCHEMA["properties"])
    obj = {"assessment": "a", "next_actions": [{"type": "ASK", "content": "q"}]}
    s = json.dumps(obj, ensure_ascii=False)
    assert consult.extract_json("```json\n" + s + "\n```", keys) == obj
    assert consult.extract_json("앞말 " + s[:-10], keys)["assessment"] == "a"  # truncated
    assert consult.extract_json('{"x": 1}', keys) is None and consult.extract_json(None, keys) is None
    assert consult.extract_json("[" * 100_000, keys) is None  # pathological input: no exception
    assert parse_advocate('<think>{"verdict": "유지"}</think>{"verdict": "재검토", "note": "근거 부족"}', "A").ok


def test_radiology_reader_uses_last_items_or_summary_object():
    items = {"items": [{"finding": "기흉", "status": "있음", "critical": True}], "summary": "기흉"}
    res = parse_radiology("analysis 생각 assistantfinal" + json.dumps(items, ensure_ascii=False) + ' {"x": 1}')
    assert res.ok and res.hint_ko == "기흉"
    assert not parse_radiology("판독 불가").ok


def test_as_list():
    assert P.as_list(None) == [] and P.as_list("") == [] and P.as_list("a") == ["a"]
    assert P.as_list([1, 2]) == [1, 2] and P.as_list((1,)) == [1] and P.as_list({"a": 1}) == [{"a": 1}]
