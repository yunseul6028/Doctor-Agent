"""'반대 의견 담당' (devil's advocate) sub-agent: a diagnostic time-out before the diagnosis is submitted.

Owned by clinical-strategist. Same doctor LLM, one call: given the proposed diagnosis and the grounded case summary,
it names the two best alternative explanations, the positive findings the leading diagnosis does not explain, the one
result that would refute it, and dangerous diagnoses not yet excluded. It never decides; the framework decides what to
do with the result (interface in `base.py`).

Grounding: the idea of a deliberate "diagnostic time-out" / forced consideration of alternatives against premature
closure and anchoring comes from the cognitive-debiasing literature (Croskerry 2003; Croskerry, Singhal & Mamede 2013;
Ely, Graber & Croskerry 2011 checklists; Graber 2005 on premature closure as the most common cognitive error). The
questions and wording are ours. Bibliographic data checked against PubMed E-utilities esummary on 2026-09-28; ledger
rows in docs/licenses.md. Stdlib only; no LLM call here.
"""
from __future__ import annotations

from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult
from doctor_agent.agent.subagents.consult import (
    HINT_MAX_CHARS,
    _cap,
    _s,
    case_context,
    extract_json,
    fit_lines,
    norm_actions,
    norm_named,
    norm_strings,
    patient_profile,
    render_profile,
    render_resources,
)
from doctor_agent.knowledge.clinical_rules import Citation
from doctor_agent.safety.protocols import cant_miss_for

C_CROSKERRY_2003 = Citation(
    "Croskerry P.", "The importance of cognitive errors in diagnosis and strategies to minimize them",
    "Acad Med", 2003, "78(8):775-780", doi="10.1097/00001888-200308000-00003", pmid="12915363", verified=True,
)
C_DEBIASING_1 = Citation(
    "Croskerry P, Singhal G, Mamede S.", "Cognitive debiasing 1: origins of bias and theory of debiasing",
    "BMJ Qual Saf", 2013, "22 Suppl 2:ii58-ii64", doi="10.1136/bmjqs-2012-001712", pmid="23882089", verified=True,
)
C_DEBIASING_2 = Citation(
    "Croskerry P, Singhal G, Mamede S.", "Cognitive debiasing 2: impediments to and strategies for change",
    "BMJ Qual Saf", 2013, "22 Suppl 2:ii65-ii72", doi="10.1136/bmjqs-2012-001713", pmid="23996094", verified=True,
)
C_CHECKLISTS = Citation(
    "Ely JW, Graber ML, Croskerry P.", "Checklists to reduce diagnostic errors",
    "Acad Med", 2011, "86(3):307-313", doi="10.1097/ACM.0b013e31820824cd", pmid="21248608", verified=True,
)
C_GRABER_2005 = Citation(
    "Graber ML, Franklin N, Gordon R.", "Diagnostic error in internal medicine",
    "Arch Intern Med", 2005, "165(13):1493-1499", doi="10.1001/archinte.165.13.1493", pmid="16009864", verified=True,
)
CITATIONS: tuple[Citation, ...] = (C_CROSKERRY_2003, C_DEBIASING_1, C_DEBIASING_2, C_CHECKLISTS, C_GRABER_2005)

NAME = "advocate"

ADVOCATE_SYSTEM = """당신은 '반대 의견 담당' 의사입니다. 주치의가 제출하려는 진단에 대해 진단 타임아웃을 합니다.
목표는 한 진단에 일찍 꽂히는 오류(조기 종결·고착)를 잡는 것입니다. 억지로 반대하지 말고, 근거가 충분하면 "유지"라고 하세요.

규칙:
1. [처음 정보], [소견 장부], [최근 대화]에 있는 소견만 인용하세요. 없는 검사 결과나 소견을 지어내지 마세요.
2. alternatives: 같은 소견을 설명할 수 있는 다른 진단 2개(제안 진단과 다른 질환)와 각각 근거 소견.
3. unexplained: 제안 진단으로 설명되지 않거나 맞지 않는 양성 소견. 단순히 "없는" 소견은 넣지 마세요. 없으면 빈 목록.
4. refuting_test: 제안 진단이 틀렸다면 그것을 드러낼 단 하나의 문진·진찰·검사. 이미 한 것과 결과 없는 검사는 제외. type은 ASK, EXAM, TEST 중 하나. 필요 없으면 null.
5. dangers_not_excluded: 아직 확인·배제하지 않은 위험 질환만.
6. verdict: "유지"(제안 진단이 충분히 뒷받침됨) 또는 "재검토"(위 문제가 중요함). note에 이유 한 문장.
한국어로 짧게, JSON 한 줄만 출력하세요.
{"alternatives": [{"name": "...", "why": "..."}], "unexplained": ["..."], "refuting_test": {"type": "ASK|EXAM|TEST", "content": "...", "why": "..."}, "dangers_not_excluded": ["..."], "verdict": "유지|재검토", "note": "..."}"""

_ACTION = {"type": "object", "properties": {
    "type": {"type": "string", "enum": ["ASK", "EXAM", "TEST"]}, "content": {"type": "string"},
    "why": {"type": "string"}}, "required": ["type", "content"]}
ADVOCATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "alternatives": {"type": "array", "maxItems": 2, "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name"]}},
        "unexplained": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "refuting_test": {"anyOf": [_ACTION, {"type": "null"}]},
        "dangers_not_excluded": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["유지", "재검토"]},
        "note": {"type": "string"},
    },
    "required": ["alternatives", "verdict"],
}
VERDICTS = {"유지": "유지", "keep": "유지", "maintain": "유지", "accept": "유지",
            "재검토": "재검토", "reconsider": "재검토", "revise": "재검토", "review": "재검토"}


def build_advocate(state, proposed_dx: str, reason: str = "", resources: dict | None = None) -> SubagentCall:
    """One devil's-advocate call on `proposed_dx` (with the doctor's stated `reason`)."""
    blocks = [f"[제안 진단] {_cap(proposed_dx, 80)}" + (f"\n[주치의 근거] {_cap(reason, 200)}" if reason else ""),
              render_profile(patient_profile(state))]
    if cant := cant_miss_for(state.initial_info or ""):
        blocks.append("[이 주호소에서 배제할 위험 질환(안전 프로토콜)] " + ", ".join(cant))
    if res := render_resources(resources):
        blocks.append(res)
    blocks.append(case_context(state))
    blocks.append("위 제안 진단에 대한 반대 의견 JSON을 쓰세요.")
    return SubagentCall(name=NAME,
                        messages=[{"role": "system", "content": ADVOCATE_SYSTEM},
                                  {"role": "user", "content": "\n\n".join(blocks)}],
                        json_schema=ADVOCATE_SCHEMA, max_chars_out=HINT_MAX_CHARS)


def parse_advocate(text: str | None, proposed_dx: str = "", max_chars: int = HINT_MAX_CHARS) -> SubagentResult:
    """Parse the advocate answer. ok=False (empty hint) when no JSON object is found or it has no usable content.
    Alternatives equal to `proposed_dx` are dropped. raw keeps the object; raw["verdict"] is normalised to 유지/재검토
    ("" when missing). The hint only restates the JSON. Never raises."""
    try:
        obj = extract_json(text, frozenset(ADVOCATE_SCHEMA["properties"]))
        if obj is None:
            return SubagentResult(NAME, False, "", [], [], [], {})
        alts = norm_named(obj.get("alternatives"), [proposed_dx] if proposed_dx else None, n=2)
        unexplained = norm_strings(obj.get("unexplained"), n=4, width=60)
        actions = norm_actions(obj.get("refuting_test") if isinstance(obj.get("refuting_test"), dict)
                               else obj.get("refuting_test") or [], n=1)
        dangers = norm_strings(obj.get("dangers_not_excluded"))
        verdict = VERDICTS.get(_s(obj.get("verdict"), 10).lower(), "")
        note = _s(obj.get("note"), 120)
        obj = dict(obj, verdict=verdict)
        if not (alts or unexplained or actions or dangers or verdict):
            return SubagentResult(NAME, False, "", [], [], [], obj)
        head = "[반대 의견]" + (f" 제안 진단 '{_cap(proposed_dx, 40)}'" if proposed_dx else "")
        head += f": {verdict}" if verdict else ""
        head += f" — {note}" if note else ""
        lines = [head]
        if unexplained:
            lines.append("- 설명 안 되는 소견: " + ", ".join(unexplained))
        if actions:
            a = actions[0]
            lines.append(f"- 반증할 한 가지: {a['type']} {a['content']}" + (f" — {a['why']}" if a["why"] else ""))
        if dangers:
            lines.append("- 아직 배제 안 된 위험: " + ", ".join(dangers))
        if alts:
            lines.append("- 대안 진단: " + ", ".join(d["name"] + (f"({d['why']})" if d["why"] else "") for d in alts))
        return SubagentResult(NAME, True, fit_lines(lines, max_chars), alts, actions, dangers, obj)
    except Exception:  # noqa: BLE001 - parser must never raise
        return SubagentResult(NAME, False, "", [], [], [], {})
