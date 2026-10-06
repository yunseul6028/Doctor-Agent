"""Measure the real gpt-oss token counts of the doctor prompts offline (no LLM or API calls).

Cases are replayed through the real `Policy` (hints, ledgers, capped view, safety layers, KB hints) with a scripted
doctor that never diagnoses early, so every case runs the full 60 turns against the keyword patient. Every prompt the
policy builds is captured (step prompts including retries, the forced final prompt at turn 60), and at checkpoint turns
the pre-diagnosis review prompt and the final-diagnosis prompt are probed on a copy of the state. Each prompt is
rendered in the harmony chat format exactly as the gpt-oss chat template does (system message with the reasoning
effort, our system prompt as the developer "# Instructions", user message, `<|start|>assistant` generation prefix) and
counted with the gpt-oss tokenizer.

    python scripts/token_budget.py                               # all data/cases_aug cases, tiktoken o200k_harmony
    python scripts/token_budget.py --limit 20 --json-out eval/results/token_budget.json
    python scripts/token_budget.py --tokenizer hf:/path/to/tokenizer.json   # the Hugging Face gpt-oss-20b tokenizer
    python scripts/token_budget.py --subagents --jobs 8          # sub-agent prompts + per-case totals, off vs on

Sub-agent mode (--subagents): the scripted doctor narrows its DDx to one specialty once 5 turns are done and keeps
its confidence at 0.3, so the routed / stuck consult can fire; sub-agent prompts get plausible JSON answers so their
hints reach later main prompts. Each case runs per case length (--diagnose-at: 60 = never diagnoses, 20, 10) with
sub-agents off, on (calibrated triggers) and on with the pre-review advocate forced. Reported: every sub-agent prompt
probed at turns 6/10/20/40/57 (consult for all specialties, advocate, radiology on the longest result and on all
result texts up to the 3,000-char cap), calls and prompt tokens per case, and the predicted wall time per case from
estimate_call_s (below) under the throughput assumptions (CONSERVATIVE / MODERATE or --prefill-tps,
--decode-tps, --overhead-s, --reasoning-tokens).

Tokenizer (dev only, requirements-dev.txt): `tiktoken` encoding `o200k_harmony` (MIT; the BPE file is downloaded once
into the tiktoken cache). It is the encoding `openai-harmony` uses for gpt-oss and gives identical ids to the
`openai/gpt-oss-20b` tokenizer.json (Apache-2.0) on our prompts (checked 2026-09-28).

This is a measurement tool: it is not part of the agent package and it does not change the agent.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent import prompts  # noqa: E402
from doctor_agent.agent.policy import Policy  # noqa: E402
from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.config import AgentConfig, LLMConfig  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402

CHECKPOINTS = (1, 5, 10, 20, 40, 60)
# gpt-oss-20b config.json max_position_embeddings (128k, YaRN). The serving side may configure less (vLLM
# --max-model-len, Ollama num_ctx), so the report also checks smaller budgets.
CONTEXT_WINDOW = 131_072
BUDGETS = (4096, 8192, 16384)
DEFAULT_DATE = "2026-09-28"  # the template puts today's date in the system message; the digit count is what matters


# ---------------------------------------------------------------------------------------------- time model
# (offline only; moved here from agent/runtime.py on 2026-09-30: the agent never estimates call times)
@dataclass(frozen=True)
class Throughput:
    """Serving-speed ASSUMPTIONS for gpt-oss-20b on a serving endpoint (nothing here is measured or sourced: the
    server, GPU, batching and prefix caching depend on the deployment). The defaults are deliberately
    slow so that a budget planned with them has headroom; replace them with numbers measured on the target server
    (GuardedLLM.stats()["latency_main_s"] + the prompt token counts of scripts/token_budget.py).

    prefill_tps: prompt tokens processed per second (no prefix-cache credit: the system prompt is re-counted per call)
    decode_tps: generated tokens per second for one request
    overhead_s: fixed cost per call (HTTP, queueing, scheduling)
    reasoning_tokens: hidden analysis-channel tokens per call at reasoning effort "low", on top of the visible JSON"""
    prefill_tps: float = 1000.0
    decode_tps: float = 20.0
    overhead_s: float = 1.0
    reasoning_tokens: int = 300


CONSERVATIVE = Throughput()
# a second, faster ASSUMPTION for the what-if tables (also unmeasured)
MODERATE = Throughput(prefill_tps=4000.0, decode_tps=60.0, overhead_s=0.5, reasoning_tokens=300)


def estimate_call_s(prompt_tokens: float, output_tokens: float, tp: Throughput = CONSERVATIVE,
                    reasoning_tokens: float | None = None) -> float:
    """Predicted wall time of one call: overhead + prompt / prefill_tps + (output + reasoning) / decode_tps."""
    reasoning = tp.reasoning_tokens if reasoning_tokens is None else reasoning_tokens
    return (tp.overhead_s + max(0.0, prompt_tokens) / max(1e-9, tp.prefill_tps)
            + (max(0.0, output_tokens) + max(0.0, reasoning)) / max(1e-9, tp.decode_tps))


def estimate_case_s(calls: Iterable[tuple[float, float]], tp: Throughput = CONSERVATIVE) -> float:
    """Predicted wall time of one case = sum over its sequential calls of estimate_call_s(prompt, output). The agent
    makes its calls one after another (sub-agents included), so the times add up; CPU work between calls is ignored
    (measured separately by tests/perf.py)."""
    return sum(estimate_call_s(p, o, tp) for p, o in calls)


# ------------------------------------------------------------------------------------------------ harmony rendering

MODEL_IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."


def harmony_system(reasoning_effort: str = "low", date: str = DEFAULT_DATE, knowledge_cutoff: str = "2024-06") -> str:
    """Body of the harmony system message the gpt-oss chat template writes (no tools)."""
    effort = reasoning_effort if reasoning_effort and reasoning_effort != "none" else "medium"  # template default
    return (f"{MODEL_IDENTITY}\nKnowledge cutoff: {knowledge_cutoff}\nCurrent date: {date}\n\nReasoning: {effort}\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.")


def render_harmony(messages: list[dict], reasoning_effort: str = "low", date: str = DEFAULT_DATE) -> str:
    """The prompt string gpt-oss sees for our single-turn messages: system message, developer message (our `system`
    role content under "# Instructions"), user message(s) and the `<|start|>assistant` generation prefix.
    Identical to `openai-harmony` render_conversation_for_completion (what vLLM uses); the HF chat_template.jinja adds
    "\\n\\n" after the instructions (0-1 token difference). Checked 2026-09-28."""
    out = f"<|start|>system<|message|>{harmony_system(reasoning_effort, date)}<|end|>"
    rest = list(messages)
    if rest and rest[0].get("role") in ("system", "developer"):
        out += f"<|start|>developer<|message|># Instructions\n\n{rest.pop(0)['content']}<|end|>"
    for m in rest:
        if m.get("role") == "user":
            out += f"<|start|>user<|message|>{m['content']}<|end|>"
        elif m.get("role") == "assistant":  # final-channel answer (not used by the agent; kept for completeness)
            out += f"<|start|>assistant<|channel|>final<|message|>{m['content']}<|end|>"
    return out + "<|start|>assistant"


# ------------------------------------------------------------------------------------------------ token counters

class TiktokenCounter:
    """tiktoken `o200k_harmony` (gpt-oss). Harmony control tokens (<|start|> etc.) count as one token each."""

    def __init__(self, encoding: str = "o200k_harmony"):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def __call__(self, text: str) -> int:
        return len(self.enc.encode(text, allowed_special="all"))


class HFTokenizerCounter:
    """A Hugging Face tokenizer.json (e.g. openai/gpt-oss-20b), via the `tokenizers` package."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)
        self.name = f"hf:{Path(path).name}"

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


class CharCounter:
    """Rough fallback (no tokenizer installed): gpt-oss spends ≈0.55 tokens per Korean-heavy character on our prompts.
    Only for smoke tests; the report says which counter was used."""

    name = "chars*0.55 (estimate)"

    def __call__(self, text: str) -> int:
        return math.ceil(len(text) * 0.55)


def get_counter(spec: str = "tiktoken") -> Callable[[str], int]:
    if spec.startswith("hf:"):
        return HFTokenizerCounter(spec[3:])
    if spec == "chars":
        return CharCounter()
    if spec.startswith("tiktoken"):
        return TiktokenCounter(spec.split(":", 1)[1] if ":" in spec else "o200k_harmony")
    raise SystemExit(f"unknown --tokenizer {spec!r} (tiktoken | tiktoken:<encoding> | hf:<tokenizer.json> | chars)")


# ------------------------------------------------------------------------------------------------ prompt anatomy

_VIEW_SECTIONS = {"[처음 정보]": "view.initial", "[소견 장부]": "view.findings", "[감별 진단 장부]": "view.ddx",
                  "[이전 행동]": "view.older_actions", "[최근 대화]": "view.recent"}
_VIEW_SPLIT = re.compile(r"\n\n(?=\[(?:소견 장부|감별 진단 장부|이전 행동|최근 대화)\])")

HINT_TYPES = [  # (prefix, label) in the order Policy._hints builds them, then the retry hints
    ("반드시 배제할 위험 질환", "hint.cant_miss"),
    ("아직 안 한 최소 안전 확인", "hint.safety_checks"),
    ("결과가 제공되지 않은 요청", "hint.unavailable"),
    ("[임상 결정 규칙", "hint.clinical_rules"),
    (prompts.RESULT_HINT.split("(")[0], "hint.result_interp"),
    ("목표 턴 수", "hint.target_turns"),
    ("고려해 볼 다른 질환", "hint.kb_candidates"),
    ("감별 포인트", "hint.kb_discriminator"),
    (prompts.LOW_TIME_HINT[:12], "hint.low_time"),
    ("정해진 형식", "retry.parse"),
    ("진단 전에 아직 안 한", "retry.safety_pushback"),
    ("검토의 지적", "retry.review_hold"),
    ("참고: ", "retry.danger_confirmed"),
]


def classify_hint(hint: str) -> str:
    for prefix, label in HINT_TYPES:
        if hint.startswith(prefix):
            return label
    if "이미 했습니다" in hint:
        return "retry.duplicate"
    if hint.endswith("다른 행동을 고르세요."):
        return "retry.precondition"
    return "hint.other"


def split_view(view: str) -> dict[str, str]:
    """{section label: text} for a CaseState.view() string (a hard-cut view is reported as one 'view.cut' part)."""
    out: dict[str, str] = {}
    if "\n…\n" in view:  # step 5 of CaseState.view(): head of the initial info + "\n…\n" + newest text
        return {"view.cut": view}
    for chunk in _VIEW_SPLIT.split(view):
        label = next((lab for head, lab in _VIEW_SECTIONS.items() if chunk.startswith(head)), "view.other")
        out[label] = out.get(label, "") + chunk
    return out


def anatomy(kind: str, messages: list[dict], count: Callable[[str], int], effort: str, *, view: str = "",
            hints: list[str] | None = None, extras: dict[str, str] | None = None) -> dict:
    """Token breakdown of one prompt. `total` is the exact count of the rendered harmony prompt; the parts are counted
    one by one (BPE merges across part borders make their sum differ from `total` by a few tokens)."""
    total = count(render_harmony(messages, effort))
    system_txt = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    user_txt = "".join(m["content"] for m in messages if m.get("role") == "user")
    parts: dict[str, int] = {"system_prompt": count(system_txt)}
    for label, text in split_view(view).items():
        parts[label] = count(text)
    for h in hints or []:
        label = classify_hint(h)
        parts[label] = parts.get(label, 0) + count("\n- " + h)
    for label, text in (extras or {}).items():
        if text:
            parts[label] = count(text)
    parts["harmony_overhead"] = total - count(system_txt) - count(user_txt)
    listed = sum(v for k, v in parts.items() if k != "harmony_overhead")
    parts["user_template"] = max(0, total - parts["harmony_overhead"] - listed)
    return {"kind": kind, "tokens": total, "user_tokens": count(user_txt), "chars": len(system_txt) + len(user_txt),
            "view_chars": len(view), "parts": parts}


# ------------------------------------------------------------------------------------------------ scripted doctor

PADDING = [  # generic requests after the case's own items are used up (mostly answered "제공되지 않습니다")
    ("ASK", "최근 해외여행을 다녀오신 적이 있나요?"), ("TEST", "갑상선기능검사(TSH, free T4)"),
    ("ASK", "가족 중에 비슷한 병을 앓은 분이 있나요?"), ("TEST", "혈액 배양 검사"), ("EXAM", "직장수지검사"),
    ("ASK", "술은 일주일에 얼마나 드시나요?"), ("TEST", "뇌 MRI"), ("ASK", "담배는 피우시나요?"),
    ("TEST", "동맥혈가스분석"), ("EXAM", "피부 병변 자세히 관찰"), ("ASK", "최근에 새로 시작한 약이 있나요?"),
    ("TEST", "혈청 페리틴"), ("ASK", "밤에 식은땀이 나나요?"), ("TEST", "복부 CT 조영증강"),
    ("EXAM", "안저 검사"), ("ASK", "소변 색깔이 변했나요?"), ("TEST", "D-dimer"), ("ASK", "잠은 잘 주무시나요?"),
    ("TEST", "혈청 칼슘과 인"), ("EXAM", "갑상선 촉진 재확인"), ("ASK", "직업이 무엇인가요?"), ("TEST", "비타민 B12"),
    ("ASK", "성생활과 관련해 걱정되는 점이 있나요?"), ("TEST", "HIV 항체 검사"), ("EXAM", "림프절 전신 촉진"),
    ("ASK", "체중이 최근에 변했나요?"), ("TEST", "심장초음파"), ("ASK", "반려동물을 키우시나요?"),
    ("TEST", "혈중 암모니아"), ("EXAM", "관절 가동 범위 검사"), ("ASK", "예방접종은 다 맞으셨나요?"),
    ("TEST", "요 단백/크레아티닌 비"), ("ASK", "증상이 하루 중 언제 심한가요?"), ("TEST", "척추 MRI"),
    ("EXAM", "보행 관찰"), ("ASK", "최근에 다친 적이 있나요?"), ("TEST", "혈청 단백 전기영동"),
    ("ASK", "음식을 삼킬 때 불편한가요?"), ("TEST", "위내시경"), ("EXAM", "음낭 진찰"), ("ASK", "기분이 우울하신가요?"),
    ("TEST", "대장내시경"), ("ASK", "숨이 찬 적이 있나요?"), ("TEST", "PET-CT"), ("EXAM", "유방 진찰"),
]
_DDX_POOL = ["급성 위장염", "폐렴", "울혈성 심부전", "갑상선기능저하증", "철결핍빈혈", "요로감염", "전신홍반루푸스",
             "결핵", "림프종", "당뇨병"]


def plan_actions(case: dict) -> list[tuple[str, str]]:
    """Requests the keyword patient can answer (first alias of each case key), history first, then exams and tests,
    followed by the generic PADDING list."""
    def first(key: str) -> str:
        return key.split("|")[0].strip()

    def question(key: str, i: int) -> str:  # alias-heavy wording: a shared template would trip the dedupe check
        aliases = [a.strip() for a in key.split("|") if a.strip()][:3]
        return ", ".join(aliases) + " " + ("어떠세요?", "있으셨나요?", "말씀해 주세요.")[i % 3]

    asks = [("ASK", question(k, i)) for i, k in enumerate(case.get("history", {}))]
    exams = [("EXAM", first(k)) for k in case.get("exam", {})]
    tests = [("TEST", first(k)) for k in case.get("tests", {})]
    mixed, i = [], 0
    while exams or tests:  # alternate exams and tests after the history, like a typical work-up
        src = exams if (i % 2 == 0 and exams) or not tests else tests
        mixed.append(src.pop(0))
        i += 1
    return asks + mixed + PADDING


# One specialty's diagnoses (Korean names that knowledge/specialty.specialty_of maps to that id; checked by
# tests/test_token_budget.py). The focused scripted DDx uses them so the routed consult can fire.
SPECIALTY_POOL: dict[str, list[str]] = {
    "cardio": ["급성 심근경색", "대동맥 박리", "심방세동", "울혈성 심부전", "급성 심낭염"],
    "resp_id": ["지역사회획득 폐렴", "천식", "만성폐쇄성폐질환", "폐결핵", "급성 기관지염"],
    "gi_liver": ["급성 충수염", "급성 췌장염", "담낭염", "위궤양", "급성 간염"],
    "neuro": ["허혈성 뇌졸중", "지주막하출혈", "편두통", "세균성 수막염", "길랭-바레 증후군"],
    "rheum_immune": ["전신홍반루푸스", "류마티스 관절염", "통풍", "거대세포 동맥염", "강직성 척추염"],
    "peds_obgyn": ["자궁외 임신", "전자간증", "난소 염전", "태반 조기박리", "골반염"],
    "heme_onc": ["철결핍빈혈", "급성 골수성 백혈병", "림프종", "다발골수종", "혈전성 혈소판감소성 자반증"],
    "renal_uro": ["급성 신우신염", "요로결석", "급성 신손상", "신증후군", "고환 염전"],
    "endo_metab": ["당뇨병케톤산증", "갑상선중독증", "부신기능저하증", "고칼슘혈증", "갑상선기능저하증"],
    "psych": ["주요우울장애", "양극성 장애", "공황장애", "알코올 금단", "조현병"],
}


def focus_specialty(case: dict) -> str:
    """The specialty of the case's gold diagnosis, or a fixed pick from the initial text when it maps to none."""
    from doctor_agent.knowledge.specialty import specialty_of

    try:
        sp = specialty_of(str(case.get("diagnosis") or ""))
    except Exception:  # noqa: BLE001
        sp = None
    if sp in SPECIALTY_POOL:
        return sp
    keys = list(SPECIALTY_POOL)
    return keys[sum(map(ord, str(case.get("initial", "")))) % len(keys)]


def subagent_name(messages: list[dict]) -> str:
    """"consult:<id>" / "advocate" / "radiology" from the sub-agent's system prompt ("subagent" if unknown)."""
    sys_ = messages[0].get("content", "") if messages else ""
    if sys_ == prompts.RESULT_INTERPRETER_PROMPT:
        return "radiology"
    try:
        from doctor_agent.agent.subagents import advocate, consult

        if sys_ == advocate.ADVOCATE_SYSTEM:
            return "advocate"
        for sid, spec in consult.SPECIALTIES.items():
            if sys_ == consult.CONSULT_SYSTEM.format(name_ko=spec.name_ko, role_ko=spec.role_ko):
                return f"consult:{sid}"
    except Exception:  # noqa: BLE001 — content module missing: unnamed
        pass
    return "subagent"


def scripted_subagent_answer(name: str, dx: list[str], messages: list[dict]) -> dict:
    """Short but complete answers in each sub-agent's schema (consult.CONSULT_SCHEMA, advocate.ADVOCATE_SCHEMA,
    orchestrator.RADIOLOGY_SCHEMA). Real gpt-oss answers may be shorter or longer; only the visible JSON is counted."""
    top, second = dx[0], dx[1] if len(dx) > 1 else "다른 질환"
    danger = dx[3] if len(dx) > 3 else second
    if name.startswith("consult"):
        return {"assessment": f"현재 소견은 {top}에 가장 합당하나 {second}과 {danger}의 배제가 아직 부족합니다.",
                "ddx_add": [{"name": second + " 합병증", "why": "현재 소견 일부를 설명할 수 있음"}],
                "missed_dangers": [danger],
                "next_actions": [
                    {"type": "TEST", "content": "혈액 배양 검사", "why": f"{top}와 {second}를 가장 잘 가름"},
                    {"type": "EXAM", "content": "활력징후 재측정", "why": "불안정 여부 확인"},
                    {"type": "ASK", "content": "증상이 언제 가장 심한가요?", "why": "경과 확인"}],
                "confidence_note": "핵심 검사 결과가 나오기 전까지는 확신도가 중간 정도입니다."}
    if name == "advocate":
        return {"alternatives": [{"name": second, "why": "발열과 기침을 함께 설명함"},
                                 {"name": danger, "why": "아직 배제하지 않은 위험 질환"}],
                "unexplained": ["체중 감소"],
                "refuting_test": {"type": "TEST", "content": "흉부 CT", "why": f"{top}이 아니라면 다른 소견이 보임"},
                "dangers_not_excluded": [danger], "verdict": "재검토",
                "note": f"{top}을 뒷받침하는 확인 검사가 아직 없습니다."}
    if name == "radiology":
        user = messages[-1].get("content", "") if messages else ""
        body = user.split("\n", 1)[-1]
        phrases = [x.strip() for x in re.split(r"[,.\n]\s*", body) if len(x.strip()) > 4][:3] or ["특이 소견"]
        return {"items": [{"finding": ph[:30], "status": "있음" if i == 0 else "없음", "site": "", "value": "",
                           "critical": False} for i, ph in enumerate(phrases)],
                "normal": False, "unavailable": False, "summary": "; ".join(ph[:30] for ph in phrases)}
    return {"hint_ko": "추가 확인이 필요합니다."}


class ScriptedDoctor:
    """Stands in for gpt-oss: returns realistic action JSON (findings from the last response, a 5-entry DDx with
    for/against evidence, a question/exam/test from the plan) and never diagnoses until asked for a final answer.
    Captures the prompt builder arguments of every call so the prompt can be broken down (see capture_builders).

    Options for the sub-agent measurement (all off by default = the original replay):
    - focus_after: once this many turns are done, the DDx narrows to five diagnoses of one specialty
      (focus_specialty; the earlier candidates are marked 배제), so the routed consult can fire
    - confidence: model confidence in every step answer (< consult_low_conf arms the stuck-consult trigger)
    - diagnose_at: from this turn on every step answer proposes DIAGNOSE (top DDx), so the real safety pushback,
      gate, confidence pushback, pre-review advocate and review path runs and the case ends early
    - subagent_answers: answer consult / advocate / radiology prompts with plausible JSON (else "{}", a failed
      call), so their hints reach the following main prompts"""

    def __init__(self, case: dict, seed_dx: str = "", *, focus_after: int | None = None, confidence: float = 0.4,
                 diagnose_at: int | None = None, subagent_answers: bool = False):
        self.call_count = 0
        self.plan = plan_actions(case)
        self.pos = 0
        self.n_steps = 0
        self.last_proposed = ""
        self.dx = [seed_dx or case.get("diagnosis") or "원인 미상"]
        pool = [d for d in _DDX_POOL if d not in self.dx]
        start = sum(map(ord, str(case.get("initial", "")))) % len(pool)
        self.dx += (pool[start:] + pool[:start])[:4]
        self.focus_after, self.confidence, self.diagnose_at = focus_after, confidence, diagnose_at
        self.subagent_answers = subagent_answers
        self.focus = focus_specialty(case) if focus_after is not None else ""
        self.dropped: list[str] = []  # candidates ruled out when the DDx narrowed
        self.last_action: Action | None = None
        self.last_response = str(case.get("initial", ""))
        self.pending: list[tuple] = []  # builder captures not yet consumed by chat()
        self.calls: list[dict] = []  # {kind, turn, attempt, messages, view, hints, extras, output}
        self.turn = 0
        self.attempt = 0
        self.probe = False
        self.outputs: list[str] = []

    # --- what the model would answer
    def _maybe_focus(self) -> None:
        if self.focus_after is None or self.dropped or self.turn - 1 < self.focus_after:
            return
        names = SPECIALTY_POOL[self.focus]
        head = [self.dx[0]] if focus_specialty({"diagnosis": self.dx[0], "initial": ""}) == self.focus else []
        new = head + [n for n in names if n not in head][:5 - len(head)]
        self.dropped = [d for d in self.dx if d not in new]
        self.dx = new

    def _findings(self) -> list[dict]:
        item = (self.last_action.content if self.last_action else "주호소")[:20].rstrip(" ?")
        resp = self.last_response.strip()
        na = "제공되지 않습니다" in resp or resp.startswith("잘 모르겠")
        out = [{"item": item, "status": "결과없음" if na else "양성", "detail": "" if na else resp[:60]}]
        second = re.split(r"[,.]\s*", resp)
        if not na and len(second) > 2 and len(second[1]) > 4:
            out.append({"item": second[1][:20], "status": "음성" if "없" in second[1] else "양성", "detail": ""})
        return out

    def _ddx(self, finding: str) -> list[dict]:
        n = self.n_steps  # step answers only, so that probes never change the replay
        rows = []
        for i, name in enumerate(self.dx):
            status = "위험" if i == 3 else "배제" if i == 4 and n > 30 else "유력"
            p = round(max(0.05, 0.45 - 0.1 * i + (0.01 * (n % 5) if i == 0 else 0)), 2)
            rows.append({"dx": name, "p": p, "status": status,
                         "for": [finding] if (n + i) % 2 == 0 and i < 3 else [],
                         "against": [finding] if i >= 2 and n % 3 == 0 else []})
        rows += [{"dx": d, "p": 0.02, "status": "배제", "for": [], "against": []} for d in self.dropped]
        return rows

    def _step_json(self, refused: bool = False) -> dict:
        if not self.probe:
            self._maybe_focus()
        if self.diagnose_at is not None and self.turn >= self.diagnose_at and not self.probe:
            self.n_steps += 1
            f = self._findings()
            return {"findings": f, "ddx": self._ddx(f[0]["item"]), "type": "DIAGNOSE", "content": self.dx[0],
                    "reason": f"지금까지의 소견이 {self.dx[0]}에 가장 합당합니다.",
                    "confidence": max(self.confidence, 0.6)}
        if refused and self.last_proposed:  # "already done": like a model would, switch to another kind of action
            while self.pos < len(self.plan) - 1 and self.plan[self.pos][0] == self.last_proposed:
                self.pos += 1
        typ, content = self.plan[min(self.pos, len(self.plan) - 1)]
        self.pos += 1
        self.last_proposed = typ
        self.n_steps += 1
        f = self._findings()
        return {"findings": f, "ddx": self._ddx(f[0]["item"]), "type": typ, "content": content,
                "reason": f"{content[:20]}(으)로 {self.dx[0]}와 {self.dx[1]}을 감별하려고 합니다.",
                "confidence": self.confidence}

    def chat(self, messages: list[dict]) -> str:
        self.call_count += 1
        cap = self.pending.pop() if self.pending else ("other", "", [], {})
        kind, view, hints, extras = cap
        name = ""
        if kind == "review":
            out = {"key_findings": [{"finding": self.dx[0], "status": "설명됨"}], "contradicting": [],
                   "confirmation": "없음", "unresolved_danger": [self.dx[3]],
                   "next": {"type": "TEST", "content": "추가 검사", "reason": "위험 질환 배제"},
                   "final_diagnosis": "", "refine_evidence": ""}
        elif kind == "final":
            out = {"findings": [], "ddx": self._ddx("최종"), "type": "DIAGNOSE", "content": self.dx[0],
                   "reason": "지금까지의 소견으로 가장 가능성이 높음", "confidence": 0.6}
        elif kind == "other" and messages and messages[0].get("content") not in (prompts.SYSTEM, prompts.REVIEW_SYSTEM):
            # a specialist sub-agent call (no builder captured it, other role): the script itself is unchanged;
            # "{}" (a failed call) unless subagent_answers
            kind, name = "subagent", subagent_name(messages)
            out = scripted_subagent_answer(name, self.dx, messages) if self.subagent_answers else {}
        else:
            out = self._step_json(refused=bool(hints) and "이미 했습니다" in hints[-1])
        text = json.dumps(out, ensure_ascii=False)
        self.calls.append({"kind": kind, "name": name, "turn": self.turn, "attempt": self.attempt, "probe": self.probe,
                           "messages": messages, "view": view, "hints": list(hints), "extras": dict(extras),
                           "output": text})
        if kind == "step":
            self.attempt += 1
        return text


@contextmanager
def capture_builders(doctor: ScriptedDoctor, state_ref: dict):
    """Wrap the prompt builders so each call records (kind, view, hints, extras) for the anatomy breakdown."""
    orig = (prompts.build_step_messages, prompts.build_review_messages, prompts.build_final_messages)

    def step(view, turn, max_turns, hints, alert=""):
        doctor.pending.append(("step", view, list(hints), {"step.triage_alert": alert} if alert else {}))
        return orig[0](view, turn, max_turns, hints, alert)

    def review(view, diagnosis, reason, turn, max_turns):
        base, extras = split_review_view(view, state_ref["state"].view() if state_ref.get("state") is not None else "")
        extras["review.proposal"] = f"[제출하려는 진단] {diagnosis}\n[근거] {reason}"
        doctor.pending.append(("review", base, [], extras))
        return orig[1](view, diagnosis, reason, turn, max_turns)

    def final(view):
        doctor.pending.append(("final", view, [], {}))
        return orig[2](view)

    prompts.build_step_messages, prompts.build_review_messages, prompts.build_final_messages = step, review, final
    try:
        yield
    finally:
        prompts.build_step_messages, prompts.build_review_messages, prompts.build_final_messages = orig


def split_review_view(view: str, base: str) -> tuple[str, dict[str, str]]:
    """(state view, {review.kb_warning, review.criteria}) from the view Policy._review passes to the builder
    (state.view() + optional "\\n\\n[지식베이스 경고] …" + optional "\\n\\n[진단 기준 대조 …")."""
    if not base or not view.startswith(base):
        return view, {}
    tail = view[len(base):]
    kb, crit = "", ""
    at = tail.find("\n\n[진단 기준 대조")
    if at >= 0:
        tail, crit = tail[:at], tail[at:]
    if "[지식베이스 경고]" in tail:
        kb, tail = tail, ""
    return base + tail, {"review.kb_warning": kb, "review.criteria": crit}


# ------------------------------------------------------------------------------------------------ replay

def _env_for(case: dict):
    from eval.simulator import CaseFileEnvironment

    return CaseFileEnvironment(case)


def replay_case(case: dict, cfg: AgentConfig | None = None, checkpoints: tuple[int, ...] = CHECKPOINTS, *,
                doctor_opts: dict | None = None, sub_checkpoints: tuple[int, ...] = ()) -> list[dict]:
    """Runs one case with the scripted doctor (cfg.max_turns turns unless doctor_opts["diagnose_at"] ends it earlier).
    Returns every captured call: {kind, name, turn (1-based turn the prompt is for), attempt, probe, messages, view,
    hints, extras, output, view_chars_uncapped}. Probes (review / final at `checkpoints`, sub-agent prompts at
    `sub_checkpoints`) run on deep copies and never change the replay."""
    cfg = cfg or AgentConfig()
    env = _env_for(case)
    doctor = ScriptedDoctor(case, **(doctor_opts or {}))
    state = CaseState(initial_info=env.reset().text, view_max_chars=cfg.max_view_chars)
    policy = Policy(doctor, cfg)
    ref = {"state": state}
    with capture_builders(doctor, ref):
        while state.turn_count < cfg.max_turns:
            turn = state.turn_count + 1
            doctor.turn, doctor.attempt = turn, 0
            uncapped = len(state.view(max_chars=0))
            if turn in checkpoints and turn < cfg.max_turns:
                _probe(doctor, policy, state, cfg, ref, uncapped)
            if turn in sub_checkpoints and turn < cfg.max_turns:
                _probe_subagents(doctor, state, case, turn)
            n_before = len(doctor.calls)
            action = policy.next_action(state)
            for c in doctor.calls[n_before:]:
                c["view_chars_uncapped"] = uncapped
            obs = env.step(action)
            state.turns.append(Turn(action, obs.text, list(state.ddx)))
            doctor.last_action, doctor.last_response = action, obs.text
            if action.type == ActionType.DIAGNOSE:
                break
    return doctor.calls


def _probe(doctor: ScriptedDoctor, policy: Policy, state: CaseState, cfg: AgentConfig, ref: dict, uncapped: int) -> None:
    """Review and final prompts as they would look if the doctor proposed/forced a diagnosis at this turn."""
    snap = copy.deepcopy(state)
    probe = Policy(doctor, cfg)
    probe._kb_seen = set(policy._kb_seen)
    ref["state"] = snap
    doctor.probe = True
    n_before = len(doctor.calls)
    try:
        top = doctor.dx[0]
        probe._review(snap, Action(ActionType.DIAGNOSE, top, f"소견이 {top}에 합당함"))
        probe._final_diagnosis(copy.deepcopy(state))
    finally:
        doctor.probe = False
        ref["state"] = state
    for c in doctor.calls[n_before:]:
        c["view_chars_uncapped"] = uncapped


def _result_texts(case: dict) -> list[tuple[str, str]]:
    """(test name, result text) of the case's exams and tests, longest first."""
    items = [(k.split("|")[0].strip(), str(v)) for sec in ("tests", "exam") for k, v in (case.get(sec) or {}).items()
             if isinstance(v, str) and v.strip()]
    return sorted(items, key=lambda kv: -len(kv[1]))


def _probe_subagents(doctor: ScriptedDoctor, state: CaseState, case: dict, turn: int) -> None:
    """Every sub-agent prompt as it would look if it fired at this turn, built on a copy of the state by the real
    content modules: consult for each of the specialties (with knowledge/specialty.resources), the advocate on the
    top DDx, and the radiology reading of (a) the case's longest exam/test result and (b) all its result texts joined
    up to the orchestrator's RADIOLOGY_TEXT_CHARS cap (the worst the case allows). No LLM call: the scripted answer is
    recorded as the output."""
    from doctor_agent.agent import result_interpreter
    from doctor_agent.agent.subagents import advocate, consult
    from doctor_agent.agent.subagents.orchestrator import RADIOLOGY_TEXT_CHARS
    from doctor_agent.knowledge import specialty

    snap = copy.deepcopy(state)
    top = doctor.dx[0]
    calls: list[tuple[str, list[dict]]] = []
    for sid in consult.SPECIALTIES:
        calls.append((f"consult:{sid}", consult.build_consult(snap, sid, specialty.resources(sid, snap)).messages))
    calls.append(("advocate", advocate.build_advocate(snap, top, f"소견이 {top}에 합당함").messages))
    texts = _result_texts(case)
    if texts:
        joined = "\n".join(t for _, t in texts)[:RADIOLOGY_TEXT_CHARS]
        for label, (name, text) in (("radiology", texts[0]), ("radiology@cap", (texts[0][0], joined))):
            interp = result_interpreter.interpret(name, text, {"initial_info": state.initial_info})
            line = result_interpreter.render_for_prompt(
                interp, max(60, AgentConfig().result_hint_chars - len(prompts.RESULT_HINT.format(line=""))))
            calls.append((label, prompts.build_result_interpreter_messages(name[:40], text[:RADIOLOGY_TEXT_CHARS],
                                                                          line or "(없음)")))
    for name, messages in calls:
        out = scripted_subagent_answer(name.split("@")[0], doctor.dx, messages)
        doctor.calls.append({"kind": "subagent", "name": name, "turn": turn, "attempt": 0, "probe": True,
                             "messages": messages, "view": "", "hints": [], "extras": {},
                             "output": json.dumps(out, ensure_ascii=False),
                             "view_chars_uncapped": len(state.view(max_chars=0))})


# ------------------------------------------------------------------------------------------------ summary

def pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def dist(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1) if xs else 0.0, "p50": pct(xs, 50),
            "p95": pct(xs, 95), "max": max(xs) if xs else 0}


_BLOCK_HEAD = re.compile(r"^\[([^\]\s(:]+)")


def subagent_blocks(messages: list[dict]) -> dict[str, str]:
    """{label: text} of a sub-agent user message, split on blank lines and labelled by the leading "[header]"
    ("sub.<header>"; "sub.other" without one). Same-label blocks are joined."""
    user = "".join(m["content"] for m in messages if m.get("role") == "user")
    out: dict[str, str] = {}
    for block in user.split("\n\n"):
        m = _BLOCK_HEAD.match(block)
        label = f"sub.{m.group(1)}" if m else "sub.other"
        out[label] = out.get(label, "") + "\n\n" + block
    return out


def measure(cases: list[dict], count: Callable[[str], int], *, effort: str = "low", cfg: AgentConfig | None = None,
            checkpoints: tuple[int, ...] = CHECKPOINTS, progress: bool = False, doctor_opts: dict | None = None,
            sub_checkpoints: tuple[int, ...] = ()) -> list[dict]:
    """Anatomy rows for every captured call of every case. Sub-agent prompts are rendered with
    cfg.subagent_reasoning_effort (what runner.run sends) and broken down by their "[header]" blocks."""
    cfg = cfg or AgentConfig()
    sub_effort = getattr(cfg, "subagent_reasoning_effort", effort) or effort
    rows = []
    for i, case in enumerate(cases, 1):
        for c in replay_case(case, cfg, checkpoints, doctor_opts=doctor_opts, sub_checkpoints=sub_checkpoints):
            if c["kind"] == "subagent":
                a = anatomy("subagent", c["messages"], count, sub_effort, extras=subagent_blocks(c["messages"]))
            else:
                a = anatomy(c["kind"], c["messages"], count, effort, view=c["view"], hints=c["hints"],
                            extras=c["extras"])
            a.update(case=case.get("_id", i), turn=c["turn"], attempt=c["attempt"], probe=c["probe"],
                     name=c.get("name", ""), view_chars_uncapped=c.get("view_chars_uncapped", len(c["view"])),
                     output_tokens=count(c["output"]))
            rows.append(a)
        if progress:
            print(f"  {i}/{len(cases)} cases", file=sys.stderr, flush=True)
    return rows


def summarize(rows: list[dict], *, checkpoints: tuple[int, ...] = CHECKPOINTS, max_view_chars: int = 12000,
              max_tokens: int = 2048, retry_tokens: int = 4096) -> dict:
    """Distributions per prompt type and checkpoint turn, part contributions, cap/budget checks, per-turn mean curve."""
    def first_step(r):  # the first attempt of a real (non-probe) step prompt
        return r["kind"] == "step" and not r["probe"] and r["attempt"] == 0

    groups = {
        "step": [r for r in rows if first_step(r)],
        "step_retry": [r for r in rows if r["kind"] == "step" and not r["probe"] and r["attempt"] > 0],
        "review": [r for r in rows if r["kind"] == "review"],
        "final": [r for r in rows if r["kind"] == "final"],
    }
    by_turn: dict[str, dict] = {}
    for kind, rs in groups.items():
        if kind == "step_retry":
            continue
        by_turn[kind] = {}
        for t in checkpoints:
            sel = [r for r in rs if r["turn"] == t]
            if sel:
                by_turn[kind][str(t)] = {"tokens": dist([r["tokens"] for r in sel]),
                                         "view_chars": dist([r["view_chars"] for r in sel])}
    parts: dict[str, dict] = {}
    for kind, rs in groups.items():
        keys = sorted({k for r in rs for k in r["parts"]})
        parts[kind] = {k: dist([r["parts"].get(k, 0) for r in rs]) for k in keys}
        # share of the total prompt, and how often the part is present at all
        tot = sum(r["tokens"] for r in rs) or 1
        for k in keys:
            parts[kind][k]["share"] = round(sum(r["parts"].get(k, 0) for r in rs) / tot, 3)
            parts[kind][k]["present"] = round(sum(1 for r in rs if r["parts"].get(k)) / max(1, len(rs)), 3)
    everything = [r for r in rows]
    worst = max(everything, key=lambda r: r["tokens"]) if everything else None
    real = [r for r in rows if not r["probe"]]
    curve = {}
    for t in sorted({r["turn"] for r in groups["step"]}):
        sel = [r["tokens"] for r in groups["step"] if r["turn"] == t]
        curve[str(t)] = round(statistics.fmean(sel), 1)
    ratio = [r["tokens"] / max(1, r["chars"]) for r in groups["step"]]
    view_ratio = [sum(v for k, v in r["parts"].items() if k.startswith("view.")) / r["view_chars"]
                  for r in groups["step"] if r["view_chars"] >= 1000]
    return {
        "n_calls": len(rows), "n_real_calls": len(real),
        "by_turn": by_turn,
        "all": {k: dist([r["tokens"] for r in rs]) for k, rs in groups.items()},
        "parts": parts,
        "output_json_tokens": dist([r["output_tokens"] for r in groups["step"]]),
        # everything that is not the view: hints (incl. retry hints) and the fixed part (system prompt + wrappers)
        "hints_total": dist([sum(v for k, v in r["parts"].items() if k.startswith(("hint.", "retry.")))
                             for r in groups["step"] + groups["step_retry"]]),
        "fixed_total": dist([r["parts"]["system_prompt"] + r["parts"]["harmony_overhead"] + r["parts"]["user_template"]
                             for r in groups["step"]]),
        "tokens_per_char": dist([round(x, 3) for x in ratio]),
        "view_cap": {
            "max_view_chars": max_view_chars,
            "uncapped_view_chars": dist([r["view_chars_uncapped"] for r in groups["step"]]),
            "share_capped": round(sum(1 for r in groups["step"] if r["view_chars_uncapped"] > max_view_chars)
                                  / max(1, len(groups["step"])), 3) if max_view_chars else 0.0,
            "view_tokens_per_char": dist([round(x, 3) for x in view_ratio]),
            "view_tokens_at_cap_est": (round(max_view_chars * max(view_ratio)) if view_ratio and max_view_chars
                                       else None),
        },
        "budget": {
            "context_window": CONTEXT_WINDOW, "max_tokens": max_tokens, "retry_tokens": retry_tokens,
            "max_prompt": worst["tokens"] if worst else 0,
            "max_prompt_where": {k: worst[k] for k in ("kind", "turn", "attempt", "case")} if worst else {},
            "fits": {str(b): {"prompt_only": sum(r["tokens"] <= b for r in everything) / max(1, len(everything)),
                              "with_max_tokens": sum(r["tokens"] + max_tokens <= b for r in everything) / max(1, len(everything)),
                              "with_retry": sum(r["tokens"] + retry_tokens <= b for r in everything) / max(1, len(everything))}
                     for b in (*BUDGETS, CONTEXT_WINDOW)},
        },
        "step_mean_by_turn": curve,
        "per_call_mean_by_case_length": per_call_means(curve),
        "retry_rate": round(len(groups["step_retry"]) / max(1, len(groups["step"])), 3),
    }


def per_call_means(curve: dict, lengths: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 59)) -> dict[str, float]:
    """Mean step-prompt tokens per call for cases of the given lengths (the number eval/experiment.py uses)."""
    from eval.experiment import measured_prompt_tokens_per_call

    return {str(n): round(measured_prompt_tokens_per_call(n, curve) or 0.0, 1) for n in lengths}


# ------------------------------------------------------------------------------------------------ sub-agents

SUB_CHECKPOINTS = (6, 10, 20, 40, 57)  # consult from 5 finished turns on; no sub-agent with <= 3 turns left (57 = last)
SUB_KINDS = ("consult", "advocate", "radiology")
# Scripted doctor used for the sub-agent measurement: the DDx narrows to one specialty once consult_min_turns turns are
# done and the model confidence stays at 0.3 (< consult_low_conf 0.5), so the routed / stuck consult can fire; the
# sub-agents get plausible answers so their hints reach the later main prompts.
SUB_DOCTOR = {"focus_after": 5, "confidence": 0.3, "subagent_answers": True}


def summarize_subagents(rows: list[dict]) -> dict:
    """Sub-agent prompt tokens: probes by name (consult per specialty, advocate, radiology, radiology@cap), all consult
    probes pooled, block contributions of the consult prompts, and the real (fired) calls by kind."""
    probes = [r for r in rows if r["kind"] == "subagent" and r["probe"]]
    real = [r for r in rows if r["kind"] == "subagent" and not r["probe"]]
    by_name: dict[str, dict] = {}
    for name in sorted({r["name"] for r in probes}):
        sel = [r for r in probes if r["name"] == name]
        worst = max(sel, key=lambda r: r["tokens"])
        by_name[name] = {"tokens": dist([r["tokens"] for r in sel]), "output_json": dist([r["output_tokens"] for r in sel]),
                         "worst_at": {"case": worst["case"], "turn": worst["turn"]},
                         "by_turn": {str(t): dist([r["tokens"] for r in sel if r["turn"] == t])["max"]
                                     for t in sorted({r["turn"] for r in sel})}}
    consult = [r for r in probes if r["name"].startswith("consult:")]
    parts: dict[str, dict] = {}
    if consult:
        tot = sum(r["tokens"] for r in consult) or 1
        for k in sorted({k for r in consult for k, v in r["parts"].items() if v}):
            parts[k] = dist([r["parts"].get(k, 0) for r in consult])
            parts[k]["share"] = round(sum(r["parts"].get(k, 0) for r in consult) / tot, 3)
    return {
        "probes": by_name,
        "consult_any": dist([r["tokens"] for r in consult]),
        "consult_parts": parts,
        "real": {k: dist([r["tokens"] for r in real if r["name"].split(":")[0] == k]) for k in SUB_KINDS},
        "real_output_json": {k: dist([r["output_tokens"] for r in real if r["name"].split(":")[0] == k])
                             for k in SUB_KINDS},
    }


def case_totals(rows: list[dict], tp=None) -> list[dict]:
    """Per case (real calls only, probes excluded): calls by kind (step includes retries), prompt / output tokens and
    the predicted wall time (estimate_call_s with throughput assumption `tp`)."""

    tp = tp or CONSERVATIVE
    out: dict = {}
    for r in rows:
        if r["probe"]:
            continue
        kind = r["name"].split(":")[0] if r["kind"] == "subagent" else r["kind"]
        c = out.setdefault(r["case"], {"case": r["case"], "calls": {}, "prompt_main": 0, "prompt_sub": 0,
                                       "output": 0, "est_s_main": 0.0, "est_s_sub": 0.0, "last_turn": 0})
        c["calls"][kind] = c["calls"].get(kind, 0) + 1
        sub = r["kind"] == "subagent"
        c["prompt_sub" if sub else "prompt_main"] += r["tokens"]
        c["output"] += r["output_tokens"]
        c["est_s_sub" if sub else "est_s_main"] += estimate_call_s(r["tokens"], r["output_tokens"], tp)
        c["last_turn"] = max(c["last_turn"], r["turn"])
    for c in out.values():
        c["n_main"] = sum(v for k, v in c["calls"].items() if k not in SUB_KINDS)
        c["n_sub"] = sum(c["calls"].get(k, 0) for k in SUB_KINDS)
        c["prompt_total"] = c["prompt_main"] + c["prompt_sub"]
        c["est_s"] = round(c["est_s_main"] + c["est_s_sub"], 1)
    return list(out.values())


def summarize_totals(totals: list[dict]) -> dict:
    kinds = ("step", "review", "final", *SUB_KINDS)
    n = max(1, len(totals))
    return {
        "cases": len(totals),
        "last_turn": dist([c["last_turn"] for c in totals]),
        "calls": {k: dist([c["calls"].get(k, 0) for c in totals]) for k in kinds},
        "fired": {k: round(sum(1 for c in totals if c["calls"].get(k)) / n, 3) for k in SUB_KINDS},
        "n_main": dist([c["n_main"] for c in totals]),
        "n_sub": dist([c["n_sub"] for c in totals]),
        "n_total": dist([c["n_main"] + c["n_sub"] for c in totals]),
        "prompt_total": dist([c["prompt_total"] for c in totals]),
        "prompt_sub": dist([c["prompt_sub"] for c in totals]),
        "sub_share": round(sum(c["prompt_sub"] for c in totals) / max(1, sum(c["prompt_total"] for c in totals)), 3),
        "est_s": dist([c["est_s"] for c in totals]),
        "est_s_sub": dist([round(c["est_s_sub"], 1) for c in totals]),
    }


def scenarios(diagnose_at: tuple[int | None, ...] = (None, 20, 10)) -> list[tuple[str, dict, dict]]:
    """(label, doctor options, AgentConfig overrides): per case length, sub-agents off, on (calibrated triggers) and
    on with the pre-review advocate forced (advocate_conf_below > 1: it fires at every first review)."""
    out = []
    for d in diagnose_at:
        tag = "60 turns" if d is None else f"DIAGNOSE from turn {d}"
        opts = dict(SUB_DOCTOR, diagnose_at=d)
        out.append((f"{tag}, sub-agents off", opts, {"use_subagents": False}))
        out.append((f"{tag}, sub-agents on", opts, {}))
        if d is not None:
            out.append((f"{tag}, on + advocate forced", opts, {"advocate_conf_below": 1.01}))
    return out


def _scenario_rows(case: dict, count: Callable[[str], int], effort: str, base: AgentConfig,
                   diagnose_at: tuple[int | None, ...]) -> dict[str, list[dict]]:
    """{scenario label: anatomy rows} for one case. Sub-agent prompt probes ride on the first 'on' scenario."""
    out: dict[str, list[dict]] = {}
    probe_label = next(lbl for lbl, _, ov in scenarios(diagnose_at) if "use_subagents" not in ov)
    for label, opts, over in scenarios(diagnose_at):
        c = copy.deepcopy(base)
        c.use_subagents = True
        for k, v in over.items():
            setattr(c, k, v)
        out[label] = measure([case], count, effort=effort, cfg=c, checkpoints=(), doctor_opts=opts,
                             sub_checkpoints=SUB_CHECKPOINTS if label == probe_label else ())
    return out


_WORKER_COUNTERS: dict = {}


def _scenario_worker(args: tuple) -> dict[str, list[dict]]:
    case, spec, effort, base, diagnose_at = args
    if spec not in _WORKER_COUNTERS:
        _WORKER_COUNTERS[spec] = get_counter(spec)
    return _scenario_rows(case, _WORKER_COUNTERS[spec], effort, base, diagnose_at)


def measure_subagents(cases: list[dict], count: Callable[[str], int], *, effort: str = "low",
                      cfg: AgentConfig | None = None, diagnose_at: tuple[int | None, ...] = (None, 20, 10),
                      tps: dict | None = None, progress: bool = False, jobs: int = 1,
                      counter_spec: str | None = None) -> dict:
    """Runs every scenario on every case (jobs > 1: cases in parallel processes, each builds its own counter from
    counter_spec). Returns {"prompts": summarize_subagents of the probe scenario, "scenarios": {label: {throughput
    name: summarize_totals}}}. Cases are replayed independently, so the result does not depend on `jobs`."""

    tps = tps or {"conservative": CONSERVATIVE}
    base = cfg or AgentConfig()
    labels = [lbl for lbl, _, _ in scenarios(diagnose_at)]
    probe_label = next(lbl for lbl, _, ov in scenarios(diagnose_at) if "use_subagents" not in ov)
    rows: dict[str, list[dict]] = {lbl: [] for lbl in labels}
    if jobs > 1 and counter_spec:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            args = [(case, counter_spec, effort, base, diagnose_at) for case in cases]
            for i, per in enumerate(ex.map(_scenario_worker, args, chunksize=1), 1):
                for lbl, rs in per.items():
                    rows[lbl] += rs
                if progress:
                    print(f"  {i}/{len(cases)} cases", file=sys.stderr, flush=True)
    else:
        for i, case in enumerate(cases, 1):
            for lbl, rs in _scenario_rows(case, count, effort, base, diagnose_at).items():
                rows[lbl] += rs
            if progress:
                print(f"  {i}/{len(cases)} cases", file=sys.stderr, flush=True)
    return {"doctor": dict(SUB_DOCTOR), "throughput": {k: vars(v) for k, v in tps.items()},
            "prompts": summarize_subagents(rows[probe_label]),
            "scenarios": {lbl: {name: summarize_totals(case_totals(rows[lbl], tp)) for name, tp in tps.items()}
                          for lbl in labels}}


def format_subagent_report(s: dict) -> str:
    L = ["\nSub-agent prompts (probes at turns " + ", ".join(map(str, SUB_CHECKPOINTS))
         + "; full harmony prompt tokens p50 / p95 / max, visible JSON answer max):"]
    for name, d in s.get("prompts", {}).get("probes", {}).items():
        t = d["tokens"]
        L.append(f"  {name:<22} {t['p50']:>6} / {t['p95']:>6} / {t['max']:>6}   answer {d['output_json']['max']:>4}   "
                 f"max by turn {d['by_turn']}")
    ca = s.get("prompts", {}).get("consult_any")
    if ca:
        L.append(f"  consult (all specialties) p50 {ca['p50']} p95 {ca['p95']} max {ca['max']} (n={ca['n']})")
    parts = s.get("prompts", {}).get("consult_parts", {})
    if parts:
        L.append("  consult prompt blocks (mean / max / share): " + "; ".join(
            f"{k} {d['mean']:.0f}/{d['max']}/{d['share']:.0%}" for k, d in sorted(parts.items(), key=lambda kv: -kv[1]["mean"])))
    real = s.get("prompts", {}).get("real", {})
    if real:
        L.append("  fired calls in that scenario: " + "; ".join(
            f"{k} n={d['n']} p50 {d['p50']} max {d['max']}" for k, d in real.items() if d["n"]))
    for label, per_tp in s.get("scenarios", {}).items():
        for tp_name, d in per_tp.items():
            c = d["calls"]
            L.append(f"\n[{label}] ({d['cases']} cases, last turn p50 {d['last_turn']['p50']}, throughput {tp_name})")
            L.append("  calls/case mean (max): " + ", ".join(f"{k} {c[k]['mean']} ({c[k]['max']})" for k in c)
                     + f"; total p50 {d['n_total']['p50']} p95 {d['n_total']['p95']} max {d['n_total']['max']}")
            L.append("  fired in % of cases: " + ", ".join(f"{k} {v:.0%}" for k, v in d["fired"].items()))
            L.append(f"  prompt tokens/case p50 {d['prompt_total']['p50']} p95 {d['prompt_total']['p95']} max "
                     f"{d['prompt_total']['max']} (sub-agent share {d['sub_share']:.1%}); predicted s/case p50 "
                     f"{d['est_s']['p50']:.0f} p95 {d['est_s']['p95']:.0f} max {d['est_s']['max']:.0f} (sub-agents "
                     f"max {d['est_s_sub']['max']:.0f})")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ report

def format_report(s: dict, counter_name: str, n_cases: int, effort: str) -> str:
    L = [f"gpt-oss prompt tokens — {n_cases} cases, tokenizer {counter_name}, harmony format, reasoning effort "
         f"'{effort}', {s['n_real_calls']} real calls + {s['n_calls'] - s['n_real_calls']} probe calls"]
    L.append("\nTokens per prompt (full harmony prompt) by turn: p50 / p95 / max   [view chars p50 / max]")
    for kind in ("step", "review", "final"):
        for t, d in s["by_turn"].get(kind, {}).items():
            tk, vc = d["tokens"], d["view_chars"]
            L.append(f"  {kind:<7} turn {t:>2}: {tk['p50']:>6} / {tk['p95']:>6} / {tk['max']:>6}   "
                     f"[{vc['p50']:>6} / {vc['max']:>6}]  n={tk['n']}")
    L.append("\nAll calls: " + "; ".join(f"{k} p50 {d['p50']} p95 {d['p95']} max {d['max']} (n={d['n']})"
                                          for k, d in s["all"].items() if d["n"]))
    L.append(f"Retry step prompts per first attempt: {s['retry_rate']}")
    L.append("Mean step-prompt tokens per call for a case of N turns: " + ", ".join(
        f"N={n}: {v:.0f}" for n, v in s.get("per_call_mean_by_case_length", {}).items()))
    for kind in ("step", "review"):
        L.append(f"\nContributions ({kind} prompts): mean tokens / p95 / max / share of prompt / present")
        rows = sorted(s["parts"].get(kind, {}).items(), key=lambda kv: -kv[1]["mean"])
        for k, d in rows:
            L.append(f"  {k:<24} {d['mean']:>7} / {d['p95']:>6} / {d['max']:>6} / {d['share']:>5.1%} / {d['present']:>5.0%}")
    vc = s["view_cap"]
    L.append(f"\nView cap: max_view_chars={vc['max_view_chars']}; uncapped view chars p50 {vc['uncapped_view_chars']['p50']} "
             f"p95 {vc['uncapped_view_chars']['p95']} max {vc['uncapped_view_chars']['max']}; share of step prompts over "
             f"the cap {vc['share_capped']:.1%}; view tokens/char p50 {vc['view_tokens_per_char']['p50']} max "
             f"{vc['view_tokens_per_char']['max']} → a view at the cap ≈ {vc['view_tokens_at_cap_est']} tokens")
    L.append(f"Hints per step prompt (incl. retries): p50 {s['hints_total']['p50']} p95 {s['hints_total']['p95']} "
             f"max {s['hints_total']['max']} tokens; fixed part (system prompt + harmony + template) max "
             f"{s['fixed_total']['max']}")
    L.append(f"Tokens per character (step prompts): p50 {s['tokens_per_char']['p50']} max {s['tokens_per_char']['max']}")
    L.append(f"Action JSON output (scripted, no reasoning): p50 {s['output_json_tokens']['p50']} "
             f"p95 {s['output_json_tokens']['p95']} max {s['output_json_tokens']['max']} tokens")
    b = s["budget"]
    L.append(f"\nLargest prompt: {b['max_prompt']} tokens at {b['max_prompt_where']}; gpt-oss-20b context "
             f"{b['context_window']}")
    for size, f in b["fits"].items():
        L.append(f"  budget {size:>6}: prompt fits {f['prompt_only']:.1%}; + max_tokens {b['max_tokens']}: "
                 f"{f['with_max_tokens']:.1%}; + length retry {b['retry_tokens']}: {f['with_retry']:.1%}")
    return "\n".join(L)


def load_cases(spec: str, limit: int | None = None) -> list[dict]:
    p = Path(spec)
    p = p if p.is_absolute() else ROOT / p
    paths = [p] if p.is_file() else sorted(p.rglob("*.json"))
    cases = []
    for path in paths:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("initial"):
            d["_id"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            cases.append(d)
    return cases[:limit] if limit else cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="data/cases_aug", help="case dir (recursive) or one case file")
    ap.add_argument("--limit", type=int, help="first N cases only")
    ap.add_argument("--tokenizer", default="tiktoken", help="tiktoken | tiktoken:<encoding> | hf:<tokenizer.json> | chars")
    ap.add_argument("--effort", default=LLMConfig().reasoning_effort, help="reasoning effort written in the system message")
    ap.add_argument("--turns", default=",".join(map(str, CHECKPOINTS)), help="checkpoint turns")
    ap.add_argument("--max-view-chars", type=int, help="override AGENT_MAX_VIEW_CHARS for the replay")
    ap.add_argument("--json-out", help="write the summary JSON here (eval/experiment.py reads eval/results/token_budget.json)")
    ap.add_argument("--subagents", action="store_true",
                    help="sub-agent mode: sub-agent prompt probes + per-case totals with sub-agents off / on (focused, "
                         "low-confidence scripted doctor) instead of the main-prompt report")
    ap.add_argument("--diagnose-at", default="60,20,10",
                    help="sub-agent mode: case lengths (turn the scripted doctor starts to DIAGNOSE; 60 = never)")
    ap.add_argument("--prefill-tps", type=float, help="time model: prompt tokens/s (default: CONSERVATIVE)")
    ap.add_argument("--decode-tps", type=float, help="time model: output tokens/s")
    ap.add_argument("--overhead-s", type=float, help="time model: fixed seconds per call")
    ap.add_argument("--reasoning-tokens", type=int, help="time model: hidden reasoning tokens per call")
    ap.add_argument("--jobs", type=int, default=1, help="sub-agent mode: cases in parallel processes")
    args = ap.parse_args(argv)

    counter = get_counter(args.tokenizer)
    cfg = AgentConfig()
    if args.max_view_chars is not None:
        cfg.max_view_chars = args.max_view_chars
    checkpoints = tuple(int(x) for x in args.turns.split(",") if x.strip())
    cases = load_cases(args.cases, args.limit)
    if not cases:
        raise SystemExit(f"no cases under {args.cases}")
    t0 = time.time()
    if args.subagents:
        from dataclasses import replace


        over = {k: v for k, v in (("prefill_tps", args.prefill_tps), ("decode_tps", args.decode_tps),
                                  ("overhead_s", args.overhead_s), ("reasoning_tokens", args.reasoning_tokens))
                if v is not None}
        tps = {"conservative": CONSERVATIVE, "moderate": MODERATE}
        if over:
            tps = {"custom": replace(Throughput(), **over)}
        dx_at = tuple(None if int(x) >= cfg.max_turns else int(x) for x in args.diagnose_at.split(",") if x.strip())
        summary = measure_subagents(cases, counter, effort=args.effort, cfg=cfg, diagnose_at=dx_at, tps=tps,
                                    progress=True, jobs=args.jobs, counter_spec=args.tokenizer)
        summary["meta"] = {"tokenizer": counter.name, "effort": args.effort,
                           "subagent_effort": cfg.subagent_reasoning_effort, "cases": len(cases), "source": args.cases,
                           "max_view_chars": cfg.max_view_chars, "prompt_version": prompts.PROMPT_VERSION,
                           "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1)}
        print(f"gpt-oss sub-agent prompt tokens — {len(cases)} cases, tokenizer {counter.name}, harmony format")
        print(format_subagent_report(summary))
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nwrote {out}")
        return 0
    rows = measure(cases, counter, effort=args.effort, cfg=cfg, checkpoints=checkpoints, progress=True)
    llm = LLMConfig()
    summary = summarize(rows, checkpoints=checkpoints, max_view_chars=cfg.max_view_chars, max_tokens=llm.max_tokens,
                        retry_tokens=min(int(llm.max_tokens * llm.length_retry_factor), llm.max_tokens_cap))
    summary["meta"] = {"tokenizer": counter.name, "effort": args.effort, "cases": len(cases), "source": args.cases,
                       "max_view_chars": cfg.max_view_chars, "prompt_version": prompts.PROMPT_VERSION,
                       "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1)}
    print(format_report(summary, counter.name, len(cases), args.effort))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
