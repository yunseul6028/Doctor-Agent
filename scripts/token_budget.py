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

Tokenizer (dev only, requirements-dev.txt): `tiktoken` encoding `o200k_harmony` (MIT; the BPE file is downloaded once
into the tiktoken cache). It is the encoding `openai-harmony` uses for gpt-oss and gives identical ids to the
`openai/gpt-oss-20b` tokenizer.json (Apache-2.0) on our prompts (checked 2026-09-28).

This is a measurement tool: it is not part of the submission and it does not change the agent.
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
from pathlib import Path
from typing import Callable

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


class ScriptedDoctor:
    """Stands in for gpt-oss: returns realistic action JSON (findings from the last response, a 5-entry DDx with
    for/against evidence, a question/exam/test from the plan) and never diagnoses until asked for a final answer.
    Captures the prompt builder arguments of every call so the prompt can be broken down (see capture_builders)."""

    def __init__(self, case: dict, seed_dx: str = ""):
        self.call_count = 0
        self.plan = plan_actions(case)
        self.pos = 0
        self.n_steps = 0
        self.last_proposed = ""
        self.dx = [seed_dx or case.get("diagnosis") or "원인 미상"]
        pool = [d for d in _DDX_POOL if d not in self.dx]
        start = sum(map(ord, str(case.get("initial", "")))) % len(pool)
        self.dx += (pool[start:] + pool[:start])[:4]
        self.last_action: Action | None = None
        self.last_response = str(case.get("initial", ""))
        self.pending: list[tuple] = []  # builder captures not yet consumed by chat()
        self.calls: list[dict] = []  # {kind, turn, attempt, messages, view, hints, extras, output}
        self.turn = 0
        self.attempt = 0
        self.probe = False
        self.outputs: list[str] = []

    # --- what the model would answer
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
        return rows

    def _step_json(self, refused: bool = False) -> dict:
        if refused and self.last_proposed:  # "already done": like a model would, switch to another kind of action
            while self.pos < len(self.plan) - 1 and self.plan[self.pos][0] == self.last_proposed:
                self.pos += 1
        typ, content = self.plan[min(self.pos, len(self.plan) - 1)]
        self.pos += 1
        self.last_proposed = typ
        self.n_steps += 1
        f = self._findings()
        return {"findings": f, "ddx": self._ddx(f[0]["item"]), "type": typ, "content": content,
                "reason": f"{content[:20]}(으)로 {self.dx[0]}와 {self.dx[1]}을 감별하려고 합니다.", "confidence": 0.4}

    def chat(self, messages: list[dict]) -> str:
        self.call_count += 1
        cap = self.pending.pop() if self.pending else ("other", "", [], {})
        kind, view, hints, extras = cap
        if kind == "review":
            out = {"key_findings": [{"finding": self.dx[0], "status": "설명됨"}], "contradicting": [],
                   "confirmation": "없음", "unresolved_danger": [self.dx[3]],
                   "next": {"type": "TEST", "content": "추가 검사", "reason": "위험 질환 배제"},
                   "final_diagnosis": "", "refine_evidence": ""}
        elif kind == "final":
            out = {"findings": [], "ddx": self._ddx("최종"), "type": "DIAGNOSE", "content": self.dx[0],
                   "reason": "지금까지의 소견으로 가장 가능성이 높음", "confidence": 0.6}
        else:
            out = self._step_json(refused=bool(hints) and "이미 했습니다" in hints[-1])
        text = json.dumps(out, ensure_ascii=False)
        self.calls.append({"kind": kind, "turn": self.turn, "attempt": self.attempt, "probe": self.probe,
                           "messages": messages, "view": view, "hints": list(hints), "extras": dict(extras),
                           "output": text})
        if kind == "step":
            self.attempt += 1
        return text


@contextmanager
def capture_builders(doctor: ScriptedDoctor, state_ref: dict):
    """Wrap the prompt builders so each call records (kind, view, hints, extras) for the anatomy breakdown."""
    orig = (prompts.build_step_messages, prompts.build_review_messages, prompts.build_final_messages)

    def step(view, turn, max_turns, hints):
        doctor.pending.append(("step", view, list(hints), {}))
        return orig[0](view, turn, max_turns, hints)

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


def replay_case(case: dict, cfg: AgentConfig | None = None, checkpoints: tuple[int, ...] = CHECKPOINTS) -> list[dict]:
    """Runs one case for cfg.max_turns turns with the scripted doctor. Returns every captured call:
    {kind, turn (1-based turn the prompt is for), attempt, probe, messages, view, hints, extras, output,
    view_chars_uncapped}. Probes (review / final at checkpoints) run on deep copies and never change the replay."""
    cfg = cfg or AgentConfig()
    env = _env_for(case)
    doctor = ScriptedDoctor(case)
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


def measure(cases: list[dict], count: Callable[[str], int], *, effort: str = "low", cfg: AgentConfig | None = None,
            checkpoints: tuple[int, ...] = CHECKPOINTS, progress: bool = False) -> list[dict]:
    """Anatomy rows for every captured call of every case."""
    cfg = cfg or AgentConfig()
    rows = []
    for i, case in enumerate(cases, 1):
        for c in replay_case(case, cfg, checkpoints):
            a = anatomy(c["kind"], c["messages"], count, effort, view=c["view"], hints=c["hints"], extras=c["extras"])
            a.update(case=case.get("_id", i), turn=c["turn"], attempt=c["attempt"], probe=c["probe"],
                     view_chars_uncapped=c.get("view_chars_uncapped", len(c["view"])),
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
