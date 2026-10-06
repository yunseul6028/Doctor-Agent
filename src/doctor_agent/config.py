import os
from dataclasses import dataclass, field


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env(prefix: str, key: str, default: str) -> str:
    # Per-role setting (DOCTOR_LLM_MODEL, etc.) → shared setting (LLM_MODEL) → default
    return os.getenv(f"{prefix}_{key}") or os.getenv(f"LLM_{key}") or default


@dataclass
class LLMConfig:
    # Defaults: local Ollama gpt-oss-20b. Swap providers via env vars (.env) only; see .env.example.
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"
    model: str = "gpt-oss:20b"
    reasoning_effort: str = "low"  # "none" means the parameter is not sent
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: float = 60.0
    max_retries: int = 2
    # gpt-oss may spend the whole max_tokens on reasoning (finish_reason="length", empty content): retry once with
    # max_tokens * length_retry_factor (capped) and reasoning_effort="low"
    length_retry_factor: float = 2.0
    max_tokens_cap: int = 8192
    # "off" | "json_schema" (OpenAI response_format) | "guided_json" (vLLM extra_body). Only used when the caller passes
    # a schema; disabled for the rest of the run if the server rejects it (HTTP 400).
    structured_output: str = "off"

    @classmethod
    def from_env(cls, prefix: str = "DOCTOR_LLM") -> "LLMConfig":
        d = cls()
        return cls(
            base_url=_env(prefix, "BASE_URL", d.base_url),
            api_key=_env(prefix, "API_KEY", d.api_key),
            model=_env(prefix, "MODEL", d.model),
            reasoning_effort=_env(prefix, "REASONING_EFFORT", d.reasoning_effort),
            temperature=float(_env(prefix, "TEMPERATURE", str(d.temperature))),
            max_tokens=int(_env(prefix, "MAX_TOKENS", str(d.max_tokens))),
            timeout_s=float(_env(prefix, "TIMEOUT", str(d.timeout_s))),
            max_retries=int(_env(prefix, "MAX_RETRIES", str(d.max_retries))),
            max_tokens_cap=int(_env(prefix, "MAX_TOKENS_CAP", str(d.max_tokens_cap))),
            structured_output=_env(prefix, "STRUCTURED_OUTPUT", d.structured_output).strip().lower(),
        )


@dataclass
class AgentConfig:
    max_turns: int = 60          # hard cap on actions per case
    target_turns: int = 20       # soft target for Efficiency
    # knowledge-base hints (candidates, discriminators, diagnosis normalisation); AGENT_USE_KB=0 turns them off
    use_kb: bool = field(default_factory=lambda: _flag("AGENT_USE_KB", "1"))
    # safety layers (see docs/architecture.md): each can be switched off for ablation experiments
    use_grounding: bool = field(default_factory=lambda: _flag("AGENT_USE_GROUNDING", "1"))
    use_danger_gate: bool = field(default_factory=lambda: _flag("AGENT_USE_DANGER_GATE", "1"))
    use_preconditions: bool = field(default_factory=lambda: _flag("AGENT_USE_PRECONDITIONS", "1"))
    max_gate_turns: int = 3  # actions the can't-miss gate may force per case
    # advisors (code-only helpers that add prompt hints / one pushback; see docs/architecture.md "Advisors"). Each can be
    # switched off for ablation experiments (condition v6-no-advisors turns all four off).
    use_confidence: bool = field(default_factory=lambda: _flag("AGENT_USE_CONFIDENCE", "1"))
    use_anchoring: bool = field(default_factory=lambda: _flag("AGENT_USE_ANCHORING", "1"))
    # premature-closure check from this many turns on (anchoring.MIN_TURNS is the same default; calibrated, see below)
    anchoring_min_turns: int = field(default_factory=lambda: int(_num("AGENT_ANCHORING_MIN_TURNS", 5)))
    use_planner: bool = field(default_factory=lambda: _flag("AGENT_USE_PLANNER", "1"))  # also needs use_kb
    use_triage: bool = field(default_factory=lambda: _flag("AGENT_USE_TRIAGE", "1"))
    # a proposed diagnosis whose code-computed confidence is below this gets one pushback per case
    confidence_pushback_below: float = field(default_factory=lambda: _num("AGENT_CONFIDENCE_PUSHBACK_BELOW", 0.3))
    planner_k: int = 3  # suggestions shown by the question planner
    # code-first result interpreter (agent/result_interpreter.py): reads each EXAM/TEST result into the findings ledger,
    # a one-line hint on the next prompt and a one-time alert for critical results (condition v6-no-interp turns it off)
    use_result_interpreter: bool = field(default_factory=lambda: _flag("AGENT_USE_RESULT_INTERPRETER", "1"))
    result_hint_chars: int = 300  # cap on the result-reading hint (part of max_advisor_chars)
    # total characters the advisor hints may add to one step prompt
    # (triage/critical-result alert > triage hint > result reading > anchoring > starting DDx > planner)
    max_advisor_chars: int = field(default_factory=lambda: int(_num("AGENT_MAX_ADVISOR_CHARS", 900)))

    # specialist sub-agents (agent/subagents/; docs/architecture.md "Specialist sub-agents"): extra calls of the same
    # doctor LLM in another role, only when triggered. Master switch (condition v6-no-subagents turns it off) + one per kind
    use_subagents: bool = field(default_factory=lambda: _flag("AGENT_USE_SUBAGENTS", "1"))
    use_consult: bool = field(default_factory=lambda: _flag("AGENT_USE_CONSULT", "1"))
    use_advocate: bool = field(default_factory=lambda: _flag("AGENT_USE_ADVOCATE", "1"))
    use_llm_radiology: bool = field(default_factory=lambda: _flag("AGENT_USE_LLM_RADIOLOGY", "1"))  # + result interpreter
    max_subagent_calls: int = field(default_factory=lambda: int(_num("AGENT_MAX_SUBAGENT_CALLS", 3)))  # per case
    max_llm_radiology: int = 1  # LLM result readings per case
    # separate prompt budget for sub-agent hints (the advisor budget above is untouched)
    max_subagent_chars: int = field(default_factory=lambda: int(_num("AGENT_MAX_SUBAGENT_CHARS", 600)))
    # reasoning effort of sub-agent calls ("none" = parameter not sent)
    subagent_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("AGENT_SUBAGENT_REASONING_EFFORT", "low").strip().lower() or "low")
    # Trigger thresholds: calibrated on replayed trajectories of non-gpt-oss dev models (eval/offline/eval_triggers.py,
    # docs/experiments.md "trigger calibration"); re-check on gpt-oss-20b runs before trusting them.
    # routed consult: from consult_min_turns turns on, when one specialty holds >= consult_min_share of the top-DDx mass
    consult_min_turns: int = field(default_factory=lambda: int(_num("AGENT_CONSULT_MIN_TURNS", 5)))
    consult_min_share: float = field(default_factory=lambda: _num("AGENT_CONSULT_MIN_SHARE", 0.6))
    # stuck consult: model confidence < consult_low_conf on consult_low_conf_turns consecutive parsed turns, from
    # consult_low_conf_after turns on (not replayable offline: the result files do not keep the model's confidence)
    consult_low_conf: float = field(default_factory=lambda: _num("AGENT_CONSULT_LOW_CONF", 0.5))
    consult_low_conf_turns: int = field(default_factory=lambda: int(_num("AGENT_CONSULT_LOW_CONF_TURNS", 3)))
    consult_low_conf_after: int = field(default_factory=lambda: int(_num("AGENT_CONSULT_LOW_CONF_AFTER", 6)))
    # pre-review advocate: code confidence of the proposed diagnosis below this
    advocate_conf_below: float = field(default_factory=lambda: _num("AGENT_ADVOCATE_CONF_BELOW", 0.4))
    # also call the advocate at the anchoring moment (when the premature-closure check fires); off: that moment did
    # not pick out wrong cases in the replay, the pre-review confidence did
    advocate_on_anchoring: bool = field(default_factory=lambda: _flag("AGENT_ADVOCATE_ON_ANCHORING", "0"))
    subagent_min_remaining_turns: int = 4  # skip all sub-agents when fewer turns than this remain (i.e. <= 3)
    # pre-call time check (runtime.GuardedLLM.subagent_time_block, only with a case time budget): skip a sub-agent
    # call when the exploratory time left < subagent_time_factor x the slowest of the last latency_window main calls
    subagent_time_factor: float = field(default_factory=lambda: _num("AGENT_SUBAGENT_TIME_FACTOR", 3.0))
    latency_window: int = 3

    # --- runtime robustness (see docs/architecture.md "Runtime") ---
    # robust mode: never raise out of run_case (log instead); dev mode: fail loudly (≥1 LLM call guarantee, billing)
    robust: bool = field(default_factory=lambda: _flag("AGENT_ROBUST", "0"))
    # wall-clock budget per case in seconds; 0 = unlimited. Set it a little below any per-case time limit you run under.
    case_time_budget_s: float = field(default_factory=lambda: _num("AGENT_CASE_TIME_BUDGET_S", 0.0))
    # past this fraction of the budget: skip the pre-diagnosis review, shorten hints, lower reasoning effort
    degrade_at_frac: float = field(default_factory=lambda: _num("AGENT_DEGRADE_AT_FRAC", 0.6))
    # when less than this many seconds remain: stop exploring and force the final DIAGNOSE
    final_reserve_s: float = field(default_factory=lambda: _num("AGENT_FINAL_RESERVE_S", 45.0))
    # below this many seconds the final LLM call is skipped too (top DDx fallback)
    min_call_s: float = 5.0
    # consecutive failed LLM calls in one case before giving up on the LLM for that case
    max_llm_failures: int = 3
    # consecutive environment errors in one case before forcing the diagnosis (robust mode)
    max_env_failures: int = 3
    # cap on the state view (prompt body) in characters; 0 = unlimited. Oldest material is truncated first.
    max_view_chars: int = field(default_factory=lambda: int(_num("AGENT_MAX_VIEW_CHARS", 12000)))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    agent: AgentConfig = field(default_factory=AgentConfig)
