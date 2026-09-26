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
    max_turns: int = 60          # competition cap
    target_turns: int = 20       # soft target for Efficiency
    confidence_to_diagnose: float = 0.8
    ddx_size: int = 5
    # knowledge-base hints (candidates, discriminators, diagnosis normalisation); AGENT_USE_KB=0 turns them off
    use_kb: bool = field(default_factory=lambda: _flag("AGENT_USE_KB", "1"))

    # --- runtime robustness (see docs/architecture.md "Runtime") ---
    # submission mode: never raise out of run_case (log instead); dev mode: fail loudly (≥1 LLM call rule, billing)
    submission: bool = field(default_factory=lambda: _flag("AGENT_SUBMISSION", "0"))
    # wall-clock budget per case in seconds; 0 = unlimited. Set once the official per-case time limit is known.
    case_time_budget_s: float = field(default_factory=lambda: _num("AGENT_CASE_TIME_BUDGET_S", 0.0))
    # past this fraction of the budget: skip the pre-diagnosis review, shorten hints, lower reasoning effort
    degrade_at_frac: float = field(default_factory=lambda: _num("AGENT_DEGRADE_AT_FRAC", 0.6))
    # when less than this many seconds remain: stop exploring and force the final DIAGNOSE
    final_reserve_s: float = field(default_factory=lambda: _num("AGENT_FINAL_RESERVE_S", 45.0))
    # below this many seconds the final LLM call is skipped too (top DDx fallback)
    min_call_s: float = 5.0
    # consecutive failed LLM calls in one case before giving up on the LLM for that case
    max_llm_failures: int = 3
    # consecutive environment errors in one case before forcing the diagnosis (submission mode)
    max_env_failures: int = 3
    # cap on the state view (prompt body) in characters; 0 = unlimited. Oldest material is truncated first.
    max_view_chars: int = field(default_factory=lambda: int(_num("AGENT_MAX_VIEW_CHARS", 12000)))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    agent: AgentConfig = field(default_factory=AgentConfig)
