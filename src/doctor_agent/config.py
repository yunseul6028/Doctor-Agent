import os
from dataclasses import dataclass, field


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
        )


@dataclass
class AgentConfig:
    max_turns: int = 60          # competition cap
    target_turns: int = 20       # soft target for Efficiency
    confidence_to_diagnose: float = 0.8
    ddx_size: int = 5
    # knowledge-base hints (candidates, discriminators, diagnosis normalisation); AGENT_USE_KB=0 turns them off
    use_kb: bool = field(default_factory=lambda: os.getenv("AGENT_USE_KB", "1").strip().lower() not in ("0", "false", "no", "off"))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    agent: AgentConfig = field(default_factory=AgentConfig)
