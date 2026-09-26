"""Pick the case source: `local` (case files) or `official` (competition API adapter)."""
import os
from collections.abc import Callable, Iterator

from doctor_agent.env.interface import Environment

CaseSource = Callable[..., Iterator[tuple[str, Environment]]]
ENV_NAMES = ("local", "official")


def default_env_name() -> str:
    return (os.getenv("DOCTOR_ENV") or "local").strip().lower()


def case_source(name: str) -> CaseSource:
    if name == "local":
        from doctor_agent.env.local import iter_cases
    elif name == "official":
        from doctor_agent.env.official import iter_cases
    else:
        raise ValueError(f"unknown environment {name!r} (choose from {ENV_NAMES})")
    return iter_cases
