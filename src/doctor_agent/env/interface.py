"""Environment abstraction.

The agent talks to this interface only. To plug in another case source (e.g. a remote patient-simulator API),
implement an Environment subclass plus an `iter_cases` in a new module and register it in `env/factory.py`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    ASK = "ASK"
    EXAM = "EXAM"
    TEST = "TEST"
    DIAGNOSE = "DIAGNOSE"


@dataclass
class Action:
    type: ActionType
    content: str
    reason: str = ""  # agent's rationale; kept locally, not sent to the environment

    def to_dict(self) -> dict:
        return {"type": self.type.value, "content": self.content}


@dataclass
class Observation:
    text: str
    done: bool = False
    info: dict = field(default_factory=dict)


class Environment(ABC):
    """One case = one Environment instance."""

    @abstractmethod
    def reset(self) -> Observation:
        """Returns the initial information (demographics, chief complaint)."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Runs one ASK/EXAM/TEST/DIAGNOSE action and returns the response."""
