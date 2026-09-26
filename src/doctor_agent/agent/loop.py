import re

from doctor_agent.agent.policy import Policy
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config
from doctor_agent.env.interface import ActionType, Environment
from doctor_agent.llm.client import LLMClient


_DX_WRAPPERS = [
    re.compile(r"^(최종\s*)?(진단|진단명)\s*[:：]\s*"),
    re.compile(r"^(당신의|환자의|이 환자의)?\s*(최종\s*)?진단은\s*"),
]
_DX_TAIL = re.compile(r"\s*(입니다|으로 진단합니다|로 진단합니다|으로 판단됩니다|로 판단됩니다)[.!。]?\s*$")


def clean_diagnosis(text: str) -> str:
    """Keep only the diagnosis name: strip sentence wrappers like '당신의 진단은 … 입니다.'"""
    t = (text or "").strip().strip("\"'")
    for pat in _DX_WRAPPERS:
        t = pat.sub("", t)
    return _DX_TAIL.sub("", t).strip().rstrip(".")


def run_case(env: Environment, llm: LLMClient, cfg: Config) -> dict:
    """Runs one case to the end. State is created inside this function only (case independence)."""
    calls_before = llm.call_count
    obs = env.reset()
    state = CaseState(initial_info=obs.text)
    policy = Policy(llm, cfg.agent)

    diagnosis = None
    while state.turn_count < cfg.agent.max_turns:
        action = policy.next_action(state)
        obs = env.step(action)
        state.turns.append(Turn(action, obs.text, list(state.ddx)))
        if action.type == ActionType.DIAGNOSE:
            diagnosis = clean_diagnosis(action.content)
            break
        if obs.done:
            break

    # Rule: at least one LLM call per case
    assert llm.call_count > calls_before, "Fixed LLM was not called for this case"
    return {
        "diagnosis": diagnosis,
        "turns": [{**t.action.to_dict(), "reason": t.action.reason, "ddx": t.ddx, "response": t.response} for t in state.turns],
        "n_turns": state.turn_count,
        "ddx": state.ddx,
        "findings": state.findings.as_list(),
        "reviews": state.reviews,
        "llm_calls": llm.call_count - calls_before,
    }
