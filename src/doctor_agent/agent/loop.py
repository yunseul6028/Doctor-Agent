import logging
import re
import time
from typing import Callable

from doctor_agent.agent import kb_hints
from doctor_agent.agent.policy import Policy
from doctor_agent.agent.runtime import BudgetExceeded, CaseBudget, GuardedLLM, LLMUnavailable
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType, Environment
from doctor_agent.llm.client import BillingError, LLMClient

log = logging.getLogger("doctor_agent.loop")


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


FALLBACK_DIAGNOSIS = "진단 불가"


def _top_ddx(state: CaseState) -> str:
    top = state.ddx[0].get("dx") if state.ddx and isinstance(state.ddx[0], dict) else None
    return str(top).strip() if top and str(top).strip() else FALLBACK_DIAGNOSIS


def _forced_diagnosis(state: CaseState, policy: Policy, guard: GuardedLLM, budget: CaseBudget, cfg: Config) -> Action:
    """Final DIAGNOSE when the normal loop could not produce one: one last LLM call if time and the LLM allow it
    (always attempted when this case has not called the LLM yet), else the top DDx / "진단 불가"."""
    guard.final_mode = True
    enough_time = not budget.enabled or budget.remaining() >= cfg.agent.min_call_s
    if not guard.dead and (enough_time or guard.attempts == 0):
        try:
            action = policy._final_diagnosis(state)
            if action.type == ActionType.DIAGNOSE and action.content.strip():
                return action
        except Exception as e:  # noqa: BLE001
            if isinstance(e, BillingError) and not cfg.agent.submission:
                raise
            guard.errors.append(f"final: {type(e).__name__}: {e}"[:300])
    return Action(ActionType.DIAGNOSE, _top_ddx(state), "강제 종료: 최상위 감별 진단")


def run_case(env: Environment, llm: LLMClient, cfg: Config, clock: Callable[[], float] = time.monotonic) -> dict:
    """Runs one case to the end. State is created inside this function only (case independence).

    Always ends with a DIAGNOSE sent to the environment (unless the environment itself ended the case). In submission
    mode (cfg.agent.submission) no exception escapes after reset(); in dev mode bugs and billing errors propagate and
    the "≥1 LLM call per case" rule is asserted."""
    budget = CaseBudget(cfg.agent.case_time_budget_s, cfg.agent.degrade_at_frac, cfg.agent.final_reserve_s, clock)
    guard = GuardedLLM(llm, cfg, budget)
    submission = cfg.agent.submission
    calls_before = llm.call_count
    obs = env.reset()
    state = CaseState(initial_info=obs.text, view_max_chars=cfg.agent.max_view_chars)
    policy = Policy(guard, cfg.agent)

    diagnosis = None
    forced: str | None = None
    env_done = False
    env_errors: list[str] = []
    while state.turn_count < cfg.agent.max_turns:
        guard.degraded = policy.degraded = budget.degraded()
        if budget.must_finish():
            forced = "time_budget"
            break
        try:
            action = policy.next_action(state)
        except BudgetExceeded:
            forced = "time_budget"
            break
        except LLMUnavailable:
            forced = "llm_unavailable"
            break
        except Exception as e:  # noqa: BLE001
            if not submission:
                raise
            log.exception("policy error")
            guard.errors.append(f"policy: {type(e).__name__}: {e}"[:300])
            forced = "policy_error"
            break
        try:
            obs = env.step(action)
        except Exception as e:  # noqa: BLE001
            if not submission:
                raise
            log.warning("env.step failed: %s", e)
            env_errors.append(f"{type(e).__name__}: {e}"[:300])
            state.turns.append(Turn(action, "(환경 응답 오류)", list(state.ddx)))
            if action.type == ActionType.DIAGNOSE:
                diagnosis = clean_diagnosis(action.content)
                break
            if len(env_errors) >= cfg.agent.max_env_failures:
                forced = "env_error"
                break
            continue
        state.turns.append(Turn(action, obs.text, list(state.ddx)))
        if action.type == ActionType.DIAGNOSE:
            diagnosis = clean_diagnosis(action.content)
            break
        if obs.done:
            env_done = True
            break

    if diagnosis is None:
        action = _forced_diagnosis(state, policy, guard, budget, cfg)
        forced = forced or ("env_done" if env_done else "no_diagnosis")
        if not env_done and state.turn_count < cfg.agent.max_turns:
            try:
                obs = env.step(action)
                state.turns.append(Turn(action, obs.text, list(state.ddx)))
            except Exception as e:  # noqa: BLE001
                if not submission:
                    raise
                env_errors.append(f"final step: {type(e).__name__}: {e}"[:300])
        diagnosis = clean_diagnosis(action.content)
    diagnosis = diagnosis or FALLBACK_DIAGNOSIS

    # Rule: at least one LLM call per case
    called = llm.call_count > calls_before
    if not called:
        if not submission:
            raise AssertionError("Fixed LLM was not called for this case")
        log.error("rule violation risk: no successful LLM call for this case (attempts=%d)", guard.attempts)
    try:
        normalized = kb_hints.normalize_hint(diagnosis, state.initial_info) if cfg.agent.use_kb else {}
    except Exception as e:  # noqa: BLE001 — record-only field
        if not submission:
            raise
        normalized = {"error": str(e)}
    return {
        "diagnosis": diagnosis,
        "turns": [{**t.action.to_dict(), "reason": t.action.reason, "ddx": t.ddx, "response": t.response} for t in state.turns],
        "n_turns": state.turn_count,
        "ddx": state.ddx,
        "findings": state.findings.as_list(),
        "reviews": state.reviews,
        "llm_calls": llm.call_count - calls_before,
        # record only: the submitted diagnosis above is not changed
        "diagnosis_normalized": normalized,
        "runtime": {"elapsed_s": round(budget.elapsed(), 2), "budget_s": budget.budget_s, "degraded": policy.degraded,
                    "forced": forced, "env_errors": env_errors, "llm_called": called, **guard.stats()},
    }
