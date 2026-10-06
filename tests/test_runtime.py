"""Runtime robustness: gpt-oss response handling, structured output fallback, time budget, prompt cap, never-crash,
run.py incremental output. No network: every LLM here is a fake."""
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from doctor_agent.agent import prompts
from doctor_agent.agent.loop import FALLBACK_DIAGNOSIS, run_case
from doctor_agent.agent.parser import ACTION_SCHEMA, parse_action
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.config import Config, LLMConfig
from doctor_agent.env.interface import Action, ActionType, Environment, Observation
from doctor_agent.llm import client as client_mod
from doctor_agent.llm.client import BillingError, LLMTimeout, OpenAICompatClient, extract_text
from doctor_agent.llm.harmony import split_harmony
from perf import limit

ROOT = Path(__file__).resolve().parents[1]
ACT = json.dumps({"type": "ASK", "content": "열이 있나요?", "confidence": 0.2}, ensure_ascii=False)


# --- fakes ---------------------------------------------------------------------------------------------------------
def _resp(content=None, reasoning=None, finish="stop", key="reasoning_content"):
    msg = SimpleNamespace(content=content, **({key: reasoning} if reasoning is not None else {}))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


class _FakeOpenAI:
    """Stands in for openai.OpenAI: .chat.completions.create(**kw) returns/raises scripted items."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.items[min(len(self.calls) - 1, len(self.items) - 1)]
        if callable(item):
            item = item(kw)
        if isinstance(item, Exception):
            raise item
        return item


class _BadRequest(Exception):
    status_code = 400


def _client(*items, **cfg):
    fake = _FakeOpenAI(*items)
    c = OpenAICompatClient(LLMConfig(base_url=cfg.pop("base_url", "http://fake"), **cfg), client=fake,
                           sleep=lambda s: None)
    return c, fake


@pytest.fixture(autouse=True)
def _reset_structured():
    client_mod._STRUCTURED_REJECTED.clear()
    yield
    client_mod._STRUCTURED_REJECTED.clear()


class _Env(Environment):
    """Keyword-free fake environment that records every action."""

    def __init__(self, fail_steps: bool = False):
        self.actions: list[Action] = []
        self.fail_steps = fail_steps

    def reset(self):
        return Observation("35세 남성. 주호소: 발열과 기침")

    def step(self, action):
        self.actions.append(action)
        if self.fail_steps and action.type != ActionType.DIAGNOSE:
            raise ConnectionError("env down")
        if action.type == ActionType.DIAGNOSE:
            return Observation("진단이 제출되었습니다.", done=True)
        return Observation("잘 모르겠어요.")


def _cfg(**agent):
    cfg = Config(llm=LLMConfig(), agent=Config().agent)
    for k, v in agent.items():
        setattr(cfg.agent, k, v)
    return cfg


# --- 1. gpt-oss response handling -----------------------------------------------------------------------------------
def test_split_harmony_variants():
    leaked = ("<|start|>assistant<|channel|>analysis<|message|>생각 {\"type\": \"ASK\", \"content\": \"초안\"}<|end|>"
              "<|start|>assistant<|channel|>final<|message|>" + ACT + "<|return|>")
    final, analysis = split_harmony(leaked)
    assert final == ACT and "생각" in analysis
    assert split_harmony("<|channel|>final <|constrain|>json<|message|>" + ACT)[0] == ACT
    final, analysis = split_harmony("<|channel|>analysis<|message|>still thinking")
    assert final == "" and analysis == "still thinking"
    final, analysis = split_harmony("analysisThe patient has fever.assistantfinal" + ACT)
    assert final == ACT and analysis.startswith("The patient")
    assert split_harmony("그냥 답변입니다.") == ("그냥 답변입니다.", "")


def test_parse_action_through_harmony_leak():
    parsed = parse_action("<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>" + ACT)
    assert parsed and parsed[0].content == "열이 있나요?"
    # only an analysis channel (cut off): the action JSON written in the reasoning is still used
    parsed = parse_action("<|channel|>analysis<|message|>결론: " + ACT)
    assert parsed and parsed[0].type == ActionType.ASK


def test_extract_text_prefers_content_then_reasoning():
    assert extract_text(_resp(ACT, "생각"))[0] == ACT
    text, reasoning, _ = extract_text(_resp(None, "생각 끝. " + ACT), expect_json=True)
    assert parse_action(text)[0].content == "열이 있나요?" and "생각" in reasoning
    text, _, _ = extract_text(_resp("", "r " + ACT, key="reasoning"), expect_json=True)
    assert parse_action(text) is not None
    # plain-text roles (patient) are left alone even if the reasoning has JSON
    assert extract_text(_resp("네, 열이 나요.", "{\"x\": 1}"))[0] == "네, 열이 나요."


def test_length_finish_retries_once_with_more_tokens_and_low_effort():
    c, fake = _client(_resp(None, "긴 생각...", finish="length"), _resp(ACT), max_tokens=1000, reasoning_effort="medium")
    out = c.chat([{"role": "user", "content": "x"}], expect_json=True)
    assert out == ACT and len(fake.calls) == 2
    assert fake.calls[0]["max_tokens"] == 1000 and fake.calls[1]["max_tokens"] == 2000
    assert fake.calls[0]["reasoning_effort"] == "medium" and fake.calls[1]["reasoning_effort"] == "low"
    assert c.call_count == 2


def test_length_retry_only_once_then_reasoning_fallback():
    c, fake = _client(_resp(None, "생각만 " + ACT, finish="length"), max_tokens=6000, max_tokens_cap=8192)
    out = c.chat([{"role": "user", "content": "x"}], expect_json=True)
    assert len(fake.calls) == 2 and fake.calls[1]["max_tokens"] == 8192
    assert parse_action(out)[0].content == "열이 있나요?"


def test_billing_error_is_not_retried():
    c, fake = _client(RuntimeError("Error code: 402 credits are depleted"))
    with pytest.raises(BillingError):
        c.chat([{"role": "user", "content": "x"}])
    assert len(fake.calls) == 1


def test_deadline_stops_retries_and_caps_timeout():
    now = [0.0]
    fake = _FakeOpenAI(RuntimeError("503"))
    c = OpenAICompatClient(LLMConfig(timeout_s=60, max_retries=5), client=fake, clock=lambda: now[0],
                           sleep=lambda s: now.__setitem__(0, now[0] + s))
    with pytest.raises(LLMTimeout):
        c.chat([{"role": "user", "content": "x"}], deadline=10.0)
    assert fake.calls[0]["timeout"] == 10.0 and len(fake.calls) == 3 and now[0] < 10.0


def test_sdk_retries_disabled_in_real_client(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    OpenAICompatClient(LLMConfig())
    assert captured["max_retries"] == 0


# --- 2. structured output --------------------------------------------------------------------------------------------
def test_structured_output_off_by_default():
    c, fake = _client(_resp(ACT))
    c.chat([{"role": "user", "content": "x"}], json_schema=ACTION_SCHEMA)
    assert "response_format" not in fake.calls[0] and "extra_body" not in fake.calls[0]


def test_structured_output_json_schema_and_guided_json():
    c, fake = _client(_resp(ACT), structured_output="json_schema")
    c.chat([{"role": "user", "content": "x"}], json_schema=ACTION_SCHEMA)
    rf = fake.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["required"] == ["type", "content"]
    c.chat([{"role": "user", "content": "x"}])  # no schema passed (e.g. patient/review) → plain request
    assert "response_format" not in fake.calls[1]
    c, fake = _client(_resp(ACT), structured_output="guided_json", base_url="http://other")
    c.chat([{"role": "user", "content": "x"}], json_schema=ACTION_SCHEMA)
    assert fake.calls[0]["extra_body"] == {"guided_json": ACTION_SCHEMA}


def test_structured_output_rejected_is_disabled_for_the_run():
    c, fake = _client(_BadRequest("response_format not supported"), _resp(ACT), structured_output="json_schema")
    assert c.chat([{"role": "user", "content": "x"}], json_schema=ACTION_SCHEMA) == ACT
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]
    c2, fake2 = _client(_resp(ACT), structured_output="json_schema")  # next case, same server
    c2.chat([{"role": "user", "content": "x"}], json_schema=ACTION_SCHEMA)
    assert "response_format" not in fake2.calls[0]


def test_bad_request_without_structured_output_is_a_normal_failure():
    c, fake = _client(_BadRequest("bad"), max_retries=1)
    with pytest.raises(RuntimeError):
        c.chat([{"role": "user", "content": "x"}])
    assert len(fake.calls) == 2


# --- 3. time budget ----------------------------------------------------------------------------------------------------
class _ClockLLM:
    """Every call advances a fake clock; step prompts get distinct ASKs, the final prompt a DIAGNOSE."""

    def __init__(self, clock, per_call=10.0):
        self.clock, self.per_call, self.call_count, self.messages = clock, per_call, 0, []

    def chat(self, messages):
        self.clock[0] += self.per_call
        self.call_count += 1
        self.messages.append(messages)
        user = messages[-1]["content"]
        if "남은 턴이 없습니다" in user:
            return json.dumps({"type": "DIAGNOSE", "content": "지역사회획득 폐렴", "confidence": 0.6})
        return json.dumps({"type": "ASK", "content": f"질문 {self.call_count}번: 증상 {self.call_count}?",
                           "ddx": [{"dx": "폐렴", "p": 0.5}], "confidence": 0.3}, ensure_ascii=False)


def test_time_budget_degrades_then_forces_diagnosis():
    now = [0.0]
    llm = _ClockLLM(now)
    env = _Env()
    cfg = _cfg(case_time_budget_s=100.0, degrade_at_frac=0.5, final_reserve_s=20.0, robust=True)
    result = run_case(env, llm, cfg, clock=lambda: now[0])
    rt = result["runtime"]
    assert rt["forced"] == "time_budget" and rt["degraded"]
    assert env.actions[-1].type == ActionType.DIAGNOSE and result["diagnosis"] == "지역사회획득 폐렴"
    assert now[0] <= 100.0  # never ran past the budget
    assert "남은 턴이 없습니다" in llm.messages[-1][-1]["content"]  # final call was the final-diagnosis prompt
    step_prompts = [m[-1]["content"] for m in llm.messages if m[0]["content"] == prompts.SYSTEM]
    assert prompts.LOW_TIME_HINT not in step_prompts[0] and prompts.LOW_TIME_HINT in step_prompts[-2]


def test_time_budget_too_short_for_final_call_uses_top_ddx():
    now = [0.0]
    llm = _ClockLLM(now, per_call=30.0)
    env = _Env()
    cfg = _cfg(case_time_budget_s=60.0, final_reserve_s=25.0, min_call_s=40.0, robust=True)
    result = run_case(env, llm, cfg, clock=lambda: now[0])
    assert result["diagnosis"] == "폐렴" and result["runtime"]["forced"] == "time_budget"
    assert env.actions[-1].type == ActionType.DIAGNOSE and llm.call_count >= 1


def test_degraded_policy_skips_review():
    from doctor_agent.agent.policy import Policy

    class Once:
        call_count = 0

        def chat(self, messages):
            self.call_count += 1
            return json.dumps({"type": "DIAGNOSE", "content": "폐렴", "reason": "r", "confidence": 0.9})

    llm, state = Once(), CaseState(initial_info="35세. 주호소: 기침")
    policy = Policy(llm, Config().agent)
    policy.degraded = True
    action = policy.next_action(state)
    assert action.type == ActionType.DIAGNOSE and llm.call_count == 1 and not state.reviews


def test_guarded_llm_records_latency_on_the_budget_clock():
    from doctor_agent.agent.runtime import CaseBudget, GuardedLLM

    now = [0.0]

    class Slow:
        call_count = 0

        def chat(self, messages):
            now[0] += 7.0 if messages[0]["content"] == prompts.SYSTEM else 3.0
            self.call_count += 1
            return ACT

    guard = GuardedLLM(Slow(), _cfg(), CaseBudget(0, 0.6, 45, clock=lambda: now[0]))
    guard.chat(prompts.build_step_messages("[처음 정보] x", 0, 60, []))
    guard.chat([{"role": "system", "content": "sub-agent"}, {"role": "user", "content": "x"}])
    guard.chat(prompts.build_review_messages("[처음 정보] x", "폐렴", "r", 1, 60))
    assert guard.latencies == [(True, 7.0), (False, 3.0), (True, 3.0)]  # the review counts as a main call
    assert guard.recent_main_call_s() == 7.0  # slowest of the last 3 main calls
    assert guard.stats()["latency_main_s"] == [7.0, 3.0] and guard.stats()["latency_sub_s"] == [3.0]


def test_watchdog_abandons_hung_call():
    from doctor_agent.agent.runtime import CaseBudget, GuardedLLM

    class Hung:
        call_count = 0

        def chat(self, messages):
            time.sleep(2)
            return ACT

    guard = GuardedLLM(Hung(), _cfg(), CaseBudget(0, 0.6, 45))
    guard.call_timeout_s = 0.1
    t0 = time.monotonic()
    assert guard.chat([{"role": "user", "content": "x"}]) == ""
    # 1 s strict (load-scaled), but always < 1.9 s so the 2 s hang still proves the timeout fired.
    assert time.monotonic() - t0 < min(limit(1.0), 1.9) and "timeout" in guard.errors[0]


# --- 4. prompt length control -----------------------------------------------------------------------------------------
def _long_state(n=40):
    st = CaseState(initial_info="60세 여성. 주호소: 호흡곤란")
    for i in range(n):
        st.turns.append(Turn(Action(ActionType.ASK, f"질문 {i}: " + "가" * 50), f"답변 {i}: " + "나" * 500))
    st.findings.update([{"item": f"소견{i}" + "다" * 30, "status": "양성"} for i in range(40)], 1)
    return st


def test_view_is_capped_and_keeps_initial_info_and_latest_turn():
    st = _long_state()
    full = st.view()
    assert len(full) > 6000
    for cap in (6000, 2500, 800):
        v = st.view(max_chars=cap)
        assert len(v) <= cap and "주호소: 호흡곤란" in v and "질문 39" in v, cap
    st.view_max_chars = 3000
    assert len(st.view()) <= 3000 and "앞선 행동" in st.view()


def test_view_unchanged_when_under_cap():
    st = _long_state(3)
    assert st.view(max_chars=100000) == st.view(max_chars=0) == st.view()


def test_prompt_sizes_are_logged_in_result():
    result = run_case(_Env(), _ClockLLM([0.0]), _cfg(max_view_chars=1500, max_turns=8, robust=True))
    rt = result["runtime"]
    assert rt["llm_attempts"] == result["llm_calls"] >= 1 and 0 < rt["prompt_chars_max"] <= rt["prompt_chars_total"]


# --- 5. never crash, always answer ---------------------------------------------------------------------------------
class _Broken:
    def __init__(self):
        self.call_count = 0
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        raise ConnectionError("server down")


def test_llm_always_failing_still_diagnoses_in_robust_mode():
    env, llm = _Env(), _Broken()
    result = run_case(env, llm, _cfg(robust=True))
    assert result["diagnosis"] == FALLBACK_DIAGNOSIS and env.actions[-1].type == ActionType.DIAGNOSE
    rt = result["runtime"]
    assert rt["forced"] == "llm_unavailable" and rt["llm_disabled"] and not rt["llm_called"]
    assert llm.calls == Config().agent.max_llm_failures  # gave up quickly instead of burning the budget


def test_llm_always_failing_is_loud_in_dev_mode():
    with pytest.raises(AssertionError):
        run_case(_Env(), _Broken(), _cfg(robust=False))


def test_intermittent_llm_failure_recovers():
    class Flaky:
        call_count = 0
        n = 0

        def chat(self, messages):
            self.n += 1
            if self.n % 2:
                raise TimeoutError("blip")
            self.call_count += 1
            return json.dumps({"type": "DIAGNOSE", "content": "폐렴", "confidence": 0.9})

    env = _Env()
    result = run_case(env, Flaky(), _cfg(robust=True, max_turns=4))
    assert result["diagnosis"] and env.actions[-1].type == ActionType.DIAGNOSE and result["llm_calls"] >= 1


def test_billing_error_in_robust_mode_does_not_raise():
    class Billing:
        call_count = 0

        def chat(self, messages):
            raise BillingError("402")

    env = _Env()
    result = run_case(env, Billing(), _cfg(robust=True))
    assert result["diagnosis"] == FALLBACK_DIAGNOSIS and env.actions[-1].type == ActionType.DIAGNOSE
    with pytest.raises(BillingError):
        run_case(_Env(), Billing(), _cfg(robust=False))


def test_env_errors_force_a_diagnosis():
    env = _Env(fail_steps=True)
    result = run_case(env, _ClockLLM([0.0]), _cfg(robust=True))
    assert result["runtime"]["forced"] == "env_error" and len(result["runtime"]["env_errors"]) == 3
    assert env.actions[-1].type == ActionType.DIAGNOSE and result["diagnosis"]


def test_policy_crash_is_caught_in_robust_mode(monkeypatch):
    from doctor_agent.agent import policy as policy_mod

    def boom(self, state):
        raise KeyError("bug")

    monkeypatch.setattr(policy_mod.Policy, "_hints", boom)
    env, llm = _Env(), _ClockLLM([0.0])
    result = run_case(env, llm, _cfg(robust=True))
    assert result["runtime"]["forced"] == "policy_error" and env.actions[-1].type == ActionType.DIAGNOSE
    assert llm.call_count == 1  # the final diagnosis call still satisfies the ≥1 LLM call rule
    with pytest.raises(KeyError):
        run_case(_Env(), _ClockLLM([0.0]), _cfg(robust=False))


# --- 6. run.py / env adapter ------------------------------------------------------------------------------------------
def _load_run():
    spec = importlib.util.spec_from_file_location("run_entry", ROOT / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_py_writes_one_line_per_case(tmp_path):
    run = _load_run()
    out = tmp_path / "preds.json"
    preds = run.main(["--cases", str(ROOT / "data/sample_cases"), "--out", str(out), "--llm", "dummy"])
    n = len(list((ROOT / "data/sample_cases").glob("*.json")))
    lines = [json.loads(x) for x in (tmp_path / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(preds) == n and all(r["diagnosis"] for r in lines)
    assert json.loads(out.read_text(encoding="utf-8")) == preds


def test_run_py_survives_a_crashing_case_and_resumes(tmp_path, monkeypatch):
    run = _load_run()
    real = run.run_case
    seen = []

    def flaky(env, llm, cfg):
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("unexpected")
        return real(env, llm, cfg)

    monkeypatch.setattr(run, "run_case", flaky)
    out = tmp_path / "p.json"
    preds = run.main(["--cases", str(ROOT / "data/sample_cases"), "--out", str(out), "--llm", "dummy"])
    lines = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()]
    bad = [r for r in lines if r["error"]]
    assert len(bad) == 1 and bad[0]["diagnosis"] == FALLBACK_DIAGNOSIS and len(lines) == len(preds)
    # resume: drop the last line (simulated crash) and a half-written line; only the missing case runs again
    kept = (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()[:-1]
    (tmp_path / "p.jsonl").write_text("\n".join(kept) + "\n{\"case_id\": \"trunc", encoding="utf-8")
    seen.clear()
    preds2 = run.main(["--cases", str(ROOT / "data/sample_cases"), "--out", str(out), "--llm", "dummy", "--resume"])
    assert len(seen) == 1 and set(preds2) == set(preds)


def test_unknown_env_is_logged_not_raised(tmp_path, monkeypatch):
    from doctor_agent.env.factory import case_source

    with pytest.raises(ValueError):
        case_source("nope")
    monkeypatch.setenv("DOCTOR_ENV", "nope")
    run = _load_run()
    assert run.main(["--out", str(tmp_path / "o.json")]) == {}  # robust mode: logged, no crash
