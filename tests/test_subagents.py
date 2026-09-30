"""Specialist sub-agents (agent/subagents/): runner, triggers, caps, skips, guards, radiology merge, result summary.
Scripted fake LLM and fake content modules (sys.modules); no real LLM calls."""
import json
import sys
import types

import pytest

from doctor_agent.agent import anchoring, confidence, prompts
from doctor_agent.agent.confidence import Assessment
from doctor_agent.agent.ledger import CODE_SOURCE, LLM_RADIOLOGY_SOURCE, DdxLedger
from doctor_agent.agent.policy import MAX_REVIEWS, Policy
from doctor_agent.agent.runtime import CaseBudget, GuardedLLM
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.agent.subagents import runner
from doctor_agent.agent.parser import as_list, find_json
from doctor_agent.agent.subagents.base import SubagentCall, SubagentResult, failed
from doctor_agent.config import Config
from doctor_agent.env.interface import Action, ActionType

CONSULT_SYS, ADVOCATE_SYS = "FAKE-CONSULT", "FAKE-ADVOCATE"
LONG_CXR = ("우측 하엽에 경결 소견이 관찰됨. 좌측 폐는 깨끗함. 심장 크기는 정상. 늑골 골절 없음. 흉수 소량 의심. "
            "종격동 이상 없음. 기관지 벽 비후 소견. 횡격막 정상.")


def _step(type_, content, conf=0.5, ddx=None):
    return json.dumps({"type": type_, "content": content, "reason": "r", "confidence": conf, "findings": [],
                       "ddx": ddx if ddx is not None else [{"dx": "A", "p": 0.5}, {"dx": "B", "p": 0.4}]},
                      ensure_ascii=False)


ASK = _step("ASK", "언제부터 아팠나요?")
DX = _step("DIAGNOSE", "A")
REVIEW_OK = json.dumps({"key_findings": [], "contradicting": [], "confirmation": "흉부 X선 소견", "unresolved_danger": [],
                        "next": None, "final_diagnosis": "", "refine_evidence": ""}, ensure_ascii=False)
CONSULT_OK = json.dumps({"hint_ko": "심장 원인을 먼저 배제하세요: 트로포닌 확인.", "ddx_add": [{"name": "급성 심근경색", "why": "흉통"}],
                         "suggested_actions": [{"type": "TEST", "content": "트로포닌", "why": "배제"}],
                         "red_flags": ["흉통"]}, ensure_ascii=False)
ADVOCATE_OK = json.dumps({"hint_ko": "A 말고 C도 설명됩니다.", "ddx_add": [{"name": "C", "why": "발열"}]}, ensure_ascii=False)
RADIOLOGY_OK = json.dumps({"items": [
    {"finding": "우측 하엽 경결", "status": "있음", "site": "우측 하엽", "value": "", "critical": False},
    {"finding": "흉막 삼출", "status": "없음", "site": "", "value": "", "critical": False},  # code read it as uncertain
    {"finding": "긴장성 기흉", "status": "있음", "site": "", "value": "", "critical": True}],
    "normal": False, "unavailable": False, "summary": "우측 하엽 경결, 소량 흉수 의심"}, ensure_ascii=False)


_GENERIC_KEYS = ("hint_ko", "hint", "summary", "ddx_add", "suggested_actions", "red_flags")


def _generic(name):
    """Generic parser of the fake content modules: hint_ko | hint | summary, ddx_add, suggested_actions, red_flags from
    the last JSON object that has one of them (the runner itself has no default parser)."""
    def parse(text):
        obj = find_json(text, lambda o: any(k in o for k in _GENERIC_KEYS))
        if not obj:
            return failed(name, "no JSON object in the answer")
        hint = obj.get("hint_ko") or obj.get("hint") or obj.get("summary") or ""
        return SubagentResult(name, True, " ".join(str(hint).split())[:2000], as_list(obj.get("ddx_add")),
                              as_list(obj.get("suggested_actions")), as_list(obj.get("red_flags")), obj)
    return parse


class Router:
    """Fake fixed LLM: answers by role (system prompt). Records (kind, messages, opts)."""

    def __init__(self, step=ASK, review=REVIEW_OK, consult=CONSULT_OK, advocate=ADVOCATE_OK, radiology=RADIOLOGY_OK,
                 raise_for=()):
        self.out = {"step": step, "review": review, "consult": consult, "advocate": advocate, "radiology": radiology}
        self.raise_for, self.call_count, self.log = set(raise_for), 0, []

    def chat(self, messages, **opts):
        sys_ = messages[0]["content"]
        kind = {prompts.SYSTEM: "step", prompts.REVIEW_SYSTEM: "review", prompts.RESULT_INTERPRETER_PROMPT: "radiology",
                CONSULT_SYS: "consult", ADVOCATE_SYS: "advocate"}.get(sys_, "other")
        self.log.append((kind, messages, opts))
        if kind in self.raise_for:
            raise RuntimeError(f"{kind} down")
        self.call_count += 1
        out = self.out[kind]
        return out.pop(0) if isinstance(out, list) else out

    def kinds(self):
        return [k for k, _, _ in self.log]

    def user(self, kind="step", i=-1):
        return [m for k, m, _ in self.log if k == kind][i][1]["content"]


class OptRouter(Router):
    supports_options = True


def _consult_mod(calls, route=("cardio", 0.7, ["흉통"]), boom=None):
    spec = types.ModuleType("doctor_agent.knowledge.specialty")

    def route_fn(state):
        calls.append(("route", state.turn_count))
        if boom == "route":
            raise ValueError("route bug")
        return route

    spec.route = route_fn
    spec.resources = lambda s, state: {"criteria": ["HEART"], "text": "짧은 기준"}
    mod = types.ModuleType("doctor_agent.agent.subagents.consult")
    mod.SPECIALTIES = {"cardio": types.SimpleNamespace(name_ko="순환기")}

    def build(state, specialty, resources):
        calls.append(("build", specialty, resources.get("criteria")))
        if boom == "build":
            raise KeyError("build bug")
        return SubagentCall(f"consult:{specialty}", [{"role": "system", "content": CONSULT_SYS},
                                                     {"role": "user", "content": state.initial_info}], {"type": "object"})

    def parse(text):
        if boom == "parse":
            raise TypeError("parse bug")
        return _generic("consult")(text)

    mod.build_consult, mod.parse_consult = build, parse
    return spec, mod


def _advocate_mod(calls):
    mod = types.ModuleType("doctor_agent.agent.subagents.advocate")

    def build(state, proposed, reason):
        calls.append(("advocate", proposed, reason))
        return SubagentCall("advocate", [{"role": "system", "content": ADVOCATE_SYS},
                                         {"role": "user", "content": f"{state.initial_info} / {proposed}"}], None)

    mod.build_advocate = build
    mod.parse_advocate = _generic("advocate")
    return mod


@pytest.fixture
def mods(monkeypatch):
    """Installs fake content modules; returns the call log."""
    calls: list = []

    def install(route=("cardio", 0.7, ["흉통"]), boom=None, consult=True, advocate=True):
        spec, cons = _consult_mod(calls, route, boom)
        if consult:
            monkeypatch.setitem(sys.modules, "doctor_agent.knowledge.specialty", spec)
            monkeypatch.setitem(sys.modules, "doctor_agent.agent.subagents.consult", cons)
        else:
            monkeypatch.setitem(sys.modules, "doctor_agent.agent.subagents.consult", None)  # import → ImportError
            monkeypatch.setitem(sys.modules, "doctor_agent.knowledge.specialty", None)  # the real one exists now
        if advocate:
            monkeypatch.setitem(sys.modules, "doctor_agent.agent.subagents.advocate", _advocate_mod(calls))
        else:
            monkeypatch.setitem(sys.modules, "doctor_agent.agent.subagents.advocate", None)
        return calls
    return install


def _cfg(**over):
    cfg = Config().agent
    cfg.use_kb = False
    cfg.use_danger_gate = cfg.use_preconditions = cfg.use_grounding = False
    cfg.use_confidence = cfg.use_anchoring = cfg.use_planner = cfg.use_triage = False
    cfg.use_result_interpreter = True
    cfg.use_subagents = cfg.use_consult = cfg.use_advocate = cfg.use_llm_radiology = True
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _state(n_turns=5, conf=0.9, results=(), initial="55세 남성. 주호소: 흉통"):
    st = CaseState(initial_info=initial, confidence=conf)
    for i in range(n_turns):
        st.turns.append(Turn(Action(ActionType.ASK, f"질문 {i + 1}번 무엇인가요?"), "네"))
    for name, text in results:
        st.turns.append(Turn(Action(ActionType.TEST, name), text))
    return st


def _sub(st, kind=None):
    return [e for e in st.safety_log if e["layer"] == "subagent" and (kind is None or e.get("kind") == kind)]


# ---------------------------------------------------------------- runner

def test_runner_parses_caps_and_sanitises():
    call = SubagentCall("consult:cardio", [{"role": "system", "content": CONSULT_SYS}], None, max_chars_out=10)
    res = runner.run(Router(), call, parse=_generic(call.name))
    assert res.ok and res.name == "consult:cardio" and len(res.hint_ko) <= 10 and res.hint_ko.endswith("…")
    assert res.ddx_add == [{"name": "급성 심근경색", "why": "흉통"}] and res.red_flags == ["흉통"]
    assert res.suggested_actions == [{"type": "TEST", "content": "트로포닌", "why": "배제"}]
    # harmony-leaked / reasoning-wrapped JSON is still found
    wrapped = "<analysis>생각 중 {\"x\": 1}</analysis>" + CONSULT_OK
    assert runner.run(Router(consult=wrapped), call, parse=_generic(call.name)).ok


@pytest.mark.parametrize("answer", ["", "JSON 아님", "{\"type\": \"ASK\", \"content\": \"x\"}", "{broken"])
def test_runner_bad_json_is_not_ok(answer):
    res = runner.run(Router(consult=answer), SubagentCall("consult:x", [{"role": "system", "content": CONSULT_SYS}], None),
                     parse=_generic("consult:x"))
    assert not res.ok and res.raw.get("error") and res.hint_ko == ""


def test_runner_never_raises():
    call = SubagentCall("consult:x", [{"role": "system", "content": CONSULT_SYS}], None)
    assert not runner.run(Router(raise_for={"consult"}), call, parse=_generic(call.name)).ok
    # no parser: not ok, and the LLM is not called
    llm = Router()
    assert runner.run(llm, call).raw["error"] == "no parser" and llm.log == []

    def bad_parse(text):
        raise ValueError("bug")
    assert "parser" in runner.run(Router(), call, parse=bad_parse).raw["error"]
    assert not runner.run(Router(), call, parse=lambda t: "not a result").ok
    # deadline already passed: no call at all
    llm = Router()
    assert not runner.run(llm, call, deadline=5.0, clock=lambda: 10.0, parse=_generic(call.name)).ok and llm.log == []


def test_runner_passes_gpt_oss_options_through_guarded_llm():
    inner = OptRouter()
    cfg = Config()
    guard = GuardedLLM(inner, cfg, CaseBudget(0, 0.6, 45))
    call = SubagentCall("advocate", [{"role": "system", "content": ADVOCATE_SYS}], {"type": "object"})
    assert runner.run(guard, call, reasoning_effort="low", parse=_generic(call.name)).ok
    _, _, opts = inner.log[-1]
    assert opts["json_schema"] == {"type": "object"} and opts["expect_json"] and opts["reasoning_effort"] == "low"
    assert guard.subagent_calls == 1 and guard.stats()["subagent_calls"] == 1
    cfg.llm.reasoning_effort = "none"  # never sent when the deployment disables it
    runner.run(guard, call, reasoning_effort="low", parse=_generic(call.name))
    assert inner.log[-1][2].get("reasoning_effort") is None


# ---------------------------------------------------------------- consult

def test_consult_routed_fires_once_hint_on_same_step_and_refs(mods):
    calls = mods()
    llm, st = Router(), _state(n_turns=5)
    pol = Policy(llm, _cfg())
    assert pol.next_action(st).type == ActionType.ASK
    assert llm.kinds() == ["consult", "step"]
    user = llm.user()
    assert "전문의 자문(순환기) (참고 의견, 사실 근거 아님): 심장 원인을 먼저 배제하세요" in user
    assert "참고(자문, 미확인): 급성 심근경색(흉통)" in user  # DDx ledger view
    assert all(d["dx"] != "급성 심근경색" for d in st.ddx)  # never a live DDx
    assert ("build", "cardio", ["HEART"]) in calls
    e = _sub(st, "call")
    assert len(e) == 1 and e[0]["name"] == "consult:cardio" and e[0]["trigger"] == "routed" and e[0]["ok"]
    st.turns.append(Turn(Action(ActionType.ASK, "네 번째 질문입니다"), "네"))
    pol.next_action(st)
    assert llm.kinds() == ["consult", "step", "step"]  # once per case; the hint is gone on the next step
    assert "전문의 자문" not in llm.user()


def test_consult_not_before_min_turns_nor_below_share(mods):
    calls = mods(route=("cardio", 0.5, []))
    llm = Router()
    Policy(llm, _cfg()).next_action(_state(n_turns=4))  # consult_min_turns = 5
    assert llm.kinds() == ["step"] and not [c for c in calls if c[0] == "route"]
    llm = Router()
    Policy(llm, _cfg()).next_action(_state(n_turns=5, conf=0.2))  # share 0.5, low confidence but before turn 6
    assert llm.kinds() == ["step"]


def test_consult_on_low_confidence_streak_after_turn_6(mods):
    mods(route=("cardio", 0.3, []))
    llm, st = Router(), _state(n_turns=6, conf=0.9)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    assert llm.kinds() == ["step"]  # confident: no consult
    llm2, st2 = Router(), _state(n_turns=6, conf=0.2)
    Policy(llm2, _cfg()).next_action(st2)
    assert llm2.kinds() == ["consult", "step"] and _sub(st2, "call")[0]["trigger"] == "low_confidence"


def test_consult_no_specialty_is_skipped_and_logged_once(mods):
    mods(route=(None, 0.0, []))
    llm, st = Router(step=_step("ASK", "언제부터 아팠나요?", conf=0.1)), _state(n_turns=6, conf=0.1)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    st.turns.append(Turn(Action(ActionType.ASK, "추가 질문입니다"), "네"))
    pol.next_action(st)
    assert llm.kinds() == ["step", "step"]
    assert [e["reason"] for e in _sub(st, "skip")] == ["no_specialty"]
    assert pol.subagent_summary(st)["skipped"] == {"consult:no_specialty": 2}


# ---------------------------------------------------------------- global guards

def test_degraded_mode_skips_everything(mods):
    calls = mods()
    llm, st = Router(), _state(n_turns=4, results=[("흉부 X선", LONG_CXR)])
    pol = Policy(llm, _cfg())
    pol.degraded = True
    pol.next_action(st)
    assert llm.kinds() == ["step"] and not [c for c in calls if c[0] == "route"]
    assert [e["reason"] for e in _sub(st, "skip")] == ["low_time"]  # the radiology trigger fired but was skipped
    assert "LLM 판독 필요(many_sentences; 호출 안 함)" in [e for e in st.safety_log if e.get("kind") == "reading"][0]["msg"]


def test_few_remaining_turns_skip(mods):
    mods()
    llm, st = Router(), _state(n_turns=57)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    assert llm.kinds() == ["step"] and [e["reason"] for e in _sub(st, "skip")] == ["few_turns"]


def test_call_cap(mods):
    mods()
    llm, st = Router(), _state(n_turns=5, results=[("흉부 X선", LONG_CXR)])
    pol = Policy(llm, _cfg(max_subagent_calls=1))
    pol.next_action(st)
    assert llm.kinds() == ["radiology", "step"]  # radiology used the only call; the routed consult is skipped
    assert [(e["name"], e["reason"]) for e in _sub(st, "skip")] == [("consult", "call_cap")]
    s = pol.subagent_summary(st)
    assert s["calls"] == {"radiology": 1} and s["ok"] == 1 and s["fail"] == 0


def test_master_switch_off(mods):
    calls = mods()
    llm, st = Router(), _state(n_turns=5, results=[("흉부 X선", LONG_CXR)])
    pol = Policy(llm, _cfg(use_subagents=False))
    pol.next_action(st)
    assert llm.kinds() == ["step"] and not calls and not _sub(st)
    assert pol.subagent_summary(st)["enabled"] is False


# ---------------------------------------------------------------- content-module failures are guarded

@pytest.mark.parametrize("boom", ["route", "build"])
def test_content_module_exception_disables_it_and_case_continues(mods, boom):
    mods(boom=boom)
    llm, st = Router(), _state(n_turns=5)
    pol = Policy(llm, _cfg())
    assert pol.next_action(st).type == ActionType.ASK
    st.turns.append(Turn(Action(ActionType.ASK, "추가 질문입니다"), "네"))
    pol.next_action(st)
    assert llm.kinds() == ["step", "step"]
    err = _sub(st, "error")
    assert len(err) == 1 and err[0]["name"] == "consult" and "consult" in pol.subagent_summary(st)["disabled"]


def test_parser_bug_and_bad_json_give_not_ok_and_case_continues(mods):
    mods(boom="parse")
    llm, st = Router(), _state(n_turns=5)
    pol = Policy(llm, _cfg())
    assert pol.next_action(st).type == ActionType.ASK
    e = _sub(st, "call")[0]
    assert not e["ok"] and "parser" in e["error"] and "전문의 자문" not in llm.user()
    assert pol.subagent_summary(st)["fail"] == 1
    mods()
    llm, st = Router(consult="엉망 {"), _state(n_turns=5)
    assert Policy(llm, _cfg()).next_action(st).type == ActionType.ASK and not _sub(st, "call")[0]["ok"]
    llm, st = Router(raise_for={"consult"}), _state(n_turns=5)
    assert Policy(llm, _cfg()).next_action(st).type == ActionType.ASK and not _sub(st, "call")[0]["ok"]


def test_missing_modules_are_skipped(mods):
    mods(consult=False, advocate=False)
    llm, st = Router(), _state(n_turns=5)
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    pol.next_action(st)
    assert llm.kinds() == ["step", "step"]
    assert [(e["name"], e["reason"]) for e in _sub(st, "skip")] == [("consult", "module_missing")]


# ---------------------------------------------------------------- advocate

def _fake_check(state, **kw):
    return {"dx": "A", "p": 0.8, "reasons": ["weak_support"], "why_ko": "근거 부족", "prompt_ko": "[반론 점검] A 고정?",
            "suggested_actions": []}


def test_advocate_at_anchoring_moment_once(mods, monkeypatch):
    calls = mods(route=(None, 0.0, []))
    monkeypatch.setattr(anchoring, "anchoring_check", _fake_check)
    monkeypatch.setattr(anchoring, "initial_differential", lambda info: [])
    llm, st = Router(), _state(n_turns=5)
    pol = Policy(llm, _cfg(use_anchoring=True, advocate_on_anchoring=True))
    pol.next_action(st)
    assert llm.kinds() == ["advocate", "step"]
    user = llm.user()
    assert "[반론 점검] A 고정?" in user and "반대 의견 검토 (참고 의견, 사실 근거 아님): A 말고 C도 설명됩니다." in user
    assert ("advocate", "A", "'A' 조기 고정 의심: 근거 부족") in calls
    assert _sub(st, "call")[0]["trigger"] == "anchoring" and st.ddx_ledger.refs_list()[0]["dx"] == "C"


def test_advocate_not_at_anchoring_moment_by_default(mods, monkeypatch):
    calls = mods(route=(None, 0.0, []))
    monkeypatch.setattr(anchoring, "anchoring_check", _fake_check)
    monkeypatch.setattr(anchoring, "initial_differential", lambda info: [])
    llm, st = Router(), _state(n_turns=5)
    Policy(llm, _cfg(use_anchoring=True)).next_action(st)
    assert llm.kinds() == ["step"] and "[반론 점검] A 고정?" in llm.user()  # the check is shown, no advocate call
    assert not [c for c in calls if c[0] == "advocate"] and not _sub(st, "call")


def _assess(score):
    def fn(state, proposed_dx=None, cfg=None, params=None):
        return Assessment(score, {}, "continue", [], proposed_dx or "")
    return fn


def test_advocate_before_review_goes_into_review_view(mods, monkeypatch):
    mods(route=(None, 0.0, []))
    monkeypatch.setattr(confidence, "assess", _assess(0.3))  # < advocate_conf_below (0.4)
    llm, st = Router(step=DX), _state(n_turns=4)
    st.safety_pushback = True
    pol = Policy(llm, _cfg())
    act = pol.next_action(st)
    assert act.type == ActionType.DIAGNOSE and llm.kinds() == ["step", "advocate", "review"]
    assert prompts.ADVOCATE_REVIEW_NOTE.format(text="A 말고 C도 설명됩니다.") in llm.user("review")
    assert _sub(st, "call")[0]["trigger"] == "pre_review"
    # at most one advocate call per case
    st.turns.append(Turn(Action(ActionType.ASK, "추가 질문입니다"), "네"))
    pol.next_action(st)
    assert llm.kinds().count("advocate") == 1


def test_confidence_assessed_once_for_pushback_check_and_advocate(mods, monkeypatch):
    mods(route=(None, 0.0, []))
    calls = []

    def counting(state, proposed_dx=None, cfg=None, params=None):
        calls.append(proposed_dx)
        return Assessment(0.3, {}, "continue", [], proposed_dx or "")
    monkeypatch.setattr(confidence, "assess", counting)
    llm, st = Router(step=DX), _state(n_turns=4)
    st.safety_pushback = True
    Policy(llm, _cfg(use_confidence=True)).next_action(st)
    assert llm.kinds() == ["step", "advocate", "review"]
    assert calls == ["A"]  # the confidence check logged it and the advocate trigger reused it
    assert [e["score"] for e in st.safety_log if e.get("layer") == "confidence"] == [0.3]


def test_advocate_not_called_when_confident_or_no_review(mods, monkeypatch):
    mods(route=(None, 0.0, []))
    for score in (0.9, 0.4):  # 0.4: not below the default cut-off
        monkeypatch.setattr(confidence, "assess", _assess(score))
        llm, st = Router(step=DX), _state(n_turns=4)
        st.safety_pushback = True
        Policy(llm, _cfg()).next_action(st)
        assert llm.kinds() == ["step", "review"]
    monkeypatch.setattr(confidence, "assess", _assess(0.1))
    llm, st = Router(step=DX), _state(n_turns=4)
    st.safety_pushback, st.reviews = True, [{}] * MAX_REVIEWS  # the review is not run: neither is the advocate
    Policy(llm, _cfg()).next_action(st)
    assert llm.kinds() == ["step"]


# ---------------------------------------------------------------- LLM radiology

def test_radiology_merges_findings_marked_and_keeps_code_reading(mods):
    mods(route=(None, 0.0, []))
    llm, st = Router(), _state(n_turns=3, results=[("흉부 X선", LONG_CXR)])
    pol = Policy(llm, _cfg())
    pol.next_action(st)
    assert llm.kinds() == ["radiology", "step"]
    rad_user = llm.user("radiology")
    assert "[결과 원문]" in rad_user and "[코드 판독(참고)] [흉부 X선]" in rad_user
    by_item = {f.item: f for f in st.findings.items}
    f = by_item["우측 하엽 경결"]
    assert f.source == LLM_RADIOLOGY_SOURCE and f.status == "양성" and "LLM 판독" in f.detail and f.turn == 4
    # the code reading of the same item is not replaced by the LLM reading
    assert by_item["흉막 삼출"].source == CODE_SOURCE and by_item["흉막 삼출"].status == "양성"
    crit = [c for c in st.result_criticals if c["concept"].startswith("llm:")]
    assert crit and crit[0]["label"] == "긴장성 기흉" and crit[0]["polarity"] == "present"
    user = llm.user()
    assert "결과 판독 보조(LLM) (참고 의견, 사실 근거 아님): [흉부 X선] 우측 하엽 경결" in user
    assert "긴장성 기흉 (LLM 판독)" in user  # one-time critical alert
    reading = [e for e in st.safety_log if e.get("kind") == "reading"][0]
    assert reading["llm_called"] and "호출함" in reading["msg"]
    assert pol.subagent_summary(st)["llm_radiology_findings"] == 3


def test_radiology_cap_one_per_case_and_bad_json(mods):
    mods(route=(None, 0.0, []))
    llm, st = Router(radiology="판독 불가"), _state(n_turns=5, results=[("흉부 X선", LONG_CXR)])
    pol = Policy(llm, _cfg())
    assert pol.next_action(st).type == ActionType.ASK
    assert not _sub(st, "call")[0]["ok"] and not [f for f in st.findings.items if f.source == LLM_RADIOLOGY_SOURCE]
    st.turns.append(Turn(Action(ActionType.TEST, "복부 CT"), LONG_CXR.replace("흉수", "복수")))
    pol.next_action(st)
    assert llm.kinds() == ["radiology", "step", "step"]
    assert [(e["name"], e["reason"]) for e in _sub(st, "skip")] == [("radiology", "radiology_cap")]


def test_radiology_needs_result_interpreter(mods):
    mods(route=(None, 0.0, []))
    llm, st = Router(), _state(n_turns=5, results=[("흉부 X선", LONG_CXR)])
    Policy(llm, _cfg(use_result_interpreter=False)).next_action(st)
    assert llm.kinds() == ["step"]


# ---------------------------------------------------------------- ledger refs

def test_ddx_refs_are_reference_only_until_the_model_adopts_them():
    led = DdxLedger()
    led.update([{"dx": "A", "p": 0.6}])
    assert led.add_ref("C", "발열") and not led.add_ref("C") and not led.add_ref("A")
    assert [d["dx"] for d in led.as_list()] == ["A"] and "참고(자문, 미확인): C(발열)" in led.render()
    led.update([{"dx": "C", "p": 0.3}])
    assert led.refs == [] and [d["dx"] for d in led.as_list()] == ["A", "C"]
    for i in range(10):
        led.add_ref(f"후보{i}")
    assert len(led.refs) == 4


# ---------------------------------------------------------------- whole case

def test_run_case_summary(mods):
    from pathlib import Path

    from doctor_agent.agent.loop import run_case
    from eval.simulator import CaseFileEnvironment

    mods()
    case = Path(__file__).resolve().parents[1] / "data/sample_cases/synthetic_001.json"
    steps = [_step("ASK", f"질문 {i}번 있나요?") for i in range(6)] + [DX]  # consult from turn 5
    cfg = Config()
    cfg.agent = _cfg()
    llm = Router(step=steps)
    res = run_case(CaseFileEnvironment.from_file(case), llm, cfg)
    s = res["subagents"]
    assert res["diagnosis"] and s["enabled"] and s["calls"].get("consult:cardio") == 1 and s["ok"] >= 1
    assert res["llm_calls"] == llm.call_count and {"calls", "ok", "fail", "skipped", "refs"} <= set(s)
    cfg.agent = _cfg(use_subagents=False)
    llm = Router(step=[_step("ASK", f"질문 {i}번 있나요?") for i in range(4)] + [DX])
    res = run_case(CaseFileEnvironment.from_file(case), llm, cfg)
    assert res["subagents"]["calls"] == {} and "consult" not in llm.kinds()


def test_whole_case_with_subagent_internals_raising(mods, monkeypatch):
    """A bug in the orchestrator itself is caught by the policy guard (dev mode would otherwise propagate it)."""
    from doctor_agent.agent.subagents import orchestrator

    mods()

    def boom(*a, **k):
        raise RuntimeError("orchestrator bug")
    monkeypatch.setattr(orchestrator.SubagentManager, "before_step", boom)
    monkeypatch.setattr(orchestrator.SubagentManager, "radiology", boom)
    monkeypatch.setattr(orchestrator.SubagentManager, "advocate_for_review", boom)
    llm, st = Router(), _state(n_turns=5, results=[("흉부 X선", LONG_CXR)])
    assert Policy(llm, _cfg()).next_action(st).type == ActionType.ASK
    assert {e["layer"] for e in st.safety_log if e.get("error")} == {"subagent"}


def test_trigger_defaults_and_env_overrides(monkeypatch):
    """Calibrated defaults (docs/experiments.md "trigger calibration") and their environment overrides."""
    cfg = Config().agent
    assert (cfg.anchoring_min_turns, cfg.consult_min_turns, cfg.consult_min_share) == (5, 5, 0.6)
    assert (cfg.advocate_conf_below, cfg.advocate_on_anchoring) == (0.4, False)
    assert cfg.anchoring_min_turns == anchoring.MIN_TURNS
    for k, v in {"AGENT_ANCHORING_MIN_TURNS": "3", "AGENT_CONSULT_MIN_TURNS": "4", "AGENT_CONSULT_MIN_SHARE": "0.8",
                 "AGENT_CONSULT_LOW_CONF": "0.3", "AGENT_CONSULT_LOW_CONF_TURNS": "2", "AGENT_CONSULT_LOW_CONF_AFTER": "8",
                 "AGENT_ADVOCATE_CONF_BELOW": "0.65", "AGENT_ADVOCATE_ON_ANCHORING": "1"}.items():
        monkeypatch.setenv(k, v)
    cfg = Config().agent
    assert (cfg.anchoring_min_turns, cfg.consult_min_turns, cfg.consult_min_share) == (3, 4, 0.8)
    assert (cfg.consult_low_conf, cfg.consult_low_conf_turns, cfg.consult_low_conf_after) == (0.3, 2, 8)
    assert (cfg.advocate_conf_below, cfg.advocate_on_anchoring) == (0.65, True)


def test_consult_min_turns_is_configurable(mods):
    mods()
    llm = Router()
    Policy(llm, _cfg(consult_min_turns=3)).next_action(_state(n_turns=3))
    assert llm.kinds() == ["consult", "step"]


def test_subagent_result_contract_fields():
    r = SubagentResult("x", True)
    assert (r.hint_ko, r.ddx_add, r.suggested_actions, r.red_flags, r.raw) == ("", [], [], [], {})
    assert SubagentCall("x", [], None).max_chars_out == 600


# ---------------------------------------------------------------- time budget (fake clock)

class ClockRouter(OptRouter):
    """Router on a fake clock: a main call (step / review / final) takes main_s, a sub-agent call sub_s. A call that
    would end after its deadline is cut there and raises LLMTimeout, which is what the real watchdog does."""

    def __init__(self, now, main_s, sub_s, **kw):
        super().__init__(**kw)
        self.now, self.main_s, self.sub_s, self.timeouts = now, main_s, sub_s, []

    def chat(self, messages, **opts):
        from doctor_agent.llm.client import LLMTimeout

        main = messages[0]["content"] in (prompts.SYSTEM, prompts.REVIEW_SYSTEM)
        dur, dl = (self.main_s if main else self.sub_s), opts.get("deadline")
        if dl is not None and self.now[0] + dur > dl + 1e-9:
            self.now[0] = max(self.now[0], dl)
            self.timeouts.append("main" if main else "sub")
            raise LLMTimeout("cut at the deadline")
        self.now[0] += dur
        return super().chat(messages, **opts)


def _asks(n=70):
    return [_step("ASK", f"질문 {i}번 증상이 있나요?") for i in range(n)]


def test_subagent_time_block_uses_live_budget_and_main_latency():
    now = [0.0]
    cfg = Config()
    cfg.agent = _cfg(subagent_time_factor=3.0)
    budget = CaseBudget(200.0, 0.9, 40.0, clock=lambda: now[0])
    guard = GuardedLLM(ClockRouter(now, main_s=20.0, sub_s=50.0, step=_asks()), cfg, budget)
    why, info = guard.subagent_time_block()  # nothing measured yet: min_call_s (5 s) is the estimate
    assert why == "" and info == {"left_s": 160.0, "est_call_s": cfg.agent.min_call_s, "factor": 3.0}
    guard.chat(prompts.build_step_messages("[처음 정보] x", 0, 60, []))
    guard.chat([{"role": "system", "content": CONSULT_SYS}, {"role": "user", "content": "x"}])  # sub-agent: 50 s
    why, info = guard.subagent_time_block()
    assert why == "" and info["est_call_s"] == 20.0 and info["left_s"] == 90.0  # sub-agent latency is not the estimate
    assert guard.stats()["latency_main_s"] == [20.0] and guard.stats()["latency_sub_s"] == [50.0]
    now[0] = 101.0  # 59 s left before the reserve < 3 x 20 s
    assert guard.subagent_time_block()[0] == "low_time_est"
    now[0] = 180.0  # degraded now (0.9 x 200), even if the policy's flag was set earlier in the turn
    assert guard.subagent_time_block()[0] == "low_time"
    assert GuardedLLM(Router(), cfg, CaseBudget(0, 0.6, 45)).subagent_time_block() == ("", {})


def test_orchestrator_skips_when_time_left_is_short(mods):
    mods()
    now = [0.0]
    cfg = Config()
    cfg.agent = _cfg(degrade_at_frac=0.99)
    budget = CaseBudget(200.0, 0.99, 40.0, clock=lambda: now[0])
    inner = ClockRouter(now, main_s=20.0, sub_s=20.0, step=_asks())
    guard = GuardedLLM(inner, cfg, budget)
    guard.latencies.append((True, 20.0))
    now[0] = 101.0  # 59 s before the reserve: < 3 x 20 → skipped, the step still runs
    st = _state(n_turns=5)
    Policy(guard, cfg.agent).next_action(st)
    assert inner.kinds() == ["step"] and not inner.timeouts
    skip = _sub(st, "skip")[0]
    assert skip["reason"] == "low_time_est" and skip["time"] == {"left_s": 59.0, "est_call_s": 20.0, "factor": 3.0}
    assert "예상 호출 시간" in skip["msg"]
    now[0] = 100.0  # 60 s >= 3 x 20 → the consult runs, then the step
    Policy(guard, cfg.agent).next_action(_state(n_turns=5))
    assert inner.kinds()[-2:] == ["consult", "step"] and not inner.timeouts


@pytest.mark.parametrize("factor", [3.0, 0.0])
def test_subagent_call_never_costs_the_step_or_overruns(mods, factor):
    """Whole case on a fake clock (budget 200 s, reserve 40 s → exploratory deadline 160 s, main calls 20 s, the consult
    25 s). The routed consult becomes due at turn 8 (t = 140 s, 20 s before the deadline). With the pre-call check
    (factor 3) it is skipped and the step still runs; without it (factor 0) the consult is cut at the deadline and the
    step of that turn is lost. The case ends inside the budget either way (the final reserve is never touched)."""
    from pathlib import Path

    from doctor_agent.agent.loop import run_case
    from eval.simulator import CaseFileEnvironment

    mods()
    now = [0.0]
    cfg = Config()
    cfg.agent = _cfg(case_time_budget_s=200.0, final_reserve_s=40.0, degrade_at_frac=0.99, consult_min_turns=7,
                     subagent_time_factor=factor)
    llm = ClockRouter(now, main_s=20.0, sub_s=25.0, step=_asks())
    case = Path(__file__).resolve().parents[1] / "data/sample_cases/synthetic_001.json"
    res = run_case(CaseFileEnvironment.from_file(case), llm, cfg, clock=lambda: now[0])
    assert now[0] <= 200.0 and res["runtime"]["forced"] == "time_budget" and res["diagnosis"]
    s = res["subagents"]
    if factor:
        assert s["calls"] == {} and s["skipped"] == {"consult:low_time_est": 1}
        assert llm.timeouts == [] and res["n_turns"] == 9  # 8 steps + the final diagnosis
    else:
        assert s["calls"] == {"consult:cardio": 1} and s["fail"] == 1
        assert llm.timeouts == ["sub"] and res["n_turns"] == 8  # the step of turn 8 was lost


def test_pre_review_advocate_sees_live_degraded_mode(mods, monkeypatch):
    """The Policy's degraded flag is set once per turn; the main call of this turn crosses the threshold, so the
    pre-review advocate (called after it) must skip on the live budget."""
    mods(route=(None, 0.0, []))
    monkeypatch.setattr(confidence, "assess", _assess(0.1))
    now = [0.0]
    cfg = Config()
    cfg.agent = _cfg()
    budget = CaseBudget(300.0, 0.6, 45.0, clock=lambda: now[0])
    inner = ClockRouter(now, main_s=30.0, sub_s=30.0, step=DX)
    guard = GuardedLLM(inner, cfg, budget)
    now[0] = 170.0  # not degraded at the start of the turn (< 180 s) ...
    pol = Policy(guard, cfg.agent)
    pol.degraded = budget.degraded()
    st = _state(n_turns=4)
    st.safety_pushback = True
    pol.next_action(st)  # ... but the step call ends at 200 s
    assert "advocate" not in inner.kinds()
    assert [(e["name"], e["reason"]) for e in _sub(st, "skip")] == [("advocate", "low_time")]
