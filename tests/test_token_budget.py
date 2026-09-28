"""scripts/token_budget.py (offline gpt-oss prompt token measurement) and the measured-token cost estimate in
eval/experiment.py. Small synthetic inputs; no LLM or network calls (tokenizer tests skip when tiktoken or its cached
encoding is unavailable)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import token_budget as tb  # noqa: E402
from doctor_agent.agent import prompts  # noqa: E402
from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.config import AgentConfig  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402
from eval import experiment as ex  # noqa: E402

CASE = {
    "initial": "45세 여성. 주호소: 2주 전부터 시작된 발열과 기침",
    "history": {"기침|가래": "2주 전부터 기침과 노란 가래가 있어요.", "발열|열": "저녁마다 38도까지 열이 나요.",
                "흡연|담배": "담배는 안 피워요."},
    "exam": {"폐 청진|호흡음": "우하엽에서 수포음이 들림", "활력징후|혈압": "혈압 120/80, 맥박 98회/분, 체온 38.2도"},
    "tests": {"흉부 x선|cxr": "우하엽 경화 소견", "혈액검사|cbc": "WBC 14,000/μL, 호중구 82%"},
    "diagnosis": "지역사회획득 폐렴",
}


def chars(text: str) -> int:  # deterministic stand-in tokenizer: one token per character
    return len(text)


# ---------------------------------------------------------------- harmony rendering / counters

def test_render_harmony_structure():
    msgs = prompts.build_step_messages("[처음 정보] 45세 여성", 0, 60, ["반드시 배제할 위험 질환: 폐색전증"])
    out = tb.render_harmony(msgs, "low", date="2026-09-28")
    assert out.startswith("<|start|>system<|message|>You are ChatGPT")
    assert "Reasoning: low" in out and "Current date: 2026-09-28" in out
    assert f"<|start|>developer<|message|># Instructions\n\n{prompts.SYSTEM}<|end|>" in out
    assert f"<|start|>user<|message|>{msgs[1]['content']}<|end|>" in out
    assert out.endswith("<|start|>assistant")
    assert "Reasoning: medium" in tb.render_harmony(msgs, "none")  # parameter not sent -> template default


def _tiktoken_or_skip():
    pytest.importorskip("tiktoken")
    try:
        return tb.TiktokenCounter()
    except Exception as e:  # noqa: BLE001 — encoding file not cached and no network
        pytest.skip(f"o200k_harmony unavailable: {e}")


def test_tiktoken_counts_control_tokens_once():
    count = _tiktoken_or_skip()
    assert all(count(x) == 1 for x in ("<|start|>", "<|message|>", "<|end|>", "<|channel|>"))
    assert count("<|start|>user<|message|><|end|>") == 4
    assert count("안녕하세요") < len("안녕하세요") * 2


def test_render_matches_openai_harmony():
    count = _tiktoken_or_skip()
    oh = pytest.importorskip("openai_harmony")
    msgs = prompts.build_review_messages("[처음 정보] 45세 여성", "폐렴", "수포음", 3, 60)
    enc = oh.load_harmony_encoding(oh.HarmonyEncodingName.HARMONY_GPT_OSS)
    conv = oh.Conversation.from_messages([
        oh.Message.from_role_and_content(oh.Role.SYSTEM, oh.SystemContent.new().with_reasoning_effort(
            oh.ReasoningEffort.LOW).with_conversation_start_date("2026-09-28")),
        oh.Message.from_role_and_content(oh.Role.DEVELOPER, oh.DeveloperContent.new().with_instructions(msgs[0]["content"])),
        oh.Message.from_role_and_content(oh.Role.USER, msgs[1]["content"])])
    ids = enc.render_conversation_for_completion(conv, oh.Role.ASSISTANT)
    assert enc.decode(ids) == tb.render_harmony(msgs, "low", "2026-09-28")
    assert len(ids) == count(tb.render_harmony(msgs, "low", "2026-09-28"))


def test_get_counter_specs():
    assert tb.get_counter("chars")("가나다라") == 3  # ceil(4 * 0.55)
    with pytest.raises(SystemExit):
        tb.get_counter("nope")


# ---------------------------------------------------------------- anatomy helpers

def _state() -> CaseState:
    s = CaseState(initial_info=CASE["initial"])
    s.findings.update([{"item": "기침", "status": "양성", "detail": "2주"}, {"item": "흡연", "status": "음성"}], 1)
    s.ddx_ledger.update([{"dx": "폐렴", "p": 0.6, "status": "유력", "for": ["수포음"]},
                         {"dx": "폐색전증", "p": 0.1, "status": "위험"}])
    for i in range(6):
        s.turns.append(Turn(Action(ActionType.ASK, f"질문 {i}번 내용입니다"), f"답변 {i} " * 10))
    return s


def test_split_view_sections_and_hard_cut():
    s = _state()
    parts = tb.split_view(s.view())
    assert set(parts) == {"view.initial", "view.findings", "view.ddx", "view.older_actions", "view.recent"}
    assert "".join(parts.values()).replace("[", "").count("처음 정보") == 1
    cut = s.view(max_chars=120)
    assert "\n…\n" in cut and tb.split_view(cut) == {"view.cut": cut}


def test_classify_hint_covers_policy_hints():
    assert tb.classify_hint("반드시 배제할 위험 질환: 폐색전증") == "hint.cant_miss"
    assert tb.classify_hint("아직 안 한 최소 안전 확인 (근거 지침): 심전도") == "hint.safety_checks"
    assert tb.classify_hint("결과가 제공되지 않은 요청 (다시 …): 뇌 MRI") == "hint.unavailable"
    assert tb.classify_hint("[임상 결정 규칙: …]\n■ Wells") == "hint.clinical_rules"
    assert tb.classify_hint("목표 턴 수를 넘었습니다. 충분히 확신하면 진단하세요.") == "hint.target_turns"
    assert tb.classify_hint("고려해 볼 다른 질환 (참고용 …): 결핵") == "hint.kb_candidates"
    assert tb.classify_hint("감별 포인트 A vs B (참고용 …)") == "hint.kb_discriminator"
    assert tb.classify_hint(prompts.LOW_TIME_HINT) == "hint.low_time"
    assert tb.classify_hint("정해진 형식의 JSON 한 줄로만 출력하세요.") == "retry.parse"
    assert tb.classify_hint("'흉부 CT'은(는) 이미 했습니다. 다른 행동을 고르세요.") == "retry.duplicate"
    assert tb.classify_hint("조영제 전 신기능 확인 필요 다른 행동을 고르세요.") == "retry.precondition"
    assert tb.classify_hint("무엇이든") == "hint.other"


def test_split_review_view():
    base = _state().view()
    view = base + "\n\n[지식베이스 경고] 여성 전용 상병" + "\n\n[진단 기준 대조: X] 판정: 충족"
    got, extras = tb.split_review_view(view, base)
    assert got == base
    assert extras["review.kb_warning"].startswith("\n\n[지식베이스 경고]")
    assert extras["review.criteria"].startswith("\n\n[진단 기준 대조")
    assert tb.split_review_view("other", base) == ("other", {})


def test_anatomy_parts_and_overhead():
    view = _state().view()
    hints = ["반드시 배제할 위험 질환: 폐색전증", "목표 턴 수를 넘었습니다."]
    msgs = prompts.build_step_messages(view, 6, 60, hints)
    a = tb.anatomy("step", msgs, chars, "low", view=view, hints=hints)
    assert a["tokens"] == len(tb.render_harmony(msgs, "low"))
    assert a["parts"]["system_prompt"] == len(prompts.SYSTEM)
    assert a["parts"]["harmony_overhead"] == a["tokens"] - len(prompts.SYSTEM) - len(msgs[1]["content"])
    assert a["parts"]["hint.cant_miss"] == len("\n- " + hints[0])
    # with a character counter the parts add up exactly
    assert sum(a["parts"].values()) == a["tokens"]


# ---------------------------------------------------------------- replay / summary

def test_scripted_doctor_plan_and_json():
    plan = tb.plan_actions(CASE)
    assert plan[0][0] == "ASK" and "기침" in plan[0][1]
    assert {t for t, _ in plan[:7]} == {"ASK", "EXAM", "TEST"} and len(plan) > 40
    d = tb.ScriptedDoctor(CASE)
    obj = json.loads(d.chat(prompts.build_step_messages("[처음 정보] x", 0, 60, [])))
    assert obj["type"] != "DIAGNOSE" and len(obj["ddx"]) == 5 and obj["findings"]
    assert d.dx[0] == CASE["diagnosis"] and d.call_count == 1


def test_replay_captures_step_review_final():
    cfg = AgentConfig(max_turns=8)
    calls = tb.replay_case(dict(CASE), cfg, checkpoints=(1, 5, 8))
    steps = [c for c in calls if c["kind"] == "step" and not c["probe"]]
    assert {c["turn"] for c in steps if c["attempt"] == 0} == set(range(1, 8))  # turn 8 = last turn -> final prompt
    finals = [c for c in calls if c["kind"] == "final"]
    assert {c["turn"] for c in finals} == {1, 5, 8} and all(c["probe"] for c in finals if c["turn"] < 8)
    reviews = [c for c in calls if c["kind"] == "review"]
    assert {c["turn"] for c in reviews} == {1, 5} and all(c["probe"] for c in reviews)
    assert all("review.proposal" in c["extras"] for c in reviews)
    # the prompt the model saw is what the builder produced from the captured view and hints
    s5 = next(c for c in steps if c["turn"] == 5 and c["attempt"] == 0)
    assert s5["messages"] == prompts.build_step_messages(s5["view"], 4, 8, s5["hints"])
    assert all(c["view_chars_uncapped"] >= len(c["view"]) or c["view_chars_uncapped"] > 0 for c in calls)


def test_replay_probes_do_not_change_the_run():
    cfg = AgentConfig(max_turns=6)
    with_probes = [c["messages"] for c in tb.replay_case(dict(CASE), cfg, (2, 4)) if c["kind"] == "step" and not c["probe"]]
    without = [c["messages"] for c in tb.replay_case(dict(CASE), cfg, ()) if c["kind"] == "step" and not c["probe"]]
    assert with_probes == without


def test_measure_summarize_report():
    cfg = AgentConfig(max_turns=6)
    rows = tb.measure([dict(CASE, _id="c1"), dict(CASE, _id="c2")], chars, cfg=cfg, checkpoints=(1, 5, 6))
    s = tb.summarize(rows, checkpoints=(1, 5, 6), max_view_chars=cfg.max_view_chars)
    assert s["by_turn"]["step"]["1"]["tokens"]["n"] == 2 and "6" not in s["by_turn"]["step"]
    assert s["by_turn"]["final"]["6"]["tokens"]["n"] == 2
    assert s["step_mean_by_turn"]["5"] > s["step_mean_by_turn"]["1"]  # the view grows with the turns
    assert s["parts"]["step"]["system_prompt"]["share"] > 0
    assert s["budget"]["max_prompt"] == max(r["tokens"] for r in rows)
    assert s["budget"]["fits"][str(tb.CONTEXT_WINDOW)]["prompt_only"] == 1.0
    assert s["per_call_mean_by_case_length"]["5"] > 0
    text = tb.format_report(s, "chars", 2, "low")
    assert "step    turn  1" in text and "Contributions (step prompts)" in text


def test_pct_and_dist():
    assert tb.pct([5, 1, 3, 2, 4], 50) == 3 and tb.pct([1, 2, 3, 4, 100], 95) == 100 and tb.pct([], 50) == 0
    assert tb.dist([1, 2, 3])["max"] == 3


def test_cli_writes_json(tmp_path, capsys):
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(CASE, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "tb.json"
    assert tb.main(["--cases", str(case_file), "--tokenizer", "chars", "--turns", "1,3", "--json-out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["cases"] == 1 and data["step_mean_by_turn"]["1"] > 0
    assert "gpt-oss prompt tokens" in capsys.readouterr().out


# ---------------------------------------------------------------- experiment.py cost estimate

def test_measured_prompt_tokens_per_call():
    curve = {1: 1000, 11: 2000}
    assert ex.measured_prompt_tokens_per_call(1, curve) == 1000
    assert ex.measured_prompt_tokens_per_call(11, curve) == pytest.approx(1500)  # mean of 1000..2000 step 100
    assert ex.measured_prompt_tokens_per_call(21, curve) == pytest.approx((1500 * 11 + 2000 * 10) / 21)
    assert ex.measured_prompt_tokens_per_call(5, {}) is None


def test_committed_measurement_table():
    curve = ex.MEASURED_STEP_TOKENS
    assert 1 in curve and len(curve) >= 5 and all(v > 0 for v in curve.values())
    assert list(curve.values()) == sorted(curve.values())  # grows with the turn


def test_load_measured_file_and_fallback(tmp_path):
    p = tmp_path / "tb.json"
    p.write_text(json.dumps({"step_mean_by_turn": {"1": 800, "10": 1800},
                             "meta": {"date": "2026-09-28", "cases": 3, "tokenizer": "t"}}), encoding="utf-8")
    m = ex.load_measured(p)
    assert m["curve"] == {1: 800.0, 10: 1800.0} and "3 cases" in m["source"]
    fb = ex.load_measured(tmp_path / "missing.json")
    assert fb["curve"] == ex.MEASURED_STEP_TOKENS and fb["source"] == ex.MEASURED_SOURCE


def _run(d: Path, name: str, model: str, rows: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"run_{name}.json").write_text(json.dumps({"doctor_model": model, "cases": rows}), encoding="utf-8")


def test_history_stats_uses_measured_tokens_for_gpt_oss(tmp_path):
    _run(tmp_path, "g", "gemini", [{"llm_calls": 10, "n_turns": 11,
                                    "usage": {"calls": 10, "prompt_tokens": 30000, "completion_tokens": 9000}}])
    measured = {"curve": {1: 1000, 11: 2000}, "source": "measured test"}
    st = ex.history_stats([tmp_path], "openai/gpt-oss-20b", measured=measured, effort="low")
    assert st["prompt_tokens_per_call"] == pytest.approx(1500)  # at the 11 turns/case of the history
    assert st["completion_tokens_per_call"] == ex.ASSUMED_COMPLETION_TOKENS["low"]
    assert st["token_source"].startswith("measured test") and "effort low" in st["token_source"]
    assert ex.history_stats([tmp_path], "gpt-oss:20b", measured=measured, effort="medium")[
        "completion_tokens_per_call"] == ex.ASSUMED_COMPLETION_TOKENS["medium"]
    # another model keeps its own recorded usage
    other = ex.history_stats([tmp_path], "gemini-2.5-flash", measured=measured)
    assert other["prompt_tokens_per_call"] == 3000 and other["token_source"].startswith("other model")
    # recorded gpt-oss usage beats the measurement
    _run(tmp_path, "o", "openai/gpt-oss-20b", [{"llm_calls": 4, "n_turns": 3,
                                               "usage": {"calls": 4, "prompt_tokens": 8000, "completion_tokens": 2000}}])
    same = ex.history_stats([tmp_path], "openai/gpt-oss-20b", measured=measured)
    assert same["prompt_tokens_per_call"] == 2000 and same["token_source"].startswith("same model")


def test_estimate_with_measured_default_curve(tmp_path):
    st = ex.history_stats([tmp_path / "empty"], "openai/gpt-oss-20b", measured={"curve": ex.MEASURED_STEP_TOKENS,
                                                                                  "source": ex.MEASURED_SOURCE})
    est = ex.estimate([10], st, margin=1.0)
    turns = ex.DEFAULTS["patient_calls_per_case"]
    assert est["basis"]["prompt_tokens_per_call"] == round(ex.measured_prompt_tokens_per_call(turns, ex.MEASURED_STEP_TOKENS))
    assert "measured" in ex.format_estimate(est, ["v6"])
