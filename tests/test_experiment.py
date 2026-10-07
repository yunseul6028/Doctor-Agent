"""eval/experiment.py: profiles, case lists, cost estimate, endpoint presets, CLI wiring, usage metering. No LLM calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval import experiment as ex  # noqa: E402
from eval import run_local  # noqa: E402
from eval.usage import UsageMeter, attach  # noqa: E402

SAMPLE = ROOT / "data/sample_cases/synthetic_001.json"


# ---------------------------------------------------------------- profiles / case lists

def test_committed_profiles_resolve():
    cfg = ex.load_config()
    smoke = ex.resolve_profile(cfg, "smoke")
    dev = ex.resolve_profile(cfg, "dev")
    assert len(smoke["cases"]) == 5 and [c["name"] for c in smoke["conditions"]] == ["v6"]
    assert len(dev["cases"]) == 50 and [c["name"] for c in dev["conditions"]] == ["v6", "v6-no-kb", "v6-no-safety",
                                                                                   "v6-no-advisors"]
    assert dev["conditions"][1]["env"] == {"AGENT_USE_KB": "0"}
    assert dev["conditions"][3]["env"] == {"AGENT_USE_CONFIDENCE": "0", "AGENT_USE_ANCHORING": "0",
                                           "AGENT_USE_PLANNER": "0", "AGENT_USE_TRIAGE": "0"}
    # the result-interpreter ablation exists but is not part of dev (cost); it can be named explicitly
    assert "v6-no-interp" not in [c["name"] for c in dev["conditions"]]
    no_interp = ex.resolve_profile(cfg, "smoke", ["v6-no-interp"])["conditions"][0]
    assert no_interp["env"] == {"AGENT_USE_RESULT_INTERPRETER": "0"} and no_interp["commit"] is None
    # same for the specialist sub-agent ablation
    assert "v6-no-subagents" not in [c["name"] for c in dev["conditions"]]
    no_sub = ex.resolve_profile(cfg, "smoke", ["v6-no-subagents"])["conditions"][0]
    assert no_sub["env"] == {"AGENT_USE_SUBAGENTS": "0"} and no_sub["commit"] is None
    # smoke is a subset of dev and covers every set
    dev_paths = {p.resolve() for _, p in dev["cases"]}
    assert all(p.resolve() in dev_paths for _, p in smoke["cases"])
    assert {s for s, _ in smoke["cases"]} == {s for s, _ in dev["cases"]}
    full = ex.resolve_profile(cfg, "full")
    assert len(full["cases"]) > len(dev["cases"])


def test_condition_override_and_unknown():
    cfg = ex.load_config()
    p = ex.resolve_profile(cfg, "smoke", ["v5-baseline", "v6"])
    assert p["conditions"][0]["commit"] and p["conditions"][1]["commit"] is None
    with pytest.raises(SystemExit):
        ex.resolve_profile(cfg, "smoke", ["nope"])
    with pytest.raises(SystemExit):
        ex.resolve_profile(cfg, "nope")


def test_committed_case_lists_match_generator(tmp_path):
    """The committed lists are exactly what --regen-case-lists produces (deterministic)."""
    cfg = ex.load_config()
    for prof in ("smoke", "dev"):
        (tmp_path / Path(cfg["profiles"][prof]["cases_file"]).parent).mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").symlink_to(ROOT / "data")
    ex.regen_case_lists(cfg, root=tmp_path)
    for prof in ("smoke", "dev"):
        rel = cfg["profiles"][prof]["cases_file"]
        assert ex.read_case_list(tmp_path / rel) == ex.read_case_list(ROOT / rel)


def test_stratified_sample():
    items = [("a", f"a{i}") for i in range(60)] + [("b", f"b{i}") for i in range(30)] + [("c", f"c{i}") for i in range(3)]
    s = ex.stratified_sample(items, 10, seed=1)
    assert s == ex.stratified_sample(items, 10, seed=1) and len(s) == 10 and s == sorted(s)
    counts = {k: sum(1 for x, _ in s if x == k) for k in "abc"}
    assert counts["c"] >= 1 and counts["a"] > counts["b"] > 0
    assert ex.stratified_sample(items, 10, seed=2) != s
    assert ex.stratified_sample(items, 500, seed=1) == sorted(items)


# ---------------------------------------------------------------- estimate

def _write_run(d: Path, name: str, model: str, rows: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"run_{name}.json").write_text(json.dumps({"doctor_model": model, "cases": rows}), encoding="utf-8")


def test_history_stats_and_estimate(tmp_path):
    _write_run(tmp_path, "1", "gemini", [{"llm_calls": 6, "n_turns": 5}, {"llm_calls": 10, "n_turns": 9}])
    _write_run(tmp_path, "2", "dummy", [{"llm_calls": 100, "n_turns": 100}])  # ignored
    _write_run(tmp_path, "3", "gpt-oss", [{"llm_calls": 8, "n_turns": 7,
                                           "usage": {"calls": 8, "prompt_tokens": 16000, "completion_tokens": 4000}}])
    st = ex.history_stats([tmp_path], "gpt-oss")
    assert st["files"] == 2 and st["doctor_calls_per_case"] == 8 and st["patient_calls_per_case"] == 7
    assert st["prompt_tokens_per_call"] == 2000 and st["completion_tokens_per_call"] == 500
    assert st["token_source"].startswith("same model")

    est = ex.estimate([10, 10], st, margin=1.0, price_in=100.0, price_out=1000.0)
    assert est["per_condition"][0]["doctor_calls"] == 80 and est["total"]["doctor_calls"] == 160
    assert est["total"]["prompt_tokens"] == 320_000 and est["total"]["completion_tokens"] == 80_000
    assert est["cost"] == pytest.approx(0.32 * 100 + 0.08 * 1000)
    assert est["total"]["patient_calls"] == 0 and est["total"]["judge_calls"] == 0
    both = ex.estimate([10], st, patient="llm", judge="llm", margin=1.0)
    assert both["total"]["patient_calls"] == 70 and both["total"]["judge_calls"] == 10 and both["cost"] is None
    assert ex.estimate([10], st, doctor="dummy")["total"]["doctor_calls"] == 0


def test_estimate_defaults_and_guard(tmp_path):
    st = ex.history_stats([tmp_path / "empty"])
    est = ex.estimate([50, 50], st, margin=1.0)
    assert est["basis"]["calls_source"] == "default"
    assert est["total"]["doctor_calls"] == round(50 * ex.DEFAULTS["doctor_calls_per_case"]) * 2
    assert ex.needs_confirmation(est, max_calls=300, max_cost=None)
    assert not ex.needs_confirmation(ex.estimate([5], st, margin=1.0), max_calls=300, max_cost=None)
    priced = ex.estimate([5], st, margin=1.0, price_in=1e6, price_out=0.0)
    assert any("KRW" in r for r in ex.needs_confirmation(priced, max_calls=10_000, max_cost=100.0))
    text = ex.format_estimate(est, ["v6", "v6-no-kb"])
    assert "TOTAL" in text and "price unknown" in text and "EXPERIMENT_PRICE_IN_PER_M" in text and "KRW" not in text
    assert "KRW" in ex.format_estimate(priced, ["v6"]) and "price unknown" not in ex.format_estimate(priced, ["v6"])


# ---------------------------------------------------------------- endpoints

def test_endpoint_presets_never_leak_keys():
    env = {"LLM_BASE_URL": "https://gem.example/v1/", "LLM_API_KEY": "gem-secret", "LLM_MODEL": "gemini-x",
           "LOCAL_LLM_BASE_URL": "https://gpu.example/v1", "LOCAL_LLM_API_KEY": "gpu-secret",
           "LOCAL_LLM_MODEL": "openai/gpt-oss-20b"}
    remote = ex.doctor_endpoint_env("local", env)
    assert remote == {"DOCTOR_LLM_BASE_URL": "https://gpu.example/v1", "DOCTOR_LLM_API_KEY": "gpu-secret",
                      "DOCTOR_LLM_MODEL": "openai/gpt-oss-20b"}
    # gemini preset: Pro doctor, Flash patient/judge; the shared LLM_MODEL (flash in .env.example) is not the doctor
    gem = ex.doctor_endpoint_env("gemini", {**env, "DOCTOR_LLM_MODEL": "should-be-overridden"})
    assert gem["DOCTOR_LLM_MODEL"] == "gemini-3.1-pro-preview" and gem["DOCTOR_LLM_API_KEY"] == "gem-secret"
    assert gem["DOCTOR_LLM_BASE_URL"] == "https://gem.example/v1/"
    for role in ("PATIENT", "JUDGE"):
        assert gem[f"{role}_LLM_MODEL"] == "gemini-3.6-flash" and gem[f"{role}_LLM_API_KEY"] == "gem-secret"
        assert gem[f"{role}_LLM_BASE_URL"] == "https://gem.example/v1/"
    gem = ex.doctor_endpoint_env("gemini", {"LLM_API_KEY": "k", "GEMINI_DOCTOR_LLM_MODEL": "gemini-pro-x",
                                            "GEMINI_PATIENT_LLM_MODEL": "flash-x", "JUDGE_LLM_MODEL": "my-judge"})
    assert gem["DOCTOR_LLM_MODEL"] == "gemini-pro-x" and gem["PATIENT_LLM_MODEL"] == "flash-x"
    assert gem["DOCTOR_LLM_BASE_URL"].startswith("https://generativelanguage.googleapis.com/")
    assert not any(k.startswith("JUDGE_") for k in gem)  # an explicit per-role model is kept
    assert ex.doctor_endpoint_env("gemini", {"LLM_API_KEY": "k", "GEMINI_LLM_MODEL": "legacy"})["DOCTOR_LLM_MODEL"] == "legacy"
    loc = ex.doctor_endpoint_env("local", {})
    assert loc["DOCTOR_LLM_BASE_URL"].startswith("http://localhost") and loc["DOCTOR_LLM_MODEL"] == "gpt-oss:20b"
    assert ex.doctor_endpoint_env("env", env) == {} and ex.doctor_endpoint_env("dummy", env) == {}
    with pytest.raises(SystemExit):
        ex.doctor_endpoint_env("gemini", {"LLM_MODEL": "x"})  # no API key
    assert ex.describe_doctor("env", {}) == ("gemini-3.1-pro-preview", "generativelanguage.googleapis.com")
    model, host = ex.describe_doctor("local", {**env, **remote})
    assert (model, host) == ("openai/gpt-oss-20b", "gpu.example") and "secret" not in model + host


def test_parse_env_pairs():
    assert ex.parse_env_pairs(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        ex.parse_env_pairs(["nokey"])


# ---------------------------------------------------------------- CLI wiring

def test_build_command():
    cmd = ex.build_command(ROOT, [("sample", SAMPLE)], doctor="llm", patient="keyword", judge="none",
                           persona="standard", workers=3, label="dev:v6", out=Path("/tmp/o"), meta={"condition": "v6"})
    assert cmd[1].endswith("eval/run_local.py") and "--no-view" in cmd
    assert cmd[cmd.index("--label") + 1] == "dev:v6" and cmd[cmd.index("--workers") + 1] == "3"
    assert json.loads(cmd[cmd.index("--meta") + 1]) == {"condition": "v6"}
    assert cmd[-1] == str(SAMPLE.resolve())


def test_estimate_only_and_threshold(capsys):
    assert ex.main(["--profile", "dev", "--doctor-endpoint", "local", "--estimate-only"]) == 0
    out = capsys.readouterr().out
    assert "cost estimate" in out and "v6-no-kb" in out
    # above --max-calls without --yes: refuse (runs nothing)
    assert ex.main(["--profile", "dev", "--doctor-endpoint", "local", "--max-calls", "1"]) == 2
    assert "NOT RUNNING" in capsys.readouterr().out


def test_end_to_end_dummy(tmp_path, monkeypatch):
    """Two conditions via real subprocesses (dummy doctor, keyword patient), compare, log, viewer, share page."""
    cases = tmp_path / "cases_alpha"
    cases.mkdir()
    for n in ("a1", "a2"):
        (cases / f"{n}.json").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = {"profiles": {"t": {"cases": [str(cases)], "conditions": ["v6", "v6-no-kb"], "workers": 2}},
           "conditions": {"v6": {"env": {}}, "v6-no-kb": {"env": {"AGENT_USE_KB": "0"}}}}
    pf = tmp_path / "profiles.json"
    pf.write_text(json.dumps(cfg), encoding="utf-8")
    md = tmp_path / "exp.md"
    md.write_text("# Log\n\n| Date | Change |\n|---|---|\n| 2026-01-01 | x |\n\n## Open issues\n- y\n", encoding="utf-8")
    out = tmp_path / "res"
    from eval import viewer

    monkeypatch.setattr(viewer, "RESULTS", viewer.RESULTS)  # restored after the test
    code = ex.main(["--profile", "t", "--profiles-file", str(pf), "--doctor-endpoint", "dummy", "--out", str(out),
                    "--no-view", "--log", "--experiments-md", str(md), "--share", str(tmp_path / "share.html"),
                    "--note", "unit"])
    assert code == 0
    runs = sorted(out.glob("run_*.json"))
    assert len(runs) == 2
    datas = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
    assert {d["experiment"]["condition"] for d in datas} == {"v6", "v6-no-kb"}
    assert all(d["usage"] == {} and d["doctor_model"] == "dummy" for d in datas)
    text = md.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = lines.index("| 2026-01-01 | x |")
    assert "experiment `t` / v6" in lines[i + 1] and "experiment `t` / v6-no-kb" in lines[i + 2]
    assert text.index("## Open issues") < text.index(ex.AUTO_HEADING) and "Shared cases: 2" in text
    assert (out / "viewer.html").exists() and (tmp_path / "share.html").read_text(encoding="utf-8").startswith("<title>")


def test_rescore_uses_current_scorer(tmp_path):
    case = json.loads(SAMPLE.read_text(encoding="utf-8"))
    row = {"case": "synthetic_001", "set": "sample", "path": str(SAMPLE), "diagnosis": case["diagnosis"], "n_turns": 2,
           "turns": [{"type": "ASK", "content": "x", "response": "y"}], "judge": None,
           "scores": {"accuracy": None, "efficiency": None, "safety": None, "case_checks": None}, "score_error": "old"}
    p = tmp_path / "run_x.json"
    p.write_text(json.dumps({"cases": [row]}), encoding="utf-8")
    d = ex.rescore(p)
    r = d["cases"][0]
    assert r["scores"]["accuracy"] == 1.0 and r["scores"]["efficiency"] is not None and "score_error" not in r
    assert d["avg"]["accuracy"] == 1.0 and d["rescored"]["by"] == "eval/scorer.py"
    assert r["path"] == "data/sample_cases/synthetic_001.json"


# ---------------------------------------------------------------- usage metering

class _FakeCompletions:
    def __init__(self, usage):
        self.usage = usage

    def create(self, **kwargs):
        return SimpleNamespace(usage=self.usage, choices=[])


def _fake_llm(usage):
    return SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(usage)),
                                                  base_url="u"))


def test_usage_meter_wraps_sdk_client():
    meter = UsageMeter()
    llm = _fake_llm(SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140,
                                    completion_tokens_details=SimpleNamespace(reasoning_tokens=30)))
    assert attach(llm, meter)
    llm.client.chat.completions.create(model="m", messages=[])
    llm.client.chat.completions.create(model="m", messages=[])
    assert llm.client.base_url == "u"  # other attributes forwarded
    t = meter.totals()
    assert t["calls"] == 2 and t["prompt_tokens"] == 200 and t["completion_tokens"] == 80 and t["reasoning_tokens"] == 60
    nousage = UsageMeter()
    llm2 = _fake_llm(None)
    attach(llm2, nousage)
    llm2.client.chat.completions.create()
    assert nousage.totals()["no_usage"] == 1
    total = UsageMeter()
    total.merge(meter)
    total.merge({"calls": 1})
    assert total.totals()["calls"] == 3
    assert not attach(SimpleNamespace(), UsageMeter())  # DummyLLM-like: nothing to meter


def test_run_local_records_doctor_usage(tmp_path, monkeypatch):
    """run_local with an LLM doctor whose SDK is faked: per-case and run-level usage end up in the result file."""
    from doctor_agent.llm import client as client_mod

    script = iter([json.dumps({"type": "DIAGNOSE", "content": "급성 충수염", "confidence": 0.95, "ddx": []})] * 50)

    class FakeSDK:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            msg = SimpleNamespace(content=next(script), reasoning_content=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050))

    orig = client_mod.OpenAICompatClient.__init__

    def init(self, cfg, client=None, **kw):
        orig(self, cfg, client=FakeSDK(), **kw)

    monkeypatch.setattr(client_mod.OpenAICompatClient, "__init__", init)
    cases = tmp_path / "cases_alpha"
    cases.mkdir()
    (cases / "a1.json").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    out = run_local.main(["--doctor", "llm", "--patient", "keyword", "--judge", "none", "--no-view",
                          "--cases", str(cases), "--out", str(tmp_path / "res"), "--meta", '{"condition": "t"}'])
    d = json.loads(out.read_text(encoding="utf-8"))
    row = d["cases"][0]
    assert row["usage"]["calls"] >= 1 and row["usage"]["prompt_tokens"] == 1000 * row["usage"]["calls"]
    assert d["usage"]["doctor"]["calls"] == row["usage"]["calls"] and d["experiment"] == {"condition": "t"}
