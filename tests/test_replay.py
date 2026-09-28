"""eval/replay.py: LLM record/replay cache. Fake SDK objects only; no network."""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.config import LLMConfig  # noqa: E402
from doctor_agent.llm.client import OpenAICompatClient  # noqa: E402
from eval import experiment as ex  # noqa: E402
from eval import replay  # noqa: E402
from eval import run_local  # noqa: E402
from eval.usage import UsageMeter, attach as meter_attach  # noqa: E402

SAMPLE = ROOT / "data/sample_cases/synthetic_001.json"


class FakeSDK:
    """Stands in for openai.OpenAI: answers with the last user message echoed, counts calls."""

    def __init__(self, answer=None):
        self.chat = SimpleNamespace(completions=self)
        self.calls = 0
        self.lock = threading.Lock()
        self.answer = answer

    def create(self, **kwargs):
        with self.lock:
            self.calls += 1
        text = self.answer if self.answer is not None else "echo:" + kwargs["messages"][-1]["content"]
        msg = SimpleNamespace(content=text, reasoning_content="thinking")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                                                     completion_tokens_details=SimpleNamespace(reasoning_tokens=5)))


def _client(sdk, cache, model="m", base_url="http://host-a/v1"):
    llm = OpenAICompatClient(LLMConfig(base_url=base_url, model=model, max_retries=2), client=sdk,
                             sleep=lambda s: None)
    replay.attach(llm, cache)
    return llm


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_auto_miss_then_hit_and_stats(tmp_path):
    sdk = FakeSDK()
    cache = replay.LLMCache("patient", "auto", tmp_path)
    llm = _client(sdk, cache)
    assert llm.chat(_msgs("hi")) == "echo:hi"
    assert llm.chat(_msgs("hi")) == "echo:hi"  # replayed through extract_text
    assert llm.chat(_msgs("other")) == "echo:other"
    assert sdk.calls == 2
    s = cache.stats()
    assert (s["hits"], s["misses"], s["recorded"], s["calls"]) == (1, 2, 2, 2)
    assert s["saved_prompt_tokens"] == 100 and s["saved_completion_tokens"] == 20 and s["saved_reasoning_tokens"] == 5
    assert s["hit_rate"] == pytest.approx(0.333, abs=1e-3) and s["entries"] == 2
    # persisted: a fresh instance (next run) replays without calling
    sdk2 = FakeSDK()
    llm2 = _client(sdk2, replay.LLMCache("patient", "replay", tmp_path))
    assert llm2.chat(_msgs("other")) == "echo:other" and sdk2.calls == 0
    entry = json.loads((tmp_path / "patient.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["role"] == "patient" and entry["model"] == "m" and entry["response"]["reasoning"] == "thinking"
    assert entry["response"]["usage"]["prompt_tokens"] == 100


def test_replay_miss_is_loud_and_never_calls(tmp_path):
    sdk = FakeSDK()
    cache = replay.LLMCache("judge", "replay", tmp_path)
    llm = _client(sdk, cache)
    # CacheMiss is a BaseException: OpenAICompatClient's retry loop (except Exception) must not swallow it
    with pytest.raises(replay.CacheMiss):
        llm.chat(_msgs("x"))
    assert sdk.calls == 0 and cache.stats()["misses"] == 1 and len(cache.miss_keys) == 1
    assert not issubclass(replay.CacheMiss, Exception)


def test_record_always_calls_and_last_entry_wins(tmp_path):
    sdk = FakeSDK(answer="first")
    rec = replay.LLMCache("doctor", "record", tmp_path)
    llm = _client(sdk, rec)
    llm.chat(_msgs("q"))
    sdk.answer = "second"
    llm.chat(_msgs("q"))
    assert sdk.calls == 2 and rec.stats()["hits"] == 0
    assert _client(FakeSDK(), replay.LLMCache("doctor", "replay", tmp_path)).chat(_msgs("q")) == "second"


def test_key_depends_on_salt_sample_idx_params_but_not_timeout():
    base = {"model": "m", "messages": _msgs("a"), "temperature": 0.2, "max_tokens": 100, "reasoning_effort": "low"}
    k = replay.request_key(base)
    assert replay.request_key({**base, "timeout": 12.3}) == k
    assert replay.request_key(dict(reversed(list(base.items())))) == k  # order-independent
    for other in (replay.request_key(base, salt="s1"), replay.request_key(base, sample_idx=1),
                  replay.request_key(base, endpoint="host-b/v1"), replay.request_key({**base, "temperature": 0.7}),
                  replay.request_key({**base, "max_tokens": 200}), replay.request_key({**base, "model": "m2"}),
                  replay.request_key({**base, "reasoning_effort": "high"}),
                  replay.request_key({**base, "response_format": {"type": "json_schema"}}),
                  replay.request_key({**base, "messages": _msgs("b")})):
        assert other != k


def test_salt_and_sample_idx_give_separate_entries(tmp_path):
    sdk = FakeSDK()
    _client(sdk, replay.LLMCache("patient", "auto", tmp_path)).chat(_msgs("q"))
    _client(sdk, replay.LLMCache("patient", "auto", tmp_path, salt="fresh")).chat(_msgs("q"))
    _client(sdk, replay.LLMCache("patient", "auto", tmp_path, sample_idx=1)).chat(_msgs("q"))
    assert sdk.calls == 3
    _client(sdk, replay.LLMCache("patient", "auto", tmp_path, salt="fresh")).chat(_msgs("q"))
    assert sdk.calls == 3
    # other endpoint = other key (same model name served elsewhere)
    _client(sdk, replay.LLMCache("patient", "auto", tmp_path), base_url="http://host-b/v1").chat(_msgs("q"))
    assert sdk.calls == 4


def test_concurrent_writers(tmp_path):
    sdk = FakeSDK()
    cache = replay.LLMCache("patient", "auto", tmp_path)
    errors = []

    def worker(t):
        try:
            llm = _client(sdk, cache)
            for i in range(40):
                llm.chat(_msgs(f"t{t}-{i}"))
                llm.chat(_msgs(f"shared-{i % 5}"))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    lines = (tmp_path / "patient.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(ln)["key"] for ln in lines)  # no interleaved/torn lines
    s = cache.stats()
    assert s["hits"] + s["misses"] == 8 * 80 and s["recorded"] == len(lines) == sdk.calls
    assert len(replay.LLMCache("patient", "replay", tmp_path)) == 8 * 40 + 5


def test_torn_line_skipped(tmp_path):
    _client(FakeSDK(), replay.LLMCache("judge", "auto", tmp_path)).chat(_msgs("q"))
    with (tmp_path / "judge.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"key": "abc", "resp')
    assert len(replay.LLMCache("judge", "auto", tmp_path)) == 1


def test_cache_sits_above_meter(tmp_path):
    """Hits never reach the usage meter: `usage` stays 'tokens actually billed'."""
    sdk = FakeSDK()
    meter = UsageMeter()
    llm = OpenAICompatClient(LLMConfig(), client=sdk)
    meter_attach(llm, meter)
    replay.attach(llm, replay.LLMCache("doctor", "auto", tmp_path))
    llm.chat(_msgs("q"))
    llm.chat(_msgs("q"))
    assert meter.totals()["calls"] == 1 and meter.totals()["prompt_tokens"] == 100


def test_make_caches_and_summary(tmp_path):
    c = replay.make_caches("auto", directory=tmp_path)
    assert c["doctor"] is None and c["patient"].mode == "auto" and c["judge"].mode == "auto"
    c = replay.make_caches("off", doctor_mode="replay", directory=tmp_path)
    assert c["doctor"].mode == "replay" and c["patient"] is None
    c = replay.make_caches("record", cache_doctor=True, directory=tmp_path)
    assert c["doctor"].mode == "record"
    assert replay.summary({"doctor": None}) is None and replay.format_summary(None) == "llm cache: off"
    c["doctor"]._hit({"usage": {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000}})
    block = replay.summary(c, price_in=100.0, price_out=400.0)
    assert block["saved_krw_doctor"] == 600.0 and "600 KRW" in replay.format_summary(block)
    assert not replay.attach(SimpleNamespace(), c["doctor"])  # DummyLLM-like
    assert not replay.attach(SimpleNamespace(client=FakeSDK()), None)


# ---------------------------------------------------------------- run_local / experiment wiring

def _patch_doctor_sdk(monkeypatch, sdk):
    from doctor_agent.llm import client as client_mod

    orig = client_mod.OpenAICompatClient.__init__

    def init(self, cfg, client=None, **kw):
        orig(self, cfg, client=sdk, **kw)

    monkeypatch.setattr(client_mod.OpenAICompatClient, "__init__", init)


def test_run_local_doctor_cache_record_then_replay(tmp_path, monkeypatch):
    dx = json.dumps({"type": "DIAGNOSE", "content": "급성 충수염", "confidence": 0.95, "ddx": []})
    sdk = FakeSDK(answer=dx)
    _patch_doctor_sdk(monkeypatch, sdk)
    cases = tmp_path / "cases_alpha"
    cases.mkdir()
    (cases / "a1.json").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    common = ["--doctor", "llm", "--patient", "keyword", "--judge", "none", "--no-view", "--cases", str(cases),
              "--out", str(tmp_path / "res"), "--cache-dir", str(tmp_path / "cache")]
    first = json.loads(run_local.main(common + ["--llm-cache", "auto", "--cache-doctor"]).read_text(encoding="utf-8"))
    n = sdk.calls
    assert n >= 1
    d = first["llm_cache"]["roles"]["doctor"]
    assert d["misses"] == n and d["recorded"] == n and set(first["llm_cache"]["roles"]) == {"doctor"}
    # identical rerun in replay mode: zero API calls, same result, usage reports nothing billed
    second = json.loads(run_local.main(common + ["--llm-cache", "replay", "--cache-doctor"]).read_text(encoding="utf-8"))
    assert sdk.calls == n
    assert second["llm_cache"]["roles"]["doctor"]["hits"] == n
    assert second["llm_cache"]["roles"]["doctor"]["saved_prompt_tokens"] == 100 * n
    assert second["usage"]["doctor"]["calls"] == 0
    assert second["cases"][0]["diagnosis"] == first["cases"][0]["diagnosis"]
    # replay with a new salt: miss -> batch aborted, no API call, exit code 4
    assert run_local.main(common + ["--doctor-cache", "replay", "--cache-salt", "new"]) is None
    assert run_local._last_exit == run_local.REPLAY_MISS_EXIT and sdk.calls == n
    # cache off: result JSON has llm_cache None
    off = json.loads(run_local.main(common).read_text(encoding="utf-8"))
    assert off["llm_cache"] is None and sdk.calls > n


def test_experiment_cache_flags_and_compare_line(tmp_path):
    assert ex.cache_args("off") == []
    a = ex.cache_args("auto", directory=tmp_path)
    assert a[:2] == ["--llm-cache", "auto"] and "--doctor-cache" not in a
    a = ex.cache_args("off", cache_doctor=True, salt="s", sample_idx=2, directory=tmp_path)
    assert a[a.index("--doctor-cache") + 1] == "auto" and a[a.index("--llm-cache") + 1] == "off"
    assert a[a.index("--cache-salt") + 1] == "s" and a[a.index("--cache-sample-idx") + 1] == "2"
    cmd = ex.build_command(ROOT, [("s", SAMPLE)], doctor="llm", patient="llm", judge="llm", persona="standard",
                           workers=1, label="x", out=tmp_path, meta={}, cache=ex.cache_args("replay"))
    assert "--llm-cache" in cmd and cmd.index("--llm-cache") < cmd.index("--cases")
    args = ex.build_parser().parse_args([])
    assert args.llm_cache == "auto" and not args.cache_doctor
    cfg = ex.load_config()
    assert all(c["cache_doctor"] is False for c in ex.resolve_profile(cfg, "smoke")["conditions"])
    data = {"llm_cache": {"roles": {"patient": {"hits": 3, "misses": 1, "saved_prompt_tokens": 300,
                                                "saved_completion_tokens": 60}}, "saved_krw_doctor": None}}
    assert ex.cache_line(data) == "patient 3/4 hit, saved 300 in / 60 out" and ex.cache_line({}) == "–"
