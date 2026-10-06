"""Standard experiments in one command: profile -> cost estimate -> runs -> compare -> (log) -> viewer + share page.

python eval/experiment.py --profile smoke --doctor-endpoint dummy              # free wiring check (no LLM at all)
python eval/experiment.py --profile smoke --doctor-endpoint local              # 5 cases on gpt-oss-20b (Ollama)
python eval/experiment.py --profile dev --doctor-endpoint env --yes --log      # DOCTOR_LLM_* from .env
python eval/experiment.py --profile dev --conditions v6,v5-baseline --estimate-only
python eval/experiment.py --list-profiles
python eval/experiment.py --regen-case-lists                                   # rewrite eval/case_lists/*.txt

Profiles and conditions live in eval/experiment_profiles.json. Each condition runs eval/run_local.py in a subprocess
(env overrides per condition; conditions with a `commit` run inside a temporary git worktree at that commit with the
current eval/ copied in and are rescored with the current scorer). The first condition is the comparison baseline
(or --compare-with FILE). Cheapest default: keyword patient + no judge (only the doctor spends credits).
Keys from .env are passed to the subprocess environment only; they are never printed.

LLM record/replay cache (eval/replay.py): patient/judge default to `--llm-cache auto` (replay on hit, else call +
record), the doctor to off unless `--cache-doctor` (or `"cache_doctor": true` on a condition, e.g. a frozen baseline).
`--llm-cache replay` makes zero API calls and stops on the first miss. Hit/miss counts and tokens avoided are printed
per run and in the comparison.
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval import compare as cmp  # noqa: E402

PROFILES_FILE = ROOT / "eval/experiment_profiles.json"
RESULTS = ROOT / "eval/results"
EXPERIMENTS_MD = ROOT / "docs/experiments.md"
BILLING_ABORT_EXIT = 3  # same as eval/run_local.py
REPLAY_MISS_EXIT = 4  # same as eval/run_local.py
CACHE_DIR = ROOT / "eval/cache"

# Conservative fallbacks when no past result file tells us better (see estimate()).
DEFAULTS = {
    "doctor_calls_per_case": 12.0,      # observed 4.7-12.2 on Gemini/Gemma (2026-09-25..26); reviewer adds calls
    "prompt_tokens_per_call": 3000.0,   # prompt 1.2-2k chars at turn 1, capped at AGENT_MAX_VIEW_CHARS=12000 chars
    "completion_tokens_per_call": 1000.0,  # gpt-oss reasoning (low effort) + action JSON; max_tokens 2048
    "patient_calls_per_case": 10.0,     # one patient answer per non-diagnosis turn
}
# Measured gpt-oss prompt tokens (scripts/token_budget.py; see docs/experiments.md "Prompt token budget"): mean tokens
# of the step prompt at turn t, full harmony prompt (system + developer + user wrappers), tokenizer o200k_harmony.
# eval/results/token_budget.json (written by `scripts/token_budget.py --json-out eval/results/token_budget.json`)
# overrides this table when present. Used for the prompt side when no same-model usage has been recorded yet.
TOKEN_BUDGET_FILE = RESULTS / "token_budget.json"
MEASURED_STEP_TOKENS = {1: 987.5, 5: 1582.8, 10: 2019.7, 15: 2489.4, 20: 2901.0, 30: 3713.7, 40: 4283.1, 50: 4702.5,
                        59: 5082.4}  # 2026-09-28, prompt v6-kb-strict-review, 267 data/cases_aug cases, effort low
MEASURED_SOURCE = "measured gpt-oss tokenizer (scripts/token_budget.py 2026-09-28, 267 cases)"
# Output side is an assumption until a gpt-oss run records usage: action JSON ≈ 300 tokens (measured on scripted answers
# with 3-5 DDx entries) + reasoning at the given effort; capped by max_tokens 2048 (a length retry may add more).
ASSUMED_COMPLETION_TOKENS = {"low": 1000.0, "medium": 1800.0, "high": 2048.0}
DEFAULT_MARGIN = 1.3
DEFAULT_MAX_CALLS = 300  # doctor calls above which --yes is required
DEFAULT_MAX_COST = 5000.0  # KRW (10% of the ~50,000 KRW credits) above which --yes is required (when prices are known)


# ---------------------------------------------------------------- config / profiles

def load_config(path: Path = PROFILES_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_case_list(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def resolve_profile(cfg: dict, name: str, conditions: list[str] | None = None, root: Path = ROOT) -> dict:
    """-> {"name", "description", "cases": [(set, Path)], "conditions": [{"name", "env", "commit"}], "workers"}."""
    from eval.run_local import collect_cases

    profiles = cfg["profiles"]
    if name not in profiles:
        raise SystemExit(f"unknown profile {name!r}; choose from {', '.join(profiles)}")
    prof = profiles[name]
    if "cases_file" in prof:
        specs = [str(root / p) for p in read_case_list(root / prof["cases_file"])]
    else:
        specs = [p if Path(p).is_absolute() else str(root / p) for p in prof["cases"]]
    cases = collect_cases(specs)
    missing = [s for s in specs if not glob.has_magic(s) and not Path(s).exists()]
    if missing:
        raise SystemExit(f"profile {name!r}: {len(missing)} case files missing, e.g. {missing[0]} "
                         "(regenerate with --regen-case-lists)")
    names = conditions or prof["conditions"]
    unknown = [c for c in names if c not in cfg["conditions"]]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; choose from {', '.join(cfg['conditions'])}")
    conds = [{"name": c, "env": dict(cfg["conditions"][c].get("env") or {}), "commit": cfg["conditions"][c].get("commit"),
              "cache_doctor": bool(cfg["conditions"][c].get("cache_doctor", False))}
             for c in names]
    return {"name": name, "description": prof.get("description", ""), "cases": cases, "conditions": conds,
            "workers": int(prof.get("workers", 4))}


def stratified_sample(items: list[tuple[str, str]], n: int, seed: int) -> list[tuple[str, str]]:
    """[(set, path)] -> n items, proportional per set (largest remainder), at least one per set when n allows.
    Deterministic for a given seed and input; output sorted by (set, path)."""
    by_set: dict[str, list[str]] = {}
    for s, p in items:
        by_set.setdefault(s, []).append(p)
    total = sum(len(v) for v in by_set.values())
    if n >= total:
        return sorted(items)
    sets = sorted(by_set)
    quota = {s: n * len(by_set[s]) / total for s in sets}
    alloc = {s: min(len(by_set[s]), max(1 if n >= len(sets) else 0, math.floor(quota[s]))) for s in sets}
    while sum(alloc.values()) > n:  # min-1 rule overshot: take from the largest allocation
        s = max(sets, key=lambda k: (alloc[k], k))
        alloc[s] -= 1
    while sum(alloc.values()) < n:
        s = max((k for k in sets if alloc[k] < len(by_set[k])), key=lambda k: (quota[k] - alloc[k], k))
        alloc[s] += 1
    out = []
    for s in sets:
        pool = sorted(by_set[s])
        out += [(s, p) for p in random.Random(f"{seed}:{s}").sample(pool, alloc[s])]
    return sorted(out)


def regen_case_lists(cfg: dict, root: Path = ROOT) -> dict[str, list[str]]:
    """Rewrite the committed case lists: dev = stratified sample of the source; smoke = stratified subset of dev."""
    from eval.run_local import collect_cases

    seed = int(cfg.get("case_list_seed", 0))
    source = [(s, _rel_to(p, root)) for s, p in collect_cases([str(root / cfg["case_list_source"])])]
    written: dict[str, list[str]] = {}
    dev_prof, smoke_prof = cfg["profiles"]["dev"], cfg["profiles"]["smoke"]
    dev = stratified_sample(source, int(dev_prof["size"]), seed)
    smoke = stratified_sample(dev, int(smoke_prof["size"]), seed)
    for prof, picked in ((dev_prof, dev), (smoke_prof, smoke)):
        path = root / prof["cases_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = {}
        for s, _ in picked:
            counts[s] = counts.get(s, 0) + 1
        header = [f"# generated by `python eval/experiment.py --regen-case-lists` (seed {seed}, source "
                  f"{cfg['case_list_source']}, stratified by set); do not edit by hand",
                  "# " + ", ".join(f"{s}={c}" for s, c in sorted(counts.items())) + f" (n={len(picked)})"]
        path.write_text("\n".join(header + [p for _, p in picked]) + "\n", encoding="utf-8")
        written[str(path)] = [p for _, p in picked]
    return written


def _rel_to(p: Path, root: Path) -> str:
    for a, b in ((Path(p), Path(root)), (Path(p).resolve(), Path(root).resolve())):
        try:
            return str(a.relative_to(b))
        except ValueError:
            continue
    return str(p)


# ---------------------------------------------------------------- endpoints / env

def read_dotenv(path: Path) -> dict:
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


ENDPOINTS = ("env", "gemini", "local", "dummy")


def doctor_endpoint_env(preset: str, env: dict) -> dict:
    """DOCTOR_LLM_* overrides for a preset, from `env` (os.environ + .env). Raises SystemExit when keys are missing.

    gemini:      GEMINI_LLM_*, else the shared LLM_*
    local:       LOCAL_LLM_*, else Ollama http://localhost:11434/v1 + gpt-oss:20b
    env / dummy: no overrides (env = whatever .env says; dummy = scripted doctor, no LLM)
    """
    g = lambda *keys, default=None: next((env[k] for k in keys if env.get(k)), default)  # noqa: E731
    if preset in ("env", "dummy"):
        return {}
    if preset == "gemini":
        over = {"DOCTOR_LLM_BASE_URL": g("GEMINI_LLM_BASE_URL", "LLM_BASE_URL"),
                "DOCTOR_LLM_API_KEY": g("GEMINI_LLM_API_KEY", "LLM_API_KEY"),
                "DOCTOR_LLM_MODEL": g("GEMINI_LLM_MODEL", "LLM_MODEL")}
    elif preset == "local":
        over = {"DOCTOR_LLM_BASE_URL": g("LOCAL_LLM_BASE_URL", default="http://localhost:11434/v1"),
                "DOCTOR_LLM_API_KEY": g("LOCAL_LLM_API_KEY", default="EMPTY"),
                "DOCTOR_LLM_MODEL": g("LOCAL_LLM_MODEL", default="gpt-oss:20b")}
    else:
        raise SystemExit(f"unknown --doctor-endpoint {preset!r}")
    missing = [k for k, v in over.items() if not v]
    if missing:
        raise SystemExit(f"--doctor-endpoint {preset}: missing {', '.join(missing)} (set them in .env; "
                         "see .env.example)")
    return over


def describe_doctor(preset: str, env: dict) -> tuple[str, str]:
    """(model, host) for display. Never includes keys."""
    if preset == "dummy":
        return "dummy", "-"
    model = env.get("DOCTOR_LLM_MODEL") or env.get("LLM_MODEL") or "gpt-oss:20b"
    base = env.get("DOCTOR_LLM_BASE_URL") or env.get("LLM_BASE_URL") or "http://localhost:11434/v1"
    return model, urlparse(base).netloc or base


def parse_env_pairs(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


# ---------------------------------------------------------------- cost estimate

def load_measured(path: Path | None = None) -> dict:
    """{"curve": {turn: mean step-prompt tokens}, "source": str} from scripts/token_budget.py output, else the
    committed MEASURED_STEP_TOKENS table."""
    path = Path(path) if path else TOKEN_BUDGET_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        curve = {int(k): float(v) for k, v in (data.get("step_mean_by_turn") or {}).items()}
        if curve:
            m = data.get("meta") or {}
            return {"curve": curve, "source": f"measured {path.name} ({m.get('date', '?')}, {m.get('cases', '?')} cases, "
                                              f"{m.get('tokenizer', '?')})"}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return {"curve": dict(MEASURED_STEP_TOKENS), "source": MEASURED_SOURCE}


def measured_prompt_tokens_per_call(n_turns: float, curve: dict) -> float | None:
    """Mean step-prompt tokens per call for a case that lasts `n_turns` turns: mean of the per-turn curve over turns
    1..n (turns between measured points interpolated linearly; beyond the last point the last value)."""
    pts = sorted((int(k), float(v)) for k, v in (curve or {}).items())
    if not pts:
        return None
    n = max(1, int(round(n_turns)))

    def at(t: int) -> float:
        if t <= pts[0][0]:
            return pts[0][1]
        for (a, va), (b, vb) in zip(pts, pts[1:]):
            if a <= t <= b:
                return va + (vb - va) * (t - a) / (b - a)
        return pts[-1][1]

    return sum(at(t) for t in range(1, n + 1)) / n


def _is_gpt_oss(model: str | None) -> bool:
    return model is None or "gpt-oss" in model.lower()


def history_stats(dirs: list[Path], doctor_model: str | None = None, *, measured: dict | None = None,
                  effort: str | None = None) -> dict:
    """Per-case doctor calls / tokens and patient turns from past result files (dummy runs ignored).

    Calls and turns are model-independent enough to pool. Tokens per call, in order of preference: recorded usage of
    runs with the same doctor model; for a gpt-oss doctor (or unknown model) the measured gpt-oss prompt tokens
    (`measured`, default load_measured(), evaluated at the mean case length) and ASSUMED_COMPLETION_TOKENS[effort];
    usage of runs with any other model; DEFAULTS."""
    calls, turns, tok_same, tok_any, files = [], [], [], [], 0
    for d in dirs:
        for p in sorted(Path(d).glob("run_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("doctor_model") in (None, "dummy"):
                continue
            files += 1
            for r in data.get("cases") or []:
                if r.get("llm_calls"):
                    calls.append(r["llm_calls"])
                if r.get("n_turns") is not None:
                    turns.append(r["n_turns"])
                u = r.get("usage") or {}
                if u.get("calls") and u.get("prompt_tokens"):
                    item = (u["prompt_tokens"] / u["calls"], u["completion_tokens"] / u["calls"])
                    tok_any.append(item)
                    if doctor_model and data.get("doctor_model") == doctor_model:
                        tok_same.append(item)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    out = {
        "files": files,
        "n_cases": len(calls),
        "doctor_calls_per_case": mean(calls),
        "patient_calls_per_case": mean(turns),
    }
    if tok_same or not _is_gpt_oss(doctor_model):
        tok = tok_same or tok_any
        return {**out, "prompt_tokens_per_call": mean([t[0] for t in tok]),
                "completion_tokens_per_call": mean([t[1] for t in tok]),
                "token_source": ("same model" if tok_same else "other model") + f" ({len(tok)} cases)" if tok else "default"}
    # gpt-oss without recorded usage: measured prompt tokens + assumed output tokens
    measured = measured if measured is not None else load_measured()
    n_turns = out["patient_calls_per_case"] or DEFAULTS["patient_calls_per_case"]
    pin = measured_prompt_tokens_per_call(n_turns, measured.get("curve") or {})
    if pin is None:
        tok = tok_any
        return {**out, "prompt_tokens_per_call": mean([t[0] for t in tok]),
                "completion_tokens_per_call": mean([t[1] for t in tok]),
                "token_source": f"other model ({len(tok)} cases)" if tok else "default"}
    effort = (effort or os.getenv("DOCTOR_LLM_REASONING_EFFORT") or os.getenv("LLM_REASONING_EFFORT") or "low").lower()
    pout = ASSUMED_COMPLETION_TOKENS.get(effort, DEFAULTS["completion_tokens_per_call"])
    return {**out, "prompt_tokens_per_call": pin, "completion_tokens_per_call": pout,
            "token_source": f"{measured.get('source', 'measured')} at {n_turns:.0f} turns/case; output assumed "
                            f"(effort {effort})"}


def estimate(n_cases: list[int], stats: dict, *, doctor: str = "llm", patient: str = "keyword", judge: str = "none",
             margin: float = DEFAULT_MARGIN, price_in: float | None = None, price_out: float | None = None) -> dict:
    """Estimated doctor calls/tokens (the credit-consuming part) and patient/judge calls, per condition and total.
    Every per-case number gets `margin` on top (past means are from Gemini/Gemma; gpt-oss may take more turns)."""
    pick = lambda k: stats.get(k) if stats.get(k) else DEFAULTS[k]  # noqa: E731
    cpc = pick("doctor_calls_per_case") * margin
    pin, pout = pick("prompt_tokens_per_call") * margin, pick("completion_tokens_per_call") * margin
    ppc = pick("patient_calls_per_case") * margin
    per = []
    for n in n_cases:
        calls = 0 if doctor == "dummy" else round(n * cpc)
        row = {"cases": n, "doctor_calls": calls, "prompt_tokens": round(calls * pin),
               "completion_tokens": round(calls * pout),
               "patient_calls": round(n * ppc) if patient == "llm" else 0, "judge_calls": n if judge == "llm" else 0}
        per.append(row)
    total = {k: sum(r[k] for r in per) for k in ("cases", "doctor_calls", "prompt_tokens", "completion_tokens",
                                                  "patient_calls", "judge_calls")}
    cost = None
    if price_in is not None and price_out is not None:
        cost = round(total["prompt_tokens"] / 1e6 * price_in + total["completion_tokens"] / 1e6 * price_out, 1)
    return {"per_condition": per, "total": total, "cost": cost, "margin": margin,
            "basis": {"doctor_calls_per_case": round(cpc, 1), "prompt_tokens_per_call": round(pin),
                      "completion_tokens_per_call": round(pout), "patient_calls_per_case": round(ppc, 1),
                      "calls_source": f"{stats.get('n_cases', 0)} past cases" if stats.get("doctor_calls_per_case") else "default",
                      "token_source": stats.get("token_source", "default")}}


def needs_confirmation(est: dict, max_calls: int, max_cost: float | None) -> list[str]:
    reasons = []
    if est["total"]["doctor_calls"] > max_calls:
        reasons.append(f"{est['total']['doctor_calls']} doctor calls > --max-calls {max_calls}")
    if est["cost"] is not None and max_cost is not None and est["cost"] > max_cost:
        reasons.append(f"~{est['cost']:.0f} KRW > --max-cost {max_cost:.0f}")
    return reasons


def format_estimate(est: dict, names: list[str]) -> str:
    b = est["basis"]
    lines = [f"cost estimate (margin x{est['margin']}; {b['doctor_calls_per_case']} doctor calls/case from "
             f"{b['calls_source']}; {b['prompt_tokens_per_call']} in + {b['completion_tokens_per_call']} out tokens/call "
             f"from {b['token_source']})"]
    for name, r in zip(names, est["per_condition"]):
        lines.append(f"  {name:<14} cases={r['cases']:<4} doctor_calls~{r['doctor_calls']:<6} "
                     f"tokens~{r['prompt_tokens']:,} in / {r['completion_tokens']:,} out"
                     + (f"  patient_calls~{r['patient_calls']}" if r["patient_calls"] else "")
                     + (f"  judge_calls={r['judge_calls']}" if r["judge_calls"] else ""))
    t = est["total"]
    lines.append(f"  {'TOTAL':<14} cases={t['cases']:<4} doctor_calls~{t['doctor_calls']:<6} "
                 f"tokens~{t['prompt_tokens']:,} in / {t['completion_tokens']:,} out"
                 + (f"  ~{est['cost']:.0f} KRW" if est["cost"] is not None else "  (no prices given: --price-in/--price-out)"))
    return "\n".join(lines)


# ---------------------------------------------------------------- running

@contextmanager
def baseline_worktree(commit: str):
    """Temporary detached worktree at `commit` with the current eval/ copied in (old run_local lacked set/usage/etc)."""
    parent = Path(tempfile.mkdtemp(prefix="doctor-agent-baseline-"))
    wt = parent / "wt"
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    try:
        shutil.rmtree(wt / "eval", ignore_errors=True)
        shutil.copytree(ROOT / "eval", wt / "eval", ignore=shutil.ignore_patterns("results", "cache", "__pycache__"))
        yield wt
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT, capture_output=True, text=True)
        shutil.rmtree(parent, ignore_errors=True)


def cache_args(mode: str = "off", *, cache_doctor: bool = False, salt: str = "", sample_idx: int = 0,
               directory: Path = CACHE_DIR) -> list[str]:
    """run_local flags for the LLM cache (the cache dir is always the current checkout's, also for worktrees).
    `mode` is the patient/judge mode; with cache_doctor the doctor uses the same mode (auto when mode is off)."""
    if mode == "off" and not cache_doctor:
        return []
    out = ["--llm-cache", mode, "--cache-dir", str(Path(directory).resolve())]
    if cache_doctor:
        out += ["--doctor-cache", mode if mode != "off" else "auto"]
    if salt:
        out += ["--cache-salt", salt]
    if sample_idx:
        out += ["--cache-sample-idx", str(sample_idx)]
    return out


def build_command(root: Path, cases: list[tuple[str, Path]], *, doctor: str, patient: str, judge: str, persona: str,
                  workers: int, label: str, out: Path, meta: dict, cache: list[str] | None = None) -> list[str]:
    return [sys.executable, str(root / "eval/run_local.py"), "--doctor", doctor, "--patient", patient,
            "--judge", judge, "--persona", persona, "--workers", str(workers), "--label", label, "--out", str(out),
            "--no-view", "--meta", json.dumps(meta, ensure_ascii=False), *(cache or []),
            "--cases", *[str(Path(p).resolve()) for _, p in cases]]


def run_subprocess(cmd: list[str], cwd: Path, env: dict) -> tuple[int, Path | None]:
    """Streams the child's output; returns (exit code, saved result path)."""
    saved = None
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write("    " + line)
        sys.stdout.flush()
        if line.startswith("saved: "):
            saved = Path(line[len("saved: "):].strip())
    return proc.wait(), saved


def rescore(path: Path, case_root: Path = ROOT) -> dict:
    """Re-score a result file with the current scorer (used for old-commit baselines). Judge scores are kept."""
    from doctor_agent.config import AgentConfig
    from eval.scorer import missed_checks, score_case

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    max_turns = AgentConfig().max_turns
    for row in data.get("cases") or []:
        p = Path(row["path"])
        p = p if p.is_absolute() else case_root / p
        case = json.loads(p.read_text(encoding="utf-8"))
        judged = (row.get("judge") or {}).get("score")
        row["scores"] = score_case(case, row, max_turns, judged)
        row["missed_checks"] = missed_checks(case, row)
        row.pop("score_error", None)
        row["path"] = _rel(p)
    rows = data.get("cases") or []
    overall = cmp.summarize(rows)
    data["overall"] = overall
    data["avg"] = {k: overall[k] for k in ("accuracy", "efficiency", "safety", "case_checks")}
    data["avg_by_set"] = cmp.summarize_by_set(rows)
    data["rescored"] = {"by": "eval/scorer.py", "commit": _git_commit(), "at": time.strftime("%Y-%m-%d %H:%M")}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _git_commit() -> str | None:
    from eval.run_local import git_commit

    return git_commit()


# ---------------------------------------------------------------- reporting

def usage_line(data: dict) -> str:
    u = (data.get("usage") or {}).get("doctor")
    if not u:
        return "–"
    return f"{u['calls']} calls, {u['prompt_tokens']:,} in / {u['completion_tokens']:,} out"


def cache_line(data: dict) -> str:
    """Per-role cache hits/misses and tokens avoided from a result file ('–' when the cache was off)."""
    block = data.get("llm_cache") or {}
    roles = block.get("roles") or {}
    if not roles:
        return "–"
    parts = [f"{r} {s['hits']}/{s['hits'] + s['misses']} hit, saved {s['saved_prompt_tokens']:,} in / "
             f"{s['saved_completion_tokens']:,} out" for r, s in roles.items()]
    if block.get("saved_krw_doctor") is not None:
        parts.append(f"~{block['saved_krw_doctor']:.0f} KRW doctor credits saved")
    return "; ".join(parts)


def markdown_log(res: dict, runs: list[dict], ctx: dict) -> tuple[list[str], str]:
    """(table rows for the main experiment table, detailed section) for docs/experiments.md."""
    date = time.strftime("%Y-%m-%d")
    rows = []
    for info, data, ov in zip(res["runs"], runs, res["overall"]):
        cond = (data.get("experiment") or {}).get("condition") or info.get("label") or info["name"]
        f = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
        notes = [f"n/a rate {f(ov.get('not_provided'))}", f"doctor usage: {usage_line(data)}", f"`{info['name']}`"]
        if data.get("llm_cache"):
            notes.insert(2, f"cache: {cache_line(data)}")
        if ctx.get("note"):
            notes.insert(0, ctx["note"])
        rows.append(f"| {date} | experiment `{ctx['profile']}` / {cond} | {info.get('prompt_version') or '–'} | "
                    f"{info.get('doctor_model') or '–'} | {ctx['profile']} {ov.get('n')} | {ctx['patient_type']} | "
                    f"{f(ov.get('accuracy'))} | {f(ov.get('efficiency'))} | {f(ov.get('safety'))} | "
                    f"{f(ov.get('turns'), 1)} | {'; '.join(notes)} |")
    section = "\n".join([
        f"### {date} · profile `{ctx['profile']}` · doctor {ctx['doctor_model']} · patient {ctx['patient_type']} · "
        f"judge {ctx['judge']}",
        "",
        f"Conditions: {', '.join(ctx['conditions'])}. Commit {ctx['commit']}."
        + (f" Note: {ctx['note']}" if ctx.get("note") else ""),
        "",
        cmp.to_markdown(res),
        "",
    ])
    return rows, section


AUTO_HEADING = "## Auto-logged experiment runs"


def append_log(md_path: Path, table_rows: list[str], section: str) -> None:
    """Insert rows after the last row of the first experiment table; append the section under AUTO_HEADING."""
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("| Date |")), None)
    if start is not None:
        end = start
        while end + 1 < len(lines) and lines[end + 1].startswith("|"):
            end += 1
        lines[end + 1:end + 1] = table_rows
    else:
        lines += [""] + table_rows
    text = "\n".join(lines).rstrip("\n") + "\n"
    if AUTO_HEADING not in text:
        text += f"\n{AUTO_HEADING}\n\nWritten by `python eval/experiment.py --log` (newest last).\n"
    text += "\n" + section.rstrip("\n") + "\n"
    Path(md_path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", help="smoke | dev | full (see eval/experiment_profiles.json)")
    ap.add_argument("--conditions", help="comma-separated override, e.g. v6,v6-no-kb,v5-baseline (first = baseline)")
    ap.add_argument("--doctor-endpoint", choices=ENDPOINTS, default="env",
                    help="DOCTOR_LLM_* preset: gemini | local | dummy | env (= .env as is)")
    ap.add_argument("--patient", choices=["keyword", "llm"], default="keyword", help="llm = virtual patient LLM (PATIENT_LLM_*)")
    ap.add_argument("--judge", choices=["none", "llm"], default="none", help="llm = LLM judge (JUDGE_LLM_*)")
    ap.add_argument("--llm-cache", choices=["off", "record", "replay", "auto"], default="auto",
                    help="record/replay cache for patient/judge (eval/replay.py); replay = no API calls, stop on a miss")
    ap.add_argument("--cache-doctor", action="store_true",
                    help="cache the doctor too (default off; replays identical requests = deterministic doctor)")
    ap.add_argument("--cache-salt", default="", help="part of every cache key; a new salt forces fresh answers")
    ap.add_argument("--cache-sample-idx", type=int, default=0, help="k-th stored sample per request (variance runs)")
    ap.add_argument("--persona", default="standard")
    ap.add_argument("--workers", type=int, help="override the profile's parallel cases")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra env for every condition, e.g. AGENT_CASE_TIME_BUDGET_S=240 (repeatable)")
    ap.add_argument("--baseline-commit", help="override the commit of conditions that run in a worktree (v5-baseline)")
    ap.add_argument("--compare-with", metavar="FILE", help="an older result JSON to use as the comparison baseline")
    ap.add_argument("--note", default="", help="free text for the log")
    ap.add_argument("--log", action="store_true", help="append the summary to docs/experiments.md")
    ap.add_argument("--share", metavar="PATH", help="share page path (default eval/results/share_<time>.html)")
    ap.add_argument("--no-view", action="store_true", help="do not open the viewer in the browser")
    ap.add_argument("--yes", action="store_true", help="proceed even when the estimate is above the thresholds")
    ap.add_argument("--estimate-only", action="store_true", help="print the plan and cost estimate, run nothing")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="doctor calls above which --yes is needed")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST, help="KRW above which --yes is needed")
    ap.add_argument("--price-in", type=float, default=_env_float("EXPERIMENT_PRICE_IN_PER_M"),
                    help="KRW per 1M prompt tokens (or EXPERIMENT_PRICE_IN_PER_M)")
    ap.add_argument("--price-out", type=float, default=_env_float("EXPERIMENT_PRICE_OUT_PER_M"),
                    help="KRW per 1M completion tokens (or EXPERIMENT_PRICE_OUT_PER_M)")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="safety factor on per-case estimates")
    ap.add_argument("--history", action="append", default=[], metavar="DIR",
                    help="extra result dirs for the estimate (default: --out and the main checkout's eval/results)")
    ap.add_argument("--out", default=str(RESULTS), help="result directory")
    ap.add_argument("--profiles-file", default=str(PROFILES_FILE))
    ap.add_argument("--experiments-md", default=str(EXPERIMENTS_MD), help=argparse.SUPPRESS)
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--regen-case-lists", action="store_true", help="rewrite the committed case lists and exit")
    return ap


def _env_float(name: str) -> float | None:
    v = os.getenv(name)
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _history_dirs(args) -> list[Path]:
    dirs = [Path(args.out)] + [Path(d) for d in args.history]
    main_results = _main_checkout() / "eval/results" if _main_checkout() else None
    if main_results and main_results.exists():
        dirs.append(main_results)
    seen, out = set(), []
    for d in dirs:
        if d.exists() and d.resolve() not in seen:
            seen.add(d.resolve())
            out.append(d)
    return out


def _main_checkout() -> Path | None:
    """The primary checkout when running from a linked worktree (its eval/results holds the past runs)."""
    try:
        r = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        common = Path(r.stdout.strip())
        common = common if common.is_absolute() else ROOT / common
        return common.resolve().parent if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(Path(args.profiles_file))
    if args.list_profiles:
        for name, p in cfg["profiles"].items():
            print(f"{name:<6} conditions={','.join(p['conditions'])}  {p.get('description', '')}")
        for name, c in cfg["conditions"].items():
            print(f"  condition {name:<12} {c.get('description', '')}")
        return 0
    if args.regen_case_lists:
        for path, items in regen_case_lists(cfg).items():
            print(f"wrote {path} ({len(items)} cases)")
        return 0

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.conditions else None
    prof = resolve_profile(cfg, args.profile, conds)
    if args.baseline_commit:
        for c in prof["conditions"]:
            if c["commit"]:
                c["commit"] = args.baseline_commit
    dotenv = read_dotenv(ROOT / ".env")
    merged = {**dotenv, **os.environ}
    doctor_over = doctor_endpoint_env(args.doctor_endpoint, merged)
    extra = parse_env_pairs(args.env)
    view = {**merged, **doctor_over}
    doctor_model, host = describe_doctor(args.doctor_endpoint, view)
    doctor = "dummy" if args.doctor_endpoint == "dummy" else "llm"
    workers = args.workers or prof["workers"]
    names = [c["name"] for c in prof["conditions"]]

    print(f"profile={prof['name']} cases={len(prof['cases'])} conditions={','.join(names)} workers={workers}")
    print(f"doctor={doctor_model} @ {host}  patient={args.patient}  judge={args.judge}  persona={args.persona}"
          + (f"  env={sorted(extra)}" if extra else ""))
    print(f"llm cache: patient/judge={args.llm_cache}  doctor="
          + ((args.llm_cache if args.llm_cache != "off" else "auto") if args.cache_doctor else "off"
             + (" (on for " + ",".join(c["name"] for c in prof["conditions"] if c["cache_doctor"]) + ")"
                if any(c["cache_doctor"] for c in prof["conditions"]) else ""))
          + (f"  salt={args.cache_salt!r}" if args.cache_salt else "")
          + ("  (estimate below ignores cache hits)" if args.llm_cache != "off" else ""))
    stats = history_stats(_history_dirs(args), doctor_model,
                          effort=view.get("DOCTOR_LLM_REASONING_EFFORT") or view.get("LLM_REASONING_EFFORT"))
    est = estimate([len(prof["cases"])] * len(names), stats, doctor=doctor, patient=args.patient, judge=args.judge,
                   margin=args.margin, price_in=args.price_in, price_out=args.price_out)
    print(format_estimate(est, names))
    if args.estimate_only:
        return 0
    reasons = needs_confirmation(est, args.max_calls, args.max_cost) if doctor == "llm" else []
    if reasons and not args.yes:
        print("\nNOT RUNNING: estimate above threshold (" + "; ".join(reasons) + "). Re-run with --yes to proceed.")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_env = {**merged, **doctor_over, **extra}
    base_env["PYTHONIOENCODING"] = "utf-8"
    paths: list[Path] = []
    aborted = False
    for c in prof["conditions"]:
        env = {**base_env, **c["env"]}
        meta = {"profile": prof["name"], "condition": c["name"], "batch": stamp, "doctor_endpoint": args.doctor_endpoint,
                "env": {**c["env"], **extra}, "commit_override": c["commit"]}
        cache_doctor = args.cache_doctor or c["cache_doctor"]
        cache = cache_args(args.llm_cache, cache_doctor=cache_doctor, salt=args.cache_salt, sample_idx=args.cache_sample_idx)
        label = f"{prof['name']}:{c['name']}"
        print(f"\n=== {label} ===" + (f" (worktree at {c['commit']})" if c["commit"] else ""), flush=True)
        kw = dict(doctor=doctor, patient=args.patient, judge=args.judge, persona=args.persona, workers=workers,
                  label=label, out=out.resolve(), meta=meta, cache=cache)
        if c["commit"]:
            with baseline_worktree(c["commit"]) as wt:
                code, saved = run_subprocess(build_command(wt, prof["cases"], **kw), wt, env)
            if saved:
                rescore(saved)
                print(f"    rescored with the current scorer: {saved}")
        else:
            code, saved = run_subprocess(build_command(ROOT, prof["cases"], **kw), ROOT, env)
        if code == BILLING_ABORT_EXIT:
            print("\nABORTED: billing error (credits depleted?). Remaining conditions skipped.")
            aborted = True
            break
        if code == REPLAY_MISS_EXIT:
            print("\nABORTED: llm cache miss in replay mode (no API calls made). Remaining conditions skipped; "
                  "run once with --llm-cache auto to fill the cache.")
            aborted = True
            break
        if code != 0 or not saved:
            print(f"condition {c['name']} failed (exit {code}); continuing")
            continue
        paths.append(saved)

    if not paths:
        print("no results")
        return BILLING_ABORT_EXIT if aborted else 1
    files = ([Path(args.compare_with)] if args.compare_with else []) + paths
    runs = [cmp.load(p) for p in files]
    res = cmp.compare(runs)
    print("\n" + cmp.to_text(res))
    for p, d in zip(files, runs):
        print(f"  usage {p.name}: doctor {usage_line(d)}")
        if d.get("llm_cache"):
            print(f"  cache {p.name}: {cache_line(d)}")

    if args.log:
        ctx = {"profile": prof["name"], "patient_type": "keyword" if args.patient == "keyword" else args.persona,
               "judge": args.judge, "doctor_model": doctor_model, "conditions": names, "commit": _git_commit(),
               "note": args.note}
        table_rows, section = markdown_log(res, runs, ctx)
        append_log(Path(args.experiments_md), table_rows, section)
        print(f"logged to {args.experiments_md}")

    from eval import viewer

    viewer.RESULTS = out  # the viewer lists every run_*.json in the result directory
    page = viewer.build()
    share = viewer.build_share(Path(args.share) if args.share else out / f"share_{stamp}.html")
    print(f"viewer: {page}\nshare page: {share}")
    if not args.no_view:
        viewer.webbrowser.open(page.as_uri())
    return BILLING_ABORT_EXIT if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
