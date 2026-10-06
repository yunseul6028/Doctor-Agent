"""Local batch evaluation.

python eval/run_local.py                                  # all roles on LLM (reads .env), data/sample_cases
python eval/run_local.py --doctor dummy --patient keyword --judge none   # smoke test without an LLM
python eval/run_local.py --cases data/cases_aug/clinicalqa data/cases_aug/sample \
    --workers 8 --label "v5 baseline" --no-view         # multi-set run, 8 cases in parallel
python eval/run_local.py --cases 'data/cases_*' --sample 30 --seed 1   # quick random subset (globs allowed)
python eval/compare.py --latest 2                         # compare the two newest runs
python eval/viewer.py                                    # view the results viewer only
python eval/experiment.py --profile dev --doctor-endpoint local   # standard profiles + cost guard (wraps this)

Real LLM clients are metered (eval/usage.py): doctor token usage per case (`usage`) and per role for the run.
Exit code 3 when the batch stopped on a billing error.

LLM record/replay cache (eval/replay.py, files under eval/cache/):
python eval/run_local.py --llm-cache auto                 # patient/judge: replay on hit, else call + record
python eval/run_local.py --llm-cache replay --cache-doctor   # zero API calls; a miss aborts the batch (exit 4)
    --cache-salt S (new salt = fresh samples), --cache-sample-idx K (K-th stored sample per request),
    --doctor-cache MODE (doctor mode independent of --llm-cache), --cache-dir DIR
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.loop import run_case  # noqa: E402
from doctor_agent.config import Config, LLMConfig  # noqa: E402
from doctor_agent.llm.client import DummyLLM, OpenAICompatClient  # noqa: E402
from eval.compare import summarize, summarize_by_set  # noqa: E402
from eval.judge import judge_diagnosis  # noqa: E402
from eval.llm_patient import PERSONA_CHOICES, PERSONAS, LLMPatientEnvironment  # noqa: E402
from eval.scorer import missed_checks, score_case  # noqa: E402
from eval.simulator import CaseFileEnvironment  # noqa: E402
from eval.usage import UsageMeter, attach  # noqa: E402
from eval import replay  # noqa: E402

try:
    from doctor_agent.llm.client import BillingError  # noqa: E402
except ImportError:  # older commits (v5 baseline worktrees) have no BillingError
    class BillingError(RuntimeError):  # type: ignore[no-redef]
        pass

BILLING_ABORT_EXIT = 3  # process exit code when the batch stopped on a billing error (see eval/experiment.py)
REPLAY_MISS_EXIT = 4  # --llm-cache replay found no stored answer: the batch stopped without calling the API
_last_aborted = False
_last_exit = 1


def _env_float(name: str) -> float | None:
    try:
        return float(os.environ[name]) if os.environ.get(name) else None
    except ValueError:
        return None


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def set_name(directory: Path) -> str:
    """data/cases_clinicalqa -> clinicalqa, data/sample_cases -> sample."""
    name = directory.name
    name = re.sub(r"^cases_", "", name)
    name = re.sub(r"_cases$", "", name)
    return name or directory.name


def collect_cases(specs: list[str]) -> list[tuple[str, Path]]:
    """Directories, globs, or single .json files -> [(set, path)], deduplicated, in argument order (sorted within a dir)."""
    out, seen = [], set()
    for spec in specs:
        matches = sorted(glob.glob(spec)) if glob.has_magic(spec) else [spec]
        for m in matches:
            p = Path(m)
            if p.is_dir():
                items = [(set_name(p), f) for f in sorted(p.glob("*.json"))]
            elif p.suffix == ".json" and p.exists():
                items = [(set_name(p.parent), p)]
            else:
                print(f"warning: no cases at {m}", file=sys.stderr)
                items = []
            for s, f in items:
                if f.resolve() not in seen:
                    seen.add(f.resolve())
                    out.append((s, f))
    return out


def select_cases(cases: list, sample: int | None = None, seed: int = 0, limit: int | None = None) -> list:
    if sample is not None and sample < len(cases):
        picked = set(random.Random(seed).sample(range(len(cases)), sample))
        cases = [c for i, c in enumerate(cases) if i in picked]  # keep original order
    if limit is not None:
        cases = cases[:limit]
    return cases


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def prompt_version() -> str | None:
    try:
        from doctor_agent.agent.prompts import PROMPT_VERSION

        return PROMPT_VERSION
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> Path | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor", choices=["dummy", "llm"], default="llm")
    ap.add_argument("--patient", choices=["keyword", "llm"], default="llm")
    ap.add_argument("--judge", choices=["none", "llm"], default="llm")
    ap.add_argument("--persona", choices=PERSONA_CHOICES, default="standard",
                    help="virtual patient type (mixed = one hard type per case)")
    ap.add_argument("--cases", nargs="+", default=[str(ROOT / "data/sample_cases")],
                    help="case directories, globs (quote them), or .json files; set name = directory name")
    ap.add_argument("--workers", type=int, default=1, help="cases run concurrently (threads)")
    ap.add_argument("--label", default="", help='free-text run label, e.g. "v6 kb+review"')
    ap.add_argument("--sample", type=int, help="random subset of N cases (after collecting all sets)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample")
    ap.add_argument("--limit", type=int, help="first N cases (after --sample)")
    ap.add_argument("--out", default=str(ROOT / "eval/results"))
    ap.add_argument("--no-view", action="store_true", help="do not open the results viewer when done")
    ap.add_argument("--meta", default=None, help="JSON object stored as `experiment` in the result file")
    ap.add_argument("--llm-cache", choices=replay.MODES, default="off",
                    help="record/replay cache for patient+judge (and the doctor with --cache-doctor); see eval/replay.py")
    ap.add_argument("--cache-doctor", action="store_true", help="apply --llm-cache to the doctor too")
    ap.add_argument("--doctor-cache", choices=replay.MODES, default=None, help="doctor cache mode (overrides the above)")
    ap.add_argument("--cache-salt", default="", help="part of every cache key; change it to force fresh answers")
    ap.add_argument("--cache-sample-idx", type=int, default=0, help="k-th stored sample per request (variance runs)")
    ap.add_argument("--cache-dir", default=str(replay.DEFAULT_DIR))
    args = ap.parse_args(argv)
    global _last_aborted, _last_exit
    _last_aborted = False
    _last_exit = 1

    load_dotenv(ROOT / ".env")
    cfg = Config()
    cases = select_cases(collect_cases(args.cases), args.sample, args.seed, args.limit)
    pv, commit = prompt_version(), git_commit()
    print(f"doctor={cfg.llm.model if args.doctor == 'llm' else 'dummy'} "
          f"patient={LLMConfig.from_env('PATIENT_LLM').model if args.patient == 'llm' else 'keyword'} "
          f"judge={LLMConfig.from_env('JUDGE_LLM').model if args.judge == 'llm' else 'none'} persona={args.persona} "
          f"prompt={pv} commit={commit} cases={len(cases)} workers={args.workers}" + (f" label={args.label!r}" if args.label else ""))
    patient_llm = OpenAICompatClient(LLMConfig.from_env("PATIENT_LLM")) if args.patient == "llm" else None
    judge_llm = OpenAICompatClient(LLMConfig.from_env("JUDGE_LLM")) if args.judge == "llm" else None
    # token usage per role (doctor also per case); only for real LLM clients
    meters = {"doctor": UsageMeter(), "patient": UsageMeter(), "judge": UsageMeter()}
    metered = {"doctor": args.doctor == "llm",
               "patient": bool(patient_llm) and attach(patient_llm, meters["patient"]),
               "judge": bool(judge_llm) and attach(judge_llm, meters["judge"])}
    # the cache wraps the metered SDK: hits never reach the meter, so `usage` stays "tokens actually billed"
    caches = replay.make_caches(args.llm_cache, cache_doctor=args.cache_doctor, doctor_mode=args.doctor_cache,
                                directory=args.cache_dir, salt=args.cache_salt, sample_idx=args.cache_sample_idx)
    in_use = {"doctor": args.doctor == "llm", "patient": patient_llm is not None, "judge": judge_llm is not None}
    caches = {r: (c if in_use[r] else None) for r, c in caches.items()}  # no stats for roles without an LLM
    for role, llm in (("patient", patient_llm), ("judge", judge_llm)):
        if llm is not None:
            replay.attach(llm, caches[role])
    if any(caches.values()):
        print("llm cache: " + "  ".join(f"{r}={c.mode}({len(c)} entries)" for r, c in caches.items() if c)
              + (f" salt={args.cache_salt!r}" if args.cache_salt else "")
              + (f" sample_idx={args.cache_sample_idx}" if args.cache_sample_idx else ""))

    def run_one(set_: str, path: Path) -> dict:
        case = json.loads(path.read_text(encoding="utf-8"))
        doctor = DummyLLM() if args.doctor == "dummy" else OpenAICompatClient(cfg.llm)  # new instance per case
        case_meter = UsageMeter()
        has_usage = attach(doctor, case_meter)
        replay.attach(doctor, caches["doctor"])
        env = LLMPatientEnvironment(case, patient_llm, args.persona) if patient_llm else CaseFileEnvironment(case)
        t0 = time.time()
        try:
            result = run_case(env, doctor, cfg)
        finally:
            meters["doctor"].merge(case_meter)  # failed cases still spent tokens
        judged = judge_diagnosis(judge_llm, case, result["diagnosis"]) if judge_llm else None
        try:
            scores = score_case(case, result, cfg.agent.max_turns, judged["score"] if judged else None)
            missed, score_error = missed_checks(case, result), None
        except Exception as e:  # noqa: BLE001 — e.g. an old-commit worktree whose safety module differs; rescored later
            scores = {"accuracy": judged["score"] if judged else None, "efficiency": None, "safety": None,
                      "case_checks": None}
            missed, score_error = [], f"{type(e).__name__}: {e}"[:300]
        persona = PERSONAS[env.persona]["label"] if patient_llm else "keyword"
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(path)
        return {"case": path.stem, "set": set_, "path": rel, "persona": persona, "initial": case["initial"],
                "answer": case["diagnosis"], **result, "judge": judged, "scores": scores,
                "missed_checks": missed, "sec": round(time.time() - t0, 1),
                **({"usage": case_meter.totals()} if has_usage else {}),
                **({"score_error": score_error} if score_error else {})}

    done: dict[int, dict] = {}
    failed: list[dict] = []
    lock = threading.Lock()
    abort = threading.Event()  # set on billing errors / replay misses: stop the whole batch
    miss: list[str] = []
    t_start = time.time()

    def report(i: int, set_: str, path: Path, row: dict | None, err: Exception | None) -> None:
        with lock:
            n = len(done) + len(failed)
            tag = f"[{n}/{len(cases)}] {set_}/{path.stem}"
            if err is not None:
                print(f"{tag}: FAILED {err}", flush=True)
            else:
                s = row["scores"]
                mark = "O" if (s["accuracy"] or 0) >= 1 else ("~" if s["accuracy"] else "X")
                print(f"{tag} [{row['persona']}] {mark} dx={row['diagnosis']!r} turns={row['n_turns']} "
                      f"acc={s['accuracy']} eff={s['efficiency']} safety={s['safety']} {row['sec']}s", flush=True)

    def finish(i: int, set_: str, path: Path, fut_result) -> None:
        try:
            row = fut_result()
        except replay.CacheMiss as e:
            # replay mode must never fall back to the API: stop the batch
            abort.set()
            with lock:
                miss.append(str(e))
                failed.append({"case": path.stem, "set": set_, "error": str(e)})
            report(i, set_, path, None, e)
            return
        except BillingError as e:
            # out of credits: every remaining case would fail too
            abort.set()
            with lock:
                failed.append({"case": path.stem, "set": set_, "error": str(e)})
            report(i, set_, path, None, e)
            return
        except Exception as e:  # noqa: BLE001 — one case failing must not stop the batch
            with lock:
                failed.append({"case": path.stem, "set": set_, "error": str(e)})
            report(i, set_, path, None, e)
            return
        with lock:
            done[i] = row
        report(i, set_, path, row, None)

    if args.workers <= 1:
        for i, (s, p) in enumerate(cases):
            if abort.is_set():
                break
            finish(i, s, p, lambda s=s, p=p: run_one(s, p))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, s, p): (i, s, p) for i, (s, p) in enumerate(cases)}
            for fut in as_completed(futs):
                finish(*futs[fut], fut.result)
                if abort.is_set():
                    for f in futs:
                        f.cancel()
                    break
    cache_block = replay.summary(caches, salt=args.cache_salt, sample_idx=args.cache_sample_idx,
                                 price_in=_env_float("EXPERIMENT_PRICE_IN_PER_M"),
                                 price_out=_env_float("EXPERIMENT_PRICE_OUT_PER_M"))
    if abort.is_set():
        if miss:
            print(f"\n{replay.format_summary(cache_block)}")
            print(f"\nABORTED: llm cache miss in replay mode ({miss[0]}). No result file written.", flush=True)
            _last_exit = REPLAY_MISS_EXIT
            return None
        print("\nABORTED: LLM billing error (credits depleted?). No result file written.", flush=True)
        _last_aborted = True
        _last_exit = BILLING_ABORT_EXIT
        return None

    rows = [done[i] for i in sorted(done)]
    if not rows:
        return None
    overall = summarize(rows)
    avg = {k: overall[k] for k in ("accuracy", "efficiency", "safety", "case_checks")}
    by_set = summarize_by_set(rows)
    print(f"\nAVG {avg} turns={overall['turns']} not_provided={overall['not_provided']} n={len(rows)}"
          + (f" failed={len(failed)}" if failed else "") + f" ({time.time() - t_start:.0f}s)")
    if len(by_set) > 1:
        for s, m in by_set.items():
            print(f"  {s:<16} n={m['n']:<4} acc={m['accuracy']} eff={m['efficiency']} safety={m['safety']} "
                  f"turns={m['turns']} not_provided={m['not_provided']}")
    usage = {k: meters[k].totals() for k in meters if metered[k]}
    if usage:
        print("usage: " + "  ".join(f"{k} calls={u['calls']} in={u['prompt_tokens']} out={u['completion_tokens']}"
                                    for k, u in usage.items()))
    if cache_block:
        print(replay.format_summary(cache_block))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.label).strip("-")[:40]
    stem = f"run_{time.strftime('%Y%m%d_%H%M%S')}{'_' + slug if slug else ''}"
    path, k = out / f"{stem}.json", 1
    while path.exists():  # two runs finishing in the same second
        k += 1
        path = out / f"{stem}_{k}.json"
    path.write_text(
        json.dumps({"doctor_model": cfg.llm.model if args.doctor == "llm" else "dummy", "persona": args.persona,
                    "label": args.label or None, "prompt_version": pv, "commit": commit,
                    "case_specs": args.cases, "sample": args.sample, "seed": args.seed if args.sample else None,
                    "limit": args.limit, "workers": args.workers,
                    "experiment": json.loads(args.meta) if args.meta else None, "usage": usage, "llm_cache": cache_block,
                    "avg": avg, "avg_by_set": by_set, "overall": overall, "failed": failed, "cases": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {path}")
    if not args.no_view and args.doctor != "dummy":  # dummy runs are hidden from the viewer
        from eval import viewer

        viewer.webbrowser.open(viewer.build().as_uri())
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else _last_exit)
