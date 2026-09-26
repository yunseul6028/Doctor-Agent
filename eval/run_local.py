"""Local batch evaluation.

python eval/run_local.py                                  # all roles on LLM (reads .env), data/sample_cases
python eval/run_local.py --doctor dummy --patient keyword --judge none   # smoke test without an LLM
python eval/run_local.py --cases data/cases_clinicalqa data/cases_agentclinic data/cases_diagnosisarena \
    --workers 8 --label "v5 baseline" --no-view         # multi-set run, 8 cases in parallel
python eval/run_local.py --cases 'data/cases_*' --sample 30 --seed 1   # quick random subset (globs allowed)
python eval/compare.py --latest 2                         # compare the two newest runs
python eval/viewer.py                                    # view the results viewer only
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
    args = ap.parse_args(argv)

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

    def run_one(set_: str, path: Path) -> dict:
        case = json.loads(path.read_text(encoding="utf-8"))
        doctor = DummyLLM() if args.doctor == "dummy" else OpenAICompatClient(cfg.llm)  # new instance per case
        env = LLMPatientEnvironment(case, patient_llm, args.persona) if patient_llm else CaseFileEnvironment(case)
        t0 = time.time()
        result = run_case(env, doctor, cfg)
        judged = judge_diagnosis(judge_llm, case, result["diagnosis"]) if judge_llm else None
        scores = score_case(case, result, cfg.agent.max_turns, judged["score"] if judged else None)
        persona = PERSONAS[env.persona]["label"] if patient_llm else "keyword"
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(path)
        return {"case": path.stem, "set": set_, "path": rel, "persona": persona, "initial": case["initial"],
                "answer": case["diagnosis"], **result, "judge": judged, "scores": scores,
                "missed_checks": missed_checks(case, result), "sec": round(time.time() - t0, 1)}

    done: dict[int, dict] = {}
    failed: list[dict] = []
    lock = threading.Lock()
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
            finish(i, s, p, lambda s=s, p=p: run_one(s, p))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, s, p): (i, s, p) for i, (s, p) in enumerate(cases)}
            for fut in as_completed(futs):
                finish(*futs[fut], fut.result)

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
                    "avg": avg, "avg_by_set": by_set, "overall": overall, "failed": failed, "cases": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {path}")
    if not args.no_view:
        from eval import viewer

        viewer.webbrowser.open(viewer.build().as_uri())
    return path


if __name__ == "__main__":
    main()
