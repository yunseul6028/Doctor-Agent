"""Submission entry point (run by the competition server).

    python run.py --cases data/sample_cases                 # local case files (keyword simulator)
    python run.py --env official                            # official API adapter (src/doctor_agent/env/official.py)
    DOCTOR_ENV=official python run.py

Crash-proof by design:
  - submission mode (default here; --dev turns it off): run_case never raises, every case ends with a DIAGNOSE
  - each case also has an outer guard here: on an unexpected error the case still gets a fallback diagnosis
  - results are written incrementally: one JSON line per case to <out>.jsonl (flushed + fsynced) and the full
    {case_id: diagnosis} map rewritten atomically to <out> after every case, so a crash mid-run keeps prior results
  - --resume skips cases already present in the .jsonl file

TODO(agent-engineer): once the participant guide is published, implement env/official.py and adapt write_outputs() if
the official output format differs.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.agent.loop import FALLBACK_DIAGNOSIS, run_case  # noqa: E402
from doctor_agent.config import Config  # noqa: E402
from doctor_agent.env.factory import ENV_NAMES, case_source, default_env_name  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402
from doctor_agent.llm.client import DummyLLM, OpenAICompatClient  # noqa: E402

log = logging.getLogger("doctor_agent.run")


def make_llm(kind: str, cfg: Config):
    """New client per case (no state shared across cases)."""
    return DummyLLM() if kind == "dummy" else OpenAICompatClient(cfg.llm)


def load_done(jsonl: Path) -> dict[str, str | None]:
    done: dict[str, str | None] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[str(row["case_id"])] = row.get("diagnosis")
            except (ValueError, KeyError, TypeError):
                continue  # a half-written last line from a crash
    return done


def append_line(jsonl: Path, row: dict) -> None:
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_outputs(out: Path, preds: dict[str, str | None]) -> None:
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)


def run_one(case_id: str, env, llm_kind: str, cfg: Config) -> dict:
    t0 = time.monotonic()
    try:
        result = run_case(env, make_llm(llm_kind, cfg), cfg)
        return {"case_id": case_id, "diagnosis": result["diagnosis"], "n_turns": result["n_turns"],
                "llm_calls": result["llm_calls"], "forced": result["runtime"]["forced"],
                "sec": round(time.monotonic() - t0, 1), "error": None}
    except Exception as e:  # noqa: BLE001 — one case failing must not stop the rest
        if not cfg.agent.submission:
            raise
        log.exception("[%s] failed", case_id)
        try:  # still hand the environment an answer
            env.step(Action(ActionType.DIAGNOSE, FALLBACK_DIAGNOSIS))
        except Exception:  # noqa: BLE001
            pass
        return {"case_id": case_id, "diagnosis": FALLBACK_DIAGNOSIS, "n_turns": None, "llm_calls": None,
                "forced": "run_error", "sec": round(time.monotonic() - t0, 1), "error": f"{type(e).__name__}: {e}"[:500]}


def main(argv: list[str] | None = None) -> dict[str, str | None]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=ENV_NAMES, default=default_env_name(), help="case source (env DOCTOR_ENV)")
    ap.add_argument("--cases", help="local env: case directory or .json file")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--jsonl", help="incremental per-case log (default: <out without .json>.jsonl)")
    ap.add_argument("--resume", action="store_true", help="skip cases already in the .jsonl log")
    ap.add_argument("--llm", choices=["openai", "dummy"], default="openai", help="dummy = scripted smoke test")
    ap.add_argument("--dev", action="store_true", help="dev mode: exceptions propagate (not for submission)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    cfg.agent.submission = not args.dev
    out = Path(args.out)
    jsonl = Path(args.jsonl) if args.jsonl else out.with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(jsonl) if args.resume else {}
    if not args.resume and jsonl.exists():
        jsonl.unlink()
    preds: dict[str, str | None] = dict(done)
    if cfg.agent.use_subagents:
        try:  # process-level warm-up of the static KB indexes only (no case data), so the first consult is not slow
            from doctor_agent.knowledge import specialty

            specialty.warm()
        except Exception:  # noqa: BLE001
            log.warning("specialty warm-up failed", exc_info=True)

    try:
        cases = iter(case_source(args.env)(cases=args.cases))
    except Exception:  # noqa: BLE001
        if not cfg.agent.submission:
            raise
        log.exception("could not open the case source %r", args.env)
        cases = iter(())
    while True:
        try:
            case_id, env = next(cases)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 — the case source itself broke: keep what we have
            if not cfg.agent.submission:
                raise
            log.exception("case source failed; stopping")
            break
        if case_id in done:
            continue
        row = run_one(case_id, env, args.llm, cfg)
        append_line(jsonl, row)
        preds[case_id] = row["diagnosis"]
        write_outputs(out, preds)
        log.info("[%s] dx=%r turns=%s calls=%s forced=%s %.1fs", case_id, row["diagnosis"], row["n_turns"],
                 row["llm_calls"], row["forced"], row["sec"])
    write_outputs(out, preds)
    return preds


if __name__ == "__main__":
    main()
