"""Compare local evaluation runs (eval/results/run_*.json) on the cases they share.

python eval/compare.py A.json B.json [C.json ...]   # first file is the baseline
python eval/compare.py --latest 2                    # the two newest runs in eval/results
python eval/compare.py A.json B.json --md            # Markdown for docs/experiments.md

Reports, on the intersection of cases: overall and per-set accuracy / efficiency / safety / avg turns,
the share of EXAM/TEST turns answered "제공되지 않습니다", and cases that flipped vs the baseline.
A case counts as "correct" when accuracy >= CORRECT (judge partial credit 0.5 counts as wrong).
Old result files without a `set` field are matched by case name only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval/results"
CORRECT = 1.0
NOT_PROVIDED = "제공되지 않습니다"
METRICS = ["accuracy", "efficiency", "safety", "case_checks"]


def is_correct(row: dict, threshold: float = CORRECT) -> bool:
    acc = (row.get("scores") or {}).get("accuracy")
    return acc is not None and acc >= threshold


def not_provided_counts(row: dict) -> tuple[int, int]:
    """(# EXAM/TEST turns answered 'not provided', # EXAM/TEST turns)."""
    et = [t for t in row.get("turns") or [] if t.get("type") in ("EXAM", "TEST")]
    return sum(NOT_PROVIDED in (t.get("response") or "") for t in et), len(et)


def summarize(rows: list[dict]) -> dict:
    """Averages over rows: score metrics (None values skipped), avg turns, not-provided rate, n."""
    out: dict = {}
    for k in METRICS:
        vals = [r["scores"][k] for r in rows if (r.get("scores") or {}).get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    out["turns"] = round(sum(r.get("n_turns", 0) for r in rows) / len(rows), 1) if rows else None
    np_hit, np_all = map(sum, zip(*(not_provided_counts(r) for r in rows))) if rows else (0, 0)
    out["not_provided"] = round(np_hit / np_all, 3) if np_all else None
    out["n"] = len(rows)
    return out


def summarize_by_set(rows: list[dict]) -> dict[str, dict]:
    sets: dict[str, list] = {}
    for r in rows:
        sets.setdefault(r.get("set") or "-", []).append(r)
    return {s: summarize(rs) for s, rs in sorted(sets.items())}


def load(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["_name"] = Path(path).stem
    return d


def case_key(row: dict, with_set: bool) -> str:
    return f"{row.get('set')}/{row['case']}" if with_set else row["case"]


def compare(runs: list[dict], threshold: float = CORRECT) -> dict:
    """runs[0] is the baseline. Returns overall/per-set summaries on the shared cases and flips vs baseline."""
    with_set = all(r.get("set") for run in runs for r in run.get("cases", []))
    maps = [{case_key(r, with_set): r for r in run.get("cases", [])} for run in runs]
    shared = sorted(set.intersection(*(set(m) for m in maps))) if maps else []
    subsets = [[m[k] for k in shared] for m in maps]
    result = {
        "runs": [{"name": run["_name"], "label": run.get("label"), "prompt_version": run.get("prompt_version"),
                  "commit": run.get("commit"), "doctor_model": run.get("doctor_model"), "n_total": len(run.get("cases", []))}
                 for run in runs],
        "n_shared": len(shared),
        "overall": [summarize(rows) for rows in subsets],
        "by_set": [summarize_by_set(rows) for rows in subsets],
        "flips": [],
    }
    base = maps[0]
    for m in maps[1:]:
        lost, gained = [], []
        for k in shared:
            a, b = is_correct(base[k], threshold), is_correct(m[k], threshold)
            if a != b:
                item = {"case": k, "answer": m[k].get("answer"), "before": base[k].get("diagnosis"), "after": m[k].get("diagnosis")}
                (lost if a else gained).append(item)
        result["flips"].append({"right_to_wrong": lost, "wrong_to_right": gained})
    return result


def _f(v, digits=2) -> str:
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def _run_name(info: dict) -> str:
    bits = [info["label"] or info["name"]]
    if info.get("prompt_version"):
        bits.append(info["prompt_version"])
    if info.get("commit"):
        bits.append(info["commit"])
    return " · ".join(bits)


COLS = [("n", "n"), ("accuracy", "acc"), ("efficiency", "eff"), ("safety", "safety"), ("turns", "turns"), ("not_provided", "n/a rate")]


def _table_rows(res: dict) -> list[tuple[str, str, dict]]:
    rows = []
    for i, info in enumerate(res["runs"]):
        rows.append(("ALL", _run_name(info), res["overall"][i]))
    sets = sorted(set().union(*(bs.keys() for bs in res["by_set"])))
    if len(sets) > 1 or (sets and sets[0] != "-"):
        for s in sets:
            for i, info in enumerate(res["runs"]):
                rows.append((s, _run_name(info), res["by_set"][i].get(s, {})))
    return rows


def to_text(res: dict) -> str:
    lines = [f"shared cases: {res['n_shared']}  (" + ", ".join(f"{_run_name(r)}: {r['n_total']}" for r in res["runs"]) + ")"]
    rows = _table_rows(res)
    header = ["set", "run"] + [c[1] for c in COLS]
    body = [[s, name] + [_f(m.get(k), 1 if k == "turns" else 2) if k != "n" else _f(m.get(k)) for k, _ in COLS] for s, name, m in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    lines += [fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in body]
    for info, fl in zip(res["runs"][1:], res["flips"]):
        lines.append(f"\n{_run_name(res['runs'][0])} → {_run_name(info)}: "
                     f"right→wrong {len(fl['right_to_wrong'])}, wrong→right {len(fl['wrong_to_right'])}")
        for tag, items in (("-", fl["right_to_wrong"]), ("+", fl["wrong_to_right"])):
            for it in items:
                lines.append(f"  {tag} {it['case']}: 정답={it['answer']} | {it['before']} → {it['after']}")
    return "\n".join(lines)


def to_markdown(res: dict) -> str:
    lines = [f"Shared cases: {res['n_shared']}", "",
             "| Set | Run | " + " | ".join(c[1] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for s, name, m in _table_rows(res):
        cells = [_f(m.get(k), 1 if k == "turns" else 2) if k != "n" else _f(m.get(k)) for k, _ in COLS]
        lines.append(f"| {s} | {name} | " + " | ".join(cells) + " |")
    for info, fl in zip(res["runs"][1:], res["flips"]):
        lines += ["", f"**{_run_name(res['runs'][0])} → {_run_name(info)}**: "
                      f"right→wrong {len(fl['right_to_wrong'])}, wrong→right {len(fl['wrong_to_right'])}"]
        if fl["right_to_wrong"] or fl["wrong_to_right"]:
            lines += ["", "| | Case | Answer | Before | After |", "|---|---|---|---|---|"]
            for tag, items in (("−", fl["right_to_wrong"]), ("+", fl["wrong_to_right"])):
                for it in items:
                    lines.append(f"| {tag} | {it['case']} | {it['answer']} | {it['before']} | {it['after']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="result JSONs; the first is the baseline")
    ap.add_argument("--latest", type=int, help="use the N newest run_*.json in --results (oldest first = baseline)")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--md", action="store_true", help="print Markdown")
    ap.add_argument("--threshold", type=float, default=CORRECT, help="accuracy >= this counts as correct")
    args = ap.parse_args(argv)
    files = list(args.files)
    if args.latest:
        files += [str(p) for p in sorted(Path(args.results).glob("run_*.json"))[-args.latest:]]
    if len(files) < 2:
        ap.error("need at least two result files (or --latest N)")
    res = compare([load(f) for f in files], args.threshold)
    sys.stdout.write((to_markdown(res) if args.md else to_text(res)) + "\n")


if __name__ == "__main__":
    main()
