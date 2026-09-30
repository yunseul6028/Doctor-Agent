"""Offline evaluation of knowledge/specialty.py (no LLM, no network).

    python eval/offline/eval_specialty.py [--results DIR] [--json OUT] [--fuzzy] [-v]

1. Gold accuracy of specialty_of on data/labels/specialty_gold_v1.jsonl (six ids), specialty_gold_v2.jsonl (eight
   ids: new names + corrected copies of v1 items) and specialty_gold_v3.jsonl (ten ids, endo_metab + psych: new names
   + corrected copies of v1/v2 items); each hand-labelled before any output was seen;
   labelling guide in docs/architecture.md "Specialty routing". strict = predicted == label; lenient = predicted is
   the label or one of its "also" alternatives (None counts as correct when listed in "also").
2. Specialty distribution of the gold diagnoses of data/cases_aug per source (diagnosis first, then aliases), and the
   share outside the eight ids by bucket (which specialties to add next).
3. Routing simulation on past runs (eval/results/run_*.json by default, doctor_model "dummy" skipped): the per-turn
   DDx snapshots are replayed through route(); a consult "fires" at the first snapshot taken after >= MIN_TURNS turns
   with share >= MIN_SHARE (config.AgentConfig.consult_min_turns / consult_min_share, the runtime gate). Reports how often it fires and whether the routed specialty is the gold diagnosis's one.
   Also times route() and resources() on those states. --gold-from OUT.json scores the routing against the gold
   specialties recorded in another run's --json output (like-for-like before/after when the id set changes).
Paths are relative to the repo root (this file's grandparent's parent).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from doctor_agent.agent.ledger import Finding  # noqa: E402
from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402
from doctor_agent.config import AgentConfig  # noqa: E402
from doctor_agent.knowledge import specialty as sp  # noqa: E402

# the orchestrator's routed-consult gate (AGENT_CONSULT_MIN_TURNS / AGENT_CONSULT_MIN_SHARE)
MIN_TURNS, MIN_SHARE = AgentConfig().consult_min_turns, AgentConfig().consult_min_share


def _load(name: str) -> list[dict]:
    return [json.loads(x) for x in open(os.path.join(ROOT, "data/labels", name), encoding="utf-8") if x.strip()]


def gold_sets() -> dict[str, list[dict]]:
    """v1 as labelled (six ids), v1 with the v2 then v3 corrected copies applied, the v2 new names (v3 corrections
    applied), the v3 new names, and all of them."""
    def opt(name: str) -> list[dict]:
        return _load(name) if os.path.exists(os.path.join(ROOT, "data/labels", name)) else []
    v1, v2, v3 = _load("specialty_gold_v1.jsonl"), opt("specialty_gold_v2.jsonl"), opt("specialty_gold_v3.jsonl")
    corr2 = {r["name"]: r for r in v2 if r.get("source") == "v1_corrected"}
    corr3 = {r["name"]: r for r in v3 if r.get("source") == "v3_corrected"}
    v1c = [corr3.get(r["name"], corr2.get(r["name"], r)) for r in v1]
    new2 = [corr3.get(r["name"], r) for r in v2 if r.get("source") == "v2_new"]
    new3 = [r for r in v3 if r.get("source") == "v3_new"]
    out = {"v1": v1, "v1_corrected": v1c, "v2_new": new2}
    if new3:
        out["v3_new"] = new3
    out["all"] = v1c + new2 + new3
    return out


def gold_accuracy(verbose: bool, rows: list[dict] | None = None, tag: str = "gold") -> dict:
    rows = rows if rows is not None else _load("specialty_gold_v1.jsonl")
    strict = lenient = 0
    by_label = collections.Counter()
    by_label_ok = collections.Counter()
    errors = []
    for r in rows:
        d = sp.specialty_detail(r["name"])
        pred = d["specialty"]
        ok_s = pred == r["specialty"]
        ok_l = ok_s or pred in (r.get("also") or [])
        strict += ok_s
        lenient += ok_l
        lab = r["specialty"] or "none"
        by_label[lab] += 1
        by_label_ok[lab] += ok_s
        if not ok_s:
            errors.append({"name": r["name"], "gold": r["specialty"], "also": r.get("also"), "pred": pred,
                           "how": d["how"], "code": d["code"], "kb_name": d["name"], "lenient_ok": ok_l})
    n = len(rows)
    out = {"n": n, "strict": round(strict / n, 3), "lenient": round(lenient / n, 3),
           "per_label": {k: f"{by_label_ok[k]}/{by_label[k]}" for k in sorted(by_label)}, "errors": errors}
    print(f"[{tag}] n={n} strict={strict}/{n} ({strict / n:.1%}) lenient={lenient}/{n} ({lenient / n:.1%})")
    print(f"[{tag}] per label:", out["per_label"])
    for e in errors if verbose else []:
        print("   ", e)
    return out


def _case_detail(c: dict) -> dict:
    names = [c.get("diagnosis") or c.get("answer") or ""] + list(c.get("aliases") or [])
    first = None
    for n in names:
        d = sp.specialty_detail(n)
        first = first or d
        if d["how"] not in ("none", "error"):
            return d
    return first or sp.specialty_detail("")


def distribution(verbose: bool) -> dict:
    per_src: dict[str, collections.Counter] = {}
    groups = collections.Counter()
    how = collections.Counter()
    listing = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/cases_aug/*/*.json"))):
        c = json.load(open(f, encoding="utf-8"))
        src = os.path.basename(os.path.dirname(f))
        d = _case_detail(c)
        key = d["specialty"] or "none"
        per_src.setdefault(src, collections.Counter())[key] += 1
        groups[d["group"]] += 1
        how[d["how"]] += 1
        listing.append((src, c["diagnosis"], d["specialty"], d["group"], d["how"], d["code"]))
    total = collections.Counter()
    for cnt in per_src.values():
        total.update(cnt)
    n = sum(total.values())
    outside = total["none"]
    out_groups = {g: v for g, v in groups.most_common() if g not in sp.SPECIALTIES}
    print(f"[dist] cases={n}")
    for src, cnt in sorted(per_src.items()):
        print(f"   {src:15s} n={sum(cnt.values()):3d} " + " ".join(f"{k}={cnt[k]}" for k in (*sp.SPECIALTIES, 'none')))
    print(f"   {'all':15s} n={n:3d} " + " ".join(f"{k}={total[k]}" for k in (*sp.SPECIALTIES, 'none')))
    print(f"[dist] outside the {len(sp.SPECIALTIES)} ids: {outside}/{n} ({outside / n:.1%}); by bucket: {out_groups}")
    print(f"[dist] resolved by: {dict(how)}")
    if verbose:
        for row in listing:
            print("   ", row)
    return {"n": n, "per_source": {s: dict(c) for s, c in per_src.items()}, "all": dict(total),
            "outside_share": round(outside / n, 3), "outside_by_group": out_groups, "how": dict(how),
            "cases": [dict(zip(("source", "diagnosis", "specialty", "group", "how", "code"), r)) for r in listing]}


def _state(initial: str, turns: list[dict], upto: int, ddx: list[dict], findings: list[dict] | None = None) -> CaseState:
    st = CaseState(initial_info=initial)
    for f in findings or []:  # the run keeps only the final ledger: replay the items recorded up to this turn
        if isinstance(f, dict) and f.get("item") and int(f.get("turn") or 0) <= upto:
            st.findings.add(Finding(f["item"], f.get("status", "양성"), f.get("detail", ""), int(f.get("turn") or 0),
                                    f.get("verified"), f.get("span", ""), f.get("source", "")))
    for t in turns[:upto]:
        try:
            typ = ActionType(t.get("type", "ASK"))
        except ValueError:
            typ = ActionType.ASK
        st.turns.append(Turn(Action(typ, t.get("content", "")), t.get("response", "")))
    st.ddx_ledger.update(ddx)  # ledger path of route(): same as at runtime
    st.ddx = list(ddx)
    return st


def _gold_for(case: dict) -> dict:
    names = [case.get("answer", "")]
    path = case.get("path")
    if not path:
        hits = glob.glob(os.path.join(ROOT, "data", "**", f"{case.get('case', '')}.json"), recursive=True)
        path = hits[0] if hits else ""
    if path:
        try:
            c = json.load(open(os.path.join(ROOT, path) if not os.path.isabs(path) else path, encoding="utf-8"))
            names = [c.get("diagnosis") or names[0]] + list(c.get("aliases") or [])
        except OSError:
            pass
    return _case_detail({"diagnosis": names[0], "aliases": names[1:]})


def simulate(results_dir: str, verbose: bool, gold_map: dict | None = None) -> dict:
    files = sorted(glob.glob(os.path.join(results_dir, "run_*.json")))
    n_traj = fired = fired_ok = gold_in_six = fired_gold_in_six = fired_ok_in_six = 0
    final_ok = final_n = 0
    fire_turns: list[int] = []
    wrong_fired = wrong_fired_ok = 0
    fired_by = collections.Counter()
    t_route: list[float] = []
    t_res: list[float] = []
    examples = []
    gold_by_case: dict[str, str | None] = {}
    for f in files:
        run = json.load(open(f, encoding="utf-8"))
        if run.get("doctor_model") == "dummy":
            continue
        for case in run.get("cases", []):
            turns = case.get("turns") or []
            snaps = [(i, t.get("ddx")) for i, t in enumerate(turns) if t.get("ddx")]
            if not snaps:
                continue
            n_traj += 1
            g = _gold_for(case)
            key = f"{case.get('case', '')}|{case.get('answer', '')}"
            if gold_map is not None and key in gold_map:  # like-for-like: the gold ids of another run
                g = {**g, "specialty": gold_map[key]}
            gold_by_case[key] = g["specialty"]
            gold_in_six += g["specialty"] is not None
            hit = None
            last = None
            for i, ddx in snaps:  # snapshot on turn i was made after i earlier turns
                st = _state(case.get("initial", ""), turns, i, ddx, case.get("findings"))
                t0 = time.perf_counter()
                spec, share, why = sp.route(st)
                t_route.append((time.perf_counter() - t0) * 1000)
                last = (spec, share)
                if hit is None and i >= MIN_TURNS and spec and share >= MIN_SHARE:
                    hit = (i, spec, share, why)
                    if len(t_res) < 400:
                        t0 = time.perf_counter()
                        sp.resources(spec, st)
                        t_res.append((time.perf_counter() - t0) * 1000)
            if last and last[0]:
                final_n += 1
                final_ok += last[0] == g["specialty"]
            if hit:
                fired += 1
                fired_by[hit[1]] += 1
                fire_turns.append(hit[0])
                ok = hit[1] == g["specialty"]
                if (case.get("scores") or {}).get("accuracy", 1.0) < 1.0:  # the doctor's final answer was wrong
                    wrong_fired += 1
                    wrong_fired_ok += ok
                fired_ok += ok
                if g["specialty"] is not None:
                    fired_gold_in_six += 1
                    fired_ok_in_six += ok
                if verbose or (not ok and len(examples) < 12):
                    examples.append({"run": os.path.basename(f), "case": case.get("case"), "gold": case.get("answer"),
                                     "gold_spec": g["specialty"], "turn": hit[0], "routed": hit[1],
                                     "share": hit[2], "why": hit[3][:1]})

    def pct(a, b):
        return f"{a}/{b} ({a / b:.1%})" if b else f"{a}/0"

    def stats(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        return {"n": len(xs), "mean_ms": round(statistics.mean(xs), 2), "p95_ms": round(xs[int(0.95 * (len(xs) - 1))], 2),
                "max_ms": round(xs[-1], 2)}
    out = {"trajectories": n_traj, "fired": fired, "fired_share": round(fired / n_traj, 3) if n_traj else 0.0,
           "fired_correct": fired_ok, "fired_precision": round(fired_ok / fired, 3) if fired else 0.0,
           "gold_in_six": gold_in_six, "fired_gold_in_six": fired_gold_in_six,
           "fired_precision_gold_in_six": round(fired_ok_in_six / fired_gold_in_six, 3) if fired_gold_in_six else 0.0,
           "final_snapshot_agreement": round(final_ok / final_n, 3) if final_n else 0.0,
           "fired_by_specialty": dict(fired_by),
           "fire_turn_mean": round(statistics.mean(fire_turns), 2) if fire_turns else None,
           "fired_on_wrong_final": wrong_fired, "fired_on_wrong_final_correct_specialty": wrong_fired_ok, "route_time": stats(t_route), "resources_time": stats(t_res),
           "examples_wrong": examples, "gold_by_case": gold_by_case}
    print(f"[route] trajectories={n_traj} (gold mapped to an id: {gold_in_six})  consult fires (>= {MIN_TURNS} "
          f"turns, share "
          f">= {MIN_SHARE}): {pct(fired, n_traj)}")
    print(f"[route] routed == gold specialty: {pct(fired_ok, fired)}; among gold mapped to an id: "
          f"{pct(fired_ok_in_six, fired_gold_in_six)}; final snapshot agreement: {pct(final_ok, final_n)}")
    print(f"[route] fired by specialty: {dict(fired_by)}; mean firing turn {out['fire_turn_mean']}")
    print(f"[route] fired on cases whose final diagnosis was wrong: {wrong_fired}; routed to the gold specialty: "
          f"{pct(wrong_fired_ok, wrong_fired)}")
    print(f"[time] route: {out['route_time']}  resources: {out['resources_time']}")
    for e in examples[:12] if verbose else examples[:6]:
        print("   ", e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "eval", "results"))
    ap.add_argument("--json", default="")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--fuzzy", action="store_true", help="turn on the KB fuzzy name step (off at runtime)")
    ap.add_argument("--gold-from", default="", help="score routing against the gold ids in another run's --json")
    a = ap.parse_args()
    sp.FUZZY = a.fuzzy
    t0 = time.perf_counter()
    sp.warm()
    print(f"[load] KB + indexes {time.perf_counter() - t0:.2f}s")
    gs = gold_sets()
    res = {"gold": {k: gold_accuracy(a.v, rows, "gold " + k) for k, rows in gs.items() if rows},
           "distribution": distribution(a.v)}
    if os.path.isdir(a.results):
        gm = None
        if a.gold_from:
            with open(a.gold_from, encoding="utf-8") as f:
                gm = json.load(f).get("routing", {}).get("gold_by_case")
        res["routing"] = simulate(a.results, a.v, gm)
    else:
        print(f"[route] no results directory {a.results}: skipped")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
