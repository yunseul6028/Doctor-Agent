"""Offline calibration of the three "second look" triggers on past real-LLM runs (no LLM, no network).

    python eval/offline/eval_triggers.py [--results DIR] [--folds 5] [--repeats 20] [--json OUT] [--cache FILE] [-v]

Triggers (docs/architecture.md "Advisors" / "Specialist sub-agents"):
  anchoring  agent/anchoring.anchoring_check, checked before every step from MIN_TURNS on, shown once per case
  consult    agent/subagents/orchestrator._consult_trigger (routed branch; the stuck branch reads the model's own
             confidence, which the result files do not keep, so it is not replayed)
  advocate   the anchoring moment, or before the pre-diagnosis review when the code confidence of the proposal is low

Replay. Every case of every eval/results/run_*.json except doctor_model "dummy" is rebuilt at each decision point j
(= turn_count when the policy chooses action j+1, j = 1..n, n = index of the final DIAGNOSE) exactly as the runtime
sees it: turns[:j], the DDx ledger updated with the snapshots of actions 1..j (the ledger the policy holds before
the LLM call of step j+1), findings reported before that step. The pre-review point is the state at the final
DIAGNOSE (ledger includes that action's snapshot) with the final diagnosis as the proposal.
Label: the case "ends wrong" when the judged accuracy < 1 (0 or 0.5). Name matching / judge noise applies.

Variants are "first decision point j >= t0 where every atom holds" (per-turn triggers) or "atoms hold at the
pre-review point" (advocate). Selection inside the training folds: among variants with fire rate <= the target,
the most wrong cases caught, then the higher precision, then the lower fire rate. Out-of-sample numbers come from
grouped cross-validation by case id (the same case appears in several runs; a case is never in train and test at once),
repeated with different fold assignments, and from leave-one-run-out. With ~12 distinct wrong cases every number
here is noisy: read differences under ~5 wrong cases as ties.

CURRENT names the rules before the 2026-09-29 calibration, CHOSEN the ones now in the code (config.AgentConfig
defaults); results and reasoning: docs/experiments.md "trigger calibration". Calibrated on Gemini/Gemma runs only:
re-run on gpt-oss-20b result files (--results) before trusting the thresholds.

Paths are relative to the repo root (this file's grandparent's parent).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from doctor_agent.agent import anchoring as A  # noqa: E402
from doctor_agent.agent import confidence as C  # noqa: E402
from doctor_agent.agent.ledger import Finding  # noqa: E402
from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.config import AgentConfig  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402
from doctor_agent.knowledge import specialty as sp  # noqa: E402

TARGET = {"anchoring": 0.30, "consult": 0.35, "advocate": 0.40}
MAX_TURNS = 60
MIN_REMAINING = 4  # AgentConfig.subagent_min_remaining_turns


# ------------------------------------------------------------------------------------------------------ replay
def state_at(case: dict, j: int, pre_review: bool = False) -> CaseState:
    turns = case["turns"]
    st = CaseState(case.get("initial", "") or "")
    for t in turns[:j]:
        try:
            typ = ActionType(t.get("type", "ASK"))
        except ValueError:
            typ = ActionType.ASK
        st.turns.append(Turn(Action(typ, t.get("content", ""), t.get("reason", "")), t.get("response", ""),
                             t.get("ddx") or []))
    for t in turns[:j + 1 if pre_review else j]:
        if t.get("ddx"):
            st.ddx_ledger.update(t["ddx"])
    st.ddx = st.ddx_ledger.as_list()
    lim = j if pre_review else j - 1  # findings reported with action t carry turn t-1
    for f in case.get("findings") or []:
        if int(f.get("turn", 0) or 0) <= lim and str(f.get("item", "")).strip():
            st.findings.items.append(Finding(f["item"], f.get("status", "양성"), f.get("detail", ""),
                                             int(f.get("turn", 0) or 0), f.get("verified")))
    return st


def point_features(st: CaseState, proposal: str | None = None) -> dict:
    """Everything the runtime can compute at this decision point (no look-ahead)."""
    live = C._live(st)
    dx = proposal or (live[0].dx if live else "")
    p1 = live[0].p if live else 0.0
    p2 = live[1].p if len(live) > 1 else 0.0
    comp, det = C.features(st, dx, AgentConfig.target_turns)
    e = C._entry(st, dx) if dx else None
    spec, share, _ = sp.route(st)
    n_spec = 0
    if spec:
        n_spec = sum(1 for n, _w in sp._candidates(st) if sp.specialty_of(n) == spec)
    chk = A.anchoring_check(st, min_turns=1)  # the start turn is part of each variant (t0)
    return {"j": len(st.turns), "dx": dx, "p1": round(p1, 3), "margin12": round(max(0.0, p1 - p2), 3),
            "n_live": len(live), "conf": round(C.score_of(comp), 4),
            "conf_margin": comp.get("margin") or 0.0, "support_g": len(det.get("support") or []),
            "against_g": len(det.get("against") or []), "confirm": comp.get("confirmatory_test") or 0.0,
            "criteria": comp.get("criteria_met"), "dangers": len(det.get("dangers") or []),
            "support_raw": len(e.support) if e else 0, "against_raw": len(e.against) if e else 0,
            "spec": spec, "share": round(float(share or 0.0), 3), "n_spec": n_spec,
            "anch": list(chk["reasons"]) if chk else []}


def load_cases(results_dir: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
        run = json.load(open(f, encoding="utf-8"))
        if run.get("doctor_model") == "dummy":
            continue
        for case in run.get("cases") or []:
            turns = case.get("turns") or []
            acc = (case.get("scores") or {}).get("accuracy")
            if not turns or acc is None or not any(t.get("ddx") for t in turns):
                continue
            out.append({"run": os.path.basename(f)[4:-5], "model": run.get("doctor_model"), "case": case.get("case"),
                        "wrong": float(acc) < 1.0, "raw": case})
    return out


def extract(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        case = c["raw"]
        turns = case["turns"]
        n = len(turns) - 1 if turns[-1].get("type") == "DIAGNOSE" else len(turns)
        pts = [point_features(state_at(case, j)) for j in range(1, n + 1)]
        pre = None
        if turns[-1].get("type") == "DIAGNOSE":
            pre = point_features(state_at(case, n, pre_review=True), turns[-1].get("content") or case.get("diagnosis"))
        rows.append({k: c[k] for k in ("run", "model", "case", "wrong")} | {"n": n, "points": pts, "pre": pre})
    return rows


# ------------------------------------------------------------------------------------------------------ rules
def _atoms() -> list[tuple[str, callable]]:
    """(name, predicate over one decision point's features). Every variant is a conjunction of <= 2 of these."""
    a: list[tuple[str, callable]] = []
    for s in (0.6, 0.7, 0.8, 0.9, 1.0):
        a.append((f"share>={s}", lambda f, s=s: f["share"] >= s))
    for s in (0.5, 0.6, 0.7, 0.8):
        a.append((f"share<{s}", lambda f, s=s: f["share"] < s))
    for k in (2, 3, 4):
        a.append((f"n_spec>={k}", lambda f, k=k: f["n_spec"] >= k))
    a.append(("n_spec<=1", lambda f: f["n_spec"] <= 1))
    for x in (0.3, 0.4, 0.5, 0.6, 0.7):
        a.append((f"p1<={x}", lambda f, x=x: f["p1"] <= x))
    for x in (0.5, 0.6, 0.7, 0.8):
        a.append((f"p1>={x}", lambda f, x=x: f["p1"] >= x))
    for x in (0.0, 0.1, 0.2):
        a.append((f"margin<={x}", lambda f, x=x: f["margin12"] <= x))
    for x in (0.3, 0.4, 0.5):
        a.append((f"margin>={x}", lambda f, x=x: f["margin12"] >= x))
    for x in (0.2, 0.3, 0.4, 0.5, 0.65):
        a.append((f"conf<{x}", lambda f, x=x: f["conf"] < x))
    for x in (0.5, 0.65, 0.8):
        a.append((f"conf>={x}", lambda f, x=x: f["conf"] >= x))
    for k in (0, 1):
        a.append((f"support_g<={k}", lambda f, k=k: f["support_g"] <= k))
    for k in (6, 8, 10):
        a.append((f"n_live>={k}", lambda f, k=k: f["n_live"] >= k))
    a += [("support_g>=3", lambda f: f["support_g"] >= 3),
          ("against_g>=1", lambda f: f["against_g"] >= 1), ("against_raw>=2", lambda f: f["against_raw"] >= 2),
          ("no_confirm", lambda f: not f["confirm"]), ("confirm", lambda f: bool(f["confirm"])),
          ("dangers>=1", lambda f: f["dangers"] >= 1), ("dangers==0", lambda f: f["dangers"] == 0),
          ("anch_any", lambda f: bool(f["anch"])), ("anch_untested", lambda f: "stable_untested" in f["anch"]),
          ("anch_weak", lambda f: "weak_support" in f["anch"]), ("anch_contra", lambda f: "contradicted" in f["anch"])]
    return a


def _masks(rows: list[dict], atoms) -> list[dict[str, int]]:
    """Per trajectory: atom name -> bitmask over its decision points (bit i = point i satisfies the atom);
    "_pre" + name for the pre-review point."""
    out = []
    for r in rows:
        m: dict[str, int] = {}
        for name, pred in atoms:
            bits = 0
            for i, f in enumerate(r["points"]):
                if pred(f):
                    bits |= 1 << i
            m[name] = bits
            m["_pre" + name] = int(bool(r["pre"] and pred(r["pre"])))
        out.append(m)
    return out


def _first(row: dict, bits: int, t0: int, need_spec: bool) -> int | None:
    for i, f in enumerate(row["points"]):
        if bits >> i & 1 and f["j"] >= t0 and MAX_TURNS - f["j"] >= MIN_REMAINING and (f["spec"] or not need_spec):
            return f["j"]
    return None


def turn_rule(rows, masks, t0: int, names: tuple[str, ...], need_spec: bool = False) -> list[int | None]:
    out = []
    for r, m in zip(rows, masks):
        bits = -1
        for n in names:
            bits &= m[n]
        out.append(_first(r, bits, t0, need_spec))
    return out


def pre_rule(rows, masks, names: tuple[str, ...]) -> list[int | None]:
    """Fires at the pre-review point (the review runs only with > 3 turns left)."""
    return [r["pre"]["j"] if r["pre"] and MAX_TURNS - r["pre"]["j"] > 3 and all(m["_pre" + n] for n in names) else None
            for r, m in zip(rows, masks)]


def either(a: list, b: list) -> list:
    return [x if x is not None else y for x, y in zip(a, b)]


def evaluate(rows: list[dict], fire_at: list, idx: list[int] | None = None) -> dict:
    idx = range(len(rows)) if idx is None else idx
    n = nw = fw = fr = 0
    turns = []
    for i in idx:
        w = rows[i]["wrong"]
        n += 1
        nw += w
        if fire_at[i] is not None:
            fw += w
            fr += not w
            turns.append(fire_at[i])
    fired = fw + fr
    base = nw / n if n else 0.0
    prec = fw / fired if fired else 0.0
    turns.sort()
    return {"n": n, "wrong": nw, "fired": fired, "fire_rate": round(fired / n, 3) if n else 0.0,
            "fired_wrong": fw, "fired_right": fr,
            "rate_on_wrong": round(fw / nw, 3) if nw else 0.0, "rate_on_right": round(fr / (n - nw), 3) if n - nw else 0.0,
            "precision": round(prec, 3), "recall": round(fw / nw, 3) if nw else 0.0,
            "lift": round(prec / base, 2) if base and fired else 0.0,
            "median_turn": turns[len(turns) // 2] if turns else None}


def pick(rows, variants: list[tuple[str, list]], target: float, idx: list[int] | None = None) -> tuple[str, list]:
    """Among variants with fire rate <= target on idx: most wrong cases caught, then precision, then lower fire rate,
    then the shorter (simpler) rule. Falls back to the first variant (the current rule) when none qualifies."""
    best, key = variants[0], None
    for name, fire in variants:
        m = evaluate(rows, fire, idx)
        if m["fire_rate"] > target or not m["fired"]:
            continue
        k = (m["fired_wrong"], m["precision"], -m["fire_rate"], -name.count("&"), -len(name))
        if key is None or k > key:
            best, key = (name, fire), k
    return best


def _fold(case_id: str, k: int, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{case_id}".encode()).hexdigest(), 16) % k


def crossval(rows, variants, target: float, folds: int, repeats: int) -> dict:
    """Out-of-sample estimate of "pick the best variant on the training part": grouped k-fold by case id (pooled over
    the folds, averaged over `repeats` fold assignments) and leave-one-run-out (training part = the other runs minus
    every case that also appears in the held-out run)."""
    reps, picks = [], {}
    for seed in range(repeats):
        fire: list = [None] * len(rows)
        for k in range(folds):
            tr = [i for i, r in enumerate(rows) if _fold(r["case"], folds, seed) != k]
            name, vec = pick(rows, variants, target, tr)
            picks[name] = picks.get(name, 0) + 1
            for i, r in enumerate(rows):
                if _fold(r["case"], folds, seed) == k:
                    fire[i] = vec[i]
        reps.append(evaluate(rows, fire))
    cv = {m: round(statistics.mean(r[m] for r in reps), 3) for m in
          ("fire_rate", "fired_wrong", "fired_right", "rate_on_wrong", "rate_on_right", "precision", "lift")}
    cv["wrong"] = reps[0]["wrong"]
    fire = [None] * len(rows)
    loro = {}
    for run in sorted({r["run"] for r in rows}):
        held = {r["case"] for r in rows if r["run"] == run}
        tr = [i for i, r in enumerate(rows) if r["run"] != run and r["case"] not in held]
        name, vec = pick(rows, variants, target, tr)
        loro[run] = name
        for i, r in enumerate(rows):
            if r["run"] == run:
                fire[i] = vec[i]
    return {"cv": cv, "cv_picks": dict(sorted(picks.items(), key=lambda x: -x[1])[:8]), "loro": evaluate(rows, fire),
            "loro_picks": loro}


# ------------------------------------------------------------------------------------------------------ families
# Three nested families per trigger, searched separately so that the out-of-sample cost of a bigger search shows:
#   threshold  the current rule with its own thresholds moved (start turn, share, confidence cut-off)
#   gated      the current rule AND one extra atom (keeps the trigger's meaning, adds a condition)
#   full       any conjunction of <= 2 atoms (+ start turn)
T0 = (3, 4, 5, 6, 8)
_SPEC_ATOMS = ("share", "n_spec")


def _combos(names: list[str]):
    return itertools.chain(((n,) for n in names), itertools.combinations(names, 2))


def anchoring_family(rows, masks, atoms, mode: str) -> list[tuple[str, list]]:
    out = [(CURRENT["anchoring"], turn_rule(rows, masks, 3, ("anch_any",)))]
    reasons = ("anch_any", "anch_untested", "anch_weak", "anch_contra")
    for t0 in T0:
        for reason in reasons:
            out.append((f"j>={t0} & {reason}", turn_rule(rows, masks, t0, (reason,))))
    names = [n for n, _ in atoms if not n.startswith(_SPEC_ATOMS + ("anch",))]
    for t0 in T0:
        if mode == "gated":
            out += [(f"j>={t0} & {n} & anch_any", turn_rule(rows, masks, t0, (n, "anch_any"))) for n in names]
        elif mode == "full":
            out += [(f"j>={t0} & " + " & ".join(c), turn_rule(rows, masks, t0, c))
                    for c in _combos(names + list(reasons))]
    return out


def consult_family(rows, masks, atoms, mode: str) -> list[tuple[str, list]]:
    """A routed specialty is always required (the consult needs one)."""
    out = [(CURRENT["consult"], turn_rule(rows, masks, 3, ("share>=0.6",), True))]
    for t0 in T0 + (10,):
        for s in (0.6, 0.7, 0.8, 0.9, 1.0):
            out.append((f"j>={t0} & share>={s}", turn_rule(rows, masks, t0, (f"share>={s}",), True)))
    names = [n for n, _ in atoms]
    for t0 in T0:
        if mode == "gated":
            for s in (0.6, 0.7, 0.8):
                out += [(f"j>={t0} & share>={s} & {n}", turn_rule(rows, masks, t0, (f"share>={s}", n), True))
                        for n in names if not n.startswith("share")]
        elif mode == "full":
            out += [(f"j>={t0} & " + " & ".join(c), turn_rule(rows, masks, t0, c, True)) for c in _combos(names)]
    return out


def advocate_family(rows, masks, atoms, anch: list, mode: str) -> list[tuple[str, list]]:
    """Advocate = anchoring moment (`anch`: fire turns of the anchoring rule in use) or the pre-review predicate; the
    "pre:" variants drop the anchoring moment."""
    out = [(CURRENT["advocate"], either(anch, pre_rule(rows, masks, ("conf<0.65",))))]
    for x in (0.2, 0.3, 0.4, 0.5, 0.65):
        out.append((f"anchoring | pre: conf<{x}", either(anch, pre_rule(rows, masks, (f"conf<{x}",)))))
        out.append((f"pre: conf<{x}", pre_rule(rows, masks, (f"conf<{x}",))))
    names = [n for n, _ in atoms if not n.startswith(_SPEC_ATOMS + ("anch", "conf"))]
    if mode == "gated":
        for x in (0.4, 0.5, 0.65):
            for n in names:
                out.append((f"anchoring | pre: conf<{x} & {n}", either(anch, pre_rule(rows, masks, (f"conf<{x}", n)))))
                out.append((f"pre: conf<{x} & {n}", pre_rule(rows, masks, (f"conf<{x}", n))))
    elif mode == "full":
        names = [n for n, _ in atoms if not n.startswith(_SPEC_ATOMS + ("anch",))]
        for c in _combos(names):
            out.append(("anchoring | pre: " + " & ".join(c), either(anch, pre_rule(rows, masks, c))))
            out.append(("pre: " + " & ".join(c), pre_rule(rows, masks, c)))
    return out


# ------------------------------------------------------------------------------------------------------ main
CURRENT = {"anchoring": "j>=3 & anch_any (current)", "consult": "j>=3 & share>=0.6 (current)",
           "advocate": "anchoring | pre: conf<0.65 (current)"}
# rules implemented in the code (variant names above); None = report only
CHOSEN: dict[str, str | None] = {"anchoring": "j>=5 & anch_any", "consult": "j>=5 & share>=0.6",
                                 "advocate": "pre: conf<0.4"}
MODES = ("threshold", "gated", "full")


def _line(tag: str, m: dict) -> str:
    return (f"  {tag:46s} fire {m['fire_rate']:6.1%} | on wrong {m['rate_on_wrong']:6.1%} ({m['fired_wrong']}/{m['wrong']})"
            f" | on right {m['rate_on_right']:6.1%} | precision {m['precision']:.2f} | lift {m['lift']:.2f}"
            + (f" | median turn {m['median_turn']}" if m.get("median_turn") is not None else ""))


def report(rows, name: str, key: str, families: dict, target, folds, repeats, verbose: bool) -> dict:
    cur_name, cur_vec = families["threshold"][0]
    cur = evaluate(rows, cur_vec)
    print(f"\n[{name}] target fire rate <= {target:.0%}; variants: "
          + ", ".join(f"{m} {len(v)}" for m, v in families.items()))
    print(_line("current: " + cur_name, cur))
    out: dict = {"current": cur}
    for fam, variants in families.items():
        pname, pvec = pick(rows, variants, target)
        ins = evaluate(rows, pvec)
        oos = crossval(rows, variants, target, folds, repeats)
        print(f" {fam}: in-sample pick = {pname}")
        print(_line("   in-sample", ins))
        print(_line(f"   out-of-sample ({folds}-fold by case, x{repeats})", oos["cv"]))
        print(_line("   out-of-sample (leave-one-run-out)", oos["loro"]))
        if verbose:
            print("     CV picks:", oos["cv_picks"])
            print("     LORO picks:", oos["loro_picks"])
        out[fam] = {"pick": pname, "in_sample": ins, **oos}
    if chosen := CHOSEN.get(key):
        vec = dict(v for fam in families.values() for v in fam)[chosen]
        m = evaluate(rows, vec)
        print(_line("chosen: " + chosen, m))
        out["chosen"] = {"rule": chosen, **m}
        for mod in sorted({r["model"] for r in rows}):
            mm = evaluate(rows, vec, [i for i, r in enumerate(rows) if r["model"] == mod])
            out["chosen"].setdefault("by_model", {})[mod] = mm
            print(_line(f"   {mod} (current {evaluate(rows, cur_vec, [i for i, r in enumerate(rows) if r['model'] == mod])['fire_rate']:.0%})", mm))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "eval", "results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--json", default="", help="write the report as JSON")
    ap.add_argument("--cache", default="", help="decision-point features JSON: read when it exists, else written")
    ap.add_argument("-v", action="store_true", help="list the rules picked in each fold")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.cache and os.path.exists(a.cache):
        rows = json.load(open(a.cache, encoding="utf-8"))
    else:
        sp.warm()
        rows = extract(load_cases(a.results))
        if a.cache:
            json.dump(rows, open(a.cache, "w", encoding="utf-8"), ensure_ascii=False)
    if not rows:
        sys.exit(f"no non-dummy trajectories with DDx snapshots under {a.results} (use --results)")
    models = sorted({r["model"] for r in rows})
    print(f"[load] {len(rows)} trajectories, {len({r['run'] for r in rows})} runs, models {models}; "
          f"{sum(r['wrong'] for r in rows)} end wrong; {len({r['case'] for r in rows})} distinct cases, "
          f"{len({r['case'] for r in rows if r['wrong']})} of them wrong at least once; "
          f"{sum(len(r['points']) for r in rows)} decision points ({time.perf_counter() - t0:.1f}s)")
    atoms = _atoms()
    masks = _masks(rows, atoms)
    res: dict = {"n": len(rows), "wrong": sum(r["wrong"] for r in rows)}
    fam = {m: anchoring_family(rows, masks, atoms, m) for m in MODES}
    res["anchoring"] = report(rows, "anchoring", "anchoring", fam, TARGET["anchoring"], a.folds, a.repeats, a.v)
    all_anch = dict(v for f in fam.values() for v in f)
    fam = {m: consult_family(rows, masks, atoms, m) for m in MODES}
    res["consult"] = report(rows, "consult", "consult", fam, TARGET["consult"], a.folds, a.repeats, a.v)
    # the advocate shares its once-per-case budget with the anchoring moment: evaluated with the anchoring rule in use
    for tag, anch_rule in (("current", CURRENT["anchoring"]), ("chosen", CHOSEN["anchoring"])):
        if anch_rule is None:
            continue
        fam = {m: advocate_family(rows, masks, atoms, all_anch[anch_rule], m) for m in MODES}
        res[f"advocate ({tag} anchoring)"] = report(
            rows, f"advocate (anchoring moment = {tag} anchoring rule)", "advocate" if tag == "chosen" else "",
            fam, TARGET["advocate"], a.folds, a.repeats, a.v)
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
