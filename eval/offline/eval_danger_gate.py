"""Offline measurement of the can't-miss gate (safety/danger_gate.py) on local case files (no LLM, no network).

    python eval/offline/eval_danger_gate.py [--cases DIR] [--no-interp] [--json OUT] [-v] [--blocked]

For each case file (default: every data/cases_aug/*/*.json):
  (a) raised_initial  dangers on the gate's checked list from the initial information only
  (b) full reveal     every history / exam / test entry of the case added as an ASK / EXAM / TEST turn (action text
                      = the entry's key alternatives), then gate(state, gold diagnosis): blocked or not, which
                      danger, the action it demands, the danger's rule-out criteria
  (c) true catches    raised dangers that ARE the gold diagnosis (lookup(gold) == danger) vs not
  forced steps        the gate is replayed as the runtime does (max_gate_turns=3): while it blocks, its action is
                      sent to the case-file environment (eval/simulator.py) and the response appended. Counted from
                      the full-reveal state ("full") and from the initial-only state ("initial", upper bound: the
                      doctor proposes the gold diagnosis without any work-up).

The code result interpreter (agent/result_interpreter.py, on by default in AgentConfig) is run on every EXAM/TEST
turn so critical results reach the gate as at runtime; --no-interp skips it.

Caveat: this is the case set the gate was designed against; read improvements as in-sample.
Paths are relative to the repo root (this file's grandparent's parent).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from doctor_agent.agent.state import CaseState, Turn  # noqa: E402
from doctor_agent.env.interface import Action, ActionType  # noqa: E402
from doctor_agent.safety import danger_gate as g  # noqa: E402

MAX_GATE_TURNS = 3
_TABLES = (("history", ActionType.ASK), ("exam", ActionType.EXAM), ("tests", ActionType.TEST))


def _interp(state: CaseState, turn_no: int, t: Turn) -> None:
    """Critical results as the policy stores them (policy._interpret_turn)."""
    from doctor_agent.agent import result_interpreter as ri
    if t.action.type not in (ActionType.EXAM, ActionType.TEST) or not (t.response or "").strip():
        return
    try:
        interp = ri.interpret(t.action.content, t.response, {"initial_info": state.initial_info})
    except Exception:  # noqa: BLE001
        return
    for i in interp.critical():
        if i.polarity == "absent":
            continue
        state.result_criticals.append({"turn": turn_no, "test": interp.test_name, "concept": i.concept,
                                       "label": i.label, "polarity": i.polarity, "kind": i.kind,
                                       "laterality": i.laterality, "summary": i.summary_ko.lstrip("⚠")})


def _add(state: CaseState, typ: ActionType, content: str, response: str, interp: bool) -> None:
    t = Turn(Action(typ, content), response)
    state.turns.append(t)
    if interp:
        _interp(state, len(state.turns), t)


def initial_state(case: dict) -> CaseState:
    return CaseState(case.get("initial", "") or "")


def full_state(case: dict, interp: bool = True) -> CaseState:
    st = initial_state(case)
    for key, typ in _TABLES:
        for k, v in (case.get(key) or {}).items():
            _add(st, typ, " / ".join(a.strip() for a in k.split("|") if a.strip()), str(v), interp)
    return st


def _respond(case: dict, action: Action) -> str:
    from eval.simulator import CaseFileEnvironment
    return CaseFileEnvironment(case).step(action).text


def gold_names(case: dict) -> set[str]:
    out = set()
    for n in [case.get("diagnosis", "")] + list(case.get("aliases") or []):
        r = g.lookup(n)
        if r is not None:
            out.add(r.name)
    return out


def replay(case: dict, st: CaseState, gold: str, interp: bool) -> list[dict]:
    """Gate replay: forced actions until allowed or MAX_GATE_TURNS used."""
    forced = []
    for used in range(MAX_GATE_TURNS):
        res = g.gate(st, gold, remaining_turns=100, max_gate_turns=MAX_GATE_TURNS, gate_turns_used=used)
        if res["allow"]:
            break
        typ, content, _ = res["action"]
        resp = _respond(case, Action(typ, content))
        forced.append({"danger": res["danger"], "type": typ.value, "action": content,
                       "available": "제공되지 않" not in resp and resp != "잘 모르겠어요."})
        _add(st, typ, content, resp, interp)
    return forced


def measure(path: str, interp: bool = True) -> dict:
    case = json.load(open(path, encoding="utf-8"))
    gold = case.get("diagnosis", "")
    golds = gold_names(case)
    s0 = initial_state(case)
    raised0 = g.raised_dangers(s0)
    sf = full_state(case, interp)
    raised_f = g.raised_dangers(sf)
    res = g.gate(sf, gold, remaining_turns=100, max_gate_turns=MAX_GATE_TURNS, gate_turns_used=0)
    status_f = {n: g._status(g.RULE_OUT[n], g._Ctx(sf))[0] for n in raised_f}
    block = None
    if not res["allow"]:
        block = {"danger": res["danger"], "action": list(res["action"][1:2])[0], "type": res["action"][0].value,
                 "criteria": g.RULE_OUT[res["danger"]].criteria}
    return {
        "case": os.path.relpath(path, ROOT), "initial": case.get("initial", ""), "gold": gold,
        "gold_danger": sorted(golds),
        "raised_initial": sorted(raised0), "raised_full": {k: raised_f[k] for k in sorted(raised_f)},
        "status_full": status_f, "blocked": block,
        "forced_full": replay(case, full_state(case, interp), gold, interp),
        "forced_initial": replay(case, initial_state(case), gold, interp),
    }


def summarise(rows: list[dict]) -> dict:
    raised0 = collections.Counter(n for r in rows for n in r["raised_initial"])
    raised_f = collections.Counter(n for r in rows for n in r["raised_full"])
    gold = collections.Counter(n for r in rows for n in r["gold_danger"])
    catch0 = collections.Counter(n for r in rows for n in r["raised_initial"] if n in r["gold_danger"])
    catch_f = collections.Counter(n for r in rows for n in r["raised_full"] if n in r["gold_danger"])
    blocked = collections.Counter(r["blocked"]["danger"] for r in rows if r["blocked"])
    forced_by = collections.Counter(f["danger"] for r in rows for f in r["forced_initial"])
    return {
        "n_cases": len(rows),
        "gold_blocked": sum(1 for r in rows if r["blocked"]),
        "gold_blocked_by_danger": dict(blocked.most_common()),
        "forced_steps_full": sum(len(r["forced_full"]) for r in rows),
        "forced_steps_full_unavailable": sum(1 for r in rows for f in r["forced_full"] if not f["available"]),
        "forced_steps_initial": sum(len(r["forced_initial"]) for r in rows),
        "forced_steps_initial_by_danger": dict(forced_by.most_common()),
        "per_danger": {n: {"raised_initial": raised0[n], "raised_full": raised_f[n], "gold": gold[n],
                           "true_catch_initial": catch0[n], "true_catch_full": catch_f[n],
                           "gold_blocked": blocked[n], "forced_initial": forced_by[n]}
                       for n in g.RULE_OUT},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases_aug"),
                    help="directory of case JSON files (searched recursively; default data/cases_aug)")
    ap.add_argument("--no-interp", action="store_true", help="do not run the code result interpreter")
    ap.add_argument("--json", help="write per-case rows + summary to this file")
    ap.add_argument("--blocked", action="store_true", help="print every case whose gold diagnosis is blocked")
    ap.add_argument("-v", "--verbose", action="store_true", help="print one line per case")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.cases, "**", "*.json"), recursive=True))
    rows = [measure(f, interp=not a.no_interp) for f in files]
    summ = summarise(rows)
    for r in rows:
        if a.verbose or (a.blocked and r["blocked"]):
            b = r["blocked"]
            print(f"{r['case']}\t{r['gold']}\t{r['initial'][:40]}\t"
                  + (f"BLOCK {b['danger']} → {b['type']} {b['action']}" if b else "allow")
                  + f"\traised={','.join(r['raised_full'])}\tstatus={r['status_full']}")
    print(f"cases {summ['n_cases']}  gold blocked (full reveal) {summ['gold_blocked']}  "
          f"forced steps: full {summ['forced_steps_full']} (unavailable {summ['forced_steps_full_unavailable']}), "
          f"initial-only {summ['forced_steps_initial']}")
    print(f"{'danger':<22}{'raised0':>8}{'raisedF':>8}{'gold':>6}{'catch0':>7}{'catchF':>7}{'blocked':>8}"
          f"{'forced0':>8}")
    for n, d in summ["per_danger"].items():
        print(f"{n:<22}{d['raised_initial']:>8}{d['raised_full']:>8}{d['gold']:>6}{d['true_catch_initial']:>7}"
              f"{d['true_catch_full']:>7}{d['gold_blocked']:>8}{d['forced_initial']:>8}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summ, "cases": rows}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
