"""Simulate anchoring_check on reconstructed trajectories from eval/results/run_*.json (read-only, no LLM).

Usage: python eval/offline/eval_anchoring.py [PARAM=value ...] [--ledger] [-v]
"""
import sys, json, glob, collections, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, ROOT + '/src')
from doctor_agent.agent import anchoring as A
from doctor_agent.agent.anchoring import anchoring_check, initial_differential
from doctor_agent.agent.ledger import DdxLedger, FindingsLedger, Finding
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.agent.text import same_dx
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge import kb

_k3 = {}


def k3(name):
    if name not in _k3:
        try:
            r = kb.normalize_diagnosis(name)
        except Exception:
            r = None
        _k3[name] = (r or {}).get("code", "").replace(".", "")[:3]
    return _k3[name]


def gold_in(names, gold):
    gk = k3(gold)
    return any(same_dx(n, gold) or (gk and k3(n) == gk) for n in names)


def build_state(case, k):
    """State after k completed turns: turns[:k] with their snapshots; ledger = snapshot the model held when choosing
    action k+1 (reflects responses 1..k); findings with turn <= k."""
    st = CaseState(initial_info=case.get('initial', ''))
    for t in case['turns'][:k]:
        st.turns.append(Turn(Action(ActionType(t['type']), t['content'], t.get('reason', '')), t.get('response', ''),
                             t.get('ddx') or []))
    nxt = case['turns'][k].get('ddx') or []
    led = DdxLedger()
    led.update([{"dx": d['dx'], "p": d.get('p', 0), "status": d.get('status', ''), "for": d.get('for'),
                 "against": d.get('against')} for d in nxt])
    st.ddx_ledger = led
    st.ddx = nxt
    fl = FindingsLedger()
    for f in case.get('findings') or []:
        if f.get('turn', 0) <= k:
            fl.items.append(Finding(f['item'], f.get('status', '양성'), f.get('detail', ''), f.get('turn', 0),
                                    f.get('verified')))
    st.findings = fl
    return st


def main():
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    for a in sys.argv[1:]:
        if '=' in a:
            k, v = a.split('=')
            setattr(A, k, float(v) if '.' in v else int(v))

    LEDGER_ONLY = '--ledger' in sys.argv
    stats = collections.Counter()
    reasons = {"right": collections.Counter(), "wrong": collections.Counter()}
    fire_turn = collections.defaultdict(list)
    examples = []
    base_t1 = union_t1 = n_all = 0
    for f in sorted(glob.glob(ROOT + '/eval/results/run_*.json')):
        if LEDGER_ONLY and ('204021' not in f and '204404' not in f):
            continue
        d = json.load(open(f, encoding='utf-8'))
        for c in d['cases']:
            turns = c.get('turns') or []
            if not turns or not turns[0].get('ddx') and len(turns) < 2:
                continue
            acc = (c.get('scores') or {}).get('accuracy')
            if acc is None:
                continue
            grp = "right" if acc >= 1 else "wrong"
            stats[grp] += 1
            # baseline: gold in the model's turn-1 DDx vs. union with initial_differential
            t1 = [x['dx'] for x in turns[0].get('ddx') or []]
            ini = [x['dx'] for x in initial_differential(c.get('initial', ''))]
            n_all += 1
            base_t1 += gold_in(t1, c['answer']) if t1 else 0
            union_t1 += gold_in(t1 + ini, c['answer'])
            # the check runs before choosing action k+1 (k turns done), never before the final DIAGNOSE decision point
            last = len(turns) - 1
            if turns[-1]['type'] == 'DIAGNOSE':
                last = len(turns) - 1  # decision for turn index `last` is the DIAGNOSE: still a valid (last) check point
            fired = None
            for k in range(A.MIN_TURNS, last + 1):
                st = build_state(c, k)
                r = anchoring_check(st)
                if r:
                    fired = (k, r)
                    break
            if fired:
                k, r = fired
                stats[grp + "_fired"] += 1
                fire_turn[grp].append(k)
                for x in r['reasons']:
                    reasons[grp][x] += 1
                top_wrong = not (same_dx(r['dx'], c['answer']) or (k3(c['answer']) and k3(r['dx']) == k3(c['answer'])))
                stats[grp + "_fired_topwrong"] += top_wrong
                live = [x['dx'] for x in c['turns'][k].get('ddx') or []]
                stats[grp + "_fired_gold_in_ddx"] += gold_in(live, c['answer'])
                if len(examples) < 400:
                    examples.append((grp, c['case'], k, len(turns), r['dx'], c['answer'], c['diagnosis'], r['reasons'],
                                     top_wrong))
    for g in ("right", "wrong"):
        n, fz = stats[g], stats[g + "_fired"]
        print(f"{g}: n={n} fired={fz} ({fz / max(n, 1):.1%}) top_at_fire_wrong={stats[g + '_fired_topwrong']} "
              f"gold_in_ddx_at_fire={stats[g + '_fired_gold_in_ddx']} reasons={dict(reasons[g])} "
              f"median_turn={sorted(fire_turn[g])[len(fire_turn[g]) // 2] if fire_turn[g] else None}")
    print(f"turn-1 DDx gold: model {base_t1}/{n_all} ({base_t1 / max(n_all, 1):.1%}); model+initial_differential {union_t1}/{n_all} "
          f"({union_t1 / max(n_all, 1):.1%})")
    if '-v' in sys.argv:
        for e in examples:
            print(e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
