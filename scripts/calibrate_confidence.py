"""Calibrate the code-computed confidence (src/doctor_agent/agent/confidence.py) on local result files. Dev only;
this offline code is not part of the agent package (moved out of agent/confidence.py on 2026-09-30).

    python scripts/calibrate_confidence.py eval/results/run_*.json            # print the report
    python scripts/calibrate_confidence.py --write eval/results/run_*.json    # also write data/labels/confidence_params.json

The written file is not shipped; to use it at runtime set AGENT_CONFIDENCE_PARAMS=data/labels/confidence_params.json,
or copy the rounded params into DEFAULT_PARAMS.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.agent.confidence import (  # noqa: E402
    FEATURES,
    MISSING,
    PARAMS_PATH,
    ConfidenceParams,
    _live,
    _sigmoid,
    decide,
    features,
    score_of,
)
from doctor_agent.agent.text import same_dx  # noqa: E402

# the first hand-set weights (before any data): the L2 prior of calibrate(), so refits do not drift towards themselves
HAND_SET = ConfidenceParams(-1.0, {
    "margin": 2.0, "verified_support": 1.0, "contradictions": -1.5, "confirmatory_test": 1.0, "criteria_met": 0.5,
    "dangers_unresolved": -1.0, "kb_agreement": 0.5, "turns_used": 0.5}, 0.75, 0.45, "hand-set")


def state_from_result(case: dict, upto: int | None = None):
    """Rebuild a CaseState from a saved case record. `upto` = number of completed turns (default: all but the final
    DIAGNOSE). The DDx ledger is the snapshot saved with action `upto` (what the model believed after those turns);
    findings are those recorded at or before `upto`."""
    from doctor_agent.agent.ledger import Finding
    from doctor_agent.agent.state import CaseState, Turn
    from doctor_agent.env.interface import Action, ActionType

    turns = case.get("turns") or []
    n = len(turns) - (1 if turns and turns[-1].get("type") == "DIAGNOSE" else 0)
    k = n if upto is None else max(0, min(upto, n))
    st = CaseState(case.get("initial", "") or "")
    for t in turns[:k]:
        try:
            typ = ActionType(t.get("type", "ASK"))
        except ValueError:
            typ = ActionType.ASK
        st.turns.append(Turn(Action(typ, t.get("content", ""), t.get("reason", "")), t.get("response", ""),
                             t.get("ddx") or []))
    snap = turns[k].get("ddx") if k < len(turns) and turns[k].get("ddx") else (case.get("ddx") or [])
    st.ddx_ledger.update(snap)
    st.ddx = st.ddx_ledger.as_list()
    for f in case.get("findings") or []:
        if int(f.get("turn", 0) or 0) <= k and str(f.get("item", "")).strip():
            st.findings.items.append(Finding(f["item"], f.get("status", "양성"), f.get("detail", ""), f.get("turn", 0)))
    return st


def same_disease(a: str, b: str) -> bool:
    """Label proxy for intermediate decision points (dev only): same_dx, one core name contained in the other
    ("결핵" / "폐결핵"), or the same KB entry ("긴장성 두통" / "긴장형 두통")."""
    from doctor_agent.agent.text import dx_keys
    if not a or not b:
        return False
    if same_dx(a, b):
        return True
    ca, cb = dx_keys(a)[0], dx_keys(b)[0]
    if min(len(ca), len(cb)) >= 2 and (ca in cb or cb in ca):
        return True
    try:
        from doctor_agent.knowledge import kb
        x, y = (kb.lookup(a), kb.lookup(b)) if kb.available() else (None, None)
        return bool(x and y and x["id"] == y["id"])
    except Exception:  # noqa: BLE001
        return False


def _correct(case: dict) -> float:
    return float((case.get("scores") or {}).get("accuracy") or 0.0)


def samples(paths, intermediate: bool = True, target_turns: int = 20) -> list[dict]:
    """One sample per (case, decision point): final = the state at the DIAGNOSE action with the final diagnosis
    (label = judged accuracy == 1); intermediate = earlier decision points with the top live DDx as the proposal
    (label proxy, `same_disease`: the top DDx names the answer, or the final diagnosis when that was judged correct)."""
    out = []
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for ci, case in enumerate(data.get("cases") or []):
            turns = case.get("turns") or []
            if not turns:
                continue
            n = len(turns) - (1 if turns[-1].get("type") == "DIAGNOSE" else 0)
            acc = _correct(case)
            answer, final = case.get("answer", ""), case.get("diagnosis", "")
            cid = f"{Path(path).stem}/{case.get('case', ci)}"
            points = range(0, n + 1) if intermediate else [n]
            for k in points:
                st = state_from_result(case, k)
                if k == n:
                    dx, label = final, float(acc >= 1.0)
                else:
                    live = _live(st)
                    if not live:
                        continue
                    dx = live[0].dx
                    label = float(same_disease(dx, answer) or (acc >= 1.0 and same_disease(dx, final)))
                comp, det = features(st, dx, target_turns)
                out.append({"case": cid, "run": Path(path).stem, "k": k, "n": n, "final": k == n, "dx": dx,
                            "answer": answer, "final_dx": final,
                            "label": label, "acc": acc, "comp": comp, "dangers": det.get("dangers", []),
                            "gate_turns": 0})
    return out


def _x(s: dict) -> list[float]:
    return [1.0] + [MISSING.get(k, 0.0) if s["comp"].get(k) is None else s["comp"][k] for k in FEATURES]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


# Sign of each weight that makes clinical sense; calibrate() drops a weight to 0 rather than flip it. turns_used is a
# budget knob, not evidence (harder cases take longer, so fitting it would learn "long case = wrong"): kept fixed.
SIGNS = {"contradictions": -1, "dangers_unresolved": -1}
FIXED = ("turns_used",)


def fit_logistic(xs: list[list[float]], ys: list[float], l2: float = 2.0, iters: int = 25,
                 prior: list[float] | None = None, free: list[int] | None = None) -> list[float]:
    """L2-regularised logistic regression by Newton steps (stdlib). Column 0 is the bias (not penalised); the penalty
    pulls the other weights towards `prior` (the hand-set defaults), so few labels move them only a little. Only the
    columns in `free` are fitted, the rest keep their `prior` value. Classes are balanced by sample weights."""
    d = len(xs[0])
    w = list(prior) if prior else [0.0] * d
    pr = list(w)
    idx = list(range(d)) if free is None else sorted(set(free) | {0})
    npos = sum(ys) or 1.0
    nneg = (len(ys) - sum(ys)) or 1.0
    sw = [len(ys) / (2 * npos) if y > 0.5 else len(ys) / (2 * nneg) for y in ys]
    for _ in range(iters):
        m = len(idx)
        g = [0.0] * m
        h = [[0.0] * m for _ in range(m)]
        for x, y, s in zip(xs, ys, sw):
            p = _sigmoid(sum(a * b for a, b in zip(w, x)))
            for a, i in enumerate(idx):
                g[a] += s * (p - y) * x[i]
                for b, j in enumerate(idx):
                    h[a][b] += s * p * (1 - p) * x[i] * x[j]
        for a, i in enumerate(idx):
            if i:
                g[a] += l2 * (w[i] - pr[i])
                h[a][a] += l2
        step = _solve(h, g)
        for a, i in enumerate(idx):
            w[i] -= step[a]
        if max(abs(v) for v in step) < 1e-6:
            break
    return w


def fit_constrained(xs: list[list[float]], ys: list[float], l2: float = 2.0) -> list[float]:
    """fit_logistic with FIXED features held at their defaults and SIGNS enforced (a weight that would take the
    wrong sign is set to 0 and the rest refitted)."""
    prior = [HAND_SET.bias] + [HAND_SET.weights[k] for k in FEATURES]
    free = [i + 1 for i, k in enumerate(FEATURES) if k not in FIXED]
    while True:
        w = fit_logistic(xs, ys, l2, prior=prior, free=free)
        bad = [i for i in free if w[i] * SIGNS.get(FEATURES[i - 1], 1) < 0]
        if not bad:
            return w
        for i in bad:
            prior[i] = 0.0
            free.remove(i)


def auc(scores: list[float], labels: list[float]) -> float | None:
    """Mann-Whitney AUC (ties count half); None when one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def simulate(smp: list[dict], params: ConfidenceParams, max_turns: int = 60, target_turns: int = 20) -> dict:
    """Replay each case's decision points with the stop rule: stop at the first 'diagnose' (correct = the sample's
    label, turns = k + 1), else keep the actual outcome (the replay cannot see what extra turns would have found).
    Returns accuracy / mean turns vs. the actual run, and how often the rule would still say 'continue' at the
    point where the agent actually diagnosed (and how many of those diagnoses were wrong)."""
    by_case: dict[str, list[dict]] = {}
    for s in smp:
        by_case.setdefault(s["case"], []).append(s)
    acc_sim = acc_act = t_sim = t_act = 0.0
    n = earlier = cont_end = cont_wrong = 0
    for pts in by_case.values():
        pts.sort(key=lambda s: s["k"])
        fin = pts[-1]
        if not fin["final"]:
            continue
        n += 1
        acc_act += fin["label"]
        t_act += fin["n"] + 1
        stop = None
        for s in pts:
            rec, _ = decide(score_of(s["comp"], params), s["k"], max_turns, target_turns, s["dangers"], params)
            if rec == "diagnose":
                stop = s
                break
        if stop is not None and not stop["final"]:
            earlier += 1
            acc_sim += stop["label"]
            t_sim += stop["k"] + 1
        else:
            acc_sim += fin["label"]
            t_sim += fin["n"] + 1
            if stop is None:
                cont_end += 1
                cont_wrong += fin["label"] < 0.5
    n = max(1, n)
    return {"cases": n, "acc_actual": round(acc_act / n, 4), "acc_sim": round(acc_sim / n, 4),
            "turns_actual": round(t_act / n, 3), "turns_sim": round(t_sim / n, 3), "stopped_earlier": earlier,
            "would_continue_at_end": cont_end, "would_continue_and_wrong": cont_wrong}


THETA_GRID = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
THETA_GAP = 0.2  # theta_low = theta_high - THETA_GAP


def choose_thresholds(smp: list[dict], params: ConfidenceParams, tolerance: float = 0.005) -> tuple[float, float]:
    """Lowest theta_high on a coarse grid whose replay keeps accuracy within `tolerance` of the actual runs
    (matched accuracy); theta_low = theta_high - THETA_GAP. Coarse grid + fixed gap = little to overfit."""
    for th in THETA_GRID:
        q = ConfidenceParams(params.bias, params.weights, th, round(th - THETA_GAP, 2))
        r = simulate(smp, q)
        if r["acc_sim"] >= r["acc_actual"] - tolerance:
            return th, round(th - THETA_GAP, 2)
    return THETA_GRID[-1], round(THETA_GRID[-1] - THETA_GAP, 2)


def _cv_replay(smp: list[dict], l2: float, tolerance: float) -> dict:
    """Leave-one-run-out for the whole pipeline (weights + thresholds fitted without the run, then its replay)."""
    agg = {"cases": 0, "acc_actual": 0.0, "acc_sim": 0.0, "turns_actual": 0.0, "turns_sim": 0.0, "stopped_earlier": 0,
           "would_continue_at_end": 0, "would_continue_and_wrong": 0}
    for r in sorted({s["run"] for s in smp}):
        tr = [s for s in smp if s["run"] != r]
        te = [s for s in smp if s["run"] == r]
        if len({s["label"] for s in tr}) < 2 or not any(s["final"] for s in te):
            continue
        w = fit_constrained([_x(s) for s in tr], [s["label"] for s in tr], l2)
        p = ConfidenceParams(w[0], dict(zip(FEATURES, w[1:])))
        p.theta_high, p.theta_low = choose_thresholds(tr, p, tolerance)
        res = simulate(te, p)
        for k in agg:  # accuracy / turns are means: weight by the run's case count
            agg[k] += res[k] * res["cases"] if k.startswith(("acc", "turns")) else res[k]
    n = max(1, agg["cases"])
    for k in ("acc_actual", "acc_sim", "turns_actual", "turns_sim"):
        agg[k] = round(agg[k] / n, 4)
    return agg


def calibrate(paths, l2: float = 2.0, tolerance: float = 0.005, smp: list[dict] | None = None) -> dict:
    """Fit bias/weights (all decision points, class-balanced, L2 towards the hand-set prior, sign-constrained) and the
    stop thresholds (matched accuracy) on local result files. Leave-one-run-out AUC / replay are the honest estimates.
    Returns the report ({"params": ..., metrics}); main() below writes it (shipped code never
    writes files)."""
    smp = smp if smp is not None else samples(paths)
    if not smp:
        return {}
    xs, ys = [_x(s) for s in smp], [s["label"] for s in smp]
    w = fit_constrained(xs, ys, l2)
    cv_s, cv_y, cvf_s, cvf_y = [], [], [], []
    for r in sorted({s["run"] for s in smp}):  # leave one run out
        tr = [i for i, s in enumerate(smp) if s["run"] != r]
        te = [i for i, s in enumerate(smp) if s["run"] == r]
        if not tr or not te or len({ys[i] for i in tr}) < 2:
            continue
        wr = fit_constrained([xs[i] for i in tr], [ys[i] for i in tr], l2)
        for i in te:
            sc = _sigmoid(sum(a * b for a, b in zip(wr, xs[i])))
            cv_s.append(sc)
            cv_y.append(ys[i])
            if smp[i]["final"]:
                cvf_s.append(sc)
                cvf_y.append(ys[i])
    fitted = ConfidenceParams(round(w[0], 4), {k: round(v, 4) for k, v in zip(FEATURES, w[1:])}, source="calibrate")
    fitted.theta_high, fitted.theta_low = choose_thresholds(smp, fitted, tolerance)
    hand_th = ConfidenceParams(HAND_SET.bias, HAND_SET.weights, source="hand-set+matched")
    hand_th.theta_high, hand_th.theta_low = choose_thresholds(smp, hand_th, tolerance)
    fin = [i for i, s in enumerate(smp) if s["final"]]
    allidx = list(range(len(smp)))

    def pauc(ps: ConfidenceParams, idx: list[int]) -> float | None:
        return auc([score_of(smp[i]["comp"], ps) for i in idx], [ys[i] for i in idx])

    report = {
        "params": fitted.to_dict(),
        "n_samples": len(smp), "n_positive": int(sum(ys)), "n_final": len(fin),
        "n_final_positive": int(sum(ys[i] for i in fin)),
        "auc_hand_set_all": pauc(HAND_SET, allidx), "auc_hand_set_final": pauc(HAND_SET, fin),
        "auc_fitted_all_insample": pauc(fitted, allidx), "auc_fitted_final_insample": pauc(fitted, fin),
        "auc_cv_all": auc(cv_s, cv_y), "auc_cv_final": auc(cvf_s, cvf_y),
        "feature_auc_final": {k: auc([_x(smp[i])[j + 1] for i in fin], [ys[i] for i in fin])
                              for j, k in enumerate(FEATURES)},
        "simulate_hand_set": simulate(smp, HAND_SET),
        "simulate_hand_set_matched": {"theta_high": hand_th.theta_high, **simulate(smp, hand_th)},
        "simulate_fitted_insample": simulate(smp, fitted),
        "simulate_cv": _cv_replay(smp, l2, tolerance),
        "method": f"class-balanced logistic regression, L2={l2} towards the hand-set defaults, sign-constrained, "
                  f"turns_used fixed; theta_high = lowest of {THETA_GRID} keeping replay accuracy within {tolerance} "
                  f"of the actual runs, theta_low = theta_high - {THETA_GAP}; leave-one-run-out CV AUC",
        "labels": "final decision point: judged accuracy == 1; earlier points: top live DDx names the answer (or the "
                  "final diagnosis when judged correct) by same_disease (a proxy)",
        "files": [Path(p).name for p in paths],
    }
    return report


def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    report = calibrate(paths)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if "--write" in argv:
        PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PARAMS_PATH.write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
