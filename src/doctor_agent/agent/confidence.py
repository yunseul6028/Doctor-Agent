"""Code-computed diagnostic confidence and stop rule.

The LLM's own "confidence" field is not trusted as a stopping signal. `assess(state, proposed_dx)` computes a score
from transparent case features (no learned model beyond a logistic link with a handful of weights), then a stop
recommendation:

    score = sigmoid(bias + sum_i w_i * x_i)          every x_i in [0, 1]; a missing feature uses MISSING[i]

    x_i                 what it reads (this case only)
    margin              p(proposed) - p(best other live DDx) from the DDx ledger, clipped to [0, 1]
    verified_support    ledger "for" items of the proposed dx that the grounding checker finds in the environment's
                        text, /3 (capped at 1)
    contradictions      grounded "against" items of the proposed dx, /2 (capped)            (negative weight)
    confirmatory_test   1 when a grounded support item comes from a TEST/EXAM response, the can't-miss table marks
                        the proposed dx confirmed, or a criteria set decides this diagnosis/subtype; else 0
    criteria_met        published criteria (knowledge/diagnostic_criteria): 1 = decision matches, else met share of
                        the set x 0.6, 0 = the findings pick a conflicting subtype; None when no set applies
    dangers_unresolved  can't-miss dangers (safety/danger_gate) other than the proposed dx that are unresolved and
                        still have a rule-out action, /2 (capped); weight 0 after calibration, acts through the
                        must_continue rule instead
    kb_agreement        1/rank of the proposed dx among KB candidates for the positive findings (0 = not in top 10);
                        None when the KB is missing or does not know the name; weight 0 after calibration (reported)
    turns_used          turn_count / target_turns (capped at 1)

    recommendation      "diagnose" | "continue" | "must_continue"
      - remaining turns <= 1                                   → diagnose (turn cap)
      - actionable unresolved danger, remaining > min_remaining,
        gate budget left                                        → must_continue
      - score >= theta_high                                     → diagnose
      - score <  theta_low                                      → continue
      - in between                                              → diagnose once turn_count >= target_turns

Everything is computed from the given CaseState (initial info + this case's responses + its ledgers); nothing is kept
between calls or cases. CPU only, stdlib only, no LLM calls, never raises (a failing feature is reported as missing).

`calibrate(paths)` refits bias/weights (L2-regularised logistic regression, Newton steps, stdlib) and the two
thresholds on past local result files (eval/results/run_*.json) and returns a report; scripts/calibrate_confidence.py
writes it to data/labels/confidence_params.json (dev only, not shipped: the submission uses DEFAULT_PARAMS below
unless AGENT_CONFIDENCE_PARAMS points to such a JSON file).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from doctor_agent.agent.text import same_dx

FEATURES = ("margin", "verified_support", "contradictions", "confirmatory_test", "criteria_met", "dangers_unresolved",
            "kb_agreement", "turns_used")
# value used for a feature that could not be computed (no criteria set, KB does not know the name, ...)
MISSING = {"criteria_met": 0.5, "kb_agreement": 0.3}
PARAMS_PATH = Path(__file__).resolve().parents[3] / "data" / "labels" / "confidence_params.json"
KB_TOP_K = 10
MIN_REMAINING = 2  # same as danger_gate.gate: with <= 2 turns left a diagnosis always goes ahead


@dataclass
class ConfidenceParams:
    # Weights act on features in [0, 1]. Values: calibrate() on 11 local runs (226 cases, 1446 decision points,
    # two non-competition dev doctor models, 2026-09-28), rounded; prior = HAND_SET below. dangers_unresolved and kb_agreement fitted to 0
    # (dangers are handled by the must_continue rule instead; KB rank did not separate right from wrong). Re-validate
    # after the switch to gpt-oss-20b. See docs/architecture.md "Confidence and stop rule".
    bias: float = -2.6
    weights: dict = field(default_factory=lambda: {
        "margin": 4.3, "verified_support": 0.45, "contradictions": -0.85, "confirmatory_test": 0.6,
        "criteria_met": 2.0, "dangers_unresolved": 0.0, "kb_agreement": 0.0, "turns_used": 0.5})
    theta_high: float = 0.85
    theta_low: float = 0.65
    source: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceParams":
        """Accepts the params dict or a whole calibrate() report ({"params": {...}, ...})."""
        d = d.get("params", d) if isinstance(d.get("params"), dict) else d
        base = cls()
        w = dict(base.weights)
        w.update({k: float(v) for k, v in (d.get("weights") or {}).items() if k in FEATURES})
        return cls(float(d.get("bias", base.bias)), w, float(d.get("theta_high", base.theta_high)),
                   float(d.get("theta_low", base.theta_low)), str(d.get("source", "file")))


DEFAULT_PARAMS = ConfidenceParams()
# the first hand-set weights (before any data): the L2 prior of calibrate(), so refits do not drift towards themselves
HAND_SET = ConfidenceParams(-1.0, {
    "margin": 2.0, "verified_support": 1.0, "contradictions": -1.5, "confirmatory_test": 1.0, "criteria_met": 0.5,
    "dangers_unresolved": -1.0, "kb_agreement": 0.5, "turns_used": 0.5}, 0.75, 0.45, "hand-set")


def load_params(path: str | os.PathLike | None = None) -> ConfidenceParams:
    """Parameters from `path` or $AGENT_CONFIDENCE_PARAMS (JSON written by calibrate); defaults on any problem."""
    p = path or os.getenv("AGENT_CONFIDENCE_PARAMS", "")
    if not p:
        return DEFAULT_PARAMS
    try:
        return ConfidenceParams.from_dict(json.loads(Path(p).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return DEFAULT_PARAMS


@dataclass
class Assessment:
    score: float
    components: dict  # feature -> value in [0, 1] or None (missing)
    recommendation: str  # diagnose | continue | must_continue
    reasons_ko: list[str]
    proposed: str = ""
    dangers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _live(state) -> list:
    return [e for e in state.ddx_ledger.ranked() if e.status != "배제"]


def _entry(state, dx: str):
    """Ledger entry for the proposed name: same_dx, else the live entry whose core name contains / is contained in it
    or is most similar (>= 0.5) — the final name is often a refined form of a ledger entry ("담도 폐쇄 (악성 종양 등)" →
    "담도 폐쇄성 악성 종양")."""
    from doctor_agent.agent.text import dx_keys, similarity
    entries = state.ddx_ledger.entries
    if (e := next((e for e in entries if same_dx(e.dx, dx)), None)) is not None:
        return e
    core = dx_keys(dx)[0]
    best, best_sim = None, 0.5
    for e in _live(state):
        c = dx_keys(e.dx)[0]
        if min(len(c), len(core)) >= 2 and (c in core or core in c):
            return e
        if (s := similarity(c, core)) >= best_sim:
            best, best_sim = e, s
    return best


def _margin(state, dx: str) -> float:
    live = _live(state)
    e = _entry(state, dx)
    p = e.p if e is not None and e.status != "배제" else 0.0
    other = max((x.p for x in live if x is not e), default=0.0)
    return round(_clip(p - other), 4)


def _evidence(state):
    from doctor_agent.agent import grounding
    return grounding._state_evidence(state)


def _grounded(items: list[str], ev) -> list[str]:
    from doctor_agent.agent import grounding
    out = []
    for it in items:
        try:
            if grounding.is_grounded(it, ev)[0]:
                out.append(it)
        except Exception:  # noqa: BLE001
            continue
    return out


def _test_text(state) -> str:
    return "\n".join(t.response for t in state.turns
                     if getattr(t.action.type, "value", t.action.type) in ("TEST", "EXAM")
                     and "제공되지 않" not in t.response)


def _case_text(state) -> str:
    """Environment facts + verified/unchecked findings (same text the pre-diagnosis review gives the criteria)."""
    return "\n".join([state.initial_info] + [t.response for t in state.turns]
                     + [state.findings.render(exclude_unverified=True)])


def _criteria(dx: str, text: str) -> tuple[float | None, bool]:
    """(criteria_met value or None, decision confirms dx)."""
    from doctor_agent.knowledge import diagnostic_criteria as dc
    best: float | None = None
    for c in dc.criteria_for(dx, related=False)[:2]:
        r = dc.evaluate(c.id, text)
        if r.decision and dc.decision_matches(c, r.decision, dx):
            return 1.0, True
        if r.decision and dc.conflicting_subtype(c, r.decision, dx):
            return 0.0, False
        n = len(r.met) + len(r.not_met) + len(r.unknown)
        v = 0.6 * len(r.met) / n if n else 0.0
        best = v if best is None else max(best, v)
    return best, False


def _kb_rank(state, dx: str, support: list[str]) -> float | None:
    from doctor_agent.knowledge import kb
    if not kb.available():
        return None
    target = kb.lookup(dx)
    if not target:
        return None
    pos = [f.item for f in state.findings.items if f.status == "양성" and f.verified is not False]
    neg = [f.item for f in state.findings.items if f.status == "음성" and f.verified is not False]
    if not pos:  # no findings ledger (older states): the grounded evidence the model cited for any live DDx
        pos = list(dict.fromkeys(s for e in _live(state) for s in e.support)) or support
    if not pos:
        return 0.0
    sex, age = kb.patient_profile(state.initial_info or "")
    cands = kb.candidates(pos, k=KB_TOP_K, negatives=neg, sex=sex or None, age=age)
    for r, c in enumerate(cands, 1):
        if c["id"] == target["id"]:
            return 1.0 / r
    return 0.0


def _dangers(state, dx: str) -> list[str]:
    from doctor_agent.safety import danger_gate
    out = []
    for d in danger_gate.unresolved_dangers(state, exclude=dx or None):
        if d["status"] == "unresolved" and danger_gate.next_rule_out_action(d, state) is not None:
            out.append(d["dx"])
    return out


def _confirmed_danger(state, dx: str) -> bool:
    from doctor_agent.safety import danger_gate
    return danger_gate.status(dx, state)[0] == "confirmed"


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return default


def features(state, proposed_dx: str | None, target_turns: int = 20) -> tuple[dict, dict]:
    """(components, details). Components are in [0, 1] or None; details hold the evidence behind them."""
    dx = (proposed_dx or "").strip()
    if not dx:
        live = _live(state)
        dx = live[0].dx if live else ""
    comp: dict = {k: None for k in FEATURES}
    det: dict = {"proposed": dx, "support": [], "against": [], "dangers": [], "confirm": ""}
    comp["turns_used"] = _clip(state.turn_count / max(1, target_turns))
    if not dx:
        return comp, det
    comp["margin"] = _safe(_margin, state, dx, default=0.0)
    e = _entry(state, dx)
    ev = _safe(_evidence, state)
    sup = list(e.support) if e else []
    ag = list(e.against) if e else []
    if ev is not None:
        sup, ag = _grounded(sup, ev), _grounded(ag, ev)
    else:  # grounding unavailable: count items at half value
        sup, ag = sup[: len(sup) // 2], ag[: len(ag) // 2]
    det["support"], det["against"] = sup, ag
    comp["verified_support"] = _clip(len(sup) / 3)
    comp["contradictions"] = _clip(len(ag) / 2)
    crit, decided = _safe(_criteria, dx, _safe(_case_text, state, default=state.initial_info), default=(None, False))
    comp["criteria_met"] = crit
    confirm = ""
    if decided:
        confirm = "진단 기준 충족"
    elif _safe(_confirmed_danger, state, dx, default=False):
        confirm = "위험 질환 확진 소견"
    elif sup and (tt := _safe(_test_text, state, default="")):
        hit = _grounded(sup, tt)
        confirm = f"검사 결과: {hit[0]}" if hit else ""
    det["confirm"] = confirm
    comp["confirmatory_test"] = 1.0 if confirm else 0.0
    dangers = _safe(_dangers, state, dx, default=[]) or []
    det["dangers"] = dangers
    comp["dangers_unresolved"] = _clip(len(dangers) / 2)
    comp["kb_agreement"] = _safe(_kb_rank, state, dx, sup, default=None)
    return comp, det


def score_of(comp: dict, params: ConfidenceParams = DEFAULT_PARAMS) -> float:
    z = params.bias
    for k in FEATURES:
        v = comp.get(k)
        z += params.weights.get(k, 0.0) * (MISSING.get(k, 0.0) if v is None else v)
    return _sigmoid(z)


def decide(score: float, turn_count: int, max_turns: int, target_turns: int, dangers: list[str],
           params: ConfidenceParams = DEFAULT_PARAMS, gate_left: bool = True, has_dx: bool = True) -> tuple[str, str]:
    """(recommendation, Korean reason) from the score and the turn budget."""
    remaining = max_turns - turn_count
    if remaining <= 1:
        return "diagnose", f"남은 턴 {remaining}개: 지금 진단"
    if dangers and remaining > MIN_REMAINING and gate_left:
        return "must_continue", "배제 안 된 위험 질환: " + ", ".join(dangers[:3])
    if not has_dx:
        return "continue", "제안 진단 없음"
    if score >= params.theta_high:
        return "diagnose", f"확신도 {score:.2f} ≥ {params.theta_high:.2f}"
    if score < params.theta_low:
        return "continue", f"확신도 {score:.2f} < {params.theta_low:.2f}"
    if turn_count >= target_turns:
        return "diagnose", f"확신도 {score:.2f} 중간, 목표 턴({target_turns}) 도달"
    return "continue", f"확신도 {score:.2f} 중간, 목표 턴({target_turns}) 전"


def assess(state, proposed_dx: str | None = None, cfg=None, params: ConfidenceParams | None = None) -> Assessment:
    """Confidence in `proposed_dx` (default: top live DDx) for this case state, and a stop recommendation.
    `cfg` is an AgentConfig (max_turns, target_turns, max_gate_turns); defaults are used when None. Never raises."""
    params = params or load_params()
    max_turns = getattr(cfg, "max_turns", 60)
    target = getattr(cfg, "target_turns", 20)
    gate_left = getattr(state, "gate_turns", 0) < getattr(cfg, "max_gate_turns", 3)
    try:
        comp, det = features(state, proposed_dx, target)
    except Exception as e:  # noqa: BLE001
        comp, det = {k: None for k in FEATURES}, {"proposed": proposed_dx or "", "dangers": [], "error": str(e)[:200]}
    score = score_of(comp, params)
    rec, why = decide(score, state.turn_count, max_turns, target, det.get("dangers", []), params, gate_left,
                      has_dx=bool(det.get("proposed")))
    reasons = [why]
    if det.get("proposed"):
        reasons.append(f"감별 우위 {comp['margin'] or 0:.2f}")
        reasons.append(f"확인된 지지 소견 {len(det.get('support', []))}개"
                       + (f" ({', '.join(det['support'][:2])})" if det.get("support") else ""))
        if det.get("against"):
            reasons.append(f"반대 소견 {len(det['against'])}개 ({', '.join(det['against'][:2])})")
        reasons.append(f"확진 근거: {det['confirm']}" if det.get("confirm") else "확진 근거 없음")
        if comp.get("criteria_met") is not None:
            reasons.append(f"진단 기준 {comp['criteria_met']:.2f}")
        if comp.get("kb_agreement") is not None:
            reasons.append(f"지식베이스 일치 {comp['kb_agreement']:.2f}")
    return Assessment(round(score, 4), comp, rec, reasons, det.get("proposed", ""), list(det.get("dangers", [])))


# ---------------------------------------------------------------------------------------------------------------
# Offline calibration on local result files (dev only)
# ---------------------------------------------------------------------------------------------------------------

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
    Returns the report ({"params": ..., metrics}); writing it is left to scripts/calibrate_confidence.py (shipped code
    never writes files)."""
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
                  "final diagnosis when judged correct) by confidence.same_disease (a proxy)",
        "files": [Path(p).name for p in paths],
    }
    return report
