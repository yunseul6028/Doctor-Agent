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

The offline calibration (refit of bias/weights by L2-regularised logistic regression and of the two thresholds on
past local result files eval/results/run_*.json) lives in scripts/calibrate_confidence.py, which can write
data/labels/confidence_params.json (dev only, not shipped: the agent uses DEFAULT_PARAMS below unless
AGENT_CONFIDENCE_PARAMS points to such a JSON file).
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
    # Weights act on features in [0, 1]. Values: scripts/calibrate_confidence.py on 11 local runs (226 cases, 1446
    # decision points, two earlier dev doctor models (Gemini Flash / Gemma), 2026-09-28), rounded; prior = its
    # HAND_SET.
    # dangers_unresolved and kb_agreement fitted to 0 (dangers are handled by the must_continue rule instead; KB rank did not separate right from wrong). Re-validate
    # on runs of the current doctor model (Gemini Pro). See docs/architecture.md "Confidence and stop rule".
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


def load_params(path: str | os.PathLike | None = None) -> ConfidenceParams:
    """Parameters from `path` or $AGENT_CONFIDENCE_PARAMS (JSON written by scripts/calibrate_confidence.py);
    defaults on any problem."""
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

