"""Offline evaluation of agent/question_planner.py on data/cases_aug (no LLM).

For each case and N in NS: partial state = initial info + first N history answers; DDx ledger seeded from KB candidates
(variant "kb") or KB candidates with the gold diagnosis inserted at rank 2 (variant "kb+gold"). Then top-3 suggestions
are matched against the case file (whitespace-insensitive version of eval/simulator.py keyword matching).

Metrics per variant:
  inf@3   share of states whose top-3 include an action that reveals a not-yet-revealed, non-normal case entry
  inf/3   mean number of such actions among the 3
  ans@3   mean number of actions the case file has any entry for
  dec@3   share of states (cases with a decisive entry) whose top-3 include a decisive test: a case entry whose result
          kb_tests.detect() reads abnormal for a curated result linked to the gold diagnosis with weight >= 2 (w3: = 3)
Baselines: best fixed triple of common exams/tests (chosen on this data: optimistic), and "top-1 confirm" (the top KB
DDx's own strongest features, no information gain).
"""
import glob
import json
import statistics
import sys
import time
from itertools import combinations

from doctor_agent.agent import question_planner as qp
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge import kb, kb_tests
from doctor_agent.nlp import parse

NS = (0, 2, 4)
GENERIC_TOK = {"증상", "주호소", "질환", "검사", "진찰"}
K = kb.get_kb()


def norm(s):
    return "".join((s or "").lower().split())


def match_keys(case, typ, content):
    table = {"ASK": case.get("history", {}), "EXAM": case.get("exam", {}), "TEST": case.get("tests", {})}[typ]
    c = norm(content)
    out = []
    for key in table:
        toks = [norm(t) for t in key.split("|")]
        if any(len(t) >= 2 and t not in GENERIC_TOK and t in c for t in toks):
            out.append(key)
    return out


def abnormal(value, typ):
    if any(p > 0 for p, _ in kb_tests.detect(value).values()):
        return True
    src = {"ASK": "patient", "EXAM": "exam", "TEST": "test"}[typ]
    return any(f.polarity == "present" and f.subject == "patient" and not f.hypothetical and f.direction != "normal"
               for f in parse(value, src))


def decisive_keys(case, min_w):
    p = kb.lookup(case["diagnosis"])
    if not p:
        return set(), False
    tfs = {x["id"][3:] for x in p["findings_from_tests"] if x["weight"] >= min_w}
    out = set()
    for typ in ("TEST", "EXAM"):
        for key, v in case.get({"TEST": "tests", "EXAM": "exam"}[typ], {}).items():
            if any(fid in tfs and pol > 0 for fid, (pol, _) in kb_tests.detect(v).items()):
                out.add((typ, key))
    return out, True


def build_state(case, n, variant):
    s = CaseState(initial_info=case["initial"])
    items = list(case["history"].items())
    revealed = set()
    for key, v in items:
        if key.startswith("주호소"):
            revealed.add(("ASK", key))
    for key, v in items[:n]:
        s.turns.append(Turn(Action(ActionType.ASK, key.split("|")[0]), v))
        revealed.add(("ASK", key))
    sex, age = kb.patient_profile(case["initial"])
    cands = K.candidates([case["initial"]] + [v for _, v in items[:n]], k=4, sex=sex or None, age=age)
    tot = sum(max(c["score"], 0) for c in cands) or 1
    ddx = [(c["name_ko"], max(c["score"], 0) / tot) for c in cands]
    if variant == "kb+gold":
        from doctor_agent.agent.text import same_dx
        gi = K._resolve(case["diagnosis"])
        if not any(same_dx(d, case["diagnosis"]) or (gi is not None and K._resolve(d) == gi) for d, _ in ddx):
            ddx = ddx[:1] + [(case["diagnosis"], 0.3)] + ddx[1:3]
        tot = sum(p for _, p in ddx) or 1
        ddx = [(d, p / tot) for d, p in ddx]
    s.ddx_ledger.update([{"dx": d, "p": p} for d, p in ddx])
    return s, revealed


def concept_hit(case, typ, feats, revealed):
    """A feature of the action is stated present in a not-yet-revealed entry of the matching section."""
    from doctor_agent.nlp.lexicon import LEXICON
    want = set()
    for t in feats:
        for c in LEXICON.by_kb(t):
            want.add(c)
            want.update(LEXICON.descendants(c))
    if not want:
        return False
    sec = {"ASK": "history", "EXAM": "exam", "TEST": "tests"}[typ]
    src = {"ASK": "patient", "EXAM": "exam", "TEST": "test"}[typ]
    for k, v in case[sec].items():
        if (typ, k) in revealed:
            continue
        if any(f.concept in want and f.polarity == "present" and f.subject == "patient" for f in parse(v, src)):
            return True
    return False


def score(case, acts, revealed, dec, concept=False):
    inf = ans = 0
    hit_dec = False
    for a in acts:
        typ, content = a[0], a[1]
        feats = a[2] if len(a) > 2 else []
        keys = match_keys(case, typ, content)
        if keys:
            ans += 1
        tbl = case[{"ASK": "history", "EXAM": "exam", "TEST": "tests"}[typ]]
        if any((typ, k) not in revealed and abnormal(tbl[k], typ) for k in keys) or (
                concept and concept_hit(case, typ, feats, revealed)):
            inf += 1
        if any((typ, k) in dec for k in keys):
            hit_dec = True
    return inf, ans, hit_dec


COMMON = [("EXAM", "활력징후 측정"), ("TEST", "일반혈액검사(CBC)"), ("TEST", "흉부 X선"), ("TEST", "기초 생화학(전해질, 신기능)"),
          ("TEST", "간기능 검사"), ("TEST", "CRP"), ("TEST", "소변검사"), ("TEST", "심전도"), ("EXAM", "신경학적 진찰"),
          ("EXAM", "복부 진찰"), ("EXAM", "폐 청진"), ("EXAM", "심장 청진")]


def top1_confirm(state):
    hyps = qp._hypotheses(K, state)
    if not hyps or hyps[0].idx is None:
        return []
    h = hyps[0]
    feats = sorted(list(h.tst.items()) + list(h.sym.items()), key=lambda x: (-x[1], x[0]))
    out = []
    for tid, _ in feats:
        if tid.startswith("TF:"):
            req, _t, _k, typ = qp.request_for_result(tid[3:], K.terms[tid]["ko"])
        else:
            r = qp._ask_text(K, tid)
            if not r:
                continue
            typ, req, _s = r
        if all((typ, req) != o[:2] for o in out) and not state.asked(Action(ActionType(typ), req)):
            out.append((typ, req, [tid]))
        if len(out) == 3:
            break
    return out


def main():
    files = sorted(glob.glob("data/cases_aug/*/*.json"))
    cases = [json.load(open(f, encoding="utf-8")) for f in files]
    lat = []
    res = {}
    resolvable = sum(K._resolve(c["diagnosis"]) is not None for c in cases)
    dec2 = {i: decisive_keys(c, 2)[0] for i, c in enumerate(cases)}
    dec3 = {i: decisive_keys(c, 3)[0] for i, c in enumerate(cases)}
    common_rows = []  # per state: which COMMON actions are informative
    examples = []
    for variant in ("kb", "kb+gold"):
        for n in NS:
            rows = {"planner": [], "top1": []}
            for i, case in enumerate(cases):
                s, revealed = build_state(case, n, variant)
                t = time.perf_counter()
                sug = qp.suggest(s, k=3, include_safety=False)
                lat.append((time.perf_counter() - t) * 1000)
                acts = [(x.type, x.content_ko, x.features) for x in sug]
                if variant == "kb+gold" and n == 2 and len(examples) < 6 and i % 40 == 0:
                    examples.append((case["diagnosis"], [e.dx for e in s.ddx_ledger.ranked()], acts))
                for name, a in (("planner", acts), ("top1", top1_confirm(s))):
                    inf, ans, _ = score(case, a, revealed, set())
                    infc = score(case, a, revealed, set(), concept=True)[0]
                    rows[name].append((inf, ans, score(case, a, revealed, dec2[i])[2] if dec2[i] else None,
                                       score(case, a, revealed, dec3[i])[2] if dec3[i] else None, infc))
                if variant == "kb":
                    common_rows.append((n, [score(case, [a], revealed, set()) for a in COMMON],
                                        [bool(score(case, [a], revealed, dec2[i])[2]) if dec2[i] else None
                                         for a in COMMON]))
            for name, r in rows.items():
                res[(variant, n, name)] = r
    # best fixed triple of common actions (per N, chosen on the data)
    for n in NS:
        cr = [(x, d) for m, x, d in common_rows if m == n]
        best = max(combinations(range(len(COMMON)), 3), key=lambda t: sum(any(x[j][0] for j in t) for x, _ in cr))
        r = [(sum(x[j][0] for j in best), sum(x[j][1] for j in best),
              (any(d[j] for j in best) if d[0] is not None else None), None, sum(x[j][0] for j in best))
             for x, d in cr]
        res[("fixed", n, "best-common:" + "+".join(COMMON[j][1] for j in best))] = r

    def fmt(r):
        m = len(r)
        inf = sum(x[0] > 0 for x in r) / m
        d2 = [x[2] for x in r if x[2] is not None]
        d3 = [x[3] for x in r if x[3] is not None]
        infc = sum(x[4] > 0 for x in r) / m
        return (f"inf@3={inf:.3f} inf@3(+concept)={infc:.3f} inf/3={statistics.mean(x[0] for x in r):.2f} ans@3={statistics.mean(x[1] for x in r):.2f} "
                f"dec(w>=2)@3={sum(d2) / max(len(d2), 1):.3f} (n={len(d2)}) "
                + (f"dec(w3)@3={sum(d3) / max(len(d3), 1):.3f} (n={len(d3)})" if d3 else ""))
    print(f"cases={len(cases)} gold resolvable in KB={resolvable}")
    for key in sorted(res, key=lambda k: (k[1], k[0], k[2])):
        print(f"N={key[1]} {key[0]:8s} {key[2][:60]:60s} {fmt(res[key])}")
    lat.sort()
    print(f"latency ms: mean={statistics.mean(lat):.2f} p50={lat[len(lat) // 2]:.2f} p95={lat[int(len(lat) * .95)]:.2f} "
          f"max={lat[-1]:.2f} (n={len(lat)})")
    for e in examples:
        print("EX", e)


if __name__ == "__main__":
    sys.exit(main())
