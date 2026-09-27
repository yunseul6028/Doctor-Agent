#!/usr/bin/env python3
"""Offline benchmark for the knowledge base (no LLM, no network, stdlib only).

For every case in data/cases_aug/{sample,clinicalqa,agentclinic,diagnosisarena}:
  1. extract positive findings from the case text (initial complaint + history/exam/test answers), dropping
     clauses with Korean/English negation or "normal" wording (a stand-in for the findings the agent collects);
  2. rank the gold diagnosis (diagnosis + aliases resolved in the KB) in kb.candidates(findings, k=50).
Reports top-1/3/10/50 hit rates and MRR overall and per set, KB coverage of gold diagnoses, and a
normalize_diagnosis() benchmark on gold names + aliases. Split: dev = sample + clinicalqa (tune here only),
held-out = agentclinic + diagnosisarena (report only).

    python scripts/eval_kb.py                 # prints a summary, writes data/labels/kb_eval_<date>.json
    python scripts/eval_kb.py --no-write --show-misses 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.knowledge import kb  # noqa: E402

CASES = ROOT / "data" / "cases_aug"
SETS = {"sample": "dev", "clinicalqa": "dev", "agentclinic": "heldout", "diagnosisarena": "heldout"}
KS = (1, 3, 10, 50)

# clause splitting: sentence ends (not decimal points), semicolons, newlines and clause connectives
_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|\.(?=\s)|[;\n!?]|(?<=[가-힣])(?:고|며|면서|지만|는데|으나)\s|\s(?:but|and)\s")
_REF = re.compile(r"\((?:[^)]*(?:정상|참고|normal|ref)[^)]*)\)", re.I)
_NEG_WORDS = ("없", "않", "아니", "안 ", "못 ", "음성", "정상", "부인", "특이", "(-)", "( - )", "no ", "not ", "denies",
              "denied", "without", "never", "negative", "normal", "unremarkable",
              # normal exam wording ("복부 평탄", "의식 명료", "근력 5/5")
              "평탄", "명료", "촉촉", "규칙적", "5/5", "양호", "잘 촉지", "깨끗", "적절", "대칭적")


def load_cases(sets=tuple(SETS)) -> list[dict]:
    out = []
    for s in sets:
        for f in sorted((CASES / s).glob("*.json")):
            c = json.loads(f.read_text(encoding="utf-8"))
            c["_set"], c["_id"] = s, f"{s}/{f.stem}"
            out.append(c)
    return out


ALL_SECTIONS = ("history", "exam", "tests")


def _units(case: dict, sections=ALL_SECTIONS) -> list[str]:
    ini = case.get("initial", "")
    units = [ini.split("주호소:", 1)[1] if "주호소:" in ini else ini]
    for sec in sections:
        for key, val in (case.get(sec) or {}).items():
            val = str(val)
            if sec == "tests" and len(val) < 20:  # "양성(증가함)" alone loses the analyte name
                val = key.split("|")[0] + ": " + val
            units.append(val)
    return units


def _is_neg(clause: str) -> bool:
    low = clause.lower() + " "
    return any(w in low for w in _NEG_WORDS)


def extract_findings(case: dict, sections=ALL_SECTIONS) -> tuple[list[str], list[str]]:
    """(positive clauses, negated clauses) of the case text. Clauses with numbers are split at commas so a normal
    value next to an abnormal one does not hide it ("Na 139, K 6.1")."""
    pos, neg = [], []
    for u in _units(case, sections):
        u = _REF.sub("", u)
        for cl in _SPLIT.split(u):
            cl = (cl or "").strip(" ,:·-")
            if len(cl) < 2:
                continue
            parts = [p.strip(" ,:·-") for p in re.split(r",\s", cl)] if re.search(r"\d", cl) else [cl]
            for p in parts:
                if len(p) >= 2:
                    (neg if _is_neg(p) else pos).append(p)
    return pos, neg


def gold_ids(case: dict) -> set[str]:
    k = kb.get_kb()
    ids: set[str] = set()
    for name in [case.get("diagnosis", "")] + list(case.get("aliases") or []):
        i = k._resolve(name)
        if i is not None:
            ids.add(k.diseases[i]["id"])
    return ids


def rank_case(case: dict, k: int = 50, use_neg: bool = True, use_demo: bool = True,
              sections=ALL_SECTIONS) -> dict:
    findings, negs = extract_findings(case, sections)
    gold = gold_ids(case)
    sex, age = kb.patient_profile(case.get("initial", "")) if use_demo else ("", None)
    t0 = time.perf_counter()
    res = kb.candidates(findings, k=k, negatives=negs if use_neg else None, sex=sex or None, age=age)
    ms = (time.perf_counter() - t0) * 1000
    rank = next((r for r, c in enumerate(res, 1) if c["id"] in gold), None)
    kbk = kb.get_kb()
    has_sx = any(kbk.diseases[kbk.by_id[g]]["symptoms"] for g in gold)
    has_orpha = any(kbk.diseases[kbk.by_id[g]].get("orpha_freq") for g in gold)
    return {"id": case["_id"], "set": case["_set"], "dx": case.get("diagnosis", ""), "gold": sorted(gold),
            "in_kb": bool(gold), "gold_has_symptoms": has_sx, "gold_has_orphanet": has_orpha, "n_findings": len(findings), "rank": rank,
            "top3": [c["name_ko"] for c in res[:3]], "ms": round(ms, 2)}


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    out = {"n": len(rows)}
    for kk in KS:
        out[f"top{kk}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= kk) / n, 4)
    out["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 4)
    out["coverage_in_kb"] = round(sum(r["in_kb"] for r in rows) / n, 4)
    out["coverage_with_symptoms"] = round(sum(r["gold_has_symptoms"] for r in rows) / n, 4)
    out["coverage_with_orphanet"] = round(sum(r.get("gold_has_orphanet", False) for r in rows) / n, 4)
    cov = [r for r in rows if r["gold_has_symptoms"]]
    out["top10_given_covered"] = round(sum(1 for r in cov if r["rank"] and r["rank"] <= 10) / (len(cov) or 1), 4)
    return out


def bench_ranking(cases: list[dict], **kw) -> tuple[dict, list[dict]]:
    rows = [rank_case(c, **kw) for c in cases]
    rep = {"overall": summarize(rows)}
    for split in ("dev", "heldout"):
        rep[split] = summarize([r for r in rows if SETS[r["set"]] == split])
    for s in SETS:
        rep[s] = summarize([r for r in rows if r["set"] == s])
    ms = sorted(r["ms"] for r in rows)
    rep["latency_ms"] = {"mean": round(sum(ms) / len(ms), 2), "p95": ms[int(0.95 * (len(ms) - 1))], "max": ms[-1]}
    return rep, rows


def bench_normalize(cases: list[dict]) -> dict:
    """Hit rate of normalize_diagnosis() on gold names and aliases, and whether a case's names agree on one code."""
    out = {}
    for split in ("dev", "heldout", "overall"):
        sub = [c for c in cases if split == "overall" or SETS[c["_set"]] == split]
        names = hit = with_code = prim = prim_code = agree = groups = 0
        for c in sub:
            codes = []
            for j, name in enumerate([c.get("diagnosis", "")] + list(c.get("aliases") or [])):
                n = kb.normalize_diagnosis(name)
                names += 1
                hit += bool(n)
                with_code += bool(n and n.get("code"))
                if j == 0:
                    prim += bool(n)
                    prim_code += bool(n and n.get("code"))
                if n and n.get("code"):
                    codes.append(n["code"][:3])
            if len(codes) >= 2:
                groups += 1
                top = max(set(codes), key=codes.count)
                agree += codes.count(top) == len(codes)
        m = len(sub) or 1
        out[split] = {"cases": len(sub), "names": names, "hit_rate": round(hit / (names or 1), 4),
                      "code_rate": round(with_code / (names or 1), 4), "primary_hit_rate": round(prim / m, 4),
                      "primary_code_rate": round(prim_code / m, 4),
                      "alias_code_agreement": round(agree / (groups or 1), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--show-misses", type=int, default=0)
    ap.add_argument("--split", default="dev", help="split to show misses from (dev/heldout)")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-neg", action="store_true", help="do not pass negated findings")
    ap.add_argument("--no-demo", action="store_true", help="do not pass sex/age")
    ap.add_argument("--prev-w", type=float, default=None, help="override the prevalence prior weight (0 = off)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    kbk = kb.get_kb()
    load_s = time.perf_counter() - t0
    if args.prev_w is not None:
        kbk.PREV_W = args.prev_w
    cases = load_cases()
    rank, rows = bench_ranking(cases, use_neg=not args.no_neg, use_demo=not args.no_demo)
    # early-interview view: initial complaint + history + exam only (no test results), i.e. when the agent's KB
    # candidate hints actually fire
    rank_hx, _ = bench_ranking(cases, use_neg=not args.no_neg, use_demo=not args.no_demo,
                               sections=("history", "exam"))
    norm = bench_normalize(cases)
    report = {"date": dt.date.today().isoformat(), "kb_load_s": round(load_s, 2), "ranking": rank,
              "ranking_history_exam_only": {s: rank_hx[s] for s in ("dev", "heldout", "overall")},
              "normalize": norm, "cases": rows,
              "method": "findings = positive clauses of initial/history/exam/tests (negated/normal clauses dropped); "
                        "gold = KB profiles resolved from diagnosis + aliases; candidates(k=50)."}
    for split in ("dev", "heldout", "overall"):
        r = rank[split]
        print(f"{split:8s} n={r['n']:3d} top1={r['top1']:.3f} top3={r['top3']:.3f} top10={r['top10']:.3f} "
              f"top50={r['top50']:.3f} mrr={r['mrr']:.3f} cov={r['coverage_in_kb']:.3f} "
              f"cov_sx={r['coverage_with_symptoms']:.3f} top10|cov={r['top10_given_covered']:.3f}")
    for s in SETS:
        r = rank[s]
        print(f"  {s:15s} n={r['n']:3d} top1={r['top1']:.3f} top10={r['top10']:.3f} top50={r['top50']:.3f} "
              f"mrr={r['mrr']:.3f} cov={r['coverage_in_kb']:.3f}")
    for split in ("dev", "heldout"):
        r = rank_hx[split]
        print(f"  hx+exam only {split:8s} top1={r['top1']:.3f} top10={r['top10']:.3f} top50={r['top50']:.3f} "
              f"mrr={r['mrr']:.3f}")
    print("latency ms", rank["latency_ms"], "load s", round(load_s, 2))
    for split in ("dev", "heldout"):
        print("normalize", split, norm[split])
    if args.show_misses:
        miss = [r for r in rows if SETS[r["set"]] == args.split and r["gold_has_symptoms"]
                and not (r["rank"] and r["rank"] <= 10)]
        for r in miss[: args.show_misses]:
            print(f"  MISS {r['id']} {r['dx']} rank={r['rank']} nf={r['n_findings']} top3={r['top3']}")
    if not args.no_write:
        out = Path(args.out) if args.out else ROOT / "data" / "labels" / f"kb_eval_{report['date']}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
