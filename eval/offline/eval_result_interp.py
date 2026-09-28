"""Offline evaluation of agent/result_interpreter.py on the hand-labelled gold set (no LLM, no network).

    python eval/offline/eval_result_interp.py            # metrics
    python eval/offline/eval_result_interp.py --errors   # + per-item misses / extras
    python eval/offline/eval_result_interp.py --write    # also write data/labels/result_interp_gold_v1_metrics.json

Gold: data/labels/result_interp_gold_v1.jsonl ("dev": the rules were tuned on it) and
data/labels/result_interp_gold_fresh_v1.jsonl ("fresh": written after the rules, scored once before any further change),
one report snippet per line (mixed Korean / English, negations, hedges,
comparisons, recommendation sections, normal studies, labs, exam findings), labelled by hand by the clinical
strategist on 2026-09-28 from clinical knowledge (no LLM). Each gold item: lexicon concept, polarity, optional
laterality. "normal" marks a whole-normal study. Only concepts that exist in data/lexicon are labelled; abnormal wording
the lexicon has no concept for is not scored (the interpreter reports it as concept "").

Metrics: precision / recall per polarity on (concept, polarity) pairs (an item counts for the polarity it was
predicted / labelled with), overall micro P/R/F1, laterality accuracy on matched gold items that state a side,
normal-flag accuracy, per-tag recall, needs_llm() rate. Caveat: the gold set and the rules were written by the same
author (development set, not a held-out test).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.agent.result_interpreter import interpret, needs_llm, render_for_prompt  # noqa: E402

GOLDS = {"dev": ROOT / "data" / "labels" / "result_interp_gold_v1.jsonl",
         "fresh": ROOT / "data" / "labels" / "result_interp_gold_fresh_v1.jsonl"}
OUT = ROOT / "data" / "labels" / "result_interp_gold_v1_metrics.json"


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def run(gold_path: Path, errors: bool = False) -> dict:
    rows = [json.loads(x) for x in gold_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_pol = {p: Counter() for p in ("present", "absent", "uncertain")}
    tot = Counter()
    lat = Counter()
    normal = Counter()
    tags: dict[str, Counter] = defaultdict(Counter)
    n_llm = 0
    for r in rows:
        it = interpret(r["test"], r["text"])
        pred = {(i.concept, i.polarity) for i in it.items if i.concept}
        gold = {(g["concept"], g["polarity"]) for g in r["gold"]}
        for c, p in pred:
            (by_pol[p].__setitem__("tp", by_pol[p]["tp"] + 1) if (c, p) in gold else
             by_pol[p].__setitem__("fp", by_pol[p]["fp"] + 1))
        for c, p in gold - pred:
            by_pol[p]["fn"] += 1
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        tot.update(tp=tp, fp=fp, fn=fn)
        for t in r.get("tags", []) or ["plain"]:
            tags[t].update(tp=tp, fn=fn, fp=fp)
        for g in r["gold"]:
            if g.get("laterality") and (g["concept"], g["polarity"]) in pred:
                got = {i.laterality for i in it.items if i.concept == g["concept"] and i.polarity == g["polarity"]}
                lat["ok" if g["laterality"] in got else "bad"] += 1
        normal["ok" if it.normal == bool(r.get("normal")) else "bad"] += 1
        n_llm += needs_llm(it)
        if errors and (pred != gold or it.normal != bool(r.get("normal"))):
            print(f"{r['id']} [{r['test']}] {r['text']}")
            print(f"   missing: {sorted(gold - pred)}")
            print(f"   extra:   {sorted(pred - gold)}")
            print(f"   normal: gold={r.get('normal')} pred={it.normal}  |  {render_for_prompt(it)}")
    res = {
        "gold": str(gold_path.relative_to(ROOT)), "n_items": len(rows), "n_gold_findings": sum(len(r["gold"]) for r in rows),
        "overall": _prf(tot["tp"], tot["fp"], tot["fn"]),
        "by_polarity": {p: _prf(c["tp"], c["fp"], c["fn"]) for p, c in by_pol.items()},
        "laterality_accuracy": round(lat["ok"] / max(1, lat["ok"] + lat["bad"]), 3), "laterality_n": lat["ok"] + lat["bad"],
        "normal_flag_accuracy": round(normal["ok"] / max(1, normal["ok"] + normal["bad"]), 3),
        "needs_llm_rate": round(n_llm / max(1, len(rows)), 3),
        "by_tag_recall": {t: round(c["tp"] / max(1, c["tp"] + c["fn"]), 3) for t, c in sorted(tags.items())},
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    res = {name: run(path, args.errors) for name, path in GOLDS.items() if path.exists()}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.write:
        res["notes"] = {
            "fresh_first_pass": "fresh set scored once before any change it prompted (2026-09-28): concept+polarity "
                                "P 0.879 R 0.879 (tp 58, fp 8, fn 8); the misses then fixed were generic (English list "
                                "negation, comparison header scope, SYM/SIGN words inside imaging reports, testis flow "
                                "wording, qualifier scope, side-less duplicate merge)",
            "caveat": "rules and labels by the same author; dev = tuned on, fresh = partly tuned after first pass",
        }
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
