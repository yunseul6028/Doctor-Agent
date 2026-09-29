#!/usr/bin/env python3
"""Audit of diagnosis-name normalisation (knowledge/kb.py normalize_diagnosis). Offline, stdlib only, no network.

Runs normalize_diagnosis() over
  - every gold diagnosis + alias of data/cases_aug (267 cases),
  - every name of the specialty gold sets (data/labels/specialty_gold_v*.jsonl),
  - every can't-miss / rule-out name of safety/protocols.py (Protocol.cant_miss), safety/danger_gate.py
    (RULE_OUT_TABLE names + aliases) and the consult specs (agent/subagents/consult.py must_not_miss,
    rule_out_names; "A·B" / "A/B" lists are split into their parts) — read-only imports,
and flags suspicious mappings with cheap checks:
  chapter   the KCD code's specialty (knowledge/specialty.py KCD_TABLE, no name overrides) disagrees with the
            specialty gold label and its "also" alternatives (specialty gold only);
  semantic  a keyword class of the input names a code range the result is outside of (poisoning/overdose words →
            T36-T65, cancer words → C/D00-D48, fracture words → S/T/M80-M84), see SEMANTIC;
  weak      a non-exact match (contained / superstring / backoff / fuzzy) whose char-bigram Dice between the input
            and the best of the result's names is below WEAK_DICE;
  overlap   no shared content token (Korean word ≥2 chars or its 2-char stem, English word ≥4 chars) between the
            input and the result's names, for non-exact matches;
  generic   a multi-word input mapped by a non-exact step to a residual code ("기타"/"상세불명"/other/unspecified,
            or a 3-character category of a chapter-summary code);
  pair      names of one case (diagnosis + aliases) or one danger entry (name + aliases) whose KCD codes fall in
            different specialty groups (specialty.py KCD_TABLE; Korean/English pairs disagreeing on the chapter).
Hand-reviewed exceptions live in data/labels/normalize_audit_allow.json ({"check|input": "reason"}); tests/
test_normalize_audit.py runs the same checks on the gold sets and fails on any flag outside that allow-list.

    python scripts/audit_normalize.py                 # summary + every flag
    python scripts/audit_normalize.py --json OUT.json # also write all rows
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doctor_agent.knowledge import kb  # noqa: E402
from doctor_agent.knowledge import specialty as sp  # noqa: E402

ALLOW_PATH = ROOT / "data" / "labels" / "normalize_audit_allow.json"
WEAK_DICE = 0.35
EXACT = {"code", "kcd_exact", "profile", "curated", "abbreviation"}  # curated/abbreviation: hand-made tables
_NON = re.compile(r"[^0-9a-z가-힣]")
_TOK = re.compile(r"[a-z]{4,}|[가-힣]{2,}")
_RESIDUAL = re.compile(r"기타|상세불명|달리 분류되지 않은|\bother\b|unspecified|\bNEC\b|\bNOS\b", re.I)
# keyword class of the input → allowed KCD code ranges (3-character categories, inclusive)
SEMANTIC: tuple[tuple[str, re.Pattern, tuple[tuple[str, str], ...]], ...] = (
    ("poisoning", re.compile(r"overdose|poisoning|(?<!neuro)toxicity|intoxication|과다\s?복용|음독|에 의한 중독"
                             r"|(?:제|약|물질|가스|탄소|농약|유기인\S*|리튬|디곡신|아세트아미노펜|파라세타몰)\s?중독", re.I),
     (("T36", "T65"), ("F10", "F19"), ("X40", "X49"), ("E00", "E90"))),  # E: thyrotoxicosis ("갑상선 중독")
    ("cancer", re.compile(r"(?<![가-힣])[가-힣]{1,6}암(?:$|\s)|carcinoma|cancer|malignan(?!t (syndrome|hypert))|lymphoma|leuk[a]?emia|sarcoma"
                          r"|림프종|백혈병|육종|골수종|myeloma", re.I),
     (("C00", "D48"),)),
    ("fracture", re.compile(r"fracture|골절", re.I), (("S00", "T14"), ("M80", "M84"), ("M48", "M48"))),
)


def _n(s: str) -> str:
    return _NON.sub("", (s or "").lower())


def _dice(a: str, b: str) -> float:
    ga, gb = kb._bigrams(a), kb._bigrams(b)
    return 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0


def _toks(s: str) -> set[str]:
    out = set()
    for t in _TOK.findall((s or "").lower()):
        out.add(t)
        if kb._HANGUL.search(t) and len(t) > 2:
            out.add(t[:2])
    return out


def split_names(s: str) -> list[str]:
    """'패혈증·패혈성 쇼크' → its parts; 'TTP/HUS' → its parts (list displays, not single diagnoses); parenthesised
    text is left to the normaliser."""
    parts, depth, cur = [], 0, ""
    for i, ch in enumerate(s):  # split on "·", "/", ", " outside parentheses
        depth += ch == "(" and 1 or ch == ")" and -1 or 0
        if depth == 0 and (ch in "·/" or (ch == "," and s[i + 1:i + 2] == " ")):
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts = [p.strip() for p in parts + [cur] if p.strip()]
    return list(dict.fromkeys(parts)) if len(parts) > 1 else [s]


# --------------------------------------------------------------------------------------------------------------------
# name sources
# --------------------------------------------------------------------------------------------------------------------

def case_groups() -> list[dict]:
    """[{"src": "cases", "id", "names": [diagnosis, *aliases]}] for data/cases_aug."""
    out = []
    for f in sorted((ROOT / "data" / "cases_aug").glob("*/*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        names = [c.get("diagnosis", "")] + list(c.get("aliases") or [])
        out.append({"src": "cases", "id": f"{f.parent.name}/{f.stem}", "names": [x for x in names if x.strip()]})
    return out


def specialty_gold() -> list[dict]:
    """Unique names of specialty_gold_v*.jsonl, the latest version's label winning (as eval_specialty's corrections)."""
    rows: dict[str, dict] = {}
    for p in sorted((ROOT / "data" / "labels").glob("specialty_gold_v*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["name"]] = {"src": "specialty_gold", "id": p.stem, "names": [r["name"]],
                                   "specialty": r["specialty"], "also": list(r.get("also") or [])}
    return list(rows.values())


def safety_groups() -> list[dict]:
    from doctor_agent.agent.subagents import consult
    from doctor_agent.safety import danger_gate, protocols
    out = []
    for r in danger_gate.RULE_OUT_TABLE:
        out.append({"src": "danger_gate", "id": r.name, "names": [r.name] + [a for a in r.aliases if a.strip()]})
    seen = set()
    for p in protocols.PROTOCOLS:
        for dx in p.cant_miss:
            for x in split_names(dx):
                if x not in seen:
                    seen.add(x)
                    out.append({"src": "protocols", "id": p.category, "names": [x]})
    for s in consult.SPECIALTIES.values():
        for dx in tuple(s.must_not_miss) + tuple(s.rule_out_names):
            for x in split_names(dx):
                if x not in seen:
                    seen.add(x)
                    out.append({"src": "consult", "id": s.id, "names": [x]})
    return out


# --------------------------------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------------------------------

def result_names(k, res: dict) -> list[str]:
    names = [res.get("name", ""), res.get("kcd_name", ""), res.get("name_en", "")]
    code = (res.get("code") or "").replace(".", "")
    kcd, _ = k._kcd_table()
    if code in kcd:
        names += kcd[code][0] + [kcd[code][1]]
    i = k.by_id.get(res.get("id", ""))
    if i is not None:
        d = k.diseases[i]
        names += [n for n, _ in d["names_ko"]] + [n for n, _ in d["names_en"]]
    return [x for x in names if x]


def _in_ranges(code3: str, ranges) -> bool:
    return any(lo <= code3 <= hi for lo, hi in ranges)


def check_name(k, name: str, res: dict | None, gold: dict | None = None) -> list[str]:
    """Flags for one (input, result) pair; gold = {"specialty", "also"} for the chapter check."""
    flags = []
    if not res:
        return flags
    code = (res.get("code") or "").replace(".", "")
    how = res.get("match", "")
    if gold is not None and code:
        e = sp._code_entry(code)
        spec = e[0] if e else None
        ok = {gold["specialty"], *gold.get("also", [])}
        if spec not in ok:
            flags.append("chapter")
    low = name.lower()
    for tag, rx, ranges in SEMANTIC:
        if rx.search(low) and code and not _in_ranges(code[:3], ranges):
            flags.append("semantic")
            break
    if how not in EXACT:
        names = result_names(k, res)
        key = _n(kb._PAREN.sub(" ", name))
        best = max((_dice(key, _n(x)) for x in names), default=0.0)
        if best < WEAK_DICE:
            flags.append("weak")
        qt = _toks(kb._PAREN.sub(" ", name))
        if qt and not any(qt & _toks(x) for x in names):
            flags.append("overlap")
        if code and len(name.split()) >= 2:
            kcd, _ = k._kcd_table()
            ko, en, _sx = kcd.get(code, ([], "", ""))
            if _RESIDUAL.search(" ".join(ko[:1]) + " " + en) and not _RESIDUAL.search(name):
                flags.append("generic")
    return flags


def _group(code: str) -> str:
    e = sp._code_entry(code)
    return e[1] if e else "?"


def run(groups: list[dict]) -> list[dict]:
    """One row per (group, name): result + flags; a "pair" flag goes on every name of a disagreeing group."""
    k = kb.get_kb()
    rows = []
    for g in groups:
        gold = {"specialty": g["specialty"], "also": g["also"]} if "specialty" in g else None
        grp = []
        for name in g["names"]:
            res = k.normalize_diagnosis(name)
            grp.append({"src": g["src"], "id": g["id"], "input": name, "code": (res or {}).get("code", ""),
                        "name": (res or {}).get("name", ""), "name_en": (res or {}).get("name_en", ""),
                        "match": (res or {}).get("match", ""), "flags": check_name(k, name, res, gold)})
        groups = {_group(r["code"]) for r in grp if r["code"]}
        if len(groups) > 1:
            for r in grp:
                if r["code"]:
                    r["flags"].append("pair")
        rows += grp
    return rows


def load_allow() -> dict[str, str]:
    return json.loads(ALLOW_PATH.read_text(encoding="utf-8")) if ALLOW_PATH.exists() else {}


def unallowed(rows: list[dict], allow: dict[str, str]) -> list[tuple[str, dict]]:
    return [(f, r) for r in rows for f in r["flags"] if f"{f}|{r['input']}" not in allow]


def all_groups() -> list[dict]:
    return case_groups() + specialty_gold() + safety_groups()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--all", action="store_true", help="also list allow-listed flags")
    a = ap.parse_args()
    rows = run(all_groups())
    allow = load_allow()
    by_src = Counter(r["src"] for r in rows)
    flagged = [r for r in rows if r["flags"]]
    print(f"names: {len(rows)} {dict(by_src)}; resolved: {sum(bool(r['match']) for r in rows)}; "
          f"with code: {sum(bool(r['code']) for r in rows)}")
    print(f"flagged names: {len(flagged)}; flags: {dict(Counter(f for r in rows for f in r['flags']))}")
    bad = unallowed(rows, allow)
    print(f"flags outside the allow-list: {len(bad)} ({dict(Counter(f for f, _ in bad))})")
    for r in rows:
        for f in r["flags"]:
            allowed = f"{f}|{r['input']}" in allow
            if allowed and not a.all:
                continue
            print(f"  [{f}{'*' if allowed else ''}] {r['src']}:{r['id']}  {r['input']!r} -> {r['code']} "
                  f"{r['name']} / {r['name_en']} ({r['match']})")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
