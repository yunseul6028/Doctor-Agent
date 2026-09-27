"""Knowledge-base hints for the policy: shown to the LLM as *hints, not evidence*, with a small prompt budget.

Three hooks, all fail-safe (any KB problem → None / {} so the agent runs exactly as without the KB):
    candidate_hint(state, seen)     other diseases to consider, at most twice per case (positives reach 3, then 6)
    discriminator_hint(state, seen) differing features/tests when the top-2 live DDx are within 0.15, once per pair
    normalize_hint(dx, initial)     standard Korean name + KCD code, and a sex-restriction mismatch warning

`seen` is a per-case set owned by the caller (the Policy instance is created per case), used only to avoid
repeating a hint; nothing is shared across cases. Prompt text for these hints lives here, not in prompts.py.
"""
from __future__ import annotations

import re

from doctor_agent.agent.text import similarity
from doctor_agent.knowledge import kb

CANDIDATE_THRESHOLDS = (3, 6)  # number of positive findings at which candidate hints are shown
MAX_CANDIDATES = 3
CLOSE_P = 0.15                 # top-2 DDx probability gap that triggers a discriminator hint
MAX_FEATURES = 4
MAX_CHARS = 400
SRC = {"DDXPlus": "DDXPlus", "DO": "DO", "WD": "Wikidata", "MedlinePlus+LLM": "MedlinePlus", "MedlinePlus": "MedlinePlus",
       "KCD": "KCD", "LLM": "LLM", "curated": "curated"}
_GENERIC_TESTS = {"medical history", "physical examination", "history", "physical exam", "medical diagnosis",
                  "diagnosis", "symptom", "symptoms"}
_SEX = re.compile(r"(남성|남자|남아|소년|여성|여자|여아|소녀|임신부|임산부)|\d+\s*(?:세|살)\s*(남|여)|\b(male|female)\b",
                  re.IGNORECASE)


def _clip(text: str, n: int = MAX_CHARS) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip(" ,;") + "…"


def _positives(state) -> list[str]:
    return [f.item for f in state.findings.items if f.status == "양성"]


def _negatives(state) -> list[str]:
    return [f.item for f in state.findings.items if f.status == "음성"]


def _in_ledger(name: str, kb_id: str, ledger_ids: set[str], ledger_names: list[str]) -> bool:
    return kb_id in ledger_ids or any(similarity(name, n) >= 0.6 or (len(name) >= 2 and name in n) for n in ledger_names)


def candidate_hint(state, seen: set) -> str | None:
    """KB candidates matching the positive findings that are not in the DDx ledger yet (≤3)."""
    try:
        pos = _positives(state)
        level = sum(len(pos) >= t for t in CANDIDATE_THRESHOLDS)
        if level == 0 or ("cand", level) in seen or not kb.available():
            return None
        seen.add(("cand", level))
        names = [e.dx for e in state.ddx_ledger.entries]
        ids = {p["id"] for p in (kb.lookup(n) for n in names) if p}
        rows = []
        sex, age = kb.patient_profile(getattr(state, "initial_info", "") or "")
        for c in kb.candidates(pos, k=12, negatives=_negatives(state), sex=sex or None, age=age):
            # one match is enough when it is a test/lab result link (decisive findings like 리파아제 상승)
            by_test = any(str(m.get("id", "")).startswith("TF:") for m in c["matched"])
            if (len(c["matched"]) < 2 and not by_test) or _in_ledger(c["name_ko"], c["id"], ids, names):
                continue
            code = c["kcd"][0] if c["kcd"] else ""
            code = f" {code[:3]}.{code[3:]}" if len(code) > 3 else f" {code}" if code else ""
            hit = "·".join(dict.fromkeys(m["finding"] for m in c["matched"][:4]))
            src = ", ".join(sorted({SRC.get(s, s) for s in c["sources"]}))
            rows.append(f"{c['name_ko']}{code} (일치: {hit})" + (f" [{src}]" if src else ""))
            if len(rows) >= MAX_CANDIDATES:
                break
        if not rows:
            return None
        return _clip("고려해 볼 다른 질환 (참고용, 확진 근거 아님. 소견과 맞는지 문진·검사로 확인): " + "; ".join(rows))
    except Exception:
        return None


def _known(state) -> set[str]:
    """KB term ids already covered by the findings ledger (positive or negative)."""
    k = kb.get_kb()
    return {t for f in state.findings.items for t in k.match_terms(f.item)}


def discriminator_hint(state, seen: set) -> str | None:
    """When the top-2 live DDx are close, features/tests that differ between them (≤4) and one suggested question."""
    try:
        live = [e for e in state.ddx_ledger.ranked() if e.status != "배제"][:2]
        if len(live) < 2 or live[0].p <= 0 or live[0].p - live[1].p > CLOSE_P:
            return None
        key = ("dsc", live[0].dx, live[1].dx)
        if key in seen or not kb.available():
            return None
        seen.add(key)
        d = kb.discriminators(live[0].dx, live[1].dx, n=6)
        if not d or d["a"] == d["b"]:
            return None
        known = _known(state)
        terms = kb.get_kb().terms

        def trusted(x: dict) -> bool:  # curated (DDXPlus) or confirmed by ≥2 sources, shown before single-source ones
            return "DDXPlus" in x.get("src", []) or len(x.get("src", [])) >= 2

        def pick(side: str) -> list[dict]:
            sym = sorted((x for x in d[f"symptoms_{side}_only"] if x["id"] not in known), key=lambda x: not trusted(x))[:2]
            tst = [x for x in d[f"tests_{side}_only"] if x["en"].lower() not in _GENERIC_TESTS and x["id"] not in known][:1]
            return sym + tst
        a, b = pick("a"), pick("b")
        chosen: dict[str, list[dict]] = {"a": [], "b": []}
        for i in range(3):  # alternate sides, cap the total
            for side, lst in (("a", a), ("b", b)):
                if i < len(lst) and sum(map(len, chosen.values())) < MAX_FEATURES:
                    chosen[side].append(lst[i])
        if not chosen["a"] and not chosen["b"]:
            return None
        parts = [f"{d[s]} 쪽 — {', '.join(x['ko'] for x in chosen[s])}" for s in ("a", "b") if chosen[s]]
        text = f"감별 포인트 {d['a']} vs {d['b']} (참고용, 확진 근거 아님): " + "; ".join(parts)
        q = next((terms[x["id"]]["q"]["ko"] for s in ("a", "b") for x in chosen[s]
                  if "DDXPlus" in x.get("src", []) and (terms.get(x["id"]) or {}).get("q", {}).get("ko")), None)
        if q:
            text += f". 질문 예: \"{q}\""
        return _clip(text + " [KB]")
    except Exception:
        return None


def patient_sex(initial_info: str) -> str:
    m = _SEX.search(initial_info or "")
    if not m:
        return ""
    s = (m.group(1) or m.group(2) or m.group(3) or "").lower()
    return "여성" if s in ("female", "소녀") or s[0] in "여임" else "남성"


def normalize_hint(dx: str, initial_info: str = "") -> dict:
    """{"input", "name", "code", "sex", "patient_sex", "warning"?}; {} when the KB cannot normalise it."""
    try:
        if not dx or not kb.available():
            return {}
        n = kb.normalize_diagnosis(dx)
        if not n:
            return {}
        out = {"input": dx, "name": n.get("name", ""), "code": n.get("code", ""), "match": n.get("match", "")}
        restrict, sex = n.get("sex", ""), patient_sex(initial_info)
        if restrict:
            out["sex_restriction"] = restrict
        if sex:
            out["patient_sex"] = sex
        if restrict in ("남성", "여성") and sex and restrict != sex:
            out["warning"] = (f"'{dx}'(KCD {out['code']} {out['name']})은 {restrict}에게만 쓰는 상병인데 환자는 {sex}입니다. "
                              "진단명이 환자와 맞는지 다시 확인하세요.")
        return out
    except Exception:
        return {}


def step_hints(state, seen: set) -> list[str]:
    """All KB hints for this step (each ≤ ~400 chars; usually none)."""
    return [h for h in (candidate_hint(state, seen), discriminator_hint(state, seen)) if h]
