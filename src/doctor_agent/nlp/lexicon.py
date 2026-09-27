"""Clinical-finding lexicon: one concept inventory for every module that reads findings from text.

Concepts have stable ids by category prefix:
    SYM:  symptoms (what the patient reports)          SIGN: examination / vital-sign findings
    LAB:  laboratory abnormalities                      IMG:  imaging / pathology / microbiology findings
    ECG:  ECG findings                                  HX:   history items and risk factors
    QUAL: pain quality / timing modifiers               GRP:  groups ("소화기 증상") that negate their members
Each concept carries Korean lay expressions (구어체), Korean medical terms, English synonyms and abbreviations, flexible
regexes, "absence" expressions ("잘 먹어요" -> no anorexia), parents, links to existing KB term ids (data/kb) and to
protocol categories / trigger tuples, and per-entry provenance (seed, curated, kb, grounding, clinical_rules,
protocols, danger_gate, preconditions, kb_tests, casefreq).

The data file data/lexicon/concepts.json is built offline by scripts/build_lexicon.py (no LLM, no network). It is
read-only static data, loaded once at import (no per-case state). Stdlib only, CPU only.

API
    LEXICON                       the loaded Lexicon
    normalize(text)               NFKC + lower case + unit/sign clean-up used by every matcher here
    Lexicon.scan(text)            raw mentions in normalised text, longest match wins (see Mention)
    Lexicon.lookup(term)          concept ids whose surface forms equal the term
    Lexicon.concept(cid)          Concept record;  ancestors(cid) / descendants(cid) over the parent links
    Lexicon.by_kb(term_id)        concept ids linked to a KB term id (e.g. "TF:lipase_high", "WD:Q38933")
    Lexicon.by_protocol(name)     concept ids linked to a protocol category or trigger tuple name
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[3] / "data" / "lexicon" / "concepts.json"

# 2-character Korean forms must start a word, or follow one of these glued prefixes ("우측흉통", "잔기침")
_GLUE_PREFIXES = ("우측", "좌측", "양측", "우", "좌", "양", "급성", "만성", "심한", "간헐적", "지속적", "반복적", "잔", "헛",
                  "마른", "전", "하", "상", "편측", "일측", "경미한", "경한", "중등도", "미약한", "약간의", "심해진")
_KO_NUM = {"한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5", "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9"}
_KO_COUNT = re.compile(r"(?<![가-힣])(한|두|세|네|다섯|여섯|일곱|여덟|아홉) ?(번|차례|개|잔|갑|병|알)(?![가-힣])")


def normalize(text: str | None) -> str:
    """Lower-cased NFKC text with signs and units unified: "(-)" -> 음성, "(+)" -> 양성, "14,200" -> "14200",
    "두 번" -> "2번", "비정상" -> "이상" (so "정상" inside it is not read as normal). Offsets of findings refer to this."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("µ", "μ").replace("β", "b")
    t = re.sub(r"[–—−~]", lambda m: "~" if m.group() == "~" else "-", t)
    t = re.sub(r"\(\s*-\s*\)", " 음성 ", t)
    t = re.sub(r"\(\s*(?:\+{1,4}|[1-4]\s?\+)\s*\)", " 양성 ", t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    t = t.replace("비정상", "이상")
    t = _KO_COUNT.sub(lambda m: _KO_NUM[m.group(1)] + m.group(2), t)
    return re.sub(r"\s+", " ", t).strip()


def _cls(ch: str) -> str | None:
    if "가" <= ch <= "힣":
        return "h"
    if "a" <= ch <= "z":
        return "a"
    if "0" <= ch <= "9":
        return "d"
    return None


def compact(text: str) -> tuple[str, list[int], list[int]]:
    """(word characters only, index of each in `text`, index in the compact string of the word start covering each
    compact character). A change of script (Hangul / Latin / digit) also starts a word: "38도", "5kg"."""
    chars: list[str] = []
    where: list[int] = []
    wstart: list[int] = []
    prev = None
    cur = 0
    for i, ch in enumerate(text):
        c = _cls(ch)
        if c is None:
            prev = None
            continue
        if c != prev:
            cur = len(chars)
        chars.append(ch)
        where.append(i)
        wstart.append(cur)
        prev = c
    return "".join(chars), where, wstart


def compact_key(form: str) -> str:
    return compact(normalize(form))[0]


@dataclass(frozen=True)
class Concept:
    id: str
    cat: str
    ko: str
    en: str
    parents: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    kb: tuple[str, ...] = ()
    protocol: tuple[str, ...] = ()

    @property
    def group(self) -> bool:
        return "group" in self.flags

    @property
    def needs_cue(self) -> bool:
        return "cue" in self.flags


@dataclass(frozen=True)
class Mention:
    """One surface match in normalised text. kind: lay | med | en | abbr (form), re (regex; a negation inside the
    span negates it), rei (idiomatic regex: the negation word is part of the concept, "입맛이 없"), neg / negre
    (expression asserting absence), block (consumes the span, no concept)."""

    start: int
    end: int
    cid: str
    kind: str
    fig: bool
    text: str

    @property
    def is_absence(self) -> bool:
        return self.kind in ("neg", "negre")


_NEG_IN_PATTERN = re.compile(r"없|않|아니|안 |안\)|안\||못")
_RE_KINDS = ("re", "rei", "negre")


@dataclass
class Lexicon:
    concepts: dict[str, Concept]
    meta: dict = field(default_factory=dict)
    _ix: dict = field(default_factory=dict)  # first 2 compact chars -> [(key, cid, kind, fig, latin)]
    _re: list = field(default_factory=list)  # [(compiled, cid, kind, fig)]
    _forms: dict = field(default_factory=dict)  # compact key -> [cid]
    _anc: dict = field(default_factory=dict)
    _desc: dict = field(default_factory=dict)
    _kb: dict = field(default_factory=dict)
    _proto: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_data(cls, data: dict) -> "Lexicon":
        concepts: dict[str, Concept] = {}
        lex = cls(concepts, data.get("meta", {}))
        for c in data["concepts"]:
            concepts[c["id"]] = Concept(c["id"], c.get("cat", c["id"].split(":")[0]), c.get("ko", ""), c.get("en", ""),
                                        tuple(c.get("parents", ())), tuple(c.get("flags", ())), tuple(c.get("kb", ())),
                                        tuple(c.get("protocol", ())))
            all_fig = "fig" in c.get("flags", ())
            for f in c.get("forms", ()):
                form, kind = f[0], f[1]
                fig = all_fig or (len(f) > 3 and bool(f[3]))
                lex._add_form(form, c["id"], kind, fig)
            for r in c.get("re", ()):
                lex._add_re(r[0], c["id"], "re", all_fig)
            for r in c.get("negre", ()):
                lex._add_re(r[0], c["id"], "negre", all_fig)
        for w in data.get("block", ()):
            lex._add_form(w, "", "block", False)
        for cid, c in concepts.items():
            for k in c.kb:
                lex._kb.setdefault(k, []).append(cid)
            for p in c.protocol:
                lex._proto.setdefault(p, []).append(cid)
        for cid in concepts:
            seen: list[str] = []
            stack = list(concepts[cid].parents)
            while stack:
                p = stack.pop()
                if p in seen or p == cid or p not in concepts:
                    continue
                seen.append(p)
                stack.extend(concepts[p].parents)
            lex._anc[cid] = tuple(seen)
            for p in seen:
                lex._desc.setdefault(p, []).append(cid)
        return lex

    def _add_form(self, form: str, cid: str, kind: str, fig: bool) -> None:
        key = compact_key(form)
        if len(key) < 2:
            return
        latin = bool(re.fullmatch(r"[a-z0-9]+", key)) and not key.isdigit()
        self._ix.setdefault(key[:2], []).append((key, cid, kind, fig, latin))
        if cid and kind not in ("neg",):
            lst = self._forms.setdefault(key, [])
            if cid not in lst:
                lst.append(cid)

    def _add_re(self, pattern: str, cid: str, kind: str, fig: bool) -> None:
        # gaps never cross a comma or a sentence end: "소변은 괜찮고, 배가 아파요" is not dysuria
        pat = re.sub(r"(?<!\\)\.\{", "[^,;]{", pattern)
        # bounded gaps are lazy: "림프절 비대나 갑상선 종대" ends the lymph-node match at "비대"
        pat = re.sub(r"(?<=[\].])(\{\d+,\d+\})(?![?+])", r"\1?", pat)
        idiom = kind == "re" and bool(_NEG_IN_PATTERN.search(pattern))
        try:
            rx = re.compile(pat)
        except re.error:
            return
        self._re.append((rx, cid, "rei" if idiom else kind, fig))

    # ---------------------------------------------------------------- queries
    def concept(self, cid: str) -> Concept | None:
        return self.concepts.get(cid)

    def ancestors(self, cid: str) -> tuple[str, ...]:
        return self._anc.get(cid, ())

    def descendants(self, cid: str) -> tuple[str, ...]:
        return tuple(self._desc.get(cid, ()))

    def by_kb(self, term_id: str) -> list[str]:
        return list(self._kb.get(term_id, ()))

    def by_protocol(self, name: str) -> list[str]:
        return list(self._proto.get(name, ()))

    def lookup(self, term: str) -> list[str]:
        """Concept ids whose surface form equals the term (space/case-insensitive), else those whose label does."""
        key = compact_key(term)
        out = list(self._forms.get(key, ()))
        if not out:
            out = [cid for cid, c in self.concepts.items() if key in (compact_key(c.ko), compact_key(c.en))]
        return out

    def label(self, cid: str) -> str:
        c = self.concepts.get(cid)
        return (c.ko or c.en) if c else cid

    def scan(self, text: str, barriers: list[int] | tuple = ()) -> list[Mention]:
        """Mentions in already-normalised text. Overlapping matches: the longest wins; equal spans keep every concept.
        Block words consume their span. Korean forms of 2 characters and Latin forms must start a word (Latin forms
        under 6 letters must also end one); longer Korean forms may sit inside a word. Regex matches that straddle a
        barrier (a clause start) are discarded before the overlap resolution."""
        comp, where, wstart = compact(text)
        n = len(comp)
        cands: list[tuple[int, int, str, str, bool]] = []
        ix = self._ix
        for p in range(n - 1):
            bucket = ix.get(comp[p:p + 2])
            if not bucket:
                continue
            ws = wstart[p]
            at_start = ws == p
            glued = not at_start and comp[ws:p] in _GLUE_PREFIXES
            for key, cid, kind, fig, latin in bucket:
                ln = len(key)
                if latin or ln <= 2:
                    if not at_start and not (glued and not latin):
                        continue
                if not comp.startswith(key, p):
                    continue
                if latin and ln < 6 and p + ln < n and wstart[p + ln] != p + ln:
                    continue
                # a match that starts inside a word must stay inside that word ("춥고 열이" is not "고열이")
                if not at_start and any(wstart[q] == q for q in range(p + 1, p + ln)):
                    continue
                cands.append((where[p], where[p + ln - 1] + 1, cid, kind, fig))
        for rx, cid, kind, fig in self._re:
            for m in rx.finditer(text):
                if m.end() > m.start() and not any(m.start() < x < m.end() - 1 for x in barriers):
                    cands.append((m.start(), m.end(), cid, kind, fig))
        if not cands:
            return []
        # longest first; on equal spans a block word beats everything and a lexicon form beats a regex. Quality /
        # timing modifiers (QUAL:) compete only among themselves, so "쥐어짜듯이 아파요" keeps both concepts.
        cands.sort(key=lambda c: (c[0] - c[1], c[0], c[3] != "block", c[3] in _RE_KINDS))
        taken = self._resolve([c for c in cands if not c[2].startswith("QUAL:")])
        taken += self._resolve([c for c in cands if c[2].startswith("QUAL:") or c[3] == "block"])
        taken = [t for i, t in enumerate(taken) if t not in taken[:i]]
        # the same span naming a concept and its ancestor keeps the specific one
        spans: dict[tuple[int, int], list[str]] = {}
        for s, e, cid, kind, _fig in taken:
            spans.setdefault((s, e), []).append(cid)
        out = [Mention(s, e, cid, kind, fig, text[s:e]) for s, e, cid, kind, fig in taken
               if kind != "block" and not any(cid in self._anc.get(o, ()) for o in spans[(s, e)] if o != cid)]
        out.sort(key=lambda m: (m.start, m.end))
        return out

    @staticmethod
    def _resolve(cands: list) -> list:
        taken: list[tuple[int, int, str, str, bool]] = []
        for c in cands:
            ok = True
            for t in taken:
                if c[0] < t[1] and t[0] < c[1]:
                    if (c[0], c[1]) == (t[0], t[1]) and t[3] != "block" and c[3] != "block" and c[2] != t[2] \
                            and not (c[3] in _RE_KINDS and t[3] not in _RE_KINDS):
                        continue
                    ok = False
                    break
            if ok:
                taken.append(c)
        return taken

    def stats(self) -> dict:
        by_cat: dict[str, int] = {}
        for c in self.concepts.values():
            by_cat[c.cat] = by_cat.get(c.cat, 0) + 1
        n_forms = sum(len(v) for v in self._ix.values())
        return {"concepts": len(self.concepts), "by_category": by_cat, "surface_forms": n_forms, "regexes": len(self._re)}


def load(path: Path = LEXICON_PATH) -> Lexicon:
    with open(path, encoding="utf-8") as f:
        return Lexicon.from_data(json.load(f))


# Read-only static data, loaded once at import from data/lexicon (not case content; never modified afterwards).
LEXICON: Lexicon = load() if LEXICON_PATH.exists() else Lexicon({})
