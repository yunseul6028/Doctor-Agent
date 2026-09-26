"""Citable medical knowledge base (disease profiles) for the Doctor Agent. Content is owned by knowledge-rag.

Built offline by scripts/build_kb.py from openly licensed sources (DDXPlus CC BY 4.0, HIRA KCD master KOGL-1,
Disease Ontology CC0, Wikidata CC0, MedlinePlus public-domain summaries). Every field keeps its source tag; see
data/kb/SOURCES.md and docs/licenses.md. Stdlib only, CPU only, no network, loaded lazily once per process.
Nothing here depends on other cases (read-only static data).

API
    lookup(name)                     -> profile dict | None      (Korean/English names, synonyms, codes; fuzzy)
    candidates(findings, k=10)       -> ranked diseases matching positive findings, with the matched terms
    discriminators(dx_a, dx_b)       -> features that differ between two diseases
    normalize_diagnosis(text)        -> {"name", "code", ...} standard Korean name + KCD code | None
    render_for_prompt(findings, dx)  -> short Korean text with source tags (<= ~800 chars) for the small LLM
    patient_profile(text)            -> (sex "남성"/"여성"/"", age or None) parsed from "35세 여성" style text

Finding → term matching uses the KB labels plus the curated tables in kb_curated.py (Korean synonyms, generic-term
stop list, vitals/lab value parsing). Offline benchmark: scripts/eval_kb.py.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import threading
from collections import defaultdict
from pathlib import Path

from doctor_agent.knowledge import kb_curated

KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
SRC_SHORT = {"DDXPlus": "DDXPlus", "DO": "DO", "WD": "Wikidata", "KCD": "KCD", "MedlinePlus": "MedlinePlus",
             "MedlinePlus+LLM": "MedlinePlus", "LLM": "LLM번역", "curated": "curated"}

_NON = re.compile(r"[^0-9a-z가-힣]")
_EN_SPACE = re.compile(r"[^0-9a-z가-힣]+")
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_CODE = re.compile(r"\b([A-Z])\s?(\d{2})(?:\.?(\d{1,2}))?\b")
# negated finding: Korean negation at the end ("발열 없음", "기침은 부인") or English negation words
_NEG = re.compile(r"((없음|없다|없어요?|없습니다|음성|부인|부인함|아님|아니요|정상|않음|않다|않아요)[.\s]*$)"
                  r"|\(\s*-\s*\)|\b(no|not|denies|denied|negative|absent)\b")
_HANGUL = re.compile(r"[가-힣]")
# KCD qualifiers stripped for an extra name key ("달리 분류되지 않은 세균성 수막염" → "세균성 수막염")
_QUAL = re.compile(r"^(달리 분류되지 않은|상세불명의|상세불명 병원체의|기타)\s*|\s*(NOS|NEC)$")
_GENERIC_TESTS = {"medical history", "physical examination", "history", "physical exam", "medical diagnosis",
                  "diagnosis", "symptom", "symptoms"}
# generic labels that would match almost any complaint
_GENERIC = {"통증", "고통", "증상", "징후", "불편감", "이상", "pain", "aches", "ache", "symptom", "symptoms", "sign"}


_PEDIATRIC = re.compile(r"소아|영아|신생아|유아기|childhood|infantile|neonatal|juvenile|pediatric|of infancy|newborn")
_ELDERLY = re.compile(r"노인성|노년|senile")
_KO_PREFIX = ("우측", "좌측", "양측", "우하", "좌하", "우상", "좌상", "급성", "만성", "마른", "심한", "전신", "간헐적",
              "지속적", "반복적", "우", "좌", "양", "상", "하", "잔")
_NEG_TAIL = re.compile(r"\s*(은|는|이|가|도)?\s*(없음|없다|없어요?|없고|없습니다|없었(?:어요|음|다)?|음성|부인함?|아님|아니요|"
                       r"않음|않다|않아요|안 함|none|absent|negative)[.\s]*$", re.I)
_SEX_RE = re.compile(r"(남성|남자|남아|소년|여성|여자|여아|소녀|임신부|임산부)|\d+\s*(?:세|살)\s*(남|여)|\b(male|female|man|woman|boy|girl)\b",
                     re.I)
_AGE_RE = re.compile(r"(\d{1,3})\s*(세|살|개월|months?|years?)", re.I)


_GENERIC_NAMES = {"증후군", "질환", "장애", "종양", "신생물", "감염", "감염증", "염증", "손상", "기타", "결핍", "결핍증",
                  "syndrome", "disease", "disorder", "disorders", "infection", "neoplasm", "tumor", "tumour", "cancer",
                  "carcinoma", "lesion", "deficiency", "insufficiency", "failure", "injury", "inflammation"}


_CONTEXT = re.compile(r"임신|분만|산후|산욕|신생아|태아|소아|영아|턱|치아|동반|의한|제외|pregnan|neonat|newborn|fetal|child|"
                      r"infant|due|with|without")


def _strip_neg(s: str) -> str:
    """"발열 없음" → "발열" (for matching the negative findings list)."""
    return _NEG_TAIL.sub("", s or "").strip()


def patient_profile(text: str) -> tuple[str, float | None]:
    """(sex "남성"/"여성"/"", age in years or None) from a short intro like "35세 여성" / "7개월 남아"."""
    t = text or ""
    sex = ""
    m = _SEX_RE.search(t)
    if m:
        v = (m.group(1) or m.group(2) or m.group(3) or "").lower()
        sex = "여성" if v in ("female", "woman", "girl", "소녀") or v[0] in "여임" else "남성"
    age = None
    m = _AGE_RE.search(t)
    if m:
        age = float(m.group(1)) / (12 if m.group(2).startswith(("개월", "month")) else 1)
    return sex, age


def _n(s: str) -> str:
    return _NON.sub("", (s or "").lower())


def _spaced(s: str) -> str:
    return " " + _EN_SPACE.sub(" ", (s or "").lower()).strip() + " "


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


class KnowledgeBase:
    K1, B = 1.2, 0.3  # BM25 parameters for candidates()
    PRIOR = {"ddx": 0.5, "mplus": 0.4, "kcd": 0.15, "rare": -0.25}
    SRC_W = {"DDXPlus": 1.0, "WD": 1.0, "DO": 1.0, "MedlinePlus+LLM": 1.0}  # per-source feature weight
    BACKOFF = 0.4    # weight of a general term implied by a specific disease feature
    QX = 0.75        # weight of a general term implied by a specific finding
    COVER_A = 0.5    # score × (COVER_A + (1 - COVER_A) × share of matched findings the disease explains)
    NEG_W = 0.5      # penalty per negated feature (× idf)
    AGE_PEN = 0.3    # pediatric-named profile for an adult (or the reverse)
    FUZZY_MIN = 0.6  # minimum char-bigram Dice for a fuzzy name match (Korean) ...
    FUZZY_SURE = 0.7  # ... below which the first two characters must agree
    FUZZY_MIN_LATIN = 0.75

    def __init__(self, kb_dir: Path = KB_DIR):
        with gzip.open(kb_dir / "kb.json.gz", "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.kb_dir = kb_dir
        self.meta = data["meta"]
        self.terms: dict[str, dict] = data["terms"]
        self.diseases: list[dict] = data["diseases"]
        self.by_id = {d["id"]: i for i, d in enumerate(self.diseases)}
        # exact name index: normalised name → [(disease idx, is primary name)]
        self.names: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for i, d in enumerate(self.diseases):
            for lst in ("names_ko", "names_en"):
                for j, (name, _src) in enumerate(d[lst]):
                    key = _n(_PAREN.sub("", name)) or _n(name)
                    if key:
                        self.names[key].append((i, j == 0))
                    # "세균성 수막염, 달리 분류되지 않은" → also index "세균성 수막염"
                    for extra in (_n(name.split(",")[0]) if "," in name else "", _n(_QUAL.sub("", name))):
                        if len(extra) >= 2 and extra != key:
                            self.names[extra].append((i, False))
        # ---- finding → term labels: KB labels + curated synonyms (kb_curated.py), minus generic/ambiguous ones
        en_ix: dict[str, str] = {}
        for tid, t in self.terms.items():
            en_ix.setdefault(t["en"].strip().lower(), tid)
        for tid, t in self.terms.items():
            for s in t["syn"]:
                if not _HANGUL.search(s):
                    en_ix.setdefault(s.strip().lower(), tid)
        self.en_ix = en_ix
        self.stop = {en_ix[x] for x in kb_curated.STOP_TERMS if x in en_ix}
        bad = {(en_ix[en], _n(lab)) for en, labs in kb_curated.BAD_LABELS.items() if en in en_ix for lab in labs}
        raw: list[tuple[str, str]] = []
        for tid, t in self.terms.items():
            if tid not in self.stop:
                raw += [(lab, tid) for lab in dict.fromkeys([t["en"], t["ko"], *t["syn"]])]
        for en, syns in kb_curated.SYNONYMS.items():
            tid = en_ix.get(en)
            if tid and tid not in self.stop:
                raw += [(x, tid) for x in syns]
        self.labels: list[tuple[str, str, bool]] = []  # (label key, tid, is_english)
        seen_lab: set[tuple[str, str]] = set()
        for lab, tid in raw:
            if not lab or lab.strip().lower() in _GENERIC:
                continue
            if _HANGUL.search(lab):
                key = _n(lab)
                keys = [key] if len(key) >= 2 else []
                if len(key) >= 4 and key[-1] in "증감":  # 구강건조증 → 구강건조 (not 심근증 → 심근)
                    keys.append(key[:-1])
                for kk in keys:
                    if (tid, kk) not in bad and (kk, tid) not in seen_lab:
                        seen_lab.add((kk, tid))
                        self.labels.append((kk, tid, False))
            else:
                key = _spaced(lab)
                # no short abbreviations ("GAD" is also a lab antibody, "MS" a valve lesion)
                if len(key.strip()) < 5 and (lab.strip().isupper() or len(key.strip()) < 4):
                    continue
                if len(key.strip()) >= 3 and (key, tid) not in seen_lab:
                    seen_lab.add((key, tid))
                    self.labels.append((key, tid, True))
        self._ko_ix: dict[str, list[tuple[str, str]]] = defaultdict(list)  # first 2 chars → Korean labels
        self._en_first: dict[str, list[tuple[str, str]]] = defaultdict(list)  # first word → English labels
        for lab, tid, is_en in self.labels:
            if is_en:
                self._en_first[lab.split()[0]].append((lab, tid))
            else:
                self._ko_ix[lab[:2]].append((lab, tid))

        # ---- term backoff: a specific term implies the general terms named inside its label
        # ("sudden numbness or weakness of the face" → weakness; "우하복부 통증" → 하복부 통증). Derived from labels.
        self.implies: dict[str, set[str]] = defaultdict(set)
        used = {t for d in self.diseases for t, _ in d["symptoms"] + d.get("risk", []) if t not in self.stop}
        generic = {t for t in used if _n(self.terms[t]["ko"]) in _GENERIC or self.terms[t]["en"].lower() in _GENERIC}
        for tid in used:
            t = self.terms[tid]
            if "history" in t["en"].lower():  # "family history of asthma" does not imply asthma
                continue
            for g in set(self._match(t["ko"])) | set(self._match(t["en"])):
                if g != tid and g in used and g not in generic:
                    self.implies[tid].add(g)
        self._build_postings()
        self._compute_prior()
        # age group from names ("childhood ...", "소아 ...", "senile ...")
        self._age: list[str] = []
        for d in self.diseases:
            names = " ".join(n for n, _ in d["names_ko"][:3] + d["names_en"][:3]).lower()
            self._age.append("child" if _PEDIATRIC.search(names) else "old" if _ELDERLY.search(names) else "")
        self._sex: list[str] | None = None
        self._bix: dict[str, tuple[dict[str, set[str]], dict[str, int]]] = {}
        self._kcd: dict[str, tuple[list[str], str, str]] | None = None
        self._kcd_names: dict[str, str] | None = None

    # ------------------------------------------------------------------ helpers
    def _richness(self, i: int) -> float:
        d = self.diseases[i]
        return (2.0 * bool(d["symptoms"]) + min(len(d["symptoms"]), 20) / 10 + bool(d["names_ko"])
                + bool(d["codes"].get("kcd")) + 0.5 * bool(d.get("summary_ko")))

    def term(self, tid: str) -> dict:
        t = self.terms[tid]
        return {"id": tid, "ko": t["ko"] or t["en"], "en": t["en"]}

    def _bigram_index(self, pool: str) -> tuple[dict[str, set[str]], dict[str, int]]:
        """Char-bigram index over profile names ("names") or KCD names ("kcd"), built lazily."""
        if pool not in self._bix:
            keys = self.names if pool == "names" else self._kcd_table()[1]
            ix: dict[str, set[str]] = defaultdict(set)
            nbig: dict[str, int] = {}
            for key in keys:
                bg = _bigrams(key)
                nbig[key] = len(bg)
                for g in bg:
                    ix[g].add(key)
            self._bix[pool] = (ix, nbig)
        return self._bix[pool]

    def _superstring(self, keys: list[str], pool: str) -> str | None:
        """Shortest known name that ends with the query and is not much longer (query ≥60% of it), unless the extra
        words change the context (pregnancy, newborn, ...): "유착성 관절낭염" → "어깨의 유착성 관절낭염"."""
        ix, _ = self._bigram_index(pool)
        best = None
        for key in keys:
            if len(key) < (4 if _HANGUL.search(key) else 8):
                continue
            cand = None
            for g in _bigrams(key):
                got = ix.get(g, set())
                cand = set(got) if cand is None else cand & got
                if not cand:
                    break
            for k in cand or ():
                if (k != key and k.endswith(key) and len(key) >= 0.6 * len(k)
                        and not _CONTEXT.search(k[: len(k) - len(key)])):
                    if best is None or (len(k), k) < (len(best), best):
                        best = k
        return best

    def _variants(self, text: str) -> tuple[list[str], list[str], list[str]]:
        """Normalised name keys to try: (same concept, backed-off concept with qualifiers dropped, spaced forms of
        the same-concept variants for word boundaries).
        Spelling pairs and qualifiers come from kb_curated (NAME_SUBS, NAME_MODIFIERS)."""
        base = _PAREN.sub(" ", text or "").lower().replace("-", " ")
        base = re.sub(r"\s+", " ", base).strip() or (text or "").lower()

        def spell(s: str) -> list[str]:
            out = [s]
            for a, b in kb_curated.NAME_SUBS:
                for x in list(out):
                    for src, dst in ((a, b), (b, a)):
                        if src in x:
                            y = x.replace(src, dst)
                            if y not in out:
                                out.append(y)
                if len(out) > 24:
                    break
            for x in list(out):  # adjectival "-성" dropped ("자가면역성 간염" → "자가면역 간염")
                y = re.sub(r"(?<=[가-힣])성(?=\s)", "", x)
                if y != x and y not in out:
                    out.append(y)
            return out
        same = spell(base)
        words = base.split()
        mods = set(kb_curated.NAME_MODIFIERS)
        core = [w for w in words if w not in mods]
        core_s = " ".join(core)
        for m in kb_curated.NAME_MODIFIERS:  # glued Korean qualifiers ("급성충수염")
            if _HANGUL.search(m) and core_s.startswith(m) and len(core_s) > len(m) + 1:
                core_s = core_s[len(m):].strip()
        back = spell(core_s) if core_s and core_s != base else []
        keys = lambda xs: list(dict.fromkeys(k for k in (_n(x) for x in xs) if k))
        return keys(same), [k for k in keys(back) if k not in keys(same)], same

    def _best_hit(self, key: str) -> int | None:
        hits = self.names.get(key)
        return max(hits, key=lambda h: (h[1], self._richness(h[0])))[0] if hits else None

    def _contained(self, keys: list[str], pool) -> str | None:
        """Longest known name inside the query ("급성 st분절상승 심근경색" → "심근경색"), ending at a word end.
        Diagnosis names put the head noun last, so the name must end the query (whole last word(s), or ≥4 chars, or
        ≥1/3 of it) or cover ≥60% of it; Latin-only names need ≥8 chars; generic heads ("증후군") never count."""
        best = None
        for spaced in keys:
            key, ends, starts = "", set(), set()
            for w in spaced.split():
                starts.add(len(key))
                key += _n(w)
                ends.add(len(key))
            n = len(key)
            for i in range(n):
                for j in range(n, i + 2, -1):
                    if best is not None and j - i <= len(best):
                        break
                    sub = key[i:j]
                    if not (j in ends and sub != key and sub in pool and sub not in _GENERIC_NAMES):
                        continue
                    hangul = bool(_HANGUL.search(sub))
                    whole_word = i in starts and j == n  # the last word(s) of the query: "지역사회 획득 폐렴" → 폐렴
                    if i not in starts and (not hangul or len(sub) < 4):  # "근무력증" is not "무력증"
                        continue
                    if ((hangul and (whole_word or (j == n and (len(sub) >= 4 or 3 * len(sub) >= n))
                                     or (len(sub) >= 3 and len(sub) >= 0.6 * n)))
                            or (not hangul and len(sub) >= 8 and ((j == n and 3 * len(sub) >= n)
                                                                  or len(sub) >= 0.6 * n))):
                        best = sub
                        break
        return best

    def _resolve(self, name_or_id: str) -> int | None:
        if name_or_id in self.by_id:
            return self.by_id[name_or_id]
        if not name_or_id or not _n(name_or_id):
            return None
        same, back, spaced = self._variants(name_or_id)
        for key in same:
            i = self._best_hit(key)
            if i is not None:
                return i
        sup = self._superstring(same, "names")
        if sup:
            return self._best_hit(sup)
        for key in back:
            i = self._best_hit(key)
            if i is not None:
                return i
        sub = self._contained(spaced, self.names)
        if sub:
            return self._best_hit(sub)
        return self._fuzzy(same[0] if same else _n(name_or_id))

    def _fuzzy(self, key: str) -> int | None:
        """Char-bigram Dice over profile names. Short keys/abbreviations never go fuzzy; Korean matches below 0.7
        must share the first two characters (spelling variants, not "추간판 탈출증" → "승모판 탈출증")."""
        hangul = bool(_HANGUL.search(key))
        if len(key) < (5 if hangul else 6):  # "고환염전" (torsion) must not fuzz into "고환염" (orchitis)
            return None
        ix, nbig = self._bigram_index("names")
        qg = _bigrams(key)
        cand: dict[str, int] = defaultdict(int)
        for g in qg:
            for k in ix.get(g, ()):
                cand[k] += 1
        best, best_rank = None, (0.0, 0, "")
        lo = self.FUZZY_MIN if hangul else self.FUZZY_MIN_LATIN
        for k, inter in cand.items():
            s = 2 * inter / (len(qg) + nbig[k])
            if s < lo or s < best_rank[0] - 1e-9:
                continue
            if hangul and s < self.FUZZY_SURE and k[:2] != key[:2]:
                continue
            rank = (round(s, 9), -len(k), k)  # deterministic tie-break: shorter, then lexicographic
            if best is None or rank > best_rank:
                best, best_rank = k, rank
        return self._best_hit(best) if best is not None else None

    def profile(self, i: int) -> dict:
        d = self.diseases[i]

        def feats(field):
            return [{**self.term(t), "src": src} for t, src in sorted(
                d.get(field, []), key=lambda x: (-len(x[1]), -self.idf.get(x[0], 0)))]
        out = {
            "id": d["id"],
            "name_ko": d["names_ko"][0][0] if d["names_ko"] else "",
            "name_en": d["names_en"][0][0] if d["names_en"] else "",
            "names_ko": d["names_ko"], "names_en": d["names_en"], "codes": d["codes"],
            "symptoms": feats("symptoms"), "risk_factors": feats("risk"), "tests": feats("tests"),
            "questions": [{"ko": self.terms[t]["q"]["ko"], "en": self.terms[t]["q"]["en"], "src": ["DDXPlus"]}
                          for t, s in d["symptoms"] + d.get("risk", []) if "DDXPlus" in s and "q" in self.terms[t]],
        }
        for k in ("def", "summary_ko", "severity", "medlineplus", "parents"):
            if k in d:
                out["definition" if k == "def" else k] = d[k]
        srcs = {s for lst in ("names_ko", "names_en") for _, s in d[lst]}
        srcs |= {s for f in ("symptoms", "risk", "tests") for _, ss in d.get(f, []) for s in ss}
        out["sources"] = sorted(srcs)
        return out

    # ------------------------------------------------------------------ public API
    def lookup(self, name: str) -> dict | None:
        i = self._resolve(name)
        return self.profile(i) if i is not None else None

    def _compute_prior(self) -> None:
        """Source prior: conditions curated for common presentations (DDXPlus, MedlinePlus topics, KCD codes) and
        profiles described by several independent sources go up; rare-disease-only profiles go down."""
        self.prior = []
        w = self.PRIOR
        for d in self.diseases:
            ddx = any(s == "DDXPlus" for _, s in d["names_en"])
            rare = bool(d["codes"].get("orpha") or d["codes"].get("omim")) and not ddx and "medlineplus" not in d
            nsrc = len({x for _, ss in d["symptoms"] for x in ss})
            self.prior.append(max(0.1, 1.0 + w["ddx"] * ddx + w["mplus"] * ("medlineplus" in d)
                                  + w["kcd"] * bool(d["codes"].get("kcd")) + w["rare"] * rare
                                  + w.get("nsrc", 0.0) * max(0, nsrc - 1)))

    def _build_postings(self) -> None:
        """Symptom postings for candidates(). Weight of a feature = source-weighted count of the independent sources
        that list it (consensus ≈ typical feature); general terms implied by a feature get BACKOFF × its weight."""
        self.post: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.dlen: list[int] = []
        lens = []
        sw = self.SRC_W
        for i, d in enumerate(self.diseases):
            feats: dict[str, float] = {}
            for t, srcs in d["symptoms"] + d.get("risk", []):
                if t in self.stop:
                    continue
                feats[t] = max(feats.get(t, 0.0), sum(sw.get(s, 1.0) for s in srcs))
            implied: dict[str, float] = {}
            for t, w in feats.items() if self.BACKOFF > 0 else ():
                for g in self.implies.get(t, ()):
                    if g not in feats:
                        implied[g] = max(implied.get(g, 0.0), w * self.BACKOFF)
            for t, w in list(feats.items()) + list(implied.items()):
                self.post[t].append((i, w))
            self.dlen.append(len(d["symptoms"]) or len(feats))  # risk factors do not dilute symptom matches
            if feats:
                lens.append(self.dlen[-1])
        self.avg_len = sum(lens) / len(lens) if lens else 1.0
        n = len(lens) or 1
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def _sex_table(self) -> list[str]:
        """Per profile: "남성"/"여성" when every KCD code of the profile is restricted to that sex (KCD 성별구분)."""
        if self._sex is None:
            kcd, _ = self._kcd_table()
            out = []
            for d in self.diseases:
                sexes = {kcd.get(c, ([], "", ""))[2] for c, _ in d["codes"].get("kcd", [])}
                out.append({"X": "여성", "Y": "남성"}.get(next(iter(sexes)), "") if len(sexes) == 1 else "")
            self._sex = out
        return self._sex

    def _match(self, text: str) -> dict[str, str]:
        """tid → matched label for one finding text (no negation handling)."""
        return {t: lab for t, (lab, _span) in self._match_spans(text).items()}

    def _match_spans(self, text: str) -> dict[str, tuple[str, tuple]]:
        """tid → (matched label, span). Spans let candidates() treat terms matched on the same words as one concept
        ("안절부절못함" → restlessness, agitation, psychomotor agitation counts once)."""
        low = (text or "").lower()
        hits: dict[str, tuple[str, tuple]] = {}

        def take(lab: str, tid: str, span: tuple) -> None:
            if len(lab) > len(hits.get(tid, ("", ()))[0]):
                hits[tid] = (lab, span)
        for pat, en in kb_curated.REGEX_C:  # flexible phrasings; the span is consumed
            m = pat.search(low)
            if m and en in self.en_ix:
                take("~" + m.group(0), self.en_ix[en], ("k", m.start(), m.end()))
                low = low[:m.start()] + " " * (m.end() - m.start()) + low[m.end():]
        for w in kb_curated.BLOCK_WORDS:
            if w in low:
                low = low.replace(w, " " * len(w))
        # no-space string + word starts (a script change also starts a word: "38도", "bt38")
        chars: list[str] = []
        where: list[int] = []  # index in `low` of each char of fk
        starts: list[int] = []
        prev = None
        for x, ch in enumerate(low):
            cls = "h" if "가" <= ch <= "힣" else "l" if ("a" <= ch <= "z" or "0" <= ch <= "9") else None
            if cls is None:
                prev = None
                continue
            if cls != prev:
                starts.append(len(chars))
            chars.append(ch)
            where.append(x)
            prev = cls
        fk = "".join(chars)
        # Korean labels must start at a word start (or after a short laterality/acuity prefix): no "호흡음" → 서호흡
        for s in starts:
            pos = [s] + [s + len(p) for p in _KO_PREFIX if fk.startswith(p, s) and len(fk) > s + len(p) + 1]
            for p in pos:
                for lab, tid in self._ko_ix.get(fk[p:p + 2], ()):
                    if fk.startswith(lab, p):
                        take(lab, tid, ("k", where[p], where[p + len(lab) - 1] + 1))
        # a short finding may be the label minus its last char ("구역" → "구역질")
        if 2 <= len(fk) <= 4:
            for lab, tid in self._ko_ix.get(fk[:2], ()):
                if len(lab) == len(fk) + 1 and lab.startswith(fk):
                    take(lab, tid, ("k", where[0], where[-1] + 1))
        fs = _spaced(low)
        for wd in set(fs.split()):
            for lab, tid in self._en_first.get(wd, ()):
                x = fs.find(lab)
                if x >= 0:
                    take(lab, tid, ("e", x, x + len(lab)))
        for en in kb_curated.lab_terms(low):
            tid = self.en_ix.get(en)
            if tid:
                take("#" + en, tid, ("#", en))
        for tid in [t for t in hits if t in self.stop]:
            del hits[tid]
        return hits

    def _groups(self, text: str) -> list[list[str]]:
        """Matched terms of one finding grouped into concepts (terms whose matched spans overlap)."""
        hits = self._match_spans(text)
        items = sorted(((span, t) for t, (_lab, span) in hits.items() if t in self.post),
                       key=lambda x: (x[0][0], x[0][1:]))
        groups: list[list[str]] = []
        last_space, last_end = None, -1
        for span, t in items:
            if span[0] != "#" and span[0] == last_space and span[1] < last_end:
                groups[-1].append(t)
                last_end = max(last_end, span[2])
            else:
                groups.append([t])
                last_space, last_end = span[0], (span[2] if span[0] != "#" else -1)
        return groups

    def match_terms(self, finding: str, allow_negated: bool = False) -> list[str]:
        """Term ids mentioned in one finding text (negated findings match nothing unless allow_negated)."""
        if not finding or (not allow_negated and _NEG.search(finding.lower())):
            return []
        return list(self._match(finding))

    def candidates(self, findings: list[str], k: int = 10, negatives: list[str] | None = None,
                   sex: str | None = None, age: float | None = None) -> list[dict]:
        """Diseases ranked by BM25 over the matched symptom terms × finding coverage × source prior.

        negatives: findings reported absent (penalise diseases listing them); sex ("남성"/"여성"): drops profiles whose
        KCD codes are all restricted to the other sex; age (years): down-weights pediatric/geriatric-named profiles."""
        matched: dict[str, list[str]] = defaultdict(list)  # tid → findings
        qw: dict[str, float] = {}
        concepts: list[list[str]] = []  # one entry per concept: its terms (+ implied general terms)
        n_q = 0
        for f in findings or []:
            if not f or _NEG.search(f.lower()):
                continue
            groups = self._groups(f)
            n_q += bool(groups)
            for grp in groups:
                fresh = [t for t in grp if t not in qw]
                if not fresh:  # the same concept was already reported by an earlier finding
                    for t in grp:
                        matched[t].append(f)
                    continue
                concept = list(grp)
                for t in grp:
                    matched[t].append(f)
                    qw[t] = 1.0
                for t in grp if self.QX > 0 else ():  # query-side backoff: "우하복부 통증" also counts as 하복부 통증
                    for g in self.implies.get(t, ()):
                        if g in self.post and g not in qw:
                            matched[g].append(f)
                            qw[g] = self.QX
                            concept.append(g)
                concepts.append(concept)
        neg = {t for f in negatives or [] for t, lab in self._match(_strip_neg(f)).items()
               if not lab.startswith("#")} - set(qw)  # values ("#fever" from 36.5℃) never negate
        scores: dict[int, float] = defaultdict(float)
        hits: dict[int, set[str]] = defaultdict(set)
        k1, b = self.K1, self.B
        for concept in concepts:  # a concept adds its best-matching term per disease (no synonym double counting)
            best: dict[int, float] = {}
            for t in concept:
                for i, w in self.post.get(t, ()):
                    norm = 1 - b + b * self.dlen[i] / self.avg_len
                    c = qw[t] * self.idf[t] * (k1 + 1) * w / (w + k1 * norm)
                    if c > best.get(i, 0.0):
                        best[i] = c
                    hits[i].add(t)
            for i, c in best.items():
                scores[i] += c
        negpen: dict[int, float] = defaultdict(float)
        for t in neg:
            for i, w in self.post.get(t, ()):
                if i in scores:
                    negpen[i] += self.idf[t] * min(w, 2.0)
        sex_t = self._sex_table() if sex in ("남성", "여성") else None
        out = []
        a = self.COVER_A
        for i, s in scores.items():
            if sex_t is not None and sex_t[i] and sex_t[i] != sex:
                continue
            cover = len({f for t in hits[i] for f in matched[t]}) / max(n_q, 1)
            score = s * (a + (1 - a) * cover) * self.prior[i] - self.NEG_W * negpen[i]
            if age is not None and self._age[i] and ((self._age[i] == "child") != (age < 18)):
                score *= self.AGE_PEN
            out.append((score, i))
        out.sort(key=lambda x: (-x[0], x[1]))
        res = []
        for score, i in out[:k]:
            d = self.diseases[i]
            direct = {t for t, _ in d["symptoms"]} | {t for t, _ in d.get("risk", [])}
            res.append({
                "id": d["id"], "name_ko": d["names_ko"][0][0] if d["names_ko"] else d["names_en"][0][0],
                "name_en": d["names_en"][0][0] if d["names_en"] else "", "score": round(score, 3),
                "kcd": [c for c, _ in d["codes"].get("kcd", [])][:2],
                "matched": [{**self.term(t), "finding": matched[t][0]}
                            for t in sorted(hits[i], key=lambda t: (t not in direct, -self.idf[t]))],
                "sources": sorted({x for t, ss in d["symptoms"] + d.get("risk", [])
                                   if t in hits[i] or self.implies.get(t, set()) & hits[i] for x in ss}),
            })
        return res

    def discriminators(self, dx_a: str, dx_b: str, n: int = 6) -> dict | None:
        ia, ib = self._resolve(dx_a), self._resolve(dx_b)
        if ia is None or ib is None:
            return None
        a, b = self.diseases[ia], self.diseases[ib]

        def fs(d, field):
            return {t: s for t, s in d.get(field, [])}
        out = {"a": self.profile(ia)["name_ko"] or dx_a, "b": self.profile(ib)["name_ko"] or dx_b}
        for field, key in (("symptoms", "symptoms"), ("risk", "risk_factors"), ("tests", "tests")):
            fa, fb = fs(a, field), fs(b, field)
            order = lambda ts: sorted(ts, key=lambda t: -self.idf.get(t, 0))[:n]
            out[f"{key}_a_only"] = [{**self.term(t), "src": fa[t]} for t in order(set(fa) - set(fb))]
            out[f"{key}_b_only"] = [{**self.term(t), "src": fb[t]} for t in order(set(fb) - set(fa))]
            out[f"{key}_shared"] = [self.term(t) for t in order(set(fa) & set(fb))]
        return out

    def _kcd_table(self):
        if self._kcd is None:
            self._kcd, self._kcd_names = {}, {}
            with gzip.open(self.kb_dir / "kcd.tsv.gz", "rt", encoding="utf-8") as f:
                next(f)
                for line in f:
                    code, ko, en, sex = line.rstrip("\n").split("\t")
                    names = ko.split("|") if ko else []
                    self._kcd[code] = (names, en, sex)
                    for nm in names:
                        self._kcd_names.setdefault(_n(nm), code)
                    for nm in names:
                        self._kcd_names.setdefault(_n(_QUAL.sub("", nm)), code)
        return self._kcd, self._kcd_names

    def _kcd_result(self, code: str, how: str, profile: dict | None = None) -> dict:
        kcd, _ = self._kcd_table()
        names, en, sex = kcd.get(code, ([], "", ""))
        name = (profile or {}).get("name_ko") or (names[0] if names else "")
        res = {"name": name, "code": _fmt_code(code), "code_system": "KCD", "kcd_name": names[0] if names else "",
               "name_en": (profile or {}).get("name_en") or en, "match": how, "source": "KCD (HIRA 상병마스터)"}
        if profile:
            res["id"] = profile["id"]
        if sex:  # KCD 성별구분: X = female only, Y = male only
            res["sex"] = {"X": "여성", "Y": "남성"}.get(sex, sex)
        return res

    def normalize_diagnosis(self, text: str) -> dict | None:
        """Standard Korean name + KCD code. Order: explicit code → exact KCD/profile name (spelling variants) →
        same with qualifiers dropped ("backoff") → longest known name inside the text ("contained") → fuzzy."""
        t = (text or "").strip()
        if not t:
            return None
        kcd, kcd_names = self._kcd_table()
        m = _CODE.search(t.upper())
        if m and (m.group(1) + m.group(2) + (m.group(3) or "")) in kcd:
            return self._kcd_result(m.group(1) + m.group(2) + (m.group(3) or ""), "code")
        same, back, spaced = self._variants(t)

        def from_kcd(key: str, how: str) -> dict:
            code = kcd_names[key]
            i = self._best_hit(key)
            same_code = i is not None and code in [c for c, _ in self.diseases[i]["codes"].get("kcd", [])]
            prof = self.profile(i) if same_code else None
            res = self._kcd_result(code, how, prof)
            res["name"] = kcd[code][0][0] if kcd[code][0] else res["name"]
            return res

        def from_profile(i: int, how: str) -> dict:
            p = self.profile(i)
            code = _best_kcd(p["codes"])
            if code:
                return self._kcd_result(code, how, p)
            return {"name": p["name_ko"] or p["name_en"], "code": "", "code_system": "", "name_en": p["name_en"],
                    "id": p["id"], "match": how, "source": ", ".join(p["sources"])}
        for keys, how_k, how_p in ((same, "kcd_exact", "profile"), (back, "kcd_backoff", "profile_backoff")):
            for key in keys:
                if key in kcd_names:
                    return from_kcd(key, how_k)
            for key in keys:
                i = self._best_hit(key)
                if i is not None:
                    return from_profile(i, how_p)
            if keys is same:  # a slightly longer official name containing the text ("어깨의 유착성 관절낭염")
                sup_k, sup_p = self._superstring(same, "kcd"), self._superstring(same, "names")
                if sup_k and len(sup_k) <= len(sup_p or sup_k):
                    return from_kcd(sup_k, "superstring")
                if sup_p:
                    return from_profile(self._best_hit(sup_p), "superstring")
        sub_k = self._contained(spaced, kcd_names)
        sub_p = self._contained(spaced, self.names)
        if sub_k and len(sub_k) >= len(sub_p or ""):
            return from_kcd(sub_k, "contained")
        if sub_p:
            return from_profile(self._best_hit(sub_p), "contained")
        i = self._resolve(t)
        return from_profile(i, "fuzzy") if i is not None else None

    def render_for_prompt(self, findings: list[str] | None = None, dx: list[str] | None = None, k: int = 3,
                          max_chars: int = 800) -> str:
        rows: list[tuple[dict, list[str]]] = []
        if dx:
            for name in dx[:k]:
                i = self._resolve(name)
                if i is not None:
                    rows.append((self.profile(i), []))
        elif findings:
            for c in self.candidates(findings, k=k):
                rows.append((self.profile(self.by_id[c["id"]]), [m["ko"] for m in c["matched"]]))
        if not rows:
            return ""
        lines = ["[지식베이스 참고: 출처 있는 질환 정보. 확진 근거가 아니라 문진·검사 계획용]"]
        for n, (p, hit) in enumerate(rows, 1):
            code = _best_kcd(p["codes"])
            head = f"{n}. {p['name_ko'] or p['name_en']}" + (f" ({_fmt_code(code)})" if code else "")
            parts = []
            if hit:
                parts.append("일치: " + ", ".join(hit[:4]))
            sx = [s["ko"] for s in p["symptoms"] if s["ko"] not in hit][:5]
            if sx:
                parts.append("전형 증상: " + ", ".join(sx))
            ts = [s["ko"] for s in p["tests"] if s["en"].lower() not in _GENERIC_TESTS][:3]
            if ts:
                parts.append("검사: " + ", ".join(ts))
            src = sorted({SRC_SHORT.get(x, x) for f in ("symptoms", "tests") for s in p[f][:5] for x in s["src"]})
            lines.append(head + " — " + " | ".join(parts) + (f" [{', '.join(src)}]" if src else ""))
        if len(rows) >= 2:
            dsc = self.discriminators(rows[0][0]["id"], rows[1][0]["id"], n=3)
            if dsc and (dsc["symptoms_a_only"] or dsc["symptoms_b_only"]):
                lines.append(f"감별 {dsc['a']} vs {dsc['b']}: " +
                             f"{dsc['a']}만 — {', '.join(x['ko'] for x in dsc['symptoms_a_only']) or '없음'}; " +
                             f"{dsc['b']}만 — {', '.join(x['ko'] for x in dsc['symptoms_b_only']) or '없음'}")
        out = ""
        for line in lines:
            if len(out) + len(line) + 1 > max_chars:
                break
            out += ("\n" if out else "") + line
        return out


def _best_kcd(codes: dict) -> str:
    """Most supported KCD category across sources (DO/Wikidata/DDXPlus ICD-10 codes), then its shortest code."""
    kcd = [c for c, _ in codes.get("kcd", [])]
    if not kcd:
        return ""
    votes: dict[str, int] = defaultdict(int)
    for c, _ in codes.get("icd10", []):
        votes[re.sub(r"[^0-9A-Z]", "", c.upper())[:3]] += 1
    return min(kcd, key=lambda c: (-votes.get(c[:3], 0), len(c), c))


def _fmt_code(code: str) -> str:
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 and "." not in code else code


_KB: KnowledgeBase | None = None
_LOCK = threading.Lock()


def get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        with _LOCK:
            if _KB is None:
                _KB = KnowledgeBase()
    return _KB


def available() -> bool:
    return (KB_DIR / "kb.json.gz").exists()


def lookup(name: str) -> dict | None:
    return get_kb().lookup(name)


def candidates(findings: list[str], k: int = 10, negatives: list[str] | None = None, sex: str | None = None,
               age: float | None = None) -> list[dict]:
    return get_kb().candidates(findings, k, negatives, sex=sex, age=age)


def discriminators(dx_a: str, dx_b: str, n: int = 6) -> dict | None:
    return get_kb().discriminators(dx_a, dx_b, n)


def normalize_diagnosis(text: str) -> dict | None:
    return get_kb().normalize_diagnosis(text)


def render_for_prompt(findings: list[str] | None = None, dx: list[str] | None = None, k: int = 3,
                      max_chars: int = 800) -> str:
    return get_kb().render_for_prompt(findings, dx, k, max_chars)
