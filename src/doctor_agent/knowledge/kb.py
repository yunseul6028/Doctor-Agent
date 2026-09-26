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
"""
from __future__ import annotations

import gzip
import json
import math
import re
import threading
from collections import defaultdict
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
SRC_SHORT = {"DDXPlus": "DDXPlus", "DO": "DO", "WD": "Wikidata", "KCD": "KCD", "MedlinePlus": "MedlinePlus",
             "MedlinePlus+LLM": "MedlinePlus", "LLM": "LLM번역"}

_NON = re.compile(r"[^0-9a-z가-힣]")
_EN_SPACE = re.compile(r"[^0-9a-z가-힣]+")
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_CODE = re.compile(r"\b([A-Z])\s?(\d{2})(?:\.?(\d{1,2}))?\b")
# negated finding: Korean negation at the end ("발열 없음", "기침은 부인") or English negation words
_NEG = re.compile(r"((없음|없다|없어요?|없습니다|음성|부인|부인함|아님|아니요|정상|않음|않다|않아요)[.\s]*$)"
                  r"|\b(no|not|denies|denied|negative|absent)\b")
_HANGUL = re.compile(r"[가-힣]")
# KCD qualifiers stripped for an extra name key ("달리 분류되지 않은 세균성 수막염" → "세균성 수막염")
_QUAL = re.compile(r"^(달리 분류되지 않은|상세불명의|상세불명 병원체의|기타)\s*|\s*(NOS|NEC)$")
_GENERIC_TESTS = {"medical history", "physical examination", "history", "physical exam", "medical diagnosis",
                  "diagnosis", "symptom", "symptoms"}
# generic labels that would match almost any complaint
_GENERIC = {"통증", "고통", "증상", "징후", "불편감", "이상", "pain", "aches", "ache", "symptom", "symptoms", "sign"}


def _n(s: str) -> str:
    return _NON.sub("", (s or "").lower())


def _spaced(s: str) -> str:
    return " " + _EN_SPACE.sub(" ", (s or "").lower()).strip() + " "


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


class KnowledgeBase:
    K1, B = 1.2, 0.3  # BM25 parameters for candidates()
    PRIOR = {"ddx": 0.5, "mplus": 0.4, "kcd": 0.15, "rare": -0.25}

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
        # symptom postings for candidates()
        # weight of a feature = number of independent sources that list it (consensus ≈ typical feature)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dlen: list[int] = []
        lens = []
        for i, d in enumerate(self.diseases):
            feats: dict[str, int] = {}
            for t, srcs in d["symptoms"] + d.get("risk", []):
                feats[t] = max(feats.get(t, 0), len(srcs))
            for t, w in feats.items():
                self.post[t].append((i, w))
            self.dlen.append(len(d["symptoms"]) or len(feats))  # risk factors do not dilute symptom matches
            if feats:
                lens.append(self.dlen[-1])
        self.avg_len = sum(lens) / len(lens) if lens else 1.0
        n = len(lens) or 1
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}
        # term labels for finding → term matching: (label, tid, is_english)
        self.labels: list[tuple[str, str, bool]] = []
        for tid, t in self.terms.items():
            for lab in {t["en"], t["ko"], *t["syn"]}:
                if not lab or lab.strip().lower() in _GENERIC:
                    continue
                if _HANGUL.search(lab):
                    key = _n(lab)
                    if len(key) >= 2:
                        self.labels.append((key, tid, False))
                    if len(key) >= 3 and key[-1] in "증감":  # 다뇨증 → 다뇨, 오심감 → 오심
                        self.labels.append((key[:-1], tid, False))
                else:
                    key = _spaced(lab)
                    if len(key.strip()) >= 3:
                        self.labels.append((key, tid, True))
        # prior: conditions curated for common presentations (DDXPlus, MedlinePlus topics) up, rare-only down
        self.prior = []
        for d in self.diseases:
            ddx = any(s == "DDXPlus" for _, s in d["names_en"])
            rare = bool(d["codes"].get("orpha") or d["codes"].get("omim")) and not ddx and "medlineplus" not in d
            w = self.PRIOR
            self.prior.append(1.0 + w["ddx"] * ddx + w["mplus"] * ("medlineplus" in d)
                              + w["kcd"] * bool(d["codes"].get("kcd")) + w["rare"] * rare)
        self._fuzzy: dict[str, set[str]] | None = None
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

    def _fuzzy_index(self) -> dict[str, set[str]]:
        if self._fuzzy is None:
            ix: dict[str, set[str]] = defaultdict(set)
            for key in self.names:
                for g in _bigrams(key):
                    ix[g].add(key)
            self._fuzzy = ix
        return self._fuzzy

    def _resolve(self, name_or_id: str) -> int | None:
        if name_or_id in self.by_id:
            return self.by_id[name_or_id]
        if not name_or_id or not _n(name_or_id):
            return None
        key = _n(_PAREN.sub("", name_or_id)) or _n(name_or_id)
        hits = self.names.get(key)
        if hits:
            return max(hits, key=lambda h: (h[1], self._richness(h[0])))[0]
        # fuzzy: char-bigram Dice over names sharing a bigram
        qg = _bigrams(key)
        ix = self._fuzzy_index()
        cand: dict[str, int] = defaultdict(int)
        for g in qg:
            for k in ix.get(g, ()):
                cand[k] += 1
        best, best_s = None, 0.0
        for k, inter in cand.items():
            s = 2 * inter / (len(qg) + len(_bigrams(k)))
            if len(key) >= 3 and key in k:  # query contained in a longer official name
                s = max(s, 0.7 + 0.3 * len(key) / len(k))
            if s > best_s or (s == best_s and best is not None and len(k) < len(best)):
                best, best_s = k, s
        if best is None or best_s < 0.6:
            return None
        return max(self.names[best], key=lambda h: (h[1], self._richness(h[0])))[0]

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

    def match_terms(self, finding: str) -> list[str]:
        """Term ids mentioned in one finding text (negated findings match nothing)."""
        if not finding or _NEG.search(finding.lower()):
            return []
        fk, fs = _n(finding), _spaced(finding)
        hits: dict[str, str] = {}
        for lab, tid, is_en in self.labels:
            # Korean: label inside the finding, or the finding is the label minus one trailing char (구역 → 구역질)
            if (lab in fs) if is_en else (lab in fk or (len(fk) >= 2 and len(lab) == len(fk) + 1 and lab.startswith(fk))):
                if len(lab) > len(hits.get(tid, "")):
                    hits[tid] = lab
        return list(hits)

    def candidates(self, findings: list[str], k: int = 10, negatives: list[str] | None = None) -> list[dict]:
        matched: dict[str, list[str]] = defaultdict(list)  # tid → findings
        n_q = 0
        for f in findings or []:
            tids = self.match_terms(f)
            n_q += bool(tids)
            for t in tids:
                matched[t].append(f)
        neg = {t for f in negatives or [] for t in self.match_terms(f)}
        scores: dict[int, float] = defaultdict(float)
        hits: dict[int, set[str]] = defaultdict(set)
        k1, b = self.K1, self.B
        for t in matched:
            for i, w in self.post.get(t, ()):
                norm = 1 - b + b * self.dlen[i] / self.avg_len
                scores[i] += self.idf[t] * (k1 + 1) * w / (w + k1 * norm)
                hits[i].add(t)
        out = []
        for i, s in scores.items():
            d = self.diseases[i]
            cover = len({f for t in hits[i] for f in matched[t]}) / max(n_q, 1)
            feats = {t for t, _ in d["symptoms"]} | {t for t, _ in d.get("risk", [])}
            penalty = sum(self.idf.get(t, 0) for t in neg & feats) * 0.5
            score = s * (0.4 + 0.6 * cover) * self.prior[i] - penalty
            out.append((score, i))
        out.sort(key=lambda x: (-x[0], x[1]))
        res = []
        for score, i in out[:k]:
            d = self.diseases[i]
            res.append({
                "id": d["id"], "name_ko": d["names_ko"][0][0] if d["names_ko"] else d["names_en"][0][0],
                "name_en": d["names_en"][0][0] if d["names_en"] else "", "score": round(score, 3),
                "kcd": [c for c, _ in d["codes"].get("kcd", [])][:2],
                "matched": [{**self.term(t), "finding": matched[t][0]} for t in sorted(hits[i], key=lambda t: -self.idf[t])],
                "sources": sorted({x for t, ss in d["symptoms"] + d.get("risk", []) if t in hits[i] for x in ss}),
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
        t = (text or "").strip()
        if not t:
            return None
        kcd, kcd_names = self._kcd_table()
        m = _CODE.search(t.upper())
        if m and (m.group(1) + m.group(2) + (m.group(3) or "")) in kcd:
            return self._kcd_result(m.group(1) + m.group(2) + (m.group(3) or ""), "code")
        key = _n(_PAREN.sub("", t)) or _n(t)
        if key in kcd_names:
            code = kcd_names[key]
            i = self._resolve(t)
            prof = self.profile(i) if i is not None and code in [c for c, _ in self.diseases[i]["codes"].get("kcd", [])] else None
            res = self._kcd_result(code, "kcd_exact", prof)
            res["name"] = kcd[code][0][0] if kcd[code][0] else res["name"]
            return res
        i = self._resolve(t)
        if i is not None:
            p = self.profile(i)
            code = _best_kcd(p["codes"])
            if code:
                return self._kcd_result(code, "profile", p)
            return {"name": p["name_ko"] or p["name_en"], "code": "", "code_system": "", "name_en": p["name_en"],
                    "id": p["id"], "match": "profile", "source": ", ".join(p["sources"])}
        return None

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


def candidates(findings: list[str], k: int = 10, negatives: list[str] | None = None) -> list[dict]:
    return get_kb().candidates(findings, k, negatives)


def discriminators(dx_a: str, dx_b: str, n: int = 6) -> dict | None:
    return get_kb().discriminators(dx_a, dx_b, n)


def normalize_diagnosis(text: str) -> dict | None:
    return get_kb().normalize_diagnosis(text)


def render_for_prompt(findings: list[str] | None = None, dx: list[str] | None = None, k: int = 3,
                      max_chars: int = 800) -> str:
    return get_kb().render_for_prompt(findings, dx, k, max_chars)
