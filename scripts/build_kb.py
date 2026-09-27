"""Build the shipped medical knowledge base (data/kb/) from openly licensed sources. Offline, dev only.

    source .venv/bin/activate
    python scripts/build_kb.py                 # download missing inputs, fill the LLM cache, assemble data/kb/
    python scripts/build_kb.py --no-llm        # assemble from cached LLM outputs only (no API calls)
    python scripts/build_kb.py --refresh       # re-download every input

Inputs (git-ignored) go to data/external/kb_raw/. Every source and its license is listed in SOURCES below and in
docs/licenses.md. The Human Phenotype Ontology file is deliberately NOT used (its license forbids altering file
content, and our KB reorganises and translates it); Orphanet's CC BY 4.0 HPO-coded phenotype annotations are used and
re-keyed to KB terms (map_orpha / orpha_terms). The HIRA patients-per-KCD-code table (KOGL type 1) becomes
data/kb/kcd3_prev.tsv.gz for a prevalence prior. Neither step uses an LLM.

LLM use (offline only, KB_LLM_* in .env, falls back to LLM_*): Korean labels for English terms, short Korean question
wording for DDXPlus evidences, and extraction of symptoms/tests + a one-line Korean summary from public-domain
MedlinePlus summaries. Every LLM output is cached in data/labels/kb_llm_cache.json (so rebuilding is deterministic
and needs no API calls) and the prompts/model/date are recorded in data/labels/kb_build_meta.json.
Inference code (src/doctor_agent/knowledge/kb.py) never calls an LLM or the network.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import html
import io
import json
import re
import sys
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from doctor_agent.knowledge import kb_tests  # noqa: E402  (curated test-finding table; stdlib only)

RAW = ROOT / "data/external/kb_raw"
OUT = ROOT / "data/kb"
CACHE = ROOT / "data/labels/kb_llm_cache.json"
META = ROOT / "data/labels/kb_build_meta.json"
UA = "DoctorAgentKB-build/0.1 (offline research build)"

MPLUS_DATE = "2026-09-26"  # MedlinePlus publishes a dated XML daily; pinned for reproducibility
SOURCES = {
    "DDXPlus": {
        "title": "DDXPlus Dataset (English) knowledge files: release_conditions.json, release_evidences.json",
        "license": "CC BY 4.0",
        "license_url": "https://api.figshare.com/v2/articles/22687585",
        "url": "https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585",
        "attribution": "Fansi Tchango A, et al. DDXPlus: A New Dataset For Automatic Medical Diagnosis. NeurIPS 2022 "
                       "Datasets and Benchmarks. Changes: evidences reduced to short terms and translated to Korean.",
    },
    "KCD": {
        "title": "건강보험심사평가원_상병마스터 (KCD disease code master)",
        "license": "KOGL Type 1 (공공누리 제1유형: 출처표시)",
        "license_url": "https://www.data.go.kr/data/15067467/fileData.do",
        "url": "https://www.data.go.kr/data/15067467/fileData.do",
        "attribution": "출처: 건강보험심사평가원, 상병마스터 (공공데이터포털). 필요한 열만 발췌.",
    },
    "DO": {
        "title": "Human Disease Ontology (doid.obo)",
        "license": "CC0 1.0",
        "license_url": "https://github.com/DiseaseOntology/HumanDiseaseOntology/blob/main/LICENSE",
        "url": "https://github.com/DiseaseOntology/HumanDiseaseOntology",
        "attribution": "Schriml LM, et al. Human Disease Ontology. disease-ontology.org",
    },
    "WD": {
        "title": "Wikidata structured data (P780 symptoms, P923 medical examinations, P699/P494/P486 ids, en/ko labels)",
        "license": "CC0 1.0 (structured data namespaces)",
        "license_url": "https://www.wikidata.org/wiki/Wikidata:Licensing",
        "url": "https://query.wikidata.org/",
        "attribution": "Wikidata (CC0).",
    },
    "MedlinePlus": {
        "title": "MedlinePlus Health Topics XML (health topic summaries only)",
        "license": "Public domain (U.S. government work; health topic summaries). A.D.A.M./ASHP content not used",
        "license_url": "https://medlineplus.gov/about/using/usingcontent/",
        "url": f"https://medlineplus.gov/xml/mplus_topics_{MPLUS_DATE}.xml",
        "attribution": "Courtesy of MedlinePlus from the National Library of Medicine. Symptoms/tests extracted and "
                       "summarised in Korean by an LLM (see data/labels/kb_build_meta.json).",
    },
    "Orphanet": {
        "title": "Orphadata Science: rare disease phenotypes with frequency classes (en_product4.xml) and "
                 "nomenclature + cross-references (en_product1.xml)",
        "license": "CC BY 4.0",
        "license_url": "https://sciences.orphadata.com/phenotypes/",
        "url": "https://www.orphadata.com/data/xml/en_product4.xml",
        "attribution": "Orphadata Science: Free access data from Orphanet. (c) INSERM 1999. Available on "
                       "http://sciences.orphadata.com/. Data version 2026-06-23 (release July 2026). Changes: "
                       "phenotype annotations re-keyed to KB terms (via English label or the Wikidata HPO-id "
                       "bridge), frequency classes kept as codes; Orphanet disorder names/synonyms and exact "
                       "ICD-10/OMIM mappings used for profiles. The Human Phenotype Ontology file itself is not used.",
    },
    "HIRA-stats": {
        "title": "건강보험심사평가원_주부상병(3단) 성별 연령군별 건강보험 진료 통계_20251231 (patients per KCD "
                 "3-character code x sex x 5-year age group, 2025)",
        "license": "KOGL Type 1 (공공누리 제1유형: 출처표시)",
        "license_url": "https://www.data.go.kr/data/15118806/fileData.do",
        "url": "https://www.data.go.kr/data/15118806/fileData.do",
        "attribution": "출처: 건강보험심사평가원, 주부상병(3단) 성별 연령군별 건강보험 진료 통계 (공공데이터포털). "
                       "환자수만 발췌해 질환 빈도 사전확률로 사용.",
    },
    "curated": {
        "title": "Our own tables: CURATED_SYN (build time), src/doctor_agent/knowledge/kb_curated.py (runtime "
                 "synonyms, phrase patterns, generic-term stop list, vital/lab thresholds, diagnosis-name spelling "
                 "pairs) and src/doctor_agent/knowledge/kb_tests.py (test/lab/imaging finding -> disease links)",
        "license": "Self-authored",
        "license_url": "",
        "url": "",
        "attribution": "Written by the team. kb_curated.py only maps wording to KB entries; kb_tests.py adds "
                       "test-result -> disease links (field findings_from_tests), each citing a guideline/review "
                       "(PMID, see meta.test_refs) or marked 'textbook' (unverified). No text copied from any source.",
    },
    "LLM": {
        "title": "Offline LLM processing (Korean labels, extraction from MedlinePlus)",
        "license": "Our derived annotations",
        "license_url": "",
        "url": "",
        "attribution": "Generated offline; prompts, model and date in data/labels/kb_build_meta.json.",
    },
}

WD_QUERIES = {
    "wd_symptoms.tsv": "SELECT ?d ?s WHERE { ?d wdt:P780 ?s }",
    "wd_exams.tsv": "SELECT ?d ?t WHERE { ?d wdt:P923 ?t }",
    "wd_ids.tsv": ("SELECT ?d ?doid ?icd ?mesh WHERE { { ?d wdt:P699 ?doid } UNION { ?d wdt:P780 [] } "
                   "OPTIONAL { ?d wdt:P494 ?icd } OPTIONAL { ?d wdt:P486 ?mesh } OPTIONAL { ?d wdt:P699 ?doid } }"),
    "wd_labels.tsv": ("SELECT ?e ?p ?l WHERE { { ?e wdt:P699 [] } UNION { ?e wdt:P780 [] } UNION { [] wdt:P780 ?e } "
                      "UNION { [] wdt:P923 ?e } VALUES ?p { rdfs:label skos:altLabel } ?e ?p ?l . "
                      "FILTER(LANG(?l) = \"en\" || LANG(?l) = \"ko\") }"),
    # bridges for Orphanet: Orphanet disorder ids of Wikidata diseases, and HPO ids of Wikidata items (CC0) with their
    # labels (so phenotype annotations can be re-keyed to KB/Wikidata terms without using the HPO file)
    "wd_orpha.tsv": "SELECT ?d ?orpha WHERE { ?d wdt:P1550 ?orpha }",
    "wd_hpo.tsv": "SELECT ?e ?hpo WHERE { ?e wdt:P3841 ?hpo }",
    "wd_hpo_labels.tsv": ("SELECT ?e ?p ?l WHERE { ?e wdt:P3841 [] . VALUES ?p { rdfs:label skos:altLabel } ?e ?p ?l . "
                          "FILTER(LANG(?l) = \"en\" || LANG(?l) = \"ko\") }"),
}


# ------------------------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------------------------

def _get(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _hira(pk: str = "15067467", head: str = "상병기호") -> bytes:
    """File download from data.go.kr (HIRA datasets; the page must state 공공누리 제1유형)."""
    page = f"https://www.data.go.kr/data/{pk}/fileData.do"
    body = _get(page).decode("utf-8", "replace")
    m = re.search(rf"fn_fileDataDown\('{pk}', '([^']+)'", body)
    if not m:
        raise RuntimeError("HIRA page layout changed: download id not found")
    if "제 1유형" not in body:
        raise RuntimeError(f"data.go.kr {pk}: license is no longer KOGL type 1")
    form = urllib.parse.urlencode({"publicDataDetailPk": m.group(1), "publicDataPk": pk, "atchFileId": "",
                                   "fileDetailSn": "1", "publicDataTyCode": "PR0051"}).encode()
    info = json.loads(_get("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do", form, {"Referer": page}))
    url = (f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={info['atchFileId']}"
           f"&fileDetailSn={info['fileDetailSn']}&dataNm=kcd")
    raw = _get(url, headers={"Referer": page})
    if not raw[:60].decode("cp949", "replace").startswith(head):
        raise RuntimeError(f"HIRA download {pk} is not the expected CSV")
    return raw


def download(refresh: bool) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    targets = {
        "ddx_conditions.json": lambda: _get("https://ndownloader.figshare.com/files/62561569"),
        "ddx_evidences.json": lambda: _get("https://ndownloader.figshare.com/files/40278013"),
        "hira_kcd.csv": _hira,
        "doid.obo": lambda: _get("https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"),
        "mplus_topics.xml": lambda: _get(SOURCES["MedlinePlus"]["url"]),
        "orpha_product4.xml": lambda: _get("https://www.orphadata.com/data/xml/en_product4.xml"),
        "orpha_product1.xml": lambda: _get("https://www.orphadata.com/data/xml/en_product1.xml"),
        "hira_kcd3_stats.csv": lambda: _hira("15118806", "진료년도"),
    }
    for name, q in WD_QUERIES.items():
        targets[name] = (lambda q=q: _get("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q}),
                                          headers={"Accept": "text/tab-separated-values"}))
    files = {}
    for name, fetch in targets.items():
        p = RAW / name
        if refresh or not p.exists():
            print("download", name)
            p.write_bytes(fetch())
        files[name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size,
                       "retrieved": dt.date.fromtimestamp(p.stat().st_mtime).isoformat()}
    return files


# ------------------------------------------------------------------------------------------------
# Parsers
# ------------------------------------------------------------------------------------------------

def norm_en(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", s.lower())).strip()


def parse_do() -> tuple[dict, str]:
    terms, cur, version = {}, None, ""
    for line in (RAW / "doid.obo").read_text(encoding="utf-8").splitlines():
        if line.startswith("data-version:"):
            version = line.split(":", 1)[1].strip()
        if line.startswith("["):
            cur = {"syn": [], "xref": [], "is_a": []} if line == "[Term]" else None
            if cur is not None:
                terms[id(cur)] = cur
            continue
        if cur is None or ": " not in line:
            continue
        k, v = line.split(": ", 1)
        if k == "id":
            cur["id"] = v
        elif k == "name":
            cur["name"] = v
        elif k == "def":
            cur["def"] = re.sub(r'\\"', '"', re.match(r'"(.*)"\s*\[', v).group(1)) if re.match(r'"(.*)"\s*\[', v) else ""
        elif k == "synonym":
            m = re.match(r'"(.*)" (EXACT|RELATED|NARROW|BROAD)', v)
            if m and m.group(2) == "EXACT":
                cur["syn"].append(m.group(1))
        elif k == "xref":
            cur["xref"].append(v.split()[0])
        elif k == "is_a":
            cur["is_a"].append(v.split()[0])
        elif k == "is_obsolete" and v == "true":
            cur["obsolete"] = True
    out = {}
    for t in terms.values():
        if t.get("obsolete") or "id" not in t or "name" not in t or not t["id"].startswith("DOID:"):
            continue
        t["has_symptom"] = [s.strip() for s in re.findall(r"has_symptom ([^,;.()]+?)(?=,| and |\.|;|$)", t.get("def", ""))]
        t["def"] = re.sub(r"\b(has_symptom|results_in|located_in|has_material_basis_in|transmitted_by|"
                          r"derives_from|has_onset|occurs_with|realized_by|has_feature)\b",
                          lambda m: m.group(1).replace("_", " "), t.get("def", ""))
        out[t["id"]] = t
    return out, version


def _qid(uri: str) -> str:
    return uri.strip().strip("<>").rsplit("/", 1)[-1]


def _tsv_rows(name: str) -> list[list[str]]:
    return [r.split("\t") for r in (RAW / name).read_text(encoding="utf-8").splitlines()[1:] if r.strip()]


def parse_wd() -> dict:
    ids = defaultdict(lambda: {"doid": set(), "icd": set(), "mesh": set()})
    for d, doid, icd, mesh in _tsv_rows("wd_ids.tsv"):
        for k, v in (("doid", doid), ("icd", icd), ("mesh", mesh)):
            if v.strip():
                ids[_qid(d)][k].add(v.strip().strip('"'))
    labels = defaultdict(lambda: {"en": "", "ko": "", "en_alt": [], "ko_alt": []})
    for e, p, lit in _tsv_rows("wd_labels.tsv"):
        m = re.match(r'"(.*)"@(en|ko)$', lit)
        if not m:
            continue
        text, lang = m.group(1).replace('\\"', '"'), m.group(2)
        rec = labels[_qid(e)]
        if p.endswith("#label>"):
            rec[lang] = text
        elif text not in rec[lang + "_alt"]:
            rec[lang + "_alt"].append(text)
    sym, exams = defaultdict(list), defaultdict(list)
    for d, s in _tsv_rows("wd_symptoms.tsv"):
        sym[_qid(d)].append(_qid(s))
    for d, t in _tsv_rows("wd_exams.tsv"):
        exams[_qid(d)].append(_qid(t))
    # keep only real diseases: items with a Disease Ontology id or an ICD-10 code (drops chemical-exposure items)
    diseases = {q for q in set(ids) | set(sym) if ids[q]["doid"] or ids[q]["icd"]}
    return {"ids": ids, "labels": labels, "sym": sym, "exams": exams, "diseases": diseases}


def parse_kcd() -> dict:
    rows = list(csv.reader(io.StringIO((RAW / "hira_kcd.csv").read_bytes().decode("cp949"))))
    head, rows = rows[0], rows[1:]
    ci = {h: i for i, h in enumerate(head)}
    kcd: dict[str, dict] = {}
    for r in rows:
        if r[ci["양한방구분"]] == "한방":
            continue  # Korean-medicine-only codes
        code = r[ci["상병기호"]].strip()
        rec = kcd.setdefault(code, {"ko": [], "en": []})
        for lang, col in (("ko", "한글명"), ("en", "영문명")):
            v = r[ci[col]].strip()
            if v and v not in rec[lang]:
                rec[lang].append(v)
        if r[ci["법정감염병구분"]].strip():
            rec["notifiable"] = r[ci["법정감염병구분"]].strip()
        sex = r[ci["성별구분"]].strip()
        if sex:
            rec["sex"] = sex
    return kcd


def parse_ddx() -> tuple[dict, dict]:
    cond = json.loads((RAW / "ddx_conditions.json").read_text(encoding="utf-8"))
    ev = json.loads((RAW / "ddx_evidences.json").read_text(encoding="utf-8"))
    return cond, ev


def _strip_html(s: str) -> str:
    s = re.sub(r"<li>\s*", "\n- ", s)
    s = re.sub(r"</p>|<br\s*/?>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def parse_mplus() -> tuple[dict, str]:
    root = ET.parse(RAW / "mplus_topics.xml").getroot()
    out = {}
    for t in root.findall("health-topic"):
        if t.get("language") != "English":
            continue
        summ = t.findtext("full-summary") or ""
        out[t.get("id")] = {
            "title": t.get("title"), "url": t.get("url"),
            "also": [a.text for a in t.findall("also-called") if a.text],
            "mesh": [d.get("id") for d in t.findall("mesh-heading/descriptor")],
            "summary": _strip_html(summ),
        }
    return out, root.get("date-generated", "")


# Orphanet frequency classes → short codes stored in profiles (field orpha_freq). EX = excluded (0%).
ORPHA_FREQ = {"Obligate": "O", "Very frequent": "VF", "Frequent": "F", "Occasional": "OC", "Very rare": "VR",
              "Excluded": "EX"}
ORPHA_FREQ_RANK = {"O": 5, "VF": 4, "F": 3, "OC": 2, "VR": 1, "EX": 0}


def _orpha_root(name: str):
    root = ET.parse(RAW / name).getroot()
    lic = root.findtext("Availability/Licence/ShortIdentifier") or ""
    if lic != "CC-BY-4.0":
        raise RuntimeError(f"{name}: unexpected license {lic!r} (expected CC-BY-4.0)")
    return root


def parse_orpha() -> dict:
    """Orphadata product 1 (names, synonyms, cross-references with mapping relation) and product 4 (HPO-coded
    phenotype annotations with frequency class). Returns {"dis": {orphacode: {...}}, "pheno": {orphacode: [(hpo id,
    label, freq code, diagnostic criterion)]}, "version": date}."""
    r1 = _orpha_root("orpha_product1.xml")
    dis = {}
    for d in r1.iter("Disorder"):
        code = d.findtext("OrphaCode")
        if not code:
            continue
        xref = []
        for x in d.iter("ExternalReference"):
            rel = (x.findtext("DisorderMappingRelation/Name") or "").split(" ", 1)[0]
            xref.append((x.findtext("Source") or "", (x.findtext("Reference") or "").strip(), rel))
        dis[code] = {"name": d.findtext("Name") or "", "syn": [s.text for s in d.iter("Synonym") if s.text],
                     "type": d.findtext("DisorderType/Name") or "", "xref": xref}
    r4 = _orpha_root("orpha_product4.xml")
    pheno = {}
    for st in r4.iter("HPODisorderSetStatus"):
        d = st.find("Disorder")
        code = d.findtext("OrphaCode")
        rows = []
        for a in d.iter("HPODisorderAssociation"):
            fname = a.findtext("HPOFrequency/Name") or ""
            freq = next((v for k, v in ORPHA_FREQ.items() if fname.startswith(k)), "")
            if not freq:
                continue
            rows.append((a.findtext("HPO/HPOId"), a.findtext("HPO/HPOTerm") or "", freq,
                         bool(a.findtext("DiagnosticCriteria/Name"))))
        if rows:
            pheno[code] = rows
            if code not in dis:
                dis[code] = {"name": d.findtext("Name") or "", "syn": [], "type": d.findtext("DisorderType/Name") or "",
                             "xref": []}
    return {"dis": dis, "pheno": pheno, "version": r4.get("date", "")[:10]}


def parse_wd_bridges() -> dict:
    """Wikidata (CC0): Orphanet id → disease items, HPO id → items, and en/ko labels of the HPO-id items."""
    orpha = defaultdict(set)
    for d, o in _tsv_rows("wd_orpha.tsv"):
        orpha[o.strip().strip('"')].add(_qid(d))
    hpo = defaultdict(set)
    for e, h in _tsv_rows("wd_hpo.tsv"):
        hpo[h.strip().strip('"')].add(_qid(e))
    labels = defaultdict(lambda: {"en": "", "ko": "", "en_alt": [], "ko_alt": []})
    for e, p, lit in _tsv_rows("wd_hpo_labels.tsv"):
        m = re.match(r'"(.*)"@(en|ko)$', lit)
        if not m:
            continue
        text, lang = m.group(1).replace('\\"', '"'), m.group(2)
        rec = labels[_qid(e)]
        if p.endswith("#label>"):
            rec[lang] = text
        elif text not in rec[lang + "_alt"]:
            rec[lang + "_alt"].append(text)
    return {"orpha": orpha, "hpo": hpo, "labels": labels}


def parse_prevalence() -> dict:
    """HIRA 2025 patients per KCD 3-character code (main + secondary diagnoses) by sex and 5-year age group:
    {code: [male counts x 18 age groups, female counts x 18]} (age group 0 = 0-4 y, ..., 17 = 85+)."""
    rows = list(csv.reader(io.StringIO((RAW / "hira_kcd3_stats.csv").read_bytes().decode("cp949"))))
    head, rows = rows[0], rows[1:]
    ci = {h: i for i, h in enumerate(head)}
    out: dict[str, list[int]] = {}
    for r in rows:
        code, sex, band = r[ci["상병코드"]].strip(), r[ci["성별"]].strip(), r[ci["연령군"]].strip()
        m = re.match(r"(\d+)_", band)
        if not code or sex not in ("남", "여") or not m:
            continue
        g = int(m.group(1)) - 1
        if not 0 <= g < 18:
            continue
        out.setdefault(code, [0] * 36)[g + (18 if sex == "여" else 0)] += int(r[ci["환자수"]] or 0)
    return out


# ------------------------------------------------------------------------------------------------
# LLM (offline) with a persistent cache
# ------------------------------------------------------------------------------------------------

TERM_PROMPT = """당신은 의학 용어 번역기입니다. 영어 증상·징후·위험인자·검사 용어 목록을 한국 임상 현장에서 쓰는 한국어 용어로 옮기세요.
규칙:
1. ko: 한국 의사가 차트에 쓰는 짧은 표준 의학 용어 (예: dyspnea→호흡곤란, polyuria→다뇨, chest X-ray→흉부 X선).
2. lay: 환자가 말할 법한 표현 0~3개. 어미를 뺀 짧은 어간 형태로 쓰세요 (예: "숨이 차", "숨참", "가슴이 아프"). 검사·위험인자는 빈 배열.
3. 용어가 모호하면 의학적으로 가장 흔한 뜻으로 옮기세요. 설명을 덧붙이지 마세요.
4. 입력의 번호와 개수를 그대로 유지하세요.
출력: JSON 객체 하나만. {"items": [{"i": 1, "ko": "...", "lay": ["..."]}, ...]}"""

EVID_PROMPT = """당신은 의학 문진 항목 정리 도구입니다. 영어 문진 질문 목록(DDXPlus 데이터셋)을 받아 각각을 짧은 소견 용어로 바꾸세요.
규칙:
1. en: 질문이 확인하려는 소견의 짧은 영어 임상 용어 (예: "Do you have a fever?" → "fever"; 병력 질문은 "history of ..." 형태, 예: "history of diabetes").
2. ko: 그 소견의 짧은 한국어 의학 용어 (예: 발열, 당뇨병 병력).
3. lay: 환자가 말할 법한 표현 0~3개, 어미를 뺀 어간 형태 (예: "열이 나").
4. q_ko: 의사가 환자에게 묻는 자연스러운 한국어 질문 한 문장 (원문의 뜻을 그대로).
5. 입력의 번호와 개수를 그대로 유지하세요.
출력: JSON 객체 하나만. {"items": [{"i": 1, "en": "...", "ko": "...", "lay": ["..."], "q_ko": "..."}, ...]}"""

MPLUS_PROMPT = """당신은 의학 정보 추출 도구입니다. 입력은 미국 국립의학도서관 MedlinePlus의 질환 요약문(공공 도메인)입니다.
각 요약문에서 **원문에 명시된 내용만** 추출하세요. 원문에 없는 증상·검사를 추가하지 마세요.
1. symptoms: 이 질환의 증상·징후로 원문에 적힌 것. en은 짧은 영어 임상 용어(원문 단어를 최대한 유지), ko는 짧은 한국어 의학 용어. 없으면 빈 배열. 최대 12개.
2. tests: 진단을 위해 원문에 언급된 검사·진찰 (치료는 제외). en/ko 형식 동일. 없으면 빈 배열. 최대 8개.
3. summary_ko: 이 질환이 무엇인지 한국어 한 문장 (40~110자). 원문 내용만 쓰고 치료 권고는 넣지 마세요.
출력: JSON 객체 하나만.
{"topics": [{"id": "<입력 id>", "summary_ko": "...", "symptoms": [{"en": "...", "ko": "..."}], "tests": [{"en": "...", "ko": "..."}]}]}"""

NAME_PROMPT = """당신은 의학 용어 번역기입니다. 영어 질환명 목록을 한국 임상에서 쓰는 표준 한국어 질환명으로 옮기세요.
규칙: 대한의학회·KCD 스타일의 공식 명칭을 우선하고, 없으면 가장 자연스러운 의학 번역을 쓰세요. 설명 없이 이름만. 번호와 개수를 유지하세요.
출력: JSON 객체 하나만. {"items": [{"i": 1, "ko": "..."}, ...]}"""

_HANGUL = re.compile(r"[가-힣]")


class LLMCache:
    def __init__(self, enabled: bool, workers: int):
        self.data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        self.enabled, self.workers = enabled, workers
        self.lock = threading.Lock()
        self.client = None
        self.cfg = None
        self.calls = 0
        self.failures: list[str] = []

    def _client(self):
        if self.client is None:
            from doctor_agent.config import LLMConfig
            from doctor_agent.llm.client import OpenAICompatClient
            from eval.run_local import load_dotenv

            load_dotenv(ROOT / ".env")
            cfg = LLMConfig.from_env("KB_LLM")
            cfg.temperature, cfg.max_tokens, cfg.timeout_s = 0.0, 16000, 180.0
            self.cfg, self.client = cfg, OpenAICompatClient(cfg)
        return self.client

    def save(self) -> None:
        with self.lock:
            CACHE.write_text(json.dumps(self.data, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")

    def run(self, task: str, keys: list[str], system: str, render, parse, batch: int) -> dict:
        """Returns {key: output} for keys, calling the LLM only for keys missing from the cache."""
        store = self.data.setdefault(task, {})
        todo = [k for k in keys if k not in store]
        if todo and self.enabled:
            from doctor_agent.agent.parser import _json_objects

            chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
            print(f"LLM {task}: {len(todo)} items in {len(chunks)} calls")

            def one(chunk: list[str]) -> None:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": render(chunk)}]
                for attempt in range(2):
                    try:
                        raw = self._client().chat(msgs)
                    except RuntimeError as e:
                        self.failures.append(f"{task}: {e}")
                        return
                    with self.lock:
                        self.calls += 1
                    objs = _json_objects(raw or "")
                    got = parse(chunk, objs[-1] if objs else {})
                    if got:
                        with self.lock:
                            store.update(got)
                        if len(got) >= len(chunk) * 0.8:
                            return
                self.failures.append(f"{task}: incomplete chunk starting {chunk[0]}")

            with ThreadPoolExecutor(self.workers) as ex:
                for n, _ in enumerate(ex.map(one, chunks)):
                    if n % 10 == 9:
                        self.save()
            self.save()
        return {k: store[k] for k in keys if k in store}


def _numbered(chunk: list[str], fmt=lambda k: k) -> str:
    return "\n".join(f"{i + 1}. {fmt(k)}" for i, k in enumerate(chunk))


def _by_index(chunk: list[str], obj: dict, check) -> dict:
    out = {}
    for it in obj.get("items", []) if isinstance(obj, dict) else []:
        try:
            i = int(it.get("i")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(chunk) and check(it):
            out[chunk[i]] = it
    return out


def _clean_lay(xs) -> list[str]:
    return [str(x).strip() for x in (xs if isinstance(xs, list) else []) if _HANGUL.search(str(x)) and len(str(x)) <= 20][:3]


def llm_terms(cache: LLMCache, terms: list[str]) -> dict:
    def parse(chunk, obj):
        got = _by_index(chunk, obj, lambda it: bool(_HANGUL.search(str(it.get("ko", "")))) and len(str(it["ko"])) <= 40)
        return {k: {"ko": str(v["ko"]).strip(), "lay": _clean_lay(v.get("lay"))} for k, v in got.items()}
    return cache.run("term_ko", terms, TERM_PROMPT, lambda c: _numbered(c), parse, batch=60)


def llm_evidences(cache: LLMCache, ev: dict) -> dict:
    keys = sorted(ev)

    def parse(chunk, obj):
        ok = lambda it: all(str(it.get(f, "")).strip() for f in ("en", "ko", "q_ko")) and _HANGUL.search(str(it["ko"]))
        return {k: {"en": str(v["en"]).strip().lower(), "ko": str(v["ko"]).strip(), "lay": _clean_lay(v.get("lay")),
                    "q_ko": str(v["q_ko"]).strip()} for k, v in _by_index(chunk, obj, ok).items()}
    return cache.run("ddx_evidence", keys, EVID_PROMPT, lambda c: _numbered(c, lambda k: ev[k]["question_en"]),
                     parse, batch=40)


def _grounded(term: str, text: str) -> bool:
    """An extracted English term counts only if one of its content words (or its stem) occurs in the source text."""
    words = [w for w in re.findall(r"[a-z]{4,}", term.lower()) if w not in {"with", "that", "from", "pain"}]
    if not words:
        words = re.findall(r"[a-z]{3,}", term.lower())
    low = text.lower()
    return any(w[:max(4, len(w) - 3)] in low for w in words)


def llm_mplus(cache: LLMCache, topics: dict) -> dict:
    keys = sorted(topics)

    def render(chunk):
        return "\n\n".join(f"[id {k}] {topics[k]['title']}\n{topics[k]['summary'][:2500]}" for k in chunk)

    def parse(chunk, obj):
        out = {}
        for t in obj.get("topics", []) if isinstance(obj, dict) else []:
            k = str(t.get("id", "")).strip()
            if k not in chunk or not isinstance(t, dict):
                continue
            text = topics[k]["title"] + " " + topics[k]["summary"]
            rec = {"summary_ko": str(t.get("summary_ko", "")).strip()[:160], "symptoms": [], "tests": [], "dropped": []}
            for field in ("symptoms", "tests"):
                for x in t.get(field) or []:
                    if not isinstance(x, dict):
                        continue
                    en, ko = str(x.get("en", "")).strip().lower(), str(x.get("ko", "")).strip()
                    if not en or not _HANGUL.search(ko):
                        continue
                    (rec[field] if _grounded(en, text) else rec["dropped"]).append({"en": en, "ko": ko})
            out[k] = rec
        return out
    return cache.run("mplus_extract", keys, MPLUS_PROMPT, render, parse, batch=6)


def llm_names(cache: LLMCache, names: list[str]) -> dict:
    def parse(chunk, obj):
        got = _by_index(chunk, obj, lambda it: bool(_HANGUL.search(str(it.get("ko", "")))) and len(str(it["ko"])) <= 60)
        return {k: str(v["ko"]).strip() for k, v in got.items()}
    return cache.run("disease_ko", names, NAME_PROMPT, lambda c: _numbered(c), parse, batch=80)


# Our own short list of common Korean clinical wordings (authored by us, source tag "curated"), keyed by the English
# term label. Helps match findings written by the doctor model to KB terms.
CURATED_SYN = {
    "nausea": ["오심", "구역", "메스꺼움"], "vomiting": ["구토", "토함"], "neck stiffness": ["목 경직", "경부 경직", "항부 강직", "목이 뻣뻣"],
    "abdominal pain": ["복부 통증", "배 통증", "배가 아프"], "dyspnea": ["숨참", "숨이 차", "호흡 곤란", "숨가쁨"],
    "chest pain": ["가슴 통증", "가슴이 아프", "흉부 통증"], "polydipsia": ["다음", "갈증 증가", "물을 많이 마심"],
    "polyuria": ["다뇨", "소변량 증가", "소변을 많이 봄"], "weight loss": ["체중 감소", "살이 빠짐", "체중감소"],
    "fever": ["발열", "고열", "미열", "열이 나"], "cough": ["기침"], "hemoptysis": ["객혈", "피 섞인 가래"],
    "diarrhea": ["설사", "묽은 변"], "jaundice": ["황달", "눈이 노랗"], "syncope": ["실신", "기절", "의식 소실"],
    "palpitations": ["두근거림", "심계항진"], "dizziness": ["어지럼", "어지러움"], "fatigue": ["피로", "피곤", "기운 없음"],
    "edema": ["부종", "붓기", "부음"], "rash": ["발진", "피부 발진"], "photophobia": ["눈부심", "광선공포증", "빛 공포"],
    "confusion": ["혼돈", "착란", "의식 혼탁"], "seizure": ["경련", "발작"], "dysuria": ["배뇨통", "배뇨 시 통증", "소변 볼 때 아프"],
    "hematuria": ["혈뇨", "소변에 피"], "melena": ["흑색변", "짜장면 같은 변"], "hematochezia": ["혈변", "선혈변"],
    "back pain": ["요통", "허리 통증"], "arthralgia": ["관절통", "관절 통증"], "myalgia": ["근육통"],
    "sore throat": ["인후통", "목 통증", "목이 아프"], "rhinorrhea": ["콧물"], "nasal congestion": ["코막힘"],
    "chills": ["오한", "춥고 떨림"], "night sweats": ["야간 발한", "식은땀"], "wheezing": ["천명", "쌕쌕거림"],
    "tachycardia": ["빈맥", "맥박이 빠름"], "hypotension": ["저혈압", "혈압 저하"], "headache": ["두통", "머리가 아프"],
    "sputum": ["가래", "객담"], "anorexia": ["식욕 부진", "식욕 저하", "입맛 없음"], "heartburn": ["속쓰림", "가슴 쓰림"],
    "constipation": ["변비"], "abdominal distension": ["복부 팽만", "배가 부름"], "tachypnea": ["빈호흡", "호흡수 증가"],
    "altered mental status": ["의식 저하", "의식 변화"], "hemiparesis": ["편마비", "한쪽 마비", "반신 마비"],
    "blurred vision": ["시야 흐림", "흐린 시야"], "tinnitus": ["이명", "귀울림"], "pruritus": ["가려움", "소양감"],
}

# ------------------------------------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------------------------------------

# DDXPlus condition → name used to find the matching DO/Wikidata disease (only where the DDXPlus label differs)
DDX_ALIAS = {
    "Spontaneous pneumothorax": "pneumothorax", "HIV (initial infection)": "HIV infection",
    "Viral pharyngitis": "pharyngitis", "PSVT": "paroxysmal supraventricular tachycardia",
    "Allergic sinusitis": "allergic rhinitis", "Chagas": "Chagas disease", "SLE": "systemic lupus erythematosus",
    "Stable angina": "angina pectoris", "Acute otitis media": "otitis media", "Panic attack": "panic disorder",
    "Bronchospasm / acute asthma exacerbation": "asthma", "Acute COPD exacerbation / infection": "chronic obstructive pulmonary disease",
    "URTI": "upper respiratory tract infection", "Acute rhinosinusitis": "sinusitis",
    "Chronic rhinosinusitis": "chronic sinusitis", "Pulmonary neoplasm": "lung cancer",
    "Possible NSTEMI / STEMI": "myocardial infarction", "Acute pulmonary edema": "pulmonary edema",
    "Larygospasm": "laryngospasm", "Boerhaave": "Boerhaave syndrome", "Spontaneous rib fracture": "rib fracture",
    "Scombroid food poisoning": "scombroid food poisoning", "Localized edema": "edema",
    "Acute dystonic reactions": "acute dystonia", "Guillain-Barré syndrome": "Guillain-Barre syndrome",
}


def kcd_code(icd: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", icd.upper())


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # keep a DO node as the root when there is one
        if rb.startswith("DO:") and not ra.startswith("DO:"):
            ra, rb = rb, ra
        self.p[rb] = ra


def add_test_links(profiles: list[dict], terms: dict) -> dict:
    """Curated test/lab/imaging finding → disease links (src/doctor_agent/knowledge/kb_tests.py, source "curated").

    Adds one term per finding ("TF:<id>") and, per linked profile, findings_from_tests = [[term id, ["curated"],
    weight 1-3, reference key, "R" if a normal result argues against the disease else ""]]. Every linked profile id
    must exist (fails loudly otherwise), so a KB rebuild cannot silently drop links."""
    by_id = {p["id"]: p for p in profiles}
    missing = sorted({pid for _f, pid, _w, _r, _ref in kb_tests.links() if pid not in by_id})
    if missing:
        raise RuntimeError(f"kb_tests.DX points at profiles missing from the KB: {missing}")
    for f in kb_tests.FINDINGS:
        terms["TF:" + f.id] = {"en": f.en, "ko": f.ko, "syn": [], "src": {"en": "curated", "ko": "curated"},
                               "kind": "test_finding"}
    n = 0
    for fid, pid, w, rule_out, ref in kb_tests.links():
        lst = by_id[pid].setdefault("findings_from_tests", [])
        if all(x[0] != "TF:" + fid for x in lst):
            lst.append(["TF:" + fid, ["curated"], w, ref, "R" if rule_out else ""])
            n += 1
    for p in profiles:
        if "findings_from_tests" in p:
            p["findings_from_tests"].sort(key=lambda x: (-x[2], x[0]))
    return {"test_findings": len(kb_tests.FINDINGS), "test_links": n,
            "profiles_with_test_findings": sum(1 for p in profiles if p.get("findings_from_tests")),
            "test_link_refs_with_pmid": sum(1 for *_x, ref in kb_tests.links() if kb_tests.REFS[ref]["pmid"])}


def map_orpha(orpha: dict, do: dict, wd: dict, bridges: dict, uf: UF, match_name) -> dict:
    """Join each Orphanet disorder that has phenotype annotations to one existing disease group, trying in order:
    DO xref ORDO:<code>, Wikidata P1550, exact (E) OMIM mapping ↔ DO MIM xref, exact English name (primary name or a
    synonym of ≥ 6 chars). A code route counts only when it points at a single group. Unjoined disorders get their
    own node. Returns {orphacode: joined root or None}."""
    do_orpha, do_omim = defaultdict(set), defaultdict(set)
    for d, t in do.items():
        for x in t["xref"]:
            if x.startswith("ORDO:"):
                do_orpha[x[5:]].add("DO:" + d)
            elif x.startswith("MIM:"):
                do_omim[x[4:]].add("DO:" + d)
    out = {}
    for oc in sorted(orpha["pheno"], key=int):
        rec = orpha["dis"][oc]
        routes = [
            do_orpha.get(oc, set()),
            {"WD:" + q for q in bridges["orpha"].get(oc, ()) if q in wd["diseases"]},
            {n for s, ref, rel in rec["xref"] if s == "OMIM" and rel == "E" for n in do_omim.get(ref, ())},
        ]
        hit = None
        for nodes in routes:
            roots = {uf.find(n) for n in nodes}
            if len(roots) == 1:
                hit = roots.pop()
                break
        if hit is None:
            hit = match_name(*[n for n in [rec["name"], *rec["syn"]] if len(norm_en(n)) >= 6])
        node = "ORPHA:" + oc
        uf.find(node)
        if hit:
            uf.union(hit, node)
        out[oc] = hit
    return out


def orpha_terms(orpha: dict, codes: set[str], bridges: dict, kcd: dict, terms: dict, en_ix: dict, add_term
                ) -> tuple[dict, dict]:
    """HPO id → KB term id for the phenotypes of the given Orphanet disorders. An existing KB term is reused when the
    Orphanet label (or its Wikidata-bridged item's label, or a simple plural/singular variant) equals a KB term label;
    otherwise a new term "HP:<id>" is added with the Wikidata English label when bridged (else Orphanet's label) and
    a Korean label only from Wikidata or an exact English-name match in the KCD master (no LLM)."""
    kcd_en: dict[str, str] = {}
    for c in sorted(kcd, key=lambda c: (len(c), c)):
        if c[0] in "VWXYZU":  # external causes, contact factors, special codes
            continue
        for e in kcd[c]["en"][:1]:
            kcd_en.setdefault(norm_en(e), c)
            # "Hepatomegaly, NEC" / "Ataxia, unspecified" also stand for the bare English term
            bare = re.sub(r",\s*(unspecified|nec|not elsewhere classified)$", "", e.strip(), flags=re.I)
            if bare != e.strip():
                kcd_en.setdefault(norm_en(bare), c)
    labels = bridges["labels"]
    hpo_lab = {hid: lab for oc in codes for hid, lab, _f, _d in orpha["pheno"].get(oc, [])}
    out, st = {}, defaultdict(int)
    for hid in sorted(hpo_lab):
        lab = hpo_lab[hid]
        qs = sorted(bridges["hpo"].get(hid, ()))
        cands = [lab] + [labels[q]["en"] for q in qs if labels[q]["en"]]
        keys = []
        for c in cands:
            k = norm_en(c)
            keys += [k, k[:-1] if k.endswith("s") else k + "s"]
        # an alias-only match (e.g. a Wikidata alias "skin ulcer" on "abscess") is trusted only when Wikidata has no
        # item of its own for this HPO id; a match on the term's own label, or on the bridged item, always counts
        bridged = {"WD:" + q for q in qs}
        tid = next((en_ix[k] for k in keys if k in en_ix and en_ix[k] in terms
                    and (not qs or en_ix[k] in bridged or norm_en(terms[en_ix[k]]["en"]) == k)), None)
        if tid:
            out[hid] = tid
            st["hpo_to_existing_term"] += 1
            continue
        wd_en = next((labels[q]["en"] for q in qs if labels[q]["en"]), "")
        ko, ko_src = next(((labels[q]["ko"], "WD") for q in qs if _HANGUL.search(labels[q]["ko"])), ("", ""))
        if not ko and norm_en(lab) in kcd_en:
            ko = re.sub(r"^(상세불명의|달리 분류되지 않은|기타)\s*", "", kcd[kcd_en[norm_en(lab)]]["ko"][0])
            ko_src = "KCD"
        syn = [a for q in qs for a in labels[q]["ko_alt"][:4] + labels[q]["en_alt"][:4]]
        tid = add_term("HP:" + hid.split(":", 1)[1], wd_en or lab, "", syn if qs else [], "WD" if wd_en else "Orphanet")
        if ko and not terms[tid]["ko"]:
            terms[tid]["ko"], terms[tid]["src"]["ko"] = ko, ko_src
        out[hid] = tid
        st["hpo_new_term" + ("_ko_" + ko_src if ko else "_en_only")] += 1
    return out, dict(st)


def assemble(cache: LLMCache, files: dict) -> dict:
    do, do_version = parse_do()
    wd = parse_wd()
    kcd = parse_kcd()
    ddx_cond, ddx_ev = parse_ddx()
    mplus, mplus_date = parse_mplus()
    orpha = parse_orpha()
    bridges = parse_wd_bridges()
    print(f"parsed DO {len(do)}, WD diseases {len(wd['diseases'])}, KCD codes {len(kcd)}, DDXPlus {len(ddx_cond)}, "
          f"MedlinePlus {len(mplus)}, Orphanet disorders with phenotypes {len(orpha['pheno'])}")

    uf = UF()
    for d in do:
        uf.find("DO:" + d)
    for q in wd["diseases"]:
        uf.find("WD:" + q)
        doids = sorted(x for x in wd["ids"][q]["doid"] if x in do)
        if doids:
            uf.union("DO:" + doids[0], "WD:" + q)

    # name index for matching DDXPlus and MedlinePlus titles
    name_ix: dict[str, set[str]] = defaultdict(set)
    for d, t in do.items():
        for n in [t["name"], *t["syn"]]:
            name_ix[norm_en(n)].add("DO:" + d)
    for q in wd["diseases"]:
        lab = wd["labels"][q]
        for n in [lab["en"], *lab["en_alt"]]:
            if n:
                name_ix[norm_en(n)].add("WD:" + q)

    def match_name(*names: str) -> str | None:
        for n in names:
            hits = sorted({uf.find(x) for x in name_ix.get(norm_en(n), ())})
            if len(hits) == 1:
                return hits[0]
            if hits:
                do_hits = [h for h in hits if h.startswith("DO:")]
                return (do_hits or hits)[0]
        return None

    ddx_map = {}
    for c, rec in ddx_cond.items():
        hit = match_name(rec["cond-name-eng"], DDX_ALIAS.get(c, ""))
        node = "DDX:" + c
        uf.find(node)
        if hit:
            uf.union(hit, node)
        ddx_map[c] = hit
    mesh_ix: dict[str, set[str]] = defaultdict(set)
    for d, t in do.items():
        for x in t["xref"]:
            if x.startswith("MESH:"):
                mesh_ix[x[5:]].add("DO:" + d)
    for q in wd["diseases"]:
        for m in wd["ids"][q]["mesh"]:
            mesh_ix[m].add("WD:" + q)
    mplus_map = {}
    for tid, t in mplus.items():
        hit = match_name(t["title"], *t["also"])
        if not hit:
            for m in t["mesh"]:
                roots = sorted({uf.find(x) for x in mesh_ix.get(m, ())})
                if len(roots) == 1:
                    hit = roots[0]
                    break
        if hit:
            uf.union(hit, "MP:" + tid)
            mplus_map[tid] = hit

    # Orphanet disorders with phenotype annotations: join an existing disease (DO ORDO xref → Wikidata P1550 →
    # exact OMIM mapping → exact English name), else become a new profile "ORPHA:<code>"
    orpha_map = map_orpha(orpha, do, wd, bridges, uf, match_name)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in list(uf.p):
        groups[uf.find(node)].append(node)

    # ---------------- terms (symptoms / risk factors / tests) ----------------
    terms: dict[str, dict] = {}
    en_ix: dict[str, str] = {}

    def add_term(tid: str, en: str, ko: str = "", syn: list[str] | None = None, src: str = "") -> str:
        key = norm_en(en)
        if tid not in terms and key in en_ix:
            tid = en_ix[key]
        rec = terms.setdefault(tid, {"en": en, "ko": "", "syn": [], "src": {}})
        ko = ko if _HANGUL.search(ko or "") else ""  # some Wikidata "ko" labels are English
        if ko and not rec["ko"]:
            rec["ko"] = ko
            rec["src"]["ko"] = src
        for s in syn or []:
            if s and s not in rec["syn"] and s != rec["en"] and s != rec["ko"]:
                rec["syn"].append(s)
        rec["src"].setdefault("en", src)
        en_ix.setdefault(key, tid)
        return tid

    used_wd = {s for q in wd["diseases"] for s in wd["sym"].get(q, []) + wd["exams"].get(q, [])}
    for s in sorted(used_wd):
        lab = wd["labels"][s]
        if not lab["en"]:
            continue
        tid = add_term("WD:" + s, lab["en"], lab["ko"], lab["en_alt"][:6] + lab["ko_alt"][:6], "WD")
        for alt in lab["en_alt"]:
            en_ix.setdefault(norm_en(alt), tid)

    ev_llm = llm_evidences(cache, {k: v for k, v in ddx_ev.items() if v["data_type"] == "B"})
    ev_tid = {}
    for k, v in ev_llm.items():
        tid = add_term("DDX:" + k, v["en"], "", [], "DDXPlus")
        terms[tid].setdefault("q", {"en": ddx_ev[k]["question_en"], "ko": v["q_ko"]})
        terms[tid]["src"].setdefault("q", "DDXPlus")
        if not terms[tid]["ko"]:
            terms[tid]["ko"] = v["ko"]
            terms[tid]["src"]["ko"] = "LLM"
        terms[tid]["syn"] += [x for x in v["lay"] if x not in terms[tid]["syn"]]
        ev_tid[k] = tid

    do_sym_tid = {}
    for d, t in do.items():
        for ph in t["has_symptom"]:
            if 2 < len(ph) <= 60:
                do_sym_tid[(d, ph)] = add_term("T:" + norm_en(ph), ph, "", [], "DO")

    mp_topics = {k: mplus[k] for k in mplus_map if mplus[k]["summary"]}
    mp_llm = llm_mplus(cache, mp_topics)
    mp_tids: dict[str, dict] = {}
    for k, rec in mp_llm.items():
        # the Korean label comes from the dedicated term translation below (more reliable than the in-context one,
        # which is kept only as a fallback)
        mp_tids[k] = {f: [add_term("T:" + norm_en(x["en"]), x["en"], "", [], "MedlinePlus") for x in rec[f]]
                      for f in ("symptoms", "tests")}
        for f in ("symptoms", "tests"):
            for x, tid in zip(rec[f], mp_tids[k][f]):
                terms[tid].setdefault("_ko_fallback", x["ko"])

    # Korean labels + lay synonyms for every term (keep the Wikidata Korean label as primary when it exists)
    tr = llm_terms(cache, sorted({t["en"] for t in terms.values()}))
    for tid, t in terms.items():
        got = tr.get(t["en"])
        fallback = t.pop("_ko_fallback", "")
        if not got:
            if not t["ko"] and fallback:
                t["ko"], t["src"]["ko"] = fallback, "LLM"
            continue
        if not t["ko"]:
            t["ko"], t["src"]["ko"] = got["ko"], "LLM"
        elif got["ko"] != t["ko"] and got["ko"] not in t["syn"]:
            t["syn"].append(got["ko"])
        t["syn"] += [x for x in got["lay"] if x not in t["syn"] and x != t["ko"]]
        t["src"].setdefault("syn", "WD+LLM")

    for en, syns in CURATED_SYN.items():
        tid = en_ix.get(norm_en(en))
        if tid and tid in terms:
            terms[tid]["syn"] = [x for x in syns if x != terms[tid]["ko"]] + [x for x in terms[tid]["syn"] if x not in syns]
            terms[tid]["src"]["curated"] = "curated"

    # Orphanet phenotypes (HPO-coded) → KB terms. No LLM: labels come from Orphanet (English), Wikidata (en/ko,
    # CC0) and exact English-name matches in the KCD master (Korean). Added after the LLM translation step on purpose.
    hpo_tid, hpo_stats = orpha_terms(orpha, {c for c in orpha_map}, bridges, kcd, terms, en_ix, add_term)

    # merge terms that ended up with the same Korean label (fever/fevers, cough variants) into one concept
    rank = lambda tid: (not tid.startswith("WD:"), not tid.startswith("DDX:"), len(terms[tid]["en"]), tid)
    by_ko: dict[str, list[str]] = defaultdict(list)
    for tid, t in terms.items():
        by_ko[re.sub(r"[^0-9a-z가-힣]", "", t["ko"].lower())].append(tid)
    canon = {}
    for key, tids in by_ko.items():
        tids.sort(key=rank)
        for tid in tids:
            canon[tid] = tids[0] if key else tid
        if not key:  # no Korean label: nothing to merge (used to pour every unlabelled term into one head's synonyms)
            continue
        head = terms[tids[0]]
        for tid in tids[1:]:
            t = terms[tid]
            for x in [t["en"], *t["syn"]]:
                if x not in head["syn"] and x not in (head["en"], head["ko"]):
                    head["syn"].append(x)
            if "q" in t and "q" not in head:
                head["q"], head["src"]["q"] = t["q"], "DDXPlus"

    # ---------------- disease profiles ----------------
    profiles = []
    for root, nodes in groups.items():
        dos = sorted(n[3:] for n in nodes if n.startswith("DO:"))
        wds = sorted(n[3:] for n in nodes if n.startswith("WD:"))
        ddxs = sorted(n[4:] for n in nodes if n.startswith("DDX:"))
        mps = sorted(n[3:] for n in nodes if n.startswith("MP:"))
        ors = sorted((n[6:] for n in nodes if n.startswith("ORPHA:")), key=int)
        if not (dos or wds or ddxs or ors):
            continue
        p = {"id": root.replace("DO:", "", 1) if root.startswith("DO:") else root, "names_en": [], "names_ko": [],
             "codes": {}, "symptoms": {}, "risk": {}, "tests": {}}

        def name(lst, n, src):
            n = (n or "").strip()
            if n and all(n.lower() != x[0].lower() for x in p[lst]):
                p[lst].append([n, src])

        def code(kind, v, src):
            p["codes"].setdefault(kind, [])
            if all(v != x[0] for x in p["codes"][kind]):
                p["codes"][kind].append([v, src])

        for d in dos:
            t = do[d]
            name("names_en", t["name"], "DO")
            for s in t["syn"][:8]:
                name("names_en", s, "DO")
            code("doid", d, "DO")
            for x in t["xref"]:
                pre, _, val = x.partition(":")
                kind = {"ICD10CM": "icd10", "MESH": "mesh", "MIM": "omim", "ORDO": "orpha"}.get(pre)
                if kind:
                    code(kind, val, "DO")
            if t.get("def") and "def" not in p:
                p["def"] = [t["def"][:400], "DO"]
            if t["is_a"]:
                p["parents"] = [[x, "DO"] for x in t["is_a"][:3]]
            for ph in t["has_symptom"]:
                if (d, ph) in do_sym_tid:
                    p["symptoms"].setdefault(do_sym_tid[(d, ph)], set()).add("DO")
        for q in wds:
            lab = wd["labels"][q]
            name("names_en", lab["en"], "WD")
            name("names_ko", lab["ko"], "WD")
            for a in lab["ko_alt"][:5]:
                name("names_ko", a, "WD")
            for a in lab["en_alt"][:5]:
                name("names_en", a, "WD")
            code("wikidata", q, "WD")
            for x in sorted(wd["ids"][q]["icd"]):  # sorted: set order depends on PYTHONHASHSEED
                code("icd10", x, "WD")
            for x in sorted(wd["ids"][q]["mesh"]):
                code("mesh", x, "WD")
            for s in wd["sym"].get(q, []):
                tid = en_ix.get(norm_en(wd["labels"][s]["en"])) if wd["labels"][s]["en"] else None
                if tid:
                    p["symptoms"].setdefault(tid, set()).add("WD")
            for s in wd["exams"].get(q, []):
                tid = en_ix.get(norm_en(wd["labels"][s]["en"])) if wd["labels"][s]["en"] else None
                if tid:
                    p["tests"].setdefault(tid, set()).add("WD")
        for c in ddxs:
            rec = ddx_cond[c]
            name("names_en", rec["cond-name-eng"], "DDXPlus")
            for x in re.split(r"[,\s]+", rec["icd10-id"]):
                if x:
                    code("icd10", x.upper(), "DDXPlus")
            p["severity"] = [rec["severity"], "DDXPlus"]
            for ek in rec["symptoms"]:
                if ek in ev_tid:
                    p["symptoms"].setdefault(ev_tid[ek], set()).add("DDXPlus")
            for ek in rec["antecedents"]:
                if ek in ev_tid:
                    p["risk"].setdefault(ev_tid[ek], set()).add("DDXPlus")
        for m in mps:
            t = mplus[m]
            name("names_en", t["title"], "MedlinePlus")
            for a in t["also"][:5]:
                name("names_en", a, "MedlinePlus")
            p["medlineplus"] = [t["url"], "MedlinePlus"]
            ext = mp_llm.get(m)
            if ext:
                if ext["summary_ko"] and "summary_ko" not in p:
                    p["summary_ko"] = [ext["summary_ko"], "MedlinePlus+LLM"]
                for tid in mp_tids[m]["symptoms"]:
                    p["symptoms"].setdefault(tid, set()).add("MedlinePlus+LLM")
                for tid in mp_tids[m]["tests"]:
                    p["tests"].setdefault(tid, set()).add("MedlinePlus+LLM")
        freq: dict[str, str] = {}
        for oc in ors:
            rec = orpha["dis"][oc]
            name("names_en", rec["name"], "Orphanet")
            for s in rec["syn"][:6]:
                name("names_en", s, "Orphanet")
            code("orpha", oc, "Orphanet")
            for src, ref, rel in rec["xref"]:
                if src == "OMIM" and rel == "E":
                    code("omim", ref, "Orphanet")
                elif src == "ICD-10" and rel == "E" and ref:
                    code("icd10", ref, "Orphanet")
                elif src == "ICD-10" and rel == "NTBT" and ref:
                    # the disorder is narrower than the code: keep the code, not the (broader) KCD name
                    code("icd10_broad", ref, "Orphanet")
            for hid, _lab, fq, _dc in orpha["pheno"].get(oc, []):
                tid = hpo_tid.get(hid)
                if not tid:
                    continue
                tid = canon.get(tid, tid)
                if ORPHA_FREQ_RANK[fq] > ORPHA_FREQ_RANK.get(freq.get(tid, ""), -1):
                    freq[tid] = fq
        own_names = {norm_en(n) for n, _ in p["names_en"]} | {re.sub(r"\s+", "", n) for n, _ in p["names_ko"]}
        for tid in [t for t in freq if norm_en(terms[t]["en"]) in own_names
                    or (terms[t]["ko"] and re.sub(r"\s+", "", terms[t]["ko"]) in own_names)]:
            del freq[tid]  # the disease itself listed as its own phenotype ("Zollinger-Ellison syndrome")
        for tid, fq in freq.items():
            if fq != "EX":
                p["symptoms"].setdefault(tid, set()).add("Orphanet")
        if freq:
            p["orpha_freq"] = [[tid, fq] for tid, fq in sorted(freq.items())]
        # KCD: exact code matches of ICD-10 codes (WHO ICD-10 and ICD-10-CM share the category structure)
        for icd, _src in p["codes"].get("icd10", []):
            k = kcd_code(icd)
            if k in kcd:
                code("kcd", k, "KCD")
                for n in kcd[k]["ko"][:3]:
                    name("names_ko", n, "KCD")
        for icd, _src in p["codes"].get("icd10_broad", []):
            k = kcd_code(icd)
            if k in kcd and all(k != c for c, _ in p["codes"].get("kcd", [])):
                code("kcd_broad", k, "KCD")
        # a KCD name whose English equals our English name goes first (it is the standard Korean name)
        en_set = {norm_en(n) for n, _ in p["names_en"]}
        std = [n for n in p["names_ko"] if n[1] == "KCD" and any(
            norm_en(e) in en_set for k in p["codes"].get("kcd", []) for e in kcd[k[0]]["en"][:1]
            if kcd[k[0]]["ko"][0] == n[0])]
        if std:
            p["names_ko"] = std[:1] + [n for n in p["names_ko"] if n is not std[0]]
        for f in ("symptoms", "risk", "tests"):
            merged: dict[str, set] = defaultdict(set)
            for tid, srcs in p[f].items():
                merged[canon.get(tid, tid)] |= srcs
            p[f] = [[tid, sorted(srcs)] for tid, srcs in sorted(merged.items())]
        profiles.append(p)

    # stable profile order (the union-find groups come from a set of Wikidata ids, i.e. PYTHONHASHSEED-dependent);
    # candidates() breaks score ties by profile index, so the order must be reproducible
    profiles.sort(key=lambda p: p["id"])

    # LLM Korean names for profiles with clinical content but no Korean name
    need = sorted({p["names_en"][0][0] for p in profiles if not p["names_ko"] and p["names_en"]
                   and (p["symptoms"] or p["tests"])})
    ko_names = llm_names(cache, need)
    for p in profiles:
        if not p["names_ko"] and p["names_en"] and p["names_en"][0][0] in ko_names:
            p["names_ko"].append([ko_names[p["names_en"][0][0]], "LLM"])

    used = {tid for p in profiles for f in ("symptoms", "risk", "tests", "orpha_freq") for tid, _ in p.get(f, [])}
    terms = {tid: t for tid, t in terms.items() if tid in used}
    for t in terms.values():
        t["syn"] = t["syn"][:14]
    test_stats = add_test_links(profiles, terms)

    stats = {
        "profiles": len(profiles),
        "profiles_with_symptoms": sum(1 for p in profiles if p["symptoms"]),
        "profiles_with_korean_name": sum(1 for p in profiles if p["names_ko"]),
        "profiles_with_kcd": sum(1 for p in profiles if p["codes"].get("kcd")),
        "profiles_with_tests": sum(1 for p in profiles if p["tests"]),
        "profiles_with_summary_ko": sum(1 for p in profiles if "summary_ko" in p),
        "terms": len(terms),
        "terms_with_korean": sum(1 for t in terms.values() if t["ko"]),
        "ddxplus_mapped": sum(1 for v in ddx_map.values() if v), "ddxplus_total": len(ddx_map),
        "medlineplus_mapped": len(mplus_map),
        "kcd_codes": len(kcd),
        **test_stats,
        "orphanet_disorders": len(orpha_map),
        "orphanet_joined_existing": sum(1 for v in orpha_map.values() if v),
        "profiles_orphanet_only": sum(1 for p in profiles if p["id"].startswith("ORPHA:")),
        "profiles_with_orpha_freq": sum(1 for p in profiles if p.get("orpha_freq")),
        "orpha_freq_links": sum(len(p.get("orpha_freq", [])) for p in profiles),
        "orpha_excluded_links": sum(1 for p in profiles for _t, f in p.get("orpha_freq", []) if f == "EX"),
        **hpo_stats,
        "terms_orphanet_new": sum(1 for tid in terms if tid.startswith("HP:")),
        "terms_orphanet_new_with_korean": sum(1 for tid, t in terms.items() if tid.startswith("HP:") and t["ko"]),
    }
    prev = parse_prevalence()
    stats["prevalence_kcd3_codes"] = len(prev)
    stats["profiles_with_prevalence"] = sum(1 for p in profiles if any(c[:3] in prev for c, _ in
                                                                      p["codes"].get("kcd", [])))
    versions = {"DO": do_version, "MedlinePlus": mplus_date, "Orphanet": orpha["version"], "files": files,
                "built": dt.date.today().isoformat()}
    return {"profiles": profiles, "terms": terms, "kcd": kcd, "stats": stats, "versions": versions,
            "ddx_map": ddx_map, "prev": prev}


def write(kb: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"schema": 1, "sources": SOURCES, "versions": kb["versions"], "stats": kb["stats"],
            "test_refs": kb_tests.REFS,
            "orpha_freq": {"O": "Obligate (100%)", "VF": "Very frequent (99-80%)", "F": "Frequent (79-30%)",
                           "OC": "Occasional (29-5%)", "VR": "Very rare (<4-1%)", "EX": "Excluded (0%)"},
            "hpo": "Human Phenotype Ontology files not included (license forbids altering file content); Orphanet "
                   "(CC BY 4.0) HPO-coded annotations are re-keyed to KB terms."}
    body = {"meta": meta, "terms": kb["terms"], "diseases": kb["profiles"]}
    def gz(path: Path):  # mtime=0: identical inputs give byte-identical files (reproducible rebuilds)
        return io.TextIOWrapper(gzip.GzipFile(str(path), "wb", compresslevel=9, mtime=0), encoding="utf-8")
    with gz(OUT / "kb.json.gz") as f:
        json.dump(body, f, ensure_ascii=False, separators=(",", ":"))
    with gz(OUT / "kcd.tsv.gz") as f:
        f.write("code\tko_names\ten_name\tsex\n")
        for code in sorted(kb["kcd"]):
            r = kb["kcd"][code]
            f.write(f"{code}\t{'|'.join(r['ko'][:4])}\t{r['en'][0] if r['en'] else ''}\t{r.get('sex', '')}\n")
    with gz(OUT / "kcd3_prev.tsv.gz") as f:  # HIRA 2025 patients per KCD 3-char code: 18 male + 18 female age groups
        f.write("code\tmale_0-4..85+\tfemale_0-4..85+\n")
        for code in sorted(kb["prev"]):
            v = kb["prev"][code]
            f.write(f"{code}\t{','.join(map(str, v[:18]))}\t{','.join(map(str, v[18:]))}\n")
    lines =["# data/kb sources and attribution", "",
             "Built by scripts/build_kb.py. See docs/licenses.md for the full license ledger.", ""]
    for k, s in SOURCES.items():
        lines.append(f"- **{k}**: {s['title']}. License: {s['license']}. {s['url']} — {s['attribution']}")
    lines += ["", f"Versions: DO {kb['versions']['DO']}; MedlinePlus XML {kb['versions']['MedlinePlus']}; "
              f"Orphadata {kb['versions']['Orphanet']}; built {kb['versions']['built']}.",
              "Human Phenotype Ontology file (hp.obo) and HPO annotations (phenotype.hpoa): not used. Orphanet "
              "phenotype annotations are HPO-coded; they are stored against KB term ids, with HPO ids only as "
              "identifiers of terms that have no other KB equivalent.",
              "kcd3_prev.tsv.gz: HIRA patients per KCD 3-character code by sex and age group (source HIRA-stats)."]
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in OUT.iterdir():
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="use cached LLM outputs only")
    ap.add_argument("--refresh", action="store_true", help="re-download all inputs")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    files = download(args.refresh)
    cache = LLMCache(enabled=not args.no_llm, workers=args.workers)
    kb = assemble(cache, files)
    write(kb)
    print(json.dumps(kb["stats"], indent=1))
    unmapped = [c for c, v in kb["ddx_map"].items() if not v]
    if unmapped:
        print("DDXPlus conditions kept as standalone profiles:", unmapped)
    if cache.failures:
        print("LLM failures:", cache.failures[:10])
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {"runs": []}
    last = meta["runs"][-1] if meta["runs"] else {}
    # log every build that calls the LLM, and every cache-only build whose output changed (new sources, new stats)
    if cache.calls or not last or last.get("stats") != kb["stats"] or last.get("versions", {}).get("built") != \
            kb["versions"]["built"]:
        cfg = cache.cfg
        meta["runs"].append({
            "date": dt.datetime.now().isoformat(timespec="seconds"),
            # cache-only build: the LLM outputs it uses come from the earlier runs' model
            "model": cfg.model if cfg else next((r["model"] for r in reversed(meta["runs"]) if r.get("model")), None),
            "temperature": cfg.temperature if cfg else None,
            "reasoning_effort": cfg.reasoning_effort if cfg else None, "workers": args.workers,
            "llm_calls": cache.calls, "failures": cache.failures,
            "mode": "cache-only (--no-llm)" if args.no_llm else "llm",
            "prompts": {"term_ko": TERM_PROMPT, "ddx_evidence": EVID_PROMPT, "mplus_extract": MPLUS_PROMPT,
                        "disease_ko": NAME_PROMPT},
            "cache": str(CACHE.relative_to(ROOT)), "stats": kb["stats"], "versions": kb["versions"],
            "note": "Outputs are cached per item in the cache file; MedlinePlus extractions keep only English terms "
                    "grounded in the source summary text (others are listed under 'dropped'). Orphanet phenotypes, "
                    "Wikidata HPO/Orphanet bridges and the HIRA prevalence table are processed without any LLM.",
        })
        META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
