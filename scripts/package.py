"""Builds the submission ZIP + rule checks. Owned by compliance-release.

    python scripts/package.py              # checks + ZIP + smoke run inside an extracted copy of the ZIP
    python scripts/package.py --strict     # submission day: submission blockers (official adapter, time budget) fail too
    python scripts/package.py --no-smoke   # skip the extracted-ZIP smoke run (faster; not for submission)

Output: dist/submission_YYYYMMDD_HHMMSS.zip and a machine-readable report next to it (.report.json).

Three severities:
  ERROR    the ZIP breaks a competition rule or is not self-contained -> NO-GO, nothing is kept
  BLOCKER  fine for a dev build, but must be fixed before a real submission (e.g. official env adapter missing)
           -> "dev GO / submission NO-GO"; with --strict these count as errors
  WARN     worth a look (reproducibility gaps, hygiene); never blocks

Checks (see docs/submission-checklist.md for the item numbers):
  1  run.py + requirements.txt at the ZIP root, ZIP <= 50MB
  2  no network libraries / provider SDKs / non-allowlisted hosts / provider names / .env reads in shipped code
  3  no weight or adapter files
  4  every case calls the LLM (loop guard present, dummy smoke run shows llm_calls >= 1 for every case)
  5  no cross-case state (module-level mutable caches that get mutated, memoisation decorators, file writes)
  6  requirements pinned, minimal and declared; every shipped citation / KB source / package in docs/licenses.md
  7  UTF-8 for every shipped text file (including gzipped KB files)
  8  smoke run of run.py from a clean extracted copy of the ZIP (isolated interpreter, scrubbed environment)
  +  reproducibility of LLM-derived artifacts in data/labels (WARN only; not shipped)
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["run.py", "requirements.txt", "src", "eval/__init__.py", "eval/simulator.py", "data/kb", "data/lexicon"]
MAX_BYTES = 50 * 1024 * 1024
SMOKE_TIMEOUT_S = 300

# Dev-tree paths that must never be shipped. Exceptions: eval/simulator.py (+ package marker) backs
# `run.py --env local` (the keyword case-file environment). It only reads a case JSON; no network, no LLM.
DEV_ONLY_PREFIXES = ("eval/", "scripts/", "tests/", "data/cases", "data/sample_cases", "data/labels", "data/external",
                     "dist/", ".claude/", ".venv/", ".git/")
SHIPPED_DEV_EXCEPTIONS = {"eval/__init__.py", "eval/simulator.py"}
SECRET_FILES = re.compile(r"(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa.*|credentials.*\.json)$")

# Network / provider modules. `openai` is the one allowed client (fixed LLM endpoint only).
FORBIDDEN_MODULES = {
    "requests", "httpx", "aiohttp", "urllib3", "urllib.request", "http.client", "socket", "ssl", "ftplib", "smtplib",
    "telnetlib", "xmlrpc", "websocket", "websockets", "grpc", "pycurl", "anthropic", "google", "vertexai", "cohere",
    "mistralai", "together", "groq", "litellm", "langchain", "langchain_core", "langchain_openai", "llama_index",
    "dotenv", "huggingface_hub", "transformers", "peft", "torch", "vllm", "subprocess", "webbrowser",
}
FORBIDDEN_CALLS = {"os.system", "os.popen", "os.exec", "os.spawn"}
# Provider names that must not appear anywhere (code or comments) in shipped files.
PROVIDER_NAMES = re.compile(r"generativelanguage|googleapis|gemini|anthropic|openrouter|api\.openai\.com", re.I)
# URL hosts allowed as string literals: the local default endpoint and citation links (displayed, never fetched).
ALLOWED_URL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "doi.org", "pubmed.ncbi.nlm.nih.gov"}
URL = re.compile(r"https?://([A-Za-z0-9.\-]+|\{[^}]*\})")

WEIGHT_FILES = re.compile(r"\.(safetensors|bin|pt|pth|gguf|ggml|ckpt|onnx|h5|hdf5|pb|tflite|msgpack|pkl|pickle|joblib|"
                          r"npy|npz|faiss|index)$", re.I)
ADAPTER_FILES = re.compile(r"(^|/)(adapter_config\.json|adapter_model\.[a-z]+|lora[^/]*|pytorch_model[^/]*|"
                           r"model\.safetensors\.index\.json|tokenizer\.json|config\.json)$", re.I)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".tsv", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini"}

# Distribution name -> import name, where they differ.
IMPORT_NAMES = {"openai": "openai"}
LOCAL_PACKAGES = {"doctor_agent", "eval"}

# Module-level mutable objects that persist across cases, reviewed by hand. Anything not listed fails check 5.
CROSS_CASE_ALLOWLIST = {
    "src/doctor_agent/llm/client.py:_STRUCTURED_REJECTED":
        "(base_url, model) pairs whose server rejected structured output: server capability only, never case content",
    "src/doctor_agent/knowledge/kb.py:_KB":
        "read-only KnowledgeBase singleton loaded from data/kb; its lazy indexes are derived from data/kb files only",
}
MUTATORS = {"append", "add", "update", "setdefault", "extend", "insert", "pop", "popitem", "clear", "remove",
            "discard", "__setitem__", "appendleft", "extendleft"}
MEMO_DECORATORS = {"lru_cache", "cache", "cached_property", "memoize"}
# Files allowed to write to disk: run.py writes the predictions (+ its own per-run .jsonl log, read back only with
# --resume to skip already-finished case ids; no content of one case feeds another).
WRITE_ALLOWED = {"run.py"}

DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'`<>|]+")
PMID = re.compile(r"(?:pmid[\"']?\s*[:=]\s*[\"']?|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{6,9})", re.I)
NC_ND = re.compile(r"\b(CC[- ]BY[- ](NC|ND|NC[- ]ND|NC[- ]SA))\b|non-?commercial|no-?deriv|unclear|unknown license|"
                   r"license\s*(tbd|unknown)", re.I)


@dataclass
class Finding:
    level: str  # ERROR | BLOCKER | WARN
    check: str  # checklist item, e.g. "2-network"
    msg: str


# ----------------------------------------------------------------------------------------------------------- collect
def collect(root: Path = ROOT, include: list[str] | None = None) -> list[Path]:
    files = []
    for item in include or INCLUDE:
        p = root / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts
                            and f.name != ".DS_Store" and f.suffix != ".pyc")
    return files


def _rel(f: Path, root: Path) -> str:
    return f.relative_to(root).as_posix()


def _read(f: Path) -> str | None:
    try:
        return f.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _module_root(name: str) -> str:
    return name.split(".")[0]


# -------------------------------------------------------------------------------------------- 1, 3, 7: layout/files
def check_layout(files: list[Path], root: Path) -> list[Finding]:
    out = []
    names = {_rel(f, root) for f in files}
    for required in ("run.py", "requirements.txt"):
        if required not in names:
            out.append(Finding("ERROR", "1-layout", f"missing {required} at the ZIP root"))
    for rel in sorted(names):
        if rel.startswith(DEV_ONLY_PREFIXES) and rel not in SHIPPED_DEV_EXCEPTIONS:
            out.append(Finding("ERROR", "1-dev-only", f"dev-only / evaluation / labeling file must not ship: {rel}"))
        if SECRET_FILES.search(rel):
            out.append(Finding("ERROR", "2-secrets", f"secret/env file must not ship: {rel}"))
        if WEIGHT_FILES.search(rel) or ADAPTER_FILES.search(rel):
            out.append(Finding("ERROR", "3-weights", f"weight/adapter-like file not allowed: {rel}"))
    return out


def check_utf8(files: list[Path], root: Path) -> list[Finding]:
    out = []
    for f in files:
        rel = _rel(f, root)
        try:
            if f.suffix in TEXT_SUFFIXES:
                f.read_bytes().decode("utf-8")
            elif f.suffix == ".gz":
                with gzip.open(f, "rb") as g:
                    g.read().decode("utf-8")
        except UnicodeDecodeError as e:
            out.append(Finding("ERROR", "7-utf8", f"not UTF-8: {rel} ({e.reason} at byte {e.start})"))
        except OSError as e:
            out.append(Finding("ERROR", "7-utf8", f"unreadable: {rel} ({e})"))
    return out


# ------------------------------------------------------------------------------------------------- 2, 6: code/deps
def parse_requirements(path: Path) -> tuple[dict[str, str], list[Finding]]:
    reqs: dict[str, str] = {}
    out = []
    if not path.exists():
        return reqs, [Finding("ERROR", "6-requirements", "requirements.txt missing")]
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([A-Za-z0-9_.+\-]+)", line)
        if not m:
            out.append(Finding("ERROR", "6-requirements", f"unpinned or unparsable requirement: {line}"))
            continue
        reqs[m.group(1).lower()] = m.group(3)
    return reqs, out


def _imports(tree: ast.AST) -> list[tuple[str, int]]:
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append((node.module, node.lineno))
            mods += [(f"{node.module}.{a.name}", node.lineno) for a in node.names]
    return mods


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check_code(files: list[Path], root: Path, reqs: dict[str, str]) -> list[Finding]:
    out = []
    declared = {IMPORT_NAMES.get(d, d.replace("-", "_")) for d in reqs}
    used_roots: set[str] = set()
    for f in files:
        rel = _rel(f, root)
        if f.suffix not in (".py", ".md", ".txt", ".json"):
            continue
        src = _read(f)
        if src is None:
            continue  # reported by check_utf8
        if m := PROVIDER_NAMES.search(src):
            line = src.count("\n", 0, m.start()) + 1
            out.append(Finding("ERROR", "2-provider", f"other-provider name {m.group(0)!r} in shipped file {rel}:{line}"))
        if f.suffix != ".py":
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as e:
            out.append(Finding("ERROR", "8-syntax", f"syntax error in {rel}: {e}"))
            continue
        for mod, line in _imports(tree):
            top = _module_root(mod)
            used_roots.add(top)
            hit = next((x for x in FORBIDDEN_MODULES if mod == x or mod.startswith(x + ".")), None)
            if hit:
                out.append(Finding("ERROR", "2-network", f"forbidden import {mod!r} in {rel}:{line}"))
            elif top not in sys.stdlib_module_names and top not in LOCAL_PACKAGES and top not in declared:
                out.append(Finding("ERROR", "6-undeclared",
                                   f"import {mod!r} in {rel}:{line} is neither stdlib nor in requirements.txt"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if any(name == c or name.startswith(c) for c in FORBIDDEN_CALLS) or name in ("__import__", "eval", "exec"):
                    out.append(Finding("ERROR", "2-network", f"forbidden call {name}() in {rel}:{node.lineno}"))
                if name.endswith("load_dotenv"):
                    out.append(Finding("ERROR", "2-dotenv", f".env loading in {rel}:{node.lineno}"))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if s.strip() in (".env", "dotenv") or s.endswith("/.env"):
                    out.append(Finding("ERROR", "2-dotenv", f"reference to {s!r} in {rel}:{node.lineno}"))
                for host in URL.findall(s):
                    if not host.startswith("{") and host.lower() not in ALLOWED_URL_HOSTS:
                        out.append(Finding("ERROR", "2-hosts", f"non-allowlisted host {host!r} in {rel}:{node.lineno}"))
            if isinstance(node, ast.JoinedStr):
                for v in node.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        for host in URL.findall(v.value):
                            if host and not host.startswith("{") and host.lower() not in ALLOWED_URL_HOSTS:
                                out.append(Finding("ERROR", "2-hosts",
                                                   f"non-allowlisted host {host!r} in {rel}:{node.lineno}"))
    for dist in reqs:
        if IMPORT_NAMES.get(dist, dist.replace("-", "_")) not in used_roots:
            out.append(Finding("ERROR", "6-minimal",
                               f"requirement {dist!r} is not imported by any shipped file (dev-only? use "
                               f"requirements-dev.txt)"))
    return out


# ------------------------------------------------------------------------------------------ 5: cross-case state
_MUTABLE_CALLS = {"dict", "list", "set", "defaultdict", "OrderedDict", "Counter", "deque", "bytearray"}


def _is_mutable_value(v: ast.AST | None) -> bool:
    if v is None:
        return False
    if isinstance(v, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(v, ast.Call):
        return _dotted(v.func).split(".")[-1] in _MUTABLE_CALLS
    return isinstance(v, ast.Constant) and v.value is None  # lazily filled singleton (needs `global` to mutate)


_WRITE_METHODS = {"write_text", "write_bytes", "touch", "unlink", "mkdir", "rmdir", "symlink_to", "hardlink_to"}
_WRITE_FUNCS = {"os.replace", "os.rename", "os.remove", "os.unlink", "os.makedirs", "os.mkdir", "shutil.copy",
                "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.rmtree", "json.dump", "pickle.dump",
                "marshal.dump", "shelve.open", "sqlite3.connect", "dbm.open"}


def _is_file_write(call: ast.Call) -> bool:
    name = _dotted(call.func)
    last = name.split(".")[-1]
    if name in _WRITE_FUNCS or (isinstance(call.func, ast.Attribute) and last in _WRITE_METHODS):
        return True
    if last == "open":
        # open(path, mode) / gzip.open(path, mode) vs. Path.open(mode)
        path_method = isinstance(call.func, ast.Attribute) and name.split(".")[0] not in (
            "gzip", "bz2", "lzma", "io", "codecs", "tarfile", "zipfile", "os", "builtins")
        pos = 0 if path_method else 1
        mode = call.args[pos] if len(call.args) > pos else next(
            (k.value for k in call.keywords if k.arg == "mode"), None)
        if mode is None:
            return False
        if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
            return True  # computed mode: assume it may write
        return any(c in mode.value for c in "wax+")
    return False


def check_cross_case(files: list[Path], root: Path) -> list[Finding]:
    out = []
    for f in files:
        if f.suffix != ".py":
            continue
        rel = _rel(f, root)
        src = _read(f)
        if src is None:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        module_mut: dict[str, int] = {}
        for node in tree.body:
            targets, value = [], None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            if _is_mutable_value(value):
                for t in targets:
                    if isinstance(t, ast.Name):
                        module_mut[t.id] = node.lineno
        mutated: dict[str, int] = {}
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if not isinstance(fn, ast.Lambda):
                for dec in fn.decorator_list:
                    name = _dotted(dec.func if isinstance(dec, ast.Call) else dec).split(".")[-1]
                    if name in MEMO_DECORATORS and f"{rel}:{fn.name}" not in CROSS_CASE_ALLOWLIST:
                        out.append(Finding("ERROR", "5-cross-case",
                                           f"memoisation decorator @{name} on {fn.name} in {rel}:{fn.lineno} "
                                           f"(results would persist across cases)"))
            for node in ast.walk(fn):
                if isinstance(node, ast.Global):
                    for n in node.names:
                        mutated.setdefault(n, node.lineno)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and isinstance(node.func.value, ast.Name) and node.func.attr in MUTATORS:
                    mutated.setdefault(node.func.value.id, node.lineno)
                elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                    tgts = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                    for t in tgts:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                            mutated.setdefault(t.value.id, node.lineno)
        for name, line in module_mut.items():
            if name in mutated and f"{rel}:{name}" not in CROSS_CASE_ALLOWLIST:
                out.append(Finding("ERROR", "5-cross-case",
                                   f"module-level mutable {name!r} ({rel}:{line}) is modified at line {mutated[name]} "
                                   f"and would carry state across cases"))
        if rel not in WRITE_ALLOWED:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _is_file_write(node):
                    out.append(Finding("ERROR", "5-file-write", f"file write {_dotted(node.func)}() in "
                                                                f"{rel}:{node.lineno} (only run.py may write outputs)"))
    return out


# ------------------------------------------------------------------------------------ 4 + knobs: static rule checks
def check_rules(root: Path) -> list[Finding]:
    out = []

    def text(rel: str) -> str:
        p = root / rel
        return p.read_text(encoding="utf-8") if p.exists() else ""

    run, loop, cfg = text("run.py"), text("src/doctor_agent/agent/loop.py"), text("src/doctor_agent/config.py")
    policy = text("src/doctor_agent/agent/policy.py")
    if "run_case(" not in run:
        out.append(Finding("ERROR", "4-llm-call", "run.py does not route cases through run_case()"))
    if "Fixed LLM was not called" not in loop or "llm_calls" not in loop:
        out.append(Finding("ERROR", "4-llm-call", "loop.run_case lost the >=1 LLM call per case guard"))
    if "self.llm.chat(" not in policy or "state.view()" not in policy:
        out.append(Finding("ERROR", "4-llm-call", "policy no longer sends the case view (state.view()) to the LLM"))
    if "guard.attempts == 0" not in loop:
        out.append(Finding("ERROR", "4-llm-call",
                           "forced-diagnosis path no longer guarantees an LLM attempt when the case had none"))
    if not re.search(r"cfg\.agent\.submission\s*=\s*not\s+args\.dev", run):
        out.append(Finding("ERROR", "8-submission-mode", "run.py must default to submission mode (not args.dev)"))
    if not re.search(r'"--llm".*default="openai"', run):
        out.append(Finding("ERROR", "8-submission-mode", "run.py --llm must default to the real (openai) client"))
    if "case_time_budget_s" not in cfg or "AGENT_CASE_TIME_BUDGET_S" not in cfg:
        out.append(Finding("ERROR", "8-time-budget", "per-case time budget knob (AGENT_CASE_TIME_BUDGET_S) missing"))
    elif re.search(r'AGENT_CASE_TIME_BUDGET_S",\s*0(\.0)?\)', cfg) and not os.getenv("AGENT_CASE_TIME_BUDGET_S"):
        out.append(Finding("BLOCKER", "8-time-budget",
                           "per-case time budget defaults to 0 (unlimited): set it below the official limit "
                           "(default in config.py or the server's env) once the guide publishes it"))
    official = text("src/doctor_agent/env/official.py")
    if "NotImplementedError" in official:
        out.append(Finding("BLOCKER", "8-official-env", "env/official.py is still a stub (participant guide pending)"))
    if re.search(r'DOCTOR_ENV"\)\s*or\s*"local"', text("src/doctor_agent/env/factory.py")):
        out.append(Finding("BLOCKER", "8-official-env",
                           "default environment is 'local': the server run must select the official adapter "
                           "(default or documented DOCTOR_ENV/--env official in the run command)"))
    if "TODO" in run and "write_outputs" in run and "official output format" in run:
        out.append(Finding("BLOCKER", "8-output-format",
                           "run.py output format not yet confirmed against the participant guide (TODO in run.py)"))
    return out


# ------------------------------------------------------------------------------------------ 6: license ledger audit
def check_ledger(files: list[Path], root: Path, reqs: dict[str, str]) -> list[Finding]:
    out = []
    ledger_path = root / "docs/licenses.md"
    if not ledger_path.exists():
        return [Finding("ERROR", "6-ledger", "docs/licenses.md missing")]
    ledger = ledger_path.read_text(encoding="utf-8")
    low = ledger.lower()
    # packages
    for dist, ver in reqs.items():
        rows = [r for r in ledger.splitlines() if r.startswith("|") and dist in r.lower()]
        if not rows:
            out.append(Finding("ERROR", "6-ledger", f"requirement {dist}=={ver} has no row in docs/licenses.md"))
        elif not any(ver in r for r in rows):
            out.append(Finding("ERROR", "6-ledger", f"ledger row for {dist} does not record the pinned version {ver}"))
    # citations in shipped code: a line citing a DOI/PMID is covered when any of its ids is in the ledger
    n_cites = 0
    for f in files:
        if f.suffix != ".py" or (src := _read(f)) is None:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            ids = [d.rstrip(".,;") for d in DOI.findall(line) if "{" not in d] + PMID.findall(line)
            if not ids:
                continue
            n_cites += 1
            if not any(x.lower() in low for x in ids):
                out.append(Finding("ERROR", "6-ledger",
                                   f"citation {ids} in {_rel(f, root)}:{i} has no row in docs/licenses.md"))
    # KB sources and test references recorded in the shipped KB
    kb = root / "data/kb/kb.json.gz"
    if kb.exists():
        with gzip.open(kb, "rt", encoding="utf-8") as g:
            meta = json.load(g).get("meta", {})
        for key, s in (meta.get("sources") or {}).items():
            lic = str(s.get("license", ""))
            if not lic:
                out.append(Finding("ERROR", "6-ledger", f"KB source {key} has no license in kb meta"))
            if NC_ND.search(lic):
                out.append(Finding("ERROR", "6-ledger", f"KB source {key} is shipped under a restricted license: {lic}"))
            urls = [u for u in (s.get("url"), s.get("license_url")) if u]
            if urls and not any(u.lower().rstrip("/") in low for u in urls):
                out.append(Finding("ERROR", "6-ledger", f"KB source {key} ({urls[0]}) has no row in docs/licenses.md"))
        for key, ref in (meta.get("test_refs") or {}).items():
            pm = str(ref.get("pmid", "")) if isinstance(ref, dict) else ""
            if pm and pm not in low:
                out.append(Finding("ERROR", "6-ledger", f"KB test reference {key} (PMID {pm}) not in docs/licenses.md"))
    # shipped ledger rows: license present, no NC/ND/unclear, URL present unless self-authored/derived
    for line in ledger.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 7 or cells[0] in ("Resource", "---") or set(cells[0]) <= {"-"}:
            continue
        name, lic, url, shipped = cells[0], cells[3], cells[4], cells[-1]
        if not shipped.startswith("✓"):
            continue
        if not lic or lic in ("–", "-", "?"):
            out.append(Finding("ERROR", "6-ledger", f"shipped ledger row without license: {name[:80]}"))
        elif NC_ND.search(lic):
            out.append(Finding("ERROR", "6-ledger", f"shipped ledger row with restricted/unclear license: {name[:80]} "
                                                    f"({lic[:80]})"))
        if (not url or url in ("–", "-")) and not re.search(r"self-authored|derived|our own", lic, re.I):
            out.append(Finding("WARN", "6-ledger", f"shipped ledger row without source URL: {name[:80]}"))
    if n_cites == 0:
        out.append(Finding("WARN", "6-ledger", "no citations found in shipped code (citation scan broken?)"))
    return out


# ------------------------------------------------------------------------------- reproducibility (data/labels, WARN)
# (case directory, glob, labels meta file, how meta lists item ids)
REPRO_SETS = [
    ("data/cases_clinicalqa", "*.json", "clinicalqa_conversion_meta.json", "items:cqa_"),
    ("data/cases_agentclinic", "*.json", "agentclinic_conversion_meta.json", "items:ac_"),
    ("data/cases_diagnosisarena", "*.json", "diagnosisarena_conversion_meta.json", "items:da_"),
    ("data/cases_aug", "*/*.json", "augmentation_full_meta.json", "runs"),
    ("data/cases_clinicalqa_aug", "*.json", "augmentation_meta.json", "runs"),
]
PROMPT_KEYS = ("system_prompt", "prompts", "user_template")


def _has_provenance(block: dict) -> list[str]:
    missing = [k for k in ("model", "date") if not block.get(k)]
    if not any(block.get(k) for k in PROMPT_KEYS):
        missing.append("prompt")
    return missing


def check_repro(root: Path) -> list[Finding]:
    out = []
    labels = root / "data/labels"
    for case_dir, pattern, meta_name, how in REPRO_SETS:
        d = root / case_dir
        if not d.exists():
            continue
        meta_path = labels / meta_name
        if not meta_path.exists():
            out.append(Finding("WARN", "repro", f"{case_dir}: LLM-derived cases without {meta_path.relative_to(root)}"))
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids: set[str] = set()
        if how.startswith("items:"):
            prefix = how.split(":", 1)[1]
            if miss := _has_provenance(meta):
                out.append(Finding("WARN", "repro", f"{meta_name}: missing {', '.join(miss)}"))
            ids = {f"{prefix}{k}" for k, v in (meta.get("items") or {}).items()
                   if not isinstance(v, dict) or v.get("status", "converted") == "converted"}
        else:
            for i, run in enumerate(meta.get("runs") or []):
                if miss := _has_provenance(run):
                    out.append(Finding("WARN", "repro", f"{meta_name} run {i}: missing {', '.join(miss)}"))
                ids |= {str(it.get("id")) for it in run.get("items") or [] if isinstance(it, dict)}
        stems = {p.stem for p in d.glob(pattern)}
        if unrecorded := sorted(stems - ids):
            out.append(Finding("WARN", "repro", f"{case_dir}: {len(unrecorded)} file(s) with no generation record in "
                                                f"{meta_name}: {', '.join(unrecorded[:5])}"))
    kb_meta = labels / "kb_build_meta.json"
    if (root / "data/kb/kb.json.gz").exists():
        if not kb_meta.exists():
            out.append(Finding("WARN", "repro", "data/kb built with LLM labels but data/labels/kb_build_meta.json missing"))
        else:
            runs = json.loads(kb_meta.read_text(encoding="utf-8")).get("runs") or []
            for i, run in enumerate(runs):
                if miss := _has_provenance(run):
                    out.append(Finding("WARN", "repro", f"kb_build_meta.json run {i}: missing {', '.join(miss)}"))
            with gzip.open(root / "data/kb/kb.json.gz", "rt", encoding="utf-8") as g:
                built = json.load(g).get("meta", {}).get("versions", {}).get("built", "")
            if runs and built and not any(str(r.get("date", "")).startswith(built) for r in runs):
                out.append(Finding("WARN", "repro", f"shipped KB built {built} has no matching run in kb_build_meta.json "
                                                    f"(cache-only rebuild: LLM outputs come from kb_llm_cache.json; "
                                                    f"consider logging no-LLM rebuilds too)"))
    if (root / "data/sample_cases").exists():
        out.append(Finding("WARN", "repro", "data/sample_cases: written by Claude (self-authored), no generation "
                                            "prompt/model record; documented in docs/licenses.md only"))
    return out


# --------------------------------------------------------------------------------------------- build + smoke run
def build_zip(files: list[Path], root: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, _rel(f, root))


def check_zip(zip_path: Path) -> list[Finding]:
    out = []
    size = zip_path.stat().st_size
    if size > MAX_BYTES:
        out.append(Finding("ERROR", "1-size", f"ZIP is {size / 1e6:.1f}MB > 50MB"))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if bad := z.testzip():
            out.append(Finding("ERROR", "1-layout", f"corrupt ZIP member: {bad}"))
    for required in ("run.py", "requirements.txt"):
        if required not in names:
            out.append(Finding("ERROR", "1-layout", f"{required} not at the ZIP root"))
    return out


def _clean_env() -> dict[str, str]:
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "LOG_LEVEL": "WARNING"})
    return env  # no DOCTOR_*/LLM_*/AGENT_*/API keys, no PYTHONPATH


def _snapshot(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def smoke_test(zip_path: Path, cases: Path, reqs: dict[str, str], python: str = sys.executable,
               timeout: float = SMOKE_TIMEOUT_S) -> tuple[list[Finding], dict]:
    """Extract the ZIP into a clean temp dir and run `run.py --llm dummy` there with an isolated interpreter (-I: no
    PYTHONPATH, no user site, no cwd on sys.path) and a scrubbed environment. Proves the ZIP is self-contained, every
    case ends with a diagnosis after >=1 LLM call, and nothing is written inside the code tree."""
    out: list[Finding] = []
    info: dict = {}
    n_cases = 1 if cases.is_file() else len(list(cases.glob("*.json")))
    with tempfile.TemporaryDirectory(prefix="nova_smoke_") as tmp:
        code, work = Path(tmp) / "submission", Path(tmp) / "out"
        work.mkdir()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(code)
        before = _snapshot(code)
        imports = sorted({IMPORT_NAMES.get(d, d.replace("-", "_")) for d in reqs})
        probe = ("import sys, pkgutil, importlib; sys.path[:0] = ['src', '.']; import doctor_agent; "
                 "[importlib.import_module(m.name) for m in pkgutil.walk_packages(doctor_agent.__path__, 'doctor_agent.')]; "
                 + "".join(f"import {m}; " for m in imports))
        try:
            p = subprocess.run([python, "-I", "-c", probe], cwd=code, env=_clean_env(), capture_output=True,
                               text=True, timeout=60)
            if p.returncode != 0:
                out.append(Finding("ERROR", "8-smoke", f"import probe failed in the extracted ZIP: {p.stderr[-800:]}"))
        except subprocess.TimeoutExpired:
            out.append(Finding("ERROR", "8-smoke", "import probe timed out"))
        preds = work / "predictions.json"
        cmd = [python, "-I", "run.py", "--env", "local", "--llm", "dummy", "--cases", str(cases.resolve()),
               "--out", str(preds)]
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=code, env=_clean_env(), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return out + [Finding("ERROR", "8-smoke", f"smoke run exceeded {timeout:.0f}s")], info
        info["smoke_sec"] = round(time.monotonic() - t0, 1)
        if p.returncode != 0:
            return out + [Finding("ERROR", "8-smoke", f"run.py exited {p.returncode}: {p.stderr[-1500:]}")], info
        try:
            pred_map = json.loads(preds.read_text(encoding="utf-8"))
            rows = [json.loads(x) for x in (work / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x]
        except (OSError, ValueError) as e:
            return out + [Finding("ERROR", "8-smoke", f"smoke outputs unreadable: {e}")], info
        info.update({"smoke_cases": n_cases, "smoke_predictions": len(pred_map),
                     "smoke_llm_calls": [r.get("llm_calls") for r in rows]})
        if len(pred_map) != n_cases or len(rows) != n_cases:
            out.append(Finding("ERROR", "8-smoke", f"{len(pred_map)} predictions / {len(rows)} log rows for "
                                                   f"{n_cases} cases"))
        for r in rows:
            if not (isinstance(r.get("llm_calls"), int) and r["llm_calls"] >= 1):
                out.append(Finding("ERROR", "4-llm-call", f"case {r.get('case_id')}: llm_calls={r.get('llm_calls')}"))
            if r.get("error"):
                out.append(Finding("ERROR", "8-smoke", f"case {r.get('case_id')} errored: {r['error']}"))
            if not r.get("diagnosis"):
                out.append(Finding("ERROR", "8-smoke", f"case {r.get('case_id')}: empty diagnosis"))
        if written := sorted(_snapshot(code) - before):
            out.append(Finding("ERROR", "5-file-write", f"run wrote files into the code tree: {written[:5]}"))
    return out, info


# ------------------------------------------------------------------------------------------------------------ main
def run_checks(root: Path = ROOT, files: list[Path] | None = None) -> list[Finding]:
    files = collect(root) if files is None else files
    reqs, findings = parse_requirements(root / "requirements.txt")
    findings += check_layout(files, root)
    findings += check_utf8(files, root)
    findings += check_code(files, root, reqs)
    findings += check_cross_case(files, root)
    findings += check_rules(root)
    findings += check_ledger(files, root, reqs)
    findings += check_repro(root)
    return findings


def check(files: list[Path]) -> list[str]:
    """Backward-compatible: blocking error messages only."""
    return [f.msg for f in run_checks(ROOT, files) if f.level == "ERROR"]


def _print(findings: list[Finding]) -> None:
    for level in ("ERROR", "BLOCKER", "WARN"):
        items = [f for f in findings if f.level == level]
        if items:
            print(f"{level} ({len(items)})")
            for f in items:
                print(f"  [{f.check}] {f.msg}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="submission day: BLOCKER findings also fail")
    ap.add_argument("--no-smoke", action="store_true", help="skip the extracted-ZIP smoke run")
    ap.add_argument("--cases", default=str(ROOT / "data/sample_cases"), help="cases for the smoke run")
    ap.add_argument("--out-dir", default=str(ROOT / "dist"))
    args = ap.parse_args(argv)

    files = collect(ROOT)
    findings = run_checks(ROOT, files)
    reqs, _ = parse_requirements(ROOT / "requirements.txt")
    info: dict = {"files": len(files)}
    out = Path(args.out_dir) / f"submission_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    if not any(f.level == "ERROR" for f in findings):
        build_zip(files, ROOT, out)
        findings += check_zip(out)
        info.update({"zip": str(out), "zip_bytes": out.stat().st_size,
                     "sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
        if not args.no_smoke:
            smoke, sinfo = smoke_test(out, Path(args.cases), reqs)
            findings += smoke
            info.update(sinfo)
        else:
            findings.append(Finding("BLOCKER", "8-smoke", "smoke run skipped (--no-smoke)"))

    errors = [f for f in findings if f.level == "ERROR"]
    blockers = [f for f in findings if f.level == "BLOCKER"]
    fail = bool(errors) or (args.strict and bool(blockers))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", None
    info.update({"commit": commit, "dirty": dirty})
    if dirty:
        findings.append(Finding("WARN", "history", "working tree has uncommitted changes: the ZIP does not match a "
                                                   "commit (commit before a real submission)"))
    _print(findings)
    if fail and out.exists():
        out.unlink()
    if out.exists():
        report = out.with_suffix(".report.json")
        report.write_text(json.dumps({"info": info, "findings": [asdict(f) for f in findings]}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")
    verdict_dev = "NO-GO" if errors else "GO"
    verdict_sub = "NO-GO" if errors or blockers else "GO"
    size = f"{info['zip_bytes'] / 1e3:.1f}KB, " if "zip_bytes" in info else ""
    where = f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} " if out.exists() else ""
    print(f"\nDEV BUILD: {verdict_dev}   REAL SUBMISSION: {verdict_sub}   "
          f"({where}{size}{len(files)} files, commit {commit}{'+dirty' if dirty else ''}, "
          f"{len(errors)} errors / {len(blockers)} blockers / {sum(f.level == 'WARN' for f in findings)} warnings)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
