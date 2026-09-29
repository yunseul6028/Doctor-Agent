"""Tests for scripts/package.py (submission build + rule checks)."""
import gzip
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nova_package", ROOT / "scripts/package.py")
pkg = importlib.util.module_from_spec(_spec)
sys.modules["nova_package"] = pkg
_spec.loader.exec_module(pkg)

LEDGER_HEAD = ("| Resource | Type | Version / revision | License | Source URL | Use | Included in submission |\n"
               "|---|---|---|---|---|---|---|\n"
               "| openai (python) | Library | 1.109.1 | Apache-2.0 | https://github.com/openai/openai-python | client | ✓ |\n")


def make_tree(tmp_path: Path, files: dict[str, str | bytes], ledger_extra: str = "",
              requirements: str = "openai==1.109.1\n") -> Path:
    root = tmp_path / "repo"
    base = {
        "run.py": "from doctor_agent.agent import loop\n",
        "requirements.txt": requirements,
        "src/doctor_agent/__init__.py": "",
        "src/doctor_agent/llm/client.py": "def make():\n    from openai import OpenAI\n    return OpenAI\n",
        "docs/licenses.md": LEDGER_HEAD + ledger_extra,
    }
    base.update(files)
    for rel, content in base.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    return root


def findings(root: Path, include=None, levels=("ERROR",)) -> list:
    files = pkg.collect(root, include or ["run.py", "requirements.txt", "src", "eval", "data"])
    reqs, out = pkg.parse_requirements(root / "requirements.txt")
    out += pkg.check_layout(files, root) + pkg.check_utf8(files, root) + pkg.check_code(files, root, reqs)
    out += pkg.check_cross_case(files, root) + pkg.check_ledger(files, root, reqs)
    return [f for f in out if f.level in levels]


def checks(fs) -> set[str]:
    return {f.check for f in fs}


def test_clean_tree_passes(tmp_path):
    assert findings(make_tree(tmp_path, {})) == []


@pytest.mark.parametrize("line", ["import requests", "from httpx import Client", "import urllib.request",
                                  "from urllib import request", "import socket", "import subprocess",
                                  "from google import genai", "import anthropic", "from dotenv import load_dotenv"])
def test_network_and_provider_imports_blocked(tmp_path, line):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": line + "\n"})
    assert "2-network" in checks(findings(root))


def test_os_system_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": "import os\nos.system('curl x')\n"})
    assert "2-network" in checks(findings(root))


@pytest.mark.parametrize("text", ["# fallback to gemini when slow\n", "URL = 'https://generativelanguage.googleapis.com'\n",
                                  "NAME = 'Anthropic'\n"])
def test_provider_names_blocked(tmp_path, text):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": text})
    assert "2-provider" in checks(findings(root))


def test_dotenv_reference_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": "from pathlib import Path\nP = Path('.env')\n"})
    assert "2-dotenv" in checks(findings(root))


def test_env_mentioned_in_comment_is_fine(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": "# set values in .env (dev only)\nX = 1\n"})
    assert findings(root) == []


def test_foreign_host_blocked_but_citations_and_localhost_allowed(tmp_path):
    ok = make_tree(tmp_path, {"src/doctor_agent/x.py":
                              "A = 'http://localhost:11434/v1'\nB = 'https://doi.org/10.1/x'\npmid = '1'\n"
                              "def u(p):\n    return f'https://pubmed.ncbi.nlm.nih.gov/{p}/'\n"},
                   ledger_extra="| x | Paper | – | Facts | https://doi.org/10.1/x | code | ✓ (code) |\n")
    assert findings(ok) == []
    bad = make_tree(tmp_path / "b", {"src/doctor_agent/x.py": "U = 'https://api.example.com/v1'\n"})
    assert "2-hosts" in checks(findings(bad))


def test_undeclared_third_party_import_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": "import numpy\nimport json\nfrom doctor_agent import llm\n"})
    msgs = [f.msg for f in findings(root)]
    assert len(msgs) == 1 and "'numpy'" in msgs[0]


def test_requirements_pinned_and_minimal(tmp_path):
    root = make_tree(tmp_path, {}, requirements="openai==1.109.1\npytest\npyarrow==25.0.1\n",
                     ledger_extra="| pyarrow | Library | 25.0.1 | Apache-2.0 | https://x | dev | ✗ |\n")
    fs = findings(root)
    assert any("unpinned" in f.msg and "pytest" in f.msg for f in fs)
    assert any(f.check == "6-minimal" and "pyarrow" in f.msg for f in fs)


def test_requirement_missing_from_ledger(tmp_path):
    root = make_tree(tmp_path, {"docs/licenses.md": "# empty\n"})
    assert any("openai" in f.msg for f in findings(root) if f.check == "6-ledger")


@pytest.mark.parametrize("rel", ["eval/run_local.py", "eval/judge.py", "scripts/build_kb.py", "data/labels/x.json",
                                 "data/cases_aug/a.json", "data/sample_cases/s.json", "data/external/raw.txt"])
def test_dev_only_paths_blocked(tmp_path, rel):
    root = make_tree(tmp_path, {rel: "{}"})
    fs = findings(root, include=["run.py", "requirements.txt", "src", rel.split("/")[0] + "/" + rel.split("/")[1]])
    assert "1-dev-only" in checks(fs)


def test_simulator_exception_allowed(tmp_path):
    root = make_tree(tmp_path, {"eval/__init__.py": "", "eval/simulator.py": "import json\n"})
    assert findings(root, include=["run.py", "requirements.txt", "src", "eval/__init__.py", "eval/simulator.py"]) == []


@pytest.mark.parametrize("rel", ["data/kb/model.safetensors", "data/kb/lora.bin", "data/kb/adapter_config.json",
                                 "data/kb/emb.pt", "data/kb/index.pkl", "data/kb/w.gguf"])
def test_weights_and_adapters_blocked(tmp_path, rel):
    root = make_tree(tmp_path, {rel: b"\x00\x01"})
    assert "3-weights" in checks(findings(root))


def test_secret_env_file_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/.env": "KEY=1\n"})
    assert "2-secrets" in checks(findings(root))


def test_non_utf8_blocked_including_gzip(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": "X = '한글'\n".encode("cp949"),
                                "data/kb/t.tsv.gz": gzip.compress("a\tb\n".encode("utf-16"))})
    bad = [f.msg for f in findings(root) if f.check == "7-utf8"]
    assert any("x.py" in m for m in bad) and any("t.tsv.gz" in m for m in bad)


def test_mutated_module_cache_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py":
                                "_SEEN = {}\n\ndef f(k, v):\n    _SEEN[k] = v\n\n"
                                "_LOG = []\n\ndef g(x):\n    _LOG.append(x)\n\n"
                                "_ONE = None\n\ndef h():\n    global _ONE\n    _ONE = 1\n"})
    msgs = [f.msg for f in findings(root) if f.check == "5-cross-case"]
    assert len(msgs) == 3


def test_read_only_module_tables_are_fine(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py":
                                "TABLE = {'a': 1}\nLST = [1, 2]\n\ndef f(k):\n    d = {}\n    d[k] = TABLE.get(k)\n"
                                "    return d, list(LST)\n"})
    assert findings(root) == []


def test_allowlisted_cache_is_fine(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/llm/client.py":
                                "_STRUCTURED_REJECTED = set()\n\ndef f(k):\n    _STRUCTURED_REJECTED.add(k)\n"
                                "    from openai import OpenAI\n"})
    assert findings(root) == []


def test_memoisation_decorator_blocked(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py":
                                "import functools\n\n@functools.lru_cache(maxsize=None)\ndef f(x):\n    return x\n"})
    assert "5-cross-case" in checks(findings(root))


@pytest.mark.parametrize("code", ["from pathlib import Path\nPath('c.json').write_text('x')\n",
                                  "open('c.json', 'w').write('x')\n",
                                  "from pathlib import Path\nPath('c.jsonl').open('a')\n",
                                  "import json\njson.dump({}, open('x', 'w'))\n"])
def test_file_writes_outside_run_py_blocked(tmp_path, code):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py": code})
    assert "5-file-write" in checks(findings(root))


def test_file_reads_are_fine(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py":
                                "import gzip\nfrom pathlib import Path\n\ndef f(p):\n"
                                "    with gzip.open(p, 'rt', encoding='utf-8') as g:\n        g.read()\n"
                                "    Path(p).open().read()\n    open(p).read()\n    Path(p).read_text()\n"})
    assert findings(root) == []


def test_citation_missing_from_ledger(tmp_path):
    root = make_tree(tmp_path, {"src/doctor_agent/x.py":
                                "R = [dict(doi='10.1000/abc', pmid='12345678')]\nS = [dict(pmid='87654321')]\n"},
                     ledger_extra="| paper | Paper | – | Facts | https://pubmed.ncbi.nlm.nih.gov/12345678/ | code | ✓ |\n")
    msgs = [f.msg for f in findings(root) if f.check == "6-ledger"]
    assert len(msgs) == 1 and "87654321" in msgs[0]


def test_nc_license_shipped_blocked(tmp_path):
    root = make_tree(tmp_path, {}, ledger_extra="| Some KB | Data | v1 | CC BY-NC 4.0 | https://x.org | KB | ✓ |\n"
                                                "| Dev set | Data | v1 | CC BY-NC-SA 4.0 | https://y.org | eval | ✗ |\n")
    msgs = [f.msg for f in findings(root) if f.check == "6-ledger"]
    assert len(msgs) == 1 and "Some KB" in msgs[0]


def test_kb_meta_sources_checked(tmp_path):
    meta = {"meta": {"sources": {"A": {"license": "CC0 1.0", "url": "https://a.org/data"},
                                 "B": {"license": "CC BY-ND 4.0", "url": "https://b.org"}},
                     "test_refs": {"r": {"pmid": "99999999"}}}}
    root = make_tree(tmp_path, {"data/kb/kb.json.gz": gzip.compress(json.dumps(meta).encode())},
                     ledger_extra="| A | Data | – | CC0 1.0 | https://a.org/data | kb | ✓ |\n")
    msgs = " ".join(f.msg for f in findings(root) if f.check == "6-ledger")
    assert "B" in msgs and "restricted" in msgs and "99999999" in msgs and "source A" not in msgs


def test_repro_flags_unrecorded_llm_cases(tmp_path):
    meta = {"model": "m", "date": "2026-09-26", "system_prompt": "p", "items": {"1": {"status": "converted"}}}
    root = make_tree(tmp_path, {"data/cases_clinicalqa/cqa_1.json": "{}", "data/cases_clinicalqa/cqa_2.json": "{}",
                                "data/labels/clinicalqa_conversion_meta.json": json.dumps(meta)})
    msgs = [f.msg for f in pkg.check_repro(root)]
    assert any("cqa_2" in m for m in msgs) and not any("missing model" in m for m in msgs)
    meta.pop("model")
    (root / "data/labels/clinicalqa_conversion_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert any("missing model" in f.msg for f in pkg.check_repro(root))


# ------------------------------------------------------------------------------------------------ real repository
def test_repository_has_no_blocking_errors():
    errors = [f"{f.check}: {f.msg}" for f in pkg.run_checks(ROOT) if f.level == "ERROR"]
    assert errors == []


def test_repository_rule_guards_present():
    fs = pkg.check_rules(ROOT)
    assert not [f for f in fs if f.level == "ERROR"], fs


def test_built_zip_is_self_contained_and_every_case_calls_llm(tmp_path):
    files = pkg.collect(ROOT)
    out = tmp_path / "sub.zip"
    pkg.build_zip(files, ROOT, out)
    assert pkg.check_zip(out) == []
    reqs, _ = pkg.parse_requirements(ROOT / "requirements.txt")
    cases = tmp_path / "cases"
    cases.mkdir()
    for name in ("synthetic_001.json", "synthetic_002.json"):
        shutil.copy(ROOT / "data/sample_cases" / name, cases / name)
    fs, info = pkg.smoke_test(out, cases, reqs, timeout=120)
    assert fs == [], fs
    assert info["smoke_predictions"] == 2 and all(n >= 1 for n in info["smoke_llm_calls"])


def test_smoke_test_catches_a_broken_zip(tmp_path):
    files = [f for f in pkg.collect(ROOT) if "knowledge/kb.py" not in f.as_posix()]
    out = tmp_path / "broken.zip"
    pkg.build_zip(files, ROOT, out)
    reqs, _ = pkg.parse_requirements(ROOT / "requirements.txt")
    fs, _ = pkg.smoke_test(out, ROOT / "data/sample_cases/synthetic_001.json", reqs, timeout=120)
    assert any(f.level == "ERROR" for f in fs)


# --- offline eval scripts (eval/offline/*.py; dev-only, never shipped) -------------------------------------------

def test_offline_script_check_flags_absolute_paths_and_missing_guard(tmp_path):
    d = tmp_path / "eval/offline"
    d.mkdir(parents=True)
    (d / "eval_bad.py").write_text('CASES = "/Users/someone/repo/data/cases_aug"\nprint(CASES)\n', encoding="utf-8")
    (d / "eval_ok.py").write_text('import sys\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n'
                                  'def main():\n    return 0\n\nif __name__ == "__main__":\n    sys.exit(main())\n',
                                  encoding="utf-8")
    problems = pkg.offline_script_problems(tmp_path, run_help=True, timeout=60)
    assert any("eval_bad.py:1: absolute path" in p for p in problems), problems
    assert any("eval_bad.py: no `if __name__" in p for p in problems), problems
    assert not any("eval_ok.py" in p for p in problems), problems
    assert {f.level for f in pkg.check_offline(tmp_path)} == {"WARN"}


def test_offline_scripts_have_no_absolute_paths_and_help_runs():
    """Every eval/offline script resolves paths from the repo root and `python -I <script> --help` (run from a temp
    dir, no PYTHONPATH) exits 0 without running the evaluation."""
    assert list(ROOT.glob(pkg.OFFLINE_GLOB))
    assert pkg.offline_script_problems(ROOT, run_help=True) == []
