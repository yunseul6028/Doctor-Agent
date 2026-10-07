"""Static check: the agent's runtime code makes no network calls except through the OpenAI-compatible LLM client.

Mirrors the AST check of the former scripts/package.py on the files the agent runs with (run.py, src/, and
eval/simulator.py which backs `run.py --env local`). No code is executed: imports, calls and string literals are read
from the syntax tree.

- imports: only the standard library, the project's own packages, and `openai` (the one allowed client; it only
  talks to the configured LLM endpoint) — no HTTP/socket libraries, provider SDKs, model runtimes or .env loaders;
- calls: no os.system/os.popen/os.exec*/os.spawn*, no eval/exec/__import__, no load_dotenv;
- string literals: URLs only for the local endpoint, citation links (displayed, never fetched) and, in config.py
  only, the host of the default doctor-LLM endpoint (the one allowed endpoint; used only through the client).
"""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([ROOT / "run.py", ROOT / "eval" / "simulator.py", *(ROOT / "src").rglob("*.py")])

ALLOWED_THIRD_PARTY = {"openai"}
LOCAL_PACKAGES = {"doctor_agent", "eval"}
FORBIDDEN_MODULES = {
    "requests", "httpx", "aiohttp", "urllib3", "urllib.request", "http.client", "socket", "ssl", "ftplib", "smtplib",
    "telnetlib", "xmlrpc", "websocket", "websockets", "grpc", "pycurl", "anthropic", "google", "vertexai", "cohere",
    "mistralai", "together", "groq", "litellm", "langchain", "langchain_core", "langchain_openai", "llama_index",
    "dotenv", "huggingface_hub", "transformers", "peft", "torch", "vllm", "subprocess", "webbrowser",
}
FORBIDDEN_CALLS = ("os.system", "os.popen", "os.exec", "os.spawn")
ALLOWED_URL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "doi.org", "pubmed.ncbi.nlm.nih.gov"}
# the default doctor-LLM endpoint (Gemini OpenAI-compatible API) may be named in the config defaults only
ALLOWED_URL_HOSTS_BY_FILE = {"src/doctor_agent/config.py": {"generativelanguage.googleapis.com"}}
URL = re.compile(r"https?://([A-Za-z0-9.\-]+|\{[^}]*\})")


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.lineno
            for a in node.names:
                yield f"{node.module}.{a.name}", node.lineno


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _problems(path: Path) -> list[str]:
    rel = path.relative_to(ROOT)
    allowed_hosts = ALLOWED_URL_HOSTS | ALLOWED_URL_HOSTS_BY_FILE.get(rel.as_posix(), set())
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
    out = []
    for mod, line in _imports(tree):
        top = mod.split(".")[0]
        if any(mod == x or mod.startswith(x + ".") for x in FORBIDDEN_MODULES):
            out.append(f"{rel}:{line}: forbidden import {mod!r}")
        elif top not in sys.stdlib_module_names and top not in LOCAL_PACKAGES and top not in ALLOWED_THIRD_PARTY:
            out.append(f"{rel}:{line}: import {mod!r} is neither stdlib, local nor the allowed LLM client")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name.startswith(FORBIDDEN_CALLS) or name in ("__import__", "eval", "exec"):
                out.append(f"{rel}:{node.lineno}: forbidden call {name}()")
            if name.endswith("load_dotenv"):
                out.append(f"{rel}:{node.lineno}: .env loading")
        strings = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            strings += [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        for s in strings:
            for host in URL.findall(s):
                if host and not host.startswith("{") and host.lower() not in allowed_hosts:
                    out.append(f"{rel}:{node.lineno}: non-allowlisted URL host {host!r}")
    return out


def test_llm_endpoint_host_allowed_only_in_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    for rel in ("src/doctor_agent/config.py", "src/doctor_agent/other.py"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('URL = "https://generativelanguage.googleapis.com/v1beta/openai/"\n', encoding="utf-8")
    assert _problems(tmp_path / "src/doctor_agent/config.py") == []
    assert _problems(tmp_path / "src/doctor_agent/other.py")


def test_runtime_files_found():
    assert (ROOT / "run.py") in FILES and len(FILES) > 20


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_network_outside_llm_client(path):
    assert _problems(path) == []


def test_requirements_are_only_the_llm_client():
    reqs = [ln.split("==")[0].strip().lower() for ln in (ROOT / "requirements.txt").read_text("utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    assert set(reqs) <= ALLOWED_THIRD_PARTY, reqs


def test_checker_catches_violations(tmp_path, monkeypatch):
    bad = tmp_path / "bad.py"
    bad.write_text("import requests\nfrom urllib.request import urlopen\nimport os\nos.system('x')\n"
                   "URL = 'https://example.com/api'\nOK = 'http://localhost:11434/v1'\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    found = "\n".join(_problems(bad))
    assert "requests" in found and "urllib.request" in found and "os.system" in found
    assert "example.com" in found and "localhost" not in found
