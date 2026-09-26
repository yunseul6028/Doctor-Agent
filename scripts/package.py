"""Builds the submission ZIP + rule checks. Owned by compliance-release.

python scripts/package.py  →  dist/submission_YYYYMMDD_HHMMSS.zip
"""
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["run.py", "requirements.txt", "src", "eval/__init__.py", "eval/simulator.py", "data/kb"]
MAX_BYTES = 50 * 1024 * 1024
FORBIDDEN_IMPORTS = re.compile(r"^\s*(import|from)\s+(requests|httpx|aiohttp|urllib\.request|anthropic|google\.generativeai|google\.genai|dotenv)\b", re.M)
FORBIDDEN_FILES = re.compile(r"\.(safetensors|bin|pt|pth|gguf|ckpt)$")


def collect() -> list[Path]:
    files = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    return files


def check(files: list[Path]) -> list[str]:
    errors = []
    names = {f.relative_to(ROOT).as_posix() for f in files}
    for required in ("run.py", "requirements.txt"):
        if required not in names:
            errors.append(f"missing {required}")
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        if rel.startswith("data/cases") or rel.startswith("data/sample_cases") or rel.startswith("data/labels"):
            errors.append(f"evaluation/labeling data must not be packaged: {rel}")
        if FORBIDDEN_FILES.search(rel):
            errors.append(f"weight-like file not allowed: {rel}")
        if f.suffix == ".py":
            try:
                src = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"not UTF-8: {rel}")
                continue
            if m := FORBIDDEN_IMPORTS.search(src):
                errors.append(f"external network import in {rel}: {m.group(0).strip()}")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and "==" not in line:
            errors.append(f"unpinned requirement: {line}")
    return errors


def main() -> None:
    files = collect()
    errors = check(files)
    if errors:
        print("NO-GO\n  " + "\n  ".join(errors))
        sys.exit(1)
    out = ROOT / "dist" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    size = out.stat().st_size
    if size > MAX_BYTES:
        print(f"NO-GO: {size / 1e6:.1f}MB > 50MB")
        sys.exit(1)
    print(f"GO: {out.relative_to(ROOT)} ({size / 1e3:.1f}KB, {len(files)} files)")


if __name__ == "__main__":
    main()
