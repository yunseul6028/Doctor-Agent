# Submission Checklist (one submission per day)

Owner: `compliance-release`. Every submission needs a **GO** on every line below. Any NO-GO blocks the submission;
record the reason in the history table instead.

## How to run

```bash
pytest                                   # includes tests/test_package.py (the checks below, on fixtures + this repo)
python scripts/package.py                # dev build: ERROR -> NO-GO; prints "DEV BUILD: … REAL SUBMISSION: …"
python scripts/package.py --strict       # submission day: BLOCKER findings fail too (exit code 1)
```

`package.py` writes `dist/submission_<ts>.zip` and `dist/submission_<ts>.report.json` (findings, ZIP size, sha256,
commit, smoke-run stats). Severities: **ERROR** = rule violation or broken ZIP (no ZIP kept); **BLOCKER** = OK for a
dev build, not for a real submission; **WARN** = review, never blocks.

## A. Automated (scripts/package.py — check id in brackets)

| # | Item | Check id | How it is verified |
|---|---|---|---|
| 1 | `run.py` + `requirements.txt` at ZIP root, ZIP ≤ 50MB, ZIP not corrupt | `1-layout`, `1-size` | file list + `zipfile.testzip()` |
| 1 | No dev-only files shipped (`eval/*`, `scripts/`, `tests/`, `data/cases*`, `data/sample_cases`, `data/labels`, `data/external`) | `1-dev-only` | path prefixes. **Only exception:** `eval/__init__.py` + `eval/simulator.py` (keyword case-file env behind `run.py --env local`; reads a JSON file, no network, no LLM) |
| 2 | No network libraries / provider SDKs (`requests`, `httpx`, `urllib.request`, `socket`, `subprocess`, `google.*`, `anthropic`, `dotenv`, …), no `os.system`/`eval`/`exec` | `2-network` | AST scan of every import (incl. function-level) and call |
| 2 | No other-provider names anywhere in shipped files (`gemini`, `generativelanguage`, `googleapis`, `anthropic`, `openrouter`, `api.openai.com`) | `2-provider` | text scan incl. comments |
| 2 | No `.env` reads / `load_dotenv`; no `.env`/key files shipped | `2-dotenv`, `2-secrets` | AST string constants + file names |
| 2 | URL literals only for allowlisted hosts: `localhost` (dev default endpoint) and citation links `doi.org`, `pubmed.ncbi.nlm.nih.gov` (displayed, never fetched) | `2-hosts` | AST string/f-string scan |
| 3 | No weights / adapters / pickles / vector indexes (`.safetensors .bin .pt .gguf .onnx .pkl .npz .faiss`, `adapter_config.json`, …) | `3-weights` | file names |
| 4 | Every case calls the LLM ≥ 1: `run.py` → `run_case`; `run_case` keeps the ≥1-call guard; policy sends `state.view()` (the case) to the LLM; forced-diagnosis path still attempts an LLM call when the case had none | `4-llm-call` | static guards + **smoke run: `llm_calls ≥ 1` in every per-case row** |
| 5 | No cross-case state: module-level mutable objects that are mutated, `@lru_cache`/`@cache`, file writes outside `run.py` | `5-cross-case`, `5-file-write` | AST scan + smoke run checks nothing was written into the code tree |
| 6 | `requirements.txt` pinned (`==`), minimal (every package imported by shipped code), every third-party import declared | `6-requirements`, `6-minimal`, `6-undeclared` | parse + AST |
| 6 | License ledger: every package (with pinned version), every DOI/PMID cited in shipped code, every KB source and KB test reference (`kb.json.gz` meta) has a row in `docs/licenses.md`; no shipped row / KB source under NC/ND/unclear license | `6-ledger` | cross-reference |
| 7 | UTF-8: every shipped text file, gzipped KB files decompressed | `7-utf8` | decode |
| 8 | Self-contained smoke run: ZIP extracted to a clean temp dir, `python -I run.py --env local --llm dummy --cases data/sample_cases` with a scrubbed environment (no `DOCTOR_*`/`LLM_*`/`AGENT_*`, no `PYTHONPATH`), 300 s timeout; import probe of every `doctor_agent` module + every requirement | `8-smoke` | subprocess |
| 8 | Submission mode is the `run.py` default (`--dev` turns it off); `--llm` defaults to `openai`; time-budget knob `AGENT_CASE_TIME_BUDGET_S` exists | `8-submission-mode`, `8-time-budget` | static |
| – | Reproducibility of LLM-derived artifacts (`data/labels/*meta*.json` has model + date + prompt; every converted/augmented case file has a record) | `repro` (WARN) | cross-reference (not shipped) |
| – | Offline eval scripts (`eval/offline/*.py`, not shipped): no machine-specific absolute paths (`/Users/`, `/home/`, `/tmp/`, `C:\Users\`), parse, `__main__` guard | `offline` (WARN) | text + AST scan; `tests/test_package.py` also runs `python -I <script> --help` from a temp dir and **fails** on any problem |

Reviewed cross-case allowlist (`CROSS_CASE_ALLOWLIST` in `package.py`; anything new must be reviewed and added there):

| Object | Why it is allowed |
|---|---|
| `llm/client.py:_STRUCTURED_REJECTED` | set of `(base_url, model)` whose server rejected structured output — a server capability, never case content |
| `knowledge/kb.py:_KB` | read-only `KnowledgeBase` singleton loaded from `data/kb`; its lazy indexes (`_bix`, `_kcd`) are built only from `data/kb` files |
| `run.py` file writes | predictions + per-run `.jsonl` log; the log is read back only with `--resume`, to skip case ids already finished (no content from one case reaches another) |

## B. Manual (tick on submission day; BLOCKER ids are what `--strict` enforces)

- [ ] `8-official-env` — `src/doctor_agent/env/official.py` implemented per the participant guide, and the server run selects it (`--env official` / `DOCTOR_ENV=official`, or make it the default)
- [ ] `8-time-budget` — `AGENT_CASE_TIME_BUDGET_S` set **below** the official per-case limit (default in `config.py` or the server env; keep `AGENT_FINAL_RESERVE_S` room for the final DIAGNOSE)
- [ ] `8-output-format` — `run.py` output (`write_outputs()`) matches the official format; remove the TODO in the `run.py` docstring once confirmed
- [ ] Doctor LLM settings point at the competition endpoint/model (`openai/gpt-oss-20b` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`) through the server's env, not `.env`; `DOCTOR_LLM_STRUCTURED_OUTPUT` / `REASONING_EFFORT` validated on that server
- [ ] Dry run on the **official sample cases** through the official adapter with the real LLM: every case has a diagnosis, `llm_calls ≥ 1`, no `run_error`, wall time per case within budget
- [ ] Prompts/thresholds re-validated on gpt-oss-20b (not only Gemini); scores recorded per metric (Accuracy / Efficiency / Safety) in `docs/experiments.md`
- [ ] `python scripts/package.py --strict` → `DEV BUILD: GO   REAL SUBMISSION: GO`, on a **clean, committed** tree (no `+dirty`)
- [ ] New dependencies / data / citations have rows in `docs/licenses.md` (the automated check enforces shipped ones)
- [ ] Row added to the submission history below (ZIP name, sha256 from the report, commit)

## Current audit (2026-09-29, commit `f2099cd`, clean tree)

**Dev build: GO. Real submission: NO-GO** (4 blockers, all waiting on the participant guide).

- ERROR: none. ZIP 4,066 KB (60 files: `run.py`, `requirements.txt`, `src/` (49), `eval/{__init__,simulator}.py`,
  `data/kb/` (5), `data/lexicon/` (3)), sha256 `66a0f42c…61ed4`. (09-27: 2,247 KB, 37 files; growth is the lexicon
  and the safety/knowledge modules.)
- Smoke run from the extracted ZIP: 16/16 sample cases, `llm_calls` 8–12 per case (dummy LLM), no errors, 3.9 s.
- Ledger check (`6-ledger`): no findings — every shipped citation, KB source, KB test reference and `openai==1.109.1`
  is in `docs/licenses.md`; no NC/ND/unclear license among shipped rows.
- BLOCKER: `env/official.py` stub; default env is `local`; per-case time budget is 0 (unlimited); output format
  unconfirmed (TODO in `run.py`).
- WARN (reproducibility, not shipped): `data/sample_cases` has no generation prompt/model record (written by Claude,
  documented in the ledger). The 09-27 KB cache-only-rebuild warning is gone. New `offline` check: no findings.
- Non-blocking hygiene: `KnowledgeBase.tpost` is a `defaultdict` read with `self.tpost[tid]`, so a lookup of a missing
  (curated, static) finding id inserts an empty list into the singleton. Keys come from the fixed `kb_tests` table, not
  from case text, so nothing case-specific leaks; switching to `.get(tid, ())` would make it strictly read-only.
  `data/lexicon/seed.tsv` (177 KB, build input for `scripts/build_lexicon.py`) ships but is not read at runtime;
  dropping it from `INCLUDE` would trim the ZIP (size is far under the limit, so no action needed).

## Submission history
| Date | ZIP | Commit | Local score | Official score |
|---|---|---|---|---|
