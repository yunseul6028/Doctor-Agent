# Doctor-Agent — conversational diagnosis agent for a small open-weight LLM

Personal project. A doctor agent interviews a virtual patient (ASK), picks physical exams (EXAM) and tests (TEST),
and commits to a diagnosis (DIAGNOSE). The design target is a **small open-weight model, `openai/gpt-oss-20b`**,
chosen on purpose: the question is how far a ~20B model gets when **the model reasons and the code verifies**.
Core idea: **efficient hard-coding** — the code keeps the memory (ledgers), reads results, checks safety and decides
verdicts; the model proposes one next action per turn and fills fixed fields. Rules should be few, sourced, shared
and generalisable (README chapter 2.1).
Overview and evidence: `README.md`. Architecture: `docs/architecture.md`.

## Design invariants (keep these when changing code)
- **The doctor LLM is always in the loop**: every case calls the doctor model at least once with a prompt containing
  that case's information (`agent/loop.py` guarantees it; dev mode asserts it). No rule-only paths.
- **No network calls from inference code** other than the doctor LLM endpoint (OpenAI-compatible). External LLMs are
  used only offline (case conversion, KB build, labelling), and their outputs are cached and recorded.
- **The GPU belongs to the LLM.** KB, lexicon, result interpreter, routing and safety layers run on CPU/RAM with the
  standard library only.
- **Cases are independent**: state is created per case; no cross-case caches, predictions or statistics
  (read-only static data such as the KB is fine).
- **Robust mode** (default in `run.py`; `--dev` turns it off): exceptions never leave a case and every case ends
  with a DIAGNOSE.
- **Licenses**: every external data source, model and tool has an entry in `docs/licenses.md` before it is used.
  Unclear, NC or ND licenses stay out of anything the runtime loads.
- **Internal-only data**: the AgentClinic- and DiagnosisArena-derived case sets (and gold labels quoting them) are
  for internal evaluation only and must not be redistributed (`docs/data-sources.md`, "공개 저장소로 만들 때").
- **Reproducibility**: self-labelling code, prompts and model versions are kept (`data/labels/` + generation scripts).

## Team agents (`.codex/agents/`)
| Agent | Owns | Main paths |
|---|---|---|
| `clinical-strategist` | Clinical reasoning: history-taking strategy, DDx, exam/test selection, safety red flags | `src/doctor_agent/agent/prompts.py`, `src/doctor_agent/safety/` |
| `agent-engineer` | Agent loop, state management, LLM client, output parsing, turn budget | `src/doctor_agent/agent/`, `src/doctor_agent/llm/`, `run.py` |
| `knowledge-rag` | Medical knowledge base, CPU retrieval, license ledger | `src/doctor_agent/knowledge/`, `data/kb/`, `docs/licenses.md` |
| `eval-simulator` | Virtual patient simulator, local scorer, experiment logs | `eval/`, `data/sample_cases/` |

Offline scripts in `scripts/` belong to the agent whose area they serve (KB build → `knowledge-rag`, labelling and
offline evaluation → `eval-simulator`, token budget → `agent-engineer`).
Team setup: 1 person + agent team. The main session is the team lead: it splits work, assigns agents, and integrates results.
Independent work runs in parallel. When an interface changes, update `docs/architecture.md` first.

## Development commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                          # enter the API key / endpoint
python eval/run_local.py                      # doctor, patient, judge all on LLM (reads .env) → viewer opens when done
python eval/viewer.py                         # view past results in the browser
python eval/play.py [--role patient]          # interactive test: you play the doctor (or the patient)
python eval/run_local.py --doctor dummy --patient keyword --judge none   # smoke test without an LLM
python eval/experiment.py --profile smoke --doctor-endpoint local        # 5 cases on a local gpt-oss-20b (Ollama)
pytest
```

## LLM setup
- Roles: doctor agent (`DOCTOR_LLM_*`), virtual patient (`PATIENT_LLM_*`), judge (`JUDGE_LLM_*`); falls back to the shared `LLM_*` when unset
- **Design target for the doctor**: `openai/gpt-oss-20b` through any OpenAI-compatible endpoint (vLLM, Ollama
  `gpt-oss:20b`, a hosted API) — edit `.env` only, no code changes. The code default is local Ollama `gpt-oss:20b`.
- **So far**: every LLM-scored run used Gemini/Gemma models as the doctor (development stand-ins); gpt-oss-20b has not
  been scored yet.
- ⚠️ Prompts and thresholds tuned on Gemini must be re-validated on gpt-oss-20b. Keep experiment records separated by model.
- Keep Gemini-specific code (SDKs, etc.) out of `src/`. The runtime only uses the shared OpenAI-compatible client.

## Conventions
- Put every environment dependency behind `env/interface.py`. A new case source or patient environment is only a new adapter.
- Put every prompt in `agent/prompts.py`, and record version changes in `docs/experiments.md`.
- Include per-metric scores (Accuracy / Efficiency / Safety — our evaluation framework, `eval/scorer.py`) in every experiment result.
- Every component has an ablation switch (`AGENT_USE_*`); a component stays only if it earns its place on measured scores.
