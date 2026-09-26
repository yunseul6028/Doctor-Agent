# ⚠️ 응답 언어 규칙 (최우선)
**요약과 중요한 설명은 반드시 한국어로 쓴다.** 작업 결과 요약, 분석, 제안, 사용자에게 묻는 질문이 해당한다.
- 작업 중간의 짧은 진행 메모 같은 잡다한 부분은 언어를 신경 쓰지 않아도 된다.
- 긴 작업이나 보조 에이전트 보고 직후에 요약이 영어로 새기 쉽다. 요약을 보내기 전에 한국어인지 확인한다.
- 사용자가 여러 번 지적한 사항이다.

# Doctor-Agent — N.O.V.A. 2026 Submission

N.O.V.A. 2026 (Bundang Seoul National University Hospital, Medical AI Center) conversational medical diagnosis agent competition.
We build a Doctor Agent that interviews a virtual patient, picks exams and tests, and reaches a diagnosis.
Full summary of the competition site: `docs/competition.md`. Architecture: `docs/architecture.md`.

## Hard rules (violating these can invalidate the submission)
- Preliminary LLM is fixed: `openai/gpt-oss-20b` (revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`). Fine-tuning, LoRA, and loading weights or adapters are forbidden.
- **Every case must call the fixed LLM at least once with a prompt containing that case's information.** Rule-only paths are invalid.
- **The submitted inference code must not call external LLMs or APIs** (network calls forbidden). External LLMs may be used only for offline data processing and labeling.
- The GPU is reserved for the fixed LLM. RAG indexes and embedding models must run on **CPU/RAM within the time limit**.
- Cases are independent: never use information, predictions, or statistics from other cases (no cross-case caching or learning).
- Submission: ZIP containing `run.py` + `requirements.txt`, **≤ 50MB**, Python, UTF-8. **One submission per day.**
- All external data, models, and tools must have a license that permits research publication → record them in `docs/licenses.md`.
- Reproducibility: self-labeling code, prompts, and model version must be kept (`data/labels/` + generation scripts).

## Team agents (`.claude/agents/`)
| Agent | Owns | Main paths |
|---|---|---|
| `clinical-strategist` | Clinical reasoning: history-taking strategy, DDx, exam/test selection, safety red flags | `src/doctor_agent/agent/prompts.py`, `src/doctor_agent/safety/` |
| `agent-engineer` | Agent loop, state management, LLM client, output parsing, turn budget | `src/doctor_agent/agent/`, `src/doctor_agent/llm/`, `run.py` |
| `knowledge-rag` | Medical knowledge base, CPU retrieval, license ledger | `src/doctor_agent/knowledge/`, `data/kb/`, `docs/licenses.md` |
| `eval-simulator` | Virtual patient simulator, local scorer, experiment logs | `eval/`, `data/sample_cases/` |
| `compliance-release` | Rule checks, packaging, submission checklist | `scripts/`, `docs/submission-checklist.md` |

Team setup: 1 person + agent team. The main session is the team lead: it splits work, assigns agents, and integrates results.
Independent work runs in parallel. When an interface changes, update `docs/architecture.md` first.

## Development commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                          # enter the API key
python eval/run_local.py                      # doctor, patient, judge all on LLM (reads .env) → viewer opens when done
python eval/viewer.py                         # view past results in the browser (not a submission artifact)
python eval/play.py [--role patient]          # interactive test: you play the doctor (or the patient)
python eval/run_local.py --doctor dummy --patient keyword --judge none   # smoke test without an LLM
python scripts/package.py                     # build the submission ZIP + rule checks
pytest
```

## LLM setup (current: temporary Gemini)
- Roles: doctor agent (`DOCTOR_LLM_*`), virtual patient (`PATIENT_LLM_*`), judge (`JUDGE_LLM_*`); falls back to the shared `LLM_*` when unset
- **Current**: all roles on Gemini (temporary, until the competition GPT API arrives)
- **Later**: switch only the doctor agent to the competition model `openai/gpt-oss-20b` (edit `.env` only, no code changes)
- ⚠️ Prompts and thresholds tuned on Gemini must be re-validated after switching models. Keep experiment records separated by model.
- Keep Gemini-specific code (SDKs, etc.) out of `src/`. The submission only uses the shared connection code.

## Conventions
- Put every environment dependency behind `env/interface.py`. Once the official participant guide API is published, only add an adapter.
- Put every prompt in `agent/prompts.py`, and record version changes in `docs/experiments.md`.
- Include per-metric scores (Accuracy / Efficiency / Safety) in every experiment result.
