# Architecture

```
run.py ──> Environment adapter (env/)          ← replaced once the official API is published
             │  observe() / step(action)
             ▼
        DoctorAgent (agent/loop.py)
             │
   ┌─────────┼───────────────┬──────────────┐
   ▼         ▼               ▼              ▼
CaseState  Policy          LLM client     Knowledge
(state.py) (policy.py)     (llm/)         (knowledge/)
 findings,  decides next    gpt-oss-20b    CPU BM25/small
 DDx,       action, applies (OpenAI-       embeddings
 turn count budget/safety   compatible)
             │
             ▼
        Safety rules (safety/) — red-flag checks, blocks premature diagnoses
```

## Core loop (per case)
1. `observe()` → initial information (demographics, chief complaint)
2. Repeat:
   - Update the DDx and uncertainty in `CaseState`
   - `Policy` asks the LLM for the next action as JSON: `{"type": "ASK|EXAM|TEST|DIAGNOSE", "content": ...}`
   - Enforce with rules: remaining turn budget, no repeated questions, unresolved red flags, forced DIAGNOSE on the last turn
   - `env.step(action)` → record the response as a finding
3. DIAGNOSE → end the case

## Grounding (verified sources)
- `safety/protocols.py`: per chief complaint, can't-miss diagnoses + minimum safety checks, each with a guideline citation (verification level recorded per item). 26 categories: chest pain, dyspnea, headache, acute neuro (stroke), fever, abdominal pain, allergy + (2026-09-26) syncope, palpitations, hemoptysis/chronic cough, jaundice (adult vs newborn), acute hot joint, low back pain, acute rash, chronic generalized pruritus, edema/foamy urine, amenorrhea/abnormal vaginal bleeding, fatigue ≥2 weeks, cognitive decline, psychiatric symptoms + (2026-09-27) chronic urticaria ≥6 weeks, hearing loss, adult neck mass, bleeding tendency, chronic limb weakness, infant bilious vomiting. Categories come from the chief complaint (`clinical_rules.detect_categories`); checks can be limited by duration (`acute_only`, `min_duration`), age (`min_age`), negation-aware triggers or a predicate (e.g. neonate, unilateral leg, postmenopausal bleeding)
- `knowledge/clinical_rules.py`: 12 published clinical decision rules (Wells PE, HEART, qSOFA, Ottawa SAH, …) implemented by us, with a citation and DOI each
- `knowledge/kb.py` (+ `knowledge/kb_curated.py` matching tables): citable disease profiles. `candidates(findings, k=10, negatives=None, sex=None, age=None)` (sex "남성"/"여성" and age in years are optional keyword args; older calls unchanged), `patient_profile(initial_info) -> (sex, age)`, `normalize_diagnosis(text)` (result `match` says which stage matched: `kcd_exact`, `profile`, `superstring`, `kcd_backoff`, `profile_backoff`, `contained`, `fuzzy`), `lookup`, `discriminators`. Offline benchmark: `scripts/eval_kb.py`
- Policy puts can't-miss diagnoses, **pending** safety checks, and at most 2 relevant rules into the hint every turn (based on initial info + everything learned so far)
- If a diagnosis is proposed while safety checks are still pending, push back once (diagnosing without them requires a stated reason)
- The agent cites rule/guideline names in `reason` → visible per turn in the viewer
- Scoring: `safety` = share of protocol checks done (guideline-based); `case_checks` = the case author's checklist (reference only)

## Design principles
- **Budget**: at most 60 turns, but Efficiency counts, so aim for a target turn count (e.g. 15–25) and stop at a confidence threshold.
- **Safety**: rule out dangerous differentials in the DDx (MI, PE, aortic dissection, sepsis, etc.) with questions and tests before diagnosing.
- **Robustness**: on LLM output parse failure, retry, then fall back to a rule. Keep a hard stop so the time limit is never exceeded.
- **Independence**: the agent object is created fresh per case. No global cache.

## Runtime (competition robustness)
- **Entry point** `run.py`: case source chosen by `--env local|official` or `DOCTOR_ENV` (`env/factory.py`).
  `env/local.py` = case files via the keyword simulator; `env/official.py` = TODO adapter (fill in once the guide is out).
  Submission mode by default (`--dev` turns it off). Output is incremental: one JSON line per case in `<out>.jsonl`
  (fsynced) + the `{case_id: diagnosis}` map rewritten atomically after each case; `--resume` skips finished cases.
- **Never crash, always answer** (`agent/loop.py`, `agent/runtime.py`): the policy talks to a per-case `GuardedLLM`.
  A failed LLM call returns "" (treated as a parse failure); after `max_llm_failures` consecutive failures the LLM is
  dropped for that case. Every case ends with a DIAGNOSE: forced final-diagnosis call if possible, else top DDx, else
  "진단 불가". In submission mode no exception leaves `run_case` (env errors, policy bugs, billing) and `run.py` has a
  second per-case guard. Dev mode (`AgentConfig.submission=False`, the eval harness default) still raises on bugs and
  billing errors and asserts the "≥1 LLM call per case" rule; submission mode logs that instead.
- **Time budget**: `AGENT_CASE_TIME_BUDGET_S` (0 = unlimited; set it below the official limit once known). Past
  `AGENT_DEGRADE_AT_FRAC` of the budget: no pre-diagnosis review / safety pushback, only the first 2 hints +
  `prompts.LOW_TIME_HINT`, reasoning effort "low". When `AGENT_FINAL_RESERVE_S` is left: forced final diagnosis
  (skipped below `min_call_s` → top DDx). Exploratory calls get a deadline that leaves the reserve untouched; every call
  also has a hard watchdog thread; the OpenAI SDK's own retries are off (ours are deadline-aware).
- **gpt-oss responses** (`llm/client.py`, `llm/harmony.py`): content preferred; harmony markers
  (`<|channel|>analysis/final<|message|>…`, `analysis…assistantfinal…`) stripped; reasoning read from
  `reasoning_content`/`reasoning`; if an action prompt's content has no JSON the reasoning is appended as
  `<analysis>…</analysis>` so the parser can fall back to the JSON written there. `finish_reason=length` with no
  answer → one retry with ×2 max_tokens (cap `max_tokens_cap`) and effort "low".
- **Structured output** (optional): `DOCTOR_LLM_STRUCTURED_OUTPUT=json_schema|guided_json` sends
  `parser.ACTION_SCHEMA` for step/final prompts only. A 400/422 disables it for the rest of the run (per server).
- **Prompt length**: `AGENT_MAX_VIEW_CHARS` (default 12000) caps `CaseState.view()`; cut order: oldest compact actions →
  long verbatim responses → fewer verbatim exchanges → oldest part of ledger lines → hard cut. Prompt sizes and runtime
  events are recorded in `result["runtime"]` (`prompt_chars_max/total`, `forced`, `degraded`, `llm_errors`, …).

## Open interfaces (confirm after the participant guide is published)
- Action/response format, diagnosis format (free text vs. code such as ICD), time limit, LLM endpoint on the server
- The current `env/interface.py` is an assumption. Only add an adapter once the guide arrives.
