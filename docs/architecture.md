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
- `safety/protocols.py`: per chief complaint, can't-miss diagnoses + minimum safety checks, each with a guideline citation (verification level recorded per item)
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

## Open interfaces (confirm after the participant guide is published)
- Action/response format, diagnosis format (free text vs. code such as ICD), time limit, LLM endpoint on the server
- The current `env/interface.py` is an assumption. Only add an adapter once the guide arrives.
