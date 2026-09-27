# Architecture

Current as of 2026-09-27 (prompt `v6-kb-strict-review`). When an interface changes, update this file first.

```
run.py ──> case source (env/factory.py: local | official)          ← official.py is a TODO until the guide is out
             │  env.reset() / env.step(action)
             ▼
        run_case (agent/loop.py) ── CaseBudget + GuardedLLM (agent/runtime.py) ── LLM client (llm/)
             │                                                                    OpenAI-compatible, gpt-oss aware
             ▼
        Policy.next_action (agent/policy.py)
             │ hints ◄── safety/protocols.py        can't-miss dx + pending minimum checks (26 categories)
             │       ◄── knowledge/clinical_rules.py ≤2 applicable decision rules (12 rules)
             │       ◄── agent/kb_hints.py ◄── knowledge/kb.py (+ kb_curated.py, kb_tests.py, data/kb/)
             ▼
        CaseState (agent/state.py): findings ledger + DDx ledger (agent/ledger.py), turns, reviews
```

## Module map

| Path | Role |
|---|---|
| `run.py` | Submission entry point. Picks the case source, runs every case in submission mode, writes results incrementally. |
| `src/doctor_agent/config.py` | `LLMConfig` (per-role env: `DOCTOR_LLM_*` → `LLM_*` → defaults) and `AgentConfig` (turn cap, KB switch, runtime knobs). |
| `env/interface.py` | `Action`, `ActionType` (ASK/EXAM/TEST/DIAGNOSE), `Observation`, `Environment` — the only environment dependency. Assumed format until the guide is published. |
| `env/factory.py`, `env/local.py`, `env/official.py` | Case sources: `local` = case files through the keyword simulator (`eval/simulator.py`); `official` = placeholder adapter that raises a clear "not published yet" error. |
| `agent/loop.py` | `run_case`: one case end to end, never-crash wrapper, forced final diagnosis, result record. |
| `agent/policy.py` | Next-action decision: hints, LLM call, post-processing rules, structured pre-diagnosis review. |
| `agent/state.py`, `agent/ledger.py` | Per-case state; `FindingsLedger` (양성/음성/결과없음) and `DdxLedger` (p, status 유력/위험/배제, for/against) merged across turns; capped prompt view. |
| `agent/parser.py` | Extracts the last action JSON (skips `<think>/<analysis>` blocks); `ACTION_SCHEMA` for structured output. |
| `agent/text.py` | Char-bigram similarity, history-question detection, DDx name-variant matching (`same_dx`). |
| `agent/prompts.py` | All prompts (`SYSTEM`, `REVIEW_SYSTEM`, final prompt, `LOW_TIME_HINT`). Version changes → `docs/experiments.md`. |
| `agent/kb_hints.py` | KB → short hints (candidates, discriminators, diagnosis normalisation). Fail-safe: any KB error = no hint. |
| `agent/runtime.py` | `CaseBudget` (wall clock), `GuardedLLM` (failure cap, watchdog, deadlines, gpt-oss options, prompt-size stats). |
| `llm/client.py`, `llm/harmony.py` | `OpenAICompatClient` (retries, 429 wait, billing detection, length retry, structured output), `DummyLLM`; harmony-format cleanup. |
| `safety/protocols.py` | Chief-complaint safety protocols (can't-miss diagnoses + minimum checks with citations). `safety/rules.py` is a backward-compatible wrapper (`red_flags_for`). |
| `knowledge/clinical_rules.py` | 12 decision rules, category detection from the chief complaint, duration/age parsing, negation handling. |
| `knowledge/kb.py`, `kb_curated.py`, `kb_tests.py` | Knowledge base runtime (stdlib only, CPU), our matching tables, curated test-result → disease links. |
| `knowledge/retriever.py` | Generic `BM25Retriever`, not used by the agent (kb.py has its own weighted BM25). |

## Per-case flow

1. `run.py` gets `(case_id, env)` from the case source and calls `run_case(env, llm, cfg)` with a fresh LLM client.
2. `run_case` creates `CaseBudget`, `GuardedLLM`, `CaseState(initial_info=env.reset().text)` and a new `Policy`.
   Nothing is shared between cases (the KB is read-only data loaded once per process).
3. Each turn (up to `max_turns` = 60):
   - `budget.degraded()` switches the policy to low-time mode; `budget.must_finish()` ends exploration.
   - `Policy.next_action(state)` returns one action (see below); `env.step(action)` → `Turn(action, response, ddx snapshot)`.
   - DIAGNOSE (or `obs.done`) ends the loop.
4. No diagnosis yet (budget, LLM unavailable, policy/env errors, turn cap) → `_forced_diagnosis`: one final LLM call
   if time allows (always tried when the case has made no LLM call yet), else the top DDx, else "진단 불가".
5. The result records diagnosis, turns (action, reason, DDx, response), ledgers, reviews, `llm_calls`,
   `diagnosis_normalized` (KB name + KCD code, record only — the submitted text is unchanged) and `runtime` stats.
   Dev mode asserts the "≥1 LLM call per case" rule; submission mode logs it.

## Policy decision order (`Policy.next_action`)

Remaining turns ≤ 1 → final-diagnosis prompt. Otherwise the hints are built once, then up to 4 attempts
(`MAX_ATTEMPTS`); each retry adds one corrective hint:

1. **Parse** — `parse_action(raw)`: last valid action JSON; thinking blocks skipped. Failure → hint "JSON 한 줄로만".
2. **Normalize type** — `normalize_type`: EXAM/TEST that is really a history question (ends in "?/나요/세요…" or
   `looks_like_history_question`) becomes ASK.
3. **Ledger update** — `findings` and `ddx` from the same JSON are merged into the ledgers (same item/DDx by
   similarity or `same_dx`; later information wins; for/against lists keep the last 6).
4. **Dedupe** — a non-DIAGNOSE action of the same type with char-bigram similarity ≥ 0.7 to an earlier one is refused.
5. **Safety pushback** — DIAGNOSE while non-treatment protocol checks are pending, more than 5 turns left, not
   degraded, not pushed back yet → one hint listing the pending checks (diagnosing anyway requires a stated reason).
6. **Pre-diagnosis review** — DIAGNOSE with < 2 reviews so far (`MAX_REVIEWS`), more than 3 turns left, not degraded:
   the same LLM in a reviewer role fills fixed fields (`key_findings` 설명됨/설명 안 됨, `contradicting`,
   `confirmation`, `unresolved_danger`, `next`, optional `final_diagnosis` + `refine_evidence`). **Code decides**:
   hold (보류) only if there is at least one issue (unexplained finding, contradiction, no confirmation, open danger)
   *and* a usable non-DIAGNOSE `next` action; an unparseable review never blocks. On hold the reviewer's action is
   returned (or, if already done, the loop retries with the issues as a hint). On approval a renaming is accepted only
   if its cited evidence is grounded in this case's text; added location qualifiers, unsupported cause/trigger
   qualifiers and manifestation-on-cause names are refused. A KB sex-restriction warning is appended to the review view,
   and so is the criteria check of the proposed diagnosis (`diagnostic_criteria.render_for_review`), whose subtype
   decision can accept or refuse a subtype renaming.

All attempts used → final-diagnosis prompt.

## Hint composition (`Policy._hints`, in this order)

1. Can't-miss diagnoses for the detected categories (`protocols.cant_miss_for(initial_info)`).
2. Pending minimum safety checks with condition and citation (`protocols.pending_checks(initial, actions, learned)`;
   treatment-kind checks are excluded from pushback).
3. Requests answered "제공되지 않습니다" (up to the last 6): do not repeat, verify another way.
4. At most 2 applicable clinical rules (`clinical_rules.rules_for(initial)` → `render_for_prompt`).
5. Target-turn notice once `turn_count ≥ target_turns` (20).
6. KB hints (`kb_hints.step_hints`, only when `AGENT_USE_KB` ≠ 0): candidate hint and discriminator hint.

In low-time mode only the first 2 hints are kept, plus `prompts.LOW_TIME_HINT`. All hints are labelled as reference,
not evidence. Categories are detected from the **initial information only**; conditional checks may also use what was
learned later (triggers, predicates).

## Safety protocols and clinical rules

- `safety/protocols.py`: **26 categories**, 88 checks (59 test, 18 exam, 9 ask, 2 treatment), 112 can't-miss
  diagnoses, 43 guideline citations (bibliographic data checked against PubMed). Each check records
  `verification` = primary (34) / secondary (17) / unverified (37). Categories: chest pain, dyspnea, headache, acute
  neuro (stroke), fever, abdominal pain, allergy, syncope, palpitations, hemoptysis/chronic cough, jaundice (adult vs
  newborn), acute hot joint, low back pain, acute rash, chronic generalized pruritus, edema/foamy urine,
  amenorrhea/abnormal vaginal bleeding, fatigue ≥ 2 weeks, cognitive decline, psychiatric symptoms, chronic urticaria
  ≥ 6 weeks, hearing loss, adult neck mass, bleeding tendency, chronic limb weakness, infant bilious vomiting.
- A check can be limited by `acute_only`, `min_duration`, `min_age`, negation-aware `triggers` or one of 19 named
  `predicates` (e.g. sepsis suspicion, neonate, unilateral leg, postmenopausal bleeding, current pregnancy).
- `knowledge/clinical_rules.py`: **12 rules** — Wells PE, PERC, HEART, ADD-RS, qSOFA, CURB-65, Centor, Ottawa SAH,
  Canadian CT Head, ABCD2, Alvarado, BISAP — each with citation/DOI. `Rule.applies_to` = (category or rule keyword)
  AND `requires_any` AND NOT `excludes_any` AND duration below `chronic_cutoff` AND age ≥ `min_age`;
  `Rule.applicability` states which population sentence of the original abstract each condition comes from.
- Coverage on the 267 `data/cases_aug` cases (initial info only): 198 match a category, 173 have ≥ 1 applicable
  non-treatment check, 53 have an applicable rule.

### Diagnostic / classification criteria (`knowledge/diagnostic_criteria.py`, 2026-09-27)

- 15 sets: 2019 EULAR/ACR SLE, 2010 ACR/EULAR RA, 2022 ACR/EULAR Takayasu and GCA, AHA 2017 Kawasaki, 2023
  Duke-ISCVID IE, KDIGO 2012 AKI staging, 2024 DKA/HHS consensus, Light's criteria, Sepsis-3, 2015 Jones, 2017
  McDonald, ICHD-3 migraine vs tension-type, bipolar I vs II (DSM-5-TR content), 2015 ACR/EULAR gout. Each has a
  PubMed/Crossref-checked citation, `verification` (primary 8 / secondary 3 / unverified 4) and our own Korean wording.
- API: `criteria_for(dx_name)` (Korean/English names, synonyms, subtype names; `related` cross-checks such as an
  arterial stenosis → Takayasu; `exclude` e.g. 가성통풍, 알코올성 케톤산증), `evaluate(id, findings_text)` →
  `CriteriaResult` (per-item met / not_met / unknown, band, `decision` = diagnosis or subtype only when explicitly
  satisfied), `render_for_review(dx_name, findings_text)` (≤ 500 chars).
- Extraction is conservative: affirmed keyword (clause negation, `- 음성:` ledger lines) or a labelled number with the
  right unit; doctor questions and `- 결과없음:` lines are ignored. Headache/gout features are read only from
  sentences about the head/joints unless that is the chief complaint; secondary-headache red flags block a
  migraine/tension-type decision.
- Used only in the pre-diagnosis review: `_review` appends `render_for_review(proposed, initial info + responses +
  findings ledger)` to the reviewer's view; `_refinement_problem` first asks `_criteria_refinement`, which accepts a
  renaming to the subtype the criteria selected (no quoted evidence needed) and refuses a renaming to a different
  subtype of the same set (e.g. II형 when the findings show admission for mania). Otherwise the usual evidence checks.
- KDCA 법정감염병 진단·신고 기준 not implemented: KOGL type 4 (no modification).

## Knowledge base (CPU, stdlib only)

Data: `data/kb/kb.json.gz` (1.63 MB) + `kcd.tsv.gz` (0.46 MB), loaded once (≈0.3 s), no network. Sources: DDXPlus,
HIRA KCD master, Disease Ontology, Wikidata, MedlinePlus (+ our `curated` tables). 12,597 profiles, 2,996 terms
(incl. 262 `TF:` test-result terms), 21,124 KCD codes. Every field carries its source tags. Build and benchmark
details: `docs/data-sources.md`.

Interfaces (`knowledge/kb.py`, all backward compatible):
- `candidates(findings, k=10, negatives=None, sex=None, age=None) -> list[dict]` — weighted BM25 over symptom/risk
  terms with term backoff and synonym-span dedupe, source prior, negative-finding penalty, optional sex (KCD
  restriction) and age filters, plus test-result scoring: each detected positive result adds
  `TEST_W(20) × {3: 1.0, 2: 0.5, 1: 0.15}` spread over linked diseases; a normal result subtracts from "R" (rule-out)
  links. Rows: `id, name_ko, kcd, matched[{id, finding}], sources, …` (`TF:` ids mark test-result matches).
- `discriminators(dx_a, dx_b, n=6)` → `symptoms_*`, `risk_factors_*`, `tests_*`, and `test_findings_a_only/b_only/shared`
  (weight ≥ 2 results are also put first in `tests_*_only`).
- `normalize_diagnosis(text)` → `{name, code, sex, match}`; `match` ∈ `kcd_exact, profile, superstring, kcd_backoff,
  profile_backoff, contained, fuzzy`.
- `lookup(name)` (profile incl. `findings_from_tests`), `render_for_prompt(...)`, `patient_profile(text) -> (sex, age)`,
  `available()`, `get_kb().match_terms(finding)` (includes test-result terms).
- `knowledge/kb_tests.py`: 262 result concepts, 452 links (186 decisive / 127 strong / 139 nonspecific; 41 rule-out)
  to 229 profiles, 96 reference keys (95 PMID-verified + `textbook`). `detect(text)` parses numbers vs. reference
  ranges (incl. reported ranges, fold-of-ULN, titres, `<`), qualitative +/−, imaging/ECG/pathology keywords with
  negation, and ignores history, rule-out intent and pending results. Build stores
  `findings_from_tests = [[term_id, ["curated"], weight, ref_key, "R"|""]]` per profile and `meta.test_refs`.

KB hints (`agent/kb_hints.py`, per case `seen` set, each ≤ 400 chars):
- `candidate_hint`: when positive findings reach 3, then 6 (at most twice per case), up to 3 KB candidates not already
  in the DDx ledger. A candidate needs ≥ 2 matched findings, or a single test-result (`TF:`) match.
  `sex`/`age` are not passed to `candidates()` yet.
- `discriminator_hint`: when the top-2 live DDx are within p 0.15, once per pair: up to 4 differing features/tests
  (trusted sources first, already-known terms skipped) and one DDXPlus question wording if available.
- `normalize_hint`: standard Korean name + KCD code; warning when the code is sex-restricted and the patient's sex differs
  (used in the review view and in the result record).

## Runtime (competition robustness)

- **Entry point** `run.py`: case source by `--env local|official` or `DOCTOR_ENV`. Submission mode by default
  (`--dev` turns it off). Output is incremental: one JSON line per case in `<out>.jsonl` (flushed + fsynced) and the
  `{case_id: diagnosis}` map rewritten atomically after each case; `--resume` skips finished cases. A second per-case
  guard in `run.py` still sends a fallback DIAGNOSE if `run_case` ever raises.
- **Never crash, always answer**: the policy talks to a per-case `GuardedLLM`. A failed LLM call returns "" (treated as
  a parse failure); after `max_llm_failures` (3) consecutive failures the LLM is dropped for that case; env errors are
  tolerated up to `max_env_failures` (3). Every case ends with a DIAGNOSE. In submission mode no exception leaves
  `run_case`; dev mode (the eval harness default) raises on bugs and billing errors.
- **Time budget**: `AGENT_CASE_TIME_BUDGET_S` (0 = unlimited; set below the official limit once known). Past
  `AGENT_DEGRADE_AT_FRAC` (0.6): no review / safety pushback, first 2 hints + `LOW_TIME_HINT`, reasoning effort "low".
  When `AGENT_FINAL_RESERVE_S` (45 s, at most half the budget) is left: forced final diagnosis (skipped below
  `min_call_s` = 5 s → top DDx). Exploratory calls get a deadline that leaves the reserve untouched; every call also has
  a watchdog thread; the OpenAI SDK's own retries are off (ours are deadline-aware).
- **gpt-oss responses** (`llm/client.py`, `llm/harmony.py`): content preferred; harmony markers
  (`<|channel|>analysis/final<|message|>…`, `analysis…assistantfinal…`) stripped; reasoning read from
  `reasoning_content`/`reasoning`; if an action prompt's content has no JSON the reasoning is appended as
  `<analysis>…</analysis>` so the parser can fall back to JSON written there. `finish_reason=length` with no answer →
  one retry with ×2 max_tokens (cap `max_tokens_cap` 8192) and effort "low".
- **Structured output** (optional): `DOCTOR_LLM_STRUCTURED_OUTPUT=json_schema|guided_json` sends `parser.ACTION_SCHEMA`
  for step/final prompts only. A 400/422 disables it for the rest of the run.
- **Prompt length**: `AGENT_MAX_VIEW_CHARS` (default 12000) caps `CaseState.view()` (ledgers + older actions compact +
  last 4 exchanges verbatim). Cut order: oldest compact actions → long verbatim responses → fewer verbatim exchanges →
  oldest part of ledger lines → hard cut keeping the initial info. Prompt sizes and runtime events are recorded in
  `result["runtime"]` (`prompt_chars_max/total`, `forced`, `degraded`, `llm_errors`, …).

## Evaluation harness (not packaged)

`eval/simulator.py` (keyword patient; also used by `env/local.py`), `eval/llm_patient.py` (LLM patient with hidden
answer fields and personas standard / vague / anxious / minimizer / poor_historian / mixed), `eval/scorer.py`
(accuracy via judge or string match, efficiency = 1 − turns/60, safety = share of applicable protocol checks done),
`eval/judge.py`, `eval/run_local.py` (multi-set, parallel, usage metering), `eval/experiment.py` (profiles, cost
guard), `eval/compare.py`, `eval/viewer.py` (viewer + share page), `eval/play.py` (interactive). See
`docs/experiments.md` for the API-day procedure.

## Design principles
- **Budget**: at most 60 turns; Efficiency counts, so a soft target of 20 turns and diagnosis at sufficient confidence.
- **Safety**: rule out can't-miss diagnoses with questions and tests before diagnosing (pushback + review).
- **Robustness**: parse failures retry, then fall back; the time limit is never exceeded; every case gets an answer.
- **Independence**: all state is created per case; no cross-case cache (the KB is static data).
- **Small-model friendly**: the code keeps the memory (ledgers) and decides verdicts; the LLM fills fixed fields.

## Open interfaces (confirm after the participant guide is published)
- Action/response format, diagnosis format (free text vs. code such as ICD/KCD), per-case time limit, LLM endpoint.
- `env/interface.py` is an assumption; add only an adapter in `env/official.py` (and adapt `run.py` output if needed).
