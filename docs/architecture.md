# Architecture

Current as of 2026-09-28 (prompt `v9-subagents`). When an interface changes, update this file first.

```
run.py ──> case source (env/factory.py: local | official)          ← official.py is a TODO until the guide is out
             │  env.reset() / env.step(action)
             ▼
        run_case (agent/loop.py) ── CaseBudget + GuardedLLM (agent/runtime.py) ── LLM client (llm/)
             │                                                                    OpenAI-compatible, gpt-oss aware
             ▼
        Policy.next_action (agent/policy.py)
             │ hints ◄── safety/protocols.py        can't-miss dx + pending minimum checks (26 categories)
             │       ◄── knowledge/clinical_rules.py ≤2 applicable decision rules (24 rules)
             │       ◄── agent/kb_hints.py ◄── knowledge/kb.py (+ kb_curated.py, kb_tests.py, data/kb/)
             │ advisors ◄── safety/triage.py (alert on top), agent/anchoring.py (turn-1 DDx, anchoring check),
             │          ◄── agent/question_planner.py (next-action suggestions), agent/confidence.py (DIAGNOSE pushback)
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
| `agent/prompts.py` | All prompts (`SYSTEM`, `REVIEW_SYSTEM`, final prompt, `LOW_TIME_HINT`, advisor wording `TRIAGE_ALERT` / `CONFIDENCE_PUSHBACK`). Version changes → `docs/experiments.md`. |
| `agent/confidence.py`, `agent/anchoring.py`, `agent/question_planner.py`, `safety/triage.py` | Advisors (code only): confidence score + stop rule, starting DDx + anchoring check, information-gain next-action planner, unstable-patient triage. Wiring: "Advisors wired into the policy". |
| `agent/subagents/` | Specialist sub-agents (same fixed LLM, other role): `base.py` contract, `runner.py` one never-raising call, `orchestrator.py` triggers/caps/logs; content modules `consult.py`, `advocate.py` (+ `knowledge/specialty.py`). See "Specialist sub-agents". |
| `agent/kb_hints.py` | KB → short hints (candidates, discriminators, diagnosis normalisation). Fail-safe: any KB error = no hint. |
| `agent/runtime.py` | `CaseBudget` (wall clock), `GuardedLLM` (failure cap, watchdog, deadlines, gpt-oss options, prompt-size stats). |
| `llm/client.py`, `llm/harmony.py` | `OpenAICompatClient` (retries, 429 wait, billing detection, length retry, structured output), `DummyLLM`; harmony-format cleanup. |
| `safety/protocols.py` | Chief-complaint safety protocols (can't-miss diagnoses + minimum checks with citations). `safety/rules.py` is a backward-compatible wrapper (`red_flags_for`). |
| `knowledge/clinical_rules.py` | 24 decision rules, category detection from the chief complaint, duration/age parsing, negation handling. |
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

0. **Read new results** (`AGENT_USE_RESULT_INTERPRETER`) — every EXAM/TEST response not read yet goes through
   `result_interpreter.interpret` (see "Result interpreter wired into the policy"): reading on the `Turn`, code
   findings into the findings ledger, critical items kept for the alert and the gate; a `needs_llm` result may get
   the LLM radiology sub-agent reading. After the advisors, the consult / anchoring-moment advocate triggers run
   (`Policy._subagent_hints`); the pre-review advocate runs just before step 6. See "Specialist sub-agents".

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
   Then the **can't-miss gate** (see "Safety layers" below).
5b. **Confidence pushback** (`AGENT_USE_CONFIDENCE`) — DIAGNOSE that passed the gate, more than 3 turns left, not
   degraded, at least 2 attempts left, not pushed back yet: `confidence.assess(state, dx, cfg)`; score <
   `confidence_pushback_below` (0.3) and recommendation ≠ `must_continue` → one hint (`prompts.confidence_pushback`, with
   `reasons_ko`) asking for the most discriminating remaining step. `must_continue` is left to the gate (no double block).
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
7. Advisor hints (`Policy._advisors`, see "Advisors wired into the policy"): triage (concerning), code reading of the
   latest result (`prompts.RESULT_HINT`), anchoring check, turn-1 starting DDx, question planner — together ≤
   `max_advisor_chars`. The triage alert for an unstable patient and new critical results are placed above the case
   view instead (`build_step_messages(alert=...)`, triage text first, then `prompts.RESULT_CRITICAL_ALERT`).
8. Sub-agent hints (`Policy._subagent_hints`, see "Specialist sub-agents"): radiology > consult > advocate, ≤
   `max_subagent_chars` (600, separate budget), only on the step right after the call; none in low-time mode.

In low-time mode only the first 2 hints are kept, plus `prompts.LOW_TIME_HINT` (and the alert, if any). All hints are labelled as reference,
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
- `knowledge/clinical_rules.py`: **24 rules** — Wells PE, PERC, HEART, ADD-RS, qSOFA, CURB-65, Centor, Ottawa SAH,
  Canadian CT Head, ABCD2, Alvarado, BISAP; added 2026-09-27: PECARN head (<2 y, ≥2 y), NEXUS, Canadian C-spine,
  SF Syncope, Canadian Syncope Risk Score, Glasgow-Blatchford, Kocher, PAS, sPESI, PECARN febrile infant,
  McIsaac — each with citation/DOI. `Rule.applies_to` = (category or rule keyword)
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

Next-question planner (`agent/question_planner.py`, wired 2026-09-28 as a per-turn hint; CPU, ≈3 ms/call warm, p95 6 ms):
- `suggest(state, k=3, include_safety=True) -> list[Suggestion]`; `Suggestion(type "ASK"|"EXAM"|"TEST", content_ko,
  targets[dx], expected_value, cost_tier ask|exam|lab|imaging|invasive, source, citation, safety, features, note)`.
- Hypotheses = top 4 live DDx-ledger entries (else `state.ddx`, else KB candidates) resolved to KB profiles + an "other"
  hypothesis (p 0.2); can't-miss ("위험") entries ×1.5. P(feature|dx) from profile symptoms/risks (Orphanet frequency
  class or source consensus) and curated test results (link weight 3/2/1 → 0.9/0.65/0.35), with a small leak.
- Value = expected information gain over the action's joint outcomes × tier weight (1/.95/.9/.75/.5). Several results
  of one test (e.g. ECG: STEMI / pericarditis pattern) form one action (`TEST_RULES`: result → request wording).
- Excluded: already done (`state.asked`, earlier TEST/EXAM naming the test), features already known from the case text
  or findings ledger (nlp concepts → KB terms, `kb_tests.detect`), questions about a hypothesis itself, and TEST/EXAM
  that `preconditions.check` blocks (warn → `note`). Pending protocol checks are appended with `source="protocol:<id>"`,
  `safety=True`; a ranked action that completes one is also marked.
- `render_for_prompt(suggestions) -> str`: "추천 다음 행동 (참고): 1) [검사] 심전도 (감별: …) …" ≤ 300 chars, protocol
  items left out (the protocol hint already shows them).
- Offline check (cases_aug, 267 cases, no LLM, DDx seeded from KB candidates ± gold dx): see the planner commit message.

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

## Safety layers wired into the policy (2026-09-27)
Order inside `Policy.next_action` for each proposed action:
1. parse → normalize type → merge ledgers → **grounding** (`agent/grounding.apply`: findings / DDx evidence never said by the environment are marked unverified; unverified findings are excluded from renaming/criteria evidence) → dedupe.
2. DIAGNOSE only: protocol safety pushback (once) → **can't-miss gate** (`safety/danger_gate.gate`: forces the next rule-out action for an unresolved can't-miss diagnosis, ≤ `max_gate_turns`=3 per case, never when ≤2 turns remain; a confirmed *other* danger only yields a one-time hint) → pre-diagnosis review (with diagnostic criteria).
3. TEST/EXAM (incl. gate/review follow-ups): **pre-test preconditions** (`safety/preconditions.check`: block → swap in the prerequisite, e.g. brain CT before LP, β-hCG before abdominal CT; block without alternative → ask the model for another action; warn → annotate the reason).
All three are guarded (exceptions are logged, never raised), recorded in `result["safety_log"]` (shown per turn in the viewer), and switchable for ablations: `AGENT_USE_GROUNDING`, `AGENT_USE_DANGER_GATE`, `AGENT_USE_PRECONDITIONS` (experiment condition `v6-no-safety`).

## Advisors wired into the policy (2026-09-28, prompt `v7-advisors`)
Four code-only helpers (CPU, no LLM call of their own). They add prompt text or one pushback; they never pick the
action. Each is guarded (an exception is logged as `{"layer", "error"}` in `safety_log` and the case goes on as if the
advisor were off), switchable, and logged to `result["safety_log"]` with a Korean `msg` shown per turn in the viewer.

| Advisor | Switch (env, default on) | When | Effect | Log (`layer`) |
|---|---|---|---|---|
| Triage (`safety/triage.py`) | `AGENT_USE_TRIAGE` | every turn | `unstable`, or `unknown` (vitals missing) with a red flag (`TRIAGE_RED_FLAGS`: ams, chest_pain, syncope, bleeding, anaphylaxis, airway, respiratory, seizure, sepsis_suspected, trauma) → `render_for_prompt` (≤ 250 chars) as an alert **above the case view** (`prompts.TRIAGE_ALERT`), kept in low-time mode; `concerning` → ordinary hint; `stable` / `unknown` without red flags → nothing | `triage`, on level change only |
| Starting DDx (`anchoring.initial_differential`) | `AGENT_USE_ANCHORING` | turn 1 only | `render_for_prompt` (≤ 300 chars) as a hint | `anchoring` / `initial_ddx` |
| Anchoring check (`anchoring.anchoring_check`) | `AGENT_USE_ANCHORING` | from turn 3, each turn until it fires | its `prompt_ko` as a hint in the next prompt, **once per case** (`state.anchoring_shown`, set only when the hint really made it into the prompt; a raising check is not retried) | `anchoring` / `premature_closure` |
| Question planner (`question_planner.suggest(k=planner_k=3)`) | `AGENT_USE_PLANNER` **and** `AGENT_USE_KB` | every turn | `render_for_prompt` (≤ 300 chars, protocol items left out) as a hint | `planner`, when the suggestions change |
| Confidence (`confidence.assess`) | `AGENT_USE_CONFIDENCE` | DIAGNOSE proposal after the protocol pushback and the can't-miss gate | score < `AGENT_CONFIDENCE_PUSHBACK_BELOW` (0.3) and not `must_continue` → one retry hint (`prompts.confidence_pushback`) per case (`state.confidence_pushback`); only with > 3 turns left, not in low-time mode, and ≥ 2 attempts of `MAX_ATTEMPTS` left | `confidence`, every assessment (score, components, recommendation, pushback) |

Prompt budget: the advisor hints of one step prompt (plus the triage alert) are ≤ `AGENT_MAX_ADVISOR_CHARS` (900),
filled in priority order triage > result reading (since `v8-result-interp`) > anchoring check > starting DDx > planner; a hint that does not fit is dropped whole
(and not logged / not counted as shown). In low-time mode only the triage alert is kept. Ablation: experiment condition
`v6-no-advisors` (all four switches off).

## Confidence and stop rule (`agent/confidence.py`, 2026-09-28; wired as a one-time pushback, see "Advisors")
Code-computed replacement for the LLM's self-reported confidence. Pure code over one `CaseState` (no LLM, CPU, stdlib,
never raises, nothing kept between calls or cases).
- API: `assess(state, proposed_dx=None, cfg=AgentConfig, params=None) -> Assessment(score, components, recommendation,
  reasons_ko, proposed, dangers)`; `recommendation` ∈ `diagnose` | `continue` | `must_continue`. Lower-level:
  `features(state, dx)`, `score_of(components, params)`, `decide(score, turn_count, max_turns, target_turns, dangers,
  params, gate_left)`.
- Score = `sigmoid(bias + Σ w·x)` over 8 features in [0, 1]: `margin` (ledger p of the proposal − best other live DDx),
  `verified_support` (grounded "for" items /3), `contradictions` (grounded "against" /2), `confirmatory_test` (support
  grounded in a TEST/EXAM response, a confirmed can't-miss dx, or a criteria decision), `criteria_met`
  (`diagnostic_criteria`), `dangers_unresolved` (actionable unresolved can't-miss dangers /2), `kb_agreement`
  (1/rank in `kb.candidates`), `turns_used` (turns / target_turns). Missing features use neutral values.
- Rule: ≤1 turn left → diagnose; actionable unresolved danger with >2 turns left and gate budget left → must_continue;
  score ≥ θ_high (0.85) → diagnose; < θ_low (0.65) → continue; in between → diagnose from `target_turns` on.
- Parameters: defaults in code (`DEFAULT_PARAMS`); `AGENT_CONFIDENCE_PARAMS=<json>` overrides (dev). `calibrate(paths)`
  (`scripts/calibrate_confidence.py [--write] eval/results/run_*.json`) rebuilds states at every decision
  point of saved runs (`state_from_result`), fits a class-balanced logistic regression (Newton, L2 towards the hand-set
  prior, sign-constrained, `turns_used` fixed), picks θ_high on a coarse grid at matched replay accuracy, and writes
  `data/labels/confidence_params.json` (dev only, not shipped).
- 2026-09-28 calibration (11 runs, 226 cases, 1446 decision points, Gemini/Gemma doctors): final-diagnosis AUC
  (correct vs. not) 0.745 hand-set → 0.796 leave-one-run-out; `margin` alone 0.747. `dangers_unresolved` and
  `kb_agreement` fitted to weight 0. Replay (leave-one-run-out, stop at the first "diagnose"): accuracy 0.885 → 0.876,
  mean turns 6.40 → 6.08. Earlier labels are a name-matching proxy (`same_disease`). **Re-validate on gpt-oss-20b.**

## Broad starting DDx and anchoring check (`agent/anchoring.py`, 2026-09-28; wired, see "Advisors")
Two pure functions for the policy (the lead wires them in; no state kept between calls or cases):
- `initial_differential(initial_info) -> list[{"dx", "tag": 위험|흔함|KB, "source", "category"?, "kcd"?}]` (≤ 8, deduplicated
  with `same_dx`): up to 3 can't-miss diagnoses (`safety/protocols` via `detect_categories`), 4 common causes from the
  `COMMON` table (per chief-complaint category + 12 extra complaint categories such as dizziness, diarrhea, vision loss;
  each category cites an AAFP review or the protocol guideline, PubMed-checked 2026-09-28; pediatric rows and 3 categories
  are marked "미검증"), 1 KB candidate (`kb.candidates` on the chief complaint), then fill. Sex/age filters
  (`kb.patient_profile` + `clinical_rules.rule_age_years` on the demographics). `render_for_prompt(ddx)` ≤ 300 chars,
  meant to be shown once at turn 1.
- `anchoring_check(state) -> {"dx", "p", "reasons", "why_ko", "prompt_ko", "suggested_actions"} | None`: only after
  `MIN_TURNS`=3 turns, top live DDx p ≥ 0.4 and the same top in the last snapshot. Reasons: `stable_untested` (top since
  turn ≤ 2 and no EXAM/TEST named it, its KB tests/decisive results or its discriminating results vs. the 2nd candidate),
  `weak_support` (p ≥ 0.6 with ≤ 1 distinct supporting item found in the initial info + responses and not marked
  unverified), `contradicted` (≥ 2 distinct "against" items that are grounded findings, not "결과 없음/확인 필요").
  `prompt_ko` is a devil's-advocate request (two alternatives that explain the findings + the result that would refute
  the current top, then pick that action). The caller must show it at most once per case.

## Unstable-patient triage (`safety/triage.py`, 2026-09-28; wired, see "Advisors")
- `assess(state) -> dict`: `level` = `unstable` (any critical signal) / `concerning` (any warning) / `stable` (no
  warning and core vitals known or requested) / **`unknown`** (no warning yet but vitals missing — missing vitals are
  never "stable"; treat as "measure vitals first"). Also `signals` [{key, ko, severity critical|warning|unknown,
  where, cite}], `why_ko`, `citation`, `news2`, `shock_index`, `map`, `qsofa`, `gcs`, `vitals`, `missing`,
  `unavailable`, `pediatric`, `flags` (ams, chest_pain, sepsis_suspected, bleeding, anaphylaxis, airway, ...).
- Reads the initial info + every environment response (not the doctor's questions); vitals are the nlp layer's
  measured values with the patient's age (Fleming 2011 centiles for children), worst value per vital.
- Thresholds: NEWS2 bands/triggers (adults), qSOFA, shock index (≥1.0 warning, ≥1.4 with SBP ≤100 critical), SIPA and
  PALS hypotension (children), MAP <65, SpO2 <90, RR ≥30/≤8, GCS ≤8/AVPU P-U, NIAID/FAAN anaphylaxis, airway
  swelling/stridor, bleeding + instability, chest pain + instability, reproductive-age abdominal pain + SI ≥1.
  Sources and verification levels are in the module docstring.
- `priority_actions(state, level_or_assessment)` → [(ActionType, content_ko, reason_ko)] in ABCDE order, skipping
  actions already done or results already in the text: airway exam, vitals / SpO2, breathing exam, ABGA, ECG,
  lactate + blood cultures, CBC, β-hCG, bedside ultrasound, glucose, GCS/neuro exam, skin exam. `stable` → [].
- `render_for_prompt(state, assessment=None, actions=None, max_chars=250)` → one Korean line, "" when stable.
- Physiologically stable emergencies (STEMI, dissection, SAH with normal vitals) stay `stable` here by design; the
  can't-miss gate (`danger_gate.py`) and protocols cover them.

## Result interpreter (`agent/result_interpreter.py`, 2026-09-28; wired, see "Result interpreter wired into the policy")
A code-first reader for one EXAM/TEST result text, separate from the diagnosing agent (it never diagnoses). Pure code:
stdlib, CPU, deterministic, never raises, no LLM or network call, nothing kept between calls or cases (~0.6 ms per
text on the 1,420 exam/test texts in `data/cases_*`).
- API: `interpret(test_name, result_text, patient_ctx=None) -> Interpretation`; `patient_ctx` = `{"age_years",
  "sex", "initial_info"}` of the same case (age switches the nlp layer's children's HR/RR ranges).
  `Interpretation`: `kind` (lab / imaging / ecg / exam / other), `items`, `normal` (whole-normal statement and nothing
  abnormal), `unavailable` ("결과가 제공되지 않습니다": **not normal**), `pending` ("대기 중": not a result), `ignored`
  (dropped sections / recommendation parts), `llm_reasons`, helpers `present()/absent()/uncertain()/critical()/
  concepts()/as_dict()`.
  `Item`: `kind`, `concept` (lexicon id; `""` = abnormal wording the code could not map), `label`, `polarity`
  (present / absent / uncertain), `span`, `site`, `laterality`, `value`/`unit`/`direction` (labs, vitals), `critical`,
  `comparison` (new / improved / worsened / stable / resolved), `hedge`, `supports` ((dx key, Korean name, weight) from
  `kb_tests` links, **supportive only**, present IMG/ECG items only), `source`, `confidence`, `summary_ko`.
- `render_for_prompt(interp, max_chars=300)`: one Korean line — `[검사] 정상 / 있음: ⚠critical first … / 의심: … /
  없음: … / 지지 가능 질환(확진 아님): …`. `needs_llm(interp)` / `llm_reasons(interp)`: long text (>500 chars),
  >6 sentences, serial time points, ≥2 unmapped abnormal phrases (or unmapped only), nothing read from an imaging/ECG
  report, conflicting polarity, ≥3 hedged items.
- Reading: sections (Indication / History / Technique / Comparison / Recommendation / 권고 / 임상 정보 dropped;
  short header sections end at their first sentence), recommendation parts cut, comparison phrases that look like
  negations masked ("no interval change in", "이전과 비교하여 변화 없음"); `nlp.findings.parse` (lexicon, vitals, labs
  with reference ranges, kb_tests); kb_tests imaging/ECG readings re-anchored on their own span; organ-dependent report
  words mapped with the organ next to them or the test name (`_DESCRIPTORS`: "출혈" on a brain CT = `IMG:ct_ich`,
  "혈전" in a leg vein = `IMG:doppler_dvt`, "비후" of the gallbladder = `IMG:us_cholecystitis`, ...); impression words
  ("급성 충수염 의심") mapped to the imaging concept; HX concepts re-mapped in imaging context (`HX:prior_vte` →
  `IMG:ctpa_pe` / `IMG:doppler_dvt`); SYM/QUAL and non-imaging SIGN concepts dropped from imaging reports ("반점상 경화"
  is not a rash). Polarity: the nlp cue rules (`assess_spans`), then per comma part hedges → uncertain, "배제할 수 없음 /
  cannot be excluded" → uncertain, "배제됨 / was excluded" → absent, English list negation ("No A, B, or C"), "A without
  B" keeps A. Lab values next to a printed range are re-checked against that range (kb_tests missed "D-dimer 750 ng/mL
  (<500)"). Pending parts are skipped.
- Critical: urgent imaging/ECG concepts (pneumothorax, free air, dissection, ICH/SAH/SDH, mass effect, PE, DVT,
  tamponade, torsion, empty uterus, STEMI, long QT, ...; our selection after the ACR communication parameter) read as
  present/uncertain, or values beyond adult critical limits (K ≥6.0/<2.8, Na <120/>160, glucose <50/>450, Hb <7,
  platelets <20k, WBC <2k/>30k, INR ≥5, HCO3 <10 — Kost 1990; SBP <90, SpO2 <90, RR ≥30/≤8, adult HR ≥130/<40).
- Lexicon: `data/lexicon/seed.tsv` gained report wording (Korean + English) for existing kb_tests IMG/ECG concepts and
  15 new concepts (`IMG:normal_study`, `ECG:normal_ecg`, `IMG:lung_nodule`, `IMG:atelectasis`, `IMG:subdural_hematoma`,
  `IMG:mass_effect`, `IMG:abscess`, `IMG:mass`, `IMG:free_fluid`, `IMG:wall_motion_abnormality`,
  `IMG:valve_regurgitation`, `IMG:lvh`, `IMG:hyperinflation`, `IMG:bone_lesion`, `IMG:intrauterine_pregnancy`).
  Disease names ("대동맥 박리", "기흉") are deliberately **not** lexicon forms (the grounding checker would treat a
  diagnosis in the doctor's reason as a finding claim); the interpreter maps them itself in imaging context.
- Offline evaluation: `eval/offline/eval_result_interp.py` on `data/labels/result_interp_gold_v1.jsonl` (66 snippets,
  116 labels; dev set) and `result_interp_gold_fresh_v1.jsonl` (32 snippets, 66 labels; written after the rules):
  fresh set before the fixes it prompted P 0.879 / R 0.879; both sets 1.00 after (same author for rules and labels →
  optimistic). Metrics in `data/labels/result_interp_gold_v1_metrics.json`. Tests: `tests/test_result_interpreter.py`.
- Future LLM hook: `prompts.RESULT_INTERPRETER_PROMPT` + `build_result_interpreter_messages(test_name, text,
  code_reading)` (JSON items/normal/unavailable/summary). Not called anywhere.

## Result interpreter wired into the policy (2026-09-28, prompt `v8-result-interp`)
Switch `AGENT_USE_RESULT_INTERPRETER` (default on; `AgentConfig.use_result_interpreter`). Ablation condition
`v6-no-interp` in `eval/experiment_profiles.json` (**not** in the `dev` profile; name it explicitly).
1. **Read** (`Policy._interpret_results`, first thing in `next_action`): each EXAM/TEST turn not read yet (each turn
   exactly once, also when it fails; "(환경 응답 오류)" skipped) → `interpret(action.content, response,
   {"initial_info": ...})`. `Turn.interp` = `interp.as_dict()` + `prompt` (the rendered line), `needs_llm`, `pending`;
   the loop result carries it per turn (`result["turns"][i]["interp"]`).
2. **Hint**: the reading of the **latest** turn (only on the prompt right after the result) as
   `prompts.RESULT_HINT` (≤ `result_hint_chars` = 300 incl. prefix), inside the shared advisor budget
   (`max_advisor_chars`), priority: triage hint > result reading > anchoring > starting DDx > planner. Dropped whole
   when it does not fit; not shown in low-time mode (hints are cut there anyway).
3. **Findings ledger** (`policy._interp_findings` → `FindingsLedger.add`): `Finding.source = "result_interpreter"`,
   `verified=True` (grounding keeps them verified: read from the text by construction). present → 양성; uncertain → 양성
   with detail "의심"; absent → 음성 (a normal measurement once as its span, e.g. "혈압 120/80" 음성 "정상"); whole-normal
   study → the test 음성 "특이 소견 없음"; **"not provided" → the test 결과없음, pending-only → the test 결과없음 "결과 대기 중"
   (never 음성)**; concept "" stays prompt-only; abnormal first, ≤ 6 items per result. Merge semantics: later
   information wins as before, except a model-reported finding does not replace a code reading of the same (or a
   later) turn. KB hints and the question planner read the ledger, so the result concepts reach `kb.candidates`
   through their Korean labels (plan item 5). TODO: hand `interp.concepts("present")` (lexicon ids) to
   `kb.candidates` / `question_planner` directly instead of via labels (needs a concept-id entry point in the KB).
4. **Critical results**: items with `critical` (present or uncertain) go to `state.result_criticals`. Each (concept,
   side, polarity) is shown **once per case** in the top-of-prompt alert slot (`TRIAGE_ALERT` wrapper; triage text
   first, then `prompts.RESULT_CRITICAL_ALERT`), also in low-time mode; its length counts against the advisor budget
   but it is never dropped. **Danger gate**: `danger_gate.CRITICAL_RESULT_DANGER` maps critical IMG/ECG concepts that
   are the disease itself (pneumothorax, dissection, AAA, STEMI, SAH, PE, free air, ICH/SDH, ischemic stroke, testicular
   no-flow) to rule-out entries; `_Ctx.critical` (from `state.result_criticals`, **present only**) makes `_status` return
   `confirmed` with the reading as evidence, and `_dangers` adds such a danger to the checked list even when neither
   the chief complaint nor the DDx ledger raised it (source `critical_result`) → a different proposed diagnosis gets
   the one-time `confirmed_other` hint; a proposed diagnosis that matches is allowed as a confirmed danger.
   `confidence.assess` sees it through `danger_gate`. Labs (K, glucose, ...) are alerted but not mapped to the gate.
5. **LLM reading** (plan item 6): since `v9-subagents` the "radiology" sub-agent (see "Specialist sub-agents") reads
   a `needs_llm` result with `RESULT_INTERPRETER_PROMPT`, at most `max_llm_radiology` (1) per case. `needs_llm`
   results are still counted: `result["result_interp"]` = `{n, needs_llm, llm_reasons: {reason: count}, critical,
   unavailable, errors}` per case; the reading's `safety_log` entry has `llm_called` and its message says
   "LLM 판독 필요(…; 호출함 | 호출 안 함)".
6. **Logs** (`safety_log`, viewer label "결과 판독"): `{"layer": "result_interp", "kind": "reading", "turn": <result
   turn>, test, test_kind, critical, needs_llm, llm_reasons, msg}` per result; `{"kind": "critical_alert", items, msg}`
   when an alert is shown; `{"layer": "result_interp", "error"}` on any exception (the case goes on as if off).

## Specialist sub-agents (2026-09-28, prompt `v9-subagents`)
Runtime "specialists" under the main doctor LLM: the **same fixed gpt-oss-20b** through the same `GuardedLLM`, with a
different role prompt and a different evidence slice, called **only when triggered**. They never pick the action; their
output is a short hint for the next main prompt, "참고" DDx candidates and log entries. The "≥ 1 LLM call per case with
case information" rule stays satisfied by the main loop; sub-agent calls are extra.

### Interface contract (shared with the content branches; do not deviate)
```python
# src/doctor_agent/agent/subagents/base.py
@dataclass
class SubagentCall:
    name: str                 # e.g. "consult:cardio", "advocate", "radiology"
    messages: list[dict]      # chat messages for the fixed LLM
    json_schema: dict | None  # optional structured output
    max_chars_out: int = 600  # cap of the rendered hint

@dataclass
class SubagentResult:
    name: str
    ok: bool
    hint_ko: str              # short Korean text injected into the NEXT main prompt (<= max_chars_out)
    ddx_add: list[dict]       # [{"name": str, "why": str}] added to the DDx ledger as "참고" (never auto-confirmed)
    suggested_actions: list[dict]  # [{"type": "ASK|EXAM|TEST", "content": str, "why": str}]
    red_flags: list[str]
    raw: dict                 # parsed JSON for logging
```
Content modules (owned by the content branches):
- `agent/subagents/consult.py`: `SPECIALTIES: dict[str, SpecialtySpec]`, `build_consult(state, specialty, resources) ->
  SubagentCall`, `parse_consult(text) -> SubagentResult`.
- `agent/subagents/advocate.py`: `build_advocate(state, proposed_dx: str | None, reason: str) -> SubagentCall`,
  `parse_advocate(text) -> SubagentResult`.
- `knowledge/specialty.py`: `route(state) -> (specialty id | None, share of top-DDx mass, reasons)` with ids cardio,
  resp_id, gi_liver, neuro, rheum_immune, peds_obgyn; `resources(specialty, state) -> dict` (criteria/rule names + short
  texts, KB slice).

Framework (pipeline branch):
- `agent/subagents/runner.py`: `run(llm, call, deadline=None, *, parse=None, clock=time.monotonic) -> SubagentResult`.
  One call of the fixed LLM with `json_schema`, `expect_json=True`, `reasoning_effort=AgentConfig.
  subagent_reasoning_effort` ("low") and the deadline (GuardedLLM also applies the case budget deadline; the earlier
  one wins). `parse` defaults to a generic JSON reader (`parser._json_objects` on the harmony-cleaned text: keys
  `hint_ko`/`hint`/`summary`, `ddx_add`, `suggested_actions`, `red_flags`); every result is sanitised (types, lengths,
  `hint_ko` ≤ `max_chars_out`). **Never raises**: LLM errors, time-outs, budget exhaustion, empty/bad JSON or an
  exception in the parser → `ok=False` (the main loop hits the same guard on its next call if the LLM is really gone).
  Counts `GuardedLLM.subagent_calls` (in `result["runtime"]`).
- `agent/subagents/orchestrator.py`: `SubagentManager` (one per `Policy`, i.e. per case): triggers, caps, skip
  reasons, logging, hint queue, summary. Content modules are imported lazily; a missing module or an exception inside
  one (route / resources / build / parse) is logged once and disables that sub-agent for the case.
- `DdxLedger.refs` (`agent/ledger.py`): "참고" candidates from sub-agents, rendered as one line of the DDx ledger view
  ("참고(자문, 미확인): …"). They are **not** in `entries`, so confidence, anchoring, the gate, the planner and the
  forced-diagnosis fallback never see them; a ref is dropped from `refs` when the main model itself puts the same
  diagnosis in its `ddx` (the model's own decision, not an automatic promotion). ≤ 4 refs.

### Triggers (each at most once per case; all switches default on)
| Sub-agent | Switch | Trigger | Output goes to |
|---|---|---|---|
| Consult (`consult:<specialty>`) | `AGENT_USE_CONSULT` | start of `next_action`: (a) `turn_count ≥ 3` and `specialty.route(state)` returns a specialty with share ≥ 0.6, or (b) `turn_count ≥ 6` and the model's own `confidence` < 0.5 on each of the last 3 parsed turns (then the routed specialty is used whatever its share; none → skip `no_specialty`) | hint on the main prompt of this `next_action` (the prompt right after the call), `ddx_add` → refs |
| Advocate (`advocate`) | `AGENT_USE_ADVOCATE` | (a) the turn the anchoring (premature-closure) hint is really shown (`proposed_dx` = the anchored top DDx, reason = the check's `why_ko`) → hint on the same prompt, next to the anchoring hint; else (b) a DIAGNOSE proposal that is about to go to the pre-diagnosis review (same guards: < 2 reviews, > 3 turns left, not degraded) with code confidence (`confidence.assess`) < 0.65 → its hint is appended to the **review view** as `[반대 의견 검토(자문)]` (the reviewer decides hold/approve; no extra turn is spent) | as left; `ddx_add` → refs |
| LLM radiology (`radiology`) | `AGENT_USE_LLM_RADIOLOGY` (+ `AGENT_USE_RESULT_INTERPRETER`) | a result read by the code interpreter with `result_interpreter.needs_llm(interp)` true; ≤ `max_llm_radiology` (1) per case | `prompts.build_result_interpreter_messages` (`RESULT_INTERPRETER_PROMPT`, code reading as draft, result text ≤ 3000 chars). Items → findings ledger with `source="llm_radiology"` (있음 → 양성, 의심 → 양성 "의심", 없음 → 음성; detail site/value/test/"LLM 판독"; ≤ 6 items). They go through the **grounding check like model findings** (not trusted by construction) and never replace a code reading of the same item (`FindingsLedger.add`). Critical present items → `state.result_criticals` with concept `llm:<finding>` (one-time alert; not mapped to the danger gate). Summary → hint `prompts.SUBAGENT_HINT` on the next main prompt |

Why the advocate has two moments: (a) is cheap (no hold, the hint only redirects exploration) and coincides with the
moment code already suspects anchoring; (b) is the last chance before a low-confidence diagnosis and only enriches the
existing review (it cannot add turns: the review is skipped with ≤ 3 turns left or in low-time mode). The cap is one
advocate call per case, whichever comes first.

### Global guards and budget
- Master switch `AGENT_USE_SUBAGENTS` (default on; ablation condition `v6-no-subagents`, **not** in `dev`).
- Skip all sub-agents when: low-time mode (`policy.degraded`), remaining turns ≤ 3, or the per-case cap
  `AGENT_MAX_SUBAGENT_CALLS` (3) of attempted calls is reached. A trigger that fires but is skipped is logged once per
  (sub-agent, reason).
- Hints: **separate budget** `AGENT_MAX_SUBAGENT_CHARS` (600) per step prompt, after the advisor hints (the triage /
  critical-result alert stays on top; the advisor budget of 900 is untouched, so no safety hint is ever displaced by a
  sub-agent). Order inside it: radiology > consult > advocate; a hint longer than what is left is cut (not dropped: it
  cost an LLM call). A hint is shown on one `next_action` (all its attempts) and then cleared. Not shown in low-time
  mode.
- Reasoning effort `AGENT_SUBAGENT_REASONING_EFFORT` (default `low`; `none` keeps the parameter unsent).
- Expected extra calls: 0 in most cases; ≤ 3 per case (consult 1 + advocate 1 + radiology 1).

### Logs and result
- `safety_log` entries with `layer="subagent"`: `kind="call"` (`name`, `trigger`, `ok`, `elapsed_s`, `hint`, `ddx_add`,
  `red_flags`, `suggested_actions`, `error`, `msg`), `kind="skip"` (`name`, `reason`, `msg`), `kind="error"` (content
  module failure). Viewer label "전문 자문".
- `result["subagents"]` = `{"enabled", "calls": {name: n}, "ok", "fail", "skipped": {"name:reason": n},
  "refs": [...], "llm_radiology_findings": n}`.
- Tests: `tests/conftest.py` sets `AGENT_USE_SUBAGENTS=0` (older tests pin the scripted LLM call sequence);
  `tests/test_subagents.py` switches them on per test and installs fake content modules via `sys.modules`.
  `scripts/token_budget.py`'s scripted doctor answers sub-agent prompts with `{}` (not counted as steps).

## Specialist sub-agent content (`agent/subagents/consult.py`, `advocate.py`, 2026-09-28; content only, not wired)

Owned by clinical-strategist. Kept separate from the framework section (when to call, merging, budgets), which the
framework owner writes. Interface: `agent/subagents/base.py` (`SubagentCall`, `SubagentResult`). Same gpt-oss-20b,
one call each; nothing here calls an LLM.

**Consult** — `build_consult(state, specialty, resources=None) -> SubagentCall`, `parse_consult(text, specialty="",
known_ddx=None, max_chars=600) -> SubagentResult`.
- `SPECIALTIES` (`SpecialtySpec`): `cardio` 심장·혈관, `resp_id` 호흡기·감염, `gi_liver` 소화기·간, `neuro` 신경,
  `rheum_immune` 류마티스·면역, `peds_obgyn` 소아·산부인과. Each: Korean role, must-not-miss list, discriminating
  questions / exams / tests, pitfalls, per-branch notes, and references by id into existing modules (not copied):
  `rule_ids` (clinical_rules RULES), `criteria_ids` (diagnostic_criteria), `protocol_categories` (safety/protocols),
  `rule_out_names` (danger_gate RULE_OUT_TABLE). Tests check every id exists. New claims cite `SpecialtySpec.citations`
  or are marked reviewer knowledge in `note`.
- Patient branch (`patient_profile`): `peds` (triage age reader; renders the age's Fleming 2011 1st–99th centile HR/RR
  and the PALS hypotension floor), `pregnant` (protocols predicate `current_pregnancy`, or postpartum words;
  gestational weeks → before/after 20 weeks), `female_repro` (predicate `pregnancy_test_abdominal`: pregnancy test before
  radiation or drugs), `adult`. Age from the initial info; pregnancy from the initial info + usable responses.
- User message order: specialty block (+ branch notes) → patient branch → can't-miss of the chief-complaint protocols
  owned by this specialty → rules in their validated population (`rules_for`) owned by it → criteria matching a current
  DDx name (`criteria_for`) owned by it → optional `resources` (from knowledge/specialty.py; any dict of str/list/dict,
  ≤ 900 chars) → case context (initial info; findings ledger with `exclude_unverified=True`; last 2 exchanges; DDx
  ledger; actions done, newest first, "(결과 없음)" marked).
- Output JSON (`CONSULT_SCHEMA`): `assessment`, `ddx_add[{name, why}]`, `missed_dangers[]`, `next_actions[≤3 {type
  ASK|EXAM|TEST, content, why}]`, `confidence_note`. Mapping: `ddx_add` → `ddx_add`, `next_actions` →
  `suggested_actions`, `missed_dangers` → `red_flags`, whole object → `raw`.

**Advocate ("반대 의견 담당", diagnostic time-out; Croskerry cognitive debiasing)** — `build_advocate(state,
proposed_dx, reason="", resources=None)`, `parse_advocate(text, proposed_dx="", max_chars=600)`. JSON
(`ADVOCATE_SCHEMA`): `alternatives[2 {name, why}]`, `unexplained[]` (positive findings the proposed dx does not
explain), `refuting_test` (one ASK/EXAM/TEST or null), `dangers_not_excluded[]`, `verdict` 유지|재검토, `note`. Mapping:
alternatives → `ddx_add` (the proposed dx itself dropped), refuting_test → `suggested_actions` (≤ 1), dangers →
`red_flags`, `raw["verdict"]` normalised (keep/reconsider → 유지/재검토, "" if missing). The user message adds the
chief-complaint can't-miss list (`protocols.cant_miss_for`).

**Parsing (both)**: `extract_json` = harmony split (`llm/harmony.py`) + reasoning-tag strip + code-fence strip, last
top-level object having a schema key (`parser._json_objects`), then a truncated-JSON repair (close strings/brackets,
cut back to earlier commas). Actions: type normalised (lowercase, Korean 문진/진찰/검사), anything else (DIAGNOSE,
PLAN, empty content) dropped, near-duplicates dropped, ≤ 3. `ddx_add` deduplicated with `same_dx` and against
`known_ddx`. `hint_ko` restates only what the JSON said (header + non-empty lines), capped at `max_chars` (600 =
`SubagentCall.max_chars_out`). `ok=False` with an empty hint when no object is found or it has no usable content.
Never raises.

**Sizes** (tiktoken `o200k_harmony`, messages only): consult system 374 tokens; specialty block 348–577; full consult
832–1,090 tokens at turn 0 and ≤ 2,759 in the worst case (60 turns, 80 findings, 8 DDx, full resources; user message
≤ 6,500 chars by test); advocate 584 / 2,361.

**Added specialties (2026-09-28): `heme_onc` 혈액·종양, `renal_uro` 신장·비뇨** (same `SpecialtySpec` pattern; 8 ids
in `SPECIALTY_IDS`; `knowledge/specialty.py` routing for them is added separately).
- `heme_onc`: neutropenic fever, acute leukaemia, TTP/HUS, DIC, HIT, TLS, malignant cord compression, SVC syndrome,
  hypercalcaemia of malignancy, haemolysis, aplastic anaemia; smear / reticulocytes / haemolysis panel / ADAMTS13 /
  SPEP·FLC / flow cytometry·biopsy. Owns protocol categories `fatigue` and `neck_mass` (previously unowned; now every
  category has an owner, checked by test) plus `bleeding`, `jaundice`, `pruritus`, `fever`, `back_pain` (shared).
  New citations: Cairo-Bishop 2004 (TLS), IMWG 2014 (myeloma), ASH 2018 (HIT), ISTH 2020 (TTP).
- `renal_uro`: AKI (pre/intra/post-renal), hyperkalaemia, obstructed infection / urosepsis, RPGN / pulmonary-renal,
  rhabdomyolysis, retention, testicular torsion (shared with `peds_obgyn`, which keeps it), AAA mimicking colic.
  `kdigo_aki_2012` is now owned only here (removed from `gi_liver` and `rheum_immune`, where it was parked). Reuses
  C_AKI, G_GLOMERULAR, C_SCROTAL, G_SEPSIS; `qsofa` rule; protocols `edema`, `abdominal_pain`, `back_pain`.
- Sizes: specialty block 467–517 tokens (render ≤ 675 chars); full consult turn 0 950–1,104 tokens, worst case ≤ 2,929
  (same measurement for the six existing specialties now reads 2,769–2,915).

## Specialty routing (`knowledge/specialty.py`, 2026-09-28; not wired yet)

Routing and evidence slicing for runtime specialist consults (code only, stdlib, CPU, no network; the consult prompt
and the call site belong to agent-engineer / clinical-strategist). Six fixed ids: `cardio`, `resp_id`, `gi_liver`,
`neuro`, `rheum_immune`, `peds_obgyn` (`SPECIALTIES`, Korean labels `SPECIALTY_KO`).

API (every function catches all errors):
- `specialty_of(dx_name) -> str | None`; `specialty_detail(dx_name) -> {specialty, group, how, code, name}`. `group` also
  names buckets outside the six (`endo_metab, renal_uro, heme_onc, psych, derm, ent_eye, msk_ortho, tox_trauma,
  symptom, other`); `how` ∈ `override, kcd, do, keyword, none`.
  Order: `OVERRIDES` (short curated regex list where the ICD chapter misleads: endocarditis → cardio, pregnancy/neonatal
  words → peds_obgyn, meningitis/encephalitis → neuro, pneumonia/pleurisy/ILD → resp_id, hepatitis/haemochromatosis/
  Wilson/mesenteric ischaemia → gi_liver, sepsis → resp_id, anaphylaxis/serum sickness → rheum_immune) →
  `kb.normalize_diagnosis()` KCD code → `KCD_TABLE` (chapter/block ranges, most specific first; e.g. A00–A09 and
  B15–B19 → gi_liver, other A/B → resp_id, I60–I69 → neuro, M00–M19/M30–M36/M45–M46 → rheum_immune, O/P, N70–N98,
  most Q → peds_obgyn) → nearest mapped Disease Ontology ancestor (`DO_MAP`, organ classes before system-wide ones) →
  `FALLBACK` organ keywords → None.
  Latency: the KB fuzzy step runs only for short Korean names (`FUZZY_MAX_KEY` 14 compact chars); it costs 40–100 ms on
  long English names (`kb.normalize_diagnosis(text, fuzzy=False)` / `_resolve(..., fuzzy=False)` added for this).
- `route(state) -> (specialty | None, share, [Korean reasons])`. Candidates = live `ddx_ledger` entries (else
  `state.ddx`), top `TOP_K` 5; mass = p (else rank weights 1/(r+1)); share = mass per specialty / total mass (unmapped
  candidates stay in the total). Patient-context override first: infant (< 1 y) → peds_obgyn, share 1.0; child (< 18 y,
  `nlp.findings.age_from_text`, then `clinical_rules.rule_age_years`) or current pregnancy
  (`safety.protocols.PREDICATES["current_pregnancy"]` on initial info + responses) → peds_obgyn when the top-1 candidate
  is age/pregnancy-relevant or the relevant share ≥ `CONTEXT_SHARE` 0.3 (relevant = maps to peds_obgyn; child: KCD P/Q,
  pediatric-named KB profile or `PEDIATRIC_DX`; pregnancy: KCD O or a pregnancy word). Otherwise argmax over the six.
  Suggested caller gate: `MIN_TURNS` 3 and share ≥ `MIN_SHARE` 0.6.
- `resources(specialty, state) -> {specialty, criteria, rules, protocols, kb_candidates}` ({} on error / unknown id),
  bounded by `MAX_ITEMS` (3/4/3/4): criteria = `diagnostic_criteria.CRITERIA` sets tagged for the specialty
  (`CRITERIA_SPECIALTY`) or naming a current candidate, candidate-named first; the first such set is evaluated
  (`evaluate`, band + met/not_met/unknown counts) on initial info + last 8 responses + verified findings ledger;
  rules = `clinical_rules.RULES` tagged for the specialty (`RULE_SPECIALTY`) with `applies` = `Rule.applies_to(chief
  complaint)` (applicable first; the two first tagged rules when none applies); protocols = `protocols.protocols_for`
  on the chief complaint with can't-miss list, pending checks (`pending_checks`) and `in_specialty`
  (`CATEGORY_SPECIALTY`); kb_candidates = the ledger's own candidates in the specialty, then `kb.candidates()` (last 8
  positive / 4 negative findings, sex/age) filtered to the specialty, each with 3 typical findings and 2 decisive tests.
  `render_resources(res, max_chars=700)` → Korean block "[… 분과 참고 자료: 확진 근거가 아니라 감별·검사 계획용]".
- `warm()`: load the KB and build its lazy indexes once at start-up (normalize/fuzzy bigram indexes, sex table,
  prevalence), so the first case does not pay ~1–2 s.
- The slice tables must cover every criteria id, rule id and protocol category (`tests/test_specialty.py` fails when
  a new one is added elsewhere without an entry; an empty tuple is allowed).

Offline numbers (`eval/offline/eval_specialty.py`, metrics `data/labels/specialty_gold_v1_metrics.json`):
- Gold set `data/labels/specialty_gold_v1.jsonl`, 116 names (100 Korean, 16 English; 22 outside the six), labelled
  before any output. Labelling guide: organ system first; infections → resp_id unless the organ is GI/liver (gi_liver)
  or the CNS (neuro); stroke → neuro; systemic autoimmune, vasculitis, inflammatory arthritis, gout, allergy/
  anaphylaxis → rheum_immune; pregnancy, female genital, neonatal and congenital syndromes → peds_obgyn; everything
  else → None with a bucket. `also` lists acceptable alternatives. First pass: strict 110/116 (94.8%), lenient 113/116
  (97.4%); after fixes made while reading those errors: strict 113/116 (97.4%), lenient 116/116.
- cases_aug (267): cardio 26, resp_id 32, gi_liver 36, neuro 37, rheum_immune 19, peds_obgyn 20, outside 97 (36.3%):
  heme_onc 25, renal_uro 13, other 13, psych 11, derm 11, endo_metab 10, msk_ortho 9, ent_eye 5.
- Routing replay (224 non-dummy trajectories of 2026-09-25/26 runs): a consult fires (≥ 3 turns, share ≥ 0.6) in 151
  (67.4%), mean firing turn 3.1; routed = gold specialty 130/151 (86.1%), 130/139 (93.5%) when the gold is inside the six.
  Warm latency (shared machine, load 7–20): route mean 0.8 ms, p95 1.7 ms; resources mean 11 ms, p95 24 ms.
