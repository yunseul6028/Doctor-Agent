# Experiment Log

- Safety = share of the guideline-based checklist (`safety/protocols.py`) completed. Chief-complaint categories are detected from the initial info only (since 2026-09-25; earlier results were rescored).
- Patient/judge LLM: gemini-3.6-flash. The doctor model is listed per row.
- **Model setup (2026-10-07)**: the doctor is now Gemini Pro (`gemini-3.1-pro-preview`, pinned) via the OpenAI-compatible
  Gemini endpoint; patient and judge stay on `gemini-3.6-flash`. `openai/gpt-oss-20b` (local Ollama,
  `--doctor-endpoint local`) was the original design target and is now an optional preset. **Every scored row below
  used a Gemini Flash / Flash-Lite / Gemma doctor — none used Gemini Pro, and none used gpt-oss-20b.** Thresholds
  (confidence 0.3 / 0.4, turn 5, share 0.6) were calibrated on those trajectories and must be re-checked on Pro.
  Dated entries below that say "measure on gpt-oss-20b" are kept as written; read them as "measure on the doctor
  model" (now Gemini Pro). Keep Pro rows separate from the earlier models.
- Open question for the Pro runs: does code verification still help a strong model, or do the hints become noise?
  An earlier review found code hints ≈ 43% of the prompt, with some contradictions. Answer it with the ablation
  conditions (`v6-no-kb`, `v6-no-safety`, `v6-no-advisors`, `v6-no-interp`, `v6-no-subagents`).
- The token-budget sections (2026-09-28/29) were measured with the gpt-oss tokenizer for the original target. Character
  counts and shares carry over; Gemini token counts may differ somewhat. Gemini Pro's context is far larger than
  gpt-oss-20b's 131k, so the 12,000-char view cap is conservative; but Pro is a thinking model (slower, more output
  tokens), so the throughput assumptions of the time model do not apply and time estimates must be re-measured on Pro.
- Local scores only reflect **relative change**. They are not an absolute performance estimate.
- Runs and offline numbers on "267 cases" / "held-out" include the private AgentClinic- and DiagnosisArena-derived sets (156 cases), which are not in the public repository; the public set is 111 cases (ClinicalQA-derived + synthetic). See `docs/data-sources.md` ch. 8.

| Date | Change | Prompt version | Doctor model | Cases | Patient type | Accuracy | Efficiency | Safety | Avg turns | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | Skeleton | v0 | dummy | synthetic_001 | keyword | – | – | – | – | smoke test |
| 2026-09-25 | Korean conversion + per-turn reasoning | v2-ko-reason | gemini-3.6-flash | synthetic 16 | standard | 1.00 | 0.92 | 0.39 | 4.7 | too easy; frequent early diagnoses |
| 2026-09-25 | Decision rules + guideline checks + pushback before diagnosis | v3-ko-rules | gemma-4-26b-a4b-it | synthetic 15 (1 timeout) | standard | 1.00 | 0.88 | 0.79 | 7.5 | safety 0.39→0.79 |
| 2026-09-25 | Same | v3-ko-rules | gemma-4-26b-a4b-it | synthetic 13 (3 failed) | mixed tricky | 1.00 | 0.88 | 0.86 | 7.2 | failures: rate limit/timeout from parallel runs |
| 2026-09-25 | Same | v3-ko-rules | gemma-4-26b-a4b-it | ClinicalQA medium/hard 14 | standard | 0.61 | 0.86 | 0.70 | 8.9 | first real errors. 2 were simulator artifacts ("missing result" = "normal") |
| 2026-09-26 | "Result not provided" separated from "normal" + pushback on low confidence | v4-ko-confidence | gemma-4-26b-a4b-it | 6 previous misses only | standard | 0.67 | 0.82 | 0.67 | 10.7 | 3 of 6 recovered. Estimated ~0.86 across all 14 |
| 2026-09-26 | Near-duplicate blocking + question/exam type correction + unavailable-result hint (Gemma run stopped for speed) | v4-ko-confidence | gemini-3.5-flash-lite | ClinicalQA 40 (original) | standard | 0.91 | 0.89 | 0.82 | – | flash-lite baseline |
| 2026-09-26 | Same | v4-ko-confidence | gemini-3.5-flash-lite | ClinicalQA 40 (augmented) | standard | 0.94 | 0.91 | 0.86 | – | augmentation effect +0.03 |
| 2026-09-26 | Findings ledger + DDx ledger + pre-diagnosis review | v5-ko-ledger-review | gemini-3.5-flash-lite | ClinicalQA 40 (original) | standard | 0.90 | 0.89 | 0.85 | – | reviewer: 0 holds, 6 diagnosis names revised |
| 2026-09-26 | Same | v5-ko-ledger-review | gemini-3.5-flash-lite | ClinicalQA 40 (augmented) | standard | **0.99** | 0.92 | 0.72 | – | reviewer: 0 holds, 11 names revised (bipolar II, ATN, long QT fixed). Safety drop is mostly protocol false triggers |

Prompt `v6-kb-strict-review` (KB hints, code-decided review, evidence-gated renaming) has **no LLM run yet**.
The first rows for it should come from `eval/experiment.py` with the Gemini Pro doctor (see "How to measure" below).

### 2026-09-28 · prompt `v6-kb-strict-review` → `v7-advisors` (no LLM run yet)
The four advisor modules are wired into the policy (`docs/architecture.md` "Advisors wired into the policy"): triage
alert above the case view for unstable patients (or missing vitals with a red flag), turn-1 starting DDx, one-time
anchoring (premature-closure) hint, per-turn question-planner suggestions (KB on), and one code-confidence pushback when
a proposed diagnosis scores < 0.3. `SYSTEM` / `REVIEW_SYSTEM` are unchanged; new wording is `prompts.TRIAGE_ALERT` and
`prompts.CONFIDENCE_PUSHBACK`. Advisor text ≤ `AGENT_MAX_ADVISOR_CHARS` = 900 per step prompt. Ablation condition:
`v6-no-advisors` (added to the `dev` profile). Dummy smoke only (16 sample cases, keyword patient, no judge; the dummy
doctor ignores hints, so scores cannot move): mean max prompt 2,322 → 2,619 chars, mean LLM calls per case 7.56 → 8.56
(the confidence pushback), Accuracy / Efficiency / Safety 0.062 / 0.91 / 0.385 in both. **Measure v6 vs.
v6-no-advisors on gpt-oss-20b before relying on it**; the 0.3 pushback threshold was not calibrated for the new model.

### 2026-09-28 · prompt `v7-advisors` → `v8-result-interp` (no LLM run yet)
The code-first result interpreter is wired into the policy (`docs/architecture.md` "Result interpreter wired into the
policy"): every EXAM/TEST result is read by code into the findings ledger (verified, source `result_interpreter`;
"not provided" / pending → 결과없음, never 음성), the latest reading is a hint (`prompts.RESULT_HINT`, ≤ 300 chars, in the
advisor budget after the triage hint), critical results get a one-time top-of-prompt alert
(`prompts.RESULT_CRITICAL_ALERT`, in the triage alert slot) and critical IMG/ECG readings confirm the matching
can't-miss danger in `danger_gate`. `SYSTEM` / `REVIEW_SYSTEM` unchanged. The optional LLM reading
(`RESULT_INTERPRETER_PROMPT`) is still not called; results that would need it are counted in `result["result_interp"]`.
Ablation condition `v6-no-interp` (`AGENT_USE_RESULT_INTERPRETER=0`), **not** in `dev`. Dummy smoke (16 sample cases,
keyword patient, no judge; the dummy doctor ignores hints, so scores cannot move): Accuracy / Efficiency / Safety
0.062 / 0.91 / 0.385 with the interpreter on and off, 5.4 turns, 8.56 LLM calls per case in both; 54 results read,
2 would need the LLM reading (both `unmapped_findings`), 3 critical, 13 "not provided", 0 errors, 130 code findings in
the ledgers. `scripts/token_budget.py --tokenizer chars --limit 3`: the result hint adds 30.5 tokens mean (max 102;
present in 72% of step prompts). **Measure v6 vs. v6-no-interp on gpt-oss-20b before relying on it.**

### 2026-09-28 · prompt `v8-result-interp` → `v9-subagents` (framework only, no LLM run yet)
Specialist sub-agent framework wired into the policy (`docs/architecture.md` "Specialist sub-agents"): extra calls of
the same fixed gpt-oss-20b in another role, only when triggered, each at most once per case and ≤ 3 per case in total
(`AGENT_MAX_SUBAGENT_CALLS`): **consult** (routed specialty share ≥ 0.6 from turn 3, or model confidence < 0.5 on 3
consecutive turns from turn 6), **advocate** (the turn the anchoring hint fires, else before a pre-diagnosis review of a
proposal with code confidence < 0.65 → note in the review view), **LLM radiology** (a `needs_llm` result, 1 per case;
findings into the ledger as source `llm_radiology`, grounding-checked, never replacing a code reading). Skipped in
low-time mode, with ≤ 3 turns left or at the cap. Hints: separate budget `AGENT_MAX_SUBAGENT_CHARS` = 600 after the
advisor hints; new wording `prompts.SUBAGENT_HINT`, `ADVOCATE_REVIEW_NOTE`; sub-agent DDx candidates are shown as a
"참고(자문, 미확인)" line of the DDx ledger view but kept out of the live DDx. `SYSTEM` / `REVIEW_SYSTEM` unchanged.
Content modules (`consult.py`, `advocate.py`, `knowledge/specialty.py`) come from the content branches; without them
consult / advocate are skipped (`module_missing`). Ablation condition `v6-no-subagents` (`AGENT_USE_SUBAGENTS=0`),
**not** in `dev`. Tests switch sub-agents off by default (`tests/conftest.py`); `tests/test_subagents.py` turns them on.
Dummy smoke on this branch (16 sample cases, keyword patient, no judge; content modules absent; the dummy doctor
ignores roles, so the 2 radiology calls got an action JSON back and failed as bad JSON — handled, case went on):
Accuracy / Efficiency / Safety 0.062 / 0.91 / 0.385 (unchanged), 5.4 turns, 8.69 LLM calls per case (8.56 before: the
2 radiology calls). **Measure v6 vs. v6-no-subagents on gpt-oss-20b before relying on it**; the trigger thresholds
(0.6 share, 0.5 / 0.65 confidence) are uncalibrated.

### 2026-09-28 · specialist sub-agent content (not wired, no LLM run)
Content only (`agent/subagents/consult.py`, `advocate.py`; see `docs/architecture.md` "Specialist sub-agent content"):
six specialist consult prompts (cardio, resp_id, gi_liver, neuro, rheum_immune, peds_obgyn with a pediatric /
pregnant / reproductive-age-female branch) and a devil's-advocate diagnostic time-out prompt, with tolerant parsers.
Not part of any prompt version until the framework wires it. Sizes (tiktoken `o200k_harmony`, messages only, without
the harmony wrapper): consult system 374 tokens; full consult turn 0 = 832–1,090 tokens, worst case (60 turns,
80 findings, 8 DDx, 2 kB resources) 2,657–2,759 tokens; advocate 584 (turn 0) / 2,361 (worst). Each call adds one
gpt-oss call. **Measure on gpt-oss-20b as ablations** (consult on/off, advocate on/off) with per-metric scores before
relying on them; the parsers were tested only on hand-written outputs.

## 2026-09-26 → 09-27: changes since the last LLM run (no LLM calls)

Everything below was measured offline (rules, KB, case files). None of it has an Accuracy/Efficiency/Safety score yet.

| Area | Change | Measured |
|---|---|---|
| Agent | v6: KB hints (candidates, discriminators, dx normalisation); structured pre-diagnosis review with code-decided verdict and evidence-gated renaming; EXAM/TEST history questions → ASK; DDx name-variant dedupe | unit tests only |
| Runtime | gpt-oss harmony handling, length retry, optional structured output, per-case time budget with degrade/forced final answer, never-crash robust mode, prompt cap, incremental `run.py` output | `tests/test_runtime.py` (28 tests) |
| Safety protocols | 7 → 20 → **26** categories; false triggers fixed from a per-case audit (periarticular pain ≠ hot joint, urticaria ≠ chronic pruritus, pregnancy test populations, adult-only dyspnea checks, negation window, …); rules got population conditions from the source abstracts | cases in `data/cases_aug` (267) with ≥ 1 applicable check: **87 → 159 → 173** (7 / 20 / 26 categories, re-measured on the current case files) |
| Cases | Full augmentation of all sets → `data/cases_aug` (267); rule-based quality gate `scripts/check_cases.py` | hard issues **48 → 0** (45 placeholder search terms, 2 sex/age-inconsistent tests, 1 vital conflict); 15 cases / 57 edits; 2,278 soft issues left as a review list (`data/labels/case_quality_2026-09-27.json`) |
| KB | Matching/ranking/normalisation rework (09-26), then curated test-result → disease links `kb_tests.py` (262 concepts, 452 links, 95 PMID-verified refs) | see below |
| Eval | One-command experiment runner (profiles smoke/dev/full, cost guard, token metering, compare, log, share page) | `tests/test_experiment.py` |
| Result interpreter (09-28; wired later the same day, prompt `v8-result-interp`) | `agent/result_interpreter.py`: code-first reading of EXAM/TEST result text (negation, hedges, comparison, sections, organ-aware report words, critical values, supportive kb_tests links); new prompt `RESULT_INTERPRETER_PROMPT` added to `prompts.py` but **not called** (PROMPT_VERSION unchanged) | gold (concept+polarity): fresh set P/R 0.879/0.879 before the fixes it prompted, dev + fresh 1.00 after (same author, optimistic); `tests/test_result_interpreter.py` |

KB offline benchmark (`scripts/eval_kb.py`, gold diagnosis rank in `candidates(k=50)` from the case text; dev = sample +
clinicalqa 111, held-out = agentclinic + diagnosisarena 156; `data/labels/kb_eval_2026-09-27.json`):

| Split | top-1 | top-3 | top-10 | top-50 | MRR | note |
|---|---|---|---|---|---|---|
| held-out, before test links | 0.045 | 0.109 | 0.211 | 0.295 | 0.091 | after the 09-26 rework |
| **held-out, now** | **0.090** | **0.179** | **0.269** | **0.359** | **0.147** | the realistic estimate |
| dev, now | 0.460 | 0.604 | 0.685 | 0.775 | 0.542 | tuned here, inflated |
| held-out, history + exam only | 0.058 | 0.109 | 0.186 | 0.263 | 0.096 | what the hints can use before tests |

Latency 8.8 ms mean / 10.1 ms p95 per `candidates()` call, KB load 0.32 s. Diagnosis normalisation (held-out): KCD
code for 56.4% of primary names. Details: `docs/data-sources.md` (knowledge base section).

## How to measure

One command runs a standard profile, estimates cost first, compares conditions, and rebuilds the viewer + share page
(`eval/experiment.py`; profiles in `eval/experiment_profiles.json`; fixed case lists in `eval/case_lists/`).

The doctor is Gemini Pro (`gemini-3.1-pro-preview`, pinned); patient and judge are `gemini-3.6-flash`.

0. Top up API credits first: Pro costs more than Flash, and a billing error stops the batch (see the guard below).
1. Point the doctor at the endpoint (keys stay in `.env`, the script never prints them): `--doctor-endpoint gemini`
   (`GEMINI_LLM_*`, else the shared `LLM_*`; the preset uses Pro for the doctor), or `--doctor-endpoint env` with
   `DOCTOR_LLM_BASE_URL=…`, `DOCTOR_LLM_API_KEY=…`, `DOCTOR_LLM_MODEL=…` (any OpenAI-compatible server).
   Optional preset: `--doctor-endpoint local` (`LOCAL_LLM_*` or Ollama `gpt-oss:20b`, the original design target).
2. Free wiring check: `python eval/experiment.py --profile smoke --doctor-endpoint dummy --no-view`
3. Plan + cost only: `python eval/experiment.py --profile dev --doctor-endpoint gemini --estimate-only`
   (add `--price-in/--price-out` in KRW per 1M tokens, or `EXPERIMENT_PRICE_{IN,OUT}_PER_M`, to get a KRW figure).
4. Connection check on the real model (5 cases, ~50 doctor calls): `python eval/experiment.py --profile smoke --doctor-endpoint gemini`
   → check `usage:` lines (real tokens/call, including thinking tokens), latency and parse failures before anything
   bigger; later estimates use them automatically. Pro is a thinking model: fit the time model
   (`result["runtime"]["latency_main_s"]`) and set `AGENT_CASE_TIME_BUDGET_S` from these calls.
5. Dev comparison and ablations (50 cases × v6 / v6-no-kb / v6-no-safety / v6-no-advisors, needs `--yes`):
   `python eval/experiment.py --profile dev --doctor-endpoint gemini --yes --log --note "first Gemini Pro run"`
   Then `--conditions v6,v6-no-subagents,v6-no-interp`. Drop components that do not raise Accuracy or Safety.
6. Re-calibrate on the Pro runs: confidence weights and the 0.3 / 0.4 thresholds (`scripts/calibrate_confidence.py`),
   sub-agent triggers (`python eval/offline/eval_triggers.py --results <Pro result folder>`).
   Optional: `--conditions v6,v5-baseline` (v5 runs in a temporary git worktree at `c3ddecd`, rescored with the current
   scorer), `--env AGENT_CASE_TIME_BUDGET_S=240`, `--env DOCTOR_LLM_STRUCTURED_OUTPUT=guided_json`, `--compare-with FILE`.
7. `full` (111 public cases) only when a dev result justifies it.

Defaults are the cheapest mode: keyword patient + no judge, so only the doctor spends API budget
(`--patient llm --judge llm` opt in; those use `PATIENT_LLM_*`/`JUDGE_LLM_*`, i.e. Gemini). Guard: `--yes` is required
above `--max-calls` (300 doctor calls) or `--max-cost` (5,000 KRW, when prices are given). A billing error (402 /
credits depleted on a paid endpoint) stops the running batch and skips the remaining conditions (exit code 3).
Cost estimate = cases × doctor calls/case (mean of past result files, else 12) × tokens/call, all × margin 1.3.
Tokens/call: recorded usage of the same doctor model; else, for gpt-oss (optional preset), the **measured** prompt tokens (see "Prompt
token budget" below; mean step prompt over a case of the past mean length) + an assumed 1,000 / 1,800 / 2,048 output
tokens at effort low / medium / high; else usage of any other model; else 3,000 in / 1,000 out. Doctor token usage is recorded per case
(`usage` in each row) and per run by wrapping the SDK client in `eval/usage.py` (src untouched).
`--log` appends one table row per condition above and a comparison section (overall, per set, flips, n/a rate) under
"Auto-logged experiment runs" at the end. The share page path is printed (`eval/results/share_<time>.html` or `--share`).

### LLM record/replay cache (stretching the API budget)
`eval/replay.py` wraps the SDK's `chat.completions.create` at the eval layer (like `eval/usage.py`; src untouched, never
shipped). Key = sha256(endpoint, model, messages, temperature, max_tokens, reasoning_effort, response_format/extra_body,
`--cache-salt`, `--cache-sample-idx`); `timeout` is ignored. One append-only JSONL per role in `eval/cache/`
(git-ignored; `doctor.jsonl`, `patient.jsonl`, `judge.jsonl`; content, reasoning, finish_reason, usage per entry;
fcntl-locked appends, safe with `--workers N`).
- Modes (`--llm-cache`): `off`, `record` (always call, store; latest entry wins), `auto` (hit → replay, miss → call +
  record), `replay` (no API calls at all; the first miss aborts the batch, exit code 4, no result file).
- `eval/experiment.py` defaults: patient/judge `auto`, doctor `off`; `--cache-doctor` (or `"cache_doctor": true` on a
  condition in `experiment_profiles.json`) caches the doctor too. `run_local.py` defaults to `off`
  (`--llm-cache auto [--cache-doctor | --doctor-cache MODE]`).
- Result JSON: `llm_cache` = per role hits / misses / recorded / tokens avoided (`saved_*`), plus `saved_krw_doctor`
  when `EXPERIMENT_PRICE_{IN,OUT}_PER_M` are set. `usage` still counts only tokens actually billed (the cache sits above
  the meter). The comparison prints a `cache` line per run and `--log` adds it to the notes.
- Where it saves: the judge prompt is (case answer, predicted diagnosis), so every repeated diagnosis across conditions
  and reruns is free. Patient answers hit only while the conversation prefix is identical (same doctor question
  sequence), i.e. unchanged conditions and the doctor-cached replays. The doctor (the main API cost) hits when a
  condition sends exactly the same requests again: reruns of the same code/prompt (baseline conditions, rescoring or
  scorer/viewer changes, parser-only changes that do not alter the prompts). A prompt change misses from its first
  differing call onward, so the saving there is the unchanged prefix only.
- Caveats: replay is deterministic — with temperature > 0 it reuses the first sample, so a replayed rerun hides sampling
  variance. For variance runs run with `--cache-sample-idx k`, k = 0, 1, 2… (each k is a separate stored sample per request) or a new
  `--cache-salt` to force fresh answers. Time-budget branches (degraded mode, deadlines) can differ between a live run
  and a fast replay, which shows up as misses, not as wrong hits. The cost estimate ignores expected hits (upper bound).

## Prompt token budget (gpt-oss tokenizer, 2026-09-28, no LLM calls)

`python scripts/token_budget.py [--limit N] [--max-view-chars N] [--json-out eval/results/token_budget.json]`
replays cases through the real `Policy` (hints, ledgers, capped view, safety layers, KB hints) with a scripted doctor
that never diagnoses early, so every case runs all 60 turns against the keyword patient. Every prompt is rendered in
the harmony format exactly as gpt-oss receives it (system message with `Reasoning: low`, our system prompt as the
developer `# Instructions`, user message, `<|start|>assistant`; identical to `openai-harmony`, which vLLM uses) and
counted with tiktoken `o200k_harmony` (same ids as the HF `openai/gpt-oss-20b` tokenizer.json on our prompts).
Review and final prompts are probed on a copy of the state at the checkpoint turns. Dev-only dependencies: `tiktoken`
(MIT), `openai-harmony` (Apache-2.0, test cross-check). Scripted answers report 1–2 findings per turn (detail ≤ 60
chars) and a 5-entry DDx with for/against evidence; real gpt-oss answers may be shorter or longer.

All 267 `data/cases_aug` cases, prompt `v6-kb-strict-review`, `AGENT_MAX_VIEW_CHARS=12000` (16,264 real calls + 4,272
probes, 315 s). Tokens of the full prompt, p50 / p95 / max:

| Turn | step | review | final |
|---|---|---|---|
| 1 | 903 / 1,521 / 1,874 | 727 / 755 / 1,031 | 781 / 791 / 802 |
| 5 | 1,503 / 2,169 / 2,706 | 1,274 / 1,424 / 1,703 | 1,322 / 1,453 / 1,717 |
| 10 | 1,964 / 2,580 / 3,109 | 1,800 / 1,994 / 2,281 | 1,847 / 2,028 / 2,112 |
| 20 | 2,846 / 3,406 / 4,192 | 2,693 / 2,908 / 3,192 | 2,739 / 2,932 / 3,227 |
| 40 | 4,230 / 4,874 / 5,428 | 4,001 / 4,409 / 4,741 | 4,051 / 4,454 / 4,614 |
| 59 / 60 | 5,017 / 5,718 / 6,312 (59) | 4,787 / 5,302 / 5,720 (59) | 4,887 / 5,385 / 5,734 (60) |

Mean step-prompt tokens per call over a whole case: 1,320 (5 turns), 1,578 (10), 2,051 (20), 2,876 (40), 3,464 (59).
Retry prompts (parse / duplicate / pushback hints) were 1.5% of step prompts.

Largest contributors (step prompts, mean tokens / share / present): findings ledger 1,147 / 33% / 98%; older-actions
list 940 / 27% / 92%; system prompt 643 / 19% / 100% (fixed); DDx ledger 245 / 7%; last 4 exchanges 151 / 4%;
clinical rules hint 95 / 2.7% / 22% (541 tokens p95 when present, the largest hint); harmony wrappers 74 (fixed);
safety-check hint 41 (max 471); can't-miss hint 38 (max 147); unavailable-requests hint 23; target-turn notice 13;
KB candidate / discriminator hints ≤ 257 / 124 but present in only 3% / 1% of prompts. All hints together: p50 122,
p95 717, max 1,361 tokens. The review adds the criteria block (≤ 367 tokens, 4% of reviews).

Budget checks:
- The 12,000-char view cap was **never reached** (uncapped view chars p50 4,270, p95 6,417, max 7,715 at turn 60).
  View tokens/char p50 0.64, max 0.71 → a view at the cap would be ≈ 8,550 tokens; worst prompt at the cap ≈
  743 (fixed) + 1,361 (hints) + 8,553 ≈ 10.7k tokens.
- Largest measured prompt 6,312 tokens: tiny against the 131,072-token gpt-oss-20b context, but the server may be
  configured smaller. With `max_tokens` 2048: every prompt fits 8k; with the length retry (2 × 2048 = 4096) only 63%
  fit 8k, 100% fit 16k. At 4k: 63% of prompts alone fit, 21% with max_tokens.
- Stress check with `--max-view-chars 4000` (30 cases): 55% of step prompts were cut, max prompt 3,986 tokens; the
  cap bounds the prompt as expected (fixed + hints + cap × 0.71).
- Scripted action JSON (no reasoning) is 307 p50 / 446 max tokens; the output assumption above adds the reasoning.

Recommendations (not applied here):
1. Once the server context `C` (vLLM `--max-model-len`) is known, set `AGENT_MAX_VIEW_CHARS ≤ (C − max_tokens_retry −
   2,100) / 0.71` (2,100 = fixed + max hints). 16k context: keep 12,000. 8k: ≈ 5,500 with max_tokens 2048 and set
   `DOCTOR_LLM_MAX_TOKENS_CAP=2048` (no room for the 4096 retry). vLLM rejects prompt + max_tokens > context with an
   error (400), while Ollama truncates prompts longer than `num_ctx` without an error → check `num_ctx` for local runs.
2. The findings ledger is the biggest and grows without bound until the view cap (the older-actions list is cut
   first). Cheapest cuts before touching the view cap: shorter finding `detail`, one compact line for `결과없음` items,
   and a shorter older-actions line (40 + 60 chars today).
3. Hint trimming order under pressure (least safety value first): KB discriminator → KB candidates → target-turn notice →
   clinical rules (render 1 rule instead of 2; 541 tokens p95) → unavailable list; keep can't-miss and pending safety
   checks last (Safety score). Degraded mode already keeps only the first two hints.
4. The system prompt (643 tokens, 19%) is identical on every call: prefix caching on the server makes it cheap in
   latency, but it is billed per call.

## Sub-agent token and time budget (gpt-oss tokenizer, 2026-09-29, no LLM calls)

`python scripts/token_budget.py --subagents --jobs 9` (all 267 `data/cases_aug` cases, prompt `v9-subagents`,
`AGENT_MAX_VIEW_CHARS=12000`, main and sub-agent effort `low`, 1,223 s on 9 processes). Same harmony rendering and
`o200k_harmony` counting as the table above. No prompt or policy change, so no Accuracy / Efficiency / Safety row.

**Scripted doctor (how representative it is).** To make the triggers fire, the scripted doctor (a) narrows its DDx to
five diagnoses of one specialty once 5 turns are done (the gold diagnosis's specialty, `knowledge/specialty.py`; earlier
candidates marked 배제), so the routed consult fires in **100%** of cases, (b) keeps its confidence at 0.3 (< 0.5, the
stuck-consult trigger is armed too), and (c) proposes DIAGNOSE from turn 10 or 20 (or never: 60 turns). From that turn
the real pushback → gate → confidence pushback → pre-review advocate → review path runs; the scripted review always
names an unresolved danger, so both reviews (MAX_REVIEWS = 2) happen and hold → the cases end at turn 11 / 21. Sub-agents
get short, schema-complete JSON answers (consult 243, advocate 183, radiology ≤ 194 visible tokens) so their hints reach
later prompts. This is a **worst-case-leaning** load, not a forecast: real consult rates were 36% at turn ≥ 5 in the
dev-model replays, the advocate fires only at code confidence < 0.4 (46% / 79% here, 33% in the replays), and radiology
fires only where the result interpreter asks for an LLM reading (61% of 60-turn cases, 23% when diagnosing from turn
10). Dev-model runs averaged 4.7–10.7 turns, so "DIAGNOSE from turn 10" is the closest to what was seen; 60 turns is the
cap. Reasoning tokens are not in these counts (see the time model).

Sub-agent prompts, probed on a copy of the state at turns 6 / 10 / 20 / 40 / 57 (57 = last turn a sub-agent may run),
tokens of the full harmony prompt p50 / p95 / max:

| Prompt | p50 | p95 | max | max at turn 6 → 57 |
|---|---|---|---|---|
| consult, all 10 specialties pooled (n = 13,350) | 2,445 | 2,991 | 3,324 | — |
| consult worst specialty (peds_obgyn / heme_onc) | 2,479 / 2,515 | 3,007 / 3,077 | **3,324** / 3,316 | 2,055 → 3,324 |
| consult lightest (gi_liver) | 2,358 | 2,898 | 3,078 | 1,965 → 3,078 |
| advocate | 1,768 | 2,297 | 2,402 | 1,259 → 2,402 |
| radiology, longest single result of the case | 646 | 682 | 765 | (turn-independent) |
| radiology, all result texts up to the 3,000-char cap | 1,543 | 1,798 | 2,174 | (turn-independent) |

Consult prompt blocks (mean / max tokens / share): findings ledger 511 / 973 / 22%; specialty card 436 / 647 / 18%;
system prompt 375 / 397 / 16%; resources 336 / 487 / 14%; current DDx 292 / 476 / 12%; actions done 171 / 270 / 7%;
recent exchanges 76 / 188; harmony 74. Every block is capped in `consult.py` (findings 1,400 chars, DDx 700, done 450,
resources 900, initial 600), so the consult prompt is bounded (≈ 3.3k tokens) however long the case runs; it is always
smaller than the late main step prompts (6,312 max).

Per case (real calls only; step includes retries; tokens are prompt tokens summed over all calls of the case;
time = time model below):

| Case length | Sub-agents | Calls/case p50 / max (main + sub) | Sub calls/case mean (consult / advocate / radiology) | Prompt tokens/case p50 / p95 / max | Sub-agent share | Predicted s/case p50 / p95 / max, conservative | same, moderate |
|---|---|---|---|---|---|---|---|
| 60 turns | off | 61 / 67 | 0 | 249k / 294k / 342k | 0% | 2,550 / 2,670 / 2,817 | 839 / 877 / 925 |
| 60 turns | on | 62 / 69 | 1.0 / 0 / 0.6 | 255k / 298k / 349k | 0.9% | 2,597 / 2,723 / 2,876 | 854 / 892 / 944 |
| DIAGNOSE from 20 | off | 25 / 29 | 0 | 63k / 84k / 110k | 0% | 953 / 1,064 / 1,180 | 317 / 353 / 390 |
| DIAGNOSE from 20 | on | 27 / 30 | 1.0 / 0.5 / 0.4 | 67k / 88k / 114k | 4.2% | 1,004 / 1,120 / 1,209 | 333 / 371 / 400 |
| DIAGNOSE from 20 | on, advocate forced | 27 / 31 | 1.0 / 1.0 / 0.4 | 68k / 90k / 116k | 5.6% | 1,016 / 1,142 / 1,236 | 338 / 378 / 409 |
| DIAGNOSE from 10 | off | 15 / 19 | 0 | 30k / 45k / 58k | 0% | 557 / 678 / 731 | 186 / 226 / 243 |
| DIAGNOSE from 10 | on | 18 / 21 | 1.0 / 0.8 / 0.2 | 33k / 49k / 63k | 8.8% | 619 / 741 / 788 | 207 / 246 / 262 |
| DIAGNOSE from 10 | on, advocate forced | 18 / 21 | 1.0 / 1.0 / 0.2 | 34k / 49k / 63k | 9.5% | 620 / 745 / 788 | 207 / 248 / 262 |

Reading: sub-agents add at most 3 calls per case (the cap), +2–3 calls in the typical-length case, +4–10% prompt tokens
(the sub-agent prompts are shorter than the main prompts they sit between) and at most 80–82 s (conservative) /
27–28 s (moderate) per case. The main loop is ≥ 90% of the predicted time in every scenario. Main-prompt
growth from the injected hints (≤ 600 chars per step, one step each) is included in the "on" rows.

**Time model** (`scripts/token_budget.py`: `Throughput`, `estimate_call_s`, `estimate_case_s`; in `agent/runtime.py` until 2026-09-30):
`t_call = overhead_s + prompt_tokens / prefill_tps + (visible_output_tokens + reasoning_tokens) / decode_tps`, and a
case is the sum over its calls (the agent calls sequentially; CPU time between calls is not included — ~0.1 s per
step with KB on in a cProfile of this replay). **All parameters are assumptions**, not measurements or published numbers: the
server, GPU, batching, prefix caching and gpt-oss's reasoning length at effort `low` on our prompts are unknown.

| Assumption | prefill tok/s | decode tok/s | overhead s | reasoning tok/call | e.g. 3k-token prompt, 300-token answer |
|---|---|---|---|---|---|
| `CONSERVATIVE` (default) | 1,000 | 20 | 1.0 | 300 | 34 s |
| `MODERATE` | 4,000 | 60 | 0.5 | 300 | 11 s |

Decode dominates (≥ 85% of each estimate): the number that matters is decode tok/s × (answer + reasoning) length.
On the first gpt-oss-20b run: run a few cases, read `result["runtime"]["latency_main_s"]` (now recorded per call) and fit the four
parameters with the prompt token counts above, then `--prefill-tps / --decode-tps / --overhead-s / --reasoning-tokens`
re-predicts the table.

**Degraded mode vs sub-agents** (checked; code in `GuardedLLM.subagent_time_block`, `SubagentManager.blocked`, tests
in `tests/test_subagents.py` "time budget"). Sub-agent calls already went through `GuardedLLM`, so they got the
exploratory deadline (budget − final reserve) and the watchdog: a sub-agent call could never overrun the case or eat
the final reserve. Two gaps were found and fixed: (1) a sub-agent that starts close to the exploratory deadline was cut
there, and the same turn's main step then hit `BudgetExceeded` → one turn lost and the case forced to its final answer
(reproduced on a fake clock: 8 → 7 completed steps); (2) the pre-review advocate runs after the turn's main call but
read the Policy's once-per-turn `degraded` flag, so it could still run after that call crossed the degrade threshold.
Now every sub-agent call is preceded by a live check: skip if the budget is degraded now, or if the exploratory time
left < `AGENT_SUBAGENT_TIME_FACTOR` (3) × the slowest of the last 3 main-call latencies. With the default degrade point
(0.6) and reserve (45 s) that check binds before degraded mode only when main calls take > (0.4 × budget − 45) / 3:
25 s at a 300 s budget, 65 s at 600 s. Under the conservative model a late main call is ~35–41 s, so at budgets
≤ 400 s the latency check (not the degrade fraction) is what stops sub-agents. The final reserve (45 s) covers one
conservative final call (≤ ~40 s for a 6k-token prompt), but not a length retry at 20 tok/s.

Budget implications (under the assumptions, not a recommendation until measured): at conservative speed even the
typical case (~15–18 calls) needs 10–12 min, so a per-case limit below that would push most cases into degraded mode
and turn sub-agents off by themselves; at moderate speed a typical case takes 3–4 min and sub-agents cost ≤ 30 s.

## Open issues (refreshed 2026-09-27)
- **Nothing since v5 is measured with an LLM.** v6 (KB hints, code-decided review), the 26-category protocols and
  `data/cases_aug` need a first run; prompts were written against Gemini and must be re-validated on gpt-oss-20b.
- Reviewer: under v5 it **never held** (0/80) and renamed with mixed effects. v6 moves the verdict into code and refuses
  ungrounded / location / cause-qualifier renamings — whether it now holds at a useful rate is untested.
- Safety false triggers (anaphylaxis → stroke checks, etc.) were fixed by rules after a per-case audit; the Safety score
  has not been re-measured. 37 of 88 checks are still `unverified` against the guideline text (34 primary, 17 secondary).
- Time: Gemma took 5–10 min per case. A per-case time budget now exists (`AGENT_CASE_TIME_BUDGET_S`), but gpt-oss-20b
  speed on our serving setup is unknown → set the budget from the first measured run.
- Single runs of n = 40–50 have large run-to-run variance → repeat runs or the `full` profile before drawing conclusions.
- Weak at subtype discrimination (bipolar I/II, vascular stenosis vs. underlying disease).
- Augmented case entries (`data/cases_aug`) are LLM-written and not clinician-reviewed; 2,278 soft quality issues
  (mostly keyword-key collisions for the keyword simulator) remain as a review list.
- KB ranking is a hint, not evidence: held-out top-10 is 0.269. `kb_hints` does not pass sex/age to `candidates()` yet.
- Only our own environment exists (case files + simulated patient); behaviour against another patient environment or
  diagnosis format is untested.

### 2026-09-28 · specialty routing for runtime consults (`knowledge/specialty.py`, code only, no LLM, not wired)
Offline only (`python eval/offline/eval_specialty.py --results <dir with run_*.json>`; metrics in
`data/labels/specialty_gold_v1_metrics.json`). No prompt change, so no Accuracy / Efficiency / Safety row.
- `specialty_of` on the hand-labelled gold set (116 names, ko/en): first pass strict 94.8% / lenient 97.4%; after fixes
  read off those errors 97.4% / 100% (optimistic: same set).
- Gold diagnoses of `data/cases_aug` (267): 36.3% fall outside the six specialties (heme_onc 25, renal_uro 13, psych 11,
  derm 11, endo_metab 10, msk_ortho 9, ent_eye 5, other 13).
- Replay of per-turn DDx snapshots from 224 Gemini/Gemma trajectories: `route()` reaches share ≥ 0.6 after ≥ 3 turns in
  67.4% (mean turn 3.1) and names the gold diagnosis's specialty in 86.1% of those (93.5% when the gold is inside the six);
  on the 16 fired cases whose final answer was wrong, 11 were routed to the gold specialty.
- Next: wire a consult behind `MIN_TURNS`/`MIN_SHARE`, measure with gpt-oss-20b; consider adding `heme_onc` and
  `renal_uro` (together 14% of cases_aug gold diagnoses).

### 2026-09-28 · consult content: `heme_onc` and `renal_uro` specialties (`agent/subagents/consult.py`, no LLM, not wired)
Content only; the consult is not wired into the loop, so no Accuracy / Efficiency / Safety row yet.
- Two new `SpecialtySpec`s (covering ~14% of `cases_aug` gold diagnoses that fell outside the six). `fatigue` and
  `neck_mass` protocols now owned by `heme_onc`; `kdigo_aki_2012` moved to `renal_uro` only.
- Consult size: turn 0 950–1,104 tokens, worst case ≤ 2,929 tokens (o200k_harmony).
- Next: after routing emits these ids, measure consult on/off on gpt-oss-20b, focusing on heme/renal gold cases
  (Accuracy) and whether suggested tests stay within the discriminating set (Efficiency).

### 2026-09-28 · specialty routing: eight ids (`heme_onc`, `renal_uro` added; code only, no LLM)
Offline (`eval/offline/eval_specialty.py`; metrics `data/labels/specialty_gold_v2_metrics.json`). No prompt change, so
no Accuracy / Efficiency / Safety row.
- New gold `specialty_gold_v2.jsonl` (48 new names + 14 corrected v1 copies): first pass strict 44/48 (91.7%) on the new
  names, 112/116 on v1-corrected; after fixes 48/48 and 113/116 (optimistic: same set).
- cases_aug gold diagnoses outside the ids: 97/267 (36.3%) → 59/267 (22.1%) (heme_onc 24, renal_uro 14).
- Trajectory replay (224): consult fires 67.4% → 80.4%; routed = gold specialty 86.1% (130/151, old module vs eight-id
  gold) → 88.9% (160/180). The DKA case still routes to gi_liver: its DDx never contained DKA (endocrine not an id).
- Needs consult specs for the two ids in `agent/subagents/consult.py` before the routing can use them.

### 2026-09-29 · ten ids: `endo_metab` and `psych` (routing + consult content; code only, no LLM, not wired)
Offline (`eval/offline/eval_specialty.py`; metrics `data/labels/specialty_gold_v3_metrics.json`). No main prompt change,
so no Accuracy / Efficiency / Safety row.
- New gold `specialty_gold_v3.jsonl` (43 new names: 20 endo_metab, 17 psych, 6 boundary controls; + 11 corrected v1/v2
  copies), labelled before any ten-id output. First pass: new names strict 42/43 (97.7%), all 207 names 203/207; after one
  keyword fix (점액수종) 43/43 and 204/207 (optimistic: same set). Eight-id module on the same gold: 4/43 and 156/207.
- cases_aug gold diagnoses outside the ids: 59/267 (22.1%) → 38/267 (14.2%) (endo_metab 8, psych 11; derm 12, other 12,
  msk_ortho 9, ent_eye 5 remain).
- Trajectory replay (224): consult fires 80.4% → 88.4% (180 → 198); routed = gold specialty 87.8% (158/180, eight-id
  module vs ten-id gold) → 89.4% (177/198). New fires: endo_metab 10, psych 6. The DKA case still routes to gi_liver:
  its turn-3 DDx never contains DKA (routing cannot add a candidate; the gi_liver and endo_metab specs both name the
  DKA-as-abdominal-pain pitfall).
- Consult: `endo_metab` / `psych` specs appended in `consult.py` (render ≤ 643 chars; turn 0 930–1,057 tokens, worst
  case ≤ 2,965 tokens, o200k_harmony). `bipolar_dsm5tr` moved from neuro to psych. 15 new citations (PubMed-checked).
- Next: wire / measure on gpt-oss-20b with the endocrine and psychiatric gold cases (Accuracy) and check that the
  psych consult asks organic-cause tests before settling on a psychiatric diagnosis (Safety).

### 2026-09-29 · trigger calibration: anchoring check, consult, advocate (offline replay, no LLM)

> **Re-measured after endo_metab/psych (10 specialties, commit 96f3712 onward):** consult old rule 84.4% → chosen rule (turn ≥ 5, share ≥ 0.6) 40.2%, wrong 16/26, lift 1.53, out-of-sample 1.10 / 1.08. The consult numbers in the table below were measured at 1853b06 (8 specialties). Conclusion unchanged: turn 5 is a cost decision, not evidence of selecting wrong cases. README 9.3 uses the re-measured values.
`python eval/offline/eval_triggers.py [--results DIR]` (repo-relative; `-v` lists the rules picked per fold). No prompt
change and no LLM run, so no Accuracy / Efficiency / Safety row: these are trigger statistics, not scores.
**Calibrated on Gemini/Gemma trajectories only (gemini-3.5-flash-lite 160, gemma-4-26b-a4b-it 48, gemini-3.6-flash 16):
re-run the script on the first gpt-oss-20b runs and re-check every threshold below.**
- Data: 224 non-dummy trajectories of 9 runs (2026-09-25/26), 1,214 decision points. "Ends wrong" = judged accuracy < 1:
  26 trajectories but only **12 distinct cases** (56 distinct cases in all; `cqa_328` alone is wrong 5 times). Labels
  carry judge / name-matching noise. Treat differences under ~5 wrong trajectories as ties.
- Replay = what the runtime sees before step j+1: `turns[:j]`, the DDx ledger updated with the snapshots of actions
  1..j (no look-ahead to the snapshot written after response j, which `eval_anchoring.py` / `eval_specialty.py` use; hence
  57.6% / 75.0% here vs. their 45% / 80.4%), findings reported before that step. Pre-review point = state at the final
  DIAGNOSE with the final diagnosis as proposal (the runtime reviews the first DIAGNOSE that reaches review; close enough
  because these runs rarely held). The consult's stuck branch (model confidence) is **not** replayable: the result files
  do not keep the model's confidence. The code confidence (`confidence.DEFAULT_PARAMS`) was itself fitted on these runs,
  so every confidence-based rule below is optimistic even out of sample.
- Search: variants = "first decision point j ≥ t0 where all atoms hold" (anchoring, consult) or "atoms hold at the
  pre-review point" (advocate); atoms = thresholds on code confidence, top-1 p, top-1 − top-2 margin, grounded support /
  against counts, confirmatory result, unresolved can't-miss count, number of live candidates, routed specialty share,
  candidates within the routed specialty, the anchoring check's reasons. Three nested families: *threshold* (the current
  rule with its own numbers moved), *gated* (current rule ∧ one atom), *full* (any ≤ 2 atoms). Selection on the training
  part: most wrong trajectories caught with fire rate ≤ target (anchoring 30%, consult 35%, advocate 40%), then precision.
  Out of sample: 5-fold CV grouped by case id ×20 fold assignments, and leave-one-run-out (training also drops the
  held-out run's cases, so for the four flash-lite runs it is close to leave-one-model-out).

Lift = precision / base rate (26/224 = 11.6%); 1.0 = no better than random.

| Trigger | Rule | Fire | On wrong | On right | Lift | Median turn | Lift, CV by case | Lift, leave-one-run-out |
|---|---|---|---|---|---|---|---|---|
| Anchoring | before: `anchoring_check` from turn 3 | 57.6% | 46.2% (12/26) | 59.1% | 0.80 | 3 | (no fitting) | |
| | **after: same check from turn 5** | 26.8% | 34.6% (9/26) | 25.8% | 1.29 | 5 | 1.27 (threshold family) | 0.89 |
| | not adopted, gated: turn ≥ 3 ∧ margin ≤ 0.1 ∧ check | 14.3% | 34.6% | 11.6% | 2.42 | 4 | 1.47 | 0.33 |
| | not adopted, full: turn ≥ 4 ∧ top-1 p ≤ 0.6 ∧ code conf < 0.4 | 28.6% | 57.7% | 24.7% | 2.02 | 4 | 1.72 | 1.12 |
| Consult (routed) | before: share ≥ 0.6 from turn 3 | 75.0% | 73.1% (19/26) | 75.3% | 0.97 | 3 | (no fitting) | |
| | **after: share ≥ 0.6 from turn 5** | 36.2% | 42.3% (11/26) | 35.4% | 1.17 | 5 | 0.71 (threshold family) | 0.36 |
| | not adopted, gated: turn ≥ 4 ∧ share ≥ 0.6 ∧ margin ≤ 0.2 | 30.8% | 46.2% | 28.8% | 1.50 | 4 | 1.13 | 0.39 |
| | not adopted, full: turn ≥ 4 ∧ top-1 p ≤ 0.7 ∧ code conf < 0.4 | 29.5% | 57.7% | 25.8% | 1.96 | 4 | 1.39 | 0.34 |
| Advocate | before: anchoring moment (from turn 3) or pre-review code conf < 0.65 | 75.0% | 88.5% (23/26) | 73.2% | 1.18 | 3 | (no fitting) | |
| | **after: pre-review code conf < 0.4 only** | 33.0% | 53.8% (14/26) | 30.3% | 1.63 | 4 | 1.54 (threshold family) | 1.64 |
| | not adopted, gated: pre-review conf < 0.65 ∧ no unresolved can't-miss | 36.2% | 73.1% | 31.3% | 2.02 | 4 | 1.70 | 2.78 |

Reading:
- The anchoring check and the routed consult, as designed, fire *more* often on trajectories that end right (confident,
  focused DDx) than on those that end wrong. No variant of either transfers across runs (leave-one-run-out lift ≤ 1.1).
  In CV the gated / full searches catch −1.9 / +4.8 (anchoring) and +3.2 / +5.7 (consult) more of the 26 wrong
  trajectories than the threshold search, i.e. about 1–2 distinct cases, and the full rules would change what the
  trigger means (an "uncertain" detector, not an anchoring or specialty signal). So both keep their rule and only
  start later (turn 5), halving the fire rate; for the consult no
  threshold separates wrong cases out of sample (CV 0.71), the choice among thresholds is about cost, and turn 5 was
  preferred over the in-sample pick (turn 4, share ≥ 0.9; 31.2%, lift 0.98) because it changes one number and matches
  the anchoring start. The consult's stuck branch is unchanged and adds fires the replay cannot count.
- The advocate is the one trigger with a transferable signal: low code confidence at the pre-review point (lift 1.5–2.8
  in every split). The anchoring moment (its other trigger) now defaults off (`AGENT_ADVOCATE_ON_ANCHORING=0`); the
  threshold goes 0.65 → 0.4. The gated rule ("and no unresolved can't-miss") scored higher but its CV gain is < 1 wrong
  trajectory and the can't-miss count interacts with the danger gate, which changes with the model: left as a candidate.
- Fire rates depend strongly on the doctor model (anchoring from turn 5: flash 6%, flash-lite 19%, Gemma 58%; the Gemma
  lift is 0.57), which is the main reason to redo this on gpt-oss-20b.
- New defaults (env-overridable, `config.AgentConfig`): `AGENT_ANCHORING_MIN_TURNS` 5 (was 3, `anchoring.MIN_TURNS`),
  `AGENT_CONSULT_MIN_TURNS` 5 (was 3), `AGENT_CONSULT_MIN_SHARE` 0.6, `AGENT_CONSULT_LOW_CONF` / `_TURNS` / `_AFTER`
  0.5 / 3 / 6, `AGENT_ADVOCATE_CONF_BELOW` 0.4 (was 0.65), `AGENT_ADVOCATE_ON_ANCHORING` 0 (was always on). The previous
  behaviour is `AGENT_ANCHORING_MIN_TURNS=3 AGENT_CONSULT_MIN_TURNS=3 AGENT_ADVOCATE_CONF_BELOW=0.65
  AGENT_ADVOCATE_ON_ANCHORING=1`.
- Expected sub-agent calls per case on trajectories like these: consult ≈ 0.36 (+ stuck branch), advocate ≈ 0.33 (was
  ≈ 0.75 + 0.75). What a trigger *does* once fired (does the hint fix wrong cases, does it break right ones) is not
  measurable offline: run the `v6-no-subagents` ablation on gpt-oss-20b.

### 2026-09-29 · clinical claim verification pass + pneumothorax gate fix (content/safety code, no LLM run)
No prompt-loop change is wired yet for the consult content, and no case run was made, so no Accuracy / Efficiency /
Safety row; expected effect is on **Safety** (fewer false "confirmed" dangers, corrected obstetric thresholds).
- `agent/subagents/consult.py`: every reviewer-knowledge claim checked against PubMed-verified sources (42 new, in
  `consult_sources.py`; abstract unless the note says full text). Reworded where the source says less or differently:
  preeclampsia now "BP ≥ 140/90 twice ≥ 4 h apart + proteinuria or a severe feature" (ACOG PB 222 full text; the old
  line read BP alone as preeclampsia) and the "postpartum 6 weeks" limit dropped (not in PB 222; thrombosis risk to
  12 weeks, Kamel 2014); fetal shielding, "inferior MI", intussusception "leg drawing" triad, SVC "orthopnoea",
  ultrasound-first for pediatric appendicitis removed or reworded; added "normal Doppler flow does not exclude ovarian
  torsion" (ACOG CO 783). 7 of 8 specs now "secondary"; `resp_id` stays "unverified" (epiglottitis exam caution,
  chorioamnionitis: no citable abstract found).
- `safety/protocols.py`: 11 unverified checks re-read → 6 primary, 5 secondary (unverified 37 → 26).
- `safety/danger_gate.py`: a pneumothorax on imaging (`IMG:cxr_ptx`) no longer confirms 긴장성 기흉 unless the reading /
  report affirms tension (mediastinal or tracheal shift, 긴장성, tension; negation-aware incl. English) or SBP < 90;
  otherwise the danger is raised (unresolved, breath-sound check) and the finding is still shown by the critical alert.
- Verify on gpt-oss-20b (eval-simulator): cases with pneumothorax, pregnancy-related hypertension/postpartum headache,
  and the consult on/off comparison once wired; watch Safety (missed can't-miss) and whether the gate adds turns.

### 2026-09-29 · diagnosis-name normalisation audit (`knowledge/kb.py`, `kb_curated.py`; code only, no LLM)
No prompt change and no case run, so no Accuracy / Efficiency / Safety row. Expected effect: **Safety / routing**
(can't-miss names and consult routing no longer land in wrong chapters). Details: `docs/data-sources.md` 7.12.
- Audit (`scripts/audit_normalize.py`, 1,907 names: cases_aug gold + aliases, specialty gold v1-v3, danger_gate /
  protocols / consult can't-miss names): flagged names 287 → 175; every remaining flag hand-reviewed into
  `data/labels/normalize_audit_allow.json` (190 entries); `tests/test_normalize_audit.py` (56 tests) fails on new flags.
- Wrong mappings fixed, e.g. 흉강 비장 이식증 F50.8 (pica) → none; Opioid overdose F11.1 → T40.2; 혈관염 I80 → I77.6;
  두개내 출혈 S06.8 → I62.9; 폐동맥 색전증 N28.0 → I26; acs Q04.0 → I24.9; SJS Sjögren → L51.1; 저칼슘혈증 →
  고칼슘혈증 profile → E83.5; 헤노흐-쇤라인 자반증 D69.2 → D69.0; 라이터 증후군 Reye G93.7 → M02.3; 파르보바이러스
  B19 감염증 B19 (viral hepatitis) → none; 급성 용혈 / 용혈 HELLP → none; 심장 혈관육종 I51.9 → none; ambiguous bare
  abbreviations (HD, PD, MS, AS, AD, CD, PV, PG, CDI, ET, CRS, ASD, PM) → none.
- Names whose mapping changed: 93 unique inputs (54 lost a resolution, 46 changed code, 12 gained). Of the 45 unique
  inputs that lost one, ~36 were wrong or ambiguous before; ~9 correct fuzzy matches were lost to the stricter fuzzy
  guards (e.g. 원발성 담즙성 담도염 → K74.3, 선천성 풍진 감염 → P35.0, Pancreatic head cancer → C25, COPD 급성 악화 → J44).
- `scripts/eval_kb.py --no-write` before → after: dev top-1 0.477 = 0.477, top-3 0.622 → 0.631, top-10 0.730 = 0.730,
  top-50 0.784 =, MRR 0.567 → 0.568, coverage 0.964 → 0.955; held-out top-1 0.147 =, top-3 0.211 → 0.205, top-10
  0.295 → 0.282, top-50 0.397 → 0.378, MRR 0.197 → 0.194, coverage 0.865 → 0.840. Every rank change comes from the gold
  id set (gold = union of the case's names resolved by `_resolve`): spurious golds removed (da_631 "heart disease" at
  rank 10, da_858 "pica" at 37), ac_20 CMV retinitis lost its "cytomegaloviral disease" gold (rank 3 → none; a real
  loss), synthetic_008 bacterial meningitis 7 → 22 (aliases now resolve to the specific meningitis profiles instead of
  generic "meningitis"), cqa_157 12 → 3. The candidate ranking itself is unchanged.
  normalize: dev hit 0.833 → 0.814, code 0.755 → 0.741, alias code agreement 0.691 → 0.745; held-out hit 0.602 →
  0.572, code 0.536 → 0.509, agreement 0.685 → 0.785 (fewer, more consistent codes).
- `eval/offline/eval_specialty.py`: gold strict 204/207, lenient 207/207 (unchanged); routing replay 177/198 routed to
  the gold specialty (unchanged). cases_aug distribution: 3 cases moved (뼈거대세포종 rheum_immune (M31.5 giant cell
  arteritis) → heme_onc; 흉강 비장증 psych → resp_id by keyword; LED 광 황반병증 ent_eye → other).
- Latency: normalize_diagnosis mean 0.96 → 0.99 ms (fuzzy on), specialty_of 0.087 → 0.101 ms; `data/kb` unchanged.

### 2026-09-29 · second verification pass (protocols, consult, rules, criteria; no LLM run)
No case run, so no Accuracy / Efficiency / Safety row; expected effect is on **Safety** (checks now match the source
text; HHS / Jones decisions follow the published cut-offs) and marginally **Efficiency** (AAA imaging no longer asked
of young flank pain without an aortopathy clue — unchanged from before, now justified).
- Levels (primary / secondary / unverified), before -> after: `safety/protocols.py` 40/22/26 -> 61/19/8;
  `agent/subagents/consult.py` specs 0/7/3 -> 0/10/0; `knowledge/clinical_rules.py` 15/8/1 -> 16/8/0;
  `knowledge/diagnostic_criteria.py` 8/3/4 -> 11/2/2. What was read is in each note; new sources in docs/licenses.md.
- Full texts read: ESC 2023 endocarditis (≥ 3 blood-culture sets at 30-min intervals before antibiotics), ESC 2019 PE
  recommendation table 4.11, ESC 2018 syncope (supine + standing BP, ECG, cardiac-syncope features), ACEP 2019
  headache (Level B answers), ACC/AHA 2022 aorta (AAA symptoms, CT in stable rupture), WAO 2020, SSC 2021, KDIGO 2021,
  ACCP 2006 cough (TB-prevalence rule: 2-3 weeks, CXR + AFB), ACR AC tables (jaundice, hearing loss, infant vomiting,
  acute pelvic pain), CSRS (CMAJ), RA 2010 (ARD), Jones 2015, ADA 2024 Fig. 2B.
- Behaviour changes: `aaa_imaging` min_age=50 -> predicate "aaa" (age ≥ 50 or unstated, or any age with Marfan / EDS /
  Loeys-Dietz / known or familial aneurysm). Decision on the ACC/AHA "24% of ruptured AAA < 65 y": keeps the floor at
  50 because that figure concerns the 50-64 band, and in the population-based Oxford Vascular Study 0/103 acute AAA
  were aged 45-54 (Howard 2015). HHS: HCO3 ≥ 15 (was ≥ 18) and effective osmolality > 300 accepted. Jones: PR
  prolongation not counted as minor when carditis is a major criterion. Renamed checks: IE blood cultures "30분 간격
  3세트 이상", jaundice imaging "복부 초음파(또는 조영증강 복부 CT·MRCP)", retrocochlear "머리·내이도 MRI" (contrast not
  required), infant bilious vomiting "즉시 상부위장관 조영술(생후 2일 이내는 복부 X선 먼저)". Dyspnea CXR now cites
  AHA/ACC/HFSA 2022; the "tension PTX is a clinical diagnosis" note was not in BTS 2023 and was removed.
- Consult wording: epiglottitis signs "suggest" (not "="), no airway instrumentation / agitation; euglycaemic DKA in
  pregnancy; thyroid-storm features instead of heat/cold intolerance; pheochromocytoma triad "only in some";
  dropped lithium, purple striae, goitre, bitemporal hemianopia, bullying, separate interview (no accessible source).
- Still below primary and why: stroke NIHSS / chest-pain CXR / three vitals checks / abdominal exam / chronic
  weakness x2 (no readable recommendation), ADD-RS, Canadian CT Head, PECARN x2, GBS, sPESI, McIsaac (originals
  paywalled), Light 1972 and DSM-5-TR (not accessible), McDonald 2017, Kawasaki 2017 CRP/ESR entry threshold.
- Verify on gpt-oss-20b (eval-simulator): abdominal/flank-pain cases < 50 y with and without Marfan/EDS clues (AAA check
  fires only with the clue), febrile murmur cases (IE blood cultures), infant bilious vomiting, HHS cases with HCO3
  15-17; compare Safety (missed must-checks) and turns with the previous commit.

### 2026-09-30 · result interpreter: kb_tests lab polarity trusted (`agent/result_interpreter.py`; code only, no LLM)
No case run, so no Accuracy / Efficiency / Safety row; expected effect is on **Safety** (a lab value above its printed
Korean "N 미만" range is no longer read as normal) and **Accuracy** (no false "ESR > 50" / "AST > 1000" readings).
- Removed `_ref_check` (re-read value vs printed range via `nlp.findings.ref_direction` and overrode kb_tests'
  polarity, ignoring `Finding.cutoff`) and `_KB_FALSE_TAIL` (no longer changed any reading after kb_tests 89181ce).
  The value / unit is now taken with kb_tests' own number reader for display only (none for ranges "0-5", titres
  "1:160", grades "3+", or a number glued to another name "C3"); direction follows the finding's side ("low" for
  `*_low`, was "high"). One bridge kept, using kb_tests' own word-range rule: a direction-word parenthesis without a
  reference keyword ("CEA 6.5 ng/mL (경미한 상승)") makes a non-cut-off finding present.
- Sweep of 7,586 distinct result texts (data/cases_*, data/sample_cases, non-dummy eval/results runs), before -> after:
  11 texts change (concept, polarity), all `LAB:esr_very_high` present -> absent for ESR 22-48 with a printed range
  (fixes; `LAB:esr_high` stays present); 0 regressions. 1,102 value/unit-only display changes, 0 critical-flag changes.
- Gold (`eval/offline/eval_result_interp.py`): dev P/R 1.00/1.00 (116), fresh 1.00/1.00 (66), unchanged.
- Found, not fixed here (other owners): `nlp.findings.ref_direction` reads "(정상 500 미만)" / "(정상 12 이상)" as
  "normal" (only `<`/`>`/ranges parse), so lexicon labs (ESR, CRP, Hb, AST...) with that phrasing read absent (0
  occurrences in the current case texts); kb_tests reads a bare-number range "(정상치 500)" / "(정상 상한 60)" as
  says="normal" -> absent regardless of the value, and compares a unitless value with a unit-bearing range as printed
  ("D-dimer 1.2 (정상 <500 ng/mL)" -> absent); kb_tests false positives exposed by the value display:
  `anion_gap_high` from "vWF:Ag 95%", `tsat_high` in a haemodynamics report.

### 2026-09-30 · one reference-range reader + look-alike analytes (`knowledge/refrange.py`; code only, no LLM)
No case run, so no Accuracy / Efficiency / Safety row. Expected effect: **Safety** (abnormal labs with Korean /
bare-number reference ranges no longer read normal) and **Accuracy** (no anion gap from "vWF:Ag", no transferrin
saturation from a cath report, no free air from a perforated mitral leaflet). Fixes the "found, not fixed" items of
the entry below. Rules are about how ranges and short names are written, not about cases (docs/nlp.md, "Fixes
2026-09-30").
- Sweep of 7,637 distinct result texts (data/cases_*, data/sample_cases, non-dummy eval/results runs) through
  `nlp.findings.parse`, `kb_tests.detect`, `result_interpreter.interpret`, before -> after: 119 texts change. Reading
  (concept / polarity) changes: kb 5, parse 7, interpreter 11 — all fixes, 0 regressions: anion_gap_high from
  "vWF:Ag 95%" gone (3 readers); CEA 6.5 "(경미한 상승)" absent -> present (kb, parse; the interpreter already had it via
  the bridge, now removed); tsat_high from "폐동맥 포화도 66%" gone (3); a spurious glucose_very_high "normal" read from the
  word 혈당 inside "(혈당 대비 감소)" after a CSF / synovial glucose gone (3 x 2); "CRP 11.7 mg/dL (1394 nmol/L)" and
  "Cr 1.86 mg/dL (164 μmol/L)" now crp_high / cr_high present (the "l" of the SI unit was read as a low flag; parse,
  interpreter); IMG:free_air from "승모판 천공" (TEE) now an unmapped abnormal item, and "고막 천공 없음" no longer yields
  IMG:free_air absent (4 otoscopy texts). Display-only: 86 kb "normal" readings next to "(정상)" are now value-based
  (the interpreter shows the value, direction "normal"); 29 parse / 6 interpreter directions of low-side kb findings
  "high" -> "low". Removing the interpreter's direction-word bridge changes 0 of 7,637 interpretations.
- Gold: findings gold (`scripts/label_findings.py metrics`) concept+polarity .945 (P .948 / R .942) before = after;
  result interpreter gold dev 1.00 / 1.00 (116), fresh 1.00 / 1.00 (66), unchanged. `eval_danger_gate.py`: one change,
  da_494 (endocarditis) no longer "confirms" bowel perforation from the TEE leaflet perforation (fix); gold blocked 12
  unchanged. `scripts/eval_kb.py` unchanged.
- Not changed: `nlp.parse` still maps the lexicon form "천공" to IMG:free_air in any organ (lexicon-level; only the
  result interpreter checks the organ); "AG" / "포화도" in a clause without their context are not read (kb_tests is
  called per clause from nlp).

### 2026-09-30 · danger gate over-firing (`safety/danger_gate.py`; offline, no LLM)
No case run, so no Accuracy / Efficiency / Safety row. Expected effect: **Efficiency** (fewer forced rule-out turns)
with **Safety** kept (no high-value true catch lost). Measured with `eval/offline/eval_danger_gate.py` on the 267
`data/cases_aug` cases (code result interpreter on): "full reveal" = every history/exam/test entry given as a turn,
then `gate(state, gold)`; "initial-only" = forced steps when the gold diagnosis is proposed with only the initial info
(upper bound, replayed through the case-file environment, ≤ 3 per case).
- Gold diagnosis blocked with everything revealed: 28 → 12 (without the interpreter 28 → 13). Forced steps: full
  reveal 33 → 15, initial-only 144 → 122. Raised from the initial info: acute heart failure 17 → 1 (gold 0), tension
  pneumothorax 17 → 13 (gold 2, true catches 1 → 1); all other raise counts unchanged. True catches (raised danger =
  gold) unchanged for every danger; mesenteric ischaemia ac_43 went from wrongly *ruled out* to confirmed (reader fix).
- Changes: acute heart failure live from dyspnoea only with a clue (Wang 2005 JAMA; negation read by both readers);
  echo normal rules HF out. Tension PTX not live for dyspnoea of ≥ 2 weeks alone (SpO2 < 94, SBP < 90, trauma, stated
  pneumothorax still make it live). Sepsis ruled out by qSOFA 0 + all routine SOFA systems normal (Sepsis-3) as an
  alternative to lactate. Ectopic not live after stated menopause / hysterectomy. Ischaemic stroke moot once an
  intracranial haemorrhage is confirmed. Reader fixes: a TEST response headed by a result label counts as that result
  ("초음파" → "질식 초음파: ..."); explicit normal statements are negated only by a negation right after them ("호흡음
  명료함, 수포음 없음" was read as not normal); "이상 호흡음 없음" is not absent breath sounds; 명료/청명 breath sounds are
  normal; bilateral leg oedema and "한쪽 다리" weakness are not Wells DVT signs; "케토산증" spelling maps to DKA.
- Remaining 12 blocks, kept on purpose: sepsis with qSOFA ≥ 1, low platelets / high creatinine or a missing SOFA lab
  (5; one lactate request each), ACS with pneumothorax < 3 h from onset (2; serial troponin), meningitis with altered
  mental status or fever + headache (2; LP), PE with Wells > 4 from leg oedema of unstated side (1), AAA in a 62-year-old
  with flank pain (1), heart failure with crackles (1). Each costs one turn when the test is not in the case.
- Overfitting risk: the same 267 cases were used to design the gate and to measure the fix. Changes were chosen as
  general rules (clinical cues, reader bugs), no case-specific patterns. A true acute heart failure presenting as bare
  dyspnoea is now gated only after a clue appears or the model flags it 위험 in the ledger.
- Verify on gpt-oss-20b (eval-simulator): dyspnoea cases (HF and non-HF), febrile infection cases (turn count, lactate
  requests), early-pregnancy / postmenopausal abdominal pain; compare turns and Safety with the previous commit.
