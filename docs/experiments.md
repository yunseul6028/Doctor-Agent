# Experiment Log

- Safety = share of the guideline-based checklist (`safety/protocols.py`) completed. Chief-complaint categories are detected from the initial info only (since 2026-09-25; earlier results were rescored).
- Patient/judge LLM: gemini-3.6-flash. The doctor model is listed per row.
- Local scores only reflect **relative change**. They are not a prediction of the official score.

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
The first rows for it should come from `eval/experiment.py` on the competition model (see below).

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
| Runtime | gpt-oss harmony handling, length retry, optional structured output, per-case time budget with degrade/forced final answer, never-crash submission mode, prompt cap, incremental `run.py` output | `tests/test_runtime.py` (28 tests) |
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

## How to run on competition API day

One command runs a standard profile, estimates cost first, compares conditions, and rebuilds the viewer + share page
(`eval/experiment.py`; profiles in `eval/experiment_profiles.json`; fixed case lists in `eval/case_lists/`).

1. Add the competition endpoint to `.env` (keys stay in `.env`; the script never prints them):
   `COMPETITION_LLM_BASE_URL=…`, `COMPETITION_LLM_API_KEY=…` (optional `COMPETITION_LLM_MODEL`, default `openai/gpt-oss-20b`).
   The existing `DOCTOR_LLM_*` lines also work. `--doctor-endpoint gemini` uses `GEMINI_LLM_*` or the shared `LLM_*`;
   `local` uses `LOCAL_LLM_*` or Ollama `gpt-oss:20b`.
2. Free wiring check: `python eval/experiment.py --profile smoke --doctor-endpoint dummy --no-view`
3. Plan + cost only: `python eval/experiment.py --profile dev --doctor-endpoint competition --estimate-only`
   (add `--price-in/--price-out` in KRW per 1M tokens, or `EXPERIMENT_PRICE_{IN,OUT}_PER_M`, to get a KRW figure).
4. Smoke on the real model (5 cases, ~50 doctor calls): `python eval/experiment.py --profile smoke --doctor-endpoint competition`
   → check `usage:` lines (real tokens/call) before anything bigger; later estimates use them automatically.
5. Dev comparison (50 cases × v6 / v6-no-kb, ~900 calls, needs `--yes`):
   `python eval/experiment.py --profile dev --doctor-endpoint competition --yes --log --note "first gpt-oss run"`
   Optional: `--conditions v6,v5-baseline` (v5 runs in a temporary git worktree at `c3ddecd`, rescored with the current
   scorer), `--env AGENT_CASE_TIME_BUDGET_S=240`, `--env DOCTOR_LLM_STRUCTURED_OUTPUT=guided_json`, `--compare-with FILE`.
6. `full` (267 cases) only when a dev result justifies it.

Defaults are the cheapest mode: keyword patient + no judge, so only the doctor spends competition credits
(`--patient llm --judge llm` opt in; those use `PATIENT_LLM_*`/`JUDGE_LLM_*`, i.e. Gemini). Guard: `--yes` is required
above `--max-calls` (300 doctor calls) or `--max-cost` (5,000 KRW, when prices are given). A billing error (402 /
credits depleted) stops the running batch and skips the remaining conditions (exit code 3).
Cost estimate = cases × doctor calls/case (mean of past result files, else 12) × tokens/call, all × margin 1.3.
Tokens/call: recorded usage of the same doctor model; else, for gpt-oss, the **measured** prompt tokens (see "Prompt
token budget" below; mean step prompt over a case of the past mean length) + an assumed 1,000 / 1,800 / 2,048 output
tokens at effort low / medium / high; else usage of any other model; else 3,000 in / 1,000 out. Doctor token usage is recorded per case
(`usage` in each row) and per run by wrapping the SDK client in `eval/usage.py` (src untouched).
`--log` appends one table row per condition above and a comparison section (overall, per set, flips, n/a rate) under
"Auto-logged experiment runs" at the end. The share page path is printed (`eval/results/share_<time>.html` or `--share`).

### LLM record/replay cache (stretching the credits)
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
  sequence), i.e. unchanged conditions and the doctor-cached replays. The doctor (the competition credits) hits when a
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

## Open issues (refreshed 2026-09-27)
- **Nothing since v5 is measured with an LLM.** v6 (KB hints, code-decided review), the 26-category protocols and
  `data/cases_aug` need a first run; prompts were written against Gemini and must be re-validated on gpt-oss-20b.
- Reviewer: under v5 it **never held** (0/80) and renamed with mixed effects. v6 moves the verdict into code and refuses
  ungrounded / location / cause-qualifier renamings — whether it now holds at a useful rate is untested.
- Safety false triggers (anaphylaxis → stroke checks, etc.) were fixed by rules after a per-case audit; the Safety score
  has not been re-measured. 37 of 88 checks are still `unverified` against the guideline text (34 primary, 17 secondary).
- Time: Gemma took 5–10 min per case. A per-case time budget now exists (`AGENT_CASE_TIME_BUDGET_S`), but the official
  limit and gpt-oss speed on the competition server are unknown → set the budget on API day.
- Single runs of n = 40–50 have large run-to-run variance → repeat runs or the `full` profile before drawing conclusions.
- Weak at subtype discrimination (bipolar I/II, vascular stenosis vs. underlying disease).
- Augmented case entries (`data/cases_aug`) are LLM-written and not clinician-reviewed; 2,278 soft quality issues
  (mostly keyword-key collisions for the keyword simulator) remain as a review list.
- KB ranking is a hint, not evidence: held-out top-10 is 0.269. `kb_hints` does not pass sex/age to `candidates()` yet.
- The official environment (`env/official.py`), action/diagnosis format and how test results are provided are unknown.
