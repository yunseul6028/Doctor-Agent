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

Prompt `v6-kb-strict-review` (current: KB hints, code-decided review, evidence-gated renaming) has **no LLM run yet**.
The first rows for it should come from `eval/experiment.py` on the competition model (see below).

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
| Result interpreter (09-28, not wired) | `agent/result_interpreter.py`: code-first reading of EXAM/TEST result text (negation, hedges, comparison, sections, organ-aware report words, critical values, supportive kb_tests links); new prompt `RESULT_INTERPRETER_PROMPT` added to `prompts.py` but **not called** (PROMPT_VERSION unchanged) | gold (concept+polarity): fresh set P/R 0.879/0.879 before the fixes it prompted, dev + fresh 1.00 after (same author, optimistic); `tests/test_result_interpreter.py` |

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
Cost estimate = cases × doctor calls/case (mean of past result files, else 12) × tokens/call (from recorded usage of
the same model, else any model, else 3,000 in / 1,000 out), all × margin 1.3. Doctor token usage is recorded per case
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
