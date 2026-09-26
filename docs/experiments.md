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

## Open issues
- The reviewer **never holds** (0/80) → the conservative check isn't working. It only revises names, with mixed effects (better subtypes vs. over-specific names like "상행결장암"/"밀가루 의존성")
- Safety checks falsely triggered by protocol category detection (anaphylaxis→stroke checks, hypersensitivity pneumonitis→sepsis cultures) → category detection needs refinement
- Single runs of n=40 have large run-to-run variance → repeat runs or more cases needed before drawing conclusions
- Weak at subtype discrimination (bipolar I/II, vascular stenosis vs. underlying disease)
- Gemma is slow at 5–10 min per case → competition time limit is a risk (reasoning-length tuning needed)
- Converted cases lack test results, so "result not provided" responses are frequent → re-check once the official guide shows how results are provided
- About half the safety checklist has not been checked against the original guideline text (see `verification` field per item)
