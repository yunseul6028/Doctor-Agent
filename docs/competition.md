# N.O.V.A. 2026 Competition Summary

Source: https://nova.snubhai.org (collected 2026-09-25; covers the overview, evaluation, rules, schedule, prizes, FAQ, and application pages)

## Overview
- Name: N.O.V.A. 2026 — Navigating Optimal clinical decisions via Virtual-patient Agents
- Hosts: Bundang Seoul National University Hospital Medical AI Center · elice
- Format: submit Doctor Agent code → it runs on the competition server against **private cases**
- Teams: individuals or teams of up to 4 (one team per person), 100 teams
- Prize money: 5M KRW total (Grand Prize 2.5M, Excellence Award 1.5M, Encouragement Award 0.5M × 2) + paper collaboration and internship opportunities

## Schedule
| Stage | Dates | Notes |
|---|---|---|
| Registration | 09.13 – 10.03 | Tally form (https://tally.so/r/J9QXY4) |
| Online opening ceremony | 10.06 10:00–12:00 | |
| Preliminary round | 10.12 – 10.22 | Code submission, automatic scoring |
| Final round | 10.27 – 11.10 | Finalist teams only, free choice of model (≤20GB VRAM), PPT required |
| Awards ceremony | 11.20 | Venue TBA |

**Not published yet (participant guide, before the preliminary round):** API usage, runtime environment, public sample cases, submission method, deadline times, detailed metrics and weights, final round evaluation method.

## Game rules
- Starts from the patient's basic information and initial symptoms. Full information is only available through the interface.
- Actions: **ASK** (history-taking), **EXAM** (physical exam), **TEST** (labs and imaging), **DIAGNOSE** (final diagnosis)
- **Up to 60 turns per case**. ASK/EXAM/TEST can be repeated.

## Evaluation (preliminary round = automatic score only)
- **Accuracy**: how correct the final diagnosis is
- **Efficiency**: whether the needed information was obtained efficiently (→ fewer unnecessary turns and tests)
- **Safety**: whether appropriate questions and tests were chosen (→ must not miss red flags or dangerous differentials)
- Detailed weights are TBA. The FAQ mentions a "GPT model used for scoring" → **likely includes LLM-judge-based scoring**

## Model and environment
- Preliminary round: fixed `openai/gpt-oss-20b` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, provided on the server
- The submission is inference code plus static assets (search indexes, etc.) only
- No auxiliary embedding model is provided → a small embedding model or index is allowed if it runs on CPU/RAM
- 50,000 KRW of API credit (gpt-oss-20b + the scoring GPT model), for competition preparation only, logs monitored

## Data and external resources
Allowed: sample cases for any purpose; self-labeling (including with external LLMs); external datasets with publication-compatible licenses; open-source libraries; post-processing
Forbidden: unclear licenses; re-identification; **external LLM/API calls from inference code**; obtaining or leaking evaluation data; learning from private cases or pseudo-labeling; cross-case information use

## Reproducibility and verification (finalist candidates)
- Code and instructions that reproduce the preliminary score, including label-generation code and prompts
- Source and license specification for every model, tool, and data source
- Team member information
