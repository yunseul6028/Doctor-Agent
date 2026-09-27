# Clinical-finding normalisation layer (`src/doctor_agent/nlp`)

One lexicon and one parser for every module that reads findings from text (patient answers in colloquial Korean,
exam reports, test results, the doctor model's own claims). Replaces five separate dictionaries and five separate
negation heuristics. Stdlib only, CPU only, no network, no LLM, no state between calls. **Not wired yet** — see the
migration plan below.

## Files

| Path | Role | Shipped |
|---|---|---|
| `src/doctor_agent/nlp/lexicon.py` | loads `data/lexicon/concepts.json` once at import (read-only); `scan()` finds mentions | yes |
| `src/doctor_agent/nlp/findings.py` | `parse()`, `match()`, `concepts_in()`, `affirmed()`, `denied()`; `assess_spans()` (polarity/subject of spans found by another vocabulary) | yes |
| `data/lexicon/kb_links.json` | concept → KB term ids with link kind (27 KB; `python scripts/eval_kb.py --build-links`) | yes |
| `data/lexicon/concepts.json` | built lexicon (≈360 KB, 614 concepts, 6.5k surface forms, 149 regexes) | yes (**add `data/lexicon` to `scripts/package.py` INCLUDE**) |
| `data/lexicon/seed.tsv` | hand-authored seed (long format `id<TAB>field<TAB>value`) | source only |
| `scripts/build_lexicon.py` | merges seed + existing tables → `concepts.json` (`--check` for staleness) | no |
| `scripts/label_findings.py` | gold-set pipeline (extract / dev / sample / show / freeze / metrics / errors) | no |
| `data/labels/findings_gold_*` | candidates, draft, hand review, corrections, gold v1, metrics | no |
| `tests/test_nlp_findings.py` | phenomena, match, load/parse speed, gold-set F1 floor | no |

## Concept schema

Ids are `CAT:name`: `SYM` symptoms, `SIGN` exam / vital-sign findings, `LAB` lab results, `IMG` imaging / endoscopy /
physiology, `ECG`, `HX` history & risk factors, `QUAL` pain quality / timing modifiers, `GRP` groups ("소화기 증상").

```json
{"id": "SYM:fever", "cat": "SYM", "ko": "발열", "en": "fever",
 "parents": ["GRP:systemic"], "flags": [], "kb": ["WD:Q38933", "..."], "protocol": ["category:fever", "protocols:_TRIG_FEVER"],
 "forms": [["열이 나", "lay", "seed+curated", 0], ["불덩이", "lay", "seed", 1], ["fever", "en", "seed+grounding", 0]],
 "re": [["열이.{0,6}(나|났|있|오르|올라)", "curated"]], "negre": []}
```

- form kinds: `lay` (구어체), `med` (Korean medical), `en`, `abbr`, `neg` (absence expression: "잘 먹어요" → no anorexia).
  The 4th element marks a figurative form that survives the simile rule ("몸이 불덩이 같아요").
- `kb`: KB term ids whose label/synonym equals a concept label or synonym (318 concepts linked), plus `TF:` ids for
  the 262 `kb_tests` result concepts. `protocol`: clinical_rules category and safety keyword-tuple names that
  mention the concept (`LEXICON.by_protocol("protocols:_TRIG_AMS")`).
- flags: `group` (an absent group makes its members absent in `match`), `cue` (emit only with an explicit cue:
  `HX:pmh`, `LAB:hcg_pos`), `fig`, `kb_tests`.
- provenance per entry: `seed`, `casefreq`, `curated` (kb_curated), `grounding`, `kb_tests`, `clinical_rules`,
  `protocols`, `danger_gate`, `preconditions`; KB links from `data/kb`. First owner of a surface form wins
  (49 conflicts, logged by `--show-conflicts`); ambiguous forms are dropped (`DROP_FORMS`, `CURATED_RE_SKIP`).

Size by category: SYM 155, SIGN 85, LAB 198 (17 own + 181 from kb_tests), IMG 80, ECG 9, HX 59, QUAL 17, GRP 11.

## Finding

`concept, label, span, polarity (present|absent|uncertain), subject (patient|family|other), temporality
(current|past|chronic|intermittent), onset, laterality, site, severity (mild|moderate|severe [+ NRS n/10]), value,
unit, direction (high|low|normal), hedged, hypothetical, confidence, source, start/end (offsets in normalize(text)),
clause, cue` (what decided the polarity; for debugging). Note: this is not `agent/ledger.Finding`.

## Rules (in order)

1. `normalize`: NFKC, lower case, `(-)`/`(+)`/`(1+)` → 음성/양성, thousands separators, "두 번" → "2번", 비정상 → 이상.
2. Sentences; sentences saying a result is unavailable are skipped.
3. Mentions: space-insensitive form index (2-syllable forms must start a word or follow a glued prefix such as 우측/잔;
   a match starting inside a word may not span a word boundary), regexes (gaps are lazy and never cross a comma or a
   clause boundary), block words (발작성, 방사선, 종양 표지 …). Longest match wins; on equal spans a form beats a regex
   and a concept beats its ancestor. QUAL mentions are resolved separately so "쥐어짜듯이 아파요" keeps both.
4. Clauses at connectives (~고, ~는데/~데, ~지만, ~으나, ~면서, ~며, ~어서, ~다가, ~니까; not ~고 있다/~고 나서),
   comma parts inside a clause. A part that is a bare noun-phrase list item (no cue, no predicate ending, no number)
   takes the predicate of the following parts → clause-final negation covers the whole list.
5. Polarity = first cue after the mention in its unit: negation (없/않/아니/아닌/안+verb/음성/정상/괜찮/부인/~기보다/
   no/denies…), affirmation (있/양성/관찰/청진/촉지/호소…), uncertainty (모르겠/글쎄/기억이 안 나…). Masked first:
   idioms that only look negative ("수 없", "이유 없이", "문제없이"); persistence counts as present ("가라앉지 않",
   "안 멈춰"); "X 외 … 없음" keeps X present; double negation ("없지는 않아요") is present; "있는 사람은 없어요" is
   absent. A negation inside a regex match negates it ("배는 안 아파"); one inside a form is part of the concept
   ("입맛이 없"). Absence forms give absent unless immediately negated ("잘 먹지 못해요"). "X할 때 …" presupposes X.
6. Hedge (~것 같아요, ~듯, 아마, 의심/시사 in reports) → `hedged=True`, confidence lowered, polarity kept
   (deliberate: "열이 나는 것 같아요" must still trigger fever protocols). Explicit don't-know → `uncertain`.
   Questions and conjectures (~일까 봐, ~아닌지, 혹시…?) → `hypothetical`, present becomes uncertain.
   Similes (X처럼/마냥/같이) drop X unless the form is figurative by design or X is a pain / quality concept.
7. Subject: latest person word *with a subject particle* before the mention (어머니가, 가족 중, 산모에게 → family;
   남편이, 같이 먹은 동료 → other; 저는, 제가, 아이가 → patient); comitative/dative (부모님과, 엄마한테) do not count.
   A case-file key whose first label is 가족력, or a question about family, sets the default.
8. Temporality from clause markers, carried forward in the sentence until 지금은/요즘: past (예전에, ~적이 있/없,
   N년 전에, 과거력, 끊었, …), intermittent (가끔, 반복, 때마다…), chronic (만성, N개월째, 복용 중, 평소…), else
   current; "N년 전에 진단받아 … 복용" is chronic; a trailing "지금은 괜찮아졌어요" clause makes earlier findings past.
   Onset text (3일 전부터, 갑자기, 어제…) is kept separately.
9. Measurements: 체온 (≥37.8 fever, 37.5–37.7 uncertain, <37.5 absent; °F converted; ≥39 high fever), HR (>100 / <60),
   RR (>20 / <10), BP (SBP <90 hypotension, 90–99 uncertain; ≥140/90 elevated), SpO2 (<92 / 92–94 uncertain / ≥95),
   17 core labs (WBC, Hb, PLT, Na, K, glucose, Cr, BUN, AST/ALT, bilirubin, albumin, CRP, ESR, HCO3, INR, eosinophils)
   with ×10³ / 만 scaling and reference ranges in parentheses; urinalysis items. An abnormal value names one side only.
   For exam/test/claim text `knowledge/kb_tests.detect()` adds its result concepts (reused, not ported).
10. Yes/no: with `context={"question": …}`, "네/아니요/모르겠어요" (or an elliptical answer's own cue) resolves the
    question's concepts the answer does not name itself; a negatively phrased question flips "네".
11. `match(claim, evidence)`: every claim concept needs evidence with the same polarity and subject, or a present
    descendant (for a present claim) / an absent ancestor other than generic pain (for an absent claim); numbers within
    rounding (or ×1000); no laterality clash; uncertain/hypothetical evidence never supports.

## Evaluation (no LLM anywhere)

Gold set: `data/labels/findings_gold_v1.jsonl`, 397 sentences (353 from data/cases_aug stratified by surface
phenomenon with a fixed seed, excluding the 403 dev sentences I looked at while writing rules + 44 self-written yes/no
answers), 640 gold findings, hand-verified item by item (`findings_gold_review.txt`; 111 items corrected).
Findings from `kb_tests.detect()` are out of scope on both sides. Labelling conventions are in the script docstring.

| concept+polarity | P | R | F1 |
|---|---|---|---|
| draft auto-labels (parser before seeing the gold corrections; **the held-out number**) | 0.913 | 0.839 | 0.875 |
| final parser (rules fixed after review — optimistic) | 0.959 | 0.939 | 0.949 |
| final, + subject + temporality | 0.949 | 0.930 | 0.939 |

Per phenomenon (final): exam lists F1 .98, numbers .98, yes/no .97, family .96, past .96, hypothetical .95,
negation .93, list negation .93, simile .93, idiom .92, hedge .90. Parse 0.27 ms/sentence; load 0.06 s.

Old logic on the same gold findings (`data/labels/findings_gold_v1_metrics.json`):
- polarity of explicit mentions (n=287): `clinical_rules.negated` .854, `preconditions._neg` .951,
  `danger_gate.polarity` .948, `nlp.parse` .979.
- relatives' findings not attributed to the patient (n=17): `contains_affirmed` 0.0, `preconditions._self_only` .824,
  `nlp.affirmed` 1.0.
- claims written as the concept's Korean label (n=572): `grounding.is_grounded` grounds the right claim .434 and
  rejects the contradicting one .909; `nlp.match` .921 / .984. (Caveat: the claims use lexicon labels, which favours
  the new layer; the old checker reads only its own synonym groups.)

## Known gaps

Coordination across a comma with sites ("머리 양쪽이랑 이마, 뒷목까지 … 아파요"); findings split across clauses
("종아리가 붓고 당기듯 뻐근"); pediatric vital-sign norms (infant HR 150 read as tachycardia); exam "normal attribute"
statements without a lexicon entry (근력 5/5, 동공 정상); imaging terms not in the lexicon (침윤, 유체 저류, 담도 확장);
diseases not modelled as HX concepts (크론병, 건선); event amnesia vs "기억이 안 나요" (read as uncertainty);
subject when a relative reports ("아내가 쓰러져 있는 저를 발견"); the gold set is a dev set, and rules were tuned on
the dev sample and after gold review — use the draft number as the honest estimate. Retest after switching to
gpt-oss-20b claims (claim wording will differ).

## Migration plan (each call site → new layer)

Principle: switch one module at a time behind its existing function signature, compare on its own tests plus
`scripts/label_findings.py metrics`, keep the old code until the new path is at least as good.

| Call site | Today | Switch to |
|---|---|---|
| `agent/grounding.py:is_grounded`, `_ground`, `_parse_claim`, `Evidence` | 80 synonym groups, own polarity/number/clause code | `match(claim, parse(evidence))`; build evidence once per check with `parse(text, source)` per response |
| `agent/grounding.py:check_findings`, `_ground_qa` | yes/no by `_YES/_NO` regex | `parse(response, context={"question": q})` per ASK turn, then `match`; status 음성 → claim `"<item> 없음"` |
| `agent/grounding.py:check_ddx_support`, `ungrounded_in_text`, `_item_ok` | `_REASON_SPLIT` + `_FINDING_MARK` | keep splitting; `finding_like` = `bool(parse(chunk, "claim"))`, then `match` |
| `knowledge/kb.py:KnowledgeBase._match_spans`, `match_terms`, `_groups` | KB labels + `kb_curated.SYNONYMS/REGEX/BLOCK_WORDS` | **done 2026-09-28** (`_analyze`): `parse(finding, "claim")` → concept → term ids from `data/lexicon/kb_links.json` (+ nearest linked ancestor at weight 0.4); the KB-label scan stays for **all** terms (the lexicon links only 294 concepts; limiting the scan to unlinked terms would drop KB-specific labels) and each label hit takes the polarity of the lexicon mention it overlaps, else `assess_spans()` |
| `knowledge/kb.py:candidates` (`_NEG`, `_strip_neg`) | end-of-text negation regex | **done**: present → query terms; absent (also inside a positive-list finding) → negatives; uncertain / hypothetical / a relative's → nothing; a negative-list finding that states nothing absent negates what it names. `_NEG`, `_NEG_TAIL`, `_strip_neg` removed |
| `knowledge/kb_curated.py:lab_terms` | vital/lab thresholds | **done**: measured findings of `parse` (`LAB_VALUES="curated"` switch keeps the old one; dev MRR −0.01 with it). `SYNONYMS`/`REGEX` no longer read at runtime (682/689 phrases map to the same term through the lexicon; test `test_lexicon_covers_the_retired_curated_synonyms`) but stay in the file: `scripts/build_lexicon.py` and the links builder read them |
| `knowledge/kb_tests.py:detect` | test-result engine | **kept, called directly** by `kb.test_findings`: via `parse` (per clause, `TF:` → concept → `TF:`) measured dev MRR 0.5630 vs 0.5652 direct, and `parse` has no "reported absent" context (`detect(f, -1)`) for the negative list |
| `knowledge/clinical_rules.py:negated`, `contains_affirmed` | 40-char window, stops at commas | `affirmed(text, ids)` / `denied(text, ids)`; keep the keyword signature via `LEXICON.lookup(keyword)` during transition |
| `knowledge/clinical_rules.py:detect_categories` (`CATEGORY_KEYWORDS`) | substring keywords + combination rules | concept ids per category: `LEXICON.by_protocol("category:<name>")`; keep the combination/duration rules; `duration_level`, `age_years` stay (onset field can feed them later) |
| `safety/protocols.py:Check.applies` (`_TRIG_*` via `contains_affirmed`) | keyword tuples | `affirmed(text, LEXICON.by_protocol("protocols:_TRIG_X"))`; tuples whose words are not findings (female, pregnancy weeks) stay as they are |
| `safety/protocols.py:_sepsis_suspected`, `_numeric_instability` | own vital regexes | measured SIGN:hypotension/tachycardia/tachypnea findings |
| `safety/danger_gate.py:polarity`, `_affirmed_at`, `_Ctx.affirmed/denied` | own negation (before/after, affirm-first) | `affirmed`/`denied` over the context text; `_Ctx._latest` vitals (`_RE_SBP/_RE_HR/_RE_RR/_RE_SPO2/_RE_TEMP`) → measured findings (value field) |
| `safety/danger_gate.py:wells_partial`, `perc_negative`, `add_rs`, `_no_fever` | keyword tuples | concept sets (`SIGN:unilateral_leg_swelling`, `HX:recent_surgery`, `HX:prior_vte`, `SYM:hemoptysis`, `HX:hormone_use`, `QUAL:tearing` …) |
| `safety/preconditions.py:_neg`, `_list_negated`, `_affirmed`, `_self_only`, `_RE_EXTRA_NEG` | family-clause removal, list negation, extra 안+verb | `affirmed()` (subject and list scope built in) |
| `safety/preconditions.py:_lp_ct_risks` (seizure-simile mask) | regex mask 발작처럼/발작성 | simile rule + block words; `affirmed(text, "SYM:seizure")` |
| `agent/ledger.py:FindingsLedger.update` | `similarity()` ≥ 0.6 dedupe | optional: key items by `LEXICON.lookup(item)` concept id so "열"/"발열" merge |
| `scripts/package.py:INCLUDE` | ships `data/kb` | add `data/lexicon` (compliance-release); `LEXICON` is an import-time constant, so the cross-case check needs no allowlist entry |

KB migration numbers (`scripts/eval_kb.py`, dev = sample + clinicalqa tuned, held-out = agentclinic + diagnosisarena
report only; before = commit 8a9e049, after = `data/labels/kb_eval_2026-09-28_nlp.json`):

| | top-1 | top-3 | top-10 | top-50 | MRR |
|---|---|---|---|---|---|
| dev before → after | 52 → 52 | 70 → 70 | 81 → 81 | 88 → 87 | 0.559 → 0.565 |
| held-out before → after | 17 → 23 | 31 → 33 | 42 → 46 | 58 → 62 | 0.165 → 0.197 |
| hx+exam only, dev | 13 → 20 | 24 → 30 | 42 → 49 | 61 → 62 | 0.198 → 0.259 |
| hx+exam only, held-out | 11 → 10 | 20 → 20 | 30 → 35 | 46 → 50 | 0.112 → 0.113 |

Per set (top-1 / top-10 / MRR): sample 11/14/.766 → 11/15/.797, clinicalqa 41/67/.524 → 41/66/.526, agentclinic
16/35/.213 → 20/37/.238, diagnosisarena 1/7/.061 → 3/9/.107. `candidates()` mean 11.2 → 26.0 ms (p95 14.7 → 32.4 ms,
two `parse` passes per finding); KB load 0.8–1.0 s either way. Dev choices: link weights (lexicon 1.0 > 0.6; ancestor
0 ≈ 0.4 > 1.0), `LINK_MODE` all > primary/fallback, nlp values > `lab_terms`. Re-adding the curated tables on top
(`CURATED_SYN=True`) gives dev +1 top-1 / +1 top-10 / +3 top-50 but held-out −2 top-1 / −1 top-10 / MRR −0.009, and its dev wins come from
loose regexes ("변형 적혈구" → hematochezia); not taken. Changed test expectations: Graves top-2 → top-3 ("갑상선 비대"
now matches goiter, absent from the Graves profile, and kb_tests reads "TSH 수용체 항체 양성" as elevated TSH too);
troponin-negative test checks the score drop (MI now leads by more than the penalty).

Maintenance: edit `data/lexicon/seed.tsv` (or the source tables), run `python scripts/build_lexicon.py`, then
`python scripts/label_findings.py metrics`, `python scripts/eval_kb.py --build-links` (KB links; `pytest
tests/test_kb_matching.py` fails when they are stale) and `pytest tests/test_nlp_findings.py`. `docs/architecture.md` should get
a pointer to this layer when the first module switches (no interface changed yet).
