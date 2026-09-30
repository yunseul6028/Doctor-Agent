# Clinical-finding normalisation layer (`src/doctor_agent/nlp`)

One lexicon and one parser for every module that reads findings from text (patient answers in colloquial Korean,
exam reports, test results, the doctor model's own claims). Replaces five separate dictionaries and five separate
negation heuristics. Stdlib only, CPU only, no network, no LLM, no state between calls. **Not wired yet** — see the
migration plan below.

## Files

| Path | Role | Shipped |
|---|---|---|
| `src/doctor_agent/nlp/lexicon.py` | loads `data/lexicon/concepts.json` once at import (read-only); `scan()` finds mentions | yes |
| `src/doctor_agent/nlp/findings.py` | `parse()`, `match()`, `concepts_in()`, `affirmed()`, `denied()`; `assess_spans()` (polarity/subject of spans found by another vocabulary); `spans_of()`, `age_from_text()`; stable public helper names (see API) | yes |
| `data/lexicon/kb_links.json` | concept → KB term ids with link kind (27 KB; `python scripts/eval_kb.py --build-links`) | yes |
| `data/lexicon/concepts.json` | built lexicon (≈385 KB; recounted 2026-09-29: 634 concepts, 6,855 surface forms, 152 regexes (148 `re` + 4 `negre`), 57 block words) | yes (`scripts/package.py` INCLUDE) |
| `data/lexicon/seed.tsv` | hand-authored seed (long format `id<TAB>field<TAB>value`) | no (build input only, excluded from the ZIP) |
| `scripts/build_lexicon.py` | merges seed + existing tables → `concepts.json` (`--check` for staleness) | no |
| `scripts/label_findings.py` | gold-set pipeline (extract / dev / sample / show / freeze / metrics / errors) | no |
| `data/labels/findings_gold_*` | candidates, draft, hand review, corrections, gold v1, metrics | no |
| `tests/test_nlp_findings.py`, `tests/test_nlp_fixes.py` | phenomena, match, load/parse speed, gold-set F1 floor; one regression test per issue fixed on 2026-09-28 | no |

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
- `kb`: KB term ids whose label/synonym equals a concept label or synonym (333 concepts linked, 2026-09-29), plus `TF:` ids for
  the 262 `kb_tests` result concepts. `protocol`: clinical_rules category and safety keyword-tuple names that
  mention the concept (`LEXICON.by_protocol("protocols:_TRIG_AMS")`).
- flags: `group` (an absent group makes its members absent in `match`), `cue` (emit only with an explicit cue:
  `HX:pmh`, `LAB:hcg_pos`), `fig`, `kb_tests`.
- provenance per entry: `seed`, `casefreq`, `curated` (kb_curated), `grounding`, `kb_tests`, `clinical_rules`,
  `protocols`, `danger_gate`, `preconditions`; KB links from `data/kb`. First owner of a surface form wins
  (50 conflicts as of 2026-09-29, logged by `--show-conflicts`); ambiguous forms are dropped (`DROP_FORMS`, `CURATED_RE_SKIP`).

Size by category (2026-09-29): SYM 157, SIGN 88, LAB 198 (24 own + 174 from kb_tests), IMG 94, ECG 10, HX 59, QUAL 17, GRP 11.

## Finding

`concept, label, span, polarity (present|absent|uncertain), subject (patient|family|other), temporality
(current|past|chronic|intermittent), onset, laterality, site, severity (mild|moderate|severe [+ NRS n/10]), value,
unit, direction (high|low|normal), hedged, hypothetical, confidence, source, start/end (offsets in normalize(text)),
clause, cue` (what decided the polarity; for debugging), `extra_spans` (further mentions merged into this finding:
same concept, polarity and subject in one clause, "뇌출혈, 뇌경색 소견 없음"; `spans_of(f)` lists every span). Note: this
is not `agent/ledger.Finding`.

## Rules (in order)

1. `normalize`: NFKC, lower case, whitespace runs → one space (a run holding a newline → "\n", a sentence end), `(-)`/`(+)`/`(1+)` → 음성/양성, thousands separators, "두 번" → "2번", 비정상 → 이상.
2. Sentences; sentences saying a result is unavailable are skipped.
3. Mentions: space-insensitive form index (2-syllable forms must start a word or follow a glued prefix such as 우측/잔;
   a match starting inside a word may not span a word boundary), regexes (gaps are lazy and never cross a comma or a
   clause boundary), block words (발작성, 방사선, 종양 표지 …). Longest match wins; on equal spans a form beats a regex
   and a concept beats its ancestor. QUAL mentions are resolved separately so "쥐어짜듯이 아파요" keeps both.
4. Clauses at connectives (~고, ~는데/~데, ~지만, ~으나, ~면서, ~며, ~어서, ~다가, ~니까; not ~고 있다/~고 나서, not
   nouns ending in 고: 자살 사고, 교통사고, 경고), comma parts inside a clause. A mention that is a bare list item takes
   the predicate of the following parts → clause-final negation covers the whole list. Bare = no words of its own after
   it up to the next coordinator (이나/및/또는 …; "임신 32주" is not bare), and in reports no severity/qualifier in its
   part ("경미한 압통", "비특이적 ST분절 하강"). The list stops at a part with a number outside a mention ("두통, 혈압
   150/90"; "S3, S4" continues). The closing cue must be the list's predicate: at the end of its part (light words such
   as 상태/소견/보임/함 may follow) or adverbial "없이"; not an attributive negation before another noun ("통증 없는 질
   출혈") nor a normal-attribute noun phrase ("정상 축"); not when the closing part repeats the item's head with another
   state word ("ST분절 하강, ST분절 상승 없음").
5. Polarity = first cue after the mention in its unit: negation (없/않/아니/아닌/안+verb/음성/정상/괜찮/부인/~기보다/
   no/denies…), affirmation (있/양성/관찰/청진/촉지/호소…), uncertainty (모르겠/글쎄/기억이 안 나…). Masked first:
   idioms that only look negative ("수 없", "이유 없이", "문제없이"); persistence counts as present ("가라앉지 않",
   "안 멈춰"); "X 외 … 없음" keeps X present; double negation ("없지는 않아요") is present; "있는 사람은 없어요" is
   absent. A negation inside a regex match negates it ("배는 안 아파"); one inside a form is part of the concept
   ("입맛이 없"). Absence forms give absent unless immediately negated ("잘 먹지 못해요"). "X할 때 …" presupposes X,
   except before a negated change verb ("기침할 때 심해지지도 않아요" → X uncertain). "아프지 않은 곳이 없어요" / "안 아픈
   데가 없어요" are present; "지혈되지 않" is persistence (bleeding present). "발작" right after another symptom word
   ("증상 발작 시", "호흡곤란 발작", "공황 발작") is not a seizure.
6. Hedge (~것 같아요, ~듯, 아마, 의심/시사 in reports) → `hedged=True`, confidence lowered, polarity kept
   (deliberate: "열이 나는 것 같아요" must still trigger fever protocols). Explicit don't-know → `uncertain`.
   Questions and conjectures (~일까 봐, ~아닌지, 혹시…?) → `hypothetical`, present becomes uncertain.
   Similes (X처럼/마냥/같이) drop X unless the form is figurative by design or X is a pain / quality concept.
7. Subject: latest person word *with a subject particle* before the mention (어머니가, 가족 중, 산모에게 → family;
   남편이, 같이 먹은 동료 → other; 저는, 제가, 아이가 → patient); comitative/dative (부모님과, 엄마한테) do not count.
   A family-history heading without a subject particle ("가족력: 고혈압", "가족력상 당뇨", "family history of") makes the
   rest of its sentence family ("가족력은 없어요" is a statement, not a heading). A proxy tag at the start of the text
   ("(남편) …", "(보호자) …", "보호자: …") makes the person word for the patient (남편 → 아내/부인; 엄마 → 아이/딸; 딸 →
   어머니; 보호자 → the first relative named) the patient, and the speaker's own 저/제가 someone else.
   A case-file key whose first label is 가족력, or a question about family, sets the default.
8. Temporality from clause markers, carried forward in the sentence until 지금은/요즘: past (예전에, ~적이 있/없,
   N년 전에, 과거력, 끊었, …), intermittent (가끔, 반복, 때마다…), chronic (만성, N개월째, 복용 중, 평소…), else
   current; "N년 전에 진단받아 … 복용" is chronic; a trailing "지금은 괜찮아졌어요" clause makes earlier findings past.
   Onset text (3일 전부터, 갑자기, 어제…) is kept separately.
9. Measurements: 체온 (≥37.8 fever, 37.5–37.7 uncertain, <37.5 absent; °F converted; ≥39 high fever), HR (>100 / <60;
   also an unlabelled rate after a rhythm word: "동성빈맥(118회/분)"), RR (>20 / <10); children < 18 y (age from
   `context["age_years"]` or stated in the same text, `age_from_text()`): Fleming 2011 centiles (PMID 21411136) — HR/RR
   > 99th present, 90th–99th uncertain; HR < 1st bradycardia, 1st–10th uncertain; RR < 1st bradypnoea. BP (SBP <90
   hypotension, 90–99 uncertain; ≥140/90 elevated), SpO2 (<92 / 92–94 uncertain / ≥95; not a saturation named after a
   vessel or chamber: 폐동맥/우심방/혼합정맥혈 포화도),
   17 core labs (WBC, Hb, PLT, Na, K, glucose, Cr, BUN, AST/ALT, bilirubin, albumin, CRP, ESR, HCO3, INR, eosinophils)
   with ×10³ / 만 scaling and reference ranges in parentheses; urinalysis items. An abnormal value names one side only.
   For exam/test/claim text `knowledge/kb_tests.detect()` adds its result concepts (reused, not ported).
9b. Merging (per concept, subject and clause): a measured value merges into the mention and decides its polarity;
    two readings of the same polarity become one finding (`extra_spans`); opposite explicit readings stay apart
    ("승모근 압통 있음, 측두동맥 압통 없음"; "혈압 150/90, 재측정 혈압 90/60"); a lexicon mention beats kb_tests' reading
    of the same clause. In reports (not patient speech) a concept's measured values, when they agree, override
    contradicting present/absent mentions anywhere in the text ("맥박 108회/분. 빈맥은 없음" → present).
10. Yes/no: with `context={"question": …}`, "네/아니요/모르겠어요" (or an elliptical answer's own cue) resolves the
    question's concepts the answer does not name itself; a negatively phrased question flips "네".
11. `match(claim, evidence)`: every claim concept needs evidence with the same polarity and subject, or a present
    descendant (for a present claim) / an absent ancestor other than generic pain (for an absent claim); numbers within
    rounding (or ×1000); no laterality clash; uncertain/hypothetical evidence never supports.

## API for other modules

Supported names in `doctor_agent.nlp.findings` (`__all__`): `parse`, `match`, `concepts_in`, `affirmed`, `denied`,
`assess_spans`, `spans_of`, `age_from_text`, `normalize`, and the helpers other modules reuse instead of copying:
`supports` (the match() support rule for one claim/evidence pair), `first_cue`, `ref_direction`, `sentences`,
`clauses` / `Clause`, `CUE`, `IDIOM_NOT_NEG`, `PRE_NEG_EN`, `PRE_NEG_KO`, `BARE_FILLER`, `SITE`, `LAT`, `LAB_RX`,
`UNAVAILABLE`. The underscore names (`_supports`, `_first_cue`, …) stay as aliases of the same objects.
`parse(..., context={"age_years": x})` reads children's vital signs by age (pass the age read from the same case's
initial information with `age_from_text`). A merged finding covers several mentions: callers that look findings up
by offset should iterate `spans_of(f)`, not only `(f.start, f.end)`.

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

### Fixes 2026-09-28 (issues reported by the three migration agents)

Parser: newlines end sentences (normalize keeps "\n"); attributive / noun-phrase negations do not close a list
("통증 없는 질 출혈", "정상 축"); a list item with words of its own keeps its reading ("비특이적 ST분절 하강, ST분절 상승
없음"); measured value beats verdict; "지혈되지 않는 출혈"; "증상 발작 시" is not a seizure; vessel saturations are not
SpO2; family-history headings; proxy speaker; "아프지 않은 곳이 없어요"; merged mentions keep their spans and different
readings stay apart; "S3, S4 또는 …" and "… 없이 양호한" lists; nouns ending in 고 ("자살 사고 없음"); "기침할 때
심해지지도 않아요"; children's HR/RR by age. Lexicon: 기면 → altered mental status; "아팠던 적" is not HX:pmh (only
"특별히/크게 아팠던 적"); "감기 기운/증상" → new SYM:uri_symptoms (child of the respiratory group, not the group);
SIGN:brudzinski split from SIGN:kernig; SYM:self_harm split from SYM:suicidal_ideation (suicide attempt stays);
SIGN:fracture / SIGN:dislocation split from HX:trauma. kb_tests: "TSH 수용체 항체" is TRAb, not a TSH level (Graves test
back to top 2). No LLM was used; measured with the scripts named below, before = commit 5037ee8.

| concept+polarity F1 (P / R) | before | after | after, inventory-adjusted gold* |
|---|---|---|---|
| gold v1, all 397 sentences (seen while the layer was built) | .949 (.959/.939) | .946 (.954/.938) | .950 (.959/.942) |
| gold v1, 353 case sentences (no synthetic yes/no) | .947 | .944 | .949 |
| **fresh held-out sample**, 80 sentences / 135 findings, never seen (not dev, not gold; labelled by hand before any fix) | .878 (.906/.852) | .885 (.913/.859) | .890 (.921/.860) |

*Three items were labelled with concept ids that the lexicon split since (ac_12 fracture/dislocation, cqa_629
self-harm, fresh ac_79 Brudzinski); the adjusted copy relabels them only. Per phenomenon (before → after adjusted):
list negation .930 → .934, numbers .979 → .985, negation .934 → .937, all others unchanged (exam lists .982, family
.961, hedge .899, hypothetical .947, idiom .917, past .955, plain .973, simile .933, yes/no .969). Parse 0.29 → 0.31
ms/item. Old-logic comparison (`label_findings.py metrics`, raw gold): `clinical_rules.negated` .983 → .990,
`preconditions._neg` / `danger_gate.polarity` .990 = .990, `nlp.parse` polarity .979 → .972 (only the relabelled
items), `grounding.is_grounded` right claim .844 → .848 (contradicting rejected .997 = .997), `nlp.match` right claim
.921 → .920 (−3 relabelled, +2 rhythm-rate) / contradicting rejected .984 → .990. KB (`eval_kb.py`): held-out
top-1/3/10/50 23/33/46/62, MRR .197 (unchanged); dev 52/70/81/87 → 53/69/81/87, MRR .565 → .566. Safety coverage
over the 267 case files (categories, must-checks, red flags, rule-out status, preconditions, criteria): 2 cases
changed, both judged right — da_122 (syncope in the chief complaint is no longer negated by a list in the next
sentence → syncope red flags) and da_14 (the ECG's "동성빈맥 (123회/분)" is now read, Wells rises to 5.5, so a normal
D-dimer alone no longer rules out PE).

## Known gaps

Coordination across a comma with sites ("머리 양쪽이랑 이마, 뒷목까지 … 아파요") and elided heads ("간 및 비장 종대
없음" → no hepatomegaly; "관절 발적이나 종창" → joint swelling, read as edema); findings split across clauses
("종아리가 붓고 당기듯 뻐근"); children's vital signs need the age (in `context` or the same text: a test report alone,
"심박수 120회/분 … 연령에 부합하는 정상 심전도", is read with adult limits); exam "normal attribute" statements without a
lexicon entry (근력 5/5, 동공 정상, 보행 양호, 정신은 맑습니다); imaging terms not in the lexicon (침윤, 유체 저류, 담도
확장); diseases not modelled as HX concepts (크론병, 건선); event amnesia vs "기억이 안 나요" (read as uncertainty);
subject when a relative reports without a proxy tag ("남편이 저를 깨우기 힘들어했어요"); word-start forms inside longer
words (경기관지 → 경기 seizure, 피로인산염 → 피로, 눈을 깜빡 → memory loss, 망상 선 → delusion); "잠혈 약양성" not read as
positive; "T-Bili" not read. Found on the fresh sample and left unfixed so that sample stays held-out. The gold set is a
dev set, and rules were tuned on the dev sample and after gold review — use the fresh-sample number as the honest
estimate. Retest after switching to gpt-oss-20b claims (claim wording will differ).

## Migration plan (each call site → new layer)

Principle: switch one module at a time behind its existing function signature, compare on its own tests plus
`scripts/label_findings.py metrics`, keep the old code until the new path is at least as good.

| Call site | Today | Switch to |
|---|---|---|
| `agent/grounding.py:is_grounded`, `_ground`, `_parse_claim`, `Evidence` | 80 synonym groups, own polarity/number/clause code | `match(claim, parse(evidence))`; build evidence once per check with `parse(text, source)` per response. **Done** (2026-09-28): concept claims via `parse` + `findings._supports` with grounding-side guards (strict numbers, bare-item list negation, reference-range check of kb_tests values, same-response conflicts, site-narrowed absent parents, claim qualifier words, past/quit vs current); concept-free claims keep a literal-word reader (layer cues, `_LAB_RX` analyte names). `_GROUPS` moved to `scripts/build_lexicon.py` (build input only) 2026-09-30 |
| `agent/grounding.py:check_findings`, `_ground_qa` | yes/no by `_YES/_NO` regex | `parse(response, context={"question": q})` per ASK turn, then `match`; status 음성 → claim `"<item> 없음"`. **Done**: `apply()` parses each ASK answer with its question (parses cached on the case state); the (question, answer) `qa` path was only used offline and moved to `scripts/label_findings.py` (2026-09-30) |
| `agent/grounding.py:check_ddx_support`, `ungrounded_in_text`, `_item_ok` | `_REASON_SPLIT` + `_FINDING_MARK` | keep splitting; `finding_like` = `bool(parse(chunk, "claim"))`, then `match`. **Done** (`finding_like` = a non-kb_tests concept, a number or `_FINDING_MARK`: kb_tests reads disease names such as 대동맥 박리 as results) |
| `knowledge/kb.py:KnowledgeBase._match_spans`, `match_terms`, `_group_hits` | KB labels + `kb_curated.SYNONYMS/REGEX/BLOCK_WORDS` | **done 2026-09-28** (`_analyze`): `parse(finding, "claim")` → concept → term ids from `data/lexicon/kb_links.json` (+ nearest linked ancestor at weight 0.4); the KB-label scan stays for **all** terms (the lexicon links only 307 concepts in `kb_links.json` as of 2026-09-29; limiting the scan to unlinked terms would drop KB-specific labels) and each label hit takes the polarity of the lexicon mention it overlaps, else `assess_spans()` |
| `knowledge/kb.py:candidates` (`_NEG`, `_strip_neg`) | end-of-text negation regex | **done**: present → query terms; absent (also inside a positive-list finding) → negatives; uncertain / hypothetical / a relative's → nothing; a negative-list finding that states nothing absent negates what it names. `_NEG`, `_NEG_TAIL`, `_strip_neg` removed |
| `knowledge/kb_curated.py:lab_terms` | vital/lab thresholds | **done**: measured findings of `parse` (`LAB_VALUES="curated"` switch keeps the old one; dev MRR −0.01 with it). `SYNONYMS`/`REGEX` no longer read at runtime (682/689 phrases map to the same term through the lexicon; test `test_lexicon_covers_the_retired_curated_synonyms`) but stay in the file: `scripts/build_lexicon.py` and the links builder read them |
| `knowledge/kb_tests.py:detect` | test-result engine | **kept, called directly** by `kb.test_findings`: via `parse` (per clause, `TF:` → concept → `TF:`) measured dev MRR 0.5630 vs 0.5652 direct, and `parse` has no "reported absent" context (`detect(f, -1)`) for the negative list |
| `knowledge/clinical_rules.py:negated`, `contains_affirmed` | **done 2026-09-27** | `keyword_statuses(text, kw)`: each keyword occurrence takes the reading of the layer finding it overlaps (POS / UNC / NEG / OTHER); no finding, or a keyword with its own negation ("의식이 없", "지혈되지") → legacy window rule + relative-before check. Layer list negations are trusted only when the negation ends the sentence ("통증 없는 질 출혈" is not one). Lines are parsed separately (a workaround from when normalize() folded newlines; since 2026-09-28 the layer ends a sentence at a line break itself, the per-line parse is kept for its offset map). Since 2026-09-28 the layer also keeps merged mentions (`spans_of`), reads family headings and proxy speakers, and only closes a list with a predicative cue, so `_usable`, the proxy rule and the heading check are now partly redundant; they were left in place (not provably equivalent). `ReadText` = text + lazily parsed findings (parse once per case text). Proxy speaker "(남편) … 아내가" → subject not trusted. Uncertain = affirmed for triggers, never denied |
| `knowledge/clinical_rules.py:detect_categories` (`CATEGORY_KEYWORDS`) | **done 2026-09-27** | keywords stay primary; a category is dropped when every keyword hit is denied by the layer ("열은 없고 기침만"); layer concepts add 11 categories whose `category:` links name exactly that complaint (`_LAYER_CATEGORIES`; "심와부 통증" → abdominal_pain) and focal-deficit concepts ("걸음걸이가 비틀"); not from the layer: rash/edema/neuro-vision/joint/allergy/bleeding/psychiatric links (broader than the protocol) |
| `safety/protocols.py:Check.applies` (`_TRIG_*` via `contains_affirmed`) | **done 2026-09-27** | through `contains_affirmed` (keyword tuples kept; each occurrence read by the layer); `must_checks_for` passes one `ReadText` to every check |
| `safety/protocols.py:_sepsis_suspected`, `_numeric_instability` | **done 2026-09-27** | layer-measured SBP/HR/RR values pooled with the module's regexes (qSOFA / HR cut-offs unchanged) |
| `safety/danger_gate.py:polarity`, `_affirmed_at`, `_Ctx.affirmed/denied` | **done 2026-09-27** | conservative union for the rule-out gate: affirmed if the layer or `_affirmed_at` affirms, negated only if both negate, dropped if the layer attributes it to a relative; `_Ctx.affirmed` via `contains_affirmed`; vitals: per response the layer's measured values pooled with the old regexes, worst value kept (lowest SBP/SpO2, highest HR/RR/T); SpO2 only with an oxygen label ("폐동맥 포화도" is not SpO2) |
| `safety/danger_gate.py:wells_partial`, `perc_negative`, `add_rs`, `_no_fever` | **via `_Ctx.affirmed`** | keyword tuples kept; each occurrence read by the layer (relatives' cancer / VTE no longer count). Concept sets not adopted yet |
| `safety/preconditions.py:_neg`, `_list_negated`, `_affirmed`, `_self_only`, `_RE_EXTRA_NEG` | **done 2026-09-27** | `_statuses` = `keyword_statuses` (+ `_list_negated` only for keywords the layer has no finding for); `_self_only` no longer cuts clauses with a relative or an honorific "셨" (it also dropped the patient's own "진단받으셨어요"); relatives come from the layer subject / relative-before check |
| `safety/preconditions.py:_lp_ct_risks` (seizure-simile mask) | **kept + layer** | mask kept (and "증상/호흡곤란 발작" added: the layer reads "증상 발작 시" as a seizure); the masked text is read through `_affirmed` |
| `knowledge/diagnostic_criteria.py:Doc.state`, `Doc._negated` | 25-char window | **done 2026-09-27**: keyword-like matches (≤ 20 chars, no result word of their own such as "ana 음성") read by the layer (`ReadText.findings_at`); uncertain / hypothetical / relatives' mentions are "skip" (never NOT_MET); long patterns and "- 음성" ledger lines keep the window rule |
| `agent/ledger.py:FindingsLedger.update` | `similarity()` ≥ 0.6 dedupe | optional: key items by `LEXICON.lookup(item)` concept id so "열"/"발열" merge |
| `scripts/package.py:INCLUDE` | ships `data/kb` | ships `data/lexicon/concepts.json` + `kb_links.json` (not `seed.tsv`); `LEXICON` is an import-time constant, so the cross-case check needs no allowlist entry |

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
