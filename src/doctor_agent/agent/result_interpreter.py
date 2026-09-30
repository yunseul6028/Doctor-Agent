"""Result interpreter: reads one test / exam result TEXT into structured items, separately from the diagnosing agent.

The environment returns results as free text ("흉부 X선: 우하엽 경화, 기흉 없음", "WBC 15,000/μL", "ST elevation in II,
III, aVF", "우하복부 압통 있음"). This module turns such a text into items the policy can use without asking the LLM to
re-read it: kind (lab / imaging / ecg / exam / other), a lexicon concept id, polarity (present / absent / uncertain),
laterality and site, lab value + unit + direction + critical flag, comparison with a prior study, the diseases the
finding supports (curated kb_tests links; supportive only, never diagnostic) and a one-line Korean summary.
Pure code: stdlib only, CPU only, deterministic, no network, no LLM call, nothing kept between calls or cases.

    interpret(test_name, result_text, patient_ctx=None) -> Interpretation
    render_for_prompt(interp, max_chars=300) -> str        one Korean line for the doctor prompt
    needs_llm(interp) -> bool                              the text is too long / complex for code alone
    llm_reasons(interp) -> list[str]                       why (for logs)

patient_ctx (optional, same case only): {"age_years": float, "sex": "M"|"F", "initial_info": str}. The age switches
children's heart / respiratory rate ranges in the nlp layer (Fleming 2011 centiles).

How a text is read
1. normalise (nlp.lexicon.normalize); "result not provided" texts are marked `unavailable` (never read as normal);
2. report sections: Findings / Impression / 소견 / 결론 are kept; Indication / Clinical history / Technique /
   Comparison / Recommendation / 권고 / 임상 정보 sections are dropped (an indication "r/o pneumonia" is not a finding);
   inside a sentence, a comma part that starts a recommendation ("추적 CT 권고", "follow-up in 3 months") and the parts
   after it are dropped;
3. comparison phrases that look like negations ("no interval change in ...", "이전과 비교하여 변화 없음") are masked
   before parsing (the finding is still there) and recorded as `comparison` (new / improved / worsened / stable /
   resolved); "resolved / 이전 대비 소실" makes the finding absent;
4. findings: (a) nlp.findings.parse (lexicon mentions, measured vitals and labs with the age-aware / reference-range
   logic, knowledge/kb_tests result concepts); kb_tests imaging / ECG readings are re-anchored on their own pattern
   span; kb_tests lab readings keep kb_tests' polarity (printed range in its unit, cut-offs) and get their value for
   display (_kb_value); (b) organ-dependent report words ("경화", "출혈", "종괴", "확장", "비후", "혈전", ...) mapped with the organ named
   next to them or, failing that, the organ of the test name (_DESCRIPTORS; "출혈" on a brain CT = IMG:ct_ich);
   (c) disease-level impression words in an imaging report ("급성 충수염 의심") mapped to the imaging concept that
   carries them; (d) history-type concepts re-mapped in imaging context (HX:prior_vte "폐색전증" -> IMG:ctpa_pe);
5. polarity of (b)/(c)/(re-anchored kb) spans: nlp.findings.assess_spans (same cue rules as parse); then, per comma
   part: hedges ("가능성", "의심", "r/o", "possible", "likely", "versus", "배제 필요") make a present finding uncertain;
   "배제할 수 없음" / "cannot be excluded" makes any reading uncertain; "배제됨" / "was excluded" makes it absent;
6. whole-normal statements ("특이 소견 없음", "정상", "No acute cardiopulmonary process", "unremarkable") give
   `normal=True` when nothing abnormal was read (IMG:normal_study / ECG:normal_ecg item); abnormal report words left
   unmapped become items with concept "" (they make needs_llm() true);
7. critical: an urgent-result concept (_CRITICAL_CONCEPTS) read as present/uncertain, or a lab / vital value beyond
   a critical limit (_CRITICAL_VALUES).

Sources (facts only; thresholds and selections are ours, wording ours, nothing copied):
- critical lab limits: Kost GJ. Critical limits for urgent clinician notification at US medical centers. JAMA
  1990;263:704-7, doi:10.1001/jama.1990.03440050098042 - rounded adult limits, our choice of analytes (not
  re-verified against the full text); vitals follow the thresholds cited in safety/triage.py (NEWS2, shock index);
- urgent imaging results: the "critical findings -> nonroutine communication" category of the ACR Practice Parameter
  for Communication of Diagnostic Imaging Findings (American College of Radiology; see docs/licenses.md); the
  parameter gives no list, so the concept list (pneumothorax, free air, dissection, intracranial hemorrhage, PE,
  torsion, ectopic pregnancy, ...) is our selection;
- disease links: knowledge/kb_tests.py (each link cites its guideline there).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from doctor_agent.nlp import findings as F
from doctor_agent.nlp.lexicon import LEXICON, normalize

KINDS = ("lab", "imaging", "ecg", "exam", "other")
POLARITIES = ("present", "absent", "uncertain")


# ------------------------------------------------------------------------------------------------ data model
@dataclass
class Item:
    kind: str  # lab | imaging | ecg | exam | other
    concept: str  # lexicon concept id; "" = abnormal report wording the code could not map
    label: str  # short Korean label
    polarity: str  # present | absent | uncertain
    span: str = ""
    site: str = ""
    laterality: str = ""  # right | left | bilateral
    value: float | None = None
    unit: str = ""
    direction: str = ""  # high | low | normal
    critical: bool = False
    comparison: str = ""  # new | improved | worsened | stable | resolved
    hedge: str = ""  # the cue that made the reading uncertain
    supports: tuple[tuple[str, str, int], ...] = ()  # (dx key, Korean name, weight 1-3): SUPPORTIVE ONLY
    source: str = ""  # lexicon | kb_tests | value | descriptor | impression | normal | unmapped
    confidence: float = 0.8
    summary_ko: str = ""

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in ("", None, False, ()) and not k.startswith("_")}
        d["polarity"] = self.polarity
        d["concept"] = self.concept
        if self.supports:
            d["supports"] = [list(s) for s in self.supports]
        return d


@dataclass
class Interpretation:
    test_name: str
    kind: str
    items: list[Item] = field(default_factory=list)
    normal: bool = False  # a whole-normal statement and nothing abnormal read
    unavailable: bool = False  # "결과가 제공되지 않습니다": NOT normal
    ignored: list[str] = field(default_factory=list)  # dropped sections / recommendation parts
    n_sentences: int = 0
    text_len: int = 0
    serial: bool = False  # several time points in one text ("3일 후 ..., 5일 후 ...")
    pending: bool = False  # some results are still pending ("대기 중", "pending"): not results
    llm_reasons: list[str] = field(default_factory=list)

    def by_polarity(self, pol: str) -> list[Item]:
        return [i for i in self.items if i.polarity == pol]

    def present(self) -> list[Item]:
        return self.by_polarity("present")

    def absent(self) -> list[Item]:
        return self.by_polarity("absent")

    def uncertain(self) -> list[Item]:
        return self.by_polarity("uncertain")

    def critical(self) -> list[Item]:
        return [i for i in self.items if i.critical]

    def concepts(self, polarity: str | None = "present") -> set[str]:
        return {i.concept for i in self.items if i.concept and (polarity is None or i.polarity == polarity)}

    def as_dict(self) -> dict:
        return {"test_name": self.test_name, "kind": self.kind, "normal": self.normal, "unavailable": self.unavailable,
                "items": [i.as_dict() for i in self.items], "ignored": self.ignored, "llm_reasons": self.llm_reasons}


# ------------------------------------------------------------------------------------------------ test kind
_ECG_TEST = re.compile(r"심전도|(?<![a-z])(?:ecg|ekg)(?![a-z])|holter|홀터|electrocardiog")
_IMAGING_TEST = re.compile(
    r"x-?선|x-?ray|엑스레이|radiograph|(?<![a-z])(?:ct|cta|mri|mra|mrcp|us|cxr|kub|pet|tee|tte)(?![a-z])|씨티|단층|자기\s?공명"
    r"|초음파|sono|ultrasound|doppler|도플러|duplex|듀플렉스|echo|심초음파|조영술|angiog|스캔|scan|영상|촬영|투시|fluoro"
    r"|내시경|endoscop|colonoscop|(?<![a-z])egd(?![a-z])|신티|scintig")
_EXAM_TEST = re.compile(r"진찰|신체|청진|촉진|타진|시진|(?<![a-z])exam|physical|활력|vital|생체\s?징후|신경학|neurolog|auscultat"
                        r"|신경|감각|근력|반사|보행|운동\s?범위|관절|외모|피부|심장|(?<![가-힣])폐(?!기능)|복부|(?<![가-힣])배|흉부|외이|고막|이경"
                        r"|안저|(?<![가-힣])눈|인두|편도|구강|림프절|사지|하지|상지|두경부|(?<![가-힣])목|경부|직장\s?수지|유방|의식|정신\s?상태"
                        r"|mental status|(?<![a-z])gcs(?![a-z])|skin|heart|lung|abdomen|extremit|head|neck|eye|ear|throat"
                        r"|palpat|inspection|이학적")
_LAB_TEST = re.compile(r"혈액|피검사|(?<![a-z])lab|cbc|혈구|화학|전해질|소변|요검사|urinalysis|배양|culture|효소|트로포닌|troponin"
                       r"|crp|혈당|glucose|가스|abga|abg|호르몬|항체|항원|표지자|marker|응고|(?<![a-z])(?:pt|inr|lft|rft|bmp|cmp)(?![a-z])"
                       r"|간기능|신기능|수치|serum|blood|plasma")


def kind_of_test(test_name: str, text: str = "") -> str:
    """lab | imaging | ecg | exam | other, from the test name first, then from what the text holds."""
    t = normalize(test_name)
    if _ECG_TEST.search(t):
        return "ecg"
    if _IMAGING_TEST.search(t):
        return "imaging"
    if _EXAM_TEST.search(t):
        return "exam"
    if _LAB_TEST.search(t):
        return "lab"
    body = normalize(text)[:200]
    if _ECG_TEST.search(body):
        return "ecg"
    if _IMAGING_TEST.search(body):
        return "imaging"
    cats = {f.concept.split(":")[0] for f in F.parse(body, "test")}
    if "LAB" in cats:
        return "lab"
    if "IMG" in cats:
        return "imaging"
    if cats & {"SIGN", "SYM"}:
        return "exam"
    return "other"


# ------------------------------------------------------------------------------------------------ anatomy
# region words (normalised text); order matters only for ties at the same position
_REGIONS: list[tuple[str, str]] = [
    ("brain", r"뇌(?!하수체)|두개|두부|머리|brain|head\b|cranial|cerebr|intracran|소뇌|대뇌|전두엽|측두엽|두정엽|후두엽|기저핵|시상(?!하)"
              r"|뇌간|basal cistern|중대뇌|(?<![a-z])mca(?![a-z])|정맥동|venous sinus|dural sinus"),
    ("pituitary", r"뇌하수체|터키안|sella|pituitar"),
    ("heart", r"심초음파|echocardio|심장|cardiac|심실|심방|판막|valv|ventric|atri(?:um|al)|승모판|대동맥판|삼첨판|mitral|tricuspid"
              r"|심낭|pericard|(?<![a-z])(?:lv|rv|la|ra|lvot)(?![a-z])|중격|septum|septal|heart"),
    ("pulm_artery", r"폐동맥|pulmonary arter"),
    ("vein", r"정맥(?!동)|(?<![a-z])vein|venous|(?<![a-z])dvt(?![a-z])|하지|다리|대퇴|슬와|종아리|popliteal|femoral|lower extremit"
             r"|lower limb|calf"),
    ("aorta", r"대동맥(?!판)|aort(?:a|ic)(?! valve)"),
    ("mediastinum", r"종격동|mediastin"),
    ("pleura", r"흉막|늑막|pleura|늑골\s?횡격막|costophrenic"),
    ("lung", r"(?<![가-힣])폐(?!색|경|동맥)|흉부|흉곽|chest|lung|pulmonar(?!y arter)|thora|(?<![a-z])cxr(?![a-z])|상엽|중엽|하엽|설엽"
             r"|lobe|폐야|폐문|hil(?:ar|um)|기관지|bronch|lingula|(?<![a-z])(?:rul|rml|rll|lul|lll)(?![a-z])|폐첨부|폐\s?기저부"),
    ("gallbladder", r"담낭|gallbladder|(?<![a-z])gb(?![a-z])"),
    ("bile_duct", r"담관|담도|총담관|(?<![a-z])cbd(?![a-z])|bile duct|biliar|ductal"),
    ("pancreas", r"췌장|이자|췌관|pancrea"),
    ("liver", r"(?<![가-힣])간(?=[\s의에은이]|우엽|좌엽|실질|내|문)|liver|hepat"),
    ("spleen", r"비장|spleen|splen"),
    ("kidney", r"신장|콩팥|요관|방광|신우|신배|신주위|kidney|renal|ureter|bladder|요로|urinary|perinephric"),
    ("appendix", r"충수|맹장|appendi"),
    ("stomach", r"(?<![가-힣])위(?=벽|저부|체부|전정부|\s|의|에)|stomach|gastric"),
    ("bowel", r"소장|대장|결장|장관|회장|공장|직장|bowel|colon|intestin|ile(?:um|al)|jejun|rect(?:um|al)|sigmoid|cecum|맹장"
              r"|(?<![가-힣])장(?=\s?(?:폐색|확장|벽|관|내|의|고리|중첩|염전))"),
    ("gyn", r"자궁|난소|부속기|골반|uter|ovar|adnex|pelvi|transvaginal|더글라스|douglas|난관"),
    ("testis", r"고환|음낭|testi|scrot|정삭"),
    ("abdomen", r"복부|복강|복막|abdom|peritone|횡격막\s?(?:아래|하)|subdiaphragm"),
    ("bone", r"뼈|(?<![가-힣])골(?!반)|척추|요추|경추|흉추|늑골(?!\s?횡격막)|bone|spine|vertebra|osse|(?<![a-z])ribs?(?![a-z])"),
    ("thyroid", r"갑상선|thyroid"),
]
_REGION_RX = [(name, re.compile(p)) for name, p in _REGIONS]

# default region from the test name ("흉부 X선" -> lung, "뇌 CT" -> brain, "심초음파" -> heart)
_TEST_REGION: list[tuple[str, str]] = [
    ("heart", r"심초음파|echo|(?<![a-z])(?:tte|tee)(?![a-z])"),
    ("pulm_artery", r"폐동맥|ctpa|ct pulmonary angio|pulmonary angio|pe protocol"),
    ("vein", r"도플러|doppler|duplex|듀플렉스|정맥|venous|하지"),
    ("brain", r"뇌|두부|머리|brain|head|cranial"),
    ("testis", r"고환|음낭|scrot|testi"),
    ("gyn", r"골반|자궁|난소|pelvi|transvaginal|질식|산부인과|ob\b"),
    ("kidney", r"신장|콩팥|kub|renal|kidney|요로|urinary|방광"),
    ("gallbladder", r"담낭|간담도|hepatobiliar|ruq|우상복부"),
    ("aorta", r"대동맥|aort"),
    ("lung", r"흉부|chest|cxr|lung|폐|thora"),
    ("abdomen", r"복부|abdom|배"),
    ("bone", r"척추|spine|뼈|bone|골|관절|joint"),
]
_TEST_REGION_RX = [(name, re.compile(p)) for name, p in _TEST_REGION]


def _test_region(test_name: str) -> str:
    t = normalize(test_name)
    return next((name for name, rx in _TEST_REGION_RX if rx.search(t)), "")


def _regions_in(text: str) -> list[tuple[int, int, str]]:
    out = []
    for name, rx in _REGION_RX:
        for m in rx.finditer(text):
            out.append((m.start(), m.end(), name))
    out.sort()
    return out


def _region_at(sent: str, part: tuple[int, int], s: int, e: int, default: str,
               regions: list[tuple[int, int, str]]) -> str:
    """The organ a report word at sent[s:e] refers to: the nearest region word before it in its comma part, else the
    first one after it in the part, else the nearest before it in the sentence, else the test's region."""
    a, b = part
    before = [r for r in regions if a <= r[0] and r[1] <= s]
    if before:
        return before[-1][2]
    after = [r for r in regions if r[0] >= e and r[1] <= b]
    if after:
        return after[0][2]
    before = [r for r in regions if r[1] <= s]
    if before:
        return before[-1][2]
    return default


# ------------------------------------------------------------------------------------------------ report words
# (name, regex, {region: concept} ("*" = any region), qualifiers [(regex in the comma part, concept)], fixed polarity)
# A qualifier found in the same comma part picks a more specific concept ("출혈" + "지주막하" -> IMG:ct_sah). A region
# missing from the table leaves the word unmapped (reported as concept "").
_DESCRIPTORS: list[tuple[str, str, dict[str, str], list[tuple[str, str]], bool]] = [
    ("consolidation", r"경화(?!증|성|된)|consolidat\w*|(?<![가-힣])침윤(?!성\s?(?:암|병변))|infiltrat\w*|airspace opacit\w*"
                      r"|(?<![a-z])opacit(?:y|ies)|음영\s?(?:증가|증강)|폐렴|pneumonia",
     {"lung": "IMG:cxr_consolidation"}, [(r"간\s?유리|ground[- ]glass", "IMG:ggo")], False),
    ("effusion", r"삼출(?!성\s?중이염)|effusion|흉수|(?:액체|fluid)(?:\s?(?:저류|collection|고임))?",
     {"pleura": "IMG:pleural_effusion", "lung": "IMG:pleural_effusion", "heart": "IMG:echo_pericardial_effusion",
      "abdomen": "IMG:free_fluid", "gyn": "IMG:free_fluid", "gallbladder": "IMG:us_cholecystitis"},
     [(r"심낭|pericard", "IMG:echo_pericardial_effusion"), (r"흉막|늑막|pleura", "IMG:pleural_effusion"),
      (r"담낭\s?주위|pericholecyst", "IMG:us_cholecystitis")], False),
    ("hemorrhage", r"출혈|hemorrhag\w*|haemorrhag\w*|bleed\w*|혈종|hematoma|haematoma",
     {"brain": "IMG:ct_ich"},
     [(r"지주막하|거미막|subarachnoid", "IMG:ct_sah"),
      (r"경막하|경막외|subdural|epidural|extradural", "IMG:subdural_hematoma")], False),
    ("infarct", r"(?<!심근\s)(?<!심근)경색|infarct\w*|확산\s?제한|diffusion restrict\w*|restricted diffusion|허혈성\s?(?:병변|변화)",
     {"brain": "IMG:stroke_imaging"}, [], False),
    ("mass", r"종괴(?!\s?효과)|종양|(?<![가-힣])덩이|덩어리|(?<![a-z])mass(?:es)?(?!\s?effect)(?![a-z])|tumou?rs?(?![a-z])|신생물|neoplas\w*",
     {"lung": "IMG:lung_mass", "pancreas": "IMG:pancreas_mass", "bile_duct": "IMG:biliary_mass",
      "stomach": "IMG:gastric_mass", "bowel": "IMG:colon_mass", "pituitary": "IMG:pituitary_mass", "*": "IMG:mass"},
     [], False),
    ("nodule", r"결절(?!성)|nodules?(?![a-z])", {"lung": "IMG:lung_nodule"}, [], False),
    ("stone", r"결석|담석|(?<![a-z])stones?(?![a-z])|calcul(?:us|i)(?![a-z])",
     {"gallbladder": "IMG:gallstone", "bile_duct": "IMG:cbd_stone", "kidney": "IMG:stone_imaging"}, [], False),
    ("dilatation", r"확장(?!기)|dilat\w*|늘어나|늘어난|확대|widen\w*|ectasia|distend\w*|distension|팽창|팽대",
     {"bile_duct": "IMG:cbd_dilation", "kidney": "IMG:hydronephrosis", "bowel": "IMG:sbo_imaging",
      "appendix": "IMG:ct_appendicitis", "gallbladder": "IMG:us_cholecystitis", "mediastinum": "IMG:cxr_mediastinum"},
     [(r"우심실|right ventric|(?<![a-z])rv(?![a-z])", "IMG:echo_rv_strain"),
      (r"복부\s?대동맥|abdominal aort", "IMG:aaa_imaging"), (r"관상\s?동맥|coronary", "IMG:coronary_aneurysm"),
      (r"심장\s?(?:음영|크기)?|cardiac silhouette|heart", "IMG:cxr_hf")], False),
    ("thickening", r"비후|두꺼워|두꺼운|두꺼움|두껍|thicken\w*|hypertroph\w*",
     {"gallbladder": "IMG:us_cholecystitis", "appendix": "IMG:ct_appendicitis"},
     [(r"비대칭|asymmetric|중격|septal|septum", "IMG:echo_hcm"),
      (r"좌심실|left ventric|(?<![a-z])lv(?![a-z])", "IMG:lvh")], False),
    ("thrombus", r"혈전|thromb\w*|색전|embol\w*|충만\s?결손|filling defect|(?<![가-힣])폐색(?!성)|occlu\w*",
     {"vein": "IMG:doppler_dvt", "pulm_artery": "IMG:ctpa_pe", "bowel": "IMG:sbo_imaging"},
     [(r"정맥동|venous sinus|dural sinus", "IMG:cvst_imaging"), (r"폐동맥|pulmonary arter|폐\s?색전|pulmonary embol", "IMG:ctpa_pe"),
      (r"중대뇌|대뇌동맥|기저동맥|cerebral arter|(?<![a-z])mca(?![a-z])|basilar", "IMG:stroke_imaging")], False),
    ("noncompressible", r"압박되지\s?않|압박이\s?되지\s?않|압박\s?불가|압박되지\s?않는|non-?compressib\w*",
     {"vein": "IMG:doppler_dvt", "*": "IMG:doppler_dvt"}, [], True),
    ("edema", r"부종|(?<![a-z])o?edema(?![a-z])",
     {"lung": "IMG:cxr_hf", "brain": "IMG:mass_effect", "pancreas": "IMG:ct_pancreatitis", "kidney": "IMG:ct_pyelonephritis",
      "gallbladder": "IMG:us_cholecystitis"}, [], False),
    ("inflammation", r"염증|inflamm\w*|stranding|지방\s?(?:층\s?)?침윤",
     {"appendix": "IMG:ct_appendicitis", "pancreas": "IMG:ct_pancreatitis", "gallbladder": "IMG:us_cholecystitis"},
     [(r"게실|diverticul", "IMG:ct_diverticulitis"), (r"충수|appendi", "IMG:ct_appendicitis"),
      (r"췌장|pancrea", "IMG:ct_pancreatitis"), (r"담낭|gallbladder", "IMG:us_cholecystitis"),
      (r"신장|신주위|perinephric|renal", "IMG:ct_pyelonephritis")], False),
    ("heart_size", r"heart size|cardiac size|심장\s?(?:크기|음영)|cardiac silhouette|심흉곽비|(?<![a-z])ctr(?![a-z])",
     {"*": "IMG:cxr_hf"}, [], False),
    ("free_air", r"(?:유리|자유)\s?(?:공기|가스)|free (?:intraperitoneal )?(?:air|gas)|횡격막\s?(?:아래|하)\s?공기"
                 r"|복강\s?내\s?(?:공기|가스)|pneumoperitoneum", {"*": "IMG:free_air"}, [], False),
    ("rv_collapse", r"확장기\s?허탈|diastolic collapse", {"*": "IMG:echo_tamponade"}, [], False),
]
# disease-level impression words in an imaging report: the imaging concept that carries them (any region)
_IMPRESSIONS: list[tuple[str, str]] = [
    (r"충수염|appendicitis", "IMG:ct_appendicitis"),
    (r"담낭염|cholecystitis", "IMG:us_cholecystitis"),
    (r"게실염|diverticulitis", "IMG:ct_diverticulitis"),
    (r"췌장염|pancreatitis", "IMG:ct_pancreatitis"),
    (r"신우신염|pyelonephritis", "IMG:ct_pyelonephritis"),
    (r"기흉|pneumothorax", "IMG:cxr_ptx"),
    (r"대동맥\s?박리|aortic dissection|(?<![a-z])dissection(?![a-z])", "IMG:ct_dissection"),
    (r"폐\s?색전(?:증)?|pulmonary embol\w*", "IMG:ctpa_pe"),
    (r"심부\s?정맥\s?혈전(?:증)?|deep ve(?:in|nous) thrombosis|(?<![a-z])dvt(?![a-z])", "IMG:doppler_dvt"),
    (r"(?<![가-힣])장\s?폐색|소장\s?폐색|small bowel obstruction|(?<![a-z])sbo(?![a-z])|bowel obstruction", "IMG:sbo_imaging"),
    (r"담석증|cholelithiasis|gallstones?", "IMG:gallstone"),
    (r"총담관\s?결석|choledocholithiasis", "IMG:cbd_stone"),
    (r"요로\s?결석|요관\s?결석|신장?\s?결석|urolithiasis|nephrolithiasis|ureterolithiasis", "IMG:stone_imaging"),
    (r"복부\s?대동맥류|abdominal aortic aneurysm|(?<![a-z])aaa(?![a-z])", "IMG:aaa_imaging"),
    (r"심장\s?눌림증?|심낭\s?압전|tamponade", "IMG:echo_tamponade"),
    (r"뇌\s?출혈|뇌내\s?출혈|intracerebral hemorrhage|(?<![a-z])ich(?![a-z])", "IMG:ct_ich"),
    (r"뇌\s?경색|cerebral infarct\w*|ischemic stroke|acute stroke", "IMG:stroke_imaging"),
    (r"천공|perforat\w*", "IMG:free_air"),
    (r"간세포암|hepatocellular carcinoma|(?<![a-z])hcc(?![a-z])", "IMG:hcc_imaging"),
    (r"폐암|lung cancer|bronchogenic carcinoma", "IMG:lung_mass"),
    (r"자궁\s?외\s?임신|ectopic pregnancy", "IMG:us_no_iup"),
    (r"고환\s?염전|testicular torsion", "IMG:testis_no_flow"),
    (r"장\s?중첩증?|intussusception", "IMG:intussusception_sign"),
    (r"(?<![가-힣])염전|volvulus", "IMG:volvulus_sign"),
    (r"심비대|cardiomegaly|심부전|heart failure|폐부종|pulmonary edema", "IMG:cxr_hf"),
]
_DESC_RX = [(n, re.compile(p), m, [(re.compile(q), c) for q, c in quals], fx) for n, p, m, quals, fx in _DESCRIPTORS]
# descriptors whose qualifier may sit anywhere in the sentence ("S상 결장 게실 다수, 주위 염증 소견 없음")
_QUAL_SENTENCE = {"inflammation"}
# testicular blood flow: the words after "혈류" decide (소실/감소 -> torsion finding present; 정상/증가 -> absent)
_FLOW = re.compile(r"혈류|(?:blood )?flow")
_FLOW_LOST = re.compile(r"^\s?(?:가|는|이|도)?\s?(?:소실|감소|저하|없|관찰되지\s?않|보이지\s?않|확인되지\s?않|(?:is )?(?:absent|decreased|diminished|not seen))")
_FLOW_KEPT = re.compile(r"^\s?(?:가|는|이|도)?\s?(?:정상|유지|보존|증가|양호|대칭|(?:is )?(?:normal|preserved|increased|symmetric))")
_FLOW_LOST_BEFORE = re.compile(r"(?:absent|no|decreased|diminished|무)\s?$")
_IMP_RX = [(re.compile(p), c) for p, c in _IMPRESSIONS]

# history-type concepts the lexicon gives for report words, re-mapped in imaging / ECG context
_HX_REMAP_SPAN = [
    ("HX:stroke", re.compile(r"출혈|hemorrh|bleed"), "IMG:ct_ich"),
    ("HX:stroke", re.compile(r"."), "IMG:stroke_imaging"),
    ("HX:prior_vte", re.compile(r"폐|pulmonary|(?<![a-z])pe(?![a-z])"), "IMG:ctpa_pe"),
    ("HX:prior_vte", re.compile(r"."), "IMG:doppler_dvt"),
    ("HX:pregnancy", re.compile(r"자궁|intrauterine|임신낭|태낭|gestational"), "IMG:intrauterine_pregnancy"),
    ("HX:heart_failure", re.compile(r"좌심실|구혈|박출|(?<![a-z])ef(?![a-z])|ventric|systolic"), "IMG:lvef_low"),
    ("HX:atrial_fibrillation", re.compile(r"."), "ECG:ecg_af"),
    ("SIGN:ascites", re.compile(r"fluid|액체"), "IMG:free_fluid"),
    ("SYM:heartburn", re.compile(r"regurgitation|역류"), "IMG:valve_regurgitation"),
]
_DROP_IN_IMAGING = {"HX:menopause", "HX:pmh"}
# patient-level concepts an imaging report may name by accident ("반점상 경화" is not a rash, "폐야 깨끗" is not
# auscultation): in imaging reports only these SIGN concepts are kept and SYM concepts are dropped
_SIGN_IN_IMAGING = {"SIGN:lymphadenopathy", "SIGN:fracture", "SIGN:ascites", "SIGN:hepatomegaly", "SIGN:splenomegaly",
                    "SIGN:abdominal_mass", "SIGN:edema", "SIGN:neck_mass", "SIGN:pulsatile_mass"}

# abnormal report wording that should have mapped to something (left over -> concept "")
_ABNORMAL_WORD = re.compile(
    r"병변|lesion|음영|opacit|종괴|mass|결절|nodul|출혈|hemorrh|비후|thicken|확장(?!기)|dilat|협착|stenos|석회|calcif|낭종|낭성|cyst"
    r"|부종|edema|위축|atroph|농양|abscess|혈전|thromb|폐색|occlu|골절|fractur|파괴|destruct|용해|lytic|조영\s?증강|enhanc"
    r"|고신호|저신호|hyperintens|hypointens|저음영|고음영|hypodens|hyperdens|hypoecho|hyperecho|저에코|고에코|결손|defect|변형|deform"
    r"|탈출|herniat|역류|regurg|허탈|collapse|협소|narrow|비대|enlarg|종대|기종|emphysem|섬유화|fibros|반흔|scar|퇴행성|degenerat"
    r"|염증|inflamm|stranding|천공|perforat|궤양|ulcer|용종|polyp|공동|cavit|폐쇄|obstruct|이상\s?소견|abnormal")

# ------------------------------------------------------------------------------------------------ cue vocabulary
_HEDGE = re.compile(
    r"가능성|의심|의증|추정|시사|(?<![a-z])r/o(?![a-z])|rule out|배제\s?(?:필요|요함|위해|를\s?위해|목적)|감별\s?(?:필요|요함|진단)"
    r"|불확실|확실하지\s?않|애매|명확하지\s?않|모호|(?<![a-z])(?:possible|possibly|probable|probably|likely|unlikely|suspected"
    r"|suspicious|suspicion|questionable|equivocal|indeterminate|presumed|presumably)(?![a-z])|suggest(?:s|ive|ing)?"
    r"|concerning for|may (?:represent|be|reflect)|could (?:represent|be|reflect)|cannot rule out|(?<![a-z])(?:vs\.?|versus)(?![a-z])"
    r"|differential|to exclude|to rule out|evaluate for")
_CANNOT_EXCLUDE = re.compile(
    r"배제(?:할|하기|하지)?\s?(?:수\s?(?:는\s?)?없|어렵|힘들|못)|배제되지\s?않|배제\s?(?:불가|안\s?됨)|완전히\s?배제되지"
    r"|cannot (?:be )?(?:entirely |completely |definitely )?(?:excluded|ruled out|exclude|rule out)|can't be (?:excluded|ruled out)"
    r"|could not be (?:excluded|ruled out)|not (?:be )?(?:entirely |completely )?(?:excluded|ruled out)")
_EXCLUDED = re.compile(r"배제\s?(?:됨|되었|되어|됐|됩니다|되었음|함|했|하였|완료)|(?:is|was|were|are|has been|have been)\s(?:\w+\s)?"
                       r"(?:excluded|ruled out)")
_DEFINITE = re.compile(r"합당|부합|일치|확인됨|확인되었|consistent with|compatible with|diagnostic of|confirm")

# comparison with a prior study
_CMP_CTX = re.compile(r"이전|전과|비교|대비|지난|종전|prior|previous|compared|comparison|interval|since|전\s?검사|last (?:study|exam)")
_CMP = [
    ("resolved", re.compile(r"해소|사라(?:짐|졌|진)|resolved|resolution|no longer (?:seen|visible|present|evident)|cleared"), False),
    ("resolved", re.compile(r"소실(?:됨|되었|되어|됐|했|하였)?"), True),
    ("new", re.compile(r"새로\s?(?:생긴|발생|나타난|보이는)|새로운|(?<![a-z])new(?:ly)?(?![a-z])|interval (?:development|appearance)"), False),
    ("stable", re.compile(r"unchanged|(?<![a-z])stable(?![a-z])|no (?:significant |interval |definite )*change(?:s)?(?: in)?|similar"), False),
    ("stable", re.compile(r"(?:큰\s?)?(?:변화|변동|차이)(?:가|는)?\s?(?:없|거의\s?없)|유사|동일"), True),
    ("improved", re.compile(r"호전|개선|improv\w*|resolving|less prominent"), False),
    ("improved", re.compile(r"감소(?:함|됨|되었|했|하였|된|한)?|줄어|decreas\w*|smaller"), True),
    ("worsened", re.compile(r"악화|worse\w*|progress(?:ed|ion|ing)"), False),
    ("worsened", re.compile(r"증가(?:함|됨|되었|했|하였|된|한)?|늘어|진행(?:됨|되었|했|하였|된|한)|increas\w*|enlarg\w*|larger"), True),
]
# comparison phrases that look like negations: masked before parsing (only with a comparison context for Korean)
_MASK_EN = re.compile(r"no (?:significant |interval |definite )*change(?:s)?(?: in(?: the)?(?: size of(?: the)?)?)?|without (?:significant |interval )?change")
_MASK_KO = re.compile(r"(?:큰\s?)?(?:변화|변동|차이)(?:가|는|를)?\s?(?:없|거의\s?없)\w*")

# report sections
_HEADER = re.compile(
    r"(?:^|(?<=[\n.;]))\s*(?P<h>findings?|impressions?|conclusions?|recommendations?|comparisons?|techniques?|clinical (?:history|information|indication|data)"
    r"|history|indications?|reason for (?:exam|study|examination)|opinion|소견|판독\s?소견|판독|결론|인상|권고\s?사항|권고|제언|추천|비교\s?검사|비교"
    r"|검사\s?방법|기법|임상\s?(?:정보|소견|병력|정보\s?및\s?의뢰\s?사유)|의뢰\s?(?:사유|내용)|검사\s?목적|참고\s?사항|참고|결과)\s*[:：]")
_KEEP_HEADERS = re.compile(r"^(?:findings?|impressions?|conclusions?|opinion|소견|판독\s?소견|판독|결론|인상|결과)$")
_RECOMMEND = re.compile(
    r"권고|권유|권장|추천|요망|바랍니다|recommend\w*|advis\w*|suggest(?:ed)?\s(?:follow|correl|further|clinical)|follow-?up"
    r"|추적\s?(?:검사|관찰|ct|mri|촬영|영상|초음파)|임상적\s?(?:연관|상관|판단|correl)|clinical(?:ly)? correlat\w*|correlate clinically"
    r"|further (?:evaluation|imaging|assessment)|추가\s?(?:검사|평가|촬영|영상)|필요\s?시|if clinically")

# whole-normal statements (a comma part holding nothing else, after an optional "name:" prefix)
_NORMAL_PART = re.compile(
    r"^(?:[^:]{0,30}:\s?)?(?:(?:전반적으로|전체적으로|그\s?외|기타|나머지는?)\s?)?"
    r"(?:정상(?:\s?(?:소견|범위(?:\s?내)?|심전도|동리듬|동율동|동성\s?리듬|흉부\s?x선|입니다|임|이다|으로\s?보임|적인\s?소견))?"
    r"|특이\s?(?:소견|사항|이상|병변)(?:은|이|는)?\s?(?:없|관찰되지\s?않|보이지\s?않|확인되지\s?않)\S*"
    r"|이상\s?(?:소견|병변)?(?:은|이|는)?\s?(?:없|관찰되지\s?않|보이지\s?않)\S*|(?:급성\s?)?(?:이상|병변)\s?없\S*"
    r"|no (?:acute |significant |active |focal |gross |definite |evidence of )*(?:abnormalit(?:y|ies)|findings?|process|disease|pathology"
    r"|intracranial abnormality|intracranial (?:process|pathology)|cardiopulmonary (?:abnormality|process|disease))"
    r"(?: (?:is |are )?(?:seen|identified|detected|noted|demonstrated))?"
    r"|unremarkable(?: (?:study|examination|exam|ct|mri|radiograph|chest radiograph))?|(?:the )?lungs? (?:are |is )?clear"
    r"|normal(?: (?:study|examination|exam|findings|chest (?:x-ray|radiograph)|radiograph|ct|mri|ultrasound|echocardiogram|ecg|ekg"
    r"|sinus rhythm|head ct|brain mri))?|within normal limits|(?<![a-z])wnl(?![a-z])|negative(?: study| examination)?"
    r"|(?<![a-z])nsr(?![a-z]))[\s.·]*$")
_NORMAL_TAIL = re.compile(r"(?:^|(?:^|\s)[가-힣]{1,8}(?:에|은|는|이|가|에서|에서는|상|도|와|과)\s)(?:모두\s|전반적으로\s)?(?:정상(?:\s?(?:소견|범위|임|입니다|이다))?"
                          r"|(?:특이|이상)\s?(?:소견|사항)?(?:은|이|는)?\s?(?:없음|없습니다|없다|없었음|관찰되지\s?않음)"
                          r"|unremarkable|within normal limits|normal)[\s.]*$")
_EXCEPT = re.compile(r"외(?:에|에는|엔)?\s|제외|이외|except|other than|apart from|aside from|besides")
_PRE_NORMAL = re.compile(r"(?:(?<![a-z])normal|정상(?:적인)?)\s(?:[a-z]+\s)?$")

_LAT_EXTRA = [("right", re.compile(r"우(?:상|중|하)엽|(?<![a-z])(?:rul|rml|rll)(?![a-z])|우(?:측)?\s?(?:신장|난소|부속기|고환|폐)")),
              ("left", re.compile(r"좌(?:상|하)엽|설엽|lingula|(?<![a-z])(?:lul|lll)(?![a-z])|좌(?:측)?\s?(?:신장|난소|부속기|고환|폐)"))]
_SITE_WORDS = re.compile(
    r"우상엽|우중엽|우하엽|좌상엽|좌하엽|설엽|(?:right|left) (?:upper|middle|lower) lobe|lingula|폐첨부|apex|apical|기저부|basal|base|폐문|hilum|hilar"
    r"|전두엽|측두엽|두정엽|후두엽|소뇌|기저핵|시상|뇌간|basal cisterns?|frontal|temporal|parietal|occipital|cerebell\w*"
    r"|담낭|총담관|간내 담관|간 우엽|간 좌엽|충수|맹장|회장|결장|직장|s상 결장|sigmoid|신장|요관|방광|췌장 두부|췌장|비장|난소|부속기|자궁|고환"
    r"|대퇴정맥|슬와정맥|상행 대동맥|하행 대동맥|복부 대동맥|폐동맥|좌심실|우심실|승모판|대동맥판|심낭|전벽|하벽|측벽|중격"
    r"|appendix|gallbladder|common bile duct|kidney|ureter|pancrea\w*|liver|spleen|ovary|adnexa|testis|femoral vein|popliteal vein"
    r"|ascending aorta|descending aorta|pulmonary arter\w*|left ventricle|right ventricle|mitral valve|aortic valve|pericardium")

# ------------------------------------------------------------------------------------------------ critical results
_CRITICAL_CONCEPTS = {
    "IMG:free_air", "IMG:ct_dissection", "IMG:cxr_mediastinum", "IMG:ct_sah", "IMG:ct_ich", "IMG:subdural_hematoma",
    "IMG:mass_effect", "IMG:cxr_ptx", "IMG:ctpa_pe", "IMG:echo_tamponade", "IMG:testis_no_flow", "IMG:us_no_iup",
    "IMG:volvulus_sign", "IMG:intussusception_sign", "IMG:stroke_imaging", "IMG:cvst_imaging", "IMG:aaa_imaging",
    "IMG:sbo_imaging", "IMG:doppler_dvt", "IMG:echo_vegetation", "ECG:ecg_stemi", "ECG:ecg_brugada", "ECG:ecg_long_qt",
    "LAB:troponin_high", "LAB:blood_culture_pos",
}
# concept -> (comparison, limit): the value makes the finding critical (adult limits; see the module docstring)
_CRITICAL_VALUES: dict[str, tuple[str, float]] = {
    "LAB:k_high": (">=", 6.0), "LAB:k_low": ("<", 2.8), "LAB:na_low": ("<", 120), "LAB:na_high": (">", 160),
    "LAB:glucose_low": ("<", 50), "LAB:glucose_high": (">", 450), "LAB:hb_low": ("<", 7.0), "LAB:plt_low": ("<", 20000),
    "LAB:wbc_low": ("<", 2000), "LAB:wbc_high": (">", 30000), "LAB:inr_high": (">=", 5.0), "LAB:hco3_low": ("<", 10),
    "SIGN:hypotension": ("<", 90), "SIGN:hypoxemia": ("<", 90), "SIGN:tachypnea": (">=", 30), "SIGN:bradypnea": ("<=", 8),
}
_CRITICAL_ADULT_ONLY = {"SIGN:tachycardia": (">=", 130), "SIGN:bradycardia": ("<", 40)}


def _beyond(v: float, op: str, lim: float) -> bool:
    return {"<": v < lim, "<=": v <= lim, ">": v > lim, ">=": v >= lim}[op]


# Korean names of the kb_tests disease keys linked from imaging / ECG / lab results (display only)
_DX_KO = {
    "aaa": "복부대동맥류", "acromegaly": "말단비대증", "acute_pancreatitis": "급성 췌장염", "af": "심방세동", "als": "근위축성 측삭경화증",
    "ami": "급성 심근경색", "aortic_dissection": "대동맥 박리", "aortic_stenosis": "대동맥판 협착증", "appendicitis": "급성 충수염",
    "as": "대동맥판 협착증", "aspergilloma": "아스페르길루스종", "asthma": "천식", "bacterial_pneumonia": "세균성 폐렴",
    "brugada": "브루가다 증후군", "cerebral_infarction": "뇌경색", "chf": "울혈성 심부전", "cholangiocarcinoma": "담관암",
    "cholangitis": "담관염", "cholecystitis": "급성 담낭염", "choledocholithiasis": "총담관결석", "cholelithiasis": "담석증",
    "chronic_pancreatitis": "만성 췌장염", "cirrhosis": "간경변", "ckd": "만성 콩팥병", "copd": "만성 폐쇄성 폐질환",
    "covid": "코로나19", "cppd": "칼슘 피로인산염 침착증", "crc": "대장암", "crohn": "크론병", "cvst": "뇌정맥동 혈전증",
    "dcm": "확장성 심근병증", "dermatomyositis": "피부근염", "diverticulitis": "게실염", "duodenal_ulcer": "십이지장궤양",
    "ectopic": "자궁외 임신", "encephalitis": "뇌염", "endocarditis": "감염성 심내막염", "epilepsy": "뇌전증",
    "gastric_cancer": "위암", "gastric_ulcer": "위궤양", "gbs": "길랭-바레 증후군", "gerd": "위식도 역류질환",
    "gpa": "육아종증 다발혈관염", "graves": "그레이브스병", "hcc": "간세포암", "hcm": "비후성 심근병증",
    "hemorrhagic_stroke": "출혈성 뇌졸중", "hf": "심부전", "hp_pneumonitis": "과민성 폐렴", "hydronephrosis": "수신증",
    "hyperthyroidism": "갑상선기능항진증", "ie": "감염성 심내막염", "ild": "간질성 폐질환", "intussusception": "장중첩증",
    "ipf": "특발성 폐섬유증", "ischemic_stroke": "허혈성 뇌졸중", "kawasaki": "가와사키병", "long_qt": "QT 연장 증후군",
    "lung_abscess": "폐농양", "lung_cancer": "폐암", "mg": "중증근무력증", "mi": "심근경색", "mitral_stenosis": "승모판 협착증",
    "ms": "다발성 경화증", "nephrolithiasis": "신장결석", "nph": "정상압 수두증", "obstruction": "장폐색",
    "osteomyelitis": "골수염", "pancreatic_cancer": "췌장암", "pancreatitis": "췌장염", "pcos": "다낭성 난소 증후군",
    "pcp": "폐포자충 폐렴", "pe": "폐색전증", "perforation": "위장관 천공", "pericarditis": "심낭염",
    "pituitary_adenoma": "뇌하수체 선종", "pneumonia": "폐렴", "pneumothorax": "기흉", "polymyositis": "다발근염",
    "prolactinoma": "프로락틴종", "psc": "원발성 경화성 담관염", "ptb": "폐결핵", "pud": "소화성 궤양",
    "pulm_htn": "폐고혈압", "pyelonephritis": "급성 신우신염", "reflux_esophagitis": "역류성 식도염", "sah": "지주막하 출혈",
    "sarcoidosis": "사르코이드증", "subacute_thyroiditis": "아급성 갑상선염", "svt": "발작성 상심실성 빈맥",
    "takayasu": "다카야스 동맥염", "tamponade": "심장눌림증", "tb": "결핵", "testicular_torsion": "고환 염전",
    "thrombosis": "혈전증", "uc": "궤양성 대장염", "unstable_angina": "불안정 협심증", "urolithiasis": "요로결석",
    "varices": "정맥류", "viral_encephalitis": "바이러스 뇌염", "viral_pneumonia": "바이러스 폐렴", "volvulus": "장염전",
    "wernicke": "베르니케 뇌병증", "wilson": "윌슨병", "wpw": "WPW 증후군",
}

_POL_KO = {"present": "있음", "absent": "없음", "uncertain": "의심(불확실)"}
_LAT_KO = {"right": "우측", "left": "좌측", "bilateral": "양측"}
_DIR_KO = {"high": "높음", "low": "낮음", "normal": "정상"}
_CMP_KO = {"new": "새로 생김", "improved": "이전보다 호전", "worsened": "이전보다 악화", "stable": "이전과 변화 없음",
           "resolved": "이전 소견 소실"}
_POL_RANK = {"present": 3, "uncertain": 2, "absent": 1}


# ------------------------------------------------------------------------------------------------ helpers
def _short_label(cid: str) -> str:
    c = LEXICON.concept(cid)
    lab = (c.ko or c.en) if c else cid
    return lab.split(":", 1)[1].strip() if ":" in lab else lab


def _kind_of(cid: str, test_kind: str) -> str:
    pre = cid.split(":")[0] if cid else ""
    if pre == "LAB":
        return "lab"
    if pre == "ECG":
        return "ecg"
    if pre == "IMG":
        return "ecg" if test_kind == "ecg" and cid == "IMG:lvh" else "imaging"
    if pre in ("SIGN", "SYM", "QUAL"):
        return "exam"
    return test_kind if test_kind in KINDS else "other"


def _supports(cid: str) -> tuple[tuple[str, str, int], ...]:
    """Diseases the finding supports (knowledge/kb_tests curated links). Supportive only: never a diagnosis."""
    c = LEXICON.concept(cid)
    if not c:
        return ()
    try:
        from doctor_agent.knowledge import kb_tests
    except Exception:  # noqa: BLE001 - optional
        return ()
    out: dict[str, int] = {}
    for k in c.kb:
        if not k.startswith("TF:"):
            continue
        f = kb_tests.BY_ID.get(k[3:])
        if not f:
            continue
        for dx, w, _r, _ref in f.parsed_links():
            out[dx] = max(out.get(dx, 0), w)
    ranked = sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return tuple((dx, _DX_KO.get(dx, dx), w) for dx, w in ranked)


def _kb_finding(cid: str):
    """(kb_tests Finding, compiled pattern) behind a lexicon concept, or (None, None)."""
    c = LEXICON.concept(cid)
    if not c:
        return None, None
    try:
        from doctor_agent.knowledge import kb_tests
    except Exception:  # noqa: BLE001
        return None, None
    for k in c.kb:
        if k.startswith("TF:"):
            for f, pat, _not_if in kb_tests._COMPILED:  # read-only static table
                if f.id == k[3:]:
                    return f, pat
    return None, None


def _kb_pattern(cid: str):
    """(compiled pattern, fixed) of the kb_tests finding behind a lexicon concept, or (None, False)."""
    f, pat = _kb_finding(cid)
    return (pat, f.fixed) if f is not None else (None, False)


def _part_of(parts: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    for a, b in parts:
        if a <= pos < b:
            return a, b
    return parts[-1] if parts else (0, 0)


def _laterality(sent: str, part: tuple[int, int], s: int, e: int) -> str:
    a, b = part
    before = sent[a:e]
    after = sent[e:b]

    def lats(t: str) -> list[tuple[int, str]]:
        out = [(m.start(), m.lastgroup) for m in F.LAT.finditer(t)]
        for name, rx in _LAT_EXTRA:
            out += [(m.start(), name) for m in rx.finditer(t)]
        return sorted(out)

    lb = lats(before)
    if lb:
        kinds = {k for _p, k in lb}
        if {"right", "left"} <= kinds and re.search(r"right and left|left and right|좌우|양", before):
            return "bilateral"
        return lb[-1][1]
    # a side named after the word belongs to it only in English ("effusion on the right"); in Korean it names the
    # next list item ("복수 및 좌측 흉수")
    la = lats(after) if re.search(r"[a-z]{3}", sent[s:e]) else []
    if la:
        kinds = {k for _p, k in la}
        if {"right", "left"} <= kinds:
            return "bilateral"
        return la[0][1]
    return ""


def _site(sent: str, part: tuple[int, int], s: int, e: int) -> str:
    a, b = part
    before = list(_SITE_WORDS.finditer(sent, a, e))
    if before:
        return before[-1].group()
    after = _SITE_WORDS.search(sent, e, b)
    return after.group() if after else ""


# English negation scope: "No A or B" negates both (the nlp layer allows 25 characters; reports run longer:
# "without sonographic evidence of acute cholecystitis"); "A without B" / "A with no B" does not negate A
_EN_PRE_NEG = re.compile(r"(?<![a-z])(?:no|without|negative for|free of|absence of|no evidence of)(?![a-z])"
                         r"(?:(?!(?<![a-z])(?:but|however|although|except|there is|there are)(?![a-z]))[^,;])*$")
_EN_POST_NOT_OURS = re.compile(r"^[^,;]{0,40}?(?<![a-z])(?:without|with no|but no|and no|no evidence of|negative for)(?![a-z])")


_EN_LIST_NEG = re.compile(r"^(?:there (?:is|are) )?(?:no|without|negative for)\s(?:(?!(?<![a-z])(?:but|however|with|which|is|are|was|were"
                          r"|seen|noted|present|shows?|demonstrates?)(?![a-z]))[^.;:])*$")
_EN_LIST_WORDS = re.compile(r"^(?:\s|,|(?<![a-z])(?:or|and|the|any|of|signs?)(?![a-z]))*$")


_STATED = re.compile(r"^[^,]*?(?:소견|관찰됨|관찰되며|보임|보이며|있음|동반)\s?(?:임|이\s?있음|보임|관찰됨|있음)?[\s.]*$")


def _own_statement(pol: str, cue: str, masked: str, part: tuple[int, int], e: int) -> str:
    """A list item that states itself ("복수 및 좌측 흉수 소견, 장기들은 전반적으로 정상") keeps its own (present)
    reading instead of the predicate of the next comma part."""
    a, b = part
    tail = masked[e:b]
    if pol == "absent" and cue.startswith("list:") and _STATED.match(tail) and F.first_cue(tail)[0] != "neg":
        return "present"
    return pol


def _english_scope(pol: str, masked: str, part: tuple[int, int], s: int, e: int) -> str:
    a, b = part
    head, tail = masked[a:s], masked[e:b]
    if not re.search(r"[a-z]{3}", masked[s:e]):
        return pol
    if pol == "present" and _EN_PRE_NEG.search(head):
        return "absent"
    # "No consolidation, effusion, or pneumothorax": a bare list item after a sentence-initial "no"
    if pol == "present" and _EN_LIST_NEG.search(masked[:s]) and _EN_LIST_WORDS.match(head) \
            and _EN_LIST_WORDS.match(tail.rstrip(". ")):
        return "absent"
    if pol == "absent" and _EN_POST_NOT_OURS.match(tail) and not _EN_PRE_NEG.search(head) \
            and not re.search(r"(?<![a-z])(?:no|not|none|negative|absent)(?![a-z])", tail[:_EN_POST_NOT_OURS.match(tail).start() + 1]):
        return "present"
    return pol


_PENDING = re.compile(r"대기\s?중|결과\s?(?:대기|미정|안\s?나)|검사\s?중|진행\s?중|(?<![a-z])pending(?![a-z])|not yet (?:available|resulted)"
                      r"|in progress|awaiting")
_VALUE_TAIL = re.compile(r"[^\d<>≤≥,]{0,14}?[<>≤≥]?=?\s*\d[\d,.]*\s*(?:%|[a-zμµ/.³^]+)?")
_REF_PAREN = re.compile(r"\s*\([^()]*\)")
_WORD_PAREN = re.compile(r"\s*\(([^()\d]{1,20})\)")  # "(경미한 상승)": a direction word, no reference numbers


def _kb_value(cid: str, masked: str, pol: str):
    """The value kb_tests read for a numeric finding: (polarity, value as printed, unit, direction, (start, end)) or
    None. The polarity is kb_tests' own: it compares the value with a printed reference range in that range's unit and
    knows which findings are absolute cut-offs ("ESR 38 (정상 <20)" is above the range, not ESR > 50), so no number is
    re-read here. One gap is bridged with kb_tests' own rule for a word-only range ("(정상 범위 초과)" -> above the range
    for a finding that is not a cut-off): a direction-word parenthesis without a reference keyword ("CEA 6.5 ng/mL
    (경미한 상승)"), which kb_tests does not take for a range."""
    f, pat = _kb_finding(cid)
    if f is None or f.mode not in ("hi", "lo"):
        return None
    m = pat.search(masked)
    if not m:
        return None
    try:
        from doctor_agent.knowledge import kb_tests
        num = kb_tests._number(masked[m.end():m.end() + 45], f)  # read-only helper: the number kb_tests reads
    except Exception:  # noqa: BLE001
        return None
    if num is None:
        return None
    _v, _cmp, raw, unit = num
    vm = _VALUE_TAIL.match(masked, m.end())
    if vm and re.match(r"\s*(?:[-~–:]\s*\d|\+)", masked[vm.end():]):
        return None  # a range ("0-5 /HPF"), a titre ("1:160") or a grade ("3+") is not one value: no value shown
    if not vm or re.search(r"[a-zμ]\d", masked[m.end():vm.end()]):
        return None  # the number belongs to another name ("IgG 및 C3"): shown without a value
    unit = unit.rstrip("/.^")
    end = vm.end() if vm else m.end()
    ref = _REF_PAREN.match(masked, end)
    word = _WORD_PAREN.match(masked, end)
    if pol == "absent" and word and not (f.cutoff or f.need_value) and not kb_tests._REFPAREN.match(word.group().strip()) \
            and not kb_tests._REF_NORMAL_NEG.search(word.group(1)):
        says_high, says_low = kb_tests._REF_HIGH.search(word.group(1)), kb_tests._REF_LOW.search(word.group(1))
        if (says_high and not says_low and f.mode == "hi") or (says_low and not says_high and f.mode == "lo"):
            pol = "present"
    direction = ("high" if f.mode == "hi" else "low") if pol == "present" else "normal" if pol == "absent" else ""
    return pol, raw, unit, direction, (m.start(), ref.end() if ref else end)


def _mask(t: str, rx: re.Pattern) -> str:
    return rx.sub(lambda m: "·" * len(m.group()), t)


def _sections(t: str) -> tuple[list[str], list[str]]:
    """(kept chunks, dropped chunks) of a normalised report by its section headers."""
    heads = list(_HEADER.finditer(t))
    if not heads:
        return [t], []
    kept, dropped = [], []
    pre = t[:heads[0].start()].strip(" \n.;")
    if pre:
        kept.append(pre)
    for i, h in enumerate(heads):
        body = t[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(t)].strip()
        if not body:
            continue
        name = re.sub(r"\s+", " ", h.group("h").strip())
        if _KEEP_HEADERS.match(name):
            kept.append(body)
            continue
        if not _RECOMMEND.search(name) and not re.match(r"권고|제언|추천|recommend", name):
            # "Comparison: 2026-09-01. Interval increase ...": a short header section ends with its first sentence
            # (dates are not sentence ends: "2026.09.01")
            m = re.search(r"(?<!\d)\.(?!\d)|\.(?=\s)|\n", body)
            if m and body[m.end():].strip():
                dropped.append(f"{name}: {body[:m.end()].strip()}")
                kept.append(body[m.end():].strip())
                continue
        dropped.append(f"{name}: {body}")
    return kept, dropped


# ------------------------------------------------------------------------------------------------ main
class _Reader:
    def __init__(self, test_name: str, text: str, ctx: dict | None):
        self.test_name = test_name or ""
        self.raw = text or ""
        self.t = normalize(self.raw)
        self.kind = kind_of_test(self.test_name, self.raw)
        self.region = _test_region(self.test_name)
        ctx = ctx or {}
        age = ctx.get("age_years")
        if not isinstance(age, (int, float)):
            age = F.age_from_text(ctx.get("initial_info") or "") if ctx.get("initial_info") else None
        self.age = float(age) if isinstance(age, (int, float)) else None
        self.source = "exam" if self.kind == "exam" else "test"
        self.interp = Interpretation(self.test_name, self.kind, text_len=len(self.raw))
        self.normal_seen = False

    # ---------------------------------------------------------------- per text
    def run(self) -> Interpretation:
        it = self.interp
        if not self.t.strip():
            it.unavailable = True
            return it
        if F.UNAVAILABLE.search(self.t) and len(self.t) < 80:
            it.unavailable = True
            return it
        it.serial = len(_SERIAL.findall(self.t)) >= 2
        chunks, dropped = _sections(self.t)
        it.ignored += dropped
        items: list[Item] = []
        for chunk in chunks:
            for a, b, _q in F.sentences(chunk):
                sent = normalize(chunk[a:b]).strip(" ,")
                if not sent or F.UNAVAILABLE.search(sent):
                    continue
                it.n_sentences += 1
                items += self._sentence(sent)
        items = self._merge(items)
        abnormal = [i for i in items if i.polarity in ("present", "uncertain")]
        if self.normal_seen and not abnormal:
            it.normal = True
            cid = {"imaging": "IMG:normal_study", "ecg": "ECG:normal_ecg"}.get(self.kind)
            if cid:
                items.append(Item(self.kind, cid, _short_label(cid), "present", span="정상", source="normal",
                                  confidence=0.9))
        for i in items:
            self._finish(i)
        it.items = items
        it.llm_reasons = llm_reasons(it)
        return it

    # ---------------------------------------------------------------- per sentence
    def _sentence(self, sent: str) -> list[Item]:
        cls = F.clauses(sent)
        parts = [p for c in cls for p in c.parts]
        # recommendation parts (and everything after them) are not findings
        for idx, (a, b) in enumerate(parts):
            if _RECOMMEND.search(sent[a:b]):
                self.interp.ignored.append(sent[a:].strip(" ,"))
                sent = sent[:a].rstrip(" ,")
                if not sent:
                    return []
                cls = F.clauses(sent)
                parts = [p for c in cls for p in c.parts]
                break
        # results still pending ("아질산염: 대기 중") are not results
        pending = [(a, b) for a, b in parts if _PENDING.search(sent[a:b])]
        if pending:
            self.interp.pending = True
            self.interp.ignored += [sent[a:b].strip(" ,") for a, b in pending]
        has_ctx = bool(_CMP_CTX.search(sent))
        masked = _mask(sent, _MASK_EN)
        if has_ctx:
            masked = _mask(masked, _MASK_KO)
        for a, b in parts:
            ptext = sent[a:b].strip(" ,")
            # imaging reports go organ by organ: "충수는 정상" is not a normal study, only a whole-normal phrase is;
            # exam / lab texts may name their subject ("심장에 이상 소견 없음", "전신 감각 정상")
            organ_ok = self.kind not in ("imaging", "ecg")
            if _NORMAL_PART.match(ptext) or (organ_ok and _NORMAL_TAIL.search(ptext) and not _EXCEPT.search(ptext)
                                             and len(ptext) <= 40 and not re.search(r"\d", ptext)):
                self.normal_seen = True
        regions = _regions_in(sent)
        ctx = {"age_years": self.age} if self.age is not None else None
        out: list[Item] = []
        taken: list[tuple[int, int]] = []
        imaging = self.kind in ("imaging", "ecg")
        for f in F.parse(masked, self.source, ctx):
            cid = f.concept
            s, e = f.start, f.end
            pol = f.polarity
            cue = f.cue
            src = "value" if f.value is not None else "lexicon"
            if f.cue.startswith("kb_tests"):
                src = "kb_tests"
                if cid.startswith(("IMG:", "ECG:")) and f.cue == "kb_tests":  # keyword reading, not a measured value
                    pat, fixed = _kb_pattern(cid)
                    m = pat.search(masked) if pat else None
                    if m:
                        s, e = m.start(), m.end()
                        if fixed:
                            pol = "present"
                        else:
                            a = F.assess_spans(masked, [(s, e)], self.source)[0]
                            pol, cue = (a.polarity, a.cue) if a is not None else (pol, cue)
                elif cid.startswith("LAB:") and f.cue == "kb_tests-value":
                    # polarity: kb_tests' own (value vs a printed range in its unit, cut-offs); the value is display
                    got = _kb_value(cid, masked, pol)
                    if got:
                        pol, value, unit, direction, (s, e) = got
                        f = F.Finding(**{**f.__dict__, "value": value, "unit": unit, "direction": direction})
                        # kb_tests reads whole clauses: a clause with a pending part stays skipped, as before
                        if any(a <= f.start < b for a, b in pending):
                            continue
            if any(a <= s < b for a, b in pending):
                continue
            if imaging and cid in _DROP_IN_IMAGING:
                continue
            if self.kind == "imaging" and (cid.startswith(("SYM:", "QUAL:"))
                                           or (cid.startswith("SIGN:") and cid not in _SIGN_IN_IMAGING)):
                continue
            if imaging or self.kind == "other":
                cid = self._remap(cid, sent[s:e], sent, parts, s, e, regions)
            if f.hypothetical and pol == "present":
                pol = "uncertain"
            out.append(Item(_kind_of(cid, self.kind), cid, _short_label(cid), pol, span=sent[s:e], value=f.value,
                            unit=f.unit, direction=f.direction, source=src, confidence=f.confidence,
                            laterality=f.laterality))
            out[-1]._pos = (s, e)  # type: ignore[attr-defined]
            out[-1]._cue = cue  # type: ignore[attr-defined]
            taken.append((s, e))
            for xs, xe in f.extra_spans:
                taken.append((xs, xe))
        if imaging:
            taken += pending
            out += self._descriptors(sent, masked, parts, regions, taken)
            out += self._unmapped(sent, masked, parts, taken)
        starts = [getattr(i, "_pos", (0, 0))[0] for i in out]
        for i in out:
            s, e = getattr(i, "_pos", (0, 0))
            part = _part_of(parts, s)
            if i.value is None:
                i.polarity = _own_statement(i.polarity, getattr(i, "_cue", ""), masked, part, e)
                i.polarity = _english_scope(i.polarity, masked, part, s, e)
            # comparison words may follow in later comma parts that name no other finding ("우하엽 5mm 결절, 이전
            # CT와 비교하여 변화 없음", "small left effusion, improved compared to prior")
            j = parts.index(part) if part in parts else len(parts) - 1
            end = part[1]
            while j + 1 < len(parts) and not any(parts[j + 1][0] <= x < parts[j + 1][1] for x in starts):
                j += 1
                end = parts[j][1]
            self._modifiers(i, sent, part, s, e, has_ctx, sent[part[0]:end])
        return out

    def _remap(self, cid: str, span: str, sent: str, parts, s: int, e: int, regions) -> str:
        for src, rx, dst in _HX_REMAP_SPAN:
            if cid == src and rx.search(span):
                if src == "HX:prior_vte":
                    reg = _region_at(sent, _part_of(parts, s), s, e, self.region, regions)
                    if reg in ("vein",):
                        return "IMG:doppler_dvt"
                    if reg in ("pulm_artery", "lung"):
                        return "IMG:ctpa_pe"
                return dst
        if cid in ("SIGN:edema",):
            reg = _region_at(sent, _part_of(parts, s), s, e, self.region, regions)
            _n, _rx, mapping, _q, _fx = next(d for d in _DESC_RX if d[0] == "edema")
            return mapping.get(reg, cid)
        return cid

    def _descriptors(self, sent: str, masked: str, parts, regions, taken: list[tuple[int, int]]) -> list[Item]:
        out: list[Item] = []
        cands: list[tuple[int, int, str, bool, str]] = []  # start, end, concept, fixed, source
        for rx, cid in _IMP_RX:
            for m in rx.finditer(masked):
                cands.append((m.start(), m.end(), cid, False, "impression"))
        for name, rx, mapping, quals, fixed in _DESC_RX:
            for m in rx.finditer(masked):
                part = _part_of(parts, m.start())
                ptext = sent[part[0]:part[1]]
                cid = next((c for q, c in quals if q.search(ptext)), None)
                if cid is None and name in _QUAL_SENTENCE:
                    cid = next((c for q, c in quals if q.search(sent)), None)
                if cid is None:
                    reg = _region_at(sent, part, m.start(), m.end(), self.region, regions)
                    cid = mapping.get(reg) or mapping.get("*")
                if cid:
                    cands.append((m.start(), m.end(), cid, fixed, "descriptor"))
        flow_pol: dict[tuple[int, int], str] = {}
        for m in _FLOW.finditer(masked):
            part = _part_of(parts, m.start())
            if _region_at(sent, part, m.start(), m.end(), self.region, regions) != "testis":
                continue
            after = masked[m.end():m.end() + 20]
            if _FLOW_LOST.match(after) or _FLOW_LOST_BEFORE.search(masked[max(0, m.start() - 12):m.start()]):
                flow_pol[(m.start(), m.end())] = "present"
            elif _FLOW_KEPT.match(after):
                flow_pol[(m.start(), m.end())] = "absent"
            else:
                continue
            cands.append((m.start(), m.end(), "IMG:testis_no_flow", True, "descriptor"))
        cands.sort(key=lambda c: (c[0], -(c[1] - c[0])))
        for s, e, cid, fixed, src in cands:
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            if (s, e) in flow_pol:
                pol = flow_pol[(s, e)]
            elif fixed:
                pol = "present"
            else:
                a = F.assess_spans(masked, [(s, e)], self.source)[0]
                if a is None:
                    continue
                pol = a.polarity
                cue = a.cue
                if a.hypothetical and pol == "present":
                    pol = "uncertain"
                if pol == "present" and _PRE_NORMAL.search(masked[max(0, s - 20):s]):
                    pol = "absent"  # "normal heart size", "정상 심장 크기"
            it = Item(_kind_of(cid, self.kind), cid, _short_label(cid), pol, span=sent[s:e], source=src,
                      confidence=0.75)
            it._pos = (s, e)  # type: ignore[attr-defined]
            it._cue = cue if not fixed and (s, e) not in flow_pol else "fixed"  # type: ignore[attr-defined]
            out.append(it)
        return out

    def _unmapped(self, sent: str, masked: str, parts, taken) -> list[Item]:
        out: list[Item] = []
        for a, b in parts:
            ptext = sent[a:b].strip(" ,")
            if not ptext or _NORMAL_PART.match(ptext):
                continue
            for m in _ABNORMAL_WORD.finditer(masked, a, b):
                if any(m.start() < te and ts < m.end() for ts, te in taken):
                    continue
                # "특이 소견 없음" / "이상 소견 없음" inside a longer part is not an abnormal word
                if m.group().startswith(("이상", "abnormal")) and F.first_cue(masked[m.end():b])[0] == "neg":
                    continue
                f = F.assess_spans(masked, [(m.start(), m.end())], self.source)[0]
                pol = f.polarity if f is not None else "present"
                if f is not None and f.hypothetical and pol == "present":
                    pol = "uncertain"
                label = re.sub(r"\s+", " ", ptext)[:40]
                it = Item(self.kind, "", label, pol, span=sent[m.start():m.end()], source="unmapped", confidence=0.5)
                it._pos = (m.start(), m.end())  # type: ignore[attr-defined]
                out.append(it)
                taken.append((a, b))
                break
        return out

    def _modifiers(self, i: Item, sent: str, part: tuple[int, int], s: int, e: int, has_ctx: bool,
                   cmp_scope: str = "") -> None:
        a, b = part
        ptext = sent[a:b]
        if i.value is None:
            if not i.laterality:
                i.laterality = _laterality(sent, part, s, e)
            if i.kind in ("imaging", "ecg") and i.concept not in ("IMG:normal_study", "ECG:normal_ecg"):
                i.site = _site(sent, part, s, e)
        # hedges / exclusion words of the comma part (a list item without its own predicate shares the next part's)
        scope = ptext
        if not re.search(r"없|않|있|보임|관찰|확인|소견|의심|가능성|(?<![a-z])(?:no|not|seen|noted|present)(?![a-z])", ptext):
            nxt = sent[b:b + 40].split(",")[0] if b < len(sent) else ""
            scope = ptext + " " + nxt
        if _CANNOT_EXCLUDE.search(scope):
            i.polarity, i.hedge = "uncertain", _CANNOT_EXCLUDE.search(scope).group()
        elif _EXCLUDED.search(scope) and not _HEDGE.search(scope.replace(_EXCLUDED.search(scope).group(), "")):
            i.polarity, i.hedge = "absent", ""
        else:
            h = _HEDGE.search(scope)
            if h and i.polarity == "present" and i.value is None and not (_DEFINITE.search(scope) and not re.search(
                    r"가능성|의심|possible|probable|likely|suspicious|r/o", h.group())):
                i.polarity, i.hedge = "uncertain", h.group()
            elif i.polarity == "uncertain" and not i.hedge:
                i.hedge = (h.group() if h else "hypothetical")
        # comparison with a prior study
        for name, rx, needs_ctx in _CMP:
            m = rx.search(cmp_scope or ptext)
            if m and (has_ctx or not needs_ctx):
                if name == "resolved" and i.polarity == "present":
                    i.polarity = "absent"  # "이전 검사와 비교하여 우하엽 경화는 소실됨"
                i.comparison = name
                break
        if i.polarity == "uncertain":
            i.confidence = min(i.confidence, 0.6)

    # ---------------------------------------------------------------- merge + finish
    def _merge(self, items: list[Item]) -> list[Item]:
        out: list[Item] = []
        for i in items:
            if not i.concept:
                out.append(i)
                continue
            same = [o for o in out if o.concept == i.concept and (o.value == i.value or i.value is None or o.value is None)]
            hit = None
            for o in same:
                if o.laterality == i.laterality or not o.laterality or not i.laterality:
                    hit = o
                    break
            if hit is None:
                out.append(i)
                continue
            if _POL_RANK[i.polarity] > _POL_RANK[hit.polarity]:
                hit.polarity, hit.hedge, hit.span = i.polarity, i.hedge, i.span
            hit.laterality = hit.laterality or i.laterality
            hit.site = hit.site or i.site
            hit.comparison = hit.comparison or i.comparison
            if hit.value is None and i.value is not None:
                hit.value, hit.unit, hit.direction = i.value, i.unit, i.direction
        return out

    def _finish(self, i: Item) -> None:
        if i.polarity in ("present", "uncertain") and i.concept:
            if i.concept in _CRITICAL_CONCEPTS:
                i.critical = True
            lim = _CRITICAL_VALUES.get(i.concept)
            if lim is None and (self.age is None or self.age >= 18):
                lim = _CRITICAL_ADULT_ONLY.get(i.concept)
            if lim and i.value is not None:
                v = i.value * 18 if i.concept.startswith("LAB:glucose") and i.unit.startswith("mmol") else i.value
                i.critical = _beyond(v, *lim)
            if i.polarity == "present" and i.concept.startswith(("IMG:", "ECG:")):
                i.supports = _supports(i.concept)
        i.summary_ko = _summary(i)


def _summary(i: Item) -> str:
    loc = ", ".join(x for x in (_LAT_KO.get(i.laterality, ""), i.site if i.kind in ("imaging", "ecg") else "") if x)
    head = ("⚠" if i.critical else "") + i.label + (f"({loc})" if loc else "")
    if i.concept == "":
        return f"{head}: {_POL_KO[i.polarity]}(미분류 소견)"
    if i.concept in ("IMG:normal_study", "ECG:normal_ecg"):
        return f"{head}"
    s = f"{head}: {_POL_KO[i.polarity]}"
    if i.value is not None:
        v = f"{i.value:g}{(' ' + i.unit) if i.unit else ''}"
        s += f" ({v}{', ' + _DIR_KO[i.direction] if i.direction in _DIR_KO else ''})"
    if i.comparison:
        s += f", {_CMP_KO[i.comparison]}"
    return s


# ------------------------------------------------------------------------------------------------ public API
def interpret(test_name: str, result_text: str, patient_ctx: dict | None = None) -> Interpretation:
    """Structured reading of one result text. Never raises (a failure returns an Interpretation with no items and
    llm_reasons=["error"])."""
    try:
        return _Reader(test_name, result_text, patient_ctx).run()
    except Exception as e:  # noqa: BLE001 - the agent must keep running
        it = Interpretation(test_name or "", "other", text_len=len(result_text or ""))
        it.llm_reasons = [f"error:{type(e).__name__}"]
        return it


def _short(i: Item) -> str:
    loc = _LAT_KO.get(i.laterality, "")
    s = ("⚠" if i.critical else "") + (i.label if i.concept else f"'{i.label}'") + (f"({loc})" if loc else "")
    if i.value is not None:
        s += f" {i.value:g}{i.unit}"
    if i.comparison:
        s += f"[{_CMP_KO[i.comparison]}]"
    return s


def render_for_prompt(interp: Interpretation, max_chars: int = 300) -> str:
    """One Korean line: abnormal (critical first), uncertain, absent, then supportive disease links (not diagnoses)."""
    name = interp.test_name.strip() or "검사"
    if interp.unavailable:
        return f"[{name}] 결과 제공되지 않음(정상으로 보지 말 것)"[:max_chars]
    segs: list[str] = []
    if interp.normal:
        segs.append("정상(특이 소견 없음)")
    pres = [i for i in interp.present() if i.concept not in ("IMG:normal_study", "ECG:normal_ecg")]
    pres.sort(key=lambda i: (not i.critical, i.concept == ""))
    unc = sorted(interp.uncertain(), key=lambda i: not i.critical)
    absn: list[str] = []
    for i in interp.absent():
        if not i.concept:
            continue
        # a normal measurement is emitted for both sides ("혈압 120/80" -> no hypotension, no raised BP): show it once
        txt = f"{i.span}(정상)" if i.value is not None and i.span else _short(i)
        if txt not in absn:
            absn.append(txt)
    if pres:
        segs.append("있음: " + ", ".join(_short(i) for i in pres))
    if unc:
        segs.append("의심: " + ", ".join(_short(i) for i in unc))
    if absn:
        segs.append("없음: " + ", ".join(absn))
    sup: list[str] = []
    for i in pres:
        for _dx, ko, _w in i.supports:
            if ko not in sup:
                sup.append(ko)
    if sup:
        segs.append("지지 가능 질환(확진 아님): " + ", ".join(sup[:4]))
    if interp.pending:
        segs.append("일부 결과 대기 중(정상 아님)")
    if not segs:
        segs.append("판독 가능한 소견 없음(원문 확인 필요)")
    out = f"[{name}] " + " / ".join(segs)
    return out if len(out) <= max_chars else out[:max_chars - 1] + "…"


_SERIAL = re.compile(r"\d+\s?(?:일|주|개월|시간)\s?(?:후|째)|(?<![a-z])(?:day|week|hour)\s?\d+|post-?op|추적\s?\d")


def llm_reasons(interp: Interpretation) -> list[str]:
    """Why the code reading may not be enough (empty list = code alone is fine)."""
    r: list[str] = []
    if any(x.startswith("error") for x in interp.llm_reasons):
        r.append("error")
    if interp.unavailable:
        return r
    if interp.text_len > 500:
        r.append("long_text")
    if interp.n_sentences > 6:
        r.append("many_sentences")
    if interp.serial:
        r.append("serial_results")
    unm = [i for i in interp.items if not i.concept and i.polarity != "absent"]
    mapped_abn = [i for i in interp.items if i.concept and i.polarity != "absent"
                  and i.concept not in ("IMG:normal_study", "ECG:normal_ecg")]
    if len(unm) >= 2 or (unm and not mapped_abn):
        r.append("unmapped_findings")
    if interp.kind in ("imaging", "ecg") and not interp.items and interp.text_len > 15:
        r.append("nothing_read")
    by: dict[str, set[str]] = {}
    for i in interp.items:
        if i.concept:
            by.setdefault(i.concept, set()).add(i.polarity)
    if any({"present", "absent"} <= v for v in by.values()):
        r.append("conflicting_polarity")
    if len(interp.uncertain()) >= 3:
        r.append("many_hedges")
    return r


def needs_llm(interp: Interpretation) -> bool:
    """True when the report is too long or complex for the code reading alone (future hook: one extra gpt-oss call
    with prompts.RESULT_INTERPRETER_PROMPT; nothing is called here)."""
    return bool(llm_reasons(interp))


__all__ = ["Item", "Interpretation", "interpret", "render_for_prompt", "needs_llm", "llm_reasons", "kind_of_test", "KINDS",
           "POLARITIES"]
