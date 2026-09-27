"""Clinical-finding parser: text -> normalised findings (concept, polarity, subject, time, site, value, ...).

One rule set for colloquial Korean patient answers, exam reports, test results and the doctor model's own claims, so
the grounding checker, the KB matcher, the safety protocols and the danger gate can all read findings the same way.
Stdlib only, CPU only, deterministic, no network, no state between calls (every call builds its own objects).

    parse(text, source="patient"|"exam"|"test"|"claim", context=None) -> list[Finding]
    match(claim_text, findings_or_text) -> (bool, span)        grounding-style check of one claim
    concepts_in(text, ...) -> set[str]                          keyword-style triggers (affirmed concept ids)
    affirmed(text, concept_ids) / denied(text, concept_ids)     drop-in for contains_affirmed() / polarity()=="neg"

How a sentence is read (details in docs/nlp.md):
1. normalise (lexicon.normalize), split into sentences, drop "결과가 제공되지 않습니다" sentences;
2. find concept mentions with the lexicon (longest match; regex mentions may not cross a clause boundary);
3. split the sentence into clauses at Korean connectives (~고, ~는데, ~지만, ~으나, ~면서, ~며, ~어서) and each clause
   into comma parts; a "bare" part ("기침이나 가래,") takes the predicate of the next part ("열은 없어요"), so a
   clause-final negation covers the whole list;
4. polarity = the first cue after the mention in its unit (negation 없/않/아니/안+verb/음성/정상/(-)/denies ...;
   affirmation 있/양성/관찰/호소 ...; uncertainty 모르겠/글쎄 ...), after masking idioms that only look negative
   ("수 없", "가라앉지 않" = still there, "이유 없이"); a negation inside a regex match ("배는 안 아파") negates it,
   a negation inside a lexicon form is part of the concept ("입맛이 없" = anorexia); "absence" forms ("잘 먹어요")
   give absent; double negation ("없지는 않아요") gives present;
5. subject (family / other / patient) from the latest person word before the mention in the sentence (or the
   question/key in `context`); temporality (past / chronic / intermittent / current) and onset from clause markers,
   carried forward inside the sentence; hedges (~것 같아요, 의심) and hypotheticals (~일까 봐, questions) flagged;
   similes ("발작처럼") dropped unless the form is figurative by design ("몸이 불덩이");
6. numbers: vital signs and common labs give measured findings (체온 38.5 -> SYM:fever present, 36.7 -> absent,
   "WBC 14,200" -> LAB:wbc_high); for exam/test/claim text knowledge/kb_tests.detect() adds its test-result concepts;
7. a yes/no answer ("네, 좀 해요", "아니요") resolves the concepts named in context["question"].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from doctor_agent.nlp.lexicon import LEXICON, Concept, Lexicon, Mention, normalize

SOURCES = ("patient", "exam", "test", "claim")


@dataclass
class Finding:
    concept: str
    label: str
    span: str
    polarity: str = "present"  # present | absent | uncertain
    subject: str = "patient"  # patient | family | other
    temporality: str = "current"  # current | past | chronic | intermittent
    onset: str = ""  # "3일 전부터", "갑자기", ...
    laterality: str = ""  # right | left | bilateral
    site: str = ""
    severity: str = ""  # mild | moderate | severe (+ " NRS 8/10")
    value: float | None = None
    unit: str = ""
    direction: str = ""  # high | low | normal (measured value vs threshold / reference range)
    hedged: bool = False
    hypothetical: bool = False
    confidence: float = 0.9
    source: str = "patient"
    start: int = 0  # offsets in normalize(text)
    end: int = 0
    clause: str = ""
    cue: str = ""  # what decided the polarity (debugging / evaluation)

    def key(self) -> tuple[str, str]:
        return self.concept, self.polarity

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in ("", None, False) or k in ("concept", "polarity")}
        d.pop("clause", None)
        return d


# ------------------------------------------------------------------------------------------------ cue vocabulary
_SENT = re.compile(r"(?<!\d)\.(?!\d)|\.(?=\s|$)|[?!;\n]")
_UNAVAILABLE = re.compile(r"제공\s?되지\s?않|제공하지\s?않|제공\s?불가|결과가?\s?없습니다|시행되지\s?않|시행하지\s?않|확인할\s?수\s?없"
                          r"|정보가\s?없|not available|not performed")
# clause boundaries: Korean connective endings followed by a space (the auxiliaries of ~고 있다, ~고 나서, ~고 싶다 do not
# end a clause)
_CLAUSE = re.compile(
    r"(?:(?<=[가-힣])고(?!\s?(?:있|계시|계세|계셨|계신|싶|나서|나면|나니|나도|난\s|난뒤|난후|말았|지내|다니|가서|보니|보면|해서|해도"
    r"|했|한다|하셨|하던|하는|했던))"
    r"|(?<=[가-힣])데(?:도)?|지만|으나|되나|면서|으며|(?<=[가-힣])(?<![이])며"
    r"|(?<=[가-힣])(?<!에)(?<!부터)서|니까|다가)"
    r"(?=[\s,])"
)
_PARTS = re.compile(r",|\s/\s")

_NEG = (r"없(?!을까)|않|아니(?!라도)|아닌|아냐|아님|아뇨|아니요|(?:다|라)?기\s?보다|음성|(?<!비)(?<!이)정상|부인|미관찰|미검출|괜찮|멀쩡|깨끗|청명|명료|문제\s?없|무증상"
        r"|(?<![가-힣])안\s?(?=(?:나|났|해|했|하|아프|아파|아팠|먹|피우|피워|피웠|마시|마셔|마셨|들|느껴|느끼|보이|보여|와요|왔|오|붓|부어|부었|쉬|생기|생겼|차|저리|어지|메스|토|가려|결리|쑤|났|뛰|떨|받|맞|써|쓰|다쳐|다친|다쳤|넘어|올라|내려|탔|타|가요|갔|갔어|뻗|퍼|저려))"
        r"|negative|normal|unremarkable|absent|\bnone\b|\bno\b|\bnot\b|denie|without|wnl")
_POS = (r"있|양성|보임|보이(?!지)|보였|보이며|관찰됨|관찰된|관찰되(?!지)|관찰함|들림|들리(?!지)|청진됨|청진되(?!지)|청취|촉지(?!되지|\s?안|\s?않)"
        r"|확인됨|확인되(?!지)|호소|동반(?!\s?(?:되지|없|안|하지))|positive|present|noted|seen")
_UNC = r"모르겠|모르|글쎄|잘\s?기억|기억이\s?(?:잘\s?)?안\s?나|기억\s?안\s?나|기억은\s?안|확실하지|확실치|확실히는|애매|헷갈|긴가민가|잘\s?몰"
_CUE = re.compile(f"(?P<neg>{_NEG})|(?P<pos>{_POS})|(?P<unc>{_UNC})")
# look like a negation but are not one (masked before the cue search); the persistence ones are affirmations
_IDIOM_NOT_NEG = re.compile(r"(?:할|걸을|잘|먹을|참을|견딜|설|앉을|누울|일어설|움직일|쉴|볼|들|숨을\s?쉴)\s?수(?:가|도)?\s?없"
                            r"|수\s?없을\s?(?:정도|만큼)|어쩔\s?수\s?없|상관\s?없|관계\s?없|틀림\s?없|끊임\s?없|쉴\s?새\s?없|빠짐\s?없"
                            r"|이유\s?없이|원인\s?없이|까닭\s?없이|예고\s?없이|뿐(?:만)?\s?아니라|아니라\s?[가-힣]+(?:도|까지)"
                            r"|정신이\s?없|뭐라\s?할\s?수\s?없|할\s?수밖에\s?없|(?:별\s?|아무\s?)?문제\s?없이|탈\s?없이|문제없이"
                            r"|(?:기간|동안|중에?)\s?[^,]{0,10}(?:문제|이상|합병증)[는은가이]?\s?없")
_PERSIST = re.compile(r"(?:낫|좋아지|호전되|멎|멈추|그치|사라지|가라앉|떨어지|나아지|빠지|내리|줄어들|안\s?낫)(?:지|질)\s?(?:않|못|안)"
                      r"|(?:안|못)\s?(?:나아|낫|멈|멎|그치|빠지|빠져|내려|떨어|좋아지|가라앉)|계속\s?(?:돼|되|있|나|해)|여전히"
                      r"|(?<![가-힣])외(?:에|에는|에도)?\s")
_DOUBLE_NEG = re.compile(r"(?:없|않|아니)[가-힣]{0,2}\s?(?:지는|진|지도|는\s?건|는\s?것은|는\s?게|은\s?건|은\s?게|다고는|다고\s?할\s?수는)\s?(?:않|아니|안\s)")
_POS_THEN_NEG = re.compile(r"있(?:지|진|지는|지도|지를|기는|는\s?건|는\s?것은)\s?(?:않|안\s|아니)|있지\s?않|있진\s?않|있지는\s?않"
                           r"|있(?:는|었던|던|으신|으셨던)\s?(?:사람|분|경우|분들|사람들|이)[은는이가도]?\s?(?:없|아무도|한\s?명도|전혀\s?없)")
_PRE_NEG_EN = re.compile(r"\b(?:no|denies|denied|without|negative for|absence of|not|free of)\b[^,;]{0,25}$")
_PRE_NEG_KO = re.compile(r"(?:^|\s)(?:안|전혀\s?안|별로\s?안|아예\s?안|하나도\s?안)\s?$|(?<!비)정상\s?$")
_NEG_INSIDE = re.compile(r"없|않|아니|(?<![가-힣])안\s|(?<![가-힣])안(?=[가-힣])(?!에|쪽|면|구|과|정|의|으로|색|압)|음성|정상")
_HEDGE = re.compile(r"것\s?같|거\s?같|듯(?:해|하|합|한\s?느낌|싶)|싶어요|아마|같기도|인\s?듯"
                    r"|편이에요|편이다|정도인\s?것|의심|시사|가능성|추정|의증|r/o|같은\s?(?:느낌|기분|거|것|게)|같아요|같습니다")
_HYPO = re.compile(r"(?:일|할|인|있을|생길|될|걸릴|올|아닐|것일)까|아닌지|인지\s?(?:모르|궁금|걱정|확인|알)|일지|까\s?봐"
                   r"|배제|확인하고\s?싶|알고\s?싶|보고\s?싶|검사(?:를)?\s?(?:해|받아)\s?(?:보|봐)|진단받을까|되면\s?어떡|아니길"
                   r"|rule out|r/o")
_METAPHOR = re.compile(r"^\s?(?:인|한|하는|난|나는|된|되는|오는|온|이|가)?\s?(?:것|거)?\s?(?:처럼|마냥|같이(?!\s?(?:있|살|오|와|먹|가|사는|지내|일)))")
_WHEN = re.compile(r"(?:을|를)?\s?(?:하|할|해|볼|쉴|뱉을|먹을|삼킬|누를|누울)?\s?(?:때|때마다|하면서)(?![가-힣])")
_RESOLVED = re.compile(r"(?:지금은|이제는|현재는)?\s?(?:괜찮아졌|나았|좋아졌|사라졌|없어졌|멎었|멈췄|가라앉았|회복)")
_QUESTION_END = re.compile(r"(?:나요|까요|가요|인가|습니까|ㄹ까|을까|를까|걸까|건가요|거죠|죠|지요|는지)\s?\??$")

# subject: person words (followed by a particle) before the mention
_FAMILY = (r"(?:(?:외|친)?(?:할머니|할아버지)|어머니|어머님|엄마|아버지|아버님|아빠|부모님?|형(?=[이은도님제])|오빠|누나|언니|남동생|여동생|동생"
           r"|산모|형제|자매|조부모|삼촌|외삼촌|이모|고모|숙모|이모부|고모부|친척|가족|집안|사촌|큰아버지|작은아버지|큰어머니|작은어머니"
           r"|mother|father|brother|sister|sibling|parents?|family|grandmother|grandfather|aunt|uncle)"
           r"(?:께서|님(?:이|은|께서|도|의)|분(?:이|은|들)?|들(?:이|은|도)|이(?![가-힣])|가(?![가-힣])|은|는|도|의|에게|쪽|\s?중|,)")
_OTHER = (r"(?:남편|아내|와이프|배우자|애인|여자\s?친구|남자\s?친구|여친|남친|파트너|동료|친구|룸메이트|직장\s?동료|같이\s?(?:먹은|사는|일하는|간)\s?(?:사람|분|동료|친구)"
          r"|같이\s?갔던\s?(?:사람|분|친구)|주변\s?사람|옆\s?사람|반\s?친구|같은\s?반|다른\s?사람|husband|wife|partner|friend|coworker|roommate)"
          r"(?:께서|님(?:이|은|도)|분(?:이|은|들)?|들(?:이|은|도)|이(?![가-힣])|가(?![가-힣])|은|는|도|의|,)")
_SELF = (r"(?:(?<![가-힣])저(?:는|도|만|의)|(?<![가-힣])제가|(?<![가-힣])전\s|본인(?:은|이|도)|(?<![가-힣])나(?:는|도)\s|(?<![가-힣])내가|환자(?:는|가|분은)"
         r"|아이(?:가|는|도|의)|아기(?:가|는|도)|(?<![가-힣])애(?:가|는|도)|우리\s?(?:아이|애|아기)|저희\s?(?:아이|애|아기)|(?:아들|딸)(?:이|은|도)"
         r"|patient|i\s)")
_SUBJ = re.compile(f"(?P<family>{_FAMILY})|(?P<other>{_OTHER})|(?P<self>{_SELF})")
_FAMILY_CTX = re.compile(r"가족력|가족\s?중|가족분|가족\s?(?:병력|질환)|집안에|부모님|family history|형제\s?중")

# temporality / onset
_PAST = re.compile(r"예전에|옛날에|과거에|어렸을\s?때|어릴\s?때|젊었을\s?때|학생\s?때|전에\s?한\s?번|한\s?번\s?(?:있었|앓|했었)"
                   r"|(?<![가-힣])적(?:이|은|도)?\s?(?:전혀\s|한\s?번도\s|별로\s|거의\s|[0-9]+\s?번\s|몇\s?번\s|여러\s?번\s)?(?:있|없)|전에도|과거(?:에|의|\s)|앓았|앓았던|[0-9]+\s?년\s?전(?:에|쯤|경)?(?!\s?부터)"
                   r"|수\s?년\s?전(?!\s?부터)|몇\s?년\s?전(?!\s?부터)|[0-9]+\s?(?:세|살)\s?(?:때|무렵)|작년에|지난해|이전에|그\s?때는|당시(?:에)?|했었|었었|았었|였었"
                   r"|나았|사라졌|없어졌|좋아졌|멎었|멈췄|(?<!열이\s)내렸|가라앉았|회복됐|회복되었|완치|끊었|끊은\s?지|과거력|병력|기왕력|history of")
_CHRONIC = re.compile(r"(?<!미)만성|오래\s?(?:전)?부터|오래됐|오래되|수년간|수\s?년\s?(?:동안|전부터|째)|몇\s?년\s?(?:동안|간|째|전부터)|[0-9]+\s?년\s?(?:동안|간|째|넘게|이상|전부터)"
                      r"|평소(?:에|엔)?(?!\s?(?:보다|처럼|와|같이|대로))|원래|항상|지병|앓고\s?(?:있|계)|복용\s?중|복용하고|먹고\s?있|드시고|투석|치료\s?중|치료받고|관리\s?중|조절되|조절\s?중"
                      r"|있어서\s?약|수개월\s?(?:동안|째|전부터)|몇\s?달\s?(?:동안|째|전부터)|[0-9]+\s?(?:개월|달|년)\s?(?:간|동안|째|전부터|넘게)|chronic|long-standing")
_DIAG_AGO = re.compile(r"[0-9]+\s?(?:년|개월)\s?전에?\s?[^,;]{0,15}?(?:진단|발견)[^,;]{0,12}(?:복용|치료|먹고|약)")
_INTERMITTENT = re.compile(r"가끔|간헐|때때로|종종|이따금|반복|왔다\s?갔다|했다\s?안\s?했다|다가\s?괜찮아졌다가|았다가\s?괜찮|었다가\s?괜찮|때마다|수시로|주기적|몇\s?번씩|한\s?번씩|번씩"
                           r"|며칠에\s?한\s?번|intermittent|episodic|on and off")
_RESET = re.compile(r"지금은|현재는?|요즘은?|최근에?는?|이번에?는?|오늘은?")
_ONSET = re.compile(r"(?:[0-9]+|몇|수|한|두|세|며칠|하루|이틀|사흘|나흘|닷새|일주일|열흘|보름|반나절|한\s?달)\s?(?:분|시간|일|주|주일|개월|달|년)?\s?"
                    r"(?:전|째|쯤\s?전|정도\s?전|가량\s?전|전쯤|전부터|동안)(?:부터|에)?"
                    r"|어제(?:\s?(?:아침|저녁|밤|오후|낮))?(?:부터)?|오늘\s?(?:아침|새벽|오후)?(?:부터)?|그저께|지난\s?(?:주|달|밤|주말)|어젯밤|엊그제"
                    r"|갑자기|갑작스럽게|갑작스레|서서히|방금|아침에\s?일어나(?:니|보니|서)?|출생\s?(?:직후|[0-9]+\s?(?:시간|일)\s?후)|생후\s?[0-9]+\s?(?:일|주|개월)")
_SEVERE = re.compile(r"너무|매우|아주|엄청|굉장히|몹시|극심|심하게|심한|심해|심했|되게|참을\s?수\s?없|견딜\s?수\s?없|죽을\s?(?:것|거|만큼)|최악|못\s?참")
_MILD = re.compile(r"약간|조금|좀|살짝|가볍|가벼운|미약|경미|경한|미미|약하게|슬쩍")
_MODERATE = re.compile(r"꽤|제법|중등도|상당히")
_NRS = re.compile(r"(?:10\s?점\s?(?:만점|중)에?\s?([0-9]{1,2})\s?점?)|(?<![0-9/])([0-9]{1,2})\s?(?:/|점\s?(?:만점\s?)?중에?\s?|\s?out of\s)10(?![0-9])")
_LAT = re.compile(r"(?P<right>오른쪽|오른|우측|(?<![가-힣])우(?=[하상측])|\bright\b|\brt\b)|(?P<left>왼쪽|(?<![가-힣])왼|좌측|(?<![가-힣])좌(?=[하상측])|\bleft\b|\blt\b)"
                  r"|(?P<bilateral>양쪽|양측|(?<![가-힣])양(?=[손발팔다쪽측하상])|두\s?쪽\s?다|\bbilateral\b|\bboth\b)")
_SITE = re.compile(r"우하복부|우상복부|좌하복부|좌상복부|상복부|하복부|심와부|명치|윗배|아랫배|배꼽|옆구리|복부|관자놀이|뒷머리|뒤통수|정수리"
                   r"|이마|머리|얼굴|눈꺼풀|입술|뒷목|어깨|겨드랑이|가슴|흉부|허리|엉덩이|골반|사타구니|서혜부|팔꿈치|손목|손가락|상지|허벅지|무릎"
                   r"|종아리|정강이|발목|엄지발가락|발가락|발바닥|발등|다리|하지|사지|전신|온몸|관절|피부|두피|음낭|고환|항문"
                   r"|(?<![가-힣])(?:배|눈|귀|코|입|혀|턱|목|등|손|팔|발|질)(?=[이가은는을를도에의]|\s|$)")
_BARE_FILLER = re.compile(r"특이|특별한|특별히|별다른|다른|기타|명확한|뚜렷한|뚜렷이|아무런|어떤|큰|이나|거나|또는|혹은|그리고|및|이랑|하고|랑"
                          r"|최근에?|요즘|지금|현재|평소|혹시|그\s?외에?|느낌|같은|증상|소견|증세|양상|등|것|과|와|도|은|는|이|가|을|를|의|에|나"
                          r"|and|or|any|other|signs? of|\s")
# a comma part ending like a predicate ("~해요", "~없음", "~있고") is not a bare list item
_PRED_END = re.compile(r"(?:요|다|니다|고|며|서|데|지만|으나|면서|거든|네|없음|있음|않음|보임|같음|좋음|아님|함|됨|임)\s*$")

# yes/no answers
_YES = re.compile(r"^\s*(?:네|예|응|어\s|맞아|맞습니다|맞아요|그래요|그렇습니다|그런\s?것\s?같|있어요|있습니다|있었어요|있어|좀|약간|조금|가끔|자주|많이"
                  r"|심해요|그럼요|물론|yes\b)")
_NO = re.compile(r"^\s*(?:아니요|아뇨|아니오|아니|없어요|없습니다|없었어요|없어|전혀|안\s|않|별로|no\b)")
_UNSURE_ANSWER = re.compile(r"^\s*(?:글쎄|잘\s?모르|모르겠|기억이|확실하지|잘\s?기억)")
_Q_NEGATIVE = re.compile(r"(?:없|않|안\s?[가-힣]+)[가-힣]*\s?(?:으세요|세요|나요|습니까|어요|죠|지요|으시죠|으신가요|으신지)\s*\??\s*$")

# measured vital signs: (regex, concept rules)
_NUMV = r"(\d{1,3}(?:\.\d{1,2})?)"
_TEMP = re.compile(r"(?:체온|(?<![a-z])bt|temp(?:erature)?|열(?:이|은|도)?)\s*(?:은|는|이|:|=|측정)?\s*(?:약\s?)?" + _NUMV
                   + r"\s*(°\s?c|℃|°|도|°\s?f|℉)?")
_TEMP_BARE = re.compile(r"(?<![\d.])(3[4-9]\.\d|4[0-3]\.\d|3[5-9]|4[0-2])\s*(°\s?c|℃|도)(?!\s?(?:각도|각|방향|기울))")
_TEMP_F = re.compile(r"(?<![\d.])(9[5-9](?:\.\d)?|10[0-8](?:\.\d)?)\s*(°\s?f|℉)")
_HR = re.compile(r"(?:맥박(?:수)?|심박(?:수)?(?!동기)|(?<![a-z])(?:hr|pr)(?![a-z])|pulse|heart rate)\s*(?:은|는|이|:|=)?\s*(?:분당\s?)?(?:약\s?)?(\d{2,3})\s*(회|bpm|/분|/min)?")
_RR = re.compile(r"(?:호흡수|호흡\s?횟수|(?<![a-z])rr(?![a-z])|respiratory rate|호흡)\s*(?:은|는|이|:|=)?\s*(?:분당\s?)?(\d{1,2})\s*(회|/분|/min|breaths|bpm)")
_BP = re.compile(r"(?:혈압|(?<![a-z])bp(?![a-z])|blood pressure)[^\d\n]{0,14}(\d{2,3})\s*/\s*(\d{2,3})")
_SPO2 = re.compile(r"(?:산소\s?포화도|(?<![a-z])spo2|sao2|o2\s?sat\w*|포화도)[^\d%\n]{0,20}(\d{2,3})\s*%")

# labs read here (kb_tests covers the rest): key, analyte regex, high concept, high threshold, low concept, low threshold
_LABS: list[tuple[str, str, str | None, float | None, str | None, float | None]] = [
    ("wbc", r"(?<![a-z])wbc(?![a-z])|백혈구(?!\s?(?:에스테라제|원주|뇨|에스터))(?:\s?수치|\s?수)?", "LAB:wbc_high", 11000, "LAB:wbc_low", 4000),
    ("hb", r"(?<![a-z])(?:hb|hgb)(?![a-z0-9])|헤모글로빈|혈색소", None, None, "LAB:hb_low", 12.0),
    ("plt", r"(?<![a-z])(?:plt|platelets?)(?![a-z])|혈소판(?:\s?수치|\s?수)?", "LAB:plt_high", 450000, "LAB:plt_low", 150000),
    ("na", r"(?<![a-z])(?:na\+?|sodium)(?![a-z])|나트륨", "LAB:na_high", 145, "LAB:na_low", 135),
    ("k", r"(?<![a-z])(?:k\+?|potassium)(?![a-z])|칼륨", "LAB:k_high", 5.2, "LAB:k_low", 3.5),
    ("glu", r"(?<![a-z])(?:glucose|fbs|bst)(?![a-z])|공복\s?혈당|혈당|(?<![가-힣])당(?=\s?\d)|포도당", "LAB:glucose_high", 200, "LAB:glucose_low", 70),
    ("cr", r"(?<![a-z])(?:cr|creatinine)(?![a-z])|크레아티닌", "LAB:cr_high", 1.3, None, None),
    ("bun", r"(?<![a-z])bun(?![a-z])|요소\s?질소", "LAB:bun_high", 25, None, None),
    ("ast", r"(?<![a-z])(?:ast|sgot|got)(?![a-z])", "LAB:ast_alt_high", 40, None, None),
    ("alt", r"(?<![a-z])(?:alt|sgpt|gpt)(?![a-z])", "LAB:ast_alt_high", 40, None, None),
    ("tbil", r"총\s?빌리루빈|(?<![a-z])t\.?\s?bil(?:irubin)?|total bilirubin|(?<!직접\s)(?<!간접\s)(?<!직접)(?<!간접)빌리루빈(?!뇨)", "LAB:bilirubin_high", 1.2, None, None),
    ("alb", r"알부민(?!뇨)|(?<![a-z])alb(?:umin)?(?![a-z])", None, None, "LAB:albumin_low", 3.5),
    ("crp", r"(?<![a-z])(?:hs-?)?crp(?![a-z])|c-?반응성?\s?단백(?:질)?", "LAB:crp_high", 0.5, None, None),
    ("esr", r"(?<![a-z])esr(?![a-z])|적혈구\s?침강\s?속도|혈침", "LAB:esr_high", 25, None, None),
    ("hco3", r"(?<![a-z])hco3-?|중탄산(?:염)?|bicarbonate", None, None, "LAB:hco3_low", 22),
    ("inr", r"(?<![a-z])(?:pt-?)?inr(?![a-z])", "LAB:inr_high", 1.2, None, None),
    ("eos", r"호산구|(?<![a-z])(?:eos|eosinophils?)(?![a-z])", "LAB:eos_high", 7, None, None),
]
_LAB_RX = [(k, re.compile(p), hi, th, lo, tl) for k, p, hi, th, lo, tl in _LABS]
_VALUE = re.compile(r"[^\d<>≤≥\n,;]{0,14}?([<>≤≥]=?\s*)?(\d+(?:\.\d+)?)\s*(만|천)?\s*((?:x|×)\s?10\s?\^?\s?[0-9³]|k)?\s*"
                    r"(%|[a-zμ/.³^0-9]+(?:/[a-zμ.0-9]+)?)?")
_REF = re.compile(r"^\s*\(([^()]*?)\)")
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)")
_UPPER = re.compile(r"[<≤]\s*=?\s*(\d+(?:\.\d+)?)")
_LOWER = re.compile(r"[>≥]\s*=?\s*(\d+(?:\.\d+)?)")
_QUAL_NORMAL = re.compile(r"^[^\d,;]{0,10}?(?:정상|음성|normal|negative|wnl|이상\s?없|범위\s?내)")
_URINE_CTX = re.compile(r"소변|요검사|요\s?분석|urinalysis|u/a|요침사|hpf|비중|요\s?배양")
_URINE_ITEMS = [
    ("LAB:urine_blood", re.compile(r"(?:잠혈|occult blood|(?<![a-z])blood)\s*:?\s*(음성|양성|negative|positive|trace|미량|\+{1,4}|[1-4]\+)")),
    ("LAB:proteinuria", re.compile(r"(?:(?<![가-힣])단백(?:질)?|요단백|protein)\s*:?\s*(음성|양성|negative|positive|trace|미량|\+{1,4}|[1-4]\+)")),
    ("LAB:glycosuria", re.compile(r"(?:(?<![가-힣])당|요당|glucose)\s*:?\s*(음성|양성|negative|positive|trace|미량|\+{1,4}|[1-4]\+)")),
]
_URINE_RBC = re.compile(r"(?:rbc|적혈구)\s*:?\s*(\d+)(?:\s*-\s*(\d+))?\s*/\s*hpf")


# ------------------------------------------------------------------------------------------------ helpers
@dataclass
class _Clause:
    start: int  # offsets in the sentence
    end: int
    parts: list[tuple[int, int]] = field(default_factory=list)


def _sentences(t: str) -> list[tuple[int, int, bool]]:
    out, s = [], 0
    for m in _SENT.finditer(t):
        if m.start() > s:
            out.append((s, m.start(), m.group() == "?"))
        s = m.end()
    if s < len(t):
        out.append((s, len(t), False))
    return [(a, b, q) for a, b, q in out if t[a:b].strip()]


def _clauses(sent: str) -> list[_Clause]:
    cuts = [0] + [m.end() for m in _CLAUSE.finditer(sent)] + [len(sent)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        c = _Clause(a, b)
        p = a
        for m in _PARTS.finditer(sent, a, b):
            c.parts.append((p, m.start()))
            p = m.end()
        c.parts.append((p, b))
        out.append(c)
    return out


def _first_cue(tail: str) -> tuple[str, str]:
    """(kind, cue text) of the first polarity cue in the text after a mention: neg | pos | unc | persist | "" ."""
    masked = _IDIOM_NOT_NEG.sub(lambda m: "·" * len(m.group()), tail)
    pm = _PERSIST.search(masked)
    for m in _CUE.finditer(masked):
        if pm and pm.start() <= m.start():
            return "pos", "persist:" + pm.group()
        kind = m.lastgroup
        rest = masked[m.start():m.start() + 14]
        if kind == "pos" and m.group().startswith("있") and _POS_THEN_NEG.match(rest):
            return "neg", rest.strip()
        if kind == "neg" and _DOUBLE_NEG.match(rest):
            return "pos", "double-neg:" + rest.strip()
        return kind, m.group()
    if pm:
        return "pos", "persist:" + pm.group()
    return "", ""


def _is_bare(text: str, spans: list[tuple[int, int]], offset: int) -> bool:
    """A comma part made only of concept mentions and fillers ("기침이나 가래", "특이 발진") takes the next part's predicate."""
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s - offset), min(len(chars), e - offset)):
            chars[i] = " "
    rest = _BARE_FILLER.sub(" ", "".join(chars))
    return len(re.sub(r"[^가-힣a-z0-9]", "", rest)) <= 1 and not _CUE.search(text)


def _num_close(a: float, b: float) -> bool:
    small, big = sorted((abs(a), abs(b)))
    if abs(a - b) <= (0.002 * big if big >= 1000 else max(0.051, 0.03 * big)):
        return True
    return big >= 1000 and abs(small * 1000 - big) <= 0.004 * big


def _ref_direction(after: str, v: float) -> tuple[str, str]:
    """Direction from a reference range in parentheses right after a value: ("high"|"low"|"normal", ref text)."""
    m = _REF.match(after)
    if not m:
        return "", ""
    body = m.group(1)
    r = _RANGE.search(body)
    if r:
        lo, hi = float(r.group(1)), float(r.group(2))
        return ("high" if v > hi else "low" if v < lo else "normal"), m.group(0)
    u, lw = _UPPER.search(body), _LOWER.search(body)
    if u:
        return ("high" if v >= float(u.group(1)) else "normal"), m.group(0)
    if lw:
        return ("low" if v <= float(lw.group(1)) else "normal"), m.group(0)
    if re.search(r"정상|normal|범위\s?내|within", body):
        return "normal", m.group(0)
    if re.search(r"높|상승|high|↑|\bh\b", body):
        return "high", m.group(0)
    if re.search(r"낮|저하|low|↓|\bl\b", body):
        return "low", m.group(0)
    return "", ""


# ------------------------------------------------------------------------------------------------ parser
class _Parser:
    def __init__(self, text: str, source: str, context: dict | None, lex: Lexicon):
        self.t = normalize(text)
        self.source = source if source in SOURCES else "patient"
        self.ctx = context or {}
        self.lex = lex
        q = self.ctx.get("question") or ""
        key = self.ctx.get("key") or ""
        primary = normalize(key).split("|")[0].strip()
        family_key = primary in ("가족력", "가족", "family history") and not re.search(r"사회력|과거력|접촉력|흡연", key)
        self.default_subject = self.ctx.get("subject") or ("family" if family_key or _FAMILY_CTX.search(normalize(q))
                                                            else "patient")

    # -------------------------------------------------------------- per text
    def run(self) -> list[Finding]:
        out: list[Finding] = []
        for a, b, is_q in _sentences(self.t):
            sent = self.t[a:b]
            if _UNAVAILABLE.search(sent):
                continue
            out += self._sentence(sent, a, is_q)
        if self.ctx.get("question") and self.source == "patient":
            out += self._yes_no(out)
        return _dedupe(out, self.lex)

    # -------------------------------------------------------------- per sentence
    def _sentence(self, sent: str, off: int, is_q: bool) -> list[Finding]:
        clauses = _clauses(sent)
        bounds = [c.start for c in clauses[1:]]
        mentions = self.lex.scan(sent, barriers=bounds)
        question = self.source == "patient" and (is_q or bool(_QUESTION_END.search(sent.strip())) and "?" in sent)
        subj_marks = [(m.start(), m.lastgroup) for m in _SUBJ.finditer(sent)]
        found: list[Finding] = []
        carry_t, carry_onset = "", ""
        for ci, c in enumerate(clauses):
            ctext = sent[c.start:c.end]
            t_mark = self._temporality(ctext)
            if _RESET.search(ctext) and not t_mark:
                carry_t, carry_onset = "", ""
            onset_m = _ONSET.search(ctext)
            onset = onset_m.group().strip() if onset_m else carry_onset
            temporality = t_mark or carry_t or "current"
            if t_mark in ("past", "chronic"):
                carry_t = t_mark
            carry_onset = onset
            cms = [m for m in mentions if c.start <= m.end - 1 < c.end]
            for m in cms:
                f = self._mention(m, sent, c, cms, temporality, onset, subj_marks, question)
                if f:
                    f.start, f.end = f.start + off, f.end + off
                    found.append(f)
            for f in self._measures(ctext, c.start, sent):
                f.temporality, f.onset = temporality if temporality != "past" else "current", onset
                f.start, f.end = f.start + off, f.end + off
                found.append(f)
            # "열이 나고 아팠는데, 지금은 괜찮아졌어요": a clause that only says the problem resolved makes the earlier
            # findings of the sentence past
            if not cms and _RESOLVED.search(ctext) and self.source == "patient":
                for f in found:
                    if f.temporality in ("current", "intermittent") and f.polarity == "present":
                        f.temporality = "past"
        if self.source in ("test", "exam", "claim"):
            found += self._kb_tests(sent, off, clauses)
        return found

    def _temporality(self, ctext: str) -> str:
        if _DIAG_AGO.search(ctext):
            return "chronic"
        if _PAST.search(ctext):
            return "past"
        if _INTERMITTENT.search(ctext):
            return "intermittent"
        if _CHRONIC.search(ctext):
            return "chronic"
        return ""

    def _subject(self, pos: int, marks: list[tuple[int, str]], ctext_after: str) -> str:
        prev = [k for p, k in marks if p < pos]
        if prev:
            return {"family": "family", "other": "other", "self": "patient"}[prev[-1]]
        if self.source in ("exam", "test"):
            return "patient"
        m = _SUBJ.search(ctext_after)
        if m and m.lastgroup in ("family", "other"):
            return m.lastgroup
        return self.default_subject

    def _mention(self, m: Mention, sent: str, c: _Clause, cms: list[Mention], temporality: str, onset: str,
                 subj_marks, question: bool) -> Finding | None:
        concept = self.lex.concept(m.cid)
        if concept is None:
            return None
        # unit: the comma part holding the mention end, extended over the following parts while the current one is a
        # list item (a noun phrase without its own cue or predicate: "기침이나 가래, 열은 없어요")
        pi = next((i for i, (a, b) in enumerate(c.parts) if a <= m.end - 1 < b), len(c.parts) - 1)
        unit_end = c.parts[pi][1]
        inherited = False
        j = pi
        while j + 1 < len(c.parts):
            a, b = c.parts[j]
            seg = sent[max(a, m.end) if j == pi else a:b]
            if _CUE.search(seg) or _PRED_END.search(sent[a:b]) or (j > pi and len(sent[a:b].strip()) > 30):
                break
            if re.search(r"\d", sent[c.parts[j + 1][0]:c.parts[j + 1][1]]):
                break
            j += 1
            unit_end = c.parts[j][1]
            inherited = True
        tail = sent[m.end:unit_end]
        head = sent[c.parts[pi][0]:m.start]
        after = sent[m.end:c.end]

        # figurative use ("발작처럼 몸이 떨려요"): dropped, except figurative-by-design forms and pain / quality concepts
        if not m.fig and _METAPHOR.match(sent[m.end:m.end + 16]) and concept.cat not in ("QUAL",) \
                and "SYM:pain" not in self.lex.ancestors(m.cid) and m.cid != "SYM:pain":
            return None

        polarity, cue, conf = "present", "default", 0.8
        if m.is_absence:
            if re.match(r"[가-힣]{0,2}\s?(?:지|진|지는|지를|지도)\s?(?:못|않)|\s?(?:못|안)\s", tail):
                polarity, cue, conf = "present", "absence-form negated", 0.7
            else:
                polarity, cue, conf = "absent", "absence-form", 0.85
        else:
            if m.kind == "re" and _NEG_INSIDE.search(m.text):
                polarity, cue, conf = "absent", "neg-inside:" + m.text, 0.85
            elif _PRE_NEG_EN.search(head) or _PRE_NEG_KO.search(head):
                polarity, cue, conf = "absent", "pre-neg", 0.85
            elif _WHEN.match(sent[m.end:m.end + 12]):
                polarity, cue, conf = "present", "when-clause", 0.75  # "기침할 때 가래는 안 나와요": the cough is there
            else:
                kind, ctext = _first_cue(tail)
                if kind == "neg":
                    polarity, cue, conf = "absent", ("list:" if inherited else "") + ctext, 0.9 if not inherited else 0.85
                elif kind == "unc":
                    polarity, cue, conf = "uncertain", ctext, 0.6
                elif kind == "pos":
                    polarity, cue, conf = "present", ctext, 0.9
            if polarity == "present" and cue == "default" and concept.needs_cue:
                return None
        hedged = bool(_HEDGE.search(tail[:24]) or (self.source in ("exam", "test") and _HEDGE.search(after[:40])))
        if hedged and polarity == "present":
            conf = min(conf, 0.65)
        hypothetical = question or bool(_HYPO.search(tail[:20]))
        if hypothetical:
            polarity, conf = ("uncertain" if polarity == "present" else polarity), min(conf, 0.5)
        subject = self._subject(m.start, subj_marks, sent[m.end:c.end])
        part_start = max((a for a, _b in c.parts if a <= m.start), default=c.start)
        window = sent[max(part_start, m.start - 14):m.start] + " " + m.text
        lat = _LAT.search(window)
        site_m = _SITE.search(m.text) or _SITE.search(sent[max(part_start, m.start - 12):m.start])
        sev = ""
        around = sent[max(c.start, m.start - 10):min(c.end, m.end + 12)]
        if _SEVERE.search(around):
            sev = "severe"
        elif _MODERATE.search(around):
            sev = "moderate"
        elif _MILD.search(around):
            sev = "mild"
        nrs = _NRS.search(sent[c.start:c.end])
        if nrs and concept.cat in ("SYM", "QUAL"):
            n = int(nrs.group(1) or nrs.group(2))
            if 0 <= n <= 10:
                sev = ("severe" if n >= 7 else "moderate" if n >= 4 else "mild") + f" NRS {n}/10"
        return Finding(
            concept=m.cid, label=concept.ko or concept.en, span=m.text, polarity=polarity, subject=subject,
            temporality=temporality, onset=onset, laterality=lat.lastgroup if lat else "",
            site=site_m.group() if site_m else "", severity=sev, hedged=hedged, hypothetical=hypothetical,
            confidence=round(conf, 2), source=self.source, start=m.start, end=m.end, clause=sent[c.start:c.end].strip(),
            cue=cue)

    # -------------------------------------------------------------- measured values
    def _mk(self, cid: str, span: str, s: int, polarity: str, value: float | None, unit: str, direction: str,
            cue: str, clause: str) -> Finding:
        c = self.lex.concept(cid)
        return Finding(concept=cid, label=(c.ko if c else cid), span=span, polarity=polarity, value=value, unit=unit,
                       direction=direction, confidence=0.95 if polarity != "uncertain" else 0.6, source=self.source,
                       start=s, end=s + len(span), clause=clause, cue=cue)

    def _measures(self, ctext: str, coff: int, sent: str) -> list[Finding]:
        if not re.search(r"\d", ctext):
            return self._qual_labs(ctext, coff)
        out: list[Finding] = []
        clause = ctext.strip()
        seen_temp = False
        for rx in (_TEMP, _TEMP_BARE, _TEMP_F):
            if seen_temp:
                break
            for m in rx.finditer(ctext):
                v = float(m.group(1))
                unit = (m.group(2) or "").replace(" ", "")
                if "f" in unit or "℉" in unit:
                    v = round((v - 32) * 5 / 9, 1)
                if not 30 <= v <= 44:
                    continue
                if rx is _TEMP and not unit and "." not in m.group(1) and not re.match(r"체온|bt|temp", m.group()):
                    continue
                pol = "present" if v >= 37.8 else "absent" if v < 37.5 else "uncertain"
                d = "high" if v >= 37.8 else "normal" if v < 37.5 else "high"
                out.append(self._mk("SYM:fever", m.group().strip(), coff + m.start(), pol, v, "°C", d, "value", clause))
                if v >= 39.0:
                    out.append(self._mk("SYM:high_fever", m.group().strip(), coff + m.start(), "present", v, "°C", "high", "value", clause))
                if v < 35.0:
                    out.append(self._mk("SIGN:hypothermia", m.group().strip(), coff + m.start(), "present", v, "°C", "low", "value", clause))
                seen_temp = True
        for m in _HR.finditer(ctext):
            v = float(m.group(1))
            if not 20 <= v <= 250:
                continue
            span = m.group().strip()
            if v >= 60:
                out.append(self._mk("SIGN:tachycardia", span, coff + m.start(), "present" if v > 100 else "absent", v,
                                    "/min", "high" if v > 100 else "normal", "value", clause))
            if v <= 100:
                out.append(self._mk("SIGN:bradycardia", span, coff + m.start(), "present" if v < 60 else "absent", v,
                                    "/min", "low" if v < 60 else "normal", "value", clause))
        for m in _RR.finditer(ctext):
            v = float(m.group(1))
            if not 4 <= v <= 80:
                continue
            span = m.group().strip()
            out.append(self._mk("SIGN:tachypnea", span, coff + m.start(), "present" if v > 20 else "absent", v, "/min",
                                "high" if v > 20 else "normal", "value", clause))
            if v < 10:
                out.append(self._mk("SIGN:bradypnea", span, coff + m.start(), "present", v, "/min", "low", "value", clause))
        for m in _BP.finditer(ctext):
            sbp, dbp = float(m.group(1)), float(m.group(2))
            if not (40 <= sbp <= 300 and 20 <= dbp <= 200):
                continue
            span = m.group().strip()
            hypo = "present" if sbp < 90 else "absent" if sbp >= 100 else "uncertain"
            high = sbp >= 140 or dbp >= 90
            if not high:
                out.append(self._mk("SIGN:hypotension", span, coff + m.start(), hypo, sbp, "mmHg",
                                    "low" if sbp < 90 else "normal", "value", clause))
            if hypo != "present":
                out.append(self._mk("SIGN:elevated_bp", span, coff + m.start(), "present" if high else "absent", sbp,
                                    "mmHg", "high" if high else "normal", "value", clause))
        for m in _SPO2.finditer(ctext):
            v = float(m.group(1))
            if not 40 <= v <= 100:
                continue
            pol = "present" if v < 92 else "absent" if v >= 95 else "uncertain"
            out.append(self._mk("SIGN:hypoxemia", m.group().strip(), coff + m.start(), pol, v, "%",
                                "low" if v < 95 else "normal", "value", clause))
        out += self._labs(ctext, coff, clause, sent)
        return out

    def _labs(self, ctext: str, coff: int, clause: str, sent: str) -> list[Finding]:
        out: list[Finding] = []
        urine = bool(_URINE_CTX.search(sent))
        for key, rx, hi, th, lo, tl in _LAB_RX:
            for m in rx.finditer(ctext):
                after = ctext[m.end():m.end() + 60]
                if key in ("wbc",) and urine and re.match(r"[^\d]{0,6}\d+(?:\s*-\s*\d+)?\s*/\s*hpf", after):
                    continue
                if re.match(r"\s?(?:수치(?:가|는)?\s?)?(?:상승|증가|높|감소|저하|낮|↑|↓)", after):
                    continue
                if key == "wbc" and urine and re.match(r"[^\d]{0,8}(?:음성|양성|negative|positive)", after):
                    continue
                if key == "glu" and urine and re.match(r"\s*:?\s*(?:음성|양성|negative|positive|trace|미량|\+|[1-4]\+)", after):
                    continue  # urine glucose (LAB:glycosuria), not blood glucose
                vm = _VALUE.match(after)
                if not vm or re.search(r"1\s?:\s?$|[a-z]$", after[:vm.start(2)]) or re.match(r"\s?(?:일|주|개월|년|시간|분|세|살|번|회|배)(?![a-z])", after[vm.end(2):vm.end(2) + 3]):
                    if _QUAL_NORMAL.match(after):
                        span = ctext[m.start():m.end() + _QUAL_NORMAL.match(after).end()]
                        for cid in (hi, lo):
                            if cid:
                                out.append(self._mk(cid, span, coff + m.start(), "absent", None, "", "normal", "normal-word", clause))
                    continue
                v = float(vm.group(2))
                if vm.group(3) == "만":
                    v *= 10000
                elif vm.group(3) == "천":
                    v *= 1000
                if vm.group(4):
                    v *= 1000
                unit = (vm.group(5) or "").strip(".")
                if key in ("wbc", "plt") and v < 2000 and not vm.group(4):
                    v *= 1000  # x10^3/μL notation ("WBC 14.2")
                if key == "crp" and unit.startswith("mg/l"):
                    v /= 10
                if key == "eos" and unit != "%" and v > 50:  # absolute count: > 500/μL = eosinophilia
                    th_eff = 500.0
                else:
                    th_eff = th
                cmp_ = (vm.group(1) or "").strip()
                direction, ref = _ref_direction(after[vm.end():], float(vm.group(2)))
                if not direction:
                    if hi is not None and th_eff is not None and v > th_eff and not cmp_.startswith(("<", "≤")):
                        direction = "high"
                    elif lo is not None and tl is not None and v < tl and not cmp_.startswith((">", "≥")):
                        direction = "low"
                    else:
                        direction = "normal"
                span = ctext[m.start():m.end() + vm.end() + len(ref)].strip()
                for cid, want in ((hi, "high"), (lo, "low")):
                    # an abnormal value names one side only ("WBC 14,200" is leukocytosis, not "no leukopenia")
                    if cid and (direction == "normal" or direction == want):
                        out.append(self._mk(cid, span, coff + m.start(), "present" if direction == want else "absent",
                                            v, unit, direction, "value" + ("+ref" if ref else ""), clause))
        if urine:
            for cid, rx in _URINE_ITEMS:
                for m in rx.finditer(ctext):
                    val = m.group(1)
                    pol = "absent" if val in ("음성", "negative") else "present"
                    out.append(self._mk(cid, m.group().strip(), coff + m.start(), pol, None, "", "", "urine", clause))
            for m in _URINE_RBC.finditer(ctext):
                v = float(m.group(2) or m.group(1))
                out.append(self._mk("LAB:urine_blood", m.group().strip(), coff + m.start(), "present" if v >= 3 else "absent",
                                    v, "/HPF", "high" if v >= 3 else "normal", "urine", clause))
        return out

    def _qual_labs(self, ctext: str, coff: int) -> list[Finding]:
        out = []
        for key, rx, hi, th, lo, tl in _LAB_RX:
            for m in rx.finditer(ctext):
                after = ctext[m.end():m.end() + 30]
                q = _QUAL_NORMAL.match(after)
                if q:
                    span = ctext[m.start():m.end() + q.end()]
                    for cid in (hi, lo):
                        if cid:
                            out.append(self._mk(cid, span, coff + m.start(), "absent", None, "", "normal", "normal-word", ctext.strip()))
        return out

    def _kb_tests(self, sent: str, off: int, clauses: list[_Clause]) -> list[Finding]:
        try:
            from doctor_agent.knowledge import kb_tests
        except Exception:  # noqa: BLE001 - the parser works without the KB module
            return []
        out = []
        for c in clauses:
            ctext = sent[c.start:c.end]
            for fid, (pol, by_value) in kb_tests.detect(ctext).items():
                cids = self.lex.by_kb("TF:" + fid)
                if not cids:
                    continue
                f = self._mk(cids[0], ctext.strip()[:60], off + c.start, "present" if pol > 0 else "absent", None, "",
                             ("high" if pol > 0 else "normal") if by_value else "", "kb_tests" + ("-value" if by_value else ""),
                             ctext.strip())
                f.confidence = 0.85
                out.append(f)
        return out

    # -------------------------------------------------------------- yes / no answers
    def _yes_no(self, found: list[Finding]) -> list[Finding]:
        q = normalize(self.ctx.get("question", ""))
        a = self.t
        if not q or len(a) > 120:
            return []
        qms = [m for m in self.lex.scan(q) if not m.is_absence]
        if not qms:
            return []
        if _UNSURE_ANSWER.search(a):
            answer = "uncertain"
        elif _NO.search(a):
            answer = "no"
        elif _YES.search(a):
            answer = "yes"
        elif not found and len(a) <= 40:
            # elliptical answer naming no finding ("계단 오를 때만 좀 차요", "한 번도 없어요"): its own cue decides
            kind, _c = _first_cue(a)
            answer = {"neg": "no", "unc": "uncertain"}.get(kind, "yes")
        else:
            return []
        q_neg = bool(_Q_NEGATIVE.search(q))
        own = {f.concept for f in found}
        own |= {anc for f in found for anc in self.lex.ancestors(f.concept)}
        own |= {d for f in found for d in self.lex.descendants(f.concept)}
        first = _sentences(a)[0] if _sentences(a) else (0, len(a), False)
        s0 = a[first[0]:first[1]]
        out = []
        subj_q = "family" if _FAMILY_CTX.search(q) or re.search(_FAMILY, q) else self.default_subject
        for m in qms:
            if m.cid in own:
                continue
            if answer == "uncertain":
                pol = "uncertain"
            elif answer == "yes":
                pol = "absent" if q_neg else "present"
            else:
                pol = ("present" if re.search(r"있", s0) else "uncertain") if q_neg else "absent"
            c = self.lex.concept(m.cid)
            t_mark = self._temporality(s0) or self._temporality(q)
            onset = _ONSET.search(s0)
            f = Finding(concept=m.cid, label=c.ko if c else m.cid, span=s0.strip()[:40], polarity=pol, subject=subj_q,
                        temporality=t_mark or "current", onset=onset.group().strip() if onset else "",
                        hedged=bool(_HEDGE.search(s0)), confidence=0.7 if pol != "uncertain" else 0.5,
                        source=self.source, start=first[0], end=first[1], clause=s0.strip(),
                        cue=f"yes/no:{answer}{'(neg-q)' if q_neg else ''}")
            if _MILD.search(s0[:8]):
                f.severity = "mild"
            out.append(f)
        return out


def _dedupe(found: list[Finding], lex: Lexicon) -> list[Finding]:
    """One finding per (concept, clause, subject): measured values are merged into the mention of the same concept;
    an explicit cue beats the default; a generic parent (SYM:pain, GRP:...) is dropped when a child with the same
    polarity is in the same clause."""
    by_key: dict[tuple, Finding] = {}
    order: list[tuple] = []
    for f in found:
        k = (f.concept, f.clause, f.subject)
        g = by_key.get(k)
        if g is None:
            by_key[k] = f
            order.append(k)
            continue
        if f.value is not None and g.value is None:
            if g.cue == "default" or g.polarity == f.polarity:
                by_key[k] = replace(f, span=g.span if g.cue != "default" else f.span)
            else:
                g.value, g.unit, g.direction = f.value, f.unit, f.direction
        elif g.cue == "default" and f.cue != "default":
            by_key[k] = f
    out = [by_key[k] for k in order]
    keep = []
    for f in out:
        if f.concept == "HX:pmh" and f.polarity == "present" and any(
                o.clause == f.clause and o.concept.startswith("HX:") and o.concept != "HX:pmh" for o in out):
            continue
        if f.concept == "SYM:pain" and any(o.polarity == f.polarity and "SYM:pain" in lex.ancestors(o.concept)
                                           and 0 <= f.start - o.end <= 12 for o in out):
            continue  # "온몸이 쑤시고 아팠는데": the generic word repeats the specific pain just named
        if f.concept == "SYM:pain":
            if any(o is not f and o.clause == f.clause and o.polarity == f.polarity and f.concept in lex.ancestors(o.concept)
                   for o in out):
                continue
        keep.append(f)
    keep.sort(key=lambda f: (f.start, f.end))
    return keep


# ------------------------------------------------------------------------------------------------ public API
def parse(text: str, source: str = "patient", context: dict | None = None, lexicon: Lexicon | None = None) -> list[Finding]:
    """Findings in one text (a patient answer, an exam/test report or a claim).

    context (optional): {"question": the doctor's question this text answers, "key": case-file key such as
    "가족력|family history", "subject": default subject}. Only the question/key of the same case are ever used."""
    return _Parser(text or "", source, context, lexicon or LEXICON).run()


def _expand_up(cid: str, lex: Lexicon) -> tuple[str, ...]:
    return (cid,) + lex.ancestors(cid)


def concepts_in(text: str, *, source: str = "patient", polarity: str | None = "present", subject: str | None = "patient",
                expand: bool = True, include_hypothetical: bool = False, context: dict | None = None,
                lexicon: Lexicon | None = None) -> set[str]:
    """Concept ids found in the text: affirmed ones about the patient by default (polarity=None / subject=None
    disable those filters); with expand, each concept also brings its ancestors (SIGN:rlq_tenderness ->
    SIGN:abdominal_tenderness -> SIGN:tenderness)."""
    lex = lexicon or LEXICON
    out: set[str] = set()
    for f in parse(text, source, context, lex):
        if polarity and f.polarity != polarity:
            continue
        if subject and f.subject != subject:
            continue
        if f.hypothetical and not include_hypothetical:
            continue
        out.update(_expand_up(f.concept, lex) if expand else (f.concept,))
    return out


def _with_descendants(ids, lex: Lexicon) -> set[str]:
    ids = [ids] if isinstance(ids, str) else list(ids)
    out = set(ids)
    for i in ids:
        out.update(lex.descendants(i))
    return out


def affirmed(text: str, concept_ids, *, source: str = "patient", lexicon: Lexicon | None = None) -> bool:
    """Any of the concepts (or their descendants) is present for the patient: the drop-in for contains_affirmed()."""
    lex = lexicon or LEXICON
    want = _with_descendants(concept_ids, lex)
    return any(f.concept in want and f.polarity == "present" and f.subject == "patient" and not f.hypothetical
               for f in parse(text, source, None, lex))


def denied(text: str, concept_ids, *, source: str = "patient", lexicon: Lexicon | None = None) -> bool:
    """Some concept is stated absent for the patient and none is affirmed (the drop-in for polarity(...) == "neg")."""
    lex = lexicon or LEXICON
    want = _with_descendants(concept_ids, lex)
    fs = [f for f in parse(text, source, None, lex) if f.concept in want and f.subject == "patient"]
    return any(f.polarity == "absent" for f in fs) and not any(f.polarity == "present" and not f.hypothetical for f in fs)


def _supports(c: Finding, e: Finding, lex: Lexicon) -> bool:
    if e.polarity == "uncertain" or c.polarity == "uncertain" or e.hypothetical:
        return False
    if c.subject != e.subject:
        return False
    if c.polarity != e.polarity:
        return False
    if e.concept != c.concept:
        if c.polarity == "present" and c.concept in lex.ancestors(e.concept):
            pass  # a present child supports its parent ("우하복부 압통" -> "복부 압통")
        elif c.polarity == "absent" and e.concept in lex.ancestors(c.concept) and e.concept != "SYM:pain":
            pass  # an absent parent supports an absent child ("열은 없어요" -> "고열 없음"; "소화기 증상 없음" -> "구토 없음")
        else:
            return False
    if c.value is not None:
        if e.value is None or not _num_close(c.value, e.value):
            return False
    if c.laterality and e.laterality and c.laterality != e.laterality:
        return False
    return True


def match(claim_text: str, findings, *, lexicon: Lexicon | None = None, context: dict | None = None) -> tuple[bool, str]:
    """Is every finding named in the claim supported by the evidence? findings: list[Finding] or evidence text
    (parsed as patient speech). Same concept (or a present child / an absent parent), same polarity and subject,
    numbers within rounding, no laterality clash. Uncertain or hypothetical evidence never supports.
    Returns (True, evidence spans) or (False, "") — also (False, "") when the claim names no known concept."""
    lex = lexicon or LEXICON
    ev = parse(findings, "patient", context, lex) if isinstance(findings, str) else list(findings)
    claims = parse(claim_text, "claim", None, lex)
    if not claims:
        return False, ""
    # generic pain in a claim that also names a specific pain concept is redundant
    specific = {c.concept for c in claims}
    claims = [c for c in claims if not (c.concept == "SYM:pain" and any("SYM:pain" in lex.ancestors(s) for s in specific))]
    spans: list[str] = []
    for c in claims:
        hit = next((e for e in ev if _supports(c, e, lex)), None)
        if hit is None:
            return False, ""
        spans.append(hit.span)
    return True, " / ".join(dict.fromkeys(spans))


# ------------------------------------------------------------------------------------------------ other vocabularies
_SPAN_ID = "SPAN:external"
_SPAN_CONCEPT = Concept(_SPAN_ID, "SPAN", "", "")
# LEXICON plus the stub concept: read-only static data built once at import (never modified afterwards)
_SPAN_LEXICON = replace(LEXICON, concepts={**LEXICON.concepts, _SPAN_ID: _SPAN_CONCEPT})


def _span_lexicon(lex: Lexicon) -> Lexicon:
    if lex is LEXICON:
        return _SPAN_LEXICON
    return replace(lex, concepts={**lex.concepts, _SPAN_ID: _SPAN_CONCEPT})


def assess_spans(text: str, spans, source: str = "claim", context: dict | None = None,
                 lexicon: Lexicon | None = None) -> list[Finding | None]:
    """Polarity / subject / hedge / hypothetical of arbitrary spans, read with the same rules as parse(), for callers
    that find mentions with their own vocabulary (the KB term-label scan in knowledge/kb.py). spans: (start, end)
    offsets in normalize(text). Returns one Finding (concept "SPAN:external") or None (span outside any sentence or
    in a "result unavailable" sentence) per span, in order. Similes are not dropped (no figurative flag here)."""
    lex = _span_lexicon(lexicon or LEXICON)
    p = _Parser(text or "", source, context, lex)
    sents = _sentences(p.t)
    cache: dict[int, tuple] = {}
    out: list[Finding | None] = []
    for s, e in spans:
        si = next((i for i, (a, b, _q) in enumerate(sents) if a <= s < b), None)
        if si is None:
            out.append(None)
            continue
        a, b, is_q = sents[si]
        sent = p.t[a:b]
        if _UNAVAILABLE.search(sent):
            out.append(None)
            continue
        if si not in cache:
            question = p.source == "patient" and (is_q or bool(_QUESTION_END.search(sent.strip())) and "?" in sent)
            cache[si] = (_clauses(sent), [(m.start(), m.lastgroup) for m in _SUBJ.finditer(sent)], question)
        clauses, subj_marks, question = cache[si]
        ms, me = s - a, max(s - a + 1, min(e, b) - a)
        c = next((c for c in clauses if c.start <= me - 1 < c.end), clauses[-1])
        m = Mention(ms, me, _SPAN_ID, "med", True, sent[ms:me])
        f = p._mention(m, sent, c, [m], "current", "", subj_marks, question)
        if f is not None:
            f.start, f.end = f.start + a, f.end + a
        out.append(f)
    return out
