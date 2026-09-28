"""Curated finding → KB-term mapping tables (source tag "curated"; written by the team, not derived from any dataset).

Used by kb.py to map short Korean findings (doctor-model wording, patient speech, exam and lab reports) onto the
existing KB terms. Nothing here adds disease knowledge: these tables only say which *term* a finding text means.
Keys are English term labels that already exist in data/kb/kb.json.gz (resolved at load time; unknown labels are
skipped). Stdlib only.

Since the migration to the normalisation layer (doctor_agent.nlp, 2026-09-28) kb.py reads patient wording, negation
and measured values through the lexicon. SYNONYMS / REGEX / lab_terms() are no longer used at runtime (switches
KnowledgeBase.CURATED_SYN / LAB_VALUES="curated" bring them back for A/B runs); they stay as build-time sources:
scripts/build_lexicon.py merges them into data/lexicon (provenance "curated"), and kb.concept_links() turns each
phrase list into a concept → term link in data/lexicon/kb_links.json. Still read at runtime:

    STOP_TERMS    over-generic KB terms that must not be matched from free text ("observation", "increase", ...)
    BAD_LABELS    ambiguous labels removed from one term only ("가슴 통증" is chest pain, not mastodynia; "무릎의
                  열감" is not fever)
    BLOCK_WORDS   words that contain a shorter label but mean something else ("수포음" = crackles, not vesicle)
    NAME_SUBS / NAME_MODIFIERS   diagnosis-name normalisation
Not read at runtime (build-time sources, A/B switches):
    SYNONYMS      extra Korean/English phrases for a term
    REGEX         flexible phrasings for a term
    lab_terms()   numeric vitals/lab values → abnormality terms ("WBC 14,200/μL" → leukocytosis); replaced by the
                  measured findings of nlp.parse (same analytes, reference ranges read)
"""
from __future__ import annotations

import re

SOURCE = "curated"

SYNONYMS: dict[str, list[str]] = {
    # constitutional
    "fever": ["열이 나", "열이 났", "열이 있", "열이 좀", "열도 나", "열도 있", "열나", "열이 오르", "열이 올라", "발열감", "고열이", "체온 상승"],
    "high fever": ["고열", "40도", "39도"],
    "chills": ["오한", "으슬으슬", "몸이 떨리"],
    "night sweats": ["밤에 땀", "야간 발한", "자면서 땀", "도한"],
    "sweaty": ["식은땀", "땀을 많이", "발한"],
    "fatigue": ["피로", "피곤", "기운이 없", "기력이 없", "쉽게 지치", "무기력"],
    "weight loss": ["체중 감소", "체중이 감소", "체중이 줄", "살이 빠", "몸무게가 줄", "kg 빠", "kg 감소"],
    "weight gain": ["체중 증가", "체중이 늘", "살이 쪘", "살이 찌"],
    "decreased appetite": ["식욕 부진", "식욕 저하", "식욕이 없", "식욕이 떨어", "입맛이 없", "입맛이 떨어"],
    "malaise": ["전신 쇠약", "몸이 안 좋", "컨디션이 안 좋"],
    "dehydration": ["탈수", "구강 점막 건조", "점막이 건조", "피부 긴장도 저하"],
    # head / neuro
    "headache": ["두통", "머리가 아프", "머리가 아파", "머리가 깨질", "머리 통증", "편두통"],
    "severe headache": ["벼락두통", "심한 두통", "극심한 두통", "생애 최악의 두통", "망치로 맞은"],
    "dizziness": ["어지럼", "어지러", "현기증"],
    "vertigo": ["빙빙 도", "빙글빙글", "회전성 어지럼", "현훈"],
    "syncope": ["실신", "기절", "쓰러졌", "정신을 잃", "의식을 잃"],
    "seizure": ["경련", "발작", "경기", "간질 발작"],
    "mental confusion": ["혼돈", "혼란", "착란", "횡설수설", "지남력 저하", "지남력 장애", "헛소리"],
    "somnolence": ["기면", "졸려", "졸림"],
    "coma": ["혼수", "무반응"],
    "memory loss": ["기억력 저하", "기억력이 떨어", "건망", "깜빡", "기억이 안"],
    "weakness": ["근력 저하", "근력 약화", "위약", "힘이 빠", "힘이 없", "힘이 안 들어", "근력 감소"],
    "hemiparesis": ["편마비", "반신 마비", "한쪽 팔다리", "편측 마비", "편측 위약", "우측 상지 근력", "좌측 상지 근력"],
    "paralysis": ["마비"],
    "hypoesthesia": ["감각 저하", "감각이 둔", "감각 둔마", "먹먹", "무감각"],
    "paresthesia": ["저림", "저리", "저릿", "찌릿", "감각 이상", "따끔거"],
    "tremor": ["떨림", "손떨림", "진전"],
    "dysarthria": ["구음 장애", "구음장애", "발음이 어눌", "말이 어눌", "혀가 꼬"],
    "loss of speech": ["실어증", "말을 못", "말이 안 나"],
    "face drooping": ["안면 마비", "안면마비", "입이 돌아", "얼굴이 처", "입꼬리가 처", "안면 비대칭"],
    "diplopia": ["복시", "둘로 보", "두 개로 보", "겹쳐 보"],
    "ptosis": ["안검하수", "눈꺼풀 처짐", "눈꺼풀이 처", "눈꺼풀이 내려"],
    "blurred vision": ["시야 흐림", "시야가 흐", "흐릿하게 보", "흐리게 보", "뿌옇게", "침침"],
    "vision loss": ["시력 저하", "시력 소실", "시력이 떨어", "안 보여", "시야 결손"],
    "photophobia": ["눈부심", "빛이 눈부", "빛을 싫어", "광과민"],
    "neck stiffness": ["경부 강직", "목 강직", "목이 뻣뻣", "항부 강직", "경직된 목"],
    "trouble walking": ["보행 장애", "걷기 힘", "걷기가 힘", "걸음이 불안", "보행 불안", "비틀거"],
    "cerebellar ataxia": ["운동 실조", "운동실조", "실조"],
    "deafness": ["난청", "청력 저하", "청력 감소", "잘 안 들", "귀가 먹먹"],
    "tinnitus": ["이명", "귀울림", "귀에서 소리"],
    "insomnia": ["불면", "잠을 못", "잠이 안"],
    "anxiety": ["불안", "초조"],
    "sadness": ["우울", "기분이 가라앉", "의욕이 없", "흥미가 없"],
    "hallucination": ["환청", "환시", "환각", "헛것이"],
    "delusion": ["망상", "감시당", "누가 쫓"],
    "irritability": ["짜증", "예민"],
    # respiratory / cardiac
    "cough": ["기침", "마른기침", "잔기침"],
    "productive cough with abnormal sputum": ["가래", "객담", "누런 가래", "화농성 객담"],
    "hemoptysis": ["객혈", "피 섞인 가래", "가래에 피", "피가 섞인 가래", "혈담"],
    "dyspnea": ["호흡곤란", "호흡 곤란", "숨이 차", "숨이 찬", "숨참", "숨쉬기 힘", "숨쉬기가 힘", "숨이 가쁘", "숨 가쁨", "숨가쁨", "숨이 막"],
    "wheeze": ["천명", "쌕쌕", "그르렁"],
    "stridor": ["협착음", "그렁거"],
    "chest pain": ["흉통", "가슴 통증", "가슴이 아프", "가슴이 아파", "가슴을 쥐어짜", "가슴이 조이", "가슴이 뻐근", "가슴 압박감", "흉부 통증", "흉부 압박"],
    "pleuritic chest pain": ["숨을 들이쉴 때", "숨을 쉴 때 아", "깊게 숨", "호흡 시 악화", "흡기 시 악화", "흉막성"],
    "chest tightness": ["가슴이 답답", "흉부 답답"],
    "palpitation": ["두근거", "심계항진", "가슴이 뛰", "심장이 빨리"],
    "tachycardia": ["빈맥", "맥박이 빠르", "맥이 빠르"],
    "bradycardia": ["서맥", "맥박이 느리", "맥이 느리"],
    "heart arrhythmia": ["불규칙한 맥", "맥박이 불규칙", "불규칙 리듬", "부정맥", "심방세동"],
    "hypotension": ["저혈압", "혈압 저하", "혈압이 낮"],
    "arterial hypertension": ["고혈압", "혈압이 높"],
    "cyanosis": ["청색증", "입술이 파랗", "입술이 퍼렇"],
    "edema": ["부종", "붓기", "부었", "부어", "붓고", "함요부종", "함요 부종", "다리가 붓"],
    "nail clubbing": ["곤봉지", "곤봉 손가락", "곤봉형"],
    "tachypnea": ["빈호흡", "호흡수 증가", "호흡이 빠르", "숨을 빠르게"],
    # GI
    "nausea": ["구역", "오심", "메스꺼", "메스껍", "속이 울렁", "울렁거", "속이 안 좋", "헛구역"],
    "vomiting": ["구토", "토했", "토함", "토하", "게워"],
    "hematemesis": ["토혈", "피를 토", "커피 찌꺼기", "커피색 구토"],
    "abdominal pain": ["복통", "배가 아프", "배가 아파", "배 아픔", "배아픔", "복부 통증", "윗배", "아랫배", "명치", "상복부 통증", "하복부 통증", "옆구리 통증"],
    "abdominal tenderness": ["복부 압통", "상복부 압통", "하복부 압통", "우하복부 압통", "우상복부 압통", "좌하복부 압통", "심와부 압통", "명치 압통"],
    "tenderness": ["압통", "누르면 아프", "누르면 아파"],
    "diarrhea": ["설사", "묽은 변", "물 같은 변", "무른 변"],
    "constipation": ["변비", "변을 못", "배변이 어렵"],
    "hematochezia": ["혈변", "선혈변", "변에 피", "대변에 피", "피가 섞인 변", "항문 출혈"],
    "melena": ["흑색변", "흑변", "검은 변", "까만 변", "짜장면 같은"],
    "bloating": ["복부 팽만", "배가 더부룩", "더부룩", "배에 가스", "배가 불러", "복부 팽창"],
    "heartburn": ["속쓰림", "속이 쓰리", "가슴 쓰림", "신물", "위산 역류", "가슴이 타"],
    "dysphagia": ["삼킴 곤란", "삼킴곤란", "연하곤란", "연하 곤란", "삼키기 힘", "삼키기가 힘", "음식이 걸리", "목에 걸리"],
    "jaundice": ["황달", "눈이 노랗", "피부가 노랗", "공막 황염", "공막 황달"],
    "ascites": ["복수", "배에 물"],
    "hepatomegaly": ["간비대", "간종대", "간 비대", "간이 커", "간이 만져"],
    "splenomegaly": ["비장비대", "비장 비대", "비종대", "비장이 커", "비장이 만져"],
    "hepatosplenomegaly": ["간비장비대", "간비종대"],
    "dark urine": ["소변 색이 진", "진한 소변", "콜라색 소변", "갈색 소변", "소변이 진해"],
    "abdominal mass": ["복부 종괴", "배에 혹", "복부에 만져지는"],
    "indigestion": ["소화불량", "소화가 안", "체한"],
    # GU / gyn
    "dysuria": ["배뇨통", "배뇨 시 통증", "소변 볼 때 아", "소변볼 때 아", "소변 볼 때 따", "소변볼 때 따", "배뇨 시 작열"],
    "frequent urination": ["빈뇨", "소변을 자주", "화장실을 자주"],
    "urinary urgency": ["요절박", "급박뇨", "소변이 급"],
    "polyuria": ["다뇨", "소변량 증가", "소변을 많이", "소변 양이 많"],
    "polydipsia": ["다음", "다갈", "갈증", "물을 많이 마", "목이 자주 마르"],
    "hematuria": ["혈뇨", "소변에 피", "붉은 소변", "소변이 빨갛"],
    "proteinuria": ["단백뇨", "거품뇨", "소변에 거품"],
    "vaginal bleeding": ["질 출혈", "질출혈", "하혈", "부정 출혈", "피가 비치", "출혈이 비치"],
    "vaginal discharge": ["질 분비물", "냉이", "대하"],
    "pelvic pain": ["골반통", "골반 통증", "골반이 아"],
    "amenorrhea": ["무월경", "생리를 안", "생리가 없", "생리가 늦", "월경이 없"],
    "menorrhagia": ["월경과다", "생리량이 많", "생리양이 많"],
    "erectile dysfunction": ["발기부전", "발기 부전", "발기가 안"],
    # MSK / skin
    "arthralgia": ["관절통", "관절 통증", "관절이 아프", "관절이 아파", "무릎이 아프", "손가락 관절", "손목이 아프"],
    "arthritis": ["관절염", "관절 부종", "관절이 붓", "관절 종창", "관절이 부"],
    "joint stiffness": ["관절 강직", "관절이 뻣뻣", "손이 뻣뻣"],
    "morning stiffness": ["아침 강직", "조조강직", "조조 강직", "아침에 뻣뻣", "아침에 손이 뻣뻣"],
    "myalgia": ["근육통", "근육 통증", "몸살", "근육이 아프", "온몸이 쑤"],
    "back pain": ["요통", "허리 통증", "허리가 아프", "허리가 아파", "등 통증", "등이 아프"],
    "neck pain": ["목 통증", "뒷목", "목이 아프"],
    "bone pain": ["뼈 통증", "뼈가 아프", "골통"],
    "limb pain": ["다리 통증", "팔 통증", "다리가 아프", "팔이 아프", "종아리 통증"],
    "muscle atrophy": ["근위축", "근육 위축", "근육이 빠"],
    "exanthem": ["발진", "붉은 반점", "뾰루지", "피부 병변"],
    "maculopapular rash": ["반구진", "홍반성 구진", "구진"],
    "erythema": ["홍반", "발적", "피부가 붉", "빨갛게 부"],
    "itch": ["가려움", "가렵", "가려워", "가려운", "소양", "간지러"],
    "urticaria": ["두드러기", "팽진이"],  # bare "팽진" also hits "피부 팽진도" (skin turgor)
    "petechia": ["점상출혈", "점상 출혈", "자반", "붉은 점"],
    "easy bruising": ["멍이 잘", "멍이 쉽게", "쉽게 멍"],
    "blisters": ["물집", "수포성", "수포가", "수포 형성"],
    "hair loss": ["탈모", "머리카락이 빠", "머리가 빠"],
    "pallor": ["창백", "얼굴이 하얗"],
    "xerostomia": ["입이 마르", "입안이 마르", "구강 건조", "입마름"],
    "dry eye": ["눈이 건조", "안구 건조", "눈이 뻑뻑", "눈이 모래"],
    "raynaud phenomenon": ["레이노", "손가락이 하얗", "손끝이 하얗", "손가락 색이 변"],
    "mouth ulcer": ["구강 궤양", "입안이 헐", "입병", "구내염", "아프타"],
    # ENT / eye
    "sore throat": ["인후통", "목이 아프", "목이 따끔", "목구멍이 아프", "삼킬 때 아", "인두 발적"],
    "runny nose": ["콧물", "비루"],
    "nasal congestion": ["코막힘", "코가 막"],
    "sneeze": ["재채기"],
    "otalgia": ["귀 통증", "귀가 아프", "이통"],
    "ocular redness": ["눈이 충혈", "안구 충혈", "결막 충혈", "눈이 빨갛"],
    "dysphonia": ["쉰 목소리", "목소리가 쉬", "애성", "목이 쉬"],
    "lymphadenopathy": ["림프절 비대", "림프절 종대", "림프절병증", "임파선이 부", "멍울", "림프절이 촉지", "림프절 촉지"],
    # endocrine / misc
    "exophthalmos": ["안구 돌출", "안구돌출", "눈이 튀어나"],
    "obesity": ["비만"],
    "tobacco smoking history": ["흡연", "담배를 피", "갑년"],
    "alcohol abuse or addiction": ["음주", "술을 매일", "소주", "과음", "알코올"],
    "malar rash": ["나비 모양 홍반", "나비모양 홍반", "나비 모양 발진", "협부 발진", "뺨의 홍반", "양 볼의 홍반"],
    "sensitivity to light": ["광과민", "광민감", "빛에 민감"],
    "sensitivity to the sun": ["햇빛 과민", "일광 과민", "광과민성 발진"],
    "flank pain": ["옆구리 통증", "옆구리가 아프", "측복부 통증", "늑골척추각 압통", "cva 압통", "갈비척추각 압통"],
    "cloudy urine": ["혼탁뇨", "소변이 탁", "소변이 뿌옇"],
    "foul-smelling urine": ["소변 냄새", "소변에서 냄새"],
    "painful swallowing": ["연하통", "삼킬 때 아프", "삼킬 때 통증"],
    "thirst": ["목마름", "목이 마르", "갈증"],
    "bruise": ["멍", "반상출혈", "피하 출혈"],
    "intestinal bleeding": ["장 출혈", "하부 위장관 출혈"],
    "gastrointestinal bleeding": ["위장관 출혈", "위장 출혈"],
    "nosebleed": ["코피", "비출혈"],
    "bleeding gums": ["잇몸 출혈", "잇몸에서 피"],
    "eye pain": ["안구 통증", "눈이 아프", "눈 통증", "안통"],
    "orthopnea": ["누우면 숨이", "누우면 숨차", "앉아서 자", "베개를 여러 개", "기좌호흡", "좌위호흡"],
    "paroxysmal nocturnal dyspnea": ["자다가 숨이 차", "자다가 숨이 막", "발작성 야간 호흡곤란"],
    "intermittent claudication": ["파행", "걸으면 다리가 아프", "걸을 때 종아리"],
    "jaw pain": ["턱 통증", "턱이 아프"],
    "suicidal ideation": ["자살 사고", "죽고 싶", "자살 생각"],
    "urinary incontinence": ["요실금", "소변이 새", "소변을 지리"],
    "fecal incontinence": ["변실금"],
    "urinary retention": ["요폐", "소변이 안 나", "소변을 못 보"],
    "testicular pain": ["고환 통증", "고환이 아프", "음낭 통증"],
    "breast lump": ["유방 멍울", "유방 종괴", "유방에 혹"],
    "nipple discharge": ["유두 분비물"],
    "muscle rigidity": ["근육 강직", "근강직", "경축", "톱니바퀴 강직"],
    "strawberry tongue": ["딸기혀", "딸기 혀"],
    "swollen joints": ["관절 부종", "관절이 붓", "관절 종창"],
    "joint effusion": ["관절 삼출", "관절액"],
    "silvery scales": ["은백색 인설", "은백색 비늘", "인설"],
    "red eye": ["결막 충혈", "눈이 충혈", "적안"],
    "watery eyes": ["눈물이 나", "눈물 흘림", "유루"],
    "change in bowel habits": ["배변 습관 변화", "배변 습관의 변화", "변이 가늘"],
    "clay-colored stools": ["회백색 변", "회색 변", "흰색 변", "무색 변"],
}

# flexible Korean phrasings (regex over the lower-cased finding) → term label; the matched span is blanked before
# label matching so its words do not also match unrelated labels ("소변 볼 때 찌릿" is dysuria, not paresthesia)
REGEX: list[tuple[str, str]] = [
    (r"(소변|오줌|배뇨).{0,8}(아프|아파|아팠|찌릿|따끔|화끈|쓰라|통증)", "dysuria"),
    (r"(소변|오줌|화장실).{0,6}자주|자주.{0,6}(소변|오줌|화장실|마려)", "frequent urination"),
    (r"열이.{0,6}(나|났|있|오르|올라)|열(나|났)", "fever"),
    (r"숨이.{0,6}(차|찬|가쁘|가빠|막히)", "dyspnea"),
    (r"(배|복부|아랫배|윗배|명치).{0,6}(아프|아파|아팠|통증|쥐어짜|쓰리)", "abdominal pain"),
    (r"(가슴|흉부).{0,6}(아프|아파|아팠|통증|쥐어짜|조이|짓누르|뻐근|답답)", "chest pain"),
    (r"(머리|두통).{0,6}(아프|아파|아팠|깨질|지끈)", "headache"),
    (r"(관절|무릎|손가락|손목|발목|어깨).{0,8}(아프|아파|아팠|통증|쑤시)", "arthralgia"),
    (r"(햇빛|햇볕|자외선|일광).{0,20}(발진|홍반|붉|빨갛|두드러기|가렵)", "sensitivity to the sun"),
    (r"(구강|입안|입 안|혀|잇몸).{0,10}(궤양|헐|염증)", "mouth ulcer"),
    (r"(체중|몸무게).{0,8}(줄|빠|감소)|(살이|살).{0,3}빠", "weight loss"),
    (r"(입맛|식욕).{0,6}(없|떨어|줄|저하)", "decreased appetite"),
    (r"(변|대변).{0,6}(검|까맣|까만|검은|짜장)", "melena"),
    (r"(변|대변).{0,8}(피|혈|빨갛)", "hematochezia"),
    (r"(소변|오줌).{0,8}(피|빨갛|붉|콜라)", "hematuria"),
    (r"(피|혈액).{0,6}토|토.{0,4}피", "hematemesis"),
    (r"(눈|피부|얼굴).{0,6}노랗", "jaundice"),
    (r"(다리|발목|종아리|하지|얼굴|눈두덩|눈꺼풀).{0,6}(붓|부었|부어|부종)", "edema"),
    (r"(힘이|기운이).{0,4}(빠|없)", "weakness"),
    (r"(말이|발음이).{0,6}(어눌|꼬|새)", "dysarthria"),
    (r"(한쪽|오른쪽|왼쪽|우측|좌측).{0,8}(팔|다리|팔다리).{0,8}(힘|마비|안 움직)", "hemiparesis"),
    (r"(목이|뒷목).{0,4}(뻣뻣|굳)", "neck stiffness"),
    (r"(물체|사물|글씨).{0,6}(둘|두 개|겹쳐)", "diplopia"),
    (r"(잠을|잠이).{0,6}(못|안 와|안 오)", "insomnia"),
    (r"(기억|깜빡).{0,8}(못|안 나|떨어|저하|잊)", "memory loss"),
    (r"(귀|청력).{0,8}(안 들|먹먹|떨어|저하|감소)", "deafness"),
    (r"(생리|월경).{0,8}(늦|없|안 해|안 하|끊)", "amenorrhea"),
    (r"(소변량|소변 양|소변이).{0,8}(줄|감소|적어)", "oliguria"),
    (r"(소변량|소변 양).{0,12}(늘|증가|많아)|하루.{0,6}소변.{0,10}리터", "polyuria"),
    (r"(밤에|야간에?|자다가).{0,8}(소변|화장실)", "nocturia"),
    (r"(소변|오줌).{0,8}거품", "proteinuria"),
    (r"(체중|몸무게).{0,10}(늘|증가|쪘)|살이.{0,3}(쪘|찌)", "weight gain"),
    (r"(가슴|명치|속).{0,8}(쓰려|쓰리|쓰림|타는 듯|화끈)", "heartburn"),
    (r"(잠들기|잠 들기).{0,4}(힘들|어렵)|자꾸 깨|자주 깨|새벽에.{0,6}깨", "insomnia"),
    (r"(기억력|기억).{0,10}(감소|저하|떨어|나빠)", "memory loss"),
    (r"(심부건반사|건반사|반사).{0,6}(저하|감소|소실)", "hyporeflexia"),
    (r"(심부건반사|건반사|반사).{0,6}(항진|증가)", "hyperreflexia"),
    (r"피부.{0,6}(건조|거칠)", "xeroderma"),
    (r"(자극|통증).{0,6}(에 대한 )?반응.{0,6}(저하|감소|없)|의식.{0,6}(저하|혼미|떨어)", "altered level of consciousness"),
]

# over-generic or mistranslated terms that must never be matched from free text
STOP_TERMS = {
    "observation", "increase", "reduction", "change", "sensation", "multiplicity", "lesion", "enlarged",
    "blood glucose", "pressure", "dryness", "maceration", "stethoscope auscultation", "irritation", "stress",
    "metastasis", "blotting", "nevus", "phlegm", "hunger", "hypervolemia", "vesicle", "acanthosis", "bruit",
    "regurgitation", "distension", "colic", "heat", "severe pain", "infection", "inflammation", "pain",
    "pain related to chief complaint", "sores", "narcolepsy", "slowed breathing", "cold", "macula",
    "anorexia", "hypochromic anemia", "psychosis", "anxiety disorder", "major depressive disorder", "mastodynia",
    "angina pectoris", "stiffness", "affected breathing", "asymptomatic", "primary affect", "scar",
}
# labels to drop from one term (term label → Korean labels)
BAD_LABELS = {"fever": ["열감", "열감 있음"], "tenderness": ["쓰라림", "아픔"], "spasm": ["통증", "발작"], "anemia": ["어지럼증"],
              "hypotension": ["어지러움"], "balance disorder": ["어지러움"], "heart failure": ["숨이 참"],
              "exanthem": ["두드러기"], "hyperaemia": ["코막힘", "가래 차오름"], "psychosis": ["망각"],
              "swelling": ["부종"], "muscle atrophy": ["살이 빠짐"], "tachycardia": ["가슴이 두근거림"],
              "rapid heartbeat": ["가슴이 두근거림", "가슴이 쿵쾅거림"], "pelvic pain": ["아랫배가 아픔"]}
# words that contain a shorter KB label but mean something else; blanked out before label matching
BLOCK_WORDS = ["수포음", "호흡음", "종격동", "반발통", "심잡음", "잡음", "관찰", "소견", "청진", "촉진", "시행",
               "검사", "다음 날", "다음날", "다음에", "다음 주", "다음번"]

REGEX_C = [(re.compile(p), en) for p, en in REGEX]

# ------------------------------------------------------------------ vitals / labs
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


# (analyte regex, low threshold, low term, high threshold, high term); value = first number after the analyte
_LABS = [
    (r"(?:wbc|백혈구)(?!\s*(?:\d+\s*-|0-))", 4000, "leukopenia", 11000, "leukocytosis"),
    (r"(?:hb|hgb|헤모글로빈|혈색소)", 11.0, "anemia", None, None),
    (r"(?:platelet|plt|혈소판)", 130000, "thrombocytopenia", None, None),
    (r"(?:na|나트륨|sodium)", 133, "hyponatremia", None, None),
    (r"(?:k|칼륨|potassium)", 3.3, "hypokalemia", 5.5, "hyperkalemia"),
    (r"(?:혈당|glucose|fbs|공복혈당)", 60, "hypoglycemia", 200, "hyperglycemia"),
    (r"(?:호산구|eosinophil|eos)", None, None, 7.0, "eosinophilia"),
    (r"(?:알부민|albumin|alb)", 3.0, "hypoalbuminemia", None, None),
]
_VITALS = re.compile(r"(?:체온|bt|temp)\s*:?\s*(\d{2}(?:\.\d)?)", re.I)
_HR = re.compile(r"(?:맥박|심박수|hr|pr|pulse)\s*:?\s*(\d{2,3})", re.I)
_RR = re.compile(r"(?:호흡수|호흡|rr)\s*:?\s*(\d{1,2})\s*(?:회|/)", re.I)
_BP = re.compile(r"(?:혈압|bp)[^\d]{0,12}(\d{2,3})\s*/\s*(\d{2,3})", re.I)
_SPO2 = re.compile(r"(?:산소포화도|spo2|sao2|o2 sat)[^\d]{0,12}(\d{2,3})\s*%", re.I)
_AST_ALT = re.compile(r"(?:ast|alt|got|gpt)\s*:?\s*(\d{2,5})", re.I)
_TBIL = re.compile(r"(?:총\s*빌리루빈|t\.?\s*bil(?:irubin)?|total bilirubin|빌리루빈)\s*:?\s*(\d+(?:\.\d+)?)", re.I)
_CR = re.compile(r"(?:\bcr\b|크레아티닌|creatinine)\s*:?\s*(\d+(?:\.\d+)?)", re.I)
_URINE_BLOOD = re.compile(r"(?:잠혈|occult blood|rbc)\s*:?\s*(양성|\+|\d{2,}|[1-4]\+)", re.I)
_URINE_PROT = re.compile(r"(?:단백|protein)\s*:?\s*(양성|[1-4]\+|\+{1,4})", re.I)
_URINE_KET = re.compile(r"(?:케톤|ketone)\s*:?\s*(양성|[1-4]\+|\+{1,4})", re.I)


def _lab_value(text: str, analyte: str) -> float | None:
    m = re.search(r"(?<![a-z가-힣])" + analyte + r"(?![a-z])\s*[:=]?\s*(?:수치|수)?\s*:?\s*" + _NUM, text, re.I)
    return _f(m.group(1)) if m else None


def lab_terms(text: str) -> list[str]:
    """Abnormality term labels (English) implied by numeric values in a finding ("체온 38.6℃" → fever)."""
    t = (text or "").lower()
    out: list[str] = []
    if not re.search(r"\d", t):
        return out
    m = _VITALS.search(t)
    if m and 30 <= _f(m.group(1)) <= 43:
        v = _f(m.group(1))
        if v >= 37.8:
            out.append("fever")
        if v >= 39.0:
            out.append("high fever")
    m = _HR.search(t)
    if m:
        v = _f(m.group(1))
        out += ["tachycardia"] if v > 100 else ["bradycardia"] if 20 <= v < 55 else []
    m = _RR.search(t)
    if m and _f(m.group(1)) >= 22:
        out.append("tachypnea")
    m = _BP.search(t)
    if m:
        sbp, dbp = _f(m.group(1)), _f(m.group(2))
        if sbp < 90:
            out.append("hypotension")
        elif sbp >= 160 or dbp >= 100:
            out.append("arterial hypertension")
    m = _SPO2.search(t)
    if m and _f(m.group(1)) < 92:
        out.append("hypoxia")
    for analyte, lo, lo_t, hi, hi_t in _LABS:
        v = _lab_value(t, analyte)
        if v is None:
            continue
        if analyte.startswith("(?:wbc") and v < 200:  # ×10³/μL notation
            v *= 1000
        if analyte.startswith("(?:platelet") and v < 2000:
            v *= 1000
        if lo is not None and v < lo:
            out.append(lo_t)
        if hi is not None and v > hi:
            out.append(hi_t)
    m = _AST_ALT.search(t)
    if m and _f(m.group(1)) >= 120:
        out.append("elevated transaminases")
    m = _TBIL.search(t)
    if m and _f(m.group(1)) >= 3.0:
        out.append("jaundice")
    m = _CR.search(t)
    if m and _f(m.group(1)) >= 2.0:
        out.append("kidney failure")
    if re.search(r"소변|요검사|urinalysis|u/a|요 ", t):
        if _URINE_BLOOD.search(t):
            out.append("hematuria")
        if _URINE_PROT.search(t):
            out.append("proteinuria")
        if _URINE_KET.search(t):
            out.append("ketonuria")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ diagnosis-name normalisation
# Korean spelling pairs used in diagnosis names (older ↔ KCD-8 terms); tried in both directions
NAME_SUBS = [("지주막", "거미막"), ("갑상선", "갑상샘"), ("담관", "쓸개관"), ("담낭", "쓸개"), ("췌장", "이자"),
             ("구균", "알균"), ("간균", "막대균"), ("임파", "림프"), ("늑막", "흉막"), ("심낭", "심장막"),
             ("뇌막", "수막"), ("자발성", "자연"), ("질환", "병"), ("신부전", "콩팥병"),
             ("신질환", "콩팥병"), ("신장", "콩팥"), ("요로 감염", "요로감염"), ("위장관염", "위장염"),
             ("대퇴골", "넙다리뼈"), ("경색", "경색증"), ("결핍", "결핍증"), ("협착증", "협착"),
             ("stemi", "st분절상승 심근경색"), ("nstemi", "st분절비상승 심근경색")]
# leading/trailing qualifiers that can be dropped to back off to the general concept
NAME_MODIFIERS = ["급성", "만성", "아급성", "원발성", "일차성", "이차성", "속발성", "특발성", "재발성", "우측", "좌측",
                  "양측", "파열된", "파열", "의증", "추정", "진행성", "acute", "chronic", "subacute", "primary",
                  "secondary", "idiopathic", "recurrent", "left", "right", "bilateral", "ruptured", "suspected"]
