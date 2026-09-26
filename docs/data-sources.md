# 외부 데이터 소스 조사 (지식 근거 + 증례)

작성: knowledge-rag 에이전트 · 조사일 2026-09-25
목적: Doctor Agent의 판단 근거를 LLM이 만든 내용이 아니라 **출처를 밝힐 수 있는 검증된 의료 데이터**에 두고, Claude가 작성한 합성 연습 증례보다 나은 증례를 확보하기 위함.

판정 기준 (대회 규칙 기반)
- **연구 발표 사용 허용** 라이선스여야 하며 출처를 밝힐 수 있어야 함. 라이선스가 불명확하면 제외.
- 제출 ZIP에 넣는 것은 **합계 50MB 이하, CPU/RAM만, 추론 시 네트워크 호출 없음**.
- 외부 LLM API(Gemini 등)로 오프라인 가공은 허용. 단 데이터 약관이 제3자 LLM API 전송을 금지하면 불가.
- 라이선스는 가능한 한 **1차 출처**(저장소 LICENSE, 공식 페이지, HF 데이터셋 메타데이터)에서 확인했고, 확인한 URL을 표에 적었다. 2차 출처에만 근거한 항목은 "미확인"으로 표시했다.
- 이 문서는 조사 결과이며, 실제 사용 시에는 `docs/licenses.md`에 항목을 추가해야 한다 (이 작업에서는 수정하지 않음).

용어: NC = 비상업, ND = 변경금지, SA = 동일조건변경허락, KOGL = 공공누리.
NC 라이선스는 연구 발표 용도로는 일반적으로 허용되지만, 상금이 있는 대회 참가가 "상업적 이용"인지는 판단이 필요하다 (아래 5장).

---

## 1. 추천 요약

### 지식 자료 Top 3

| 순위 | 자료 | 라이선스 | 이유 |
|---|---|---|---|
| 1 | **공개된 임상 결정 규칙·점수 직접 구현** (Wells, PERC, HEART, qSOFA, CURB-65, Centor/McIsaac, Ottawa ankle/knee, Canadian CT Head, NEXUS, Alvarado, ABCD2, CHA2DS2-VASc, GCS, NIHSS 등) | 공식 자체는 공개 논문의 방법(사실)이고, 원 논문을 인용해 **우리 코드·우리 문장으로 구현** | Safety 점수와 직접 연결(놓치면 안 되는 질환 배제/확진에 필요한 질문·검사 결정). 용량이 거의 0이고 CPU만 쓰며 결정적임. 항목마다 원 논문을 인용할 수 있음 |
| 2 | **DDXPlus 지식 파일** (`release_conditions.json`, `release_evidences.json`) | CC-BY (저장소 README 기준, 버전 표기 없음) | 질환 49개 × 증상 110개 + 병력 113개의 구조화된 관계, 질환별 ICD-10 코드, 의사가 검토한 증상별 **질문 문구(영/불)**. ASK 전략(다음에 무엇을 물을지)에 바로 쓸 수 있고 용량이 작음. 단, 호흡기·흉통·이비인후 중심이라 범위가 좁음 |
| 3 | **HIRA 상병마스터** (KCD 코드 + 한글명 + 영문명) | 공공누리 제1유형 (출처표시) | 한국어 증례일 경우 최종 진단명을 **표준 한글 진단명과 KCD 코드**로 정규화하는 데 가장 안전하고 상업 이용 제한도 없음. 47,798행 CSV, 소용량 |

보조: 영문 질환 설명이 필요하면 **MedlinePlus Health Topics 요약**(미국 정부 저작물, 공공 도메인, A.D.A.M. 백과사전과 약물 정보는 제외)과 **Disease Ontology**(CC0)를 쓴다. 한국어 설명은 질병관리청 국가건강정보포털(KOGL 4유형, 비상업·변경금지)을 원문 그대로 인용하는 방식으로만 조건부 사용을 권한다.

### 증례 데이터 Top 3 (연습/평가용, 제출물에는 넣지 않음)

| 순위 | 자료 | 라이선스 | 이유 |
|---|---|---|---|
| 1 | **AgentClinic-MedQA / MedQA-Extended** (107 → 215 증례) | 저장소 MIT (AgentClinic), 원천 MedQA 저장소도 MIT | 구조가 **병력(환자 역할)·신체검사 소견·검사 결과·정답 진단으로 이미 분리된 OSCE 형식**이라 우리 ASK/EXAM/TEST/DIAGNOSE 스키마로 거의 그대로 변환 가능. 대화형 진단 에이전트 벤치마크 그 자체임 |
| 2 | **MedR-Bench** (진단 957건, 그중 희귀질환 491건) 및 **MedCaseReasoning** (14,489건) | MedR-Bench: CC BY-SA (LICENSE 파일, 버전 미표기) / MedCaseReasoning: GitHub README는 CC-BY 4.0, HF 메타데이터는 MIT | 실제 PMC Open Access 증례 보고 기반이라 Claude 합성 증례보다 현실적. MedR-Bench는 "기본 정보 → 검사 추천 → 진단"의 다회 턴 평가를 이미 설계해 둠. MedCaseReasoning은 의사가 쓴 감별진단 추론 문장이 있어 Safety/감별 평가 기준 만들기에 좋음 |
| 3 | **snuh/ClinicalQA** (1,045문항, 한국어) | Apache-2.0 (HF 메타데이터) | **한국어**, 주호소(chief complaint) 기반, 의사 3명 검토. 대회 주최(분당서울대병원)와 같은 서울대병원 계열 자료라 증례 문체가 비슷할 가능성. 단, 초안을 LLM이 생성한 객관식 형식이라 증례로 쓰려면 변환 필요 |

대안: **DiagnosisArena**(915건, MIT, 필드가 증례 정보/신체검사/검사/최종진단으로 분리되어 스키마 적합도는 가장 높음)는 원천이 저널 증례 보고(JAMA, Cell 등)라 MIT 표기가 원문 저작권까지 해결하는지 불명확 → 내부 평가용으로만 조건부 사용.

### 라이선스 주의 (요약)
- **MIMIC 계열(AgentClinic-MIMIC-IV, MIMIC-IV-Ext 등)**: PhysioNet 자격 필요 + **일반 Gemini API/ChatGPT 전송 금지**(Vertex AI 등 승인 서비스만 허용). 현재 개발 구성(ai.google.dev Gemini API)으로는 사용 불가.
- **NEJM 기반 증례(AgentClinic-NEJM, NEJM Case Records)**: 원문 저작권이 NEJM에 있고 별도 라이선스 표기가 없음 → 제외.
- **MedQA 데이터 묶음 안의 영어 교과서 18권**: 상용 교과서 텍스트로 보이며 저작권 해결 근거가 없음 → 문항만 쓰고 교과서는 제외.
- **대한의사협회 의학용어(term.kma.org)**: 무단 전재·재배포 금지 명시 → 제외.
- **UMLS / SNOMED CT**: 라이선스 계약 필요, 재배포 제한 → 제출물 포함 불가.

---

## 2. 지식 자료 비교표

| 자료 | 내용 | 규모 | 언어 | 라이선스 (확인 URL) | 연구 발표 | 제출물(≤50MB, CPU) | 외부 LLM API 전송 | 적합도 |
|---|---|---|---|---|---|---|---|---|
| **임상 결정 규칙·점수** (원 논문 기반 자체 구현) | 점수 공식, 임계값, 적용 대상 | 규칙 수십 개, 수 KB | 원문 영어 → 우리가 한국어로 작성 | 공식·방법은 저작권 대상이 아님(사실). 원 논문 인용 필수. **MDCalc 등의 설명문은 복사하지 말 것**. MMSE(PAR 저작권), MoCA(사용 허가 필요) 같은 저작권 설문 도구는 제외 | 가능 | 가능 (코드) | 제한 없음 | **High**: Safety와 검사 선택 근거를 결정적으로 제공 |
| **DDXPlus 지식 파일** | 질환 49개(ICD-10 코드, 불/영 이름), 근거(증상 110 + 병력 113, 이진/범주/다중선택), 근거별 질문 문구(영/불), 질환별 증상 관계 | JSON 2개, 소용량 (정확한 크기 미확인, 수백 KB~수 MB로 추정) | 영어, 프랑스어 | CC-BY, 버전 미표기 ("We are releasing under the CC-BY licence", https://github.com/mila-iqia/ddxplus README). GitHub에는 LICENSE 파일 없음. figshare 페이지는 접근 차단(403)으로 미확인 | 가능 (출처 표기) | 가능 | 제한 없음 | **High**: 증상-질환 관계와 질문 문구가 구조화됨. 단 호흡기/흉통/이비인후 중심. 원 지식베이스(Dialogue Health Technologies)는 비공개이고 공개된 파일만 CC-BY |
| **HIRA 상병마스터** (KCD) | 상병코드, 한글명, 영문명, 주상병 사용 여부, 법정감염병, 성별·연령 제한 | 47,798행 CSV (API도 제공) | 한국어/영어 | 공공누리 제1유형 "공공저작물: 출처표시" (https://www.data.go.kr/data/15067467/fileData.do) | 가능 | 가능 (필요 열만 추려도 수 MB 이하 추정) | 제한 없음 | **High** (한국어 증례일 때): 진단명 정규화, 성별·연령 부적합 진단 거르기 |
| **MedlinePlus Health Topics 요약** | 질환·검사 소비자용 요약 | 약 1,000여 토픽 (미확인), XML 제공 | 영어 (일부 스페인어) | 건강 주제 요약·검사 정보·유전학 요약은 미국 정부 저작물 = 공공 도메인. **A.D.A.M. 의학 백과사전, ASHP 약물 정보는 저작권 있음 → 제외** (https://medlineplus.gov/about/using/usingcontent/) | 가능 ("Courtesy of MedlinePlus from the National Library of Medicine" 표기 권장) | 가능 (요약만 추리면 소용량) | 제한 없음 | Medium: 신뢰도 높지만 소비자용이라 진단 기준 수준의 깊이가 부족 |
| **Human Disease Ontology (DO)** | 질환 용어, 정의, 계층, ICD/UMLS/MeSH 교차참조 | 약 1만 개 이상 질환 (정확한 수 미확인) | 영어 | CC0-1.0 (https://github.com/DiseaseOntology/HumanDiseaseOntology/blob/main/LICENSE) | 가능 | 가능 (부분 추출) | 제한 없음 | Medium: 진단명 동의어·계층(정답 판정 시 상·하위 질환 매칭)에 유용. 증상 관계는 빈약 |
| **Human Phenotype Ontology + 주석(HPOA)** | 표현형 용어, 질환-표현형 주석 (OMIM/Orphanet 중심, 약 8천 개 희귀질환) | hp.json·phenotype.hpoa 수십 MB 수준 (미확인) | 영어 (일부 번역 존재) | HPO 자체 라이선스: 무료 사용, **Consortium 인용, 공개 시 버전/날짜 표기, 파일 내용과 관계를 변경하지 말 것**, 서비스에 HPO 사용 고지 (http://human-phenotype-ontology.github.io/license.html; GitHub LICENSE.md는 이 페이지로 연결) | 가능 (인용) | 부분 추출 시 가능. "변경 금지" 조항 때문에 추출본은 원 ID·버전을 유지해야 함 | 제한 명시 없음 | Low~Medium: 희귀질환 중심이라 일반 진료 증례에는 적합도 낮음 |
| **Orphadata Science** | 희귀질환 임상 징후·빈도, 역학 | 미확인 | 영어 외 다국어 | CC BY 4.0 (Orphadata Science 한정. Orphadata Products는 별도 계약) (https://www.orphadata.com/legal-notice/) | 가능 | 가능 (부분) | 제한 없음 | Low: 희귀질환 증례가 나오면 보조 |
| **MeSH** | 의학 주제어, 질환 트리(C), 동의어 | 약 3만 개 descriptor (미확인) | 영어 | NLM 약관: 무료, 사용료 없음, **NLM 출처를 명확히 표기**, NLM 보증을 암시하지 말 것, 최신판 아님을 고지 (https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html) | 가능 | 가능 (부분) | 제한 없음 | Low~Medium: 동의어 사전 용도. 공식 한국어 MeSH 라이선스는 미확인 |
| **질병관리청 국가건강정보포털** | 질환별 한국어 건강정보(학회 감수) | 미확인 | 한국어 | 공공누리 제4유형 (출처표시 + 상업적 이용금지 + 변경금지) (https://health.kdca.go.kr/healthinfo/biz/health/portalUseGuidance/hlthinsReqst/hlthinsReqstMth.do?index=2). Open API는 신청·승인 필요 | 가능 (비상업) | 원문 그대로 인용하는 경우에만. **LLM 요약·재작성본 배포는 변경금지 위반 소지** | 약관상 명시 금지 없음. 다만 가공물 배포는 불가 | Medium: 한국어 근거로 좋지만 ND 제약 |
| **StatPearls** (NCBI Bookshelf) | 질환별 임상 리뷰 (감별, 검사, 진단 기준) | 수천 개 문서 | 영어 | CC BY-NC-ND 4.0 (NCBI Bookshelf 개별 문서 표기, 검색 결과로 확인. Bookshelf 페이지는 reCAPTCHA로 직접 열람 실패 → **1차 확인 부분 미완**) | 가능 (비상업) | 원문 청크 그대로만. 요약본 배포는 ND 위반 소지 | 약관상 명시 금지 없음 | Medium: 내용은 가장 임상적이나 NC+ND, 대량 다운로드 경로 미확인 |
| **Wikipedia** | 질환 문서 (한국어·영어) | 대용량 | 다국어 (한국어 포함) | CC BY-SA 4.0 (+GFDL) (https://en.wikipedia.org/wiki/Wikipedia:Copyrights) | 가능 | 부분 추출 가능. 배포 인덱스는 CC BY-SA로 공개해야 하고 저작자 표기 필요 | 제한 없음 | Low~Medium: 품질 편차, 검증된 근거로 보기 어려움 |
| **WikiDoc** | 의사 작성 위키 | 대용량 | 영어 | "Creative Commons Attribution/Share-Alike" (버전 미표기) (https://www.wikidoc.org/index.php/Main_Page 하단) | 가능 | 가능 (SA 조건) | 제한 없음 | Low: 버전 불명, 품질 편차 |
| **openFDA** | 약물 라벨, 부작용 보고, 리콜 | 대용량 | 영어 | 공공 도메인, CC0 1.0 (GMDN 내용 제외) (https://open.fda.gov/license/) | 가능 | 가능 | 제한 없음 | Low: 진단 과제와 관련이 적음 |
| **WHO ICD-11** | 분류 코드·명칭 | 대용량 | 다국어 (공식 한국어판 여부 미확인) | CC BY-ND 3.0 IGO. 소프트웨어 포함은 허용, 내용 변경·다른 분류 개발 금지, 매핑·번역은 별도 계약 (https://icd.who.int/en/docs/icd11-license.pdf) | 가능 | 코드·명칭 그대로 사용 시 가능 | 제한 없음 | Low: 한국은 KCD(ICD-10 기반)를 쓰므로 HIRA 상병마스터가 우선 |
| **WHO ICD-10** | 분류 | – | – | WHO 저작권 정책 페이지는 ICD-11/ICF/ICHI만 다루고 **ICD-10 라이선스는 확인 못 함** (https://www.who.int/about/policies/publishing/copyright) | 미확인 | – | – | 제외 (대신 HIRA 상병마스터 사용) |

---

## 3. 증례 데이터 비교표

모두 **연습·평가 전용**이며 제출 ZIP에는 넣지 않는다 (대회 규칙상 제출물에 증례를 넣을 이유가 없고, 교차 증례 정보 사용 금지와도 충돌할 수 있음).

| 자료 | 내용 | 규모 | 언어 | 라이선스 (확인 URL) | 연구 발표 | 외부 LLM API 전송 | 스키마 적합도 | 적합도 |
|---|---|---|---|---|---|---|---|---|
| **AgentClinic-MedQA / -Extended** | MedQA 문항을 OSCE 형식(환자 정보·병력, 신체검사 소견, 검사 결과, 정답 진단)으로 변환한 대화형 진단 증례 | 107 → 215건 (extended) | 영어 | 저장소 MIT (https://github.com/SamuelSchmidgall/AgentClinic/blob/main/LICENSE.txt). 원천 MedQA 저장소도 MIT (https://github.com/jind11/MedQA/blob/master/LICENSE) | 가능 | 제한 없음 | 매우 높음 (ASK/EXAM/TEST 분리 완료) | **High** |
| **AgentClinic-NEJM / -Extended** | NEJM 증례(이미지 문제) | 15 → 120건 | 영어 | 저장소는 MIT이나 **원문은 NEJM 저작권**, 데이터에 대한 별도 허락 표기 없음 (README 확인) | 불명확 | – | 높음 | **제외** |
| **AgentClinic-MIMIC-IV** | MIMIC-IV 실제 환자 기반 | 미확인 | 영어 | PhysioNet 자격·DUA 필요 (AgentClinic README) | 가능 (DUA 준수 시) | **일반 API 금지**. PhysioNet 정책: OpenAI API·ChatGPT 등 제3자 전송 금지, Azure OpenAI·Bedrock·Vertex AI Gemini·Anthropic Claude 등 조건부 허용 (https://physionet.org/news/post/gpt-responsible-use) | 높음 | Low (절차 부담, 현재 Gemini API 구성과 충돌) |
| **MedQA (USMLE)** | USMLE 형식 객관식 임상 문항 | 영어 12,723문항 (+중국어) | 영어, 중국어 | GitHub 저장소 MIT ("Code and data for MedQA"). 단 HF 미러별 표기가 제각각(bigbio: unknown, GBaker: CC-BY-4.0). **묶음에 포함된 교과서 18권은 저작권 해결 근거 없음 → 제외** | 가능 (MIT 기준) | 제한 없음 | 중간 (객관식 → 증례 변환 필요, 정보가 한 문단에 섞여 있음) | Medium |
| **MedR-Bench** | PMC-OA 증례 보고 기반, 진단 957건(희귀 491) + 치료 496건. 원샷·1턴·자유 다회 턴 검사 추천 평가 코드 포함 | JSON 약 15MB + 8MB | 영어 | LICENSE 파일 "CC BY-SA" (버전 미표기) (https://github.com/MAGIC-AI4Med/MedRBench/blob/main/LICENSE) | 가능 (SA) | 제한 없음 | 높음 (검사 추천 → 진단 흐름이 우리 과제와 유사) | **High** |
| **MedCaseReasoning** | PMC-OA 증례 보고, 증례 제시문 + 의사 작성 감별진단 추론 + 최종 진단 | 14,489건 (train 13,092 / test 897) | 영어 | GitHub README: 데이터 CC-BY 4.0 ("derived from the PMC Open Access Subset"), 코드 MIT (https://github.com/kevinwu23/Stanford-MedCaseReasoning). **HF 메타데이터는 MIT** (https://huggingface.co/datasets/zou-lab/MedCaseReasoning) → 표기 불일치 | 가능 | 제한 없음 | 중간 (증례 제시문을 병력/검사/검사결과로 나눠야 함) | **High**: 실제 증례 + 감별 근거 |
| **DiagnosisArena** | 저널 증례 보고(10개 주요 저널)를 구조화: Case Information, Physical Examination, Diagnostic Tests, Final Diagnosis, 4지선다 | 공개 915건 (전체 1,113건), 1.56MB | 영어 | MIT (https://github.com/SPIRAL-MED/DiagnosisArena/blob/main/LICENSE, HF 카드 동일). 카드에 "Cell, JAMA 등 공개 문헌을 각색, 일부 저널은 저작권 때문에 제외"라고 명시 → **원문 저작권이 MIT로 해결되는지 불명확** | 조건부 | 제한 없음 | 매우 높음 | Medium (내부 평가용 조건부) |
| **snuh/ClinicalQA** | 주호소 기반 한국 의사국시 수준 문항 (문제, 소견, 보기, 정답, 해설, 출처) | 1,045문항, 약 1.7MB | 한국어 | Apache-2.0 (https://huggingface.co/datasets/snuh/ClinicalQA, HF 메타데이터) | 가능 | 제한 없음 | 중간 (객관식 → 증례 변환 필요) | **High** (한국어): GPT-4o 등으로 초안 생성 후 의사 3명 검토 |
| **KorMedMCQA** | 한국 보건의료인 국가시험 기출(의사 2,489, 간호 1,751, 약사 1,817, 치과 1,412), 2012~2024 | 7,469문항 | 한국어 | CC-BY-NC-2.0 (https://huggingface.co/datasets/sean0042/KorMedMCQA). **원 기출문항 저작권(한국보건의료인국가시험원) 허락 여부는 카드·논문 초록에 없음 → 미확인** | 가능 (비상업) | 제한 없음 | 낮음~중간 (의사 과목 일부만 증례형) | Medium: 한국어 진단 표현 연습 |
| **PMC-Patients (V2)** | PMC 증례 보고에서 추출한 환자 요약 | 약 16.7만 → V2 25만 건, 1.38GB | 영어 | CC BY-NC-SA 4.0 (https://github.com/zhao-zy15/PMC-Patients/blob/master/LICENSE, HF 동일) | 가능 (비상업) | 제한 없음 | 낮음 (요약문, 진단 라벨 별도 추출 필요) | Medium: 규모는 크나 가공 부담 |
| **RareArena** | PMC-Patients 기반 희귀질환 진단 과제 | 약 5만 건, 4천여 희귀질환 | 영어 | CC BY-NC-SA 4.0 (https://huggingface.co/datasets/THUMedInfo/RareArena) | 가능 (비상업) | 제한 없음 | 중간 | Low: 희귀질환 편중 |
| **DDXPlus 환자** | 합성 환자: 인구정보, 증상·병력, 감별진단, 정답 질환 | 약 130만 명 | 영어/프랑스어 | CC-BY (README) | 가능 | 제한 없음 | 낮음~중간 (신체검사·검사 결과 없음, ASK만 평가 가능) | Medium: 문진 효율성 평가용 |
| **MedMCQA** | 인도 AIIMS/NEET PG 객관식 | 약 19.4만 문항 | 영어 | Apache-2.0 (HF, https://huggingface.co/datasets/openlifescienceai/medmcqa). GitHub 저장소는 MIT | 가능 | 제한 없음 | 낮음 (지식 문항 위주) | Low |
| **MediQ** | MedQA/Craft-MD 기반 대화형 문진 벤치마크 | 미확인 | 영어 | 저장소 CC-BY-4.0 (GitHub license API, https://github.com/stellalisy/mediQ/blob/main/LICENSE) | 가능 | 제한 없음 | 중간 (문진 중심) | Medium: 세부 내용 미확인 |
| **NEJM Case Records (MGH)** | 증례 기록 | – | 영어 | NEJM 저작권, 공개 라이선스 없음 | 불가 | – | – | **제외** |

---

## 4. 제외한 자료와 이유

| 자료 | 제외 이유 |
|---|---|
| NEJM Case Records, AgentClinic-NEJM(-Extended) | NEJM(Massachusetts Medical Society) 저작권. 재사용 라이선스 표기 없음 |
| AgentClinic-MIMIC-IV, MIMIC-IV-Ext 계열 전반 | PhysioNet 자격 증명 필요 + DUA가 OpenAI API·ChatGPT 등 제3자 전송 금지. 우리 개발 환경의 Gemini API(ai.google.dev)는 PhysioNet이 허용한 서비스(Vertex AI 경유 Gemini 등)가 아님. 제출물 포함도 불가 |
| MedQA에 포함된 영어 교과서 18권 | 상용 교과서 텍스트로 보이며 재배포 허락 근거가 없음 (문항 데이터와 분리해서 사용하지 말 것) |
| UMLS Metathesaurus | 라이선스 계약 필요. 하위 집합 포함 재배포 금지(애플리케이션 일부로만 예외), 소스별 추가 제한(Category 3: 내부 연구만, Category 4: 미국 내 한정) (https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html) |
| SNOMED CT | 한국은 SNOMED International 회원국이라 국내 사용은 무료이나 Affiliate License 등록 필요, 제출 ZIP 배포 조건 불명확 (https://www.snomed.org/get-snomed, https://www.snomed.org/members). 우리 과제에 비해 부담이 큼 |
| 대한의사협회 의학용어위원회 용어 (term.kma.org) | "모든 정보의 저작권은 대한의사협회 의학용어위원회에 있으며" 사전 승인 없는 전재·재배포 금지. 다운로드 제공 없음 |
| MedlinePlus A.D.A.M. 의학 백과사전, ASHP 약물 정보 | 저작권 보유(MedlinePlus 공식 안내). 공공 도메인인 건강 주제 요약만 사용 |
| WHO ICD-10 원본 | 라이선스를 1차 출처에서 확인하지 못함. HIRA 상병마스터(KOGL 1유형)로 대체 |
| MDCalc 등 계산기 사이트의 설명 문구 | 사이트 저작권. 공식은 원 논문에서 직접 가져오고 설명은 직접 작성 |
| MMSE, MoCA 등 저작권 있는 평가 도구 | 사용 허가·라이선스 필요 |
| snuh/specialist-level·essential-level_medical_knowledge_dataset_sft | CC-BY-ND-4.0 (HF 메타데이터). AI Hub 데이터에서 파생되고 Qwen3로 추론을 생성한 SFT용 데이터. ND라 가공본 배포 불가이고, 대회는 파인튜닝 금지라 활용처가 적음. 원천 AI Hub 약관(해외 반출·제3자 제공 제한 가능성)도 미확인 |
| AI Hub 의료 데이터 (전문 의학지식 등) | 이용 신청·약관 동의가 필요하고, 해외 서버(외부 LLM API) 전송 허용 여부를 확인하지 못함 → 확인 전까지 제외 |

---

## 5. 확인 못 한 점

1. **NC 라이선스와 대회 상금**: 상금이 있는 대회 참가가 CC BY-NC / KOGL 4유형의 "비상업"에 해당하는지 대회 측 해석이 없다. KorMedMCQA, PMC-Patients, StatPearls, 국가건강정보포털을 제출물에 넣기 전에 주최 측 FAQ나 참가자 가이드 확인이 필요하다. 연습·평가용(제출물 제외)으로만 쓰면 위험이 낮다.
2. **DDXPlus 라이선스 버전**: README에 "CC-BY"만 있고 버전이 없으며 GitHub에 LICENSE 파일이 없다. figshare 페이지(원 배포처)는 403으로 열람하지 못했다. 지식 파일의 정확한 크기도 확인하지 못했다.
3. **MedCaseReasoning 라이선스 불일치**: GitHub README는 CC-BY 4.0, HF 메타데이터는 MIT. 또 PMC OA Subset에는 CC BY-NC 논문도 섞여 있는데(라이선스가 논문마다 다르다고 PMC가 명시), 비상업 논문을 걸러냈는지 확인하지 못했다. MedR-Bench도 같은 문제가 있고 "CC BY-SA"의 버전이 없다.
4. **MedQA 원문 저작권**: 저장소는 MIT지만 문항은 웹에서 수집한 USMLE 대비 문제은행에서 왔다(논문 기준). 원 문제은행의 권리 관계는 확인하지 못했다. AgentClinic-MedQA도 이 문제를 그대로 물려받는다.
5. **DiagnosisArena 원문 저작권**: MIT 표기와 달리 원 저널 증례의 재배포 허락 여부가 불명확하다. 제작자 스스로 일부 저널을 저작권 때문에 뺐다고 적었다.
6. **KorMedMCQA 원 기출문항 권리**: 한국보건의료인국가시험원 허락 여부가 데이터 카드와 논문 초록에 없다. 논문 본문은 확인하지 못했다.
7. **StatPearls 라이선스**: NCBI Bookshelf 페이지가 reCAPTCHA로 막혀 1차 페이지를 직접 보지 못했고, 검색 결과로만 CC BY-NC-ND 4.0을 확인했다. 대량 다운로드 경로(Bookshelf OA 여부)도 미확인이다.
8. **임상 규칙 원 논문 서지**: 규칙 목록은 제안 단계이며, 구현할 때 각 규칙의 원 논문(예: Wells PE Thromb Haemost 2000, HEART Neth Heart J 2008, qSOFA JAMA 2016, Ottawa ankle JAMA 1993, Canadian CT Head Lancet 2001, CURB-65 Thorax 2003 등)을 원문에서 다시 확인해 `docs/licenses.md`에 인용과 함께 기록해야 한다. 이번 조사에서는 원문을 열람하지 않았다.
9. **규모 수치**: Disease Ontology 용어 수, MedlinePlus 토픽 수, MeSH descriptor 수, HPO 파일 크기, HIRA CSV 실제 용량은 확인하지 못했다(표에 "미확인"으로 표시).
10. **한국어 임상 진료지침**: 대한의학회 임상진료지침정보센터 등 학회 지침은 학회별 저작권이라 일괄 라이선스를 찾지 못했다. KTAS(한국형 응급환자 분류도구) 같은 한국 도구의 라이선스도 확인하지 못했다.
11. **대회 증례 언어·형식**: 참가자 가이드가 아직 공개되지 않아 한국어 자료의 우선순위는 가정에 기반한다.
