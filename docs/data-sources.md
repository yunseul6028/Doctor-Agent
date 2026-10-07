# 외부 데이터 소스 조사 (지식 근거 + 증례)

작성: knowledge-rag 에이전트 · 조사일 2026-09-25
목적: Doctor Agent의 판단 근거를 LLM이 만든 내용이 아니라 **출처를 밝힐 수 있는 검증된 의료 데이터**에 두고, Claude가 작성한 합성 연습 증례보다 나은 증례를 확보하기 위함.

판정 기준 (이 프로젝트의 설계 원칙)
- **연구 발표 사용 허용** 라이선스여야 하며 출처를 밝힐 수 있어야 함. 라이선스가 불명확하면 제외.
- 실행 코드가 읽는 데이터는 **작게(현재 수 MB), CPU/RAM만, 추론 시 의사 모델 엔드포인트 말고는 네트워크 호출 없음**.
- 외부 LLM API(Gemini 등)로 오프라인 가공은 허용. 단 데이터 약관이 제3자 LLM API 전송을 금지하면 불가.
- (2026-10-07 추가) 의사 모델이 제미나이 Pro API로 바뀌었으므로, 평가 증례의 내용도 진료 중에 제3자 LLM API로 전송된다. 평가 증례로 쓰는 데이터 역시 같은 기준(제3자 LLM API 전송 허용)을 지켜야 한다.
- 라이선스는 가능한 한 **1차 출처**(저장소 LICENSE, 공식 페이지, HF 데이터셋 메타데이터)에서 확인했고, 확인한 URL을 표에 적었다. 2차 출처에만 근거한 항목은 "미확인"으로 표시했다.
- 이 문서는 조사 결과이며, 실제 사용 시에는 `docs/licenses.md`에 항목을 추가해야 한다 (이 작업에서는 수정하지 않음).

용어: NC = 비상업, ND = 변경금지, SA = 동일조건변경허락, KOGL = 공공누리.
NC 라이선스는 연구 발표 용도로는 일반적으로 허용되지만, 가공본을 공개 저장소에 올리는 것은 따로 판단해야 한다 (아래 5장, 8장).

---

## 1. 추천 요약

### 지식 자료 Top 3

| 순위 | 자료 | 라이선스 | 이유 |
|---|---|---|---|
| 1 | **공개된 임상 결정 규칙·점수 직접 구현** (Wells, PERC, HEART, qSOFA, CURB-65, Centor/McIsaac, Ottawa ankle/knee, Canadian CT Head, NEXUS, Alvarado, ABCD2, CHA2DS2-VASc, GCS, NIHSS 등) | 공식 자체는 공개 논문의 방법(사실)이고, 원 논문을 인용해 **우리 코드·우리 문장으로 구현** | Safety 점수와 직접 연결(놓치면 안 되는 질환 배제/확진에 필요한 질문·검사 결정). 용량이 거의 0이고 CPU만 쓰며 결정적임. 항목마다 원 논문을 인용할 수 있음 |
| 2 | **DDXPlus 지식 파일** (`release_conditions.json`, `release_evidences.json`) | CC-BY (저장소 README 기준, 버전 표기 없음) | 질환 49개 × 증상 110개 + 병력 113개의 구조화된 관계, 질환별 ICD-10 코드, 의사가 검토한 증상별 **질문 문구(영/불)**. ASK 전략(다음에 무엇을 물을지)에 바로 쓸 수 있고 용량이 작음. 단, 호흡기·흉통·이비인후 중심이라 범위가 좁음 |
| 3 | **HIRA 상병마스터** (KCD 코드 + 한글명 + 영문명) | 공공누리 제1유형 (출처표시) | 한국어 증례일 경우 최종 진단명을 **표준 한글 진단명과 KCD 코드**로 정규화하는 데 가장 안전하고 상업 이용 제한도 없음. 47,798행 CSV, 소용량 |

보조: 영문 질환 설명이 필요하면 **MedlinePlus Health Topics 요약**(미국 정부 저작물, 공공 도메인, A.D.A.M. 백과사전과 약물 정보는 제외)과 **Disease Ontology**(CC0)를 쓴다. 한국어 설명은 질병관리청 국가건강정보포털(KOGL 4유형, 비상업·변경금지)을 원문 그대로 인용하는 방식으로만 조건부 사용을 권한다.

### 증례 데이터 Top 3 (연습/평가용, 실행 코드는 읽지 않음)

| 순위 | 자료 | 라이선스 | 이유 |
|---|---|---|---|
| 1 | **AgentClinic-MedQA / MedQA-Extended** (107 → 215 증례) | 저장소 MIT (AgentClinic), 원천 MedQA 저장소도 MIT | 구조가 **병력(환자 역할)·신체검사 소견·검사 결과·정답 진단으로 이미 분리된 OSCE 형식**이라 우리 ASK/EXAM/TEST/DIAGNOSE 스키마로 거의 그대로 변환 가능. 대화형 진단 에이전트 벤치마크 그 자체임 |
| 2 | **MedR-Bench** (진단 957건, 그중 희귀질환 491건) 및 **MedCaseReasoning** (14,489건) | MedR-Bench: CC BY-SA (LICENSE 파일, 버전 미표기) / MedCaseReasoning: GitHub README는 CC-BY 4.0, HF 메타데이터는 MIT | 실제 PMC Open Access 증례 보고 기반이라 Claude 합성 증례보다 현실적. MedR-Bench는 "기본 정보 → 검사 추천 → 진단"의 다회 턴 평가를 이미 설계해 둠. MedCaseReasoning은 의사가 쓴 감별진단 추론 문장이 있어 Safety/감별 평가 기준 만들기에 좋음 |
| 3 | **snuh/ClinicalQA** (1,045문항, 한국어) | Apache-2.0 (HF 메타데이터) | **한국어**, 주호소(chief complaint) 기반, 의사 3명 검토. 한국어 임상 문체를 연습하기에 가장 가까운 공개 자료. 단, 초안을 LLM이 생성한 객관식 형식이라 증례로 쓰려면 변환 필요 |

대안: **DiagnosisArena**(915건, MIT, 필드가 증례 정보/신체검사/검사/최종진단으로 분리되어 스키마 적합도는 가장 높음)는 원천이 저널 증례 보고(JAMA, Cell 등)라 MIT 표기가 원문 저작권까지 해결하는지 불명확 → 내부 평가용으로만 조건부 사용.

### 라이선스 주의 (요약)
- **MIMIC 계열(AgentClinic-MIMIC-IV, MIMIC-IV-Ext 등)**: PhysioNet 자격 필요 + **일반 Gemini API/ChatGPT 전송 금지**(Vertex AI 등 승인 서비스만 허용). 현재 개발 구성(ai.google.dev Gemini API)으로는 사용 불가.
- **NEJM 기반 증례(AgentClinic-NEJM, NEJM Case Records)**: 원문 저작권이 NEJM에 있고 별도 라이선스 표기가 없음 → 제외.
- **MedQA 데이터 묶음 안의 영어 교과서 18권**: 상용 교과서 텍스트로 보이며 저작권 해결 근거가 없음 → 문항만 쓰고 교과서는 제외.
- **대한의사협회 의학용어(term.kma.org)**: 무단 전재·재배포 금지 명시 → 제외.
- **UMLS / SNOMED CT**: 라이선스 계약 필요, 재배포 제한 → 저장소·실행 데이터에 넣을 수 없음.

---

## 2. 지식 자료 비교표

| 자료 | 내용 | 규모 | 언어 | 라이선스 (확인 URL) | 연구 발표 | 실행 데이터 포함 (소용량, CPU) | 외부 LLM API 전송 | 적합도 |
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

모두 **연습·평가 전용**이며 실행 코드가 읽는 데이터에는 넣지 않는다 (실행 데이터에 증례를 넣을 이유가 없고, 증례 간 정보를 쓰지 않는다는 설계 원칙과도 충돌할 수 있음).

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
| AgentClinic-MIMIC-IV, MIMIC-IV-Ext 계열 전반 | PhysioNet 자격 증명 필요 + DUA가 OpenAI API·ChatGPT 등 제3자 전송 금지. 우리 개발 환경의 Gemini API(ai.google.dev)는 PhysioNet이 허용한 서비스(Vertex AI 경유 Gemini 등)가 아님. 저장소 포함도 불가 |
| MedQA에 포함된 영어 교과서 18권 | 상용 교과서 텍스트로 보이며 재배포 허락 근거가 없음 (문항 데이터와 분리해서 사용하지 말 것) |
| UMLS Metathesaurus | 라이선스 계약 필요. 하위 집합 포함 재배포 금지(애플리케이션 일부로만 예외), 소스별 추가 제한(Category 3: 내부 연구만, Category 4: 미국 내 한정) (https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html) |
| SNOMED CT | 한국은 SNOMED International 회원국이라 국내 사용은 무료이나 Affiliate License 등록 필요, 저장소 배포 조건 불명확 (https://www.snomed.org/get-snomed, https://www.snomed.org/members). 우리 과제에 비해 부담이 큼 |
| 대한의사협회 의학용어위원회 용어 (term.kma.org) | "모든 정보의 저작권은 대한의사협회 의학용어위원회에 있으며" 사전 승인 없는 전재·재배포 금지. 다운로드 제공 없음 |
| MedlinePlus A.D.A.M. 의학 백과사전, ASHP 약물 정보 | 저작권 보유(MedlinePlus 공식 안내). 공공 도메인인 건강 주제 요약만 사용 |
| WHO ICD-10 원본 | 라이선스를 1차 출처에서 확인하지 못함. HIRA 상병마스터(KOGL 1유형)로 대체 |
| MDCalc 등 계산기 사이트의 설명 문구 | 사이트 저작권. 공식은 원 논문에서 직접 가져오고 설명은 직접 작성 |
| MMSE, MoCA 등 저작권 있는 평가 도구 | 사용 허가·라이선스 필요 |
| snuh/specialist-level·essential-level_medical_knowledge_dataset_sft | CC-BY-ND-4.0 (HF 메타데이터). AI Hub 데이터에서 파생되고 Qwen3로 추론을 생성한 SFT용 데이터. ND라 가공본 배포 불가이고, 이 프로젝트는 파인튜닝을 하지 않으므로 활용처가 적음. 원천 AI Hub 약관(해외 반출·제3자 제공 제한 가능성)도 미확인 |
| AI Hub 의료 데이터 (전문 의학지식 등) | 이용 신청·약관 동의가 필요하고, 해외 서버(외부 LLM API) 전송 허용 여부를 확인하지 못함 → 확인 전까지 제외 |

---

## 5. 확인 못 한 점

1. **NC 라이선스와 공개 저장소**: KorMedMCQA, PMC-Patients, StatPearls, 국가건강정보포털은 비상업(NC)·변경 금지(ND) 조건이 있어, 가공본을 공개 저장소나 실행 데이터에 넣기 전에 조건을 다시 확인해야 한다. 연습·평가용으로만 쓰고 배포하지 않으면 위험이 낮다. (현재 이 자료들은 하나도 쓰지 않는다.)
2. **DDXPlus 라이선스 버전**: README에 "CC-BY"만 있고 버전이 없으며 GitHub에 LICENSE 파일이 없다. figshare 페이지(원 배포처)는 403으로 열람하지 못했다. 지식 파일의 정확한 크기도 확인하지 못했다.
3. **MedCaseReasoning 라이선스 불일치**: GitHub README는 CC-BY 4.0, HF 메타데이터는 MIT. 또 PMC OA Subset에는 CC BY-NC 논문도 섞여 있는데(라이선스가 논문마다 다르다고 PMC가 명시), 비상업 논문을 걸러냈는지 확인하지 못했다. MedR-Bench도 같은 문제가 있고 "CC BY-SA"의 버전이 없다.
4. **MedQA 원문 저작권**: 저장소는 MIT지만 문항은 웹에서 수집한 USMLE 대비 문제은행에서 왔다(논문 기준). 원 문제은행의 권리 관계는 확인하지 못했다. AgentClinic-MedQA도 이 문제를 그대로 물려받는다.
5. **DiagnosisArena 원문 저작권**: MIT 표기와 달리 원 저널 증례의 재배포 허락 여부가 불명확하다. 제작자 스스로 일부 저널을 저작권 때문에 뺐다고 적었다.
6. **KorMedMCQA 원 기출문항 권리**: 한국보건의료인국가시험원 허락 여부가 데이터 카드와 논문 초록에 없다. 논문 본문은 확인하지 못했다.
7. **StatPearls 라이선스**: NCBI Bookshelf 페이지가 reCAPTCHA로 막혀 1차 페이지를 직접 보지 못했고, 검색 결과로만 CC BY-NC-ND 4.0을 확인했다. 대량 다운로드 경로(Bookshelf OA 여부)도 미확인이다.
8. **임상 규칙 원 논문 서지**: 규칙 목록은 제안 단계이며, 구현할 때 각 규칙의 원 논문(예: Wells PE Thromb Haemost 2000, HEART Neth Heart J 2008, qSOFA JAMA 2016, Ottawa ankle JAMA 1993, Canadian CT Head Lancet 2001, CURB-65 Thorax 2003 등)을 원문에서 다시 확인해 `docs/licenses.md`에 인용과 함께 기록해야 한다. 이번 조사에서는 원문을 열람하지 않았다.
9. **규모 수치**: Disease Ontology 용어 수, MedlinePlus 토픽 수, MeSH descriptor 수, HPO 파일 크기, HIRA CSV 실제 용량은 확인하지 못했다(표에 "미확인"으로 표시).
10. **한국어 임상 진료지침**: 대한의학회 임상진료지침정보센터 등 학회 지침은 학회별 저작권이라 일괄 라이선스를 찾지 못했다. KTAS(한국형 응급환자 분류도구) 같은 한국 도구의 라이선스도 확인하지 못했다.
11. **증례 언어·형식**: 평가 증례는 한국어 대화형을 기본으로 정했다. 다른 언어·형식의 환경에서는 한국어 자료의 우선순위를 다시 봐야 한다.

---

## 6. 로컬 평가용 증례 변환 (eval-simulator, 2026-09-26)

실행 코드가 읽지 않는 **로컬 평가 전용** 증례. 실행 시 읽는 데이터는 `src`, `data/kb`, `data/lexicon`뿐이다. AgentClinic·DiagnosisArena 변환본은 비공개로만 썼고 공개 저장소에는 없다(8장). 원본 다운로드는 `data/external/`(git-ignored)에만 둔다.
영어 원문은 증례 파일에 넣지 않는다(환자 LLM에 정답이 새지 않도록). 증례에는 `source {dataset, id, license, url}`만 남긴다.

| 출처 | 라이선스 (1차 확인) | 출력 폴더 | 선택 기준 | 변환 스크립트 / 기록 |
|---|---|---|---|---|
| AgentClinic-MedQA | MIT, repo LICENSE.txt (https://github.com/SamuelSchmidgall/AgentClinic, commit b6570ed) | `data/cases_agentclinic/ac_<id>.json` | 기본 세트 107건 전부(`agentclinic_medqa_extended.jsonl` 0-106행 = `agentclinic_medqa.jsonl`). NEJM·MIMIC-IV 하위 세트는 쓰지 않음 | `scripts/convert_english_cases.py agentclinic` / `data/labels/agentclinic_conversion_meta.json` |
| DiagnosisArena | MIT, HF 카드 + GitHub LICENSE (https://huggingface.co/datasets/shzyk/DiagnosisArena, rev 7163704) | `data/cases_diagnosisarena/da_<id>.json` | 915건 중 50건: 본문 700-2200자, 단일 진단(복합 병인·병기·숫자 제외), 피부 병변으로 시작하는 증례 제외(영상·조직 사진 의존), 본문에 정답명이 이미 나오는 증례 제외 → id 순 10개 층에서 5건씩 무작위(seed 20260926). 난이도는 모두 "어려움"으로 고정 | `scripts/convert_english_cases.py diagnosisarena` / `data/labels/diagnosisarena_conversion_meta.json` |
| snuh/ClinicalQA (2차분) | Apache-2.0 | `data/cases_clinicalqa/cqa_<id>.json` (기존 40건은 수정하지 않음) | 진단형 문항 중 정답이 질환인 55건 추가(`EXTRA_IDS`) | `scripts/convert_clinicalqa.py --set extra` / `data/labels/clinicalqa_conversion_meta.json` |

라이선스 주의
- **AgentClinic-MedQA**: 저장소는 MIT이지만 증례는 MedQA(USMLE형) 문항을 LLM으로 확장한 것이다. MedQA 저장소도 MIT이나 문항 자체는 웹의 USMLE 대비 문제은행에서 수집되어 원 권리자의 허락 근거가 없다 → 출처 논란이 있으므로 **내부 평가 전용**이다. 발표물에 쓰지 않고, 변환본을 공개 저장소에 올리지 않는다 (8장).
- **DiagnosisArena**: 저널 증례 보고(Cell, JAMA 등)를 각색한 데이터이고 제작자가 일부 저널을 저작권 때문에 뺐다고 적었다. MIT가 원문 저작권까지 해결하는지 불명확 → **내부 평가 전용**, 변환본 재배포 금지 (8장).
- MedR-Bench는 이번에 쓰지 않았다(DiagnosisArena로 충분하고, CC BY-SA 버전 미표기).

변환 규칙 (공통)
- LLM: `.env`의 `CONVERT_LLM_*`(없으면 `LLM_*`), temperature 0, 병렬 2개 이하(API 할당량 공유), 429는 클라이언트가 30초 대기 후 재시도(최대 6회).
- 원문에 있는 사실만 옮긴다. 검증에 실패하면 오류를 붙여 한 번 재시도하고, 또 실패하면 건너뛰고 로그에 남긴다.
- 검증: JSON 스키마, 정답 일치(영어 출처는 원문 영어 진단명이 `aliases`에 그대로 있어야 함), `initial`과 모든 키에 정답 누출 금지, exam/tests에 원문에 없는 숫자 금지(영어 숫자 단어 포함), exam/tests에 원문에 없는 단위 금지. 경고(로그만): 값 안의 정답명, 원문 숫자 누락, history의 추가 숫자.
- 증례 형식: `category`(한국어 주호소, ClinicalQA 주호소 목록에서 선택), `difficulty`, `teaching_point`, `must_check: []`, `source`.

결과 (2026-09-26, gemini-3.6-flash)

| 세트 | 변환 / 건너뜀 | 재시도 | 난이도 (쉬움/보통/어려움) | 많은 주호소 |
|---|---|---|---|---|
| AgentClinic-MedQA | 107 / 0 | 4 | 38 / 68 / 1 | 피부발진 15, 팔다리 근력약화 9, 관절통 9, 피로 7, 기분장애 6 |
| DiagnosisArena | 49 / 1 (id 178: history에 원문에 없는 숫자, 재시도 후에도 실패) | 8 | 0 / 0 / 49 (고정) | 호흡곤란·피부발진·발열 각 5, 의식변화 4 |
| ClinicalQA 2차분 | 55 / 0 | 0 | 37 / 18 / 0 | 관절통 6, 가려움증·만성복통 각 5 |

검토 결과와 알려진 한계
- 사람이 원문과 대조해 18건을 확인했다(ClinicalQA 7, AgentClinic 7, DiagnosisArena 4). 없는 사실을 지어낸 것은 2건이었고, 둘 다 프롬프트·검증을 고친 뒤 전체를 다시 변환했다. (1) `11 pound` 뒤에 "약 5kg" 환산값을 덧붙임 → 환산 금지 규칙을 넣고 history 숫자 불일치를 오류로 격상. (2) 원문에 단위가 없는 수치(BMI 26)에 단위를 붙이고, 혈당의 원문 단위를 고침 → 단위 불일치 검사를 추가.
- 한국어 용어 오역: 진단명과 모든 영어-한국어 쌍을 검토해 오역 13건(예: pemphigus vulgaris → "보통천식창", GFAP astrocytopathy → "…난소세포병증")과 한자가 섞인 출력 9건을 찾았다. `scripts/convert_english_cases.py`의 `OVERRIDES`에 기록하고 `--apply-overrides`로 LLM 없이 고쳤다. 이후 실행부터는 한자를 검증 오류로 처리한다.
- 남은 사소한 문제: 검색 키 일부의 오타·무의미한 토큰, 어색한 표현("맥박 111/회"), 원문 수치 일부 누락(주로 °F 병기, 출생체중·산모 나이 등 13+16건, meta에 경고로 기록). 사실 오류는 아니다.
- 원문 검사 결과 문장에 정답명이 들어 있는 경우(예: "MRI: PML에 부합하는 병변")는 원문 그대로 두었다(AgentClinic 21건, meta 경고). 이 증례들은 해당 검사를 하면 사실상 정답이 드러난다.
- AgentClinic은 LLM이 매긴 난이도상 "어려움"이 1건뿐이다. 판별력이 필요하면 DiagnosisArena나 ClinicalQA의 어려움 증례와 섞어 쓴다.
- DiagnosisArena의 `initial`은 원문 첫 문장 탓에 병력 정보가 많이 들어간 경우가 있다(예: da_854, da_858).

재실행
```bash
source .venv/bin/activate
# 원본 다운로드 (data/external/, git-ignored)
mkdir -p data/external/agentclinic data/external/diagnosisarena
curl -sSL https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/b6570edefb940857a7c334350656b29f9d984f24/agentclinic_medqa_extended.jsonl -o data/external/agentclinic/agentclinic_medqa_extended.jsonl
curl -sSL https://huggingface.co/datasets/shzyk/DiagnosisArena/resolve/7163704e580ba643647483307f64d426eef33ef0/data/test-00000-of-00001.parquet -o data/external/diagnosisarena/test.parquet
# 변환 (기존 출력은 건너뜀, --overwrite로 다시 생성)
python scripts/convert_clinicalqa.py --set extra
python scripts/convert_english_cases.py agentclinic            # --set extended: 나머지 107건
python scripts/convert_english_cases.py diagnosisarena
python scripts/convert_english_cases.py agentclinic --apply-overrides      # 검토자 수정만 다시 적용 (LLM 호출 없음)
python scripts/convert_english_cases.py diagnosisarena --apply-overrides
# 평가에 쓰기
python eval/run_local.py --cases data/cases_agentclinic
```

전체 보강과 품질 검사 (2026-09-26~27)
- `scripts/augment_cases.py --mode full`이 네 세트 전부(sample 16, clinicalqa 95, agentclinic 107, diagnosisarena 49)에 표준 진찰·검사 묶음과 결정적·감별 검사 결과를 정답과 모순 없이 채워 **`data/cases_aug/<세트>/` 267건**을 만들었다(공개 저장소에는 sample·clinicalqa 111건만 있음, 8장). 추가 항목은 `augmented`로 표시하고, 보강 메타데이터는 환자 LLM에게 보이지 않는다(`eval/llm_patient.py`의 `HIDDEN_KEYS`). 기록: `data/labels/augmentation_full_meta.json`. 표준 실험(`eval/experiment.py`)은 이 세트를 쓴다. (`data/cases_clinicalqa_aug/` 40건은 이전의 결정적 검사만 보강한 판이다.)
- `scripts/check_cases.py`(LLM 없음)가 정답 누출, 원본과 보강 항목 사이의 모순(활력징후·검사값·단위·발열·있음/없음), 성별·나이에 맞지 않는 검사, 비현실적인 수치, 키워드 시뮬레이터의 검색어 문제를 찾는다. 반드시 0이어야 하는 hard 문제는 48 → 0 (검색어 자리표시자 45, 성별·나이 부적합 검사 2, 활력징후 불일치 1; 15건 57곳 수정). 판단이 애매한 soft 문제 2,278건은 검토 목록으로 남겼다. 기록: `data/labels/case_quality_2026-09-27.json`, 테스트: `tests/test_case_quality.py`.
- 보강 항목은 LLM이 쓴 것이고 의료진 검토를 받지 않았다. 원본 사실은 고치지 않는다.

---

## 7. 구축한 지식베이스 (data/kb/, 2026-09-26 구축 · 09-27 검사 연결, Orphanet 빈도·유병률 추가)

`scripts/build_kb.py`가 원자료를 `data/external/kb_raw/`(git 제외)에 내려받아 `data/kb/`(실행 시 읽음)를 만든다. 추론 코드는 `src/doctor_agent/knowledge/kb.py` (표준 라이브러리만, CPU, 네트워크 없음, 처음 호출할 때 한 번 적재). 에이전트는 `src/doctor_agent/agent/kb_hints.py`를 통해 **참고용 힌트**로만 쓴다 (`AGENT_USE_KB=0`이면 끔). 인터페이스는 `docs/architecture.md`의 Knowledge base 절 참조.

### 7.1 포함한 자료와 재확인 결과
| 자료 | 라이선스 재확인 | 쓰임 |
|---|---|---|
| DDXPlus 지식 파일 (영문판) | figshare API 메타데이터로 **CC BY 4.0** 확인 (앞선 조사의 "버전 미확인" 해소). https://api.figshare.com/v2/articles/22687585 | 질환 49개의 증상·병력, ICD-10, 중증도, 문진 질문 (LLM으로 한국어 질문 문구 생성) |
| HIRA 상병마스터 20250930 | 데이터 페이지에 "공공저작물: 출처표시 (제 1유형)". 47,798행, 한방 전용 행 제외 | KCD 코드·한글/영문명·성별 제한 (`kcd.tsv.gz`), 프로필 한국어 이름 |
| Disease Ontology releases/2026-08-31 | GitHub LICENSE = CC0 1.0 | 질환 뼈대 12,282개: 이름, 동의어, 정의, ICD-10-CM/MeSH/OMIM/ORDO 교차참조, `has_symptom` 문구 |
| Wikidata (SPARQL, 2026-09-26) | 구조화 데이터 CC0 (https://www.wikidata.org/wiki/Wikidata:Licensing) | 질환↔증상(P780), 질환↔검사(P923), 한국어 라벨. DOID 또는 ICD-10이 있는 항목만 사용 (화학물질 노출 항목 제거) |
| MedlinePlus Health Topics XML 2026-09-26 | 건강 주제 요약은 공공 도메인 (https://medlineplus.gov/about/using/usingcontent/). A.D.A.M.·ASHP·이미지는 사용 안 함 | 질환 토픽 548개에서 LLM으로 증상·검사·한 줄 한국어 요약 추출. 원문에 없는 영어 용어는 버림 |
| Orphadata (Orphanet) product 4 + product 1, JDBOR 2026-06-23 | **CC BY 4.0** — https://sciences.orphadata.com/phenotypes/ 페이지와 XML 안 `<Licence>`에 명시 (법적 고지 https://www.orphadata.com/legal-notice/). 출처 표기: "Orphadata Science: Free access data from Orphanet. © INSERM 1999. … Data version 2026-06-23" | 희귀질환 4,355개의 표현형과 **빈도 등급**(항상/매우 흔함/흔함/가끔/매우 드묾/**없음(0%)**), 이름·동의어, OMIM·ICD-10 대응 (7.9) |
| Wikidata P1550·P3841 (SPARQL, 2026-09-27) | CC0 | Orphanet id → 기존 질환, HPO id → 기존 용어 다리, 새 표현형 용어의 영/한 라벨 |
| HIRA 주부상병(3단) 성별 연령군별 건강보험 진료 통계 20251231 | 데이터 페이지에 "공공저작물: 출처표시 (제 1유형)", 포털 메타데이터 `rsrchPblictnAt=Y`. https://www.data.go.kr/data/15118806/fileData.do | KCD 3자리 코드별·성별·5세 연령군별 환자수(2025년) → 약한 유병률 사전확률 (7.10) |
| **자체 작성 표** (출처 태그 `curated`) | 우리가 쓴 문구·정규식·기준값 | `kb_curated.py`: 소견 표현 → 기존 KB 용어 대응 (7.4). `kb_tests.py`: 검사 결과 → 질환 연결, 연결마다 참고문헌 (7.5) |

### 7.2 제외
- **Human Phenotype Ontology 파일(hp.obo) + phenotype.hpoa**: 여전히 쓰지 않는다 (2026-09-27). 대신 Orphanet이 CC BY 4.0으로 배포하는 HPO 코드 주석을 쓰고, 표현형은 기존 KB 용어(영어 라벨 일치 또는 Wikidata P3841 다리)에 붙인다. 대응이 없는 표현형만 `HP:<id>` 용어가 되며 영어 라벨은 Wikidata 라벨(있으면) 또는 Orphanet이 준 라벨을 **바꾸지 않고** 쓴다. 원래 제외 사유: 라이선스가 "HPO 파일의 내용과 논리적 관계를 어떤 식으로든 변경하지 말 것"을 요구한다 (http://human-phenotype-ontology.github.io/license.html; reusabledata.org는 "restrictive"로 분류). 우리 KB는 재구성·번역한 파생물이라 조건 충족이 불명확하고, 희귀질환 중심이라 가치도 낮아서 제외했다. 주석의 원천인 OMIM도 별도 라이선스가 있다.
- **Human Symptoms–Disease Network (Zhou 2014, Nat Commun)**: Europe PMC 기준 오픈 액세스 라이선스가 확인되지 않아 제외.
- NC/ND 자료(StatPearls, 국가건강정보포털 등)는 KB에 넣지 않았다.
- 검사 결과 연결을 만들 때 검토한 자료 (2026-09-27):

| 자료 | 판정 | 이유 |
|---|---|---|
| Wikidata P923 "medical examination" (CC0) | 이미 사용 중, 추가 없음 | 검사 **이름**만 있고 결과·방향(상승/양성)이 없다 |
| Wikidata P5131 "possible medical findings" (CC0) | 쓰지 않음 | 전체 139개 진술뿐이고 대부분 심잡음·진찰 징후. 새 용어의 한국어 라벨을 만들려면 LLM이 필요한데 이번 작업은 LLM 호출 금지 |
| Wikidata P2293 genetic association (CC0) | 쓰지 않음 | 유전자 연관은 진료 현장 감별에 쓸모가 적다 |
| Human Disease Ontology 논리 정의 (CC0) | 추가 없음 | `has_symptom`만 있고 검사 결과 관계가 없다 (이미 반영) |
| MedlinePlus 검사 페이지 (NLM 작성분 공공 도메인) | 이번엔 쓰지 않음 | 서술형 문장이라 구조화하려면 LLM 추출이 필요. 다음 후보 (원문 근거 확인 방식으로) |
| LOINC | 쓰지 않음 | 자체 라이선스 조건이 있고, 검사 코드 체계일 뿐 질환 연결이 없다 |
| HPO | 계속 제외 | 위 참조 (Orphanet 주석은 사용, 7.9) |
| 국민건강보험공단 질병 소분류(3단/4단상병) 통계 (data.go.kr 15138832/15138833) | 쓰지 않음 | 이용허락범위가 "제한 없음"으로만 적혀 있고 공공누리 유형 표시가 없음 → 불명확하면 제외 원칙. HIRA 15118806(제1유형)으로 대체 |
| HIRA 다빈도질병통계 (data.go.kr 15065488, 제1유형) | 쓰지 않음 | 라이선스는 가능하지만 상위 질병만 있어 15118806(전체 3단 코드 2,029개, 성·연령별)이 상위호환 |

### 7.3 규모 (현재 빌드, 2026-09-27 Orphanet·유병률 추가 후)
- `kb.json.gz` 2.88MB + `kcd.tsv.gz` 0.46MB + `kcd3_prev.tsv.gz` 0.12MB, `data/kb` 전체 약 3.5MB (한도 20MB). 적재 약 0.9초 (이전 0.3초; 늘어난 용어의 계층 계산이 대부분).
- 질환 프로필 14,517개 (이전 12,597; Orphanet 단독 `ORPHA:` 프로필 1,920개 추가). 증상 있음 5,369 (이전 1,418), 한국어 이름 4,709, KCD 코드 4,151, 검사 이름 473, 한국어 요약 505, 검사 결과 연결 229, Orphanet 빈도 있음 4,267.
- 용어 10,953개 (이전 2,996; 새 표현형 용어 `HP:` 7,961개 중 한국어 라벨 있는 것 525개 — 만들 때 Wikidata 371, KCD 영문명 일치 176이었고 같은 한국어 라벨의 기존 용어와 합치거나 질환 자기 이름인 것을 뺀 뒤 525 — 나머지는 영어 라벨만). KCD 코드 21,124개.
- DDXPlus 49개 중 40개는 DO/Wikidata 질환에 연결, 9개는 독립 프로필 (Boerhaave, 자발 늑골골절, 스콤브로이드 중독, 후두경련, 급성 근긴장이상, 국소 부종, 안정형 협심증, URTI, 만성 비부비동염).
- 빌드 LLM 호출: gemini-3.6-flash, temperature 0, 154회, 동시 작업 2개. 기록은 `data/labels/kb_build_meta.json`, 항목별 결과는 `data/labels/kb_llm_cache.json`에 있다. 검사 결과 연결(7.5)은 LLM 없이 만들었다.
- 빌드 버그 수정 (09-27): 한국어 라벨이 빈 용어들이 "같은 한국어 라벨" 병합 규칙에 걸려 한 용어의 동의어로 몰리던 문제 (이전엔 모든 용어에 한국어가 있어 드러나지 않음), 질환 자신이 자기 표현형으로 들어가던 문제(졸린거-엘리슨 증후군)를 고쳤다.
- 모든 필드에 출처가 있다: 이름·코드·정의는 `[값, 출처]`, 증상·검사는 `[용어 id, [출처...]]` 형태로 저장한다. 출처 태그는 `DO`, `WD`, `KCD`, `DDXPlus`, `MedlinePlus+LLM`(요약문에서 LLM이 뽑은 것), `LLM`(한국어 번역), `curated`(우리가 쓴 표)이다. `tests/test_kb.py`가 이를 검사한다.

### 7.4 소견 매칭과 순위 (kb.py + kb_curated.py, 2026-09-26)
- **매칭**: 한국어 라벨은 단어 시작(또는 우측/좌측/급성 같은 짧은 접두어 뒤)에서만 맞춘다 ("호흡음"이 "서호흡"으로, "상부 종격동"이 "부종"으로 잡히던 문제). "수포음", "다음 날" 같은 오인 단어는 먼저 지운다. 일반어 용어("관측", "증가하다", "혈당", "통증" 등 50개)는 매칭에서 뺀다. 4자 미만·대문자 약어 영어 라벨("GAD", "MS")은 쓰지 않는다.
- **curated 표** (`src/doctor_agent/knowledge/kb_curated.py`): 기존 KB 용어 170개에 한국어 표현 691개, 유연한 구문 정규식 39개("소변 볼 때 좀 찌릿" → 배뇨통, "열이 좀 났어요" → 발열), 활력징후·검사값 판정("WBC 14,200" → 백혈구증가증, "헤모글로빈 8.2" → 빈혈, "Na 128" → 저나트륨혈증, 체온·맥박·호흡수·혈압·SpO2 등), 진단명 철자 쌍 24개(지주막하↔거미막하, 갑상선↔갑상샘 등). 이 표는 **질환–증상 연결을 추가하지 않는다** (소견을 기존 용어에 대응시킬 뿐).
- **용어 계층(backoff)**: 구체적 용어의 라벨 안에 일반 용어가 있으면 그 일반 용어도 가진 것으로 본다 ("갑작스러운 안면 및 사지 위약" → 근력 약화, 가중치 0.4; 소견 쪽 "우하복부 통증" → 하복부 통증·복통, 0.75). 라벨에서 자동으로 만든다 (993개 용어, 1,750개 연결). 가족력·병력 용어는 제외.
- **같은 말 중복 제거**: 한 소견의 같은 글자 범위에 걸린 용어들(안절부절못함/초조/정신운동성 초조)은 하나의 개념으로 보고 질환마다 가장 잘 맞는 하나만 점수에 넣는다. 이미 앞선 소견에서 나온 개념은 다시 세지 않는다.
- **음성 소견·성별·나이**: `candidates(findings, k, negatives, sex=, age=)`. 음성 소견은 "없음" 같은 꼬리를 떼고 매칭해 해당 증상을 가진 질환을 깎는다. `sex`("남성"/"여성")를 주면 KCD 코드가 모두 다른 성별 전용인 질환을 뺀다. `age`는 이름에 소아/childhood 등이 붙은 질환을 성인에서 낮춘다. `patient_profile("35세 여성")` → `("여성", 35.0)`. **`kb_hints.py`는 아직 sex/age를 넘기지 않는다.**
- **진단명 정규화** (`normalize_diagnosis`): 코드 → 철자 변형 포함 정확 일치(KCD 먼저) → 조금 더 긴 공식명("유착성 관절낭염" → "어깨의 유착성 관절낭염", 임신·신생아 등 맥락어가 붙는 경우 제외) → 수식어(급성/원발성/좌측…) 제거 후 정확 일치 → 문장 끝 단어 기준으로 포함된 가장 긴 이름("급성 ST분절상승 심근경색(전벽)" → 급성 심근경색증 I21) → 글자 bigram 퍼지(한국어 5자 이상·0.6 이상이며 0.7 미만이면 앞 두 글자 일치 필요, 영어 0.75 이상, 약어는 퍼지 안 함). 결과의 `match`가 어느 단계인지 알려준다 (`kcd_exact`, `profile`, `superstring`, `kcd_backoff`, `profile_backoff`, `contained`, `fuzzy`). 동점 처리가 해시 순서에 따라 달라지던 비결정성도 고쳤다.

### 7.5 검사 결과 ↔ 질환 연결 (kb_tests.py, 2026-09-27)
09-26 벤치마크에서 가장 큰 병목은 KB 내용이었다: 공개 자료(DDXPlus, DO, Wikidata, MedlinePlus)는 질환을 증상과 **검사 이름**("CT", "혈액 검사")에만 연결하고, **검사 결과**("리파아제 3배 상승", "AMA 양성", "CT 충수 비후")와는 연결하지 않는다. 그래서 문진 후반의 결정적 검사 결과가 순위에 쓰이지 못했다. 공개 자료 중 이를 채울 것이 없어서(7.2 표) 자체 표를 만들었다.

- **자체 작성 표** `src/doctor_agent/knowledge/kb_tests.py` (출처 태그 `curated`): 사실은 교과서·진료지침 수준이며 연결마다 참고문헌 키를 단다. 참고문헌 95개는 PubMed E-utilities로 PMID·제목·저자·학술지·연도를 대조해 확인했다 (`docs/licenses.md`). 특정 지침에 묶지 못한 연결은 `textbook`(미검증)으로 표시. 유료·비상업(NC) 자료의 문장은 복사하지 않았다.
- **규모**: 검사 결과 개념 262개 (정량 상승 49, 정량 저하 17, 정성 양성/음성 52, 영상·병리·심전도 판독 키워드 144), 질환 연결 452개 (결정적 3점 186, 강한 근거 2점 127, 비특이 1점 139), 연결된 질환 프로필 229개. 연결 340개는 PMID가 있는 지침·리뷰를, 112개는 `textbook`을 근거로 한다. 정상 결과가 질환을 반대하는 "배제 가능(R)" 연결은 41개 (트로포닌→심근경색, 리파아제→급성 췌장염, D-dimer→폐색전증·심부정맥혈전증, ANA→SLE, HbA1c→당뇨병, 케톤→DKA, β-hCG→자궁외임신, CT 충수→충수염 등).
- **빌드**: `scripts/build_kb.py`가 표를 읽어 연결마다 대상 프로필 id가 KB에 있는지 확인하고(없으면 빌드 실패), 프로필에 `findings_from_tests = [[용어 id, ["curated"], 가중치, 참고문헌 키, "R"|""]]`, 용어 사전에 `TF:<id>` 용어(한국어·영어 이름)를 넣는다. 참고문헌 목록은 `meta.test_refs`. 같은 입력에서도 `PYTHONHASHSEED`에 따라 코드·프로필 순서가 바뀌던 문제를 고치고 gzip 헤더 시각을 0으로 고정해, `--no-llm` 재빌드는 바이트 단위로 같은 파일을 만든다.
- **인식** (`kb_tests.detect`, 표준 라이브러리 정규식): 소견을 쉼표·세미콜론 단위로 나눈 뒤 검사명을 찾고, 바로 뒤 문구로 방향을 정한다. 숫자는 기준값과 비교하고(단위 환산: 트로포닌 ng/L, D-dimer ng/mL, HbA1c mmol/mol 등), "(정상 0.4-4.0)" 같은 참고치가 적혀 있으면 그것을 쓰며, "정상 상한의 5배", 역가 "1:320", "< 0.01" 같은 비교 기호도 처리한다. 판독 키워드는 뒤따르는 "없음/관찰되지 않음/정상"으로 음성 처리하고, 병력("기흉 병력")·배제 목적("배제 위해 시행")·결과 대기("진행 중")는 무시한다. 음성 소견 목록으로 들어온 글은 글 안에 부정어가 없으면 전체를 정상으로 본다.
- **순위** (`candidates()`): 양성 결과 하나가 질환마다 `TEST_W(20) × {3: 1.0, 2: 0.5, 1: 0.15} / (1 + 0.2 × (연결된 질환 수 − 1))`를 더한다. 증상 점수와 같은 설명 비율·출처 prior를 거친다. 정상 결과는 R 연결 질환에서 `5 × 가중치 비율`을 뺀다. 인자와 반환 형식은 그대로이고, `matched`에 `TF:` 용어가, `sources`에 `curated`가 추가된다.
- **감별·조회**: `discriminators()`가 `test_findings_a_only/b_only/shared`를 새로 주고, 가중치 2 이상 결과를 `tests_a_only/b_only` 맨 앞에도 넣는다 → 감별 힌트가 "CT" 대신 "리파아제 상승" 같은 결정적 결과를 보여 준다. `lookup()` 프로필에 `findings_from_tests`, `render_for_prompt()`에 "결정적 검사:" 항목, `match_terms()`에 검사 결과 용어가 생겼다.
- **힌트**: `kb_hints.candidate_hint`는 일치 항목이 2개 이상인 후보만 보여 주되, **검사 결과(`TF:`) 하나만 맞은 후보도 보여 준다** (커밋 ca5e704).
- **인자 조정은 dev에서만**: `TEST_W` 6→50은 dev에서 단조 증가하다 20 이후 평평해 20으로 정했다. 나머지 인자(2·1점 비율, 퍼짐, 음성 벌점)는 dev에서 ±3증례 안이라 중간값을 골랐다. DO 상위 질환의 연결을 하위 질환이 물려받게 해 봤지만 dev top-10이 0.685→0.658로 떨어져 넣지 않았다.

### 7.6 오프라인 벤치마크 (`scripts/eval_kb.py`)
- **방법**: `data/cases_aug`의 267 증례에서 초기 정보·병력·진찰·검사 답변을 절로 나누고, 부정("없", "음성", "(-)")·정상("정상", "평탄", "명료", "5/5" 등) 절은 음성 소견으로, 나머지를 양성 소견으로 넣어 `candidates(k=50)`에서 정답 질환의 순위를 본다. 정답은 `diagnosis` + `aliases`를 KB에서 찾은 프로필 전부. LLM 호출 없음.
- **과적합 방지**: **dev = sample + clinicalqa (111)** 에서만 조정하고, **held-out = agentclinic + diagnosisarena (156)** 는 보고만 했다. held-out 증례 내용은 보지 않았다.

현재 결과 (`data/labels/kb_eval_2026-09-28_nlp.json`; Orphanet 단계는 `kb_eval_2026-09-27_orphanet.json`, 검사 연결 단계는 `kb_eval_2026-09-27.json`)와 단계별 변화:

| | top-1 | top-3 | top-10 | top-50 | MRR |
|---|---|---|---|---|---|
| held-out, 09-26 이전 kb.py (커밋 705efb8) | 0.038 | 0.109 | 0.192 | 0.288 | 0.092 |
| held-out, 09-26 매칭 개선 후 (재측정) | 0.045 | 0.109 | 0.211 | 0.295 | 0.091 |
| held-out, 09-27 검사 연결 후 | 0.090 | 0.179 | 0.269 | 0.359 | 0.147 |
| dev, 09-26 이전 | 0.063 | 0.162 | 0.270 | 0.396 | 0.133 |
| dev, 09-26 매칭 개선 후 | 0.108 | 0.207 | 0.333 | 0.495 | 0.183 |
| held-out, 09-27 Orphanet 빈도 + 유병률 사전확률 | 0.109 | 0.199 | 0.269 | 0.372 | 0.165 |
| dev, 09-27 검사 연결 후 | 0.460 | 0.604 | 0.685 | 0.775 | 0.542 |
| dev, 09-27 Orphanet 빈도 + 유병률 사전확률 | 0.469 | 0.631 | 0.730 | 0.793 | 0.559 |
| **held-out, 09-28 소견 정규화 계층(nlp) 도입 (현재)** | **0.147** | **0.211** | **0.295** | **0.397** | **0.197** |
| dev, 09-28 소견 정규화 계층 도입 (현재) | 0.469 | 0.631 | 0.730 | 0.784 | 0.565 |

- 두 번째 줄은 빌드 재현성 수정 뒤 검사 연결 없이 다시 잰 값이다 (순위 동점 처리가 프로필 순서에 달려 있어 09-26 기록의 held-out top-10 0.205와 조금 다르다).
- 세트별 현재 top-10: sample 0.875, clinicalqa 0.705, agentclinic 0.327, diagnosisarena 0.143 (희귀 증례 보고라 거의 안 맞는다; Orphanet 추가 뒤에도 top-1 0.020, top-10 0.143 그대로, MRR 0.051→0.060, top-50 0.265→0.245. 이유는 7.9). agentclinic은 top-1 0.121→0.149, top-50 0.402→0.430.
- 병력+진찰만 (검사 결과 제외, 힌트가 실제로 뜨는 문진 단계): 현재 dev top-10 0.351, held-out 0.186. 09-26 개선으로 dev 0.261 → 0.342, held-out 0.180 → 0.186이 됐고, 검사 연결은 여기에 거의 영향이 없다 (진찰 징후 일부만 해당).
- 정답이 KB에 있는 비율: dev 95.5% (증상까지 있는 비율 73.0%), held-out 85.3% (55.1%). 증상까지 있는 정답만 보면 현재 top-10은 dev 0.78, held-out 0.47 (09-26: 0.46, 0.37).
- 09-26 개선에서 무엇이 효과가 있었나 (하나씩 뺀 결과, dev/held-out top-10): 용어 계층을 빼면 0.333→0.261 / 0.205→0.167로 가장 크게 떨어진다. curated 정규식을 빼면 dev top-1 0.108→0.081. 검사값 판정, 음성 소견, 성별·나이는 ±1~2증례 수준(잡음 범위)이다. BM25 인자, 출처 prior, 설명 비율 가중치는 어떤 값을 줘도 dev에서 ±1증례 안이라 거의 그대로 두었다.
- **정규화** (현재): 대표 진단명의 KCD 코드 부여율 dev 78.4%, held-out 56.4%. 같은 증례의 이름·별칭이 같은 KCD 3자리로 모이는 비율 dev 68.4%, held-out 67.9%. 09-26 개선 때 held-out 전체 이름 적중률이 67.0% → 59.6%로 줄었는데, 대부분 틀린 퍼지·약어 추정이 없어진 것이다 (바뀐 결과 45개를 직접 확인: 개선 28, 악화 6, 비슷함 11. 예: "STEMI"가 NSTEMI I21.4로, "중추성 요붕증"이 신성 요붕증 N25.1로 가던 오류가 없어짐).
- **지연**: `candidates()` 평균 26.0ms, p95 32.4ms (09-28 정규화 계층 도입 전 11.2ms / 14.7ms; Orphanet 추가 뒤 12.1ms / 16.0ms, Orphanet 추가 전 8.8ms / 10.1ms, 검사 결과 정규식 추가 전 2.6ms). KB 적재 0.9초 (이전 0.32초).
- **솔직한 평가**: 09-26 개선과 09-27 검사 연결 모두 dev 실패를 보며 고친 부분이 있다 (예: "LAD"가 LA 등급 D로, "반월상 연골"이 반월체로, "Hb 12.4"가 B12로 잡히던 인식 오류, 혈우병 인자·작은 신장·과립 원주·위·담관 종괴 개념 추가). 그래서 dev 향상은 부풀려져 있고 **held-out 향상(09-26 이전 대비 top-1 +5.2%p, top-10 +7.7%p, MRR +0.055)이 실제 기대치**다.
- **회귀 테스트**: `tests/test_kb_matching.py`가 dev 고정 50증례(sample 16 + clinicalqa 앞 34)에서 top-1 ≥ 22, top-10 ≥ 32, top-50 ≥ 36 (작성 시 25, 35, 39)을 확인한다. `tests/test_kb_tests.py`는 인식 방향 36개, 오인식 8개, 표↔KB 일치, 결정적 결과 순위 9개, 정상 결과 벌점, 반환 형식, 감별·프로필·렌더, 크기·지연을 본다. 조사용 소형 벤치마크(고전적 증상 조합 15개; 1위 5개, 5위 이내 10개, 09-26)는 참고용으로만 남긴다.

- **09-28 소견 정규화 계층 도입** (7.11): 세트별 top-1/top-10/MRR — sample 11/14/0.766 → 11/15/0.797, clinicalqa 41/67/0.524 → 41/66/0.526, agentclinic 16/35/0.213 → 20/37/0.238, diagnosisarena 1/7/0.061 → 3/9/0.107. 병력+진찰만: dev top-10 0.378 → 0.441, held-out 0.192 → 0.224 (top-1 11 → 10증례). `candidates()` 평균 11.2 → 26.0ms (p95 14.7 → 32.4ms).

### 7.7 한계
- **후보 목록은 힌트일 뿐 진단 근거가 아니다.** held-out top-1은 15%다. 증상 빈도는 Orphanet이 다루는 희귀질환(4,267개 프로필)에만 있다; 흔한 질환은 여전히 빈도가 없어 여러 출처가 같은 증상을 들면 가중치를 높이는 방식이다. Orphanet 이전에는 증상 있는 프로필 1,418개 중 절반이 증상 3개 이하였다. DO 정의문·한국어 요약에서 용어를 추가로 뽑거나 증상 없는 질환이 상위 개념의 증상을 물려받게 해 봤지만 dev에서 이득이 없어 넣지 않았다.
- **의사 검토 전이다.** 한국어 라벨 상당수(약 2/3)와 MedlinePlus 추출은 LLM 산출물이다. Wikidata의 증상 연결에는 이상한 항목도 섞여 있다 (예: 폐렴의 "코골이"). curated 표의 검사값 기준은 일반 성인 기준을 우리가 고른 것이다 (성별·나이·검사실별 참고치 없음; 적혀 있는 참고치가 있으면 그것을 우선).
- **검사 결과 인식은 정규식이다.** 문장 구조가 복잡하면 놓치거나 방향을 틀릴 수 있다 (예: 뇌척수액 소견이 "WBC 1,200 (호중구 90%)"처럼 검체 이름 없이 오면 세균성 수막염으로 연결 못 함). 결정적 결과 하나가 증상 점수 전체보다 커질 수 있게 가중치를 줬으므로, 인식이 틀리면 순위도 크게 틀린다.
- **연결은 KB에 이미 있는 프로필에만 걸 수 있다.** 횡문근융해증, 헤파린 유발 혈소판감소증, 난소 염전, 흉막삼출, 추간판 탈출증, 간농양, 미세다발혈관염 등은 프로필이 없어 빠졌다. 일부 프로필 이름이 어긋나 있다 (DOID:446은 영어로 원발성 알도스테론증인데 한국어 이름이 쿠싱증후군이라 두 검사군을 모두 걸었다; `lookup("원발성 담즙성 담관염")`은 담관염으로 풀린다).
- **남은 실패 유형**: (1) 정답 프로필에 증상이 없거나 1~3개뿐 (held-out의 약 45%); (2) 흔한 증상(발열·구토·복통)만 겹치는 감염·중독 질환(장염, 뎅기, 스트리크닌 중독 등)이 위로 올라옴; (3) 흡연·음주 같은 위험인자가 물질 관련 질환을 끌어올림.
- 벤치마크 소견은 증례 원문을 절 단위로 자른 것이라 실제 에이전트가 모으는 짧은 소견보다 길고 잡음이 많다.

### 7.8 재빌드와 평가
```bash
source .venv/bin/activate
python scripts/build_kb.py --no-llm      # 캐시된 LLM 출력만 사용 (API 호출 없음, 바이트 단위로 같은 결과)
python scripts/eval_kb.py                # 벤치마크 → data/labels/kb_eval_<date>.json
python scripts/eval_kb.py --no-write --show-misses 20 --split heldout
python scripts/eval_kb.py --no-write --prev-w 0   # 유병률 사전확률 끈 비교
```

### 7.9 Orphanet 표현형 빈도 (2026-09-27, LLM 호출 없음)
- **자료**: Orphadata `en_product4.xml`(질환 ↔ HPO 코드 표현형 + 빈도 등급, 4,357개 질환 116,664개 주석)과 `en_product1.xml`(이름·동의어·교차참조 + 대응 관계 E/NTBT/BTNT). 빌드가 XML 안의 라이선스가 CC-BY-4.0인지 확인한다.
- **질환 연결** (`build_kb.map_orpha`): DO의 ORDO 교차참조 → Wikidata P1550 → Orphanet의 정확(E) OMIM 대응 ↔ DO MIM → 영어 이름 정확 일치 순서로, 한 그룹만 가리킬 때만 붙인다. 2,435개는 기존 프로필에 붙고 1,920개는 새 `ORPHA:<코드>` 프로필이 된다 (영어 이름·동의어, 출처 `Orphanet`). 정확(E) ICD-10 대응만 `icd10`→KCD 코드·한국어 이름이 되고, "Orphanet이 더 좁음(NTBT)" 대응은 `kcd_broad`에만 넣는다 (넓은 KCD 이름을 한국어 이름으로 쓰지 않기 위해). 새 프로필의 한국어 이름은 LLM 없이는 만들 수 없어 대부분 비어 있다.
- **표현형 → 용어** (`build_kb.orpha_terms`): Orphanet 라벨 또는 Wikidata P3841로 이어진 항목의 영어 라벨(단·복수 변형 포함)이 기존 KB 용어와 같으면 그 용어를 쓴다 (HPO 용어 731개; Wikidata에 따로 항목이 있는 HPO는 기존 용어의 별칭만 같을 때 붙이지 않는다 — Wikidata "abscess"의 별칭 "ulcer" 같은 오류 방지). 아니면 `HP:<id>` 새 용어: 영어 라벨은 Wikidata(있으면) 또는 Orphanet 라벨 그대로, 한국어 라벨은 Wikidata 한국어 라벨(371개) 또는 KCD 영문명과 정확히 같은 경우의 KCD 한글명(176개, "상세불명의"·"달리 분류되지 않은" 제거)만. **LLM 번역 없음**. 한국어가 같은 용어는 기존 규칙대로 하나로 합친다.
- **저장**: 프로필 `symptoms`에 출처 `Orphanet`, 새 필드 `orpha_freq = [[용어 id, "O"|"VF"|"F"|"OC"|"VR"|"EX"]]` (114,913개, 그중 "없음(0%)" 731개). 여러 하위 질환이 한 프로필에 붙으면 가장 높은 빈도를 쓰고, 양성이 있으면 EX는 버린다. 등급 설명은 `meta.orpha_freq`.
- **점수** (`kb.py`): Orphanet 출처의 특징 가중치 = `ORPHA_W` {항상 0.6, 매우 흔함 0.5, 흔함 0.35, 가끔 0.15, 매우 드묾 0.05} (다른 출처는 1.0씩 더함). BM25 길이에는 Orphanet 단독 특징을 `ORPHA_W × 0.25`로만 센다 (표현형 60~80개인 희귀질환 프로필이 흔한 증상 하나하나로는 크게 오르지 않게, 그러면서 기존 프로필의 길이 정규화는 그대로). 환자가 가진 소견이 그 질환에서 "없음(0%)"이면 `EXCL_W(1.0) × idf`를 뺀다. `profile()`의 증상에 `freq`("매우 흔함(80-99%)" 등), `excluded` 목록이 생기고, 증상 순서는 출처 수 → 빈도 → 특이도, `render_for_prompt`는 한국어 라벨 있는 증상을 먼저 보인다.
- **인자 조정 (dev만)**: 빈도 가중치를 그대로(1.2/1.0/0.7/0.35/0.1) 쓰고 길이도 전부 세면 dev top-1 0.460→0.432로 떨어졌다. 길이 계수 0/0.25/0.5/1/2, 가중치 ×0.5/×1/×0.25, 기존 프로필에 붙은 Orphanet 가중치 축소(0/0.5/1), EX 벌점 0/1/3을 dev에서 비교했고 길이 0~0.25, 가중치 ×0.5 근처가 평평한 최고(dev top-1 0.486~0.495, MRR 0.576; 아래 빌드 버그 수정 전 측정)라 그 중간값을 골랐다. EX 벌점은 dev에서 효과가 없었다 (해당 증례 없음; 원리상 넣음).
- **예**: "소화성 궤양, 설사, 식도염" → 졸린거-엘리슨 증후군 순위 없음 → 1위. "소뇌실조증, 근긴장저하, 안구운동 실행증" → 주베르 증후군 없음 → 1위. "안구진탕증, 눈부심, 난시" → 백색증 없음 → 1위 (`tests/test_kb_orphanet.py`).
- **diagnosisarena가 거의 그대로인 이유** (구조 통계만 확인, 증례 내용은 보지 않음): 49증례 중 정답이 KB에 있는 것 39, 정답 프로필에 Orphanet 자료가 있는 것 9, 정답이 Orphanet 단독 프로필인 것 0. 즉 이 세트의 희귀 증례는 대부분 Orphanet 질환이 아니거나(드문 발현의 흔한 질환, 종양 등) 정답 이름이 KB에 없다. 그리고 새 표현형 용어 7,961개 중 7,436개가 영어 라벨만 있어 한국어 소견과 맞지 않는다.
- **다음 후보**: 새 표현형 용어(특히 주석이 많은 상위 1,000개)의 한국어 라벨을 오프라인 LLM으로 만들고 원문 라벨 대조로 검증하기 (이번 작업은 LLM 금지라 하지 않음), Orphanet 단독 프로필 한국어 이름.

### 7.10 한국 유병률 사전확률 (2026-09-27)
- **자료**: 건강보험심사평가원 주부상병(3단) 성별 연령군별 건강보험 진료 통계 2025 (공공누리 제1유형). 주·부상병 전체 기준 KCD 3자리 코드 2,029개 × 성 2 × 5세 연령군 18의 환자수. 빌드가 데이터 페이지에 "제 1유형"이 있는지 확인하고 내려받아 `data/kb/kcd3_prev.tsv.gz`(0.12MB, 환자수만)에 넣는다.
- **적용**: 프로필의 KCD 코드 3자리의 환자수 중 최대값 (없으면 Orphanet `kcd_broad` 코드의 환자수를 그 코드를 쓰는 프로필 수로 나눈 값), 환자 성별·나이가 있으면 그 칸의 값. `z = clamp((log10(1+환자수) − 3) / 3, −1, 1)`, 코드 없음은 z = −0.5. **증상 점수 부분에만** `1 + PREV_W × z`를 곱한다 → 결정적 검사 결과 점수는 그대로라 희귀질환도 검사 결과로 1위를 지킬 수 있다. `PREV_W = 0.15` → 영향은 ±15%로 제한.
- **인자 조정 (dev만)**: `PREV_W` 0→0.8에서 dev top-10은 0.2까지 오르고(0.703→0.748) top-1은 0.1을 넘으면 떨어져(0.495→0.460) 약한 값 0.15를 골랐다. 성·연령 칸을 쓰는 쪽이 전체 합보다 dev top-10이 나았다 (0.2에서 0.748 vs 0.721). 코드 없음 값(0/−0.25/−0.5/−1), 기준점(2/3/4)은 ±1증례.
- **held-out 분리 효과** (보고만): 기존 KB + 유병률만 = top-1 0.083, MRR 0.143 (효과 없음); Orphanet만(`--prev-w 0`) = top-1 0.109, top-3 0.186, top-10 0.269, top-50 0.359, MRR 0.160; 둘 다 = top-1 0.109, top-3 0.199, top-10 0.269, top-50 0.372, MRR 0.165. 유병률은 Orphanet과 함께일 때만 top-3·MRR을 조금 올린다.
- **한계**: 환자수는 부상병·의심 진단(R/O)까지 포함한 청구 기준이라 실제 유병률보다 흔한 질환 쪽으로 부풀려져 있다. 이 사전확률은 **다른 증례의 정보가 아니라 고정된 공개 통계**이며 증례 사이에 공유되는 학습·캐시가 없다.

### 7.11 소견 정규화 계층으로 이전 (2026-09-28, LLM 호출 없음)
- **무엇이 바뀌었나**: 소견 → KB 용어 매칭과 부정 처리를 `doctor_agent.nlp`(어휘집 + 파서)로 옮겼다. 소견마다 `parse(소견, "claim")`로 개념·극성(있음/없음/불확실)·주어(환자/가족/타인)·측정값을 읽고, 개념 → KB 용어는 `data/lexicon/kb_links.json`(27KB, 294개 개념, 링크 724개: curated 198, label 100, lexicon 359, form 2, 상위개념 65)으로 잇는다. KB 자체 라벨 스캔은 모든 용어에 그대로 두되, 각 적중은 겹치는 어휘집 언급의 극성을, 없으면 같은 규칙으로 읽은 자기 구간의 극성(`nlp.assess_spans`)을 받는다. 있음 → 질의 용어, 없음 → 음성(양성 목록 안의 "열은 없음"도), 불확실·가정·가족의 소견 → 무시. 옛 끝부분 부정 정규식(`_NEG`, `_strip_neg`)은 지웠다.
- **kb_curated**: `SYNONYMS`/`REGEX`/`lab_terms()`는 실행 시 더 이상 읽지 않는다 (어휘집이 689개 구절 중 682개를 같은 용어로 잇는다). `scripts/build_lexicon.py`와 링크 빌더가 원천으로 읽으므로 파일에는 남긴다. `STOP_TERMS`, `BAD_LABELS`(예: "무릎의 열감"은 발열 아님), `BLOCK_WORDS`, 진단명 정규화 표는 계속 쓴다.
- **링크 재생성**: `python scripts/eval_kb.py --build-links` (결정적; 어휘집·KB·kb_curated가 바뀌면 다시 돌린다. 오래되면 `tests/test_kb_matching.py::test_links_file_is_fresh`가 실패).
- **결과**: 7.6 표. held-out top-1 17 → 23, top-10 42 → 46, top-50 58 → 62, MRR 0.165 → 0.197; dev는 top-1·top-10 그대로, top-50 −1, MRR +0.006. 세부 조정 기록은 `docs/nlp.md` "KB migration numbers".
- **크기**: `data/kb` 3.3MB (변화 없음), `data/lexicon` 552KB (`kb_links.json` 27KB 추가).

### 7.12 진단명 정규화 감사 (2026-09-29, LLM·네트워크 없음, KB 재빌드 없음)
- **도구**: `python scripts/audit_normalize.py [--json OUT]`. `normalize_diagnosis()`를 증례 정답 이름(`data/cases_aug` 267증례의 진단명+별칭 1,391개), 분과 정답(`specialty_gold_v*.jsonl` 205개), 위험 배제 이름(`danger_gate` 119, `protocols` 117, consult 75; "A·B" 목록은 나눠서)에 돌리고 값싼 검사로 의심 대응을 표시한다: KCD 장이 분과 정답과 다름(chapter), 중독·암·골절 낱말과 코드 범위 불일치(semantic), 정확 일치가 아닌데 이름 유사도가 낮음(weak)·공통 낱말 없음(overlap), 여러 낱말 이름이 "기타/상세불명" 코드로(generic), 한 증례 이름들이 다른 분과 그룹으로 갈림(pair).
- **결과**: 같은 감사 스크립트로 이전 코드 287개 이름 표시 → 수정 후 175개. 남은 200개 표시는 모두 사람이 검토해 `data/labels/normalize_audit_allow.json`에 이유와 함께 적었다 (대부분 "상위 개념으로 물러남, 같은 장"; 고치지 못한 한계는 "known limitation"으로 표시). `tests/test_normalize_audit.py`가 같은 검사를 돌려 목록 밖 표시가 생기면 실패한다.
- **원인별 수정 (kb.py 매처 논리)**: ① 쉼표 이름 "X, Y"의 X를 별칭으로 넣을 때 Y가 순수 수식어이거나 X가 여러 낱말 영어 구·3자 이상 한국어 명사일 때만 ("심장성, 심장 또는 심근부전 NOS" → "심장성", "용혈, 간효소상승 …" → HELLP의 "용혈" 같은 조각 제거). ② KCD 괄호 이름을 2순위 표로 ("상세불명의 두개내출혈(비외상성)" → "두개내출혈", "헤노흐(-쇤라인)자반"). ③ 상위 이름(superstring) 단계에서 외상·유전 아형 낱말은 개념을 바꾸는 것으로 보고, 영어는 낱말 경계에서만 ("ventricular tachycardia" ≠ "supraventricular …"). ④ 퍼지 일치: 한국어는 첫 글자(장기)와 머리 명사가 같아야 하고, 영어는 질의의 각 낱말이 이름에 대부분 있어야 하며, 반대말 형태소(고/저, hyper/hypo, clast/blast)만 다른 이름은 거부. ⑤ 여러 KCD 범주가 같은 표를 받으면 KCD 제목과 프로필 이름의 일치도, 그다음 출처 순서로 고른다 (예전 "짧은 코드→알파벳" 규칙은 혈관염을 I80 정맥염으로 보냈다). ⑥ 문장 속 3자리 코드는 단독·괄호·"KCD" 표시일 때만 코드로 읽는다 ("파르보바이러스 B19" ≠ B19 간염). ⑦ 한국어 하이픈 복합어는 가운데서 자르지 않는다 ("폐-신장 증후군" ≠ 신증후군). ⑧ MedlinePlus 주제 제목 중 과다복용·중독 제목은 중독 코드가 없는 프로필에 색인하지 않는다 ("Opioid Overdose" → F11.1 남용).
- **문서화된 일회성 표 (kb_curated.py)**: `CONTAINED_BLOCK`(동형이의어 "이식증" = 이식증(異食症) 대 이식(移植)), `NAME_ANTONYMS`, `POISON_CODES`(약물 + 중독 낱말 → T36-T65; 리튬처럼 ICD 약물표에서 코드가 갈리는 것은 제외), `ABBREVIATIONS`/`ABBR_AMBIGUOUS`(ACS가 acrocallosal 증후군, SJS가 쇼그렌으로 가던 문제; HD·PD·MS 같은 다의어 약어는 정규화하지 않음), `NAME_CODES`(급성 관상동맥 증후군 I24.9, 저·고칼슘혈증 E83.5, 당뇨병성 신증 E14.2, 면역 혈소판 감소증 D69.3), `PROFILE_KCD`(혈관염 I77.6), `NAME_SUBS` 추가 철자쌍(다카야수/타카야수, 라이터 증후군, 폐동맥 색전, 뇌수막염/수막염 등).
- **크기·속도**: `data/kb` 변화 없음. `normalize_diagnosis` 평균 0.96 → 0.99 ms (퍼지 켬), `specialty_of` 0.087 → 0.101 ms, 로드 시간 차이 없음.

---

## 8. 공개 저장소로 만들 때 (재배포 금지 자료)

**공개 저장소에 들어 있는 증례**는 두 종류뿐이다.

| 경로 | 개수 | 출처·라이선스 |
|---|---|---|
| `data/sample_cases/`, `data/cases_aug/sample/` | 16 | 이 프로젝트가 쓴 합성 연습 증례. 의료진 검토 없음 |
| `data/cases_clinicalqa/`, `data/cases_clinicalqa_aug/`, `data/cases_aug/clinicalqa/` | 95 (보강 초기판 40) | [snuh/ClinicalQA](https://huggingface.co/datasets/snuh/ClinicalQA), **Apache License 2.0**. 이 프로젝트가 **수정한 파생물**이다: 원 문항을 언어모델로 대화형 한국어 증례로 변환하고(`scripts/convert_clinicalqa.py`), 원문에 없던 진찰·검사 결과를 언어모델로 보강했다(`scripts/augment_cases.py`, `augmented` 표시). 변환·보강 결과는 의료진이 따로 검토하지 않았다. 파일마다 `source {dataset, id, license}`와 `_note`에 출처와 변경 사실을 적었다(Apache-2.0 4조의 변경 고지) |

**AgentClinic-MedQA·DiagnosisArena에서 만든 증례는 비공개로 평가에만 썼고, 공개 저장소에는 넣지 않았다.** 두 원천 모두 저장소 라이선스는 MIT이지만, 변환·보강본은 원문을 옮긴 파생물이라 재배포 근거가 없다(3장, 6장).

| 뺀 경로 | 내용 | 이유 |
|---|---|---|
| `data/cases_agentclinic/` (107건), `data/cases_aug/agentclinic/` (107건), `data/labels/agentclinic_conversion_meta.json` | AgentClinic-MedQA 증례의 한국어 변환·보강본과 변환 기록 | 원 문항(MedQA, USMLE 대비 문제은행)의 권리 근거가 없음 |
| `data/cases_diagnosisarena/` (49건), `data/cases_aug/diagnosisarena/` (49건), `data/labels/diagnosisarena_conversion_meta.json` | DiagnosisArena 증례의 한국어 변환·보강본과 변환 기록 | 원천이 학술지 증례 보고이고, 제작자도 "연구·모델 평가 목적"으로 한정. MIT가 원문 저작권을 해결하는지 불명확 |

**두 세트의 줄·항목만 걸러 낸 파일** (ClinicalQA·자체 증례 줄만 남김, 2026-10-06):

| 파일 | 전체 → 공개판 |
|---|---|
| `data/labels/findings_gold_candidates.jsonl` | 9,176 → 3,456줄 |
| `data/labels/findings_gold_draft.jsonl`, `findings_gold_v1.jsonl` | 397 → 173줄 (`label_findings.py freeze`로 다시 만들어도 같음) |
| `data/labels/findings_gold_fresh_v1.jsonl` | 80 → 34줄 |
| `data/labels/findings_gold_review.txt`, `findings_gold_corrections.json` | 검토 항목 353 → 129 |
| `data/labels/findings_dev_ids.json` | 403 → 159 |
| `data/labels/findings_gold_v1_metrics.json` | 공개판 173줄로 다시 계산 (`label_findings.py metrics`) |
| `data/labels/augmentation_full_meta.json` | 항목 기록 353 → 146 (프롬프트·검사 묶음 정의는 그대로) |
| `data/labels/case_quality_2026-09-27.json` | 수정 기록 15 → 3증례, 남은 문제 목록 262 → 109증례. 전체 요약(267건)은 집계로 남기고 공개분 요약 `summary_after_public`을 추가 |
| `data/labels/kb_eval_2026-09-2*.json` (4개) | 증례별 줄 267 → 111. 세트별·전체 집계는 원문이 없으므로 그대로(비공개 세트 포함 값) |
| `eval/case_lists/dev.txt`, `smoke.txt` | 공개 111건에서 다시 뽑음 (`eval/experiment.py --regen-case-lists`, 시드 2026: clinicalqa 43 + sample 7 / clinicalqa 4 + sample 1) |
| `tests/` 일부 | 두 세트의 문장을 옮긴 시험 입력과 주석의 `ac_…`/`da_…` 번호를 같은 뜻의 다른 문장으로 바꿈 |
| `scripts/check_cases.py` | 두 세트에만 해당하던 수동 수정표(`FIXES`) 3건을 비움 |

- 시험 하한 조정: 증례 품질 검사 대상 ≥ 250 → ≥ 100(공개 111건), 어휘집 정답 세트 크기 ≥ 300 → ≥ 170(F1 하한 0.93은 그대로, 공개판 0.948), 진단명 정규화 감사 이름 수 > 1,500 → > 1,000(공개판 1,083).
- **README와 문서의 "267증례"·"held-out" 수치는 비공개 세트를 포함해 잰 기록값**이다. 공개 데이터만으로는 dev(= 공개 111건) 쪽만 재현된다.
- 변환 스크립트(`scripts/convert_english_cases.py`)는 남겼다. 원본을 각 데이터셋의 조건에 따라 직접 내려받으면 로컬에서 다시 만들 수 있다(6장 명령). 만든 파일은 커밋하지 않는다.
- 지식 베이스(`data/kb/`)는 CC BY 4.0·CC0·공공누리 1유형·미국 정부 저작물·자체 작성만 담고 있어 출처 표기(`data/kb/SOURCES.md`)를 지키면 배포할 수 있다.
- **git 기록**: 위 파일의 이전 판은 과거 커밋에 남아 있으므로, 공개 전에 기록에서 지우거나 새 저장소로 옮긴다.
- 집계 수치(예: held-out 지식 베이스 순위)와 증례 번호·진단명만 적은 문서 기록은 원문을 담지 않으므로 그대로 둔다.
