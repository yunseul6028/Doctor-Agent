# Doctor-Agent

**N.O.V.A. 2026** (분당서울대학교병원 의료인공지능센터 주최) **대화형 의료 진단 AI 에이전트 대회** 출품작입니다.
가상 환자에게 질문하고(ASK), 신체진찰(EXAM)과 검사(TEST)를 골라, 최종 진단(DIAGNOSE)에 도달하는 의사 에이전트를 만듭니다.

- 대회 사이트: https://nova.snubhai.org
- 평가 기준: 진단 정확도 · 정보 획득 효율성 · 임상적 안전성 (증례당 최대 60턴)
- 예선 모델: `openai/gpt-oss-20b` 고정 (파인튜닝 금지, 추론 코드에서 외부 API 호출 금지)

> 대회 요약은 [`docs/competition.md`](docs/competition.md), 설계는 [`docs/architecture.md`](docs/architecture.md)에 있습니다.

---

## 에이전트 구조

```
환경(가상 환자) ⇄ 진료 루프 (agent/loop.py)
                     │
        ┌────────────┼──────────────────────────────┐
        ▼            ▼                              ▼
   진료 상태      행동 결정 (agent/policy.py)       근거 자료
   ─ 소견 장부    1. 모델이 다음 행동 제안           ─ 위험 질환·최소 안전 확인 (safety/protocols.py)
   ─ 감별 장부    2. 규칙으로 보정                    ─ 임상 결정 규칙 12종 (knowledge/clinical_rules.py)
   ─ 대화 기록       · 반복·유사 행동 차단               Wells, HEART, qSOFA, Ottawa SAH …
                     · 질문/진찰 유형 교정
                     · 필수 안전 확인 전 진단 시 되묻기
                  3. 진단 전 검토의 승인
```

**핵심 설계**
| 기능 | 하는 일 |
|---|---|
| **소견 장부** | 매 턴 새 소견을 양성 / 음성 / 결과없음으로 정리. 모델에게는 긴 대화 대신 장부를 보여줌 |
| **감별 진단 장부** | 후보별 확률·지지 소견·반대 소견·상태(유력/위험/배제)를 턴마다 이어서 관리 |
| **진단 전 검토** | 진단 직전, 같은 모델이 검토의 역할로 "모든 소견 설명? 모순? 위험 질환 배제? 더 구체적 진단? 확진 근거?"를 확인 |
| **지침 기반 안전 확인** | 주호소별 반드시 배제할 질환과 최소 확인 항목을 출처(지침·논문)와 함께 안내 |
| **근거 인용** | 매 행동에 근거를 쓰고, 판단 기준을 쓸 때는 이름과 출처를 인용 |
| **튼튼한 출력 해석** | 모델이 생각 과정을 먼저 출력해도 마지막 행동 JSON만 정확히 읽음 |

---

## 빠른 시작

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # API 키 입력 (.env는 git에 올라가지 않음)
```

| 할 일 | 명령 |
|---|---|
| 모델 없이 동작 확인 | `python eval/run_local.py --doctor dummy --patient keyword --judge none` |
| 연습 증례로 평가 | `python eval/run_local.py --cases data/sample_cases` |
| 까다로운 환자로 평가 | `python eval/run_local.py --persona mixed` |
| 결과 보기 (브라우저) | `python eval/viewer.py` |
| 직접 진료해 보기 | `python eval/play.py` (내가 의사) · `python eval/play.py --role patient` (내가 환자) |
| 제출 파일 만들기 + 규칙 검사 | `python scripts/package.py` |
| 자동 시험 | `pytest` |

macOS에서는 `진료 테스트.command`를 더블클릭하면 직접 진료 모드가 열립니다.

### 모델 설정
역할별로 다른 모델을 쓸 수 있습니다. `DOCTOR_LLM_*`(의사), `PATIENT_LLM_*`(가상 환자), `JUDGE_LLM_*`(채점)을 따로 지정하고, 지정하지 않으면 공통 `LLM_*`을 씁니다. 모두 OpenAI 호환 접속 방식이라 코드 수정 없이 `.env`만 바꾸면 됩니다.

---

## 평가 환경 (로컬)

- **가상 환자**: 증례 정보만으로 답하는 언어모델 환자. 정답과 해설은 가려져 있음. 증례에 없는 검사는 "결과가 제공되지 않습니다"로 답함 (정상과 구분)
- **까다로운 환자 4종**: 모호한 환자, 불안한 환자, 증상을 축소하는 환자, 기억이 흐린 환자. 여러 질문을 한 번에 하면 첫 질문에만 답함
- **채점**: 정확도(언어모델 채점자) · 효율성(사용 턴) · 안전성(지침 기반 확인 목록 수행률)
- **뷰어**: 실험 비교표 + 증례별 진료 대화, 턴마다 근거·감별 후보·빠뜨린 확인 표시

### 증례 세트
| 경로 | 내용 | 출처 |
|---|---|---|
| `data/sample_cases/` | 연습용 가상 증례 16개 (흉통·두통·복통 등) | 자체 제작 (의료진 검수 없음) |
| `data/cases_clinicalqa/` | 서울대병원 임상 문항 40개를 대화형 증례로 변환 | [snuh/ClinicalQA](https://huggingface.co/datasets/snuh/ClinicalQA) (Apache-2.0), 변환 과정에서 수정됨 |
| `data/cases_clinicalqa_aug/` | 위 40개에 빠진 검사 결과를 보강한 버전 | 보강 항목은 언어모델 생성 (`augmented` 필드에 표시, 의료진 검수 없음) |

변환·보강에 쓴 프롬프트, 모델, 날짜는 `data/labels/`에 기록되어 있습니다 (대회 재현성 규칙).

---

## 대회 규칙 준수

- 추론 코드에서 외부 네트워크 호출 금지 → `scripts/package.py`가 금지된 라이브러리 사용을 검사
- 증례마다 고정 모델을 최소 1회 호출 → 진료 루프에서 강제
- 증례 간 정보 공유 금지 → 증례마다 상태를 새로 생성
- 제출 파일 50MB 이하, 모델 가중치 파일 포함 금지 → 패키징 시 자동 검사
- 사용한 모든 데이터·모델·도구의 출처와 라이선스 → [`docs/licenses.md`](docs/licenses.md)

## 문서
| 문서 | 내용 |
|---|---|
| [`docs/competition.md`](docs/competition.md) | 대회 요약 (일정, 규칙, 평가) |
| [`docs/architecture.md`](docs/architecture.md) | 구조 설계 |
| [`docs/experiments.md`](docs/experiments.md) | 실험 기록 |
| [`docs/data-sources.md`](docs/data-sources.md) | 검증된 데이터 출처 조사 |
| [`docs/licenses.md`](docs/licenses.md) | 출처·라이선스 장부 |
| [`docs/submission-checklist.md`](docs/submission-checklist.md) | 제출 전 점검표 |

## 현재 상태
- 대회 참가자 가이드(공식 환경 인터페이스, 샘플 증례, 세부 채점 기준)가 **예선 전 공개 예정**입니다. 공개되면 `src/doctor_agent/env/`에 공식 연결부만 추가합니다.
- 개발 중에는 대회 모델 대신 제미나이·젬마 계열 모델로 실험하고 있습니다. 모델을 바꾸면 프롬프트를 다시 검증해야 합니다.
