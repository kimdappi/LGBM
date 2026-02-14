# 에이전트 설계 및 활용 조건 정리

## 1. 전체 파이프라인 개요

```
[Phase 0] 에피소딕 메모리 검색 (진단+임베딩 유사도)
              ↓ episodic_lessons
[1] Chart Structurer ──→ [2] Evidence Agent (1차)
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    [3] Diagnosis Agent              [4] Treatment Agent
              └───────────────┬───────────────┘
                              ↓
              [5] Evidence Agent (2차) ──→ [6] Intervention Checker
                              ↓
              [7] Critic Agent (서브그래프)
                              ↓
                         [END]
[Phase Final] 에피소딕 메모리 저장
```

---

## 2. 에이전트 목록 및 형식

| # | M&M 역할 | 에이전트명 | 노드/위치 | 주요 역할 |
|---|----------|------------|-----------|-----------|
| 1 | 발표 전공의 | **Chart Structurer** | `chart_structurer` | 케이스 전체 구조화, 타임라인·주요 소견 정리 |
| 2 | 문헌 리뷰어 | **Evidence Agent** | `evidence`, `evidence_2nd` | 임상 패턴 기반 문헌 탐색, 2-Pass CRAG, 품질 관리 |
| 3 | 진단 전문의 | **Diagnosis Agent** | `diagnosis` | 진단 누락·감별 검토, 결정적 오류·논리 결함 분석 |
| 4 | 치료 전문의 | **Treatment Agent** | `treatment` | 치료 적절성, 퇴원 결정·위험도, 대안 제시 |
| 5 | 간호사/약사 | **Intervention Checker** | `intervention_checker` | 실제 시행 치료 검증, 허위 비판 차단 |
| 6 | 부서장/좌장 | **Critic Agent** | `critic` | 전체 논의 종합, 핵심 문제·근거 품질·개선안 도출 |

---

## 3. 각 에이전트 활용 조건

### 3.1 Chart Structurer (발표 전공의)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. 그래프 진입점(첫 노드). |
| **입력** | `patient_case` (특히 `clinical_text` 또는 `text`) |
| **출력** | `structured_chart` (demographics, vitals, symptoms, interventions_given, outcome 등) |
| **조건별 동작** | • 텍스트 길이 &lt; 50자 → 기본 구조만 반환, 분석 계속 진행<br>• LLM/JSON 파싱 실패 → 기본 구조 반환 (분석 중단 없음) |
| **비고** | 설계상 "실패 시 전체 분석 중단"은 원칙이며, 코드는 fallback으로 기본 구조를 넣어 후속 노드가 동작하도록 함. |

---

### 3.2 Evidence Agent (문헌 리뷰어)

#### 1차 (evidence 노드)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Chart Structurer 직후. |
| **입력** | `patient_case`, `structured_chart`, `rag_retriever`(선택) |
| **출력** | `evidence` (내부/외부 검색 결과, retrieval_mode, quality_evaluation 등) |
| **조건별 동작** | • **RAG 없음** → 내부 검색 스킵, PubMed 등 외부만 사용 가능<br>• **유사 케이스 유사도 ≥ 0.7** → CRAG: 내부+외부 혼합 사용<br>• **유사 케이스 없거나 유사도 &lt; 0.7** → 외부(PubMed)만 사용<br>• **임상 패턴 감지** (VTE 고위험, ACS, 진단불명+호흡곤란 등) → 패턴별 맞춤 검색 쿼리 생성 |

#### 2차 (evidence_2nd 노드)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Diagnosis + Treatment 완료 후. |
| **입력** | `diagnosis_analysis`, `treatment_analysis`에서 나온 비판/이슈, 기존 `evidence` |
| **출력** | 기존 evidence에 **비판 기반 타겟 PubMed 검색 결과 병합** |
| **조건별 동작** | • 비판/이슈가 없으면 2차 검색량 적음<br>• 있으면 해당 내용으로 타겟 쿼리 생성 후 PubMed 검색·병합 |

---

### 3.3 Diagnosis Agent (진단 전문의)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Evidence 1차 직후, Treatment와 **병렬**. |
| **입력** | `patient_case`, `structured_chart`, `evidence`, `episodic_lessons` |
| **출력** | `diagnosis_analysis` (issues, procedural_safety 등) |
| **조건별 동작** | • **episodic_lessons 있음** → 프롬프트에 과거 유사 케이스 교훈 주입<br>• **episodic_lessons 없음** → "없음"으로 넣고 동일 로직 실행<br>• Rule-based: 시술 안전 수식어, 사망 원인 정렬 등은 텍스트 기반으로 항상 수행 |

---

### 3.4 Treatment Agent (치료 전문의)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Evidence 1차 직후, Diagnosis와 **병렬**. |
| **입력** | `patient_case`, `structured_chart`, `evidence`, `episodic_lessons` |
| **출력** | `treatment_analysis` (medication_issues, disposition_evaluation, recommendations 등) |
| **조건별 동작** | • **episodic_lessons 있음** → 프롬프트에 과거 교훈 주입<br>• **episodic_lessons 없음** → "없음"으로 넣고 동일 로직 실행<br>• Disposition: expired 케이스는 조기퇴원 비판 금지, 고위험 생존 케이스만 퇴원 적절성 평가 |

---

### 3.5 Intervention Checker (간호사/약사)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Evidence 2차 직후. |
| **입력** | `structured_chart`, `diagnosis_analysis`, `treatment_analysis` |
| **출력** | `intervention_coverage` (실제 시행 치료 coverage, 필터된 이슈 등) |
| **조건별 동작** | • 구조화 차트에서 `interventions_given`(medications, oxygen_therapy 등) 추출<br>• Diagnosis/Treatment 이슈 중 "이미 시행된 치료 부재" 주장 → 제거 또는 보정<br>• 별도 조건 분기 없음. 항상 동일 로직. |

---

### 3.6 Critic Agent (부서장/좌장)

Critic은 **서브그래프** 하나로 묶여 있으며, 내부에서 전처리·Router·도구·CritiqueBuilder·Feedback이 순서대로(및 조건부 반복) 실행된다.

#### 3.6.1 실행 조건 (메인 그래프 기준)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **항상** 실행. Intervention Checker 직후, 그래프의 마지막 노드. |
| **입력** | 메인 state 전체 (patient_case, similar_cases, structured_chart, diagnosis/treatment_analysis, evidence, intervention_coverage, preprocessing 등) → adapter로 Critic용 state 변환 후 invoke. |

#### 3.6.2 Critic 내부: 전처리(preprocess)

| 항목 | 내용 |
|------|------|
| **실행** | **항상** timeline, evidence, record_gaps 3개 도구 실행. |
| **추가 조건** | **similar_cases가 1개 이상 있으면** → `behavior_topk_direct_compare` 1회 **추가 실행** (예산 1 소모). |

#### 3.6.3 Critic 내부: Router

| 항목 | 내용 |
|------|------|
| **기본** | **HeuristicRouter** 사용. |
| **조건** | **환경변수 `CARE_CRITIC_ROUTER_LLM=1`** 이면 **LLMRouter** 사용 (ToolRAG + LLM으로 도구 선택). |
| **Heuristic 기준** | • 텍스트 길이 버킷(very_short / short / medium / long)<br>• 중증 키워드(shock, sepsis, intub 등) 또는 patient.status in ("dead","alive") → lens_severity_risk, lens_monitoring_response<br>• 진단 키워드 또는 medium/long → lens_diagnostic_consistency<br>• very_short/short → behavior_topk_direct_compare |

#### 3.6.4 Critic 내부: run_tools / run_requested_tools

| 항목 | 내용 |
|------|------|
| **실행** | Router가 선택한 도구만 실행. **이미 executed_tools에 있으면 스킵.** |
| **조건** | **executed_budget &lt; max_tools(기본 8)** 인 동안만 추가 실행. 초과 시 budget_stop. |

#### 3.6.5 Critic 내부: CritiqueBuilder

| 항목 | 내용 |
|------|------|
| **실행** | **항상** 1회 이상 (feedback 루프 시 재실행 가능). |
| **조건** | **OPENAI_API_KEY 있음** → LLM(gpt-4o-mini)으로 비판점 생성<br>**OPENAI_API_KEY 없음** → 휴리스틱(lens/behavior 결과만으로 요약 비판) |

#### 3.6.6 Critic 내부: Feedback

| 항목 | 내용 |
|------|------|
| **실행** | CritiqueBuilder 직후 **항상** 1회. |
| **조건** | • **ok == True** → END<br>• **ok == False 이고 feedback_round &lt; max_rounds(기본 2)** → run_requested_tools → router로 **반복**<br>• **feedback_round >= max_rounds** → 무조건 END |

#### 3.6.7 Verifier (Critic 노드 내, 서브그래프 밖)

| 항목 | 내용 |
|------|------|
| **실행 조건** | **similar_cases가 1개 이상** 이고 **OPENAI_API_KEY 있음** 일 때만 실행. |
| **동작** | 비판점 + 유사 케이스 top-3로 solutions(issue, solution, evidence, priority) 생성 → `updates["solutions"]` 갱신. |

---

## 4. 파이프라인 외부 활용 조건 (run 전)

| 항목 | 조건 | 효과 |
|------|------|------|
| **에피소딕 메모리 검색** | `episodic_store`가 로드되어 있고, `patient_case`에 clinical_text 등 있음 | 진단+임베딩 유사도로 과거 에피소드 검색, top_k=2, min_similarity=0.3 → `episodic_lessons` 문자열 생성 → Diagnosis/Treatment 프롬프트에 주입 |
| **유사 케이스 검색** | `rag_retriever` 존재, `run_agent_critique.py`에서 top_k=3 검색 | 유사도 ≥ 0.7만 `similar_cases`로 전달. 0개면 CRAG는 외부만, Critic의 BehaviorTopK·Verifier는 비활성화에 가깝게 동작 |
| **에피소딕 메모리 저장** | `episodic_store` 존재, 그래프 실행 완료 후 | 이번 케이스의 critique/solutions/진단 등으로 episode 추가 (LLM 요약 → MedCPT 임베딩 → FAISS 등) |

---

## 5. 요약 표: "어떤 조건에서 활용되는가"

| 에이전트 | 항상 실행 여부 | 조건부 동작 요약 |
|----------|----------------|------------------|
| Chart Structurer | ✅ 예 | 입력 부족/실패 시 기본 구조 반환 |
| Evidence (1차) | ✅ 예 | RAG·유사도에 따라 내부/외부/혼합 선택 |
| Evidence (2차) | ✅ 예 | 비판 유무에 따라 타겟 검색량 차이 |
| Diagnosis | ✅ 예 | episodic_lessons 유무로 프롬프트만 차이 |
| Treatment | ✅ 예 | episodic_lessons 유무로 프롬프트만 차이 |
| Intervention Checker | ✅ 예 | 조건 분기 없음 |
| Critic 전체 | ✅ 예 | 내부에서 similar_cases·Router·API키·feedback 라운드에 따라 도구 선택·반복·Verifier 여부 결정 |

이 문서는 현재 코드(`graph.py`, `critic_graph.py`, `run_agent_critique.py` 등) 기준으로 작성되었으며, 변경 시 함께 갱신하는 것을 권장합니다.
