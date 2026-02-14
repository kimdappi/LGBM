# 최적 Critic 분석 에이전트 프로세스 설계

현재 코드베이스 에이전트 + 제안된 조건부 에이전트를 조합한, **조화로운 Critic 분석**을 위한 단일 프로세스 설계안입니다.  
에이전트는 **필수 실행** 또는 **조건 충족 시 실행**으로 구분됩니다.

---

## 1. 에이전트 목록 및 역할 정리

| 구분 | 에이전트 | 역할 요약 | 실행 방식 |
|------|----------|-----------|-----------|
| 기존 | **Chart Structurer** | 케이스 전체 구조화, 타임라인·주요 소견 | **필수** |
| 기존 | **Evidence Agent (1차)** | 임상 패턴·문헌·유사 케이스 검색 (CRAG) | **필수** |
| 기존 | **Diagnosis Agent** | 진단 누락·감별·결정적 오류·논리 결함 | **필수** |
| 기존 | **Treatment Agent** | 치료 적절성·퇴원 결정·위험도·대안 | **필수** |
| 기존 | **Evidence Agent (2차)** | 비판 기반 타겟 문헌 검색 | **필수** |
| 기존 | **Intervention Checker** | 실제 시행 치료 검증, 허위 비판 차단 | **필수** |
| 기존 | **Critic Agent** | 논의 종합·핵심 문제·근거 품질·개선안 | **필수** (모드만 조건부) |
| 신규 | **Critique Eligibility (게이트)** | 풀 비판 적합 여부 판단 → Critic 모드 결정 | **필수** (규칙만 실행) |
| 신규 | **Risk Factor Agent** | 고위험 케이스 시 위험 인자·동반질환 집중 분석 | **조건부** |
| 신규 | **Process Contributor Agent** | 이상치/프로세스 기여 의심 시 지연·불일치 분석 | **조건부** |
| 신규 | **Alternative Explanation Agent** | 해석 애매 시 대안 설명·불확실성 정리 | **조건부** |

---

## 2. 전체 프로세스 흐름

```
[Phase 0] 에피소딕 메모리 검색
              ↓ episodic_lessons
[Phase 1] Chart Structurer ──→ Evidence (1차)
              ↓
[Phase 2] Diagnosis Agent ∥ Treatment Agent
              ↓
         Evidence (2차) ──→ Intervention Checker
              ↓
[Phase 3] Risk Factor Agent*     (* 조건 충족 시만 실질 실행)
              ↓
         Process Contributor Agent*   (* 조건 충족 시만 실질 실행)
              ↓
         Critique Eligibility       (규칙: similarity·진단 패턴 → full/standard)
              ↓
[Phase 4] Critic Agent (full 또는 standard 모드)
              ↓
         Alternative Explanation Agent*   (* 조건 충족 시만 실질 실행)
              ↓
         [END] → 최종 보고서 조합
```

---

## 3. 단계별 상세 설계

### Phase 0 (기존 유지)

| 항목 | 내용 |
|------|------|
| **에피소딕 메모리 검색** | 진단 + 임베딩 유사도로 과거 에피소드 검색 → `episodic_lessons` 생성 → Diagnosis/Treatment 프롬프트 주입 |

---

### Phase 1–2 (기존 필수 에이전트, 변경 없음)

- **Chart Structurer** → **Evidence 1차** → **Diagnosis ∥ Treatment** → **Evidence 2차** → **Intervention Checker**  
- 입력·출력·순서는 현재 `graph.py`와 동일.

---

### Phase 3: 조건부·게이트 노드

#### 3.1 Risk Factor Agent (조건부)

| 항목 | 내용 |
|------|------|
| **실행** | Intervention Checker 직후 **항상 노드 진입**. 내부에서 조건 검사. |
| **활성화 조건** | `comorbidity(동반질환) ≥ 2` **및** `cohort 사망률이 높음` (예: cohort 내 expired 비율 ≥ 임계값 또는 상대적 고위험) |
| **입력** | `patient_case`, `structured_chart`, `diagnosis_analysis`, `treatment_analysis`, `similar_cases`, `evidence` |
| **출력** | `risk_factor_analysis: Optional[Dict]` (조건 미충족 시 `None` 또는 `{"active": false}`) |
| **내용** | 동반질환·cohort 대비 위험도·고위험 시 권고사항 요약. Critic 비판점·최종 보고서에 “위험 인자” 섹션으로 병합. |

**조건 판단 예시 (규칙)**  
- `secondary_diagnoses` + `key_conditions` 개수 ≥ 2  
- cohort에서 `hospital_expire_flag==1` 또는 `status=="expired"` 비율이 설정값 이상

---

#### 3.2 Process Contributor Agent (조건부)

| 항목 | 내용 |
|------|------|
| **실행** | Risk Factor Agent 직후 **항상 노드 진입**. 내부에서 조건 검사. |
| **활성화 조건** | (1) 치료 지연·누락 키워드 포함 (`delay`, `missed`, `late`, `지연`, `누락` 등) **또는** (2) 해당 케이스 outcome이 cohort 평균과 반대 방향 (예: cohort는 대부분 생존인데 이 케이스만 사망, 또는 그 반대) |
| **입력** | `patient_case`, `structured_chart`, `similar_cases`, `evidence`, `intervention_coverage` |
| **출력** | `process_contributor_analysis: Optional[Dict]` (미충족 시 `None`) |
| **내용** | 이상치/프로세스 기여 의심 근거, 지연·불일치 요약. 보고서에 “프로세스·이상치” 섹션으로 병합. |

**조건 판단 예시**  
- 텍스트/구조화 차트에서 키워드 검색  
- `patient_case.outcome` vs `similar_cases` 내 expired 비율 비교

---

#### 3.3 Critique Eligibility (게이트, 필수 실행)

| 항목 | 내용 |
|------|------|
| **실행** | Process Contributor Agent 직후 **항상 실행**. LLM 없이 규칙만. |
| **입력** | `similar_cases`, `patient_case` (진단·패턴), `evidence` |
| **출력** | `critique_mode: "full" | "standard"` (및 선택적 `critique_eligibility_reason: str`) |
| **규칙** | • `similarity` 평균 ≥ threshold (예: 0.7) **그리고** 주요 진단 패턴 존재(진단명 비공백 등) → `full`  
  • 그 외 → `standard` |
| **사용처** | Critic 노드에서 `critique_mode` 참고: `full`이면 Verifier·풀 feedback 라운드, `standard`이면 Verifier 생략 또는 감소된 라운드. |

---

### Phase 4: Critic + 사후 조건부

#### 4.1 Critic Agent (필수, 모드만 조건부)

| 항목 | 내용 |
|------|------|
| **실행** | **항상** 실행. 기존 서브그래프(preprocess → router → run_tools → critique_builder → feedback → …) 유지. |
| **모드** | `critique_mode == "full"`일 때: Verifier 실행, feedback 라운드 기본값 유지. `"standard"`일 때: Verifier 생략 또는 feedback 1라운드만 등으로 경량화. |
| **입력** | 기존과 동일 + `risk_factor_analysis`, `process_contributor_analysis` 있으면 Critic 프롬프트/문맥에 요약으로 넣어 종합 비판에 반영. |

---

#### 4.2 Alternative Explanation Agent (조건부)

| 항목 | 내용 |
|------|------|
| **실행** | Critic 완료 후 **항상 노드 진입**. 내부에서 조건 검사. |
| **활성화 조건** | (1) 유사 케이스 similarity **분산**이 큼 **또는** (2) cohort 패턴이 일관되지 않음 (진단/결과 분포가 산만) **또는** (3) Critic 출력의 불확실성 점수가 높음 (예: confidence 낮음, 또는 “불확실”/“가능성” 등 표현 다수) |
| **입력** | `similar_cases`, `critique`, `confidence`, `evidence`, `diagnosis_analysis`, `treatment_analysis` |
| **출력** | `alternative_explanations: Optional[Dict]` (미충족 시 `None`) |
| **내용** | 해석이 애매한 부분, 대안적 설명 1~3개, 주의사항·한계. 보고서 “대안 해석·불확실성” 섹션으로 병합. |

---

## 4. 상태(State) 확장 제안

`AgentState`에 추가할 필드 예시:

```python
# 조건부 에이전트 출력
risk_factor_analysis: Optional[Dict]      # Risk Factor Agent
process_contributor_analysis: Optional[Dict]  # Process Contributor Agent
critique_mode: Optional[str]             # "full" | "standard" (Critique Eligibility)
alternative_explanations: Optional[Dict] # Alternative Explanation Agent
```

기존 `critique`, `solutions`, `confidence` 등은 그대로 두고, 최종 보고서 생성 시 위 필드를 섹션별로 합치면 됨.

---

## 5. 최종 보고서 구조 (조합 방식)

| 섹션 | 출처 |
|------|------|
| 진단·치료 요약 | Diagnosis Agent, Treatment Agent |
| 근거·문헌 | Evidence (1차·2차) |
| 시행 치료 검증 | Intervention Checker |
| **위험 인자** | Risk Factor Agent (활성화 시) |
| **프로세스·이상치** | Process Contributor Agent (활성화 시) |
| 비판점·개선안 | Critic Agent (critique, solutions) |
| **대안 해석·불확실성** | Alternative Explanation Agent (활성화 시) |
| Critic 신뢰도·모드 | confidence, critique_mode |

---

## 6. 구현 시 권장 순서

1. **Critique Eligibility** 노드 추가: 규칙만 구현, `critique_mode` state 필드 설정.  
2. **Critic 노드 수정**: `critique_mode` 읽어서 Verifier/feedback 라운드 분기.  
3. **Risk Factor Agent** 노드: 조건 판단 + LLM(또는 규칙) 분석, `risk_factor_analysis` 반환.  
4. **Process Contributor Agent** 노드: 조건 판단 + 분석, `process_contributor_analysis` 반환.  
5. **Alternative Explanation Agent** 노드: 조건 판단 + 분석, `alternative_explanations` 반환.  
6. **메인 그래프 엣지**: intervention_checker → risk_factor → process_contributor → critique_eligibility → critic → alternative_explanation → END.  
7. **리포트 생성**: 위 state 필드들을 한 보고서 템플릿에 조합.

이 순서로 적용하면 기존 에이전트와 제안 에이전트가 **필수/조건부**로 조화되며, 좋은 Critic 분석을 위한 **단일 최적 프로세스**로 확장할 수 있습니다.
