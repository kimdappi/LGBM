# LangGraph 전체 프로세스 및 에이전트/도구 목록

## 1. 활용되는 모든 에이전트(Agent) · 한 줄 요약

### 메인 그래프 — 에이전트 (노드 단위)

| 구분 | 이름 | 한 줄 요약 |
|------|------|------------|
| 필수 | **chart_structurer** | 차트 원문을 구조화 JSON(타임라인·소견·시술·결과 등)으로 추출한다. |
| 필수 | **evidence** | 1차 Evidence: RAG·PubMed로 유사 케이스·문헌 검색 후 evidence를 채운다. |
| 필수 | **diagnosis** | 진단 적절성·누락·감별·결정적 오류·논리 결함을 분석해 diagnosis_analysis를 낸다. |
| 필수 | **treatment** | 치료 적절성·퇴원 결정·위험도를 분석해 treatment_analysis를 낸다. |
| 필수 | **evidence_2nd** | 비판 기반 2차 검색: Diagnosis/Treatment 비판으로 타겟 쿼리를 만들어 evidence를 보강한다. |
| 필수 | **intervention_checker** | 구조화 차트에서 이미 시행된 치료를 확인해, “치료 부재” 비판을 “적절성 평가”로 정리한다. |
| 필수 | **agent_router** | LLM이 에이전트 문서와 state를 보고 실행할 에이전트 목록(selected_agents)을 선택한다. |
| 조건 | **risk_factor** | 고위험 케이스일 때 위험 인자·동반질환을 집중 분석해 risk_factor_analysis를 낸다. (router 선택 시만) |
| 조건 | **process_contributor** | 이상치·프로세스 기여 의심 시 지연·불일치를 분석해 process_contributor_analysis를 낸다. (router 선택 시만) |
| 필수 | **critic** | Critic 서브그래프를 한 번 돌려 전처리·도구·CritiqueBuilder로 비판(critique)을 만들고, 유사 케이스 있으면 Verifier로 solutions를 붙인다. |
| 필수 | **run_alternative_explanation** | Critic 이후 항상 대안 설명·불확실성·주의사항을 정리해 alternative_explanations를 낸다. |
| 조건 | **Verifier** | critic 노드 내부: 유사 케이스가 있을 때만 비판점에 대해 유사 케이스 기반 solutions(근거)를 붙인다. |

---

## 2. Critic 서브그래프 — 도구(Tool) · 한 줄 요약

| 구분 | 이름 | 한 줄 요약 |
|------|------|------------|
| 필수 | **timeline** | 차트에서 시간순 이벤트·타임라인을 뽑아 preprocessing에 넣는다. |
| 필수 | **evidence** | 환자 텍스트에서 증거 구간(evidence spans)을 추출해 preprocessing에 넣는다. |
| 필수 | **record_gaps** | 기록 공백·누락 구간을 찾아 preprocessing에 넣는다. |
| 조건 | **behavior_topk_direct_compare** | 유사 케이스와의 Top-K 직접 비교로 행동/결과 차이를 분석한다. (preprocess: 유사 케이스·예산 있을 때; run_tools: router 선택 시) |
| 조건 | **lens_severity_risk** | 중증도·위험도 관점으로 소견을 해석한다. (router가 선택한 경우만 run_tools에서 실행) |
| 조건 | **lens_diagnostic_consistency** | 진단·소견 간 일관성을 검토한다. (router가 선택한 경우만 run_tools에서 실행) |
| 조건 | **lens_monitoring_response** | 모니터링·대응 적절성을 검토한다. (router가 선택한 경우만 run_tools에서 실행) |

※ Critic 내 **router**는 위 조건부 도구들 중 어떤 것을 쓸지 selected_tools로 정하고, **run_tools**가 그 목록만 예산 안에서 실행한다. **critique_builder**는 항상 실행되어 전처리·도구 결과를 묶어 비판을 생성한다.

---

## 2.1 Critic 이전 결과 → Critic 전달·활용 검증

**전달 경로:** 메인 그래프 state → `clean_state_to_agent_state()` (adapter) → `cohort_data`에 앞단 결과를 모두 넣음 → Critic 서브그래프 `initial_critic_dict["cohort_data"]`로 전달.

| Critic 이전 결과물 (메인 state 키) | Critic에서 전달 위치 | Critic 내 활용처 |
|----------------------------------|----------------------|------------------|
| **structured_chart** | cohort_data.structured_chart | **preprocess 노드**: 있으면 timeline/evidence 도구를 스킵하고, `_structured_chart_to_timeline_and_evidence()`로 procedures_performed, clinical_course, outcome, evidence_spans → preprocessing.timeline / preprocessing.evidence 생성. |
| **similar_cases** | cohort_data.similar_cases | **preprocess**: behavior_topk_direct_compare 도구 입력. **메인 critic 노드**: Verifier 호출 시 비판점 + 유사 케이스로 solutions 생성. |
| **diagnosis_analysis** | cohort_data.diagnosis_analysis | **CritiqueBuilder** LLM payload의 `reference_only_prior_results.diagnosis_analysis` (참고용). **lens_diagnostic_consistency**: `reference_diagnosis_analysis` / `reference_only_prior_diagnosis_analysis` (참고용). **behavior_topk_direct_compare**: `reference_diagnosis_analysis` / `reference_only_prior_diagnosis_analysis` (참고용). |
| **treatment_analysis** | cohort_data.treatment_analysis | **CritiqueBuilder** `reference_only_prior_results.treatment_analysis`. **behavior_topk_direct_compare** `reference_only_prior_treatment_analysis`. |
| **evidence** | cohort_data.evidence | **CritiqueBuilder** `reference_only_prior_results.evidence`. **behavior_topk_direct_compare** `reference_only_prior_evidence`. |
| **intervention_coverage** | cohort_data.intervention_coverage | **CritiqueBuilder** `reference_only_prior_results.intervention_coverage`. |
| **risk_factor_analysis** | cohort_data.risk_factor_analysis | **lens_severity_risk**: `reference_risk_factor_analysis` (참고용). |
| **process_contributor_analysis** | cohort_data.process_contributor_analysis | **lens_monitoring_response**: `reference_process_contributor_analysis` / `reference_only_prior_process_contributor_analysis` (참고용). |

**결론:** Critic 이전의 비판 근거·문서 분석 결과(구조화 차트, 진단/치료 분석, evidence, intervention_coverage, risk_factor, process_contributor)는 모두 **adapter를 통해 cohort_data로 Critic에 전달**되며, **Critic 서브그래프의 preprocess·router·run_tools·CritiqueBuilder 및 메인 critic 노드의 Verifier**에서 **참고용(reference_only)** 또는 **직접 입력**으로 활용된다. (프롬프트에는 “reference만 사용, 의존 금지”로 되어 있어 앞단이 없어도 Critic은 동작한다.)

---

## 2.2 Critic 내부의 timeline / evidence는 어떻게 채워지나?

Critic의 **preprocess** 노드에서 `preprocessing["timeline"]`과 `preprocessing["evidence"]`를 채우는 방식은 **두 경로** 중 하나다.

| 조건 | timeline | evidence |
|------|----------|----------|
| **cohort_data.structured_chart 있음** (chart_structurer 결과가 넘어온 경우) | **도구 미실행.** `_structured_chart_to_timeline_and_evidence(structured_chart)`로 **변환만** 수행. → `procedures_performed`, `clinical_course.events`, `outcome.critical_events_leading_to_outcome`를 events 리스트로 만듦. | **도구 미실행.** 같은 함수에서 `structured_chart.evidence_spans` 리스트를 `E1`, `E2`, … 키의 `evidence_spans` 딕셔너리로 변환. |
| **structured_chart 없음** | **PreprocessTimelineTool** 실행. `patient.text`(원문)를 문장 스팬으로 쪼갠 뒤, 각 문장을 event로 태깅(event_id, type, time_hint, text, start, end). | **PreprocessEvidenceTool** 실행. `patient.text`를 문장 단위로 나누고, claimable한 문장만 골라 `evidence_spans` (E1, E2, …; category, quote, start, end) 생성. |

- **정리:** 앞단에서 **구조화 차트가 오면** Critic은 timeline/evidence **도구를 돌리지 않고** 차트 필드만 Critic용 형식으로 변환해 쓴다. **구조화 차트가 없으면** Critic이 **환자 원문(patient.text)**만 보고 두 도구를 실행해 같은 구조를 스스로 만든다.
- **record_gaps**는 두 경로 공통으로 항상 실행된다. 이후 lens/behavior 도구와 **CritiqueBuilder**는 이 `preprocessing.timeline`·`preprocessing.evidence`를 입력으로 사용한다.

---

## 3. LangGraph 형식 전체 프로세스

### 3.1 메인 그래프 (MedicalCritiqueGraph)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     메인 그래프 (AgentState)              │
                    └─────────────────────────────────────────────────────────┘

[START]
    │
    ▼
┌─────────────────────┐
│ chart_structurer     │  차트 → 구조화 JSON
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ evidence            │  1차 RAG·PubMed 검색
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│diagnosis│  │treatment│  진단·치료 분석 (병렬)
└────┬────┘  └────┬────┘
     └─────┬─────┘
           ▼
┌─────────────────────┐
│ evidence_2nd        │  비판 기반 2차 검색
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ intervention_checker│  시행 치료 확인
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ agent_router        │  selected_agents 선택 (LLM)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ run_conditional_    │  risk_factor / process_contributor (선택 시만)
│ agents              │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ critic              │  서브그래프 invoke + (유사 케이스 있으면 Verifier)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ run_alternative_    │  대안 해석·불확실성 (항상)
│ explanation         │
└──────────┬──────────┘
           ▼
        [END]
```

- 모든 노드 간은 **선형 엣지** (한 방향으로만 진행).
- **분기**는 노드 **내부**에서만: `run_conditional_agents`(risk_factor/process_contributor), `critic`(Verifier는 유사 케이스 있을 때만 실행).

---

### 3.2 Critic 서브그래프 (get_critic_graph)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              Critic 서브그래프 (CriticGraphState)         │
                    └─────────────────────────────────────────────────────────┘

[START]
    │
    ▼
┌─────────────────────┐
│ preprocess          │  timeline, evidence, record_gaps (항상)
│                     │  + behavior_topk_direct_compare (유사 케이스·예산 있을 때)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ router              │  selected_tools 선택 (LLM 또는 휴리스틱)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ run_tools           │  selected_tools 안의 도구만 예산 내 실행
│                     │  (lens_severity_risk, lens_diagnostic_consistency,
│                     │   lens_monitoring_response, behavior_topk_direct_compare 등)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ critique_builder    │  전처리·도구 결과 종합 → critique 생성 (LLM)
└──────────┬──────────┘
           ▼
        [END]
```

- 메인 그래프의 **critic** 노드가 이 서브그래프를 **한 번** invoke한 뒤, 나온 critique(및 유사 케이스 있을 때 Verifier의 solutions)를 메인 state에 반영한다.

---

## 4. 한눈에 보는 필수 vs 조건

| 레벨 | 필수 (항상 활용) | 조건 (선택/조건 시만 활용) |
|------|------------------|----------------------------|
| **메인 에이전트** | chart_structurer, evidence, diagnosis, treatment, evidence_2nd, intervention_checker, agent_router, run_conditional_agents(노드), critic, run_alternative_explanation | risk_factor, process_contributor, Verifier(유사 케이스 있을 때) |
| **Critic 도구** | timeline, evidence, record_gaps, router(노드), run_tools(노드), critique_builder | behavior_topk_direct_compare, lens_severity_risk, lens_diagnostic_consistency, lens_monitoring_response |
