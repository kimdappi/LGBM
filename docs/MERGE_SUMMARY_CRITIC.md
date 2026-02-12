# LGBM-clean + LGBM Critic 통합 요약

Phase 1~6 완료 후 **무엇이 달라졌는지**, **어떻게 융합되어 있는지**, **전체 구조**를 정리한 문서입니다.

---

## 1. Phase별로 무엇이 달라졌는지

### Phase 1: State와 Adapter

| 변경 | 내용 |
|------|------|
| **state.py** | 그래프 상태에 Critic용 필드 추가: `preprocessing`, `lens_results`, `behavior_results`, `router`, `trace`, `similar_case_patterns`. 기존 필드(patient_case, structured_chart, diagnosis/treatment_analysis, evidence, intervention_coverage, critique, solutions, iteration, confidence, memory) 유지. |
| **critic_adapter.py** | **그래프 state ↔ Critic 전용 state** 변환: `clean_state_to_agent_state(clean_state)` → LGBM 형식의 `CriticAgentState`(patient, cohort_data, preprocessing 등). 역방향: `agent_state_to_clean_updates(critic_state, critique_result)` → 그래프에 넣을 업데이트 dict. |

→ **그래프 한 개의 state**로 흐르면서, Critic 구간만 **LGBM이 기대하는 형태**로 변환해 쓰도록 했음.

---

### Phase 2: Critic 엔진 이식

| 변경 | 내용 |
|------|------|
| **src/critic_agent/** | LGBM의 agent 로직 이식: `types`, `tool_base`, `toolrag`, `registry`, `router`, `critique_builder`, `feedback`, `runner`. |
| **src/critic_agent/tools/** | 전처리 3개(timeline, evidence, record_gaps), lens 3개(severity_risk, diagnostic_consistency, monitoring_response), behavior 1개(behavior_topk_direct_compare). |
| **src/llm/** | `openai_chat.py` 이식 (OpenAIChatConfig, call_openai_chat_completions, safe_json_loads). Critic·Verifier·도구에서 사용. |

→ **Critic은 “한 번의 GPT 호출”이 아니라**, 전처리 → Router → 도구 실행 → CritiqueBuilder → Feedback 루프까지 **LGBM과 동일한 파이프라인**으로 동작.

---

### Phase 3: 그래프에 Critic 연결

| 변경 | 내용 |
|------|------|
| **critic 노드** | 기존 `run_critic_agent()` 제거. `clean_state_to_agent_state(state)` → `run_agent(...)` → `agent_state_to_clean_updates(...)` 로 **LGBM run_agent 전체**를 한 노드에서 실행. |
| **반복 정책** | reflect 후 **풀 리셋이 아니라 Critic만 재실행**: 엣지 `reflect` → `critic`. |
| **confidence / iteration** | Feedback의 `ok` → confidence 0.8/0.5, iteration은 매 critic 실행 시 +1. `_should_continue`에서 iteration ≥ max 또는 confidence > 0.8이면 END. |

→ **오케스트레이션은 LangGraph**, **비판 생성은 LGBM 스타일 Agent** 한 덩어리로 붙음.

---

### Phase 4: Solutions

| 변경 | 내용 |
|------|------|
| **normalize_solutions()** | CritiqueBuilder의 `recommendations`(문자열 리스트)와 Verifier의 `solution/evidence/priority` 형식을 **리포트 형식** `{ action, citation, priority }` 로 통일. |
| **Verifier 이식** | `src/critic_agent/verifier.py`. critique + 유사 케이스 top-k → 유사 케이스 근거 solutions. `openai_chat` 사용. |
| **critic 노드 내 선택 호출** | 유사 케이스가 있고 API 키가 있으면 Verifier 호출 후 solutions를 그 결과로 교체; 실패 시 recommendations 기반 solutions 유지. |

→ **solutions**는 항상 **동일한 스키마**로 그래프/리포트에 전달됨.

---

### Phase 5: 의존성·설정

| 변경 | 내용 |
|------|------|
| **requirements.txt** | `requests` 추가, Critic/LLM 관련 주석. |
| **환경변수** | README에 `CARE_CRITIC_ROUTER_LLM=1` 설명 (1이면 도구 선택에 LLM 라우터, 미설정 시 휴리스틱). |
| **src/llm/__init__.py** | openai_chat(Critic용) vs get_llm(그래프 노드용) 역할 docstring. |

→ **실행 환경·설정**이 문서와 코드에 맞춰짐.

---

### Phase 6: 출력 형식 호환

| 변경 | 내용 |
|------|------|
| **critique 정규화** | `point` → `issue`, `severity` "high" → "critical", `category` 기본값 "process". |
| **initial_state** | `confidence: None` 추가로 AgentState 정의와 일치. |

→ **run_agent_critique.py**가 기대하는 `issue`, `severity`(critical/medium/low), `category`, solutions의 `action`/`citation`/`priority`와 **그대로 호환**.

---

## 2. 어떻게 융합되어 있는지 (두 세계의 연결)

- **세계 1: LangGraph (LGBM-clean)**  
  - State: `AgentState` (patient_case, similar_cases, structured_chart, diagnosis/treatment_analysis, evidence, intervention_coverage + **critic용 필드**).  
  - 노드: chart_structurer → evidence → diagnosis/treatment → evidence_2nd → intervention_checker → **critic** → reflect(조건부) → END 또는 critic.

- **세계 2: Critic Agent (LGBM 스타일)**  
  - State: `CriticAgentState` / `AgentState`(types.py): patient, cohort_data, preprocessing, lens_results, behavior_results, router, trace.  
  - 파이프라인: 전처리(항상) → Router(휴리스틱 또는 LLM) → 도구 실행(예산 내) → CritiqueBuilder → Feedback(품질 판단, 필요 시 도구 추가 실행 후 재빌드).

**연결점**

1. **진입**: critic 노드에서 `clean_state_to_agent_state(state)`로 **그래프 state → Critic state** 변환 (patient_case→patient, similar_cases→cohort_data 등).
2. **실행**: `run_agent(state=critic_state, registry, preprocessing_tools, config)` 한 번 호출로 **전체 Critic 파이프라인** 수행. 결과는 `(state, critique)`.
3. **복귀**: `agent_state_to_clean_updates(critic_state, critique_result)`로 **critique, solutions, preprocessing, lens_results, behavior_results, router, trace**를 그래프에 넣을 dict로 변환.
4. **추가 처리**: 그래프 쪽에서 Verifier(선택), point→issue / high→critical / category, confidence·iteration 설정 후 **한 번에 merge** → 다음 노드(reflect 또는 END)로 이어짐.

그래프 state는 **한 종류**만 유지하고, Critic 구간만 **adapter를 통해 LGBM 형식으로 바꿨다가 다시 그래프 형식으로 되돌리는** 구조라서, **데이터가 한쪽 형식에 묶이지 않고** 깔끔하게 융합되어 있습니다.

---

## 3. 전체 구조 (아키텍처)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MedicalCritiqueGraph (LangGraph)                                            │
│  진입: run(patient_case, similar_cases)  →  initial_state                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [LGBM-clean 파이프라인]                                                     │
│  chart_structurer → evidence(1차) → diagnosis + treatment(병렬)              │
│       → evidence_2nd(비판 기반) → intervention_checker                       │
│       → critic ─────────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  critic 노드                      │
                    │  1. clean_state_to_agent_state()  │
                    │  2. run_agent(...)  ←─────────────┼── [LGBM Critic Agent]
                    │  3. agent_state_to_clean_updates()│
                    │  4. (선택) Verifier → solutions     │
                    │  5. critique 정규화(issue, severity, category)          │
                    │  6. confidence, iteration 설정     │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [LGBM Critic Agent - run_agent 내부]                                        │
│  • Preprocessing: timeline, evidence, record_gaps (항상)                     │
│  • (선택) behavior_topk_direct_compare (유사 케이스 있으면)                   │
│  • Router: Heuristic 또는 LLM(CARE_CRITIC_ROUTER_LLM=1) → lens_* / behavior_* │
│  • 도구 실행 (max_tools 예산) → lens_results, behavior_results               │
│  • CritiqueBuilder(LLM) → critique_points, risk_factors, recommendations     │
│  • Feedback: OK이면 종료, 아니면 requested_tools 실행 후 CritiqueBuilder 재호출│
│  • 반환: (state, critique)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  critic 노드 반환값이 graph state에 merge                                  │
│  → 조건부: _should_continue → "reflect" 또는 "end"                          │
│  → reflect → memory에 critical 이슈 저장 → critic (Critic만 재실행)         │
│  → end → END                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**디렉터리 역할**

| 경로 | 역할 |
|------|------|
| **src/agents/** | LangGraph 정의(graph.py), 그래프 state(state.py), 노드(nodes/), **그래프↔Critic 변환(critic_adapter.py)**. |
| **src/critic_agent/** | LGBM 스타일 Critic 전체: runner, router, registry, critique_builder, feedback, **tools/** (전처리·lens·behavior), **verifier**. |
| **src/llm/** | Critic/Verifier/도구용 **openai_chat**. (그래프 노드용 get_llm은 src/agents/llm.py.) |
| **scripts/run_agent_critique.py** | `MedicalCritiqueGraph.run()` 호출, 결과를 critique/solutions 기준으로 출력·저장. |

---

## 4. 데이터 흐름 요약

1. **입력**: `patient_case`, `similar_cases` → initial_state.
2. **그래프**: 각 노드가 state를 읽고, 반환 dict만 merge (LangGraph 기본).
3. **critic 노드**:
   - 그래프 state → **adapter** → Critic state → **run_agent** → (전처리, 라우팅, 도구, CritiqueBuilder, Feedback).
   - 나온 critique_result + 갱신된 critic_state → **adapter** → 그래프용 업데이트.
   - (선택) Verifier로 solutions 보강.
   - critique 필드 정규화(issue, severity, category), confidence, iteration 설정 → merge.
4. **반복**: reflect 시 **critic만** 다시 실행(이전 그래프 state 유지, memory 추가).
5. **출력**: `run()` 반환값의 `critique`, `solutions` 등이 **run_agent_critique.py**와 **리포트 형식**과 호환.

이렇게 **Phase 1~6**이 반영된 상태에서, **어떤 부분이 달라졌는지**, **어떻게 한 시스템으로 융합되었는지**, **전체 구조**가 위와 같습니다.
