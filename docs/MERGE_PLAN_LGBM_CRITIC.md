# LGBM-clean 베이스 + LGBM Critic Agent 통합 계획

## 1. 현재 구조 요약

### 1.1 LGBM-clean (베이스)
- **오케스트레이션**: LangGraph (`MedicalCritiqueGraph`)
- **플로우**: `chart_structurer` → `evidence`(1차) → `diagnosis` / `treatment`(병렬) → `evidence_2nd` → `intervention_checker` → **`critic`** → `reflect`(조건부) → **종료** 또는 **Critic만 재실행** (reflect → critic)
- **State**: `AgentState` (TypedDict) — `patient_case`, `similar_cases`, `structured_chart`, `diagnosis_analysis`, `treatment_analysis`, `evidence`, `intervention_coverage`, `critique`, `solutions`, `iteration`, `confidence`, `memory`
- **Critic**: 단일 노드 `run_critic_agent()` — **한 번의 GPT-4o 호출**로 `critique_points` + `solutions` + `confidence` 생성. 프롬프트에 진단/치료/근거/Intervention Coverage/Evidence Quality 전부 포함.

### 1.2 LGBM (Critic Agent 쪽)
- **Critic 파트**: Pipeline 내 “Agent” 단계 = `run_agent()`
- **구성요소**:
  - **Preprocessing** (항상 실행): `timeline`, `evidence`, `record_gaps` (각각 Tool)
  - **Router**: Heuristic 또는 LLM — **어떤 도구를 돌릴지 선택** (lens_*, behavior_* 등)
  - **Tool 실행**: 선택된 도구만 예산(`max_tools`) 내에서 실행 → `lens_results`, `behavior_results` 적재
  - **CritiqueBuilder**: `preprocessing` + `lens_results` + `behavior_results` → LLM으로 **구조화된 critique** (point, span_id, severity, cohort_comparison) 생성
  - **Feedback**: critique 품질 판단 → 불충분하면 `requested_tools` + `patch_instructions` → 추가 도구 실행 후 CritiqueBuilder **재호출** (feedback_rounds)
- **추가**: `critique_engine.Verifier` — critique 기반으로 **solutions** 별도 생성 (유사 케이스 top-k 근거)

---

## 2. 통합 목표

- **베이스**: LGBM-clean 유지 (LangGraph, chart_structurer ~ intervention_checker, state, 진단/치료/2-pass evidence, reflexion).
- **Critic 구간만**: LGBM 스타일로 교체
  - **Router**로 “어떤 lens/behavior 도구를 쓸지” 결정
  - **Tool 레지스트리**로 preprocessing + lens + behavior 도구 실행
  - **CritiqueBuilder**로 구조화된 critique 생성 (span_id, cohort_comparison 등)
  - **Feedback** 루프로 품질 부족 시 추가 도구 실행 후 critique 재생성
  - (선택) **Verifier**로 solutions 생성 또는 기존 critic의 solutions 생성 방식 유지

---

## 3. 단계별 계획

### Phase 1: LGBM-clean 쪽에서 “Critic 진입 전 State” 정리

1. **LGBM-clean state에 critic용 필드 추가**
   - LGBM의 `AgentState`와 맞추기 위해, critic 노드 입력으로 쓸 수 있도록:
     - `preprocessing`: dict (timeline, evidence, record_gaps 등) — 비어 있어도 됨, LGBM 도구가 채움
     - `lens_results`: dict
     - `behavior_results`: dict
     - `router`: dict (선택된 도구, 사유)
     - `trace`: list (도구 실행 이력)
   - 기존 필드(`structured_chart`, `diagnosis_analysis`, `treatment_analysis`, `evidence`, `intervention_coverage`)는 유지.

2. **LGBM-clean state → LGBM AgentState “뷰” 매핑**
   - LGBM의 `AgentState`는 `patient`, `cohort_data`, `similar_case_patterns`, `preprocessing`, `lens_results`, `behavior_results`, `router`, `trace`.
   - LGBM-clean의 `patient_case` → `patient`, `similar_cases` → `cohort_data["similar_cases"]`, `structured_chart`/`diagnosis_analysis`/`treatment_analysis`/`evidence`는 cohort_data 또는 별도 필드로 넘길지 결정.
   - **Adapter 함수** 한 개: `clean_state_to_agent_state(clean_state) -> AgentState` (LGBM의 `AgentState` 타입).  
     역방향: `agent_state_to_clean_updates(agent_state, critique_result)` → graph state에 반영할 업데이트 dict.

---

### Phase 2: LGBM critic 컴포넌트 이식

1. **디렉터리 구조**
   - LGBM-clean 내에 `src/agents/critic/` 또는 `src/agent/` (LGBM과 동일 이름) 생성.
   - 이식할 모듈:
     - `types.py` (ToolCard, ToolSelection, AgentState — 여기서는 “critic용 내부 state”로 사용)
     - `tool_base.py`
     - `toolrag.py`
     - `registry.py` → `build_default_registry()` (도구 목록은 그대로 또는 LGBM-clean 데이터에 맞게 조정)
     - `router.py` (HeuristicRouter, LLMRouter)
     - `critique_builder.py`
     - `feedback.py`
     - `runner.py` (`run_agent`)

2. **도구(Tools)**
   - `tools/`: `preprocess_timeline`, `preprocess_evidence`, `preprocess_gaps`, `lens_severity_risk`, `lens_diagnostic_consistency`, `lens_monitoring_response`, `behavior_topk_direct_compare` 복사.
   - 각 도구의 `run(state: AgentState)`이 기대하는 입력:
     - `state.patient`: LGBM-clean의 patient_case + text 등
     - `state.cohort_data["similar_cases"]`: LGBM-clean의 similar_cases
     - `state.preprocessing`: timeline, evidence, record_gaps
   - LGBM-clean에서 이미 만드는 `evidence`(CRAG 결과)와 LGBM의 `preprocess_evidence` 출력 형식이 다를 수 있음 → **preprocess_evidence**를 “LGBM-clean evidence를 evidence_spans 형태로 변환”하는 래퍼로 바꾸거나, LGBM-clean evidence를 그대로 쓰는 **새 도구**를 둘 수 있음.

3. **CritiqueBuilder**
   - LGBM 그대로 이식하되, 입력 `state`는 LGBM `AgentState` (adapter 통해 전달).
   - 출력 형식: `critique_points`, `risk_factors`, `recommendations`, `analysis` 등 — LGBM-clean의 `critique`(리스트)와 맞추기 위해 **변환 레이어** 한 번:  
     `critique_builder_output → { "critique": [...], "solutions": [] }`  
     (solutions는 아직 비우거나, Phase 3에서 Verifier/기존 방식으로 채움)

4. **Runner + Feedback**
   - `run_agent()` 그대로 가져와서, LGBM-clean 그래프의 **한 노드**에서 호출.
   - 입력: adapter로 만든 `AgentState` + `registry` + `preprocessing_tools` + `AgentConfig`.
   - 출력: `(state, critique)` → adapter로 `critique` 및 필요 시 `solutions`를 LGBM-clean state에 반영.

---

### Phase 3: LangGraph에 “Critic 서브그래프” 연결

1. **기존 `critic` 노드 교체**
   - 현재: `_critic_node` → `run_critic_agent(state)` 직접 호출.
   - 변경: `_critic_node` 내부에서
     - `clean_state_to_agent_state(state)` 호출
     - `run_agent(state=agent_state, registry=..., preprocessing_tools=[...], config=...)` 호출
     - 반환된 `critique`를 LGBM-clean 형식으로 변환 (`critique_points` → `critique` 리스트, severity/category 등 매핑)
     - (선택) Verifier 호출해 `solutions` 생성하거나, LGBM-clean 기존처럼 CritiqueBuilder 확장으로 solutions까지 생성

2. **반복(Reflexion)과의 관계**
   - LGBM-clean: `critic` → `_should_continue` → reflect 또는 end.
   - LGBM: Feedback 루프는 “critic 노드 내부”에서만 (도구 재실행 + critique 재생성).
   - **재시작 정책 (확정)**: reflect 후 **Critic만 재실행** (chart_structurer부터 풀 리셋 아님).
     - 엣지: `reflect` → **`critic`** (기존 `reflect` → `chart_structurer` 제거 또는 “풀 리셋” 옵션으로만 유지).
     - 재실행 **근거/기준**: `confidence ≤ 0.8` 및 `iteration < max_iterations`. (선택) 품질 메트릭·Feedback 불충분 보조. → 상세는 `LGBM-clean/docs/CRITIC_RERUN_DESIGN.md` 참고.
     - **동일 결과 방지**: memory를 “반드시 개선할 critical 이슈 목록”으로 쓰고, 재실행 시(`iteration >= 2`) revision 전용 프롬프트 + 출력 제약 적용. → 상세는 `CRITIC_RERUN_DESIGN.md` 참고.

3. **confidence / iteration**
   - LGBM runner는 `confidence`를 직접 반환하지 않음.  
     LGBM-clean의 `_should_continue`를 유지하려면:
     - CritiqueBuilder/Feedback 결과에서 “품질 지표”를 하나 정해 `confidence`로 매핑하거나,
     - Feedback에서 `decision.ok == True`이면 confidence 높음으로 고정하는 식으로 단순화.

---

### Phase 4: Solutions 생성 방식 결정

- **옵션 A**: LGBM **Verifier** 이식  
  - critique만 Critic 노드에서 생성하고, solutions는 `critique_engine.Verifier`로 생성 (유사 케이스 top-k 사용).  
  - LGBM-clean의 `evidence`/similar_cases를 Verifier에 넘기면 됨.

- **옵션 B**: LGBM-clean 기존 방식 유지  
  - CritiqueBuilder 출력에 “recommendations”를 solutions와 동일한 형식으로 매핑하거나, CritiqueBuilder 프롬프트를 확장해 solutions 필드까지 생성.

- **옵션 C**: 혼합  
  - 1차: CritiqueBuilder에서 recommendations → solutions 후보.  
  - 2차: Verifier로 검증/보강 (선택).

---

### Phase 5: 의존성 및 설정

- LGBM의 `src/llm/openai_chat.py` (OpenAIChatConfig, call_openai_chat_completions, safe_json_loads) → LGBM-clean에 없으면 이식하거나, LGBM-clean의 `get_llm()`/기존 LLM 래퍼로 대체하는 **얇은 래퍼** 작성.
- `requirements.txt`: LGBM 기준으로 sklearn (toolrag), 기타 필요한 것 추가.
- 환경변수: `CARE_CRITIC_ROUTER_LLM=1` 등 LGBM과 동일하게 두면 Router가 LLM 모드로 동작.

---

### Phase 6: 스크립트 및 출력 형식

- `run_agent_critique.py`는 그대로 `MedicalCritiqueGraph.run()` 호출.
- 반환 `result`의 `critique` / `solutions` 형식은 기존 LGBM-clean 리포트와 **호환**되도록 변환 레이어에서 맞춤 (필드명: issue, severity, category, action, citation, priority 등).

---

## 4. 요약 체크리스트

| 단계 | 내용 |
|------|------|
| 1 | LGBM-clean state에 preprocessing, lens_results, behavior_results, router, trace 필드 추가 |
| 2 | clean_state ↔ LGBM AgentState adapter 함수 작성 |
| 3 | LGBM의 agent/ (types, tool_base, toolrag, registry, router, critique_builder, feedback, runner) + tools/ 이식 |
| 4 | Preprocess evidence 도구를 LGBM-clean evidence 형식과 연결 (또는 래퍼 도구) |
| 5 | LangGraph critic 노드를 run_agent() 호출 + adapter로 교체 |
| 6 | confidence/iteration 정책 결정 및 _should_continue 연동; reflect → **critic** 엣지로 Critic만 재실행 구현 |
| 6b | Critic 재실행 시 근거/기준·동일 결과 방지 반영 (CRITIC_RERUN_DESIGN.md 참고) |
| 7 | Solutions: Verifier 이식 vs CritiqueBuilder 확장 결정 후 구현 |
| 8 | LLM 호출 레이어 통일 (openai_chat 이식 또는 래퍼) |
| 9 | requirements 및 환경변수 정리 |
| 10 | run_agent_critique.py 출력 형식 호환 유지 |

---

## 5. 리스크 및 주의사항

- **Evidence 형식**: LGBM-clean의 2-pass CRAG `evidence`와 LGBM의 `preprocess_evidence`(evidence_spans) 구조가 다름.  
  → 초기에는 LGBM preprocessing을 “차트/텍스트에서 추출”하는 쪽으로 두고, LGBM-clean evidence는 CritiqueBuilder에 “추가 컨텍스트”로만 넣는 방식이 구현 부담이 적음.  
  → 이후에 evidence_spans를 LGBM-clean evidence로부터 생성하는 공통 레이어를 두면 일원화 가능.

- **Patient 필드명**: LGBM은 `patient.text`, LGBM-clean은 `patient_case.clinical_text` 등 차이 있을 수 있음. Adapter에서 필드 매핑 명시.

- **Reflexion 메모리**: LGBM에는 없음. LGBM-clean의 `memory`를 adapter에서 `AgentState`에 넣어줄지, 아니면 CritiqueBuilder 프롬프트에만 별도로 주입할지 결정.

- **재시작 지점**: 구현 선택은 **Critic만 재실행** (reflect → critic). Evidence 2nd부터 재시작·chart_structurer 풀 리셋은 확장 옵션으로 둠. 재실행 근거/기준·동일 결과 방지는 `LGBM-clean/docs/CRITIC_RERUN_DESIGN.md`에 정리됨.

이 계획대로 진행하면 LGBM-clean의 전체 플로우는 유지하면서, critic 구간만 LGBM의 Router + Tools + CritiqueBuilder + Feedback 구조로 교체할 수 있습니다.
