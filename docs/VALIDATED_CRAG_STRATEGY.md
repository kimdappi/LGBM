# 하이브리드 검증 기반 CRAG 전략 (Hybrid Validated CRAG)

**최종 업데이트:** 2026-02-04

---

## 📋 개요

기존 표준 CRAG는 "내부 유사도 >= 0.7 & >= 2개"만 확인했습니다.  
하지만 **유사도가 높아도 M&M 비판/해결에 무의미한 케이스**가 포함될 수 있었습니다.

**하이브리드 검증 기반 CRAG**는:
1. **내부 케이스 >= 1개**만 있어도 LLM 검증 수행
2. 검증 통과 시 **내부 + 외부 모두 사용** (하이브리드)
3. 검증 실패 시 외부만 사용

즉, **1개만 있어도 유용하면 사용하되, 외부 근거로 보강**합니다.

---

## ❌ 기존 문제

### Case Study: PE (Pulmonary Embolism) 케이스

**입력 환자:**
- 3주 전 무릎 수술
- 갑작스러운 흉막성 흉통
- 저산소증 + 빈맥
- 우측 종아리 비대 + 압통 (DVT sign)

**FAISS 검색 결과:**
1. Case A: Crohn disease (similarity: 0.78)
2. Case B: H. pylori gastritis (similarity: 0.76)
3. Case C: Feline virus (similarity: 0.74)

**문제:**
- 모두 유사도 >= 0.7 ✅
- 모두 >= 2개 ✅
- **하지만 PE 진단 실패를 비판하는데 전혀 도움 안됨 ❌**

---

## ✅ 해결 방법: 검증 기반 CRAG

### 핵심 아이디어

**"내부 케이스가 충분하다고 판단되면, LLM으로 한번 더 검증"**

- ✅ 검증 통과 → 내부만 사용 (믿을 수 있음)
- ❌ 검증 실패 → 외부(PubMed)만 사용 (내부는 무의미)

---

## 🔄 전체 Flow

```
┌─────────────────────────────────────────────────────────┐
│ Evidence Agent: 하이브리드 검증 기반 CRAG               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Step 1: LLM으로 임상 맥락 분석                          │
│   → 주요 진단, 임상 우선순위, 위험 패턴 추출            │
│                                                         │
│ Step 2: 내부 RAG 검색 (similarity >= 0.7 필터링)        │
│   → FAISS + Reranker로 top-3 케이스 추출                │
│                                                         │
│ Step 3: 품질 평가 (유효 케이스 >= 1개?)                 │
│   ├─ NO (0개) → 외부(PubMed)만 사용 (바로 종료)        │
│   └─ YES (1개 이상) → Step 4로 진행                     │
│                                                         │
│ Step 4: LLM 검증 (비판/해결에 유용한가?)                │
│   ├─ 통과 → 내부 + 외부 모두 사용 ✅ (하이브리드)      │
│   │   - 검증된 내부 케이스 사용                         │
│   │   - PubMed도 검색하여 외부 근거 추가                │
│   └─ 실패 → 외부(PubMed)만 사용 ❌                     │
│       - 내부 케이스 전부 폐기                           │
│       - PubMed에서만 검색                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 LLM 검증 기준

### 프롬프트 포커스

```
Task: Evaluate if these similar cases are USEFUL for M&M CRITIQUE and SOLUTION generation.

CRITICAL: These cases will be used to:
- Find DIAGNOSTIC ERRORS (missed diagnoses, delayed diagnosis)
- Find TREATMENT ERRORS (wrong medication, inappropriate intervention)
- Generate EVIDENCE-BASED SOLUTIONS (how to prevent, what to do better)

Consider:
1. Are diagnoses/conditions truly similar (not just superficial text match)?
2. Do cases contain ACTIONABLE LESSONS for critique?
   - Complications, adverse events, errors
   - Management decisions (good or bad)
   - Outcomes (expired, improved, complications)
3. Are there any MISLEADING or IRRELEVANT cases that would dilute critique quality?
```

### 검증 통과 조건

1. **is_valid = true**
2. **confidence >= 0.5**
3. **적어도 1개 이상의 유효 케이스 존재**

### 검증 실패 조건

다음 중 하나라도 해당:
- 케이스가 증상만 유사하고 진단/치료 맥락이 다름
- 비판점을 찾는데 도움이 안됨
- 해결책을 도출하는데 기여하지 못함
- LLM confidence < 0.5

---

## 📊 시나리오별 동작

### Scenario 1: 내부 없음 (0개) → 바로 외부

```
[Step 2] Internal search
  → Found 0 cases (all similarity < 0.7)

[Step 3] Quality evaluation
  → 0 < 1 (insufficient)

[CRAG] Internal insufficient → EXTERNAL ONLY

[External] PubMed search
  → Found 5 articles

Result:
  - Retrieval mode: external_only
  - Internal: 0 cases
  - External: 5 articles
```

---

### Scenario 2: 내부 1개 + 검증 통과 → 하이브리드

```
[Step 2] Internal search
  → Found 1 case (similarity: 0.75)

[Step 3] Quality evaluation
  → 1 >= 1 (sufficient)

[Step 4] LLM validation
  → ✅ PASSED (confidence: 0.80)
  → Valid cases: 1/1
  → Reason: "Case provides relevant PE diagnostic lessons"

[CRAG] Internal validated → HYBRID (internal + external)

[Internal] 1 case (validated)
[External] PubMed search
  → Found 5 articles

Result:
  - Retrieval mode: hybrid
  - Internal: 1 case (validated)
  - External: 5 articles
  - Total: 6 sources
```

---

### Scenario 3: 내부 3개 + 검증 통과 → 하이브리드

```
[Step 2] Internal search
  → Found 3 cases (similarity >= 0.7)

[Step 3] Quality evaluation
  → 3 >= 1 (sufficient)

[Step 4] LLM validation
  → ✅ PASSED (confidence: 0.85)
  → Valid cases: 2/3
  → Reason: "Cases provide relevant diagnostic error lessons"

[CRAG] Internal validated → HYBRID (internal + external)

[Internal] 2 cases (validated)
[External] PubMed search
  → Found 5 articles

Result:
  - Retrieval mode: hybrid
  - Internal: 2 cases (validated)
  - External: 5 articles
  - Total: 7 sources
```

---

### Scenario 4: 내부 있음 + 검증 실패 → 외부만

```
[Step 2] Internal search
  → Found 3 cases (similarity >= 0.7)
    - Crohn disease (0.78)
    - H. pylori (0.76)
    - Feline virus (0.74)

[Step 3] Quality evaluation
  → 3 >= 1 (sufficient)

[Step 4] LLM validation
  → ❌ FAILED
  → Reason: "Cases are about GI diseases, not relevant to PE diagnosis"
  → Confidence: 0.25

[CRAG] Internal not useful for critique → EXTERNAL ONLY

[External] PubMed search
  → Query: "pulmonary embolism diagnostic error post-operative..."
  → Found 5 articles

Result:
  - Retrieval mode: external_only_after_validation
  - Internal: 0
  - External: 5 articles
```

---

## 🎯 기대 효과

### 1. 비판 품질 향상

**Before (검증 없음):**
```
Internal cases:
  - Crohn disease (0.78)
  - H. pylori (0.76)

Critique:
  - "Crohn disease 감별 필요"
  - "H. pylori 검사 권고"
  → ❌ 전혀 도움 안됨 (PE 진단 실패를 놓침)
```

**After (하이브리드 검증):**
```
LLM validation: FAILED (Crohn/H. pylori are not relevant to PE)
  → Switch to external only

External (PubMed):
  - "PE diagnosis after orthopedic surgery"
  - "Wells score for PE risk stratification"
  - "CTPA in suspected PE"

Critique:
  - "고위험 PE 감별 실패 (Wells score 미사용)"
  - "CTPA 미시행은 중대한 진단 누락"
  → ✅ 정확하고 유용한 비판
```

**또는 (검증 통과 시):**
```
Internal: 1 case (similar PE case after surgery)
LLM validation: PASSED
  → Use internal + external (hybrid)

Internal:
  - "Similar PE case: delayed diagnosis after orthopedic surgery"

External (PubMed):
  - "PE diagnosis after orthopedic surgery"
  - "Wells score for PE risk stratification"

Critique:
  - "유사 케이스에서도 동일한 진단 지연 발생"
  - "Wells score 사용 권고 (문헌 근거)"
  - "CTPA 미시행은 중대한 진단 누락 (외부 가이드라인)"
  → ✅ 내부 + 외부 근거로 강화된 비판
```

### 2. 근거 신뢰성 증가

- 내부 케이스를 사용할 때: LLM이 검증한 신뢰할 수 있는 케이스만 사용
- 외부 문헌도 함께 사용: M&M 목적(오류, 합병증, 예방)에 맞춰 검색
- **하이브리드 전략**: 내부(유사 케이스) + 외부(최신 문헌) 병행으로 비판 품질 극대화

### 3. 1개만 있어도 활용

- 기존: >= 2개 필요
- 현재: >= 1개만 있어도 검증 후 사용 가능
- **유연성 증가**: 희귀 질환이나 특수 상황에서도 내부 케이스 활용 가능

### 4. 외부 근거로 보강

- 내부 케이스가 검증 통과 → 외부도 검색하여 보강 (품질 최우선)
- 내부 케이스가 무의미 → 외부만 검색 (품질 보장)

---

## 🔧 구현 위치

### 코드: `src/agents/evidence_agent.py`

```python
def run_evidence_agent(state, rag_retriever, similarity_threshold=0.7):
    """하이브리드 검증 기반 CRAG"""
    
    # Step 1: Clinical analysis
    clinical_analysis = analyze_clinical_context_with_llm(patient, structured_chart)
    
    # Step 2: Internal search
    internal_results = search_internal_rag(query, rag_retriever, top_k=3)
    internal_results = [r for r in internal_results if r['score'] >= 0.7]
    
    # Step 3: Quality evaluation (>= 1개?)
    quality = evaluate_internal_quality(internal_results, threshold=0.7)
    
    if not quality["is_sufficient"]:
        # Case 1: 내부 없음 (0개) → 바로 외부만
        return external_search_only()
    
    # Step 4: LLM validation
    validation = validate_internal_evidence_with_llm(internal_results, patient)
    
    if validation["is_valid"]:
        # Case 2: 검증 통과 → 내부 + 외부 (하이브리드)
        external_results = search_pubmed(query)
        return {
            "retrieval_mode": "hybrid",
            "internal": validation["filtered_results"],
            "external": external_results
        }
    else:
        # Case 3: 검증 실패 → 외부만
        return external_search_only()
```

**핵심 변경:**
- `evaluate_internal_quality`: `>= 2` → `>= 1`로 완화
- 검증 통과 시: `internal_only` → `hybrid` (내부 + 외부)
- 외부 검색이 조건부가 아닌, 검증 통과시 항상 수행

### 검증 함수: `validate_internal_evidence_with_llm()`

- **입력:** 내부 케이스 리스트 + 환자 데이터
- **LLM 모델:** GPT-4o
- **출력:** `is_valid`, `reason`, `confidence`, `filtered_results`

---

## 📚 관련 문서

- `README.md`: 전체 시스템 아키텍처
- `docs/MM_FOCUSED_SEARCH.md`: M&M 목적 검색 전략
- `docs/QUERY_PRIORITY_FIX.md`: 진단 우선순위 쿼리 생성

---

## 🎓 학습 포인트

1. **유사도 ≠ 유용성**
   - FAISS 유사도가 높아도 비판/해결에 무의미할 수 있음
   - LLM 검증으로 실제 유용성 확인 필수

2. **M&M 목적 명확화**
   - 일반적인 유사 케이스 찾기 ❌
   - 비판점/해결책 도출에 유용한 케이스 찾기 ✅

3. **1개만 있어도 활용**
   - 기존: 최소 2개 필요
   - 현재: 1개만 있어도 유용하면 사용
   - 희귀 질환/특수 상황에서도 활용 가능

4. **하이브리드 전략**
   - 내부가 유용하면 → 외부도 검색 (보강)
   - 내부가 무의미하면 → 외부만 사용 (품질 보장)
   - **내부 + 외부 = 최대 품질**

---

**하이브리드 검증 기반 CRAG = 유연함 + 품질 보장 + 외부 보강**
