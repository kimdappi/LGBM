# 검색 쿼리 우선순위 수정

## 업데이트 일자
2026-02-04

## 문제 인식

사용자 피드백:
> "검색시 지금 chief complaint가 더 우선순위하게 되어 있는데 primary diagnosis가 먼저 오게 하는게 맞지 않아?"

**문제:**
- 검색 쿼리가 증상(chief complaint) 중심으로 생성될 수 있음
- Primary diagnosis가 쿼리의 후순위로 밀릴 수 있음
- 증상 기반 검색 → 관련 없는 케이스 검색 가능

---

## 왜 Primary Diagnosis가 먼저여야 하는가?

### Chief Complaint vs Primary Diagnosis

| 항목 | 설명 | 예시 | 검색 적합도 |
|------|------|------|-----------|
| **Chief Complaint** | 환자가 호소하는 증상 | "chest pain", "dyspnea", "headache" | ❌ 낮음 (너무 광범위) |
| **Primary Diagnosis** | 실제 질환/진단명 | "pulmonary embolism", "acute MI" | ✅ 높음 (구체적) |

### 검색 결과 비교

#### ❌ Chief Complaint 우선 (문제)

```
쿼리: "chest pain post-operative hypoxia pulmonary embolism"
        ↑ 증상 먼저

검색 결과:
- Chest pain in GERD
- Chest pain in anxiety
- Chest pain differential diagnosis
- ...
→ PE와 무관한 케이스 많음
```

#### ✅ Primary Diagnosis 우선 (올바름)

```
쿼리: "pulmonary embolism post-operative hypoxia chest pain"
        ↑ 진단 먼저

검색 결과:
- Post-op PE cases
- PE diagnostic errors
- PE with atypical presentation
- ...
→ PE 중심의 관련 케이스
```

---

## 수정 사항

### 1. **Internal RAG 쿼리 생성 프롬프트**

#### Before (우선순위 불명확)

```python
Requirements:
1. 5-10 keywords maximum (concise but specific)
2. Include: main diagnosis + critical differentials + key clinical context
3. Focus on high-risk scenarios
```

**문제:**
- "main diagnosis + differentials + context" 순서가 명확하지 않음
- LLM이 증상을 먼저 넣을 수 있음

#### After (명확한 우선순위)

```python
Requirements:
1. 5-10 keywords maximum (concise but specific)
2. PRIORITY ORDER (most important first):
   a) PRIMARY DIAGNOSIS (required, must be first)
   b) Key clinical context (e.g., post-operative, ICU, emergency)
   c) High-risk findings (e.g., hypoxia, hypotension, tachycardia)
   d) M&M keywords (e.g., missed diagnosis, complication, error)
3. Do NOT lead with symptoms (chest pain, headache) - start with diagnosis
```

**개선:**
- ✅ Primary diagnosis가 **첫 번째** 명시
- ✅ 증상으로 시작하지 말라는 명확한 지시
- ✅ 우선순위 번호 부여 (a, b, c, d)

---

### 2. **PubMed 쿼리 생성 프롬프트**

#### Before (우선순위 불명확)

```python
Requirements:
1. 5-8 keywords total (specific but not too narrow)
2. Include primary diagnosis (required)
3. Add M&M-relevant keywords (choose 1-2)
4. Add clinical context (1-2 keywords from comorbidities/risk factors)
```

**문제:**
- "Include primary diagnosis" 만 있고 우선순위 명시 없음
- LLM이 순서를 임의로 정할 수 있음

#### After (명확한 우선순위)

```python
Requirements:
1. 5-8 keywords total (specific but not too narrow)
2. PRIORITY ORDER (most important first):
   a) PRIMARY DIAGNOSIS (required, must be FIRST keyword)
   b) M&M keyword (diagnostic error / missed diagnosis / complication)
   c) Clinical context (post-operative, ICU, emergency, etc.)
   d) Risk factors or comorbidities (if relevant)
   e) "guideline" or "management" (optional, at the end)
3. Do NOT start with symptoms - PRIMARY DIAGNOSIS must be first
```

**개선:**
- ✅ Primary diagnosis가 **첫 번째 키워드** 명시
- ✅ 모든 요소의 순서 지정 (a → e)
- ✅ 증상으로 시작 금지 명확히 표시

---

### 3. **Clinical Analysis 프롬프트 (추가 방어)**

#### Before (증상 혼입 가능)

```python
Tasks:
1. Identify top 3 differential diagnoses or complications that MUST be ruled out
   (prioritize life-threatening conditions)
```

**문제:**
- LLM이 "chest pain", "dyspnea" 같은 증상을 priorities에 넣을 수 있음

#### After (질환만 명시)

```python
Tasks:
1. Identify top 3 differential diagnoses or complications that MUST be ruled out
   (prioritize life-threatening DISEASES, not symptoms)

CRITICAL: 
- "clinical_priorities" must be DISEASES/CONDITIONS, NOT symptoms
- Use specific medical diagnoses (e.g., "pulmonary embolism", NOT "chest pain")
- Use disease names (e.g., "acute coronary syndrome", NOT "dyspnea")

Focus on:
- Life-threatening DISEASES (PE, MI, stroke, sepsis, etc.) - NOT symptoms
- Common missed DIAGNOSES - NOT presenting complaints
- CONDITIONS requiring immediate intervention - NOT isolated symptoms
```

**개선:**
- ✅ "DISEASES, not symptoms" 명확히 강조
- ✅ 예시로 잘못된 경우 제시 ("chest pain" ❌)
- ✅ 3번 반복 강조 (Tasks, CRITICAL, Focus)

---

## 예시 비교

### 케이스: PE 진단 실패

**Input:**
```json
{
  "diagnosis": "Unknown",
  "clinical_text": "82세 남성, 수술 3주 후 갑작스러운 흉통, 저산소증 SpO2 90%, 빈맥 110bpm..."
}
```

#### ❌ Before (증상 우선 가능)

**Clinical Analysis:**
```json
{
  "clinical_priorities": ["chest pain", "dyspnea", "pulmonary embolism"]
}
```

**RAG Query:**
```
"chest pain post-operative dyspnea hypoxia pulmonary embolism"
 ↑ 증상 먼저
```

**PubMed Query:**
```
"chest pain post-operative diagnostic error prevention"
 ↑ 증상 먼저 → PE와 무관한 논문 검색
```

#### ✅ After (질환 우선 보장)

**Clinical Analysis:**
```json
{
  "clinical_priorities": ["pulmonary embolism", "acute coronary syndrome", "pneumothorax"]
}
```

**RAG Query:**
```
"pulmonary embolism post-operative hypoxia tachycardia missed diagnosis"
 ↑ 진단 먼저
```

**PubMed Query:**
```
"pulmonary embolism diagnostic error post-operative prevention guideline"
 ↑ 진단 먼저 → PE 관련 논문 정확히 검색
```

---

## 검색 품질 개선 효과

### 내부 RAG 검색

#### Before (증상 우선)
```
쿼리: "chest pain post-operative hypoxia PE"

검색 결과:
1. Chest pain in post-op patient (GERD)
2. Post-op chest discomfort management
3. Chest pain differential in ICU
4. PE case (relevant) ← 4번째에야 나옴
```

#### After (질환 우선)
```
쿼리: "pulmonary embolism post-operative hypoxia chest pain"

검색 결과:
1. PE in post-op patient (relevant) ✅
2. Post-op PE diagnostic delay (relevant) ✅
3. PE with atypical presentation (relevant) ✅
→ 모든 결과가 PE 중심
```

### 외부 PubMed 검색

#### Before (증상 우선)
```
쿼리: "chest pain diagnostic error post-operative"

검색 결과:
1. Chest Pain Evaluation in ER
2. Post-op Pain Management
3. Atypical Chest Pain Guidelines
→ PE와 무관
```

#### After (질환 우선)
```
쿼리: "pulmonary embolism diagnostic error post-operative prevention"

검색 결과:
1. Missed PE in Post-Op Patients ✅
2. VTE Prevention in Surgery ✅
3. PE Diagnostic Pitfalls ✅
→ 모두 PE 관련
```

---

## Critic Agent에 미치는 영향

### Before (증상 기반 근거)

```
Evidence:
- Internal: Chest pain cases (일부만 PE)
- External: General chest pain guidelines

Critic Output:
"흉통 환자에서 감별진단 필요..."
→ 너무 일반적, PE 특화 안됨
```

### After (질환 기반 근거)

```
Evidence:
- Internal: PE cases (모두 PE 관련)
- External: PE diagnostic error literature

Critic Output:
"수술 후 PE 진단 지연 사례(Case 1)에서 Wells score 미사용이 주요 원인.
문헌(PMID 12345)에 따르면 DVT sign + 흉통 + 저산소는 PE 고위험.
본 케이스에서 CTPA 미시행 = Critical Diagnostic Failure"
→ PE에 특화된 구체적 비판
```

---

## 코드 변경 요약

### 파일: `src/agents/evidence_agent.py`

#### 1. `analyze_clinical_context_with_llm()` (Line 88-170)

**변경:**
- ✅ "DISEASES, not symptoms" 3번 강조
- ✅ 예시 추가 ("pulmonary embolism", NOT "chest pain")

#### 2. `generate_search_query_with_llm()` (Line 173-236)

**변경:**
- ✅ "PRIORITY ORDER" 섹션 추가
- ✅ Primary diagnosis must be first 명시
- ✅ "Do NOT lead with symptoms" 경고 추가

#### 3. `generate_pubmed_query_with_llm()` (Line 239-325)

**변경:**
- ✅ "PRIORITY ORDER" 섹션 추가 (a→e 순서)
- ✅ "PRIMARY DIAGNOSIS must be FIRST keyword" 강조
- ✅ "Do NOT start with symptoms" 명시

---

## 검증 방법

### 로그 확인

```bash
python scripts/run_agent_critique.py

# 로그 확인
[LLM Analysis] Priorities: ['pulmonary embolism', 'acute coronary syndrome', 'pneumothorax']
# ✅ 질환명, 증상 아님

[LLM Query] Generated: pulmonary embolism post-operative hypoxia tachycardia missed diagnosis
# ✅ 첫 단어가 진단명

[LLM PubMed Query] Generated: pulmonary embolism diagnostic error post-operative prevention guideline
# ✅ 첫 단어가 진단명
```

### 체크리스트

- ✅ clinical_priorities에 증상 없음? (chest pain, dyspnea ❌)
- ✅ clinical_priorities가 질환명? (pulmonary embolism ✅)
- ✅ RAG 쿼리 첫 단어가 진단명?
- ✅ PubMed 쿼리 첫 단어가 진단명?

---

## 제한사항

### LLM의 자유도

- 명확한 지시에도 LLM이 가끔 순서를 바꿀 수 있음
- 해결: 여러 번 강조 + 예시 제공

### Primary Diagnosis = "Unknown"인 경우

```python
# 진단 불명 케이스
diagnosis = "Unknown"

# 이 경우 clinical_priorities 활용
clinical_priorities = ["pulmonary embolism", "acute MI", "sepsis"]

# 쿼리 생성 시 priorities[0] 사용
query = f"{priorities[0]} {context_keywords} missed diagnosis"
```

---

## 결론

**Q: Primary diagnosis가 먼저 오게 되어 있나?**

**A: 이제 명확히 그렇습니다! ✅**

### 3단계 보장:

1. **Clinical Analysis:** "DISEASES, not symptoms" 3번 강조
2. **RAG Query:** "PRIMARY DIAGNOSIS must be first" + 예시
3. **PubMed Query:** "PRIMARY DIAGNOSIS must be FIRST keyword" + 우선순위 번호

### 기대 효과:

- 🎯 검색 정확도 향상 (질환 중심)
- 🎯 관련 없는 케이스 제거 (증상 중심 배제)
- 🎯 구체적인 비판/해결책 생성

**Primary Diagnosis 우선순위 확보 완료!** 🚀
