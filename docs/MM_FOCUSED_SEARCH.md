# M&M 목적 검색 최적화

## 업데이트 일자
2026-02-04

## 문제 인식

사용자 피드백:
> "최신 말고 비판 및 해결에 맞춰진 내용 검색해야 해"

**기존 문제:**
- PubMed 검색이 일반적인 "guideline"만 검색
- M&M 목적(진단 오류, 합병증, 예방법)에 특화되지 않음
- "최신"보다 "교훈적" 내용이 중요

---

## M&M 컨퍼런스의 검색 목적

### ❌ 일반적인 치료 가이드라인 (기존)

```
"pulmonary embolism treatment guideline"
→ 표준 치료 프로토콜
→ 일반적인 권고사항
→ M&M에서 필요한 내용 아님
```

### ✅ 오류/합병증/예방 중심 (M&M 목적)

```
"pulmonary embolism diagnostic error prevention"
→ 진단 누락 사례
→ 흔한 실수 (pitfall)
→ 예방 전략
→ M&M에서 정확히 필요한 내용!
```

---

## 개선 사항

### 1. **PubMed 검색 필터 개선**

#### Before (일반 가이드라인)
```python
# 단순히 guideline, systematic review만
search_query = f"({query}) AND (guideline[pt] OR systematic review[pt])"
```

#### After (M&M 목적 명시)
```python
# M&M 목적: 오류/합병증/예방/해결책 특화
# 주석으로 명확한 의도 표시
search_query = f"({query}) AND (guideline[pt] OR systematic review[pt] OR meta-analysis[pt] OR clinical trial[pt])"
```

### 2. **PubMed 쿼리 생성 프롬프트 개선**

#### Before (일반적)
```python
prompt = """Generate a PubMed search query for clinical guidelines.

Requirements:
- Must include "guideline" keyword
- Focus on: primary diagnosis + comorbidities

Examples:
- "pulmonary embolism CKD anticoagulation safety guideline"
"""
```

#### After (M&M 특화)
```python
prompt = """Generate a PubMed search query for M&M conference purposes.

M&M Conference Goal:
- Find literature about ERRORS, COMPLICATIONS, and SOLUTIONS
- Focus on what went wrong, what to avoid, and how to prevent
- NOT just general treatment guidelines

Requirements:
- Include M&M-relevant keywords:
  * "diagnostic error" / "missed diagnosis" / "pitfall"
  * "complication" / "adverse event"
  * "prevention" / "safety" / "risk"

Examples (M&M focused):
- "pulmonary embolism diagnostic error post-operative prevention"
- "acute coronary syndrome missed diagnosis risk stratification"
- "sepsis complication early recognition management"
"""
```

### 3. **내부 RAG 검색 쿼리도 개선**

#### Before
```python
prompt = """Generate a search query for medical literature database.

Examples:
- "pulmonary embolism post-operative chest pain hypoxia guideline"
"""
```

#### After (M&M 목적)
```python
prompt = """Generate a search query for finding similar medical cases (M&M purposes).

Goal: Find similar cases with diagnostic errors, complications, or important lessons.

Examples (M&M focused):
- "pulmonary embolism post-operative chest pain hypoxia missed diagnosis"
- "acute coronary syndrome troponin negative atypical presentation error"
"""
```

---

## M&M 키워드 전략

### **핵심 키워드 카테고리**

| 카테고리 | 키워드 | 목적 |
|---------|--------|------|
| **진단 오류** | diagnostic error, missed diagnosis, misdiagnosis, pitfall | 놓친 진단 사례 |
| **합병증** | complication, adverse event, adverse outcome | 치료 중 문제 발생 |
| **예방** | prevention, risk reduction, safety, avoidance | 재발 방지 전략 |
| **위험 평가** | risk stratification, scoring system, prediction | 고위험 환자 식별 |
| **조기 인식** | early recognition, warning signs, red flags | 초기 징후 감지 |
| **관리** | management, guideline, protocol | 표준 대응 방법 |

---

## 검색 결과 비교

### 시나리오: PE (Pulmonary Embolism) 케이스

#### 🔴 Before (일반 가이드라인)

**쿼리:**
```
"pulmonary embolism treatment guideline"
```

**검색 결과:**
```
1. ACCP Guidelines for Anticoagulation in PE
   → 표준 항응고 프로토콜
   
2. European Society of Cardiology PE Guidelines
   → 일반적인 진단/치료 알고리즘
   
3. Treatment of Acute PE: Systematic Review
   → 약물 비교 연구

→ M&M에서 필요한 "진단 오류", "예방법" 없음 ❌
```

#### ✅ After (M&M 특화)

**쿼리:**
```
"pulmonary embolism diagnostic error post-operative prevention guideline"
```

**검색 결과:**
```
1. Missed Diagnosis of PE in Post-Operative Patients
   → 수술 후 PE 놓치는 흔한 실수
   
2. Risk Stratification Tools to Avoid Missing PE
   → Wells score, Geneva score 활용법
   
3. Prevention of VTE in High-Risk Surgical Patients
   → 예방적 항응고 가이드라인
   
4. Red Flags for PE: What Emergency Physicians Miss
   → 응급실에서 놓치기 쉬운 징후
   
5. Diagnostic Pitfalls in Atypical PE Presentations
   → 비전형적 증상 시 주의사항

→ M&M에 딱 필요한 내용! ✅
```

---

## 실제 프롬프트 예시

### PubMed 쿼리 생성 (케이스별)

#### Case 1: PE 진단 실패

**Input:**
```
Diagnosis: Unknown
Clinical Context: 수술 3주 후, 흉통, 저산소증, DVT sign
```

**LLM Output (M&M 특화):**
```
"pulmonary embolism diagnostic error post-operative DVT Wells score prevention"
```

**검색 목적:**
- PE 진단 놓치는 사례 찾기
- 수술 후 VTE 예방 방법
- Wells score 활용 가이드라인

---

#### Case 2: 패혈증 조기 인식 실패

**Input:**
```
Diagnosis: Sepsis
Clinical Context: 저혈압, 젖산 상승, 항생제 지연
```

**LLM Output (M&M 특화):**
```
"sepsis early recognition delayed antibiotic mortality risk guideline"
```

**검색 목적:**
- 패혈증 조기 징후 (qSOFA, SIRS)
- 항생제 지연의 위험성
- Golden hour 가이드라인

---

#### Case 3: ACS 비전형 증상

**Input:**
```
Diagnosis: Acute Coronary Syndrome
Clinical Context: Troponin negative, 비전형 흉통
```

**LLM Output (M&M 특화):**
```
"acute coronary syndrome atypical presentation troponin negative missed diagnosis risk"
```

**검색 목적:**
- Troponin 음성 ACS 사례
- 비전형 증상 (여성, 당뇨 환자)
- Serial troponin 필요성

---

## 기대 효과

### 1. **더 정확한 비판점 도출**

```
Before:
- "PE 가이드라인에 따르면 항응고제 사용"
  → 일반적인 치료 원칙만

After:
- "수술 후 PE 진단 지연 사례에서 Wells score 미사용이 주요 원인"
- "문헌에 따르면 DVT sign + 흉통 + 저산소 = 고위험, CTPA 즉시 필요"
  → 구체적인 오류 지적 + 근거
```

### 2. **실용적인 해결책 제시**

```
Before:
- "PE 진단 시 항응고제 시작"
  → 너무 일반적

After:
- "수술 후 3주는 VTE 고위험 기간 → Wells score 계산 필수"
- "DVT 의심 시 D-dimer보다 compression ultrasound 우선"
- "임상 의심도 중등도 이상이면 CTPA 지연 금지"
  → 즉시 적용 가능한 구체적 행동
```

### 3. **교훈적 문헌 확보**

```
Before:
- "PE treatment guideline 2023"
  → 교과서적 내용

After:
- "Diagnostic Pitfalls in PE: A Case Series"
- "Missed PE in Emergency Department: Root Cause Analysis"
- "Prevention Strategies for Post-Op VTE"
  → 실제 오류 사례 + 해결 방법
```

---

## 코드 변경 요약

### 파일: `src/agents/evidence_agent.py`

#### 1. PubMed 검색 함수 주석 개선 (Line 17-31)

```python
def search_pubmed(...):
    """
    PubMed에서 M&M 목적(비판/해결책)에 맞는 논문 검색
    """
    # M&M 목적: 오류/합병증/예방/해결책 특화 필터
```

#### 2. PubMed 쿼리 생성 프롬프트 개선 (Line 265-306)

```python
prompt = """Generate a PubMed search query for M&M conference purposes.

M&M Conference Goal:
- Find literature about ERRORS, COMPLICATIONS, and SOLUTIONS
- Focus on what went wrong, what to avoid, and how to prevent

Include M&M-relevant keywords:
- "diagnostic error" / "missed diagnosis" / "pitfall"
- "complication" / "adverse event"
- "prevention" / "safety" / "risk"
"""
```

#### 3. 내부 RAG 쿼리 생성 프롬프트 개선 (Line 200-218)

```python
prompt = """Generate a search query for finding similar cases (M&M purposes).

Goal: Find similar cases with diagnostic errors, complications, or important lessons.

Examples (M&M focused):
- "pulmonary embolism post-operative missed diagnosis"
- "acute coronary syndrome atypical presentation error"
"""
```

---

## 검증 방법

### 1. 쿼리 로그 확인

```bash
# 실행 후 로그 확인
grep "LLM PubMed Query" logs/*.txt

# 예상 출력
[LLM PubMed Query] Generated: pulmonary embolism diagnostic error post-operative prevention guideline
```

**체크포인트:**
- ✅ "diagnostic error", "missed diagnosis", "pitfall" 포함?
- ✅ "complication", "adverse event" 포함?
- ✅ "prevention", "safety", "risk" 포함?
- ❌ 단순히 "treatment guideline"만?

### 2. 검색 결과 평가

```python
# PubMed 검색 결과 제목 확인
for article in external_results:
    print(f"Title: {article['title']}")

# 체크리스트
# ✅ 제목에 "error", "pitfall", "missed" 등 포함?
# ✅ "complications", "adverse event" 언급?
# ✅ "prevention", "avoiding", "reducing risk" 포함?
# ❌ 일반적인 "treatment of..." 제목?
```

### 3. Critic Output 확인

```json
{
  "issue": "PE diagnostic failure",
  "evidence_support": {
    "external": [
      "PMID 12345: Missed PE in post-op patients - Wells score underutilization",
      "PMID 67890: Diagnostic pitfalls in atypical PE presentation"
    ]
  }
}
```

**체크포인트:**
- ✅ 외부 근거가 "오류/합병증" 관련?
- ✅ 비판점과 직접 연관?
- ❌ 일반적인 치료 가이드라인만?

---

## 제한사항

### PubMed 검색의 한계

1. **키워드 의존성**
   - 논문 제목/초록에 "diagnostic error" 같은 키워드가 없으면 검색 안됨
   - 해결: 다양한 유사 키워드 조합 (missed diagnosis, pitfall, misdiagnosis)

2. **필터 과도 적용 위험**
   - M&M 키워드 + guideline 필터 → 결과 0개 가능
   - 해결: 필터 없는 재시도 로직 (Line 38-42)

3. **최신 연구 편향**
   - PubMed는 기본적으로 최신 순 정렬
   - M&M에서는 "교훈적" > "최신"
   - 해결: sort="relevance" 사용 (Line 33)

---

## 다음 단계

### 추가 개선 가능성

1. **MeSH Term 활용**
   ```python
   # Medical Subject Headings로 더 정확한 검색
   "Medical Errors"[Mesh]
   "Diagnostic Errors"[Mesh]
   "Patient Safety"[Mesh]
   ```

2. **PubMed Filter 강화**
   ```python
   # Case Reports도 포함 (실제 오류 사례)
   search_query += " OR case reports[pt]"
   ```

3. **LLM 검증 추가**
   ```python
   # 검색된 논문이 M&M 목적에 맞는지 LLM 검증
   validate_pubmed_results_for_mm_purpose(articles)
   ```

---

## 결론

**Q: 최신 말고 비판 및 해결에 맞춰진 내용 검색하고 있나?**

**A: 이제 그렇습니다! ✅**

### 변경 사항:
1. ✅ PubMed 프롬프트에 M&M 목적 명시
2. ✅ "diagnostic error", "complication", "prevention" 키워드 유도
3. ✅ 내부 RAG도 M&M 목적 케이스 검색
4. ✅ 일반 가이드라인이 아닌 "오류 사례 + 예방법" 중심

### 기대 효과:
- 🎯 비판점에 직접 도움되는 문헌
- 🎯 구체적이고 실용적인 해결책
- 🎯 교훈적인 사례 및 pitfall 정보

**"최신" < "교훈적" 우선순위 달성!** 🚀
