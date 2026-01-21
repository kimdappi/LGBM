# LLM 통합 계획

## 현재 문제점

현재 코드는 **수적 비교**에 의존:
- 집합 연산 (`&`, `-`)으로 단순 비교
- 정규표현식으로 패턴 매칭
- 임계값 기반 규칙 (`>= 2`, `>= 0.6`)
- 의미 이해 없이 문자열 매칭

## LLM 통합 우선순위

### 🔴 **최우선: critical_agent.py**

**이유**: 비판형 분석의 핵심, 의미 기반 이해가 가장 중요

**현재 문제**:
```python
# 수적 비교만 수행
if num_supporting >= 2:
    assessment = "충분한 근거 있음"
else:
    assessment = "추가 검토 필요"

# 단순 개수 비교
if len(similarities) >= len(differences):
    assessment = "표준 치료와 일치"
```

**LLM 개선 방안**:
1. **의사결정 분석** (`_analyze_decisions`)
   - LLM이 의사결정의 적절성, 근거의 질 평가
   - 단순 개수가 아닌 **의미적 근거 강도** 평가
   - 예: "이 의사결정은 유사 사례와 비교했을 때 적절한가? 왜?"

2. **치료 분석** (`_analyze_treatment`)
   - 약물/시술의 **의학적 적절성** 평가
   - 단순 일치가 아닌 **치료 전략의 합리성** 평가
   - 예: "이 치료 선택이 임상적으로 타당한가?"

3. **결과 분석** (`_analyze_outcome`)
   - 결과의 **임상적 의미** 해석
   - 단순 통계가 아닌 **맥락 기반 평가**
   - 예: "이 결과가 예상 가능한가? 왜 이례적인가?"

4. **위험 요소 식별** (`_identify_risk_factors`)
   - **임상적 위험도** 평가
   - 단순 개수가 아닌 **의학적 중요성** 평가
   - 예: "이 케이스의 주요 위험 요소는 무엇인가?"

5. **권고사항 생성** (`_generate_recommendations`)
   - **구체적이고 실행 가능한** 권고 생성
   - 단순 템플릿이 아닌 **맥락 기반 제안**
   - 예: "이 케이스에서 개선할 수 있는 점은?"

**구현 예시**:
```python
def _analyze_decisions_with_llm(self, evidence: Dict, llm) -> List[Dict]:
    """LLM 기반 의사결정 분석"""
    prompt = f"""
    다음 의사결정을 분석하세요:
    - 의사결정: {decision['title']}
    - 설명: {decision['description']}
    - 지원 사례 수: {num_supporting}
    
    이 의사결정이 임상적으로 적절한지, 근거가 충분한지 평가하세요.
    """
    
    analysis = llm.generate(prompt)
    # 구조화된 평가 반환
```

---

### 🟡 **중요: evidence_tracker.py**

**이유**: 비교 분석의 정확도 향상, 의미 기반 유사성 판단

**현재 문제**:
```python
# 단순 문자열 집합 비교
common = input_diag & sc_diag  # {"HCV cirrhosis"} == {"HCV cirrhosis"}만 매칭
diff = sc_diag - input_diag    # 의미적으로 유사해도 다르다고 판단
```

**LLM 개선 방안**:
1. **진단 비교** (`_compare_diagnosis`)
   - **의학적 동의어/관련성** 인식
   - 예: "Ascites"와 "Portal HTN with ascites"는 관련 있음
   - 의미적 유사성 평가

2. **치료 비교** (`_compare_treatment`)
   - **약물의 임상적 등가성** 판단
   - 예: "Furosemide 20mg"와 "Furosemide 40mg"는 용량 차이지만 같은 약물
   - 치료 전략의 유사성 평가

3. **근거 추출** (`_extract_evidence`)
   - **의사결정의 의미적 유사성** 판단
   - 단순 제목 매칭이 아닌 **맥락 기반 매칭**
   - 예: "ASCITES"와 "Diuretic refractory ascites"는 관련 있음

**구현 예시**:
```python
def _compare_diagnosis_with_llm(self, input_extracted, similar_extracted, llm):
    """LLM 기반 진단 비교"""
    for sc in similar_extracted:
        prompt = f"""
        다음 두 진단 목록을 비교하세요:
        입력 케이스: {input_diag}
        유사 사례: {sc_diag}
        
        의학적으로 관련된 진단, 동의어, 하위/상위 개념을 찾아주세요.
        """
        
        semantic_similarity = llm.analyze(prompt)
        # 의미적 유사성 반환
```

---

### 🟢 **보조: extractor.py**

**이유**: 정보 추출 정확도 향상, 맥락 이해

**현재 문제**:
```python
# 정규표현식으로만 추출
primary_match = re.search(r'(?:Primary|PRIMARY)[:\s]*(.*?)', text)
# 형식이 다르면 추출 실패
```

**LLM 개선 방안**:
1. **진단 추출** (`_extract_diagnosis`)
   - 형식에 구애받지 않는 추출
   - 맥락 기반 진단명 정규화
   - 예: "HCV cirrhosis complicated by ascites" → ["HCV cirrhosis", "ascites"]

2. **의사결정 포인트 추출** (`_extract_decision_points`)
   - "#" 없이도 의사결정 포인트 인식
   - 의미 기반 섹션 구분
   - 예: "Goals of care", "DNR/DNI 결정" 등

**구현 예시**:
```python
def _extract_diagnosis_with_llm(self, case: Dict, llm):
    """LLM 기반 진단 추출"""
    prompt = f"""
    다음 퇴원 진단 텍스트에서 모든 진단을 추출하세요:
    {case['discharge_diagnosis']}
    
    Primary, Secondary 구분 없이 모든 진단명을 리스트로 반환하세요.
    """
    
    diagnoses = llm.extract(prompt)
    # 구조화된 진단 리스트 반환
```

---

### 🔵 **선택적: retriever.py**

**이유**: 유사도 검색 품질 향상 (하지만 TF-IDF도 충분히 효과적)

**현재 문제**:
```python
# TF-IDF는 단어 기반, 의미 이해 부족
# "liver disease"와 "hepatic disorder"는 다르게 인식
```

**LLM 개선 방안**:
- **임베딩 기반 검색**: LLM 임베딩으로 의미적 유사도 계산
- 또는 TF-IDF + LLM 하이브리드

**구현 예시**:
```python
def retrieve_with_embeddings(self, query_case, llm):
    """LLM 임베딩 기반 검색"""
    query_embedding = llm.embed(query_text)
    case_embeddings = [llm.embed(case_text) for case_text in all_texts]
    
    similarities = cosine_similarity([query_embedding], case_embeddings)
    # 의미 기반 유사도
```

---

## 통합 전략

### Phase 1: Critical Agent 우선 (최우선)
```
critical_agent.py
├─ _analyze_decisions() → LLM 기반 평가
├─ _analyze_treatment() → LLM 기반 평가
├─ _analyze_outcome() → LLM 기반 해석
├─ _identify_risk_factors() → LLM 기반 식별
└─ _generate_recommendations() → LLM 기반 생성
```

### Phase 2: Evidence Tracker 개선
```
evidence_tracker.py
├─ _compare_diagnosis() → LLM 기반 의미 비교
├─ _compare_treatment() → LLM 기반 의미 비교
└─ _extract_evidence() → LLM 기반 맥락 매칭
```

### Phase 3: Extractor 보강 (선택적)
```
extractor.py
├─ _extract_diagnosis() → LLM 보조 추출
└─ _extract_decision_points() → LLM 기반 추출
```

### Phase 4: Retriever 개선 (선택적)
```
retriever.py
└─ retrieve() → LLM 임베딩 기반 검색
```

## 구현 고려사항

### LLM 선택
- **API 기반**: OpenAI GPT-4, Claude (빠른 프로토타이핑)
- **로컬 모델**: Llama 3, Mistral (비용 절감, 프라이버시)
- **의료 특화**: BioBERT, ClinicalBERT (도메인 특화)

### 비용 최적화
- **하이브리드 접근**: 규칙 기반 + LLM (필요한 곳만)
- **배치 처리**: 여러 케이스 한 번에 처리
- **캐싱**: 동일한 쿼리 재사용

### 구조 설계
```python
# LLM 래퍼 클래스
class LLMAnalyzer:
    def analyze_decision(self, decision, context):
        """의사결정 분석"""
        pass
    
    def compare_semantically(self, item1, item2):
        """의미 기반 비교"""
        pass

# CriticalAgent에 통합
class CriticalAnalysisAgent:
    def __init__(self, llm_analyzer=None):
        self.llm = llm_analyzer
    
    def _analyze_decisions(self, evidence):
        if self.llm:
            return self._analyze_decisions_with_llm(evidence)
        else:
            return self._analyze_decisions_rule_based(evidence)
```

## 결론

**최우선 통합 지점**: `critical_agent.py`
- 비판형 분석의 핵심
- 의미 기반 이해가 가장 중요
- 수적 비교의 한계가 가장 명확

**다음 우선순위**: `evidence_tracker.py`
- 비교 분석의 정확도 향상
- 의미적 유사성 판단

**보조 통합**: `extractor.py`, `retriever.py`
- 정확도 향상이지만 규칙 기반도 충분히 작동
