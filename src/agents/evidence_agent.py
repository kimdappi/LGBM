"""Evidence Agent - CRAG + 2-Pass 타겟 검색.

구성:
  1. PubMed / 내부 RAG 검색
  2. LLM 임상 분석 및 쿼리 생성
  3. 품질 평가 및 LLM 검증
  4. Evidence Agent 1차 (CRAG)
  5. Evidence Agent 2차 (비판 기반 타겟 검색)
"""

from Bio import Entrez
from typing import Dict, List
from openai import OpenAI
import os
import json
import re

# PubMed 설정
Entrez.email = os.getenv("PUBMED_EMAIL", "researcher@example.com")

# CRAG 임계치
SIMILARITY_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# 1. PubMed / 내부 RAG 검색
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 5, use_mesh: bool = True) -> List[Dict]:
    """
    PubMed에서 M&M 목적(비판/해결책)에 맞는 논문 검색
    
    Args:
        query: 검색 질의
        max_results: 최대 결과 수
        use_mesh: MeSH term 사용 여부 (기본: True)
    """
    try:
        # M&M 목적: 오류/합병증/예방/해결책 특화 필터
        if use_mesh:
            # 우선순위 1: 가이드라인 + 안전/오류/합병증 키워드
            # 우선순위 2: Systematic review (근거 수준 높음)
            # 우선순위 3: Clinical trial (실제 증거)
            search_query = f"({query}) AND (guideline[pt] OR systematic review[pt] OR meta-analysis[pt] OR clinical trial[pt])"
        else:
            search_query = query
        
        handle = Entrez.esearch(db="pubmed", term=search_query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        # Fallback: MeSH 필터로 결과 없으면 일반 검색 재시도
        if not record["IdList"] and use_mesh:
            print(f"  [PubMed] No results with filters, retrying without filters...")
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
        
        if not record["IdList"]:
            return []
        
        # 초록 가져오기
        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="abstract", retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        results = []
        for article in articles.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            
            results.append({
                "pmid": str(medline.get("PMID", "")),
                "title": article_data.get("ArticleTitle", ""),
                "abstract": article_data.get("Abstract", {}).get("AbstractText", [""])[0] if article_data.get("Abstract") else "",
                "source": "pubmed"
            })
        
        return results

    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def search_internal_rag(query: str, rag_retriever, top_k: int = 3) -> List[Dict]:
    """내부 RAG에서 유사 케이스 검색"""
    try:
        if hasattr(rag_retriever, 'retrieve_with_patient'):
            cohort = rag_retriever.retrieve_with_patient({"clinical_text": query}, top_k=top_k)
            cases = cohort.get("similar_cases", [])
            return [{"content": c.get("text", ""), "score": c.get("similarity", 0), "source": "internal", "case_id": c.get("id")} for c in cases]
        else:
            results = rag_retriever.search(query, top_k=top_k)
            return [{"content": r["text"], "score": r["score"], "source": "internal"} for r in results]
    except Exception as e:
        print(f"Internal RAG search error: {e}")
        return []


# ---------------------------------------------------------------------------
# 2. LLM 임상 분석 및 쿼리 생성
# ---------------------------------------------------------------------------

def analyze_clinical_context_with_llm(patient: Dict, structured_chart: Dict = None) -> Dict:
    """
    🎯 LLM으로 임상 맥락 분석 및 검색 전략 생성
    
    Returns:
        {
            "clinical_priorities": ["PE", "ACS", ...],  # 의심되는 진단들
            "key_findings": ["hypoxia", "chest pain", ...],  # 주요 소견
            "risk_factors": ["post-op", "immobilization", ...],  # 위험 인자
            "search_strategy": str,  # 검색 전략 설명
            "urgency_level": "critical/high/moderate/low"
        }
    """
    diagnosis = patient.get("diagnosis", "Unknown")
    
    # 기본 결과 준비
    default_result = {
        "clinical_priorities": [diagnosis],
        "key_findings": ["diagnosis confirmed"],
        "risk_factors": [],
        "urgency_level": "moderate",
        "search_strategy": "Evidence-based guideline search",
        "reasoning": "Basic analysis based on primary diagnosis"
    }
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        clinical_text = (patient.get("clinical_text", "") or patient.get("text", ""))[:2000]
        
        # Structured chart 정보 추가
        vitals_info = ""
        if structured_chart:
            vitals = structured_chart.get('vitals', {})
            if vitals:
                vitals_info = f"\n\nVital Signs:\n{json.dumps(vitals, indent=2)}"
        
        prompt = f"""You analyze a clinical case to guide evidence search for M&M critique/solutions.

Primary diagnosis: {diagnosis}

Clinical text (may be truncated):
{clinical_text}{vitals_info}

Return ONLY valid JSON:
{{
  "clinical_priorities": ["life-threatening disease/complication", "…", "…"],
  "key_findings": ["key symptom/vital/lab", "..."],
  "risk_factors": ["risk factor/context", "..."],
  "urgency_level": "critical|high|moderate|low",
  "search_strategy": "what evidence is needed (errors/complications/prevention)",
  "reasoning": "one sentence"
}}

Rules:
- Priorities must be DISEASES/complications (e.g., PE/ACS/sepsis), not symptoms.
- Prefer commonly missed diagnoses and time-sensitive conditions."""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        
        # JSON 파싱
        response_text = response.choices[0].message.content.strip()
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        result = json.loads(response_text)
        
        # 필수 필드 검증
        required_fields = ["clinical_priorities", "key_findings", "risk_factors", "urgency_level", "search_strategy"]
        for field in required_fields:
            if field not in result:
                print(f"  [Warning] Missing field '{field}', using default")
                result[field] = default_result[field]
        
        print(f"  [LLM Analysis] Priorities: {result.get('clinical_priorities', [])}")
        print(f"  [LLM Analysis] Urgency: {result.get('urgency_level', 'unknown')}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parsing failed: {e}")
        print(f"  [Strategy] Using default clinical analysis")
        return default_result
    except Exception as e:
        print(f"  [ERROR] LLM clinical analysis failed: {e}")
        print(f"  [Strategy] Using default clinical analysis")
        return default_result


def generate_search_query_with_llm(patient: Dict, clinical_analysis: Dict = None) -> str:
    """
    🎯 LLM으로 최적화된 검색 쿼리 생성
    
    Args:
        patient: 환자 정보
        clinical_analysis: analyze_clinical_context_with_llm() 결과
    
    Returns:
        str: 생성된 검색 쿼리 (실패 시 기본 쿼리)
    """
    diagnosis = patient.get("diagnosis", "Unknown")
    
    # 기본 쿼리 준비 (LLM 실패 시 사용)
    default_query = f"{diagnosis} complications diagnostic error"
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Clinical analysis 정보 포함
        analysis_info = ""
        if clinical_analysis:
            priorities = clinical_analysis.get("clinical_priorities", [])
            findings = clinical_analysis.get("key_findings", [])
            risk_factors = clinical_analysis.get("risk_factors", [])
            
            analysis_info = f"""
Clinical Analysis:
- Differential diagnoses: {', '.join(priorities[:3])}
- Key findings: {', '.join(findings[:5])}
- Risk factors: {', '.join(risk_factors[:3])}
"""
        
        prompt = f"""Generate ONE short search query for similar cases/solutions (M&M purpose).

Primary diagnosis (must be first): {diagnosis}
{analysis_info}

Goal: Find similar cases or solutions about the Primary Diagnosis with diagnostic errors, complications, or important lessons.

Return ONLY the query string (no quotes, no explanation)."""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        
        query = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # 빈 쿼리 방지
        if not query or len(query.strip()) == 0:
            print(f"  [LLM Query] Empty result, using default query")
            return default_query
        
        print(f"  [LLM Query] Generated: {query}")
        return query
        
    except Exception as e:
        print(f"  [ERROR] LLM query generation failed: {e}")
        print(f"  [Strategy] Using default query: {default_query}")
        return default_query


def generate_pubmed_query_with_llm(patient: Dict, clinical_analysis: Dict = None) -> str:
    """
    🎯 LLM으로 PubMed 검색 쿼리 생성
    
    PubMed는 max_results=5만 가져오므로 더 구체적일수록 좋음
    
    Returns:
        str: 생성된 검색 쿼리 (실패 시 기본 쿼리)
    """
    diagnosis = patient.get("diagnosis", "Unknown")
    
    # 기본 쿼리 준비 (LLM 실패 시 사용)
    default_query = f"{diagnosis} complication prevention guideline"
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        secondary = patient.get("secondary_diagnoses", [])
        key_conditions = patient.get("key_conditions", [])
        
        # Clinical analysis 정보
        analysis_info = ""
        if clinical_analysis:
            priorities = clinical_analysis.get("clinical_priorities", [])
            risk_factors = clinical_analysis.get("risk_factors", [])
            
            analysis_info = f"""
Clinical Priorities: {', '.join(priorities[:3])}
Risk Factors: {', '.join(risk_factors[:3])}
"""
        
        comorbidities_info = ""
        if secondary or key_conditions:
            all_conditions = secondary + key_conditions
            comorbidities_info = f"\nComorbidities: {', '.join(all_conditions[:5])}"
        
        prompt = f"""Generate ONE PubMed query for M&M (errors/complications + prevention).

Primary Diagnosis: {diagnosis}
{comorbidities_info}
{analysis_info}

M&M Conference Goal:
- Find literature about ERRORS, COMPLICATIONS, and SOLUTIONS about the Primary Diagnosis
- Focus on what went wrong, what to avoid, and how to prevent
- NOT just general treatment guidelines

Context:
- PubMed will return only TOP 5 results (max_results=5)
- More specific = higher quality top 5
- Focus on: diagnostic errors, complications, adverse events, prevention strategies

Requirements:
1. 5-8 keywords total (specific but not too narrow)
2. PRIORITY ORDER (most important first):
   a) PRIMARY DIAGNOSIS (required, must be FIRST keyword)
   b) M&M keyword (diagnostic error / missed diagnosis / complication / prevention)
   c) Clinical context (post-operative, ICU, emergency, etc.)
   d) Risk factors or comorbidities (if relevant)
   e) "guideline" or "management" (optional, at the end)

Generate the query. Return ONLY the query string, no explanations."""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        
        query = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # 빈 쿼리 방지
        if not query or len(query.strip()) == 0:
            print(f"  [LLM PubMed Query] Empty result, using default query")
            return default_query
        
        print(f"  [LLM PubMed Query] Generated: {query}")
        return query
        
    except Exception as e:
        print(f"  [ERROR] LLM PubMed query generation failed: {e}")
        print(f"  [Strategy] Using default query: {default_query}")
        return default_query


# ---------------------------------------------------------------------------
# 3. 품질 평가 및 LLM 검증
# ---------------------------------------------------------------------------

def evaluate_internal_quality(internal_results: List[Dict], threshold: float = SIMILARITY_THRESHOLD) -> Dict:
    """
    내부 근거 품질 평가 (필터링 후)
    
    Returns:
        {
            "is_sufficient": bool,
            "count": int,
            "avg_score": float,
            "reason": str
        }
    """
    if not internal_results:
        return {
            "is_sufficient": False,
            "count": 0,
            "avg_score": 0.0,
            "reason": "유사 케이스 없음 (모두 유사도 < 0.7)"
        }
    
    scores = [r.get("score", 0) for r in internal_results]
    avg_score = sum(scores) / len(scores)
    
    # 충분 조건: 최소 1개 이상의 유사 케이스 (이미 threshold 필터링 통과)
    is_sufficient = len(internal_results) >= 1
    
    return {
        "is_sufficient": is_sufficient,
        "count": len(internal_results),
        "avg_score": round(avg_score, 3),
        "reason": f"{len(internal_results)}건 (평균 유사도: {avg_score:.3f})"
    }


def extract_key_events(text: str) -> Dict:
    """케이스 텍스트에서 M&M에 중요한 핵심 이벤트 추출"""
    default_events = {"outcome": None, "procedures": [], "complications": [], "critical_events": []}
    
    # None, 빈 문자열, 공백만 있는 경우 처리
    if not text or not isinstance(text, str) or not text.strip():
        return default_events
    
    text_lower = text.lower()
    events = {"outcome": None, "procedures": [], "complications": [], "critical_events": []}
    
    # Outcome 추출 (다양한 표현 지원)
    death_keywords = [
        "expired", "died", "death", "deceased", "passed away",
        "discharge disposition:\nexpired", "discharge disposition: expired",
        "discharge location:\ndied", "discharge location: died",
        "cmo", "comfort measures", "withdrawal of care", "dnr/dni"
    ]
    survival_keywords = ["discharged home", "discharge to", "discharged to"]
    
    if any(kw in text_lower for kw in death_keywords):
        events["outcome"] = "death"
    elif any(kw in text_lower for kw in survival_keywords):
        events["outcome"] = "survived"
    elif "discharge" in text_lower:
        events["outcome"] = "survived"
    
    # 시술/프로시저 추출 (확장된 키워드)
    procedure_keywords = [
        "paracentesis", "thoracentesis", "intubation", "intubated", "extubated",
        "egd", "endoscopy", "colonoscopy", "ercp", "tips", "bronchoscopy",
        "catheterization", "cardiac cath", "pci", "cabg", "surgery", "operation",
        "transfusion", "mtp", "massive transfusion", "prbc", "ffp", "platelets",
        "dialysis", "crrt", "hemodialysis", "central line", "a-line",
        "ct scan", "ctpa", "mri", "ultrasound", "biopsy", "lumbar puncture"
    ]
    for kw in procedure_keywords:
        if kw in text_lower and kw not in events["procedures"]:
            events["procedures"].append(kw)
    
    # 합병증/critical event 추출 (확장된 키워드)
    complication_keywords = [
        "hemorrhage", "bleeding", "hemoperitoneum", "hematemesis", "melena", "hematochezia",
        "hypotension", "shock", "cardiac arrest", "code blue", "pulseless",
        "respiratory failure", "hypoxia", "hypoxemia", "ards", "respiratory distress",
        "renal failure", "aki", "acute kidney injury", "anuria", "oliguria",
        "hepatorenal", "encephalopathy", "altered mental status", "ams", "confusion",
        "sepsis", "septic shock", "bacteremia", "infection",
        "iatrogenic", "complication", "adverse event", "error",
        "hct drop", "hgb drop", "anemia", "coagulopathy", "dic",
        "pressors", "vasopressors", "norepinephrine", "vasopressin",
        "aspiration", "pneumonia", "pulmonary embolism", "pe", "dvt",
        "stroke", "mi", "myocardial infarction", "arrhythmia"
    ]
    for kw in complication_keywords:
        if kw in text_lower and kw not in events["complications"]:
            events["complications"].append(kw)
    
    # Critical 이벤트 시퀀스 패턴 (확장)
    critical_patterns = [
        # Procedure → Complication
        (["paracentesis"], ["bleeding", "hemorrhage", "hemoperitoneum"], "paracentesis → bleeding"),
        (["thoracentesis"], ["bleeding", "pneumothorax"], "thoracentesis → complication"),
        (["central line", "catheter"], ["infection", "sepsis", "bacteremia"], "line → infection"),
        (["surgery", "operation"], ["bleeding", "hemorrhage"], "surgery → bleeding"),
        (["intubation"], ["aspiration", "pneumonia"], "intubation → aspiration"),
        
        # Drug → Adverse event
        (["lorazepam", "benzodiazepine", "ativan", "midazolam"], ["encephalopathy", "confusion", "ams"], "sedative → HE worsening"),
        (["nsaid", "ketorolac", "ibuprofen"], ["renal failure", "aki", "creatinine"], "NSAID → AKI"),
        (["anticoagulant", "heparin", "warfarin"], ["bleeding", "hemorrhage"], "anticoagulation → bleeding"),
        
        # Cascade
        (["transfusion", "mtp", "prbc"], ["expired", "died", "death"], "transfusion → death"),
        (["pressors", "vasopressor", "norepinephrine"], ["expired", "died", "death"], "vasopressors → death"),
        (["intubation"], ["expired", "died", "death"], "intubation → death"),
        
        # Disease progression
        (["cirrhosis"], ["hepatorenal", "hrs"], "cirrhosis → HRS"),
        (["cirrhosis"], ["encephalopathy"], "cirrhosis → HE"),
        (["sepsis"], ["shock", "hypotension"], "sepsis → shock"),
    ]
    
    for triggers, outcomes, event_name in critical_patterns:
        if any(t in text_lower for t in triggers) and any(o in text_lower for o in outcomes):
            if event_name not in events["critical_events"]:
                events["critical_events"].append(event_name)
    
    return events


def validate_internal_evidence_with_llm(internal_results: List[Dict], patient: Dict) -> Dict:
    """
    LLM으로 내부 근거 검증 (M&M 비판/해결에 유용한지)
    
    개선사항:
    - 케이스 텍스트에서 핵심 이벤트(outcome, procedures, complications) 추출
    - 텍스트 길이 확대 (300 → 1000)
    - 유사한 결과/경과도 유효한 것으로 인정
    """
    if not internal_results:
        return {
            "is_valid": False,
            "reason": "내부 근거 없음",
            "confidence": 0.0,
            "filtered_results": []
        }
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 인덱스 케이스 핵심 이벤트 추출
        index_text = patient.get('clinical_text', '') or patient.get('text', '')
        index_events = extract_key_events(index_text)
        
        # 내부 케이스 요약 (핵심 이벤트 포함)
        cases_summary = []
        for i, case in enumerate(internal_results[:3]):
            content = case.get('content', '') or case.get('text', '')
            case_events = extract_key_events(content)
            
            # 더 많은 컨텍스트 전달 (300 → 1000)
            summary = f"""Case {i+1}:
- Outcome: {case_events['outcome'] or 'unknown'}
- Procedures: {', '.join(case_events['procedures'][:5]) or 'none'}
- Complications: {', '.join(case_events['complications'][:5]) or 'none'}
- Critical sequence: {', '.join(case_events['critical_events']) or 'none'}
- Text excerpt: {content[:1000]}..."""
            cases_summary.append(summary)
        
        prompt = f"""You judge whether retrieved cases are useful for M&M (Morbidity & Mortality) critique.

INDEX CASE (the patient being reviewed):
- Diagnosis: {patient.get('diagnosis', 'Unknown')}
- Outcome: {index_events['outcome'] or 'unknown'}
- Procedures: {', '.join(index_events['procedures'][:5]) or 'none'}
- Complications: {', '.join(index_events['complications'][:5]) or 'none'}
- Critical events: {', '.join(index_events['critical_events']) or 'none'}

CANDIDATE SIMILAR CASES:
{chr(10).join(cases_summary)}

VALIDATION CRITERIA - Mark is_valid=TRUE if ANY of the following:
1. SIMILAR OUTCOME: Same or related death/complication pathway (e.g., both died after procedure)
2. SIMILAR PROCEDURE + COMPLICATION: Same procedure with complications (e.g., paracentesis → bleeding)
3. SIMILAR DISEASE TRAJECTORY: Same disease progression (e.g., cirrhosis → HE → death)
4. LEARNING VALUE: Case shows what went wrong or how to prevent similar outcomes

CRITICAL: A case with "paracentesis → bleeding → MTP → death" is HIGHLY RELEVANT to another case with same trajectory. Do NOT reject just because the exact error keywords are missing.

Return JSON:
{{
  "is_valid": true/false,
  "reason": "one sentence explaining why useful or not for M&M learning",
  "valid_case_indices": [0,1,2],
  "confidence": 0.0-1.0
}}

Be GENEROUS with is_valid=true if the outcome/complication pattern matches."""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 필수 필드 검증
        is_valid = result.get("is_valid", False)
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "No reason provided")
        valid_indices = result.get("valid_case_indices", [])
        
        # 신뢰할 수 있는 케이스만 필터링
        filtered_results = [internal_results[i] for i in valid_indices if i < len(internal_results)]
        
        # confidence가 0.5 미만이면 is_valid를 False로
        if confidence < 0.5:
            is_valid = False
        
        print(f"  [LLM Validation] is_valid={is_valid}, confidence={confidence:.2f}, reason={reason[:80]}...")
        
        return {
            "is_valid": is_valid,
            "reason": reason,
            "confidence": confidence,
            "filtered_results": filtered_results if is_valid else []
        }
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] LLM Validation JSON parsing failed: {e}")
        return {
            "is_valid": False,
            "reason": "Validation JSON parsing failed",
            "confidence": 0.0,
            "filtered_results": []
        }
    except Exception as e:
        print(f"  [ERROR] LLM Validation failed: {e}")
        return {
            "is_valid": False,
            "reason": f"Validation error: {str(e)}",
            "confidence": 0.0,
            "filtered_results": []
        }


# ---------------------------------------------------------------------------
# 3.5 Evidence 포맷팅 (공유 함수 - Diagnosis/Treatment/Critic 공용)
# ---------------------------------------------------------------------------

def format_evidence_summary(evidence: Dict, include_abstract: bool = True) -> str:
    """
    Evidence 검색 결과를 에이전트 프롬프트용 문자열로 포맷팅.
    Diagnosis/Treatment/Critic 등 모든 에이전트가 공유.

    Args:
        evidence: run_evidence_agent() 반환 dict 전체
        include_abstract: PubMed abstract 포함 여부 (기본: True)
    """
    if not evidence:
        return "검색된 근거 없음"

    lines: List[str] = []

    internal = evidence.get("internal", {})
    external = evidence.get("external", {})

    internal_results = internal.get("results", [])
    external_results = external.get("results", [])

    # ── 내부 유사 케이스 ──
    if internal_results:
        lines.append("### 내부 유사 케이스")
        survived = [c for c in internal_results if str(c.get("status", "")).lower() in ("alive", "survived")]
        died = [c for c in internal_results if str(c.get("status", "")).lower() in ("dead", "died")]
        lines.append(f"- 생존: {len(survived)}건, 사망: {len(died)}건")
        for c in internal_results[:3]:
            score = c.get("score", c.get("similarity", 0))
            status = c.get("status", "unknown")
            content = c.get("content", c.get("text", ""))
            lines.append(f"- [유사도 {score:.2f}] [{status}] {content}...")
    else:
        lines.append("### 내부 유사 케이스: 없음 (유사도 < 0.7)")

    # ── PubMed 문헌 ──
    if external_results:
        lines.append("\n### 외부 문헌 (PubMed)")
        for e in external_results[:5]:
            lines.append(f"- [PMID: {e.get('pmid')}] {e.get('title', '')}")
            if include_abstract and e.get("abstract"):
                lines.append(f"  Abstract: {str(e['abstract'])}...")
    else:
        lines.append("\n### 외부 문헌: 없음")

    # ── 2차 비판 기반 타겟 검색 결과 ──
    critique_based = evidence.get("critique_based", {})
    critique_ext = critique_based.get("results", [])
    critique_int = critique_based.get("internal_results", [])
    if critique_ext or critique_int:
        lines.append("\n### 비판 기반 타겟 검색 (2차)")
        lines.append(f"쿼리: {critique_based.get('query', 'N/A')}")
        if critique_int:
            lines.append(f"  2차 내부 유사 케이스: {len(critique_int)}건")
            for c in critique_int[:2]:
                score = c.get("score", 0)
                content = c.get("content", c.get("text", ""))
                lines.append(f"  - [유사도 {score:.2f}] {content}...")
        for e in critique_ext[:3]:
            lines.append(f"- [PMID: {e.get('pmid')}] {e.get('title', '')}")
            if include_abstract and e.get("abstract"):
                lines.append(f"  Abstract: {str(e['abstract'])}...")

    return "\n".join(lines)


def format_clinical_analysis(evidence: Dict) -> str:
    """
    Evidence Agent의 LLM 임상 분석 결과를 프롬프트용 문자열로 포맷팅.
    """
    if not evidence:
        return "임상 분석 결과 없음"

    analysis = evidence.get("clinical_analysis", {})
    if not analysis:
        return "임상 분석 결과 없음"

    lines: List[str] = []

    priorities = analysis.get("clinical_priorities", [])
    if priorities:
        lines.append(f"임상 우선순위: {', '.join(str(p) for p in priorities[:5])}")

    findings = analysis.get("key_findings", [])
    if findings:
        lines.append(f"주요 소견: {', '.join(str(f) for f in findings[:5])}")

    risk_factors = analysis.get("risk_factors", [])
    if risk_factors:
        lines.append(f"위험 인자: {', '.join(str(r) for r in risk_factors[:5])}")

    urgency = analysis.get("urgency_level", "")
    if urgency:
        lines.append(f"긴급도: {urgency}")

    strategy = analysis.get("search_strategy", "")
    if strategy:
        lines.append(f"검색 전략: {strategy}")

    return "\n".join(lines) if lines else "임상 분석 결과 없음"


# ---------------------------------------------------------------------------
# 4. Evidence Agent 1차 (CRAG: 유사 케이스 + PubMed)
# ---------------------------------------------------------------------------

def run_evidence_agent(
    state: Dict,
    rag_retriever=None,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Dict:
    """
    Evidence Agent 실행 (하이브리드 검증 기반 CRAG)
    
    핵심 전략:
    1. 내부 RAG 검색 (유사도 >= 0.7 필터링)
    2. 품질 평가: 유효 케이스 >= 1개?
       - NO (0개) → 외부(PubMed)만 사용
       - YES (1개 이상) → LLM 검증 진행
    3. LLM 검증: 실제로 유사하고 비판/해결에 유용한가?
       - 통과 → 내부 + 외부 모두 사용 (하이브리드)
       - 실패 → 외부(PubMed)만 사용
    
    Flow:
    1. LLM으로 임상 맥락 분석
    2. 내부 RAG 검색
    3. 품질 평가 (>= 1개?)
    4. 내부 있음 → LLM 검증
       - 통과 → 내부 + 외부 (하이브리드)
       - 실패 → 외부만
    5. 내부 없음 → 바로 외부만
    """
    patient = state["patient_case"]
    similar_cases = state.get("similar_cases", [])
    structured_chart = state.get("structured_chart", {})
    
    print(f"  [Evidence Agent] Patient diagnosis: {patient.get('diagnosis')}")
    print(f"  [Evidence Agent] Similar cases provided: {len(similar_cases)}")
    
    # 1. LLM으로 임상 맥락 분석
    print(f"  [Step 1/4] Analyzing clinical context with LLM...")
    clinical_analysis = analyze_clinical_context_with_llm(patient, structured_chart)
    
    # 2. 내부 근거 수집 (유사도 필터링 적용)
    print(f"  [Step 2/4] Searching internal evidence...")
    query = generate_search_query_with_llm(patient, clinical_analysis)
    internal_results = []
    
    # query는 항상 유효한 문자열 (기본 쿼리 포함)
    if similar_cases:
        # 유사도 >= threshold인 케이스만 사용
        internal_results = [
            {
                "content": c.get("text", ""),
                "score": c.get("similarity", 0),
                "source": "internal",
                "case_id": c.get("id"),
                "status": c.get("status")
            }
            for c in similar_cases[:3]
            if c.get("similarity", 0) >= similarity_threshold
        ]
    elif rag_retriever:
        raw_results = search_internal_rag(query, rag_retriever, top_k=3)
        internal_results = [r for r in raw_results if r.get("score", 0) >= similarity_threshold]
    
    print(f"  [Internal] Found {len(internal_results)} cases above threshold {similarity_threshold}")
    
    # 3. 품질 평가
    print(f"  [Step 3/4] Evaluating quality...")
    quality = evaluate_internal_quality(internal_results, threshold=similarity_threshold)
    
    external_results = []
    retrieval_mode = ""
    final_internal = []
    validation_result = None
    
    if not quality["is_sufficient"]:
        # Case 1: 내부 없음 (0개) → 바로 외부만 사용
        print(f"  [CRAG] Internal insufficient ({quality['reason']}) → EXTERNAL ONLY")
        retrieval_mode = "external_only"
        final_internal = []
        
        # PubMed 검색 (항상 유효한 쿼리 반환)
        pubmed_query = generate_pubmed_query_with_llm(patient, clinical_analysis)
        external_results = search_pubmed(pubmed_query, max_results=5)
        print(f"  [External] Found {len(external_results)} PubMed articles")
    
    else:
        # Case 2: 내부 있음 (>= 1개) → LLM 검증 진행
        print(f"  [Step 4/4] Internal found ({quality['reason']}) → LLM validation...")
        validation_result = validate_internal_evidence_with_llm(internal_results, patient)
        
        if validation_result["is_valid"]:
            # 검증 통과 → 내부 + 외부 모두 사용 (하이브리드)
            print(f"  [LLM Validation] PASSED (confidence: {validation_result.get('confidence', 0):.2f})")
            print(f"  [LLM Validation] Reason: {validation_result.get('reason', 'N/A')}")
            print(f"  [CRAG] Internal validated → HYBRID (internal + external)")
            retrieval_mode = "hybrid"
            final_internal = validation_result.get("filtered_results", [])
            
            # 외부도 검색 (항상 유효한 쿼리 반환)
            pubmed_query = generate_pubmed_query_with_llm(patient, clinical_analysis)
            external_results = search_pubmed(pubmed_query, max_results=5)
            print(f"  [External] Found {len(external_results)} PubMed articles")
        else:
            # 검증 실패 → 외부만 사용
            print(f"  [LLM Validation] FAILED (confidence: {validation_result.get('confidence', 0):.2f})")
            print(f"  [LLM Validation] Reason: {validation_result.get('reason', 'N/A')}")
            print(f"  [CRAG] Internal not useful for critique → EXTERNAL ONLY")
            retrieval_mode = "external_only_after_validation"
            final_internal = []
            
            # PubMed 검색 (항상 유효한 쿼리 반환)
            pubmed_query = generate_pubmed_query_with_llm(patient, clinical_analysis)
            external_results = search_pubmed(pubmed_query, max_results=5)
            print(f"  [External] Found {len(external_results)} PubMed articles")
    
    # 5. 결과 반환
    evidence = {
        "retrieval_mode": retrieval_mode,
        "similarity_threshold": similarity_threshold,
        "quality_evaluation": quality,
        "validation_result": validation_result,
        "clinical_analysis": clinical_analysis,
        "internal": {
            "results": final_internal,
            "count": len(final_internal)
        },
        "external": {
            "results": external_results,
            "count": len(external_results)
        },
        "total_sources": len(final_internal) + len(external_results)
    }
    
    return {"evidence": evidence}


# ---------------------------------------------------------------------------
# 5. Evidence Agent 2차 (비판 기반 타겟 검색)
# ---------------------------------------------------------------------------

def generate_critique_based_query(patient: Dict, preliminary_issues: List[Dict]) -> str:
    """
    비판 내용 기반 PubMed 쿼리 생성
    
    Args:
        patient: 환자 정보
        preliminary_issues: Diagnosis/Treatment Agent에서 도출된 초기 이슈
    
    Returns:
        str: 비판 내용에 맞는 타겟 검색 쿼리
    """
    diagnosis = patient.get("diagnosis", "Unknown")
    
    # 기본 쿼리
    default_query = f"{diagnosis} complication management guideline"
    
    if not preliminary_issues:
        return default_query
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 이슈 요약
        issues_text = "\n".join([
            f"- [{issue.get('category', 'unknown')}] {issue.get('issue', '')}"
            for issue in preliminary_issues[:5]
        ])
        
        prompt = f"""Generate ONE PubMed query to find evidence for these clinical issues.

Patient Diagnosis: {diagnosis}

Issues Found:
{issues_text}

Goal: Find literature about these SPECIFIC issues (not general guidelines)

Rules:
1. Focus on the MOST CRITICAL issue
2. Include diagnosis + specific issue keywords
3. Add: "missed diagnosis" / "delayed diagnosis" / "complication" / "prevention" as relevant
4. 5-8 keywords total
5. Return ONLY the query string

Example:
Issues: "PE diagnosis delayed", "D-dimer not ordered"
Query: pulmonary embolism missed diagnosis pneumonia D-dimer diagnostic delay"""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        
        query = response.choices[0].message.content.strip().strip('"').strip("'")
        
        if not query or len(query.strip()) == 0:
            return default_query
        
        print(f"  [2nd Pass Query] Generated: {query}")
        return query
        
    except Exception as e:
        print(f"  [ERROR] Critique-based query generation failed: {e}")
        return default_query


def run_evidence_agent_2nd_pass(state: Dict, rag_retriever=None) -> Dict:
    """
    2차 Evidence 검색 (CRAG: 내부 RAG + PubMed, 비판 기반)

    Diagnosis/Treatment Agent의 분석 결과를 바탕으로
    비판 내용에 맞는 타겟 검색 수행 (CRAG 전략)

    Args:
        state: AgentState
        rag_retriever: 내부 RAG 검색기 (1차와 동일 인스턴스)
    """
    patient = state.get("patient_case", {})
    existing_evidence = state.get("evidence", {})

    # ── 초기 이슈 수집 (Diagnosis + Treatment Agent 결과) ──
    preliminary_issues = []

    diagnosis_analysis = state.get("diagnosis_analysis", {})
    if diagnosis_analysis:
        for issue in diagnosis_analysis.get("issues", []):
            preliminary_issues.append({
                "category": "diagnosis",
                "issue": issue.get("issue", "") if isinstance(issue, dict) else str(issue),
                "severity": issue.get("severity", "medium") if isinstance(issue, dict) else "medium"
            })
        for missed in diagnosis_analysis.get("missed_diagnoses", []):
            preliminary_issues.append({
                "category": "missed_diagnosis",
                "issue": missed.get("condition", "") if isinstance(missed, dict) else str(missed),
                "severity": "critical"
            })

    treatment_analysis = state.get("treatment_analysis", {})
    if treatment_analysis:
        for issue in treatment_analysis.get("medication_issues", []) or []:
            text = issue if isinstance(issue, str) else issue.get("issue", str(issue))
            preliminary_issues.append({
                "category": "treatment_medication",
                "issue": text,
                "severity": "medium"
            })
        for issue in treatment_analysis.get("timing_issues", []) or []:
            text = issue if isinstance(issue, str) else issue.get("issue", str(issue))
            preliminary_issues.append({
                "category": "treatment_timing",
                "issue": text,
                "severity": "medium"
            })

    # 이슈가 없으면 2차 검색 스킵
    if not preliminary_issues:
        print("  [2nd Pass] No preliminary issues found, skipping")
        return {}

    # Critical 이슈 우선 정렬
    preliminary_issues.sort(key=lambda x: 0 if x.get("severity") == "critical" else 1)

    print(f"  [2nd Pass] Found {len(preliminary_issues)} preliminary issues")
    print(f"  [2nd Pass] Top issues: {[i.get('issue', '')[:50] for i in preliminary_issues[:3]]}")

    # 비판 기반 쿼리 생성
    critique_query = generate_critique_based_query(patient, preliminary_issues)

    # === CRAG 1단계: 내부 RAG 검색 (비판 기반 쿼리) ===
    critique_internal_results: List[Dict] = []
    if rag_retriever:
        try:
            raw_results = search_internal_rag(critique_query, rag_retriever, top_k=3)
            critique_internal_results = [
                r for r in raw_results
                if r.get("score", 0) >= SIMILARITY_THRESHOLD
            ]
            print(f"  [2nd Pass CRAG] Internal: {len(critique_internal_results)} cases above threshold {SIMILARITY_THRESHOLD}")
        except Exception as e:
            print(f"  [2nd Pass CRAG] Internal RAG search failed: {e}")

    # === CRAG 2단계: PubMed 검색 ===
    critique_results = search_pubmed(critique_query, max_results=5)
    print(f"  [2nd Pass CRAG] External: {len(critique_results)} targeted PubMed articles")

    # ── 기존 evidence에 2차 검색 결과 병합 (원본 mutation 방지) ──
    updated_evidence = {
        k: v for k, v in existing_evidence.items()
        if k not in ("internal", "external", "total_sources", "critique_based")
    }

    # 2차 검색 결과 섹션 추가
    updated_evidence["critique_based"] = {
        "query": critique_query,
        "results": critique_results,
        "count": len(critique_results),
        "internal_results": critique_internal_results,
        "internal_count": len(critique_internal_results),
        "preliminary_issues": preliminary_issues[:5]
    }

    # 외부 결과 병합 (새 리스트 생성, 중복 PMID 제거)
    prev_external = existing_evidence.get("external", {}).get("results", [])
    prev_pmids = {r.get("pmid") for r in prev_external}
    new_external = [r for r in critique_results if r.get("pmid") not in prev_pmids]
    merged_external = prev_external + new_external
    updated_evidence["external"] = {
        "results": merged_external,
        "count": len(merged_external),
    }

    # 내부 결과 병합 (새 리스트 생성, 중복 case_id 제거)
    prev_internal = existing_evidence.get("internal", {}).get("results", [])
    prev_case_ids = {r.get("case_id") for r in prev_internal if r.get("case_id")}
    new_internal = [r for r in critique_internal_results if r.get("case_id") not in prev_case_ids]
    merged_internal = prev_internal + new_internal
    updated_evidence["internal"] = {
        "results": merged_internal,
        "count": len(merged_internal),
    }

    updated_evidence["total_sources"] = len(merged_internal) + len(merged_external)

    return {"evidence": updated_evidence}