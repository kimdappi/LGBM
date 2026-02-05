"""Diagnosis Agent - 진단 적절성 분석"""

from typing import Dict, List
from .llm import get_llm

SYSTEM_PROMPT = """당신은 중환자실 진단 전문의입니다. 케이스 텍스트를 꼼꼼히 읽고, 실제 존재하는 근거와 실제 문제점만 지적하세요."""

ANALYSIS_PROMPT = """
진단: {diagnosis}
구조화 소견: vitals={vitals_summary}; symptoms={symptoms_summary}; red_flags={red_flags}; PE={physical_exam}; labs/imaging={lab_imaging}
최종 결과: {outcome}

실제 케이스 텍스트 (전체 맥락 파악용):
{clinical_text_excerpt}

검색된 근거:
{evidence_summary}

## 분석 규칙 (MUST FOLLOW):

1. **텍스트에 있는 근거는 인정하라**
   - 예: asterixis 있으면 HE 근거로 인정
   - 예: 간경변 과거력 있으면 만성 간질환 근거로 인정
   - "간기능검사 없음"처럼 실제 텍스트에 있는 것을 없다고 하지 말 것

2. **Missed diagnosis는 이 환자에서 구체적 근거가 있을 때만**
   - ❌ "모든 AMS 환자에서 PE/ACS 고려해야 함" (템플릿)
   - ✅ "수술 후 + DVT sign + 저산소 → PE 감별 필요했음" (구체적 근거)
   - 근거 없이 PE/ACS/Sepsis를 나열하지 말 것

3. **이 케이스의 실제 사망/합병증 경로에 집중**
   - 환자가 실제로 어떻게 악화/사망했는지 파악
   - 그 경로에서 진단적 실패가 있었는지 분석
   - 예: "시술 합병증 출혈"이 사망 원인이면 그에 맞는 진단 이슈 분석

4. **우선순위와 근거를 붙여서 지적**
   - 각 issue/missed diagnosis에 "왜 이 환자에서" 근거 제시
   - severity: critical > medium > low

출력은 JSON만:
{{
    "diagnosis_evaluation": "적절/부적절/부분적절/근거부족",
    "issues": [
        {{"issue": "문제점 설명", "evidence_in_text": "텍스트에서 근거", "severity": "critical/medium/low"}}
    ],
    "missed_diagnoses": [
        {{"condition": "놓친 진단", "rationale": "왜 이 환자에서 고려해야 했는지 (구체적 근거)", "relevance": "실제 경과와의 관련성"}}
    ],
    "timing_assessment": "평가 내용",
    "actual_outcome_analysis": "실제 사망/합병증 경로와 진단 관련성",
    "comparison_with_similar": "유사 케이스 비교",
    "literature_support": "문헌 근거"
}}
"""


def format_evidence_summary(internal: Dict, external: Dict) -> str:
    """근거 요약 생성"""
    lines = []
    
    internal_results = internal.get("results", [])
    external_results = external.get("results", [])
    
    if internal_results:
        lines.append("### 내부 유사 케이스")
        survived = [c for c in internal_results if c.get("status") in ["alive", "survived"]]
        died = [c for c in internal_results if c.get("status") in ["dead", "died"]]
        lines.append(f"- 생존: {len(survived)}건")
        lines.append(f"- 사망: {len(died)}건")
        for c in internal_results[:2]:
            lines.append(f"- [유사도 {c.get('score', 0):.2f}] {c.get('content', '')[:150]}...")
    else:
        lines.append("### 내부 유사 케이스: 없음 (유사도 < 0.7)")
    
    if external_results:
        lines.append("\n### 외부 문헌 (PubMed)")
        for e in external_results[:3]:
            lines.append(f"- [PMID: {e.get('pmid')}] {e.get('title', '')}")
    else:
        lines.append("\n### 외부 문헌: 없음")
    
    return "\n".join(lines)


def format_structured_summary(structured: Dict) -> Dict[str, str]:
    """구조화 데이터를 요약 문자열로 변환"""
    if not structured:
        return {
            "vitals_summary": "N/A",
            "symptoms_summary": "Not documented",
            "red_flags": "None",
            "physical_exam": "N/A",
            "lab_imaging": "N/A"
        }
    
    vitals = structured.get("vitals", {})
    symptoms = structured.get("symptoms", {})
    physical = structured.get("physical_exam", {})
    lab = structured.get("laboratory", {})
    imaging = structured.get("imaging", {})
    
    # 안전하게 symptoms 리스트 합치기
    respiratory = symptoms.get("respiratory", []) if isinstance(symptoms.get("respiratory"), list) else []
    cardiovascular = symptoms.get("cardiovascular", []) if isinstance(symptoms.get("cardiovascular"), list) else []
    systemic = symptoms.get("systemic", []) if isinstance(symptoms.get("systemic"), list) else []
    all_symptoms = respiratory + cardiovascular + systemic
    
    # 안전하게 red_flags 리스트 처리
    red_flags = structured.get("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []
    
    return {
        "vitals_summary": f"SpO2 {vitals.get('oxygen_saturation', 'N/A')}%, RR {vitals.get('respiratory_rate', 'N/A')}, HR {vitals.get('heart_rate', 'N/A')}",
        "symptoms_summary": ", ".join(all_symptoms) or "Not documented",
        "red_flags": ", ".join(red_flags) or "None",
        "physical_exam": f"Lungs: {physical.get('lung_sounds', 'N/A')}, Heart: {physical.get('heart_sounds', 'N/A')}, JVD: {'Yes' if physical.get('jvd_present') else 'No'}",
        "lab_imaging": f"CXR: {imaging.get('chest_xray', 'N/A')}, WBC: {lab.get('wbc', 'N/A')}, BNP: {lab.get('bnp', 'N/A')}"
    }


def run_diagnosis_agent(state: Dict) -> Dict:
    """진단 에이전트 실행 - 구조화 데이터 + 근거 사용"""
    patient = state.get("patient_case", {})
    if not patient:
        return {"diagnosis_analysis": {"diagnosis_evaluation": "입력 데이터 없음", "issues": [], "missed_diagnoses": []}}
    
    evidence = state.get("evidence", {})
    structured = state.get("structured_chart", {})
    
    print(f"  [Diagnosis Agent] clinical_text length: {len(patient.get('clinical_text', ''))} chars")
    print(f"  [Diagnosis Agent] Using structured data")
    
    internal = evidence.get("internal", {})
    external = evidence.get("external", {})
    evidence_summary = format_evidence_summary(internal, external)
    
    # 구조화 데이터 사용 (무조건)
    summaries = format_structured_summary(structured)
    
    # 케이스 텍스트 발췌 (전체 맥락 파악용)
    clinical_text = patient.get("clinical_text", "") or patient.get("text", "")
    clinical_text_excerpt = clinical_text[:3000] if clinical_text else "N/A"
    
    # Outcome 정보
    outcome_info = structured.get("outcome", {}) if structured else {}
    outcome = f"status={outcome_info.get('status', patient.get('status', 'unknown'))}"
    if outcome_info.get("disposition"):
        outcome += f", disposition={outcome_info.get('disposition')}"
    
    prompt = ANALYSIS_PROMPT.format(
        diagnosis=patient.get("diagnosis", "Unknown"),
        vitals_summary=summaries["vitals_summary"],
        symptoms_summary=summaries["symptoms_summary"],
        red_flags=summaries["red_flags"],
        physical_exam=summaries["physical_exam"],
        lab_imaging=summaries["lab_imaging"],
        outcome=outcome,
        clinical_text_excerpt=clinical_text_excerpt,
        evidence_summary=evidence_summary
    )
    
    try:
        llm = get_llm()
        response = llm.gpt4o(prompt, system=SYSTEM_PROMPT, json_mode=True, timeout=60)
        
        import json
        try:
            analysis = json.loads(response)
            
            # 필수 필드 검증
            if "diagnosis_evaluation" not in analysis:
                analysis["diagnosis_evaluation"] = "근거부족"
            if "issues" not in analysis:
                analysis["issues"] = []
            if "missed_diagnoses" not in analysis:
                analysis["missed_diagnoses"] = []
        except json.JSONDecodeError as e:
            print(f"  [Diagnosis Agent] JSON parsing failed: {e}")
            analysis = {
                "diagnosis_evaluation": "파싱 실패",
                "issues": [],
                "missed_diagnoses": [],
                "raw_response": response[:500]
            }
        
        return {"diagnosis_analysis": analysis}
        
    except Exception as e:
        print(f"  [Diagnosis Agent] Error: {e}")
        return {
            "diagnosis_analysis": {
                "diagnosis_evaluation": "실행 실패",
                "issues": [],
                "missed_diagnoses": [],
                "error": str(e)
            }
        }
