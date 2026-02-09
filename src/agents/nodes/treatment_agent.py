"""Treatment Agent - 치료 적절성 분석"""

from typing import Dict, List
from ..llm import get_llm

SYSTEM_PROMPT = """당신은 중환자실 치료 전문의입니다. 시행된 치료를 확인한 뒤 치료/처치의 적절성(선택·용량·타이밍)과 disposition을 근거 기반으로 평가하세요."""

ANALYSIS_PROMPT = """
## 환자 정보 (구조화)
진단: {diagnosis}

## 임상 상태
퇴원 결정: {disposition}
상태: vitals={vitals_summary}; 경과={clinical_course}

이미 시행된 치료(반드시 확인):
{interventions_given}

근거(요약):
{evidence_summary}

요청:
1) 치료/처치 적절성: 선택·용량·타이밍·가이드라인 준수. **시행된 치료를 "없었다"라고 쓰지 말고** 적절성/개선점으로 평가.
2) Disposition 평가:
   - 환자 expired면: 조기퇴원/입원권고 금지. 대신 치료 과정의 문제(합병증/지연/부적절 약물/모니터링 실패) 중심으로 분석하고 `is_appropriate`는 N/A로 표기.
   - 생존 케이스에서 고위험(저산소/지속 빈맥·빈호흡/치명적 감별 미배제/진단 불명확) 조기퇴원은 부적절로 평가.

출력은 JSON만:
{{
    "treatment_evaluation": "적절/부적절/부분적절/근거부족",
    "medication_issues": ["약물 관련 문제 (부재 아닌 적절성 평가)"],
    "timing_issues": ["타이밍 관련 문제"],
    "guideline_adherence": "가이드라인 준수 평가",
    "disposition_evaluation": {{
        "is_appropriate": true/false,
        "risk_level": "low/medium/high/critical",
        "concern": "퇴원 결정의 문제점 (있는 경우)",
        "recommendation": "입원/관찰/조건부 퇴원 권고"
    }},
    "recommendations": ["추가/보완 권고사항"]
}}
"""


def format_evidence_summary(internal: Dict, external: Dict) -> str:
    """근거 요약 생성"""
    lines = []
    
    internal_results = internal.get("results", [])
    external_results = external.get("results", [])
    
    if internal_results:
        lines.append("### 내부 유사 케이스")
        for c in internal_results[:3]:
            lines.append(f"- [유사도 {c.get('score', 0):.2f}] [{c.get('status')}] {c.get('content', '')[:150]}...")
    else:
        lines.append("### 내부 유사 케이스: 없음 (유사도 < 0.7)")
    
    if external_results:
        lines.append("\n### 외부 문헌 (PubMed)")
        for e in external_results[:3]:
            lines.append(f"- [PMID: {e.get('pmid')}] {e.get('title', '')}")
            if e.get("abstract"):
                lines.append(f"  {e.get('abstract', '')[:200]}...")
    else:
        lines.append("\n### 외부 문헌: 없음")
    
    return "\n".join(lines)


def format_interventions_given(structured: Dict) -> str:
    """시행된 치료 포맷팅"""
    if not structured:
        return "**시행된 치료 정보 없음**"
    
    interventions = structured.get("interventions_given", {})
    if not interventions:
        return "**시행된 치료 정보 없음**"
    
    lines = []
    
    # 약물
    meds = interventions.get("medications", [])
    if meds and isinstance(meds, list):
        lines.append("**약물:**")
        for m in meds:
            if isinstance(m, dict):
                name = m.get('name', 'Unknown')
                timing = m.get('timing', 'N/A')
                route = m.get('route', 'N/A')
                lines.append(f"  - {name} (시점: {timing}, 경로: {route})")
            elif isinstance(m, str):
                lines.append(f"  - {m}")
    else:
        lines.append("**약물:** 기록 없음")
    
    # 산소 치료
    o2 = interventions.get("oxygen_therapy", [])
    if o2 and isinstance(o2, list):
        lines.append("**산소 치료:**")
        for o in o2:
            if isinstance(o, dict):
                o_type = o.get('type', 'Unknown')
                timing = o.get('timing', 'N/A')
                lines.append(f"  - {o_type} (시점: {timing})")
            elif isinstance(o, str):
                lines.append(f"  - {o}")
    else:
        lines.append("**산소 치료:** 기록 없음")
    
    # 수액
    fluids = interventions.get("fluids")
    if fluids:
        lines.append(f"**수액:** {fluids}")
    
    return "\n".join(lines)


def run_treatment_agent(state: Dict) -> Dict:
    """치료 에이전트 실행 - 구조화 데이터 + 시행된 치료 명시"""
    patient = state["patient_case"]
    evidence = state.get("evidence", {})
    structured = state.get("structured_chart", {})
    
    print(f"  [Treatment Agent] Using structured data")
    
    internal = evidence.get("internal", {})
    external = evidence.get("external", {})
    evidence_summary = format_evidence_summary(internal, external)
    
    # 구조화 데이터 사용 (무조건)
    vitals = structured.get("vitals", {})
    vitals_summary = f"SpO2 {vitals.get('oxygen_saturation', 'N/A')}%, O2: {vitals.get('oxygen_requirement', 'N/A')}"
    
    course = structured.get("clinical_course", {})
    clinical_course = f"호전: {'예' if course.get('improvement') else '아니오'}, 산소 추세: {course.get('oxygen_trend', 'N/A')}"
    
    interventions_given = format_interventions_given(structured)
    
    # Disposition 정보 추출 (Chart Structurer 결과 사용)
    outcome = structured.get("outcome", {})
    disposition_status = outcome.get("disposition", "Unknown")
    discharge_location = patient.get("discharge_location", "Unknown")
    disposition = f"{disposition_status} ({discharge_location})"
    
    prompt = ANALYSIS_PROMPT.format(
        diagnosis=patient.get("diagnosis", "Unknown"),
        disposition=disposition,
        vitals_summary=vitals_summary,
        clinical_course=clinical_course,
        interventions_given=interventions_given,
        evidence_summary=evidence_summary
    )
    
    try:
        llm = get_llm()
        response = llm.gpt4o(prompt, system=SYSTEM_PROMPT, json_mode=True, timeout=60)
        
        import json
        try:
            analysis = json.loads(response)
            
            # 필수 필드 검증
            if "treatment_evaluation" not in analysis:
                analysis["treatment_evaluation"] = "근거부족"
            if "disposition_evaluation" not in analysis:
                analysis["disposition_evaluation"] = {
                    "is_appropriate": True,
                    "risk_level": "low",
                    "concern": "평가 불가",
                    "recommendation": "N/A"
                }
        except json.JSONDecodeError as e:
            print(f"  [Treatment Agent] JSON parsing failed: {e}")
            analysis = {
                "treatment_evaluation": "파싱 실패",
                "medication_issues": [],
                "timing_issues": [],
                "raw_response": response[:500]
            }
        
        return {"treatment_analysis": analysis}
        
    except Exception as e:
        print(f"  [Treatment Agent] Error: {e}")
        return {
            "treatment_analysis": {
                "treatment_evaluation": "실행 실패",
                "medication_issues": [],
                "timing_issues": [],
                "error": str(e)
            }
        }
