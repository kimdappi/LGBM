"""Critic Agent - 최종 검증 및 종합 (GPT-4o)"""

from typing import Dict, List
from .llm import get_llm

SYSTEM_PROMPT = """당신은 의료 품질 관리 전문가입니다. 입력된 분석/근거만 바탕으로 최종 critique와 실행 가능한 구체적 해결책을 JSON으로 산출하세요."""

SYNTHESIS_PROMPT = """
환자 요약:
- Diagnosis: {diagnosis}
- Text: {patient_summary}
- Outcome(structured): {structured_outcome}

이미 시행된 치료:
{intervention_coverage}

진단 분석:
{diagnosis_analysis}

치료/Disposition 분석:
{treatment_analysis}

근거:
{evidence_summary}

Evidence quality:
{evidence_quality}

Reflexion memory:
{memory}

## 규칙:

### Critique 규칙:
- 임상 위험도에 맞춰 severity 부여 (critical/medium/low)
- expired 케이스면 조기퇴원/입원권고 비판 금지. 사망원인/타임라인/의인성/시스템 실패 중심.
- Evidence retrieval이 무관하면 evidence_quality 카테고리로 명시.
- Citation: 근거 섹션에 제공된 PMID 또는 명시된 guideline명만 사용. 없으면 "N/A".

### Solutions 규칙 (MUST FOLLOW):
- **구체적 실행 방안 필수**: "중단하고 대체" 대신 "어떤 대체 옵션이 있는지" 명시
- **Multi-step action 포함**: 단순 약물 변경이 아니라 유발요인 교정 등 전체 접근법

예시 - 간경변 환자에서 벤조 사용 문제:
❌ "Lorazepam 중단 및 Lactulose 사용"
✅ "1) Lorazepam 즉시 중단, 2) HE 유발요인 교정(감염 R/O, GI 출혈 R/O, 전해질 교정, 변비 해결, 약물 검토), 3) 불면증 필요 시 대체 전략: 저용량 trazodone 또는 비약물적 접근(수면위생)"

예시 - 간경변+HRS에서 NSAID 대체:
❌ "Ketorolac 대체 진통제 사용"
✅ "1) Ketorolac 즉시 중단 (HRS 악화 위험), 2) 대체 진통제 옵션: a) 제한적 acetaminophen (≤2g/day), b) 필요시 저용량 opioid (tramadol 25mg PRN), c) 국소 치료 (lidocaine patch), 3) 통증 원인 평가 및 근본 치료"

- 출력은 JSON만. critique_points 3-6개, solutions 3-6개.

JSON 형식:
{{
    "critique_points": [
        {{
            "issue": "문제점 (구체적으로)",
            "severity": "critical/medium/low",
            "category": "diagnosis/treatment/procedure/disposition/evidence_quality/timing/monitoring",
            "evidence_support": "근거 설명"
        }}
    ],
    "solutions": [
        {{
            "target_issue": "해결할 문제",
            "action": "구체적 multi-step 조치 (1단계, 2단계, 대체 옵션 등 포함)",
            "specific_alternatives": "대체 옵션이 있는 경우 구체적 약물/용량/방법 나열",
            "precipitant_correction": "유발요인 교정이 필요한 경우 체크리스트 (감염/출혈/전해질/변비/약물 등)",
            "rationale": "이유",
            "citation": "근거 출처 (PMID 또는 가이드라인명)",
            "priority": "immediate/short-term/long-term"
        }}
    ],
    "overall_assessment": "종합 평가",
    "confidence_score": 0.0-1.0,
    "limitations": "분석의 한계점 (근거 부족 등)"
}}
"""


def format_evidence(evidence: Dict) -> str:
    """근거 요약 포맷팅 (CRAG 형식)"""
    if not evidence:
        return "수집된 근거 없음"
    
    lines = []
    
    # CRAG 모드 표시
    mode = evidence.get("retrieval_mode", "unknown")
    quality = evidence.get("quality_evaluation", {})
    lines.append(f"[검색 모드: {mode}]")
    lines.append(f"[내부 근거 품질: avg_score={quality.get('avg_score', 0)}, threshold={evidence.get('similarity_threshold', 0.7)}]")
    
    # 내부 근거
    internal = evidence.get("internal", {})
    if internal.get("results"):
        lines.append(f"\n[내부 유사 케이스 ({internal.get('count', 0)}건)]")
        for r in internal["results"][:3]:
            score = r.get("score", 0)
            status = r.get("status", "unknown")
            lines.append(f"  - [유사도: {score:.2f}] [{status}] {r.get('content', '')[:200]}...")
    
    # 외부 근거 (CRAG로 보강된 경우)
    external = evidence.get("external", {})
    if external.get("results"):
        triggered = "CRAG 보강" if external.get("triggered") else "추가 검색"
        lines.append(f"\n[PubMed ({triggered}, {external.get('count', 0)}건)]")
        for r in external["results"][:3]:
            lines.append(f"  - [PMID: {r.get('pmid', '')}] {r.get('title', '')}")
    
    return "\n".join(lines) if lines else "근거 없음"


def format_intervention_coverage(coverage: Dict) -> str:
    """시행된 치료 확인 결과 포맷팅"""
    if not coverage:
        return "**시행된 치료 정보 없음**"
    
    cov = coverage.get("coverage", {})
    if not cov:
        return "**시행된 치료 카테고리 정보 없음**"
    
    lines = []
    lines.append("**시행된 치료 카테고리:**")
    lines.append(f"  - Bronchodilator: {'[OK] Done' if cov.get('bronchodilator') else '[X] Not done'}")
    lines.append(f"  - Corticosteroid: {'[OK] Done' if cov.get('corticosteroid') else '[X] Not done'}")
    lines.append(f"  - Antibiotic: {'[OK] Done' if cov.get('antibiotic') else '[X] Not done'}")
    lines.append(f"  - Oxygen support: {'[OK] Done' if cov.get('oxygen_support') else '[X] Not done'}")
    
    blocked = coverage.get("blocked_count", 0)
    if blocked > 0:
        lines.append(f"\n[WARN] {blocked} 'absence' critiques blocked (already performed)")
    
    return "\n".join(lines)


def run_critic_agent(state: Dict) -> Dict:
    """Critic 에이전트 실행 (GPT-4o 사용) - Intervention Coverage + Evidence Quality 반영"""
    patient = state["patient_case"]
    intervention_coverage = state.get("intervention_coverage", {})
    evidence = state.get("evidence", {})
    
    # Evidence quality 평가
    evidence_quality_lines = []
    retrieval_mode = evidence.get("retrieval_mode", "unknown")
    quality_eval = evidence.get("quality_evaluation", {})
    external_results = evidence.get("external", {}).get("results", [])
    
    evidence_quality_lines.append(f"**Retrieval Mode:** {retrieval_mode}")
    evidence_quality_lines.append(f"**Internal Cases:** {quality_eval.get('count', 0)}건")
    evidence_quality_lines.append(f"**Quality Reason:** {quality_eval.get('reason', 'N/A')}")
    
    # 외부 문헌 관련성 체크
    if retrieval_mode == "external_only" and external_results:
        titles = [r.get("title", "").lower() for r in external_results[:3]]
        # 무관한 키워드 체크
        irrelevant_keywords = ["crohn", "h. pylori", "helicobacter", "cat", "feline", "colitis", "gastroenterology"]
        found_irrelevant = [k for k in irrelevant_keywords if any(k in t for t in titles)]
        
        if found_irrelevant:
            evidence_quality_lines.append(f"[ALERT] **Irrelevant literature detected** - {', '.join(found_irrelevant)}")
            evidence_quality_lines.append("   -> High probability of evidence retrieval failure")
    
    evidence_quality = "\n".join(evidence_quality_lines)
    
    # Structured chart에서 outcome 정보 추출
    structured_chart = state.get("structured_chart", {})
    
    if not structured_chart:
        structured_outcome = "**Outcome 정보 없음**"
    else:
        outcome = structured_chart.get("outcome", {})
        if not outcome:
            structured_outcome = "**Outcome 정보 없음**"
        else:
            structured_outcome_lines = []
            structured_outcome_lines.append(f"**Status:** {outcome.get('status', 'unknown')}")
            structured_outcome_lines.append(f"**Discharge Condition:** {outcome.get('discharge_condition', 'unknown')}")
            structured_outcome_lines.append(f"**Discharge Location:** {outcome.get('discharge_location', 'unknown')}")
            
            if outcome.get('cause_of_death'):
                structured_outcome_lines.append(f"**🚨 Cause of Death:** {outcome.get('cause_of_death')}")
            
            critical_events = outcome.get('critical_events_leading_to_outcome', [])
            if critical_events and isinstance(critical_events, list):
                structured_outcome_lines.append(f"**🚨 Critical Events:**")
                for event in critical_events:
                    if event:  # None이나 빈 문자열 체크
                        structured_outcome_lines.append(f"   - {event}")
            
            structured_outcome = "\n".join(structured_outcome_lines)
    
    prompt = SYNTHESIS_PROMPT.format(
        diagnosis=patient.get("diagnosis", "Unknown"),
        patient_summary=patient.get("clinical_text", ""),
        structured_outcome=structured_outcome,
        intervention_coverage=format_intervention_coverage(intervention_coverage),
        diagnosis_analysis=str(state.get("diagnosis_analysis", {})),
        treatment_analysis=str(state.get("treatment_analysis", {})),
        evidence_summary=format_evidence(evidence),
        evidence_quality=evidence_quality,
        memory="\n".join(state.get("memory", [])) or "없음"
    )
    
    try:
        llm = get_llm()
        response = llm.gpt4o(prompt, system=SYSTEM_PROMPT, json_mode=True, timeout=60)
        
        # JSON 파싱
        import json
        import re
        
        # JSON 블록 추출
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
                
                # 필수 필드 검증
                critique_points = result.get("critique_points", [])
                solutions = result.get("solutions", [])
                confidence = result.get("confidence_score", 0.5)
                
                # 타입 검증
                if not isinstance(critique_points, list):
                    critique_points = []
                if not isinstance(solutions, list):
                    solutions = []
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                
                return {
                    "critique": critique_points,
                    "solutions": solutions,
                    "confidence": confidence
                }
            except json.JSONDecodeError as e:
                print(f"  [Critic Agent] JSON 파싱 오류: {e}")
                print(f"  [Critic Agent] Response sample: {response[:300]}...")
        else:
            print(f"  [Critic Agent] JSON 블록을 찾을 수 없음")
            print(f"  [Critic Agent] Response sample: {response[:300]}...")
        
        return {
            "critique": [{"issue": "JSON 파싱 실패", "severity": "low", "category": "system_error"}],
            "solutions": [],
            "confidence": 0.0
        }
        
    except Exception as e:
        print(f"  [Critic Agent] 실행 오류: {e}")
        return {
            "critique": [{"issue": f"Critic Agent 실행 실패: {str(e)}", "severity": "low", "category": "system_error"}],
            "solutions": [],
            "confidence": 0.0
        }