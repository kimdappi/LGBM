"""Intervention Coverage Checker - 이미 시행된 치료 확인"""

from typing import Dict, List


def check_intervention_coverage(state: Dict) -> Dict:
    """
    구조화된 차트에서 이미 시행된 치료를 확인
    - Diagnosis/Treatment Agent가 제안한 비판/해결책 중 중복 제거
    - "치료 부재"류 비판을 "적절성 평가"로 전환
    """
    structured = state.get("structured_chart", {})
    diagnosis_analysis = state.get("diagnosis_analysis", {})
    treatment_analysis = state.get("treatment_analysis", {})
    
    print(f"  [Intervention Checker] Checking interventions from structured chart")
    
    # 1. 시행된 치료 목록 추출
    interventions_given = structured.get("interventions_given", {})
    
    # 안전하게 medications 추출 (name이 없는 경우 대비)
    medications_raw = interventions_given.get("medications", [])
    medications_given = []
    for m in medications_raw:
        if isinstance(m, dict) and "name" in m:
            medications_given.append(m["name"].lower())
        elif isinstance(m, str):
            medications_given.append(m.lower())
    
    # 안전하게 oxygen_therapy 추출 (type이 없는 경우 대비)
    oxygen_raw = interventions_given.get("oxygen_therapy", [])
    oxygen_given = []
    for o in oxygen_raw:
        if isinstance(o, dict) and "type" in o:
            oxygen_given.append(o["type"])
        elif isinstance(o, str):
            oxygen_given.append(o)
    
    # 2. 주요 치료 카테고리 체크
    coverage = {
        "bronchodilator": any(
            keyword in " ".join(medications_given) 
            for keyword in ["albuterol", "duoneb", "bronchodilator", "beta-agonist"]
        ),
        "corticosteroid": any(
            keyword in " ".join(medications_given)
            for keyword in ["steroid", "prednisone", "methylprednisolone", "dexamethasone"]
        ),
        "antibiotic": any(
            keyword in " ".join(medications_given)
            for keyword in ["antibiotic", "ceftriaxone", "azithromycin", "levofloxacin", "cefepime"]
        ),
        "diuretic": any(
            keyword in " ".join(medications_given)
            for keyword in ["lasix", "furosemide", "diuretic"]
        ),
        "oxygen_support": len(oxygen_given) > 0,
        "niv": any("niv" in o.lower() or "bipap" in o.lower() for o in oxygen_given),
        "all_medications": medications_given,
        "all_oxygen": oxygen_given
    }
    
    # 3. 비판 풀 구성: Diagnosis + Treatment 에이전트 이슈 (docstring: "Diagnosis/Treatment Agent가 제안한 비판 중 중복 제거")
    filtered_issues = []
    
    medication_issues = treatment_analysis.get("medication_issues", []) or []
    timing_issues = treatment_analysis.get("timing_issues", []) or []
    diagnosis_issues_raw = diagnosis_analysis.get("issues", []) or []
    
    # Treatment: 문자열 리스트
    treatment_issue_texts = []
    for x in medication_issues + timing_issues:
        treatment_issue_texts.append(x if isinstance(x, str) else (x.get("issue", "") if isinstance(x, dict) else str(x)))
    
    # Diagnosis: issues[] 항목에서 issue 텍스트만 추출
    diagnosis_issue_texts = []
    for item in diagnosis_issues_raw:
        if isinstance(item, dict):
            diagnosis_issue_texts.append(item.get("issue", "") or "")
        elif isinstance(item, str):
            diagnosis_issue_texts.append(item)
        else:
            diagnosis_issue_texts.append(str(item))
    
    original_issues = [t for t in treatment_issue_texts + diagnosis_issue_texts if t and t.strip()]
    
    for issue in original_issues:
        issue_lower = issue.lower()
        
        # "부재", "없음", "미시행", "not given" 등 키워드 체크
        is_absence_claim = any(
            keyword in issue_lower
            for keyword in ["부재", "없음", "미시행", "not given", "not administered", "missing"]
        )
        
        # 이미 시행된 치료에 대한 "부재" 비판인지 확인
        already_given = False
        if is_absence_claim:
            if "bronchodilator" in issue_lower and coverage["bronchodilator"]:
                already_given = True
            elif "steroid" in issue_lower and coverage["corticosteroid"]:
                already_given = True
            elif "antibiotic" in issue_lower and coverage["antibiotic"]:
                already_given = True
            elif "oxygen" in issue_lower and coverage["oxygen_support"]:
                already_given = True
        
        if not already_given:
            filtered_issues.append(issue)
        else:
            print(f"  [Intervention Checker] 차단: '{issue}' (이미 시행됨)")
    
    result = {
        "coverage": coverage,
        "filtered_issues": filtered_issues,
        "blocked_count": len(original_issues) - len(filtered_issues)
    }
    
    print(f"  [Intervention Checker] 완료")
    print(f"    - Performed treatments: {sum(1 for v in coverage.values() if isinstance(v, bool) and v)} categories")
    print(f"    - Blocked critiques: {result['blocked_count']} items")
    
    return {"intervention_coverage": result}
