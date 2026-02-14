"""Diagnosis Agent - 진단 적절성 분석

강화 로직:
  1. Procedural Safety Check: 시술에 안전 수식어 동반 여부
  2. Cause of Death Alignment: 입원 사유 ≠ 사망 원인 시 마지막 24h 가중
  3. Severity Hierarchy: Iatrogenic trauma > medication error
"""

import re
from typing import Dict, List, Optional
from .llm import get_llm


# ──────────────────────────────────────────────
# Rule-based: Procedural Safety Check
# ──────────────────────────────────────────────

# 시술 키워드 (대소문자 무시)
_PROCEDURE_KEYWORDS = [
    "paracentesis", "thoracentesis", "central line", "central venous",
    "chest tube", "thoracostomy", "lumbar puncture", "LP",
    "arterial line", "art line", "a-line",
    "intubation", "tracheostomy", "pericardiocentesis",
    "percutaneous drain", "pigtail catheter", "PICC",
    "feeding tube", "NG tube placement", "dialysis catheter",
]

# 안전 수식어 (있어야 하는 것)
_SAFETY_MODIFIERS = [
    "ultrasound.?guided", "US.?guided", "ultra.?sound",
    "fluoroscop", "CT.?guided", "image.?guided",
    "sterile", "aseptic", "full barrier",
]

# 위험 수식어 (있으면 즉시 Red Flag)
_DANGER_MODIFIERS = [
    r"\bblind(?:ly)?\b", r"without\s+(?:ultrasound|imaging|guidance)",
    r"landmark.?based",
]


def detect_procedural_safety_issues(clinical_text: str) -> List[Dict]:
    """
    텍스트에서 시술 키워드를 찾고, 안전 수식어 동반 여부를 확인.
    
    Returns:
        [{"procedure": str, "has_safety": bool, "danger_flag": str|None,
          "context": str}]
    """
    if not clinical_text:
        return []
    
    text_lower = clinical_text.lower()
    findings = []
    
    for proc in _PROCEDURE_KEYWORDS:
        # 시술 키워드 위치 찾기
        pattern = re.compile(re.escape(proc.lower()), re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text_lower), match.end() + 200)
            context = clinical_text[start:end]
            context_lower = context.lower()
            
            # 안전 수식어 체크
            has_safety = any(
                re.search(mod, context_lower, re.IGNORECASE)
                for mod in _SAFETY_MODIFIERS
            )
            
            # 위험 수식어 체크
            danger_flag = None
            for danger in _DANGER_MODIFIERS:
                if re.search(danger, context_lower, re.IGNORECASE):
                    danger_flag = re.search(danger, context_lower, re.IGNORECASE).group()
                    break
            
            findings.append({
                "procedure": proc,
                "has_safety": has_safety,
                "danger_flag": danger_flag,
                "context": context.strip()[:300],
            })
    
    # 중복 제거 (같은 시술 여러 번 잡힐 수 있음)
    seen = set()
    unique = []
    for f in findings:
        key = f["procedure"]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    
    return unique


# ──────────────────────────────────────────────
# Rule-based: Cause of Death Alignment
# ──────────────────────────────────────────────

def detect_death_cause_mismatch(
    patient: Dict,
    structured: Dict,
) -> Optional[Dict]:
    """
    입원 사유와 사망 원인이 다른 경우를 감지.
    
    Returns:
        {"admission_reason": str, "death_cause": str, "critical_events": list,
         "mismatch": bool, "iatrogenic": bool} or None
    """
    if not structured:
        return None
    
    outcome = structured.get("outcome", {})
    status = (outcome.get("status") or patient.get("outcome") or "").lower()
    
    if "expir" not in status and "dead" not in status and "died" not in status:
        return None  # 생존 케이스는 해당 없음
    
    admission_reason = (
        patient.get("diagnosis", "")
        or structured.get("demographics", {}).get("chief_complaint", "")
    )
    death_cause = outcome.get("cause_of_death") or ""
    critical_events = outcome.get("critical_events_leading_to_outcome") or []
    
    if not death_cause and not critical_events:
        return None
    
    # 입원 사유와 사망 원인 비교 (단순 포함 관계)
    admission_lower = admission_reason.lower()
    death_lower = death_cause.lower()
    
    mismatch = True
    if admission_lower and death_lower:
        # 키워드 겹침 체크
        admission_words = set(re.findall(r'\w{4,}', admission_lower))
        death_words = set(re.findall(r'\w{4,}', death_lower))
        overlap = admission_words & death_words
        if len(overlap) >= 1:
            mismatch = False
    elif not death_cause:
        mismatch = False  # 사망 원인이 명시 안 되면 판단 불가
    
    # Iatrogenic 키워드 체크
    iatrogenic_keywords = [
        "iatrogenic", "procedur", "complication", "puncture",
        "hemoperitoneum", "hemorrhage from", "bleeding from",
        "perforation", "laceration", "organ injury",
    ]
    iatrogenic = any(
        kw in death_lower or any(kw in str(ev).lower() for ev in critical_events)
        for kw in iatrogenic_keywords
    )
    
    return {
        "admission_reason": admission_reason,
        "death_cause": death_cause,
        "critical_events": critical_events,
        "mismatch": mismatch,
        "iatrogenic": iatrogenic,
    }

SYSTEM_PROMPT = """당신은 중환자실 진단 전문의입니다. 케이스 텍스트를 꼼꼼히 읽고, 실제 존재하는 근거와 실제 문제점만 지적하세요."""

ANALYSIS_PROMPT = """
진단: {diagnosis}
구조화 소견: vitals={vitals_summary}; symptoms={symptoms_summary}; red_flags={red_flags}; PE={physical_exam}; labs/imaging={lab_imaging}
최종 결과: {outcome}

시술 정보: {procedures_info}

사망 원인 정렬: {death_alignment_info}

시술 안전성 사전 검사 결과: {procedural_safety_findings}

실제 케이스 텍스트 (전체 맥락 파악용):
{clinical_text_excerpt}

검색된 근거:
{evidence_summary}

과거 유사 케이스 분석 경험:
{episodic_lessons}

## 분석 규칙 (MUST FOLLOW):

1. **텍스트에 있는 근거는 인정하라**
   - 예: asterixis 있으면 HE 근거로 인정
   - "간기능검사 없음"처럼 실제 텍스트에 있는 것을 없다고 하지 말 것

2. **Missed diagnosis는 이 환자에서 구체적 근거가 있을 때만**
   - ❌ "모든 AMS 환자에서 PE/ACS 고려해야 함" (템플릿)
   - ✅ "수술 후 + DVT sign + 저산소 → PE 감별 필요했음" (구체적 근거)

3. **★ Procedural Safety Check (최우선):**
   - 시술(Paracentesis, Central Line, Thoracentesis, Chest Tube, LP 등)이 있으면:
     a) "Ultrasound-guided" 또는 "sterile technique" 수식어가 동반되었는지 확인
     b) "Blind/Blindly" → **즉시 CRITICAL Red Flag** (iatrogenic 위험)
     c) 시술 후 합병증(bleeding, Hct drop, organ injury)이 발생했으면 → CRITICAL
   - 시술 안전성 문제가 약물 오류보다 **항상 우선순위 높음**
   - 시술 안전성 사전 검사 결과를 참고하되, 텍스트 전체 맥락도 반드시 확인

4. **★ Cause of Death Alignment (사망 원인 정렬):**
   - 환자가 사망한 경우, 입원 사유(admission reason)와 실제 사망 원인(cause of death)을 비교
   - **두 원인이 다를 경우**: 마지막 24시간의 이벤트(Hct drop, unstable vitals, 시술 합병증)에 가중치를 두어 사망 원인 추론
   - 예: 입원은 HE(hepatic encephalopathy)인데 사망은 시술 후 출혈 → "사망 경로는 HE가 아닌 iatrogenic hemoperitoneum"
   - 입원 사유만 분석하고 실제 사망 원인을 놓치면 안 됨

5. **★ Severity Hierarchy (중증도 위계):**
   - Priority 1: **Iatrogenic trauma/organ injury/출혈** (시술로 인한 장기 손상, 합병증 출혈)
   - Priority 2: **사망 직접 원인에 기여한 진단/치료 실패**
   - Priority 3: **약물 오류** (Benzos, NSAID 등 부적절 약물)
   - Priority 4: 기타 진단/치료 이슈
   - Iatrogenic trauma가 있으면 반드시 issues의 **첫 번째**로 배치하고 severity="critical"

6. **과거 유사 케이스 교훈이 있으면 반드시 참고**

출력은 JSON만:
{{
    "diagnosis_evaluation": "적절/부적절/부분적절/근거부족",
    "issues": [
        {{"issue": "문제점 설명", "evidence_in_text": "텍스트에서 근거", "severity": "critical/medium/low", "category": "iatrogenic_trauma/procedural_safety/death_alignment/medication_error/diagnostic_failure/other"}}
    ],
    "missed_diagnoses": [
        {{"condition": "놓친 진단", "rationale": "왜 이 환자에서 고려해야 했는지", "relevance": "실제 경과와의 관련성"}}
    ],
    "procedural_safety_assessment": {{
        "procedures_found": ["시술 목록"],
        "safety_concerns": ["안전 문제"],
        "overall": "safe/unsafe/not_documented"
    }},
    "death_cause_alignment": {{
        "admission_reason": "입원 사유",
        "actual_death_cause": "실제 사망 원인",
        "mismatch": true/false,
        "last_24h_key_events": ["마지막 24시간 핵심 이벤트"],
        "primary_death_pathway": "사망 경로 설명"
    }},
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


# ──────────────────────────────────────────────
# Severity Hierarchy 재정렬
# ──────────────────────────────────────────────

_SEVERITY_PRIORITY = {
    "iatrogenic_trauma": 0,
    "procedural_safety": 1,
    "death_alignment": 2,
    "medication_error": 3,
    "diagnostic_failure": 4,
    "other": 5,
}


def _rerank_issues_by_severity_hierarchy(issues: List[Dict]) -> List[Dict]:
    """
    Severity Hierarchy에 따라 issues를 재정렬.
    
    Priority:
      0. Iatrogenic trauma/organ injury
      1. Procedural safety violations
      2. Death cause alignment issues
      3. Medication errors
      4. Diagnostic failures
      5. Other
    
    같은 카테고리 내에서는 severity (critical > medium > low) 순.
    """
    severity_order = {"critical": 0, "medium": 1, "low": 2}
    
    def sort_key(issue: Dict):
        cat = issue.get("category", "other")
        cat_priority = _SEVERITY_PRIORITY.get(cat, 5)
        sev = severity_order.get(issue.get("severity", "low"), 2)
        
        # 텍스트에서 iatrogenic 키워드가 있으면 카테고리 무관하게 최우선
        text = (issue.get("issue", "") + issue.get("evidence_in_text", "")).lower()
        iatrogenic_keywords = ["iatrogenic", "hemoperitoneum", "organ injury", "puncture.*bleed", "procedur.*complication"]
        if any(re.search(kw, text) for kw in iatrogenic_keywords):
            cat_priority = 0
            if issue.get("severity") != "critical":
                issue["severity"] = "critical"
        
        return (cat_priority, sev)
    
    return sorted(issues, key=sort_key)


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
    
    # 에피소딕 메모리 교훈 (과거 유사 케이스 경험)
    episodic_lessons = state.get("episodic_lessons", "") or "없음"
    
    # ── Procedural Safety Check (룰 기반) ──
    procedural_findings = detect_procedural_safety_issues(clinical_text)
    if procedural_findings:
        safety_lines = []
        for f in procedural_findings:
            status = "DANGER" if f["danger_flag"] else ("SAFE" if f["has_safety"] else "NOT DOCUMENTED")
            line = f"- {f['procedure']}: [{status}]"
            if f["danger_flag"]:
                line += f" ⚠ '{f['danger_flag']}' detected → CRITICAL Red Flag"
            if not f["has_safety"] and not f["danger_flag"]:
                line += " (ultrasound/sterile technique not documented)"
            safety_lines.append(line)
        procedural_safety_findings = "\n".join(safety_lines)
        print(f"  [Diagnosis Agent] Procedural Safety: {len(procedural_findings)} procedures found")
    else:
        procedural_safety_findings = "시술 키워드 미감지"
    
    # ── Cause of Death Alignment ──
    death_alignment = detect_death_cause_mismatch(patient, structured)
    if death_alignment:
        da = death_alignment
        alignment_lines = [
            f"입원 사유: {da['admission_reason']}",
            f"사망 원인: {da['death_cause'] or 'Not explicitly stated'}",
            f"불일치: {'YES - 입원사유와 사망원인이 다름. 마지막 24시간 이벤트에 가중치를 두어 분석할 것' if da['mismatch'] else 'No'}",
            f"Iatrogenic: {'YES - 의인성 합병증 의심. 최우선 분석 필요' if da['iatrogenic'] else 'No'}",
        ]
        if da['critical_events']:
            alignment_lines.append(f"마지막 critical events: {'; '.join(str(e) for e in da['critical_events'][:5])}")
        death_alignment_info = "\n".join(alignment_lines)
        if da['mismatch'] or da['iatrogenic']:
            print(f"  [Diagnosis Agent] Death Alignment: MISMATCH={da['mismatch']}, IATROGENIC={da['iatrogenic']}")
    else:
        death_alignment_info = "해당 없음 (생존 케이스 또는 사망 정보 미기재)"
    
    # ── Procedures Info (구조화 데이터에서) ──
    procedures = structured.get("procedures_performed", []) if structured else []
    if procedures:
        proc_lines = []
        for p in procedures:
            if isinstance(p, dict):
                proc_lines.append(
                    f"- {p.get('name', 'Unknown')}: technique={p.get('technique', 'N/A')}, "
                    f"complications={p.get('complications', 'none')}, "
                    f"safety_flags={p.get('safety_flags', [])}"
                )
        procedures_info = "\n".join(proc_lines) if proc_lines else "시술 정보 없음"
    else:
        procedures_info = "구조화 데이터에 시술 정보 없음"
    
    prompt = ANALYSIS_PROMPT.format(
        diagnosis=patient.get("diagnosis", "Unknown"),
        vitals_summary=summaries["vitals_summary"],
        symptoms_summary=summaries["symptoms_summary"],
        red_flags=summaries["red_flags"],
        physical_exam=summaries["physical_exam"],
        lab_imaging=summaries["lab_imaging"],
        outcome=outcome,
        procedures_info=procedures_info,
        death_alignment_info=death_alignment_info,
        procedural_safety_findings=procedural_safety_findings,
        clinical_text_excerpt=clinical_text_excerpt,
        evidence_summary=evidence_summary,
        episodic_lessons=episodic_lessons,
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
            
            # ── Severity Hierarchy 후처리: iatrogenic > death_alignment > medication ──
            analysis["issues"] = _rerank_issues_by_severity_hierarchy(analysis.get("issues", []))
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
