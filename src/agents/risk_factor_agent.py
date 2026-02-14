"""
Risk Factor Agent: Router가 선택했을 때만 위험 인자·동반질환 집중 분석. OpenAI LLM 사용.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

from .llm import get_llm


def run_risk_factor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """selected_agents에 risk_factor가 있을 때만 LLM 실행."""
    selected = state.get("selected_agents") or []
    if "risk_factor" not in selected:
        return {"risk_factor_analysis": None}
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"risk_factor_analysis": None}

    patient_case = state.get("patient_case") or {}
    structured = state.get("structured_chart") or {}
    diagnosis_analysis = state.get("diagnosis_analysis") or {}
    treatment_analysis = state.get("treatment_analysis") or {}
    similar_cases = state.get("similar_cases") or []
    evidence = state.get("evidence") or {}

    secondary = patient_case.get("secondary_diagnoses") or []
    key_conditions = patient_case.get("key_conditions") or []
    text = (patient_case.get("clinical_text") or patient_case.get("text") or "")[:4000]

    prompt = f"""You are a risk factor analyst for a clinical critique. This case has multiple comorbidities and/or high cohort mortality. Analyze risk factors and summarize for the report.

Patient: diagnosis={patient_case.get('diagnosis')}, secondary_diagnoses={secondary}, key_conditions={key_conditions}
Structured chart (outcome): {structured.get('outcome', {})}
Diagnosis analysis (summary): {str(diagnosis_analysis)[:500]}
Treatment analysis (summary): {str(treatment_analysis)[:500]}
Similar cases count: {len(similar_cases)}
Evidence (mode): {evidence.get('retrieval_mode', 'N/A')}

Clinical text excerpt:
{text}

Output JSON only:
{{"active": true, "summary": "2-4 sentences on risk factors and high-risk context", "key_risk_factors": ["factor1", "factor2"], "recommendations": ["one line each"]}}
"""

    try:
        llm = get_llm()
        response = llm.gpt4o(prompt=prompt, temperature=0.2, max_tokens=600, json_mode=True, timeout=45)
        response = response.strip().replace("```json", "").replace("```", "").strip()
        analysis = json.loads(response)
        analysis["active"] = True
        print("  [Risk Factor Agent] ran")
        return {"risk_factor_analysis": analysis}
    except Exception as e:
        print(f"  [Risk Factor Agent] LLM failed: {e}")
        return {"risk_factor_analysis": None}
