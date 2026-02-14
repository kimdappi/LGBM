"""
Process Contributor Agent: Router가 선택했을 때만 프로세스·이상치 분석. OpenAI LLM 사용.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

from .llm import get_llm


def run_process_contributor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """selected_agents에 process_contributor가 있을 때만 LLM 실행."""
    selected = state.get("selected_agents") or []
    if "process_contributor" not in selected:
        return {"process_contributor_analysis": None}
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"process_contributor_analysis": None}

    patient_case = state.get("patient_case") or {}
    structured = state.get("structured_chart") or {}
    similar_cases = state.get("similar_cases") or []
    intervention_coverage = state.get("intervention_coverage") or {}
    text = (patient_case.get("clinical_text") or patient_case.get("text") or "")[:4000]

    prompt = f"""You are a process contributor / outlier analyst for a clinical critique. This case may show treatment delay/missed care or outcome opposite to similar cases. Analyze and summarize for the report.

Patient outcome: {patient_case.get('outcome')} / {patient_case.get('status')}
Structured chart (outcome): {structured.get('outcome', {})}
Similar cases count: {len(similar_cases)}
Intervention coverage (summary): {str(intervention_coverage)[:400]}

Clinical text excerpt:
{text}

Output JSON only:
{{"active": true, "summary": "2-4 sentences on process/outlier suspicion", "delay_or_missed_findings": ["finding1"], "outcome_vs_cohort": "one sentence", "recommendations": ["one line each"]}}
"""

    try:
        llm = get_llm()
        response = llm.gpt4o(prompt=prompt, temperature=0.2, max_tokens=600, json_mode=True, timeout=45)
        response = response.strip().replace("```json", "").replace("```", "").strip()
        analysis = json.loads(response)
        analysis["active"] = True
        print("  [Process Contributor Agent] ran")
        return {"process_contributor_analysis": analysis}
    except Exception as e:
        print(f"  [Process Contributor Agent] LLM failed: {e}")
        return {"process_contributor_analysis": None}
