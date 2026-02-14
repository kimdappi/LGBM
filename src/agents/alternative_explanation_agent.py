"""
Alternative Explanation Agent: Critic 실행 후 항상 대안 해석·불확실성 정리. OpenAI LLM 사용.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

from .llm import get_llm


def run_alternative_explanation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Critic 이후 항상 LLM으로 대안 해석·불확실성 정리 시도. API 키 없으면 None 반환."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"alternative_explanations": None}

    patient_case = state.get("patient_case") or {}
    similar_cases = state.get("similar_cases") or []
    critique = state.get("critique") or []
    confidence = state.get("confidence", 0.5)
    diagnosis_analysis = state.get("diagnosis_analysis") or {}
    treatment_analysis = state.get("treatment_analysis") or {}
    text = (patient_case.get("clinical_text") or patient_case.get("text") or "")[:3000]

    critique_summary = "\n".join(
        f"- {c.get('issue', c.get('point', ''))[:120]}" for c in critique[:8] if isinstance(c, dict)
    ) or "None"

    prompt = f"""You are an alternative explanation analyst. This case has ambiguous interpretation (high variance in similar cases or low confidence in critique). Provide alternative explanations and uncertainty notes for the report.

Patient diagnosis: {patient_case.get('diagnosis')}
Similar cases count: {len(similar_cases)}
Critique confidence: {confidence}
Critique points (summary):
{critique_summary}

Diagnosis analysis (brief): {str(diagnosis_analysis)[:300]}
Treatment analysis (brief): {str(treatment_analysis)[:300]}

Clinical text excerpt:
{text}

Output JSON only:
{{"active": true, "summary": "2-4 sentences on interpretation ambiguity", "alternative_explanations": ["explanation1", "explanation2"], "uncertainty_notes": ["note1"], "caveats": ["one line each"]}}
"""

    try:
        llm = get_llm()
        response = llm.gpt4o(prompt=prompt, temperature=0.2, max_tokens=600, json_mode=True, timeout=45)
        response = response.strip().replace("```json", "").replace("```", "").strip()
        analysis = json.loads(response)
        analysis["active"] = True
        print("  [Alternative Explanation Agent] ran")
        return {"alternative_explanations": analysis}
    except Exception as e:
        print(f"  [Alternative Explanation Agent] LLM failed: {e}")
        return {"alternative_explanations": None}
