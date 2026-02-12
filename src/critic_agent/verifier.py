"""
Verifier: critique 기반으로 유사 케이스 top-k 근거의 solutions 생성.
LGBM critique_engine.verifier 이식, openai_chat 사용.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _bullets(xs: List[Any]) -> str:
    if not xs:
        return "- None"
    return "\n".join(
        f"- {x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)}" for x in xs
    )


class Verifier:
    """Critique + 유사 케이스 top-k → 구체적 solutions (action, evidence, priority)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def verify(
        self,
        critique: Dict[str, Any],
        similar_cases_topk: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Args:
            critique: critique_points, risk_factors, recommendations (및 patient_id 등)
            similar_cases_topk: 유사 케이스 리스트 (text, age, sex, status 등)

        Returns:
            { "patient_id": ..., "solutions": [ { "issue", "solution", "evidence", "priority" } ], "raw": ... }
        """
        prompt = self._build_prompt(critique, similar_cases_topk)
        cfg = OpenAIChatConfig(model=self.model, temperature=0.3, max_tokens=4000)
        content = call_openai_chat_completions(
            messages=[{"role": "user", "content": prompt}],
            config=cfg,
        )
        if not content:
            return {
                "patient_id": critique.get("patient_id"),
                "solutions": [],
                "raw": "",
            }
        obj = safe_json_loads(content) or {}
        solutions = obj.get("solutions", [])
        if not isinstance(solutions, list):
            solutions = []
        return {
            "patient_id": critique.get("patient_id"),
            "solutions": solutions,
            "raw": content,
        }

    def _build_prompt(
        self,
        critique: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
    ) -> str:
        critique_block = f"""
[CRITIQUE]
Critique Points:
{_bullets(critique.get("critique_points", []))}

Risk Factors:
{_bullets(critique.get("risk_factors", []))}

Recommendations:
{_bullets(critique.get("recommendations", []))}
"""

        case_block = ""
        for i, c in enumerate(similar_cases[:3], 1):
            age = c.get("age", c.get("anchor_age", "N/A"))
            sex = c.get("sex", c.get("gender", "N/A"))
            status = str(c.get("status", "")).lower()
            outcome = "DIED" if status == "dead" else "SURVIVED"
            if not status and c.get("hospital_expire_flag") == 1:
                outcome = "DIED"
            case_block += f"""
[Similar Case {i}]
- Age/Sex: {age}/{sex}
- Admission: {c.get("admission_type")} / {c.get("admission_location")}
- Outcome: {outcome}
- Clinical Note:
{(c.get("text") or "")}
"""

        return f"""
You are a senior clinical decision verifier.

Task:
Based on the critique and the TOP-3 similar cases,
generate concrete SOLUTIONS that directly address the critique points.

Rules:
- Each solution MUST reference at least one similar case as evidence
- Do NOT invent new medical facts
- Be concise, actionable, and clinically realistic
- Output JSON ONLY

Output format:
{{
  "solutions": [
    {{
      "issue": "critique issue",
      "solution": "concrete action",
      "evidence": "Similar Case 1 / 2 / 3",
      "priority": "high | medium | low"
    }}
  ]
}}

{critique_block}

[SIMILAR CASES]
{case_block}
"""
