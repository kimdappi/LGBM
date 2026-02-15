"""
Verifier: critique 기반으로 유사 케이스 top-k 근거의 solutions 생성.
LGBM critique_engine.verifier 이식, openai_chat 사용.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads
from ..agents.evidence_agent import format_evidence_summary


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
        evidence: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            critique: critique_points, risk_factors, recommendations (및 patient_id 등)
            similar_cases_topk: 유사 케이스 리스트 (text, age, sex, status 등)
            evidence: Evidence Agent 검색 결과 (1차 + 2차 PubMed/내부 근거)

        Returns:
            { "patient_id": ..., "solutions": [ { "issue", "solution", "evidence", "priority" } ], "raw": ... }
        """
        prompt = self._build_prompt(critique, similar_cases_topk, evidence=evidence)
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
        evidence: Dict[str, Any] = None,
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

        # 문헌 근거 블록 (1차 + 2차 검색 결과)
        evidence_block = ""
        if evidence:
            formatted = format_evidence_summary(evidence)
            if formatted and formatted != "검색된 근거 없음":
                evidence_block = f"""
[LITERATURE EVIDENCE]
{formatted}
"""

        return f"""
You are a senior clinical decision verifier.

Task:
Based on the critique, the TOP-3 similar cases, and available literature evidence,
generate concrete SOLUTIONS that directly address the critique points.

Rules:
- Each solution MUST reference at least one similar case OR literature (PMID) as evidence
- Cite PubMed literature ONLY when it directly supports the specific solution for this patient's issue
- Do NOT cite literature just because it is available; relevance to the specific critique point is required
- If no literature is relevant, use similar cases only — that is perfectly acceptable
- Do NOT invent new medical facts
- Be concise, actionable, and clinically realistic
- Output JSON ONLY

Output format:
{{
  "solutions": [
    {{
      "issue": "critique issue",
      "solution": "concrete action",
      "evidence": "Similar Case 1 / PMID: 12345678 / etc.",
      "priority": "high | medium | low"
    }}
  ]
}}

{critique_block}

[SIMILAR CASES]
{case_block}
{evidence_block}"""
