"""
Verifier
- critique_reasoner가 생성한 critique를 입력으로 받아
- 유사 케이스(topk=3)를 근거로 해결책(solution) 생성
- OpenAI GPT-4o 사용
"""

from typing import Dict, List, Any
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Verifier:
    """
    Critique 결과를 검증하고
    유사 케이스를 근거로 해결책(Solution)을 생성하는 모듈
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_llm(self, prompt: str) -> str:
        """OpenAI GPT-4o 호출"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical verifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"[ERROR] {e}"

    def verify(
        self,
        critique: Dict,
        similar_cases_topk: List[Dict]
    ) -> Dict:
        """
        Args:
            critique: critique_reasoner 결과
            similar_cases_topk: 유사 케이스 topk=3 (RAG 결과 원본)

        Returns:
            {
              patient_id,
              solutions,
              raw
            }
        """

        prompt = self._build_prompt(critique, similar_cases_topk)
        llm_response = self._call_llm(prompt)

        solutions = self._safe_parse_json(llm_response).get("solutions", [])

        return {
            "patient_id": critique.get("patient_id"),
            "solutions": solutions,
            "raw": llm_response
        }

    def _build_prompt(
        self,
        critique: Dict,
        similar_cases: List[Dict]
    ) -> str:
        """해결책 생성 프롬프트"""

        def bullets(xs):
            if not xs:
                return "- None"
            return "\n".join(
                [f"- {x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)}" for x in xs]
            )

        critique_block = f"""
[CRITIQUE]
Critique Points:
{bullets(critique.get("critique_points", []))}

Risk Factors:
{bullets(critique.get("risk_factors", []))}

Recommendations:
{bullets(critique.get("recommendations", []))}
"""

        case_block = ""
        for i, c in enumerate(similar_cases[:3], 1):
            case_block += f"""
[Similar Case {i}]
- Age/Sex: {c.get("anchor_age")}/{c.get("gender")}
- Admission: {c.get("admission_type")} / {c.get("admission_location")}
- Outcome: {"DIED" if c.get("hospital_expire_flag") == 1 else "SURVIVED"}
- Clinical Note:
{c.get("text", "")[:800]}
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

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """JSON 파싱 (깨져도 시스템 안 죽게)"""

        if not text:
            return {}

        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception:
            pass

        return {}
