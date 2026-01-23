"""
Verifier
- critique_reasoner가 생성한 critique를 입력으로 받아
- 유사 케이스(topk=3)를 근거로 해결책(solution) 생성
- LLM 호출은 CritiqueReasoner와 동일 (HF Inference API)
"""

from typing import Dict, List
import requests
import json
import re


class Verifier:
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"

    def _call_hf_api(self, prompt: str) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1200,
                "temperature": 0.3,
                "return_full_text": False
            }
        }

        try:
            res = requests.post(self.api_url, json=payload)
            res.raise_for_status()
            out = res.json()

            if isinstance(out, list) and out:
                return out[0].get("generated_text", "")
            if isinstance(out, dict):
                return out.get("generated_text", "")
            return str(out)

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
            similar_cases_topk: 유사 케이스 topk=3 (rag 결과 원본)

        Returns:
            solution dict
        """

        prompt = self._build_prompt(critique, similar_cases_topk)
        llm_response = self._call_hf_api(prompt)

        solutions = self._parse_solution(llm_response)

        return {
            "patient_id": critique.get("patient_id"),
            "solution": solutions,
            "raw": llm_response
        }

    def _build_prompt(
        self,
        critique: Dict,
        similar_cases: List[Dict]
    ) -> str:
        """해결책 생성용 프롬프트"""

        def bullets(xs):
            return "\n".join([f"- {x}" for x in xs]) if xs else "- None"

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
- Outcome: {"DIED" if c.get("hospital_expire_flag")==1 else "SURVIVED"}
- Note:
{c.get("text","")[:800]}
"""

        return f"""
You are a clinical verifier.

Task:
Based on the critique and the TOP-3 similar cases,
generate concrete SOLUTIONS that address the critique.

Rules:
- Each solution must reference at least one similar case as evidence
- Be concise and actionable
- Output JSON only

Output format:
{{
  "solutions": [
    {{
      "issue": "...",
      "solution": "...",
      "evidence": "which similar case supports this",
      "priority": "high|medium|low"
    }}
  ]
}}

{critique_block}

[SIMILAR CASES]
{case_block}
"""

    def _parse_solution(self, text: str) -> List[Dict]:
        """JSON 파싱 (실패해도 죽지 않게)"""

        try:
            # JSON 부분만 추출
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start:end+1])
                return data.get("solutions", [])
        except Exception:
            pass

        return []
