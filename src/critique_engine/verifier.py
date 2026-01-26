"""
Verifier
- critique_reasoner가 생성한 critique를 입력으로 받아
- 유사 케이스(topk=3)를 근거로 해결책(solution) 생성
- OpenAI GPT-4o 사용
"""

import requests
from typing import Dict, List, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 경로를 프로젝트 루트 기준으로 설정
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

class Verifier:

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):

        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"  # 올바른 엔드포인트
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    
    def _call_llm_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        
        if not self.api_key:
            print("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"OpenAI API 호출 실패: {e}")
            return None


    def verify(
        self,
        critique: Dict,
        similar_cases_topk: List[Dict]
    ) -> Dict:

        prompt = self._build_prompt(critique, similar_cases_topk)
        llm_response = self._call_llm_api(prompt)

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
