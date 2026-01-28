"""
Verifier 모듈 
"solution": critique_reasoner 가 비판한 것에 대한 해결점 생성 (유사 케이스 topk =3 근거를 가지고) 생성, 이후 solution 이라는 변수에 저장
llm 은 동일한걸로 불러오기


critique = {
    'analysis': llm_response,
    'critique_points': [],
    'risk_factors': [],
    'recommendations': []
}


        line_lower = line.lower()
            if 'critique' in line_lower and 'point' in line_lower:
                current_section = 'critique_points'
            elif 'risk' in line_lower and 'factor' in line_lower:
                current_section = 'risk_factors'
            elif 'recommendation' in line_lower:
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                # 리스트 아이템
                item = line.lstrip('-•* ').strip()
                if current_section and item:
                    critique[current_section].append(item)
        
        # 메타 정보 추가
        critique['patient_id'] = patient_data.get('id')
        critique['cohort_size'] = patterns.get('cohort_size', 0)
        critique['survival_rate'] = patterns.get('survival_stats', {}).get('survival_rate', 0)
        
        return critique

        

verifier는 "solution": "해결책 (verifier, 유사 케이스 근거 기반)"을 위쪽 critique를 기반으로 생성합니다.

"""

import requests
from typing import Dict, List, Any
import json
import os
import time
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
    
    def _call_llm_api(self, prompt: str, max_retries: int = 5) -> str:
        """OpenAI API 호출"""
        
        if not self.api_key:
            raise Exception("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                actual_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    self.api_url, 
                    headers=actual_headers, 
                    json=payload,
                    timeout=120
                )

                if response.status_code == 401:
                    raise Exception("API 키가 유효하지 않습니다. .env 파일의 OPENAI_API_KEY를 확인하세요.")
                
                elif response.status_code == 429:
                    if attempt < max_retries:
                        wait_time = min(attempt * 20, 120)  # 최대 2분
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Rate limit: {max_retries}회 시도 후에도 실패")
                
                elif response.status_code >= 500:
                    if attempt < max_retries:
                        wait_time = attempt * 10
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"서버 에러 ({response.status_code}): {max_retries}회 시도 후에도 실패")
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" not in result or len(result["choices"]) == 0:
                    raise Exception("API 응답에 choices가 없습니다.")
                
                content = result["choices"][0]["message"]["content"]
                if not content:
                    raise Exception("API 응답의 content가 비어있습니다.")
                return content
                
            except requests.exceptions.Timeout as e:
                if attempt < max_retries:
                    wait_time = attempt * 5
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"타임아웃: {max_retries}회 시도 후에도 실패. 마지막 에러: {e}")
            
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    wait_time = attempt * 5
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"연결 실패: {max_retries}회 시도 후에도 실패. 네트워크를 확인하세요. 마지막 에러: {e}")
            
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries:
                    wait_time = attempt * 10
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"HTTP 에러: {max_retries}회 시도 후에도 실패. 마지막 에러: {e}")
            
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    wait_time = 5
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"JSON 파싱 실패: {max_retries}회 시도 후에도 실패. 응답 형식이 예상과 다릅니다. 마지막 에러: {e}")
            
            except Exception as e:
                if attempt < max_retries:
                    wait_time = attempt * 5
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API 호출 실패: {max_retries}회 시도 후에도 실패. 마지막 에러: {type(e).__name__}: {e}")
        
        # 이 코드는 도달하지 않아야 하지만 안전장치
        raise Exception(f"API 호출 최종 실패: {max_retries}회 시도 후에도 성공하지 못함")


    def verify(
        self,
        critique: Dict,
        similar_cases_topk: List[Dict]
    ) -> Dict:

        prompt = self._build_prompt(critique, similar_cases_topk)
        
        try:
            llm_response = self._call_llm_api(prompt)
            if not llm_response:
                raise Exception("Verifier API 호출 실패: 응답이 비어있음")
        except Exception as e:
            raise

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
            # patient.json 스키마 기준: age, sex, status
            age = c.get("age", c.get("anchor_age", "N/A"))
            sex = c.get("sex", c.get("gender", "N/A"))
            
            # status 기준으로 outcome 결정
            status = str(c.get("status", "")).lower()
            if status:
                outcome = "DIED" if status == "dead" else "SURVIVED"
            else:
                # fallback: hospital_expire_flag
                outcome = "DIED" if c.get("hospital_expire_flag") == 1 else "SURVIVED"
            
            case_block += f"""
[Similar Case {i}]
- Age/Sex: {age}/{sex}
- Admission: {c.get("admission_type")} / {c.get("admission_location")}
- Outcome: {outcome}
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

        # JSON 코드 블록에서 추출 시도
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()

        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    extracted = text[start:end + 1]
                    parsed = json.loads(extracted)
                    return parsed
                except json.JSONDecodeError:
                    pass
            return {}
