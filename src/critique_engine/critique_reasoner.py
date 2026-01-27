"""
Critique Reasoner - 비판적 분석 모듈
환자 정보를 cohort_comparator의 similar_case_patterns 결과와 비교하여 비판 포인트 생성
## 추후 모델명을 조절하는 modelinfo.py 생성하는 것이 나을 것으로 보임.
"""

from typing import Dict, List, Any
import json
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 경로를 프로젝트 루트 기준으로 설정
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class CritiqueReasoner:
    """
    LLM 기반 비판적 분석
    
    환자 케이스를 유사 케이스 패턴과 비교하여
    잠재적 문제점, 개선 사항, 주의사항 등을 분석
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """CritiqueReasoner 초기화 (OpenAI 사용)"""
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    
    def _call_llm_api(self, prompt: str, stream: bool = False, max_retries: int = 5) -> str:
        """OpenAI API 호출"""
        if not self.api_key:
            raise Exception("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4000
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=120
                )

                if response.status_code == 401:
                    raise Exception("API 키가 유효하지 않습니다. .env 파일의 OPENAI_API_KEY를 확인하세요.")
                elif response.status_code == 429:
                    if attempt < max_retries:
                        time.sleep(min(attempt * 20, 120))
                        continue
                    raise Exception(f"Rate limit: {max_retries}회 시도 후에도 실패")
                elif response.status_code == 404:
                    raise Exception(f"모델을 찾을 수 없음: {self.model}")
                elif response.status_code >= 500:
                    if attempt < max_retries:
                        time.sleep(attempt * 10)
                        continue
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
                    time.sleep(attempt * 5)
                    continue
                raise Exception(f"타임아웃: {max_retries}회 시도 후에도 실패. 마지막 에러: {e}")

            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    time.sleep(attempt * 5)
                    continue
                raise Exception(f"연결 실패: {max_retries}회 시도 후에도 실패. 네트워크를 확인하세요. 마지막 에러: {e}")

            except requests.exceptions.HTTPError as e:
                if attempt < max_retries:
                    time.sleep(attempt * 10)
                    continue
                raise Exception(f"HTTP 에러: {max_retries}회 시도 후에도 실패. 마지막 에러: {e}")

            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                raise Exception(f"JSON 파싱 실패: {max_retries}회 시도 후에도 실패. 응답 형식이 예상과 다릅니다. 마지막 에러: {e}")

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(attempt * 5)
                    continue
                raise Exception(f"API 호출 실패: {max_retries}회 시도 후에도 실패. 마지막 에러: {type(e).__name__}: {e}")
        
        # 이 코드는 도달하지 않아야 하지만 안전장치
        raise Exception(f"API 호출 최종 실패: {max_retries}회 시도 후에도 성공하지 못함")
                
    
    def critique(
        self,
        patient_data: Dict,
        similar_case_patterns: Dict,
        stream: bool = False
    ) -> Dict:
        """
        환자 케이스 비판적 분석
        
        Args:
            patient_data: 환자 정보
                {
                    'id': str,
                    'status': str,
                    'sex': str,
                    'age': int,
                    'admission_type': str,
                    'admission_location': str,
                    'discharge_location': str,
                    'arrival_transport': str,
                    'disposition': str,
                    'text': str
                }
            similar_case_patterns: cohort_comparator 분석 결과
                {
                    'cohort_size': int,
                    'survival_stats': {...},
                    'demographic_patterns': {...},
                    'admission_patterns': {...},
                    'clinical_patterns': {...},
                    'outcome_comparison': {...}
                }
        
        Returns:
            critique: {
                'analysis': str,
                'critique_points': List[str],
                'risk_factors': List[str],
                'recommendations': List[str]
            }
        """
        # Stage 1: 환자 근거 추출
        stage1_result = self._extract_patient_evidence(patient_data, stream=stream)

        # Stage 2: 코호트 기반 비판 생성
        critique = self._generate_critique(stage1_result, similar_case_patterns, stream=stream)
        
        return critique
    
    def _build_stage1_prompt(
        self,
        patient_data: Dict
    ) -> str:
        """Stage 1 프롬프트 생성: 근거/결정지점 추출"""

        patient_summary = f"""
Patient Case:
- ID: {patient_data.get('id')}
- Age/Sex: {patient_data.get('age')}/{patient_data.get('sex')}
- Admission Type: {patient_data.get('admission_type')}
- Admission Location: {patient_data.get('admission_location')}
- Status: {patient_data.get('status', 'unknown').upper()}
- Clinical Summary: {patient_data.get('text', 'No clinical summary available')}
"""

        return f"""You are an evidence extractor. Do NOT critique.

{patient_summary}

Task: Extract the following from the clinical text:

1. **EVIDENCE_SPANS**: Key clinical findings mentioned in the text
   - Symptoms, vital signs, lab results, imaging findings, diagnoses
   - Format: {{span_id: {{section: "category", quote: "exact text"}}}}

2. **DECISION_POINTS**: Treatment/diagnostic decisions made by clinicians
   - Examples: medication orders, procedures, consultations, discharge plans
   - For each decision, identify:
     - What was decided
     - When (time_hint)
     - Which evidence_spans justify this decision
     - What information was missing but needed

3. **UNCERTAINTIES**: Information gaps or ambiguous findings in the record

Return JSON only:
{{
  "evidence_spans": {{}},
  "decision_points": [],
  "uncertainties": []
}}
"""

    def _build_stage2_prompt(
        self,
        stage1_result: Dict,
        patterns: Dict
    ) -> str:
        """Stage 2 프롬프트 생성: 코호트 기반 비판 생성"""

        survival_stats = patterns.get('survival_stats', {})
        clinical_patterns = patterns.get('clinical_patterns', {})
        outcome_comparison = patterns.get('outcome_comparison', {})

        cohort_summary = f"""
Similar Cases Analysis:
- Cohort Size: {patterns.get('cohort_size', 0)} cases
- Survival Rate: {survival_stats.get('survival_rate', 0)}%
  ({survival_stats.get('survived', 0)} survived, {survival_stats.get('died', 0)} died)

Clinical Patterns from Similar Cases:
{clinical_patterns.get('llm_analysis', 'No clinical pattern analysis available')}

Outcome Comparison:
{outcome_comparison.get('comparison_analysis', 'No outcome comparison available')}
"""

        return f"""You are a critical medical reviewer. Use ONLY provided evidence.

Stage 1 Evidence (JSON):
{json.dumps(stage1_result, ensure_ascii=False)}

{cohort_summary}

Task: Generate a critical review comparing this patient to similar cases. Return JSON with:

1. **analysis** (string): 
   - Overall assessment of the patient's clinical course
   - Compare patient's treatment decisions with patterns from similar cases
   - Highlight key differences between this patient and the cohort

2. **critique_points** (array of objects):
   - Each critique must identify a potential issue, gap, or concern in patient care
   - Structure: {{"point": "description", "span_id": "E1 or record_uncertainty", "severity": "high/medium/low", "cohort_comparison": "how similar cases differed"}}
   - Examples of critiques: delayed treatment, missing diagnostic tests, deviation from common successful patterns, incomplete documentation

3. **risk_factors** (array of strings):
   - Patient-specific factors that increase risk (from evidence_spans)
   - Factors where this patient differs negatively from survived cases in cohort

4. **recommendations** (array of strings):
   - Actionable suggestions based on successful patterns from similar cases
   - Address each critique_point with a specific recommendation
   - Prioritize recommendations by potential impact on outcome

Rules:
- Every critique_point MUST cite at least one span_id from Stage 1 OR use "record_uncertainty" if based on missing information.
- Support critiques with cohort pattern evidence when available.
- Do NOT assume or claim death/survival as ground truth - focus on process, not outcome.
- Return valid JSON only, no additional text.
"""

    def _extract_patient_evidence(self, patient_data: Dict, stream: bool = False) -> Dict:
        """Stage 1: 환자 근거 추출"""
        prompt = self._build_stage1_prompt(patient_data)
        try:
            llm_response = self._call_llm_api(prompt, stream=stream)
            if not llm_response or llm_response == "No response":
                raise Exception("Stage 1 API 호출 실패: 응답이 비어있음")
        except Exception as e:
            raise
        
        result = self._safe_json_parse(llm_response, fallback={
            "evidence_spans": {},
            "decision_points": [],
            "uncertainties": []
        })
        result["patient_id"] = patient_data.get("id")
        return result

    def _generate_critique(self, stage1_result: Dict, patterns: Dict, stream: bool = False) -> Dict:
        """Stage 2: 코호트 기반 비판 생성"""
        prompt = self._build_stage2_prompt(stage1_result, patterns)
        try:
            llm_response = self._call_llm_api(prompt, stream=stream)
            if not llm_response or llm_response == "No response":
                raise Exception("Stage 2 API 호출 실패: 응답이 비어있음")
        except Exception as e:
            raise
        
        critique = self._safe_json_parse(llm_response, fallback={
            "analysis": llm_response,
            "critique_points": [],
            "risk_factors": [],
            "recommendations": []
        })

        critique["patient_id"] = stage1_result.get("patient_id")
        critique["cohort_size"] = patterns.get("cohort_size", 0)
        critique["survival_rate"] = patterns.get("survival_stats", {}).get("survival_rate", 0)
        return critique
    
    def _safe_json_parse(self, text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 응답을 JSON으로 파싱, 실패 시 fallback 반환"""
        if not text or text == "No response":
            return fallback
        
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
            return json.loads(text)
        except json.JSONDecodeError as e:
            # JSON 객체 찾기 시도
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    extracted = text[start:end + 1]
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
            return fallback

    def _consume_stream_response(self, response: requests.Response) -> str:
        """스트리밍 응답을 실시간 출력하고 전체 텍스트 반환"""
        output_chunks: List[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            try:
                payload = json.loads(line)
                token = (
                    payload.get("token", {}).get("text")
                    or payload.get("generated_text")
                    or payload.get("text")
                )
                if token:
                    print(token, end="", flush=True)
                    output_chunks.append(token)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                output_chunks.append(line)
        print()
        return "".join(output_chunks)