"""
Critique Reasoner - 비판적 분석 모듈
환자 정보를 cohort_comparator의 similar_case_patterns 결과와 비교하여 비판 포인트 생성
## 추후 모델명을 조절하는 modelinfo.py 생성하는 것이 나을 것으로 보임.
"""

from typing import Dict, List
import requests
import json


class CritiqueReasoner:
    """
    LLM 기반 비판적 분석
    
    환자 케이스를 유사 케이스 패턴과 비교하여
    잠재적 문제점, 개선 사항, 주의사항 등을 분석
    """
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Critique Reasoner 초기화
        
        Args:
            model: Hugging Face 모델 이름
                - "mistralai/Mistral-7B-Instruct-v0.3"
        """
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
    
    def _call_hf_api(self, prompt: str) -> str:
        """Hugging Face Inference API 호출"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1500,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 응답 형식에 따라 처리
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response')
            elif isinstance(result, dict):
                return result.get('generated_text', 'No response')
            else:
                return str(result)
        except:#빠져서 파이썬 버전에 따라 충돌날까봐 임의로 추가해둠
            pass 
                
    
    def critique(
        self,
        patient_data: Dict,
        similar_case_patterns: Dict
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
        print("\n[CritiqueReasoner] 비판적 분석 시작...")
        
        # 프롬프트 생성
        prompt = self._build_critique_prompt(patient_data, similar_case_patterns)
        
        # LLM 호출
        llm_response = self._call_hf_api(prompt)
        
        # 결과 파싱 (구조화 시도)
        critique = self._parse_critique(llm_response, patient_data, similar_case_patterns)
        
        print(" 비판적 분석 완료\n")
        
        return critique
    
    def _build_critique_prompt(
        self,
        patient_data: Dict,
        patterns: Dict
    ) -> str:
        """비판적 분석용 프롬프트 생성"""
        
        # 환자 정보 요약
        patient_summary = f"""
Patient Case:
- ID: {patient_data.get('id')}
- Age/Sex: {patient_data.get('age')}/{patient_data.get('sex')}
- Admission Type: {patient_data.get('admission_type')}
- Admission Location: {patient_data.get('admission_location')}
- Status: {patient_data.get('status', 'unknown').upper()}
- Clinical Summary: {patient_data.get('text', 'No clinical summary available')}
"""
        
        # 유사 케이스 패턴 요약
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
        
        # 전체 프롬프트
        prompt = f"""You are a critical medical reviewer analyzing patient care quality and outcomes.

{patient_summary}

{cohort_summary}

Task: Critically analyze this patient's case by comparing it with similar cases. Provide:

1. CRITIQUE POINTS: Identify potential issues, gaps in care, or areas of concern in the patient's management
2. RISK FACTORS: Based on similar cases, what risk factors does this patient have?
3. COMPARISON WITH COHORT: How does this patient's presentation and management compare to similar cases?
4. RECOMMENDATIONS: What should be monitored or improved?

Structure your response with clear sections. Be specific and clinically relevant. Focus on actionable insights.
"""
        
        return prompt
    
    def _parse_critique(
        self,
        llm_response: str,
        patient_data: Dict,
        patterns: Dict
    ) -> Dict:
        """LLM 응답을 구조화된 critique로 파싱"""
        
        # 기본 구조
        critique = {
            'analysis': llm_response,
            'critique_points': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # 간단한 섹션 파싱 시도
        lines = llm_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 섹션 감지
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