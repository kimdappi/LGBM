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
from typing import List, Dict, Optional
import re


class ExpertBase:
    """전문가 베이스 클래스"""
    
    def __init__(self, name: str):
        self.name = name
  
