from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """전체 에이전트 시스템의 상태"""
    
    # 입력
    patient_case: Dict
    similar_cases: List[Dict]
    
    # 구조화 데이터 (Information Extraction)
    structured_chart: Optional[Dict]  # Vitals, 증상, 검사, 치료, 경과 등
    
    # 각 에이전트 출력
    diagnosis_analysis: Optional[Dict]
    treatment_analysis: Optional[Dict]
    evidence: Optional[Dict]
    intervention_coverage: Optional[Dict]  # 이미 시행된 치료 확인
    
    # 최종 출력
    critique: Optional[List[Dict]]  # Critic의 critique_points 리스트
    solutions: Optional[List[Dict]]
    
    # 메타
    iteration: int
    confidence: Optional[float]  # Critic 신뢰도 (반복 종료 조건용)
    memory: Annotated[List[str], add]  # Reflexion memory
