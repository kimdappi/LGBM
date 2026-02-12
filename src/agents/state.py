from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    """전체 에이전트 시스템의 상태"""
    
    # 입력
    patient_case: Dict
    similar_cases: List[Dict]
    
    # 에피소딕 메모리 (크로스런 학습: 과거 분석 경험)
    episodic_lessons: Optional[str]  # format_for_prompt() 결과 → 각 노드 프롬프트에 주입
    
    # 구조화 데이터 (Information Extraction)
    structured_chart: Optional[Dict]  # Vitals, 증상, 검사, 치료, 경과 등
    
    # 각 에이전트 출력
    diagnosis_analysis: Optional[Dict]
    treatment_analysis: Optional[Dict]
    evidence: Optional[Dict]
    intervention_coverage: Optional[Dict]  # 이미 시행된 치료 확인
    
    # Critic 서브그래프 상태
    preprocessing: Dict
    lens_results: Dict
    behavior_results: Dict
    router: Dict
    trace: List[Dict]
    similar_case_patterns: Dict
    
    # 최종 출력
    critique: Optional[List[Dict]]  # Critic의 critique_points 리스트
    solutions: Optional[List[Dict]]
    iteration: int
    confidence: Optional[float]
