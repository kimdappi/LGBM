from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """그래프 전체 상태. Critic 노드에서 쓰는 필드는 critic_adapter로 넘긴 뒤 run_agent가 채움."""

    patient_case: Dict
    similar_cases: List[Dict]
    structured_chart: Optional[Dict]
    diagnosis_analysis: Optional[Dict]
    treatment_analysis: Optional[Dict]
    evidence: Optional[Dict]
    intervention_coverage: Optional[Dict]

    # Critic용 (run_agent가 채움)
    preprocessing: Optional[Dict]
    lens_results: Optional[Dict]
    behavior_results: Optional[Dict]
    router: Optional[Dict]
    trace: Optional[List[Dict]]
    similar_case_patterns: Optional[Dict]

    critique: Optional[List[Dict]]
    solutions: Optional[List[Dict]]
    iteration: int
    confidence: Optional[float]
    memory: Annotated[List[str], add]
