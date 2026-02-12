"""
그래프 state ↔ Critic용 state 변환.

사용 순서(현재 구조 기준):
  1. critic_state = clean_state_to_agent_state(graph_state)
  2. critic_dict 초기화 후 Critic LangGraph 서브그래프(get_critic_graph)를 invoke
  3. dict_to_critic_agent_state(critic_dict)로 복원
  4. updates = agent_state_to_clean_updates(critic_state, result)  → graph에 merge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CriticAgentState:
    """run_agent()에 넘기는 state. LGBM AgentState와 필드 동일."""

    patient: Dict[str, Any]
    cohort_data: Dict[str, Any]
    similar_case_patterns: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    lens_results: Dict[str, Any] = field(default_factory=dict)
    behavior_results: Dict[str, Any] = field(default_factory=dict)
    router: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_trace(self, *, tool: str, status: str, detail: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {"tool": tool, "status": status}
        if detail is not None:
            entry["detail"] = detail
        self.trace.append(entry)


def dict_to_critic_agent_state(d: Dict[str, Any]) -> CriticAgentState:
    """LangGraph 등에서 쓰는 dict state → CriticAgentState (adapter/runner 호환)."""
    return CriticAgentState(
        patient=d.get("patient") or {},
        cohort_data=d.get("cohort_data") or {},
        similar_case_patterns=d.get("similar_case_patterns") or {},
        preprocessing=d.get("preprocessing") or {},
        lens_results=d.get("lens_results") or {},
        behavior_results=d.get("behavior_results") or {},
        router=d.get("router") or {},
        trace=list(d.get("trace") or []),
    )


def clean_state_to_agent_state(clean_state: Dict[str, Any]) -> CriticAgentState:
    """그래프 state → run_agent가 받을 state."""
    pc = clean_state.get("patient_case") or {}
    similar = clean_state.get("similar_cases") or []
    cohort = dict(clean_state.get("cohort_data") or {})
    cohort["similar_cases"] = similar

    return CriticAgentState(
        patient={
            "id": pc.get("patient_id") or pc.get("id"),
            "text": pc.get("clinical_text") or pc.get("text", ""),
            "age": pc.get("age"),
            "sex": pc.get("sex"),
            "status": pc.get("outcome") or pc.get("status"),
            "admission_type": pc.get("admission_type"),
            "admission_location": pc.get("admission_location"),
        },
        cohort_data=cohort,
        similar_case_patterns=clean_state.get("similar_case_patterns") or {},
        preprocessing=clean_state.get("preprocessing") or {},
        lens_results=clean_state.get("lens_results") or {},
        behavior_results=clean_state.get("behavior_results") or {},
        router=clean_state.get("router") or {},
        trace=clean_state.get("trace") or [],
    )


def normalize_solutions(solutions: List[Any]) -> List[Dict[str, Any]]:
    """
    리포트 호환 형식으로 통일: [{ "action", "citation", "priority" }].
    - recommendations(문자열 리스트) → action에 넣고 citation/priority 기본값
    - Verifier 형식(issue, solution, evidence, priority) → action/citation/priority로 매핑
    """
    out: List[Dict[str, Any]] = []
    for s in solutions or []:
        if isinstance(s, str):
            out.append({
                "action": s,
                "citation": "Critic recommendation",
                "priority": "medium",
            })
        elif isinstance(s, dict):
            action = s.get("action") or s.get("solution", "")
            target_issue = s.get("target_issue") or s.get("issue", "")
            citation = s.get("citation") or s.get("evidence", "N/A")
            priority = s.get("priority", "medium")
            if action:
                out.append({"action": action, "target_issue": target_issue, "citation": citation, "priority": priority})
    return out


def agent_state_to_clean_updates(
    critic_state: CriticAgentState, critique_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Critic 결과 → 그래프 state에 넣을 dict (critique, solutions + 부가 필드)."""
    critique = critique_result.get("critique_points") or critique_result.get("critique", [])
    raw_solutions = critique_result.get("solutions") or critique_result.get("recommendations", [])
    solutions = normalize_solutions(raw_solutions)
    return {
        "critique": critique,
        "solutions": solutions,
        "preprocessing": critic_state.preprocessing,
        "lens_results": critic_state.lens_results,
        "behavior_results": critic_state.behavior_results,
        "router": critic_state.router,
        "trace": critic_state.trace,
    }
