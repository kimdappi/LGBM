"""
Router가 선택한 에이전트 중 risk_factor, process_contributor를 순차 실행하고 state에 반영.
"""

from __future__ import annotations

from typing import Dict, Any

from .risk_factor_agent import run_risk_factor_agent
from .process_contributor_agent import run_process_contributor_agent


def run_conditional_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    selected_agents에 따라 risk_factor, process_contributor만 실행.
    선택되지 않은 에이전트는 출력을 None으로 둠.
    """
    updates: Dict[str, Any] = {}
    selected = state.get("selected_agents") or []

    if "risk_factor" in selected:
        updates.update(run_risk_factor_agent(state))
    else:
        updates["risk_factor_analysis"] = None

    state_after = {**state, **updates}
    if "process_contributor" in selected:
        updates.update(run_process_contributor_agent(state_after))
    else:
        updates["process_contributor_analysis"] = None

    return updates
