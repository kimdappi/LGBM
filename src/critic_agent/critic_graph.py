"""
Critic 파이프라인을 LangGraph 서브그래프로 구현.

흐름: preprocess → router → run_tools → critique_builder → feedback
      → (조건) 반복 시: run_requested_tools → router → run_tools → critique_builder → feedback
      → (조건) 종료 시: END
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .critique_builder import CritiqueBuilder
from .feedback import CritiqueFeedback, FeedbackConfig
from .registry import build_default_registry
from .router import LLMRouter
from .runner import AgentConfig, ToolRegistry
from .toolrag import build_toolrag_index
from .types import AgentState


# ---------------------------------------------------------------------------
# Critic 서브그래프 State (dict 기반, LangGraph merge용)
# ---------------------------------------------------------------------------

class CriticGraphState(TypedDict, total=False):
    patient: Dict[str, Any]
    cohort_data: Dict[str, Any]
    similar_case_patterns: Dict[str, Any]
    preprocessing: Dict[str, Any]
    lens_results: Dict[str, Any]
    behavior_results: Dict[str, Any]
    router: Dict[str, Any]
    trace: List[Dict[str, Any]]
    critique: Dict[str, Any]
    patch_instructions: str
    feedback_round: int
    last_feedback_ok: bool
    last_requested_tools: List[str]
    executed_tools: List[str]
    executed_budget: int


def _dict_to_agent_state(d: Dict[str, Any]) -> AgentState:
    """Graph state dict → AgentState (mutable, for tools/feedback)."""
    return AgentState(
        patient=d.get("patient") or {},
        cohort_data=d.get("cohort_data") or {},
        similar_case_patterns=d.get("similar_case_patterns") or {},
        preprocessing=dict(d.get("preprocessing") or {}),
        lens_results=dict(d.get("lens_results") or {}),
        behavior_results=dict(d.get("behavior_results") or {}),
        router=dict(d.get("router") or {}),
        trace=list(d.get("trace") or []),
    )


def _agent_state_to_updates(s: AgentState) -> Dict[str, Any]:
    """AgentState 변경분 → graph merge용 dict."""
    return {
        "preprocessing": s.preprocessing,
        "lens_results": s.lens_results,
        "behavior_results": s.behavior_results,
        "router": s.router,
        "trace": s.trace,
    }


# ---------------------------------------------------------------------------
# 노드 팩토리 (registry, config 클로저로 보유)
# ---------------------------------------------------------------------------

def _make_preprocess_node(registry: ToolRegistry, config: AgentConfig):
    def preprocess_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        for tool_name in ["timeline", "evidence", "record_gaps"]:
            tool = registry.get(tool_name)
            out = tool.safe_run(s)
            s.preprocessing[tool_name] = out
        # runner.py 기존 동작과 동일하게: preprocessing은 budget에 포함하지 않음.
        executed = ["timeline", "evidence", "record_gaps"]
        budget = 0
        if "behavior_topk_direct_compare" in registry.available_tool_names:
            similar = (s.cohort_data.get("similar_cases") or []) if isinstance(s.cohort_data.get("similar_cases"), list) else []
            if similar and budget < config.max_tools:
                tool = registry.get("behavior_topk_direct_compare")
                out = tool.safe_run(s)
                s.behavior_results["behavior_topk_direct_compare"] = out
                executed.append("behavior_topk_direct_compare")
                budget += 1
        return {
            **_agent_state_to_updates(s),
            "executed_tools": executed,
            "executed_budget": budget,
        }
    return preprocess_node


def _make_router_node(registry: ToolRegistry, config: AgentConfig):
    index = None
    cards_by_name = None

    def router_node(state: CriticGraphState) -> Dict[str, Any]:
        nonlocal index, cards_by_name
        if index is None:
            index = build_toolrag_index(registry.cards)
            cards_by_name = {c.name: c for c in registry.cards}
        s = _dict_to_agent_state(state)
        round_idx = state.get("feedback_round", 0)
        selection = LLMRouter(model=config.router_llm_model).select(
            state=s,
            available_tools=registry.available_tool_names,
            toolrag_index=index,
            tool_cards_by_name=cards_by_name,
        )
        s.router = {
            "round": round_idx,
            "selected_tools": selection.tools,
            "reason": selection.reason,
            "retrieved_cards": selection.retrieved_cards,
        }
        return {"router": s.router, "trace": s.trace}
    return router_node


def _make_run_tools_node(registry: ToolRegistry, config: AgentConfig):
    def run_tools_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        executed_set = set(state.get("executed_tools") or [])
        budget = state.get("executed_budget", 0)
        selected = (state.get("router") or {}).get("selected_tools") or []
        for tool_name in selected:
            if tool_name in executed_set:
                continue
            if budget >= config.max_tools:
                s.add_trace(tool="runner", status="budget_stop", detail={"max_tools": config.max_tools})
                break
            tool = registry.get(tool_name)
            out = tool.safe_run(s)
            if tool_name.startswith("lens_"):
                s.lens_results[tool_name] = out
            else:
                s.behavior_results[tool_name] = out
            executed_set.add(tool_name)
            budget += 1
        return {
            **_agent_state_to_updates(s),
            "executed_tools": list(executed_set),
            "executed_budget": budget,
        }
    return run_tools_node


def _make_critique_builder_node(registry: ToolRegistry, config: AgentConfig):
    builder = CritiqueBuilder(model=config.critique_model)

    def critique_builder_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        prev = state.get("critique") or {}
        patch = state.get("patch_instructions") or ""
        critique = builder.build(s, previous_critique=prev if prev else None, patch_instructions=patch)
        return {"critique": critique, "trace": s.trace}
    return critique_builder_node


def _make_feedback_node(registry: ToolRegistry, config: AgentConfig):
    feedback = CritiqueFeedback(
        FeedbackConfig(max_rounds=config.feedback_rounds, model=config.feedback_model)
    )

    def feedback_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        critique = state.get("critique") or {}
        round_idx = state.get("feedback_round", 0)
        decision = feedback.decide(
            state=s,
            critique=critique,
            available_tools=registry.available_tool_names,
        )
        s.add_trace(
            tool="feedback",
            status="ok",
            detail={
                "round": round_idx,
                "ok": decision.ok,
                "reason": decision.reason,
                "requested_tools": decision.requested_tools,
            },
        )
        next_round = round_idx + 1
        return {
            "trace": s.trace,
            "last_feedback_ok": decision.ok,
            "last_requested_tools": decision.requested_tools or [],
            "patch_instructions": decision.patch_instructions or (state.get("patch_instructions") or ""),
            "feedback_round": next_round,
        }
    return feedback_node


def _make_run_requested_tools_node(registry: ToolRegistry, config: AgentConfig):
    def run_requested_tools_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        executed_set = set(state.get("executed_tools") or [])
        budget = state.get("executed_budget", 0)
        requested = state.get("last_requested_tools") or []
        for tool_name in requested:
            if tool_name in executed_set:
                continue
            if budget >= config.max_tools:
                s.add_trace(tool="runner", status="budget_stop", detail={"max_tools": config.max_tools})
                break
            tool = registry.get(tool_name)
            out = tool.safe_run(s)
            if tool_name.startswith("lens_"):
                s.lens_results[tool_name] = out
            else:
                s.behavior_results[tool_name] = out
            executed_set.add(tool_name)
            budget += 1
        return {
            **_agent_state_to_updates(s),
            "executed_tools": list(executed_set),
            "executed_budget": budget,
        }
    return run_requested_tools_node


def _make_should_continue_feedback(config: AgentConfig):
    max_rounds = max(config.feedback_rounds, 1)

    def _should_continue_feedback(state: CriticGraphState) -> str:
        """반복 조건: 아직 ok 아니고, 라운드 한도 내이면 'continue', 아니면 'end'."""
        ok = state.get("last_feedback_ok", True)
        round_idx = state.get("feedback_round", 0)
        if ok or round_idx >= max_rounds:
            return "end"
        return "continue"
    return _should_continue_feedback


# ---------------------------------------------------------------------------
# 그래프 빌드 및 컴파일
# ---------------------------------------------------------------------------

def build_critic_graph(
    registry: Optional[ToolRegistry] = None,
    config: Optional[AgentConfig] = None,
):
    registry = registry or build_default_registry()
    config = config or AgentConfig()
    graph = StateGraph(CriticGraphState)

    graph.add_node("preprocess", _make_preprocess_node(registry, config))
    graph.add_node("router", _make_router_node(registry, config))
    graph.add_node("run_tools", _make_run_tools_node(registry, config))
    graph.add_node("critique_builder", _make_critique_builder_node(registry, config))
    graph.add_node("feedback", _make_feedback_node(registry, config))
    graph.add_node("run_requested_tools", _make_run_requested_tools_node(registry, config))

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "router")
    graph.add_edge("router", "run_tools")
    graph.add_edge("run_tools", "critique_builder")
    graph.add_edge("critique_builder", "feedback")
    graph.add_conditional_edges(
        "feedback",
        _make_should_continue_feedback(config),
        {
            "continue": "run_requested_tools",
            "end": END,
        },
    )
    graph.add_edge("run_requested_tools", "router")

    return graph.compile()


# 기본 컴파일된 그래프 (메인 그래프에서 사용)
_compiled_critic_graph = None


def get_critic_graph():
    global _compiled_critic_graph
    if _compiled_critic_graph is None:
        _compiled_critic_graph = build_critic_graph()
    return _compiled_critic_graph
