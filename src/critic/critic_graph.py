"""
Critic 파이프라인을 LangGraph 서브그래프로 구현.

흐름: preprocess → router → run_tools → critique_builder → END (단일 패스)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .critique_builder import CritiqueBuilder
from .registry import build_default_registry
from .router import LLMRouter
from .runner import AgentConfig, ToolRegistry
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
    executed_tools: List[str]
    executed_budget: int


def _dict_to_agent_state(d: Dict[str, Any]) -> AgentState:
    """Graph state dict → AgentState (mutable, for tools)."""
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
# structured_chart → timeline/evidence 형식 변환 (앞단 결과 있으면 전처리 스킵용)
# ---------------------------------------------------------------------------

def _structured_chart_to_timeline_and_evidence(structured_chart: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """structured_chart가 있으면 timeline·evidence 형식으로 변환. 없으면 ({}, {})."""
    if not structured_chart or not isinstance(structured_chart, dict):
        return {}, {}
    events: List[Dict[str, Any]] = []
    # procedures_performed
    for i, proc in enumerate(structured_chart.get("procedures_performed") or [], 1):
        if isinstance(proc, dict):
            name = proc.get("name") or "procedure"
            timing = proc.get("timing") or ""
            events.append({
                "event_id": f"T{i}",
                "type": "procedure",
                "time_hint": timing,
                "text": f"{name} ({timing})" if timing else name,
                "start": 0,
                "end": 0,
            })
    base = len(events)
    # clinical_course.events
    course = structured_chart.get("clinical_course") or {}
    for i, ev in enumerate((course.get("events") or []) if isinstance(course.get("events"), list) else [], 1):
        text = ev if isinstance(ev, str) else str(ev.get("text", ev))[:400]
        events.append({
            "event_id": f"T{base + i}",
            "type": "other",
            "time_hint": None,
            "text": text,
            "start": 0,
            "end": 0,
        })
    base = len(events)
    # outcome.critical_events_leading_to_outcome
    outcome = structured_chart.get("outcome") or {}
    for i, ev in enumerate((outcome.get("critical_events_leading_to_outcome") or []) if isinstance(outcome.get("critical_events_leading_to_outcome"), list) else [], 1):
        text = ev if isinstance(ev, str) else str(ev)[:400]
        events.append({
            "event_id": f"T{base + i}",
            "type": "deterioration" if "death" in text.lower() or "expir" in text.lower() else "other",
            "time_hint": None,
            "text": text,
            "start": 0,
            "end": 0,
        })
    timeline_out = {"events": events, "event_count": len(events)} if events else {"events": [], "event_count": 0}
    # evidence_spans: list of {field, text_span} → dict E1 -> {category, quote, start, end}
    spans_in = structured_chart.get("evidence_spans") or []
    evidence_spans: Dict[str, Dict[str, Any]] = {}
    if isinstance(spans_in, list):
        for i, sp in enumerate(spans_in, 1):
            if isinstance(sp, dict):
                field = sp.get("field") or "other"
                quote = sp.get("text_span") or str(sp)[:500]
                evidence_spans[f"E{i}"] = {"category": field, "quote": quote, "start": 0, "end": 0}
    evidence_out = {"evidence_spans": evidence_spans} if evidence_spans else {"evidence_spans": {}}
    return timeline_out, evidence_out


# ---------------------------------------------------------------------------
# 노드 팩토리 (registry, config 클로저로 보유)
# ---------------------------------------------------------------------------

def _make_preprocess_node(registry: ToolRegistry, config: AgentConfig):
    def preprocess_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        executed: List[str] = []
        # structured_chart 있으면 timeline/evidence 도구 스킵, 변환으로 채움 (의존 아님)
        structured_chart = s.cohort_data.get("structured_chart") if isinstance(s.cohort_data, dict) else None
        if structured_chart and isinstance(structured_chart, dict):
            timeline_out, evidence_out = _structured_chart_to_timeline_and_evidence(structured_chart)
            s.preprocessing["timeline"] = timeline_out
            s.preprocessing["evidence"] = evidence_out
            executed.extend(["timeline", "evidence"])
        else:
            for tool_name in ["timeline", "evidence"]:
                tool = registry.get(tool_name)
                out = tool.safe_run(s)
                s.preprocessing[tool_name] = out
                executed.append(tool_name)
        # record_gaps는 항상 실행
        tool = registry.get("record_gaps")
        out = tool.safe_run(s)
        s.preprocessing["record_gaps"] = out
        executed.append("record_gaps")
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
    def router_node(state: CriticGraphState) -> Dict[str, Any]:
        s = _dict_to_agent_state(state)
        selection = LLMRouter(model=config.router_llm_model).select(
            state=s,
            available_tools=registry.available_tool_names,
            tool_cards=registry.cards,
        )
        s.router = {
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
        critique = builder.build(s, previous_critique=None, patch_instructions="")
        return {"critique": critique, "trace": s.trace}
    return critique_builder_node


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

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "router")
    graph.add_edge("router", "run_tools")
    graph.add_edge("run_tools", "critique_builder")
    graph.add_edge("critique_builder", END)

    return graph.compile()


# 기본 컴파일된 그래프 (메인 그래프에서 사용)
_compiled_critic_graph = None


def get_critic_graph():
    global _compiled_critic_graph
    if _compiled_critic_graph is None:
        _compiled_critic_graph = build_critic_graph()
    return _compiled_critic_graph
