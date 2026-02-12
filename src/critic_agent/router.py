from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from .toolrag import ToolRAGIndex, retrieve_tool_cards
from .types import AgentState, ToolCard, ToolSelection
from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _text_len_bucket(text: str) -> str:
    n = len(text or "")
    if n < 600:
        return "very_short"
    if n < 1500:
        return "short"
    if n < 4000:
        return "medium"
    return "long"


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


@dataclass
class HeuristicRouter:
    """
    Router v1: no extra LLM calls.
    Picks lens/tools based on simple triggers.
    """

    def select(self, state: AgentState, available_tools: List[str]) -> ToolSelection:
        text = str(state.patient.get("text", "") or "")
        bucket = _text_len_bucket(text)

        severe_kw = [
            "shock",
            "hypotension",
            "intub",
            "vent",
            "respiratory failure",
            "sepsis",
            "lactate",
            "pressors",
            "icu",
            "rapid response",
            "code blue",
        ]
        dx_kw = ["diagnosis", "impression", "assessment", "a/p", "ddx", "differential"]

        has_severe = _contains_any(text, severe_kw)
        has_dx_claim = _contains_any(text, dx_kw)

        tools: List[str] = []

        # Lens selection (V1 minimal set)
        if has_severe or state.patient.get("status") in ("dead", "alive"):
            tools += ["lens_severity_risk", "lens_monitoring_response"]
        if has_dx_claim or bucket in ("medium", "long"):
            tools += ["lens_diagnostic_consistency"]

        # Top-K direct compare is treated as shared evidence base and can be run earlier by runner.
        # Still keep it as a selectable tool for backward-compat / optional execution.
        if bucket in ("very_short", "short"):
            tools += ["behavior_topk_direct_compare"]

        # Deduplicate, keep only available
        uniq = []
        for t in tools:
            if t in available_tools and t not in uniq:
                uniq.append(t)

        reason = f"heuristic_router bucket={bucket} severe={has_severe} dx_claim={has_dx_claim}"
        return ToolSelection(tools=uniq, reason=reason, retrieved_cards=[])


@dataclass
class LLMRouter:
    """
    Router v1: ToolRAG (retrieve tool cards) + LLM picks tools.
    Enabled when CARE_CRITIC_ROUTER_LLM=1.
    """

    model: str = "gpt-4o-mini"
    top_cards: int = 6

    def _enabled(self) -> bool:
        return os.environ.get("CARE_CRITIC_ROUTER_LLM", "").strip() == "1"

    def select(
        self,
        state: AgentState,
        available_tools: List[str],
        toolrag_index: ToolRAGIndex,
        tool_cards_by_name: Dict[str, ToolCard],
    ) -> ToolSelection:
        if not self._enabled():
            return HeuristicRouter().select(state, available_tools)

        text = str(state.patient.get("text", "") or "")
        pre = state.preprocessing or {}

        query = "\n".join(
            [
                "Select the minimal set of tools to run for critical process review.",
                f"text_len={len(text)}",
                f"timeline_events={len((pre.get('timeline') or {}).get('events', [])) if isinstance(pre.get('timeline'), dict) else 0}",
                f"evidence_spans={len((pre.get('evidence') or {}).get('evidence_spans', {})) if isinstance(pre.get('evidence'), dict) else 0}",
                text[:2000],
            ]
        )

        retrieved = retrieve_tool_cards(toolrag_index, query=query, top_n=self.top_cards)
        card_text = "\n\n".join([c.to_text() for c, _ in retrieved])
        retrieved_names = [c.name for c, _ in retrieved]

        prompt = f"""You are a tool router for a clinical critique agent.
You MUST select only from AVAILABLE_TOOLS.
Select the MINIMAL set of tools needed (1~6 tools).

AVAILABLE_TOOLS: {available_tools}

Retrieved tool cards:
{card_text}

Patient (truncated):
{text[:2500]}

Output JSON only:
{{
  "tools": ["tool_name", "..."],
  "reason": "one short sentence"
}}
"""

        cfg = OpenAIChatConfig(model=self.model, temperature=0.1, max_tokens=400)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}
        tools = obj.get("tools") if isinstance(obj.get("tools"), list) else []
        tools = [t for t in tools if isinstance(t, str)]

        # Strict filtering
        tools = [t for t in tools if t in available_tools]
        if not tools:
            return HeuristicRouter().select(state, available_tools)

        reason = str(obj.get("reason", "") or "")
        return ToolSelection(tools=tools, reason=reason, retrieved_cards=retrieved_names)

