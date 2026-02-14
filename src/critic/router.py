from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

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
    Router that uses LLM (agent-style): reads tool cards (documents) and current
    context, then selects which agents/tools to run next. Fallback to HeuristicRouter
    when OPENAI_API_KEY is not set.
    """

    model: str = "gpt-4o-mini"

    def _llm_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())

    def select(
        self,
        state: AgentState,
        available_tools: List[str],
        tool_cards: List[ToolCard],
    ) -> ToolSelection:
        if not self._llm_available():
            return HeuristicRouter().select(state, available_tools)

        text = str(state.patient.get("text", "") or "")
        card_text = "\n\n".join([c.to_text() for c in tool_cards]) if tool_cards else ""

        prompt = f"""You are a tool router for a clinical critique agent. Read the tool cards below and the patient context, then select the MINIMAL set of tools (1~6) whose triggers or description fit the case.

AVAILABLE_TOOLS: {available_tools}

Tool cards:
{card_text}

Patient context (truncated):
{text[:2500]}

Output JSON only:
{{"tools": ["tool_name", "..."], "reason": "one short sentence"}}
"""

        cfg = OpenAIChatConfig(model=self.model, temperature=0.1, max_tokens=400)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}
        tools = obj.get("tools") if isinstance(obj.get("tools"), list) else []
        tools = [t for t in tools if isinstance(t, str) and t in available_tools]
        if not tools:
            return HeuristicRouter().select(state, available_tools)
        reason = str(obj.get("reason", "") or "")
        return ToolSelection(tools=tools, reason=reason, retrieved_cards=tools)

