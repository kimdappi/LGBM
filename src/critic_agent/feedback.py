from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import AgentState, JsonDict
from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _count_points(critique: Dict[str, Any]) -> int:
    pts = critique.get("critique_points")
    return len(pts) if isinstance(pts, list) else 0


def _missing_span_ids(critique: Dict[str, Any]) -> int:
    pts = critique.get("critique_points")
    if not isinstance(pts, list):
        return 0
    missing = 0
    for p in pts:
        if not isinstance(p, dict):
            continue
        sid = p.get("span_id")
        if not sid or not isinstance(sid, str):
            missing += 1
    return missing


@dataclass
class FeedbackConfig:
    max_rounds: int = 2
    min_points: int = 3
    model: str = "gpt-4o-mini"


@dataclass
class FeedbackDecision:
    ok: bool
    reason: str
    requested_tools: List[str]
    patch_instructions: str = ""
    raw: Optional[str] = None


class CritiqueFeedback:
    """
    Given the current critique report, decide whether it is sufficient.
    If insufficient, propose additional tools to run and/or revision hints.

    - If OPENAI_API_KEY exists: LLM-based judge with tool suggestions.
    - Else: heuristic judge.
    """

    def __init__(self, cfg: FeedbackConfig):
        self.cfg = cfg

    def decide(self, *, state: AgentState, critique: JsonDict, available_tools: List[str]) -> FeedbackDecision:
        if os.environ.get("OPENAI_API_KEY", ""):
            return self._decide_llm(state=state, critique=critique, available_tools=available_tools)
        return self._decide_heuristic(critique=critique, available_tools=available_tools)

    def _decide_heuristic(self, *, critique: JsonDict, available_tools: List[str]) -> FeedbackDecision:
        n = _count_points(critique)
        miss = _missing_span_ids(critique)
        requested: List[str] = []

        if n < self.cfg.min_points:
            # try to enrich evidence and comparisons
            if "behavior_topk_direct_compare" in available_tools:
                requested.append("behavior_topk_direct_compare")
            if "lens_diagnostic_consistency" in available_tools:
                requested.append("lens_diagnostic_consistency")

        ok = (n >= self.cfg.min_points) and (miss == 0)
        reason = f"heuristic_feedback points={n} missing_span_id={miss}"
        return FeedbackDecision(ok=ok, reason=reason, requested_tools=requested, patch_instructions="")

    def _decide_llm(self, *, state: AgentState, critique: JsonDict, available_tools: List[str]) -> FeedbackDecision:
        payload = {
            "router": state.router,
            "preprocessing_keys": list((state.preprocessing or {}).keys()),
            "lens_keys": list((state.lens_results or {}).keys()),
            "behavior_keys": list((state.behavior_results or {}).keys()),
            "critique": critique,
            "available_tools": available_tools,
        }
        prompt = f"""You are a quality judge for a clinical critique report.
Decide if the report is sufficient for a user-facing output.
If insufficient, request additional tools (from available_tools) and provide short patch instructions.

Input JSON:
{payload}

Output JSON only:
{{
  "ok": true|false,
  "reason": "one short sentence",
  "requested_tools": ["tool_name", "..."],
  "patch_instructions": "how to revise the critique after tools run (1-3 sentences)"
}}

Rules:
- requested_tools MUST be subset of available_tools.
- Prefer minimal additional tools.
- If evidence spans are missing, prioritize preprocessing already done (do not ask for it again).
"""
        cfg = OpenAIChatConfig(model=self.cfg.model, temperature=0.1, max_tokens=400)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}

        ok = bool(obj.get("ok", False))
        requested = obj.get("requested_tools") if isinstance(obj.get("requested_tools"), list) else []
        requested = [t for t in requested if isinstance(t, str) and t in available_tools]
        return FeedbackDecision(
            ok=ok,
            reason=str(obj.get("reason", "") or ""),
            requested_tools=requested,
            patch_instructions=str(obj.get("patch_instructions", "") or ""),
            raw=content,
        )

