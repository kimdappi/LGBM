from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .types import ToolCard
from .tool_base import Tool


@dataclass
class AgentConfig:
    """Critic 에이전트 공통 설정 (LangGraph 서브그래프에서도 공유)."""

    max_tools: int = 8
    router_llm_model: str = "gpt-4o-mini"
    critique_model: str = "gpt-4o-mini"


@dataclass
class ToolRegistry:
    """도구 레지스트리: critic_graph 및 기타 유틸에서 사용."""

    tools: Dict[str, Tool]

    @property
    def available_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    @property
    def cards(self) -> List[ToolCard]:
        return [t.card for t in self.tools.values()]

    def get(self, name: str) -> Tool:
        return self.tools[name]

