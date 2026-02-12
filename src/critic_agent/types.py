from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class ToolCard:
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    anti_triggers: List[str] = field(default_factory=list)
    input_contract: JsonDict = field(default_factory=dict)
    output_contract: JsonDict = field(default_factory=dict)

    def to_text(self) -> str:
        return "\n".join(
            [
                f"[Tool] {self.name}",
                f"Description: {self.description}",
                f"Triggers: {', '.join(self.triggers) if self.triggers else '-'}",
                f"Anti-triggers: {', '.join(self.anti_triggers) if self.anti_triggers else '-'}",
            ]
        )


@dataclass
class ToolSelection:
    tools: List[str]
    reason: str = ""
    retrieved_cards: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    patient: JsonDict
    cohort_data: JsonDict
    similar_case_patterns: JsonDict = field(default_factory=dict)

    preprocessing: JsonDict = field(default_factory=dict)
    lens_results: JsonDict = field(default_factory=dict)
    behavior_results: JsonDict = field(default_factory=dict)

    router: JsonDict = field(default_factory=dict)
    trace: List[JsonDict] = field(default_factory=list)

    def add_trace(self, *, tool: str, status: str, detail: Optional[JsonDict] = None) -> None:
        payload: JsonDict = {"tool": tool, "status": status}
        if detail:
            payload["detail"] = detail
        self.trace.append(payload)

