from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import AgentState, JsonDict, ToolCard


@dataclass
class Tool:
    """
    Base class for critic agent tools.
    Each tool should be single-responsibility and write results into AgentState.
    """

    name: str
    card: ToolCard

    def run(self, state: AgentState) -> JsonDict:
        raise NotImplementedError

    def safe_run(self, state: AgentState) -> JsonDict:
        try:
            out = self.run(state)
            state.add_trace(tool=self.name, status="ok")
            return out
        except Exception as e:
            state.add_trace(tool=self.name, status="error", detail={"error": f"{type(e).__name__}: {e}"})
            return {"error": f"{type(e).__name__}: {e}"}


def get_patient_text(state: AgentState) -> str:
    return str(state.patient.get("text", "") or "")


def ensure_preprocessing(state: AgentState, key: str) -> Optional[Dict[str, Any]]:
    obj = state.preprocessing.get(key)
    return obj if isinstance(obj, dict) else None

