from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from ..tool_base import Tool, get_patient_text
from ..types import AgentState, JsonDict, ToolCard


def _missing_vitals(text: str) -> List[str]:
    t = (text or "").lower()
    vitals = {
        "bp": [r"\bbp\b", "blood pressure", "mmhg", "hypotension"],
        "hr": [r"\bhr\b", "heart rate", "tachy", "brady"],
        "rr": [r"\brr\b", "resp rate", "respiratory rate"],
        "spo2": ["spo2", "o2 sat", "sat%"],
        "temp": ["temp", "temperature", "febrile"],
    }
    missing = []
    for k, pats in vitals.items():
        ok = False
        for p in pats:
            if p.startswith("\\b") or p.startswith("(") or "[" in p or "\\" in p:
                if re.search(p, t):
                    ok = True
                    break
            else:
                if p in t:
                    ok = True
                    break
        if not ok:
            missing.append(k)
    return missing


@dataclass
class PreprocessRecordGapTool(Tool):
    def __init__(self):
        super().__init__(
            name="record_gaps",
            card=ToolCard(
                name="record_gaps",
                description="기록 공백(누락/불확실성) 태깅.",
                triggers=["텍스트가 짧음", "근거 부족", "불확실성 표기 필요"],
                input_contract={"patient.text": "str"},
                output_contract={"missing": "List[str]", "uncertainty_markers": "List[str]"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        text = get_patient_text(state)
        t = (text or "").lower()
        uncertainty_markers: List[str] = []
        for phrase in ["not documented", "unknown", "unable to", "limited history", "poor historian", "unclear", "cannot confirm"]:
            if phrase in t:
                uncertainty_markers.append(phrase)
        missing_vitals = _missing_vitals(text)
        missing: List[str] = []
        if missing_vitals:
            missing.append(f"vitals_missing: {', '.join(missing_vitals)} (텍스트 상 명시 부족)")
        if len(text) < 600:
            missing.append("clinical_text_short: 요약/근거가 빈약할 가능성")
        return {"missing": missing, "uncertainty_markers": uncertainty_markers}
