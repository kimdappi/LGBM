from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..tool_base import Tool
from ..types import AgentState, JsonDict, ToolCard


SEVERITY_SIGNALS: List[Tuple[str, List[str]]] = [
    ("shock_or_hypotension", ["shock", "hypotension", "map", "pressors", "vasopress"]),
    ("resp_failure", ["intub", "vent", "respiratory failure", "bipap", "cpap", "desat", "spo2"]),
    ("sepsis", ["sepsis", "septic", "lactate", "broad spectrum", "abx", "antibiotic"]),
    ("cardiac_instability", ["vtach", "vfib", "arrest", "troponin", "stemi", "nstemi"]),
]


def _find_span_for_keyword(evidence_spans: Dict[str, Dict], kw: str) -> Optional[str]:
    kw_l = kw.lower()
    for sid, sp in evidence_spans.items():
        if kw_l in str(sp.get("quote", "")).lower():
            return sid
    return None


@dataclass
class LensSeverityRiskTool(Tool):
    def __init__(self):
        super().__init__(
            name="lens_severity_risk",
            card=ToolCard(
                name="lens_severity_risk",
                description="중증도/리스크 신호 태깅 및 누락된 중증도 평가 탐지.",
                triggers=["악화/쇼크/호흡부전 단서", "레벨오브케어 판단 필요"],
                input_contract={"preprocessing.evidence": "Dict", "patient.metadata": "Dict"},
                output_contract={"severity_flags": "List", "missing_severity_assessment": "List[str]"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        evidence = state.preprocessing.get("evidence") or {}
        evidence_spans = (evidence.get("evidence_spans") or {}) if isinstance(evidence, dict) else {}
        record_gaps = state.preprocessing.get("record_gaps") or {}
        missing = record_gaps.get("missing", []) if isinstance(record_gaps, dict) else []
        flags: List[Dict] = []
        for signal, kws in SEVERITY_SIGNALS:
            for kw in kws:
                span_id = _find_span_for_keyword(evidence_spans, kw)
                if span_id:
                    flags.append({"signal": signal, "kw": kw, "span_id": span_id, "confidence": "medium"})
                    break
        missing_sev: List[str] = []
        if any("vitals_missing" in str(x) for x in missing):
            missing_sev.append("vitals_not_documented_in_text: 중증도 판단에 필요한 활력징후가 텍스트에서 충분히 확인되지 않음")
        return {"severity_flags": flags, "missing_severity_assessment": missing_sev}
