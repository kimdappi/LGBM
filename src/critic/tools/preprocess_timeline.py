from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..tool_base import Tool, get_patient_text
from ..types import AgentState, JsonDict, ToolCard


def _sentence_spans(text: str) -> List[Dict]:
    """
    Very lightweight sentence splitter that preserves offsets.
    """
    if not text:
        return []
    spans: List[Dict] = []
    # split on newline or period followed by whitespace
    pattern = re.compile(r"(.+?)(?:\n+|(?<=\.)\s+|$)", flags=re.DOTALL)
    cursor = 0
    for m in pattern.finditer(text):
        s = m.group(1)
        if not s:
            continue
        start = text.find(s, cursor)
        if start == -1:
            continue
        end = start + len(s)
        cursor = end
        s_clean = s.strip()
        if not s_clean:
            continue
        spans.append({"start": start, "end": end, "text": s_clean})
    return spans


def _extract_time_hint(s: str) -> Optional[str]:
    if not s:
        return None
    patterns = [
        r"\b\d{1,2}:\d{2}\b",
        r"\b(?:day|hd|hospital day)\s*\d+\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _tag_event_type(s: str) -> str:
    t = s.lower()
    if any(k in t for k in ["intub", "vent", "cpap", "bipap", "o2", "oxygen", "desat", "spo2"]):
        return "respiratory"
    if any(k in t for k in ["bp", "hypotens", "press", "shock", "tachy", "brady", "hr "]):
        return "hemodynamic"
    if any(k in t for k in ["ct", "x-ray", "mri", "ultrasound", "imaging"]):
        return "imaging"
    if any(k in t for k in ["lab", "wbc", "cr", "creatin", "lactate", "troponin", "abg"]):
        return "lab"
    if any(k in t for k in ["antibi", "heparin", "warfarin", "insulin", "steroid", "vasopress"]):
        return "medication"
    if any(k in t for k in ["procedure", "line", "catheter", "dialysis", "surgery"]):
        return "procedure"
    if any(k in t for k in ["icu", "ward", "transfer", "stepdown", "discharge"]):
        return "level_of_care"
    if any(k in t for k in ["worsen", "deterior", "declin", "arrest", "code blue", "rapid response"]):
        return "deterioration"
    return "other"


@dataclass
class PreprocessTimelineTool(Tool):
    def __init__(self):
        super().__init__(
            name="timeline",
            card=ToolCard(
                name="timeline",
                description="입원~퇴원 흐름을 event list로 정규화(오프셋 포함).",
                triggers=["긴 텍스트", "시간 순 사건 파악 필요", "악화/반응 분석 전처리"],
                input_contract={"patient.text": "str"},
                output_contract={"events": "List[{event_id,type,time_hint,text,start,end}]"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        text = get_patient_text(state)
        spans = _sentence_spans(text)
        events: List[Dict] = []
        for i, sp in enumerate(spans, 1):
            s = sp["text"]
            events.append(
                {
                    "event_id": f"T{i}",
                    "type": _tag_event_type(s),
                    "time_hint": _extract_time_hint(s),
                    "text": s[:4000],
                    "start": sp["start"],
                    "end": sp["end"],
                }
            )
            if len(events) >= 120:
                break
        return {"events": events, "event_count": len(events)}

