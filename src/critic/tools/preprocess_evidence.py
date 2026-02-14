from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..tool_base import Tool, get_patient_text
from ..types import AgentState, JsonDict, ToolCard


def _iter_sentence_offsets(text: str) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    cursor = 0
    for chunk in re.split(r"\n+", text):
        if not chunk.strip():
            cursor += len(chunk) + 1
            continue
        start = text.find(chunk, cursor)
        if start == -1:
            continue
        end = start + len(chunk)
        cursor = end
        parts = re.split(r"(?<=[\.\?!])\s+", chunk.strip())
        local_cursor = start
        for p in parts:
            if not p.strip():
                continue
            ps = p.strip()
            ps_start = text.find(ps, local_cursor)
            if ps_start == -1:
                continue
            ps_end = ps_start + len(ps)
            local_cursor = ps_end
            out.append((ps_start, ps_end, ps))
    return out


def _category(sentence: str) -> str:
    s = sentence.lower()
    if re.search(r"\b(bp|hr|rr|spo2|sat|temp)\b", s):
        return "vital_signs"
    if re.search(r"\b(wbc|hgb|plt|cr|creatin|bun|lactate|na|k|ast|alt|bilir|abg)\b", s):
        return "labs"
    if any(k in s for k in ["ct", "x-ray", "mri", "ultrasound", "echo"]):
        return "imaging"
    if any(k in s for k in ["diagnosis", "impression", "assessment", "a/p", "problem", "plan"]):
        return "assessment"
    if any(k in s for k in ["antibi", "heparin", "warfarin", "insulin", "steroid", "vasopress", "fluid"]):
        return "therapy"
    return "other"


def _is_claimable(sentence: str) -> bool:
    s = sentence.lower()
    if len(s) < 20:
        return False
    if re.search(r"\d", s):
        return True
    return any(k in s for k in ["diagnosis", "impression", "assessment", "plan", "started", "given", "administered", "intub", "icu", "transfer", "worsen", "deterior"])


@dataclass
class PreprocessEvidenceTool(Tool):
    def __init__(self):
        super().__init__(
            name="evidence",
            card=ToolCard(
                name="evidence",
                description="clinical text에서 근거 문장을 span으로 구조화.",
                triggers=["근거 인용 필요", "비판 포인트에 span_id 부여"],
                input_contract={"patient.text": "str"},
                output_contract={"evidence_spans": "Dict[Eid->{category,quote,start,end}]"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        text = get_patient_text(state)
        spans: Dict[str, Dict] = {}
        sent_offsets = _iter_sentence_offsets(text)
        eid = 1
        for start, end, sent in sent_offsets:
            if not _is_claimable(sent):
                continue
            spans[f"E{eid}"] = {"category": _category(sent), "quote": sent[:600], "start": start, "end": end}
            eid += 1
            if eid > 60:
                break
        return {"evidence_spans": spans, "span_count": len(spans)}
