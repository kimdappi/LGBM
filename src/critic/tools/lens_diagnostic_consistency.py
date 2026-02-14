from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..tool_base import Tool
from ..types import AgentState, JsonDict, ToolCard
from ...llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _pick_assessment_spans(evidence_spans: Dict[str, Dict[str, Any]]) -> List[str]:
    return [sid for sid, sp in evidence_spans.items() if sp.get("category") == "assessment"][:10]


@dataclass
class LensDiagnosticConsistencyTool(Tool):
    def __init__(self):
        super().__init__(
            name="lens_diagnostic_consistency",
            card=ToolCard(
                name="lens_diagnostic_consistency",
                description="진단 결론이 근거/논리와 정렬되는지 점검. 갭/모순 탐지.",
                triggers=["진단이 강하게 주장됨", "근거가 빈약", "대안진단 배제 근거 필요"],
                input_contract={"preprocessing.evidence": "Dict[evidence_spans]", "preprocessing.timeline": "Dict[events]"},
                output_contract={"diagnosis_claims": "List[str]", "supporting_evidence": "List[span_id]", "gaps": "List[str]", "contradictions": "List[str]"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        evidence = state.preprocessing.get("evidence") or {}
        evidence_spans = (evidence.get("evidence_spans") or {}) if isinstance(evidence, dict) else {}
        timeline = state.preprocessing.get("timeline") or {}
        out: JsonDict
        if os.environ.get("OPENAI_API_KEY", ""):
            out = self._run_llm(evidence_spans=evidence_spans, timeline=timeline, patient=state.patient, state=state)
        else:
            out = self._run_heuristic(evidence_spans=evidence_spans)
        # 참고용: 앞단 diagnosis_analysis 있으면 붙임 (의존 아님)
        if isinstance(getattr(state, "cohort_data", None), dict) and state.cohort_data.get("diagnosis_analysis"):
            out["reference_diagnosis_analysis"] = state.cohort_data.get("diagnosis_analysis")
        return out

    def _run_heuristic(self, evidence_spans: Dict[str, Dict[str, Any]]) -> JsonDict:
        claims = []
        for sid, sp in evidence_spans.items():
            if sp.get("category") != "assessment":
                continue
            q = str(sp.get("quote", "")).strip()
            if not q:
                continue
            m = re.search(r"(diagnosis|impression|assessment)\s*[:\-]\s*(.+)$", q, flags=re.IGNORECASE)
            claims.append(m.group(2)[:200] if m else q[:200])
            if len(claims) >= 6:
                break
        gaps = [] if claims else ["명시적 진단/평가 문장이 근거 span에서 충분히 확인되지 않음(기록 공백 가능)."]
        return {"diagnosis_claims": claims, "supporting_evidence": _pick_assessment_spans(evidence_spans), "gaps": gaps, "contradictions": [], "note": "LLM 미사용(키 없음)으로 휴리스틱 추정입니다."}

    def _run_llm(self, *, evidence_spans: Dict[str, Dict[str, Any]], timeline: Dict, patient: Dict, state: AgentState = None) -> JsonDict:
        payload = {
            "patient": {k: patient.get(k) for k in ["age", "sex", "admission_type", "admission_location", "status"]},
            "evidence_spans": evidence_spans,
            "timeline_events": (timeline.get("events") or [])[:40],
        }
        if state and isinstance(getattr(state, "cohort_data", None), dict) and state.cohort_data.get("diagnosis_analysis"):
            payload["reference_only_prior_diagnosis_analysis"] = state.cohort_data.get("diagnosis_analysis")
        prompt = f"""You are a diagnostic reasoning auditor. Check whether diagnostic conclusions are aligned with evidence.
Input JSON: {payload}
If "reference_only_prior_diagnosis_analysis" is present, use it only as reference; do not depend on it. Base your output on evidence_spans and timeline.
Return JSON only: {{ "diagnosis_claims": ["..."], "supporting_evidence": ["E1","E2"], "gaps": ["..."], "contradictions": ["..."] }}
Rules: Only cite span_ids that exist. If record is insufficient, say so in gaps."""
        cfg = OpenAIChatConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=900)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}
        return {"diagnosis_claims": obj.get("diagnosis_claims", []), "supporting_evidence": obj.get("supporting_evidence", []), "gaps": obj.get("gaps", []), "contradictions": obj.get("contradictions", []), "raw": content}

