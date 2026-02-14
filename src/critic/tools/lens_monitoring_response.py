from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..tool_base import Tool
from ..types import AgentState, JsonDict, ToolCard
from ...llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


DETERIORATION_KW = ["worsen", "deterior", "declin", "hypotens", "desat", "arrest", "code blue", "rapid response"]
RESPONSE_KW = ["intub", "vent", "pressor", "vasopress", "fluid", "antibi", "icu", "transfer", "dialysis"]


def _match_kw(text: str, kws: List[str]) -> bool:
    return any(k in (text or "").lower() for k in kws)


def _find_span_by_kw(evidence_spans: Dict[str, Dict[str, Any]], kws: List[str]) -> Optional[str]:
    for sid, sp in evidence_spans.items():
        q = str(sp.get("quote", "")).lower()
        for k in kws:
            if k in q:
                return sid
    return None


@dataclass
class LensMonitoringResponseTool(Tool):
    def __init__(self):
        super().__init__(
            name="lens_monitoring_response",
            card=ToolCard(
                name="lens_monitoring_response",
                description="악화 징후 발생 시 재평가/모니터링/치료 수정이 따라왔는지(지연 포함) 점검.",
                triggers=["악화 단서", "중증도 플래그", "반응/에스컬레이션 평가 필요"],
                input_contract={"preprocessing.timeline": "Dict[events]", "preprocessing.evidence": "Dict[evidence_spans]"},
                output_contract={"deterioration_points": "List", "response_actions": "List", "lags": "List"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        timeline = state.preprocessing.get("timeline") or {}
        events = timeline.get("events", []) if isinstance(timeline, dict) else []
        evidence = state.preprocessing.get("evidence") or {}
        evidence_spans = (evidence.get("evidence_spans") or {}) if isinstance(evidence, dict) else {}
        if os.environ.get("OPENAI_API_KEY", ""):
            out = self._run_llm(events=events, evidence_spans=evidence_spans, state=state)
        else:
            out = self._run_heuristic(events=events, evidence_spans=evidence_spans)
        if isinstance(getattr(state, "cohort_data", None), dict) and state.cohort_data.get("process_contributor_analysis"):
            out["reference_process_contributor_analysis"] = state.cohort_data.get("process_contributor_analysis")
        return out

    def _run_heuristic(self, *, events: List[Dict], evidence_spans: Dict[str, Dict[str, Any]]) -> JsonDict:
        det = []
        resp = []
        for ev in events[:80]:
            txt = str(ev.get("text", ""))
            if _match_kw(txt, DETERIORATION_KW):
                det.append({"event_id": ev.get("event_id"), "time_hint": ev.get("time_hint"), "text": txt[:280]})
            if _match_kw(txt, RESPONSE_KW):
                resp.append({"event_id": ev.get("event_id"), "time_hint": ev.get("time_hint"), "text": txt[:280]})
        lags = []
        if det and resp:
            idx_map = {ev.get("event_id"): i for i, ev in enumerate(events)}
            for d in det[:3]:
                di = idx_map.get(d["event_id"])
                if di is None:
                    continue
                after = [r for r in resp if idx_map.get(r["event_id"], 10**9) > di]
                if not after:
                    lags.append({
                        "deterioration_event_id": d["event_id"],
                        "response_event_id": None,
                        "lag_events": None,
                        "span_id": _find_span_by_kw(evidence_spans, DETERIORATION_KW) or "record_uncertainty",
                        "note": "악화 후 반응 이벤트를 텍스트에서 확실히 찾지 못함(기록 공백 가능)",
                    })
                    continue
                r0 = after[0]
                lag = idx_map.get(r0["event_id"], di) - di
                lags.append({
                    "deterioration_event_id": d["event_id"],
                    "response_event_id": r0["event_id"],
                    "lag_events": lag,
                    "span_id": _find_span_by_kw(evidence_spans, DETERIORATION_KW) or "record_uncertainty",
                })
        return {"deterioration_points": det[:10], "response_actions": resp[:12], "lags": lags, "note": "LLM 미사용(키 없음)으로 키워드 기반 추정입니다."}

    def _run_llm(self, *, events: List[Dict], evidence_spans: Dict[str, Dict[str, Any]], state: AgentState = None) -> JsonDict:
        payload = {"timeline_events": events[:60], "evidence_spans": evidence_spans}
        if state and isinstance(getattr(state, "cohort_data", None), dict) and state.cohort_data.get("process_contributor_analysis"):
            payload["reference_only_prior_process_contributor_analysis"] = state.cohort_data.get("process_contributor_analysis")
        prompt = f"""You are a monitoring/response auditor. Identify deterioration points and whether appropriate response followed.
Input JSON: {payload}
If "reference_only_prior_process_contributor_analysis" is present, use it only as reference; do not depend on it.
Return JSON only: {{ "deterioration_points": [{{"event_id":"T1","summary":"...","span_id":"E1"}}], "response_actions": [{{"event_id":"T2","summary":"...","span_id":"E2"}}], "lags": [{{"deterioration_event_id":"T1","response_event_id":"T2","lag_events":3,"span_id":"E1"}}] }}
Rules: span_id must exist in evidence_spans or use record_uncertainty. Be conservative if documentation is unclear."""
        cfg = OpenAIChatConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=900)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}
        obj["raw"] = content
        return obj
