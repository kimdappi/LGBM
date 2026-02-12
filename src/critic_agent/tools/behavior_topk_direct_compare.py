from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from ..tool_base import Tool
from ..types import AgentState, JsonDict, ToolCard
from ...llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _tokenize_light(text: str) -> List[str]:
    t = (text or "").lower()
    out, cur = [], []
    for ch in t:
        if ch.isalnum():
            cur.append(ch)
        else:
            if len(cur) >= 3:
                out.append("".join(cur))
            cur = []
    if len(cur) >= 3:
        out.append("".join(cur))
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:400]


@dataclass
class BehaviorTopKDirectCompareTool(Tool):
    def __init__(self):
        super().__init__(
            name="behavior_topk_direct_compare",
            card=ToolCard(
                name="behavior_topk_direct_compare",
                description="Top-K 유사 케이스와 patient를 항목별로 직접 비교해 근거 강화.",
                triggers=["텍스트가 짧음", "근거 강화 필요", "유사 케이스가 존재"],
                input_contract={"cohort_data.similar_cases": "List[Dict[text,...]]", "preprocessing.evidence": "Dict"},
                output_contract={"comparisons": "List", "summary": "str"},
            ),
        )

    def run(self, state: AgentState) -> JsonDict:
        similar_cases = (state.cohort_data.get("similar_cases") or []) if isinstance(state.cohort_data.get("similar_cases"), list) else []
        evidence = state.preprocessing.get("evidence") or {}
        evidence_spans = (evidence.get("evidence_spans") or {}) if isinstance(evidence, dict) else {}
        if not similar_cases:
            return {"comparisons": [], "summary": "similar_cases가 없어 비교를 수행하지 않았습니다."}
        if os.environ.get("OPENAI_API_KEY", ""):
            return self._run_llm(state=state, similar_cases=similar_cases, evidence_spans=evidence_spans)
        return self._run_heuristic(similar_cases=similar_cases, evidence_spans=evidence_spans)

    def _run_heuristic(self, *, similar_cases: List[Dict[str, Any]], evidence_spans: Dict[str, Dict[str, Any]]) -> JsonDict:
        patient_terms = []
        for sid, sp in list(evidence_spans.items())[:25]:
            patient_terms.extend(_tokenize_light(str(sp.get("quote", ""))))
        patient_set = set(patient_terms)
        comps = []
        for i, c in enumerate(similar_cases[:3], 1):
            case_text = str(c.get("text", "") or "")[:6000]
            overlap = list(patient_set & set(_tokenize_light(case_text)))
            comps.append({"case_rank": i, "case_id": c.get("id"), "similarity": c.get("similarity"), "overlap_terms_sample": overlap[:20], "overlap_count": len(overlap), "note": "휴리스틱(토큰 중복) 기반 비교."})
        return {"comparisons": comps, "summary": "Top-K 케이스와 patient 근거 span에서 공통 토큰을 비교했습니다(키 없음)."}

    def _run_llm(self, *, state: AgentState, similar_cases: List[Dict[str, Any]], evidence_spans: Dict[str, Dict[str, Any]]) -> JsonDict:
        cases_block = [{"rank": i, "case_id": c.get("id"), "similarity": c.get("similarity"), "status": c.get("status"), "age": c.get("age"), "sex": c.get("sex"), "text": str(c.get("text", "") or "")[:2500]} for i, c in enumerate(similar_cases[:3], 1)]
        payload = {"patient": {"id": state.patient.get("id"), "age": state.patient.get("age"), "sex": state.patient.get("sex"), "text": str(state.patient.get("text", "") or "")[:2500]}, "patient_evidence_spans": evidence_spans, "similar_cases": cases_block}
        prompt = f"""You are a contrastive comparator for clinical process review. Compare the patient to each similar case (problems, workup, therapies, monitoring/response).
Input JSON: {payload}
Return JSON only: {{ "comparisons": [{{ "case_id":"...", "key_similarities":["..."], "key_differences":["..."], "evidence_links":[{{"span_id":"E1","why":"..."}}] }}], "summary": "1-3 sentences" }}
Rules: evidence_links must use existing span_id or record_uncertainty."""
        cfg = OpenAIChatConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=1200)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {"summary": content, "comparisons": []}
        obj["raw"] = content
        return obj
