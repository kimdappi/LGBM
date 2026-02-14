from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import AgentState, JsonDict
from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions, safe_json_loads


def _as_json(x: Any) -> Any:
    if isinstance(x, (dict, list, str, int, float, bool)) or x is None:
        return x
    return str(x)


# ──────────────────────────────────────────────
# Severity Hierarchy 재정렬 (CritiqueBuilder 후처리)
# ──────────────────────────────────────────────

_IATROGENIC_PATTERNS = [
    r"iatrogenic", r"hemoperitoneum", r"organ.?injur",
    r"procedur.{0,20}complication", r"procedur.{0,20}bleed",
    r"puncture.{0,10}bleed", r"blind(?:ly)?\s+(?:paracentesis|thoracentesis|procedure)",
    r"without.{0,15}(?:ultrasound|guidance)", r"perforation",
]

_DEATH_ALIGNMENT_PATTERNS = [
    r"cause.{0,10}death", r"death.{0,10}cause",
    r"admission.{0,20}(?:differ|mismatch|vs).{0,20}(?:death|expir)",
    r"last.{0,5}24.{0,5}hour", r"terminal.{0,10}event",
    r"hct.{0,5}drop", r"hemodynamic.{0,10}instabilit",
]


def _rerank_critique_points(pts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Severity Hierarchy에 따라 critique_points를 재정렬.
    
    Priority 0: Iatrogenic trauma / procedural complications → 자동 severity=high
    Priority 1: Cause of death alignment issues
    Priority 2: Other high severity
    Priority 3: Medium
    Priority 4: Low
    """
    severity_order = {"high": 0, "critical": 0, "medium": 1, "low": 2}
    
    def sort_key(pt: Dict[str, Any]):
        text = (str(pt.get("point", "")) + str(pt.get("cohort_comparison", ""))).lower()
        
        # Priority 0: Iatrogenic
        is_iatrogenic = any(re.search(p, text) for p in _IATROGENIC_PATTERNS)
        if is_iatrogenic:
            pt["severity"] = "high"
            return (0, 0)
        
        # Priority 1: Death alignment
        is_death = any(re.search(p, text) for p in _DEATH_ALIGNMENT_PATTERNS)
        if is_death:
            return (1, severity_order.get(pt.get("severity", "low"), 2))
        
        # Normal severity ordering
        sev = severity_order.get(pt.get("severity", "low"), 2)
        return (2 + sev, sev)
    
    return sorted(pts, key=sort_key)


@dataclass
class CritiqueBuilder:
    """
    Merge preprocessing + lens + behavior into a single critique object.

    If OPENAI_API_KEY is set, uses LLM to generate structured critique points with evidence_span ids.
    Otherwise, falls back to a lightweight heuristic summary.
    """

    model: str = "gpt-4o-mini"

    def build(
        self,
        state: AgentState,
        *,
        previous_critique: Optional[JsonDict] = None,
        patch_instructions: str = "",
    ) -> JsonDict:
        # LLM path (preferred)
        if os.environ.get("OPENAI_API_KEY", ""):
            return self._build_with_llm(state, previous_critique=previous_critique, patch_instructions=patch_instructions)
        return self._build_heuristic(state)

    def _build_heuristic(self, state: AgentState) -> JsonDict:
        pts: List[Dict[str, Any]] = []

        sev = state.lens_results.get("lens_severity_risk") or {}
        mon = state.lens_results.get("lens_monitoring_response") or {}
        dx = state.lens_results.get("lens_diagnostic_consistency") or {}
        topk_cmp = state.behavior_results.get("behavior_topk_direct_compare") or {}

        if isinstance(sev, dict) and sev.get("severity_flags"):
            pts.append(
                {
                    "point": "중증도/리스크 신호가 존재합니다. 모니터링/레벨오브케어/에스컬레이션이 적절했는지 재검토가 필요합니다.",
                    "span_id": (sev.get("severity_flags") or [{}])[0].get("span_id", "record_uncertainty"),
                    "severity": "high",
                    "cohort_comparison": str(topk_cmp.get("summary", "N/A"))[:400] if isinstance(topk_cmp, dict) else "N/A",
                }
            )

        if isinstance(mon, dict) and mon.get("lags"):
            pts.append(
                {
                    "point": "악화 신호 대비 반응(재평가/치료수정)이 지연되었을 가능성이 있습니다.",
                    "span_id": (mon.get("lags") or [{}])[0].get("span_id", "record_uncertainty"),
                    "severity": "high",
                    "cohort_comparison": str(topk_cmp.get("summary", "N/A"))[:400] if isinstance(topk_cmp, dict) else "N/A",
                }
            )

        if isinstance(dx, dict) and (dx.get("gaps") or dx.get("contradictions")):
            pts.append(
                {
                    "point": "진단 결론과 근거 간 정렬이 불충분할 수 있습니다. 대안 진단 배제 근거/추가 평가 필요성을 점검하세요.",
                    "span_id": "record_uncertainty",
                    "severity": "medium",
                    "cohort_comparison": str(topk_cmp.get("summary", "N/A"))[:400] if isinstance(topk_cmp, dict) else "N/A",
                }
            )

        return {
            "patient_id": state.patient.get("id"),
            "analysis": "LLM 미사용(키 없음)으로 요약 기반 비판을 생성했습니다. (Top-K 비교 결과가 있으면 cohort_comparison에 반영)",
            "critique_points": _rerank_critique_points(pts),
            "risk_factors": [],
            "recommendations": [],
            "agent_mode": True,
        }

    def _build_with_llm(
        self,
        state: AgentState,
        *,
        previous_critique: Optional[JsonDict] = None,
        patch_instructions: str = "",
    ) -> JsonDict:
        payload = {
            "patient": {
                "id": state.patient.get("id"),
                "age": state.patient.get("age"),
                "sex": state.patient.get("sex"),
                "status": state.patient.get("status"),
                "admission_type": state.patient.get("admission_type"),
                "admission_location": state.patient.get("admission_location"),
            },
            "preprocessing": _as_json(state.preprocessing),
            "lens_results": _as_json(state.lens_results),
            "behavior_results": _as_json(state.behavior_results),
            "cohort_patterns": _as_json(state.similar_case_patterns),
            "previous_critique": _as_json(previous_critique) if previous_critique else None,
            "patch_instructions": patch_instructions or "",
        }
        cohort = getattr(state, "cohort_data", None)
        if isinstance(cohort, dict):
            ref = {}
            if cohort.get("diagnosis_analysis") is not None:
                ref["diagnosis_analysis"] = cohort.get("diagnosis_analysis")
            if cohort.get("treatment_analysis") is not None:
                ref["treatment_analysis"] = cohort.get("treatment_analysis")
            if cohort.get("evidence") is not None:
                ref["evidence"] = cohort.get("evidence")
            if cohort.get("intervention_coverage") is not None:
                ref["intervention_coverage"] = cohort.get("intervention_coverage")
            if ref:
                payload["reference_only_prior_results"] = ref

        mode_line = "Revise the previous critique using patch_instructions." if previous_critique else "Generate a critique report."

        prompt = f"""You are a critical medical process reviewer.
{mode_line}
Use ONLY the provided structured evidence. Do not invent unobserved facts.
If "reference_only_prior_results" is present in Input JSON, use it only as reference; do not depend on it. Base your critique on preprocessing, lens_results, behavior_results.

Input JSON:
{payload}

Return JSON only:
{{
  "analysis": "overall assessment (2-6 sentences)",
  "critique_points": [
    {{
      "point": "specific critique",
      "span_id": "E1 | E2 | ... | record_uncertainty",
      "severity": "high|medium|low",
      "cohort_comparison": "how top-k similar cases differed (if available)"
    }}
  ],
  "risk_factors": ["..."],
  "recommendations": ["..."]
}}

Rules:
- Every critique point MUST cite span_id OR record_uncertainty.
- If the record is insufficient, explicitly say so and prefer cautious language.
- If behavior_topk_direct_compare exists, use it to fill cohort_comparison with concrete differences.
- If previous_critique is provided, keep good parts and fix only what is requested.

★ CRITICAL SEVERITY HIERARCHY (MUST follow this order):
1. **Iatrogenic Trauma / Procedural Complications** = HIGHEST PRIORITY (severity: high)
   - If any procedure (paracentesis, central line, thoracentesis, etc.) caused bleeding, organ injury, hemoperitoneum, or perforation → this MUST be the #1 critique point
   - "Blind" or undocumented technique in high-risk procedures → severity: high
   - Procedural complications leading to death outweigh ALL other issues
2. **Cause of Death Alignment**
   - If admission reason differs from cause of death, explicitly identify the actual death pathway
   - Weight the last 24 hours' events (Hct drop, hemodynamic instability, procedure complications) heavily
3. **Medication Errors** (e.g., benzos in HE, NSAIDs in CKD) = important but LOWER priority than iatrogenic trauma
4. **Other diagnostic/process issues** = lowest priority

Do NOT let medication errors overshadow iatrogenic procedural complications in the ranking.
"""

        cfg = OpenAIChatConfig(model=self.model, temperature=0.2, max_tokens=1600)
        content = call_openai_chat_completions(messages=[{"role": "user", "content": prompt}], config=cfg)
        obj = safe_json_loads(content) or {}

        # Severity Hierarchy 후처리: rerank critique_points
        critique_pts = obj.get("critique_points", [])
        critique_pts = _rerank_critique_points(critique_pts)

        out: JsonDict = {
            "patient_id": state.patient.get("id"),
            "analysis": obj.get("analysis", content),
            "critique_points": critique_pts,
            "risk_factors": obj.get("risk_factors", []),
            "recommendations": obj.get("recommendations", []),
            "agent_mode": True,
            "raw": content,
        }
        return out

