"""
Agent Router: 문서(에이전트 설명·조건)를 보고 현재 state에 맞는 에이전트를 선택.
OpenAI LLM 사용. 선택된 에이전트만 이후 run_conditional_agents / run_alternative_explanation에서 실행.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Any

from .llm import get_llm


AGENT_DOCUMENT = """
[선택 가능한 에이전트 목록 및 활용 조건]

1. risk_factor
   - 조건: 동반질환(comorbidity) 2개 이상 AND cohort 사망률이 높음(유사 케이스 중 expired 비율 높음)
   - 역할: 고위험 케이스일 때 위험 인자·동반질환 집중 분석

2. process_contributor
   - 조건: 치료 지연·누락 키워드 포함(delay, missed, late, 지연, 누락 등) OR 해당 케이스 outcome이 cohort 평균과 반대 방향(예: cohort는 대부분 생존인데 이 케이스만 사망 또는 그 반대)
   - 역할: 이상치·프로세스 기여 의심 시 지연·불일치 분석

3. alternative_explanation
   - 조건: 유사 케이스 similarity 분산이 큼 OR cohort 패턴이 일관되지 않음(진단/결과 분포 산만) OR 비판 불확실성 예상(진단 불명, 근거 빈약)
   - 역할: 해석이 애매할 때 대안 설명·불확실성 정리 (Critic 실행 후 사용)
"""


def run_agent_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state와 에이전트 문서를 보고, 실행할 에이전트 목록(selected_agents)을 LLM으로 선택.
    API 키 없으면 기본값: risk_factor, process_contributor는 조건 만족 시에만 나중에 실행되므로 [] 가능.
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"selected_agents": []}

    patient_case = state.get("patient_case") or {}
    similar_cases = state.get("similar_cases") or []
    diagnosis_analysis = state.get("diagnosis_analysis") or {}
    treatment_analysis = state.get("treatment_analysis") or {}
    evidence = state.get("evidence") or {}

    # 유사도 통계
    sims = [c.get("similarity", 0) for c in similar_cases if isinstance(c.get("similarity"), (int, float))]
    avg_sim = sum(sims) / len(sims) if sims else 0
    var_sim = (sum((s - avg_sim) ** 2 for s in sims) / len(sims)) ** 0.5 if len(sims) > 1 else 0
    expired_count = sum(1 for c in similar_cases if str(c.get("status", "")).lower() == "dead" or c.get("hospital_expire_flag") == 1)
    cohort_mortality = expired_count / len(similar_cases) if similar_cases else 0

    secondary = patient_case.get("secondary_diagnoses") or []
    key_conditions = patient_case.get("key_conditions") or []
    comorbidity_count = len(secondary) + len(key_conditions)
    outcome = str(patient_case.get("outcome") or patient_case.get("status") or "").lower()
    text = (patient_case.get("clinical_text") or patient_case.get("text") or "").lower()
    delay_keywords = ["delay", "missed", "late", "지연", "누락", "늦", "미시행"]
    has_delay = any(kw in text for kw in delay_keywords)

    context = f"""
현재 케이스 요약:
- 진단: {patient_case.get('diagnosis', 'Unknown')}
- 동반질환/핵심 조건 수: {comorbidity_count} (secondary_diagnoses + key_conditions)
- Outcome: {outcome}
- 유사 케이스 수: {len(similar_cases)}
- 유사도 평균: {avg_sim:.3f}, 분산(표준편차): {var_sim:.3f}
- Cohort 사망 비율: {cohort_mortality:.2f} (expired 수 / 전체)
- 치료 지연·누락 키워드 포함: {has_delay}
"""

    prompt = f"""You are an agent router for a clinical critique pipeline. Read the agent document and the current context. Decide which agents should run for this case. Output JSON only.

{AGENT_DOCUMENT}

{context}

Rules:
- selected_agents: list of agent names to run. Use only: "risk_factor", "process_contributor", "alternative_explanation". Can be empty [].
- Choose risk_factor only if comorbidity_count >= 2 and cohort mortality is high (e.g. > 0.2).
- Choose process_contributor only if delay/missed/late keywords present OR outcome is opposite to cohort (e.g. only this case died while most cohort survived).
- Choose alternative_explanation only if similarity variance is high OR cohort is inconsistent OR diagnosis is unclear / evidence is weak (so critique may be ambiguous later).

Output JSON:
{{"selected_agents": ["risk_factor", ...], "reason": "one short sentence"}}
"""

    try:
        llm = get_llm()
        response = llm.gpt4o(prompt=prompt, temperature=0.1, max_tokens=400, json_mode=True, timeout=30)
        response = response.strip().replace("```json", "").replace("```", "").strip()
        obj = json.loads(response)
        selected = obj.get("selected_agents")
        if not isinstance(selected, list):
            selected = []
        selected = [s for s in selected if s in ("risk_factor", "process_contributor", "alternative_explanation")]
        print(f"  [Agent Router] selected_agents={selected}, reason={obj.get('reason', '')[:80]}")
        return {"selected_agents": selected}
    except Exception as e:
        print(f"  [Agent Router] LLM failed: {e}, using []")
        return {"selected_agents": []}
