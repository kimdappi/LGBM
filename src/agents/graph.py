"""LangGraph 기반 Medical Critique Multi-Agent System"""

import os
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from .state import AgentState
from .critic_adapter import (
    clean_state_to_agent_state,
    agent_state_to_clean_updates,
    dict_to_critic_agent_state,
    normalize_solutions,
)
from .nodes import (
    run_chart_structurer,
    run_diagnosis_agent,
    run_treatment_agent,
    run_evidence_agent,
    run_evidence_agent_2nd_pass,
    check_intervention_coverage,
)
from src.critic_agent.critic_graph import get_critic_graph
from src.critic_agent.verifier import Verifier


class MedicalCritiqueGraph:
    """
    Multi-Agent Critique 시스템 (2-Pass CRAG + LGBM-style Critic Agent)

        Orchestrator (Episodic Memory)
              │
        ┌─ 에피소딕 메모리 검색 (과거 유사 케이스 교훈)
        │
        Chart Structurer (IE)
              │
        Evidence 1st (CRAG: 유사케이스 + 일반검색)
              │
        ┌─────┴─────┐
        ↓           ↓
    Diagnosis  Treatment      ← episodic_lessons 주입
     (GPT-4o)  (GPT-4o)
        └─────┬─────┘
              ↓
        Evidence 2nd (비판 기반 타겟 검색)
              ↓
    Intervention Checker
              ↓
    ─────────────────────────────────────────────────────────────────
        Preprocessing (timeline, evidence, record_gaps)
              │
        Router (Heuristic | LLM) → 선택: lens_* / behavior_* 도구
              │
        Tool 실행 (예산 내) → lens_results, behavior_results 적재
              │
        CritiqueBuilder (LLM) → critique_points (span_id, severity, cohort_comparison)
              │
        Feedback (품질 판단)
              │
        ┌─────┴─────┐
        ↓           ↓
       OK      불충분 → requested_tools 실행 → CritiqueBuilder 재호출 (feedback_rounds)
        │           │
        └─────┬─────┘
              ↓
        (선택) Verifier → solutions (유사 케이스 근거)
              ↓
        → graph state에 critique / solutions 반영
    ─────────────────────────────────────────────────────────────────
              ↓
             END
              │
        └─ 에피소딕 메모리에 저장 (critique, solutions, 교훈)
    """
    
    def __init__(self, rag_retriever=None, episodic_store=None):
        self.rag_retriever = rag_retriever
        self.episodic_store = episodic_store
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 구성 (2-Pass CRAG):
        
        chart_structurer → evidence_1st → diagnosis/treatment (병렬) 
        → evidence_2nd (비판 기반) → intervention_checker → critic → END
        """
        
        # 그래프 생성
        graph = StateGraph(AgentState)
        
        # 노드 추가
        graph.add_node("chart_structurer", self._chart_structurer_node)
        graph.add_node("evidence", self._evidence_node)
        graph.add_node("diagnosis", self._diagnosis_node)
        graph.add_node("treatment", self._treatment_node)
        graph.add_node("evidence_2nd", self._evidence_2nd_node)
        graph.add_node("intervention_checker", self._intervention_checker_node)
        graph.add_node("critic", self._critic_node)
        
        # 엣지 정의
        graph.set_entry_point("chart_structurer")
        
        # chart_structurer → evidence (1차)
        graph.add_edge("chart_structurer", "evidence")
        
        # evidence → diagnosis/treatment 병렬
        graph.add_edge("evidence", "diagnosis")
        graph.add_edge("evidence", "treatment")
        
        # diagnosis/treatment → evidence_2nd (비판 기반 2차 검색)
        graph.add_edge("diagnosis", "evidence_2nd")
        graph.add_edge("treatment", "evidence_2nd")
        
        # evidence_2nd → intervention_checker
        graph.add_edge("evidence_2nd", "intervention_checker")
        
        # intervention_checker → critic
        graph.add_edge("intervention_checker", "critic")
        
        # critic → END
        graph.add_edge("critic", END)
        
        return graph.compile()
    
    # ──────────────────────────────────────────────
    # 노드 함수
    # ──────────────────────────────────────────────
    
    def _chart_structurer_node(self, state: AgentState) -> Dict:
        """Chart Structurer: 차트 구조화 (Information Extraction)"""
        return run_chart_structurer(state)
    
    def _evidence_node(self, state: AgentState) -> Dict:
        """Evidence Agent: 내부 RAG + PubMed CRAG"""
        return run_evidence_agent(state, rag_retriever=self.rag_retriever)
    
    def _diagnosis_node(self, state: AgentState) -> Dict:
        """Diagnosis Agent: 진단 분석 (구조화 데이터 기반)"""
        return run_diagnosis_agent(state)
    
    def _treatment_node(self, state: AgentState) -> Dict:
        """Treatment Agent: 치료 분석 (구조화 데이터 기반)"""
        return run_treatment_agent(state)
    
    def _evidence_2nd_node(self, state: AgentState) -> Dict:
        """Evidence Agent 2nd Pass: 비판 기반 타겟 검색"""
        print("\n[Evidence 2nd Pass] Starting critique-based search...")
        return run_evidence_agent_2nd_pass(state)
    
    def _intervention_checker_node(self, state: AgentState) -> Dict:
        """Intervention Checker: 이미 시행된 치료 확인"""
        return check_intervention_coverage(state)
    
    def _critic_node(self, state: AgentState) -> Dict:
        """Critic Agent: LangGraph 서브그래프 → graph state 반영"""
        critic_state = clean_state_to_agent_state(state)
        initial_critic_dict = {
            "patient": critic_state.patient,
            "cohort_data": critic_state.cohort_data,
            "similar_case_patterns": critic_state.similar_case_patterns,
            "preprocessing": critic_state.preprocessing,
            "lens_results": critic_state.lens_results,
            "behavior_results": critic_state.behavior_results,
            "router": critic_state.router,
            "trace": critic_state.trace,
            "critique": {},
            "patch_instructions": "",
            "feedback_round": 0,
            "last_feedback_ok": False,
            "last_requested_tools": [],
            "executed_tools": [],
            "executed_budget": 0,
        }
        result_state = get_critic_graph().invoke(initial_critic_dict)
        critique_result = result_state.get("critique") or {}
        critic_state = dict_to_critic_agent_state(result_state)
        updates = agent_state_to_clean_updates(critic_state, critique_result)

        # Verifier: 유사 케이스가 있으면 solutions 보강
        similar_cases = state.get("similar_cases") or []
        if similar_cases and os.environ.get("OPENAI_API_KEY"):
            try:
                verifier = Verifier()
                critique_for_verifier = {
                    "patient_id": critic_state.patient.get("id"),
                    "critique_points": [
                        c.get("point") or c.get("issue", "") for c in (updates.get("critique") or [])
                    ],
                    "risk_factors": critique_result.get("risk_factors", []),
                    "recommendations": critique_result.get("recommendations", []),
                }
                verifier_result = verifier.verify(critique_for_verifier, similar_cases)
                if verifier_result.get("solutions"):
                    updates["solutions"] = normalize_solutions(verifier_result["solutions"])
            except Exception:
                pass  # 유지: recommendations 기반 solutions

        # 리포트 호환 — point→issue, severity high→critical, category 기본값
        critique_list = updates.get("critique", [])
        normalized = []
        for c in critique_list:
            if isinstance(c, dict):
                c = dict(c)
                if "point" in c and "issue" not in c:
                    c["issue"] = c["point"]
                if c.get("severity") == "high":
                    c["severity"] = "critical"
                if "category" not in c:
                    c["category"] = "process"
            normalized.append(c)
        updates["critique"] = normalized

        # confidence
        last_feedback = [t for t in critic_state.trace if t.get("tool") == "feedback"]
        confidence = 0.8 if (last_feedback and last_feedback[-1].get("detail", {}).get("ok")) else 0.5
        updates["iteration"] = 1
        updates["confidence"] = confidence
        return updates
    
    # ──────────────────────────────────────────────
    # 에피소딕 메모리
    # ──────────────────────────────────────────────
    
    def _search_episodic_memory(self, patient_case: Dict) -> str:
        """과거 유사 케이스의 분석 경험을 검색하여 프롬프트 문자열 반환"""
        if self.episodic_store is None:
            return ""
        
        try:
            clinical_text = patient_case.get("clinical_text", "") or patient_case.get("text", "")
            episodes = self.episodic_store.search_similar_episodes(
                clinical_text=clinical_text,
                top_k=2,
                min_similarity=0.3,
                diagnosis=patient_case.get("diagnosis", ""),
                secondary_diagnoses=patient_case.get("secondary_diagnoses", []),
            )
            if episodes:
                return self.episodic_store.format_for_prompt(episodes, max_episodes=2)
        except Exception as e:
            print(f"  [EpisodicMemory] 검색 실패: {e}")
        
        return ""
    
    def _save_episodic_memory(self, patient_case: Dict, result: Dict):
        """분석 결과를 에피소딕 메모리에 저장"""
        if self.episodic_store is None:
            return
        
        try:
            self.episodic_store.add_episode(
                patient_case=patient_case,
                critique_points=result.get("critique", []),
                solutions=result.get("solutions", []),
                confidence=result.get("confidence", 0.5),
                diagnosis_analysis=result.get("diagnosis_analysis"),
                treatment_analysis=result.get("treatment_analysis"),
            )
        except Exception as e:
            print(f"  [EpisodicMemory] 저장 실패: {e}")
    
    # ──────────────────────────────────────────────
    # 실행 함수
    # ──────────────────────────────────────────────
    
    def run(self, patient_case: Dict, similar_cases: list = None) -> Dict:
        """
        Critique 생성 실행
        
        Args:
            patient_case: 환자 케이스 정보
            similar_cases: 유사 케이스 목록
        
        Returns:
            최종 critique 및 solutions
        """
        # Phase 0: 에피소딕 메모리 검색 (과거 유사 케이스 교훈)
        episodic_lessons = self._search_episodic_memory(patient_case)
        
        if episodic_lessons:
            print(f"\n[Episodic Memory] 과거 유사 경험 발견 → 각 노드 프롬프트에 주입")
        else:
            print(f"\n[Episodic Memory] 과거 유사 경험 없음 (첫 실행 또는 유사도 부족)")
        
        initial_state = {
            "patient_case": patient_case,
            "similar_cases": similar_cases or [],
            "episodic_lessons": episodic_lessons,
            "structured_chart": None,
            "diagnosis_analysis": None,
            "treatment_analysis": None,
            "evidence": None,
            "intervention_coverage": None,
            "preprocessing": {},
            "lens_results": {},
            "behavior_results": {},
            "router": {},
            "trace": [],
            "similar_case_patterns": {},
            "critique": None,
            "solutions": None,
            "iteration": 0,
            "confidence": None,
        }
        
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        
        result = {
            "patient_id": patient_case.get("patient_id") or patient_case.get("id"),
            "critique": final_state.get("critique", []),
            "solutions": final_state.get("solutions", []),
            "diagnosis_analysis": final_state.get("diagnosis_analysis"),
            "treatment_analysis": final_state.get("treatment_analysis"),
            "evidence": final_state.get("evidence"),
            "confidence": final_state.get("confidence", 0.5),
            "episodic_lessons_used": bool(episodic_lessons),
        }
        
        # Phase Final: 에피소딕 메모리에 이번 분석 경험 저장
        self._save_episodic_memory(patient_case, result)
        
        return result
