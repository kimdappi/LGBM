"""LangGraph 기반 Medical Critique Multi-Agent System"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    run_chart_structurer,
    run_diagnosis_agent,
    run_treatment_agent,
    run_evidence_agent,
    run_evidence_agent_2nd_pass,
    check_intervention_coverage,
    run_critic_agent,
)

#이게 전체 Orchestrator 처럼 사용 (전체를 감싸고 있는 구조)
class MedicalCritiqueGraph:
    """
    Multi-Agent Critique 시스템 (2-Pass CRAG)
    
    구조:
        Orchestrator (Reflexion Memory)
              │
        Chart Structurer (IE)
              │
        Evidence 1st (CRAG: 유사케이스 + 일반검색)
              │
        ┌─────┴─────┐
        ↓           ↓
    Diagnosis  Treatment
     (GPT-4o)  (GPT-4o)
        └─────┬─────┘
              ↓
        Evidence 2nd (비판 기반 타겟 검색)  ← NEW
              ↓
    Intervention Checker
              ↓
           Critic (GPT-4o)
    """
    
    def __init__(self, rag_retriever=None, max_iterations: int = 2):
        self.rag_retriever = rag_retriever
        self.max_iterations = max_iterations
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 구성 (2-Pass CRAG):
        
        chart_structurer → evidence_1st → diagnosis/treatment (병렬) 
        → evidence_2nd (비판 기반) → intervention_checker → critic → reflect (조건부)
        """
        
        # 그래프 생성
        graph = StateGraph(AgentState)
        
        # 노드 추가
        graph.add_node("chart_structurer", self._chart_structurer_node)
        graph.add_node("evidence", self._evidence_node)
        graph.add_node("diagnosis", self._diagnosis_node)
        graph.add_node("treatment", self._treatment_node)
        graph.add_node("evidence_2nd", self._evidence_2nd_node)  # NEW: 2차 검색
        graph.add_node("intervention_checker", self._intervention_checker_node)
        graph.add_node("critic", self._critic_node)
        graph.add_node("reflect", self._reflect_node)
        
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
        
        # critic → 조건부 분기 (반복 or 종료)
        graph.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "reflect": "reflect",
                "end": END
            }
        )
        
        # reflect → chart_structurer (재시작)
        graph.add_edge("reflect", "chart_structurer")
        
        return graph.compile()
    
#노드 함수
    
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
        """Critic Agent: 최종 종합 (GPT-4o) with Intervention Coverage"""
        result = run_critic_agent(state)
        return {
            "critique": result["critique"],
            "solutions": result["solutions"],
            "iteration": state.get("iteration", 0) + 1,
            "confidence": result.get("confidence", 0.5),
        }
    
    def _reflect_node(self, state: AgentState) -> Dict:
        """Reflexion: critical 이슈만 메모리에 저장 후 재시도"""
        critique = state.get("critique", [])
        issues = [c.get("issue", "") for c in critique if c.get("severity") == "critical"]
        reflection = f"Iteration {state.get('iteration', 0)}: Critical issues: {issues}"
        return {"memory": [reflection]}
    
  #조건분기
    
    def _should_continue(self, state: AgentState) -> str:
        """반복 여부 결정"""
        iteration = state.get("iteration", 0)
        confidence = state.get("confidence", 0.5)
        
        # 최대 반복 도달 또는 충분한 confidence
        if iteration >= self.max_iterations or confidence > 0.8:
            return "end"
        
        return "reflect"
    
 # 실행함수
    
    def run(self, patient_case: Dict, similar_cases: list = None) -> Dict:
        """
        Critique 생성 실행
        
        Args:
            patient_case: 환자 케이스 정보
            similar_cases: 유사 케이스 목록
        
        Returns:
            최종 critique 및 solutions
        """
        initial_state = {
            "patient_case": patient_case,
            "similar_cases": similar_cases or [],
            "structured_chart": None,  # Chart Structurer 결과
            "diagnosis_analysis": None,
            "treatment_analysis": None,
            "evidence": None,
            "intervention_coverage": None,  # Intervention Checker 결과
            "critique": None,
            "solutions": None,
            "iteration": 0,
            "memory": []
        }
        
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        
        return {
            "patient_id": patient_case.get("patient_id") or patient_case.get("id"),
            "similar_cases": similar_cases or [],  # top-k=3 유사 케이스
            "critique": final_state.get("critique", []),
            "solutions": final_state.get("solutions", []),
            "diagnosis_analysis": final_state.get("diagnosis_analysis"),
            "treatment_analysis": final_state.get("treatment_analysis"),
            "evidence": final_state.get("evidence"),
            "iterations": final_state.get("iteration", 1),
            "memory": final_state.get("memory", [])
        }

