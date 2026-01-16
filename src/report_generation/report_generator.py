"""
Report Generator - CARE-CRITIC 최종 리포트 생성
새 스키마에 맞춰 JSON 형식으로 리포트 생성
미완성
"""

import json
import os
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    """
    CARE-CRITIC 리포트 생성기
    
    최종 리포트 구조:
    {
        "patient_id": str,
        "similar_cases": List[Dict],
        "similar_case_patterns": Dict,
        "critique": Dict/str,
        "solution": Dict/str
    }
    """
    
    def __init__(self, output_dir: str = 'outputs/reports'):
        """
        리포트 생성기 초기화
        
        Args:
            output_dir: 리포트 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(
        self,
        patient_data: Dict,
        cohort_data: Dict,
        similar_case_patterns: Dict,
        critique_result: Dict,
        solution_result: Dict
    ) -> Dict:
        """
        최종 리포트 생성
        
        Args:
            patient_data: 환자 정보
                {
                    'id': str,
                    'status': str,
                    'sex': str,
                    'age': int,
                    'admission_type': str,
                    'admission_location': str,
                    'discharge_location': str,
                    'arrival_transport': str,
                    'disposition': str,
                    'text': str
                }
            cohort_data: retriever에서 반환된 유사 케이스 데이터
                {
                    'similar_cases': List[Dict],
                    'stats': Dict
                }
            similar_case_patterns: cohort_comparator 분석 결과
            critique_result: critique_reasoner 결과
            solution_result: verifier 결과
        
        Returns:
            최종 리포트 딕셔너리
        """
        
        # 환자 ID
        patient_id = patient_data.get('id', 'unknown')
        
        # 유사 케이스 정보 (text + 메타데이터)
        similar_cases = cohort_data.get('similar_cases', [])
        
        # 유사 케이스 패턴
        similar_case_patterns_summary = self._format_patterns(similar_case_patterns)
        
        # 비판 포인트
        critique_summary = self._format_critique(critique_result)
        
        # 해결책
        solution_summary = self._format_solution(solution_result)
        
        # 최종 리포트 구성
        report = {
            "report_metadata": {
                "report_id": f"CARE-CRITIC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "patient_id": patient_id
            },
            "patient_info": {
                "id": patient_id,
                "age": patient_data.get('age'),
                "sex": patient_data.get('sex'),
                "admission_type": patient_data.get('admission_type'),
                "admission_location": patient_data.get('admission_location'),
                "status": patient_data.get('status')
            },
            "similar_cases": [
                {
                    "case_id": case.get('id'),
                    "similarity": case.get('similarity'),
                    "status": case.get('status'),
                    "age": case.get('age'),
                    "sex": case.get('sex'),
                    "admission_type": case.get('admission_type'),
                    "text": case.get('text')
                }
                for case in similar_cases
            ],
            "similar_case_patterns": similar_case_patterns_summary,
            "critique": critique_summary,
            "solution": solution_summary
        }
        
        return report
    
    def _format_patterns(self, patterns: Dict) -> Dict:
        """유사 케이스 패턴 포맷팅"""
        if not patterns:
            return {}
        
        return {
            "cohort_size": patterns.get('cohort_size', 0),
            "survival_statistics": {
                "total": patterns.get('survival_stats', {}).get('total_cases', 0),
                "survived": patterns.get('survival_stats', {}).get('survived', 0),
                "died": patterns.get('survival_stats', {}).get('died', 0),
                "survival_rate": f"{patterns.get('survival_stats', {}).get('survival_rate', 0)}%"
            },
            "clinical_patterns": patterns.get('clinical_patterns', {}).get('llm_analysis', 'No analysis available'),
            "outcome_comparison": patterns.get('outcome_comparison', {}).get('comparison_analysis', 'No comparison available')
        }
    
    def _format_critique(self, critique: Dict) -> Dict:
        """비판 포인트 포맷팅"""
        if isinstance(critique, str):
            return {"analysis": critique}
        
        return critique if critique else {"analysis": "No critique available"}
    
    def _format_solution(self, solution: Dict) -> Dict:
        """해결책 포맷팅"""
        if isinstance(solution, str):
            return {"recommendations": solution}
        
        return solution if solution else {"recommendations": "No solution available"}
    
    def save(self, report: Dict, filename: str = None) -> str:
        """
        리포트를 JSON 파일로 저장
        
        Args:
            report: 리포트 딕셔너리
            filename: 파일명 (없으면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            report_id = report.get('report_metadata', {}).get('report_id', 'unknown')
            filename = f"{report_id}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n 리포트 저장 완료: {filepath}")
        return filepath
