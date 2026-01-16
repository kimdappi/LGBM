f"""
코호트 비교 모듈 - 유사 환자끼리 치료 패턴 비교

[흐름]
1. rag_retriever에서 검색된 유사 케이스를 받음 (text 및 메타데이터 모두)
2. 코호트 간 치료 패턴 비교하고 생존/사망 통계 분석
3. Hugging Face LLM으로 임상 패턴 심층 분석 (무료)
4. 비교 결과 반환 (similar_case_patterns 변수로 레포트 생성에 사용 예정)
"""

from typing import List, Dict
from collections import Counter
import requests
import json


class CohortComparator:
    """
    유사 환자 코호트 검색 및 치료 패턴 비교
    
    rag_retriever에서 검색된 유사 케이스를 받아서 
    코호트 간 치료 패턴 비교하고 생존/사망 통계 분석
    """
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """
        코호트 비교기 초기화
        """
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
    
    def _call_hf_api(self, prompt: str) -> str:
        """Hugging Face Inference API 호출 (무료, API 키 불필요)"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 6000,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 응답 형식에 따라 처리
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response')
            elif isinstance(result, dict):
                return result.get('generated_text', 'No response')
            else:
                return str(result)
                
        except Exception as e:
            print(f"  ⚠️  Hugging Face API 호출 실패: {e}")
    
    def analyze_cohort(self, cohort_data: Dict) -> Dict:
        """
        유사 케이스 코호트 분석
        
        Args:
            cohort_data: retriever에서 반환된 데이터
                {
                    'similar_cases': [...],
                    'stats': {...}
                }
        
        Returns:
            similar_case_patterns: {
                'cohort_size': int,
                'survival_stats': {...},
                'demographic_patterns': {...},
                'admission_patterns': {...},
                'clinical_patterns': {...},
                'outcome_comparison': {...}
            }
        """
        similar_cases = cohort_data['similar_cases']
        stats = cohort_data['stats']
        
        if not similar_cases:
            return self._empty_result()
        
        # 1. 생존 통계
        survival_stats = self._analyze_survival(similar_cases, stats)
        print("  ✓ 생존 통계 분석 완료")
        
        # 2. 인구통계학적 패턴
        demographic_patterns = self._analyze_demographics(similar_cases)
        print("  ✓ 인구통계학적 패턴 분석 완료")
        
        # 3. 입원 패턴
        admission_patterns = self._analyze_admission(similar_cases)
        print("  ✓ 입원 패턴 분석 완료")
        
        # 4. 임상 패턴 (HF LLM 분석)
        clinical_patterns = self._analyze_clinical_text_with_llm(similar_cases)
        print("  ✓ 임상 패턴 LLM 분석 완료")
        
        # 5. 결과 비교 (생존 vs 사망)
        outcome_comparison = self._compare_outcomes_with_llm(similar_cases)
        print("  ✓ 결과 비교 LLM 분석 완료")
        
        # 종합 결과
        similar_case_patterns = {
            'cohort_size': len(similar_cases),
            'survival_stats': survival_stats,
            'demographic_patterns': demographic_patterns,
            'admission_patterns': admission_patterns,
            'clinical_patterns': clinical_patterns,
            'outcome_comparison': outcome_comparison
        }
        
        print("✅ 코호트 분석 완료\n")
        return similar_case_patterns
    
    def _analyze_survival(self, cases: List[Dict], stats: Dict) -> Dict:
        """생존 통계 분석"""
        return {
            'total_cases': stats['total'],
            'survived': stats['alive'],
            'died': stats['dead'],
            'survival_rate': round(stats['survival_rate'] * 100, 1),  # 백분율
            'case_ids': {
                'survived': [c['id'] for c in cases if c['status'] == 'alive'],
                'died': [c['id'] for c in cases if c['status'] == 'dead']
            }
        }
    
    def _analyze_demographics(self, cases: List[Dict]) -> Dict:
        """인구통계학적 패턴 분석"""
        ages = [c['age'] for c in cases]
        sexes = [c['sex'] for c in cases]
        
        sex_count = Counter(sexes)
        
        return {
            'age_range': {
                'min': min(ages),
                'max': max(ages),
                'mean': round(sum(ages) / len(ages), 1)
            },
            'sex_distribution': dict(sex_count),
            'details': [
                {
                    'id': c['id'],
                    'age': c['age'],
                    'sex': c['sex'],
                    'status': c['status']
                }
                for c in cases
            ]
        }
    
    def _analyze_admission(self, cases: List[Dict]) -> Dict:
        """입원 패턴 분석"""
        admission_types = [c['admission_type'] for c in cases]
        admission_locations = [c['admission_location'] for c in cases]
        discharge_locations = [c['discharge_location'] for c in cases]
        
        return {
            'admission_type_distribution': dict(Counter(admission_types)),
            'admission_location_distribution': dict(Counter(admission_locations)),
            'discharge_location_distribution': dict(Counter(discharge_locations)),
            'most_common': {
                'admission_type': Counter(admission_types).most_common(1)[0] if admission_types else None,
                'admission_location': Counter(admission_locations).most_common(1)[0] if admission_locations else None,
                'discharge_location': Counter(discharge_locations).most_common(1)[0] if discharge_locations else None
            }
        }
    
    def _analyze_clinical_text_with_llm(self, cases: List[Dict]) -> Dict:
        """Hugging Face API를 사용한 임상 텍스트 패턴 분석"""
        
        # 각 케이스 요약 준비
        case_summaries = []
        for i, c in enumerate(cases, 1):
            case_summaries.append({
                'case_number': i,
                'id': c['id'],
                'status': c['status'],
                'similarity': round(c['similarity'], 3),
                'age': c['age'],
                'sex': c['sex'],
                'text': c['text']
            })
        
        # LLM 프롬프트 구성
        prompt = self._build_clinical_analysis_prompt(case_summaries)
        
        # Hugging Face API 호출
        llm_analysis = self._call_hf_api(prompt)
        
        return {
            'llm_analysis': llm_analysis,
            'case_summaries': [
                {
                    'id': c['id'],
                    'status': c['status'],
                    'similarity': c['similarity'],
                    'text_preview': c['text'][:200] + "..." if len(c['text']) > 200 else c['text']
                }
                for c in case_summaries
            ]
        }
    
    def _build_clinical_analysis_prompt(self, case_summaries: List[Dict]) -> str:
        """임상 패턴 분석용 프롬프트 생성"""
        
        prompt = "You are an expert medical analyst specializing in identifying clinical patterns across patient cohorts.\n\n"
        prompt += "Analyze the following similar patient cases and identify common clinical patterns:\n\n"
        
        for case in case_summaries:
            prompt += f"Case {case['case_number']} (ID: {case['id']}, Status: {case['status'].upper()}, Age: {case['age']}, Sex: {case['sex']}):\n"
            prompt += f"{case['text']}\n\n"
        
        prompt += """
Based on these cases, please provide:
1. Common presenting symptoms and clinical findings
2. Common diagnostic tests or procedures mentioned
3. Common treatment approaches
4. Key clinical patterns that differentiate survived vs deceased patients (if applicable)
5. Overall clinical summary of this cohort

Keep the analysis concise and focus on actionable clinical insights.
"""
        
        return prompt
    
    def _compare_outcomes_with_llm(self, cases: List[Dict]) -> Dict:
        """Hugging Face API를 사용한 생존 vs 사망 그룹 비교"""
        survived_cases = [c for c in cases if c['status'] == 'alive']
        died_cases = [c for c in cases if c['status'] == 'dead']
        
        def group_stats(group):
            if not group:
                return None
            return {
                'count': len(group),
                'avg_age': round(sum(c['age'] for c in group) / len(group), 1),
                'sex_distribution': dict(Counter(c['sex'] for c in group)),
                'admission_types': dict(Counter(c['admission_type'] for c in group))
            }
        
        survived_stats = group_stats(survived_cases)
        died_stats = group_stats(died_cases)
        
        # LLM 비교 분석
        comparison_prompt = self._build_outcome_comparison_prompt(
            survived_cases, died_cases, survived_stats, died_stats
        )
        
        llm_comparison = self._call_hf_api(comparison_prompt)
        
        return {
            'survived_group': survived_stats,
            'died_group': died_stats,
            'comparison_analysis': llm_comparison
        }
    
    def _build_outcome_comparison_prompt(
        self,
        survived: List[Dict],
        died: List[Dict],
        survived_stats: Dict,
        died_stats: Dict
    ) -> str:
        """결과 비교용 프롬프트 생성"""
        
        prompt = "You are a medical data analyst specializing in outcome comparison.\n\n"
        prompt += "Compare the following patient groups and identify key differences:\n\n"
        
        # 생존 그룹
        if survived:
            prompt += f"SURVIVED GROUP ({len(survived)} patients):\n"
            prompt += f"- Average age: {survived_stats['avg_age']}\n"
            prompt += f"- Sex distribution: {survived_stats['sex_distribution']}\n"
            prompt += "Clinical summaries:\n"
            for i, case in enumerate(survived, 1):
                text_short = case['text'][:150] + "..." if len(case['text']) > 150 else case['text']
                prompt += f"  {i}. {text_short}\n"
            prompt += "\n"
        
        # 사망 그룹
        if died:
            prompt += f"DECEASED GROUP ({len(died)} patients):\n"
            prompt += f"- Average age: {died_stats['avg_age']}\n"
            prompt += f"- Sex distribution: {died_stats['sex_distribution']}\n"
            prompt += "Clinical summaries:\n"
            for i, case in enumerate(died, 1):
                text_short = case['text'][:150] + "..." if len(case['text']) > 150 else case['text']
                prompt += f"  {i}. {text_short}\n"
            prompt += "\n"
        
        prompt += """
Please analyze:
1. Key demographic differences
2. Clinical presentation differences
3. Treatment or care differences
4. Potential factors associated with different outcomes
5. Overall comparison summary

Keep the analysis concise and clinically relevant.
"""
        
        return prompt
    
    def _generate_comparison_summary(
        self, 
        survived: List[Dict], 
        died: List[Dict]
    ) -> str:
        """생존/사망 그룹 비교 요약 생성"""
        if not survived and not died:
            return "No cases available for comparison."
        
        if not died:
            return f"All {len(survived)} similar cases survived."
        
        if not survived:
            return f"All {len(died)} similar cases died."
        
        survived_avg_age = sum(c['age'] for c in survived) / len(survived)
        died_avg_age = sum(c['age'] for c in died) / len(died)
        
        summary = (
            f"Among similar cases: {len(survived)} survived (avg age {survived_avg_age:.1f}) "
            f"and {len(died)} died (avg age {died_avg_age:.1f}). "
        )
        
        if abs(survived_avg_age - died_avg_age) > 5:
            if died_avg_age > survived_avg_age:
                summary += "Deceased patients were notably older. "
            else:
                summary += "Deceased patients were notably younger. "
        
        return summary
    
