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
from pathlib import Path
from datetime import datetime
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()


class CohortComparator:
    """
    유사 환자 코호트 검색 및 치료 패턴 비교
    
    rag_retriever에서 검색된 유사 케이스를 받아서 
    코호트 간 치료 패턴 비교하고 생존/사망 통계 분석
    OpenAI API 사용 (임시)
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        코호트 비교기 초기화
        
        Args:
            model: OpenAI 모델명 (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            api_key: OpenAI API 키 (없으면 환경변수 OPENAI_API_KEY 사용)
        """
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    
    def _call_llm_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        
        if not self.api_key:
            print("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"OpenAI API 호출 실패: {e}")
            return None
    
    def analyze_cohort(self, cohort_data: Dict) -> Dict:
        """
        유사 케이스 코호트 분석
        
        Args:
            cohort_data: retriever에서 반환된 데이터
                {
                    'similar_cases': [...], #케이스 metadata + text
                    'stats': {...} #생존통계
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
        
        # 사망 생존 통계, 인구통계학적 패턴, 입원 패턴 등 메타데이터 기준으로 텍스트 패턴 분석
        statistics_based_analysis = self._analyze_with_llm(similar_cases)
        print(" 통계 기반 분석 완료")
        
        # 임상 패턴 (HF LLM 분석)
        clinical_patterns = self._analyze_clinical_text_with_llm(similar_cases)
        print("임상 패턴 LLM 분석 완료")
        
        # 결과 패턴 비교
        outcome_comparison = self._compare_outcomes_with_llm(similar_cases)
        print("결과 비교 LLM 분석 완료")
        
        # 종합 결과
        similar_case_patterns = {
            'cohort_size': len(similar_cases),
            'statistic_based_analysis': statistics_based_analysis,
            'clinical_patterns': clinical_patterns,
            'outcome_comparison': outcome_comparison
        }
        
        # JSON 파일로 저장 (최신 파일만 유지)
        self._save_to_json(similar_case_patterns)
        
        print("코호트 분석 완료\n")
        
        return similar_case_patterns
    
    def _save_to_json(self, data: Dict) -> str:
        """
        similar_case_patterns를 JSON 파일로 저장
        outputs/similar_case_patterns/ 폴더에 최신 파일만 유지
        """
        # 출력 디렉토리 생성
        output_dir = Path("outputs/similar_case_patterns")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 파일 모두 삭제 (최신 파일만 유지)
        for old_file in output_dir.glob("*.json"):
            old_file.unlink()
        
        # 파일명 생성 (타임스탬프 같이 구분용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"similar_case_patterns_{timestamp}.json"
        filepath = output_dir / filename
        
        # JSON 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장: {filepath}")
        return str(filepath)
    
    def _analyze_with_llm(self, cases: List[Dict]) -> Dict:
        """
        메타데이터 기반 임상 텍스트 패턴 분석
        
        각 케이스의 메타데이터(생존여부, 성별, 나이, admission type, 
        admission/discharge location, arrival transport, disposition)를 
        기준으로 text를 분석하여 패턴 추출
        
        Args:
            cases: 유사 케이스 리스트 (메타데이터 + text 포함)
        
        Returns:
            메타데이터 기반 패턴 분석 결과
        """
        # 프롬프트 구성
        prompt = self._build_metadata_based_prompt(cases)
        
        # LLM 호출
        llm_response = self._call_llm_api(prompt)
        
        
        return {
            'pattern_analysis': llm_response,
        }
    
    def _build_metadata_based_prompt(self, cases: List[Dict]) -> str:
        """메타데이터 기반 패턴 분석 프롬프트 생성"""
        
        prompt = """You are an expert medical analyst. Analyze the following patient cases based on their metadata characteristics and clinical text.

=== PATIENT CASES ===
"""
        # 각 케이스별 메타데이터 + text 추가
        for i, case in enumerate(cases, 1):
            prompt += f"""
--- Case {i} ---
[METADATA]
• Patient ID: {case.get('id', 'N/A')}
• Status: {case.get('status', 'N/A').upper()}
• Sex: {case.get('sex', 'N/A')}
• Age: {case.get('age', 'N/A')}
• Admission Type: {case.get('admission_type', 'N/A')}
• Admission Location: {case.get('admission_location', 'N/A')}
• Discharge Location: {case.get('discharge_location', 'N/A')}
• Arrival Transport: {case.get('arrival_transport', 'N/A')}
• Disposition: {case.get('disposition', 'N/A')}
• Similarity Score: {case.get('similarity', 0):.3f}

[CLINICAL TEXT]
{case.get('text', 'No text available')}
"""
        
        prompt += """

=== ANALYSIS INSTRUCTIONS ===
Based on the metadata and clinical text above, identify patterns in the following categories:

1. **SURVIVAL PATTERNS**
   - What metadata factors correlate with survival vs death?
   - Age, sex, admission type patterns in each outcome group

2. **ADMISSION PATTERNS**
   - Common admission types and locations
   - How admission characteristics relate to clinical presentations in text

3. **DISCHARGE PATTERNS**
   - Discharge location patterns by outcome
   - Disposition trends and their clinical context based on metadata and text

4. **TRANSPORT & ARRIVAL PATTERNS**
   - Arrival transport methods and their correlation with severity (from text)

5. **CLINICAL TEXT PATTERNS BY METADATA**
   - Common diagnoses/conditions mentioned in survived vs deceased
   - Treatment patterns by admission type
   - Clinical severity indicators by age group

6. **KEY INSIGHTS**
   - Most significant patterns found
   - Risk factors identified
   - Recommendations for similar future cases

Provide concise, actionable insights for each category.
"""
        
        return prompt
    
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
        llm_analysis = self._call_llm_api(prompt)
        
        return {
            'llm_analysis': llm_analysis
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
        
        llm_comparison = self._call_llm_api(comparison_prompt)
        
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
        
        # 생존 그룹 - 전체 텍스트 사용
        if survived:
            prompt += f"SURVIVED GROUP ({len(survived)} patients):\n"
            prompt += f"- Average age: {survived_stats['avg_age']}\n"
            prompt += f"- Sex distribution: {survived_stats['sex_distribution']}\n"
            prompt += "Clinical records:\n"
            for i, case in enumerate(survived, 1):
                prompt += f"\n  [Case {i}]\n{case['text']}\n"
            prompt += "\n"
        
        # 사망 그룹 - 전체 텍스트 사용
        if died:
            prompt += f"DECEASED GROUP ({len(died)} patients):\n"
            prompt += f"- Average age: {died_stats['avg_age']}\n"
            prompt += f"- Sex distribution: {died_stats['sex_distribution']}\n"
            prompt += "Clinical records:\n"
            for i, case in enumerate(died, 1):
                prompt += f"\n  [Case {i}]\n{case['text']}\n"
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