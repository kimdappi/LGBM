"""Multi-Agent Critique 실행 스크립트 - LLM 기반 진단 추출 개선 버전"""

import sys
import json
from pathlib import Path
from datetime import datetime
import re

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import MedicalCritiqueGraph
from src.retrieval.rag_retriever import RAGRetriever
from src.memory import EpisodicMemoryStore


def load_patient_case(path: str) -> dict:
    """환자 케이스 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_report(result: dict, output_dir: str = "outputs/reports"):
    """결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AGENT-CRITIQUE-{timestamp}.json"
    
    with open(output_path / filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Report saved: {output_path / filename}")
    return output_path / filename


def extract_diagnosis_from_text(clinical_text: str) -> dict:
    """LLM 기반 진단 추출 - GPT-4o(Primary + Secondary + Key conditions)
    
    Returns:
        dict: {
            "diagnosis": str,  # Primary diagnosis
            "secondary_diagnoses": list,  # Secondary/comorbidities
            "key_conditions": list,  # Important conditions affecting treatment
            "confidence": str,
            "reasoning": str
        }
    """
    from openai import OpenAI
    import os
    
    # 1단계: 정규표현식으로 진단 섹션 우선 추출 (토큰 절약)
    diagnosis_section = None
    patterns = [
        r'(?:Discharge|Final|Principal|Primary)\s+Diagnosis[:\s]+(.*?)(?:\n\n|\n[A-Z][a-z]+:)',
        r'Diagnoses[:\s]+(.*?)(?:\n\n|\n[A-Z][a-z]+:)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clinical_text, re.IGNORECASE | re.DOTALL)
        if match:
            diagnosis_section = match.group(1).strip()
            break
    
    # 분석할 텍스트 결정 (섹션 있으면 섹션만, 없으면 앞부분)
    text_to_analyze = diagnosis_section if diagnosis_section else clinical_text[:2000]
    
    # 2단계: GPT-4o로 진단 추출 (Primary + Secondary)
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Extract diagnoses from the clinical text.

Clinical Text:
{text_to_analyze}

Instructions:
1. PRIMARY diagnosis: Main reason for admission/treatment (most acute/severe)
2. SECONDARY diagnoses: Active comorbidities affecting treatment (e.g., Hypertension, Diabetes, CKD)
3. KEY conditions: Important past history relevant to treatment (e.g., prior stroke, GI bleeding)

Exclude:
- Rule-out diagnoses (unless confirmed)
- Past resolved conditions

Response format (JSON):
{{
  "primary_diagnosis": "main reason for admission/treatment",
  "secondary_diagnoses": ["active comorbidities affecting treatment"],
  "key_conditions": ["relevant past history affecting decisions"],
  "confidence": "high/medium/low",
  "reasoning": "one sentence"
}}

Rules:
- Exclude rule-out diagnoses unless confirmed.
- Use empty arrays if none.

Text:
{text_to_analyze}
"""
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # JSON 파싱
        response_text = response.choices[0].message.content.strip()
        # JSON 코드 블록 제거
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        result = json.loads(response_text)
        
        return {
            "diagnosis": result.get("primary_diagnosis", "Unknown"),
            "secondary_diagnoses": result.get("secondary_diagnoses", []),
            "key_conditions": result.get("key_conditions", []),
            "confidence": result.get("confidence", "low"),
            "reasoning": result.get("reasoning", "N/A")
        }
        
    except Exception as e:
        print(f"  Warning: GPT-4o extraction failed ({e}), using fallback")
        
        # Fallback: 정규표현식으로 첫 번째 진단 추출
        if diagnosis_section:
            # 번호 목록에서 첫 줄 추출
            first_line = diagnosis_section.split('\n')[0].strip()
            # 숫자, 점, 하이픈 제거
            diagnosis = re.sub(r'^\d+[\.\)]\s*|-\s*', '', first_line).strip()
            return {
                "diagnosis": diagnosis if diagnosis else "Unknown",
                "secondary_diagnoses": [],
                "key_conditions": [],
                "confidence": "low",
                "reasoning": "Fallback: regex extraction"
            }
        
        return {
            "diagnosis": "Unknown",
            "secondary_diagnoses": [],
            "key_conditions": [],
            "confidence": "low",
            "reasoning": "Extraction failed"
        }


def run_agent_critique_pipeline( #main에 있던 코드를 함수로 묶
    patient_data: dict,
    db_path: str = "vector_db",
    top_k: int = 3,
    similarity_threshold: float = 0.7,
    max_iterations: int = 3,
    ) -> dict:
    
    # 1. RAG Retriever + Episodic Memory 로드
    print("\n[1/5] Loading RAG retriever + Episodic Memory...")
    try:
        rag = RAGRetriever(db_path=db_path)
    except Exception as e:
        print(f"  Warning: RAG not loaded ({e})")
        rag = None
    
    # Episodic Memory (RAG의 임베딩 모델 공유 → 중복 로딩 방지)
    episodic = None
    try:
        shared_embedder = rag.vector_db if rag and hasattr(rag, 'vector_db') else None
        episodic = EpisodicMemoryStore(shared_embedder=shared_embedder)
        episodic.load()
        print(f"  [EpisodicMemory] {episodic.episode_count}건의 과거 경험 로드됨")
    except Exception as e:
        print(f"  Warning: Episodic Memory not loaded ({e})")
        episodic = None
    

    # LLM 기반 진단 추출
    clinical_text = patient_data.get("text", "")
    print(f"  Clinical text length: {len(clinical_text)} chars")
    print(f"  Extracting diagnosis with LLM...")
    
    diagnosis_result = extract_diagnosis_from_text(clinical_text)
    
    patient_case = {
        "patient_id": patient_data.get("id"),
        "diagnosis": diagnosis_result["diagnosis"],
        "secondary_diagnoses": diagnosis_result.get("secondary_diagnoses", []),
        "key_conditions": diagnosis_result.get("key_conditions", []),
        "diagnosis_confidence": diagnosis_result["confidence"],
        "diagnosis_reasoning": diagnosis_result["reasoning"],
        "clinical_text": clinical_text,
        "outcome": patient_data.get("status"),
        "age": patient_data.get("age"),
        "sex": patient_data.get("sex")
    }
    
    print(f"  Patient ID: {patient_case['patient_id']}")
    print(f"  Primary Diagnosis: {diagnosis_result['diagnosis']}")
    print(f"  Secondary Diagnoses: {diagnosis_result.get('secondary_diagnoses', [])}")
    print(f"  Key Conditions: {diagnosis_result.get('key_conditions', [])}")
    print(f"  Confidence: {diagnosis_result['confidence']}")
    print(f"  Reasoning: {diagnosis_result['reasoning']}")
    print(f"  Outcome: {patient_case['outcome']}")
    


    # 3. Top-K 유사 케이스 검색 (근거용) + 품질 검증
    print("\n[3/5] Retrieving similar cases (top_k=3)...")
    similar_cases = []
    if rag:
        try:
            cohort_data = rag.retrieve_with_patient(patient_case, top_k=3)
            raw_cases = cohort_data.get("similar_cases", [])
            
            # 유사도 품질 검증
            valid_cases = []
            for case in raw_cases:
                similarity = case.get("similarity", 0)
                if similarity >= 0.7:
                    valid_cases.append(case)
                    print(f"  OKAY Case {case.get('id')}: similarity={similarity:.3f} [VALID]")
                else:
                    print(f"  BAD Case {case.get('id')}: similarity={similarity:.3f} [REJECTED - below 0.7]")
            
            similar_cases = valid_cases
            
            if len(valid_cases) == 0:
                print(f"  (Warning!!!) No valid similar cases (all below 0.7 threshold)")
                print(f"  → CRAG will use external PubMed only")
            else:
                print(f"  Found {len(valid_cases)} valid similar cases")
                
        except Exception as e:
            print(f"  Warning: Similar case retrieval failed ({e})")
    
    # 4. 그래프 생성 및 실행
    print("\n[4/5] Running agent graph...")
    graph = MedicalCritiqueGraph(rag_retriever=rag, episodic_store=episodic)
    
    result = graph.run(
        patient_case=patient_case,
        similar_cases=similar_cases  # top-k=3 유사 케이스 전달
    )
    
    # 5. 결과 출력
    print("\n[5/5] Results:")
    print("=" * 60)
    
    # 진단 정보
    print(f"\n[DIAGNOSIS EXTRACTION]:")
    print(f"  Primary Diagnosis: {patient_case['diagnosis']}")
    if patient_case.get('secondary_diagnoses'):
        print(f"  Secondary Diagnoses: {', '.join(patient_case['secondary_diagnoses'])}")
    if patient_case.get('key_conditions'):
        print(f"  Key Conditions: {', '.join(patient_case['key_conditions'])}")
    print(f"  Confidence: {patient_case.get('diagnosis_confidence', 'N/A')}")
    print(f"  Method: GPT-4o extraction")
    
    # 검색 품질
    evidence = result.get("evidence", {})
    mode = evidence.get("retrieval_mode", "unknown")
    quality = evidence.get("quality_evaluation", {})
    
    print(f"\n[EVIDENCE QUALITY]:")
    print(f"  Mode: {mode}")
    print(f"  Internal cases: {quality.get('count', 0)}")
    print(f"  Reason: {quality.get('reason', 'N/A')}")
    
    print("\n[CRITIQUE POINTS - BY SEVERITY]:")
    
    critiques = result.get("critique", [])
    critical = [c for c in critiques if c.get("severity") == "critical"]
    medium = [c for c in critiques if c.get("severity") == "medium"]
    low = [c for c in critiques if c.get("severity") == "low"]
    other = [c for c in critiques if c.get("severity") not in ["critical", "medium", "low"]]
    
    if critical:
        print("\n  [CRITICAL]:")
        for i, c in enumerate(critical, 1):
            category = c.get("category", "?")
            print(f"    {i}. [{category.upper()}] {c.get('issue', 'Unknown')}")
    
    if medium:
        print("\n  [MEDIUM]:")
        for i, c in enumerate(medium, 1):
            category = c.get("category", "?")
            print(f"    {i}. [{category.upper()}] {c.get('issue', 'Unknown')}")
    
    if low:
        print("\n  [LOW]:")
        for i, c in enumerate(low, 1):
            category = c.get("category", "?")
            print(f"    {i}. [{category.upper()}] {c.get('issue', 'Unknown')}")
    
    if other:
        print("\n  [OTHER]:")
        for i, c in enumerate(other, 1):
            category = c.get("category", "?")
            print(f"    {i}. [{category.upper()}] {c.get('issue', 'Unknown')}")
    
    print("\n[SOLUTIONS]:")
    for i, s in enumerate(result.get("solutions", []), 1):
        action = s.get("action", "Unknown")
        citation = s.get("citation", "N/A")
        priority = s.get("priority", "?")
        print(f"  {i}. [{priority}] {action}")
        print(f"     Citation: {citation}")
    
    # 에피소딕 메모리 활용 여부
    if result.get("episodic_lessons_used"):
        print(f"\n[EPISODIC MEMORY]: 과거 유사 경험 참조됨 ")
    else:
        print(f"\n[EPISODIC MEMORY]: 과거 유사 경험 없음 (이번 분석이 메모리에 저장됨)")
    
    return result

## execute.py 말고 cli로  python scripts/run_agent_critique.py 실행 가능
def main():
    print("=" * 60)
    print("Multi-Agent Medical Critique System (LLM-Enhanced)")
    print("=" * 60)

    # 기존 동작 유지: data/patient.json을 불러서 단독 실행
    patient_data = load_patient_case("data/patient.json")

    result = run_agent_critique_pipeline(
        patient_data=patient_data,
        db_path="vector_db",
        top_k=3,
        similarity_threshold=0.7,
        max_iterations=3,
    )
    # 저장
    save_report(result)

    print("\n" + "=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()