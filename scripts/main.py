"""
Input(patient_json) 스키마:
{
  "id": str,
  "status": str,
  "sex": str,
  "age": int,
  "admission_type": str,
  "admission_location": str,
  "discharge_location": str,
  "arrival_transport": str,
  "disposition": str,
  "text": str
}

Flow:
1) retriever: 코사인 유사도 기반 유사 케이스 3개
2) cohort_comparator: 유사 케이스 패턴 분석
3) critique_reasoner: 환자 vs 패턴 비교 비판 포인트 생성
4) verifier: 비판에 대한 해결점 생성 (근거: 유사 케이스 top3)
5) report json 저장


    {
    "patient_id": 환자 id,
    "similar_cases": 유사 케이스 3개 코사인 유사도 기반 text 및 해당 메타데이터 내용,
    "similar_case_patterns": 유사 케이스 끼리 분석 후 패턴 판단 (cohort_comparator 결과),
    "critique": 환자 텍스트를 cohort_comparator 결과와 비교하여 비판 포인트 생성 (critique_reasoner 결과),
    "solution": resoner 가 비판한 것에 대한 해결점 생성 (유사 케이스 topk =3 근거를 가지고) (verifier 결과),
    }


"""
# scripts/main.py  (최소 + 어댑터 포함)

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# Force CPU mode for macOS compatibility - disable CUDA and MPS
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: False

# scripts/에서 실행해도 src import 되도록
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.rag_retriever import RAGRetriever
from src.critique_engine.cohort_comparator import CohortComparator
from src.critique_engine.critique_reasoner import CritiqueReasoner
from src.critique_engine.verifier import Verifier
from src.report_generation.report_generator import ReportGenerator

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_json", default="./data/patient.json")
    parser.add_argument("--out_dir", default="./outputs/reports")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    patient: Dict[str, Any] = load_json(args.patient_json)

    required = [
        "id", "status", "sex", "age", "admission_type", "admission_location",
        "discharge_location", "arrival_transport", "disposition", "text"
    ]
    missing = [k for k in required if k not in patient]
    if missing:
        raise ValueError(f"patient_json에 필수 키가 없습니다: {missing}")

    # 1) Retriever: topk=3 cohort_data 반환
    retriever = RAGRetriever()
    cohort_data: Dict[str, Any] = retriever.retrieve_with_patient(patient, top_k=3)
    similar_cases: List[Dict[str, Any]] = cohort_data.get("similar_cases", [])

    # 2) Comparator: 입력은 cohort_data 하나
    # OpenAI API 사용 (gpt-4o-mini 기본값)
    comparator = CohortComparator(model="gpt-4o-mini")
    similar_case_patterns: Dict[str, Any] = comparator.analyze_cohort(cohort_data)

    # Reasoner / ReportGenerator가 기대하는 survival_stats를 main에서 보강
    stats = cohort_data.get("stats", {})
    # retriever stats: total/alive/dead/survival_rate
    similar_case_patterns["survival_stats"] = {
        "total_cases": stats.get("total", 0),
        "survived": stats.get("alive", 0),
        "died": stats.get("dead", 0),
        # ReportGenerator는 "% 문자열"로 다시 감싸지만,
        # Reasoner는 숫자처럼 쓰기도 해서 float 유지
        "survival_rate": round(stats.get("survival_rate", 0.0) * 100, 1),
    }

    # 3) Critique
    reasoner = CritiqueReasoner()
    critique: Dict[str, Any] = reasoner.critique(
        patient_data=patient,
        similar_case_patterns=similar_case_patterns,
        stream=args.stream
    )

    # 4) Verifier: similar_cases 스키마 어댑터 (anchor_age/gender/hospital_expire_flag 맞추기)
    def adapt_case_for_verifier(c: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(c)
        # Verifier가 보는 키로 맞춤
        out["anchor_age"] = out.get("anchor_age", out.get("age"))
        out["gender"] = out.get("gender", out.get("sex"))
        # hospital_expire_flag: 1이면 DIED, 0이면 SURVIVED
        if "hospital_expire_flag" not in out:
            status = str(out.get("status", "")).lower()
            out["hospital_expire_flag"] = 1 if status == "dead" else 0
        return out

    verifier_cases = [adapt_case_for_verifier(c) for c in similar_cases[:3]]

    verifier = Verifier()
    solution: Dict[str, Any] = verifier.verify(
        critique=critique,
        similar_cases_topk=verifier_cases
    )

    # 5) ReportGenerator: 클래스 기반 generate → save
    report_gen = ReportGenerator(output_dir=args.out_dir)
    report = report_gen.generate(
        patient_data=patient,
        cohort_data=cohort_data,
        similar_case_patterns=similar_case_patterns,
        critique_result=critique,
        solution_result=solution
    )

    # 파일명 고정(원하면) or 자동
    saved_path = report_gen.save(report)  # 자동 파일명

    print(f"[OK] report saved: {saved_path}")


if __name__ == "__main__":
    main()
