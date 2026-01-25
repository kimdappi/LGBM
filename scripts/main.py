f"""
CARE-CRITIC 실행 스크립트 
인풋으로 환자 텍스트 받아서 분석 main.py
환자 텍스트는 

                    'id':,
                    'status': ,
                    'sex': ,
                    'age': ,
                    'admission_type':,
                    'admission_location': ,
                    'discharge_location':,
                    'arrival_transport':,
                    'disposition': ,
                    'text':
이렇게 받게 만들기
retriever 에서 코사인 유사도 기반 케이스 3개 검색
이후 cohort_comparator 에서 유사케이스 끼리 분석 후 패턴 판단
critique reasoner 에서 환자 텍스트를 cohort_comparator 결과와 비교하여 비판 포인트 생성
verifier 에서 resoner 가 비판한 것에 대한 해결점 생성 (유사 케이스 topk =3 근거를 가지고)
report generator 에서 리포트 생성 json 형식으로 저장
{
"patient_id": 환자 id,
"similar_cases": 유사 케이스 3개 코사인 유사도 기반 text 및 해당 메타데이터 내용,
"similar_case_patterns": 유사 케이스 끼리 분석 후 패턴 판단 (cohort_comparator 결과),
"critique": 환자 텍스트를 cohort_comparator 결과와 비교하여 비판 포인트 생성 (critique_reasoner 결과),
"solution": resoner 가 비판한 것에 대한 해결점 생성 (유사 케이스 topk =3 근거를 가지고) (verifier 결과),
}

comparator = CohortComparator(model="mistralai/Mistral-7B-Instruct-v0.3")
"""
import sys
import os
from src.retrieval.rag_retriever import *
from src.critique_engine.cohort_comparator import *
from src.critique_engine.critique_reasoner import *
from src.critique_engine.verifier import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


