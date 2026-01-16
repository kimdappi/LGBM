# CARE-CRITIC

의료 케이스 비판적 분석 파이프라인 - 유사 케이스 기반 AI 의료 검토 시스템

## 개요

CARE-CRITIC은 환자 데이터를 입력받아 유사 케이스와 비교 분석하고, 비판적 검토 포인트 및 해결책을 제시하는 AI 시스템입니다.

## 파이프라인 흐름

```
환자 데이터 입력
       ↓
[1] RAG Retriever: 코사인 유사도 기반 유사 케이스 3개 검색
       ↓
[2] Cohort Comparator: 유사 케이스 패턴 분석
       ↓
[3] Critique Reasoner: 환자 vs 유사 케이스 비교 → 비판 포인트 생성
       ↓
[4] Verifier: 비판에 대한 해결책 생성 (유사 케이스 근거 기반)
       ↓
[5] Report Generator: JSON 리포트 저장
```

## 입력 데이터 형식

```python
patient_data = {
    'id': '환자 ID',
    'status': 'alive/dead',
    'sex': 'M/F',
    'age': 65,
    'admission_type': '입원 유형',
    'admission_location': '입원 위치',
    'discharge_location': '퇴원 위치',
    'arrival_transport': '이송 수단',
    'disposition': '처분',
    'text': '임상 텍스트'
}
```

## 출력 리포트 형식

```json
{
    "patient_id": "환자 ID",
    "similar_cases": "유사 케이스 3개 (코사인 유사도 기반, text + 메타데이터)",
    "similar_case_patterns": "유사 케이스 패턴 분석 결과 (cohort_comparator)",
    "critique": "비판 포인트 (critique_reasoner)",
    "solution": "해결책 (verifier, 유사 케이스 근거 기반)"
}
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행

### 1. 벡터 DB 빌드 (최초 1회)

```bash
python scripts/build_vector_db.py
```

### 2. 분석 파이프라인 실행

```bash
python scripts/run_critique.py
```

## 프로젝트 구조

```
CARE-CRITIC/
├── data/
│   ├── flag_0_textclean.csv      # 생존 환자 데이터
│   ├── flag_1_textclean.csv      # 사망 환자 데이터
│   └── vector_db/                # FAISS 벡터 DB
│       ├── faiss_index.idx
│       └── metadata.pkl
├── src/
│   ├── retrieval/
│   │   └── rag_retriever.py      # 유사 케이스 검색
│   ├── critique_engine/
│   │   ├── cohort_comparator.py  # 코호트 패턴 분석
│   │   ├── critique_reasoner.py  # 비판 포인트 생성
│   │   └── verifier.py           # 해결책 생성
│   └── report_generation/
│       └── report_generator.py   # 리포트 생성
├── scripts/
│   ├── build_vector_db.py        # 벡터 DB 빌드
│   └── run_critique.py           # 메인 실행 스크립트
└── outputs/                      # 생성된 리포트
```

## 핵심 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| **RAG Retriever** | BioBERT 임베딩 + FAISS로 코사인 유사도 기반 유사 케이스 검색 |
| **Cohort Comparator** | 유사 케이스들의 치료 패턴 분석 (Mistral-7B 사용) |
| **Critique Reasoner** | 환자와 유사 케이스 비교하여 비판적 검토 포인트 생성 |
| **Verifier** | 비판 포인트에 대한 해결책 제시 (유사 케이스 근거 기반) |
| **Report Generator** | 최종 분석 결과를 JSON 형식으로 저장 |
