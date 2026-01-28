# CARE-CRITIC

의료 케이스 비판적 분석 파이프라인 - 유사 케이스 기반 AI 의료 검토 시스템

## 개요

CARE-CRITIC은 환자 데이터를 입력받아 **임상적으로 유사한 케이스**와 비교 분석하고, 비판적 검토 포인트 및 해결책을 제시하는 AI 시스템입니다.

## 파이프라인 흐름

```
환자 데이터 입력 (patient.json)
       ↓
┌─────────────────────────────────────────────────────────────┐
│  [1] RAG Retriever (3단계 검색)                              │
│  ────────────────────────────────                            │
│  Stage 1: MedCPT + FAISS → top-10 후보 (코사인 유사도)        │
│  Stage 2: LLM 진단 추출 → 유사 진단 필터링                    │
│  Stage 3: Cross-Encoder Reranking → top-3 최종               │
└─────────────────────────────────────────────────────────────┘
       ↓
[2] Cohort Comparator: 유사 케이스 패턴 분석 (GPT-4o-mini)
       ↓
[3] Critique Reasoner: 환자 vs 유사 케이스 비교 → 비판 포인트 생성
       ↓
[4] Verifier: 비판에 대한 해결책 생성 (유사 케이스 근거 기반)
       ↓
[5] Report Generator: JSON 리포트 저장
```

## 3단계 검색 시스템

기존 단순 코사인 유사도의 한계(문서 형식만 유사하면 높은 점수)를 해결하기 위한 **임상 유사성 기반 검색**:

| Stage | 방법 | 역할 |
|-------|------|------|
| **Stage 1** | MedCPT + FAISS | 빠른 후보 검색 (top-10) |
| **Stage 2** | LLM 진단 추출 | 주요 진단 일치 여부 필터링 |
| **Stage 3** | Instruction-tuned Reranker | 질환 메커니즘 기반 정밀 reranking |

### Instruction-tuned Cross-Encoder Reranker

**모델**: `BAAI/bge-reranker-v2-m3` (instruction 지원)

**Instruction**: 
> "Find cases that share the same primary disease mechanism and clinical presentation"

```
Bi-Encoder (Stage 1):   Query → MedCPT-Query → Vector ─┐
                                                        ├→ cosine (상호작용 없음)
                        Doc → MedCPT-Article → Vector ──┘

Cross-Encoder (Stage 3): 
    Instruction: "Find cases with same disease mechanism"
              ↓
    [CLS] Query [SEP] Doc [SEP] → Encoder → Score
    (Query-Doc 간 Cross-Attention + Instruction 반영)
```

**기존 reranker vs Instruction-tuned**:
| 방식 | 평가 기준 |
|------|----------|
| 기존 (`bge-reranker-base`) | 텍스트 유사도 |
| Instruction-tuned (`bge-reranker-v2-m3`) | **질환 메커니즘 유사성** |

## 임베딩 모델 옵션

### MedCPT (현재 사용)

| 항목 | 설명 |
|------|------|
| **구조** | 비대칭 듀얼 인코더 (Query + Article) |
| **학습** | Contrastive Learning (PubMed 33M query-article pairs) |
| **장점** | 검색 목적에 최적화, 쿼리-문서 매칭 성능 우수 |

```
DB 구축: 문서 → MedCPT-Article-Encoder → 768차원 벡터 → FAISS
검색:    쿼리 → MedCPT-Query-Encoder  → 768차원 벡터 → FAISS 검색
```

### BioLORD (대안)

| 항목 | 설명 |
|------|------|
| **구조** | 단일 인코더 |
| **학습** | Ontology-based Contrastive Learning (UMLS, SNOMED-CT) |
| **장점** | 임상 개념 유사성 학습, "COPD"≠"Cirrhosis" 구분 |

```
DB 구축 & 검색: 모두 FremyCompany/BioLORD-2023-M 사용
```

### 왜 BioBERT 대신?

| 모델 | 학습 목표 | 검색 적합성 |
|------|----------|-------------|
| BioBERT | 단어 예측 (MLM) | ⭐⭐ 검색 최적화 안됨 |
| MedCPT | 쿼리→문서 매칭 | ⭐⭐⭐⭐ 검색 특화 |
| BioLORD | 개념 간 유사성 | ⭐⭐⭐⭐⭐ 질환 유사성 |

**BioBERT 문제점**: "COPD 환자" 검색 시 "Cirrhosis 환자 (COPD 언급됨)" 반환 가능  
**MedCPT/BioLORD**: 의미/개념 수준 유사성으로 더 정확한 결과

## 입력 데이터 형식

`data/patient.json`:
```json
{
    "id": "34719194",
    "status": "dead",
    "sex": "F",
    "age": 68,
    "admission_type": "OBSERVATION ADMIT",
    "admission_location": "EMERGENCY ROOM",
    "discharge_location": "DIED",
    "arrival_transport": "WALK IN",
    "disposition": "ADMITTED",
    "text": "임상 텍스트..."
}
```

## 출력 리포트 형식

`outputs/reports/CARE-CRITIC-YYYYMMDD-HHMMSS.json`:
```json
{
    "report_metadata": { "report_id": "...", "generated_at": "...", "patient_id": "..." },
    "patient_info": { "id": "...", "age": 68, "sex": "F", ... },
    "similar_cases": [
        { "case_id": "...", "similarity": 0.85, "status": "dead", "age": 71, ... }
    ],
    "similar_case_patterns": { "cohort_size": 3, "clinical_patterns": "...", ... },
    "critique": { "analysis": "...", "critique_points": [...], "risk_factors": [...] },
    "solution": { "solutions": [...] }
}
```

## 설치

```bash
pip install -r requirements.txt
```

### 주요 패키지
- `transformers`, `torch` - 임베딩 모델 (MedCPT/BioLORD)
- `faiss-cpu` - 벡터 검색
- `FlagEmbedding` - Instruction-tuned Reranker (bge-reranker-v2-m3)
- `sentence-transformers` - Cross-Encoder fallback
- `openai` - LLM API (GPT-4o-mini)

## 환경 설정

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-your-api-key-here
```

## 실행

### 1. 벡터 DB 빌드 (최초 1회)

```bash
python scripts/build_vector_db.py
```

### 2. 분석 파이프라인 실행

```bash
python scripts/main.py --patient_json data/patient.json
```

### 임베딩 모델 변경 시

```bash
# 1. build_vector_db.py 와 rag_retriever.py 에서 EMBEDDING_MODEL 변경
# 2. 벡터 DB 재구축 필수
python scripts/build_vector_db.py
```

## 프로젝트 구조

```
CARE-CRITIC/
├── data/
│   ├── patient.json              # 분석할 환자 데이터
│   ├── flag_0_textclean.csv      # 생존 환자 데이터
│   ├── flag_1_textclean.csv      # 사망 환자 데이터
│   └── vector_db/                # FAISS 벡터 DB
│       ├── faiss_index.idx
│       └── metadata.pkl
├── src/
│   ├── retrieval/
│   │   └── rag_retriever.py      # 3단계 검색 시스템
│   │       ├── VectorDBManager   # FAISS + MedCPT/BioLORD
│   │       ├── DiagnosisExtractor # LLM 진단 추출
│   │       └── RAGRetriever      # 통합 검색기
│   ├── critique_engine/
│   │   ├── cohort_comparator.py  # 코호트 패턴 분석
│   │   ├── critique_reasoner.py  # 비판 포인트 생성
│   │   └── verifier.py           # 해결책 생성
│   └── report_generation/
│       └── report_generator.py   # 리포트 생성
├── scripts/
│   ├── build_vector_db.py        # 벡터 DB 빌드 (MedCPT-Article/BioLORD)
│   └── main.py                   # 메인 실행 스크립트
├── outputs/
│   ├── reports/                  # 생성된 리포트
│   └── similar_case_patterns/    # 코호트 분석 결과
├── .env                          # API 키 (git 제외)
└── requirements.txt
```

## 핵심 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| **RAG Retriever** | 3단계 검색: MedCPT+FAISS → LLM 진단 필터링 → Instruction-tuned Reranking |
| **DiagnosisExtractor** | GPT-4o-mini로 임상 텍스트에서 주요 진단 추출 |
| **Instruction-tuned Reranker** | BGE v2-m3로 "질환 메커니즘 유사성" 기반 정밀 reranking |
| **Cohort Comparator** | 유사 케이스들의 치료 패턴 분석 (GPT-4o-mini) |
| **Critique Reasoner** | 환자와 유사 케이스 비교하여 비판적 검토 포인트 생성 |
| **Verifier** | 비판 포인트에 대한 해결책 제시 (유사 케이스 근거 기반) |
| **Report Generator** | 최종 분석 결과를 JSON 형식으로 저장 |

## 사용 모델

| 용도 | 모델 | 설명 |
|------|------|------|
| 문서 임베딩 (DB 구축) | `ncbi/MedCPT-Article-Encoder` | 비대칭 인코딩, 검색 특화 |
| 쿼리 임베딩 (검색) | `ncbi/MedCPT-Query-Encoder` | 비대칭 인코딩, 검색 특화 |
| 대안 임베딩 | `FremyCompany/BioLORD-2023-M` | 단일 인코더, 개념 유사성 |
| Reranking | `BAAI/bge-reranker-v2-m3` | **Instruction-tuned** Cross-Encoder |
| LLM 분석 | `gpt-4o-mini` (OpenAI API) | 진단 추출, 비평 생성 |

### Reranker Instruction
```
"Find cases that share the same primary disease mechanism and clinical presentation"
```
→ 단순 텍스트 유사도가 아닌 **질환 메커니즘 기반** 유사성 평가
