"""
RAG Retriever - FAISS 기반 유사 케이스 검색 + LLM 진단 필터링 + Reranking
이미 생성된 FAISS DB와 metadata.pkl 사용

3단계 검색:
  Stage 1: MedCPT Query Encoder + FAISS → top-10 후보 (빠름)
  Stage 2: LLM 진단 추출 → 유사 진단 필터링 (정확한 recall)
  Stage 3: Cross-encoder Reranking → top-k 최종 (정확한 ranking)
  
임베딩 모델 옵션:
  - MedCPT (현재): 비대칭 인코더, PubMed 검색 특화
  - BioLORD: 단일 인코더, UMLS 기반 임상 개념 유사성
"""

import numpy as np
import pickle
import faiss
import torch
import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# .env 로드
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# Reranker lazy import
_reranker = None #초기화

_reranker_model_name = "BAAI/bge-reranker-v2-m3"
# _reranker_model_name = "BAAI/bge-reranker-base"

# 의료 특화 리랭커 옵션 (향후 고려):
# - MedCPT는 bi-encoder만 있고 cross-encoder는 없음
# - BioLinkBERT, PubMedBERT 기반 cross-encoder 직접 파인튜닝 필요
# → 현재는 BGE-Reranker + M&M 특화 instruction 사용

# Reranker instruction - M&M 컨퍼런스 목적에 맞춘 케이스 검색
# M&M의 목표: 비판점 도출 및 해결책 학습
# → 비슷한 임상 상황에서 진단/치료 의사결정에 교훈을 줄 수 있는 케이스
RERANKER_INSTRUCTION = "Find cases with similar clinical presentations that provide lessons about diagnosis, treatment decisions, or potential complications"

def get_reranker():
    """Reranker 모델 lazy loading (FlagReranker with instruction support)"""
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            print(f"[Reranker] Loading {_reranker_model_name} (instruction-tuned)...")
            _reranker = FlagReranker(_reranker_model_name, use_fp16=True)
            print(f"[Reranker] Loaded successfully")
            print(f"[Reranker] Instruction: '{RERANKER_INSTRUCTION}'")
        except ImportError:
            print("[Warning] FlagEmbedding not installed. Trying sentence-transformers fallback...")
            try:
                from sentence_transformers import CrossEncoder
                print(f"[Reranker] Loading {_reranker_model_name} via CrossEncoder (no instruction)...")
                _reranker = ("crossencoder", CrossEncoder(_reranker_model_name, max_length=512))
                print(f"[Reranker] Loaded (fallback mode, instruction not supported)")
            except ImportError:
                print("[Warning] Neither FlagEmbedding nor sentence-transformers installed.")
                print("  Install: pip install FlagEmbedding")
                _reranker = "disabled"
        except Exception as e:
            print(f"[Warning] Failed to load reranker: {e}")
            _reranker = "disabled"
    return _reranker if _reranker != "disabled" else None


class DiagnosisExtractor:
    """LLM 기반 진단 추출기 - 입원 주 원인 vs 동반 질환 구분"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self._cache = {}  # 캐싱으로 중복 호출 방지
    
    def extract(self, text: str, use_cache: bool = True) -> Dict[str, List[str]]:
        """
        임상 텍스트에서 입원 주 원인과 동반 질환을 구분하여 추출
        
        Args:
            text: 임상 텍스트
            use_cache: 캐시 사용 여부
        
        Returns:
            {
                'chief_complaint': ['Shortness of breath'],
                'primary_diagnosis': ['COPD EXACERBATION', 'RESPIRATORY FAILURE'],
                'comorbidities': ['CAD', 'ATRIAL FIBRILLATION', 'HYPERTENSION']
            }
        """
        # 캐시 확인
        cache_key = hash(text)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        if not self.api_key:
            print("[DiagnosisExtractor] API key not found")
            return {'chief_complaint': [], 'primary_diagnosis': [], 'comorbidities': []}
        
        prompt = f"""Analyze this clinical note and extract diagnoses into THREE categories.

Definitions:
- chief_complaint: presenting symptom/reason for visit
- primary_diagnosis: main cause of THIS hospitalization
- comorbidities: pre-existing conditions, not the main cause

Return ONLY JSON:
{{"chief_complaint": [...], "primary_diagnosis": [...], "comorbidities": [...]}}

Text:
{text}"""

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # JSON 파싱
            extracted = self._parse_structured_diagnoses(content)
            
            # 캐시 저장
            if use_cache:
                self._cache[cache_key] = extracted
            
            return extracted
            
        except Exception as e:
            print(f"[DiagnosisExtractor] LLM failed: {e}")
            return {'chief_complaint': [], 'primary_diagnosis': [], 'comorbidities': []}
    
    def _parse_structured_diagnoses(self, content: str) -> Dict[str, List[str]]:
        """LLM 응답에서 구조화된 진단 정보 파싱"""
        default = {'chief_complaint': [], 'primary_diagnosis': [], 'comorbidities': []}
        
        try:
            # JSON 블록 추출
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            
            data = json.loads(content)
            
            result = {}
            for key in ['chief_complaint', 'primary_diagnosis', 'comorbidities']:
                if key in data and isinstance(data[key], list):
                    result[key] = [d.upper().strip() for d in data[key] if isinstance(d, str)]
                else:
                    result[key] = []
            
            return result
            
        except:
            return default
    
    def is_similar(
        self, 
        extracted1: Dict[str, List[str]], 
        extracted2: Dict[str, List[str]], 
        match_type: str = "primary"
    ) -> bool:
        """
        두 환자가 임상적으로 유사한지 확인 (입원 주 원인 기준)
        
        Args:
            extracted1: 첫 번째 환자의 추출 결과
            extracted2: 두 번째 환자의 추출 결과
            match_type: 매칭 기준
                - "primary": primary_diagnosis만 비교 (권장)
                - "chief": chief_complaint도 비교
                - "any": 모든 필드 비교 (기존 방식)
        
        Returns:
            유사 여부
        """
        # 이전 버전 호환성: List가 들어오면 primary_diagnosis로 처리
        if isinstance(extracted1, list):
            extracted1 = {'primary_diagnosis': extracted1, 'chief_complaint': [], 'comorbidities': []}
        if isinstance(extracted2, list):
            extracted2 = {'primary_diagnosis': extracted2, 'chief_complaint': [], 'comorbidities': []}
        
        # 추출 실패 시 통과 (필터링 건너뜀)
        if not extracted1.get('primary_diagnosis') or not extracted2.get('primary_diagnosis'):
            return True
        
        # 동의어 확장
        def expand_with_synonyms(terms: List[str]) -> Set[str]:
            synonyms = {
                "CHF": "HEART FAILURE",
                "AFIB": "ATRIAL FIBRILLATION", 
                "CAD": "CORONARY ARTERY DISEASE",
                "MI": "MYOCARDIAL INFARCTION",
                "CVA": "STROKE",
                "COPD EXACERBATION": "COPD",
                "AECOPD": "COPD EXACERBATION",
                "RESPIRATORY FAILURE": "RESPIRATORY DISTRESS",
                "LIVER FAILURE": "HEPATIC FAILURE",
                "DECOMPENSATED CIRRHOSIS": "CIRRHOSIS",
            }
            
            expanded = set(t.upper().strip() for t in terms)
            for abbrev, full in synonyms.items():
                if abbrev in expanded:
                    expanded.add(full)
                if full in expanded:
                    expanded.add(abbrev)
            return expanded
        
        # Primary diagnosis 비교 (핵심!)
        primary1 = expand_with_synonyms(extracted1.get('primary_diagnosis', []))
        primary2 = expand_with_synonyms(extracted2.get('primary_diagnosis', []))
        
        primary_overlap = len(primary1 & primary2)
        
        if match_type == "primary":
            return primary_overlap >= 1
        
        # Chief complaint도 비교
        if match_type == "chief":
            chief1 = expand_with_synonyms(extracted1.get('chief_complaint', []))
            chief2 = expand_with_synonyms(extracted2.get('chief_complaint', []))
            chief_overlap = len(chief1 & chief2)
            return primary_overlap >= 1 or chief_overlap >= 1
        
        # Any: 동반 질환도 포함 (기존 방식, 권장하지 않음)
        all1 = primary1 | expand_with_synonyms(extracted1.get('comorbidities', []))
        all2 = primary2 | expand_with_synonyms(extracted2.get('comorbidities', []))
        return len(all1 & all2) >= 1


BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DEFAULT_DB_PATH = BASE_DIR / "data" / "vector_db"

class VectorDBManager:
    """FAISS 벡터 DB 관리 클래스 - 기존 DB 로드"""
    # MedCPT (현재 사용)
    EMBEDDING_MODEL = 'ncbi/MedCPT-Query-Encoder'
    
    # BioLORD (대안) - 위 줄 주석처리 후 아래 주석해제
    # EMBEDDING_MODEL = 'FremyCompany/BioLORD-2023-M'
    
    def __init__(self, embedding_model: str = None, db_path: str = '../data/vector_db'):
        """
        임베딩 모델 옵션:
        - ncbi/MedCPT-Query-Encoder: 검색 쿼리용 (DB는 Article-Encoder로 구축)
        - FremyCompany/BioLORD-2023-M: 단일 인코더, 임상 개념 유사성 특화
        """
        self.embedding_model = embedding_model or self.EMBEDDING_MODEL
        self.db_path = Path(DEFAULT_DB_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 쿼리 임베딩용
        self.tokenizer = None
        self.model = None
        # FAISS 인덱스 및 메타데이터 
        self.index = None
        self.metadata = []
        
    def load(self):
        """기존 벡터 DB 및 임베딩 모델 로드"""
        print(f"\n[VectorDBManager]")
        print(f"  - 임베딩 모델: {self.embedding_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model).to(self.device)
        self.model.eval()
        index_path = self.db_path / "faiss_index.idx"
        self.index = faiss.read_index(str(index_path))
        
        # 3. 메타데이터 로드 (data/vector_db/metadata.pkl)
        #    이미 text 포함한 전체 record가 저장되어 있음
        metadata_path = self.db_path / "metadata.pkl"

        print(f"  - 메타데이터 로드: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ 로드 완료: {len(self.metadata)}건의 케이스")
    
    def embed_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """텍스트를 BioBERT로 임베딩 (쿼리용)"""
        # 빈 텍스트 처리
        processed_text = text if text and text.strip() else ' '
        
        # 토큰화
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.astype(np.float32)
    
    def search(
        self, 
        query_text: str, 
        top_k: int = 3, 
        exclude_id: str = None,
        use_reranker: bool = True,
        use_diagnosis_filter: bool = True,
        rerank_top_n: int = 10  # 작은 DB용
    ) -> List[Dict]:
        """
        유사 케이스 검색 - 3단계 검색 (FAISS + 진단 필터링 + Reranking)
        
        Args:
            query_text: 검색할 텍스트
            top_k: 최종 반환할 결과 수
            exclude_id: 제외할 환자 ID (자기 자신 제외용)
            use_reranker: Reranker 사용 여부
            use_diagnosis_filter: LLM 진단 필터링 사용 여부
            rerank_top_n: Stage 1에서 가져올 후보 수
        
        Returns:
            유사 케이스 리스트 (text + 전체 metadata + similarity 포함)
        """
        
        print(f"\n[Stage 1] FAISS 검색: top-{rerank_top_n} 후보")
        
        # 자기 자신이 포함될 수 있으므로 여유있게 가져옴
        fetch_k = rerank_top_n + 10 if exclude_id else rerank_top_n
        
        query_vector = self.embed_text(query_text)
        faiss.normalize_L2(query_vector)
        similarities, indices = self.index.search(query_vector, fetch_k)
        
        # 결과 구성 (자기 자신 제외)
        candidates = []
        for dist, idx in zip(similarities[0], indices[0]):
            record = self.metadata[idx].copy()
            record['similarity'] = float(dist)
            
            # 자기 자신 제외
            if exclude_id and str(record.get('id')) == str(exclude_id):
                continue
                
            candidates.append(record)
            
            if len(candidates) >= rerank_top_n:
                break
        
        print(f"  → {len(candidates)}개 후보 검색됨")
        
        # 진단 필터링
        if use_diagnosis_filter and len(candidates) > top_k:
            candidates = self._filter_by_diagnosis(query_text, candidates)
        
        # Reranker
        if use_reranker and len(candidates) > top_k:
            candidates = self._rerank(query_text, candidates, top_k)
        else:
            candidates = candidates[:top_k]
        
        return candidates
    
    def _filter_by_diagnosis(self, query_text: str, candidates: List[Dict]) -> List[Dict]:
        """
        LLM으로 진단 추출 후 유사 진단 필터링
        
        핵심: PRIMARY DIAGNOSIS (입원 주 원인)만 비교
              COMORBIDITIES (동반 질환)는 매칭에서 제외
        
        Args:
            query_text: 쿼리 텍스트
            candidates: Stage 1 후보들
        
        Returns:
            입원 주 원인이 유사한 후보들만 필터링
        """
        print(f"\n[Stage 2] LLM 진단 필터링 (Primary Diagnosis 기준)")
        
        extractor = DiagnosisExtractor()
        
        # 쿼리 환자의 진단 추출 (구조화된 형태)
        query_extracted = extractor.extract(query_text)
        print(f"  쿼리 환자:")
        print(f"    - Primary Diagnosis: {query_extracted.get('primary_diagnosis', [])} (검색 사용 ✅)")
        print(f"    - Chief Complaint: {query_extracted.get('chief_complaint', [])} (참고용)")
        print(f"    - Comorbidities: {query_extracted.get('comorbidities', [])} (매칭 제외)")
        
        if not query_extracted.get('primary_diagnosis'):
            print("  → Primary diagnosis 추출 실패, 필터링 건너뜀")
            return candidates
        
        # 후보들 필터링 (Primary Diagnosis 기준)
        filtered = []
        for c in candidates:
            candidate_extracted = extractor.extract(c.get('text', ''))
            
            # Primary Diagnosis만 비교 (핵심!)
            if extractor.is_similar(query_extracted, candidate_extracted, match_type="primary"):
                c['extracted_diagnoses'] = candidate_extracted
                filtered.append(c)
        
        print(f"  → {len(candidates)}개 → {len(filtered)}개 (Primary Diagnosis 일치)")
        
        # 필터링된 케이스 미리보기
        for i, c in enumerate(filtered[:5]):
            primary = c.get('extracted_diagnoses', {}).get('primary_diagnosis', [])
            print(f"    {i+1}. ID={c.get('id')}: {primary}")
        
        # 필터링 결과가 너무 적으면 원본 유지
        if len(filtered) < 3:
            print(f"  → 필터링 결과 부족 ({len(filtered)}개), 원본 상위 10개 유지")
            return candidates[:10]
        
        return filtered
    
    def _rerank(self, query_text: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Instruction-tuned Cross-encoder로 후보 reranking
        
        Instruction: "Find cases that share the same primary disease mechanism"
        → 단순 텍스트 유사도가 아닌 질환 메커니즘 기반 유사성 평가
        
        Args:
            query_text: 쿼리 텍스트
            candidates: Stage 1에서 검색된 후보들
            top_k: 최종 반환할 개수
        
        Returns:
            Reranking된 상위 top_k 케이스
        """
        reranker = get_reranker()
        
        if reranker is None:
            # Reranker 사용 불가 시 원래 순서 유지
            print("[Reranker] Disabled, using original order")
            return candidates[:top_k]
        
        # Query-Candidate 쌍 생성
        pairs = []
        for c in candidates:
            candidate_text = c.get('text', '')
            pairs.append([query_text, candidate_text])
        
        # Reranker 타입에 따른 점수 산출
        print(f"[Reranker] Scoring {len(pairs)} candidates...")
        
        if isinstance(reranker, tuple) and reranker[0] == "crossencoder":
            # Fallback: CrossEncoder (instruction 미지원)
            print("[Reranker] Using CrossEncoder fallback (no instruction)")
            scores = reranker[1].predict(pairs)
        else:
            # FlagReranker with instruction
            print(f"[Reranker] Instruction: '{RERANKER_INSTRUCTION}'")
            scores = reranker.compute_score(
                pairs,
                normalize=False  # Raw logit 사용 (순서만 중요, 절댓값 무의미)
            )
            # compute_score는 단일 쌍이면 float, 여러 쌍이면 list 반환
            if not isinstance(scores, list):
                scores = [scores]
        
        # 점수로 재정렬 (순서만 바꾸고 similarity는 FAISS 원본 유지)
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
            # CRITICAL: similarity는 FAISS 원본 유지 (0.7 임계치 체크용)
            # 리랭커 스코어는 순서 정렬에만 사용
        
        # 🔍 DEBUG: 전체 리랭커 스코어 분포 확인
        all_rerank_scores = [c['rerank_score'] for c in candidates]
        print(f"[Reranker] Score 분포: min={min(all_rerank_scores):.3f}, max={max(all_rerank_scores):.3f}, avg={sum(all_rerank_scores)/len(all_rerank_scores):.3f}")
        print(f"[Reranker] 전체 {len(candidates)}개 스코어: {[f'{s:.3f}' for s in sorted(all_rerank_scores, reverse=True)]}")
        
        # 높은 rerank_score 순 정렬
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        print(f"[Reranker] Top-{top_k} selected (순서=rerank, 유사도=FAISS 원본)")
        for i, c in enumerate(candidates[:top_k]):
            print(f"  {i+1}. ID={c.get('id')}, similarity={c['similarity']:.3f} (FAISS ✅), rerank={c['rerank_score']:.3f} (순서)")
        
        return candidates[:top_k]

class RAGRetriever:
    """유사 케이스 검색기 - cohort_data 반환"""
    
    def __init__(
        self,
        db_path: str = 'data/vector_db',
        embedding_model: str = 'ncbi/MedCPT-Query-Encoder'
    ):
        self.vector_db = VectorDBManager(
            embedding_model=embedding_model,
            db_path=db_path
        )
        self.is_loaded = False
    
    def load(self):
        """벡터 DB 로드"""
        self.vector_db.load()
        self.is_loaded = True
        print(f"\n RAG Retriever 로드 완료: {len(self.vector_db.metadata)}건")
    
    def retrieve(
        self, 
        query_text: str, 
        top_k: int = 3,
        exclude_id: str = None,
        use_reranker: bool = True,
        use_diagnosis_filter: bool = True,
        rerank_top_n: int = 10
    ) -> Dict:
        """
        유사 케이스 검색 및 cohort_data 반환 (3단계 검색)
        
        Args:
            query_text: 검색할 텍스트
            top_k: 최종 반환할 결과 수
            exclude_id: 제외할 환자 ID (자기 자신 제외)
            use_reranker: Reranker 사용 여부
            use_diagnosis_filter: LLM 진단 필터링 사용 여부
            rerank_top_n: Stage 1에서 가져올 후보 수
        """
        if not self.is_loaded:
            self.load()
        
        # 유사 케이스 검색 (3단계: FAISS → 진단필터링 → Reranking)
        similar_cases = self.vector_db.search(
            query_text, 
            top_k=top_k,
            exclude_id=exclude_id,
            use_reranker=use_reranker,
            use_diagnosis_filter=use_diagnosis_filter,
            rerank_top_n=rerank_top_n
        )
        
        # cohort_data 구성
        cohort_data = {
            'similar_cases': similar_cases,
            'stats': self._calculate_stats(similar_cases)
        }
        
        return cohort_data
    
    def _calculate_stats(self, similar_cases: List[Dict]) -> Dict:
        """유사 케이스 통계 계산"""
        if not similar_cases:
            return {'total': 0, 'expired_count': 0, 'survival_rate': None}
        
        total = len(similar_cases)
        expired_count = sum(
            1 for c in similar_cases 
            if c.get('status', '').lower() in ['expired', 'died', 'death']
        )
        survival_rate = (total - expired_count) / total if total > 0 else None
        
        return {
            'total': total,
            'expired_count': expired_count,
            'survival_rate': survival_rate
        }
    
    def retrieve_with_patient(
        self, 
        patient_data: Dict, 
        top_k: int = 3,
        use_reranker: bool = True,
        use_diagnosis_filter: bool = True,
        rerank_top_n: int = 10
    ) -> Dict:
        """
        환자 데이터로 유사 케이스 검색 및 cohort_data 반환
        
        Args:
            patient_data: 환자 정보 딕셔너리
                {
                    'id': str,
                    'status': str,
                    'sex': str,
                    'age': int,
                    'admission_type': str,
                    'admission_location': str,
                    'discharge_location': str,
                    'arrival_transport': str,
                    'text': str
                }
            top_k: 최종 반환할 결과 수
            use_reranker: Reranker 사용 여부
            use_diagnosis_filter: LLM 진단 필터링 사용 여부
            rerank_top_n: Stage 1에서 가져올 후보 수
        
        Returns:
            cohort_data (위와 동일)
        """
        # 'text' 또는 'clinical_text' 키 모두 지원
        query_text = patient_data.get('text', '') or patient_data.get('clinical_text', '')
        patient_id = patient_data.get('id') or patient_data.get('patient_id')  # 자기 자신 제외용
        
        return self.retrieve(
            query_text, 
            top_k=top_k,
            exclude_id=patient_id,
            use_reranker=use_reranker,
            use_diagnosis_filter=use_diagnosis_filter,
            rerank_top_n=rerank_top_n
        )