"""
RAG Retriever - FAISS 기반 유사 케이스 검색
이미 생성된 FAISS DB와 metadata.pkl 사용
텍스트 기반 유사 케이스 검색 (코사인 유사도) 및 전체 메타데이터 반환
"""

import numpy as np
import pickle
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel


class VectorDBManager:
    """FAISS 벡터 DB 관리 클래스 - 기존 DB 로드"""
    
    def __init__(self, embedding_model: str = 'dmis-lab/biobert-v1.1', db_path: str = 'data/vector_db'):
        self.embedding_model = embedding_model
        self.db_path = Path(db_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 쿼리 임베딩용
        self.tokenizer = None
        self.model = None
        # FAISS 인덱스 및 메타데이터 
        self.index = None
        self.metadata = []
        
    def load(self):
        """기존 벡터 DB 및 BioBERT 모델 로드"""
        print(f"\n[VectorDBManager]")
        
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
    
    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        유사 케이스 검색 - 기존 FAISS DB에서 검색
        
        Args:
            query_text: 검색할 텍스트
            top_k: 반환할 결과 수
        
        Returns:
            유사 케이스 리스트 (text + 전체 metadata + similarity 포함)
        """
        # 쿼리 임베딩
        query_vector = self.embed_text(query_text)
        faiss.normalize_L2(query_vector) #L2 정규화
        similarities, indices = self.index.search(query_vector, top_k)
        
        # 결과 구성
        results = []
        for dist, idx in zip(similarities[0], indices[0]):
            
            # 메타데이터에서 전체 record 가져오기
            # metadata.pkl에는 이미 text 포함된 전체 데이터가 저장되어 있음
            record = self.metadata[idx].copy()
            record['similarity'] = float(similarities)
            
            results.append(record)
        
        return results

class RAGRetriever:
    """유사 케이스 검색기 - cohort_data 반환"""
    
    def __init__(
        self,
        db_path: str = 'data/vector_db',
        embedding_model: str = 'dmis-lab/biobert-v1.1'
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
    
    def retrieve(self, query_text: str, top_k: int = 3) -> Dict:
        """
        유사 케이스 검색 및 cohort_data 반환
        
        Args:
            query_text: 검색할 텍스트
            top_k: 반환할 결과 수
        """
        if not self.is_loaded:
            self.load()
        
        # 유사 케이스 검색
        similar_cases = self.vector_db.search(query_text, top_k=top_k)
        
        # 생존 통계 계산
        stats = self._calculate_stats(similar_cases)
        
        # cohort_data 구성
        cohort_data = {
            'similar_cases': similar_cases,
            'stats': stats
        }
        
        return cohort_data
    
    def retrieve_with_patient(self, patient_data: Dict, top_k: int = 3) -> Dict:
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
                    'disposition': str,
                    'text': str
                }
            top_k: 반환할 결과 수
        
        Returns:
            cohort_data (위와 동일)
        """
        query_text = patient_data.get('text', '')
        return self.retrieve(query_text, top_k=top_k)
    
    def _calculate_stats(self, similar_cases: List[Dict]) -> Dict:
        """
        유사 케이스의 생존 통계 계산
        
        Args:
            similar_cases: 검색된 케이스 리스트
        
        Returns:
            생존 통계
        """
        if not similar_cases:
            return {
                'total': 0,
                'alive': 0,
                'dead': 0,
                'survival_rate': 0.0
            }
        
        total = len(similar_cases)
        alive_count = sum(1 for c in similar_cases if c.get('status') == 'alive')
        dead_count = total - alive_count
        
        return {
            'total': total,
            'alive': alive_count,
            'dead': dead_count,
            'survival_rate': alive_count / total if total > 0 else 0.0
        }