"""
의료 데이터 처리 및 벡터 DB 생성 파이프라인

필수 패키지 설치:
    pip install transformers torch faiss-cpu pandas tqdm --break-system-packages

임베딩 모델 옵션:
- MedCPT (현재): ncbi/MedCPT-Article-Encoder (비대칭, 검색 특화)
- BioLORD (대안): FremyCompany/BioLORD-2023-M (단일, 개념 유사성)

작업 순서:
1. CSV 데이터 로드 (df_flag0_final_processed.csv + df_flag1_final_processed.csv)
2. 두 파일 병합 (flag0 + flag1)
3. hospital_expire_flag를 status로 매핑 (0→alive, 1→dead)
4. 필요한 컬럼 추출 및 JSON 변환
5. 임베딩 모델로 텍스트 임베딩 생성
6. FAISS 벡터 데이터베이스 생성 및 저장

출력:
- data/processed_data.json (전처리된 데이터)
- data/vector_db/faiss_index.idx (FAISS 인덱스)
- data/vector_db/metadata.pkl (메타데이터)
"""

import pandas as pd
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm
import faiss
import os
FAISS_AVAILABLE = True
from transformers import AutoTokenizer, AutoModel
TRANSFORMERS_AVAILABLE = True

# 프로젝트 루트 경로 설정 (scripts/ 폴더에서 실행해도 정상 작동)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MedicalDataLoader:
    """의료 데이터 로더 클래스"""
    
    def __init__(self, flag0_file: str, flag1_file: str):
        self.flag0_file = flag0_file
        self.flag1_file = flag1_file
        
    def load_and_process(self) -> List[Dict]:
        """데이터를 로드하고 전처리하여 JSON 형식으로 반환"""
        
        # 두 데이터 파일 로드
        print(f"\nLoading flag0 data file: {self.flag0_file}")
        df_flag0 = pd.read_csv(self.flag0_file)
        print(f"Flag0 rows loaded: {len(df_flag0)}")
        
        print(f"\nLoading flag1 data file: {self.flag1_file}")
        df_flag1 = pd.read_csv(self.flag1_file)
        print(f"Flag1 rows loaded: {len(df_flag1)}")
        
        # 두 데이터프레임 병합
        df = pd.concat([df_flag0, df_flag1], ignore_index=True)
        print(f"\nTotal rows after merge: {len(df)}")
        
        # 필요한 컬럼만 선택
        required_columns = [
            'stay_id',
            'hospital_expire_flag',
            'gender', 
            'anchor_age', 
            'admission_type', 
            'admission_location', 
            'discharge_location', 
            'arrival_transport', 
            'text'
        ]
        
        # 필요한 컬럼이 데이터프레임에 있는지 확인
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 필요한 컬럼만 선택
        df = df[required_columns]
        
        # 모든 필수 컬럼에 결측치가 하나라도 있는 행 제거
        df_clean = df.dropna()
        
        removed_count = len(df) - len(df_clean)
        print(f"Removed {removed_count} rows with NaN values")
        print(f"Remaining rows: {len(df_clean)}")
        
        # JSON 형식으로 변환
        records = []
        
        for _, row in df_clean.iterrows():
            # hospital_expire_flag를 status로 매핑
            status = 'dead' if row['hospital_expire_flag'] == 1 else 'alive'
            
            record = {
                'id': str(int(row['stay_id'])),
                'status': status,
                'sex': str(row['gender']),
                'age': int(row['anchor_age']),
                'admission_type': str(row['admission_type']),
                'admission_location': str(row['admission_location']),
                'discharge_location': str(row['discharge_location']),
                'arrival_transport': str(row['arrival_transport']),
                'text': str(row['text'])
            }
            records.append(record)
        
        print(f"Successfully processed: {len(records)} records")
        
        return records
    
    def save_to_json(self, records: List[Dict], output_path: str):
        """레코드를 JSON 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print(f"데이터 저장: {output_path}")

# MedCPT (현재 사용)
EMBEDDING_MODEL = 'ncbi/MedCPT-Article-Encoder'

# BioLORD (대안) - 위 줄 주석처리 후 아래 주석해제
# EMBEDDING_MODEL = 'FremyCompany/BioLORD-2023-M'


class MedCPTEmbedder:
    """임상 텍스트 임베딩 클래스
    
    지원 모델:
    - ncbi/MedCPT-Article-Encoder: 문서 임베딩용 (비대칭)
    - FremyCompany/BioLORD-2023-M: 단일 인코더 (대칭)
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n임베딩 모델: {self.model_name}")
        print(f"Device: {self.device}")
        
        # 토크나이저와 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        print("모델 로드 완료")
    
    def embed_batch(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """여러 텍스트를 배치로 임베딩"""
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i+batch_size]
            
            # 빈 텍스트 처리
            processed_texts = [text if text and text.strip() else ' ' for text in batch_texts]
            
            # 토큰화
            inputs = self.tokenizer(
                processed_texts,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        result = np.vstack(embeddings).astype(np.float32)
        print(f"Embeddings 생성 완료: {result.shape}")
        return result

#FAISS 기반 벡터 db 구축

class FAISSVectorDB:
    """FAISS 벡터 데이터베이스 클래스"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """벡터와 메타데이터 추가"""
        faiss.normalize_L2(vectors) #L2 정규화
        self.index.add(vectors)
        self.metadata.extend(metadata)
        
        print(f"\n벡터 DB 생성 완료")
    
    def save(self, save_dir: str):
        """인덱스와 메타데이터 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = save_path / "faiss_index.idx"
        faiss.write_index(self.index, str(index_path))
        
        # 메타데이터 저장
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\n 벡터 DB 저장 완료:")


#파이프라인

def run_pipeline(
    flag0_file: str,
    flag1_file: str,
    json_output: str = "data/processed_data.json",
    vector_db_output: str = "data/vector_db",
    batch_size: int = 8,
    max_length: int = 512
):
    """FAISS DB 생성 파이프라인 실행"""
    print("\n" + "="*70)
    print("Medical Data Processing & MedCPT FAISS DB Creation Pipeline")
    print("="*70)
    
    # 데이터 로딩 (flag0 + flag1)

    loader = MedicalDataLoader(flag0_file, flag1_file)
    records = loader.load_and_process()
    loader.save_to_json(records, json_output)
    
    # 임베딩

    embedder = MedCPTEmbedder()
    texts = [record['text'] for record in records]
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, max_length=max_length)
    
    # 벡터DB

    vector_db = FAISSVectorDB(dimension=768)
    vector_db.add_vectors(embeddings, records)
    
    # db 저장
    vector_db.save(vector_db_output)
    
    print("\n" + "="*70)
    print("db 파이프라인 완료!")
    print("="*70)


def main():
    """실행"""
    run_pipeline(
        flag0_file=str(PROJECT_ROOT / 'data' / 'df_flag0_final_processed.csv'),
        flag1_file=str(PROJECT_ROOT / 'data' / 'df_flag1_final_processed.csv'),
        json_output=str(PROJECT_ROOT / 'data' / 'processed_data.json'),
        vector_db_output=str(PROJECT_ROOT / 'data' / 'vector_db'),
        batch_size=8,
        max_length=512
    )


if __name__ == '__main__':
    main()
