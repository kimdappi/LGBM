"""
의료 데이터 처리 및 BioBERT 벡터 DB 생성 파이프라인

필수 패키지 설치:
    pip install transformers torch faiss-cpu pandas tqdm --break-system-packages

작업 순서:
1. CSV 데이터 로드 (flag_1: dead, flag_0: alive)
2. hospital_expire_flag를 status로 매핑 (0→alive, 1→dead)
3. 필요한 컬럼 추출 및 JSON 변환
4. BioBERT로 텍스트 임베딩 생성
5. FAISS 벡터 데이터베이스 생성 및 저장

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
import os
from tqdm import tqdm
import faiss

# Force CPU mode for macOS compatibility - disable CUDA and MPS
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: False

FAISS_AVAILABLE = True
from transformers import AutoTokenizer, AutoModel
TRANSFORMERS_AVAILABLE = True

# 프로젝트 루트 경로 설정 (scripts/ 폴더에서 실행해도 정상 작동)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MedicalDataLoader:
    """의료 데이터 로더 클래스"""
    
    def __init__(self, dead_file_path: str, alive_file_path: str):
        self.dead_file_path = dead_file_path
        self.alive_file_path = alive_file_path
        
    def load_and_process(self) -> List[Dict]:
        """데이터를 로드하고 전처리하여 JSON 형식으로 반환"""
        
        # 데이터 로드
        print(f"\nLoading data files...")
        df_dead = pd.read_csv(self.dead_file_path)
        df_alive = pd.read_csv(self.alive_file_path)
        
        # 데이터 합치기
        df = pd.concat([df_dead, df_alive], ignore_index=True)
        
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
            'disposition', 
            'text'
        ]
        
        # JSON 형식으로 변환
        records = []
        for _, row in df.iterrows():
            # hospital_expire_flag를 status로 매핑
            status = 'dead' if row['hospital_expire_flag'] == 1 else 'alive'
            
            record = {
                'id': str(row['stay_id']),
                'status': status,
                'sex': row['gender'],
                'age': int(row['anchor_age']),
                'admission_type': row['admission_type'],
                'admission_location': row['admission_location'],
                'discharge_location': row['discharge_location'],
                'arrival_transport': row['arrival_transport'],
                'disposition': row['disposition'],
                'text': row['text']
            }
            records.append(record)
        
        return records
    
    def save_to_json(self, records: List[Dict], output_path: str):
        """레코드를 JSON 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print(f"데이터 저장: {output_path}")

class BioBERTEmbedder:
    """BioBERT를 사용한 텍스트 임베딩 클래스"""
    
    def __init__(self, model_name: str = 'dmis-lab/biobert-v1.1'):
        self.model_name = model_name
        # Force CPU mode for macOS - disable CUDA
        self.device = torch.device('cpu')
        
        print(f"\nBioBERT model on {self.device}")
        
        # 토크나이저와 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("BioBERT 모델 로드 완료")
    
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
    dead_file: str,
    alive_file: str,
    json_output: str = "data/processed_data.json",
    vector_db_output: str = "data/vector_db",
    batch_size: int = 8,
    max_length: int = 512
):
    """FAISS DB 생성 파이프라인 실행"""
    print("\n" + "="*70)
    print("Medical Data Processing & BioBERT FAISS DB Creation Pipeline")
    print("="*70)
    
    # 1. 데이터 로드 및 전처리

    loader = MedicalDataLoader(dead_file, alive_file)
    records = loader.load_and_process()
    loader.save_to_json(records, json_output)
    
    # 2. BioBERT 임베딩 생성

    embedder = BioBERTEmbedder()
    texts = [record['text'] for record in records]
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, max_length=max_length)
    
    # 3. FAISS 벡터 DB 생성

    vector_db = FAISSVectorDB(dimension=768)
    vector_db.add_vectors(embeddings, records)
    
    # 4. 벡터 DB 저장
    vector_db.save(vector_db_output)
    
    print("\n" + "="*70)
    print("db 파이프라인 완료!")
    print("="*70)


def main():
    """메인 실행"""
    run_pipeline(
        dead_file=str(PROJECT_ROOT / 'data' / 'flag_1_textclean.csv'),
        alive_file=str(PROJECT_ROOT / 'data' / 'flag_0_textclean.csv'),
        json_output=str(PROJECT_ROOT / 'data' / 'processed_data.json'),
        vector_db_output=str(PROJECT_ROOT / 'data' / 'vector_db'),
        batch_size=8,
        max_length=512
    )


if __name__ == '__main__':
    main()
