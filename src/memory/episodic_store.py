"""
Episodic Memory Store - 에피소딕 메모리 저장소

크로스런(cross-run) 학습을 위한 에피소딕 메모리.
과거 분석 경험(진단, critique, 교훈)을 저장하고,
새 케이스가 들어올 때 유사 경험을 검색하여 프롬프트에 주입.

검색 전략
  진단명 기반 사전 필터 → 같은 진단 에피소드를 확실히 잡음
  LLM 요약 임베딩 → 저장 시 GPT로 임상 요약 후 임베딩, 같은 진단 내 정밀 순위

구현:
  - FAISS 벡터 인덱스 (MedCPT 인프라 재활용)
  - JSON 메타데이터 (에피소드 정보)
  - 저장: clinical_text → LLM 요약 → MedCPT 임베딩
  - 검색: 진단명 필터 → FAISS 유사도 순위
"""

import json
import os
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModel


BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DEFAULT_EPISODIC_PATH = BASE_DIR / "data" / "episodic_db"


class EpisodicMemoryStore:
    """
    에피소딕 메모리 저장소
    
    - 과거 분석 경험을 FAISS + JSON으로 저장
    - 유사 케이스의 과거 경험을 검색 (진단 필터 + 임베딩 유사도)
    - MedCPT 임베딩 모델 재사용
    """
    
    EMBEDDING_MODEL = "ncbi/MedCPT-Query-Encoder"
    EMBEDDING_DIM = 768  # MedCPT output dimension
    
    def __init__(self, db_path: str = None, shared_embedder=None):
        """
        Args:
            db_path: 에피소딕 DB 저장 경로
            shared_embedder: RAGRetriever의 VectorDBManager를 공유하여
                             모델 중복 로딩 방지 (embed_text 메서드 필요)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_EPISODIC_PATH
        self.shared_embedder = shared_embedder
        
        # 자체 임베딩 모델 (shared_embedder 없을 때)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FAISS 인덱스 + 메타데이터
        self.index = None
        self.episodes: List[Dict] = []
        self.is_loaded = False
    
    def load(self):
        """에피소딕 DB 로드 (없으면 새로 생성)"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        index_path = self.db_path / "episodic_faiss.idx"
        meta_path = self.db_path / "episodic_meta.json"
        
        # FAISS 인덱스 로드 또는 생성
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            print(f"  [EpisodicMemory] FAISS 인덱스 로드: {self.index.ntotal}건")
        else:
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            print("  [EpisodicMemory] 새 FAISS 인덱스 생성")
        
        # 메타데이터 로드 또는 생성
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.episodes = json.load(f)
            print(f"  [EpisodicMemory] 메타데이터 로드: {len(self.episodes)}건")
        else:
            self.episodes = []
            print("  [EpisodicMemory] 새 메타데이터 생성")
        
        # 임베딩 모델 로드 (shared_embedder 없을 때만)
        if self.shared_embedder is None:
            self._load_embedding_model()
        
        self.is_loaded = True
    
    def _load_embedding_model(self):
        """MedCPT 임베딩 모델 로드"""
        print(f"  [EpisodicMemory] 임베딩 모델 로드: {self.EMBEDDING_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(self.EMBEDDING_MODEL).to(self.device)
        self.model.eval()
    
    # ──────────────────────────────────────────────
    # 임베딩
    # ──────────────────────────────────────────────
    
    def _embed_text(self, text: str) -> np.ndarray:
        """텍스트를 MedCPT로 임베딩 (저장: LLM 요약문, 검색: raw text)"""
        if self.shared_embedder is not None:
            return self.shared_embedder.embed_text(text)
        
        processed_text = text if text and text.strip() else " "
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.astype(np.float32)
    
    def save(self):
        """에피소딕 DB를 디스크에 저장"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        index_path = self.db_path / "episodic_faiss.idx"
        meta_path = self.db_path / "episodic_meta.json"
        
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)
        
        print(f"  [EpisodicMemory] 저장 완료: {len(self.episodes)}건 -> {self.db_path}")
    
    # ──────────────────────────────────────────────
    # LLM 임상 요약 (저장 시 1회)
    # ──────────────────────────────────────────────
    
    def _summarize_clinical_text(self, clinical_text: str) -> str:
        """
        GPT-4o-mini로 임상 텍스트를 핵심 요약.
        저장 시 1회만 호출. 요약문을 임베딩하여 FAISS에 저장.
        API 키 없으면 앞부분 발췌로 fallback.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("  [EpisodicMemory] API 키 없음 -> 텍스트 발췌 fallback")
            return clinical_text
        
        try:
            from ..llm.openai_chat import OpenAIChatConfig, call_openai_chat_completions
            
            prompt = f"""Summarize this clinical case in 150-200 words for case similarity matching.
Include: primary diagnosis, key comorbidities, presenting symptoms, critical lab/vital findings,
major treatments given, clinical course, and outcome.
Do NOT include patient identifiers.

Clinical text:
{clinical_text}

Summary:"""
            
            cfg = OpenAIChatConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=400)
            summary = call_openai_chat_completions(
                messages=[{"role": "user", "content": prompt}],
                config=cfg,
            )
            print(f"  [EpisodicMemory] LLM 요약 생성: {len(summary)} chars")
            return summary
        except Exception as e:
            print(f"  [EpisodicMemory] LLM 요약 실패 ({e}) -> 텍스트 발췌 fallback")
            return clinical_text
    
    # ──────────────────────────────────────────────
    # 에피소드 저장
    # ──────────────────────────────────────────────
    
    def add_episode(
        self,
        patient_case: Dict,
        critique_points: List[Dict],
        solutions: List[Dict],
        confidence: float,
        diagnosis_analysis: Optional[Dict] = None,
        treatment_analysis: Optional[Dict] = None,
    ):
        """
        분석 에피소드를 메모리에 저장
        
        저장 흐름:
          clinical_text -> LLM 요약 -> MedCPT 임베딩 -> FAISS
          메타데이터 (진단, 비판, 교훈, 솔루션) -> JSON
        """
        if not self.is_loaded:
            self.load()
        
        # LLM 요약 생성 (임베딩용)
        clinical_text = patient_case.get("clinical_text", "") or patient_case.get("text", "")
        clinical_summary = self._summarize_clinical_text(clinical_text)
        
        # 에피소드 구성
        episode = {
            "episode_id": f"EP-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_case.get("patient_id") or patient_case.get("id"),
            "diagnosis": patient_case.get("diagnosis", "Unknown"),
            "secondary_diagnoses": patient_case.get("secondary_diagnoses", []),
            "outcome": patient_case.get("outcome") or patient_case.get("status"),
            "clinical_summary": clinical_summary,
            "critique_summary": self._summarize_critiques(critique_points),
            "lessons_learned": self._extract_lessons(critique_points, solutions),
            "key_solutions": self._summarize_solutions(solutions),
            "confidence": confidence,
        }
        
        # 임베딩 생성 (LLM 요약문 기반)
        embedding = self._embed_text(clinical_summary)
        faiss.normalize_L2(embedding)
        
        # FAISS에 추가
        self.index.add(embedding)
        self.episodes.append(episode)
        
        # 자동 저장
        self.save()
        
        print(f"  [EpisodicMemory] 에피소드 저장: {episode['episode_id']} "
              f"(진단: {episode['diagnosis']}, confidence: {confidence:.2f})")
        
        return episode
    
    def _summarize_critiques(self, critique_points: List[Dict]) -> List[Dict]:
        """Critique를 요약 형태로 저장"""
        if not critique_points:
            return []
        
        summaries = []
        for cp in critique_points:
            if isinstance(cp, dict):
                summaries.append({
                    "issue": cp.get("issue", ""),
                    "severity": cp.get("severity", "low"),
                    "category": cp.get("category", "unknown"),
                })
        return summaries
    
    def _extract_lessons(
        self,
        critique_points: List[Dict],
        solutions: List[Dict],
    ) -> List[str]:
        """비판 + 솔루션에서 교훈 추출"""
        lessons = []
        
        # Critical/medium 비판에서 교훈 추출
        for cp in (critique_points or []):
            if isinstance(cp, dict):
                severity = cp.get("severity", "low")
                issue = cp.get("issue", "")
                if issue and severity in ("critical", "medium"):
                    lessons.append(f"[{severity.capitalize()}] {issue}")
        
        # 솔루션의 핵심 액션
        for sol in (solutions or []):
            if isinstance(sol, dict):
                action = sol.get("action", "")
                target = sol.get("target_issue", "")
                if action and target:
                    lessons.append(f"[Solution] {target} -> {action}")
        
        return lessons[:10]  # 최대 10개
    
    def _summarize_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Solutions를 요약 형태로 저장"""
        if not solutions:
            return []
        
        summaries = []
        for sol in solutions:
            if isinstance(sol, dict):
                summaries.append({
                    "target_issue": sol.get("target_issue", ""),
                    "action": sol.get("action", "")[:200],
                    "priority": sol.get("priority", "short-term"),
                })
        return summaries[:6]  # 최대 6개
    
    # ──────────────────────────────────────────────
    # 진단명 매칭
    # ──────────────────────────────────────────────
    
    @staticmethod
    def _diagnosis_matches(query_dx: str, episode: Dict) -> bool:
        """진단명이 매칭되는지 확인 (대소문자 무시, 부분 문자열 포함)"""
        if not query_dx:
            return False
        q = query_dx.lower().strip()
        
        # primary diagnosis 매칭
        ep_dx = (episode.get("diagnosis") or "").lower()
        if q in ep_dx or ep_dx in q:
            return True
        
        # secondary diagnoses 매칭
        for sec in (episode.get("secondary_diagnoses") or []):
            sec_lower = sec.lower()
            if q in sec_lower or sec_lower in q:
                return True
        
        return False
    
    # ──────────────────────────────────────────────
    # 유사 경험 검색 (1+3: 진단 필터 + 요약 임베딩 유사도)
    # ──────────────────────────────────────────────
    
    def search_similar_episodes(
        self,
        clinical_text: str,
        top_k: int = 3,
        min_similarity: float = 0.7,
        diagnosis: str = "",
        secondary_diagnoses: List[str] = None,
    ) -> List[Dict]:
        """
        유사 케이스의 과거 분석 경험을 검색
        
        검색 전략:
          1단계: 진단명 매칭으로 후보 필터
          2단계: FAISS 임베딩 유사도로 후보 내 순위
          fallback: 진단 매칭 없으면 전체 FAISS 검색
        
        Args:
            clinical_text: 현재 환자의 임상 텍스트
            top_k: 반환할 에피소드 수
            min_similarity: 최소 유사도 임계값
            diagnosis: 현재 환자의 주 진단명 (필터용)
            secondary_diagnoses: 현재 환자의 부 진단명 목록
        """
        if not self.is_loaded:
            self.load()
        
        if self.index.ntotal == 0:
            print("  [EpisodicMemory] 저장된 에피소드 없음")
            return []
        
        # 쿼리 임베딩 (chunk mean pooling으로 전체 텍스트 반영)
        query_vec = self._embed_text(clinical_text)
        faiss.normalize_L2(query_vec)
        
        # 전체 FAISS 검색 (넉넉히 가져옴)
        search_k = min(self.index.ntotal, max(top_k * 3, 10))
        similarities, indices = self.index.search(query_vec, search_k)
        
        # 진단명 목록 구성
        all_diagnoses = [diagnosis] + (secondary_diagnoses or [])
        all_diagnoses = [d for d in all_diagnoses if d]
        
        dx_matched = []
        dx_unmatched = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.episodes):
                continue
            if float(sim) < min_similarity:
                continue
            
            episode = self.episodes[idx].copy()
            episode["similarity"] = round(float(sim), 4)
            
            # 진단명 매칭 여부
            matched = any(self._diagnosis_matches(dx, episode) for dx in all_diagnoses)
            
            if matched:
                dx_matched.append(episode)
            else:
                dx_unmatched.append(episode)
        
        # 진단 매칭된 것 우선, 그 안에서 유사도 순
        if dx_matched:
            results = sorted(dx_matched, key=lambda x: x["similarity"], reverse=True)[:top_k]
            print(f"  [EpisodicMemory] 진단 매칭 {len(dx_matched)}건 중 top-{len(results)} 반환")
        else:
            # fallback: 전체에서 유사도 순
            all_results = sorted(dx_unmatched, key=lambda x: x["similarity"], reverse=True)
            results = all_results[:top_k]
            if results:
                print(f"  [EpisodicMemory] 진단 매칭 없음 -> 임베딩 유사도 기반 {len(results)}건 반환")
        
        if results:
            print(f"  [EpisodicMemory] 유사 경험 {len(results)}건 "
                  f"(최고 유사도: {results[0]['similarity']:.3f})")
        else:
            print("  [EpisodicMemory] 유사 경험 없음")
        
        return results


#프롬프트

    def format_for_prompt(self, episodes: List[Dict], max_episodes: int = 2) -> str:
        """검색된 에피소드를 프롬프트에 주입할 문자열로 변환"""
        if not episodes:
            return "과거 유사 경험 없음"
        
        lines = []
        for i, ep in enumerate(episodes[:max_episodes]):
            lines.append(f"--- 과거 경험 #{i+1} (유사도: {ep.get('similarity', 0):.2f}) ---")
            lines.append(f"진단: {ep.get('diagnosis', 'Unknown')}")
            lines.append(f"결과: {ep.get('outcome', 'Unknown')}")
            lines.append(f"신뢰도: {ep.get('confidence', 0):.2f}")
            
            # 핵심 교훈
            lessons = ep.get("lessons_learned", [])
            if lessons:
                lines.append("교훈:")
                for lesson in lessons[:5]:
                    lines.append(f"  - {lesson}")
            
            # 핵심 비판점
            critiques = ep.get("critique_summary", [])
            critical_ones = [c for c in critiques if c.get("severity") == "critical"]
            if critical_ones:
                lines.append("주요 비판점:")
                for c in critical_ones[:3]:
                    lines.append(f"  - [{c.get('category', '')}] {c.get('issue', '')}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    @property
    def episode_count(self) -> int:
        """저장된 에피소드 수"""
        return len(self.episodes)
