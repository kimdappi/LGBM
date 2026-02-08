"""LLM Wrapper - OpenAI API 호출을 위한 래퍼 클래스"""

import os
from openai import OpenAI
from typing import Optional


class LLMWrapper:
    """OpenAI API를 위한 래퍼 클래스"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 가져옴)
            model: 사용할 모델명 (없으면 환경변수에서 가져옴)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=self.api_key)
    
    def gpt4o(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_mode: bool = False,
        timeout: int = 60
    ) -> str:
        """
        GPT-4o 모델 호출
        
        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트 (선택)
            temperature: 온도 (0.0-2.0)
            max_tokens: 최대 토큰 수
            json_mode: JSON 모드 활성화 여부
            timeout: 타임아웃 (초)
        
        Returns:
            str: 모델 응답
        """
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        # JSON 모드 설정
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    


# 싱글톤 인스턴스
_llm_instance = None


def get_llm() -> LLMWrapper:
    """
    LLM 래퍼 인스턴스를 반환 (싱글톤 패턴)
    
    Returns:
        LLMWrapper: LLM 래퍼 인스턴스
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    
    return _llm_instance