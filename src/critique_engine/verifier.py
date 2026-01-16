"""
Verifier 모듈 
"solution": critique_reasoner 가 비판한 것에 대한 해결점 생성 (유사 케이스 topk =3 근거를 가지고) 생성, 이후 solution 이라는 변수에 저장
llm 은 동일한걸로 불러오기
"""
from typing import List, Dict, Optional
import re


class ExpertBase:
    """전문가 베이스 클래스"""
    
    def __init__(self, name: str):
        self.name = name
  
