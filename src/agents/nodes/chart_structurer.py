"""Chart Structurer Agent - Information Extraction"""

import json
from typing import Dict
from ..llm import get_llm

# LLM 인스턴스
llm = get_llm()



STRUCTURING_PROMPT = """
의료 차트에서 핵심 정보를 **구조화된 JSON**으로 추출하세요.

규칙:
- 출력은 **반드시 JSON만** (추가 설명/마크다운 금지)
- 차트에 **명시된 내용만** 추출 (추론 금지)
- 정보가 없으면 `null` 또는 `[]` 사용
- 치료/검사가 시행되었으면 반드시 `interventions_given`에 포함

입력 차트:
{clinical_text}

{{
    "demographics": {{
        "age": 숫자,
        "sex": "M/F",
        "chief_complaint": "주 호소"
    }},
    "vitals": {{
        "temperature": 숫자 (°C),
        "blood_pressure": "수축기/이완기",
        "heart_rate": 숫자,
        "respiratory_rate": 숫자,
        "oxygen_saturation": 숫자 (%),
        "oxygen_requirement": "room air / O2 via NC / NIV / intubated"
    }},
    "symptoms": {{
        "respiratory": ["dyspnea", "cough", "wheeze", "chest_pain"],
        "cardiovascular": ["edema", "JVD", "orthopnea"],
        "systemic": ["fever", "fatigue", "confusion"],
        "duration": "발현 기간"
    }},
    "red_flags": [
        "명시된 위험 징후들 (예: fever, severe hypoxia, altered mental status)"
    ],
    "physical_exam": {{
        "lung_sounds": "clear / crackles / wheeze / diminished",
        "heart_sounds": "regular / irregular / murmur",
        "extremities": "no edema / pitting edema / cyanosis",
        "jvd_present": true/false
    }},
    "laboratory": {{
        "wbc": 숫자,
        "procalcitonin": 숫자,
        "bnp": 숫자,
        "lactate": 숫자,
        "abg": {{"ph": 숫자, "pco2": 숫자, "po2": 숫자}}
    }},
    "imaging": {{
        "chest_xray": "소견 (infiltrate, cardiomegaly, clear 등)",
        "ct_chest": "소견"
    }},
    "procedures_performed": [
        {{
            "name": "시술명 (Paracentesis, Central Line, Thoracentesis, Chest Tube, LP, Intubation 등)",
            "technique": "시술 기법 (ultrasound-guided / blind / landmark-based / sterile / not documented)",
            "timing": "시점",
            "complications": "합병증 (bleeding, pneumothorax, infection, organ injury 등, 없으면 null)",
            "safety_flags": ["발견된 안전 문제 (예: 'blind technique', 'no sterile documented', 'puncture site bleeding')"]
        }}
    ],
    "interventions_given": {{
        "medications": [
            {{"name": "약물명", "timing": "시점/시간", "route": "경로"}}
        ],
        "oxygen_therapy": [
            {{"type": "NC/NIV/intubation", "timing": "시점"}}
        ],
        "fluids": "수액 투여 여부 및 양"
    }},
    "clinical_course": {{
        "improvement": true/false,
        "deterioration": true/false,
        "events": ["주요 경과 사건들"],
        "oxygen_trend": "increasing / stable / decreasing",
        "symptom_resolution": ["호전된 증상들"]
    }},
    "outcome": {{
        "status": "alive/expired/transferred",
        "discharge_condition": "텍스트 원문 그대로 (예: 'Expired', 'Improved', 'Stable')",
        "discharge_location": "HOME/DIED/ICU/등 원문 그대로",
        "disposition": "ADMITTED/DISCHARGED/TRANSFERRED",
        "cause_of_death": "사망한 경우, 직접 사망 원인 (예: Iatrogenic Hemoperitoneum, 없으면 null)",
        "critical_events_leading_to_outcome": [
            "시간 순서대로 중요 사건 (예: 'Blind paracentesis performed', 'Puncture site bleeding', 'Hct dropped to 9')"
        ],
        "length_of_stay": 숫자 (일)
    }},
    "evidence_spans": [
        {{
            "field": "해당 필드명",
            "text_span": "원문에서 발췌한 근거 문장"
        }}
    ]
}}

CRITICAL (Expired):
- 사망이면 `outcome.status="expired"`로 설정
- `discharge_condition`, `discharge_location`은 차트 원문 그대로
- `cause_of_death`와 `critical_events_leading_to_outcome`(타임라인)을 최우선 추출
- `cause_of_death`에 iatrogenic(의인성) 원인이면 반드시 명시 (예: "Iatrogenic hemoperitoneum from paracentesis")

CRITICAL (Procedures):
- 모든 시술(paracentesis, central line, thoracentesis, chest tube, LP, intubation)을 `procedures_performed`에 기록
- technique 필드: "ultrasound-guided", "US-guided", "sterile" 등 수식어가 있으면 기록. 없으면 "not documented"
- "blind", "blindly", "without guidance" 표현이 있으면 technique에 "blind"로 기록하고 safety_flags에 추가
- 시술 후 합병증(bleeding, puncture site bleeding, pneumothorax 등)이 있으면 complications에 기록
"""


def run_chart_structurer(state: Dict) -> Dict:
    """
    차트 구조화 에이전트
    - 원문에서 구조화된 정보 추출 (Information Extraction)
    - 이후 모든 Agent가 이 구조화 데이터를 사용
    """
    patient = state["patient_case"]
    clinical_text = patient.get("clinical_text", "")
    
    print(f"  [Chart Structurer] Input text length: {len(clinical_text)} chars")
    
    if not clinical_text or len(clinical_text) < 50:
        # 텍스트가 없거나 너무 짧으면 기본 구조 반환
        error_msg = f"Insufficient clinical text for structuring (length: {len(clinical_text)})"
        print(f"  [Chart Structurer] [WARN] {error_msg}")
        
        # 기본 구조 반환
        default_structure = {
            "demographics": {"age": None, "sex": None, "chief_complaint": patient.get("diagnosis", "Unknown")},
            "vitals": {},
            "symptoms": {},
            "red_flags": [],
            "physical_exam": {},
            "laboratory": {},
            "imaging": {},
            "interventions_given": {"medications": [], "oxygen_therapy": [], "fluids": None},
            "clinical_course": {},
            "outcome": {"status": "unknown", "discharge_condition": None, "discharge_location": None},
            "evidence_spans": []
        }
        return {"structured_chart": default_structure}
    
    prompt = STRUCTURING_PROMPT.format(clinical_text=clinical_text)
    
    try:
        response = llm.gpt4o(
            prompt=prompt,
            temperature=0.1,  
            max_tokens=4000,  
            json_mode=True  
        )
        
        # JSON 파싱 전 정리
        import json
        import re
        
        # JSON 코드 블록 제거
        response_clean = re.sub(r'```json\s*|\s*```', '', response).strip()
        
        # 잘린 JSON 감지 (마지막이 } 또는 ] 로 끝나지 않으면 잘린 것)
        if not response_clean.endswith('}') and not response_clean.endswith(']'):
            print(f"  [Chart Structurer] [WARN] JSON appears truncated, retrying with more tokens...")
            # 재시도 (max_tokens 더 늘리기)
            response = llm.gpt4o(
                prompt=prompt,
                temperature=0.1,
                max_tokens=6000,  # 더 늘림
                json_mode=True
            )
            response_clean = re.sub(r'```json\s*|\s*```', '', response).strip()
        
        structured = json.loads(response_clean)
        
        print("  [Chart Structurer] [OK] Structured successfully")
        return {"structured_chart": structured}
        
    except json.JSONDecodeError as e:
        print(f"  [Chart Structurer] [ERROR] JSON parsing failed: {e}")
        print(f"  [Chart Structurer] Response sample: {response_clean[:500]}...")
        
        # 기본 구조 반환
        default_structure = {
            "demographics": {"age": None, "sex": None, "chief_complaint": patient.get("diagnosis", "Unknown")},
            "vitals": {},
            "symptoms": {},
            "red_flags": [],
            "physical_exam": {},
            "laboratory": {},
            "imaging": {},
            "interventions_given": {"medications": [], "oxygen_therapy": [], "fluids": None},
            "clinical_course": {},
            "outcome": {"status": "unknown", "discharge_condition": None, "discharge_location": None},
            "evidence_spans": []
        }
        return {"structured_chart": default_structure}
        
    except Exception as e:
        print(f"  [Chart Structurer] [ERROR] Structuring failed: {e}")
        
        # 기본 구조 반환
        default_structure = {
            "demographics": {"age": None, "sex": None, "chief_complaint": patient.get("diagnosis", "Unknown")},
            "vitals": {},
            "symptoms": {},
            "red_flags": [],
            "physical_exam": {},
            "laboratory": {},
            "imaging": {},
            "interventions_given": {"medications": [], "oxygen_therapy": [], "fluids": None},
            "clinical_course": {},
            "outcome": {"status": "unknown", "discharge_condition": None, "discharge_location": None},
            "evidence_spans": []
        }
        return {"structured_chart": default_structure}
