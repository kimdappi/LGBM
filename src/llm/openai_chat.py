from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class OpenAIChatConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 2000
    timeout_s: int = 120
    max_retries: int = 4


class OpenAIChatError(RuntimeError):
    pass


def _default_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "") or ""


def call_openai_chat_completions(
    *,
    messages: List[Dict[str, str]],
    config: OpenAIChatConfig,
    api_key: Optional[str] = None,
    api_url: str = "https://api.openai.com/v1/chat/completions",
) -> str:
    key = api_key if api_key is not None else _default_api_key()
    if not key:
        raise OpenAIChatError("OPENAI_API_KEY가 설정되지 않았습니다.")

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for attempt in range(1, config.max_retries + 1):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=config.timeout_s,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < config.max_retries:
                    time.sleep(min(2**attempt, 20))
                    continue
            if resp.status_code == 401:
                raise OpenAIChatError("API 키가 유효하지 않습니다. OPENAI_API_KEY를 확인하세요.")
            if resp.status_code == 404:
                raise OpenAIChatError(f"모델을 찾을 수 없습니다: {config.model}")
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                raise OpenAIChatError("OpenAI 응답 content가 비어있습니다.")
            return str(content).strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < config.max_retries:
                time.sleep(min(attempt * 2, 10))
                continue
            raise OpenAIChatError(f"네트워크/타임아웃 오류: {type(e).__name__}: {e}") from e
        except requests.exceptions.HTTPError as e:
            if attempt < config.max_retries:
                time.sleep(min(attempt * 2, 10))
                continue
            raise OpenAIChatError(f"HTTP 오류: {e}") from e
        except json.JSONDecodeError as e:
            raise OpenAIChatError(f"응답 JSON 파싱 실패: {e}") from e
        except OpenAIChatError:
            raise
        except Exception as e:
            if attempt < config.max_retries:
                time.sleep(min(attempt * 2, 10))
                continue
            raise OpenAIChatError(f"OpenAI 호출 실패: {type(e).__name__}: {e}") from e
    raise OpenAIChatError("OpenAI 호출 최종 실패")


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1]) if isinstance(json.loads(s[start : end + 1]), dict) else None
            except Exception:
                return None
        return None
