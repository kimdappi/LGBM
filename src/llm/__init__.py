"""
LLM clients and utilities.

- openai_chat: Critic Agent 전용 (Router, CritiqueBuilder, Feedback, Verifier, tools).
  OpenAIChatConfig, call_openai_chat_completions, safe_json_loads 사용.
- 그래프 노드(Chart Structurer, Diagnosis/Treatment 등)는 src.agents.llm.get_llm() 사용.
"""
