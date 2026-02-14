"""경로 변경 후 import/상호참조 검증 (실행: python scripts/check_imports.py)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    # 1) critic 패키지 (critic -> src.llm 경로)
    from src.critic.critic_graph import get_critic_graph
    from src.critic.verifier import Verifier
    from src.critic.router import LLMRouter
    from src.critic.critique_builder import CritiqueBuilder
    print("critic -> ..llm, .types, .runner: OK")

    # 2) pipeline state/adapter (MedicalCritiqueGraph 부르지 않으므로 agents 미로드)
    from src.pipeline import AgentState, clean_state_to_agent_state, agent_state_to_clean_updates
    print("pipeline state/adapter: OK")

    # 3) pipeline.graph -> src.agents + src.critic (agents 로드 시 evidence_agent는 Bio 사용)
    try:
        from src.pipeline import MedicalCritiqueGraph
        print("pipeline.graph -> src.agents, src.critic: OK")
    except ModuleNotFoundError as e:
        if "Bio" in str(e):
            print("pipeline.graph -> src.agents, src.critic: 경로 OK (Bio 미설치로 agents 일부 스킵)")
        else:
            raise

    print("\n=== 경로/상호참조 정상 ===")

if __name__ == "__main__":
    main()
