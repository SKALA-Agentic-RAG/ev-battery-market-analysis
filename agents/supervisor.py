"""
Supervisor and Routing Logic for the Multi-Agent Battery Market Analysis System.
Manages workflow flow and centralized routing with unified critic support.
"""

from typing import Literal, Dict, Any
from state import GraphState


def supervisor_node(state: GraphState) -> GraphState:
    """
    Central supervisor node. Logs current state and determines the next step.
    """
    current_task = state.get("current_task", "start")
    print(f"\n[Supervisor] Current Task: {current_task}")
    return state


def supervisor_router(state: GraphState) -> Literal["market", "skon", "catl", "critic", "analysis", "writer", "output", "__end__"]:
    """
    Central routing logic based on state.
    Strictly controls the flow between all agents.
    """
    task = state.get("current_task", "start")
    phase = state.get("critic_phase", "data")
    
    # 1. 초기 시작 -> Market Research
    if task == "start":
        return "market"
    
    # 2. 데이터 수집 단계 (Market -> SK On -> CATL)
    elif task == "market_analysis_complete":
        return "skon"
    elif task == "skon_complete":
        return "catl"
    
    # 3. 데이터 수집 완료 -> Critic (Data Phase)
    elif task == "catl_complete":
        state["critic_phase"] = "data"
        return "critic"
    
    # 4. 검증 결과 처리 (통합)
    elif task == "critic_complete":
        if state.get("critic_pass", False):
            # PASS 시 다음 단계로 진행
            if phase == "data":
                return "analysis"
            elif phase == "analysis":
                return "writer"
            elif phase == "report":
                return "output"
        else:
            # FAIL 시 지정된 타겟으로 리트라이
            retry_target = state.get("critic_retry_target", "market")
            return retry_target # market, skon, catl, analysis, writer 중 하나
            
    # 5. 분석 완료 -> Critic (Analysis Phase)
    elif task == "analysis_complete":
        state["critic_phase"] = "analysis"
        return "critic"
    
    # 6. 작성 완료 -> Critic (Report Phase)
    elif task == "writer_complete":
        state["critic_phase"] = "report"
        return "critic"
    
    # 7. 출력 완료 -> 종료
    elif task == "workflow_complete":
        return "__end__"
        
    return "__end__"


class SupervisorAgent:
    @staticmethod
    def get_node():
        return supervisor_node
