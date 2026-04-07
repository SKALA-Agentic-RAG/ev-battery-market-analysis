"""
LangGraph workflow builder for the Multi-Agent Battery Market Analysis System.
Constructs the StateGraph with all nodes, edges, and conditional routing.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from agents.market_agent import market_agent_node
from agents.skon_agent import skon_agent_node
from agents.catl_agent import catl_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.writer_agent import writer_agent_node
from agents.critic_agent import critic1_node, critic2_node, critic3_node
from agents.output_agent import output_agent_node, output_verify_node
from agents.supervisor import route_after_critic1, route_after_critic2, route_after_critic3


def collect_all_node(state: GraphState) -> dict:
    """
    Market → SK On → CATL 순차 수집. 각 노드는 부분 업데이트 dict를 반환하므로
    여기서 누적 병합하지 않으면 이후 단계에서 상태가 잘립니다.
    """
    print("[Collect All] 시장/SK On/CATL 데이터 수집 시작 (순차 실행)...")
    merged: dict = {**state}

    delta = market_agent_node(merged)
    if delta:
        merged.update(delta)
    print("[Collect All] 시장 데이터 수집 완료")

    delta = skon_agent_node(merged)
    if delta:
        merged.update(delta)
    print("[Collect All] SK On 데이터 수집 완료")

    delta = catl_agent_node(merged)
    if delta:
        merged.update(delta)
    print("[Collect All] CATL 데이터 수집 완료")

    return {
        "market_context": merged.get("market_context", ""),
        "sk_on_data": merged.get("sk_on_data") or {},
        "catl_data": merged.get("catl_data") or {},
        "current_task": "collect_all_complete",
    }


def build_graph():
    """
    Build and compile the LangGraph StateGraph for the battery analysis workflow.

    Graph structure:
        collect_all → critic1
            ↓ (conditional)
        skon / catl / collect_all → critic1  (retry loops)
            ↓ (pass)
        analysis → critic2
            ↓ (conditional)
        analysis  (retry loop)
            ↓ (pass)
        writer → critic3 → (conditional) writer | output → output_verify → END

    Returns:
        Compiled LangGraph application with MemorySaver checkpointer
    """
    workflow = StateGraph(GraphState)

    # Register all nodes
    workflow.add_node("collect_all", collect_all_node)
    workflow.add_node("market", market_agent_node)
    workflow.add_node("skon", skon_agent_node)
    workflow.add_node("catl", catl_agent_node)
    workflow.add_node("critic1", critic1_node)
    workflow.add_node("analysis", analysis_agent_node)
    workflow.add_node("critic2", critic2_node)
    workflow.add_node("writer", writer_agent_node)
    workflow.add_node("critic3", critic3_node)
    workflow.add_node("output", output_agent_node)
    workflow.add_node("output_verify", output_verify_node)

    # Entry point: start with data collection
    workflow.set_entry_point("collect_all")

    # collect_all always proceeds to critic1
    workflow.add_edge("collect_all", "critic1")

    # Conditional routing after critic1
    # - If data passes: go to analysis
    # - If skon needs retry: go to skon node, then back to critic1
    # - If catl needs retry: go to catl node, then back to critic1
    # - If both need retry: go to collect_all again (re-runs all three)
    workflow.add_conditional_edges(
        "critic1",
        route_after_critic1,
        {
            "skon": "skon",
            "catl": "catl",
            "collect_all": "collect_all",
            "analysis": "analysis",
        },
    )

    # After individual skon/catl retry, go back to critic1 for re-evaluation
    workflow.add_edge("skon", "critic1")
    workflow.add_edge("catl", "critic1")

    # Analysis always proceeds to critic2
    workflow.add_edge("analysis", "critic2")

    # Conditional routing after critic2
    # - If analysis passes: go to writer
    # - If not: retry analysis with feedback
    workflow.add_conditional_edges(
        "critic2",
        route_after_critic2,
        {
            "analysis": "analysis",
            "writer": "writer",
        },
    )

    # Writer → 3차 검수(형식·내용) → 재작성 또는 PDF
    workflow.add_edge("writer", "critic3")
    workflow.add_conditional_edges(
        "critic3",
        route_after_critic3,
        {
            "writer": "writer",
            "output": "output",
        },
    )

    workflow.add_edge("output", "output_verify")
    workflow.add_edge("output_verify", END)

    # Compile with memory checkpointer for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
