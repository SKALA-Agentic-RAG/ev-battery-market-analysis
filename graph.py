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
from agents.output_agent import output_agent_node
from agents.supervisor import route_after_critic1, route_after_critic2, route_after_critic3


def collect_all_node(state: GraphState) -> GraphState:
    """
    Fan-out collection node that sequentially executes Market, SK On, and CATL agents.

    LangGraph does not natively support parallel execution without the Send API,
    so this node calls all three data collection agents sequentially within one node.

    Args:
        state: Current graph state

    Returns:
        Updated state with market_context, sk_on_data, and catl_data all populated
    """
    print("[Collect All] 시장/SK On/CATL 데이터 병렬 수집 시작 (순차 실행)...")

    # Merge a partial-dict result from a sub-agent back into local state
    def _merge(s: dict, result) -> dict:
        if isinstance(result, dict):
            return {**s, **result}
        return s

    # Step 1: Collect market data (skip if already collected on retry)
    if state.get("market_context"):
        print("[Collect All] 시장 데이터 기존 캐시 사용 (재수집 생략)")
    else:
        state = _merge(state, market_agent_node(state))
        print("[Collect All] 시장 데이터 수집 완료")

    # Step 2: Collect SK On data
    state = _merge(state, skon_agent_node(state))
    print("[Collect All] SK On 데이터 수집 완료")

    # Step 3: Collect CATL data
    state = _merge(state, catl_agent_node(state))
    print("[Collect All] CATL 데이터 수집 완료")

    # Return only the data keys (exclude messages to prevent add_messages duplication)
    return {
        "market_context": state.get("market_context", ""),
        "sk_on_data": state.get("sk_on_data", {}),
        "catl_data": state.get("catl_data", {}),
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
        writer → critic3 → output → END

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

    # Writer proceeds to critic3 for format/content review
    workflow.add_edge("writer", "critic3")

    # Conditional routing after critic3
    # - If report passes: go to output
    # - If not: retry writer with feedback
    workflow.add_conditional_edges(
        "critic3",
        route_after_critic3,
        {
            "writer": "writer",
            "output": "output",
        },
    )

    # Output completes the workflow
    workflow.add_edge("output", END)

    # Compile with memory checkpointer for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
