"""
State definition for the Multi-Agent Battery Market Analysis System.
GraphState TypedDict defines the shared state passed between all agents.
"""

from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Shared state for the battery market analysis multi-agent graph.

    Fields:
        messages: Conversation message history (auto-appended by add_messages)
        market_context: Formatted string of global EV/battery market background
        sk_on_data: Dict containing SK On research data with keys:
                    technology, strategy, market, risks, sources
        catl_data: Dict containing CATL research data with keys:
                   technology, strategy, market, risks, sources
        analysis_report: Comparative analysis report in Markdown format
        final_draft: Full final report draft in Markdown format
        critic1_feedback: Feedback from first critic (fairness & grounding check)
        critic1_pass: Whether first critic check passed
        critic1_retry_count: Number of retries for first critic phase
        critic1_retry_target: Which agent to retry: "skon"|"catl"|"both"|"none"
        critic2_feedback: Feedback from second critic (logic & consistency check)
        critic2_pass: Whether second critic check passed
        critic2_retry_count: Number of retries for second critic phase
        current_task: Current task being executed (for logging/debugging)
        error_log: List of error messages encountered during execution
    """
    messages: Annotated[list, add_messages]
    market_context: str
    sk_on_data: Dict[str, Any]
    catl_data: Dict[str, Any]
    analysis_report: str
    final_draft: str
    critic1_feedback: str
    critic1_pass: bool
    critic1_retry_count: int
    critic1_retry_target: str  # "skon"|"catl"|"both"|"none"
    critic2_feedback: str
    critic2_pass: bool
    critic2_retry_count: int
    current_task: str
    error_log: List[str]
