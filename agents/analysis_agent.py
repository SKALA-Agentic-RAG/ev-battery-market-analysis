"""T4: 양사 전략 비교 + Comparative SWOT."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from agents.prompts import ANALYSIS_SYSTEM
from state import GraphState

llm = get_chat_model()


def analysis_agent_node(state: GraphState) -> dict:
    updates: dict = {"current_task": "T4_analysis"}

    # 2차 검토 실패 후 재진입 시 재시도 카운트 증가 (설계: 최대 2회)
    if state.get("critic2_feedback") and not state.get("critic2_pass"):
        updates["critic2_retry_count"] = int(state.get("critic2_retry_count") or 0) + 1

    payload = {
        "market_context": state.get("market_context", ""),
        "sk_on_data": state.get("sk_on_data", {}),
        "catl_data": state.get("catl_data", {}),
        "prior_feedback": state.get("critic2_feedback") or None,
    }
    human = (
        "아래 JSON을 바탕으로 분석을 작성하세요.\n\n"
        + truncate(json.dumps(payload, ensure_ascii=False, indent=2), 8000)
    )
    if payload["prior_feedback"]:
        human += (
            "\n\n[이전 2차 검토 피드백 — 반드시 반영하여 수정]\n"
            + str(payload["prior_feedback"])
        )

    out = llm.invoke(
        [SystemMessage(content=ANALYSIS_SYSTEM), HumanMessage(content=human)]
    )
    content = out.content if isinstance(out.content, str) else str(out.content)
    updates["analysis_report"] = content
    return updates
