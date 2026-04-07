"""T5: 목차 구조 기반 보고서 초안."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from agents.prompts import WRITER_SYSTEM
from state import GraphState

llm = get_chat_model()


def writer_agent_node(state: GraphState) -> dict:
    updates: dict = {"current_task": "T5_writer"}

    # 3차 검토(초안) 실패 후 Writer 재진입 시 재시도 카운트
    if state.get("critic3_feedback") and state.get("critic3_pass") is False:
        updates["critic3_retry_count"] = int(state.get("critic3_retry_count") or 0) + 1

    bundle = {
        "market_context": state.get("market_context", ""),
        "sk_on_data": state.get("sk_on_data", {}),
        "catl_data": state.get("catl_data", {}),
        "analysis_report": state.get("analysis_report", ""),
        "critic2_pass": state.get("critic2_pass"),
        "critic2_feedback": state.get("critic2_feedback"),
    }
    prior_draft = (state.get("final_draft") or "").strip()
    c3_fb = state.get("critic3_feedback") or ""

    if prior_draft and c3_fb and state.get("critic3_pass") is False:
        human = (
            "아래는 **이전 Markdown 초안**과 **3차 검토 피드백**입니다. "
            "피드백을 반영해 전체 초안을 다시 출력하세요(동일 목차·형식 유지).\n\n"
            f"[3차 피드백]\n{c3_fb}\n\n[이전 초안]\n{truncate(prior_draft, 14000)}\n\n"
            "[참고 원자료 JSON]\n"
            + truncate(json.dumps(bundle, ensure_ascii=False, indent=2), 8000)
        )
    else:
        human = (
            "다음 JSON을 바탕으로 보고서 초안을 작성하세요.\n\n"
            + truncate(json.dumps(bundle, ensure_ascii=False, indent=2), 12000)
        )
        if not state.get("critic2_pass", False):
            human += (
                "\n\n[주의] 2차 검토 미통과. '검토 미통과 항목' 소절을 넣고 "
                "critic2_feedback을 반영하세요."
            )

    out = llm.invoke(
        [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=human)]
    )
    text = out.content if isinstance(out.content, str) else str(out.content)
    updates["final_draft"] = text
    return updates
