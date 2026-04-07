"""T5: 목차 구조 기반 보고서 초안."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from state import GraphState

llm = get_chat_model()

_WRITER_SYSTEM = """당신은 전략 보고서 작가입니다. 지정 목차에 맞춰 Markdown 보고서 초안을 작성하세요.

규칙:
- 주어진 수집·분석 내용의 사실만 사용. 새 통계·날짜·내부 정보를 invent 하지 마세요.
- SUMMARY, 본문(2~5장), REFERENCE 포함
- 4장 SWOT는 T4 표를 유지하거나 다듬기만 하세요
- REFERENCE는 데이터에 나온 출처만 중복 없이 정리

목차:
1. SUMMARY
2. 배터리 시장 환경 변화
3. 기업별 포트폴리오 다각화 전략 (3.1 SK On, 3.2 CATL)
4. 핵심 전략 비교 및 Comparative SWOT
5. 종합 시사점
6. REFERENCE
"""


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
        [SystemMessage(content=_WRITER_SYSTEM), HumanMessage(content=human)]
    )
    text = out.content if isinstance(out.content, str) else str(out.content)
    updates["final_draft"] = text
    return updates
