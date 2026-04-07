"""T4: 양사 전략 비교 + Comparative SWOT."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from state import GraphState
from config import MAX_RETRIES

llm = get_chat_model()

_ANALYSIS_SYSTEM = """당신은 배터리 산업 전략 분석가입니다.
오직 제공된 수집 데이터(market_context, sk_on_data, catl_data)만 근거로 작성하세요.
데이터에 없는 수치·날짜·시장점유율은 만들지 말고 "출처 확인 필요"라고 표시하세요.

다음을 Markdown으로 출력하세요:
1) 동일 비교축(기술/시장/생산·투자/다각화)으로 SK On vs CATL 요약 비교
2) Comparative SWOT: 3열 표 (SK On | CATL | 전략적 시사점)
3) 시계열 우위 판정: `단기(1년)`, `중기(1~3년)`, `장기(3년+)` 각각에 대해
   - "우위 기업: SK On | CATL | 경합"
   - 근거 2개 이상(수치/사실 기반)
   - 신뢰도(상/중/하)
4) 종합 판정: 아래 문장을 반드시 포함
   - "종합 판정: 현재 기준 상대적으로 더 유리한 기업은 X다."
   - 단, 전제가 바뀌면 판정이 달라질 수 있는 조건 1~2개 명시
5) 주요 주장 뒤 (source: ...) 로 출처 표기 — 데이터에 있는 URL/문서명만 사용.
"""


def analysis_agent_node(state: GraphState) -> dict:
    updates: dict = {"current_task": "T4_analysis"}

    # critic2_node가 retry_count를 관리 — 여기서 중복 증가 없음
    critic2_retry_count = int(state.get("critic2_retry_count") or 0)

    # 강제 종료: LLM 재호출 한도 초과 시 기존 분석 유지
    if critic2_retry_count >= MAX_RETRIES:
        print(f"[T4 Analysis] LLM 재호출 {MAX_RETRIES}회 도달 → 강제 종료 (기존 분석 유지)")
        existing = state.get("analysis_report", "")
        if existing:
            updates["analysis_report"] = existing
        updates["current_task"] = "T4_analysis_force_exit"
        return updates

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
        [SystemMessage(content=_ANALYSIS_SYSTEM), HumanMessage(content=human)]
    )
    content = out.content if isinstance(out.content, str) else str(out.content)
    updates["analysis_report"] = content
    return updates
