"""T1: 시장 배경 조사 (Web Search)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import topic_from_state, truncate
from agents.llm import get_chat_model
from state import GraphState
from tools.web_search import web_search

llm = get_chat_model()


def market_agent_node(state: GraphState) -> dict:
    topic = topic_from_state(state)
    queries = [
        f"global EV market growth battery demand 2024 2025 {topic}"[:220],
        "EV market slowdown reasons charging infrastructure",
    ]
    chunks = []
    sources: list[str] = []
    for q in queries:
        for r in web_search(q, max_results=4):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            chunks.append(f"- {title}: {content}")
            if url and url not in sources:
                sources.append(url)
    raw = "\n".join(chunks) if chunks else "(검색 결과 없음)"

    sys = SystemMessage(
        content=(
            "당신은 시장 조사가입니다. 웹 검색 스니펫만 사용해 "
            "글로벌 EV·배터리 시장 배경(성장 둔화, 캐즘, 다각화 필요성)을 "
            "한국어로 8~12문장으로 정리하세요. 끝에 '출처:' 로 URL 목록을 적으세요."
        )
    )
    human = HumanMessage(content=f"주제: {topic}\n\n검색 스니펫:\n{truncate(raw, 12000)}")
    out = llm.invoke([sys, human])
    text = out.content if isinstance(out.content, str) else str(out.content)
    if sources:
        text += "\n\n출처(URL):\n" + "\n".join(f"- {u}" for u in sources[:12])
    return {"market_context": text, "current_task": "T1_market"}
