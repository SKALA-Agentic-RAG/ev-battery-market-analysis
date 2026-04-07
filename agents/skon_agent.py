"""T2: SK On 데이터 수집 (RAG + Web)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import topic_from_state, truncate
from agents.llm import get_chat_model
from state import GraphState
from tools.web_search import web_search

llm = get_chat_model()


def _rag_context() -> str:
    try:
        from tools.rag import RAGTool

        tool = RAGTool()
        if not tool.load_index():
            tool.index_documents()
        parts: list[str] = []
        for q in (
            "SK On battery LFP NCM diversification ESS",
            "SK On North America EV battery supply",
        ):
            for h in tool.search(q, k=3):
                parts.append(f"[{h.get('source','')}]\n{h.get('content','')}")
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        print(f"[SKOn Agent] RAG 보조 실패(무시): {e}")
        return ""


def skon_agent_node(state: GraphState) -> dict:
    topic = topic_from_state(state)
    queries = [
        f"SK On battery business strategy diversification ESS 2024 2025 {topic}"[:220],
        "SK On EV battery production investment North America",
    ]
    chunks: list[str] = []
    sources: list[str] = []
    for q in queries:
        for r in web_search(q, max_results=5):
            url = r.get("url", "")
            if url and url not in sources:
                sources.append(url)
            chunks.append(f"- {r.get('title','')}: {r.get('content','')}")

    rag = _rag_context()
    if rag:
        chunks.append("\n[RAG 인용]\n" + truncate(rag, 4000))

    raw = "\n".join(chunks) if chunks else "(검색·RAG 결과 없음)"

    sys = SystemMessage(
        content=(
            "SK On 담당 분석가입니다. 아래 스니펫만 근거로 한국어 JSON 한 개로만 답하세요. "
            '키: technology, strategy, market, risks — 문자열 값. '
            '그리고 sources: 검색/RAG에 실제로 나온 URL·파일명 문자열 배열.'
        )
    )
    human = HumanMessage(
        content=f"주제: {topic}\n\n자료:\n{truncate(raw, 14000)}\n\nJSON만 출력."
    )
    out = llm.invoke([sys, human])
    text = out.content if isinstance(out.content, str) else str(out.content)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()

    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "technology": text[:2000],
            "strategy": "",
            "market": "",
            "risks": "",
            "sources": sources,
        }
    if isinstance(data, dict) and sources and not data.get("sources"):
        data["sources"] = sources[:20]
    return {"sk_on_data": data, "current_task": "T2_skon"}
