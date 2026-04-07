"""T3: CATL 데이터 수집 (RAG + Web)."""

from __future__ import annotations

import json

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
            "CATL sodium ion battery cost strategy diversification",
            "CATL energy storage ESS revenue",
        ):
            for h in tool.search(q, k=3):
                parts.append(f"[{h.get('source','')}]\n{h.get('content','')}")
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        print(f"[CATL Agent] RAG 보조 실패(무시): {e}")
        return ""


def catl_agent_node(state: GraphState) -> dict:
    topic = topic_from_state(state)
    queries = [
        f"CATL battery strategy diversification sodium ESS global 2024 2025 {topic}"[:220],
        "CATL market share EV battery supply chain vertical integration",
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
            "CATL 담당 분석가입니다. 아래 스니펫만 근거로 한국어 JSON 한 개로만 답하세요. "
            '키: technology, strategy, market, risks — 문자열 값. '
            "sources: 실제 URL·파일명 배열."
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
    return {"catl_data": data, "current_task": "T3_catl"}
