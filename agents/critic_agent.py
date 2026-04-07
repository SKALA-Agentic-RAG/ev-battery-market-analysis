"""T6: 1차(공정성·근거성) / 2차(논리·일관성) / 3차(초안 형식·내용) 검수."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from agents.prompts import CRITIC1_SYSTEM, CRITIC2_SYSTEM, CRITIC3_SYSTEM
from agents.schemas import Critic1Result, Critic2Result, Critic3Result
from config import MAX_RETRIES
from state import GraphState

llm = get_chat_model()
critic1_llm = llm.with_structured_output(Critic1Result)
critic2_llm = llm.with_structured_output(Critic2Result)
critic3_llm = llm.with_structured_output(Critic3Result)


def _normalize_target(raw: str) -> str:
    t = (raw or "none").strip().lower()
    if t in ("skon", "sk_on", "sk-on"):
        return "skon"
    if t in ("catl",):
        return "catl"
    if t in ("both", "all", "collect", "collect_all"):
        return "both"
    return "none"


def critic1_node(state: GraphState) -> dict:
    ctx = {
        "market_context": state.get("market_context", ""),
        "sk_on_data": state.get("sk_on_data", {}),
        "catl_data": state.get("catl_data", {}),
    }
    human = (
        "[수집 데이터]\n"
        + truncate(json.dumps(ctx, ensure_ascii=False, indent=2), 10000)
    )
    verdict: Critic1Result = critic1_llm.invoke(
        [SystemMessage(content=CRITIC1_SYSTEM), HumanMessage(content=human)]
    )
    target = _normalize_target(verdict.retry_target)
    passed = verdict.pass_all
    retry_count = int(state.get("critic1_retry_count") or 0)
    if not passed:
        retry_count += 1
    # 상한 초과 시 통과로 간주하고 다음 단계 진행 (산출물 Fallback)
    if retry_count > MAX_RETRIES:
        passed = True
        target = "none"
        retry_count = MAX_RETRIES

    return {
        "critic1_pass": passed,
        "critic1_feedback": verdict.feedback,
        "critic1_retry_count": retry_count,
        "critic1_retry_target": target,
        "current_task": "T6_critic1",
    }


def critic2_node(state: GraphState) -> dict:
    analysis = state.get("analysis_report", "")
    ctx = truncate(
        json.dumps(
            {
                "market_context": state.get("market_context", ""),
                "sk_on_data": state.get("sk_on_data", {}),
                "catl_data": state.get("catl_data", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        6000,
    )
    human = f"[수집 데이터 요약]\n{ctx}\n\n[분석 초안]\n{truncate(analysis, 8000)}"
    verdict: Critic2Result = critic2_llm.invoke(
        [SystemMessage(content=CRITIC2_SYSTEM), HumanMessage(content=human)]
    )
    return {
        "critic2_pass": verdict.pass_all,
        "critic2_feedback": verdict.feedback,
        "current_task": "T6_critic2",
    }


def critic3_node(state: GraphState) -> dict:
    """보고서 Markdown 초안의 형식·내용 검수 (T7 이전 게이트)."""
    draft = state.get("final_draft", "")
    ref_ctx = {
        "analysis_report_excerpt": truncate(state.get("analysis_report", ""), 6000),
        "market_excerpt": truncate(state.get("market_context", ""), 2000),
    }
    human = (
        "[참고: 분석·시장 요약]\n"
        + json.dumps(ref_ctx, ensure_ascii=False, indent=2)
        + "\n\n[보고서 초안 Markdown]\n"
        + truncate(draft, 12000)
    )
    verdict: Critic3Result = critic3_llm.invoke(
        [SystemMessage(content=CRITIC3_SYSTEM), HumanMessage(content=human)]
    )
    return {
        "critic3_pass": verdict.pass_all,
        "critic3_feedback": verdict.feedback,
        "current_task": "T6_critic3",
    }
