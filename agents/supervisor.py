"""Supervisor 라우팅: critic1 / critic2 이후 조건부 엣지."""

from __future__ import annotations

from config import MAX_RETRIES
from state import GraphState


class SupervisorAgent:
    """그래프 조건부 라우팅은 함수로 처리. 클래스는 패키지 API 호환용."""

    pass


def route_after_critic1(state: GraphState) -> str:
    """통과 → analysis. 미통과 → 재수집 대상 노드. 재시도 상한 → analysis 강제."""
    if state.get("critic1_pass"):
        return "analysis"
    if int(state.get("critic1_retry_count") or 0) > MAX_RETRIES:
        return "analysis"
    target = (state.get("critic1_retry_target") or "none").lower()
    if target == "skon":
        return "skon"
    if target == "catl":
        return "catl"
    if target == "both":
        return "collect_all"
    return "collect_all"


def route_after_critic2(state: GraphState) -> str:
    """통과 → writer. 미통과이고 재시도 가능 → analysis. 상한 → writer."""
    if state.get("critic2_pass"):
        return "writer"
    if int(state.get("critic2_retry_count") or 0) >= MAX_RETRIES:
        return "writer"
    return "analysis"


def route_after_critic3(state: GraphState) -> str:
    """통과 → output. 미통과이고 재시도 가능 → writer. 상한 → output(강제)."""
    if state.get("critic3_pass"):
        return "output"
    if int(state.get("critic3_retry_count") or 0) >= MAX_RETRIES:
        return "output"
    return "writer"
