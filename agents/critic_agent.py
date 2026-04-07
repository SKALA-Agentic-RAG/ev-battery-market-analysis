"""T6: 1차(완결·공정·근거) / 2차(논리·일관·정합) / 3차(형식·내용·출처) 검수.

구조적 결함은 LLM 호출 전 preflight 로 차단하고, 나머지는 루브릭·구조화 출력으로 엄밀히 판정합니다.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents._util import truncate
from agents.llm import get_chat_model
from agents.schemas import Critic1Result, Critic2Result, Critic3Result
from config import MAX_RETRIES
from state import GraphState

# --- 임계값 (결정론적 preflight) ---
_MIN_MARKET_CONTEXT_CHARS = 120
_MIN_CORPUS_TEXT_CHARS = 200
_MIN_ANALYSIS_CHARS = 400
_MIN_FINAL_DRAFT_CHARS = 600

llm = get_chat_model()
critic1_llm = llm.with_structured_output(Critic1Result)
critic2_llm = llm.with_structured_output(Critic2Result)
critic3_llm = llm.with_structured_output(Critic3Result)

_CRITIC1_SYSTEM = """당신은 **엄격한 1차 데이터 검수관**입니다. 아래 루브릭을 **모두** 적용해 판정합니다.

## 1) 완결성 (completeness_ok)
- `market_context`: 캐즘·시장 전환 등 **실질 서술**이 있는가. 한두 문장 placeholder면 False.
- `sk_on_data`, `catl_data`: dict이며, 기술/전략/시장/리스크 등 **의미 있는 문자열 값** 또는 구조화 필드가 있는가. 거의 비어 있으면 False.

## 2) 공정성 (fairness_ok)
- 한 기업만 압도적으로 길거나, 다른 기업은 빈약한 **구조적 편향**이 없는가.
- 동일한 비교축(기술·시장·리스크 등)이 양사에 **대칭적으로** 적용 가능한가.

## 3) 근거성 (grounding_ok)
- 정량·시장점유·매출 등 **검증 필요 주장**에 `sources` 또는 동등한 출처 필드가 있는가.
- 출처가 있어도 본문 주장과 **무관한 URL 나열**이면 False.

## 출력 규칙
- 각 bool은 **보수적으로**: 애매하면 False.
- `rationale_brief`: 어떤 키/문단을 근거로 판정했는지 2~4문장.
- `feedback`: 미통과 시 **번호 목록**으로 재수집 지시. 통과 시 한 줄.
- `retry_target`: 미통과 시만 `"skon"|"catl"|"both"` 중 **가장 비용 대비 효과적인** 하나. 통과 시 `"none"`.
"""

_CRITIC2_SYSTEM = """당신은 **전략 보고서용 2차 논리 검수관**입니다. 입력은 (1) 수집 원자료 JSON 요약 (2) T4 분석 Markdown 입니다.

## 1) 논리성 (logic_ok)
- 주장 → 근거 흐름이 성립하는가. **비약**(근거 없는 도약)이 있으면 False.
- 상관관계를 인과로 단정하면 False.

## 2) 내부 일관성 (consistency_ok)
- 앞 문단과 뒤 문단이 **같은 사실**에 대해 모순되지 않는가.
- SWOT/비교표와 본문 서술이 **충돌**하면 False.

## 3) 데이터 정합 (data_alignment_ok)
- 분석문에 나온 **구체 사실·수치·고유명사**가 수집 JSON/시장 맥락에 **근거가 없으면** False (환각 가능).
- 수집에 없는 새로운 통계를 단정하면 False.

## 출력 규칙
- 애매하면 해당 축은 False.
- `rationale_brief`: 문제가 되는 **문장 또는 표현**을 가리키며 2~4문장.
- `feedback`: 미통과 시 Analysis가 수정할 **번호 목록**. 통과 시 한 줄.
"""

_CRITIC3_SYSTEM = """당신은 **출판 품질 3차 검수관**입니다. Writer의 Markdown **최종 초안**을 검토합니다.

## 1) 형식 (format_ok)
- **SUMMARY** (또는 동등 한글 요약 섹션) 존재.
- **다단계 헤딩**으로 목차 구조가 보이는가 (# / ## 등).
- **Comparative SWOT**: 표(table) 또는 동등한 3열 구조.
- **REFERENCE** (또는 참고문헌) 섹션 존재.

## 2) 내용 (content_ok)
- "추가 예정", "TBD", 비어 있는 장만 있는 섹션 없음.
- T4 분석·수집 요약과 **명백히 모순**되는 단정 없음.

## 3) 출처 무결 (reference_integrity_ok)
- REFERENCE·본문의 출처가 **수집 데이터에 등장한 URL/문서** 범위를 벗어나지 않는가.
- 존재하지 않는 보고서명·링크를 지어내면 False.

## 출력 규칙
- 형식·내용·출처 중 하나라도 의심스러우면 해당 축 False.
- `rationale_brief` 2~4문장, `feedback`은 미통과 시 번호 목록.
"""


def _normalize_target(raw: str) -> str:
    t = (raw or "none").strip().lower()
    if t in ("skon", "sk_on", "sk-on"):
        return "skon"
    if t in ("catl",):
        return "catl"
    if t in ("both", "all", "collect", "collect_all"):
        return "both"
    return "none"


def _corp_text_volume(d: Any) -> int:
    if not isinstance(d, dict):
        return 0
    n = 0
    for v in d.values():
        if isinstance(v, str):
            n += len(v.strip())
        elif isinstance(v, list):
            n += sum(len(str(x)) for x in v)
    return n


def _corp_has_sources(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    src = d.get("sources")
    return isinstance(src, list) and len(src) > 0


def _dataset_health(
    market_context: str, sk_on_data: Any, catl_data: Any
) -> tuple[bool, bool, bool, list[str]]:
    """(market_ok, sk_ok, catl_ok, issue_lines)"""
    mc = (market_context or "").strip()
    sk = sk_on_data if isinstance(sk_on_data, dict) else {}
    ct = catl_data if isinstance(catl_data, dict) else {}
    issues: list[str] = []

    m_ok = len(mc) >= _MIN_MARKET_CONTEXT_CHARS
    if not m_ok:
        issues.append(
            f"시장 맥락이 {_MIN_MARKET_CONTEXT_CHARS}자 미만이거나 비어 있음 — 시장·전체 수집 재검토"
        )

    sk_vol = _corp_text_volume(sk)
    ct_vol = _corp_text_volume(ct)
    sk_ok = sk_vol >= _MIN_CORPUS_TEXT_CHARS or _corp_has_sources(sk)
    ct_ok = ct_vol >= _MIN_CORPUS_TEXT_CHARS or _corp_has_sources(ct)

    if not sk_ok:
        issues.append(
            f"SK On 데이터 본문이 {_MIN_CORPUS_TEXT_CHARS}자 미만이고 sources도 부족 — SK On 재수집"
        )
    if not ct_ok:
        issues.append(
            f"CATL 데이터 본문이 {_MIN_CORPUS_TEXT_CHARS}자 미만이고 sources도 부족 — CATL 재수집"
        )

    return m_ok, sk_ok, ct_ok, issues


def _critic1_retry_for_health(m_ok: bool, sk_ok: bool, ct_ok: bool) -> str:
    if not m_ok:
        return "both"
    if not sk_ok and not ct_ok:
        return "both"
    if not sk_ok:
        return "skon"
    if not ct_ok:
        return "catl"
    return "none"


def _finalize_critic1(
    state: GraphState,
    *,
    passed: bool,
    feedback: str,
    raw_target: str,
) -> dict:
    target = _normalize_target(raw_target if not passed else "none")
    retry_count = int(state.get("critic1_retry_count") or 0)
    if not passed:
        retry_count += 1
    if retry_count > MAX_RETRIES:
        passed = True
        target = "none"
        retry_count = MAX_RETRIES
        feedback = (
            f"{feedback}\n\n[시스템] 1차 검수 재시도 상한 도달 — 다음 단계로 진행합니다. "
            "수동 검토 권장."
        )
    return {
        "critic1_pass": passed,
        "critic1_feedback": feedback.strip(),
        "critic1_retry_count": retry_count,
        "critic1_retry_target": target,
        "current_task": "T6_critic1",
    }


def _preflight_critic1(state: GraphState) -> dict | None:
    m_ok, sk_ok, ct_ok, issues = _dataset_health(
        state.get("market_context") or "",
        state.get("sk_on_data"),
        state.get("catl_data"),
    )
    if m_ok and sk_ok and ct_ok:
        return None
    fb = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(issues))
    tgt = _critic1_retry_for_health(m_ok, sk_ok, ct_ok)
    return _finalize_critic1(state, passed=False, feedback=fb, raw_target=tgt)


def critic1_node(state: GraphState) -> dict:
    pf = _preflight_critic1(state)
    if pf is not None:
        return pf

    ctx = {
        "market_context": state.get("market_context", ""),
        "sk_on_data": state.get("sk_on_data", {}),
        "catl_data": state.get("catl_data", {}),
    }
    human = (
        "### 검수 대상 (JSON)\n"
        + truncate(json.dumps(ctx, ensure_ascii=False, indent=2), 12000)
        + "\n\n### 지시\n"
        "위 루브릭대로 completeness_ok, fairness_ok, grounding_ok를 각각 판정하고 "
        "retry_target을 선택하세요. 통과 시 retry_target은 반드시 \"none\"."
    )
    verdict: Critic1Result = critic1_llm.invoke(
        [SystemMessage(content=_CRITIC1_SYSTEM), HumanMessage(content=human)]
    )
    tgt = _normalize_target(verdict.retry_target)
    if verdict.pass_all:
        tgt = "none"
    fb = (verdict.feedback or "").strip() or (verdict.rationale_brief or "").strip()
    if verdict.pass_all and not fb:
        fb = "1차 검수 통과."
    elif not verdict.pass_all and not fb:
        fb = "1차 검수 미통과. rationale_brief 및 수집 데이터를 재검토하세요."
    return _finalize_critic1(
        state,
        passed=verdict.pass_all,
        feedback=fb,
        raw_target=tgt,
    )


def _preflight_critic2(state: GraphState) -> dict | None:
    analysis = (state.get("analysis_report") or "").strip()
    if len(analysis) >= _MIN_ANALYSIS_CHARS:
        return None
    fb = (
        f"1. 분석문이 비어 있거나 과도하게 짧습니다 (권장 최소 {_MIN_ANALYSIS_CHARS}자 이상).\n"
        "2. T4 Analysis 에이전트에서 비교축·SWOT·근거를 보강한 전체 Markdown을 다시 작성하세요."
    )
    return {
        "critic2_pass": False,
        "critic2_feedback": fb,
        "current_task": "T6_critic2_preflight",
    }


def critic2_node(state: GraphState) -> dict:
    pf = _preflight_critic2(state)
    if pf is not None:
        return pf

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
        8000,
    )
    human = (
        "### A. 수집·시장 원자료 (요약)\n"
        f"{ctx}\n\n"
        "### B. T4 분석 Markdown\n"
        f"{truncate(analysis, 10000)}\n\n"
        "### 지시\n"
        "A를 **근거 집합**으로 보고 B만 검토하세요. "
        "logic_ok / consistency_ok / data_alignment_ok 를 각각 독립 판정하세요."
    )
    verdict: Critic2Result = critic2_llm.invoke(
        [SystemMessage(content=_CRITIC2_SYSTEM), HumanMessage(content=human)]
    )
    return {
        "critic2_pass": verdict.pass_all,
        "critic2_feedback": verdict.feedback
        or verdict.rationale_brief
        or "(피드백 없음 — rationale_brief 확인)",
        "current_task": "T6_critic2",
    }


def _draft_format_signals(md: str) -> dict[str, bool]:
    """경량 휴리스틱: LLM 판정 보조용 힌트 (강제 판정 아님)."""
    lower = md.lower()
    has_summary = bool(
        re.search(r"(^|\n)#+\s*(summary|요약|executive)", md, re.I | re.M)
        or "summary" in lower[:800]
    )
    has_ref = bool(
        re.search(r"(^|\n)#+\s*(reference|references|참고|참고문헌)", md, re.I | re.M)
    )
    has_table = "|" in md and "\n|" in md
    heading_count = len(re.findall(r"(?m)^#+\s+\S+", md))
    return {
        "hint_summary_heading": has_summary,
        "hint_reference_heading": has_ref,
        "hint_markdown_table": has_table,
        "hint_heading_lines": heading_count >= 4,
    }


def _preflight_critic3(state: GraphState) -> dict | None:
    draft = (state.get("final_draft") or "").strip()
    if len(draft) >= _MIN_FINAL_DRAFT_CHARS:
        return None
    fb = (
        f"1. 최종 초안이 비어 있거나 너무 짧습니다 (권장 {_MIN_FINAL_DRAFT_CHARS}자 이상).\n"
        "2. Writer가 목차 전체를 채운 Markdown을 다시 생성하세요."
    )
    return {
        "critic3_pass": False,
        "critic3_feedback": fb,
        "current_task": "T6_critic3_preflight",
    }


def critic3_node(state: GraphState) -> dict:
    pf = _preflight_critic3(state)
    if pf is not None:
        return pf

    draft = state.get("final_draft", "")
    ref_ctx = {
        "analysis_report_excerpt": truncate(state.get("analysis_report", ""), 7000),
        "market_excerpt": truncate(state.get("market_context", ""), 2500),
        "sk_sources_excerpt": truncate(
            json.dumps(
                (state.get("sk_on_data") or {}).get("sources", []),
                ensure_ascii=False,
            ),
            1500,
        ),
        "catl_sources_excerpt": truncate(
            json.dumps(
                (state.get("catl_data") or {}).get("sources", []),
                ensure_ascii=False,
            ),
            1500,
        ),
        "format_heuristics": _draft_format_signals(draft),
    }
    human = (
        "### 참고: 분석·시장·출처 목록 (발췌)\n"
        + json.dumps(ref_ctx, ensure_ascii=False, indent=2)
        + "\n\n### 보고서 초안 (Markdown)\n"
        + truncate(draft, 14000)
        + "\n\n### 지시\n"
        "format_ok / content_ok / reference_integrity_ok 를 각각 독립 판정하세요. "
        "format_heuristics 는 참고용이며, 최종 판단은 본문 전체에 대해 하세요."
    )
    verdict: Critic3Result = critic3_llm.invoke(
        [SystemMessage(content=_CRITIC3_SYSTEM), HumanMessage(content=human)]
    )
    return {
        "critic3_pass": verdict.pass_all,
        "critic3_feedback": verdict.feedback
        or verdict.rationale_brief
        or "(피드백 없음 — rationale_brief 확인)",
        "current_task": "T6_critic3",
    }
