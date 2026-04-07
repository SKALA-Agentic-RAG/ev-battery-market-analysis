"""
Writer Agent (T5) for drafting the final structured report.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from state import GraphState
from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, MAX_RETRIES
from agents.utils._extract_utils import CITATION_ID_REGEX


WRITER_PROMPT_TEMPLATE = """당신은 전문 배터리 산업 보고서 작성자입니다.
아래 수집된 시장 데이터, 기업 데이터, 비교 분석 내용을 바탕으로 완성도 높은 보고서를 작성하십시오.

## 보고서 구조 (반드시 아래 구조를 따르십시오)

### 1. SUMMARY
- 첫 문장에 **한 줄 결론(So What)** 제시
- **핵심 메시지** 3-5개를 불릿으로 제시 (각 1-2문장, 수치/사실 근거 포함)
- **의사결정 포인트** 2-3개 제시 (우선순위/리스크/실행 시점 중심)
- 바쁜 임원이 1분 내 핵심 판단이 가능하도록 간결하게 작성
- SUMMARY 전체 분량은 **약 1/2페이지(권장 450~900자, 최대 1,100자)** 이내로 제한

### 2. 배터리 시장 환경 변화
#### 2.1 글로벌 전기차 시장 현황 및 캐즘 분석
- 전기차 시장 성장률 및 현황
- EV 캐즘(Chasm) 현상의 원인과 배경

#### 2.2 배터리 산업 트렌드 및 기술 패러다임 변화
- 주요 배터리 기술 트렌드 (LFP, NCM, 고체배터리 등)
- 원가 경쟁 심화 및 공급망 재편

### 3. 기업별 포트폴리오 다각화 전략
#### 3.1 SK On의 포트폴리오 다각화 전략
3.1.1 핵심 배터리 기술 현황
3.1.2 전기차용 배터리 사업 전략
3.1.3 ESS 및 비EV 시장 진출 현황
3.1.4 글로벌 생산 거점 및 파트너십
3.1.5 재무 현황 및 투자 계획
3.1.6 전기차 캐즘 대응 전략

#### 3.2 CATL의 포트폴리오 다각화 전략
3.2.1 핵심 배터리 기술 현황
3.2.2 전기차용 배터리 사업 전략
3.2.3 ESS 및 비EV 시장 진출 현황
3.2.4 글로벌 생산 거점 및 파트너십
3.2.5 재무 현황 및 투자 계획
3.2.6 전기차 캐즘 대응 전략

### 4. 핵심 전략 비교 및 Comparative SWOT
#### 4.1 핵심 전략 비교 분석
- 기술 포트폴리오 비교
- 시장 접근 전략 비교
- 재무/투자 전략 비교
- 다각화 성숙도 비교

#### 4.2 Comparative SWOT 분석

| 구분 | SK On | CATL |
|------|-------|------|
| **강점 (Strengths)** | (내용) | (내용) |
| **약점 (Weaknesses)** | (내용) | (내용) |
| **기회 (Opportunities)** | (내용) | (내용) |
| **위협 (Threats)** | (내용) | (내용) |

#### 4.3 임원 의사결정 요약표

| 결정 이슈 | 권고안 | 기대효과 | 주요 리스크 | 실행시점 |
|---|---|---|---|---|
| (내용) | (내용) | (내용) | (내용) | (내용) |

#### 4.4 시계열 우위 판정 (단기/중기/장기)

| 구분 | 우위 기업 | 근거 요약 | 신뢰도 |
|---|---|---|---|
| 단기(1년) | SK On/CATL/경합 중 택1 | (근거) | 상/중/하 |
| 중기(1~3년) | SK On/CATL/경합 중 택1 | (근거) | 상/중/하 |
| 장기(3년+) | SK On/CATL/경합 중 택1 | (근거) | 상/중/하 |

### 5. 종합 시사점
#### 5.1 SK On에 대한 전략적 시사점
#### 5.2 CATL의 전략적 시사점
#### 5.3 한국 배터리 산업에 대한 시사점
#### 5.4 향후 배터리 시장 전망

#### 5.5 12개월 실행 로드맵

| 분기 | 핵심 과제 | KPI | 오너 | 상태 |
|---|---|---|---|---|
| 2026 Q3 | (내용) | (내용) | (내용) | (내용) |

#### 5.6 리스크-대응 매트릭스

| 리스크 | 영향도 | 발생가능성 | 조기경보 지표 | 대응책 |
|---|---|---|---|---|
| (내용) | (상/중/하) | (상/중/하) | (내용) | (내용) |

### 6. REFERENCE
- 번호 매기기 형식으로 출처 목록 작성
- URL 포함 명시

---
## 작성에 활용할 데이터

### 시장 환경 데이터
{market_context}

### SK On 데이터
{sk_on_data}

### CATL 데이터
{catl_data}

### 비교 분석 보고서
{analysis_report}

### 인용 가능한 출처 맵 (ID -> 문헌)
{source_catalog}

---
## 작성 지침
1. 위 보고서 구조를 정확히 따르십시오
2. 각 섹션에 실제 데이터와 출처를 인용하십시오
3. 전문적이고 객관적인 어조를 유지하십시오
4. SWOT 표는 반드시 마크다운 표 형식으로 작성하십시오
5. 출처 인용 시 [출처 번호] 형식을 사용하십시오
6. 보고서 전체를 한국어로 작성하십시오
7. 각 섹션 제목은 ## 또는 ### 마크다운 헤딩으로 작성하십시오
8. 평가/해석 문장은 반드시 비교 근거(수치·사실)와 함께 제시하십시오
9. 사실(Actual) / 계획(Planned) / 전망(Forecast)을 구분하여 서술하십시오
10. Tavily 종합 답변은 직접 인용하지 마십시오 (원문 기사/보고서만 인용)
11. SUMMARY는 `한 줄 결론` → `핵심 메시지` → `의사결정 포인트` 순서를 반드시 지키십시오
12. SUMMARY에는 상세 배경 설명보다 경영 의사결정에 필요한 핵심만 압축하십시오
13. SUMMARY가 1,100자를 넘으면 자동으로 감점될 수 있으므로, 문장을 과감히 압축하십시오
14. SUMMARY를 제외한 본문(2~5장)은 충분히 상세히 작성하십시오 (총 6,000자 이상 권장)
15. 2~5장 각 하위 섹션은 원칙적으로 2개 이상 문단으로 설명하고, 단문 불릿 나열만으로 끝내지 마십시오
16. 4.3 / 5.5 / 5.6 표는 반드시 포함하고, 가능한 한 데이터 기반 값으로 채우십시오
17. 보고서 본문에 아래 문장을 반드시 포함하십시오:
   - `단기 우위: ...`
   - `중기 우위: ...`
   - `장기 우위: ...`
   - `종합 판정: 현재 기준 상대적으로 더 유리한 기업은 ...`
18. 판정 문장은 반드시 SK On/CATL 중 누구가 유리한지 명시하고, "둘 다 필요" 같은 모호한 표현만으로 끝내지 마십시오

위 지침에 따라 완성도 높은 보고서를 작성하십시오.
"""


LONGFORM_REWRITE_PROMPT_TEMPLATE = """아래 보고서 초안을 유지하되, 길이와 표 요건을 충족하도록 **재작성**하십시오.

## 재작성 목표
- SUMMARY는 유지/압축하고, 2~5장을 충분히 길고 깊게 확장
- 4.3 임원 의사결정 요약표, 5.5 실행 로드맵, 5.6 리스크-대응 매트릭스 반드시 포함
- 기존 근거와 인용을 유지하면서 설명 밀도를 높임

## 자동 검증 이슈
{issues}

## 추가 작성 규칙
1. SUMMARY 제외 본문 분량을 충분히 늘릴 것 (최소 6,000자 목표)
2. 2~5장 각 하위섹션을 최소 2문단 이상으로 작성
3. 표는 헤더만 두지 말고 최소 2행 이상 채울 것
4. 보고서 구조(1~6장)는 유지
5. `1. SUMMARY`는 절대 생략하지 말고 원문 내용을 실제 텍스트로 그대로 포함할 것
6. `"(원문 유지)"`, `"요약 생략"` 같은 플레이스홀더 문구 사용 금지
7. 4.4 시계열 우위 판정 표와 `단기 우위/중기 우위/장기 우위/종합 판정` 문장을 반드시 포함

## 기존 초안
{draft}
"""


def writer_agent_node(state: GraphState) -> GraphState:
    """
    Writer Agent node: drafts the final structured report.

    Takes all collected data and analysis to produce:
    1. SUMMARY (½ page)
    2. 배터리 시장 환경 변화 (sections 2.1, 2.2)
    3. 기업별 포트폴리오 다각화 전략 (sections 3.1 SK On, 3.2 CATL, each 6 subsections)
    4. 핵심 전략 비교 및 Comparative SWOT (sections 4.1, 4.2 with table)
    5. 종합 시사점 (sections 5.1-5.4)
    6. REFERENCE (numbered with URLs)

    Args:
        state: Current graph state

    Returns:
        Updated state with final_draft populated
    """
    print("[T5 Writer Agent] 최종 보고서 초안 작성 시작...")

    # 강제 종료: LLM 재호출 한도 초과 시 기존 초안 유지
    critic3_retry_count = state.get("critic3_retry_count", 0)
    if critic3_retry_count >= MAX_RETRIES:
        print(f"[T5 Writer Agent] LLM 재호출 {MAX_RETRIES}회 도달 → 강제 종료 (기존 초안 유지)")
        return {
            "final_draft": state.get("final_draft", ""),
            "current_task": "writer_force_exit",
        }

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        # Prepare all data
        market_context = _to_text(state.get("market_context", "시장 데이터 없음"), default="시장 데이터 없음")
        sk_on_data = _ensure_dict(state.get("sk_on_data", {}))
        catl_data = _ensure_dict(state.get("catl_data", {}))
        analysis_report = _to_text(state.get("analysis_report", "분석 보고서 없음"), default="분석 보고서 없음")

        # Format company data
        sk_on_formatted = _format_for_writer(sk_on_data, "SK On")
        catl_formatted = _format_for_writer(catl_data, "CATL")
        source_map = _build_unified_source_map(sk_on_data, catl_data)
        source_catalog = _format_source_catalog(source_map)

        # Collect all sources for references section
        all_sources = _collect_all_sources(sk_on_data, catl_data)

        prompt = WRITER_PROMPT_TEMPLATE.format(
            market_context=market_context[:2000] if len(market_context) > 2000 else market_context,
            sk_on_data=sk_on_formatted[:2500] if len(sk_on_formatted) > 2500 else sk_on_formatted,
            catl_data=catl_formatted[:2500] if len(catl_formatted) > 2500 else catl_formatted,
            analysis_report=analysis_report[:3000] if len(analysis_report) > 3000 else analysis_report,
            source_catalog=source_catalog[:3000] if len(source_catalog) > 3000 else source_catalog,
        )

        # Call LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        final_draft = _to_text(response.content, default="")
        final_draft = _strip_non_report_preface(final_draft)

        original_summary_section = _extract_summary_section(final_draft)

        longform_issues = _longform_quality_issues(final_draft)
        if longform_issues:
            print(f"[T5 Writer Agent] 장문 보강 재작성 시작 ({len(longform_issues)}개 이슈)")
            rewrite_prompt = LONGFORM_REWRITE_PROMPT_TEMPLATE.format(
                issues="\n".join(f"- {x}" for x in longform_issues[:10]),
                draft=final_draft[:12000] if len(final_draft) > 12000 else final_draft,
            )
            rewrite_response = llm.invoke([HumanMessage(content=rewrite_prompt)])
            rewritten = _to_text(rewrite_response.content, default="").strip()
            if rewritten:
                final_draft = _strip_non_report_preface(rewritten)

        # Guardrail: longform 재작성 시 SUMMARY가 플레이스홀더로 바뀌는 경우 원문 복원
        final_draft = _restore_summary_from_original(final_draft, original_summary_section)
        final_draft = _strip_non_report_preface(final_draft)

        final_draft = _normalize_citations_and_references(final_draft, source_map, all_sources)
        final_draft = _compact_summary_to_half_page(final_draft)

        # If LLM forgot to cite, fall back to plain reference list from collected sources
        if "## 6. REFERENCE" not in final_draft and all_sources:
            final_draft += "\n\n## 6. REFERENCE\n\n"
            final_draft += _format_references(all_sources)

        print("[T5 Writer Agent] 보고서 초안 완성")

        return {
            "final_draft": final_draft,
            "current_task": "writer_complete",
        }

    except Exception as e:
        error_msg = f"[T5 Writer Agent] 오류 발생: {str(e)}"
        print(error_msg)

        # Fallback: compile existing data into basic report structure
        fallback_draft = _generate_fallback_draft(state)
        source_map = _build_unified_source_map(
            _ensure_dict(state.get("sk_on_data", {})),
            _ensure_dict(state.get("catl_data", {})),
        )
        all_sources = _collect_all_sources(
            _ensure_dict(state.get("sk_on_data", {})),
            _ensure_dict(state.get("catl_data", {})),
        )
        fallback_draft = _normalize_citations_and_references(fallback_draft, source_map, all_sources)
        fallback_draft = _compact_summary_to_half_page(fallback_draft)

        return {
            "final_draft": fallback_draft,
            "current_task": "writer_error",
            "error_log": _ensure_list(state.get("error_log", [])) + [error_msg],
        }


def _format_for_writer(data: dict, company_name: str) -> str:
    """Format company data for the writer prompt."""
    if not data:
        return f"{company_name} 데이터 없음"

    parts = []
    for key in ["technology", "strategy", "market", "risks"]:
        if key in data and data[key]:
            rendered = _render_section(data[key])
            if rendered:
                parts.append(f"### {company_name} {key}\n{rendered}")

    metrics_by_horizon = data.get("metrics_by_horizon", {})
    if isinstance(metrics_by_horizon, dict):
        horizon_labels = {
            "actual": "사실(Actual)",
            "planned": "계획(Planned)",
            "forecast": "전망(Forecast)",
        }
        horizon_parts = []
        for horizon_key, label in horizon_labels.items():
            rendered = _render_section(metrics_by_horizon.get(horizon_key, []))
            if rendered:
                horizon_parts.append(f"#### {label}\n{rendered}")
        if horizon_parts:
            parts.append(f"### {company_name} 시점 구분\n" + "\n\n".join(horizon_parts))

    if not parts and "raw_content" in data:
        raw = _to_text(data["raw_content"], default="")
        parts.append(raw[:1500] if len(raw) > 1500 else raw)

    return "\n\n".join(parts) if parts else f"{company_name} 데이터 없음"


def _collect_all_sources(sk_on_data: dict, catl_data: dict) -> list:
    """Collect all source URLs from both company datasets."""
    sources = []
    seen_urls = set()

    for data in [sk_on_data, catl_data]:
        for src in _ensure_list(data.get("sources", [])):
            if not isinstance(src, dict):
                continue
            if src.get("type") == "web":
                url = src.get("url", "")
                if url and url not in seen_urls and url != "tavily://answer":
                    seen_urls.add(url)
                    sources.append({
                        "id": src.get("id", ""),
                        "title": src.get("title", "출처"),
                        "url": url,
                    })

    return sources


def _format_references(sources: list) -> str:
    """Format sources as a numbered reference list."""
    refs = []
    for i, src in enumerate(sources, 1):
        title = src.get("title", "출처")
        url = src.get("url", "")
        src_id = src.get("id", "")
        if src_id:
            refs.append(f"{i}. [{i}] {src_id} = {title}. {url}")
        else:
            refs.append(f"{i}. [{i}] {title}. {url}")
    return "\n".join(refs)


def _generate_fallback_draft(state: GraphState) -> str:
    """Generate basic fallback report when LLM call fails."""
    market_context = _to_text(state.get("market_context", ""), default="")
    sk_on_data = _ensure_dict(state.get("sk_on_data", {}))
    catl_data = _ensure_dict(state.get("catl_data", {}))
    analysis_report = _to_text(state.get("analysis_report", ""), default="")

    all_sources = _collect_all_sources(sk_on_data, catl_data)

    draft = """# 전기차 캐즘 환경에서 SK On과 CATL의 포트폴리오 다각화 전략 비교 분석

## 1. SUMMARY

**한 줄 결론:** 전기차 캐즘 국면에서는 SK On의 기술·시장 다각화 가속과 CATL의 원가·규모 우위 대응이 동시에 필요하며, 2026~2028년이 전략 전환의 결정적 구간이다.

**핵심 메시지:**
- 두 기업 모두 전기차 의존도를 낮추기 위해 포트폴리오 다각화를 추진 중이나, 실행 속도와 재무 체력에서 차이가 크다.
- CATL은 글로벌 1위 점유율과 LFP 기반 원가 경쟁력으로 ESS·신규 응용 시장 확장에 유리한 위치다.
- SK On은 NCM·전고체 등 기술 축에서 차별화 잠재력이 있으나, 단기 수익성 및 투자 효율 관리가 성패를 좌우한다.
- 향후 경쟁의 핵심은 기술 우위 자체보다 `원가-생산-시장 접근`을 결합한 실행력으로 이동하고 있다.

**의사결정 포인트:**
1. 2026~2027년 우선순위를 `수익성 방어`와 `비EV 매출 비중 확대` 중 어디에 둘지 명확히 정해야 한다.
2. SK On은 북미/유럽 거점의 가동률과 제품 믹스 최적화, CATL은 지정학 리스크 분산 투자의 속도를 점검해야 한다.
3. 전고체/나트륨이온 등 차세대 기술은 장기 옵션으로 관리하되, 단기 현금흐름을 훼손하지 않는 투자 게이트가 필요하다.

---

## 2. 배터리 시장 환경 변화

### 2.1 글로벌 전기차 시장 현황 및 캐즘 분석

"""
    draft += market_context[:1500] if market_context else "시장 데이터 수집 중\n"

    draft += """

### 2.2 배터리 산업 트렌드 및 기술 패러다임 변화

배터리 산업은 LFP 기술의 확산, 원가 경쟁 심화, ESS 시장 성장 등 주요 트렌드를 경험하고 있습니다.

---

## 3. 기업별 포트폴리오 다각화 전략

### 3.1 SK On의 포트폴리오 다각화 전략

"""
    draft += _render_section(sk_on_data.get("technology", "")) + "\n"
    draft += _render_section(sk_on_data.get("strategy", "")) + "\n"
    draft += _render_section(sk_on_data.get("market", "")) + "\n"
    draft += _render_section(sk_on_data.get("risks", "")) + "\n"

    draft += """

### 3.2 CATL의 포트폴리오 다각화 전략

"""
    draft += _render_section(catl_data.get("technology", "")) + "\n"
    draft += _render_section(catl_data.get("strategy", "")) + "\n"
    draft += _render_section(catl_data.get("market", "")) + "\n"
    draft += _render_section(catl_data.get("risks", "")) + "\n"

    draft += """

---

## 4. 핵심 전략 비교 및 Comparative SWOT

### 4.1 핵심 전략 비교 분석

"""
    draft += analysis_report[:2000] if analysis_report else "분석 데이터 없음\n"

    draft += """

### 4.2 Comparative SWOT 분석

| 구분 | SK On | CATL |
|------|-------|------|
| **강점 (Strengths)** | - NCM 파우치형 고에너지밀도<br>- 북미/유럽 현지 생산<br>- 글로벌 완성차 파트너십 | - 세계 최대 시장점유율<br>- LFP 원가 경쟁력<br>- CTP/Kirin 혁신 기술 |
| **약점 (Weaknesses)** | - 지속적 영업 적자<br>- 상대적 소규모 생산 능력<br>- 제품 다양성 부족 | - 미국 시장 접근 제한<br>- 지정학적 리스크<br>- 중국 시장 의존도 |
| **기회 (Opportunities)** | - ESS 시장 성장<br>- 고체배터리 상용화<br>- 미국 IRA 혜택 활용 | - 글로벌 ESS 시장 확대<br>- 신흥시장 성장<br>- 차세대 배터리 기술 선점 |
| **위협 (Threats)** | - 전기차 캐즘 지속<br>- CATL 등 중국 업체 가격 경쟁<br>- 원자재 가격 변동 | - 미국/유럽 관세 및 제재<br>- 한국/일본 경쟁사 기술 추격<br>- 과잉생산 우려 |

### 4.3 임원 의사결정 요약표

| 결정 이슈 | 권고안 | 기대효과 | 주요 리스크 | 실행시점 |
|---|---|---|---|---|
| SK On 수익성 회복 우선순위 | 북미/유럽 거점 가동률·제품 믹스 동시 개선 | 분기 손익 변동성 축소 | 초기 비용 증가 | 2026 Q3~Q4 |
| CATL 지정학 리스크 대응 | 지역별 생산·고객 포트폴리오 다변화 | 규제 충격 흡수력 제고 | 투자 회수 기간 장기화 | 2026 Q3~2027 Q2 |

### 4.4 시계열 우위 판정 (단기/중기/장기)

| 구분 | 우위 기업 | 근거 요약 | 신뢰도 |
|---|---|---|---|
| 단기(1년) | CATL | LFP 기반 원가 경쟁력과 대규모 출하 우위 | 중 |
| 중기(1~3년) | CATL | 생산·투자 규모 우위와 시장 지배력 지속 가능성 | 중 |
| 장기(3년+) | 경합 | SK On 차세대 기술 잠재력 vs CATL 규모 우위가 병존 | 중 |

단기 우위: CATL
중기 우위: CATL
장기 우위: 경합
종합 판정: 현재 기준 상대적으로 더 유리한 기업은 CATL이다.

---

## 5. 종합 시사점

### 5.1 SK On에 대한 전략적 시사점
- 재무 건전성 회복을 최우선 과제로 설정하고, 원가 경쟁력 강화 필요
- ESS 및 비EV 분야로의 다각화를 통해 캐즘 리스크 분산
- 고체배터리 등 차세대 기술 선점으로 중장기 경쟁력 확보

### 5.2 CATL의 전략적 시사점
- 미중 갈등에 대응한 글로벌 생산 거점 다변화 지속
- 기술 혁신을 통한 프리미엄 시장 진출로 단순 가격 경쟁 탈피
- 비EV 분야(ESS, 선박, 항공) 다각화로 포트폴리오 리스크 분산

### 5.3 한국 배터리 산업에 대한 시사점
- 한국 배터리 업체들은 기술력 고도화와 원가 경쟁력 동시 강화 필요
- 정부 차원의 배터리 산업 지원 정책 강화 요청

### 5.4 향후 배터리 시장 전망
- 전기차 캐즘 이후 ESS 주도 성장 국면으로 전환 예상
- 고체배터리 상용화가 경쟁 구도 재편의 핵심 변수

### 5.5 12개월 실행 로드맵

| 분기 | 핵심 과제 | KPI | 오너 | 상태 |
|---|---|---|---|---|
| 2026 Q3 | 북미 거점 가동률 개선 계획 확정 | 가동률 +5%p | COO | 계획 |
| 2026 Q4 | 비EV 고객 파이프라인 확대 | 신규 고객 2곳 | CSO | 계획 |
| 2027 Q1 | 제품 믹스 고도화 | 고부가 비중 +3%p | 사업부장 | 계획 |

### 5.6 리스크-대응 매트릭스

| 리스크 | 영향도 | 발생가능성 | 조기경보 지표 | 대응책 |
|---|---|---|---|---|
| EV 수요 둔화 장기화 | 상 | 중 | 주문 성장률 둔화 | ESS/비EV 매출 비중 확대 |
| 원자재 가격 급변 | 중 | 중 | 원재료 단가 변동률 | 장기 계약 및 헤지 확대 |
| 지정학 규제 강화 | 상 | 중 | 관세/제재 발표 | 생산 거점·고객 지역 다변화 |

---

## 6. REFERENCE

"""
    if all_sources:
        draft += _format_references(all_sources)
    else:
        draft += "출처 데이터 수집 중"

    return draft


def _ensure_dict(value: Any) -> Dict[str, Any]:
    """Return a dict when possible, otherwise empty dict."""
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> List[Any]:
    """Normalize value to list for safe iteration."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _to_text(value: Any, default: str = "") -> str:
    """Convert arbitrary values to readable text for prompts/reports."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "claim" in value:
            claim = _to_text(value.get("claim"), "").strip()
            metric = _to_text(value.get("value"), "").strip()
            if claim and metric:
                return f"{claim} (값: {metric})"
            return claim or metric or default
        lines = []
        for k, v in value.items():
            text = _to_text(v, "").strip()
            if text:
                lines.append(f"{k}: {text}")
        return "\n".join(lines) if lines else default
    if isinstance(value, list):
        lines = []
        for item in value:
            text = _to_text(item, "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines) if lines else default
    return str(value)


def _render_section(value: Any) -> str:
    """Render a section body from legacy/new data formats."""
    text = _to_text(value, default="")
    return text.strip()


def _build_unified_source_map(sk_on_data: dict, catl_data: dict) -> Dict[str, Dict[str, Any]]:
    """Merge source_map objects from both company datasets."""
    merged: Dict[str, Dict[str, Any]] = {}
    for data in (sk_on_data, catl_data):
        source_map = data.get("source_map", {})
        if not isinstance(source_map, dict):
            continue
        for sid, info in source_map.items():
            if not isinstance(info, dict):
                continue
            if info.get("url") == "tavily://answer":
                continue
            merged[sid] = info
    return merged


def _format_source_catalog(source_map: Dict[str, Dict[str, Any]]) -> str:
    """Render source catalog for prompt visibility."""
    if not source_map:
        return "출처 맵 없음"
    lines = []
    for sid, info in source_map.items():
        title = info.get("title", sid)
        url = info.get("url", "")
        lines.append(f"- {sid}: {title} ({url})")
    return "\n".join(lines)


def _normalize_citations_and_references(
    draft: str,
    source_map: Dict[str, Dict[str, Any]],
    fallback_sources: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Normalize citation style to [1], [2], ... and rebuild REFERENCE section
    from actually cited source IDs.
    """
    text = _to_text(draft, default="").strip()
    if not text:
        return text

    body_only = _strip_reference_section(text)

    citation_pattern = re.compile(
        rf"\[({CITATION_ID_REGEX})\]|(?<![\w/])({CITATION_ID_REGEX})(?![\w-])"
    )

    cited_ids: List[str] = []

    def collect(match: re.Match) -> str:
        sid = match.group(1) or match.group(2)
        if sid in source_map and sid not in cited_ids:
            cited_ids.append(sid)
        return match.group(0)

    citation_pattern.sub(collect, body_only)
    id_to_num = {sid: idx for idx, sid in enumerate(cited_ids, start=1)}

    def replace(match: re.Match) -> str:
        sid = match.group(1) or match.group(2)
        if sid in id_to_num:
            return f"[{id_to_num[sid]}]"
        return match.group(0)

    normalized_body = citation_pattern.sub(replace, body_only)

    ref_lines: List[str] = []
    if cited_ids:
        for sid in cited_ids:
            num = id_to_num[sid]
            info = source_map.get(sid, {})
            title = info.get("title", sid)
            url = info.get("url", "")
            if url and url != "내부 문서":
                ref_lines.append(f"{num}. [{num}] {sid} = {title}. {url}")
            else:
                ref_lines.append(f"{num}. [{num}] {sid} = {title}. (내부 문서)")
    else:
        # LLM이 [[1](URL)] / [1](URL) 형태로만 인용한 경우를 보정
        normalized_body, url_ref_lines = _normalize_numeric_url_citations(
            normalized_body,
            source_map,
            fallback_sources or [],
        )
        ref_lines = url_ref_lines

        # 본문 인용에서 URL을 찾지 못했지만 수집된 출처가 있으면 fallback으로 채움
        if not ref_lines and fallback_sources:
            seen = set()
            for src in fallback_sources:
                if not isinstance(src, dict):
                    continue
                url = _to_text(src.get("url", ""), default="").strip()
                if not url or url in seen or url == "tavily://answer":
                    continue
                seen.add(url)
                title = _to_text(src.get("title", "출처"), default="출처").strip() or "출처"
                ref_lines.append(f"{len(ref_lines)+1}. [{len(ref_lines)+1}] {title}. {url}")

    if not ref_lines:
        ref_lines = ["1. 인용된 출처를 찾지 못했습니다."]

    reference_section = "## 6. REFERENCE\n\n" + "\n".join(ref_lines)
    return _replace_reference_section(normalized_body, reference_section)


def _normalize_numeric_url_citations(
    body: str,
    source_map: Dict[str, Dict[str, Any]],
    fallback_sources: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Normalize numeric markdown URL citations to [1], [2], ...
    and build REFERENCE lines from discovered URLs.
    """
    # Matches [[1](https://...)] and [1](https://...)
    pattern = re.compile(
        r"\[\[(\d+)\]\((https?://[^\s)]+)\)\]|\[(\d+)\]\((https?://[^\s)]+)\)"
    )

    ordered_urls: List[str] = []

    def replace(match: re.Match) -> str:
        url = (match.group(2) or match.group(4) or "").strip()
        if not url:
            return match.group(0)
        if url not in ordered_urls:
            ordered_urls.append(url)
        num = ordered_urls.index(url) + 1
        return f"[{num}]"

    normalized_body = pattern.sub(replace, body)
    if not ordered_urls:
        return normalized_body, []

    title_by_url: Dict[str, str] = {}
    for sid, info in source_map.items():
        if not isinstance(info, dict):
            continue
        url = _to_text(info.get("url", ""), default="").strip()
        if not url:
            continue
        title = _to_text(info.get("title", sid), default=sid).strip() or sid
        title_by_url[url] = title

    for src in fallback_sources:
        if not isinstance(src, dict):
            continue
        url = _to_text(src.get("url", ""), default="").strip()
        if not url:
            continue
        title = _to_text(src.get("title", "출처"), default="출처").strip() or "출처"
        title_by_url.setdefault(url, title)

    ref_lines: List[str] = []
    for i, url in enumerate(ordered_urls, start=1):
        title = title_by_url.get(url, f"출처 {i}")
        ref_lines.append(f"{i}. [{i}] {title}. {url}")

    return normalized_body, ref_lines


def _replace_reference_section(body: str, new_reference_section: str) -> str:
    """Replace existing REFERENCE section or append if absent."""
    pattern = re.compile(r"(?im)^#{2,3}\s*(?:6\.\s*)?REFERENCE\b.*$")
    match = pattern.search(body)
    if not match:
        return body.rstrip() + "\n\n" + new_reference_section + "\n"
    return body[:match.start()].rstrip() + "\n\n" + new_reference_section + "\n"


def _strip_reference_section(text: str) -> str:
    """Return report body before REFERENCE section."""
    pattern = re.compile(r"(?im)^#{2,3}\s*(?:6\.\s*)?REFERENCE\b.*$")
    match = pattern.search(text)
    if not match:
        return text
    return text[:match.start()].rstrip()


def _longform_quality_issues(draft: str) -> List[str]:
    """Check whether non-summary body is sufficiently long and includes required tables."""
    text = _to_text(draft, default="")
    if not text.strip():
        return ["보고서 본문이 비어 있음"]

    body = _strip_reference_section(text)
    sec2 = _extract_numbered_section(body, 2)
    sec3 = _extract_numbered_section(body, 3)
    sec4 = _extract_numbered_section(body, 4)
    sec5 = _extract_numbered_section(body, 5)

    issues: List[str] = []
    non_summary_len = len((sec2 + sec3 + sec4 + sec5).strip())
    if non_summary_len < 6000:
        issues.append(f"SUMMARY 제외 본문 분량 부족 ({non_summary_len}자, 최소 6000자)")
    if len(sec2.strip()) < 900:
        issues.append(f"2장 분량 부족 ({len(sec2.strip())}자)")
    if len(sec3.strip()) < 1700:
        issues.append(f"3장 분량 부족 ({len(sec3.strip())}자)")
    if len(sec4.strip()) < 1200:
        issues.append(f"4장 분량 부족 ({len(sec4.strip())}자)")
    if len(sec5.strip()) < 1200:
        issues.append(f"5장 분량 부족 ({len(sec5.strip())}자)")

    required_markers = [
        "### 4.3 임원 의사결정 요약표",
        "| 결정 이슈 | 권고안 | 기대효과 | 주요 리스크 | 실행시점 |",
        "### 4.4 시계열 우위 판정 (단기/중기/장기)",
        "| 구분 | 우위 기업 | 근거 요약 | 신뢰도 |",
        "### 5.5 12개월 실행 로드맵",
        "| 분기 | 핵심 과제 | KPI | 오너 | 상태 |",
        "### 5.6 리스크-대응 매트릭스",
        "| 리스크 | 영향도 | 발생가능성 | 조기경보 지표 | 대응책 |",
    ]
    for marker in required_markers:
        if marker not in body:
            issues.append(f"필수 표/헤더 누락: {marker}")

    required_judgment_phrases = [
        "단기 우위:",
        "중기 우위:",
        "장기 우위:",
        "종합 판정:",
    ]
    for phrase in required_judgment_phrases:
        if phrase not in body:
            issues.append(f"판정 문구 누락: {phrase}")

    if not re.search(r"종합 판정:\s*현재 기준 상대적으로 더 유리한 기업은\s*(SK On|CATL)", body):
        issues.append("종합 판정에서 우위 기업이 명시되지 않음")

    return issues


def _extract_numbered_section(text: str, section_number: int) -> str:
    """Extract body text for a numbered H2 section like '## 2.'."""
    pattern = re.compile(rf"(?im)^##\s*{section_number}\.\s+")
    start_match = pattern.search(text)
    if not start_match:
        return ""
    start = start_match.start()
    next_h2 = re.search(r"(?im)^##\s+\d+\.\s+", text[start_match.end():])
    end = start_match.end() + next_h2.start() if next_h2 else len(text)
    return text[start:end]


def _extract_summary_section(text: str) -> str:
    """Extract full SUMMARY section including heading."""
    return _extract_numbered_section(text, 1).strip()


def _restore_summary_from_original(draft: str, original_summary_section: str) -> str:
    """
    Restore original SUMMARY when rewritten draft omits it or uses placeholders
    like '(원문 유지)'.
    """
    text = _to_text(draft, default="").strip()
    original = _to_text(original_summary_section, default="").strip()
    if not text or not original:
        return text

    current_summary = _extract_summary_section(text)
    if not _summary_section_is_placeholder_or_missing(current_summary):
        return text

    pattern = re.compile(r"(?im)^##\s*1\.\s*SUMMARY\b.*$")
    match = pattern.search(text)
    if not match:
        # SUMMARY가 완전히 없는 경우, 문서 선두(제목 다음)로 삽입
        h1 = re.search(r"(?m)^#\s+.+$", text)
        if h1:
            insert_pos = h1.end()
            return text[:insert_pos] + "\n\n" + original + "\n\n" + text[insert_pos:].lstrip()
        return original + "\n\n" + text

    start = match.start()
    next_h2 = re.search(r"(?im)^##\s+\d+\.\s+", text[match.end():])
    end = match.end() + next_h2.start() if next_h2 else len(text)
    return text[:start].rstrip() + "\n\n" + original + "\n\n" + text[end:].lstrip()


def _summary_section_is_placeholder_or_missing(summary_section: str) -> bool:
    """Detect placeholder-style summary output from rewrite step."""
    summary = _to_text(summary_section, default="").strip()
    if not summary:
        return True

    # Heading-only or near-empty summary
    body = re.sub(r"(?im)^##\s*1\.\s*SUMMARY\b.*$", "", summary).strip()
    if len(body) < 120:
        return True

    placeholders = [
        "(원문 유지)",
        "원문 유지",
        "요약 생략",
        "summary 생략",
        "동일",
    ]
    lowered = body.lower()
    if any(p.lower() in lowered for p in placeholders):
        return True

    # Missing actual "한 줄 결론" content
    if "한 줄 결론" in body:
        if not re.search(r"한 줄 결론(?:\s*\(.*?\))?\s*:\s*\S+", body):
            lines = [ln.strip() for ln in body.splitlines()]
            for i, ln in enumerate(lines):
                if "한 줄 결론" in ln:
                    nxt = ""
                    for cand in lines[i + 1:]:
                        if cand:
                            nxt = cand
                            break
                    if not nxt or ("핵심 메시지" in nxt) or ("의사결정 포인트" in nxt):
                        return True
                    break

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", body))
    decision_count = len(re.findall(r"(?m)^\s*\d+\.\s+", body))
    if bullet_count < 3:
        return True
    if decision_count < 2:
        return True

    return False


def _strip_non_report_preface(draft: str) -> str:
    """Remove conversational preface before the actual report heading."""
    text = _to_text(draft, default="").strip()
    if not text:
        return text

    lines = text.splitlines()
    start_idx = None
    heading_pat = re.compile(r"^\s*(#\s+|##\s*1\.\s*SUMMARY\b|1\.\s*SUMMARY\b)")

    for i, ln in enumerate(lines):
        if heading_pat.search(ln):
            start_idx = i
            break

    if start_idx is None or start_idx == 0:
        return text

    preface = "\n".join(lines[:start_idx]).strip()
    meta_markers = ["아래는 요청", "요청하신 조건", "재작성 보고서", "원문 유지"]
    if any(m in preface for m in meta_markers):
        return "\n".join(lines[start_idx:]).lstrip()

    return text


def _compact_summary_to_half_page(draft: str, max_chars: int = 1100) -> str:
    """
    Enforce executive summary length to approximately half-page.
    Keeps core structure and trims excessive verbosity.
    """
    text = _to_text(draft, default="").strip()
    if not text:
        return text

    summary_heading = re.search(r"(?im)^##\s*1\.\s*SUMMARY\b.*$", text)
    if not summary_heading:
        return text

    start = summary_heading.end()
    next_h2 = re.search(r"(?im)^##\s+\d+\.", text[start:])
    end = start + next_h2.start() if next_h2 else len(text)
    summary_body = text[start:end].strip()

    if len(summary_body) <= max_chars:
        return text

    lines = [ln.rstrip() for ln in summary_body.splitlines()]
    kept: List[str] = []
    section = ""
    core_msg_count = 0
    decision_count = 0
    so_what_content_count = 0  # "한 줄 결론" 헤딩 다음 줄 내용 카운터

    for raw in lines:
        ln = raw.strip()
        if not ln:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        if "한 줄 결론" in ln:
            section = "so_what"
            so_what_content_count = 0
            kept.append(_trim_line(raw, 180))
            continue
        if "핵심 메시지" in ln:
            section = "core"
            kept.append(raw)
            continue
        if "의사결정 포인트" in ln:
            section = "decision"
            kept.append(raw)
            continue

        # "한 줄 결론"이 별도 헤딩이고 내용이 다음 줄에 있는 경우: 최대 1줄 보존
        if section == "so_what" and so_what_content_count < 1:
            kept.append(_trim_line(raw, 180))
            so_what_content_count += 1
            continue

        if section == "core" and re.match(r"^\s*[-*]\s+", raw):
            if core_msg_count < 4:
                kept.append(_trim_line(raw, 170))
            core_msg_count += 1
            continue

        # 번호 목록(1. 2. 3.)과 불릿(-/*) 모두 의사결정 포인트로 인정
        if section == "decision" and re.match(r"^\s*([-*]|\d+\.)\s+", raw):
            if decision_count < 3:
                kept.append(_trim_line(raw, 170))
            decision_count += 1
            continue

        # Drop free-form long prose outside required blocks

    compact = "\n".join(_squash_blank_lines(kept)).strip()

    # Final hard cap
    if len(compact) > max_chars:
        compact = compact[:max_chars - 1].rstrip() + "…"

    return text[:start] + "\n\n" + compact + "\n\n" + text[end:].lstrip()


def _trim_line(line: str, max_len: int) -> str:
    stripped = line.strip()
    if len(stripped) <= max_len:
        return line
    prefix = ""
    m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)", line)
    if m:
        prefix = m.group(1)
    body = stripped[len(prefix.strip()):].strip() if prefix else stripped
    body = body[: max_len - len(prefix) - 1].rstrip() + "…"
    return f"{prefix}{body}" if prefix else body


def _squash_blank_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    prev_blank = True
    for ln in lines:
        blank = (ln.strip() == "")
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return out
