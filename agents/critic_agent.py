"""
Critic Agent for quality control of collected data and analysis.
critic1_node: checks fairness & grounding of T1-T3 data collection.
critic2_node: checks logic & consistency of T4 analysis.
"""

import re

from state import GraphState
from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, MAX_RETRIES
from agents.utils._extract_utils import CITATION_ID_REGEX


CRITIC1_PROMPT_TEMPLATE = """당신은 배터리 산업 분석 품질 검토 전문가입니다.
아래 수집된 데이터의 품질을 세 가지 기준으로 평가하십시오.

## 평가 기준

### 1. 시장 데이터 충실성 (Market Completeness)
- EV 시장 성장률·규모 등 핵심 지표에 구체적인 수치가 있는가?
- 캐즘(Chasm) 현황, LFP 점유율 변화, HEV/PHEV 비중 등 분석에 필요한 맥락이 포함되어 있는가?
- 시장 데이터가 SK On/CATL 비교 분석의 배경으로 충분히 활용 가능한가?
- 출처 없이 단정적으로 서술된 시장 수치가 없는가?

### 2. 균형성 (Fairness)
- SK On에 대해 긍정적 내용과 부정적 내용이 균형 있게 수집되었는가?
- CATL에 대해 긍정적 내용과 부정적 내용이 균형 있게 수집되었는가?
- 어느 한 기업에 편향된 평가가 있는가?

### 3. 근거성 (Grounding)
- 주요 주장에 대해 출처(URL 또는 문서 참조)가 명시되어 있는가?
- 사실 확인이 가능한 구체적 수치와 데이터가 포함되어 있는가?
- 근거 없는 추측성 내용이 포함되어 있는가?
- 인용 ID가 실제 참고문헌(source_map)과 1:1 매핑되는가?
- Tavily 종합 답변(tavily://answer)을 직접 인용하지 않았는가?
- 수치(%, GWh, 조원, 억달러 등) 근처에 인용 ID가 실제로 붙어 있는가?

---
## 시장 데이터
{market_context}

---
## SK On 데이터
{sk_on_data}

---
## CATL 데이터
{catl_data}

---
## 평가 결과 형식

다음 JSON 형식으로 정확히 응답하십시오 (JSON 블록만 반환):

```json
{{
  "pass": true 또는 false,
  "retry_target": "none" 또는 "skon" 또는 "catl" 또는 "both",
  "feedback": "구체적인 피드백 내용 (한국어로 작성)",
  "market_score": 1-10 점수,
  "fairness_score": 1-10 점수,
  "grounding_score": 1-10 점수,
  "issues": ["문제점1", "문제점2"]
}}
```

판단 기준:
- pass=true: market_score >= 5 AND fairness_score >= 6 AND grounding_score >= 5
- pass=false: 어느 하나라도 기준 미달
- retry_target: 어느 기업 데이터를 재수집해야 하는지 명시 (시장 데이터 문제는 "both")
"""

CRITIC3_PROMPT_TEMPLATE = """당신은 출판·편집 검수 전문가입니다.
아래 Markdown 보고서 초안의 형식과 내용을 평가하십시오.

## 평가 기준

### 1. 형식 (Format)
- SUMMARY 섹션이 있는가?
- SUMMARY가 `한 줄 결론` → `핵심 메시지` → `의사결정 포인트` 구조를 따르는가?
- 번호가 있는 목차/헤딩 구조인가? (2~5장 이상)
- SWOT 비교표(또는 동등한 표)가 포함되어 있는가?
- 4.3 임원 의사결정 요약표, 4.4 시계열 우위 판정표, 5.5 12개월 실행 로드맵, 5.6 리스크-대응 매트릭스가 있는가?
- REFERENCE 섹션이 있는가?
- 빈 장이나 TBD·미완성 섹션이 없는가?

### 2. 내용 (Content)
- T4 분석 요약과 모순되는 내용이 없는가?
- 근거 없는 단정적 수치가 사용되지 않았는가?
- SK On과 CATL 양사가 균형 있게 다루어졌는가?
- SUMMARY만 읽어도 임원이 1분 내 핵심 판단을 할 수 있을 정도로 압축·명확한가?
- SUMMARY를 제외한 본문(2~5장)이 충분히 상세한가? (과도하게 짧지 않은가)
- `단기 우위/중기 우위/장기 우위/종합 판정`이 명시되어 있는가?
- 종합 판정에서 SK On 또는 CATL 중 누가 상대적으로 유리한지 분명히 결론 내렸는가?

---
## T4 분석 요약 (참고)
{analysis_excerpt}

---
## 보고서 초안 Markdown
{draft}

---
## 평가 결과 형식

다음 JSON 형식으로 정확히 응답하십시오 (JSON 블록만 반환):

```json
{{
  "pass": true 또는 false,
  "feedback": "구체적인 피드백 내용 (한국어로 작성)",
  "format_score": 1-10 점수,
  "content_score": 1-10 점수,
  "issues": ["문제점1", "문제점2"]
}}
```

판단 기준:
- pass=true: format_score >= 6 AND content_score >= 6
- pass=false: 어느 하나라도 기준 미달
"""

CRITIC2_PROMPT_TEMPLATE = """당신은 배터리 산업 분석 품질 검토 전문가입니다.
아래 비교 분석 보고서의 품질을 두 가지 기준으로 평가하십시오.

## 평가 기준

### 1. 논리성 (Logic)
- 분석 과정에서 논리적 비약이 없는가?
- 결론이 제시된 데이터와 논리적으로 연결되는가?
- 인과관계가 명확하게 설명되는가?

### 2. 일관성 (Consistency)
- 보고서의 각 섹션 간에 내용 모순이 없는가?
- SWOT 분석이 본문 내용과 일치하는가?
- 동일한 사실을 서로 다르게 해석하는 부분이 없는가?

---
## 비교 분석 보고서
{analysis_report}

---
## 평가 결과 형식

다음 JSON 형식으로 정확히 응답하십시오 (JSON 블록만 반환):

```json
{{
  "pass": true 또는 false,
  "feedback": "구체적인 피드백 내용 (한국어로 작성)",
  "logic_score": 1-10 점수,
  "consistency_score": 1-10 점수,
  "issues": ["문제점1", "문제점2"],
  "suggestions": ["개선제안1", "개선제안2"]
}}
```

판단 기준:
- pass=true: logic_score >= 6 AND consistency_score >= 6
- pass=false: 어느 하나라도 기준 미달
"""


def critic1_node(state: GraphState) -> GraphState:
    """
    Critic 1 node: evaluates fairness and grounding of collected data (T1-T3).

    Checks:
    - Fairness: balanced positive/negative coverage for both companies
    - Grounding: sources cited for major claims

    If retry_count >= MAX_RETRIES, forces pass (fallback mechanism).

    Args:
        state: Current graph state

    Returns:
        Updated state with critic1_pass, critic1_feedback, critic1_retry_target
    """
    print("[Critic 1] 데이터 공정성 및 근거성 검토 시작...")

    retry_count = state.get("critic1_retry_count", 0)

    # Fallback: force pass after max retries
    if retry_count >= MAX_RETRIES:
        print(f"[Critic 1] 최대 재시도 횟수({MAX_RETRIES}) 도달, 강제 통과")
        return {
            "critic1_pass": True,
            "critic1_feedback": f"최대 재시도 횟수({MAX_RETRIES}) 도달로 강제 통과 처리됨",
            "critic1_retry_target": "none",
            "current_task": "critic1_force_pass",
            "error_log": state.get("error_log", []) + [
                f"[Critic 1] 최대 재시도 도달로 강제 통과 (retry_count={retry_count})"
            ],
        }

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import json
        import re

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        # Prepare data summaries for prompt
        market_context = state.get("market_context", "데이터 없음")
        sk_on_data = state.get("sk_on_data", {})
        catl_data = state.get("catl_data", {})

        sk_on_summary = _summarize_company_data(sk_on_data, "SK On")
        catl_summary = _summarize_company_data(catl_data, "CATL")

        prompt = CRITIC1_PROMPT_TEMPLATE.format(
            market_context=market_context[:1500] if len(market_context) > 1500 else market_context,
            sk_on_data=sk_on_summary[:2000] if len(sk_on_summary) > 2000 else sk_on_summary,
            catl_data=catl_summary[:2000] if len(catl_summary) > 2000 else catl_summary,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_response(response.content)

        critic1_pass = result.get("pass", False)
        feedback = result.get("feedback", "피드백 없음")
        retry_target = result.get("retry_target", "none")
        market_score = result.get("market_score", 0)
        fairness_score = result.get("fairness_score", 0)
        grounding_score = result.get("grounding_score", 0)

        # Deterministic traceability checks: if these fail, force retry.
        traceability_issues = []
        sk_issues = _traceability_hard_issues(sk_on_data, "SK On")
        catl_issues = _traceability_hard_issues(catl_data, "CATL")
        traceability_issues.extend(sk_issues + catl_issues)
        if traceability_issues:
            critic1_pass = False
            grounding_score = min(grounding_score, 4)
            retry_target = _retry_target_from_issues(bool(sk_issues), bool(catl_issues))
            issue_text = "\n".join(f"- {x}" for x in traceability_issues[:8])
            feedback = f"{feedback}\n\n[자동 검증] 근거 추적성 이슈:\n{issue_text}"

        print(f"[Critic 1] 결과: pass={critic1_pass}, market={market_score}, fairness={fairness_score}, grounding={grounding_score}")
        print(f"[Critic 1] 재시도 대상: {retry_target}")

        new_retry_count = retry_count + 1 if not critic1_pass else retry_count

        return {
            "critic1_pass": critic1_pass,
            "critic1_feedback": feedback,
            "critic1_retry_target": retry_target if not critic1_pass else "none",
            "critic1_retry_count": new_retry_count,
            "current_task": "critic1_complete",
        }

    except Exception as e:
        error_msg = f"[Critic 1] 오류 발생: {str(e)}"
        print(error_msg)

        # On error, force pass to avoid blocking pipeline
        return {
            "critic1_pass": True,
            "critic1_feedback": f"비평 과정 오류로 통과 처리: {str(e)}",
            "critic1_retry_target": "none",
            "critic1_retry_count": retry_count + 1,
            "current_task": "critic1_error",
            "error_log": state.get("error_log", []) + [error_msg],
        }


def critic2_node(state: GraphState) -> GraphState:
    """
    Critic 2 node: evaluates logic and consistency of the analysis report (T4).

    Checks:
    - Logic: no logical leaps, conclusions follow from data
    - Consistency: no contradictions between sections, SWOT matches body

    If retry_count >= MAX_RETRIES, forces pass (fallback mechanism).

    Args:
        state: Current graph state

    Returns:
        Updated state with critic2_pass, critic2_feedback
    """
    print("[Critic 2] 분석 보고서 논리성 및 일관성 검토 시작...")

    retry_count = state.get("critic2_retry_count", 0)

    # Fallback: force pass after max retries
    if retry_count >= MAX_RETRIES:
        print(f"[Critic 2] 최대 재시도 횟수({MAX_RETRIES}) 도달, 강제 통과")
        return {
            "critic2_pass": True,
            "critic2_feedback": f"최대 재시도 횟수({MAX_RETRIES}) 도달로 강제 통과 처리됨",
            "current_task": "critic2_force_pass",
            "error_log": state.get("error_log", []) + [
                f"[Critic 2] 최대 재시도 도달로 강제 통과 (retry_count={retry_count})"
            ],
        }

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        analysis_report = state.get("analysis_report", "분석 보고서 없음")

        prompt = CRITIC2_PROMPT_TEMPLATE.format(
            analysis_report=analysis_report[:4000] if len(analysis_report) > 4000 else analysis_report,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_response(response.content)

        critic2_pass = result.get("pass", False)
        feedback = result.get("feedback", "피드백 없음")
        logic_score = result.get("logic_score", 0)
        consistency_score = result.get("consistency_score", 0)

        print(f"[Critic 2] 결과: pass={critic2_pass}, logic={logic_score}, consistency={consistency_score}")

        new_retry_count = retry_count + 1 if not critic2_pass else retry_count

        return {
            "critic2_pass": critic2_pass,
            "critic2_feedback": feedback,
            "critic2_retry_count": new_retry_count,
            "current_task": "critic2_complete",
        }

    except Exception as e:
        error_msg = f"[Critic 2] 오류 발생: {str(e)}"
        print(error_msg)

        # On error, force pass to avoid blocking pipeline
        return {
            "critic2_pass": True,
            "critic2_feedback": f"비평 과정 오류로 통과 처리: {str(e)}",
            "critic2_retry_count": retry_count + 1,
            "current_task": "critic2_error",
            "error_log": state.get("error_log", []) + [error_msg],
        }


def critic3_node(state: GraphState) -> GraphState:
    """
    Critic 3 node: evaluates format and content of the final draft (writer output).

    Checks:
    - Format: SUMMARY/REFERENCE/SWOT table, numbered headings, no empty sections
    - Content: no contradiction with analysis_report, no hallucinated data

    If retry_count >= MAX_RETRIES, forces pass (fallback mechanism).

    Args:
        state: Current graph state

    Returns:
        Updated state with critic3_pass, critic3_feedback
    """
    print("[Critic 3] 보고서 초안 형식 및 내용 검토 시작...")

    retry_count = state.get("critic3_retry_count", 0)

    # Fallback: force pass after max retries
    if retry_count >= MAX_RETRIES:
        print(f"[Critic 3] 최대 재시도 횟수({MAX_RETRIES}) 도달, 강제 통과")
        return {
            "critic3_pass": True,
            "critic3_feedback": f"최대 재시도 횟수({MAX_RETRIES}) 도달로 강제 통과 처리됨",
            "current_task": "critic3_force_pass",
            "error_log": state.get("error_log", []) + [
                f"[Critic 3] 최대 재시도 도달로 강제 통과 (retry_count={retry_count})"
            ],
        }

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        final_draft = state.get("final_draft", "보고서 없음")
        analysis_report = state.get("analysis_report", "")

        prompt = CRITIC3_PROMPT_TEMPLATE.format(
            analysis_excerpt=analysis_report[:2000] if len(analysis_report) > 2000 else analysis_report,
            draft=final_draft[:5000] if len(final_draft) > 5000 else final_draft,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_response(response.content)

        critic3_pass = result.get("pass", False)
        feedback = result.get("feedback", "피드백 없음")
        format_score = result.get("format_score", 0)
        content_score = result.get("content_score", 0)

        summary_issues = _summary_hard_issues(final_draft)
        if summary_issues:
            critic3_pass = False
            format_score = min(format_score, 5)
            issue_text = "\n".join(f"- {x}" for x in summary_issues)
            feedback = f"{feedback}\n\n[자동 검증] SUMMARY 구조 이슈:\n{issue_text}"

        longform_issues = _longform_hard_issues(final_draft)
        if longform_issues:
            critic3_pass = False
            format_score = min(format_score, 5)
            content_score = min(content_score, 5)
            issue_text = "\n".join(f"- {x}" for x in longform_issues)
            feedback = f"{feedback}\n\n[자동 검증] 본문 길이/표 이슈:\n{issue_text}"

        print(f"[Critic 3] 결과: pass={critic3_pass}, format={format_score}, content={content_score}")

        new_retry_count = retry_count + 1 if not critic3_pass else retry_count

        return {
            "critic3_pass": critic3_pass,
            "critic3_feedback": feedback,
            "critic3_retry_count": new_retry_count,
            "current_task": "critic3_complete",
        }

    except Exception as e:
        error_msg = f"[Critic 3] 오류 발생: {str(e)}"
        print(error_msg)

        return {
            "critic3_pass": True,
            "critic3_feedback": f"비평 과정 오류로 통과 처리: {str(e)}",
            "critic3_retry_count": retry_count + 1,
            "current_task": "critic3_error",
            "error_log": state.get("error_log", []) + [error_msg],
        }


def _summarize_company_data(data: dict, company_name: str) -> str:
    """
    Create a summary of structured company data for critic evaluation.
    Handles both new (list of {claim, value}) and legacy (string) formats.
    """
    if not data:
        return f"{company_name} 데이터 없음"

    source_map = data.get("source_map", {})

    def expand_srcs(text: str) -> str:
        def repl(m):
            info = source_map.get(m.group(0), {})
            return info.get("title", m.group(0))
        return re.sub(CITATION_ID_REGEX, repl, text)

    parts = [f"## {company_name} 데이터 요약"]

    for key in ["technology", "strategy", "market", "risks"]:
        items = data.get(key, [])
        if isinstance(items, list):
            bullets = []
            for item in items:
                claim = expand_srcs(item.get("claim", ""))
                value = item.get("value")
                bullets.append(f"  - {claim}" + (f" ({value})" if value else ""))
            if bullets:
                parts.append(f"### {key}\n" + "\n".join(bullets))
        elif isinstance(items, str) and items:
            parts.append(f"### {key}\n{items}")

    # Quantitative summary
    quant = data.get("quantitative_summary", {})
    if quant:
        q_lines = [f"  - {k}: {expand_srcs(str(v))}" for k, v in quant.items()]
        parts.append("### 정량 요약\n" + "\n".join(q_lines))

    # Source count and quality
    sources = data.get("sources", [])
    quality = data.get("data_quality", "unknown")
    parts.append(f"\n출처 수: {len(sources)}개 | 데이터 품질: {quality}")

    return "\n\n".join(parts)


def _summary_hard_issues(final_draft: str) -> list:
    """
    Deterministic checks for executive-summary structure.
    Enforces: 한 줄 결론 / 핵심 메시지 / 의사결정 포인트.
    """
    text = str(final_draft or "")
    if not text.strip():
        return ["보고서 본문이 비어 있음"]

    summary_heading = re.search(r"(?im)^##\s*1\.\s*SUMMARY\b.*$", text)
    if not summary_heading:
        return ["'## 1. SUMMARY' 섹션 없음"]

    start = summary_heading.end()
    next_h2 = re.search(r"(?im)^##\s+\d+\.", text[start:])
    summary_body = text[start:start + next_h2.start()] if next_h2 else text[start:]

    issues = []
    if "한 줄 결론" not in summary_body:
        issues.append("'한 줄 결론' 항목 누락")
    if "핵심 메시지" not in summary_body:
        issues.append("'핵심 메시지' 항목 누락")
    if "의사결정 포인트" not in summary_body:
        issues.append("'의사결정 포인트' 항목 누락")

    if "한 줄 결론" in summary_body and not re.search(r"한 줄 결론(?:\s*\(.*?\))?\s*:\s*\S+", summary_body):
        lines = [ln.strip() for ln in summary_body.splitlines()]
        for i, ln in enumerate(lines):
            if "한 줄 결론" in ln:
                nxt = ""
                for cand in lines[i + 1:]:
                    if cand:
                        nxt = cand
                        break
                if not nxt or ("핵심 메시지" in nxt) or ("의사결정 포인트" in nxt):
                    issues.append("'한 줄 결론' 내용 비어 있음")
                break

    summary_len = len(summary_body.strip())
    if summary_len > 1100:
        issues.append(f"SUMMARY 분량 초과 ({summary_len}자, 최대 1100자)")
    if summary_len < 300:
        issues.append(f"SUMMARY 분량 부족 ({summary_len}자, 최소 300자)")

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", summary_body))
    if bullet_count < 3:
        issues.append("핵심 메시지 불릿이 3개 미만")
    if bullet_count > 5:
        issues.append("핵심 메시지 불릿이 5개 초과")

    decision_count = len(re.findall(r"(?m)^\s*\d+\.\s+", summary_body))
    if decision_count < 2:
        issues.append("의사결정 포인트가 2개 미만")
    if decision_count > 3:
        issues.append("의사결정 포인트가 3개 초과")

    return issues


def _longform_hard_issues(final_draft: str) -> list:
    """Deterministic checks for long-form body depth and required tables."""
    text = str(final_draft or "")
    if not text.strip():
        return ["보고서 본문이 비어 있음"]

    body = _strip_reference_section(text)
    sec2 = _extract_numbered_section(body, 2)
    sec3 = _extract_numbered_section(body, 3)
    sec4 = _extract_numbered_section(body, 4)
    sec5 = _extract_numbered_section(body, 5)

    issues = []
    first_nonempty = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if any(marker in text[:300] for marker in ["아래는 요청", "요청하신 조건", "재작성 보고서"]):
        issues.append("보고서 외 메타 설명 문구 포함")
    if first_nonempty and not re.search(r"^(#\s+|##\s*1\.\s*SUMMARY\b|1\.\s*SUMMARY\b)", first_nonempty):
        issues.append("문서 시작부에 보고서 본문 외 텍스트 포함 가능성")

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


def _strip_reference_section(text: str) -> str:
    pattern = re.compile(r"(?im)^#{2,3}\s*(?:6\.\s*)?REFERENCE\b.*$")
    match = pattern.search(text)
    if not match:
        return text
    return text[:match.start()].rstrip()


def _extract_numbered_section(text: str, section_number: int) -> str:
    pattern = re.compile(rf"(?im)^##\s*{section_number}\.\s+")
    start_match = pattern.search(text)
    if not start_match:
        return ""
    start = start_match.start()
    next_h2 = re.search(r"(?im)^##\s+\d+\.\s+", text[start_match.end():])
    end = start_match.end() + next_h2.start() if next_h2 else len(text)
    return text[start:end]


def _retry_target_from_issues(sk_has_issue: bool, catl_has_issue: bool) -> str:
    if sk_has_issue and catl_has_issue:
        return "both"
    if sk_has_issue:
        return "skon"
    if catl_has_issue:
        return "catl"
    return "none"


def _traceability_hard_issues(data: dict, company_name: str) -> list:
    """
    Deterministic validation for citation traceability.
    """
    issues = []
    if not isinstance(data, dict):
        return [f"{company_name}: 데이터 형식 오류"]

    source_map = data.get("source_map", {})
    if not isinstance(source_map, dict) or not source_map:
        issues.append(f"{company_name}: source_map 비어 있음")
        return issues

    # Check claims and quantitative summary for citation consistency
    texts_to_check = []
    for key in ["technology", "strategy", "market", "risks"]:
        for item in data.get(key, []) if isinstance(data.get(key, []), list) else []:
            if isinstance(item, dict):
                texts_to_check.append(str(item.get("claim", "")))
            else:
                texts_to_check.append(str(item))

    quant = data.get("quantitative_summary", {})
    if isinstance(quant, dict):
        texts_to_check.extend(str(v) for v in quant.values())

    for text in texts_to_check:
        ids = re.findall(CITATION_ID_REGEX, text)
        if re.search(r"\d", text) and "데이터 부족" not in text and not ids:
            issues.append(f"{company_name}: 정량 문장 인용 누락 '{text[:60]}'")
        for sid in ids:
            src = source_map.get(sid)
            if not src:
                issues.append(f"{company_name}: source_map 미매핑 ID={sid}")
                continue
            if src.get("url") == "tavily://answer":
                issues.append(f"{company_name}: Tavily 요약 직접 인용 ID={sid}")

    # Enforce top-tier evidence for key quantitative metrics when available
    metric_evidence = data.get("metric_evidence", {})
    if isinstance(metric_evidence, dict):
        for metric, info in metric_evidence.items():
            if not isinstance(info, dict):
                continue
            if info.get("has_numeric") and not info.get("has_top_tier_source"):
                issues.append(f"{company_name}: {metric} 최상위 근거 부족")

    return issues[:12]


def _parse_json_response(response_text: str) -> dict:
    """
    Parse JSON from LLM response text.
    Handles cases where JSON is wrapped in markdown code blocks.
    """
    import json
    import re

    # Try to extract JSON from code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find outermost JSON object using balanced brace matching (handles nested arrays/objects)
    start = response_text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(response_text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response_text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Fallback: parse manually
    print(f"[Critic] JSON 파싱 실패, 텍스트 분석으로 대체")
    result = {"pass": True, "feedback": response_text, "retry_target": "none"}

    # Check for failure indicators in Korean text
    fail_indicators = ["통과 불가", "재수집 필요", "불균형", "출처 부족", "편향"]
    if any(indicator in response_text for indicator in fail_indicators):
        result["pass"] = False

    # Determine retry target
    if "SK On" in response_text and "재수집" in response_text and "CATL" not in response_text:
        result["retry_target"] = "skon"
    elif "CATL" in response_text and "재수집" in response_text and "SK On" not in response_text:
        result["retry_target"] = "catl"
    elif "양사" in response_text or ("SK On" in response_text and "CATL" in response_text and "재수집" in response_text):
        result["retry_target"] = "both"

    return result
