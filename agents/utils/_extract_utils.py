"""
Shared LLM-based structured data extraction utility for SK On / CATL agents.

Goals:
  - Quantitative-first extraction
  - Source-traceable claims
  - Explicit time horizon split (actual / planned / forecast)
"""

import json
import re
from typing import Any, Dict, List, Tuple


SECTION_KEYS = ("technology", "strategy", "market", "risks")
CITATION_ID_REGEX = r"(?:[A-Z0-9_]+-)?SRC-\d+"


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

EXTRACTION_PROMPT = """당신은 배터리 산업 전문 데이터 분석가입니다.
아래 {company} 관련 검색 결과에서 구조화된 팩트 데이터를 추출하십시오.

## 추출 규칙 (반드시 준수)
1. **정량 우선**: 수치(%, GWh, 조원, 억달러, MW 등)가 있는 데이터를 우선 추출
2. **출처 인라인**: 각 claim에 [소스ID] 형식으로 출처를 반드시 명시
3. **팩트만**: 추측·일반론·홍보성 표현 제외. 확인된 사실만 기술
4. **시간축 분류**: 각 claim에 time_horizon을 actual/planned/forecast/unknown 중 하나로 지정
5. **중복 제거**: 동일 내용이 여러 출처에 있으면 신뢰도 높은 출처 1개만 사용

## 출처 목록
{source_list}

## 검색 데이터 (출처 ID 포함)
{tagged_content}

## 출력 형식 (JSON only, 설명 없이)
{{
  "technology": [
    {{"claim": "구체적 기술 내용 [소스ID]", "value": "수치 또는 null", "time_horizon": "actual/planned/forecast/unknown"}}
  ],
  "strategy": [
    {{"claim": "전략/사업 다각화 내용 [소스ID]", "value": "수치 또는 null", "time_horizon": "actual/planned/forecast/unknown"}}
  ],
  "market": [
    {{"claim": "시장/출하량/매출 내용 [소스ID]", "value": "수치 또는 null", "time_horizon": "actual/planned/forecast/unknown"}}
  ],
  "risks": [
    {{"claim": "리스크/한계 내용 [소스ID]", "value": "수치 또는 null", "time_horizon": "actual/planned/forecast/unknown"}}
  ],
  "quantitative_summary": {{
    "market_share": "N% [소스ID] 또는 데이터 부족",
    "production_capacity": "N GWh [소스ID] 또는 데이터 부족",
    "revenue": "N조원/억달러 [소스ID] 또는 데이터 부족",
    "shipments": "N GWh [소스ID] 또는 데이터 부족"
  }},
  "data_quality": "sufficient 또는 insufficient",
  "insufficient_categories": ["해당 없으면 빈 배열"]
}}"""


# ──────────────────────────────────────────────
# Source tagging helpers
# ──────────────────────────────────────────────

def _company_prefix(company: str) -> str:
    """Build deterministic citation prefix from company label."""
    slug = re.sub(r"[^A-Za-z0-9]+", "", company).upper()
    return slug or "GEN"


def build_tagged_content(
    rag_results: List[Dict],
    web_results: List[Dict],
    max_chars_per_source: int = 600,
    source_prefix: str = "GEN",
) -> Tuple[str, List[Dict], Dict[str, Dict]]:
    """
    Assign source IDs and build tagged content blocks.

    Returns:
        tagged_content : single string with [ID] labels
        sources        : list of source dicts
        source_map     : {source_id: {title, url, ...}}
    """
    source_map: Dict[str, Dict] = {}
    sources: List[Dict] = []
    blocks: List[str] = []
    idx = 1

    def next_source_id() -> str:
        nonlocal idx
        sid = f"{source_prefix}-SRC-{idx}"
        idx += 1
        return sid

    for r in rag_results:
        src_id = next_source_id()
        title = r.get("source", "내부 문서")
        url = "내부 문서"
        score = r.get("score", 0.0)
        source_map[src_id] = {"id": src_id, "type": "rag", "title": title, "url": url, "score": score}
        sources.append({"id": src_id, "type": "rag", "title": title, "url": url, "score": score})
        content = (r.get("content", "") or "")[:max_chars_per_source]
        blocks.append(f"[{src_id}] (문서: {title}, 관련도: {score:.2f})\n{content}")

    for r in web_results:
        url = (r.get("url", "") or "").strip()
        if not url or url == "tavily://answer":
            # Tavily synthesized answer is a hint, not a citable primary source.
            continue

        src_id = next_source_id()
        title = r.get("title", "제목 없음")
        domain_score = r.get("domain_score", 1)
        source_tier = r.get("source_tier", "general")
        is_top_tier = bool(r.get("is_top_tier", False))
        published_at = r.get("published_at")
        source_map[src_id] = {
            "id": src_id,
            "type": "web",
            "title": title,
            "url": url,
            "domain_score": domain_score,
            "source_tier": source_tier,
            "is_top_tier": is_top_tier,
            "published_at": published_at,
        }
        sources.append(
            {
                "id": src_id,
                "type": "web",
                "title": title,
                "url": url,
                "domain_score": domain_score,
                "source_tier": source_tier,
                "is_top_tier": is_top_tier,
                "published_at": published_at,
            }
        )
        content = (r.get("content", "") or "")[:max_chars_per_source]
        blocks.append(f"[{src_id}] (웹: {title})\n{content}")

    return "\n\n".join(blocks), sources, source_map


# ──────────────────────────────────────────────
# LLM extraction
# ──────────────────────────────────────────────

def extract_structured_data(
    company: str,
    rag_results: List[Dict],
    web_results: List[Dict],
) -> Dict[str, Any]:
    """
    Use LLM to extract structured, quantitative, source-linked company data.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

    prefix = _company_prefix(company)
    tagged_content, sources, source_map = build_tagged_content(
        rag_results,
        web_results,
        source_prefix=prefix,
    )

    if not tagged_content.strip():
        return _empty_result(company, sources, source_map)

    source_list = "\n".join(
        f"  {sid}: {info.get('title', '출처')} ({info.get('url', '')})"
        for sid, info in source_map.items()
    )

    prompt = EXTRACTION_PROMPT.format(
        company=company,
        source_list=source_list,
        tagged_content=tagged_content[:7000],
    )

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_extracted = _parse_json(response.content)
    except Exception as e:
        print(f"[extract] LLM 추출 실패 ({company}): {e}")
        raw_extracted = _empty_result(company, sources, source_map)

    extracted = _normalize_extracted(raw_extracted, source_map, company)

    cited_srcs = set(_extract_citation_ids(json.dumps(extracted, ensure_ascii=False)))
    if len(cited_srcs) < 2:
        extracted["data_quality"] = "insufficient"
        extracted.setdefault("insufficient_categories", [])
        if "출처 부족" not in extracted["insufficient_categories"]:
            extracted["insufficient_categories"].append("출처 부족")

    extracted["sources"] = sources
    extracted["source_map"] = source_map
    extracted["raw_content"] = tagged_content
    return extracted


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_citation_ids(text: str) -> List[str]:
    return re.findall(CITATION_ID_REGEX, text or "")


def _contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _infer_time_horizon(text: str) -> str:
    t = (text or "").lower()
    planned_keywords = ["계획", "목표", "예정", "착공", "증설", "달성 목표", "by ", "target", "plan", "planned"]
    forecast_keywords = ["전망", "예상", "추정", "forecast", "projected", "outlook"]
    if any(k in t for k in planned_keywords):
        return "planned"
    if any(k in t for k in forecast_keywords):
        return "forecast"
    return "actual"


def _is_top_tier_source(src: Dict[str, Any]) -> bool:
    """
    Decide whether a citation source is top-tier for critical numeric claims.
    - Internal RAG docs are treated as top-tier (curated corpus assumption).
    - Web sources require source_tier=top or domain_score>=3.
    """
    if not isinstance(src, dict):
        return False
    if src.get("type") == "rag":
        return True
    if str(src.get("source_tier", "")).lower() == "top":
        return True
    try:
        return int(src.get("domain_score", 0)) >= 3
    except Exception:
        return False


def _normalize_claim_item(
    item: Any,
    source_map: Dict[str, Dict],
    issues: List[str],
    section: str,
) -> Dict[str, Any]:
    if isinstance(item, dict):
        claim = str(item.get("claim", "")).strip()
        value = item.get("value")
        time_horizon = str(item.get("time_horizon", "")).strip().lower()
    else:
        claim = str(item).strip()
        value = None
        time_horizon = ""

    if not claim:
        claim = "데이터 부족 (출처 없음)"

    citation_ids = _extract_citation_ids(claim)
    if not citation_ids and "데이터 부족" not in claim:
        issues.append(f"{section}: 인용 ID 누락 claim='{claim[:80]}'")

    for sid in citation_ids:
        src = source_map.get(sid)
        if not src:
            issues.append(f"{section}: source_map 미매핑 ID={sid}")
            continue
        if src.get("url") == "tavily://answer":
            issues.append(f"{section}: Tavily answer를 직접 인용함 ID={sid}")

    normalized_horizon = time_horizon if time_horizon in {"actual", "planned", "forecast", "unknown"} else ""
    if not normalized_horizon:
        normalized_horizon = _infer_time_horizon(claim)

    normalized_value = None if value in ("", [], {}) else value
    return {
        "claim": claim,
        "value": normalized_value,
        "time_horizon": normalized_horizon,
    }


def _normalize_quant_summary(
    quant: Dict[str, Any],
    source_map: Dict[str, Dict],
    issues: List[str],
) -> Dict[str, str]:
    keys = ("market_share", "production_capacity", "revenue", "shipments")
    out: Dict[str, str] = {}
    for key in keys:
        text = str(quant.get(key, "데이터 부족")).strip() if isinstance(quant, dict) else "데이터 부족"
        if not text:
            text = "데이터 부족"
        citations = _extract_citation_ids(text)
        if _contains_number(text) and not citations:
            issues.append(f"quantitative_summary.{key}: 정량 수치에 인용 누락")
        for sid in citations:
            if sid not in source_map:
                issues.append(f"quantitative_summary.{key}: source_map 미매핑 ID={sid}")
        out[key] = text
    return out


def _evaluate_metric_evidence(
    quant_summary: Dict[str, str],
    source_map: Dict[str, Dict],
    issues: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate whether key quantitative metrics are supported by top-tier evidence.
    """
    evidence: Dict[str, Dict[str, Any]] = {}
    for key, text in quant_summary.items():
        metric_text = str(text or "").strip()
        citations = _extract_citation_ids(metric_text)
        has_numeric = _contains_number(metric_text) and "데이터 부족" not in metric_text
        has_top = any(_is_top_tier_source(source_map.get(sid, {})) for sid in citations)

        evidence[key] = {
            "has_numeric": has_numeric,
            "citations": citations,
            "has_top_tier_source": has_top,
        }

        if has_numeric and citations and not has_top:
            issues.append(f"quantitative_summary.{key}: 최상위 근거 부족 (SNE/Bloomberg/IEA 등)")

    return evidence


def _build_metrics_by_horizon(extracted: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {"actual": [], "planned": [], "forecast": [], "unknown": []}
    for section in SECTION_KEYS:
        for item in extracted.get(section, []):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim or "데이터 부족" in claim:
                continue
            horizon = str(item.get("time_horizon", "unknown")).lower()
            if horizon not in buckets:
                horizon = "unknown"
            buckets[horizon].append(
                {
                    "section": section,
                    "claim": claim,
                    "value": item.get("value"),
                    "citations": _extract_citation_ids(claim),
                }
            )
    return buckets


def _normalize_extracted(raw: Dict[str, Any], source_map: Dict[str, Dict], company: str) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    issues: List[str] = []
    out: Dict[str, Any] = {}

    for section in SECTION_KEYS:
        items = raw.get(section, [])
        normalized_items: List[Dict[str, Any]] = []

        if isinstance(items, list):
            for item in items:
                normalized_items.append(_normalize_claim_item(item, source_map, issues, section))
        elif isinstance(items, str) and items.strip():
            normalized_items.append(_normalize_claim_item({"claim": items}, source_map, issues, section))

        if not normalized_items:
            normalized_items = [{"claim": f"{company} 데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}]
        out[section] = normalized_items

    out["quantitative_summary"] = _normalize_quant_summary(raw.get("quantitative_summary", {}), source_map, issues)
    out["metric_evidence"] = _evaluate_metric_evidence(out["quantitative_summary"], source_map, issues)
    out["metrics_by_horizon"] = _build_metrics_by_horizon(out)

    # Data quality
    insufficient_categories = raw.get("insufficient_categories", [])
    if not isinstance(insufficient_categories, list):
        insufficient_categories = []
    insufficient_categories = [str(x) for x in insufficient_categories]

    if issues:
        if "근거 추적성 부족" not in insufficient_categories:
            insufficient_categories.append("근거 추적성 부족")

    existing_quality = str(raw.get("data_quality", "insufficient")).lower()
    data_quality = "sufficient" if existing_quality == "sufficient" and not issues else "insufficient"

    out["data_quality"] = data_quality
    out["insufficient_categories"] = insufficient_categories
    out["traceability_issues"] = issues[:20]
    return out


def _parse_json(text: str) -> Dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text or "")
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    print("[extract] JSON 파싱 실패 — 기본 구조 사용")
    return {
        "technology": [{"claim": "데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}],
        "strategy": [{"claim": "데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}],
        "market": [{"claim": "데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}],
        "risks": [{"claim": "데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}],
        "quantitative_summary": {
            "market_share": "데이터 부족",
            "production_capacity": "데이터 부족",
            "revenue": "데이터 부족",
            "shipments": "데이터 부족",
        },
        "data_quality": "insufficient",
        "insufficient_categories": ["파싱 실패"],
    }


def _empty_result(company: str, sources: List[Dict], source_map: Dict[str, Dict]) -> Dict[str, Any]:
    placeholder = [{"claim": f"{company} 데이터 부족 (출처 없음)", "value": None, "time_horizon": "unknown"}]
    return {
        "technology": placeholder,
        "strategy": placeholder,
        "market": placeholder,
        "risks": placeholder,
        "quantitative_summary": {
            "market_share": "데이터 부족",
            "production_capacity": "데이터 부족",
            "revenue": "데이터 부족",
            "shipments": "데이터 부족",
        },
        "metrics_by_horizon": {"actual": [], "planned": [], "forecast": [], "unknown": []},
        "data_quality": "insufficient",
        "insufficient_categories": ["전체 데이터 부족"],
        "traceability_issues": ["원천 데이터가 비어 있음"],
        "sources": sources,
        "source_map": source_map,
        "raw_content": "",
    }


def format_claims_for_report(claims: List[Dict], source_map: Dict[str, Dict]) -> str:
    """
    Convert claim list to readable markdown bullets with inline source links.
    """
    lines = []
    for item in claims:
        claim = str(item.get("claim", ""))
        value = item.get("value")

        def replace_src(match: re.Match) -> str:
            sid = match.group(0)
            info = source_map.get(sid, {})
            title = info.get("title", sid)
            url = info.get("url", "")
            if url and url != "내부 문서":
                return f"[[{title}]({url})]"
            return f"[{title}]"

        linked = re.sub(CITATION_ID_REGEX, replace_src, claim)
        line = f"- {linked}"
        if value:
            line += f"  *(값: {value})*"
        lines.append(line)
    return "\n".join(lines) if lines else "- 데이터 부족 (출처 없음)"
