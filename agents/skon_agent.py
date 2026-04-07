"""
SK On Agent (T2): SK On 배터리 전략 데이터 수집.

변경 사항:
  - _extract_section (키워드 필터) 제거
  - LLM 기반 구조화 추출(_extract_utils) 적용
  - 각 claim에 출처 URL 인라인 연결
  - RAG score 임계값 필터 적용
  - 소스 수 < 2 시 재검색 트리거
"""

import json
import re
from typing import Any, Dict, List

from state import GraphState
from tools.web_search import web_search
from tools.rag import RAGTool
from config import DOCS_PATH, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from agents.utils._extract_utils import extract_structured_data

# ──────────────────────────────────────────────
# Search queries
# ──────────────────────────────────────────────

SKON_PRIMARY_QUERIES = [
    "SK On 배터리 기술 포트폴리오 NCM 파우치형 에너지밀도 GWh 2024",
    "SK On 포트폴리오 다각화 전략 ESS 에너지저장장치 비EV 매출",
    "SK On 생산 능력 기가팩토리 헝가리 미국 조지아 GWh",
    "SK On 재무 실적 매출 영업손실 2023 2024 조원",
    "SK On 배터리 출하량 시장점유율 글로벌 순위 2024",
    "SK On 고체 전지 차세대 배터리 상용화 일정 2027",
    "SK On 리스크 전기차 캐즘 적자 생존 대응 전략",
    "SK On 배터리 약점 한계 CATL LG에너지솔루션 대비 경쟁력",
    "SK On 재무 위기 적자 지속 자금 조달 위험",
    "SK On 기술 혁신 성과 수상 특허 글로벌 인정",
    "SK On 고객사 포드 현대 협력 성과 수주 실적",
]

SKON_VERIFICATION_QUERIES = [
    "SK On 2024 annual report revenue operating loss debt ratio",
    "SK On 사업보고서 2024 매출 영업손실 부채비율",
    "SNE Research SK On market share shipment GWh 2024",
    "Reuters Bloomberg SK On battery shipments market share 2024",
    "SK On ESS 매출 비중 비EV 매출 비중 2024",
    "SK On 사업 포트폴리오 EV ESS 비중 매출 구조",
]

MIN_RAG_SCORE = 0.5   # RAG 관련도 임계값
MIN_SOURCE_COUNT = 2  # 최소 출처 수 (미달 시 재검색 트리거)

RETRY_QUERY_PROMPT_TEMPLATE = """당신은 배터리 산업 리서치 쿼리 설계 전문가입니다.
아래 비평 피드백을 반영해 {company} 재탐색용 검색 쿼리를 생성하십시오.

요구사항:
1) {company}를 반드시 포함
2) 정량 검증이 가능한 표현 포함 (예: GWh, %, 매출, 영업이익, 점유율, shipment)
3) 긍정/부정/리스크/계획·전망까지 균형 있게 커버
4) 중복 없이 최대 {max_queries}개
5) 한국어 중심 쿼리

기존 기본 쿼리:
{base_queries}

비평 피드백:
{critic_feedback}

출력 형식(JSON only):
{{"queries": ["쿼리1", "쿼리2"]}}
"""


# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────

def skon_agent_node(state: GraphState) -> GraphState:
    """
    SK On Agent: 구조화된 데이터 수집 및 LLM 기반 추출.

    - 정량 데이터 우선 추출 (점유율, GWh, 매출 등)
    - 각 claim에 출처 URL 인라인 연결
    - 소스 수 부족 시 추가 검색 자동 수행
    """
    print("[T2 SK On Agent] SK On 데이터 수집 시작...")

    try:
        retry_target = state.get("critic1_retry_target", "none")
        is_retry = retry_target in ("skon", "both")

        queries = list(dict.fromkeys(SKON_PRIMARY_QUERIES + SKON_VERIFICATION_QUERIES))
        if is_retry:
            print("[T2 SK On Agent] 균형성 재검토 모드: LLM 보완 쿼리 생성...")
            critic_feedback = str(state.get("critic1_feedback", ""))
            llm_queries = _generate_retry_queries(
                company="SK On",
                critic_feedback=critic_feedback,
                base_queries=SKON_PRIMARY_QUERIES,
                max_queries=5,
            )
            if llm_queries:
                print(f"[T2 SK On Agent] LLM 보완 쿼리 {len(llm_queries)}개 추가")
                queries = list(dict.fromkeys(queries + llm_queries))
            else:
                print("[T2 SK On Agent] LLM 쿼리 생성 실패/빈 결과 → primary 쿼리셋 사용")

        rag_results = _search_rag(queries[:3])
        web_results = _search_web(queries)

        # 소스 수 부족 → 추가 검색
        total_sources = len(rag_results) + len(web_results)
        if total_sources < MIN_SOURCE_COUNT:
            print(f"[T2 SK On Agent] 소스 부족({total_sources}건) → 보완 검색 실행")
            extra = web_search(
                "SK On annual report 2024 revenue operating loss debt ratio market share SNE",
                max_results=5,
                min_domain_score=2,
                require_top_tier=True,
            )
            if not extra:
                extra = web_search(
                    "SK On annual report 2024 revenue operating loss debt ratio market share SNE",
                    max_results=5,
                    min_domain_score=2,
                )
            web_results.extend(extra)

        sk_on_data = extract_structured_data("SK On", rag_results, web_results)

        quality = sk_on_data.get("data_quality", "insufficient")
        src_count = len(sk_on_data.get("sources", []))
        print(f"[T2 SK On Agent] 완료: 품질={quality}, 출처={src_count}건")

        if quality == "insufficient":
            missing = sk_on_data.get("insufficient_categories", [])
            print(f"[T2 SK On Agent] 데이터 부족 항목: {missing}")

        return {
            **state,
            "sk_on_data": sk_on_data,
            "current_task": "skon_complete",
        }

    except Exception as e:
        error_msg = f"[T2 SK On Agent] 오류 발생: {str(e)}"
        print(error_msg)
        return {
            **state,
            "sk_on_data": {
                "technology": [{"claim": "SK On 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "strategy": [{"claim": "SK On 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "market": [{"claim": "SK On 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "risks": [{"claim": "SK On 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "quantitative_summary": {
                    "market_share": "데이터 부족",
                    "production_capacity": "데이터 부족",
                    "revenue": "데이터 부족",
                    "shipments": "데이터 부족",
                },
                "metrics_by_horizon": {"actual": [], "planned": [], "forecast": [], "unknown": []},
                "data_quality": "insufficient",
                "insufficient_categories": ["오류 발생"],
                "traceability_issues": [],
                "sources": [],
                "source_map": {},
            },
            "current_task": "skon_error",
            "error_log": state.get("error_log", []) + [error_msg],
        }


# ──────────────────────────────────────────────
# Search helpers
# ──────────────────────────────────────────────

def _search_rag(queries: List[str]) -> List[Dict]:
    """RAG 검색 — score < MIN_RAG_SCORE 결과 제거."""
    try:
        rag = RAGTool()
        if not rag.load_index():
            rag.index_documents(DOCS_PATH)
        if not rag._initialized:
            return []

        results = []
        seen = set()
        for query in queries:
            for hit in rag.search(query, k=3, min_score=MIN_RAG_SCORE):
                key = hit["content"][:100]
                if key not in seen:
                    seen.add(key)
                    results.append(hit)
        return results
    except Exception as e:
        print(f"[RAG SK On] 검색 실패: {e}")
        return []


def _search_web(queries: List[str]) -> List[Dict]:
    """웹 검색 — 도메인 품질 필터 적용 (web_search 내부 처리)."""
    all_results: List[Dict] = []
    seen_urls: set = set()

    for query in queries:
        print(f"  웹 검색: {query}")
        min_score = 2 if _is_verification_query(query) else 1
        require_top_tier = _requires_top_tier_evidence(query)
        hits = web_search(
            query,
            max_results=4,
            min_domain_score=min_score,
            require_top_tier=require_top_tier,
        )
        if require_top_tier and not hits:
            # Fallback to trusted tier to avoid empty retrieval, but still track quality later.
            hits = web_search(query, max_results=4, min_domain_score=2, require_top_tier=False)

        for r in hits:
            url = r.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    return all_results


def _is_verification_query(query: str) -> bool:
    """Queries for key metrics should rely on higher-credibility domains."""
    q = (query or "").lower()
    keywords = [
        "annual report", "사업보고서", "market share", "시장점유율",
        "shipment", "출하량", "revenue", "매출", "영업손실", "영업이익",
        "부채비율", "sne", "reuters", "bloomberg", "매출 비중", "비중",
    ]
    return any(k in q for k in keywords)


def _requires_top_tier_evidence(query: str) -> bool:
    """
    Critical numeric claims should prefer top-tier sources.
    """
    q = (query or "").lower()
    keywords = [
        "market share", "시장점유율", "shipment", "출하량",
        "ess 매출 비중", "ess revenue share", "비중",
        "annual report", "사업보고서", "reuters", "bloomberg", "sne",
        "부채비율", "debt ratio", "영업이익", "영업손실",
    ]
    return any(k in q for k in keywords)


def _generate_retry_queries(
    company: str,
    critic_feedback: str,
    base_queries: List[str],
    max_queries: int = 5,
) -> List[str]:
    """Generate additional search queries via LLM for retry cycles."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        prompt = RETRY_QUERY_PROMPT_TEMPLATE.format(
            company=company,
            max_queries=max_queries,
            base_queries="\n".join(f"- {q}" for q in base_queries[:12]),
            critic_feedback=critic_feedback[:1200] if critic_feedback else "없음",
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = str(response.content)
        parsed = _parse_query_response(raw_text)

        cleaned: List[str] = []
        seen = set()
        for q in parsed:
            text = str(q).strip()
            if not text:
                continue
            if len(text) < 10 or len(text) > 180:
                continue
            if company.lower().replace(" ", "") not in text.lower().replace(" ", ""):
                text = f"{company} {text}"
            if text not in seen and text not in base_queries:
                seen.add(text)
                cleaned.append(text)
            if len(cleaned) >= max_queries:
                break

        return cleaned
    except Exception as e:
        print(f"[T2 SK On Agent] LLM 보완 쿼리 생성 실패: {e}")
        return []


def _parse_query_response(text: str) -> List[str]:
    """Parse query list from JSON or bullet text."""
    # JSON fenced block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text or "")
    if m:
        text = m.group(1)

    # Try JSON object / array
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
            return [str(x) for x in obj["queries"]]
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass

    # Fallback: bullet/line parsing
    lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        s = re.sub(r"^[\-\*\d\.\)\s]+", "", s)
        if s:
            lines.append(s)
    return lines
