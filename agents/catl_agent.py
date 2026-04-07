"""
CATL Agent (T3): CATL 배터리 전략 데이터 수집.

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
from config import DOCS_PATH, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, MAX_RETRIES
from agents.utils._extract_utils import extract_structured_data

# ──────────────────────────────────────────────
# Search queries
# ──────────────────────────────────────────────

CATL_PRIMARY_QUERIES = [
    "CATL 배터리 기술 포트폴리오 LFP CTP Kirin 에너지밀도 GWh 2024",
    "CATL 포트폴리오 다각화 전략 ESS 선박 항공 나트륨이온 매출",
    "CATL 생산 능력 기가팩토리 글로벌 헝가리 독일 GWh",
    "CATL 재무 실적 매출 순이익 2023 2024 억위안 억달러",
    "CATL 배터리 출하량 시장점유율 세계 1위 퍼센트 2024",
    "CATL 나트륨이온 배터리 고체배터리 차세대 기술 상용화",
    "CATL 리스크 미중 갈등 관세 IRA 지정학 미국 제재",
    "CATL 배터리 약점 한계 기술 보안 우려 미국 블랙리스트",
    "CATL 재무 리스크 중국 내수 의존 경쟁 심화 마진 하락",
    "CATL 혁신 성과 특허 글로벌 파트너십 수상",
    "CATL ESS 에너지 저장 사업 성장 매출 비중",
]

CATL_VERIFICATION_QUERIES = [
    "CATL annual report 2024 revenue net profit debt ratio",
    "CATL 2024 年报 营收 净利润 资产负债率",
    "SNE Research CATL market share shipment GWh 2024",
    "Reuters Bloomberg CATL battery shipments market share 2024",
    "CATL ESS revenue share non-EV revenue mix 2024",
    "CATL 사업 포트폴리오 EV ESS 선박 항공 매출 구조 비중",
]

MIN_RAG_SCORE = 0.5
MIN_SOURCE_COUNT = 2

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

def catl_agent_node(state: GraphState) -> GraphState:
    """
    CATL Agent: 구조화된 데이터 수집 및 LLM 기반 추출.

    - 정량 데이터 우선 추출 (점유율, GWh, 매출 등)
    - 각 claim에 출처 URL 인라인 연결
    - 소스 수 부족 시 추가 검색 자동 수행
    """
    print("[T3 CATL Agent] CATL 데이터 수집 시작...")

    retry_target = state.get("critic1_retry_target", "none")
    is_retry = retry_target in ("catl", "both")
    critic1_retry_count = state.get("critic1_retry_count", 0)

    # 강제 종료: RAG 재검색 한도 초과 시 기존 데이터 유지
    if is_retry and critic1_retry_count >= MAX_RETRIES:
        print(f"[T3 CATL Agent] RAG 재검색 {MAX_RETRIES}회 도달 → 강제 종료 (기존 데이터 유지)")
        return {
            "catl_data": state.get("catl_data", {}),
            "current_task": "catl_force_exit",
        }

    try:
        queries = list(dict.fromkeys(CATL_PRIMARY_QUERIES + CATL_VERIFICATION_QUERIES))
        if is_retry:
            print("[T3 CATL Agent] 균형성 재검토 모드: LLM 보완 쿼리 생성...")
            critic_feedback = str(state.get("critic1_feedback", ""))
            llm_queries = _generate_retry_queries(
                company="CATL",
                critic_feedback=critic_feedback,
                base_queries=CATL_PRIMARY_QUERIES,
                max_queries=5,
            )
            if llm_queries:
                print(f"[T3 CATL Agent] LLM 보완 쿼리 {len(llm_queries)}개 추가")
                queries = list(dict.fromkeys(queries + llm_queries))
            else:
                print("[T3 CATL Agent] LLM 쿼리 생성 실패/빈 결과 → primary 쿼리셋 사용")

        rag_results = _search_rag(queries[:3])
        web_results = _search_web(queries)

        # 소스 수 부족 → 추가 검색
        total_sources = len(rag_results) + len(web_results)
        if total_sources < MIN_SOURCE_COUNT:
            print(f"[T3 CATL Agent] 소스 부족({total_sources}건) → 보완 검색 실행")
            extra = web_search(
                "CATL annual report 2024 revenue net profit debt ratio market share SNE",
                max_results=5,
                min_domain_score=2,
                require_top_tier=True,
            )
            if not extra:
                extra = web_search(
                    "CATL annual report 2024 revenue net profit debt ratio market share SNE",
                    max_results=5,
                    min_domain_score=2,
                )
            web_results.extend(extra)

        catl_data = extract_structured_data("CATL", rag_results, web_results)

        quality = catl_data.get("data_quality", "insufficient")
        src_count = len(catl_data.get("sources", []))
        print(f"[T3 CATL Agent] 완료: 품질={quality}, 출처={src_count}건")

        if quality == "insufficient":
            missing = catl_data.get("insufficient_categories", [])
            print(f"[T3 CATL Agent] 데이터 부족 항목: {missing}")

        return {
            "catl_data": catl_data,
            "current_task": "catl_complete",
        }

    except Exception as e:
        error_msg = f"[T3 CATL Agent] 오류 발생: {str(e)}"
        print(error_msg)
        return {
            "catl_data": {
                "technology": [{"claim": "CATL 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "strategy": [{"claim": "CATL 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "market": [{"claim": "CATL 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
                "risks": [{"claim": "CATL 데이터 부족 (오류)", "value": None, "time_horizon": "unknown"}],
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
            "current_task": "catl_error",
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
        print(f"[RAG CATL] 검색 실패: {e}")
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
        "annual report", "年报", "market share", "시장점유율",
        "shipment", "출하량", "revenue", "매출", "net profit", "순이익",
        "debt ratio", "부채비율", "资产负债率", "sne", "reuters", "bloomberg",
        "매출 비중", "revenue share", "비중",
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
        "annual report", "年报", "reuters", "bloomberg", "sne",
        "debt ratio", "부채비율", "资产负债率", "순이익", "net profit",
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
        print(f"[T3 CATL Agent] LLM 보완 쿼리 생성 실패: {e}")
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
