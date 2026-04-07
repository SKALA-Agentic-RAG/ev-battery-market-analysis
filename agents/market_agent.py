"""
Market Agent for analyzing the global battery/EV market.
Uses web search to collect data and LLM to summarize findings.
"""

import openai
from state import GraphState
from tools.web_search import web_search
from config import LLM_MODEL, OPENAI_API_KEY, LLM_TEMPERATURE

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def market_agent_node(state: GraphState) -> GraphState:
    """
    Market Agent node that analyzes EV/battery market trends (pre/post chasm).
    
    Args:
        state: The current state of the graph
        
    Returns:
        Updated state with market_context
    """
    print("\n[Market Agent] 글로벌 EV/배터리 시장 데이터 분석 시작...")
    
    # 1. Search for required data
    queries = [
        "2024-2026 글로벌 EV 판매 성장률 YoY",
        "LFP 배터리 글로벌 점유율 추이",
        "하이브리드(HEV/PHEV) 판매 비중 변화"
    ]
    
    all_search_results = []
    for query in queries:
        print(f"[Market Agent] 검색 중: {query}")
        results = web_search(query, max_results=3)
        all_search_results.extend(results)
        
    # 2. Format search results for LLM
    search_context = ""
    for i, res in enumerate(all_search_results):
        search_context += f"[{i+1}] {res.get('title')}\n"
        search_context += f"URL: {res.get('url')}\n"
        search_context += f"Content: {res.get('content')[:1000]}\n\n"
        
    # 3. Construct prompt for LLM
    prompt = f"""당신은 글로벌 전기차(EV) 및 배터리 시장 전문가입니다. 
제공된 검색 결과를 바탕으로 현재 시장의 '캐즘(Chasm)' 현황을 분석하고 보고서를 작성해 주세요.

[검색 결과]
{search_context}

[요구사항]
1. 보고서 제목: 글로벌 EV/배터리 시장 캐즘 전후 현황 분석 및 전략적 시사점
2. 캐즘 전(2021-2022)과 캐즘 진행/후(2024-2026)를 명확히 비교하는 섹션을 포함하세요.
3. 다음 데이터를 반드시 구체적으로 언급하세요:
   - 2024-2026 글로벌 EV 판매 성장률 추이 (YoY)
   - LFP 배터리의 글로벌 시장 점유율 변화
   - 하이브리드(HEV/PHEV) 차량의 판매 비중 변화 및 그 의미
4. 왜 배터리 기업들이 포트폴리오 다각화(LFP 개발, ESS 시장 진출 등)를 필수적으로 선택해야 하는지 논리적 근거를 제시하세요.
5. 모든 내용은 한국어로 작성하며, 마크다운(Markdown) 형식을 사용하세요.

전문적인 분석 보고서를 작성하세요:"""

    try:
        print(f"[Market Agent] LLM 요약 중 (Model: {LLM_MODEL})...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "당신은 전문적인 시장 분석가입니다. 객관적인 데이터에 기반하여 한국어로 보고서를 작성합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        market_analysis = response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"LLM 호출 오류: {str(e)}"
        print(f"[Market Agent] {error_msg}")
        market_analysis = f"시장 데이터 분석 중 오류가 발생했습니다.\n\n오류 상세: {error_msg}"
        if "error_log" not in state:
            state["error_log"] = []
        state["error_log"].append(error_msg)

    # 4. Update state
    state["market_context"] = market_analysis
    state["current_task"] = "market_analysis_complete"
    
    return state
