[README.md](https://github.com/user-attachments/files/26530913/README.md)
# EV Battery Market Analysis (SK On vs CATL)

## Overview
- **Objective**: 전기차 배터리 시장 내 주요 플레이어인 SK On과 CATL의 전략 및 기술 경쟁력을 비교 분석하여 심층 보고서를 생성하는 멀티 에이전트 시스템입니다.
- **Method**: LangGraph 기반의 멀티 에이전트 워크플로우를 활용하여 데이터 수집, 분석, 비판, 작성 과정을 자동화합니다.
- **Tools**: LangGraph, LangChain, Tavily Search, FAISS, FlagEmbedding, WeasyPrint(PDF)

## Features
- **PDF/MD/TXT 자료 기반 정보 추출**: RAG(Retrieval-Augmented Generation)를 통해 내부 문서를 검색하고 분석에 활용합니다.
- **신뢰성 기반 웹 검색**: Tavily API를 사용하여 고신뢰도 도메인(Reuters, Bloomberg, SNE Research 등)의 최신 정보를 수집하고 필터링합니다.
- **확증 편향 방지 및 품질 관리**: Critic 에이전트(Critic1~Critic3)를 통해 데이터의 공정성, 근거의 타당성, 논리적 일관성, 보고서 형식/출처 무결성을 검증하고 필요 시 재수집/재분석/재작성 루프를 수행합니다.
- **자동 보고서 생성**: 분석된 내용을 바탕으로 Markdown 및 PDF 형식의 최종 보고서를 자동 생성합니다.
- **최종 산출물 체크**: PDF 변환 후 `report.md`/`report.pdf` 파일 존재·크기 기반의 프로그램적 검증(`output_verify`)을 수행합니다.

## Tech Stack
| Category   | Details |
|------------|---------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | OpenAI Chat Model (`config.LLM_MODEL`, default: `gpt-4.1-mini`) |
| Retrieval  | FAISS (Vector Store) |
| Embedding  | `BAAI/bge-m3` (FlagEmbedding) |
| Search     | Tavily Search API |
| Output     | `markdown2` + `weasyprint` (Markdown → PDF) |

## Agents
- **Market Agent**: 글로벌 전기차 및 배터리 시장의 일반적인 배경과 트렌드 데이터를 수집합니다.
- **SK On Agent**: SK On의 기술력, 포트폴리오, 시장 전략, 전략적 리스크 등을 조사합니다. (RAG + Web)
- **CATL Agent**: CATL의 기술적 우위, 공급망/시장 전략, 리스크 등을 조사합니다. (RAG + Web)
- **Critic 1 (Fairness/Grounding)**: 수집 데이터의 완결성·공정성·근거성을 검토하고, 필요 시 재수집을 요청합니다.
- **Analysis Agent**: 수집된 데이터를 바탕으로 양사 전략 비교 및 Comparative SWOT 분석을 수행합니다.
- **Critic 2 (Logic/Consistency)**: 분석 결과의 논리 구조·일관성·수집 데이터 정합을 검증합니다.
- **Writer Agent**: 분석된 결과를 바탕으로 목차에 따른 상세 리포트를 작성합니다.
- **Critic 3 (Draft/Format/Reference)**: 보고서 초안의 형식·내용·REFERENCE 무결성을 검증하고, 필요 시 재작성을 요청합니다.
- **Output Agent**: 최종 작성된 내용을 Markdown 및 PDF 파일로 변환하여 저장합니다.
- **Output Verify**: 산출물 파일(programmatic) 검증을 수행합니다.

## Architecture
시스템은 다음과 같은 LangGraph 워크플로우로 동작하며, `graph.py`의 `build_graph()`에 정의되어 있습니다.

1. **데이터 수집 (`collect_all`)**: 시장, SK On, CATL 데이터를 순차적으로 수집합니다.
2. **1차 검증 (`critic1`)**: 수집된 데이터의 품질을 검증하고 필요 시 수집 단계로 회귀합니다.
3. **심층 분석 (`analysis`)**: 두 기업의 전략적 차이를 분석합니다.
4. **2차 검증 (`critic2`)**: 분석 결과의 논리·일관성을 검토하고 필요 시 재분석합니다.
5. **보고서 작성 (`writer`)**: 최종 리포트 초안을 작성합니다.
6. **3차 검증 (`critic3`)**: 초안의 형식·내용·REFERENCE 무결성을 검토하고 필요 시 재작성합니다.
7. **결과 출력 (`output`)**: MD/PDF 파일을 생성합니다.
8. **산출물 검증 (`output_verify`)**: 파일 존재/크기 기반의 프로그램적 검증을 수행합니다.

## Setup & Run

### 1) 환경 변수 설정
프로젝트 루트에서:

```bash
cp .env.example .env
```

`.env`에 `OPENAI_API_KEY`, `TAVILY_API_KEY`를 채워 주세요.

### 2) 상태 점검(추천)

```bash
python main.py doctor
```

### 3) RAG 인덱스 구축(선택)
`docs/`에 PDF/MD/TXT를 넣은 뒤:

```bash
python main.py setup-rag
```

기존 인덱스를 지우고 새로 만들려면:

```bash
python main.py setup-rag --rebuild
```

### 4) 실행

```bash
python main.py run
```

주제를 바로 넘기려면:

```bash
python main.py run --topic "전기차 캐즘 환경에서 SK On과 CATL의 포트폴리오 다각화 전략 비교 분석"
```

실행이 완료되면 프로젝트 루트에 `report.md`, `report.pdf`가 생성됩니다.

## Directory Structure
```text
ev-battery-market-analysis/
├── agents/                # 에이전트별 노드 정의 및 스키마
│   ├── market_agent.py    # 글로벌 배터리 시장 환경 분석 에이전트
│   ├── skon_agent.py      # SK On 기업/기술 데이터 수집 에이전트
│   ├── catl_agent.py      # CATL 기업/기술 데이터 수집 에이전트
│   ├── analysis_agent.py  # 기업 간 비교 및 전략 분석 에이전트
│   ├── critic_agent.py    # 검증 및 피드백 (Critic1~Critic3)
│   ├── writer_agent.py    # 분석 결과를 바탕으로 리포트 작성
│   ├── output_agent.py    # 리포트를 MD/PDF로 변환 및 저장(+output_verify)
│   ├── supervisor.py      # 워크플로우 라우팅 및 흐름 제어 로직
│   ├── llm.py             # 공유 LLM 인스턴스 설정
│   ├── schemas.py         # Pydantic 스키마 정의
│   ├── _util.py           # 에이전트 공용 유틸리티
│   └── utils/             # 텍스트 추출 및 처리 보조 유틸리티
│       └── _extract_utils.py
├── rag/                   # RAG 엔진 (Embedding, Chunking, Storage)
│   ├── chunking.py
│   ├── loaders.py
│   ├── rag_tool.py
│   └── storage.py
├── tools/                 # 외부 도구 (Web Search, RAG Wrapper)
│   ├── web_search.py
│   └── rag.py
├── docs/                  # 분석에 활용될 PDF/MD/TXT 문서 (RAG 인덱싱 대상)
├── rag/vectordb/          # (기본) 벡터 인덱스 저장 경로
├── graph.py               # LangGraph 워크플로우 구축
├── state.py               # 그래프 상태(GraphState) 정의
├── main.py                # 실행 엔트리 포인트 (doctor/setup-rag/run)
├── requirements.txt
└── .env.example
```

## Contributors

- 전아린: PDF Parsing & Retrieval(RAG), Integration, Agent Design, Routing, 
- 이지수: Workflow, Supervisor Agent, Evaluation/Verification, Prompt Engineering
- 권익주: AIOps Architecture Design, Critic Agent, Writer Agent, Analysis Agent




