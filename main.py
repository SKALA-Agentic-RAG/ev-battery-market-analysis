"""
Main entry point for the Multi-Agent Battery Market Analysis System.
Runs the LangGraph workflow to produce a comparative analysis of SK On vs CATL.
"""

import sys
import os
from pathlib import Path
from graph import build_graph
from state import GraphState

# 출력 파일을 스크립트 위치 기준으로 저장
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)


def _format_step_status(node_name: str, node_output) -> str:
    """워크플로우 단계 로그를 상태 기반으로 사람이 읽기 쉽게 포맷팅."""
    if not isinstance(node_output, dict):
        return f"[{node_name}] 완료"

    critic_pass_keys = {
        "critic1": "critic1_pass",
        "critic2": "critic2_pass",
        "critic3": "critic3_pass",
    }
    pass_key = critic_pass_keys.get(node_name)

    if pass_key:
        passed = node_output.get(pass_key)
        if passed is True:
            return f"[{node_name}] 완료 (PASS)"
        if passed is False:
            retry_count = node_output.get(f"{node_name}_retry_count")
            if node_name == "critic1":
                target = node_output.get("critic1_retry_target", "none")
                if retry_count is not None:
                    return f"[{node_name}] 재시도 필요 (FAIL, target={target}, retry_count={retry_count})"
                return f"[{node_name}] 재시도 필요 (FAIL, target={target})"
            if retry_count is not None:
                return f"[{node_name}] 재시도 필요 (FAIL, retry_count={retry_count})"
            return f"[{node_name}] 재시도 필요 (FAIL)"

    current_task = str(node_output.get("current_task", ""))
    if current_task.endswith("force_pass"):
        return f"[{node_name}] 완료 (FORCE PASS)"
    if "error" in current_task.lower():
        return f"[{node_name}] 완료 (ERROR)"

    return f"[{node_name}] 완료"


def main():
    """
    Main function to run the battery market analysis multi-agent system.

    Prompts user for analysis topic, initializes state, and streams
    the LangGraph workflow to completion, producing report.md and report.pdf.
    """
    print("=" * 60)
    print("  배터리 시장 전략 분석 멀티 에이전트 시스템")
    print("=" * 60)
    print()

    # Get analysis topic from user
    topic = input("분석 주제를 입력하세요 (Enter for default): ").strip()
    if not topic:
        topic = "전기차 캐즘 환경에서 SK On과 CATL의 포트폴리오 다각화 전략 비교 분석"

    print(f"\n분석 주제: {topic}")
    print()

    # Build the workflow graph
    try:
        app = build_graph()
    except Exception as e:
        print(f"[오류] 그래프 구축 실패: {e}")
        sys.exit(1)

    # Initialize state
    initial_state = {
        "messages": [{"role": "user", "content": topic}],
        "market_context": "",
        "sk_on_data": {},
        "catl_data": {},
        "analysis_report": "",
        "final_draft": "",
        "critic1_feedback": "",
        "critic1_pass": False,
        "critic1_retry_count": 0,
        "critic1_retry_target": "none",
        "critic2_feedback": "",
        "critic2_pass": False,
        "critic2_retry_count": 0,
        "critic3_feedback": "",
        "critic3_pass": False,
        "critic3_retry_count": 0,
        "current_task": "start",
        "error_log": [],
    }

    # LangGraph requires a config with thread_id for MemorySaver
    config = {"configurable": {"thread_id": "battery-analysis-1"}}

    print("=== 배터리 시장 전략 분석 시스템 시작 ===\n")

    # Stream the workflow execution
    try:
        for step in app.stream(initial_state, config):
            for node_name, node_output in step.items():
                print(_format_step_status(node_name, node_output))

                # Print any errors logged in this step
                if isinstance(node_output, dict):
                    error_log = node_output.get("error_log", [])
                    if error_log:
                        # Only print new errors (not cumulative)
                        current_task = node_output.get("current_task", "")
                        if "error" in current_task.lower():
                            print(f"  ⚠ 오류 발생: {error_log[-1] if error_log else '알 수 없는 오류'}")

    except KeyboardInterrupt:
        print("\n[사용자 중단] 분석이 취소되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[오류] 워크플로우 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    md_path = BASE_DIR / "report.md"
    pdf_path = BASE_DIR / "report.pdf"
    print("\n=== 분석 완료 ===")
    print("\n생성된 파일:")
    if md_path.exists():
        print(f"  - {md_path}  ({md_path.stat().st_size:,} bytes)")
    if pdf_path.exists():
        print(f"  - {pdf_path}  ({pdf_path.stat().st_size:,} bytes)")
    elif not md_path.exists():
        print("  [경고] 파일 생성에 실패했습니다. error_log를 확인하세요.")


if __name__ == "__main__":
    main()
