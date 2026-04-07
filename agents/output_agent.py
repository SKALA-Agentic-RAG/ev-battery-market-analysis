"""
Output Agent for the Multi-Agent Battery Market Analysis System.
Responsible for saving the final report in Markdown and PDF formats.
"""

import os
from typing import Any, Dict
from state import GraphState


def output_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Saves the final report as report.md and report.pdf.

    Args:
        state: Current graph state containing the final_draft

    Returns:
        Final task status
    """
    print("\n[Output Agent] 최종 보고서 저장 프로세스 시작...")
    
    final_draft = state.get("final_draft", "")
    if not final_draft:
        print("[Output Agent] 저장할 보고서 내용이 없습니다.")
        return {"current_task": "output_failed", "error_log": ["No final draft content"]}

    # 1. Save as Markdown
    md_path = "report.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_draft)
        print(f"[Output Agent] Markdown 보고서 저장 완료: {os.path.abspath(md_path)}")
    except Exception as e:
        print(f"[Output Agent] Markdown 저장 실패: {e}")
        return {"current_task": "output_error", "error_log": [str(e)]}

    # 2. Attempt PDF conversion
    # Note: PDF conversion with proper Korean support requires external libraries like fpdf2 or reportlab
    # with a specific Korean font (.ttf). We'll implement a fallback here.
    pdf_path = "report.pdf"
    try:
        # Try to use fpdf2 if available
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(80)
                self.cell(30, 10, 'Battery Market Analysis Report', 0, 0, 'C')
                self.ln(20)

        pdf = PDF()
        pdf.add_page()
        # Note: Standard Arial does NOT support Korean. 
        # For real Korean PDF, we'd need: pdf.add_font('NanumGothic', '', 'NanumGothic.ttf', unicode=True)
        # and: pdf.set_font('NanumGothic', '', 12)
        pdf.set_font("Arial", size=12)
        
        # Split content by lines to avoid overflow
        for line in final_draft.split('\n'):
            # Replace non-latin characters for the fallback PDF
            clean_line = "".join([c if ord(c) < 128 else "?" for c in line])
            pdf.cell(200, 10, txt=clean_line, ln=True)
            
        pdf.output(pdf_path)
        print(f"[Output Agent] PDF 보고서 저장 완료 (텍스트 전용): {os.path.abspath(pdf_path)}")
        print("  ※ 한글 폰트가 설치되지 않아 PDF 내 한글은 '?'로 표시될 수 있습니다. report.md를 확인해 주세요.")
        
    except ImportError:
        print("[Output Agent] fpdf2 패키지가 없습니다. PDF 생성을 건너뜁니다.")
        print("  (pip install fpdf2 를 통해 설치 가능합니다)")
    except Exception as e:
        print(f"[Output Agent] PDF 생성 실패: {e}")

    return {"current_task": "workflow_complete"}


def output_verify_node(_state: GraphState) -> Dict[str, Any]:
    """출력 결과물 존재 여부를 확인하는 검증 노드."""
    import os
    md_exists = os.path.exists("report.md")
    pdf_exists = os.path.exists("report.pdf")
    if md_exists:
        print(f"[Output Verify] report.md 확인 완료")
    if pdf_exists:
        print(f"[Output Verify] report.pdf 확인 완료")
    if not md_exists:
        print("[Output Verify] 경고: report.md 없음")
    return {"current_task": "output_verified"}
