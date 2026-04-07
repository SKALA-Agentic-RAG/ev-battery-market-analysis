"""
Output Agent (T7) for converting the final Markdown report to PDF.
Saves report.md and attempts PDF conversion via weasyprint.
"""

import os
from typing import Any, Dict
from state import GraphState


def output_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Output Agent node: saves the final report as Markdown and converts to PDF.

    Steps:
    1. Save final_draft to report.md
    2. Convert Markdown to HTML using markdown2
    3. Convert HTML to PDF using weasyprint
    4. Log completion

    If PDF conversion fails, only Markdown is saved and error is logged.

    Args:
        state: Current graph state

    Returns:
        Updated state with completion logged
    """
    print("[T7 Output Agent] 보고서 저장 및 PDF 변환 시작...")

    final_draft = state.get("final_draft", "")
    error_log = list(state.get("error_log", []))

    if not final_draft:
        error_msg = "[T7 Output Agent] 최종 보고서 초안이 비어 있습니다."
        print(error_msg)
        return {
            "current_task": "output_error",
            "error_log": error_log + [error_msg],
        }

    # Step 1: Save Markdown
    md_path = "report.md"
    md_success = _save_markdown(final_draft, md_path, error_log)

    # Step 2: Convert to PDF
    pdf_path = "report.pdf"
    pdf_success = _convert_to_pdf(final_draft, pdf_path, error_log)

    if md_success and pdf_success:
        print(f"[T7 Output Agent] 완료: {md_path} 및 {pdf_path} 생성됨")
        status_msg = f"보고서 생성 완료: {md_path}, {pdf_path}"
    elif md_success:
        print(f"[T7 Output Agent] 완료 (PDF 실패): {md_path}만 생성됨")
        status_msg = f"마크다운 보고서 생성 완료: {md_path} (PDF 변환 실패)"
    else:
        status_msg = "보고서 저장 실패"

    return {
        "current_task": "output_complete",
        "error_log": error_log,
        "messages": [{"role": "assistant", "content": status_msg}],
    }


def _save_markdown(content: str, filepath: str, error_log: list) -> bool:
    """Save the report content as a Markdown file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[T7 Output Agent] Markdown 저장 완료: {filepath}")
        return True
    except Exception as e:
        error_msg = f"[T7 Output Agent] Markdown 저장 실패: {str(e)}"
        print(error_msg)
        error_log.append(error_msg)
        return False


def _convert_to_pdf(content: str, pdf_path: str, error_log: list) -> bool:
    """Convert Markdown content to PDF using markdown2 + weasyprint."""
    try:
        _prepare_weasyprint_runtime()
        import markdown2
        import weasyprint

        html_content = markdown2.markdown(
            content,
            extras=[
                "tables",
                "fenced-code-blocks",
                "header-ids",
                "strike",
                "task_list",
            ],
        )

        styled_html = _wrap_with_html_template(html_content)
        pdf = weasyprint.HTML(string=styled_html).write_pdf()

        with open(pdf_path, "wb") as f:
            f.write(pdf)

        print(f"[T7 Output Agent] PDF 저장 완료: {pdf_path}")
        return True

    except ImportError as e:
        error_msg = f"[T7 Output Agent] PDF 변환 라이브러리 없음: {str(e)}"
        print(error_msg)
        error_log.append(error_msg)
        return False
    except Exception as e:
        error_msg = f"[T7 Output Agent] PDF 변환 실패: {str(e)}"
        print(error_msg)
        error_log.append(error_msg)
        return False


def _prepare_weasyprint_runtime():
    """
    Ensure dynamic library lookup works for WeasyPrint on macOS Homebrew.

    WeasyPrint/cffi may fail to locate GLib libraries even when installed via
    Homebrew. Adding Homebrew lib paths to DYLD_FALLBACK_LIBRARY_PATH fixes it.
    """
    if os.name != "posix":
        return

    candidates = [p for p in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.exists(p)]
    if not candidates:
        return

    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    existing = [p for p in current.split(":") if p]

    changed = False
    for path in candidates:
        if path not in existing:
            existing.insert(0, path)
            changed = True

    if changed:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(existing)


def _wrap_with_html_template(html_content: str) -> str:
    """Wrap HTML content with a full HTML document including CSS styling."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>배터리 시장 전략 분석 보고서</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm 2.5cm;
        }}
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
            font-size: 11pt;
            line-height: 1.8;
            color: #333333;
            max-width: 100%;
        }}
        h1 {{
            font-size: 20pt;
            font-weight: bold;
            color: #1a1a2e;
            border-bottom: 3px solid #1a1a2e;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        h2 {{
            font-size: 15pt;
            font-weight: bold;
            color: #16213e;
            border-bottom: 1px solid #cccccc;
            padding-bottom: 5px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        h3 {{
            font-size: 13pt;
            font-weight: bold;
            color: #0f3460;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        h4 {{
            font-size: 12pt;
            font-weight: bold;
            color: #333333;
            margin-top: 15px;
        }}
        p {{
            margin-bottom: 10px;
            text-align: justify;
        }}
        ul, ol {{
            margin-bottom: 10px;
            padding-left: 25px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }}
        th {{
            background-color: #1a1a2e;
            color: white;
            font-weight: bold;
            padding: 10px 12px;
            text-align: center;
            border: 1px solid #cccccc;
        }}
        td {{
            padding: 8px 12px;
            border: 1px solid #cccccc;
            vertical-align: top;
        }}
        tr:nth-child(even) {{
            background-color: #f5f5f5;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 10pt;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #1a1a2e;
            padding-left: 15px;
            margin: 15px 0;
            color: #666666;
            font-style: italic;
        }}
        strong {{
            color: #1a1a2e;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""


def output_verify_node(_state: GraphState) -> Dict[str, Any]:
    """출력 결과물 존재 여부를 확인하는 검증 노드."""
    md_exists = os.path.exists("report.md")
    pdf_exists = os.path.exists("report.pdf")
    if md_exists:
        print("[Output Verify] report.md 확인 완료")
    if pdf_exists:
        print("[Output Verify] report.pdf 확인 완료")
    if not md_exists:
        print("[Output Verify] 경고: report.md 없음")
    return {"current_task": "output_verified"}
