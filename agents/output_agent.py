"""T7: Markdown → HTML/PDF 저장."""

from __future__ import annotations

from pathlib import Path

import markdown2
from weasyprint import HTML

from state import GraphState

BASE_DIR = Path(__file__).resolve().parent.parent


def output_agent_node(state: GraphState) -> dict:
    md_text = state.get("final_draft") or ""
    md_path = BASE_DIR / "report.md"
    pdf_path = BASE_DIR / "report.pdf"
    try:
        md_path.write_text(md_text, encoding="utf-8")
        html_body = markdown2.markdown(
            md_text,
            extras=["tables", "fenced-code-blocks", "header-ids"],
        )
        html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<style>body {{ font-family: sans-serif; margin: 2rem; max-width: 900px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; }}</style>
</head><body>{html_body}</body></html>"""
        HTML(string=html_doc, base_url=str(BASE_DIR)).write_pdf(pdf_path)
    except Exception as e:
        err = f"output_agent: {e}"
        print(f"[{err}]")
        return {
            "current_task": "T7_output_error",
            "error_log": [err],
        }
    return {"current_task": "T7_output_done"}


def output_verify_node(state: GraphState) -> dict:
    """PDF/Markdown 산출물 프로그램적 검증(LLM 없음)."""
    md_path = BASE_DIR / "report.md"
    pdf_path = BASE_DIR / "report.pdf"
    prev = list(state.get("error_log") or [])
    errs: list[str] = []
    try:
        if not md_path.is_file() or md_path.stat().st_size < 80:
            errs.append("output_verify: report.md 없음 또는 비정상적으로 짧음")
        if not pdf_path.is_file() or pdf_path.stat().st_size < 800:
            errs.append("output_verify: report.pdf 없음 또는 비정상적으로 작음")
    except OSError as e:
        errs.append(f"output_verify: 파일 확인 실패 ({e})")
    if errs:
        return {
            "current_task": "T7_verify_failed",
            "error_log": prev + errs,
        }
    return {"current_task": "T7_verify_ok"}
