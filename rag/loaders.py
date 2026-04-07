"""
Document loading utilities for RAG ingestion.
"""

import os
from typing import Dict, List

from .chunking import chunk_text, clean_text, remove_repeated_lines, split_text_sections


def read_pdf_pages(filepath: str) -> List[str]:
    """Read text from each page of a PDF file."""
    try:
        import PyPDF2

        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return [clean_text(page.extract_text() or "") for page in reader.pages]
    except ImportError:
        pass
    except Exception as e:
        print(f"[RAG] PyPDF2 PDF 읽기 실패 ({filepath}): {e}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except ImportError:
        print(f"[RAG] PDF 파싱 라이브러리 없음 (PyPDF2 또는 pypdf 필요): {filepath}")
    except Exception as e:
        print(f"[RAG] pypdf PDF 읽기 실패 ({filepath}): {e}")
    return []


def load_pdf(filepath: str, chunk_size_words: int, chunk_overlap_words: int) -> List[Dict]:
    """Load and chunk a PDF file."""
    chunks: List[Dict] = []
    try:
        page_texts = read_pdf_pages(filepath)
        if not page_texts:
            return chunks

        page_texts = remove_repeated_lines(page_texts)
        filename = os.path.basename(filepath)
        rel_source = os.path.relpath(filepath)

        for page_num, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            page_chunks = chunk_text(page_text, chunk_size_words, chunk_overlap_words)
            for local_idx, chunk in enumerate(page_chunks, start=1):
                chunks.append(
                    {
                        "content": chunk,
                        "source": f"{filename} (p.{page_num})",
                        "chunk_id": f"{filename}:p{page_num}:c{local_idx}",
                        "metadata": {
                            "doc_id": filename,
                            "source_file": rel_source,
                            "page_start": page_num,
                            "page_end": page_num,
                            "section_path": None,
                            "chunk_index": local_idx,
                        },
                    }
                )
    except Exception as e:
        print(f"[RAG] PDF 로드 실패 ({filepath}): {e}")
    return chunks


def load_text(filepath: str, chunk_size_words: int, chunk_overlap_words: int) -> List[Dict]:
    """Load and chunk a text or markdown file."""
    chunks: List[Dict] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if not text.strip():
            return chunks

        filename = os.path.basename(filepath)
        rel_source = os.path.relpath(filepath)
        sections = split_text_sections(text, filepath)

        chunk_index = 0
        for section_path, section_text in sections:
            for chunk in chunk_text(section_text, chunk_size_words, chunk_overlap_words):
                chunk_index += 1
                chunks.append(
                    {
                        "content": chunk,
                        "source": filename if section_path is None else f"{filename} ({section_path})",
                        "chunk_id": f"{filename}:c{chunk_index}",
                        "metadata": {
                            "doc_id": filename,
                            "source_file": rel_source,
                            "page_start": None,
                            "page_end": None,
                            "section_path": section_path,
                            "chunk_index": chunk_index,
                        },
                    }
                )
    except Exception as e:
        print(f"[RAG] 텍스트 로드 실패 ({filepath}): {e}")
    return chunks

