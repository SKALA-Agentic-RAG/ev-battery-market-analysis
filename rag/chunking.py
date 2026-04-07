"""
Text cleanup and chunking helpers for RAG ingestion.
"""

import re
from typing import Dict, List, Optional, Tuple


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noisy standalone page-number lines."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_text_sections(text: str, filepath: str) -> List[Tuple[Optional[str], str]]:
    """Split markdown by headings; return one section for non-markdown text."""
    if not filepath.lower().endswith(".md"):
        return [(None, text)]

    sections: List[Tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []

    for line in text.splitlines():
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
                current_lines = []
            current_heading = heading_match.group(1).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections if sections else [(None, text)]


def remove_repeated_lines(page_texts: List[str]) -> List[str]:
    """
    Remove repeated header/footer-like lines that appear on many pages.
    """
    if len(page_texts) < 4:
        return [clean_text(text) for text in page_texts]

    line_frequency: Dict[str, int] = {}
    for text in page_texts:
        unique_lines = set()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) < 4 or len(line) > 120:
                continue
            if re.fullmatch(r"\d{1,4}", line):
                continue
            unique_lines.add(line)
        for line in unique_lines:
            line_frequency[line] = line_frequency.get(line, 0) + 1

    threshold = max(3, int(len(page_texts) * 0.35))
    repeated_lines = {line for line, count in line_frequency.items() if count >= threshold}
    if not repeated_lines:
        return [clean_text(text) for text in page_texts]

    cleaned_pages = []
    for text in page_texts:
        kept_lines = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped in repeated_lines:
                continue
            kept_lines.append(raw_line)
        cleaned_pages.append(clean_text("\n".join(kept_lines)))
    return cleaned_pages


def chunk_text(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """
    Split text into overlapping chunks while preserving sentence/paragraph boundaries.
    """
    chunk_size = chunk_size or chunk_size_words
    overlap = overlap if overlap is not None else chunk_overlap_words
    overlap = max(0, min(overlap, chunk_size // 2))

    cleaned = clean_text(text)
    if not cleaned:
        return []

    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", cleaned)
        if segment and segment.strip()
    ]
    if not segments:
        segments = [cleaned]

    chunks: List[str] = []
    current_words: List[str] = []

    for segment in segments:
        segment_words = segment.split()
        if not segment_words:
            continue

        if len(segment_words) > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []

            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(segment_words):
                end = min(start + chunk_size, len(segment_words))
                chunk_words = segment_words[start:end]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
                if end >= len(segment_words):
                    break
                start += step
            continue

        if len(current_words) + len(segment_words) <= chunk_size:
            current_words.extend(segment_words)
            continue

        chunks.append(" ".join(current_words))
        carry_words = current_words[-overlap:] if overlap > 0 else []
        current_words = carry_words + segment_words

    if current_words:
        chunks.append(" ".join(current_words))

    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

