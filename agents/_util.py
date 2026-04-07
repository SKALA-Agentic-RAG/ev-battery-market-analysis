"""상태·텍스트 유틸."""

from __future__ import annotations

import json
from typing import Any


def topic_from_state(state: dict) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    return c.strip()
        ctype = getattr(m, "type", None)
        if ctype == "human" and getattr(m, "content", None):
            c = m.content
            if isinstance(c, str) and c.strip():
                return c.strip()
    return "전기차 캐즘 환경에서 SK On과 CATL의 포트폴리오 다각화 전략 비교"


def truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated]"


def dumps_compact(obj: Any, max_chars: int = 8000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    return truncate(s, max_chars)
