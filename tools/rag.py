"""
Backward-compatible RAG tool entrypoint.

Actual implementation lives in the top-level `rag/` package.
"""

from rag.rag_tool import RAGTool

__all__ = ["RAGTool"]

