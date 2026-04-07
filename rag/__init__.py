"""
RAG package.

Contains modularized retrieval pipeline:
- chunking
- document loading
- index storage
- tool orchestration
"""

from .rag_tool import RAGTool

__all__ = ["RAGTool"]

