"""
FAISS index/document persistence helpers for RAG.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple


def compute_corpus_fingerprint(filepaths: List[str]) -> str:
    """
    Compute a stable fingerprint of source files using path/size/mtime.
    """
    sha = hashlib.sha256()
    for path in filepaths:
        try:
            stat = os.stat(path)
            rel = os.path.relpath(path)
            sha.update(rel.encode("utf-8", errors="ignore"))
            sha.update(str(stat.st_size).encode("utf-8"))
            sha.update(str(stat.st_mtime_ns).encode("utf-8"))
        except OSError:
            sha.update(path.encode("utf-8", errors="ignore"))
    return sha.hexdigest()


def _paths(base_path: str) -> Tuple[str, str, str]:
    index_path = os.path.join(base_path, "index.faiss")
    docs_path = os.path.join(base_path, "documents.json")
    meta_path = os.path.join(base_path, "index_meta.json")
    return index_path, docs_path, meta_path


def save_index_bundle(index, documents: List[Dict], index_meta: Dict, base_path: str) -> None:
    """Save FAISS index, documents, and metadata."""
    import faiss

    os.makedirs(base_path, exist_ok=True)
    index_path, docs_path, meta_path = _paths(base_path)
    faiss.write_index(index, index_path)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(index_meta or {}, f, ensure_ascii=False, indent=2)


def load_index_meta(base_path: str) -> Dict:
    """Load index metadata from a path."""
    try:
        _, _, meta_path = _paths(base_path)
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_existing_db_path(primary_path: str, legacy_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve which vector DB path currently has index files.
    Prefer primary path, fallback to legacy path.
    """
    candidates = [primary_path]
    if legacy_path and legacy_path not in candidates:
        candidates.append(legacy_path)

    for path in candidates:
        index_path, docs_path, _ = _paths(path)
        if os.path.exists(index_path) and os.path.exists(docs_path):
            return path
    return None


def load_index_bundle(base_path: str) -> Optional[Tuple[object, List[Dict]]]:
    """Load FAISS index and documents from one path."""
    import faiss

    index_path, docs_path, _ = _paths(base_path)
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return None

    index = faiss.read_index(index_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    if not isinstance(documents, list):
        documents = []
    return index, documents

