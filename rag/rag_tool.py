"""
RAG tool orchestration class.
"""

import glob
import os
import time
from typing import Dict, List, Optional, Tuple

from config import DOCS_PATH, EMBEDDING_MODEL, RAG_TOP_K, VECTOR_DB_PATH
from .loaders import load_pdf, load_text
from .storage import (
    compute_corpus_fingerprint,
    load_index_bundle,
    load_index_meta,
    resolve_existing_db_path,
    save_index_bundle,
)


class RAGTool:
    """
    RAG tool that uses BAAI/bge-m3 embeddings with FAISS vector store.
    Supports indexing PDF and text documents and semantic search.
    """

    def __init__(
        self,
        chunk_size_words: int = 700,
        chunk_overlap_words: int = 120,
        embedding_batch_size: int = 16,
    ):
        self.index = None
        self.documents: List[Dict] = []
        self.embeddings_model = None
        self._initialized = False
        self._index_meta: Dict = {}

        env_chunk_size = _safe_int(os.getenv("RAG_CHUNK_SIZE_WORDS"), chunk_size_words)
        env_chunk_overlap = _safe_int(os.getenv("RAG_CHUNK_OVERLAP_WORDS"), chunk_overlap_words)
        env_embedding_batch = _safe_int(os.getenv("RAG_EMBEDDING_BATCH_SIZE"), embedding_batch_size)

        self.chunk_size_words = max(200, env_chunk_size)
        self.chunk_overlap_words = max(20, min(env_chunk_overlap, self.chunk_size_words // 2))
        self.embedding_batch_size = max(4, env_embedding_batch)

        # Save new artifacts under /rag by default, but keep legacy fallback.
        self.vector_db_path = os.getenv("RAG_VECTOR_DB_PATH", VECTOR_DB_PATH)
        self.legacy_vector_db_path = "./vectordb"

        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize the BAAI/bge-m3 embedding model."""
        try:
            from FlagEmbedding import FlagModel

            self.embeddings_model = FlagModel(
                EMBEDDING_MODEL,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=False,
            )
            print(f"[RAG] 임베딩 모델 로드 완료: {EMBEDDING_MODEL}")
        except ImportError:
            print("[RAG] FlagEmbedding 패키지가 설치되지 않았습니다.")
            self.embeddings_model = None
        except Exception as e:
            print(f"[RAG] 임베딩 모델 로드 실패: {e}")
            self.embeddings_model = None

    def _encode(self, texts: List[str]) -> Optional[object]:
        """Encode texts into embeddings."""
        if self.embeddings_model is None:
            return None
        try:
            import numpy as np

            embeddings = self.embeddings_model.encode(texts)
            return np.array(embeddings, dtype="float32")
        except Exception as e:
            print(f"[RAG] 임베딩 인코딩 실패: {e}")
            return None

    def _collect_supported_files(self, docs_path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        pdf_files = glob.glob(os.path.join(docs_path, "**/*.pdf"), recursive=True)
        txt_files = glob.glob(os.path.join(docs_path, "**/*.txt"), recursive=True)
        md_files = glob.glob(os.path.join(docs_path, "**/*.md"), recursive=True)
        all_files = pdf_files + txt_files + md_files
        return pdf_files, txt_files, md_files, all_files

    def index_documents(self, docs_path: str = DOCS_PATH):
        """
        Load and index PDF and text files from the specified directory.
        """
        if not os.path.exists(docs_path):
            print(f"[RAG] 문서 디렉토리 없음: {docs_path}")
            return

        pdf_files, txt_files, md_files, all_files = self._collect_supported_files(docs_path)
        if not all_files:
            print(f"[RAG] {docs_path} 에서 문서를 찾을 수 없습니다. RAG를 건너뜁니다.")
            return

        corpus_fingerprint = compute_corpus_fingerprint(sorted(all_files))
        meta = self._load_index_meta()
        if (
            meta
            and meta.get("corpus_fingerprint") == corpus_fingerprint
            and meta.get("chunk_size_words") == self.chunk_size_words
            and meta.get("chunk_overlap_words") == self.chunk_overlap_words
            and meta.get("embedding_model") == EMBEDDING_MODEL
        ):
            if self.load_index():
                print("[RAG] 기존 인덱스 재사용: 문서 변경 없음 (청킹/임베딩 스킵)")
                return

        raw_docs: List[Dict] = []
        for filepath in pdf_files:
            raw_docs.extend(load_pdf(filepath, self.chunk_size_words, self.chunk_overlap_words))
        for filepath in txt_files + md_files:
            raw_docs.extend(load_text(filepath, self.chunk_size_words, self.chunk_overlap_words))

        if not raw_docs:
            print("[RAG] 로드된 문서가 없습니다.")
            return

        print(
            f"[RAG] 총 {len(raw_docs)} 청크 인덱싱 시작 "
            f"(chunk={self.chunk_size_words}, overlap={self.chunk_overlap_words}, batch={self.embedding_batch_size})"
        )
        self._index_meta = {
            "created_at_unix": int(time.time()),
            "docs_path": os.path.abspath(docs_path),
            "file_count": len(all_files),
            "corpus_fingerprint": corpus_fingerprint,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_model": EMBEDDING_MODEL,
            "total_chunks": len(raw_docs),
        }
        self._build_index(raw_docs)

    def _build_index(self, raw_docs: List[Dict]):
        """Build FAISS index from document chunks with batched embedding for large corpora."""
        try:
            import faiss

            self.index = None
            total_docs = len(raw_docs)
            dimension = None
            index = None

            for start in range(0, total_docs, self.embedding_batch_size):
                end = min(start + self.embedding_batch_size, total_docs)
                batch_docs = raw_docs[start:end]
                batch_texts = [doc["content"] for doc in batch_docs]
                batch_embeddings = self._encode(batch_texts)

                if batch_embeddings is None:
                    print("[RAG] 임베딩 실패로 인덱스를 구축할 수 없습니다.")
                    return

                if index is None:
                    dimension = batch_embeddings.shape[1]
                    index = faiss.IndexFlatIP(dimension)

                faiss.normalize_L2(batch_embeddings)
                index.add(batch_embeddings)
                print(f"[RAG] 임베딩 진행: {end}/{total_docs} 청크")

            if index is None or dimension is None:
                print("[RAG] 인덱스를 구축할 수 없습니다. 청크가 비어 있습니다.")
                return

            self.index = index
            self.documents = raw_docs
            self._initialized = True
            print(f"[RAG] FAISS 인덱스 구축 완료: {len(raw_docs)} 청크, 차원={dimension}")
            self._save_index()

        except ImportError:
            print("[RAG] faiss-cpu 패키지가 설치되지 않았습니다.")
        except Exception as e:
            print(f"[RAG] FAISS 인덱스 구축 실패: {e}")

    def _save_index(self):
        """Save index/documents/metadata to the configured vector db path."""
        try:
            save_index_bundle(self.index, self.documents, self._index_meta, self.vector_db_path)
            print(f"[RAG] 인덱스 저장 완료: {self.vector_db_path}")
        except Exception as e:
            print(f"[RAG] 인덱스 저장 실패: {e}")

    def _load_index_meta(self) -> Dict:
        """
        Load index metadata from current path; fallback to legacy path if needed.
        """
        existing_path = resolve_existing_db_path(self.vector_db_path, self.legacy_vector_db_path)
        if existing_path:
            return load_index_meta(existing_path)
        return {}

    def load_index(self) -> bool:
        """Load a previously saved FAISS index from disk."""
        try:
            existing_path = resolve_existing_db_path(self.vector_db_path, self.legacy_vector_db_path)
            if not existing_path:
                return False

            bundle = load_index_bundle(existing_path)
            if bundle is None:
                return False
            self.index, self.documents = bundle
            self._initialized = True

            if existing_path != self.vector_db_path:
                print(f"[RAG] 레거시 인덱스 로드: {existing_path}")
            print(f"[RAG] 저장된 인덱스 로드 완료: {len(self.documents)} 청크")
            return True
        except Exception as e:
            print(f"[RAG] 인덱스 로드 실패: {e}")
            return False

    def search(self, query: str, k: int = RAG_TOP_K, min_score: float = 0.5) -> List[Dict]:
        """
        Perform semantic search over indexed documents.
        """
        if not self._initialized or self.index is None:
            print("[RAG] 인덱스가 초기화되지 않았습니다. 웹 검색만 사용합니다.")
            return []
        if not self.documents:
            return []

        try:
            import faiss

            query_embedding = self._encode([query])
            if query_embedding is None:
                return []

            faiss.normalize_L2(query_embedding)
            fetch_k = min(k * 3, len(self.documents))
            scores, indices = self.index.search(query_embedding, fetch_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score < min_score:
                    continue
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append(
                        {
                            "content": doc.get("content", ""),
                            "source": doc.get("source", "내부 문서"),
                            "score": float(score),
                            "chunk_id": doc.get("chunk_id"),
                            "metadata": doc.get("metadata", {}),
                        }
                    )
                if len(results) >= k:
                    break

            if not results:
                print(f"[RAG] 쿼리 '{query}' 에 대한 관련 문서 없음 (min_score={min_score})")
            return results
        except Exception as e:
            print(f"[RAG] 검색 실패 (query='{query}'): {e}")
            return []


def _safe_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default

