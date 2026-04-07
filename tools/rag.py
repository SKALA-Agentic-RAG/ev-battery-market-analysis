"""
RAG (Retrieval-Augmented Generation) tool using BAAI/bge-m3 embeddings and FAISS.
Indexes documents from the ./docs/ directory and provides semantic search.
"""

import os
import warnings
from typing import Dict, List, Optional

from config import DOCS_PATH, EMBEDDING_MODEL, RAG_TOP_K, VECTOR_DB_PATH


class RAGTool:
    """
    RAG tool that uses BAAI/bge-m3 embeddings with FAISS vector store.
    Supports indexing PDF and text documents and semantic search.
    """

    def __init__(self):
        self.index = None
        self.documents = []  # list of {"content": str, "source": str}
        self.embeddings_model = None
        self._initialized = False
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

    def index_documents(self, docs_path: str = DOCS_PATH):
        """
        Load and index PDF and text files from the specified directory.

        Args:
            docs_path: Path to directory containing documents
        """
        import glob

        if not os.path.exists(docs_path):
            print(f"[RAG] 문서 디렉토리 없음: {docs_path}")
            return

        # Collect all supported files
        pdf_files = glob.glob(os.path.join(docs_path, "**/*.pdf"), recursive=True)
        txt_files = glob.glob(os.path.join(docs_path, "**/*.txt"), recursive=True)
        md_files = glob.glob(os.path.join(docs_path, "**/*.md"), recursive=True)
        all_files = pdf_files + txt_files + md_files

        if not all_files:
            print(f"[RAG] {docs_path} 에서 문서를 찾을 수 없습니다. RAG를 건너뜁니다.")
            return

        raw_docs = []

        # Process PDF files
        for filepath in pdf_files:
            chunks = self._load_pdf(filepath)
            raw_docs.extend(chunks)

        # Process text/markdown files
        for filepath in txt_files + md_files:
            chunks = self._load_text(filepath)
            raw_docs.extend(chunks)

        if not raw_docs:
            print("[RAG] 로드된 문서가 없습니다.")
            return

        print(f"[RAG] 총 {len(raw_docs)} 청크 인덱싱 중...")

        # Build FAISS index
        self._build_index(raw_docs)

    def _load_pdf(self, filepath: str) -> List[Dict]:
        """Load and chunk a PDF file."""
        chunks = []
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            for chunk in self._chunk_text(text):
                                chunks.append({
                                    "content": chunk,
                                    "source": f"{os.path.basename(filepath)} (p.{page_num + 1})",
                                })
            except ImportError:
                # Fallback: try pypdf
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            for chunk in self._chunk_text(text):
                                chunks.append({
                                    "content": chunk,
                                    "source": f"{os.path.basename(filepath)} (p.{page_num + 1})",
                                })
                except ImportError:
                    print(f"[RAG] PDF 파싱 라이브러리 없음 (PyPDF2 또는 pypdf 필요): {filepath}")
        except Exception as e:
            print(f"[RAG] PDF 로드 실패 ({filepath}): {e}")
        return chunks

    def _load_text(self, filepath: str) -> List[Dict]:
        """Load and chunk a text or markdown file."""
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                for chunk in self._chunk_text(text):
                    chunks.append({
                        "content": chunk,
                        "source": os.path.basename(filepath),
                    })
        except Exception as e:
            print(f"[RAG] 텍스트 로드 실패 ({filepath}): {e}")
        return chunks

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(words):
                break
            start += chunk_size - overlap
        return chunks

    def _build_index(self, raw_docs: List[Dict]):
        """Build FAISS index from document chunks."""
        try:
            import faiss
            import numpy as np

            texts = [doc["content"] for doc in raw_docs]
            embeddings = self._encode(texts)

            if embeddings is None:
                print("[RAG] 임베딩 실패로 인덱스를 구축할 수 없습니다.")
                return

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            self.documents = raw_docs
            self._initialized = True

            print(f"[RAG] FAISS 인덱스 구축 완료: {len(raw_docs)} 청크, 차원={dimension}")

            # Optionally save index
            self._save_index()

        except ImportError:
            print("[RAG] faiss-cpu 패키지가 설치되지 않았습니다.")
        except Exception as e:
            print(f"[RAG] FAISS 인덱스 구축 실패: {e}")

    def _save_index(self):
        """Save the FAISS index and documents to disk."""
        try:
            import faiss
            import json

            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            faiss.write_index(self.index, os.path.join(VECTOR_DB_PATH, "index.faiss"))

            with open(os.path.join(VECTOR_DB_PATH, "documents.json"), "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            print(f"[RAG] 인덱스 저장 완료: {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"[RAG] 인덱스 저장 실패: {e}")

    def load_index(self) -> bool:
        """Load a previously saved FAISS index from disk."""
        try:
            import faiss
            import json

            index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
            docs_path = os.path.join(VECTOR_DB_PATH, "documents.json")

            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                return False

            self.index = faiss.read_index(index_path)
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            self._initialized = True
            print(f"[RAG] 저장된 인덱스 로드 완료: {len(self.documents)} 청크")
            return True
        except Exception as e:
            print(f"[RAG] 인덱스 로드 실패: {e}")
            return False

    def search(self, query: str, k: int = RAG_TOP_K) -> List[Dict]:
        """
        Perform semantic search over indexed documents.

        Args:
            query: Search query
            k: Number of top results to return

        Returns:
            List of dicts with keys: content, source, score
            Returns empty list if no index or no documents found.
        """
        if not self._initialized or self.index is None:
            print("[RAG] 인덱스가 초기화되지 않았습니다. 웹 검색만 사용합니다.")
            return []

        if not self.documents:
            return []

        try:
            import faiss
            import numpy as np

            # Encode query
            query_embedding = self._encode([query])
            if query_embedding is None:
                return []

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search
            actual_k = min(k, len(self.documents))
            scores, indices = self.index.search(query_embedding, actual_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "content": doc["content"],
                        "source": doc["source"],
                        "score": float(score),
                    })

            return results

        except Exception as e:
            print(f"[RAG] 검색 실패 (query='{query}'): {e}")
            return []
