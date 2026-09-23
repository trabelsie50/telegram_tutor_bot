"""
rag_engine.py

محرك RAG: تهيئة ChromaDB مع نموذج التضمين المحلي،
إضافة المستندات (نصوص الدروس المعتمدة)، واسترجاع القطع الأكثر صلة
بسؤال التلميذ. يضمن أن جميع الإجابات مبنية فقط على المحتوى المخزن
من المنهج التونسي (CNP).

الملفات المخططة: rag_engine.py
يصدّر: RAGEngine, add_document, retrieve_relevant, initialize_chroma
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaClientSettings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

from config import get_settings
from prompts import ANSWER_PROMPT_TEMPLATE, QUIZ_PROMPT_TEMPLATE, LESSON_EXPLANATION_PROMPT

# ── إعداد السجل ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── الثوابت الافتراضية ────────────────────────────────────────────────────
DEFAULT_COLLECTION_NAME: str = "tunisian_curriculum"
DEFAULT_EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_TOP_K_RESULTS: int = 5

# مثيل عالمي للمحرك (يُستخدم من الدوال على مستوى الوحدة)
_default_engine: Optional["RAGEngine"] = None


# ── دوال مساعدة داخلية ───────────────────────────────────────────────────

def _split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """تقسيم النص إلى قطع صغيرة مع مراعاة حدود الجمل."""
    if not text or len(text.strip()) == 0:
        return []

    sentence_endings = re.compile(r"(?<=[.!؟؟])\s+")
    sentences: List[str] = sentence_endings.split(text)

    chunks: List[str] = []
    current_chunk: str = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_chunk and len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = current_chunk + " " + sentence
        elif not current_chunk and len(sentence) <= chunk_size:
            current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(sentence) > chunk_size:
                words = sentence.split()
                temp: str = ""
                for word in words:
                    if not temp:
                        temp = word
                    elif len(temp) + len(word) + 1 <= chunk_size:
                        temp = temp + " " + word
                    else:
                        chunks.append(temp)
                        temp = word
                current_chunk = temp
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            prev_words = prev.split()
            overlap_words = prev_words[-overlap:] if len(prev_words) >= overlap else prev_words
            prefix = " ".join(overlap_words)
            if prefix not in curr:
                overlapped_chunks.append(prefix + " " + curr)
            else:
                overlapped_chunks.append(curr)
        chunks = overlapped_chunks

    return chunks


def _generate_chunk_id(prefix: str, index: int) -> str:
    """توليد معرّف فريد لقطعة نصية."""
    return f"{prefix}_chunk_{index}"


# ── الصنف الرئيسي ─────────────────────────────────────────────────────────

class RAGEngine:
    """محرك RAG: إدارة ChromaDB مع نموذج التضمين المحلي."""

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K_RESULTS,
    ):
        self.persist_directory: str = persist_directory
        self.embedding_model_name: str = embedding_model_name
        self.collection_name: str = collection_name
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.top_k: int = top_k

        self._embedding_model: Optional[SentenceTransformer] = None
        os.makedirs(self.persist_directory, exist_ok=True)

        _settings = get_settings()
        self.llm: ChatOpenAI = ChatOpenAI(
            model=_settings.openai_model,
            api_key=_settings.openai_api_key,
            temperature=0.4,
        )

        self.chroma_client: chromadb.Client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaClientSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
            ),
        )

        try:
            self.collection: chromadb.Collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info("Loaded existing ChromaDB collection: %s", self.collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created new ChromaDB collection: %s", self.collection_name)

        logger.info("RAGEngine initialized successfully.")

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info("Loading local embedding model: %s...", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Local embedding model loaded successfully.")
        return self._embedding_model

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[str]:
        if not text or not text.strip():
            return []

        chunks: List[str] = _split_text_into_chunks(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return []

        base_metadata: Dict[str, Any] = metadata or {}
        base_metadata.setdefault("source", "manual")
        base_metadata.setdefault("subject", "unknown")
        base_metadata.setdefault("lesson_name", "unknown")

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            if document_id:
                chunk_id: str = _generate_chunk_id(document_id, i)
            else:
                chunk_id = _generate_chunk_id(f"doc_{abs(hash(text)) % (10 ** 8)}", i)

            chunk_metadata: Dict[str, Any] = dict(base_metadata)
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_char_length"] = len(chunk)

            embedding: List[float] = self.embedding_model.encode(
                chunk, normalize_embeddings=True
            ).tolist()

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append(chunk_metadata)

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        return ids

    def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
        subject_filter: Optional[str] = None,
        lesson_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        k: int = top_k if top_k is not None else self.top_k
        where_filter: Optional[Dict[str, Any]] = None
        if subject_filter or lesson_filter:
            where_filter = {}
            if subject_filter:
                where_filter["subject"] = subject_filter
            if lesson_filter:
                where_filter["lesson_name"] = lesson_filter

        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
        )

        retrieved: List[Dict[str, Any]] = []
        ids: List[str] = results.get("ids", [[]])[0]
        documents: List[str] = results.get("documents", [[]])[0]
        distances: List[float] = results.get("distances", [[]])[0]
        metadatas: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            retrieved.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "distance": distances[i] if i < len(distances) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "similarity": (1.0 - distances[i] if distances[i] is not None else 0.0),
                }
            )
        return retrieved

    def retrieve_for_quiz(
        self,
        lesson_name: str,
        subject: str,
        top_k: Optional[int] = None,
    ) -> str:
        results = self.retrieve_relevant(
            query=lesson_name,
            top_k=top_k,
            subject_filter=subject,
            lesson_filter=lesson_name,
        )
        results.sort(key=lambda chunk: chunk.get("metadata", {}).get("chunk_index", 0))
        if top_k:
            results = results[:top_k]
        return "\n".join(chunk["text"] for chunk in results if chunk["text"])

    def generate_answer(
        self,
        question: str,
        subject: str,
        lesson_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        context_chunks = self.retrieve_relevant(
            query=question,
            top_k=top_k,
            subject_filter=subject,
            lesson_filter=lesson_name,
        )
        context = "\n\n".join(chunk["text"] for chunk in context_chunks if chunk["text"])
        if not context.strip():
            return f"عذراً، لم أتمكن من العثور على معلومات كافية للإجابة عن سؤال: {question}"
        
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question, subject=subject)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


# ── دوال مستوى الوحدة (تُعرّف بعد الصنف لضمان الرؤية) ─────────────────────────

def initialize_chroma(
    persist_directory: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> RAGEngine:
    settings = get_settings()
    if persist_directory is None:
        persist_directory = settings.chroma_persist_directory
    if embedding_model_name is None:
        embedding_model_name = DEFAULT_EMBEDDING_MODEL

    return RAGEngine(
        persist_directory=persist_directory,
        embedding_model_name=embedding_model_name,
        collection_name=collection_name
    )


def add_document(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
) -> List[str]:
    global _default_engine
    if _default_engine is None:
        _default_engine = initialize_chroma()
    return _default_engine.add_document(text, metadata=metadata, document_id=document_id)


def retrieve_relevant(
    query: str,
    top_k: Optional[int] = None,
    subject_filter: Optional[str] = None,
    lesson_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    global _default_engine
    if _default_engine is None:
        _default_engine = initialize_chroma()
    return _default_engine.retrieve_relevant(
        query=query,
        top_k=top_k,
        subject_filter=subject_filter,
        lesson_filter=lesson_filter,
    )
