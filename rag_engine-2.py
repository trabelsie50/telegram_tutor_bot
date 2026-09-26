"""
rag_engine.py

محرك RAG: تهيئة ChromaDB مع embeddings عبر Gemini API (مجاني، بدون نموذج
PyTorch محلي)، إضافة المستندات (نصوص الدروس المعتمدة)، واسترجاع القطع
الأكثر صلة بسؤال التلميذ. الشرح/الاختبار/تقييم الإجابات تُنفَّذ عبر Gemini
أيضاً (نموذج gemini-2.5-flash)، بديل كامل لـ OpenAI.

تمت إزالة sentence-transformers/torch بالكامل من هذا الملف — كانت السبب
الأساسي في تجاوز حد الذاكرة (Out of Memory) على Render (512MB)، وGemini
API يقوم بحساب الـ embeddings على سيرفراته بدل تحميل نموذج محلي ثقيل.
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional

import chromadb
import google.generativeai as genai

from config import get_settings
from prompts import (
    ANSWER_PROMPT_TEMPLATE,
    QUIZ_JSON_PROMPT_TEMPLATE,
    LESSON_EXPLANATION_PROMPT,
    ANSWER_EVAL_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME: str = "tunisian_curriculum"
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_TOP_K_RESULTS: int = 5

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
    return f"{prefix}_chunk_{index}"


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """يستخرج كائن JSON من رد النموذج، حتى لو أضاف Markdown أو نصاً حول الـ JSON."""
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error("تعذّر تحليل JSON من رد النموذج: %s", cleaned[:300])

    return {}


class RAGEngine:
    """محرك RAG: ChromaDB + Gemini API (embeddings وتوليد النصوص)."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K_RESULTS,
    ):
        self.persist_directory: str = persist_directory
        self.collection_name: str = collection_name
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.top_k: int = top_k

        os.makedirs(self.persist_directory, exist_ok=True)

        settings = get_settings()
        if not settings.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY غير موجود في متغيرات البيئة!")
        genai.configure(api_key=settings.gemini_api_key)

        self.model_name: str = settings.gemini_model
        self.embedding_model_name: str = settings.gemini_embedding_model
        self.generative_model = genai.GenerativeModel(self.model_name)

        self.chroma_client: chromadb.Client = chromadb.PersistentClient(
            path=self.persist_directory
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

        logger.info("RAGEngine initialized successfully (Gemini API).")

    # ── مساعدات Gemini ──────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """يستدعي Gemini لتوليد نص، مع إعادة محاولة بسيطة عند تجاوز حدود المعدل."""
        for attempt in range(2):
            try:
                response = self.generative_model.generate_content(prompt)
                return (response.text or "").strip()
            except Exception as exc:
                logger.warning("فشل استدعاء Gemini (محاولة %d): %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(2)
                else:
                    raise

    def _embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """يحسب embeddings عبر Gemini API لقائمة نصوص (بدون أي نموذج محلي)."""
        task_type = "retrieval_query" if is_query else "retrieval_document"
        embeddings: List[List[float]] = []
        for text in texts:
            result = genai.embed_content(
                model=f"models/{self.embedding_model_name}",
                content=text,
                task_type=task_type,
            )
            embeddings.append(result["embedding"])
        return embeddings

    # ── إدارة المحتوى ────────────────────────────────────────────────────

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

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)

        embeddings = self._embed(documents, is_query=False)

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

        query_embedding = self._embed([query], is_query=True)[0]

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
        if not results:
            results = self.retrieve_relevant(
                query=lesson_name,
                top_k=top_k,
                subject_filter=subject,
            )
        results.sort(key=lambda chunk: chunk.get("metadata", {}).get("chunk_index", 0))
        if top_k:
            results = results[:top_k]
        return "\n".join(chunk["text"] for chunk in results if chunk["text"])

    # ── الوظائف التربوية (Gemini) ────────────────────────────────────────

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
        return self._generate(prompt)

    def explain_lesson(self, lesson_name: str, subject: str) -> str:
        lesson_text = self.retrieve_for_quiz(lesson_name, subject, top_k=6)
        if not lesson_text.strip():
            return (
                f"عذراً يا بطل، لم أجد درس '{lesson_name}' في مادة {subject} ضمن المحتوى "
                f"المخزَّن حالياً. تأكد من اسم الدرس، أو اطلب من معلمك إضافته عبر 📸 إضافة درس (OCR)."
            )

        prompt = LESSON_EXPLANATION_PROMPT.format(lesson_text=lesson_text)
        return self._generate(prompt)

    def generate_quiz(
        self,
        lesson_name: str,
        subject: str,
        num_questions: int = 3,
        difficulty: str = "medium",
    ) -> Dict[str, Any]:
        lesson_text = self.retrieve_for_quiz(lesson_name, subject, top_k=6)
        if not lesson_text.strip():
            return {"questions": []}

        prompt = QUIZ_JSON_PROMPT_TEMPLATE.format(
            lesson_text=lesson_text,
            subject=subject,
            lesson_title=lesson_name,
            num_questions=num_questions,
        )
        content = self._generate(prompt)
        parsed = _extract_json_object(content)
        questions = parsed.get("questions", [])

        valid_questions: List[Dict[str, Any]] = []
        for q in questions:
            if isinstance(q, dict) and q.get("question") and q.get("expected_answer"):
                valid_questions.append(q)

        return {"questions": valid_questions[:num_questions]}

    def evaluate_answer(
        self,
        question: str,
        expected_answer: str,
        explanation: str,
        student_answer: str,
    ) -> Dict[str, Any]:
        prompt = ANSWER_EVAL_PROMPT_TEMPLATE.format(
            question=question,
            expected_answer=expected_answer,
            explanation=explanation,
            student_answer=student_answer,
        )
        content = self._generate(prompt)
        parsed = _extract_json_object(content)

        if "is_correct" not in parsed:
            is_correct = any(
                word.lower() in student_answer.lower()
                for word in expected_answer.split()
                if len(word) > 2
            )
            return {"is_correct": is_correct, "feedback": ""}

        return {
            "is_correct": bool(parsed.get("is_correct")),
            "feedback": parsed.get("feedback", "") or "",
        }


# ── دوال مستوى الوحدة ─────────────────────────────────────────────────────

def initialize_chroma(
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> RAGEngine:
    settings = get_settings()
    if persist_directory is None:
        persist_directory = settings.chroma_persist_directory

    return RAGEngine(
        persist_directory=persist_directory,
        collection_name=collection_name,
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
