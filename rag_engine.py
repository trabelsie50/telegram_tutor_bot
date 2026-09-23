"""rag_engine.py

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
    """تقسيم النص إلى قطع صغيرة مع مراعاة حدود الجمل.

    يُقسّم النص عند حدود الجمل (نقطة، علامة استفهام، إلخ) ثم يُجمّع
    الجمل في قطع لا يتجاوز طولها ``chunk_size`` حرفاً، مع الحفاظ على
    تداخل ``overlap`` حرف بين القطع المتتالية.
    """
    if not text or len(text.strip()) == 0:
        return []

    # فصل الجمل باستخدام التعبير النمطي
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
            # حفظ القطع الحالية والبدء بقطع جديدة
            if current_chunk:
                chunks.append(current_chunk)

            if len(sentence) > chunk_size:
                # الجملة أطول من حجم القطعة: تقسيمها كلمة بكلمة
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

    # تطبيق التداخل بين القطع
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # إضافة آخر ``overlap`` كلمة من القطعة السابقة
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


# ── دوال مستوى الوحدة ─────────────────────────────────────────────────────

def initialize_chroma(
    persist_directory: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> "RAGEngine":
    """تهيئة ChromaDB وإرجاع محرك RAG جاهز للاستخدام.

    Args:
        persist_directory: مسار تخزين ChromaDB الدائم. إذا لم يُحدَّد،
            يُؤخذ من ``settings.chroma_persist_directory()``.
        embedding_model_name: اسم نموذج التضمين المحلي. إذا لم يُحدَّد،
            يُستخدم ``paraphrase-multilingual-MiniLM-L12-v2``.
        collection_name: اسم مجموعة ChromaDB.

    Returns:
        مثيل ``RAGEngine`` مُهيَّأ بالكامل.
    """
    settings = get_settings()
    if persist_directory is None:
        persist_directory = settings.chroma_persist_directory()
    if embedding_model_name is None:
        embedding_model_name = DEFAULT_EMBEDDING_MODEL

    engine = RAGEngine(
        persist_directory=persist_directory,
        embedding_model_name=embedding_model_name,
        collection_name=collection_name,
    )
    logger.info(
        "ChromaDB initialized: persist_dir=%s, collection=%s",
        persist_directory,
        collection_name,
    )
    return engine


def add_document(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
) -> List[str]:
    """إضافة مستند (نص درس معتمد) إلى قاعدة ChromaDB.

    دالة مساعدة على مستوى الوحدة تستخدم المثيل العالمي للمحرك.

    Args:
        text: النص الكامل للدرس.
        metadata: بيانات وصفية اختيارية (المادة، اسم الدرس، الصف...).
        document_id: معرّف اختياري للمستند.

    Returns:
        قائمة بمعرّفات القطع المُضافة.
    """
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
    """استرجاع القطع الأكثر صلة بسؤال التلميذ.

    دالة مساعدة على مستوى الوحدة تستخدم المثيل العالمي للمحرك.

    Args:
        query: سؤال التلميذ.
        top_k: عدد القطع المراد استرجاعها.
        subject_filter: فلتر حسب المادة (رياضيات، فرنسية، عربية، إيقاظ علمي).
        lesson_filter: فلتر حسب اسم الدرس.

    Returns:
        قائمة قواميس تحتوي على النص والمسافة والبيانات الوصفية.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = initialize_chroma()
    return _default_engine.retrieve_relevant(
        query=query,
        top_k=top_k,
        subject_filter=subject_filter,
        lesson_filter=lesson_filter,
    )


# ── الصنف الرئيسي ─────────────────────────────────────────────────────────

class RAGEngine:
    """محرك RAG: إدارة ChromaDB مع نموذج التضمين المحلي (مع تفعيل التحميل عند الطلب).

    Args:
        persist_directory: مسار تخزين ChromaDB الدائم.
        embedding_model_name: اسم نموذج التضمين المحلي من sentence-transformers.
        collection_name: اسم مجموعة ChromaDB.
        chunk_size: الحجم الأقصى لقطعة النص (بالأحرف).
        chunk_overlap: حجم التداخل بين القطع (بالكلمات).
        top_k: عدد النتائج الافتراضي للاسترجاع.
    """

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

        # متغير داخلي لتخزين النموذج مؤقتاً عند تحميله لأول مرة
        self._embedding_model: Optional[SentenceTransformer] = None

        # التأكد من وجود دليل التخزين
        os.makedirs(self.persist_directory, exist_ok=True)

        # 🛑 تم حذف التحميل الفوري من هنا لتوفير الـ RAM عند الإقلاع
        # سيتم تحميل نموذج التضمين المحلي فقط عند أول طلب عبر الخاصية embedding_model

        # تهيئة نموذج اللغة (LLM) عبر OpenAI لتوليد الإجابات والاختبارات
        _settings = get_settings()
        self.llm: ChatOpenAI = ChatOpenAI(
            model=_settings.openai_model,
            api_key=_settings.openai_api_key,
            temperature=0.4,
        )

        # تهيئة عميل ChromaDB
        self.chroma_client: chromadb.Client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaClientSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
            ),
        )

        # الحصول على المجموعة أو إنشائها
        try:
            self.collection: chromadb.Collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(
                "Loaded existing ChromaDB collection: %s", self.collection_name
            )
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Created new ChromaDB collection: %s", self.collection_name
            )

        logger.info("RAGEngine initialized successfully (Lazy Loading enabled).")

    @property
    def embedding_model(self) -> SentenceTransformer:
        """خاصية ذكية تقوم بتحميل نموذج التضمين محلياً فقط عند استخدامه لأول مرة (Lazy Loading)."""
        if self._embedding_model is None:
            logger.info("Loading local embedding model: %s (on demand)...", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Local embedding model loaded successfully.")
        return self._embedding_model

    # ── إضافة المستندات ──────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[str]:
        """إضافة مستند (نص درس) إلى قاعدة ChromaDB بعد تقسيمه إلى قطع.

        يُقسّم النص تلقائياً إلى قطع صغيرة، ويُولّد تضميناً لكل قطعة،
        ويُخزّنها في ChromaDB مع البيانات الوصفية.

        Args:
            text: النص الكامل للدرس المستخرج من OCR والمعتمد.
            metadata: بيانات وصفية مثل:
                ``{"subject": "رياضيات", "lesson_name": "الضرب", "grade": "السنة الخامسة"}``
            document_id: معرّف اختياري للمستند الأصلي.

        Returns:
            قائمة بمعرّفات القطع المُضافة (strings).
        """
        if not text or not text.strip():
            logger.warning("Attempted to add an empty document. Skipping.")
            return []

        chunks: List[str] = _split_text_into_chunks(
            text, self.chunk_size, self.chunk_overlap
        )
        if not chunks:
            logger.warning("No chunks generated from the document text.")
            return []

        base_metadata: Dict[str, Any] = metadata or {}
        # التأكد من وجود حقول وصفية أساسية
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
                chunk_id = _generate_chunk_id(
                    f"doc_{abs(hash(text)) % (10 ** 8)}", i
                )

            chunk_metadata: Dict[str, Any] = dict(base_metadata)
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_char_length"] = len(chunk)

            # توليد التضمين باستخدام النموذج المحلي (سيتم تحفيز التحميل هنا تلقائياً عند أول استدعاء لـ self.embedding_model)
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
            logger.info(
                "Added %d chunks to collection '%s' (document: %s).",
                len(ids),
                self.collection_name,
                document_id or "auto",
            )

        return ids

    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """إضافة عدة مستندات دفعة واحدة.

        Args:
            documents: قائمة قواميس، كل منها يحتوي على:
                ``text`` (str): النص.
                ``metadata`` (dict, اختياري): البيانات الوصفية.
                ``document_id`` (str, اختياري): المعرّف.

        Returns:
            قائمة بكل معرّفات القطع المُضافة (مسطحة).
        """
        all_ids: List[str] = []
        for doc in documents:
            text: str = doc.get("text", "")
            metadata: Optional[Dict[str, Any]] = doc.get("metadata")
            document_id: Optional[str] = doc.get("document_id")
            chunk_ids = self.add_document(
                text=text, metadata=metadata, document_id=document_id
            )
            all_ids.extend(chunk_ids)
        logger.info("Batch added %d total chunks for %d documents.", len(all_ids), len(documents))
        return all_ids

    # ── استرجاع المعلومات ────────────────────────────────────────────────

    def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
        subject_filter: Optional[str] = None,
        lesson_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """استرجاع القطع الأكثر صلة بسؤال التلميذ.

        يستخدم نموذج التضمين لتحويل السؤال إلى متجه، ثم يبحث في
        ChromaDB عن القطع الأقرب تشابهاً (cosine similarity).

        Args:
            query: سؤال التلميذ (نص).
            top_k: عدد القطع المراد إرجاعها (الافتراضي: ``self.top_k``).
            subject_filter: فلتر اختياري حسب المادة
                (``رياضيات`` / ``فرنسية`` / ``عربية`` / ``إيقاظ علمي``).
            lesson_filter: فلتر اختياري حسب اسم الدرس بالضبط.

        Returns:
            قائمة من القواميس، كل منها يحتوي على:
                ``id`` (str): معرّف القطعة.
                ``text`` (str): نص القطعة.
                ``distance`` (float): المسافة cosine (أقل = أكثر صلة).
                ``similarity`` (float): درجة التشابه (أعلى = أكثر صلة).
                ``metadata`` (dict): البيانات الوصفية للقطعة.
        """
        if not query or not query.strip():
            logger.warning("Empty query received. Returning empty results.")
            return []

        k: int = top_k if top_k is not None else self.top_k

        # بناء فلتر WHERE لـ ChromaDB
        where_filter: Optional[Dict[str, Any]] = None
        if subject_filter or lesson_filter:
            where_filter = {}
            if subject_filter:
                where_filter["subject"] = subject_filter
            if lesson_filter:
                where_filter["lesson_name"] = lesson_filter

        # ملاحظة: إذا قمت بالبحث، سيتم استدعاء self.embedding_model ضمناً لتوليد متجه السؤال
        # (يمكنك أيضاً استخدام استعلام نصي مباشر عبر chroma إذا أردت، لكننا سنحافظ على طريقة التضمين المحلية لتوافق النماذج)
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        # تنفيذ الاستعلام بالمتجه المولد
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
        )

        # تحليل النتائج
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
                    "similarity": (
                        1.0 - distances[i]
                        if distances[i] is not None
                        else 0.0
                    ),
                }
            )

        logger.info(
            "Retrieved %d relevant chunks for query (subject=%s, lesson=%s).",
            len(retrieved),
            subject_filter or "none",
            lesson_filter or "none",
        )
        return retrieved

    def retrieve_for_quiz(
        self,
        lesson_name: str,
        subject: str,
        top_k: Optional[int] = None,
    ) -> str:
        """استرجاع النص الكامل لدرس معين لاستخدامه في توليد الاختبار."""
        results = self.retrieve_relevant(
            query=lesson_name,
            top_k=top_k,
            subject_filter=subject,
            lesson_filter=lesson_name,
        )
        results.sort(key=lambda chunk: chunk.get("metadata", {}).get("chunk_index", 0))
        if top_k:
            results = results[:top_k]
        return "\n".join(
            chunk["text"] for chunk in results if chunk["text"]
        )

    def generate_answer(
        self,
        question: str,
        subject: str,
        lesson_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """توليد إجابة على سؤال باستخدام السياق المسترجع."""
        context_chunks = self.retrieve_relevant(
            query=question,
            top_k=top_k,
            subject_filter=subject,
            lesson_filter=lesson_name,
        )
        context = "\n\n".join(
            chunk["text"] for chunk in context_chunks if chunk["text"]
        )
        if not context.strip():
            return (
                "عذراً، لم أتمكن من العثور على معلومات كافية "
                f"في قاعدة المعرفة للإجابة عن سؤال: {question}"
            )
        prompt = self._format_prompt(
            prompt_template=ANSWER_PROMPT_TEMPLATE,
            context=context,
            question=question,
            subject=subject,
        )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def explain_lesson(
        self,
        lesson_name: str,
        subject: str,
        top_k: Optional[int] = None,
    ) -> str:
        """شرح درس كامل للتلميذ بالاستناد إلى نصه المخزَّن في المنهج فقط."""
        reference_text = self.retrieve_for_quiz(
            lesson_name=lesson_name,
            subject=subject,
            top_k=top_k,
        )
        if not reference_text.strip():
            return (
                "عذراً يا بطل، لم أجد معلومات كافية في كتب المنهج عن "
                f"'{lesson_name}'. جرّب تكتب اسم الدرس بشكل مختلف!"
            )
        prompt = self._format_prompt(
            prompt_template=LESSON_EXPLANATION_PROMPT,
            lesson_text=reference_text,
        )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def generate_quiz(
        self,
        lesson_name: str,
        subject: str,
        num_questions: int = 3,
        difficulty: str = "medium",
    ) -> List[Dict[str, str]]:
        """توليد اختبار من محتوى درس معين."""
        reference_text = self.retrieve_for_quiz(
            lesson_name=lesson_name,
            subject=subject,
        )
        if not reference_text.strip():
            logger.warning(
                "No content found for lesson '%s' (subject=%s).",
                lesson_name,
                subject,
            )
            return []
        prompt = self._format_prompt(
            prompt_template=QUIZ_PROMPT_TEMPLATE,
            subject=subject,
            lesson_text=reference_text,
            lesson_title=lesson_name,
        )
        response = self.llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return self._parse_quiz_response(raw)

    def _parse_quiz_response(
        self, raw_text: str
    ) -> List[Dict[str, str]]:
        """تحليل نص الاستجابة إلى قائمة أسئلة منظمة."""
        import json
        import re

        try:
            match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
            if match:
                raw_text = match.group(1)
            data = json.loads(raw_text)
            if isinstance(data, dict):
                data = [data]
            questions = []
            for item in data:
                question = item.get("question", "")
                options = item.get("options", [])
                correct = item.get("correct_answer", "")
                if question and options:
                    questions.append(
                        {
                            "question": question,
                            "options": (
                                options
                                if isinstance(options, list)
                                else [options]
                            ),
                            "correct_answer": correct,
                        }
                    )
            return questions
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            logger.error("Failed to parse quiz response: %s", exc)
            return []

    def _format_prompt(
        self,
        prompt_template: str,
        context: str,
        question: str,
        subject: str,
        **kwargs,
    ) -> str:
        """إدراج القيم في قالب التلميح (prompt template)."""
        replacements = {
            "context": context,
            "question": question,
            "subject": subject,
            **kwargs,
        }
        try:
            return prompt_template.format(**replacements)
        except KeyError as exc:
            logger.warning(
                "Missing placeholder %s in prompt template.", exc
            )
            return prompt_template


def main():
    """نقطة الدخول الرئيسية لتشغيل محرك RAG من سطر الأوامر."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Engine: استرجاع وتوليد إجابات ذكية"
    )
    parser.add_argument(
        "--mode",
        choices=["answer", "quiz"],
        default="answer",
        help="وضع التشغيل: answer للإجابة على أسئلة، quiz لتوليد اختبار",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="السؤال المراد الإجابة عنه (وضع answer)",
    )
    parser.add_argument(
        "--lesson", "-l",
        type=str,
        help="اسم الدرس لاسترجاع المحتوى أو توليد اختبار",
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        default="general",
        help="المادة (الافتراضي: general)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="عدد القطع المرجعية المراد استرجاعها",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=3,
        help="عدد الأسئلة في الاختبار (وضع quiz)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="مستوى صعوبة الاختبار (وضع quiz)",
    )
    args = parser.parse_args()

    engine = RAGEngine(persist_directory="./chroma_db")

    if args.mode == "answer":
        if not args.question:
            logger.error("الوضع 'answer' يتطلب --question")
            return
        answer = engine.generate_answer(
            question=args.question,
            subject=args.subject,
            top_k=args.top_k,
        )
        print(f"\n{'=' * 60}")
        print(f"السؤال: {args.question}")
        print(f"{'=' * 60}")
        print(answer)
    elif args.mode == "quiz":
        if not args.lesson:
            logger.error("الوضع 'quiz' يتطلب --lesson")
            return
        quiz = engine.generate_quiz(
            lesson_name=args.lesson,
            subject=args.subject,
            num_questions=args.num_questions,
            difficulty=args.difficulty,
        )
        if quiz:
            print(f"\n{'=' * 60}")
            print(f"اختبار درس: {args.lesson} ({args.subject})")
            print(f"{'=' * 60}")
            for i, item in enumerate(quiz, 1):
                print(f"\n{i}. {item['question']}")
                for j, opt in enumerate(item["options"], 1):
                    print(f"   {j}. {opt}")
                print(f"   الإجابة الصحيحة: {item['correct_answer']}")
        else:
            print("لم يتم توليد أي أسئلة.")


if __name__ == "__main__":
    main()
