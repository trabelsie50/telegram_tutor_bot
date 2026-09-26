"""
bot_handlers.py

جميع معالجات رسائل بوت تيليغرام المخصص لمعلم السنة الخامسة ابتدائي في تونس.
يتضمن التعامل مع الأوامر، الرسائل النصية والصوتية، الأزرار التفاعلية،
وضع الاختبار الذكي، شرح الدروس، تقارير الأولياء، ووضع مراجعة نصوص OCR/PDF/Word.
"""

import os
import re
import asyncio
import logging
import tempfile
import uuid
from typing import Dict, Any, Optional, List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from config import get_settings, Settings
from prompts import (
    SYSTEM_PROMPT,
    CURRICULUM_REFUSAL_MESSAGE,
    LESSON_EXPLANATION_PROMPT,
    QUIZ_PROMPT_TEMPLATE,
    REVIEW_PROMPT,
    PARENT_REPORT_PROMPT,
)
from database import StudentDB, get_db_path, get_database
from rag_engine import RAGEngine, initialize_chroma
from ocr_engine import extract_text_from_image
from document_engine import extract_text_from_document
from voice_utils import transcribe_telegram_voice, text_to_speech

# إعداد السجلات Logging
logger = logging.getLogger(__name__)

# تهيئة المحركات الأساسية
settings: Settings = get_settings()
rag_engine: RAGEngine = initialize_chroma()
db_instance: StudentDB = StudentDB()

# تخزين مؤقت لمراجعة النصوص الاستخرجية عبر OCR/PDF/Word
# المفتاح: doc_id -> القيمة: dict يحتوي البيانات والصورة والنص
PENDING_REVIEWS: Dict[str, Dict[str, Any]] = {}

# تخزين المحتوى المعتمد
APPROVED_CONTENT: Dict[str, Dict[str, Any]] = {}

# صيغ الملفات المدعومة لاستخراج الدروس منها (غير الصور)
SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx")


# -----------------------------------------------------------------------------
# لوحات المفاتيح والأزرار المساعدة (Keyboards)
# -----------------------------------------------------------------------------

def get_main_keyboard() -> ReplyKeyboardMarkup:
    """توليد لوحة المفاتيح الرئيسية للبوت."""
    keyboard = [
        [KeyboardButton("📖 شرح درس"), KeyboardButton("🧠 اختبار ذكي")],
        [KeyboardButton("📊 تقرير الأولياء"), KeyboardButton("📸 إضافة درس (OCR)")],
        [KeyboardButton("⚙️ اختيار المادة")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def get_subjects_inline_keyboard() -> InlineKeyboardMarkup:
    """توليد أزرار اختيار المادة التعليمية."""
    keyboard = [
        [
            InlineKeyboardButton("🔢 رياضيات", callback_data="sub_math"),
            InlineKeyboardButton("🇹🇳 لغة عربية", callback_data="sub_arabic"),
        ],
        [
            InlineKeyboardButton("🇫🇷 لغة فرنسية", callback_data="sub_french"),
            InlineKeyboardButton("🔬 إيقاظ علمي", callback_data="sub_science"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# -----------------------------------------------------------------------------
# مساعد الرد الموحّد — نص دائماً، + صوت إذا كان التلميذ يتواصل بالصوت
# -----------------------------------------------------------------------------

async def send_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    parse_mode: Optional[str] = None,
) -> None:
    """يرسل رداً نصياً دائماً، ويضيف نسخة صوتية (TTS) إذا كان التلميذ في "وضع صوتي"
    (أي آخر رسالة بعثها كانت صوتية) — لتفعيل التفاعل الصوتي ذهاباً وعودة.
    """
    if not update.message:
        return

    await update.message.reply_text(text, parse_mode=parse_mode)

    if not context.user_data.get("voice_mode"):
        return

    audio_path: Optional[str] = None
    try:
        # تنظيف رموز Markdown قبل التحويل لصوت، وتحديد طول النص المُلفوظ
        clean_text = re.sub(r"[*_`#]", "", text).strip()
        if not clean_text:
            return
        audio_path = await asyncio.to_thread(text_to_speech, clean_text[:600], "ar")
        with open(audio_path, "rb") as audio_file:
            await update.message.reply_voice(voice=audio_file)
    except Exception as e:
        logger.error(f"خطأ أثناء توليد/إرسال الرد الصوتي: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# -----------------------------------------------------------------------------
# معالج الأمر الرئيسي /start
# -----------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    معالجة الأمر /start
    ترحيب بالتلميذ بالدارجة التونسية المشجعة وتسجيل بياناته في قاعدة البيانات.
    """
    if not update.effective_user or not update.effective_chat:
        return

    user = update.effective_user
    student_id = user.id
    first_name = user.first_name or "يا بطل"

    # تسجيل/تحديث التلميذ في قاعدة البيانات
    try:
        db_instance.add_or_update_student(
            student_id=student_id,
            name=first_name,
            grade="السنة الخامسة ابتدائي"
        )
    except Exception as e:
        logger.error(f"خطأ أثناء تسجيل التلميذ {student_id}: {e}")

    welcome_text = (
        f"أهلاً وسهلاً بك يا {first_name}! 🖐️\n"
        f"أنا أستاذك الخصوصي الذكي للسنة الخامسة ابتدائي 🇹🇳\n\n"
        f"مرحباً بك في عالم النجاح والتميز! أنا هنا لمساعدتك في أربعة مواد أساسية:\n"
        f"🔹 الرياضيات\n"
        f"🔹 اللغة العربية\n"
        f"🔹 اللغة الفرنسية (Français)\n"
        f"🔹 الإيقاظ العلمي\n\n"
        f"اختر المادة التي تريد أن نبدأ بها اليوم من الأزرار أسفله:"
    )

    context.user_data["current_subject"] = "رياضيات"  # المادة الافتراضية
    context.user_data["mode"] = "idle"
    context.user_data["voice_mode"] = False

    await update.message.reply_text(
        welcome_text,
        reply_markup=get_main_keyboard()
    )
    await update.message.reply_text(
        "اختر المادة التعليمية:",
        reply_markup=get_subjects_inline_keyboard()
    )


# -----------------------------------------------------------------------------
# معالجة استجابات الأزرار التفاعلية (Callback Queries)
# -----------------------------------------------------------------------------

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """معالجة النقر على الأزرار الشفافة Inline Buttons."""
    query = update.callback_query
    if not query:
        return

    await query.answer()
    data = query.data

    if data.startswith("sub_"):
        subject_map = {
            "sub_math": "رياضيات",
            "sub_arabic": "لغة عربية",
            "sub_french": "لغة فرنسية",
            "sub_science": "إيقاظ علمي"
        }
        selected_subject = subject_map.get(data, "رياضيات")
        context.user_data["current_subject"] = selected_subject

        response_text = f"ممتاز جـداً! اخترنا مادة: **{selected_subject}** 📚\nماذا تريد أن نفعل الآن؟"
        await query.edit_message_text(text=response_text, parse_mode="Markdown")

    elif data.startswith("review_approve_"):
        await review_approve_callback(update, context)

    elif data.startswith("review_edit_"):
        await review_edit_callback(update, context)


# -----------------------------------------------------------------------------
# وضع شرح الدروس (Lesson Explanation Mode)
# -----------------------------------------------------------------------------

async def start_lesson_explanation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """بدء وضع شرح الدرس مع التلميذ."""
    context.user_data["mode"] = "explain"
    subject = context.user_data.get("current_subject", "رياضيات")

    msg = (
        f"يعطيك الصحة! نحن الآن في وضع **شرح الدروس** لمادة **{subject}** 📖\n"
        f"اكتب لي اسم الدرس أو السؤال الذي لم تفهمه جيداً في الفصل، وسأشرحه لك خطوة بخطوة بالشيّق والمبسط!"
    )
    await send_reply(update, context, msg, parse_mode="Markdown")


# -----------------------------------------------------------------------------
# وضع الاختبار الذكي (Smart Quiz Mode)
# -----------------------------------------------------------------------------

async def start_smart_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """بدء وضع الاختبار الذكي التفاعلي."""
    context.user_data["mode"] = "awaiting_quiz_topic"
    subject = context.user_data.get("current_subject", "رياضيات")

    msg = (
        f"ممتاز! حان وقت التحدي والاختبار الذكي في مادة **{subject}** 🧠⚡\n"
        f"اكتب لي عنوان الدرس الذي تريد أن نختبر معلوماتك فيه الآن:"
    )
    await send_reply(update, context, msg, parse_mode="Markdown")


async def execute_smart_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE, lesson_name: str) -> None:
    """توليد الاختبار الذكي المكون من 3 أسئلة متدرجة استناداً إلى RAG."""
    subject = context.user_data.get("current_subject", "رياضيات")

    await update.message.reply_text(f"جاري إعداد الأسئلة المتدرجة لدرس '{lesson_name}' من المنهج الرسمـي... ⏳")

    try:
        quiz_data = await asyncio.to_thread(
            rag_engine.generate_quiz,
            lesson_name=lesson_name,
            subject=subject,
            num_questions=3,
            difficulty="medium",
        )

        questions = quiz_data.get("questions", [])
        if not questions:
            await send_reply(
                update,
                context,
                "عذراً يا بطل، لم أجد معلومات كافية في كتب المنهج لهذا الدرس لتوليد اختبار. حاول اختيار درس آخر!",
            )
            context.user_data["mode"] = "idle"
            return

        context.user_data["current_quiz"] = {
            "lesson_name": lesson_name,
            "subject": subject,
            "questions": questions,
            "current_index": 0,
            "score": 0,
            "total": len(questions),
            "errors": []
        }
        context.user_data["mode"] = "in_quiz"

        await send_next_quiz_question(update, context)

    except Exception as e:
        logger.error(f"خطأ أثناء توليد الاختبار: {e}")
        await update.message.reply_text("حدث خطأ أثناء إعداد الاختبار. يرجى المحاولة مرة أخرى.")
        context.user_data["mode"] = "idle"


async def send_next_quiz_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال السؤال التالي في الاختبار الذكي."""
    quiz_info = context.user_data.get("current_quiz")
    if not quiz_info:
        return

    idx = quiz_info["current_index"]
    questions = quiz_info["questions"]

    if idx >= len(questions):
        # انتهى الاختبار
        await finish_quiz(update, context)
        return

    q_item = questions[idx]
    q_num = idx + 1
    diff_labels = {"easy": "سهل 🟢", "medium": "متوسط 🟡", "hard": "صعب 🔴"}
    diff = diff_labels.get(q_item.get("difficulty", "medium"), "متوسط 🟡")

    text = (
        f"📌 **السؤال رقم {q_num} من {len(questions)}** ({diff}):\n\n"
        f"{q_item.get('question')}\n\n"
        f"✍️ اكتب إجابتك الآن بصوتك أو بنص عادي:"
    )

    await send_reply(update, context, text, parse_mode="Markdown")


async def process_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, user_answer: str) -> None:
    """تقييم إجابة التلميذ بالذكاء الاصطناعي (متسامح مع الصياغة المختلفة وأخطاء النطق الصوتي)
    وتسجيل الأخطاء الشائعة إن وجدت.
    """
    quiz_info = context.user_data.get("current_quiz")
    if not quiz_info:
        return

    idx = quiz_info["current_index"]
    q_item = quiz_info["questions"][idx]
    student_id = update.effective_user.id if update.effective_user else 0

    question_text = q_item.get("question", "")
    expected_answer = q_item.get("expected_answer", "")
    explanation = q_item.get("explanation", "")
    error_cat = q_item.get("error_category", "أخطاء عامة")

    try:
        eval_result = await asyncio.to_thread(
            rag_engine.evaluate_answer, question_text, expected_answer, explanation, user_answer
        )
        is_correct = bool(eval_result.get("is_correct"))
        model_feedback = eval_result.get("feedback", "")
    except Exception as e:
        logger.error(f"خطأ أثناء التقييم الذكي للإجابة، استخدام فحص احتياطي مبسّط: {e}")
        is_correct = any(
            word.lower() in user_answer.lower() for word in expected_answer.split() if len(word) > 2
        )
        model_feedback = ""

    if is_correct:
        quiz_info["score"] += 1
        feedback = f"يعطيك الصحة! إجابة صحيحة وممتازة 👏🎉\n\n💡 **توضيح:** {explanation}"
    else:
        quiz_info["errors"].append(error_cat)
        feedback = (
            f"يعطيك الصحة على المحاولة يا بطل! 👏\n"
            f"الإجابة الدقيقة هي: **{expected_answer}**\n\n"
            f"💡 **الشرح والتوضيح:** {explanation}\n"
            f"لاحظنا نقطة نحتاج لمراجعتها: ({error_cat})."
        )
        # تسجيل نوع الخطأ في SQLite
        try:
            db_instance.record_error_type(
                student_id=student_id,
                error_category=error_cat,
                lesson_name=quiz_info["lesson_name"],
                subject=quiz_info["subject"]
            )
        except Exception as e:
            logger.error(f"خطأ أثناء تسجيل نوع الخطأ: {e}")

    if model_feedback:
        feedback += f"\n\n🗣️ {model_feedback}"

    await send_reply(update, context, feedback, parse_mode="Markdown")

    quiz_info["current_index"] += 1
    await send_next_quiz_question(update, context)


async def finish_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إنهاء الاختبار وتسجيل النتيجة النهائية وإرسال التشجيع."""
    quiz_info = context.user_data.get("current_quiz")
    if not quiz_info:
        return

    student_id = update.effective_user.id if update.effective_user else 0
    score = quiz_info["score"]
    total = quiz_info["total"]
    lesson_name = quiz_info["lesson_name"]
    subject = quiz_info["subject"]
    errors = quiz_info["errors"]

    # حفظ النتيجة في قاعدة البيانات
    try:
        db_instance.record_quiz(
            student_id=student_id,
            lesson_name=lesson_name,
            subject=subject,
            score=score,
            total_questions=total,
            details={"errors_detected": errors},
            error_categories=errors
        )
    except Exception as e:
        logger.error(f"خطأ أثناء حفظ نتيجة الاختبار: {e}")

    # التقدير بالدارجة
    if score == total:
        encouragement = "ممتاز جداً ومعلم! أحسنت العلامة الكاملة 💯🌟"
    elif score >= total / 2:
        encouragement = "عمل ممتاز يا بطل! واصل واجتهد وستصبح الأفضل دائماً 💪"
    else:
        encouragement = "محاولة طيبة! لا تقلق، بالمراجعة والتمارين ستصبح ممتازاً في هذا الدرس 👍"

    summary_text = (
        f"🏁 **انتهى الاختبار الذكي!**\n\n"
        f"📚 الدرس: {lesson_name}\n"
        f"🎯 النتيجة: {score} من {total}\n\n"
        f"{encouragement}"
    )

    context.user_data["mode"] = "idle"
    context.user_data.pop("current_quiz", None)

    await send_reply(update, context, summary_text, parse_mode="Markdown")


# -----------------------------------------------------------------------------
# وضع تقارير الأولياء (Parent Report Mode)
# -----------------------------------------------------------------------------

async def request_parent_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """طلب واستخراج تقرير أداء التلميذ الموجه للولي."""
    student_id = update.effective_user.id if update.effective_user else 0
    await send_parent_report(update, context, student_id)


async def send_parent_report(update: Update, context: ContextTypes.DEFAULT_TYPE, student_id: int) -> None:
    """إرسال تقرير مبسط ومفصل لأولياء الأمور."""
    try:
        report_data = db_instance.generate_parent_report(student_id)
        if isinstance(report_data, dict):
            student_info = report_data.get("student", {})
            student_name = student_info.get("name", "التلميذ")
            total_quizzes = report_data.get("total_quizzes", 0)
            avg_score = report_data.get("average_score_pct", 0)
            common_errors = report_data.get("common_errors", [])

            errors_formatted = "\n".join([f"• {err['category']} ({err['count']} مرات)" for err in common_errors]) or "لا يوجد أخطاء متكررة ملحوظة 🎉"

            report_msg = (
                f"📊 **تقرير أداء التلميذ(ة): {student_name}**\n"
                f"🏫 المستوى: السنة الخامسة ابتدائي (CNP)\n"
                f"-----------------------------------\n"
                f"📝 عدد الاختبارات المجراة: {total_quizzes}\n"
                f"📈 معدل النجاح في الاختبارات: {avg_score:.1f}%\n\n"
                f"⚠️ **النقاط التي تحتاج إلى تعزيز ومراجعة:**\n"
                f"{errors_formatted}\n\n"
                f"💡 **توصية الأستاذ:** ينصح بمراجعة المفاهيم المتعلقة بالأخطاء المذكورة وتأكيده بالتطبيقات اليومية."
            )
        else:
            report_msg = str(report_data)

        if update.message:
            await update.message.reply_text(report_msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"خطأ أثناء إنشاء تقرير الأولياء: {e}")
        if update.message:
            await update.message.reply_text("عذراً، حدث خطأ أثناء إعداد التقرير.")


# -----------------------------------------------------------------------------
# وضع مراجعة النصوص الاستخرجية (OCR / PDF / Word Review Mode)
# -----------------------------------------------------------------------------

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    معالجة رفع صور صفحات الكتب المدرسية، استخراج النص بـ OCR،
    وإرسالها للمشرف لتدقيقها وإقرارها قبل الإضافة لـ ChromaDB.
    """
    if not update.message or not update.message.photo:
        return

    await update.message.reply_text("جاري استخراج النص من الصورة باستخدام OCR... 🔍")

    photo_file = await update.message.photo[-1].get_file()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
        img_path = tmp_img.name

    try:
        await photo_file.download_to_drive(img_path)

        ocr_result = extract_text_from_image(img_path)
        extracted_text = ocr_result.get("text", "").strip()
        confidence = ocr_result.get("confidence", 0.0)

        if not extracted_text:
            await update.message.reply_text("لم أتمكن من استخراج نص واضح من هذه الصورة. يرجى إعادة التقاط الصورة بوضوح أعلى.")
            return

        doc_id = str(uuid.uuid4())[:8]
        PENDING_REVIEWS[doc_id] = {
            "text": extracted_text,
            "confidence": confidence,
            "subject": context.user_data.get("current_subject", "عام")
        }

        await prompt_for_lesson_metadata(update, context, doc_id)

    except Exception as e:
        logger.error(f"خطأ أثناء معالجة صورة OCR: {e}")
        await update.message.reply_text("حدث خطأ أثناء معالجة الصورة.")
    finally:
        # تنظيف الصورة المؤقتة دائماً (كانت تتراكم على القرص بدون هذا)
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass


async def handle_document_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """معالجة رفع ملفات PDF أو Word (.docx): استخراج نصها لمراجعتها واعتمادها في RAG."""
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    file_name = doc.file_name or "ملف"
    ext = os.path.splitext(file_name)[1].lower()

    await update.message.reply_text(f"جاري تحليل الملف '{file_name}' واستخراج نصه... 📄⏳")

    doc_file = await doc.get_file()
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex[:8]}{ext}")

    try:
        await doc_file.download_to_drive(tmp_path)
        result = extract_text_from_document(tmp_path)
        extracted_text = result.get("text", "").strip()

        if not extracted_text:
            await update.message.reply_text(
                "لم أتمكن من استخراج أي نص من هذا الملف. تأكد أنه غير فارغ أو محمي/مشفّر."
            )
            return

        doc_id = str(uuid.uuid4())[:8]
        needs_review = result.get("needs_review", False)
        PENDING_REVIEWS[doc_id] = {
            "text": extracted_text,
            "confidence": 60.0 if needs_review else 100.0,
            "subject": context.user_data.get("current_subject", "عام"),
        }

        if needs_review:
            await update.message.reply_text(
                "⚠️ الملف يحتوي على صفحات ممسوحة (صور) تمت معالجتها بـ OCR، يُفضّل التدقيق قبل الاعتماد."
            )

        await prompt_for_lesson_metadata(update, context, doc_id)

    except ValueError as e:
        # صيغة غير مدعومة (مثلاً .doc القديم)
        await update.message.reply_text(f"⚠️ {e}")
    except Exception as e:
        logger.error(f"خطأ أثناء معالجة الملف {file_name}: {e}")
        await update.message.reply_text("حدث خطأ أثناء معالجة الملف. تأكد من أنه غير تالف.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def prompt_for_lesson_metadata(update: Update, context: ContextTypes.DEFAULT_TYPE, doc_id: str) -> None:
    """يطلب من المشرف اسم الدرس بالضبط قبل عرض معاينة الاعتماد النهائية.

    ضروري عند رفع كتب/فصول كاملة (بدل درس واحد فقط): بدون وسم lesson_name دقيق،
    البحث الدلالي في retrieve_for_quiz/explain_lesson قد يخلط بين درسين متقاربين
    في الموضوع بدل جلب الدرس المطلوب بالضبط.
    """
    context.user_data["awaiting_lesson_name_for"] = doc_id
    await update.message.reply_text(
        "✏️ شنو اسم الدرس بالضبط لهذا المحتوى؟ (مثلاً: \"القسمة على رقمين\")\n"
        "هذا يساعد البوت يلقى الدرس بدقة وقت ما التلميذ يطلبه."
    )


async def lesson_name_received_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, lesson_name: str) -> None:
    """يستقبل اسم الدرس المكتوب من المشرف، ثم يطلب تحديد نوع المحتوى (درس/دليل معلم)."""
    doc_id = context.user_data.pop("awaiting_lesson_name_for", None)
    if not doc_id or doc_id not in PENDING_REVIEWS:
        await update.message.reply_text("⚠️ انتهت جلسة الرفع. يرجى إعادة رفع الملف/الصورة من جديد.")
        return

    lesson_name = lesson_name.strip()
    if len(lesson_name) < 2:
        await update.message.reply_text("⚠️ اسم الدرس قصير جداً. أعد كتابته:")
        context.user_data["awaiting_lesson_name_for"] = doc_id
        return

    PENDING_REVIEWS[doc_id]["lesson_name"] = lesson_name

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📖 محتوى درس (للتلميذ)", callback_data=f"ctype_lesson_{doc_id}"),
            InlineKeyboardButton("👨‍🏫 دليل معلم (طريقة تدريس)", callback_data=f"ctype_guide_{doc_id}"),
        ]
    ])
    await update.message.reply_text(
        f"📚 الدرس: **{lesson_name}**\nهذا المحتوى هو:",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


async def content_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يستقبل اختيار نوع المحتوى (درس للتلميذ / دليل معلم)، ويعرض المعاينة النهائية
    مع أزرار الاعتماد/التعديل."""
    query = update.callback_query
    await query.answer()

    if query.data.startswith("ctype_lesson_"):
        doc_id = query.data.replace("ctype_lesson_", "")
        content_type = "lesson"
        content_type_label = "محتوى درس (للتلميذ)"
    else:
        doc_id = query.data.replace("ctype_guide_", "")
        content_type = "teacher_guide"
        content_type_label = "دليل معلم (طريقة تدريس)"

    review_data = PENDING_REVIEWS.get(doc_id)
    if not review_data:
        await query.edit_message_text("⚠️ لم يعد هذا المراجعة موجودة.")
        return

    review_data["content_type"] = content_type

    extracted_text = review_data["text"]
    confidence = review_data["confidence"]
    subject = review_data["subject"]
    lesson_name = review_data.get("lesson_name", "غير محدد")

    preview = extracted_text[:800] + ("…" if len(extracted_text) > 800 else "")

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ اعتماد النص", callback_data=f"review_approve_{doc_id}"),
            InlineKeyboardButton("✏️ تعديل النص", callback_data=f"review_edit_{doc_id}")
        ]
    ])

    review_msg = (
        f"🔍 **مراجعة نهائية قبل الاعتماد**\n"
        f"-----------------------------------\n"
        f"📚 المادة: {subject}\n"
        f"📖 الدرس: {lesson_name}\n"
        f"🏷️ النوع: {content_type_label}\n"
        f"📊 درجة الثقة: {confidence:.1f}%\n\n"
        f"📝 **معاينة النص:**\n\n`{preview}`\n\n"
        f"هل تعتمد هذا النص لإضافته للـ RAG أم ترغب في تعديله؟"
    )

    await query.edit_message_text(review_msg, reply_markup=keyboard, parse_mode="Markdown")


async def review_approve_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    doc_id = query.data.replace("review_approve_", "")
    review_data = PENDING_REVIEWS.get(doc_id)

    if not review_data:
        await query.edit_message_text("⚠️ لم يعد هذا المراجعة موجودة.")
        return

    approved_text = review_data["text"]
    subject = review_data["subject"]
    confidence = review_data["confidence"]
    lesson_name = review_data.get("lesson_name")
    content_type = review_data.get("content_type", "lesson")

    APPROVED_CONTENT[doc_id] = {
        "text": approved_text,
        "subject": subject,
        "confidence": confidence,
        "lesson_name": lesson_name,
        "content_type": content_type,
        "approved_by": str(query.from_user.id),
        "approved_at": __import__("datetime").datetime.now().isoformat()
    }

    try:
        await add_document_to_rag(approved_text, subject, lesson_name=lesson_name, content_type=content_type)
    except Exception as e:
        logger.error(f"خطأ أثناء إضافة المستند إلى RAG: {e}")

    del PENDING_REVIEWS[doc_id]

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 عرض المحتوى المعتمد", callback_data="view_approved")]
    ])

    await query.edit_message_text(
        f"✅ **تم اعتماد النص بنجاح**\n\n"
        f"📚 المادة: {subject}\n"
        f"📖 الدرس: {lesson_name or 'غير محدد'}\n"
        f"🏷️ النوع: {'دليل معلم' if content_type == 'teacher_guide' else 'محتوى درس'}\n"
        f"📊 درجة الثقة: {confidence:.1f}%\n"
        f"🆔 معرّف المراجعة: `{doc_id}`\n\n"
        f"تم إضافة النص إلى قاعدة RAG.",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def review_edit_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    doc_id = query.data.replace("review_edit_", "")
    review_data = PENDING_REVIEWS.get(doc_id)

    if not review_data:
        await query.edit_message_text("⚠️ لم يعد هذا المراجعة موجودة.")
        return

    context.user_data["editing_doc_id"] = doc_id
    context.user_data["editing_mode"] = True

    await query.edit_message_text(
        "✏️ **وضع التعديل**\n\n"
        "أرسل النص المعدّل الآن. سيحل محل النص المستخرج من الصورة.\n"
        "أو أرسل `/cancel` لإلغاء التعديل.",
        parse_mode="Markdown"
    )


async def edited_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("editing_mode"):
        return

    doc_id = context.user_data.get("editing_doc_id")
    if not doc_id or doc_id not in PENDING_REVIEWS:
        await update.message.reply_text("⚠️ انتهت جلسة التعديل. يرجى بدء مراجعة جديدة.")
        context.user_data["editing_mode"] = False
        return

    edited_text = update.message.text

    if len(edited_text.strip()) < 5:
        await update.message.reply_text("⚠️ النص المعدّل قصير جداً. يرجى إدخال نص صالح.")
        return

    PENDING_REVIEWS[doc_id]["text"] = edited_text
    PENDING_REVIEWS[doc_id]["confidence"] = 100.0
    context.user_data["editing_mode"] = False
    context.user_data.pop("editing_doc_id", None)

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ اعتماد النص المعدّل", callback_data=f"review_approve_{doc_id}"),
            InlineKeyboardButton("🔄 إعادة التعديل", callback_data=f"review_edit_{doc_id}")
        ]
    ])

    await update.message.reply_text(
        f"✅ **تم حفظ التعديل**\n\n"
        f"📝 النص المعدّل:\n\n`{edited_text}`\n\n"
        f"هل تريد اعتماد هذا النص أو إعادة تعديله؟",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def review_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query:
        await query.answer()
        await query.edit_message_text("❌ تم إلغاء المراجعة.")
    else:
        await update.message.reply_text("❌ تم إلغاء المراجعة.")

    context.user_data["editing_mode"] = False
    context.user_data.pop("editing_doc_id", None)


async def view_approved_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if not APPROVED_CONTENT:
        await query.edit_message_text("📭 لا يوجد محتوى معتمد حالياً.")
        return

    lines = ["📚 **المحتوى المعتمد**\n" + "-" * 30]
    for doc_id, data in APPROVED_CONTENT.items():
        lines.append(
            f"🆔 `{doc_id}` | 📚 {data['subject']} | "
            f"📊 {data['confidence']:.1f}% | ✅ {data.get('approved_by', '?')}"
        )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]
    ])

    await query.edit_message_text(
        "\n".join(lines),
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def main_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📷 إرسال صورة للـ OCR", callback_data="ocr_mode")],
        [InlineKeyboardButton("📝 إرسال نص مباشر", callback_data="text_mode")],
        [InlineKeyboardButton("📚 عرض المحتوى المعتمد", callback_data="view_approved")],
        [InlineKeyboardButton("🗑️ مسح الذاكرة", callback_data="clear_memory")]
    ])

    await query.edit_message_text(
        "🏠 **القائمة الرئيسية**\n\n"
        "اختر إجراءً:",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def ocr_mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data["mode"] = "ocr"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="main_menu")]
    ])

    await query.edit_message_text(
        "📷 **وضع OCR**\n\n"
        "أرسل صورة تحتوي على نص عربي، أو ملف PDF/Word، وسأقوم باستخراج نصه.\n"
        "بعد الاستخراج، سأعرض النص لمراجعة واعتماد قبل إضافته للـ RAG.",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def text_mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data["mode"] = "text"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="main_menu")]
    ])

    await query.edit_message_text(
        "📝 **وضع النص المباشر**\n\n"
        "أرسل النص مباشرة وسأضيفه للـ RAG فوراً.\n"
        "يمكنك تحديد المادة بإرسال:\n"
        "`/subject [اسم المادة]`",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def clear_memory_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    context.user_data.clear()
    PENDING_REVIEWS.clear()

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="main_menu")]
    ])

    await query.edit_message_text(
        "🗑️ **تم مسح الذاكرة**\n\n"
        "تم مسح بيانات الجلسة والمراجعات المعلقة.",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    mode = context.user_data.get("mode", "text")
    subject = context.user_data.get("current_subject", "عام")

    if mode == "text":
        doc_id = str(uuid.uuid4())[:8]
        APPROVED_CONTENT[doc_id] = {
            "text": text,
            "subject": subject,
            "confidence": 100.0,
            "approved_by": str(update.message.from_user.id),
            "approved_at": __import__("datetime").datetime.now().isoformat()
        }

        try:
            await add_document_to_rag(text, subject)
        except Exception as e:
            logger.error(f"خطأ أثناء إضافة المستند إلى RAG: {e}")

        await update.message.reply_text(
            f"✅ تم إضافة النص للـ RAG بنجاح.\n"
            f"🆔 المعرّف: `{doc_id}`\n"
            f"📚 المادة: {subject}",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "يرجى اختيار وضع العمل من القائمة الرئيسية أولاً.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="main_menu")]
            ])
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = context.user_data.get("mode", "ocr")

    if mode != "ocr":
        await update.message.reply_text("يرجى تفعيل وضع OCR أولاً.")
        return

    await process_ocr_image(update, context)


# -----------------------------------------------------------------------------
# وضع إدارة المحتوى (RAG Admin Mode) — للمشرف فقط، عبر أمر /admin
# -----------------------------------------------------------------------------

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يعرض قائمة إدارة محتوى RAG (OCR/نص مباشر/عرض المعتمد/مسح الذاكرة)."""
    await update.message.reply_text(
        "🏠 **القائمة الرئيسية**\n\n"
        "اختر إجراءً:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📷 إرسال صورة للـ OCR", callback_data="ocr_mode")],
            [InlineKeyboardButton("📝 إرسال نص مباشر", callback_data="text_mode")],
            [InlineKeyboardButton("📚 عرض المحتوى المعتمد", callback_data="view_approved")],
            [InlineKeyboardButton("🗑️ مسح الذاكرة", callback_data="clear_memory")],
        ]),
        parse_mode="Markdown",
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """راوتر الملفات المرفوعة (Document): PDF/Word تُعالَج دائماً كدرس جديد،
    وأي صيغة صورة أُرسلت كـ "ملف" تُعامل كصورة OCR (كانت تفشل بصمت سابقاً).
    """
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    ext = os.path.splitext(doc.file_name or "")[1].lower()

    if ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        await handle_document_upload(update, context)
        return

    if doc.mime_type and doc.mime_type.startswith("image/"):
        # صورة أُرسلت بصيغة "ملف" (بعض الهواتف تفعل هذا) — تُعامَل كصورة OCR مباشرة
        mode = context.user_data.get("mode", "ocr")
        if mode != "ocr":
            await update.message.reply_text("يرجى تفعيل وضع OCR أولاً.")
            return
        await process_ocr_image_from_document(update, context)
        return

    await update.message.reply_text(
        "⚠️ صيغة الملف غير مدعومة. المدعوم حالياً: صور (JPG/PNG)، PDF، وWord (.docx فقط)."
    )


async def process_ocr_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """نقطة دخول موحّدة لمعالجة صور OCR المُستقبلة كـ "Photo" (تُستخدم من handle_photo)."""
    await handle_photo_message(update, context)


async def process_ocr_image_from_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """معالجة صورة أُرسلت كـ "Document" (ملف) بدل "Photo": ننزّلها ثم نستخدم نفس محرك OCR."""
    if not update.message or not update.message.document:
        return

    await update.message.reply_text("جاري استخراج النص من الصورة باستخدام OCR... 🔍")

    doc = update.message.document
    ext = os.path.splitext(doc.file_name or "")[1].lower() or ".jpg"
    doc_file = await doc.get_file()
    img_path = os.path.join(tempfile.gettempdir(), f"docimg_{uuid.uuid4().hex[:8]}{ext}")

    try:
        await doc_file.download_to_drive(img_path)

        ocr_result = extract_text_from_image(img_path)
        extracted_text = ocr_result.get("text", "").strip()
        confidence = ocr_result.get("confidence", 0.0)

        if not extracted_text:
            await update.message.reply_text("لم أتمكن من استخراج نص واضح من هذه الصورة. يرجى إعادة التقاط الصورة بوضوح أعلى.")
            return

        doc_id = str(uuid.uuid4())[:8]
        PENDING_REVIEWS[doc_id] = {
            "text": extracted_text,
            "confidence": confidence,
            "subject": context.user_data.get("current_subject", "عام")
        }

        await prompt_for_lesson_metadata(update, context, doc_id)

    except Exception as e:
        logger.error(f"خطأ أثناء معالجة صورة OCR (كملف): {e}")
        await update.message.reply_text("حدث خطأ أثناء معالجة الصورة.")
    finally:
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass


async def add_document_to_rag(
    text: str,
    subject: str,
    lesson_name: Optional[str] = None,
    content_type: str = "lesson",
) -> List[str]:
    """إضافة نص معتمد إلى قاعدة RAG (تُنفَّذ في خيط مستقل حتى لا تحجب حلقة الأحداث).

    lesson_name: اسم الدرس بالضبط (ضروري للاسترجاع الدقيق عند رفع كتب كاملة).
    content_type: "lesson" (محتوى موجَّه للتلميذ) أو "teacher_guide" (دليل معلم،
    يُستعمل فقط للاستئناس في أسلوب الشرح، لا يُعرض مباشرة على التلميذ).
    """
    metadata: Dict[str, Any] = {
        "subject": subject,
        "grade": "السنة الخامسة ابتدائي",
        "content_type": content_type,
    }
    if lesson_name:
        metadata["lesson_name"] = lesson_name
    return await asyncio.to_thread(rag_engine.add_document, text, metadata)


# -----------------------------------------------------------------------------
# وضع الرسائل الصوتية (Voice Interaction) — تفريغ صوت التلميذ لنص، ورد صوتي
# -----------------------------------------------------------------------------

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يستقبل رسالة صوتية من التلميذ، يحوّلها لنص عبر Whisper المحلي، ثم يمرّرها
    لنفس راوتر الرسائل النصية — وبعدها يرسل البوت ردوده نصاً + صوتاً معاً.
    """
    if not update.message or not update.message.voice:
        return

    context.user_data["voice_mode"] = True  # من الآن، رد البوت يكون نص + صوت

    await update.message.reply_text("🎙️ جاري تحويل رسالتك الصوتية إلى نص...")

    voice_file = await update.message.voice.get_file()
    tmp_path = os.path.join(tempfile.gettempdir(), f"voice_{uuid.uuid4().hex[:8]}.ogg")

    try:
        await voice_file.download_to_drive(tmp_path)
        transcribed = await asyncio.to_thread(transcribe_telegram_voice, tmp_path)

        if not transcribed.strip():
            await update.message.reply_text(
                "عذراً، لم أفهم الرسالة الصوتية بوضوح. حاول التسجيل في مكان أهدأ أو اكتب رسالتك."
            )
            return

        await update.message.reply_text(f"📝 سمعتك تقول: \"{transcribed}\"")

        # نمرّر النص المُفرَّغ مباشرة للراوتر الرئيسي دون تعديل رسالة تيليغرام
        # (كائنات تيليغرام غير قابلة للتعديل في python-telegram-bot v20+)
        await route_text_message(update, context, override_text=transcribed)

    except Exception as e:
        logger.error(f"خطأ أثناء تفريغ الرسالة الصوتية: {e}")
        await update.message.reply_text("حدث خطأ أثناء معالجة الرسالة الصوتية.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# راوتر الرسائل النصية — يوجّه ضغطات لوحة المفاتيح ونص التلميذ حسب الوضع الحالي
# -----------------------------------------------------------------------------

MENU_BUTTON_ROUTES = {
    "📖 شرح درس": start_lesson_explanation,
    "🧠 اختبار ذكي": start_smart_quiz,
    "📊 تقرير الأولياء": request_parent_report,
}


async def route_text_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    override_text: Optional[str] = None,
) -> None:
    """المُوجِّه الرئيسي لكل رسالة نصية عادية من التلميذ (أو نص مُفرَّغ من رسالة صوتية
    عبر override_text — لأن رسائل تيليغرام غير قابلة للتعديل مباشرة).
    """
    if not update.message or (override_text is None and not update.message.text):
        return

    text = (override_text if override_text is not None else update.message.text).strip()

    if override_text is None:
        # التلميذ كتب فعلاً (لم يبعث صوتاً) — نرجع لوضع الرد النصي فقط
        context.user_data["voice_mode"] = False

    # 1) وضع تعديل نص مراجعة OCR/PDF/Word له الأولوية دائماً
    if context.user_data.get("editing_mode"):
        await edited_text_handler(update, context)
        return

    # 2) ضغطات لوحة المفاتيح الرئيسية
    if text in MENU_BUTTON_ROUTES:
        context.user_data["mode"] = "idle"
        await MENU_BUTTON_ROUTES[text](update, context)
        return

    if text == "📸 إضافة درس (OCR)":
        context.user_data["mode"] = "ocr"
        await update.message.reply_text(
            "📷 أرسل الآن صورة صفحة الدرس، أو ملف PDF/Word، وسأستخرج نصها لمراجعتها واعتمادها."
        )
        return

    if text == "⚙️ اختيار المادة":
        await update.message.reply_text(
            "اختر المادة التعليمية:",
            reply_markup=get_subjects_inline_keyboard(),
        )
        return

    # 3) حسب الوضع الحالي المخزَّن من قبل
    mode = context.user_data.get("mode", "idle")

    if mode == "awaiting_quiz_topic":
        await execute_smart_quiz(update, context, lesson_name=text)
        return

    if mode == "in_quiz":
        await process_quiz_answer(update, context, user_answer=text)
        return

    if mode == "explain":
        subject = context.user_data.get("current_subject", "رياضيات")
        await update.message.reply_text("لحظة، جاري تحضير الشرح... ⏳")
        try:
            explanation = await asyncio.to_thread(
                rag_engine.explain_lesson, lesson_name=text, subject=subject
            )
        except Exception as e:
            logger.error(f"خطأ أثناء شرح الدرس: {e}")
            explanation = "عذراً، حدث خطأ أثناء تحضير الشرح. حاول مرة أخرى."
        await send_reply(update, context, explanation, parse_mode="Markdown")
        return

    if mode == "text":
        await handle_text(update, context)
        return

    # 4) الوضع الافتراضي: توجيه التلميذ للقائمة الرئيسية
    await update.message.reply_text(
        "يرجى اختيار وضع العمل من القائمة الرئيسية أولاً 👇",
        reply_markup=get_main_keyboard(),
    )


# -----------------------------------------------------------------------------
# تسجيل جميع المعالجات
# -----------------------------------------------------------------------------

def register_all_handlers(application: Application) -> None:
    """يسجّل جميع معالجات البوت على تطبيق تيليغرام المُعطى."""

    # الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("admin", admin_command))

    # الرسائل النصية والصوتية والصور والمستندات
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, route_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # أزرار اختيار المادة + مراجعة OCR/PDF/Word (اعتماد/تعديل)
    application.add_handler(
        CallbackQueryHandler(handle_callback_query, pattern=r"^(sub_|review_approve_|review_edit_)")
    )
    application.add_handler(CallbackQueryHandler(review_cancel_callback, pattern=r"^review_cancel_"))

    # قائمة إدارة محتوى RAG (/admin)
    application.add_handler(CallbackQueryHandler(main_menu_callback, pattern=r"^main_menu$"))
    application.add_handler(CallbackQueryHandler(ocr_mode_callback, pattern=r"^ocr_mode$"))
    application.add_handler(CallbackQueryHandler(text_mode_callback, pattern=r"^text_mode$"))
    application.add_handler(CallbackQueryHandler(view_approved_callback, pattern=r"^view_approved$"))
    application.add_handler(CallbackQueryHandler(clear_memory_callback, pattern=r"^clear_memory$"))

    logger.info("✅ تم تسجيل جميع معالجات البوت بنجاح.")
