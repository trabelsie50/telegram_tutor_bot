#!/usr/bin/env python3
"""
main.py - نقطة الدخول الرئيسية لتطبيق بوت الدعم التعليمي التونسي

يقوم بتهيئة جميع المكونات (إعدادات، قاعدة بيانات، RAG، البوت)،
تسجيل المعالجات، وبدء تشغيل بوت تيليغرام.
يتضمن فحص البيئة والتأكد من وجود الملفات المطلوبة قبل التشغيل.
"""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

# ─── إعداد نظام التسجيل ───
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def check_environment(settings) -> bool:
    """
    فحص البيئة والتأكد من وجود المتطلبات الأساسية قبل التشغيل.

    يتحقق من:
    - وجود متغيرات البيئة الضرورية (توكن البوت، مفتاح OpenAI)
    - تثبيت Tesseract OCR
    - إمكانية إنشاء مجلدات ChromaDB وقاعدة البيانات
    - وجود نموذج Whisper المحلي (إن وُدد)

    Returns:
        bool: True إذا كانت البيئة جاهزة، False وإلا
    """
    errors = []
    warnings = []

    # ── التحقق من متغيرات البيئة ──
    if not settings.telegram_bot_token:
        errors.append(
            "❌ TELEGRAM_BOT_TOKEN غير محدد في متغيرات البيئة (.env)"
        )

    if not settings.openai_api_key:
        errors.append(
            "❌ OPENAI_API_KEY غير محدد في متغيرات البيئة (.env)"
        )

    # ── التحقق من Tesseract OCR ──
    if shutil.which("tesseract") is None:
        errors.append(
            "❌ Tesseract OCR غير مثبت. يرجى تثبيته:\n"
            "   Ubuntu/Debian: sudo apt-get install tesseract-ocr "
            "tesseract-ocr-ara tesseract-ocr-fra\n"
            "   macOS: brew install tesseract tesseract-lang"
        )

    # ── التحقق من مجلد ChromaDB ──
    chroma_path = settings.chroma_persist_directory
    try:
        chroma_dir = Path(chroma_path) if chroma_path else Path("chroma_db")
        if not chroma_dir.exists():
            chroma_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"تم إنشاء مجلد ChromaDB: {chroma_dir}")
    except OSError as e:
        errors.append(f"❌ تعذر إنشاء مجلد ChromaDB ({chroma_path}): {e}")

    # ── التحقق من مجلد قاعدة البيانات ──
    try:
        db_dir = Path("data")
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"تم إنشاء مجلد قاعدة البيانات: {db_dir}")
    except OSError as e:
        errors.append(f"❌ تعذر إنشاء مجلد قاعدة البيانات: {e}")

    # ── التحقق من نموذج Whisper المحلي ──
    whisper_path = settings.whisper_model_path
    if whisper_path:
        whisper_file = Path(whisper_path)
        if not whisper_file.exists():
            warnings.append(
                f"⚠️ نموذج Whisper غير موجود في: {whisper_path}\n"
                "   يرجى تحميله قبل استخدام وضع تحويل الصوت.\n"
                "   مثال: wget https://openai.com/documents/whisper-small.pt"
            )

    # ── عرض النتائج ──
    for warning in warnings:
        logger.warning(warning)

    for error in errors:
        logger.error(error)

    if errors:
        return False

    return True


def initialize_database():
    """تهيئة قاعدة بيانات SQLite وإنشاء الجداول"""
    logger.info("جاري تهيئة قاعدة البيانات...")
    from database import init_db, get_database

    init_db()
    db = get_database()
    logger.info("✅ تم تهيئة قاعدة البيانات بنجاح.")
    return db


def initialize_rag_engine(settings):
    """تهيئة محرك RAG مع ChromaDB"""
    logger.info("جاري تهيئة محرك RAG (ChromaDB)...")
    from rag_engine import initialize_chroma

    rag_engine = initialize_chroma(
        persist_directory=settings.chroma_persist_directory,
    )
    logger.info("✅ تم تهيئة محرك RAG بنجاح.")
    return rag_engine


def build_application(settings):
    """بناء تطبيق بوت تيليغرام وتسجيل المعالجات"""
    from telegram.ext import Application

    logger.info("جاري بناء تطبيق البوت...")

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .build()
    )

    # تسجيل جميع المعالجات
    from bot_handlers import register_all_handlers
    register_all_handlers(application)

    logger.info("✅ تم بناء التطبيق وتسجيل المعالجات بنجاح.")
    return application


def main():
    """
    نقطة الدخول الرئيسية للتطبيق.

    خطوات التنفيذ:
    1. تحميل الإعدادات من متغيرات البيئة
    2. فحص البيئة والتأكد من وجود المتطلبات
    3. تهيئة قاعدة البيانات
    4. تهيئة محرك RAG
    5. بناء تطبيق البوت
    6. بدء تشغيل البوت (polling)
    """

    # ── الخطوة 1: تحميل الإعدادات ──
    logger.info("=" * 60)
    logger.info("🚀 بدء تشغيل بوت الدعم التعليمي التونسي...")
    logger.info("=" * 60)

    from config import get_settings
    settings = get_settings()
    logger.info("تم تحميل الإعدادات بنجاح.")

    # ── الخطوة 2: فحص البيئة ──
    logger.info("جاري فحص البيئة...")
    if not check_environment(settings):
        logger.error(
            "\n❌ فحص البيئة فشل. يرجى مراجعة الأخطاء أعلاه "
            "وإصلاحها قبل تشغيل البوت."
        )
        sys.exit(1)

    logger.info("✅ فحص البيئة تم بنجاح.")

    # ── الخطوة 3: تهيئة قاعدة البيانات ──
    try:
        db = initialize_database()
    except Exception as e:
        logger.error(f"❌ فشل تهيئة قاعدة البيانات: {e}")
        sys.exit(1)

    # ── الخطوة 4: تهيئة محرك RAG ──
    try:
        rag_engine = initialize_rag_engine(settings)
    except Exception as e:
        logger.error(f"❌ فشل تهيئة محرك RAG: {e}")
        sys.exit(1)

    # ── الخطوة 5: بناء تطبيق البوت ──
    try:
        application = build_application(settings)
    except Exception as e:
        logger.error(f"❌ فشل بناء تطبيق البوت: {e}")
        sys.exit(1)

    # ── الخطوة 6: بدء التشغيل ──
    logger.info("=" * 60)
    logger.info("🎓 بوت الدعم التعليمي التونسي جاهز للتشغيل!")
    logger.info("📚 الأوضاع المتاحة: شرح دروس، اختبار ذكي، تقارير أولياء، وضع مراجعة")
    logger.info("=" * 60)

    try:
        from telegram import Update

        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("⏹️ تم إيقاف البوت بواسطة المستخدم.")
    except Exception as e:
        logger.error(f"❌ خطأ أثناء تشغيل البوت: {e}")
        sys.exit(1)
    finally:
        # إغلاق قاعدة البيانات عند الخروج
        try:
            db.close()
            logger.info("تم إغلاق قاعدة البيانات.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
