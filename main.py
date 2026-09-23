#!/usr/bin/env python3
"""
main.py - نقطة الدخول الرئيسية لتطبيق بوت الدعم التعليمي التونسي
"""

import asyncio
import logging
import shutil
import sys
from pathlib import Path
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── إعداد نظام التسجيل ───
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running successfully!")
    def log_message(self, format, *args):
        return

def run_dummy_server():
    import os
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    server.serve_forever()


def check_environment(settings) -> bool:
    errors = []
    warnings = []

    if not settings.telegram_bot_token:
        errors.append("❌ TELEGRAM_BOT_TOKEN غير محدد في متغيرات البيئة (.env)")

    if not settings.openai_api_key:
        errors.append("❌ OPENAI_API_KEY غير محدد في متغيرات البيئة (.env)")

    if shutil.which("tesseract") is None:
        errors.append("❌ Tesseract OCR غير مثبت.")

    chroma_path = settings.chroma_persist_directory
    try:
        chroma_dir = Path(chroma_path) if chroma_path else Path("chroma_db")
        if not chroma_dir.exists():
            chroma_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"❌ تعذر إنشاء مجلد ChromaDB: {e}")

    try:
        db_dir = Path("data")
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"❌ تعذر إنشاء مجلد قاعدة البيانات: {e}")

    for warning in warnings:
        logger.warning(warning)
    for error in errors:
        logger.error(error)

    if errors:
        return False
    return True


def initialize_database():
    logger.info("جاري تهيئة قاعدة البيانات...")
    from database import init_db, get_database
    init_db()
    db = get_database()
    logger.info("✅ تم تهيئة قاعدة البيانات بنجاح.")
    return db


def initialize_rag_engine(settings):
    logger.info("جاري تهيئة محرك RAG (ChromaDB)...")
    from rag_engine import initialize_chroma
    rag_engine = initialize_chroma(
        persist_directory=settings.chroma_persist_directory,
    )
    logger.info("✅ تم تهيئة محرك RAG بنجاح.")
    return rag_engine


def build_application(settings):
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
    from bot_handlers import register_all_handlers
    register_all_handlers(application)
    logger.info("✅ تم بناء التطبيق وتسجيل المعالجات بنجاح.")
    return application


def main():
    # تشغيل خادم الويب الوهمي لمنع خطأ No open ports في Render
    server_thread = threading.Thread(target=run_dummy_server, daemon=True)
    server_thread.start()

    logger.info("=" * 60)
    logger.info("🚀 بدء تشغيل بوت الدعم التعليمي التونسي...")
    logger.info("=" * 60)

    from config import get_settings
    settings = get_settings()

    if not check_environment(settings):
        logger.error("❌ فحص البيئة فشل.")
        sys.exit(1)

    try:
        db = initialize_database()
    except Exception as e:
        logger.error(f"❌ فشل قاعدة البيانات: {e}")
        sys.exit(1)

    try:
        rag_engine = initialize_rag_engine(settings)
    except Exception as e:
        logger.error(f"❌ فشل محرك RAG: {e}")
        sys.exit(1)

    try:
        application = build_application(settings)
    except Exception as e:
        logger.error(f"❌ فشل بناء البوت: {e}")
        sys.exit(1)

    logger.info("🎓 بوت الدعم التعليمي التونسي جاهز للتشغيل!")

    try:
        from telegram import Update
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("⏹️ تم إيقاف البوت.")
    except Exception as e:
        logger.error(f"❌ خطأ أثناء التشغيل: {e}")
        sys.exit(1)
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
