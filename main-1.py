"""
main.py

الملف الرئيسي لتشغيل بوت تيليغرام الخاص بمعلم السنة الخامسة ابتدائي في تونس.

مهم جداً: خادم الويب الوهمي (لفتح المنفذ أمام Render) يجب أن يبدأ فوراً،
قبل أي استيراد ثقيل (sentence-transformers/chromadb/torch عبر bot_handlers).
لهذا الاستيرادات الثقيلة أصبحت مؤجَّلة (داخل main()) بعد بدء تشغيل السيرفر،
وليس في أعلى الملف كما كانت — كانت هذه هي السبب الحقيقي لفشل اكتشاف
المنفذ (No open ports detected) و Timed Out عند Render.
"""

import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# خادم ويب وهمي (Dummy HTTP Server)
# لضمان أن Render يكتشف منفذاً مفتوحاً بسرعة (خلال ثوانٍ)، بدل انتظار
# تحميل مكتبات RAG/الصوت الثقيلة الذي قد يستغرق دقائق.
# -----------------------------------------------------------------------------
class SimpleHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Bot is running successfully!".encode("utf-8"))

    def log_message(self, format, *args):
        # إيقاف طباعة تفاصيل كل طلب HTTP في السجلات لتبقى نظيفة
        return


def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, SimpleHTTPHandler)
    logger.info(f"🌐 خادم الويب الوهمي يعمل على المنفذ {port} بنجاح.")
    httpd.serve_forever()


# -----------------------------------------------------------------------------
# الدالة الرئيسية لتشغيل البوت
# -----------------------------------------------------------------------------
def main() -> None:
    # 1) نفتح المنفذ فوراً، قبل أي استيراد ثقيل — هذا هو التصحيح الأساسي.
    #    Render يفحص المنفذ خلال أول دقيقة أو دقيقتين من الـ deploy؛ إذا لم
    #    يجده، يعتبر الخدمة فاشلة حتى لو البوت سيعمل بشكل سليم لاحقاً.
    server_thread = threading.Thread(target=run_dummy_server, daemon=True)
    server_thread.start()
    logger.info("✅ تم بدء خادم المنفذ الوهمي أولاً (قبل تحميل مكتبات RAG/الصوت الثقيلة).")

    # 2) الاستيرادات الثقيلة (تسحب sentence-transformers/chromadb/torch عبر
    #    bot_handlers) تجي فقط الآن، بعد ما المنفذ فاتح خلاص.
    from telegram import Update
    from telegram.ext import Application
    from config import get_settings, Settings
    from bot_handlers import register_all_handlers

    settings: Settings = get_settings()
    if not settings.telegram_bot_token:
        logger.error("❌ خطأ حرج: لم يتم العثور على TELEGRAM_BOT_TOKEN في متغيرات البيئة!")
        return

    logger.info("⚙️ جاري بناء تطبيق تيليغرام وتحميل محركات RAG/OCR (قد يستغرق دقيقة أو أكثر)...")
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    register_all_handlers(application)

    logger.info("🚀 البوت بدأ العمل الآن ويستمع لرسائل التلامذة...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
