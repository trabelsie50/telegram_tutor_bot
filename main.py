"""
main.py

الملف الرئيسي لتشغيل بوت تيليغرام الخاص بمعلم السنة الخامسة ابتدائي في تونس.
يحتوي على إعداد التطبيق، تشغيل خادم ويب وهمي (لكي يبقى نشطاً على Render)،
وتشغيل حلقة معالجة الرسائل (Polling).
"""

import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from telegram import Update
from telegram.ext import Application

from config import get_settings, Settings
from bot_handlers import register_all_handlers

# إعداد السجلات Logging بشكل احترافي
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# خادم ويب وهمي (Dummy HTTP Server)
# لضمان بقاء الخدمة المجانية على Render مستيقظة وتستقبل الطلبات
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
    # 1. قراءة الإعدادات والتحقق من التوكن
    settings: Settings = get_settings()
    if not settings.telegram_bot_token:
        logger.error("❌ خطأ حرج: لم يتم العثور على TELEGRAM_BOT_TOKEN في متغيرات البيئة!")
        return

    # 2. بدء خادم الويب الوهمي في خيط منفصل (Background Thread)
    # هذا يمنع Render من إغلاق البوت بسبب خمول الويب
    server_thread = threading.Thread(target=run_dummy_server, daemon=True)
    server_thread.start()

    # 3. بناء تطبيق تيليغرام
    logger.info("⚙️ جاري بناء تطبيق تيليغرام وبوت المعلم...")
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    # 4. تسجيل جميع المعالجات (الأوامر، النصوص، الصور، الأزرار)
    register_all_handlers(application)

    # 5. تشغيل البوت بنظام الاستعلام (Polling)
    logger.info("🚀 البوت بدأ العمل الآن ويستمع لرسائل التلامذة...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
