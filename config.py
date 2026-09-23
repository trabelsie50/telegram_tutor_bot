import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """إعدادات المشروع المركزية.

    جميع القيم تُحمَّل من متغيرات البيئة عند التوافر،
    وتُستبدل بقيم افتراضية آمنة في حال عدم وجودها.
    """

    # ── توكن البوت ──────────────────────────────────────────────
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))

    # ── مفتاح OpenAI ────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── نموذج اللغة من OpenAI ───────────────────────────────────
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # ── مسار قاعدة ChromaDB ─────────────────────────────────────
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "chroma_db"))

    # ── حجم نموذج Whisper المحلي ────────────────────────────────
    whisper_model_size: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small")
    )

    # ── محرك OCR الافتراضي ──────────────────────────────────────
    # القيم المقبولة: "tesseract" | "easyocr"
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "tesseract"))

    # ── لغة الافتراض للبوت ──────────────────────────────────────
    # "ar" = عربية (فصحى مبسطة)، "fr" = فرنسية، "ar-tn" = دارجة تونسية
    default_language: str = field(default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "ar"))

    # ── مسار قاعدة بيانات SQLite ────────────────────────────────
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "bot_database.db"))

    # ── معرّف الدردشة للمشرف (وضع المراجعة) ────────────────────
    # يُرسل إليه النص المستخرج والصورة للمراجعة
    supervisor_chat_id: Optional[int] = field(
        default_factory=lambda: int(os.getenv("SUPERVISOR_CHAT_ID", "0"))
        if os.getenv("SUPERVISOR_CHAT_ID")
        else None
    )

    # ── محرك تحويل النص إلى صوت ────────────────────────────────
    # القيم المقبولة: "gtts"
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "gtts"))

    # ── عدد الأسئلة في الاختبار الذكي ───────────────────────────
    quiz_questions_count: int = field(
        default_factory=lambda: int(os.getenv("QUIZ_QUESTIONS_COUNT", "3"))
    )

    # ── دليل الملفات الصوتية المؤقتة ────────────────────────────
    temp_dir: str = field(default_factory=lambda: os.getenv("TEMP_DIR", "temp_voices"))

    # ── مستوى التسجيل ────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ── حد الثقة الأدنى لـ OCR (أقل من هذه القيمة تُراجع يدوياً) ──
    ocr_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
    )

    # ── المواد الدراسية المدعومة ─────────────────────────────────
    supported_subjects: list = field(
        default_factory=lambda: [
            "رياضيات",
            "فرنسية",
            "عربية",
            "إيقاظ علمي",
        ]
    )

    # ── أسماء الملفات الشخصية للتلاميذ ──────────────────────────
    students_file: str = field(default_factory=lambda: os.getenv("STUDENTS_FILE", "students.json"))

    @property
    def whisper_model_path(self) -> str:
        """مسار نموذج Whisper المحمّل محلياً."""
        return os.path.join(
            os.getenv("WHISPER_MODEL_DIR", os.path.expanduser("~/.cache/whisper")),
            self.whisper_model_size,
        )

    @property
    def is_review_mode_enabled(self) -> bool:
        """ما إذا كان وضع المراجعة مفعلاً (يوجد معرّف مشرف)."""
        return self.supervisor_chat_id is not None and self.supervisor_chat_id > 0

    @property
    def chroma_persist_directory(self) -> str:
        """مسار تخزين ChromaDB الدائم."""
        return os.path.abspath(self.chroma_path)


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """إرجاع النسخة الوحيدة (singleton) من كائن الإعدادات."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings: Settings = get_settings()
